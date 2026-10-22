/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Decoding windows built from dem_chunks: request validation, instance
// ranges, window construction from a locally stitched chain, and the
// canonical workload's commit, sends and frame update. Construction runs at
// configuration time and may allocate; nothing here is on the hot path.

#include "decoding_task_decoding_window.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace cudaq::qec {
namespace {

// Throws std::invalid_argument naming the operation.
[[noreturn]] void fail(const std::string &what) {
  throw std::invalid_argument("window_from_dem_chunks: " + what);
}

// Throws std::invalid_argument for a request that breaks a construction
// rule, which config validation reports as well as construction.
[[noreturn]] void reject(const std::string &what) {
  throw std::invalid_argument("window request: " + what);
}

// Floor of a / b for b > 0.
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  return a >= 0 ? a / b : -((-a + b - 1) / b);
}

// Ceiling of a / b for b > 0.
std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
  return floor_div(a + b - 1, b);
}

// Whether a send is one of the spec's defaults, named by id alone.
bool is_default_send(const window_request::send_request &send) {
  return (send.id == "forward" || send.id == "backward") && !send.mode &&
         send.rounds.empty();
}

// The buffer zone that starts where the commit zone ends, if any.
const round_range *trailing_buffer(const window_request &request) {
  for (const auto &buffer : request.buffers)
    if (buffer.begin == request.commit->end)
      return &buffer;
  return nullptr;
}

// Every zone of the request, sorted by first round.
std::vector<round_range> sorted_zones(const window_request &request) {
  std::vector<round_range> zones = request.buffers;
  zones.push_back(*request.commit);
  std::sort(zones.begin(), zones.end(),
            [](const round_range &a, const round_range &b) {
              return a.begin < b.begin;
            });
  return zones;
}

// Checks the period and the commit zone, which every query needs.
void check_anchor(const window_request &request) {
  if (request.period < 1)
    reject("period must be at least 1");
  if (!request.commit)
    reject("no commit zone");
  if (request.commit->empty())
    reject("the commit zone is empty");
}

// Checks one request's sends against its relative rounds [begin, end).
void check_sends(const window_request &request, round_range window) {
  std::vector<std::string> ids;
  for (const auto &send : request.sends) {
    const auto name = "send '" + send.id + "'";
    auto rounds = send.rounds;
    if (std::find(ids.begin(), ids.end(), send.id) != ids.end())
      reject(name + " is repeated");
    ids.push_back(send.id);
    if (is_default_send(send)) {
      if (send.id == "forward" && !trailing_buffer(request))
        reject("a forward send needs a buffer zone after the commit zone");
      continue;
    }
    if (!send.mode || send.rounds.empty())
      reject(name + " needs a mode and rounds");
    std::sort(rounds.begin(), rounds.end(),
              [](const round_range &a, const round_range &b) {
                return a.begin < b.begin;
              });
    for (std::size_t i = 0; i < rounds.size(); ++i) {
      if (rounds[i].empty())
        reject(name + " has an empty round range");
      if (rounds[i].begin < window.begin || rounds[i].end > window.end)
        reject(name + " reaches past the window");
      if (i > 0 && rounds[i].begin < rounds[i - 1].end)
        reject(name + " repeats round " + std::to_string(rounds[i].begin));
    }
  } // end - for(send)
} // end - check_sends()

// Checks a request against the spec's construction rules: one non-empty
// commit zone, disjoint contiguous zones, a disposition on each port, a
// valid prior, imports at the output wire, and well-formed sends.
void check_request(const window_request &request) {
  check_anchor(request);
  const auto zones = sorted_zones(request);
  const auto &input = request.input;
  const auto &output = request.output;
  const round_range window{zones.front().begin, zones.back().end};
  for (const auto &buffer : request.buffers)
    if (buffer.empty())
      reject("a buffer zone is empty");
  for (std::size_t i = 1; i < zones.size(); ++i) {
    if (zones[i].begin < zones[i - 1].end)
      reject("zones overlap");
    if (zones[i].begin > zones[i - 1].end)
      reject("zones are not contiguous");
  }
  if (!input.disposition)
    reject("the input port has no disposition");
  if (!output.disposition)
    reject("the output port has no disposition");
  const bool open_input = *input.disposition == port_disposition::open;
  const bool truncated = input.boundary == input_boundary::truncate;
  if (!open_input && input.boundary)
    reject("a closed input takes no boundary");
  if (open_input && !truncated && !input.prior)
    reject("an open input needs a prior");
  if (!open_input && input.prior)
    reject("a closed input takes no prior");
  if (truncated && input.prior)
    reject("a truncated input takes no prior");
  if (input.prior && !(*input.prior > 0.0 && *input.prior < 1.0))
    reject("the input prior must be strictly between 0 and 1");
  if (output.imports) {
    if (*output.disposition == port_disposition::open)
      reject("an open output imports nothing");
    if (output.imports->empty())
      reject("the imports are empty");
    if (output.imports->begin != window.end)
      reject("imports must start at the output wire");
  }
  check_sends(request, window);
} // end - check_request()

// The rounds a window's model reads, relative to its anchor: from the round
// before its first (v_in, and a truncated input's crossing faults) to the
// end of the chunks it stitches. Past that it reads only whether the end
// round exists, which decides the last chunk's outgoing seam and the output
// port.
round_range model_reach(const experiment_template &t,
                        const window_request &request) {
  const auto zones = sorted_zones(request);
  const auto m = static_cast<std::int64_t>(t.max_span());
  const auto end = zones.back().end;
  auto imports_end = end;
  if (request.output.disposition == port_disposition::closed)
    imports_end =
        request.output.imports ? request.output.imports->end : end + m;
  return {zones.front().begin - 1, std::max(end + m, imports_end)};
}

// The first k whose model reads no prefix round.
std::int64_t first_bulk_index(const experiment_template &t,
                              const window_request &request) {
  const auto reach = model_reach(t, request);
  return ceil_div(static_cast<std::int64_t>(t.prefix_rounds()) -
                      request.offset - reach.begin,
                  request.period);
}

// Rounds the suffix keeps from the shot's end, and at least the last one,
// whose chunk has no outgoing seam.
std::int64_t end_rounds(const experiment_template &t) {
  return std::max<std::int64_t>(t.suffix_rounds(), 1);
}

// Row-major element access to a rank-2 tensor. tensor::at() allocates an
// index vector and builds a view on every call, and tensor::shape() returns
// a new vector, so loops over a window's model read both once through this.
template <typename T>
struct dense_view {
  T *data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  T &operator()(std::size_t r, std::size_t c) const {
    return data[r * cols + c];
  }
};

// A mutable view of m.
dense_view<std::uint8_t> view(cudaqx::tensor<uint8_t> &m) {
  const auto shape = m.shape();
  return {m.data(), shape[0], shape[1]};
}

// A read-only view of m.
dense_view<const std::uint8_t> view(const cudaqx::tensor<uint8_t> &m) {
  const auto shape = m.shape();
  return {m.data(), shape[0], shape[1]};
}

// Whether two dense matrices are equal, compared as one buffer.
bool same_tensor(const cudaqx::tensor<uint8_t> &a,
                 const cudaqx::tensor<uint8_t> &b) {
  return a.shape() == b.shape() &&
         std::equal(a.data(), a.data() + a.size(), b.data());
}

// Whether two affine maps are equal.
bool same_map(const affine_map &a, const affine_map &b) {
  return a.matrix.num_rows() == b.matrix.num_rows() &&
         a.matrix.num_cols() == b.matrix.num_cols() &&
         a.matrix.to_nested_csr() == b.matrix.to_nested_csr() &&
         a.offset == b.offset && a.input_sizes == b.input_sizes &&
         a.output_sizes == b.output_sizes;
}

// Rows r of a GF(2) map as sorted column lists, with repeated columns
// cancelled in pairs.
sparse_binary_matrix map_matrix(std::size_t num_rows, std::size_t num_cols,
                                std::vector<std::vector<std::uint32_t>> rows) {
  for (auto &row : rows) {
    std::vector<std::uint32_t> kept;
    std::sort(row.begin(), row.end());
    for (const auto c : row)
      if (!kept.empty() && kept.back() == c)
        kept.pop_back();
      else
        kept.push_back(c);
    row = std::move(kept);
  }
  return sparse_binary_matrix::from_nested_csr(
      static_cast<sparse_binary_matrix::index_type>(num_rows),
      static_cast<sparse_binary_matrix::index_type>(num_cols), rows);
}

// Column c of a matrix, over every row.
std::vector<std::uint8_t> column_bits(dense_view<const std::uint8_t> m,
                                      std::size_t c) {
  std::vector<std::uint8_t> bits(m.rows);
  for (std::size_t r = 0; r < m.rows; ++r)
    bits[r] = m(r, c);
  return bits;
}

// Groups the window's fault columns, closed columns first + j for zone[j],
// into built columns. A class is a zone's columns with identical full rows
// in the closed chain, which holds every row they touch. most_likely merges
// a class whose observable effects differ; merge_identical merges the
// members with equal effects. Groups are ordered by their first column, and
// list their most likely column first, ties by index.
std::vector<std::vector<std::size_t>>
group_fault_columns(const detector_error_model &closed, std::size_t first,
                    const std::vector<std::size_t> &zone,
                    const window_request::column_options &options) {
  using signature = std::pair<std::size_t, std::vector<std::uint8_t>>;
  std::map<signature, std::vector<std::size_t>> classes;
  std::vector<std::vector<std::size_t>> by_first(zone.size());
  std::vector<std::vector<std::size_t>> groups;
  const auto h = view(closed.detector_error_matrix);
  const auto o = view(closed.observables_flips_matrix);
  const auto &rates = closed.error_rates;
  // Without an option every column is its own group.
  if (!options.merge_identical &&
      options.conflicting_parallel == parallel_resolution::reject) {
    for (std::size_t j = 0; j < zone.size(); ++j)
      groups.push_back({j});
    return groups;
  }
  for (std::size_t j = 0; j < zone.size(); ++j)
    classes[{zone[j], column_bits(h, first + j)}].push_back(j);
  for (const auto &[key, members] : classes) {
    std::map<std::vector<std::uint8_t>, std::vector<std::size_t>> by_effect;
    for (const auto j : members)
      by_effect[column_bits(o, first + j)].push_back(j);
    if (options.conflicting_parallel == parallel_resolution::most_likely &&
        by_effect.size() > 1) {
      by_first[members.front()] = members;
    } else {
      for (const auto &[effect, part] : by_effect)
        if (options.merge_identical)
          by_first[part.front()] = part;
        else
          for (const auto j : part)
            by_first[j] = {j};
    }
  } // end - for(classes)
  for (auto &group : by_first) {
    if (group.empty())
      continue;
    std::stable_sort(group.begin(), group.end(),
                     [&](std::size_t a, std::size_t b) {
                       return rates[first + a] > rates[first + b];
                     });
    groups.push_back(std::move(group));
  }
  return groups;
} // end - group_fault_columns()

// The measurement layout of a window's rounds: each round's width and where
// its measurements start among the map's inputs, after v_in.
struct measurement_layout {
  std::size_t v_in = 0;
  std::vector<std::size_t> width; ///< Per round of the window
  std::vector<std::size_t> start; ///< Per round, then the end
};

// Without explicit D rows, each round measures its detectors (the memory
// convention of dem_chunks_to_d_sparse()).
std::size_t round_width(const experiment_template &t, std::int64_t r,
                        std::size_t R) {
  if (t.spec().has_D_sparse())
    return t.measurement_offset(r + 1, R) - t.measurement_offset(r, R);
  return t.row_offset(r + 1, R) - t.row_offset(r, R);
}

// Lays out the measurements of rounds [s, e), after s's incoming wire.
measurement_layout layout_measurements(const experiment_template &t,
                                       round_range rounds, std::size_t R) {
  measurement_layout out;
  out.v_in = rounds.begin > 0 ? round_width(t, rounds.begin - 1, R) : 0;
  out.start.push_back(out.v_in);
  for (auto r = rounds.begin; r < rounds.end; ++r) {
    out.width.push_back(round_width(t, r, R));
    out.start.push_back(out.start.back() + out.width.back());
  }
  return out;
}

// m2d: each own round's detectors from its D rows over (previous round,
// this round), or the memory convention, then the outgoing wires' virtual
// outcomes, which on dem_chunks are the measurements of the round before.
affine_map build_m2d(const experiment_template &t, const decoding_window &w,
                     const measurement_layout &m) {
  affine_map map;
  std::vector<std::vector<std::uint32_t>> rows;
  std::vector<std::int64_t> wires;
  const auto R = w.shot_rounds;
  const auto s = w.rounds.begin;
  for (auto r = s; r < w.rounds.end; ++r) {
    const auto i = static_cast<std::size_t>(r - s);
    const std::size_t prev_start = i == 0 ? 0 : m.start[i - 1];
    const std::size_t prev_width = i == 0 ? m.v_in : m.width[i - 1];
    const auto detectors = t.row_offset(r + 1, R) - t.row_offset(r, R);
    const auto first_row = rows.size();
    if (!t.spec().has_D_sparse()) {
      for (std::size_t j = 0; j < detectors; ++j) {
        auto &row = rows.emplace_back();
        row.push_back(static_cast<std::uint32_t>(m.start[i] + j));
        if (r > 0)
          row.push_back(static_cast<std::uint32_t>(prev_start + j));
      }
      continue;
    }
    std::vector<std::uint32_t> row;
    for (const auto value : t.spec().get_phase(t.phase(r, R)).D_sparse) {
      const auto index = static_cast<std::size_t>(value);
      if (value == -1) {
        rows.push_back(std::move(row));
        row.clear();
      } else if (index < prev_width) {
        row.push_back(static_cast<std::uint32_t>(prev_start + index));
      } else if (index - prev_width < m.width[i]) {
        row.push_back(
            static_cast<std::uint32_t>(m.start[i] + index - prev_width));
      } else {
        fail("round " + std::to_string(r) + "'s D_sparse reads measurement " +
             std::to_string(index) + ", past its round");
      }
    } // end - for(value)
    if (rows.size() - first_row != detectors)
      fail("round " + std::to_string(r) + "'s D_sparse has " +
           std::to_string(rows.size() - first_row) + " rows for " +
           std::to_string(detectors) + " detectors");
  } // end - for(r)
  map.output_sizes.push_back(rows.size());

  if (w.commit.end < static_cast<std::int64_t>(R))
    wires.push_back(w.commit.end);
  if (w.output.exists && w.rounds.end != w.commit.end)
    wires.push_back(w.rounds.end);
  for (const auto wire : wires) {
    const auto i = static_cast<std::size_t>(wire - 1 - s);
    for (std::size_t j = 0; j < m.width[i]; ++j)
      rows.push_back({static_cast<std::uint32_t>(m.start[i] + j)});
    map.output_sizes.push_back(m.width[i]);
  }
  const auto num_outputs = rows.size();
  map.input_sizes = {m.v_in, m.start.back() - m.v_in};
  map.offset.assign(num_outputs, 0);
  map.matrix = map_matrix(num_outputs, m.start.back(), std::move(rows));
  return map;
} // end - build_m2d()

// m2o: the readout round's records, from the last phase's m2o_sparse, over
// the window's measurements alone.
affine_map build_m2o(const experiment_template &t, const decoding_window &w,
                     const measurement_layout &m) {
  affine_map map;
  std::vector<std::vector<std::uint32_t>> rows;
  std::vector<std::uint32_t> row;
  const auto R = static_cast<std::int64_t>(w.shot_rounds);
  if (w.rounds.end != R)
    fail("m2o needs the readout round, but window " + std::to_string(w.k) +
         " ends at round " + std::to_string(w.rounds.end) + " of " +
         std::to_string(R));
  if (t.readout().empty())
    fail("m2o needs the last phase's m2o_sparse");
  const auto base = m.start[R - 1 - w.rounds.begin] - m.v_in;
  for (const auto value : t.readout()) {
    if (value == -1) {
      rows.push_back(std::move(row));
      row.clear();
    } else {
      row.push_back(static_cast<std::uint32_t>(base + value));
    }
  }
  const auto num_outputs = rows.size();
  map.input_sizes = {m.start.back() - m.v_in};
  map.output_sizes = {num_outputs};
  map.offset.assign(num_outputs, 0);
  map.matrix =
      map_matrix(num_outputs, m.start.back() - m.v_in, std::move(rows));
  return map;
} // end - build_m2o()

} // namespace

void experimental::check_window_request(const window_request &request) {
  check_request(request);
}

// The first k whose clipped commit zone ends after round 0, to the first
// whose commit zone starts at or past R.
index_range window_instances(const window_request &request,
                             std::size_t num_rounds) {
  check_anchor(request);
  const auto R = static_cast<std::int64_t>(num_rounds);
  const auto period = request.period;
  const auto &commit = *request.commit;
  index_range out;
  out.begin = floor_div(-request.offset - commit.end, period) + 1;
  out.end = floor_div(R - request.offset - commit.begin + period - 1, period);
  out.end = std::max(out.end, out.begin);
  return out;
}

// Implements the plan's construction: clip the zones, build the ports and
// sends, stitch the window's chunks and the imported ones, and slice the
// closed chain into the window's model and maps.
decoding_window
window_from_dem_chunks(std::shared_ptr<const experiment_template> context,
                       const window_request &request, std::int64_t k,
                       std::size_t num_rounds, bool with_m2o) {
  decoding_window w;
  std::vector<extended_dem> chunks;
  if (!context)
    fail("no template");
  const auto &t = *context;
  const auto R = static_cast<std::int64_t>(num_rounds);
  const auto anchor = request.offset + k * request.period;
  const auto clip = [&](round_range z) {
    return round_range{std::max<std::int64_t>(0, anchor + z.begin),
                       std::min<std::int64_t>(R, anchor + z.end)};
  };
  check_request(request);
  t.row_offset(num_rounds, num_rounds); // throws unless the spec admits R
  const auto relative_end = sorted_zones(request).back().end;
  const auto m_span = static_cast<std::int64_t>(t.max_span());

  // Zones and ports.
  w.context = context;
  w.k = k;
  w.shot_rounds = num_rounds;
  w.commit = clip(*request.commit);
  if (w.commit.empty())
    fail("window " + std::to_string(k) + " has no commit round in a " +
         std::to_string(R) + "-round shot");
  w.rounds = w.commit;
  for (const auto &buffer : request.buffers) {
    const auto clipped = clip(buffer);
    if (clipped.empty())
      continue;
    w.buffers.push_back(clipped);
    w.rounds.begin = std::min(w.rounds.begin, clipped.begin);
    w.rounds.end = std::max(w.rounds.end, clipped.end);
  }
  std::sort(w.buffers.begin(), w.buffers.end(),
            [](const round_range &a, const round_range &b) {
              return a.begin < b.begin;
            });
  const auto s = w.rounds.begin;
  const auto e = w.rounds.end;
  w.input.exists = s > 0;
  w.input.disposition = *request.input.disposition;
  w.output.exists = e < R;
  w.output.disposition = *request.output.disposition;
  w.input.boundary = request.input.boundary.value_or(input_boundary::columns);
  const bool open_input =
      w.input.exists && w.input.disposition == port_disposition::open;
  const bool truncated =
      open_input && w.input.boundary == input_boundary::truncate;
  const bool boundary_columns = open_input && !truncated;
  if (boundary_columns)
    w.input.prior = *request.input.prior;
  if (w.output.exists && w.output.disposition == port_disposition::closed)
    w.output.imports = clip(request.output.imports.value_or(
        round_range{relative_end, relative_end + m_span}));

  // Rows and columns, relative to round s. The chain starts at round lo: s,
  // or the round before it, whose crossing faults a truncated input keeps;
  // its rows and columns are shifted by the rounds before s.
  const auto row = [&](std::int64_t r) {
    return t.row_offset(r, num_rounds) - t.row_offset(s, num_rounds);
  };
  const auto col = [&](std::int64_t r) {
    return t.column_offset(r, num_rounds) - t.column_offset(s, num_rounds);
  };
  const auto lo = truncated ? s - 1 : s;
  const auto row_shift =
      t.row_offset(s, num_rounds) - t.row_offset(lo, num_rounds);
  const auto col_shift =
      t.column_offset(s, num_rounds) - t.column_offset(lo, num_rounds);
  // Closing drops the last chunk's outgoing band, so the chain runs past e
  // far enough to hold every row the window's faults touch, which the
  // column options compare, and past the imports.
  const auto hi =
      std::max(std::min<std::int64_t>(R, e + m_span), w.output.imports.end);
  for (auto r = lo; r < hi; ++r)
    chunks.push_back(t.chunk(r, num_rounds));
  const auto closed =
      dem_close_all(chunks, t.spec().seam.from_seam, t.spec().seam.to_seam);
  const auto closed_h = view(closed.detector_error_matrix);
  const auto closed_o = view(closed.observables_flips_matrix);

  // The window's rows of the chain: own, then imports.
  std::vector<std::size_t> rows;
  w.num_own_rows = row(e);
  for (std::size_t i = 0; i < w.num_own_rows; ++i)
    rows.push_back(row_shift + i);
  if (!w.output.imports.empty())
    for (auto i = row(w.output.imports.begin); i < row(w.output.imports.end);
         ++i)
      rows.push_back(row_shift + i);

  // Fault columns, grouped by the column options, with each one's zone: 0
  // for the commit zone, else 1 plus its buffer's index.
  std::vector<std::size_t> zone;
  std::vector<std::size_t> crossing;
  for (auto r = s; r < e; ++r) {
    std::size_t z = 0;
    for (std::size_t b = 0; b < w.buffers.size(); ++b)
      if (r >= w.buffers[b].begin && r < w.buffers[b].end)
        z = b + 1;
    zone.resize(col(r + 1), z);
  }
  w.fault_column_sources =
      group_fault_columns(closed, col_shift, zone, request.columns);
  for (std::size_t c = 0; c < col_shift; ++c)
    if (std::any_of(rows.begin(), rows.end(),
                    [&](std::size_t r) { return closed_h(r, c) != 0; }))
      crossing.push_back(c);

  // The model: fault columns, then boundary or crossing columns, then
  // logical-error columns.
  const auto &groups = w.fault_column_sources;
  const auto first_commit = static_cast<std::size_t>(col(w.commit.begin));
  w.num_fault_columns = groups.size();
  w.num_virtual_outcome_columns = boundary_columns ? row(s + 1) : 0;
  w.num_crossing_columns = crossing.size();
  w.num_logical_error_columns = boundary_columns ? t.num_observables() : 0;
  for (const auto &group : groups) {
    if (group.front() < first_commit)
      ++w.commit_columns.begin;
    if (zone[group.front()] == 0)
      ++w.commit_columns.end;
  }
  w.commit_columns.end += w.commit_columns.begin;
  const auto F = w.num_fault_columns;
  const auto V = w.num_virtual_outcome_columns;
  const auto C = w.num_crossing_columns;
  const auto K = w.num_logical_error_columns;
  const auto n_obs = closed.num_observables();
  auto &model = w.model;
  model.detector_error_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{rows.size(), F + V + C + K});
  model.observables_flips_matrix =
      cudaqx::tensor<uint8_t>(std::vector<std::size_t>{n_obs, F + V + C + K});
  model.error_rates.assign(F + V + C + K, w.input.prior);
  const auto model_h = view(model.detector_error_matrix);
  const auto model_o = view(model.observables_flips_matrix);
  for (std::size_t g = 0; g < F; ++g) {
    const auto source = col_shift + groups[g].front();
    double prior = 0.0;
    for (std::size_t i = 0; i < rows.size(); ++i)
      model_h(i, g) = closed_h(rows[i], source);
    for (std::size_t o = 0; o < n_obs; ++o)
      model_o(o, g) = closed_o(o, source);
    for (const auto j : groups[g]) {
      const auto p = closed.error_rates[col_shift + j];
      prior = prior + p - 2 * prior * p;
    }
    model.error_rates[g] = prior;
  } // end - for(g)
  for (std::size_t j = 0; j < V; ++j)
    model_h(j, F + j) = 1;
  for (std::size_t x = 0; x < C; ++x) {
    for (std::size_t i = 0; i < rows.size(); ++i)
      model_h(i, F + V + x) = closed_h(rows[i], crossing[x]);
    for (std::size_t o = 0; o < n_obs; ++o)
      model_o(o, F + V + x) = closed_o(o, crossing[x]);
    model.error_rates[F + V + x] = closed.error_rates[crossing[x]];
  }
  for (std::size_t o = 0; o < K; ++o)
    model_o(o, F + V + C + o) = 1;

  // Sends: the defaults cover the first max_span() rounds of their zone.
  for (const auto &request_send : request.sends) {
    window_send send;
    std::vector<round_range> rounds = request_send.rounds;
    send.id = request_send.id;
    send.mode = request_send.mode.value_or(
        send.id == "backward" ? send_mode::residual : send_mode::delta);
    if (is_default_send(request_send)) {
      const auto first = send.id == "forward" ? trailing_buffer(request)->begin
                                              : request.commit->begin;
      rounds = {{first, first + m_span}};
    }
    std::sort(rounds.begin(), rounds.end(),
              [](const round_range &a, const round_range &b) {
                return a.begin < b.begin;
              });
    for (const auto &relative : rounds) {
      const auto clipped = clip(relative);
      if (clipped.empty())
        continue;
      send.rounds.push_back(clipped);
      for (auto i = row(clipped.begin); i < row(clipped.end); ++i)
        send.rows.push_back(i);
    }
    w.sends.push_back(std::move(send));
  } // end - for(request_send)

  // Measurement maps.
  const auto layout = layout_measurements(t, w.rounds, num_rounds);
  w.num_virtual_outcomes = layout.v_in;
  w.num_measurements = layout.start.back() - layout.v_in;
  w.m2d = build_m2d(t, w, layout);
  if (with_m2o)
    w.m2o = build_m2o(t, w, layout);
  return w;
} // end - window_from_dem_chunks()

// Prefix and terminal positions follow from where the model's reach meets
// the shot's prefix rounds and its end.
variant_key window_variant(const experiment_template &context,
                           const window_request &request, std::int64_t k,
                           std::size_t num_rounds) {
  variant_key key;
  check_anchor(request);
  const auto reach = model_reach(context, request);
  const auto R = static_cast<std::int64_t>(num_rounds);
  const auto anchor = request.offset + k * request.period;
  if (anchor + reach.begin < static_cast<std::int64_t>(context.prefix_rounds()))
    key.position = k;
  if (anchor + reach.end > R - end_rounds(context))
    key.terminal = R - anchor;
  return key;
}

// Every prefix key has k below k_bulk and every terminal key R - anchor
// below the reach's end plus end_rounds(), so the shots up to one period
// past the bulk at k_bulk + 1 hold every key.
std::size_t variant_probe_rounds(const experiment_template &context,
                                 const window_request &request) {
  check_request(request);
  if (!context.repeats())
    return context.min_rounds() + 1;
  const auto reach = model_reach(context, request);
  const auto k = first_bulk_index(context, request) + 1;
  const auto end =
      request.offset + k * request.period + reach.end + end_rounds(context) + 1;
  return std::max<std::size_t>(
      context.min_rounds(),
      static_cast<std::size_t>(std::max<std::int64_t>(end, 0)));
}

// Compares in a fixed order, so the first mismatch named is deterministic:
// rounds, zones, ports, sends, model, then the measurement maps.
void check_same_window_up_to_shift(const decoding_window &a,
                                   const decoding_window &b) {
  const auto differ = [&](const std::string &what) {
    throw std::invalid_argument("check_same_window_up_to_shift: windows " +
                                std::to_string(a.k) + " and " +
                                std::to_string(b.k) + " differ in " + what);
  };
  const auto shift = [](round_range r, std::int64_t by) {
    return r.empty() ? round_range{} : round_range{r.begin - by, r.end - by};
  };
  const auto shifted = [&](const std::vector<round_range> &ranges,
                           std::int64_t by) {
    std::vector<round_range> out;
    for (const auto r : ranges)
      out.push_back(shift(r, by));
    return out;
  };
  const auto da = a.rounds.begin;
  const auto db = b.rounds.begin;
  const auto port_fields = [&](const window_port &p, std::int64_t by) {
    return std::tuple{p.exists, p.disposition, p.boundary, p.prior,
                      shift(p.imports, by)};
  };
  if (a.rounds.end - da != b.rounds.end - db)
    differ("their rounds");
  if (shift(a.commit, da) != shift(b.commit, db) ||
      shifted(a.buffers, da) != shifted(b.buffers, db))
    differ("their zones");
  if (port_fields(a.input, da) != port_fields(b.input, db))
    differ("their input port");
  if (port_fields(a.output, da) != port_fields(b.output, db))
    differ("their output port");
  if (a.sends.size() != b.sends.size())
    differ("their sends");
  for (std::size_t i = 0; i < a.sends.size(); ++i)
    if (a.sends[i].id != b.sends[i].id || a.sends[i].mode != b.sends[i].mode ||
        shifted(a.sends[i].rounds, da) != shifted(b.sends[i].rounds, db) ||
        a.sends[i].rows != b.sends[i].rows)
      differ("send '" + a.sends[i].id + "'");
  if (a.num_own_rows != b.num_own_rows ||
      a.num_fault_columns != b.num_fault_columns ||
      a.num_virtual_outcome_columns != b.num_virtual_outcome_columns ||
      a.num_crossing_columns != b.num_crossing_columns ||
      a.num_logical_error_columns != b.num_logical_error_columns ||
      a.commit_columns != b.commit_columns ||
      a.fault_column_sources != b.fault_column_sources ||
      !same_tensor(a.model.detector_error_matrix,
                   b.model.detector_error_matrix) ||
      !same_tensor(a.model.observables_flips_matrix,
                   b.model.observables_flips_matrix) ||
      a.model.error_rates != b.model.error_rates)
    differ("their model");
  if (a.num_virtual_outcomes != b.num_virtual_outcomes ||
      a.num_measurements != b.num_measurements || !same_map(a.m2d, b.m2d))
    differ("their m2d");
  if (a.m2o.has_value() != b.m2o.has_value() ||
      (a.m2o && !same_map(*a.m2o, *b.m2o)))
    differ("their m2o");
} // end - check_same_window_up_to_shift()

// Builds k_bulk and k_bulk + 1 in the shortest shot where both are bulk.
void check_translation_invariance(
    std::shared_ptr<const experiment_template> context,
    const window_request &request) {
  if (!context)
    fail("no template");
  check_request(request);
  if (!context->repeats())
    return;
  const auto reach = model_reach(*context, request);
  const auto k = first_bulk_index(*context, request);
  const auto R =
      std::max<std::int64_t>(static_cast<std::int64_t>(context->min_rounds()),
                             request.offset + (k + 1) * request.period +
                                 reach.end + end_rounds(*context));
  const auto num_rounds = static_cast<std::size_t>(R);
  check_same_window_up_to_shift(
      window_from_dem_chunks(context, request, k, num_rounds),
      window_from_dem_chunks(context, request, k + 1, num_rounds));
}

// H keeps the decoder's columns, the leading ones; O and the priors keep
// every column.
window_kernel_forms kernel_forms(const decoding_window &window) {
  window_kernel_forms forms;
  const auto h = view(window.model.detector_error_matrix);
  const auto o = view(window.model.observables_flips_matrix);
  const auto columns = [](dense_view<const std::uint8_t> m, std::size_t n) {
    std::vector<std::vector<sparse_binary_matrix::index_type>> nested(n);
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t r = 0; r < m.rows; ++r)
        if (m(r, c))
          nested[c].push_back(static_cast<sparse_binary_matrix::index_type>(r));
    return sparse_binary_matrix::from_nested_csc(
        static_cast<sparse_binary_matrix::index_type>(m.rows),
        static_cast<sparse_binary_matrix::index_type>(n), nested);
  };
  forms.h = columns(h, window.num_decoder_columns());
  forms.o = columns(o, o.cols);
  forms.priors = window.model.error_rates;
  return forms;
}

// Columns past H's are the ones the decoder does not see.
detector_error_model model_from_kernel_forms(const window_kernel_forms &forms) {
  detector_error_model model;
  const auto total = static_cast<std::size_t>(forms.o.num_cols());
  const auto h = forms.h.to_nested_csc();
  const auto o = forms.o.to_nested_csc();
  model.detector_error_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{forms.h.num_rows(), total});
  model.observables_flips_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{forms.o.num_rows(), total});
  const auto model_h = view(model.detector_error_matrix);
  const auto model_o = view(model.observables_flips_matrix);
  for (std::size_t c = 0; c < h.size(); ++c)
    for (const auto r : h[c])
      model_h(r, c) = 1;
  for (std::size_t c = 0; c < o.size(); ++c)
    for (const auto r : o[c])
      model_o(r, c) = 1;
  model.error_rates = forms.priors;
  return model;
}

// e' masks every column outside the commit zone; each send multiplies its
// rows of H by e', and a residual adds the window's detectors.
window_outputs commit_window(const decoding_window &window,
                             const std::vector<std::uint8_t> &detectors,
                             const std::vector<std::uint8_t> &estimate) {
  window_outputs out;
  const auto h = view(window.model.detector_error_matrix);
  const auto o = view(window.model.observables_flips_matrix);
  const auto commit = window.commit_columns;
  if (detectors.size() != h.rows || estimate.size() != h.cols)
    throw std::invalid_argument("commit_window: expected " +
                                std::to_string(h.rows) + " detectors and " +
                                std::to_string(h.cols) + " estimates, got " +
                                std::to_string(detectors.size()) + " and " +
                                std::to_string(estimate.size()));
  out.committed.assign(estimate.size(), 0);
  for (auto c = commit.begin; c < commit.end; ++c)
    out.committed[c] = estimate[c] & 1;
  for (const auto &send : window.sends) {
    auto &values = out.sends.emplace_back();
    for (const auto r : send.rows) {
      std::uint8_t delta = 0;
      for (auto c = commit.begin; c < commit.end; ++c)
        delta ^= h(r, c) & out.committed[c];
      values.push_back(send.mode == send_mode::residual
                           ? static_cast<std::uint8_t>(detectors[r] ^ delta)
                           : delta);
    }
  }
  out.frame.assign(o.rows, 0);
  for (std::size_t i = 0; i < o.rows; ++i)
    for (auto c = commit.begin; c < commit.end; ++c)
      out.frame[i] ^= o(i, c) & out.committed[c];
  return out;
} // end - commit_window()

} // namespace cudaq::qec
