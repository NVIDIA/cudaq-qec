/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Decoding windows built from dem_chunks: request validation, instance
// ranges, window placement, construction from a locally closed chain with
// its commit columns' parallel conflicts, and the canonical workload:
// receiving sends, decoding, and the commit, sends and frame update. Models
// stay sparse throughout, since a window at a large distance would take
// gigabytes as dense matrices. Construction, decoders and workspaces are made
// at configuration time and may allocate. The canonical workload runs once
// per shot: on a workspace made beforehand, receive_send(), solve_window()
// and commit_window() allocate nothing, beyond what the decoder's own
// decode() does.

#include "decoding_task_decoding_window.h"
#include "cudaq/qec/decoder_init.h"
#include <algorithm>
#include <limits>
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

// Rejects a value further than window_round_limit from zero, so window
// arithmetic on it stays within int64.
void check_within(std::int64_t value, const std::string &what) {
  if (value > window_round_limit || value < -window_round_limit)
    reject(what + " " + std::to_string(value) + " is beyond " +
           std::to_string(window_round_limit) + " rounds, past any shot");
}

// Checks the period, the offset and the commit zone, which every query
// needs.
void check_anchor(const window_request &request) {
  if (request.period < 1)
    reject("period must be at least 1");
  check_within(request.period, "period");
  check_within(request.offset, "offset");
  if (!request.commit)
    reject("no commit zone");
  check_within(request.commit->begin, "round");
  check_within(request.commit->end, "round");
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
      check_within(rounds[i].begin, "round");
      check_within(rounds[i].end, "round");
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
  for (const auto &buffer : request.buffers) {
    check_within(buffer.begin, "round");
    check_within(buffer.end, "round");
    if (buffer.empty())
      reject("a buffer zone is empty");
  }
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
    check_within(output.imports->begin, "round");
    check_within(output.imports->end, "round");
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
// a new vector, so dense_model() reads both once through this.
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

// Whether two matrices are equal; canonical ones compare entry by entry.
bool same_matrix(const sparse_binary_matrix &a, const sparse_binary_matrix &b) {
  return a.layout() == b.layout() && a.num_rows() == b.num_rows() &&
         a.num_cols() == b.num_cols() && a.ptr() == b.ptr() &&
         a.indices() == b.indices();
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

// Row or observable indices of one column, sorted.
using index_list = std::vector<sparse_binary_matrix::index_type>;

// A closed chain by column: what dem_close_all() gives, without its dense
// matrices, which a window at a large distance cannot afford.
struct chain_columns {
  std::size_t num_rows = 0;
  std::size_t num_observables = 0;
  std::vector<index_list> rows;        ///< Each fault column's rows
  std::vector<index_list> observables; ///< Each fault column's observables
  std::vector<double> priors;          ///< Each fault column's prior
};

// Closes chunks column by column: H from dem_chunks_to_pcm(), which keeps
// dem_close_all()'s rows, O from dem_chunks_to_o_sparse(), and the chunks'
// priors in order.
chain_columns close_columns(const std::vector<extended_dem> &chunks,
                            seam_id from_seam, seam_id to_seam) {
  chain_columns out;
  const auto h = dem_chunks_to_pcm(chunks, from_seam, to_seam);
  const auto o = dem_chunks_to_o_sparse(chunks);
  out.num_rows = h.num_rows();
  out.num_observables = o.size();
  out.rows = h.to_nested_csc();
  out.observables.resize(out.rows.size());
  for (std::size_t i = 0; i < o.size(); ++i)
    for (const auto c : o[i])
      out.observables[c].push_back(
          static_cast<sparse_binary_matrix::index_type>(i));
  for (const auto &chunk : chunks)
    out.priors.insert(out.priors.end(), chunk.error_rates.begin(),
                      chunk.error_rates.end());
  return out;
} // end - close_columns()

// Groups the window's fault columns, closed columns first + j for zone[j],
// into built columns. A class is a zone's columns with identical full rows
// in the closed chain, which holds every row they touch. most_likely merges
// a class whose observable effects differ, keeping the effect whose columns
// together are most likely; merge_identical merges the members with equal
// effects. Groups are ordered by their first column. Each lists first the
// column its corrections are reported against: its most likely, of the kept
// effect under most_likely; ties go to the lower index.
std::vector<std::vector<std::size_t>>
group_fault_columns(const chain_columns &closed, std::size_t first,
                    const std::vector<std::size_t> &zone,
                    const window_request::column_options &options) {
  using signature = std::pair<std::size_t, index_list>;
  std::map<signature, std::vector<std::size_t>> classes;
  std::vector<std::vector<std::size_t>> by_first(zone.size());
  std::vector<std::vector<std::size_t>> groups;
  const auto &rates = closed.priors;
  // Columns most likely first, ties by index; parts arrive in index order.
  const auto by_rate = [&](std::vector<std::size_t> part) {
    std::stable_sort(part.begin(), part.end(),
                     [&](std::size_t a, std::size_t b) {
                       return rates[first + a] > rates[first + b];
                     });
    return part;
  };
  // Without an option every column is its own group.
  if (!options.merge_identical &&
      options.conflicting_parallel == parallel_resolution::reject) {
    for (std::size_t j = 0; j < zone.size(); ++j)
      groups.push_back({j});
    return groups;
  }
  for (std::size_t j = 0; j < zone.size(); ++j)
    classes[{zone[j], closed.rows[first + j]}].push_back(j);
  for (const auto &[key, members] : classes) {
    std::map<index_list, std::vector<std::size_t>> by_effect;
    const std::vector<std::size_t> *kept = nullptr;
    double kept_rate = -1.0;
    std::vector<std::size_t> rest;
    for (const auto j : members)
      by_effect[closed.observables[first + j]].push_back(j);
    if (options.conflicting_parallel != parallel_resolution::most_likely ||
        by_effect.size() == 1) {
      for (const auto &[effect, part] : by_effect)
        if (options.merge_identical)
          by_first[part.front()] = by_rate(part);
        else
          for (const auto j : part)
            by_first[j] = {j};
      continue;
    }
    // An effect's rate is the chance that an odd number of its columns
    // fire; the most likely effect is kept, ties to the lower first index.
    for (const auto &[effect, part] : by_effect) {
      double rate = 0.0;
      for (const auto j : part)
        rate = rate + rates[first + j] - 2 * rate * rates[first + j];
      if (rate > kept_rate ||
          (rate == kept_rate && part.front() < kept->front())) {
        kept = &part;
        kept_rate = rate;
      }
    }
    for (const auto &[effect, part] : by_effect)
      if (&part != kept)
        rest.insert(rest.end(), part.begin(), part.end());
    std::sort(rest.begin(), rest.end());
    auto &group = by_first[members.front()];
    group = by_rate(*kept);
    for (const auto j : by_rate(rest))
      group.push_back(j);
  } // end - for(classes)
  for (auto &group : by_first)
    if (!group.empty())
      groups.push_back(std::move(group));
  return groups;
} // end - group_fault_columns()

// Each commit column's first parallel column, by index, that lets the
// decoder's choice between them change the commitment or the logical. A
// column touching no row of the window is never chosen, and a twin in the
// same zone with the same full rows and observable effect is no conflict.
// zone and the closed chain's columns are the unmerged fault columns, the
// window's first at closed column first.
std::vector<parallel_columns>
find_parallel_conflicts(const decoding_window &w, const chain_columns &closed,
                        std::size_t first,
                        const std::vector<std::size_t> &zone) {
  std::vector<parallel_columns> out;
  std::map<index_list, std::vector<std::size_t>> by_rows;
  const auto rows = w.model.h.to_nested_csc();
  const auto o = w.model.o.to_nested_csc();
  const auto &sources = w.fault_column_sources;
  for (std::size_t c = 0; c < rows.size(); ++c)
    if (!rows[c].empty())
      by_rows[rows[c]].push_back(c);
  for (auto c = w.commit_columns.begin; c < w.commit_columns.end; ++c) {
    if (rows[c].empty())
      continue;
    const auto source = sources[c].front();
    for (const auto other : by_rows.at(rows[c])) {
      std::optional<parallel_conflict> conflict;
      if (other >= w.num_fault_columns)
        conflict = parallel_conflict::boundary;
      else if (zone[sources[other].front()] != zone[source])
        conflict = parallel_conflict::other_zone;
      else if (closed.rows[first + sources[other].front()] !=
               closed.rows[first + source])
        conflict = parallel_conflict::outside_rows;
      else if (o[other] != o[c])
        conflict = parallel_conflict::observable;
      if (!conflict)
        continue;
      out.push_back({c, other, *conflict});
      break;
    } // end - for(other)
  } // end - for(c)
  return out;
} // end - find_parallel_conflicts()

// The measurement layout of a window's rounds: each round's width and where
// its measurements start among the map's inputs, after v_in.
struct measurement_layout {
  std::size_t v_in = 0;
  std::vector<std::size_t> width; ///< Per round of the window
  std::vector<std::size_t> start; ///< Per round, then the end
};

// Round r's measurements in the stream; without D_sparse rows the template
// gives one per detector, the memory convention of dem_chunks_to_d_sparse().
std::size_t round_width(const experiment_template &t, std::int64_t r,
                        std::size_t R) {
  return t.measurement_offset(r + 1, R) - t.measurement_offset(r, R);
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
    // The memory convention: the template gives every round the same
    // detectors, so prev_start + j stays in the previous round.
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

// Clips the zones, builds the ports, and places the sends, whose rows count
// from the window's first round.
decoding_window place_window(std::shared_ptr<const experiment_template> context,
                             const window_request &request, std::int64_t k,
                             std::size_t num_rounds) {
  decoding_window w;
  if (!context)
    throw std::invalid_argument("place_window: no template");
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
    throw std::invalid_argument("place_window: window " + std::to_string(k) +
                                " has no commit round in a " +
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
  if (w.input.exists && w.input.disposition == port_disposition::open &&
      w.input.boundary == input_boundary::columns)
    w.input.prior = *request.input.prior;
  if (w.output.exists && w.output.disposition == port_disposition::closed)
    w.output.imports = clip(request.output.imports.value_or(
        round_range{relative_end, relative_end + m_span}));
  const auto row = [&](std::int64_t r) {
    return t.row_offset(r, num_rounds) - t.row_offset(s, num_rounds);
  };

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

  return w;
} // end - place_window()

// Places the window, stitches its chunks and the imported ones, and slices
// the closed chain into the window's model and maps.
decoding_window
window_from_dem_chunks(std::shared_ptr<const experiment_template> context,
                       const window_request &request, std::int64_t k,
                       std::size_t num_rounds, bool with_m2o) {
  auto w = place_window(context, request, k, num_rounds);
  std::vector<extended_dem> chunks;
  const auto &t = *context;
  const auto R = static_cast<std::int64_t>(num_rounds);
  const auto m_span = static_cast<std::int64_t>(t.max_span());
  const auto s = w.rounds.begin;
  const auto e = w.rounds.end;
  const bool open_input =
      w.input.exists && w.input.disposition == port_disposition::open;
  const bool truncated =
      open_input && w.input.boundary == input_boundary::truncate;
  const bool boundary_columns = open_input && !truncated;

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
      close_columns(chunks, t.spec().seam.from_seam, t.spec().seam.to_seam);

  // The window's rows of the chain: own, then imports. window_row maps a
  // chain row to its window row, or to -1 outside the window; it keeps their
  // order, so mapped columns stay sorted.
  std::vector<std::size_t> rows;
  std::vector<std::int64_t> window_row(closed.num_rows, -1);
  w.num_own_rows = row(e);
  for (std::size_t i = 0; i < w.num_own_rows; ++i)
    rows.push_back(row_shift + i);
  if (!w.output.imports.empty())
    for (auto i = row(w.output.imports.begin); i < row(w.output.imports.end);
         ++i)
      rows.push_back(row_shift + i);
  for (std::size_t i = 0; i < rows.size(); ++i)
    window_row[rows[i]] = static_cast<std::int64_t>(i);
  const auto rows_in_window = [&](std::size_t column) {
    index_list out;
    for (const auto r : closed.rows[column])
      if (window_row[r] >= 0)
        out.push_back(
            static_cast<sparse_binary_matrix::index_type>(window_row[r]));
    return out;
  };

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
    if (!rows_in_window(c).empty())
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
  std::vector<index_list> h_columns(F + V + C);
  std::vector<index_list> o_columns(F + V + C + K);
  auto &priors = w.model.priors;
  priors.assign(F + V + C + K, w.input.prior);
  for (std::size_t g = 0; g < F; ++g) {
    const auto source = col_shift + groups[g].front();
    double prior = 0.0;
    h_columns[g] = rows_in_window(source);
    o_columns[g] = closed.observables[source];
    for (const auto j : groups[g]) {
      const auto p = closed.priors[col_shift + j];
      prior = prior + p - 2 * prior * p;
    }
    priors[g] = prior;
  } // end - for(g)
  for (std::size_t j = 0; j < V; ++j)
    h_columns[F + j] = {static_cast<sparse_binary_matrix::index_type>(j)};
  for (std::size_t x = 0; x < C; ++x) {
    h_columns[F + V + x] = rows_in_window(crossing[x]);
    o_columns[F + V + x] = closed.observables[crossing[x]];
    priors[F + V + x] = closed.priors[crossing[x]];
  }
  for (std::size_t o = 0; o < K; ++o)
    o_columns[F + V + C + o] = {
        static_cast<sparse_binary_matrix::index_type>(o)};
  w.model.h =
      sparse_binary_matrix::from_nested_csc(
          static_cast<sparse_binary_matrix::index_type>(rows.size()),
          static_cast<sparse_binary_matrix::index_type>(F + V + C), h_columns)
          .canonicalize();
  w.model.o =
      sparse_binary_matrix::from_nested_csc(
          static_cast<sparse_binary_matrix::index_type>(closed.num_observables),
          static_cast<sparse_binary_matrix::index_type>(F + V + C + K),
          o_columns)
          .canonicalize();

  w.parallel_conflicts = find_parallel_conflicts(w, closed, col_shift, zone);

  // Measurement maps.
  const auto layout = layout_measurements(t, w.rounds, num_rounds);
  // The maps index measurements with 32 bits; check before any is formed.
  if (layout.start.back() >
      std::numeric_limits<sparse_binary_matrix::index_type>::max())
    fail("window " + std::to_string(k) + " holds " +
         std::to_string(layout.start.back()) +
         " measurements, more than its maps can index (" +
         std::to_string(
             std::numeric_limits<sparse_binary_matrix::index_type>::max()) +
         ")");
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
      !same_matrix(a.model.h, b.model.h) ||
      !same_matrix(a.model.o, b.model.o) || a.model.priors != b.model.priors)
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

// H's columns are the leading ones; the logical-error columns after them
// stay empty.
detector_error_model dense_model(const window_model &model) {
  detector_error_model dense;
  const auto total = static_cast<std::size_t>(model.o.num_cols());
  const auto h = model.h.to_nested_csc();
  const auto o = model.o.to_nested_csc();
  dense.detector_error_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{model.h.num_rows(), total});
  dense.observables_flips_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{model.o.num_rows(), total});
  const auto dense_h = view(dense.detector_error_matrix);
  const auto dense_o = view(dense.observables_flips_matrix);
  for (std::size_t c = 0; c < h.size(); ++c)
    for (const auto r : h[c])
      dense_h(r, c) = 1;
  for (std::size_t c = 0; c < o.size(); ++c)
    for (const auto r : o[c])
      dense_o(r, c) = 1;
  dense.error_rates = model.priors;
  return dense;
} // end - dense_model()

// Each send's values are sized by its rows, so commit_window() writes every
// one in place.
window_workspace make_window_workspace(const decoding_window &window) {
  window_workspace ws;
  const auto rows = window.model.h.num_rows();
  const auto columns = window.model.o.num_cols();
  auto &out = ws.solution.outputs;
  ws.syndrome.assign(rows, 0);
  ws.explained.assign(rows, 0);
  ws.delta.assign(rows, 0);
  ws.solution.estimate.assign(columns, 0);
  out.committed.assign(columns, 0);
  out.frame.assign(window.model.o.num_rows(), 0);
  for (const auto &send : window.sends)
    out.sends.emplace_back(send.rows.size(), 0);
  return ws;
} // end - make_window_workspace()

namespace {

// Throws std::invalid_argument, for @p what, unless @p ws is sized for
// @p window. Takes a C string so that a passing check allocates nothing.
void check_workspace(const decoding_window &window, const window_workspace &ws,
                     const char *what) {
  const auto rows = window.model.h.num_rows();
  const auto columns = window.model.o.num_cols();
  const auto &out = ws.solution.outputs;
  bool fits = ws.syndrome.size() == rows && ws.explained.size() == rows &&
              ws.delta.size() == rows &&
              ws.solution.estimate.size() == columns &&
              out.committed.size() == columns &&
              out.frame.size() == window.model.o.num_rows() &&
              out.sends.size() == window.sends.size();
  for (std::size_t i = 0; fits && i < window.sends.size(); ++i)
    fits = out.sends[i].size() == window.sends[i].rows.size();
  if (!fits)
    throw std::invalid_argument(std::string(what) +
                                ": the workspace is not sized for window " +
                                std::to_string(window.k));
} // end - check_workspace()

} // namespace

// e' masks every column outside the commit zone. The committed columns'
// rows give each send's delta, to which a residual adds the window's
// detectors, and their observables give the frame update. Every buffer is
// the workspace's, cleared before use.
void commit_window(const decoding_window &window,
                   const std::vector<std::uint8_t> &detectors,
                   window_workspace &ws) {
  const auto &h = window.model.h;
  const auto &o = window.model.o;
  const auto commit = window.commit_columns;
  const auto &estimate = ws.solution.estimate;
  auto &out = ws.solution.outputs;
  if (detectors.size() != h.num_rows())
    throw std::invalid_argument(
        "commit_window: expected " + std::to_string(h.num_rows()) +
        " detectors, got " + std::to_string(detectors.size()));
  check_workspace(window, ws, "commit_window");
  std::fill(ws.delta.begin(), ws.delta.end(), 0);
  std::fill(out.committed.begin(), out.committed.end(), 0);
  std::fill(out.frame.begin(), out.frame.end(), 0);
  for (auto c = commit.begin; c < commit.end; ++c) {
    out.committed[c] = estimate[c] & 1;
    if (!out.committed[c])
      continue;
    for (auto p = h.ptr()[c]; p < h.ptr()[c + 1]; ++p)
      ws.delta[h.indices()[p]] ^= 1;
    for (auto p = o.ptr()[c]; p < o.ptr()[c + 1]; ++p)
      out.frame[o.indices()[p]] ^= 1;
  }
  for (std::size_t i = 0; i < window.sends.size(); ++i) {
    const auto &send = window.sends[i];
    auto &values = out.sends[i];
    for (std::size_t j = 0; j < send.rows.size(); ++j) {
      const auto r = send.rows[j];
      values[j] = send.mode == send_mode::residual
                      ? static_cast<std::uint8_t>(detectors[r] ^ ws.delta[r])
                      : ws.delta[r];
    }
  }
} // end - commit_window()

// The estimate goes in a workspace of its own.
window_outputs commit_window(const decoding_window &window,
                             const std::vector<std::uint8_t> &detectors,
                             const std::vector<std::uint8_t> &estimate) {
  auto ws = make_window_workspace(window);
  if (detectors.size() != window.model.h.num_rows() ||
      estimate.size() != ws.solution.estimate.size())
    throw std::invalid_argument(
        "commit_window: expected " + std::to_string(window.model.h.num_rows()) +
        " detectors and " + std::to_string(ws.solution.estimate.size()) +
        " estimates, got " + std::to_string(detectors.size()) + " and " +
        std::to_string(estimate.size()));
  ws.solution.estimate = estimate;
  commit_window(window, detectors, ws);
  return std::move(ws.solution.outputs);
} // end - commit_window()

// Rows are matched through the shot: the sender's rows count from its first
// round, the receiver's own rows from its own, and its imported rows follow
// them, counted from its first imported round. The first pass checks every
// row and the second applies them, so no list of rows is kept.
void receive_send(const decoding_window &receiver,
                  const decoding_window &sender, std::size_t send,
                  const std::vector<std::uint8_t> &values,
                  std::vector<std::uint8_t> &detectors) {
  // No receiver row: larger than any row index.
  constexpr auto none = std::numeric_limits<std::size_t>::max();
  const auto reject = [](const std::string &what) {
    throw std::invalid_argument("receive_send: " + what);
  };
  if (!receiver.context || receiver.context != sender.context ||
      receiver.shot_rounds != sender.shot_rounds)
    reject("the windows are not of one shot on one template");
  if (send >= sender.sends.size())
    reject("window " + std::to_string(sender.k) + " has no send " +
           std::to_string(send));
  const auto &s = sender.sends[send];
  if (values.size() != s.rows.size() ||
      detectors.size() != receiver.model.h.num_rows())
    reject("send '" + s.id + "' has " + std::to_string(s.rows.size()) +
           " rows and its receiver " +
           std::to_string(receiver.model.h.num_rows()) + ", given " +
           std::to_string(values.size()) + " values and " +
           std::to_string(detectors.size()) + " detectors");
  const auto &t = *receiver.context;
  const auto R = receiver.shot_rounds;
  const auto imports = receiver.output.imports;
  const auto sender_first = t.row_offset(sender.rounds.begin, R);
  const auto own_first = t.row_offset(receiver.rounds.begin, R);
  const auto import_first =
      imports.empty() ? 0 : t.row_offset(imports.begin, R);
  const auto import_end = imports.empty() ? 0 : t.row_offset(imports.end, R);
  // The receiver's row for row r of the send, or none if the receiver does
  // not take it in the send's mode.
  const auto receiver_row = [&](std::size_t r) -> std::size_t {
    const auto row = sender_first + r;
    if (s.mode == send_mode::delta && row >= own_first &&
        row - own_first < receiver.num_own_rows)
      return row - own_first;
    if (s.mode == send_mode::residual && row >= import_first &&
        row < import_end)
      return receiver.num_own_rows + row - import_first;
    return none;
  };
  for (const auto r : s.rows)
    if (receiver_row(r) == none)
      reject("send '" + s.id + "' of window " + std::to_string(sender.k) +
             " reaches shot row " + std::to_string(sender_first + r) +
             ", which window " + std::to_string(receiver.k) +
             " does not take as a " +
             (s.mode == send_mode::delta ? "delta" : "residual"));
  for (std::size_t i = 0; i < s.rows.size(); ++i)
    if (s.mode == send_mode::delta)
      detectors[receiver_row(s.rows[i])] ^= values[i] & 1;
    else
      detectors[receiver_row(s.rows[i])] = values[i] & 1;
} // end - receive_send()

// H keeps only the decoder's columns, so its priors are the leading ones.
// dem_chunks carry no error_ids, so there are none to pass. The decoder is
// checked as the memory window checks its own: it must return fault
// columns, and take the window's rows and return its decoder columns.
std::unique_ptr<decoder>
make_window_decoder(const decoding_window &window, const std::string &name,
                    const cudaqx::heterogeneous_map &params) {
  const auto columns = window.num_decoder_columns();
  const auto rows = window.model.h.num_rows();
  const auto &priors = window.model.priors;
  std::unique_ptr<decoder> dec;
  if (rows == 0 || columns == 0)
    return nullptr;
  dec = get_decoder(name,
                    decoder_init(window.model.h, std::nullopt,
                                 std::vector<double>(priors.begin(),
                                                     priors.begin() + columns)),
                    decoder_output_request{decode_result_type::errors}, params);
  if (dec->get_result_type() != decode_result_type::errors)
    throw std::invalid_argument("make_window_decoder: decoder '" + name +
                                "' returns observables; a window needs its "
                                "fault columns");
  if (dec->get_syndrome_size() != rows || dec->get_block_size() != columns)
    throw std::invalid_argument(
        "make_window_decoder: decoder '" + name + "' takes " +
        std::to_string(dec->get_syndrome_size()) + " detectors and returns " +
        std::to_string(dec->get_block_size()) + " columns, but window " +
        std::to_string(window.k) + " has " + std::to_string(rows) +
        " rows and " + std::to_string(columns) + " decoder columns");
  return dec;
} // end - make_window_decoder()

// The decoder's soft result is made hard. H e is formed from H's columns,
// which are the decoder's, and compared with the detectors. The decoder's
// input, H e and the solution are the workspace's, cleared before use.
void solve_window(const decoding_window &window, decoder *dec,
                  const std::vector<std::uint8_t> &detectors,
                  window_workspace &ws) {
  const auto &h = window.model.h;
  auto &out = ws.solution;
  if (detectors.size() != h.num_rows())
    throw std::invalid_argument(
        "solve_window: expected " + std::to_string(h.num_rows()) +
        " detectors, got " + std::to_string(detectors.size()));
  check_workspace(window, ws, "solve_window");
  std::fill(out.estimate.begin(), out.estimate.end(), 0);
  std::fill(ws.explained.begin(), ws.explained.end(), 0);
  out.converged = true;
  out.satisfied = true;
  if (dec) {
    std::copy(detectors.begin(), detectors.end(), ws.syndrome.begin());
    const auto result = dec->decode(ws.syndrome);
    if (result.result.size() != h.num_cols())
      throw std::invalid_argument("solve_window: the decoder returned " +
                                  std::to_string(result.result.size()) +
                                  " columns for a window of " +
                                  std::to_string(h.num_cols()));
    out.converged = result.converged;
    for (std::size_t c = 0; c < h.num_cols(); ++c)
      out.estimate[c] = convert_soft_to_hard(result.result[c]);
  }
  for (std::size_t c = 0; c < h.num_cols(); ++c)
    if (out.estimate[c])
      for (auto p = h.ptr()[c]; p < h.ptr()[c + 1]; ++p)
        ws.explained[h.indices()[p]] ^= 1;
  for (std::size_t r = 0; r < ws.explained.size(); ++r)
    out.satisfied = out.satisfied && ws.explained[r] == (detectors[r] & 1);
  commit_window(window, detectors, ws);
} // end - solve_window()

} // namespace cudaq::qec
