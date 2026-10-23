/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Serial parallel-window reference. The window rules are applied directly to
// the whole shot's rounds, rows and columns: zones clipped to the shot,
// ports at the window's edges unless the shot ends there, imports and sends
// as shot rows, boundary or truncated columns at an open input. Decoding runs
// the canonical workload window by window, producers first.

#include "parallel_window_reference.h"
#include "cudaq/qec/decoder_init.h"
#include "cudaq/qec/extended_dem.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <utility>

namespace cudaq::qec::pw_reference {
namespace {

using experimental::input_boundary;
using experimental::port_disposition;
using experimental::round_range;
using experimental::send_mode;
using experimental::window_request;

// Whether a sorted list holds x.
bool holds(const std::vector<std::uint32_t> &sorted, std::size_t x) {
  return std::binary_search(sorted.begin(), sorted.end(),
                            static_cast<std::uint32_t>(x));
}

// The round holding shot row r.
std::int64_t round_of_row(const shot &s, std::size_t r) {
  return std::upper_bound(s.row.begin(), s.row.end(), r) - s.row.begin() - 1;
}

// Window k of request clipped to an R-round shot: [anchor + z) within
// [0, R).
round_range clip(const window_request &request, std::int64_t k, std::int64_t R,
                 round_range z) {
  const auto anchor = request.offset + k * request.period;
  return {std::max<std::int64_t>(0, anchor + z.begin),
          std::min<std::int64_t>(R, anchor + z.end)};
}

// A window exists while its clipped commit zone holds a round.
bool exists(const window_request &request, std::int64_t k, std::int64_t R) {
  return !clip(request, k, R, *request.commit).empty();
}

// The index of shot row r among the window's rows, or -1.
std::int64_t local_row(const reference_window &w, std::size_t r) {
  const auto at = std::lower_bound(w.rows.begin(), w.rows.end(), r);
  return at != w.rows.end() && *at == r ? at - w.rows.begin() : -1;
}

// The window's rounds as shot rows.
void add_rows(const shot &s, round_range rounds,
              std::vector<std::size_t> &rows) {
  for (auto r = s.row[rounds.begin]; r < s.row[rounds.end]; ++r)
    rows.push_back(r);
}

// Applies the window rules to window k of a layer: zones, ports, imports,
// sends and dependencies, then its rows and decoder columns.
reference_window slice(const shot &s, const window_graph &graph,
                       const window_layer &layer, std::int64_t k,
                       std::int64_t R) {
  reference_window w;
  const auto &q = layer.request;
  const auto M = s.max_span;
  std::int64_t relative_end = q.commit->end;
  w.node = layer.node;
  w.k = k;
  w.commit = clip(q, k, R, *q.commit);
  w.rounds = w.commit;
  for (const auto &buffer : q.buffers) {
    const auto clipped = clip(q, k, R, buffer);
    relative_end = std::max(relative_end, buffer.end);
    if (clipped.empty())
      continue;
    w.buffers.push_back(clipped);
    w.rounds = {std::min(w.rounds.begin, clipped.begin),
                std::max(w.rounds.end, clipped.end)};
  }
  std::sort(w.buffers.begin(), w.buffers.end(),
            [](round_range a, round_range b) { return a.begin < b.begin; });
  w.input_exists = w.rounds.begin > 0;
  w.output_exists = w.rounds.end < R;
  if (w.output_exists && *q.output.disposition == port_disposition::closed)
    w.imports = clip(
        q, k, R,
        q.output.imports.value_or(round_range{relative_end, relative_end + M}));

  // Sends: a default forward covers the trailing buffer's first M rounds,
  // a default backward the commit zone's, as a residual.
  for (const auto &send : q.sends) {
    reference_send out;
    auto rounds = send.rounds;
    out.id = send.id;
    out.mode = send.mode.value_or(send.id == "backward" ? send_mode::residual
                                                        : send_mode::delta);
    if (!send.mode && send.rounds.empty()) {
      auto first = q.commit->begin;
      if (send.id == "forward")
        first = q.commit->end;
      rounds = {{first, first + M}};
    }
    std::sort(rounds.begin(), rounds.end(),
              [](round_range a, round_range b) { return a.begin < b.begin; });
    for (const auto &relative : rounds) {
      const auto clipped = clip(q, k, R, relative);
      if (clipped.empty())
        continue;
      out.rounds.push_back(clipped);
      add_rows(s, clipped, out.rows);
    }
    w.sends.push_back(std::move(out));
  } // end - for(send)

  // Each incoming edge reads producer instance k + index_offset.
  for (const auto &e : graph.edges) {
    if (e.to_node != layer.node)
      continue;
    const auto producer = std::find_if(
        graph.layers.begin(), graph.layers.end(),
        [&](const window_layer &l) { return l.node == e.from_node; });
    const auto j = k + e.index_offset;
    w.inputs.push_back({e.from_node, e.from_output, j, e.to_input,
                        exists(producer->request, j, R)});
  }

  // Rows: own rounds, then imports. Columns: the window's own faults, then a
  // truncated input's crossing faults; boundary columns follow those.
  add_rows(s, w.rounds, w.rows);
  w.num_own_rows = w.rows.size();
  if (!w.imports.empty())
    add_rows(s, w.imports, w.rows);
  for (auto c = s.column[w.rounds.begin]; c < s.column[w.rounds.end]; ++c) {
    const auto round = std::upper_bound(s.column.begin(), s.column.end(), c) -
                       s.column.begin() - 1;
    w.columns.push_back(c);
    w.committed.push_back(round >= w.commit.begin && round < w.commit.end);
  }
  w.num_faults = w.columns.size();
  const bool open_input =
      w.input_exists && *q.input.disposition == port_disposition::open;
  if (open_input && q.input.boundary == input_boundary::truncate) {
    for (auto c = s.column[w.rounds.begin - 1]; c < s.column[w.rounds.begin];
         ++c)
      if (std::any_of(s.h[c].begin(), s.h[c].end(),
                      [&](std::uint32_t r) { return local_row(w, r) >= 0; }))
        w.columns.push_back(c);
  } else if (open_input) {
    w.num_virtual = s.row[w.rounds.begin + 1] - s.row[w.rounds.begin];
  }
  return w;
} // end - slice()

} // namespace

// Round r's rows are chunk r's to_seam band and interior rows, in the order
// dem_close_all() writes them.
shot whole_shot(dem_chunks_spec spec, std::size_t rounds) {
  shot s;
  const auto to = spec.seam.to_seam;
  spec.num_rounds = rounds;
  const auto chunks = dem_chunks_from_spec(spec);
  s.dem = dem_close_all(chunks, spec.seam.from_seam, to);
  s.h = sparse_binary_matrix(s.dem.detector_error_matrix).to_nested_csc();
  s.o = sparse_binary_matrix(s.dem.observables_flips_matrix).to_nested_csc();
  s.column = dem_chunks_to_fault_offsets(chunks);
  s.row = {0};
  for (const auto &chunk : chunks)
    s.row.push_back(s.row.back() +
                    (chunk.has_seam(to) ? chunk.get_seam(to).num_rows() : 0) +
                    chunk.num_interior_rows());
  if (s.row.back() != s.dem.detector_error_matrix.shape()[0])
    throw std::logic_error("whole_shot: rows do not add up");
  for (std::size_t r = 0; r < rounds; ++r)
    for (auto c = s.column[r]; c < s.column[r + 1]; ++c)
      if (!s.h[c].empty())
        s.max_span = std::max(s.max_span, round_of_row(s, s.h[c].back()) -
                                              static_cast<std::int64_t>(r));
  return s;
} // end - whole_shot()

// XOR of each set fault's rows.
std::vector<std::uint8_t> syndrome_of(const shot &s,
                                      const std::vector<std::uint8_t> &e) {
  std::vector<std::uint8_t> detectors(s.row.back(), 0);
  for (std::size_t c = 0; c < e.size(); ++c)
    if (e[c])
      for (const auto r : s.h[c])
        detectors[r] ^= 1;
  return detectors;
}

// XOR of each set fault's observables.
std::vector<std::uint8_t> observables_of(const shot &s,
                                         const std::vector<std::uint8_t> &e) {
  std::vector<std::uint8_t> flipped(s.dem.observables_flips_matrix.shape()[0],
                                    0);
  for (std::size_t c = 0; c < e.size(); ++c)
    if (e[c])
      for (const auto o : s.o[c])
        flipped[o] ^= 1;
  return flipped;
}

// Asks for fault columns, which the windows commit and send.
std::unique_ptr<decoder>
make_decoder(const std::vector<std::vector<std::uint32_t>> &h,
             std::size_t num_rows, std::vector<double> priors,
             const std::string &name, const cudaqx::heterogeneous_map &params) {
  const auto matrix = sparse_binary_matrix::from_nested_csc(
      static_cast<std::uint32_t>(num_rows),
      static_cast<std::uint32_t>(h.size()), h);
  return get_decoder(
      name, decoder_init(matrix, std::nullopt, std::move(priors)),
      decoder_output_request{decode_result_type::errors}, params);
}

// Slices every window, builds its decoder over its columns restricted to its
// rows, then orders the windows so every producer comes first.
parallel_window_reference::parallel_window_reference(
    const dem_chunks_spec &spec, const window_graph &graph, std::size_t rounds,
    const std::string &decoder_name, const cudaqx::heterogeneous_map &params)
    : shot_(whole_shot(spec, rounds)) {
  std::map<std::pair<std::string, std::int64_t>, std::size_t> index;
  std::vector<std::vector<std::size_t>> consumers;
  std::vector<std::size_t> waiting;
  std::vector<std::size_t> ready;
  const auto R = static_cast<std::int64_t>(rounds);
  for (const auto &layer : graph.layers) {
    const auto &q = layer.request;
    if (q.columns.merge_identical ||
        q.columns.conflicting_parallel !=
            experimental::parallel_resolution::reject)
      throw std::invalid_argument(
          "parallel_window_reference: column options are not modelled");
    // Every k whose anchor puts the commit zone within reach of the shot.
    const auto reach = R + std::abs(q.offset) + std::abs(q.commit->begin) +
                       std::abs(q.commit->end) + 1;
    for (auto k = -reach; k <= reach; ++k)
      if (exists(q, k, R))
        windows_.push_back(slice(shot_, graph, layer, k, R));
  }

  for (std::size_t i = 0; i < windows_.size(); ++i) {
    const auto &w = windows_[i];
    const auto &q =
        std::find_if(graph.layers.begin(), graph.layers.end(),
                     [&](const window_layer &l) { return l.node == w.node; })
            ->request;
    auto &h = h_.emplace_back();
    std::vector<double> priors;
    index[{w.node, w.k}] = i;
    for (const auto c : w.columns) {
      auto &rows = h.emplace_back();
      for (const auto r : shot_.h[c])
        if (const auto local = local_row(w, r); local >= 0)
          rows.push_back(static_cast<std::uint32_t>(local));
      priors.push_back(shot_.dem.error_rates[c]);
    }
    for (std::size_t j = 0; j < w.num_virtual; ++j) {
      h.push_back({static_cast<std::uint32_t>(j)});
      priors.push_back(*q.input.prior);
    }
    decoders_.push_back(nullptr);
    if (!h.empty() && !w.rows.empty())
      decoders_.back() = make_decoder(h, w.rows.size(), std::move(priors),
                                      decoder_name, params);
  } // end - for(i)

  consumers.resize(windows_.size());
  waiting.assign(windows_.size(), 0);
  for (std::size_t i = 0; i < windows_.size(); ++i)
    for (const auto &input : windows_[i].inputs)
      if (input.exists) {
        consumers[index.at({input.from_node, input.k})].push_back(i);
        ++waiting[i];
      }
  for (std::size_t i = 0; i < windows_.size(); ++i)
    if (waiting[i] == 0)
      ready.push_back(i);
  while (!ready.empty()) {
    const auto i = ready.back();
    ready.pop_back();
    order_.push_back(i);
    for (const auto c : consumers[i])
      if (--waiting[c] == 0)
        ready.push_back(c);
  }
  if (order_.size() != windows_.size())
    throw std::invalid_argument(
        "parallel_window_reference: the dependencies form a cycle");
} // end - parallel_window_reference::parallel_window_reference()

// The canonical workload: d from the shot, deltas XORed into own rows and
// residuals written over imports; e decoded; e' masked to the commit zone;
// each send H[send rows] e', plus d for a residual; l = O e'.
result
parallel_window_reference::decode(const std::vector<std::uint8_t> &detectors,
                                  bool sends) {
  result r;
  const auto K = shot_.dem.observables_flips_matrix.shape()[0];
  r.errors.assign(shot_.h.size(), 0);
  r.frame.assign(K, 0);
  r.windows.resize(windows_.size());
  for (const auto i : order_) {
    const auto &w = windows_[i];
    const auto &h = h_[i];
    auto &out = r.windows[i];
    std::vector<std::uint8_t> explained(w.rows.size(), 0);
    for (const auto row : w.rows)
      out.detectors.push_back(detectors[row]);
    for (const auto &input : w.inputs) {
      if (!sends || !input.exists)
        continue;
      const auto p =
          std::find_if(windows_.begin(), windows_.end(),
                       [&](const reference_window &x) {
                         return x.node == input.from_node && x.k == input.k;
                       }) -
          windows_.begin();
      const auto &from = windows_[p];
      for (std::size_t si = 0; si < from.sends.size(); ++si) {
        const auto &send = from.sends[si];
        if (send.id != input.from_output)
          continue;
        for (std::size_t j = 0; j < send.rows.size(); ++j) {
          const auto local = local_row(w, send.rows[j]);
          const bool own =
              local >= 0 && static_cast<std::size_t>(local) < w.num_own_rows;
          const bool imported =
              local >= 0 && static_cast<std::size_t>(local) >= w.num_own_rows;
          const auto value = r.windows[p].sends[si][j];
          if (send.mode == send_mode::delta && own)
            out.detectors[local] ^= value;
          else if (send.mode == send_mode::residual && imported)
            out.detectors[local] = value;
          else
            throw std::logic_error("parallel_window_reference: send '" +
                                   send.id + "' reaches a row " + w.node +
                                   " cannot take it");
        }
      } // end - for(si)
    } // end - for(input)

    out.estimate.assign(h.size(), 0);
    if (decoders_[i]) {
      const auto decoded = decoders_[i]->decode(
          std::vector<float_t>(out.detectors.begin(), out.detectors.end()));
      out.converged = decoded.converged;
      for (std::size_t j = 0; j < h.size(); ++j)
        out.estimate[j] = convert_soft_to_hard(decoded.result[j]);
    }
    for (std::size_t j = 0; j < h.size(); ++j)
      if (out.estimate[j])
        for (const auto row : h[j])
          explained[row] ^= 1;
    out.satisfied = explained == out.detectors;
    for (std::size_t j = 0; j < w.num_faults; ++j)
      if (w.committed[j] && out.estimate[j]) {
        out.committed.push_back(w.columns[j]);
        r.errors[w.columns[j]] ^= 1;
      }

    for (const auto &send : w.sends) {
      auto &values = out.sends.emplace_back();
      for (const auto row : send.rows) {
        std::uint8_t delta = 0;
        for (const auto c : out.committed)
          delta ^= holds(shot_.h[c], row);
        if (send.mode == send_mode::residual)
          delta ^= out.detectors[local_row(w, row)];
        values.push_back(delta);
      }
    }
    out.frame.assign(K, 0);
    for (const auto c : out.committed)
      for (const auto o : shot_.o[c])
        out.frame[o] ^= 1;
    for (std::size_t o = 0; o < K; ++o)
      r.frame[o] ^= out.frame[o];
    r.converged = r.converged && out.converged;
    r.satisfied = r.satisfied && out.satisfied;
  } // end - for(i)
  r.observables = observables_of(shot_, r.errors);
  return r;
} // end - parallel_window_reference::decode()

} // namespace cudaq::qec::pw_reference
