/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Test fixtures shared by the DEM, task-graph and window unit tests: small
// dem_chunks_spec memory experiments with a repeating phase, so a test can
// expand them to any number of rounds, and the window requests of the
// parallel-window and sliding-window layouts.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include "cudaq/qec/window_request.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

/// Detectors per round of the rep-5 fixtures.
constexpr uint32_t rep5_checks = 4;

/// Rep-5 memory experiment as init, bulk and final phases. Fault columns
/// 0..4 are the five data qubits and 5..8 the four measurement errors; the
/// final phase measures destructively, so it has only the five data columns.
inline dem_chunks_spec rep5_spec(uint64_t num_rounds = 5) {
  dem_chunks_spec spec;
  const auto make = [](uint64_t nf, std::vector<std::int64_t> h,
                       std::vector<std::int64_t> o, double rate) {
    dem_chunk_spec s;
    s.num_faults = nf;
    s.H_sparse = std::move(h);
    s.O_sparse = std::move(o);
    s.error_rates.assign(nf, rate);
    return s;
  };
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_final}};
  spec.num_rounds = num_rounds;
  spec.phases = {
      {phase_name::dem_init,
       make(9, {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1}, {0, -1},
            0.02)},
      {phase_name::dem_bulk,
       make(9, {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1}, {0, -1},
            0.02)},
      {phase_name::dem_final,
       make(5, {0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1}, {0, -1}, 0.02)}};
  return spec;
} // end - rep5_spec()

/// Four timelike faults per round, one per detector; fault 0 flips
/// @p observable. With @p explicit_d, every phase carries D_sparse over 4
/// measurements a round.
inline dem_chunks_spec arbitrary_memory_spec(std::size_t rounds,
                                             bool explicit_d = false,
                                             std::size_t observable = 0) {
  auto spec = rep5_spec(rounds);
  for (auto &phase : spec.phases) {
    phase.spec.num_faults = 4;
    phase.spec.O_sparse = {static_cast<std::int64_t>(observable), -1};
    phase.spec.H_sparse = {0, -1, 1, -1, 2, -1, 3, -1};
    phase.spec.error_rates.assign(4, 0.02);
    if (explicit_d) {
      phase.spec.num_measurements = 4;
      for (std::int64_t k = 0; k < 4; ++k) {
        if (phase.id != phase_name::dem_init)
          phase.spec.D_sparse.push_back(k);
        phase.spec.D_sparse.push_back(phase.id == phase_name::dem_init ? k
                                                                       : 4 + k);
        phase.spec.D_sparse.push_back(-1);
      }
    }
  } // end - for(phase)
  return spec;
} // end - arbitrary_memory_spec()

/// arbitrary_memory_spec with explicit D and @p tail_count suffix phases.
inline dem_chunks_spec multi_terminal_memory_spec(std::size_t rounds,
                                                  std::size_t tail_count) {
  auto spec = arbitrary_memory_spec(rounds, true);
  auto previous = spec.repeating_phase();
  const auto final = spec.phases.back().id;
  for (std::size_t i = 0; i + 1 < tail_count; ++i) {
    const auto id = phase_id::from_name("terminal" + std::to_string(i));
    auto phase = spec.phases.back().spec;
    // Different observable ownership makes treating this as bulk detectable.
    phase.O_sparse = {1, -1};
    phase.error_rates.assign(4, 0.03);
    spec.phases.push_back({id, std::move(phase)});
    for (auto &connection : spec.connections)
      if (connection.from_phase == previous && connection.to_phase == final)
        connection.to_phase = id;
    spec.connections.push_back({id, final});
    previous = id;
  } // end - for(i)
  return spec;
} // end - multi_terminal_memory_spec()

/// Every fault touches its own round only: the outgoing seam band is empty.
inline dem_chunks_spec single_round_spec() {
  auto spec = arbitrary_memory_spec(5);
  for (auto &phase : spec.phases) {
    phase.spec.seam_specs = {{seam_name::prev_round, {phase.spec.H_sparse, {}}},
                             {seam_name::next_round, {{-1, -1, -1, -1}, {}}}};
    phase.spec.H_sparse.clear();
  }
  return spec;
}

/// The parallel-window layers: A[k] buffers b rounds on each side of c_a
/// commit rounds, and B[k] is the c_b-round gap before A[k].
struct layered_requests {
  experimental::window_request a;
  experimental::window_request b;
};

/// A and B windows for commit c_a, buffer b and gap c_b.
inline layered_requests layered(std::int64_t c_a, std::int64_t b,
                                std::int64_t c_b) {
  using experimental::port_disposition;
  layered_requests out;
  const std::int64_t period = c_a + c_b;
  out.a.period = period;
  out.a.commit = experimental::round_range{0, c_a};
  out.a.buffers = {{-b, 0}, {c_a, c_a + b}};
  out.a.input = {port_disposition::open, 0.01};
  out.a.output = {port_disposition::open, std::nullopt};
  out.a.sends = {{"forward", std::nullopt, {}}, {"backward", std::nullopt, {}}};
  out.b.period = period;
  out.b.offset = -c_b;
  out.b.commit = experimental::round_range{0, c_b};
  out.b.input = {port_disposition::closed, std::nullopt};
  out.b.output = {port_disposition::closed, std::nullopt};
  return out;
} // end - layered()

/// A sequential sliding window: commit step rounds, then look ahead.
inline experimental::window_request sliding(std::int64_t step,
                                            std::int64_t size) {
  using experimental::port_disposition;
  experimental::window_request w;
  w.period = step;
  w.commit = experimental::round_range{0, step};
  w.buffers = {{step, size}};
  w.input = {port_disposition::closed, std::nullopt};
  w.output = {port_disposition::open, std::nullopt};
  w.sends = {{"forward", std::nullopt, {}}};
  return w;
}

} // namespace cudaq::qec
