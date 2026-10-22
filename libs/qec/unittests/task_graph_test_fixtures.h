/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Test fixtures shared by the DEM, task-graph and window unit tests: small
// dem_chunks_spec memory experiments with a repeating phase, so a test can
// expand them to any number of rounds.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <cstdint>
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

} // namespace cudaq::qec
