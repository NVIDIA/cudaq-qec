/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// A dynamic DEM of a distance-5 repetition-code memory experiment,
// the model of yaml/task_graph_parallel_window.yaml built in C++.
//
// Compile and run with:
//   nvq++ -lcudaq-qec -lcudaq-qec-decoders dyn_dem.cpp
//   ./a.out

#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/extended_dem.h"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using namespace cudaq::qec;

// [Begin Documentation]
// A phase whose faults flip the rows each seam lists, and whose detectors read
// the measurements D_sparse lists; observable 0 reads data qubit 0.
static phase_spec_entry
phase(const char *name, std::uint64_t num_faults,
      std::vector<std::pair<seam_id, std::vector<std::int64_t>>> seams,
      std::vector<std::int64_t> D_sparse) {
  phase_spec_entry entry;
  dem_chunk_spec spec;
  spec.num_faults = num_faults;
  for (auto &[seam, rows] : seams) {
    seam_spec_entry seam_entry;
    seam_entry.id = seam;
    seam_entry.spec.H_sparse = std::move(rows);
    spec.seam_specs.push_back(std::move(seam_entry));
  }
  spec.D_sparse = std::move(D_sparse);
  spec.O_sparse = {0, -1};
  spec.error_rates.assign(num_faults, 0.02);
  entry.id = seam_id{name};
  entry.spec = std::move(spec);
  return entry;
}

int main() {
  // Faults 0-4 flip data qubits 0-4; faults 5-8 are errors in measuring checks
  // 0-3. Check i reads data qubits i and i + 1. The final round reads the
  // data qubits.
  const std::vector<std::pair<seam_id, std::vector<std::int64_t>>> checks{
      {seam_name::prev_round,
       {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1}},
      {seam_name::next_round, {5, -1, 6, -1, 7, -1, 8, -1}}};
  dem_chunks_spec model;
  model.seam = {seam_name::next_round, seam_name::prev_round};
  model.measurements_per_round = 4; // one per check
  model.phases = {
      phase("init", 9, checks, {0, -1, 1, -1, 2, -1, 3, -1}),
      phase("bulk", 9, checks, {0, 4, -1, 1, 5, -1, 2, 6, -1, 3, 7, -1}),
      phase("final", 5,
            {{seam_name::prev_round, {0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1}}},
            {0, 4, 5, -1, 1, 5, 6, -1, 2, 6, 7, -1, 3, 7, 8, -1}),
  };
  model.phases[2].spec.num_measurements = 5; // data qubits 0-4
  model.phases[2].spec.m2o_sparse = {0, -1}; // observable 0 reads qubit 0
  model.connections = {{seam_id{"init"}, seam_id{"bulk"}},
                       {seam_id{"bulk"}, seam_id{"bulk"}},
                       {seam_id{"bulk"}, seam_id{"final"}}};

  // Fix the shot length to decode complete shots: 5 rounds of checks and the
  // final readout.
  model.num_rounds = 6;
  auto graph = decoding_task_graph::from_spec(model, "pymatching");

  // Data qubit 0 flips in round 2: from then on, check 0 reads 1, and the
  // final readout reads it as 1.
  measurement_results shot;
  for (std::size_t r = 0; r < 5; ++r)
    for (std::size_t i = 0; i < 4; ++i)
      shot.bits.push_back(r >= 2 && i == 0);
  shot.bits.insert(shot.bits.end(), {1, 0, 0, 0, 0});
  const auto corrections = graph->run(shot).bits;
  std::cout << "corrections: [" << int(corrections.at(0)) << "]\n";
  // [End Documentation]
  return corrections.at(0) == 1 ? 0 : 1;
}
