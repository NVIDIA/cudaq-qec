/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Parallel-window decoding of a distance-5 repetition-code memory experiment,
// the model of yaml/task_graph_parallel_window.yaml built in C++.
//
// Compile and run with:
//   nvq++ -lcudaq-qec -lcudaq-qec-decoders task_graph_parallel_window.cpp
//   ./a.out

#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/extended_dem.h"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using namespace cudaq::qec;

// A phase whose faults flip the rows each seam lists; observable 0 reads data
// qubit 0.
static phase_spec_entry
phase(const char *name, std::uint64_t num_faults,
      std::vector<std::pair<seam_id, std::vector<std::int64_t>>> seams) {
  phase_spec_entry entry;
  dem_chunk_spec spec;
  spec.num_faults = num_faults;
  for (auto &[seam, rows] : seams) {
    seam_spec_entry seam_entry;
    seam_entry.id = seam;
    seam_entry.spec.H_sparse = std::move(rows);
    spec.seam_specs.push_back(std::move(seam_entry));
  }
  spec.O_sparse = {0, -1};
  spec.error_rates.assign(num_faults, 0.02);
  entry.id = seam_id{name};
  entry.spec = std::move(spec);
  return entry;
}

int main() {
  // Faults 0-4 flip data qubits 0-4; faults 5-8 are errors in measuring checks
  // 0-3. Check i reads data qubits i and i + 1.
  dem_chunks_spec model;
  model.seam = {seam_name::next_round, seam_name::prev_round};
  model.phases = {
      phase("bulk", 9,
            {{seam_name::prev_round,
              {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1}},
             {seam_name::next_round, {5, -1, 6, -1, 7, -1, 8, -1}}}),
      phase(
          "final", 5,
          {{seam_name::prev_round, {0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1}}}),
  };
  model.connections = {{seam_id{"bulk"}, seam_id{"bulk"}},
                       {seam_id{"bulk"}, seam_id{"final"}}};

  // [Begin parallel_window]
  std::vector<task_graph_node_config> nodes(1);
  nodes[0].id = "pw";
  nodes[0].kind = "parallel_window";
  nodes[0].dem = model;
  nodes[0].node_params.insert("commit_rounds", std::size_t{3});
  nodes[0].node_params.insert("buffer_rounds", std::size_t{3});
  nodes[0].node_params.insert("gap_rounds", std::size_t{9});
  nodes[0].node_params.insert("output", std::string{"corrections"});
  nodes[0].max_in_flight = 4;
  nodes[0].decoder_type = "pymatching";
  nodes[0].params.insert("merge_strategy", std::string{"smallest_weight"});

  task_graph_options options;
  options.async_streaming = true;
  // One node needs no edges: it is wired measurements -> pw -> output.
  auto graph =
      decoding_task_graph::from_topology(std::move(nodes), {}, options);
  // [End parallel_window]

  // [Begin stream]
  // A 40-round shot in which data qubit 0 flips in round 10: from then on,
  // check 0 reads 1. Stream one round per call and end the shot with FINAL.
  const std::size_t rounds = 40;
  std::vector<std::uint8_t> syndrome(graph->get_num_msyn_per_round(), 0);
  graph->reset_decoder();
  for (std::size_t r = 0; r < rounds; ++r) {
    syndrome[0] = r >= 10;
    graph->enqueue_syndrome(syndrome.data(), syndrome.size(),
                            r + 1 == rounds ? round_type::FINAL
                                            : round_type::BULK);
    if (r == 30) {
      // Mid-shot: the corrections cover rounds [0, frontier).
      const auto progress = graph->streaming_result().outputs.at(0);
      std::cout << "after round " << r << ": rounds ["
                << progress.coverage_begin_round << ", "
                << progress.coverage_end_round << ") committed\n";
    }
  }
  const std::uint8_t correction = graph->get_obs_corrections()[0];
  std::cout << "corrections: [" << int(correction) << "]\n";
  // [End stream]
  return correction == 1 ? 0 : 1;
}
