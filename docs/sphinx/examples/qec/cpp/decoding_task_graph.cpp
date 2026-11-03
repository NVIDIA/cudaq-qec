/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Decoding task graph: construction, one-shot, batch, and streaming interfaces.
//
// Compile and run with:
//   nvq++ -lcudaq-qec -lcudaq-qec-decoders decoding_task_graph.cpp
//   ./a.out

#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/extended_dem.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace cudaq::qec;

// ---------------------------------------------------------------------------
// Helper: rep-5 dem_chunks_spec (init / bulk / final phases).
// ---------------------------------------------------------------------------

static dem_chunks_spec rep5_spec(uint64_t num_rounds = 5) {
  auto make_phase = [](const char *name, uint64_t nf, std::vector<int64_t> H,
                       std::vector<int64_t> O, double rate) {
    phase_spec_entry e;
    e.id = seam_id{name};
    dem_chunk_spec s;
    s.num_faults = nf;
    s.H_sparse = std::move(H);
    s.O_sparse = std::move(O);
    s.error_rates.assign(nf, rate);
    e.spec = std::move(s);
    return e;
  };

  dem_chunks_spec spec;
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  spec.phases = {
      make_phase("init", 9,
                 {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1}, {0, -1},
                 0.02),
      make_phase("bulk", 9,
                 {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1}, {0, -1},
                 0.02),
      make_phase("final", 5, {0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1}, {0, -1},
                 0.02),
  };
  spec.connections = {
      {seam_id{"init"}, seam_id{"bulk"}},
      {seam_id{"bulk"}, seam_id{"bulk"}},
      {seam_id{"bulk"}, seam_id{"final"}},
  };
  spec.num_rounds = num_rounds;
  return spec;
}

// ---------------------------------------------------------------------------
// [Begin from_spec]
// Build a single-decoder task graph from a dem_chunks_spec.
//
// When num_rounds is set the graph uses a fixed-T window (one call to run()
// per shot).  When num_rounds is absent the graph uses a streaming window
// that fires once per round (stride = 1).
// ---------------------------------------------------------------------------

static void example_from_spec() {
  // Fixed-T graph: processes one 5-round shot at a time.
  auto fixed_spec = rep5_spec(/*num_rounds=*/5);
  auto fixed_graph =
      decoding_task_graph::from_spec(fixed_spec, "single_error_lut");

  std::cout << "fixed-T  msyn/decode: "
            << fixed_graph->get_num_msyn_per_decode() << "\n";
  std::cout << "fixed-T  observables: " << fixed_graph->get_num_observables()
            << "\n";

  // Streaming graph: num_rounds absent — window fires once per round.
  auto streaming_spec = rep5_spec();
  streaming_spec.num_rounds = std::nullopt;
  auto streaming_graph =
      decoding_task_graph::from_spec(streaming_spec, "single_error_lut");

  std::cout << "streaming msyn/round: "
            << streaming_graph->get_num_msyn_per_round() << "\n";
  std::cout << "streaming observables: "
            << streaming_graph->get_num_observables() << "\n";
}
// [End from_spec]

// ---------------------------------------------------------------------------
// [Begin one-shot]
// One-shot execution: run() processes one complete shot and returns a
// logical_outcome.  Does not touch the streaming accumulation buffer.
// ---------------------------------------------------------------------------

static void example_one_shot() {
  auto graph = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  const std::size_t n = graph->get_num_msyn_per_decode();

  measurement_results m;
  m.bits.assign(n, 0); // zero syndrome → no predicted errors

  logical_outcome lo = graph->run(m);
  std::cout << "converged: " << lo.converged
            << ", corrections: " << static_cast<int>(lo.bits[0]) << "\n";
}
// [End one-shot]

// ---------------------------------------------------------------------------
// [Begin batch]
// Batch execution: run_batch() processes a collection of complete shots.
// ---------------------------------------------------------------------------

static void example_batch() {
  auto graph = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  measurement_results m;
  m.bits.assign(graph->get_num_msyn_per_decode(), 0);
  std::vector<measurement_results> shots(8, m);
  auto results = graph->run_batch(shots);
  std::cout << "batch results: " << results.size() << " shots processed\n";
}
// [End batch]

// ---------------------------------------------------------------------------
// [Begin streaming]
// Stream a 10-round shot one round per call, tagging the last round FINAL.
// ---------------------------------------------------------------------------

static void example_streaming() {
  auto streaming_spec = rep5_spec();
  streaming_spec.num_rounds = std::nullopt;
  auto graph =
      decoding_task_graph::from_spec(streaming_spec, "single_error_lut");
  const std::size_t per_round = graph->get_num_msyn_per_round();
  const std::size_t num_rounds = 10;
  const std::vector<uint8_t> syndrome(per_round, 0); // no errors

  for (std::size_t r = 0; r < num_rounds; ++r)
    graph->enqueue_syndrome(syndrome.data(), per_round,
                            r + 1 < num_rounds ? round_type::BULK
                                               : round_type::FINAL);

  const auto progress = graph->streaming_result().outputs.at(0);
  std::cout << "complete: "
            << (progress.status == task_graph_streaming_status::complete)
            << "\n";
  std::cout << "corrections[0]: "
            << static_cast<int>(graph->get_obs_corrections()[0]) << "\n";

  // Start the next shot.
  graph->clear_corrections();
}
// [End streaming]

// ---------------------------------------------------------------------------
// [Begin combine]
// combine() runs two independent decoders and XORs their logical outcomes.
// Each branch applies its own D matrix.  Both input graphs are consumed.
// In streaming mode, enqueue_syndrome routes the syndrome to both branches
// simultaneously.
// ---------------------------------------------------------------------------

static void example_combine() {
  auto left = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto right = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");

  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  std::cout << "combined observables: " << combined->get_num_observables()
            << "\n";

  // Batch: run() walks both branches and XORs the results.
  const std::size_t n = combined->get_num_msyn_per_decode();
  measurement_results m;
  m.bits.assign(n, 0);
  logical_outcome lo = combined->run(m);
  std::cout << "combined corrections: " << static_cast<int>(lo.bits[0])
            << "\n"; // XOR of identical = 0

  // Streaming: both branches receive the same syndrome bytes.
  const std::vector<uint8_t> zero_syndrome(n, 0);
  if (combined->enqueue_syndrome(zero_syndrome.data(), n)) {
    const uint8_t *corr = combined->get_obs_corrections();
    std::cout << "combined streaming corrections: " << static_cast<int>(corr[0])
              << "\n";
  }
}
// [End combine]

int main() {
  std::cout << "--- from_spec ---\n";
  example_from_spec();
  std::cout << "\n--- one-shot ---\n";
  example_one_shot();
  std::cout << "\n--- batch ---\n";
  example_batch();
  std::cout << "\n--- streaming ---\n";
  example_streaming();
  std::cout << "\n--- combine ---\n";
  example_combine();
}
