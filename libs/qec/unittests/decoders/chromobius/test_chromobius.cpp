/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoding_task_graph.h"

#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char *chromobius_dem = R"DEM(
error(0.1000000000000000056) D0 D1
error(0.1000000000000000056) D0 D1 D2
error(0.1000000000000000056) D0 L0
error(0.1000000000000000056) D1 D2 D3
error(0.1000000000000000056) D2 D3
error(0.1000000000000000056) D3
detector(0, 0, 0, 1) D0
detector(1, 0, 0, 2) D1
detector(2, 0, 0, 0) D2
detector(3, 0, 0, 1) D3
)DEM";

cudaqx::tensor<uint8_t> make_H(std::size_t num_detectors = 4,
                               std::size_t num_observables = 1) {
  std::vector<uint8_t> H_data(num_detectors * num_observables, 0);
  cudaqx::tensor<uint8_t> H;
  H.copy(H_data.data(), {num_detectors, num_observables});
  return H;
}

cudaqx::heterogeneous_map make_params() { return cudaqx::heterogeneous_map{}; }

} // namespace

TEST(ChromobiusDecoder, checkAllZeroSyndrome) {
  auto decoder = cudaq::qec::decoder::get(
      "chromobius", std::string_view{chromobius_dem}, make_params());

  std::vector<cudaq::qec::float_t> syndrome = {0, 0, 0, 0};
  auto result = decoder->decode(syndrome);

  EXPECT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 1);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(decoder->get_block_size(), 6);
  EXPECT_EQ(decoder->get_syndrome_size(), 4);
  EXPECT_EQ(decoder->get_result_type(),
            cudaq::qec::decode_result_type::observables);
  EXPECT_THROW(
      (void)cudaq::qec::decoder::get(
          "chromobius",
          cudaq::qec::decoder_init::from_stim_dem(std::string{chromobius_dem}),
          cudaq::qec::decode_result_type::errors, make_params()),
      std::invalid_argument);
}

TEST(ChromobiusDecoder, checkKnownObservableFlip) {
  auto decoder = cudaq::qec::decoder::get(
      "chromobius", std::string_view{chromobius_dem}, make_params());

  std::vector<cudaq::qec::float_t> syndrome = {1, 0, 0, 0};
  auto result = decoder->decode(syndrome);

  EXPECT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 1);
  EXPECT_EQ(result.result[0], 1.0);
}

TEST(ChromobiusDecoder, checkPcmInputThrows) {
  EXPECT_THROW(
      (void)cudaq::qec::decoder::get("chromobius", make_H(), make_params()),
      std::runtime_error);
}

TEST(ChromobiusDecoder, checkMalformedDemThrows) {
  EXPECT_THROW(
      (void)cudaq::qec::decoder::get(
          "chromobius", std::string_view{"not a valid DEM"}, make_params()),
      std::runtime_error);
}

TEST(ChromobiusDecoder, declaresDecodeToObs) {
  // It returns an observable frame, so it has to say so: a caller that
  // projects through its own O (a task-graph window_node) has no other way to
  // tell, and standalone the identity O_sparse hides the difference.
  auto decoder = cudaq::qec::decoder::get(
      "chromobius", std::string_view{chromobius_dem}, make_params());
  EXPECT_EQ(decoder->get_result_type(),
            cudaq::qec::decode_result_type::observables);
}

// A task-graph node whose inner decoder is built from Stim DEM text rather
// than from the node's H. Chromobius needs that text: it reads each detector's
// colour from the `detector(...)` coordinates, which no DEM struct in this
// library stores, so the matrices alone cannot build it.
static cudaq::qec::task_graph_node_config chromobius_node() {
  cudaq::qec::task_graph_node_config node;
  node.id = "global";
  node.kind = "single_window";
  node.decoder_type = "chromobius";
  node.stim_dem = chromobius_dem;
  // One measurement per detector, and one observable row to match L0.
  node.d_sparse = {{0}, {1}, {2}, {3}};
  node.o_sparse = {{0}};
  node.node_params.insert("num_syndromes_per_round", std::size_t{4});
  return node;
}

TEST(ChromobiusDecoder, drivesATaskGraphNodeFromStimDemText) {
  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(chromobius_node());
  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes));

  // D0 alone is the `error(...) D0 L0` mechanism, so L0 flips -- the same
  // answer checkKnownObservableFlip gets from the bare decoder, now reached
  // through a task graph that projects nothing of its own.
  const uint8_t msyn[4] = {1, 0, 0, 0};
  ASSERT_TRUE(graph->enqueue_syndrome(msyn, 4));
  ASSERT_EQ(graph->get_num_observables(), 1u);
  EXPECT_EQ(graph->get_obs_corrections()[0], 1u);
}

TEST(ChromobiusDecoder, drivesTheTaskGraphBatchPathsFromStimDemText) {
  // run()/run_async() have to read the observable frame the same way the
  // streaming path does. They used to project it through O unconditionally,
  // so a node whose O is wider than the observable count threw. Give this
  // node a 4-column O (one per detector) so it is wider than the single
  // observable chromobius returns -- otherwise the two paths coincide and
  // the test cannot see the bug.
  auto node = chromobius_node();
  node.o_sparse = {{0, 1, 2, 3}};
  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes));

  cudaq::qec::measurement_results m;
  m.bits = {1, 0, 0, 0}; // D0 alone -> the `D0 L0` mechanism
  EXPECT_EQ(graph->run(m).bits, (std::vector<uint8_t>{1}));
  EXPECT_EQ(graph->run_async(m).get().bits, (std::vector<uint8_t>{1}));
}

TEST(ChromobiusDecoder, taskGraphRejectsAnErrorPatternDecoderForStimDem) {
  // The text's fault columns are not the node's, so only a decoder that
  // projects to observables itself can be driven this way.
  auto node = chromobius_node();
  node.decoder_type = "single_error_lut_example";
  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  EXPECT_THROW(
      (void)cudaq::qec::decoding_task_graph::from_topology(std::move(nodes)),
      std::invalid_argument);
}
