/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../lib/decoding_task/decoding_task_edge_value.h"
#include "../lib/decoding_task/decoding_task_graph_utils.h"
#include "../lib/decoding_task/decoding_task_memory_window_node.h"
#include "../lib/decoding_task/decoding_task_predecoder_node.h"
#include "../lib/decoding_task/decoding_task_window_node.h"
#include "../lib/realtime/decode_target.h"
#include "../lib/realtime/realtime_decoding.h"
#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/decoding_task_types.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/extended_dem.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace cudaq::qec {

/// gtest prints a mismatched type by its C++ name.
void PrintTo(const edge_value_type &t, std::ostream *os) {
  *os << t.info().name();
}

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Matrices derived from one rep-3 chunk, shared by the tests below: they all
// decode the same code and differ only in what they do with the graph.
struct rep3_inputs {
  std::vector<std::vector<uint32_t>> d;
  std::vector<std::vector<uint32_t>> o;
  detector_error_model closed;

  decoding_task_graph graph() const { return graph_with_d(d); }

  // Same graph over a caller-supplied D, for tests that perturb it.
  decoding_task_graph
  graph_with_d(const std::vector<std::vector<uint32_t>> &d_override) const {
    const cudaqx::heterogeneous_map params;
    return decoding_task_graph(
        get_decoder("single_error_lut", closed.detector_error_matrix, params),
        d_override, o);
  }

  // The same for O, which perturbs the observable count rather than the window.
  decoding_task_graph
  graph_with_o(const std::vector<std::vector<uint32_t>> &o_override) const {
    const cudaqx::heterogeneous_map params;
    return decoding_task_graph(
        get_decoder("single_error_lut", closed.detector_error_matrix, params),
        d, o_override);
  }
};

// Rep-3 CSS code: H_Z = [[1,1,0],[0,1,1]], L_Z = [[1,0,0]], X noise only.
rep3_inputs make_rep3_inputs() {
  css_code_matrices code;
  code.hz = sparse_binary_matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});
  code.lz = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  css_noise_params noise;
  noise.px = 0.01;

  const std::vector<extended_dem> chunks{
      extended_dem_from_css_matrices(code, noise)};
  return {dem_chunks_to_d_sparse(chunks, seam_name::next_round,
                                 seam_name::prev_round),
          dem_chunks_to_o_sparse(chunks),
          dem_close_all(chunks, seam_name::next_round, seam_name::prev_round)};
}

// Build the rep-5 spec used in test_extended_dem.cpp.
static dem_chunks_spec rep5_spec(uint64_t num_rounds = 5) {
  dem_chunks_spec spec;
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_final}};
  spec.num_rounds = num_rounds;

  auto make = [](uint64_t nf, std::vector<int64_t> H, std::vector<int64_t> O,
                 double rate) {
    dem_chunk_spec s;
    s.num_faults = nf;
    s.H_sparse = std::move(H);
    s.O_sparse = std::move(O);
    s.error_rates.assign(nf, rate);
    return s;
  };

  phase_spec_entry init_e;
  init_e.id = phase_name::dem_init;
  init_e.spec = make(9, {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1},
                     {0, -1}, 0.02);

  phase_spec_entry bulk_e;
  bulk_e.id = phase_name::dem_bulk;
  bulk_e.spec = make(9, {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1},
                     {0, -1}, 0.02);

  phase_spec_entry final_e;
  final_e.id = phase_name::dem_final;
  final_e.spec =
      make(5, {0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1}, {0, -1}, 0.02);

  spec.phases = {init_e, bulk_e, final_e};
  return spec;
}

// ---------------------------------------------------------------------------
// apply_d_sparse
// ---------------------------------------------------------------------------

TEST(ApplyDSparse, SingleRowXor) {
  // d_sparse[0] = {0, 2} means detector 0 = m[0] XOR m[2]
  std::vector<std::vector<uint32_t>> d = {{0, 2}, {1}};
  measurement_results m;
  m.bits = {1, 0, 1, 0}; // m[0]^m[2]=0, m[1]=0
  const auto dets = apply_d_sparse(m, d);
  ASSERT_EQ(dets.bits.size(), 2u);
  EXPECT_EQ(dets.bits[0], 0.0f); // 1^1=0
  EXPECT_EQ(dets.bits[1], 0.0f); // 0
}

TEST(ApplyDSparse, OutOfRangeThrows) {
  std::vector<std::vector<uint32_t>> d = {{5}};
  measurement_results m;
  m.bits = {0, 0};
  EXPECT_THROW(apply_d_sparse(m, d), std::out_of_range);
}

// ---------------------------------------------------------------------------
// apply_o_sparse
// ---------------------------------------------------------------------------

TEST(ApplyOSparse, FlipsObservable) {
  // o_sparse[0] = {1} means col 1 flips observable 0
  std::vector<std::vector<uint32_t>> o = {{1}};
  error_pattern e;
  e.converged = true;
  e.result = {0.0f, 1.0f}; // fault 1 predicted
  const auto lo = apply_o_sparse(e, o);
  EXPECT_EQ(lo.bits[0], 1u);
  EXPECT_TRUE(lo.converged);
}

TEST(ApplyOSparse, NoFlip) {
  std::vector<std::vector<uint32_t>> o = {{0}};
  error_pattern e;
  e.converged = true;
  e.result = {0.0f};
  const auto lo = apply_o_sparse(e, o);
  EXPECT_EQ(lo.bits[0], 0u);
}

// ---------------------------------------------------------------------------
// xor_logical
// ---------------------------------------------------------------------------

TEST(XorLogical, BasicXor) {
  logical_outcome a, b;
  a.bits = {0, 1};
  b.bits = {1, 1};
  a.converged = true;
  b.converged = true;
  const auto c = xor_logical(a, b);
  EXPECT_EQ(c.bits[0], 1u);
  EXPECT_EQ(c.bits[1], 0u);
  EXPECT_TRUE(c.converged);
}

TEST(XorLogical, SizeMismatchThrows) {
  logical_outcome a, b;
  a.bits = {0};
  b.bits = {0, 1};
  EXPECT_THROW(xor_logical(a, b), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// decoding_task_graph — construction and one-shot API
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, RunZeroSyndromesGivesZeroCorrections) {
  const auto graph = make_rep3_inputs().graph();

  // Zero syndromes → no errors → zero observables.
  measurement_results m;
  m.bits.assign(graph.get_num_msyn_per_decode(), 0);

  const auto lo = graph.run(m);
  EXPECT_TRUE(lo.converged);
  EXPECT_EQ(lo.bits.size(), graph.get_num_observables());
  for (auto bit : lo.bits)
    EXPECT_EQ(bit, 0u);
}

TEST(DecodingTaskGraph, RunBatchMatchesSequential) {
  const auto graph = make_rep3_inputs().graph();

  std::vector<measurement_results> shots(4);
  for (auto &m : shots)
    m.bits.assign(graph.get_num_msyn_per_decode(), 0);

  const auto batch = graph.run_batch(shots);
  ASSERT_EQ(batch.size(), shots.size());
  for (const auto &lo : batch)
    for (auto bit : lo.bits)
      EXPECT_EQ(bit, 0u);
}

TEST(DecodingTaskGraph, RunRejectsShortMeasurementResults) {
  const auto graph = make_rep3_inputs().graph();
  measurement_results m;
  m.bits.assign(graph.get_num_msyn_per_decode() - 1, 0);

  EXPECT_THROW(graph.run(m), std::out_of_range);
}

// ---------------------------------------------------------------------------
// decoding_task_graph — from_spec factory
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, FromSpecBuildsGraph) {
  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");
  EXPECT_GT(graph->get_num_msyn_per_decode(), 0u);
  EXPECT_GT(graph->get_num_observables(), 0u);
}

TEST(DecodingTaskGraph, FromSpecRunsZeroSyndrome) {
  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  measurement_results m;
  m.bits.assign(graph->get_num_msyn_per_decode(), 0);
  const auto lo = graph->run(m);
  EXPECT_TRUE(lo.converged);
  for (auto bit : lo.bits)
    EXPECT_EQ(bit, 0u);
}

TEST(DecodingTaskGraph, FromSpecUsesConfiguredSeamIds) {
  // Custom names make this independent of the next_round/prev_round defaults.
  // The expected D is built directly from the same configured seam pair.
  auto spec = rep5_spec(5);
  spec.seam = {seam_id{"outgoing_syndrome"}, seam_id{"incoming_syndrome"}};
  const auto chunks = dem_chunks_from_spec(spec);
  const auto expected_d = make_csr(
      dem_chunks_to_d_sparse(chunks, spec.seam.from_seam, spec.seam.to_seam),
      "D");
  const auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  EXPECT_EQ(graph->d_matrix().num_rows(), expected_d.num_rows());
  EXPECT_EQ(graph->d_matrix().num_cols(), expected_d.num_cols());
  EXPECT_EQ(graph->d_matrix().ptr(), expected_d.ptr());
  EXPECT_EQ(graph->d_matrix().indices(), expected_d.indices());
}

// ---------------------------------------------------------------------------
// decoding_task_graph — streaming (enqueue_syndrome) path
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, EnqueueSyndromeMatchesRun) {
  const auto inputs = make_rep3_inputs();
  const auto graph_one_shot = inputs.graph();
  auto graph_stream = inputs.graph();

  const std::size_t n_msyn = graph_stream.get_num_msyn_per_decode();
  std::vector<uint8_t> syndromes(n_msyn, 0);

  const bool fired = graph_stream.enqueue_syndrome(syndromes.data(), n_msyn);
  EXPECT_TRUE(fired);

  // One-shot run.
  measurement_results m;
  m.bits.assign(n_msyn, 0);
  const auto lo_one_shot = graph_one_shot.run(m);

  // streaming corrections
  const uint8_t *corr = graph_stream.get_obs_corrections();
  ASSERT_EQ(lo_one_shot.bits.size(), graph_stream.get_num_observables());
  for (std::size_t i = 0; i < lo_one_shot.bits.size(); ++i)
    EXPECT_EQ(lo_one_shot.bits[i], corr[i]);
}

TEST(DecodingTaskGraph, RepeatedColumnsCancelInStoredMatrices) {
  // The graph stores D and O canonicalized, so a column named twice in a row
  // cancels there instead of cancelling again on every decode.
  const auto inputs = make_rep3_inputs();

  // Naming an existing column twice more leaves the parity unchanged.
  auto d_dup = inputs.d;
  ASSERT_FALSE(d_dup.empty());
  ASSERT_FALSE(d_dup[0].empty());
  const uint32_t repeated = d_dup[0][0];
  d_dup[0].push_back(repeated);
  d_dup[0].push_back(repeated);

  const auto plain = inputs.graph();
  const auto duplicated = inputs.graph_with_d(d_dup);

  EXPECT_EQ(duplicated.d_matrix().num_nnz(), plain.d_matrix().num_nnz());
  EXPECT_EQ(duplicated.get_num_msyn_per_decode(),
            plain.get_num_msyn_per_decode());
  EXPECT_EQ(duplicated.d_matrix().indices(), plain.d_matrix().indices());
}

TEST(DecodingTaskGraph, ResetClearsCorrections) {
  auto graph = make_rep3_inputs().graph();

  const std::size_t n = graph.get_num_msyn_per_decode();
  std::vector<uint8_t> syndromes(n, 0);
  graph.enqueue_syndrome(syndromes.data(), n);
  graph.reset_decoder();

  const uint8_t *corr = graph.get_obs_corrections();
  for (std::size_t i = 0; i < graph.get_num_observables(); ++i)
    EXPECT_EQ(corr[i], 0u);
}

// ---------------------------------------------------------------------------
// Realtime dispatch routing: g_task_graphs takes priority over g_decoders
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, HostDispatchRoutesToTaskGraph) {
  // Verify that host::enqueue_syndromes routes to g_task_graphs[id] when set,
  // bypassing g_decoders[id] even when that slot is null.  This exercises the
  // dispatch changes in realtime_decoding.cpp; the decoding server's own ring
  // path is covered by the DecodingSession suites.

  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");
  const std::size_t n_msyn = graph->get_num_msyn_per_decode();
  const std::size_t n_obs = graph->get_num_observables();

  // Register the task graph at slot 0 (clears g_decoders[0]).
  cudaq::qec::decoding::host::register_task_graph(0, std::move(graph));

  // Zero syndromes → no errors → zero corrections.
  std::vector<uint8_t> syndromes(n_msyn, 0);
  cudaq::qec::decoding::host::enqueue_syndromes(
      /*decoder_id=*/0, syndromes.data(), n_msyn, /*tag=*/0);

  std::vector<uint8_t> corrections(n_obs, 0xff);
  cudaq::qec::decoding::host::get_corrections(
      /*decoder_id=*/0, corrections.data(), n_obs, /*reset=*/true);
  for (auto c : corrections)
    EXPECT_EQ(c, 0u);

  // reset_decoder must not throw even though g_decoders[0] is null.
  EXPECT_NO_THROW(cudaq::qec::decoding::host::reset_decoder(/*decoder_id=*/0));

  cudaq::qec::decoding::config::finalize_decoders();
}

TEST(DecodingTaskGraph, ConfigureDecodersAcceptsAFinalRowAndServesBothShapes) {
  // The YAML half: a document with a `final` row configures, and both shapes
  // are enqueueable.  With no session this runs the inline path, so it pins
  // parsing and the inline dispatch, not the dispatcher's own lookup.
  const std::string yaml = R"(decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
syndrome_mapping:
  - {id: 1, type: final}
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cudaq::qec::decoding::config::configure_decoders(config), 0);

  // id 0 survives as the pass-through BULK row even though the document only
  // declared id 1 -- it is what a producer sends unless it names a row.
  std::vector<uint8_t> syndromes(3, 0);
  EXPECT_NO_THROW(cudaq::qec::decoding::host::enqueue_syndromes(
      0, syndromes.data(), 3, /*tag=*/0));
  EXPECT_NO_THROW(cudaq::qec::decoding::host::enqueue_syndromes(
      0, syndromes.data(), 3, /*tag=*/0, /*syndrome_mapping_id=*/1));

  cudaq::qec::decoding::config::finalize_decoders();
}

TEST(DecodingTaskGraph, MappingIdResolvesToAShapeOnTheInlinePath) {
  // With no session the id never reaches a dispatcher, so the inline path
  // resolves it locally -- and must refuse an undeclared id the way the
  // dispatcher BAD_REQUESTs one.
  //
  // Break-it: have shape_for_mapping_id fall back to BULK instead of throwing.
  const std::string yaml = R"(decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
syndrome_mapping:
  - {id: 1, type: final}
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cudaq::qec::decoding::config::configure_decoders(config), 0);

  std::vector<uint8_t> syndromes(3, 0);
  auto by_id = [&](std::uint64_t tag, std::uint64_t mapping_id) {
    cudaq::qec::decoding::host::enqueue_syndromes(0, syndromes.data(), 3, tag,
                                                  mapping_id);
  };
  EXPECT_NO_THROW(by_id(/*tag=*/1, /*mapping_id=*/0)); // seeded pass-through
  EXPECT_NO_THROW(by_id(/*tag=*/2, /*mapping_id=*/1)); // the declared row
  EXPECT_THROW(by_id(/*tag=*/3, /*mapping_id=*/7), std::runtime_error);

  cudaq::qec::decoding::config::finalize_decoders();
}

TEST(DecodingTaskGraph, EndingAShotRequiresADeclaredFinalRow) {
  // A shape is a field of a row, so a FINAL round can only be had by naming a
  // row that has one.  A document with no section has only the seeded BULK row
  // under id 0; every other id is refused.
  //
  // Break-it: have shape_for_mapping_id fall back to BULK for an unknown id.
  const std::string yaml = R"(decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cudaq::qec::decoding::config::configure_decoders(config), 0);

  std::vector<uint8_t> syndromes(3, 0);
  EXPECT_NO_THROW(cudaq::qec::decoding::host::enqueue_syndromes(
      0, syndromes.data(), 3, /*tag=*/0));
  EXPECT_THROW(cudaq::qec::decoding::host::enqueue_syndromes(
                   0, syndromes.data(), 3, /*tag=*/1,
                   /*syndrome_mapping_id=*/1),
               std::runtime_error);

  cudaq::qec::decoding::config::finalize_decoders();
}

TEST(DecodingTaskGraph, HostDispatchCarriesTheRoundShapeToTheGraph) {
  // Naming a `final` row is how an in-process caller ends a stream.
  // Exercised through the export itself, not through decode_target: a forward
  // that dropped the resolved shape would otherwise pass every suite.
  //
  // A guarded window is the only shape that withholds a tail, so a FINAL round
  // must commit strictly more than the same round sent as BULK.
  const auto build = [] {
    std::vector<task_graph_node_config> nodes;
    task_graph_node_config node;
    node.id = "w";
    node.kind = "sliding_window";
    node.dem = rep5_spec(5);
    node.decoder_type = "single_error_lut";
    node.node_params.insert("window_size", std::size_t{5});
    node.node_params.insert("step_size", std::size_t{3});
    node.node_params.insert("start_guard", std::size_t{1});
    node.node_params.insert("end_guard", std::size_t{1});
    nodes.push_back(std::move(node));
    return decoding_task_graph::from_topology(std::move(nodes));
  };

  // configure_decoders installs the table shape_for_mapping_id reads, then
  // register_task_graph overrides slot 0 with the guarded window under test.
  const std::string yaml = R"(decoders:
  - id: 0
    type: single_error_lut
    block_size: 1
    syndrome_size: 1
    H_sparse: [0,-1]
    O_sparse: [0,-1]
    D_sparse: [0,-1]
syndrome_mapping:
  - {id: 1, type: final}
)";

  const auto drive = [&](std::optional<std::uint64_t> mapping_id) {
    auto config =
        cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
    EXPECT_EQ(cudaq::qec::decoding::config::configure_decoders(config), 0);
    auto graph = build();
    const std::size_t n_msyn = graph->get_num_msyn_per_decode();
    const std::size_t n_obs = graph->get_num_observables();
    const std::size_t per_round = graph->get_num_msyn_per_round();
    cudaq::qec::decoding::host::register_task_graph(0, std::move(graph));

    std::vector<uint8_t> syndromes(n_msyn, 0);
    syndromes[n_msyn - per_round] = 1; // lands in the end_guard
    if (mapping_id)
      cudaq::qec::decoding::host::enqueue_syndromes(0, syndromes.data(), n_msyn,
                                                    /*tag=*/0, *mapping_id);
    else
      cudaq::qec::decoding::host::enqueue_syndromes(0, syndromes.data(), n_msyn,
                                                    /*tag=*/0);

    std::vector<uint8_t> corrections(n_obs, 0);
    cudaq::qec::decoding::host::get_corrections(0, corrections.data(), n_obs,
                                                /*reset=*/false);
    cudaq::qec::decoding::config::finalize_decoders();
    return corrections;
  };

  // Break-it: have enqueue_syndromes_common pass round_type::BULK instead of
  // shape_for_mapping_id(...).
  EXPECT_NE(drive(/*syndrome_mapping_id=*/1), drive(std::nullopt))
      << "the FINAL row committed nothing the untagged call did not";
  // Not vacuous: the difference is the row's shape, not the extra argument --
  // id 0 is the seeded BULK row, so it must match the untagged call.
  EXPECT_EQ(drive(/*syndrome_mapping_id=*/0), drive(std::nullopt));
}

TEST(DecodingTaskGraph, HostDispatchValidatesTaskGraphBufferSizes) {
  // The task-graph dispatch path must apply the same length checks as the
  // plain-decoder path; without them an oversized get_corrections reads past
  // the graph's corrections buffer.  Syndromes are the one exception: a graph
  // meters a chunk across its nodes, so an oversized one is consumed rather
  // than refused -- see DecodeTargetBudgetsOnlyADecoder.
  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");
  const std::size_t n_msyn = graph->get_num_msyn_per_decode();
  const std::size_t n_obs = graph->get_num_observables();

  cudaq::qec::decoding::host::register_task_graph(0, std::move(graph));

  std::vector<uint8_t> syndromes(n_msyn + 1, 0);
  EXPECT_NO_THROW(cudaq::qec::decoding::host::enqueue_syndromes(
      /*decoder_id=*/0, syndromes.data(), n_msyn + 1, /*tag=*/0));

  std::vector<uint8_t> corrections(n_obs + 1, 0);
  EXPECT_THROW(cudaq::qec::decoding::host::get_corrections(
                   /*decoder_id=*/0, corrections.data(), n_obs + 1,
                   /*reset=*/false),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::decoding::host::get_corrections(
                   /*decoder_id=*/0, corrections.data(), 0, /*reset=*/false),
               std::invalid_argument);

  // An id in neither table reports the same way it does for plain decoders.
  try {
    cudaq::qec::decoding::host::reset_decoder(/*decoder_id=*/1);
    FAIL() << "expected an unregistered id to throw";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("Decoder 1 not found"),
              std::string::npos)
        << e.what();
  }

  cudaq::qec::decoding::config::finalize_decoders();
}

// ---------------------------------------------------------------------------
// Sliding-window (streaming) — from_spec with no num_rounds
// ---------------------------------------------------------------------------

// Build a streaming spec: same as rep5 but num_rounds absent.
static dem_chunks_spec rep5_streaming_spec() {
  auto spec = rep5_spec(5); // copy phases/connections/seam
  spec.num_rounds = std::nullopt;
  return spec;
}

// The closed H a rep5 streaming node ends up with, so a test-supplied
// decoder_instance has the shape window_node will actually hand it.
static sparse_binary_matrix rep5_streaming_h() {
  auto spec = rep5_streaming_spec();
  spec.num_rounds = spec.minimum_rounds();
  const auto chunks = dem_chunks_from_spec(spec);
  return sparse_binary_matrix(
      dem_close_all(chunks, spec.seam.from_seam, spec.seam.to_seam)
          .detector_error_matrix);
}

TEST(DecodingTaskGraph, StreamingFromSpecBuildsGraph) {
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");
  // Streaming builds a 2-round window; buffer > 0 and observables > 0.
  EXPECT_GT(graph->get_num_msyn_per_decode(), 0u);
  EXPECT_GT(graph->get_num_observables(), 0u);
}

TEST(DecodingTaskGraph, StreamingEnqueueFiresOncePerWindow) {
  // Fill the window in one shot — must fire.  Filling again from scratch must
  // also fire, confirming the window resets correctly.
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  const std::size_t n = graph->get_num_msyn_per_decode();
  std::vector<uint8_t> zeros(n, 0);

  EXPECT_TRUE(graph->enqueue_syndrome(zeros.data(), n));

  graph->reset_decoder();
  EXPECT_TRUE(graph->enqueue_syndrome(zeros.data(), n));
}

TEST(DecodingTaskGraph, StreamingFiresEveryRoundOnceWindowIsFull) {
  // The contract a realtime caller follows: one round per call.  The window
  // fires when it first fills, then on every round after, because it slides
  // by one round.
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  const std::size_t per_round = graph->get_num_msyn_per_round();
  const std::size_t capacity = graph->get_num_msyn_per_decode();
  ASSERT_GT(per_round, 0u);
  ASSERT_EQ(capacity % per_round, 0u);
  const std::size_t rounds_per_window = capacity / per_round;
  ASSERT_GT(rounds_per_window, 1u);

  std::vector<uint8_t> round(per_round, 0);
  for (std::size_t i = 0; i + 1 < rounds_per_window; ++i)
    EXPECT_FALSE(graph->enqueue_syndrome(round.data(), per_round))
        << "round " << i;
  EXPECT_TRUE(graph->enqueue_syndrome(round.data(), per_round));
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_TRUE(graph->enqueue_syndrome(round.data(), per_round))
        << "round after fill " << i;
}

TEST(DecodingTaskGraph, OversizedEnqueueIsConsumedRatherThanDropped) {
  // Replaces the drop contract the graph shared with
  // decoder::enqueue_syndrome: metering means more than a window's worth at
  // once drives every fire it covers and carries the remainder forward.  The
  // divergence only ever accepts more, so a caller sending whole rounds is
  // unaffected.  The node-level guard below still refuses an oversized chunk
  // -- see WindowNodeStillRefusesAnOversizedChunk.
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  const std::size_t per_round = graph->get_num_msyn_per_round();
  const std::size_t capacity = graph->get_num_msyn_per_decode();
  ASSERT_GT(capacity / per_round, 1u);

  std::vector<uint8_t> round(per_round, 0);
  std::vector<uint8_t> oversized(capacity + 1, 0);

  // A partial window, then more than a whole window in one call.
  EXPECT_FALSE(graph->enqueue_syndrome(round.data(), per_round));
  EXPECT_TRUE(graph->enqueue_syndrome(oversized.data(), oversized.size()));
}

class native_stream_probe_decoder : public decoder {
public:
  explicit native_stream_probe_decoder(bool withhold_tail = false,
                                       bool tag_aware = true)
      : decoder(decoder_init(sparse_binary_matrix::from_nested_csr(
                                 4, 1, {{0}, {0}, {0}, {0}}),
                             sparse_binary_matrix::from_nested_csr(1, 1, {{0}}),
                             {},
                             sparse_binary_matrix::from_nested_csr(
                                 4, 4, {{0}, {1}, {2}, {3}})),
                decode_result_type::observables),
        withhold_tail_(withhold_tail), tag_aware_(tag_aware) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    ++decode_calls;
    uint8_t parity = 0;
    for (const auto bit : syndrome)
      parity ^= static_cast<uint8_t>(convert_soft_to_hard(bit));
    return decoder_result{true, {static_cast<float_t>(parity)}};
  }

  bool enqueue_syndrome(const uint8_t *syndrome,
                        std::size_t syndrome_length) override {
    ++enqueue_calls;
    enqueue_lengths.push_back(syndrome_length);
    for (std::size_t i = 0; i < syndrome_length; ++i)
      parity_ ^= syndrome[i];
    received_ += syndrome_length;
    if (received_ < 4)
      return false;
    correction_[0] = parity_;
    received_ = 0;
    parity_ = 0;
    return !withhold_tail_;
  }

  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t syndrome_length,
                        round_type type) override {
    // Tag-unaware: what every decoder that overrides only the untagged form
    // gets, through the base adapter.
    if (!tag_aware_)
      return decoder::enqueue_syndrome(syndrome, syndrome_length, type);
    tags.push_back(type);
    return enqueue_syndrome(syndrome, syndrome_length);
  }

  const uint8_t *get_obs_corrections() const override {
    return correction_.data();
  }
  void clear_corrections() override { correction_[0] = 0; }
  void reset_decoder() override {
    received_ = 0;
    parity_ = 0;
    correction_[0] = 0;
  }

  std::size_t enqueue_calls = 0;
  std::size_t decode_calls = 0;
  std::vector<std::size_t> enqueue_lengths;
  std::vector<round_type> tags;

private:
  std::array<uint8_t, 1> correction_{0};
  std::size_t received_ = 0;
  uint8_t parity_ = 0;
  bool withhold_tail_ = false;
  bool tag_aware_ = true;
};

TEST(DecodingTaskGraph, StreamingNodeForwardsToNativeDecoderStream) {
  const auto kinds = registered_task_graph_node_kinds();
  EXPECT_NE(std::find(kinds.begin(), kinds.end(), "streaming"), kinds.end());

  auto decoder = std::make_unique<native_stream_probe_decoder>();
  auto *probe = decoder.get();
  task_graph_node_config node;
  node.id = "native";
  node.kind = "streaming";
  node.d_sparse = {{0}, {1}, {2}, {3}};
  node.o_sparse = {{0}};
  node.node_params.insert("num_syndromes_per_round", std::size_t{1});
  node.decoder_instance = std::move(decoder);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));

  const std::array<uint8_t, 4> shot{1, 0, 0, 0};
  EXPECT_FALSE(graph->enqueue_syndrome(nullptr, 0));
  EXPECT_EQ(probe->enqueue_calls, 0u);
  EXPECT_TRUE(
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  EXPECT_EQ(probe->enqueue_calls, 1u);
  EXPECT_EQ(probe->enqueue_lengths, (std::vector<std::size_t>{shot.size()}));
  EXPECT_EQ(probe->decode_calls, 0u);
  EXPECT_EQ(graph->get_obs_corrections()[0], 1u);

  graph->reset_decoder();
  measurement_results measurements;
  measurements.bits.assign(shot.begin(), shot.end());
  EXPECT_EQ(graph->run(measurements).bits, (std::vector<uint8_t>{1}));
  EXPECT_EQ(probe->decode_calls, 1u);

  auto async_decoder = std::make_unique<native_stream_probe_decoder>();
  auto *async_probe = async_decoder.get();
  task_graph_node_config async_node;
  async_node.id = "native";
  async_node.kind = "streaming";
  async_node.d_sparse = {{0}, {1}, {2}, {3}};
  async_node.o_sparse = {{0}};
  async_node.node_params.insert("num_syndromes_per_round", std::size_t{1});
  async_node.decoder_instance = std::move(async_decoder);
  std::vector<task_graph_node_config> async_nodes;
  async_nodes.push_back(std::move(async_node));
  task_graph_options options;
  options.async_streaming = true;
  options.async_queue_capacity = 4;
  auto async_graph =
      decoding_task_graph::from_topology(std::move(async_nodes), {}, options);
  EXPECT_TRUE(async_graph->enqueue_syndrome(shot.data(), shot.size(),
                                            round_type::FINAL));
  EXPECT_EQ(async_probe->enqueue_calls, 1u);
  EXPECT_EQ(async_probe->enqueue_lengths,
            (std::vector<std::size_t>{shot.size()}));
  EXPECT_EQ(async_probe->decode_calls, 0u);
  EXPECT_EQ(async_graph->get_obs_corrections()[0], 1u);
}

TEST(DecodingTaskGraph, StreamingNodeFlushesNativeDecoderTail) {
  const auto run = [](bool async) {
    auto decoder = std::make_unique<native_stream_probe_decoder>(
        /*withhold_tail=*/true);
    auto *probe = decoder.get();
    task_graph_node_config node;
    node.id = "native";
    node.kind = "streaming";
    node.d_sparse = {{0}, {1}, {2}, {3}};
    node.o_sparse = {{0}};
    node.node_params.insert("num_syndromes_per_round", std::size_t{1});
    node.decoder_instance = std::move(decoder);
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(
        std::move(nodes), {},
        {.async_streaming = async, .async_queue_capacity = 2});

    const std::array<uint8_t, 4> shot{1, 0, 0, 0};
    EXPECT_TRUE(
        graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
    EXPECT_EQ(probe->enqueue_calls, 1u);
    return graph->get_obs_corrections()[0];
  };

  EXPECT_EQ(run(false), 1u);
  EXPECT_EQ(run(true), 1u);
}

// A decoder that decides its length at run time learns where the shot ends
// from the round tag, so the streaming node must deliver FINAL on the round
// that carries it, not just flush() after.
TEST(DecodingTaskGraph, StreamingNodeForwardsRoundTypeToNativeDecoder) {
  const auto run = [](bool async) {
    auto decoder = std::make_unique<native_stream_probe_decoder>();
    auto *probe = decoder.get();
    task_graph_node_config node;
    node.id = "native";
    node.kind = "streaming";
    node.d_sparse = {{0}, {1}, {2}, {3}};
    node.o_sparse = {{0}};
    node.node_params.insert("num_syndromes_per_round", std::size_t{1});
    node.decoder_instance = std::move(decoder);
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(
        std::move(nodes), {},
        {.async_streaming = async, .async_queue_capacity = 4});

    const uint8_t one = 1, zero = 0;
    (void)graph->enqueue_syndrome(&one, 1, round_type::BULK);
    (void)graph->enqueue_syndrome(&zero, 1, round_type::BULK);
    (void)graph->enqueue_syndrome(&zero, 1, round_type::BULK);
    EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));
    EXPECT_EQ(graph->get_obs_corrections()[0], 1u);
    return std::pair{probe->tags, probe->enqueue_lengths};
  };

  using R = round_type;
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async ? "async" : "sync");
    const auto [tags, lengths] = run(async);
    EXPECT_EQ(tags, (std::vector<R>{R::BULK, R::BULK, R::BULK, R::FINAL}));
    EXPECT_EQ(lengths, (std::vector<std::size_t>{1, 1, 1, 1}));
  }
}

// FINAL carries the shot's final measurements, so an empty one is refused
// before any decoder sees it: one that overrides only the untagged form would
// take it as an extra, possibly null, round.
TEST(DecodingTaskGraph, EmptyFinalIsRejectedBeforeTheDecoder) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async ? "async" : "sync");
    auto decoder = std::make_unique<native_stream_probe_decoder>(
        /*withhold_tail=*/false, /*tag_aware=*/false);
    auto *probe = decoder.get();
    task_graph_node_config node;
    node.id = "native";
    node.kind = "streaming";
    node.d_sparse = {{0}, {1}, {2}, {3}};
    node.o_sparse = {{0}};
    node.node_params.insert("num_syndromes_per_round", std::size_t{1});
    node.decoder_instance = std::move(decoder);
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(
        std::move(nodes), {},
        {.async_streaming = async, .async_queue_capacity = 4});

    const std::array<uint8_t, 4> shot{1, 0, 0, 0};
    for (const auto bit : shot)
      (void)graph->enqueue_syndrome(&bit, 1, round_type::BULK);
    const auto committed = graph->streaming_result().generation;
    EXPECT_THROW(graph->enqueue_syndrome(nullptr, 0, round_type::FINAL),
                 std::invalid_argument);
    EXPECT_EQ(graph->streaming_result().generation, committed);
    EXPECT_EQ(probe->enqueue_lengths, (std::vector<std::size_t>{1, 1, 1, 1}));
    EXPECT_EQ(graph->get_obs_corrections()[0], 1u);
  }
}

// Model a native decoder whose legacy correction read seals the current shot.
// A provisional result at the template boundary must not use that accessor.
class sealing_stream_probe_decoder final
    : public native_stream_probe_decoder,
      public streaming_correction_provider {
public:
  bool enqueue_syndrome(const uint8_t *input, std::size_t length,
                        round_type type) override {
    if (sealed_)
      throw std::logic_error("provisional correction read closed the shot");
    return native_stream_probe_decoder::enqueue_syndrome(input, length, type);
  }
  const uint8_t *get_obs_corrections() const override {
    sealed_ = true;
    ++seals;
    return native_stream_probe_decoder::get_obs_corrections();
  }
  const uint8_t *peek_obs_corrections() const override {
    ++peeks;
    return native_stream_probe_decoder::get_obs_corrections();
  }
  const uint8_t *finalize_obs_corrections() override {
    return get_obs_corrections();
  }
  void reset_decoder() override {
    sealed_ = false;
    native_stream_probe_decoder::reset_decoder();
  }
  mutable std::size_t seals = 0, peeks = 0;

private:
  mutable bool sealed_ = false;
};

TEST(DecodingTaskGraph, ProvisionalNativeCorrectionsDoNotSealBeforeFinal) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    auto dec = std::make_unique<sealing_stream_probe_decoder>();
    auto *probe = dec.get();
    task_graph_node_config node;
    node.id = "native";
    node.kind = "streaming";
    node.d_sparse = {{0}, {1}, {2}, {3}};
    node.o_sparse = {{0}};
    node.node_params.insert("num_syndromes_per_round", std::size_t{1});
    node.decoder_instance = std::move(dec);
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(
        std::move(nodes), {},
        {.async_streaming = async, .async_queue_capacity = 2});
    for (std::size_t epoch = 0; epoch < 3; ++epoch) {
      const std::array<uint8_t, 4> prefix{1, 0, 0, 0};
      (void)graph->enqueue_syndrome(prefix.data(), prefix.size(),
                                    round_type::BULK);
      EXPECT_EQ(graph->get_obs_corrections()[0], 1);
      EXPECT_EQ(probe->seals, epoch);
      EXPECT_EQ(probe->peeks, epoch + 1);
      const uint8_t tail = 0;
      (void)graph->enqueue_syndrome(&tail, 1, round_type::FINAL);
      EXPECT_EQ(graph->get_obs_corrections()[0], 1);
      EXPECT_EQ(probe->seals, epoch + 1);
      // A round after FINAL is dropped, so it neither reaches the sealed
      // decoder nor publishes the shot again.
      const auto generation = graph->streaming_result().generation;
      (void)graph->enqueue_syndrome(&tail, 1, round_type::FINAL);
      EXPECT_EQ(graph->streaming_result().generation, generation);
      EXPECT_EQ(probe->seals, epoch + 1);
      graph->reset_decoder();
      EXPECT_EQ(graph->get_obs_corrections()[0], 0);
    }
  }
}

// Model a fixed-length native decoder that defers a complete shot's flips
// until the shot is sealed. A peek leaves them out; the legacy read, or the
// next shot's first round, seals them in.
class deferring_stream_probe_decoder final
    : public native_stream_probe_decoder,
      public streaming_correction_provider {
public:
  bool enqueue_syndrome(const uint8_t *input, std::size_t length) override {
    seal();
    sealed_ = false;
    if (!native_stream_probe_decoder::enqueue_syndrome(input, length))
      return false;
    deferred_ = native_stream_probe_decoder::get_obs_corrections()[0];
    complete_ = true;
    return true;
  }
  const uint8_t *get_obs_corrections() const override {
    seal();
    return frame_.data();
  }
  const uint8_t *peek_obs_corrections() const override { return frame_.data(); }
  const uint8_t *finalize_obs_corrections() override {
    return get_obs_corrections();
  }
  std::optional<std::uint64_t> finalized_round_count() const override {
    return sealed_ ? std::optional<std::uint64_t>(4) : std::nullopt;
  }
  void clear_corrections() override { frame_[0] = 0; }
  void reset_decoder() override {
    frame_[0] = 0;
    complete_ = sealed_ = false;
    native_stream_probe_decoder::reset_decoder();
  }
  mutable std::size_t seals = 0;

private:
  void seal() const {
    if (!complete_)
      return;
    frame_[0] ^= deferred_;
    complete_ = false;
    sealed_ = true;
    ++seals;
  }
  mutable std::array<uint8_t, 1> frame_{0};
  mutable bool complete_ = false, sealed_ = false;
  uint8_t deferred_ = 0;
};

// An untagged record keeps the decoder's legacy contract: the decoder ends its
// fixed-length shot, and the read that publishes it seals it. Peeking instead
// would hold each shot's deferred flips back until the next shot's first
// round, publishing them with that shot.
TEST(DecodingTaskGraph, UntaggedNativeShotsPublishTheirDeferredCorrections) {
  for (const bool async : {false, true})
    for (const bool clear : {false, true}) {
      SCOPED_TRACE(async);
      SCOPED_TRACE(clear);
      auto dec = std::make_unique<deferring_stream_probe_decoder>();
      auto *probe = dec.get();
      task_graph_node_config node;
      node.id = "native";
      node.kind = "streaming";
      node.d_sparse = {{0}, {1}, {2}, {3}};
      node.o_sparse = {{0}};
      node.node_params.insert("num_syndromes_per_round", std::size_t{1});
      node.decoder_instance = std::move(dec);
      std::vector<task_graph_node_config> nodes;
      nodes.push_back(std::move(node));
      auto graph = decoding_task_graph::from_topology(
          std::move(nodes), {},
          {.async_streaming = async, .async_queue_capacity = 2});
      const std::array<std::array<uint8_t, 4>, 4> shots{
          {{1, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, 0}, {0, 1, 1, 0}}};
      uint8_t frame = 0;
      for (std::size_t shot = 0; shot < shots.size(); ++shot) {
        SCOPED_TRACE(shot);
        uint8_t parity = 0;
        for (const auto bit : shots[shot]) {
          (void)graph->enqueue_syndrome(&bit, 1);
          parity ^= bit;
        }
        frame = clear ? parity : frame ^ parity;
        EXPECT_EQ(graph->get_obs_corrections()[0], frame);
        EXPECT_EQ(probe->seals, shot + 1);
        // A shot sealed at its configured length takes the next span of the
        // stream's schedule, not the physical length a FINAL would report.
        EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round,
                  4 * (shot + 1));
        if (clear)
          graph->clear_corrections();
      }
      EXPECT_TRUE(probe->tags.empty());
    }
}

// Model a native decoder whose untagged shots have the configured length,
// dropping a chunk that would overrun one as the base decoder does, and whose
// tagged shots run until FINAL. Each shot flips the frame by its parity.
class shot_length_probe_decoder final : public native_stream_probe_decoder,
                                        public streaming_correction_provider {
public:
  bool enqueue_syndrome(const uint8_t *input, std::size_t length) override {
    return append(input, length, /*tagged=*/false, /*final=*/false);
  }
  bool enqueue_syndrome(const uint8_t *input, std::size_t length,
                        round_type type) override {
    return append(input, length, /*tagged=*/true, type == round_type::FINAL);
  }
  const uint8_t *get_obs_corrections() const override { return frame_.data(); }
  const uint8_t *peek_obs_corrections() const override { return frame_.data(); }
  const uint8_t *finalize_obs_corrections() override {
    if (rounds_)
      end_shot();
    return frame_.data();
  }
  std::optional<std::uint64_t> finalized_round_count() const override {
    return finalized_;
  }
  void clear_corrections() override { frame_[0] = 0; }
  void reset_decoder() override {
    frame_[0] = parity_ = 0;
    rounds_ = 0;
    finalized_.reset();
  }

private:
  bool append(const uint8_t *input, std::size_t length, bool tagged,
              bool final) {
    if (!tagged && rounds_ + length > 4)
      return false;
    finalized_.reset();
    for (std::size_t i = 0; i < length; ++i)
      parity_ ^= input[i];
    rounds_ += length;
    if (!final && (tagged || rounds_ < 4))
      return false;
    end_shot();
    return true;
  }
  void end_shot() {
    frame_[0] ^= parity_;
    finalized_ = rounds_;
    rounds_ = 0;
    parity_ = 0;
  }
  std::array<uint8_t, 1> frame_{0};
  uint8_t parity_ = 0;
  std::uint64_t rounds_ = 0;
  std::optional<std::uint64_t> finalized_;
};

static std::unique_ptr<decoding_task_graph>
make_shot_length_probe_graph(bool async) {
  task_graph_node_config node;
  node.id = "native";
  node.kind = "streaming";
  node.d_sparse = {{0}, {1}, {2}, {3}};
  node.o_sparse = {{0}};
  node.node_params.insert("num_syndromes_per_round", std::size_t{1});
  node.decoder_instance = std::make_unique<shot_length_probe_decoder>();
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  return decoding_task_graph::from_topology(
      std::move(nodes), {},
      {.async_streaming = async, .async_queue_capacity = 2});
}

// The graph cuts an untagged stream at each shot's end, so packets of any
// size, and records longer than a shot, cross from one shot into the next
// rather than overrunning the decoder's fixed-length shot and being dropped.
TEST(DecodingTaskGraph, StreamingNodeCutsUntaggedRecordsAtEachShot) {
  const std::vector<uint8_t> stream{1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                    0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0};
  for (const bool async : {false, true})
    for (const std::size_t packet : {3, 6, 9}) {
      SCOPED_TRACE(::testing::Message()
                   << "async=" << async << " packet=" << packet);
      auto graph = make_shot_length_probe_graph(async);
      for (std::size_t offset = 0; offset < stream.size();) {
        const auto count = std::min(packet, stream.size() - offset);
        (void)graph->enqueue_syndrome(stream.data() + offset, count);
        offset += count;
        // Where a record ends a shot, every shot so far has published.
        if (offset % 4 != 0)
          continue;
        uint8_t frame = 0;
        for (std::size_t i = 0; i < offset; ++i)
          frame ^= stream[i];
        EXPECT_EQ(graph->get_obs_corrections()[0], frame) << offset;
        EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round,
                  offset);
      }
    }
}

// A FINAL shot ends the stream of untagged shots before it, after their
// rounds, and the shot after it begins a new stream from round 0.
TEST(DecodingTaskGraph, StreamingNodeFinalShotEndsItsStream) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async ? "async" : "sync");
    auto graph = make_shot_length_probe_graph(async);
    const auto expect_shot = [&](uint8_t parity, std::uint64_t begin,
                                 std::uint64_t end) {
      EXPECT_EQ(graph->get_obs_corrections()[0], parity);
      const auto &output = graph->streaming_result().outputs[0];
      EXPECT_EQ(output.coverage_begin_round, begin);
      EXPECT_EQ(output.coverage_end_round, end);
      graph->clear_corrections();
    };
    const auto untagged = [&](std::array<uint8_t, 4> shot,
                              std::uint64_t index) {
      (void)graph->enqueue_syndrome(shot.data(), shot.size());
      expect_shot(shot[0] ^ shot[1] ^ shot[2] ^ shot[3], 4 * index,
                  4 * (index + 1));
    };
    untagged({1, 0, 0, 0}, 0);
    untagged({0, 1, 1, 1}, 1);
    const std::array<uint8_t, 6> final_shot{1, 1, 0, 1, 0, 0};
    (void)graph->enqueue_syndrome(final_shot.data(), 5, round_type::BULK);
    (void)graph->enqueue_syndrome(final_shot.data() + 5, 1, round_type::FINAL);
    expect_shot(1, 8, 14);
    untagged({0, 0, 1, 0}, 0);
    untagged({1, 1, 0, 0}, 1);
  }
}

TEST(DecodingTaskGraph, StreamingNodeRequiresDecoderAndDWidthsToMatch) {
  task_graph_node_config node;
  node.id = "native";
  node.kind = "streaming";
  node.d_sparse = {{0}, {1}, {2}, {2}};
  node.o_sparse = {{0}};
  node.node_params.insert("num_syndromes_per_round", std::size_t{1});
  node.decoder_instance = std::make_unique<native_stream_probe_decoder>();
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));

  EXPECT_THROW(
      {
        try {
          (void)decoding_task_graph::from_topology(std::move(nodes));
        } catch (const std::invalid_argument &error) {
          EXPECT_STREQ(error.what(),
                       "streaming node decoder syndrome size must match D "
                       "column count");
          throw;
        }
      },
      std::invalid_argument);
}

// Returns a different single-fault explanation on each call and counts the
// calls.  The count is what a chunking test needs: a per-call fire flag is not
// chunking-invariant, since one big call reports one fire however many it
// drove.
class rotating_fault_decoder final : public decoder {
public:
  rotating_fault_decoder(sparse_binary_matrix h, std::size_t width)
      : decoder(decoder_init(std::move(h))), width_(width) {}

  decoder_result decode(const std::vector<float_t> &) override {
    std::vector<float_t> e(width_, 0.0f);
    e[calls_ % width_] = 1.0f;
    ++calls_;
    return decoder_result{true, std::move(e)};
  }

  // Shot-local rotation, so two shots of the same bytes accumulate the same
  // corrections however they were chunked.
  void reset_decoder() override { calls_ = 0; }

  std::size_t calls() const { return calls_; }

private:
  std::size_t width_, calls_ = 0;
};

// A fixed 4-measurement window that empties on each fire, so 4n bytes is
// exactly n fires.  Three observable rows make the corrections a 3-bit
// fingerprint of the fire sequence rather than one bit.
struct chunk_fixture {
  rotating_fault_decoder *probe = nullptr;
  std::unique_ptr<decoding_task_graph> graph;
  static constexpr std::size_t capacity = 4;

  chunk_fixture() {
    auto dec = std::make_unique<rotating_fault_decoder>(
        sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}), 4);
    probe = dec.get();
    task_graph_node_config n;
    n.id = "win";
    n.kind = "single_window";
    n.d_sparse = {{0}, {1}, {2}, {3}};
    n.o_sparse = {{0}, {1}, {2, 3}};
    n.decoder_instance = std::move(dec);
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(n));
    graph = decoding_task_graph::from_topology(std::move(nodes));
  }
};

// How many times the window fired over a whole stream, and the corrections it
// left.
struct stream_result {
  std::size_t fires = 0;
  std::vector<uint8_t> corrections;
};

// Deliver `m` in fixed-size chunks, the last one short.  Resets first, so one
// fixture can serve several chunkings.
static stream_result stream_in_chunks(chunk_fixture &fx,
                                      const std::vector<uint8_t> &m,
                                      std::size_t chunk) {
  fx.graph->reset_decoder();
  for (std::size_t off = 0; off < m.size(); off += chunk)
    fx.graph->enqueue_syndrome(m.data() + off, std::min(chunk, m.size() - off));
  const uint8_t *c = fx.graph->get_obs_corrections();
  return {fx.probe->calls(), {c, c + fx.graph->get_num_observables()}};
}

TEST(DecodingTaskGraph, ChunkingTheStreamDoesNotChangeTheResult) {
  // A chunk crossing a fire boundary must drive every fire it covers, so a
  // caller may cut the stream anywhere.  One window per call is the reference.
  chunk_fixture fx;
  const std::size_t windows = 5;
  std::vector<uint8_t> m(chunk_fixture::capacity * windows, 0);
  for (std::size_t i = 0; i < m.size(); i += 3)
    m[i] = 1;

  const auto expected = stream_in_chunks(fx, m, chunk_fixture::capacity);
  ASSERT_EQ(expected.fires, windows);
  ASSERT_NE(std::count(expected.corrections.begin(), expected.corrections.end(),
                       uint8_t{0}),
            static_cast<long>(expected.corrections.size()))
      << "reference corrections are all zero; this would pass vacuously";

  for (std::size_t chunk = 1; chunk <= m.size(); ++chunk) {
    const auto got = stream_in_chunks(fx, m, chunk);
    EXPECT_EQ(got.fires, expected.fires) << "chunk size " << chunk;
    EXPECT_EQ(got.corrections, expected.corrections) << "chunk size " << chunk;
  }
}

TEST(DecodingTaskGraph, ZeroLengthChunkChangesNothing) {
  chunk_fixture fx;
  std::vector<uint8_t> half(chunk_fixture::capacity / 2, 1);
  const uint8_t zero = 0;

  EXPECT_FALSE(fx.graph->enqueue_syndrome(&zero, 0));
  // Half a window short, so the empty calls neither fired it nor consumed the
  // bytes that follow.
  EXPECT_FALSE(fx.graph->enqueue_syndrome(half.data(), half.size()));
  EXPECT_FALSE(fx.graph->enqueue_syndrome(&zero, 0));
  EXPECT_TRUE(fx.graph->enqueue_syndrome(half.data(), half.size()));
  EXPECT_EQ(fx.probe->calls(), 1u);
}

TEST(DecodingTaskGraph, PerRoundSizeDividesTheWindow) {
  auto streaming =
      decoding_task_graph::from_spec(rep5_streaming_spec(), "single_error_lut");
  EXPECT_GT(streaming->get_num_msyn_per_decode(),
            streaming->get_num_msyn_per_round());
  EXPECT_EQ(streaming->get_num_msyn_per_decode() %
                streaming->get_num_msyn_per_round(),
            0u);

  // A fixed-T window empties on each fire, so it also keeps accepting whole
  // windows per call.
  auto fixed = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  const std::size_t n = fixed->get_num_msyn_per_decode();
  EXPECT_EQ(n % fixed->get_num_msyn_per_round(), 0u);
  std::vector<uint8_t> zeros(n, 0);
  EXPECT_TRUE(fixed->enqueue_syndrome(zeros.data(), n));
  EXPECT_TRUE(fixed->enqueue_syndrome(zeros.data(), n));
}

TEST(DecodingTaskGraph, DecodeTargetAnswersPerRoundSizeForBothKinds) {
  using cudaq::qec::detail_dispatch::decode_target;

  auto streaming =
      decoding_task_graph::from_spec(rep5_streaming_spec(), "single_error_lut");
  const decode_target graph_target(streaming.get());
  EXPECT_EQ(graph_target.num_msyn_per_round(),
            streaming->get_num_msyn_per_round());
  EXPECT_LT(graph_target.num_msyn_per_round(),
            graph_target.num_msyn_per_decode());

  // A decoder empties its measurement buffer on every fire, so the chunk it
  // always accepts is the whole buffer.
  const auto inputs = make_rep3_inputs();
  const auto n_msyn = inputs.graph().get_num_msyn_per_decode();
  const cudaqx::heterogeneous_map params;
  auto dec =
      get_decoder("single_error_lut",
                  decoder_init(inputs.closed, make_csr(inputs.d, "D")), params);
  const decode_target decoder_target(dec.get());
  EXPECT_EQ(decoder_target.num_msyn_per_decode(), n_msyn);
  EXPECT_EQ(decoder_target.num_msyn_per_round(), n_msyn);
}

TEST(DecodingTaskGraph, DecodeTargetBudgetsOnlyADecoder) {
  // The realtime admission bound has to follow the engine: a graph meters a
  // chunk across its nodes and takes any length, so bounding it there would
  // refuse the chunking the metering exists to accept.  A decoder still drops
  // a chunk that does not fit, so its buffer stays the bound.
  using cudaq::qec::detail_dispatch::decode_target;

  auto streaming =
      decoding_task_graph::from_spec(rep5_streaming_spec(), "single_error_lut");
  const decode_target graph_target(streaming.get());
  EXPECT_FALSE(graph_target.chunk_limit().has_value());

  const std::vector<uint8_t> oversized(
      graph_target.num_msyn_per_decode() * 2 + 1, 0);
  EXPECT_NO_THROW(
      graph_target.validate_syndromes(oversized.data(), oversized.size()));

  const auto inputs = make_rep3_inputs();
  const cudaqx::heterogeneous_map params;
  auto dec =
      get_decoder("single_error_lut",
                  decoder_init(inputs.closed, make_csr(inputs.d, "D")), params);
  const decode_target decoder_target(dec.get());
  ASSERT_TRUE(decoder_target.chunk_limit().has_value());
  EXPECT_EQ(*decoder_target.chunk_limit(),
            decoder_target.num_msyn_per_decode());

  const std::vector<uint8_t> too_big(decoder_target.num_msyn_per_decode() + 1,
                                     0);
  EXPECT_THROW(
      decoder_target.validate_syndromes(too_big.data(), too_big.size()),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, DecodeTargetCarriesTheRoundShapeToBothArms) {
  using cudaq::qec::detail_dispatch::decode_target;

  // The shim is the only thing the realtime path holds, so a guarded window is
  // reachable through it or not at all.
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "w";
  node.kind = "sliding_window";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  node.node_params.insert("window_size", std::size_t{5});
  node.node_params.insert("step_size", std::size_t{3});
  node.node_params.insert("start_guard", std::size_t{1});
  node.node_params.insert("end_guard", std::size_t{1});
  nodes.push_back(std::move(node));
  auto guarded = decoding_task_graph::from_topology(std::move(nodes));

  const decode_target graph_target(guarded.get());
  const std::size_t n = graph_target.num_msyn_per_decode();
  const std::size_t per_round = graph_target.num_msyn_per_round();
  const std::size_t obs = guarded->get_num_observables();
  std::vector<uint8_t> shot(n, 0);
  shot[n - per_round] = 1; // lands in the end_guard
  ASSERT_TRUE(graph_target.enqueue_syndrome(shot.data(), shot.size(),
                                            round_type::FINAL));
  const std::vector<uint8_t> tagged(guarded->get_obs_corrections(),
                                    guarded->get_obs_corrections() + obs);

  // Break-it: have the shim's graph arm drop the tag.
  auto plain_graph = decoding_task_graph::from_topology([] {
    std::vector<task_graph_node_config> ns;
    task_graph_node_config nd;
    nd.id = "w";
    nd.kind = "sliding_window";
    nd.dem = rep5_spec(5);
    nd.decoder_type = "single_error_lut";
    nd.node_params.insert("window_size", std::size_t{5});
    nd.node_params.insert("step_size", std::size_t{3});
    nd.node_params.insert("start_guard", std::size_t{1});
    nd.node_params.insert("end_guard", std::size_t{1});
    ns.push_back(std::move(nd));
    return ns;
  }());
  const decode_target plain_target(plain_graph.get());
  ASSERT_TRUE(plain_target.enqueue_syndrome(shot.data(), shot.size()));
  EXPECT_NE(tagged,
            std::vector<uint8_t>(plain_graph->get_obs_corrections(),
                                 plain_graph->get_obs_corrections() + obs));

  // The decoder arm forwards it too; the base drops it, so a plain decoder is
  // unaffected either way.
  const auto inputs = make_rep3_inputs();
  const cudaqx::heterogeneous_map params;
  auto dec =
      get_decoder("single_error_lut",
                  decoder_init(inputs.closed, make_csr(inputs.d, "D")), params);
  const decode_target decoder_target(dec.get());
  std::vector<uint8_t> one(decoder_target.num_msyn_per_decode(), 0);
  EXPECT_NO_THROW((void)decoder_target.enqueue_syndrome(one.data(), one.size(),
                                                        round_type::FINAL));
}

TEST(DecodingTaskGraph, StreamingZeroSyndromesZeroCorrections) {
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  const std::size_t n = graph->get_num_msyn_per_decode();
  std::vector<uint8_t> zeros(n, 0);
  EXPECT_TRUE(graph->enqueue_syndrome(zeros.data(), n));

  const uint8_t *corr = graph->get_obs_corrections();
  for (std::size_t i = 0; i < graph->get_num_observables(); ++i)
    EXPECT_EQ(corr[i], 0u);
}

TEST(DecodingTaskGraph, StreamingResetClearsBuffer) {
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  const std::size_t n = graph->get_num_msyn_per_decode();
  // Deliver one byte short of a full window to verify partial state is held.
  std::vector<uint8_t> partial(n - 1, 0);
  EXPECT_FALSE(graph->enqueue_syndrome(partial.data(), partial.size()));

  graph->reset_decoder();

  // After reset the partial state is gone: a full window fires cleanly.
  std::vector<uint8_t> full(n, 0);
  EXPECT_TRUE(graph->enqueue_syndrome(full.data(), full.size()));
}

TEST(DecodingTaskGraph, ManyResetCyclesProduceIdenticalCorrectionsEachTime) {
  // reset_decoder() between shots has to fully clear state -- not just once
  // (StreamingResetClearsBuffer above covers a single reset) but for
  // thousands of cycles in a row, to catch a counter or buffer remnant that
  // only misbehaves once it has accumulated or wrapped over a long run. A
  // nonzero, non-uniform pattern is deliberate: an all-zero syndrome would
  // pass this even with a decoder that silently returns zero, and would not
  // reveal a leftover buffer byte or a corr_ that failed to clear.
  const auto spec = rep5_streaming_spec();
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  // A single-bit fault is single_error_lut's solvable case (see
  // GuardedSlidingWindowMatchesMonolithicDecodeAccuracy above), so it
  // reliably decodes to a nonzero correction rather than an unconverged zero.
  const std::size_t n = graph->get_num_msyn_per_decode();
  std::vector<uint8_t> pattern(n, 0);
  pattern[0] = 1;

  std::vector<uint8_t> first_corrections;
  constexpr int num_shots = 5000;
  for (int shot = 0; shot < num_shots; ++shot) {
    ASSERT_TRUE(graph->enqueue_syndrome(pattern.data(), n)) << "shot " << shot;
    const uint8_t *corr = graph->get_obs_corrections();
    const std::vector<uint8_t> corrections(corr,
                                           corr + graph->get_num_observables());
    if (shot == 0) {
      first_corrections = corrections;
      // Not vacuous: the pattern must actually make the decoder do something.
      ASSERT_NE(
          std::count(first_corrections.begin(), first_corrections.end(), 0u),
          static_cast<long>(first_corrections.size()))
          << "pattern decodes to all-zero corrections; strengthen it";
    } else {
      ASSERT_EQ(corrections, first_corrections) << "shot " << shot;
    }
    graph->reset_decoder();
  }
}

// ---------------------------------------------------------------------------
// task_graph: YAML block — configure_decoders integration
// ---------------------------------------------------------------------------

static std::string
sliding_window_yaml(const std::string &type = "sliding_window",
                    const std::string &num_rounds = "") {
  // Minimal YAML for a sliding_window task_graph using the rep-5 streaming
  // spec. Phases match rep5_streaming_spec(); error_rates are required by the
  // inner decoder's decoder_init.  Setting num_rounds pins the round count,
  // which turns the same spec into one fixed-T window.
  return R"(
decoders:
  - id: 0
    task_graph:
      type: )" +
         type +
         R"(
      inner_type: single_error_lut
    dem_chunks:)" +
         (num_rounds.empty() ? "" : "\n      num_rounds: " + num_rounds) + R"(
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      phases:
        - name: init
          spec:
            num_faults: 9
            H_sparse: [0,1,5,-1,1,2,6,-1,2,3,7,-1,3,4,8,-1]
            O_sparse: [0,-1]
            error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
        - name: bulk
          spec:
            num_faults: 9
            H_sparse: [0,1,5,-1,1,2,6,-1,2,3,7,-1,3,4,8,-1]
            O_sparse: [0,-1]
            error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
        - name: final
          spec:
            num_faults: 5
            H_sparse: [0,1,-1,1,2,-1,2,3,-1,3,4,-1]
            O_sparse: [0,-1]
            error_rates: [0.02,0.02,0.02,0.02,0.02]
)";
}

TEST(DecodingTaskGraph, TaskGraphYAMLConfiguresViaConfigureDecoders) {
  namespace cfg = cudaq::qec::decoding::config;

  auto multi_config =
      cfg::multi_decoder_config::from_yaml_str(sliding_window_yaml());
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  // Slot 0 now holds a task graph rather than a plain decoder, and the host
  // API reaches it: an id backed only by g_decoders would report not found.
  EXPECT_NO_THROW(cudaq::qec::decoding::host::reset_decoder(/*decoder_id=*/0));

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, TaskGraphYAMLEnqueueAndGetCorrections) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace h = cudaq::qec::decoding::host;

  auto multi_config =
      cfg::multi_decoder_config::from_yaml_str(sliding_window_yaml());
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  // Determine the window size from the graph stored in g_task_graphs.
  // Use register_task_graph's slot 0 via the host API.
  // We can't query num_msyn_per_decode directly through the host API,
  // so deliver a full buffer worth of zeros to ensure a fire.
  // The streaming spec builds a (non_repeating=3)-round window; we need
  // enough bytes to fill it.  Over-deliver and check corrections are zero.
  //
  // Use from_spec with the same spec to learn the buffer size.
  const auto spec = rep5_streaming_spec();
  auto probe =
      cudaq::qec::decoding_task_graph::from_spec(spec, "single_error_lut");
  const std::size_t n = probe->get_num_msyn_per_decode();

  std::vector<uint8_t> zeros(n, 0);
  h::enqueue_syndromes(/*decoder_id=*/0, zeros.data(), n, /*tag=*/0);

  std::vector<uint8_t> corr(1, 0xff);
  h::get_corrections(/*decoder_id=*/0, corr.data(), 1, /*reset=*/true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

// ---------------------------------------------------------------------------
// task_graph: YAML block — flat form (pre-expanded H/D/O)
// ---------------------------------------------------------------------------

// A task_graph whose DEM arrives as flat matrices instead of dem_chunks: the
// inner decoder is built from H, and D/O wrap it in a fixed-T graph.
// H: detector 0 = mechanisms {0,1}, detector 1 = {1,2}.  D: one measurement
// per detector.  O: observable 0 flips with mechanism 0.
static std::string
flat_task_graph_yaml(const std::string &d_sparse = "[0, -1, 1, -1]",
                     const std::string &type = "single_window") {
  return R"(
decoders:
  - id: 0
    task_graph:
      type: )" +
         type +
         R"(
      inner_type: single_error_lut
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
    D_sparse: )" +
         d_sparse + R"(
)";
}

TEST(DecodingTaskGraph, FlatTaskGraphYAMLEnqueueAndGetCorrections) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace h = cudaq::qec::decoding::host;

  auto multi_config =
      cfg::multi_decoder_config::from_yaml_str(flat_task_graph_yaml());
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  // D has one column per measurement, so the window is two bytes wide.
  std::vector<uint8_t> zeros(2, 0);
  h::enqueue_syndromes(/*decoder_id=*/0, zeros.data(), zeros.size(),
                       /*tag=*/0);

  std::vector<uint8_t> corr(1, 0xff);
  h::get_corrections(/*decoder_id=*/0, corr.data(), 1, /*reset=*/true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, FlatTaskGraphYAMLRequiresDAndO) {
  namespace cfg = cudaq::qec::decoding::config;

  const std::string no_d = R"(
decoders:
  - id: 0
    task_graph:
      type: sliding_window
      inner_type: single_error_lut
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
)";
  // A flat DEM is only complete with D, so the config never reaches the
  // graph builder.
  EXPECT_THROW(cfg::multi_decoder_config::from_yaml_str(no_d),
               std::runtime_error);
}

TEST(DecodingTaskGraph, FlatTaskGraphYAMLRejectsOutOfRangeColumn) {
  namespace cfg = cudaq::qec::decoding::config;

  // Larger than the sparse matrix index type: narrowing it would alias the
  // entry onto a real measurement column.
  auto out_of_range = cfg::multi_decoder_config::from_yaml_str(
      flat_task_graph_yaml("[0, -1, 5000000000, -1]"));
  // configure_decoders reports a build failure by return code, logging the
  // reason, rather than by letting the exception escape.
  EXPECT_NE(cfg::configure_decoders(out_of_range), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, SlidingWindowNameAcceptedForTheFlatForm) {
  namespace cfg = cudaq::qec::decoding::config;

  // single_window is the clearer spelling for a flat DEM, but the original
  // name keeps working: only single_window carries a shape requirement.
  auto multi_config = cfg::multi_decoder_config::from_yaml_str(
      flat_task_graph_yaml("[0, -1, 1, -1]", "sliding_window"));
  EXPECT_EQ(cfg::configure_decoders(multi_config), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, SingleWindowNeedsAFixedRoundCount) {
  namespace cfg = cudaq::qec::decoding::config;

  // A repeating phase with no num_rounds is a stream of rounds, which
  // from_spec would decode with a sliding window.
  auto multi_config = cfg::multi_decoder_config::from_yaml_str(
      sliding_window_yaml("single_window"));
  EXPECT_NE(cfg::configure_decoders(multi_config), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, SingleWindowAcceptsAPinnedRoundCount) {
  namespace cfg = cudaq::qec::decoding::config;

  auto multi_config = cfg::multi_decoder_config::from_yaml_str(
      sliding_window_yaml("single_window", "5"));
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);
  EXPECT_NO_THROW(cudaq::qec::decoding::host::reset_decoder(/*decoder_id=*/0));
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, TwoKeyFormAcceptsARegisteredAlias) {
  namespace cfg = cudaq::qec::decoding::config;

  // The kind registry resolves the spelling, so an alias means here what it
  // means in the nodes form.
  auto multi_config = cfg::multi_decoder_config::from_yaml_str(
      sliding_window_yaml("SingleWindow", "5"));
  EXPECT_EQ(cfg::configure_decoders(multi_config), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, UnknownTaskGraphTypeNamesTheSupportedOnes) {
  namespace cfg = cudaq::qec::decoding::config;

  auto multi_config = cfg::multi_decoder_config::from_yaml_str(
      sliding_window_yaml("rolling_window"));
  EXPECT_NE(cfg::configure_decoders(multi_config), 0);
  cfg::finalize_decoders();
}

// ---------------------------------------------------------------------------
// task_graph: YAML block — nodes/edges form
// ---------------------------------------------------------------------------

// The rep-5 phase block, indented for a `dem_chunks:` key at `indent` spaces.
static std::string rep5_phase_block(const std::string &indent,
                                    const std::string &num_rounds = "") {
  const std::string body =
      (num_rounds.empty() ? "" : "num_rounds: " + num_rounds + "\n" + indent) +
      R"(seam: {from: next_round, to: prev_round}
)" + indent +
      R"(connections:
)" + indent +
      R"(  - {from: init, to: bulk}
)" + indent +
      R"(  - {from: bulk, to: bulk}
)" + indent +
      R"(  - {from: bulk, to: final}
)" + indent +
      R"(phases:
)" + indent +
      R"(  - name: init
)" + indent +
      R"(    spec: {num_faults: 9, H_sparse: [0,1,5,-1,1,2,6,-1,2,3,7,-1,3,4,8,-1], O_sparse: [0,-1], error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]}
)" + indent +
      R"(  - name: bulk
)" + indent +
      R"(    spec: {num_faults: 9, H_sparse: [0,1,5,-1,1,2,6,-1,2,3,7,-1,3,4,8,-1], O_sparse: [0,-1], error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]}
)" + indent +
      R"(  - name: final
)" + indent +
      R"(    spec: {num_faults: 5, H_sparse: [0,1,-1,1,2,-1,2,3,-1,3,4,-1], O_sparse: [0,-1], error_rates: [0.02,0.02,0.02,0.02,0.02]}
)";
  return indent + body;
}

// The sliding window written as a one-node graph, with
// `straddle_end_round: true` replaced by `end_guard: 1` (they are different
// axes; see the kind's parameter schema).
static std::string nodes_form_yaml(const std::string &node_body) {
  return R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: sw
          type: SlidingWindow
)" + node_body +
         R"(          inner:
            type: single_error_lut
    dem_chunks:
)" + rep5_phase_block("      ");
}

TEST(DecodingTaskGraph, NodesFormSlidingWindowConfiguresAndStreams) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace h = cudaq::qec::decoding::host;

  auto multi_config = cfg::multi_decoder_config::from_yaml_str(nodes_form_yaml(
      R"(          window_size: 3
          step_size: 2
          num_syndromes_per_round: 4
          end_guard: 1
)"));
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  // Three rounds of four check bytes fill the window; zero syndromes commit
  // nothing regardless of which rounds the guard withholds.
  std::vector<uint8_t> round(4, 0);
  for (int i = 0; i < 3; ++i)
    h::enqueue_syndromes(/*decoder_id=*/0, round.data(), round.size(),
                         /*tag=*/0);

  std::vector<uint8_t> corr(1, 0xff);
  h::get_corrections(/*decoder_id=*/0, corr.data(), 1, /*reset=*/true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, NodesFormRejectsStepLargerThanWindow) {
  namespace cfg = cudaq::qec::decoding::config;

  EXPECT_THROW(cfg::multi_decoder_config::from_yaml_str(nodes_form_yaml(
                   R"(          window_size: 3
          step_size: 4
)")),
               std::runtime_error);
}

TEST(DecodingTaskGraph, NodesFormRoundTripsThroughYAML) {
  namespace cfg = cudaq::qec::decoding::config;

  const auto yaml = nodes_form_yaml(
      R"(          window_size: 3
          step_size: 2
          end_guard: 1
          partition: local
)");
  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(parsed.decoders.size(), 1u);
  const auto &node = parsed.decoders[0].task_graph->nodes.at(0);
  EXPECT_EQ(node.id, "sw");
  EXPECT_EQ(node.resolved_kind(), "sliding_window");
  EXPECT_EQ(node.resolved_decoder_type(), "single_error_lut");
  EXPECT_EQ(node.partition_label, "local");

  // An alias is an input spelling only: emission writes the canonical name, so
  // the topology is stable from the first round trip onward.
  const auto emitted = parsed.to_yaml_str();
  EXPECT_NE(emitted.find("sliding_window"), std::string::npos) << emitted;
  EXPECT_EQ(emitted.find("SlidingWindow"), std::string::npos) << emitted;

  // dem_chunks emission cannot recover seam and phase names from their hashes,
  // so compare the topology, which is what these traits are responsible for.
  auto reparsed = cfg::multi_decoder_config::from_yaml_str(emitted);
  auto expected = parsed.decoders.at(0).task_graph;
  expected->nodes.at(0).type = "sliding_window";
  EXPECT_EQ(reparsed.decoders.at(0).task_graph, expected);

  auto again = cfg::multi_decoder_config::from_yaml_str(reparsed.to_yaml_str());
  EXPECT_EQ(again.decoders.at(0).task_graph,
            reparsed.decoders.at(0).task_graph);
}

TEST(DecodingTaskGraph, NodesFormAcceptsExplicitPartitionColumns) {
  namespace cfg = cudaq::qec::decoding::config;

  auto parsed = cfg::multi_decoder_config::from_yaml_str(nodes_form_yaml(
      R"(          window_size: 3
          partition: {columns: [0, 8, 10, 12]}
)"));
  const auto &node = parsed.decoders[0].task_graph->nodes.at(0);
  ASSERT_EQ(node.partition_columns.size(), 2u);
  EXPECT_EQ(node.partition_columns[0].lo, 0u);
  EXPECT_EQ(node.partition_columns[0].hi, 8u);
  EXPECT_EQ(node.partition_columns[1].lo, 10u);
  EXPECT_EQ(node.partition_columns[1].hi, 12u);

  // Column ownership is parsed and emitted but not yet honored, so building
  // says so rather than quietly decoding the whole DEM.
  auto config = parsed;
  EXPECT_NE(cfg::configure_decoders(config), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, NodeDemChunksOverrideTheEntryLevelDem) {
  namespace cfg = cudaq::qec::decoding::config;

  // The entry-level DEM is pinned at 5 rounds while the node describes a
  // stream, so a 7-round window can only come from the node's own DEM.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: sw
          type: sliding_window
          window_size: 7
          step_size: 1
          inner:
            type: single_error_lut
          dem_chunks:
)" + rep5_phase_block("            ") +
                           R"(    dem_chunks:
)" + rep5_phase_block("      ", "5");

  auto multi_config = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  // 7 rounds of 4 check bytes: the window fires only once the node's own DEM
  // has been filled.
  std::vector<uint8_t> round(4, 0);
  for (int i = 0; i < 7; ++i)
    cudaq::qec::decoding::host::enqueue_syndromes(0, round.data(), round.size(),
                                                  0);
  std::vector<uint8_t> corr(1, 0xff);
  cudaq::qec::decoding::host::get_corrections(0, corr.data(), 1, true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, NodesFormRunsOnEntryLevelFlatMatrices) {
  namespace cfg = cudaq::qec::decoding::config;

  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: w
          type: single_window
          inner:
            type: single_error_lut
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1]
)";
  auto multi_config = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  std::vector<uint8_t> zeros(2, 0);
  cudaq::qec::decoding::host::enqueue_syndromes(0, zeros.data(), zeros.size(),
                                                0);
  std::vector<uint8_t> corr(1, 0xff);
  cudaq::qec::decoding::host::get_corrections(0, corr.data(), 1, true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, NodesFormMultiNodeGraphRunsOnFlatMatrices) {
  namespace cfg = cudaq::qec::decoding::config;

  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: x
          type: single_window
          inner: {type: single_error_lut}
        - id: z
          type: single_window
          inner: {type: single_error_lut}
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: x}
        - {from: measurements, to: z}
        - {from: x, to: combine.a}
        - {from: z, to: combine.b}
        - {from: combine, to: output}
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1]
)";
  auto multi_config = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  std::vector<uint8_t> zeros(2, 0);
  cudaq::qec::decoding::host::enqueue_syndromes(0, zeros.data(), zeros.size(),
                                                0);
  std::vector<uint8_t> corr(1, 0xff);
  cudaq::qec::decoding::host::get_corrections(0, corr.data(), 1, true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, NodesFormMultiNodeGraphConfiguresAndStreams) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace h = cudaq::qec::decoding::host;

  // The X/Z split the user docs show: two windows fed the same shot, their
  // outcomes XORed. This is combine()'s topology written as YAML.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: x
          type: sliding_window
          window_size: 3
          step_size: 1
          inner: {type: single_error_lut}
        - id: z
          type: sliding_window
          window_size: 3
          step_size: 1
          inner: {type: single_error_lut}
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: x}
        - {from: measurements, to: z}
        - {from: x, to: combine.a}
        - {from: z, to: combine.b}
        - {from: combine, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");

  auto multi_config = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(multi_config), 0);

  std::vector<uint8_t> round(4, 0);
  for (int i = 0; i < 3; ++i)
    h::enqueue_syndromes(0, round.data(), round.size(), 0);

  std::vector<uint8_t> corr(1, 0xff);
  h::get_corrections(0, corr.data(), 1, /*reset=*/true);
  EXPECT_EQ(corr[0], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, ExampleThreeTopologyParsesAndValidates) {
  namespace cfg = cudaq::qec::decoding::config;

  // Example 3: a pre-decoder feeding a residual decoder,
  // both XORed. The plugin it names does not exist in this build, and the
  // ports it wires are not declared by any kind, so this is a parse-and-
  // validate case: the configuration surface accepts the topology.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: ising_cnn
          window_size: 8
          step_size: 2
          guard_rounds: 3
          partition: local
        - id: global
          type: pymatching
          partition: boundary
        - id: combine
          type: XOR
      edges:
        - {from: measurements, to: pre.input}
        - {from: pre.residual, to: global.input}
        - {from: pre.pre_L, to: combine.a}
        - {from: global.L, to: combine.b}
        - {from: combine, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  const auto &tg = *parsed.decoders[0].task_graph;
  ASSERT_EQ(tg.nodes.size(), 3u);
  // A type that names no registered kind is taken as a decoder plugin, and the
  // node is a window over it.
  EXPECT_EQ(tg.nodes[0].resolved_kind(), "sliding_window");
  EXPECT_EQ(tg.nodes[0].resolved_decoder_type(), "ising_cnn");
  EXPECT_EQ(tg.nodes[2].resolved_kind(), "xor");
  EXPECT_TRUE(tg.nodes[2].resolved_decoder_type().empty());
  // Edges parse into the builder's own type, with an unnamed port filled in
  // with the default the builder would have used.
  ASSERT_EQ(tg.edges.size(), 5u);
  EXPECT_EQ(tg.edges[1].from_node, "pre");
  EXPECT_EQ(tg.edges[1].from_output, "residual");
  EXPECT_EQ(tg.edges[4].to_node, "output");
  EXPECT_EQ(tg.edges[4].to_input, "input");

  // `XOR` is an accepted input spelling of the `xor` kind; emission writes the
  // canonical one. A plugin-valued type has no canonical form to write.
  const auto emitted = parsed.to_yaml_str();
  // A port equal to the default is left off, so `pre.input` comes back as the
  // `pre` the document could have written instead. Named ports stay named.
  EXPECT_EQ(emitted.find("pre.input"), std::string::npos) << emitted;
  EXPECT_NE(emitted.find("pre.residual"), std::string::npos) << emitted;
  auto reparsed = cfg::multi_decoder_config::from_yaml_str(emitted);
  auto expected = parsed.decoders.at(0).task_graph;
  expected->nodes.at(2).type = "xor";
  EXPECT_EQ(reparsed.decoders.at(0).task_graph, expected);
}

// Defined with the other from_topology helpers below.
static task_graph_node_config rep5_streaming_node(std::string id = "node0");

TEST(DecodingTaskGraph, InnerDecoderArgsUseTheirRegisteredSchema) {
  namespace cfg = cudaq::qec::decoding::config;
  using kind = cfg::param_kind;

  cfg::register_decoder_schema({"test_inner_nested", {}});
  cfg::decoder_schema schema{
      "test_inner_finalization",
      {{"required_arg", kind::uint64, /*required=*/true},
       {"limit", kind::uint64},
       {"nested_kind", kind::string},
       {"nested_params", kind::discriminated, false, "", "nested_kind",
        /*materialize_empty=*/true}}};
  schema.validate = [](const cudaqx::heterogeneous_map &args) {
    if (args.contains("limit") && args.get<std::size_t>("limit") > 7)
      throw std::runtime_error("limit must be <= 7");
  };
  cfg::register_decoder_schema(std::move(schema));

  const auto graph_yaml = [](const std::string &node) {
    return R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: w
)" + node +
           R"(    dem_chunks:
)" + rep5_phase_block("      ");
  };
  const std::string inner = graph_yaml(
      R"(          type: single_window
          inner:
            type: test_inner_finalization
            required_arg: 1
            nested_kind: test_inner_nested
)");
  const std::string direct = graph_yaml(
      R"(          type: test_inner_finalization
          decoder_custom_args:
            required_arg: 1
            nested_kind: test_inner_nested
)");

  auto inner_config = cfg::multi_decoder_config::from_yaml_str(inner);
  auto direct_config = cfg::multi_decoder_config::from_yaml_str(direct);
  const auto &inner_args =
      inner_config.decoders[0].task_graph->nodes[0].decoder_custom_args.map();
  const auto &direct_args =
      direct_config.decoders[0].task_graph->nodes[0].decoder_custom_args.map();
  EXPECT_TRUE(inner_args.contains("nested_params"));
  EXPECT_TRUE(cfg::custom_args_maps_equal(inner_args, direct_args));

  EXPECT_THROW(cfg::multi_decoder_config::from_yaml_str(graph_yaml(
                   R"(          type: single_window
          inner: {type: test_inner_finalization}
)")),
               std::runtime_error);
  EXPECT_THROW(cfg::multi_decoder_config::from_yaml_str(graph_yaml(
                   R"(          type: single_window
          inner:
            type: test_inner_finalization
            required_arg: 1
            limit: 9
)")),
               std::runtime_error);
}

TEST(DecodingTaskGraph, ExampleThreeStreamsThroughThePredecoder) {
  namespace cfg = cudaq::qec::decoding::config;

  // Example 3, now buildable: `pre` produces a residual, `global` decodes it,
  // and the two commits XOR into one correction. Both nodes share the DEM and
  // the window size, and neither slides, as the edge between them requires.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          window_size: 5
          residual_from: parity_check
          inner: {type: single_error_lut}
        - id: global
          type: single_window
          window_size: 5
          inner: {type: multi_error_lut, lut_error_depth: 2}
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: pre}
        - {from: pre.residual, to: global}
        - {from: pre.pre_L, to: combine.a}
        - {from: global, to: combine.b}
        - {from: combine, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(parsed), 0)
      << "the residual topology should build now that `predecoder` exists";
  // A zero syndrome commits nothing on either branch.
  auto probe = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  const std::size_t n = probe->get_num_msyn_per_decode();
  std::vector<uint8_t> zeros(n, 0);
  cudaq::qec::decoding::host::enqueue_syndromes(0, zeros.data(), n, 0);
  std::vector<uint8_t> corr(1, 0xff);
  cudaq::qec::decoding::host::get_corrections(0, corr.data(), 1, true);
  EXPECT_EQ(corr[0] & 1u, 0u);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, YamlPredecoderNeedsADeclaredResidualSource) {
  namespace cfg = cudaq::qec::decoding::config;

  // The configuration path is the only one that enforces a schema's `required`
  // -- from_topology does not schema-check node params, so the factory carries
  // that half. Both are worth pinning, since a document is where the key is
  // most likely to be left off.
  const std::string with_key = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          window_size: 5
          residual_from: parity_check
          inner: {type: single_error_lut}
    dem_chunks:
)" + rep5_phase_block("      ");

  auto parsed = cfg::multi_decoder_config::from_yaml_str(with_key);
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();

  // Only the one line differs, so the rejection is the required key and not
  // some other flaw in the document.
  std::string without = with_key;
  const auto at = without.find("          residual_from: parity_check\n");
  ASSERT_NE(at, std::string::npos);
  without.erase(at,
                std::string("          residual_from: parity_check\n").size());
  EXPECT_THROW(cfg::multi_decoder_config::from_yaml_str(without),
               std::runtime_error);
}

// Two nodes differing in every matrix: fault spaces of unlike width, and D read
// in opposite orders. No entry-level block or round chain describes both.
static std::string node_flat_yaml() {
  return R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: single_window
          inner: {type: single_error_lut}
          block_size: 2
          syndrome_size: 2
          H_sparse: [0, -1, 1, -1]
          O_sparse: [0, -1]
          D_sparse: [0, -1, 1, -1]
        - id: b
          type: single_window
          inner: {type: single_error_lut}
          block_size: 4
          syndrome_size: 2
          H_sparse: [0, 1, -1, 2, 3, -1]
          O_sparse: [0, -1]
          D_sparse: [1, -1, 0, -1]
      edges:
        - {from: measurements, to: a}
        - {from: measurements, to: b}
        - {from: a, to: output}
        - {from: b, to: output}
)";
}

TEST(DecodingTaskGraph, NodesCarryTheirOwnFlatMatrices) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace h = cudaq::qec::decoding::host;

  auto parsed = cfg::multi_decoder_config::from_yaml_str(node_flat_yaml());
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);

  // Measurement 0 alone. `a` reads detectors (1,0), which only fault 0
  // explains, and its O flips on it. `b`'s reversed D reads (0,1), taking
  // fault 2 or 3, and its O flips on neither. Only each node's own matrices
  // give both answers. Each commits one observable, in order.
  std::vector<uint8_t> syndrome{1, 0};
  h::enqueue_syndromes(/*decoder_id=*/0, syndrome.data(), syndrome.size(),
                       /*tag=*/0);
  std::vector<uint8_t> corr(2, 0xff);
  h::get_corrections(/*decoder_id=*/0, corr.data(), corr.size(),
                     /*reset=*/true);
  EXPECT_EQ(corr[0], 1u);
  EXPECT_EQ(corr[1], 0u);

  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, NodeFlatMatricesSurviveARoundTrip) {
  namespace cfg = cudaq::qec::decoding::config;

  auto parsed = cfg::multi_decoder_config::from_yaml_str(node_flat_yaml());
  auto reparsed =
      cfg::multi_decoder_config::from_yaml_str(parsed.to_yaml_str());
  EXPECT_EQ(parsed, reparsed);

  // Read the widths back rather than resting on struct equality.
  const auto &nodes = reparsed.decoders[0].task_graph->nodes;
  ASSERT_EQ(nodes.size(), 2u);
  EXPECT_EQ(nodes[0].block_size, 2u);
  EXPECT_EQ(nodes[1].block_size, 4u);
  EXPECT_EQ(nodes[1].H_sparse, (std::vector<std::int64_t>{0, 1, -1, 2, 3, -1}));
  EXPECT_EQ(nodes[1].D_sparse, (std::vector<std::int64_t>{1, -1, 0, -1}));
}

TEST(DecodingTaskGraph, LegacyDetsSourceEmitsAsMeasurements) {
  namespace cfg = cudaq::qec::decoding::config;
  std::string yaml = node_flat_yaml();
  std::size_t pos = 0;
  while ((pos = yaml.find("from: measurements", pos)) != std::string::npos) {
    yaml.replace(pos, std::string("from: measurements").size(), "from: dets");
    pos += std::string("from: dets").size();
  }

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  const auto &edges = parsed.decoders[0].task_graph->edges;
  ASSERT_FALSE(edges.empty());
  EXPECT_EQ(edges.front().from_node, "measurements");
  EXPECT_EQ(parsed.to_yaml_str().find("from: dets"), std::string::npos);
}

TEST(DecodingTaskGraph, NodeFlatMatricesRejectIncompleteForms) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const char *needle) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      FAIL() << "expected rejection mentioning '" << needle << "'";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
          << e.what();
    }
  };
  // Each drops one line, so the rejection is that line.
  const auto without = [](const char *line) {
    std::string yaml = node_flat_yaml();
    const auto at = yaml.find(line);
    EXPECT_NE(at, std::string::npos);
    yaml.erase(at, std::string(line).size());
    return yaml;
  };

  rejects(without("          O_sparse: [0, -1]\n"), "come as a pair");
  rejects(without("          block_size: 2\n"), "shape it");

  // Node-level, so the pair is on one node rather than one at each level.
  std::string with_chunks = node_flat_yaml();
  const std::string anchor = "          D_sparse: [0, -1, 1, -1]\n";
  with_chunks.insert(with_chunks.find(anchor) + anchor.size(),
                     "          dem_chunks:\n" +
                         rep5_phase_block("            "));
  rejects(with_chunks, "exactly one of them");

  std::string sizes_only = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: single_error_lut
          window_size: 3
          step_size: 1
          block_size: 3
      edges:
        - {from: measurements, to: a}
        - {from: a, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");
  rejects(sizes_only, "no flat matrices");

  std::string on_xor = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: single_error_lut
          window_size: 3
          step_size: 1
        - id: c
          type: xor
          D_sparse: [0, -1]
          O_sparse: [0, -1]
      edges:
        - {from: measurements, to: a}
        - {from: a, to: c.a}
        - {from: c, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");
  rejects(on_xor, "reads no DEM");
}

TEST(DecodingTaskGraph, ANodesOwnMatricesNeedItsOwnModel) {
  namespace cfg = cudaq::qec::decoding::config;

  // D and O do not carry H, and the entry-level model is indexed in another
  // fault space.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: single_window
          inner: {type: single_error_lut}
          D_sparse: [0, -1, 1, -1]
          O_sparse: [0, -1]
      edges:
        - {from: measurements, to: a}
        - {from: a, to: output}
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1]
)";

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_NE(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, YamlHeterogeneousWindowsNeedOnlyAgreePerOutput) {
  namespace cfg = cudaq::qec::decoding::config;

  // Raw-input nodes on separate outputs may hold different windows, so a
  // document saying so passes validation.
  const std::string head = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: wide
          type: single_error_lut
          window_size: 5
          step_size: 1
        - id: narrow
          type: single_error_lut
          window_size: 3
          step_size: 1
)";
  const std::string dem = R"(    dem_chunks:
)" + rep5_phase_block("      ");

  const std::string separate = head + R"(      edges:
        - {from: measurements, to: wide}
        - {from: measurements, to: narrow}
        - {from: wide, to: output}
        - {from: narrow, to: output}
)" + dem;

  auto parsed = cfg::multi_decoder_config::from_yaml_str(separate);
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();

  // Folded into one output, the same topology remains a valid C++ batch
  // program, but configure_decoders installs realtime targets only and must
  // refuse it before an experiment starts.
  const std::string folded = head + R"(        - id: combine
          type: xor
      edges:
        - {from: measurements, to: wide}
        - {from: measurements, to: narrow}
        - {from: wide, to: combine.a}
        - {from: narrow, to: combine.b}
        - {from: combine, to: output}
)" + dem;

  auto folded_parsed = cfg::multi_decoder_config::from_yaml_str(folded);
  EXPECT_NE(cfg::configure_decoders(folded_parsed), 0);
  EXPECT_THROW(cudaq::qec::decoding::host::reset_decoder(/*decoder_id=*/0),
               std::invalid_argument);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, ResidualEdgeStillNeedsAKindThatProducesOne) {
  namespace cfg = cudaq::qec::decoding::config;

  // The negative half of the pairing above: a kind that declares no `residual`
  // output still cannot carry that edge. Asserting both directions is what
  // makes the positive test meaningful -- if compile_wires stopped checking
  // ports, this one would start passing too.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: single_error_lut
          window_size: 3
          step_size: 1
        - id: global
          type: single_error_lut
          window_size: 3
          step_size: 1
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: pre}
        - {from: pre.residual, to: global}
        - {from: pre, to: combine.a}
        - {from: global, to: combine.b}
        - {from: combine, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");

  // The configuration surface accepts the topology; the builder is what stops.
  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_NE(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();

  try {
    std::vector<cudaq::qec::task_graph_node_config> nodes;
    nodes.push_back(rep5_streaming_node("pre"));
    nodes.push_back(rep5_streaming_node("global"));
    cudaq::qec::task_graph_node_config combiner;
    combiner.id = "combine";
    combiner.kind = "xor";
    nodes.push_back(std::move(combiner));
    decoding_task_graph::from_topology(
        std::move(nodes), {{"measurements", "output", "pre", "input"},
                           {"pre", "residual", "global", "input"},
                           {"pre", "output", "combine", "a"},
                           {"global", "output", "combine", "b"},
                           {"combine", "output", "output", "input"}});
    FAIL() << "expected the residual edge to be rejected";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("no output 'residual'"),
              std::string::npos)
        << e.what();
  }
}

TEST(DecodingTaskGraph, NodesFormRejectsConfigurationErrors) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const char *needle) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      FAIL() << "expected rejection mentioning '" << needle << "'";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
          << e.what();
    }
  };

  // The straddle flags move a window's decode-input columns, not its commit.
  rejects(nodes_form_yaml("          straddle_end_round: true\n"), "end_guard");

  // A guard needs a neighboring window to commit what it withholds.
  rejects(nodes_form_yaml(
              R"(          window_size: 3
          step_size: 3
          end_guard: 1
)"),
          "overlapping windows");
  rejects(nodes_form_yaml(
              R"(          window_size: 4
          step_size: 2
          end_guard: 1
)"),
          "tile the stream exactly once");
  EXPECT_NO_THROW(cfg::multi_decoder_config::from_yaml_str(nodes_form_yaml(
      R"(          window_size: 4
          step_size: 3
          end_guard: 1
)")));

  // A window smaller than the DEM's phase chain cannot be expanded from it.
  rejects(nodes_form_yaml(
              R"(          window_size: 2
          step_size: 1
)"),
          "phase chain needs");

  // A declared round layout that disagrees with the DEM is a modeling error,
  // which only the DEM itself can reveal, so it lands at construction.
  auto mismatch = cfg::multi_decoder_config::from_yaml_str(nodes_form_yaml(
      R"(          window_size: 3
          step_size: 1
          num_syndromes_per_round: 5
)"));
  EXPECT_NE(cfg::configure_decoders(mismatch), 0);
  cfg::finalize_decoders();

  rejects(R"(
decoders:
  - id: 0
    task_graph:
      type: sliding_window
      inner_type: single_error_lut
      nodes:
        - id: sw
          type: sliding_window
          inner: {type: single_error_lut}
    dem_chunks:
)" + rep5_phase_block("      "),
          "mutually exclusive");

  rejects(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: sliding_window
          inner: {type: single_error_lut}
        - id: a
          type: sliding_window
          inner: {type: single_error_lut}
      edges:
        - {from: measurements, to: a}
        - {from: a, to: output}
    dem_chunks:
)" + rep5_phase_block("      "),
          "duplicate task_graph node id");

  rejects(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: sliding_window
          inner: {type: single_error_lut}
        - id: b
          type: sliding_window
          inner: {type: single_error_lut}
    dem_chunks:
)" + rep5_phase_block("      "),
          "edges is required");

  rejects(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: sliding_window
          inner: {type: single_error_lut}
      edges:
        - {from: measurements, to: a}
    dem_chunks:
)" + rep5_phase_block("      "),
          "reaches 'output'");

  // An xor node wraps no decoder and declares no window parameters.
  rejects(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: c
          type: xor
          inner: {type: single_error_lut}
      edges:
        - {from: measurements, to: c}
        - {from: c, to: output}
    dem_chunks:
)" + rep5_phase_block("      "),
          "wraps no decoder");

  rejects(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: c
          type: xor
          window_size: 3
      edges:
        - {from: measurements, to: c}
        - {from: c, to: output}
    dem_chunks:
)" + rep5_phase_block("      "),
          "task_graph_node.xor");

  // A dot with nothing after it names no port. Endpoint errors are reported as
  // YAML diagnostics, which carry a line and column on the error stream rather
  // than in the exception, so this only asserts the rejection.
  EXPECT_ANY_THROW(cfg::multi_decoder_config::from_yaml_str(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: a
          type: sliding_window
          inner: {type: single_error_lut}
      edges:
        - {from: measurements, to: "a."}
        - {from: a, to: output}
    dem_chunks:
)" + rep5_phase_block("      ")));

  // Disagreeing window sizes are the builder's to reject, since it turns on
  // what the nodes feed: YamlHeterogeneousWindowsNeedOnlyAgreePerOutput.
}

TEST(DecodingTaskGraph, JsonSchemaDescribesTaskGraphAndItsKinds) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto schema = cfg::decoder_config_json_schema();

  EXPECT_NE(schema.find("\"task_graph\""), std::string::npos);
  EXPECT_NE(schema.find("\"task_graph_node\""), std::string::npos);
  EXPECT_NE(schema.find("\"task_graph_edge\""), std::string::npos);
  EXPECT_NE(schema.find("\"task_graph_partition\""), std::string::npos);
  // Node parameters come from the kind registry, so a kind registered by this
  // test file is exported too.
  EXPECT_NE(schema.find("\"start_guard\""), std::string::npos);
  EXPECT_NE(schema.find("\"unused_knob\""), std::string::npos);
  // A node entry may be an ensemble pattern instead of a typed node.
  EXPECT_NE(schema.find("\"ensemble\""), std::string::npos);
  EXPECT_NE(schema.find("\"join_strategy\""), std::string::npos);
  // Members hold nodes of the node schema itself; each strategy's parameters
  // follow its name.
  EXPECT_NE(schema.find("\"deadline_us\""), std::string::npos);
  EXPECT_NE(schema.find("\"soft_stop_before_us\""), std::string::npos);
}

// A flat-matrix entry whose one node is an ensemble pattern of two LUT
// windows, the second one a subgraph with explicit edges.
static std::string ensemble_yaml(const std::string &node_block) {
  return R"(
decoders:
  - id: 0
    task_graph:
      nodes:
)" + node_block +
         R"(    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1]
)";
}

TEST(DecodingTaskGraph, EnsembleYamlParsesRoundTripsAndLowers) {
  namespace cfg = cudaq::qec::decoding::config;
  auto multi_config = cfg::multi_decoder_config::from_yaml_str(ensemble_yaml(R"(
        - id: ens
          ensemble:
            join_strategy: deadline
            join_params: {deadline_us: 200000, soft_stop_before_us: 100000}
            members:
              - nodes:
                  - {id: a, type: single_window, inner: {type: single_error_lut}}
                edges: [{from: input, to: a}, {from: a, to: output}]
              - nodes:
                  - {id: b, type: single_window, inner: {type: single_error_lut}}
                  - {id: x, type: xor}
                edges:
                  - {from: input, to: b}
                  - {from: b, to: x.a}
                  - {from: b, to: x.b}
                  - {from: x, to: output}
)"));
  const auto &node = multi_config.decoders.at(0).task_graph->nodes.at(0);
  ASSERT_TRUE(node.ensemble.has_value());
  EXPECT_TRUE(node.type.empty());
  EXPECT_EQ(node.ensemble->members.size(), 2u);
  EXPECT_EQ(node.ensemble->join_strategy, "deadline");
  EXPECT_EQ(node.ensemble->join_params.map().get<std::size_t>("deadline_us"),
            200000u);
  EXPECT_EQ(node.ensemble->members[1].nodes.size(), 2u);
  EXPECT_EQ(node.ensemble->members[1].edges.size(), 4u);

  // The document round-trips through emission.
  const auto again =
      cfg::multi_decoder_config::from_yaml_str(multi_config.to_yaml_str());
  EXPECT_EQ(again.decoders.at(0).task_graph,
            multi_config.decoders.at(0).task_graph);

  // Built, the pattern is a fan-out, the members (the subgraph one with a push
  // tail) and a join; a zero syndrome decodes to the zero correction.
  auto g =
      cudaq::qec::decoding::host::build_task_graph(multi_config.decoders.at(0));
  ASSERT_NE(g, nullptr);
  EXPECT_EQ(g->run(measurement_results{{0, 0}}).bits, std::vector<uint8_t>{0});
}

TEST(DecodingTaskGraph, EnsembleStreamsOnTheServerPathBehindAPredecoder) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace h = cudaq::qec::decoding::host;

  // The server path: configure_decoders() demands a streaming graph and drives
  // it with raw rounds. The ensemble takes its windows from the predecoder's
  // residual and races two windows over it; the winner XORs with pre_L.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          window_size: 5
          residual_from: parity_check
          inner: {type: single_error_lut}
        - id: ens
          ensemble:
            join_strategy: deadline
            join_params: {deadline_us: 500000}
            members:
              - nodes:
                  - {id: a, type: single_window, window_size: 5, push_candidates: true, cancellable: true, inner: {type: single_error_lut}}
                edges: [{from: input, to: a}]
              - nodes:
                  - {id: b, type: single_window, window_size: 5, inner: {type: multi_error_lut, lut_error_depth: 2}}
                edges: [{from: input, to: b}, {from: b, to: output}]
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: pre}
        - {from: pre.residual, to: ens}
        - {from: pre.pre_L, to: combine.a}
        - {from: ens, to: combine.b}
        - {from: combine, to: output}
    dem_chunks:
)" + rep5_phase_block("      ");

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);
  auto probe = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  const std::size_t n = probe->get_num_msyn_per_decode();
  std::vector<uint8_t> zeros(n, 0);
  for (int window = 0; window < 3; ++window)
    h::enqueue_syndromes(0, zeros.data(), n, 0);
  std::vector<uint8_t> corr(1, 0xff);
  h::get_corrections(0, corr.data(), 1, true);
  EXPECT_EQ(corr[0] & 1u, 0u);
  cfg::finalize_decoders();

  // Fed raw measurements directly, the ensemble has no windows to forward and
  // the server, which only streams, rejects the graph.
  auto raw = cfg::multi_decoder_config::from_yaml_str(ensemble_yaml(R"(
        - id: ens
          ensemble:
            join_strategy: first
            members:
              - nodes: [{id: a, type: single_window, inner: {type: single_error_lut}}]
                edges: [{from: input, to: a}, {from: a, to: output}]
)"));
  EXPECT_NE(cfg::configure_decoders(raw), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, EnsembleYamlRejectsLoweredKindsAndBadPatterns) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const char *needle) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      FAIL() << "expected rejection mentioning '" << needle << "'";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
          << e.what();
    }
  };
  // The kinds an ensemble lowers to are not a `type` a user writes.
  rejects(ensemble_yaml(R"(
        - {id: fan, type: ensemble_fanout}
)"),
          "write an 'ensemble' entry instead");
  // Nor is a selection strategy a decoder type.
  rejects(ensemble_yaml(R"(
        - {id: w, type: single_window, inner: {type: deadline, deadline_us: 5}}
)"),
          "no parameter schema is registered");
  rejects(ensemble_yaml(R"(
        - id: ens
          ensemble:
            join_strategy: best
            members:
              - nodes: [{id: a, type: single_window, inner: {type: single_error_lut}}]
                edges: [{from: input, to: a}, {from: a, to: output}]
)"),
          "not a registered selection strategy");
  rejects(ensemble_yaml(R"(
        - id: ens
          type: xor
          ensemble:
            join_strategy: first
            members:
              - nodes: [{id: a, type: single_window, inner: {type: single_error_lut}}]
                edges: [{from: input, to: a}, {from: a, to: output}]
)"),
          "takes no type");
  rejects(ensemble_yaml(R"(
        - id: ens
          ensemble:
            join_strategy: deadline
            members:
              - nodes: [{id: a, type: single_window, inner: {type: single_error_lut}}]
                edges: [{from: input, to: a}, {from: a, to: output}]
)"),
          "deadline_us");
  // Member wiring is the builder's to check.
  auto unwired = cfg::multi_decoder_config::from_yaml_str(ensemble_yaml(R"(
        - id: ens
          ensemble:
            join_strategy: first
            members:
              - nodes: [{id: a, type: single_window, inner: {type: single_error_lut}}]
)"));
  try {
    cudaq::qec::decoding::host::build_task_graph(unwired.decoders.at(0));
    FAIL() << "expected the builder to reject a member without edges";
  } catch (const std::exception &e) {
    EXPECT_NE(std::string(e.what()).find("has no edges"), std::string::npos)
        << e.what();
  }
}

// ---------------------------------------------------------------------------
// combine() — two-decoder DAG with xor_node
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, CombineZeroSyndromesZeroCorrections) {
  // Two identical graphs combined: XOR of two equal outcomes == 0.
  const auto spec = rep5_spec(5);
  auto left = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right = decoding_task_graph::from_spec(spec, "single_error_lut");

  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  ASSERT_NE(combined, nullptr);

  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);
  const auto lo = combined->run(m);
  EXPECT_TRUE(lo.converged);
  for (auto bit : lo.bits)
    EXPECT_EQ(bit, 0u);
}

TEST(DecodingTaskGraph, CombineMatchesXorOfSingles) {
  // run() on the combined graph must equal xor_logical(left.run(),
  // right.run()).
  const auto spec = rep5_spec(5);
  auto left_a = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right_a = decoding_task_graph::from_spec(spec, "single_error_lut");
  // Independent copies for the single-run reference.
  auto left_b = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right_b = decoding_task_graph::from_spec(spec, "single_error_lut");

  auto combined =
      decoding_task_graph::combine(std::move(left_a), std::move(right_a));

  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);

  const auto lo_combined = combined->run(m);
  const auto lo_left = left_b->run(m);
  const auto lo_right = right_b->run(m);
  const auto lo_xor = xor_logical(lo_left, lo_right);

  ASSERT_EQ(lo_combined.bits.size(), lo_xor.bits.size());
  for (std::size_t i = 0; i < lo_combined.bits.size(); ++i)
    EXPECT_EQ(lo_combined.bits[i], lo_xor.bits[i]);
  EXPECT_EQ(lo_combined.converged, lo_xor.converged);
}

TEST(DecodingTaskGraph, CombineStreamingEnqueuesBothNodes) {
  // enqueue_syndrome on a combined graph must fire when both children fire,
  // and get_obs_corrections must return XOR of the two children's corrections.
  const auto spec = rep5_spec(5);
  auto left_a = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right_a = decoding_task_graph::from_spec(spec, "single_error_lut");
  // Reference singles for the XOR check.
  auto left_b = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right_b = decoding_task_graph::from_spec(spec, "single_error_lut");

  auto combined =
      decoding_task_graph::combine(std::move(left_a), std::move(right_a));

  const std::size_t n = combined->get_num_msyn_per_decode();
  std::vector<uint8_t> zeros(n, 0);

  EXPECT_TRUE(combined->enqueue_syndrome(zeros.data(), n));
  EXPECT_TRUE(left_b->enqueue_syndrome(zeros.data(), n));
  EXPECT_TRUE(right_b->enqueue_syndrome(zeros.data(), n));

  const uint8_t *corr_combined = combined->get_obs_corrections();
  const uint8_t *corr_left = left_b->get_obs_corrections();
  const uint8_t *corr_right = right_b->get_obs_corrections();

  const std::size_t n_obs = combined->get_num_observables();
  for (std::size_t i = 0; i < n_obs; ++i)
    EXPECT_EQ(corr_combined[i], uint8_t(corr_left[i] ^ corr_right[i]));
}

TEST(DecodingTaskGraph, CombineNullThrows) {
  auto graph = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  EXPECT_THROW(decoding_task_graph::combine(nullptr, std::move(graph)),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, CombineDifferentDMatricesMatchesSingles) {
  // combine() must route the raw shot through each branch's own D matrix.
  const auto inputs = make_rep3_inputs();
  auto right_d = inputs.d;
  ASSERT_FALSE(right_d.empty());
  right_d[0].clear();

  auto left_for_combined = inputs.graph();
  auto right_for_combined = inputs.graph_with_d(right_d);
  auto combined = decoding_task_graph::combine(
      std::make_unique<decoding_task_graph>(std::move(left_for_combined)),
      std::make_unique<decoding_task_graph>(std::move(right_for_combined)));

  auto left_reference = inputs.graph();
  auto right_reference = inputs.graph_with_d(right_d);
  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 1);

  const auto expected =
      xor_logical(left_reference.run(m), right_reference.run(m));
  const auto actual = combined->run(m);
  EXPECT_EQ(actual.converged, expected.converged);
  EXPECT_EQ(actual.bits, expected.bits);
}

TEST(DecodingTaskGraph, CombinedGraphOMatrixThrows) {
  auto left = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto right = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));

  EXPECT_THROW(combined->o_matrix(), std::logic_error);
}

TEST(DecodingTaskGraph, CombineRejectsMultiNodeGraph) {
  auto left = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto right = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  auto single =
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");

  EXPECT_THROW(
      decoding_task_graph::combine(std::move(combined), std::move(single)),
      std::invalid_argument);
}

static void expect_batch_valid_streaming_conflict(
    std::unique_ptr<decoding_task_graph> graph,
    const std::string &diagnostic_fragment);

TEST(DecodingTaskGraph, CombineDefersDifferentWindowsToStreaming) {
  const auto inputs = make_rep3_inputs();
  auto wider_d = inputs.d;
  ASSERT_FALSE(wider_d.empty());
  wider_d[0].push_back(3);

  auto left = inputs.graph();
  auto right = inputs.graph_with_d(wider_d);
  auto left_reference = inputs.graph();
  auto right_reference = inputs.graph_with_d(wider_d);
  auto combined = decoding_task_graph::combine(
      std::make_unique<decoding_task_graph>(std::move(left)),
      std::make_unique<decoding_task_graph>(std::move(right)));

  // Four is the wider branch's complete one-shot measurement width. Use
  // nonzero data and compare with independently executed branches: merely
  // running a largest-streaming-window-sized zero shot would not prove that
  // the batch DAG routes each branch through its own D.
  measurement_results shot;
  shot.bits = {1, 0, 1, 1};
  const auto expected =
      xor_logical(left_reference.run(shot), right_reference.run(shot));
  const auto actual = combined->run(shot);
  EXPECT_EQ(actual.bits, expected.bits);
  EXPECT_EQ(actual.converged, expected.converged);

  expect_batch_valid_streaming_conflict(std::move(combined), "window_size");
}

TEST(DecodingTaskGraph, CombineRejectsDifferentObservableCounts) {
  // Two branches XORing into one output must be the same width.  This built
  // fine and failed later: compile_wires() only *records* a width mismatch, to
  // keep a from_topology DAG usable as a batch program, and combine() has no
  // batch program to keep -- run() throws from xor_logical.
  const auto inputs = make_rep3_inputs();
  auto wider_o = inputs.o;
  ASSERT_FALSE(wider_o.empty());
  wider_o.push_back(wider_o.front());

  auto left = inputs.graph();
  auto right = inputs.graph_with_o(wider_o);
  try {
    (void)decoding_task_graph::combine(
        std::make_unique<decoding_task_graph>(std::move(left)),
        std::make_unique<decoding_task_graph>(std::move(right)));
    FAIL() << "expected combine to reject unequal observable counts";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("observables"), std::string::npos)
        << e.what();
  }
}

// ---------------------------------------------------------------------------
// Streaming edge routing — window_node::streaming_output / enqueue_edge_value
// ---------------------------------------------------------------------------

// Build a standalone window_node from rep3 inputs for direct method testing.
static std::unique_ptr<window_node>
make_rep3_window_node(const rep3_inputs &inp, window_spec ws, uint32_t spr) {
  auto d_mat = make_csr(inp.d, "D");
  auto o_mat = make_csr(inp.o, "O");
  auto d_ptr = std::make_shared<sparse_binary_matrix>(std::move(d_mat));
  auto o_ptr = std::make_shared<sparse_binary_matrix>(std::move(o_mat));
  cudaqx::heterogeneous_map params;
  auto dec =
      get_decoder("single_error_lut", inp.closed.detector_error_matrix, params);
  return std::make_unique<window_node>(d_ptr, o_ptr, std::move(dec), spr, ws);
}

TEST(DecodingTaskGraph, WindowNodeExposesItsCapabilities) {
  const auto inp = make_rep3_inputs();
  const auto n = static_cast<uint32_t>(make_csr(inp.d, "D").num_cols());
  auto node = make_rep3_window_node(inp, {n, n, 0u, 0u}, 1u);
  task_graph_node *core = node.get();

  EXPECT_EQ(core->streaming(), node.get());
  EXPECT_EQ(core->decoder_backed(), node.get());
}

TEST(DecodingTaskGraph, WindowNodeDeclaresPortTypes) {
  // The output contract is path-independent. The input accepts either form
  // supported by execute_indexed(), while only slot 0 supports edge-fed
  // streaming.
  const auto inp = make_rep3_inputs();
  auto d_mat = make_csr(inp.d, "D");
  const auto n_cols = static_cast<uint32_t>(d_mat.num_cols());
  auto node = make_rep3_window_node(inp, {1u, 1u, 0u, 0u}, n_cols);

  EXPECT_EQ(node->output_type(0), edge_value_type("logical_outcome"));
  EXPECT_FALSE(node->output_type(1).has_value());
  EXPECT_TRUE(
      node->accepts_input_type(0, edge_value_type("measurement_results")));
  EXPECT_TRUE(node->accepts_input_type(0, edge_value_type("detection_events")));
  EXPECT_FALSE(node->accepts_input_type(0, edge_value_type("logical_outcome")));
  EXPECT_TRUE(node->supports_raw_streaming_input(0));
  EXPECT_FALSE(node->supports_raw_streaming_input(1));
  EXPECT_TRUE(node->supports_streaming_input(0));
  EXPECT_FALSE(node->supports_streaming_input(1));
}

TEST(DecodingTaskGraph, EnqueueEdgeValueWrongTypeThrows) {
  // Passing a logical_outcome to a slot that expects detection_events must
  // throw so that callers discover mismatch immediately rather than silently
  // producing no output.
  const auto inp = make_rep3_inputs();
  auto d_mat = make_csr(inp.d, "D");
  const auto n_cols = static_cast<uint32_t>(d_mat.num_cols());
  auto node = make_rep3_window_node(inp, {1u, 1u, 0u, 0u}, n_cols);

  logical_outcome lo;
  lo.bits = {0u};
  EXPECT_THROW(node->enqueue_edge_value(edge_value{lo}, 0),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, EnqueueEdgeValueWrongSlotThrows) {
  const auto inp = make_rep3_inputs();
  auto d_mat = make_csr(inp.d, "D");
  const auto n_dets = static_cast<uint32_t>(d_mat.num_rows());
  const auto n_cols = static_cast<uint32_t>(d_mat.num_cols());
  auto node = make_rep3_window_node(inp, {1u, 1u, 0u, 0u}, n_cols);

  detection_events dets;
  dets.bits.assign(n_dets, 0.0f);
  EXPECT_THROW(node->enqueue_edge_value(edge_value{dets}, 1),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, StreamingOutputNulloptBeforeFire) {
  const auto inp = make_rep3_inputs();
  const auto n = static_cast<uint32_t>(make_csr(inp.d, "D").num_cols());
  auto node = make_rep3_window_node(inp, {n, n, 0u, 0u}, 1u);
  EXPECT_EQ(node->streaming_output(0), nullptr);
  EXPECT_EQ(node->streaming_output(1), nullptr); // always nullptr
}

TEST(DecodingTaskGraph, StreamingOutputAfterRawFire) {
  const auto inp = make_rep3_inputs();
  const auto n = static_cast<uint32_t>(make_csr(inp.d, "D").num_cols());
  auto node = make_rep3_window_node(inp, {n, n, 0u, 0u}, 1u);

  std::vector<uint8_t> zero_syndrome(n, 0);
  EXPECT_TRUE(node->enqueue(zero_syndrome.data(), n, round_type::BULK));

  const auto *val = node->streaming_output(0);
  ASSERT_NE(val, nullptr);
  const auto &lo = val->get<logical_outcome>();
  EXPECT_TRUE(lo.converged);
  for (auto bit : lo.bits)
    EXPECT_EQ(bit, 0u);
}

TEST(DecodingTaskGraph, EnqueueEdgeValueFiresAndProducesCorrections) {
  // Single-round window (ws.size=1): fires on the first detection-event frame.
  // The rep3 decoder is dimensioned for n_dets inputs — matches det_buf_ size.
  const auto inp = make_rep3_inputs();
  auto d_mat = make_csr(inp.d, "D");
  const auto n_dets = static_cast<uint32_t>(d_mat.num_rows());
  const auto n_cols = static_cast<uint32_t>(d_mat.num_cols());
  auto node = make_rep3_window_node(inp, {1u, 1u, 0u, 0u}, n_cols);

  detection_events zero_dets;
  zero_dets.bits.assign(n_dets, 0.0f);

  // Before any call: no output.
  EXPECT_EQ(node->streaming_output(0), nullptr);
  EXPECT_EQ(node->streaming_output(1), nullptr); // always nullptr

  // Fire: zero detectors → zero corrections.
  EXPECT_TRUE(node->enqueue_edge_value(edge_value{zero_dets}, 0));

  const auto *out = node->streaming_output(0);
  ASSERT_NE(out, nullptr);
  const auto &lo = out->get<logical_outcome>();
  EXPECT_TRUE(lo.converged);
  for (auto bit : lo.bits)
    EXPECT_EQ(bit, 0u);
  for (std::size_t i = 0; i < node->num_observables(); ++i)
    EXPECT_EQ(node->corrections()[i], 0u);
}

TEST(DecodingTaskGraph, EnqueueEdgeValueRejectsWrongDetectorCount) {
  const auto inp = make_rep3_inputs();
  auto d_mat = make_csr(inp.d, "D");
  const auto n_cols = static_cast<uint32_t>(d_mat.num_cols());
  auto node = make_rep3_window_node(inp, {1u, 1u, 0u, 0u}, n_cols);

  detection_events dets;
  dets.bits.assign(d_mat.num_rows() + 1, 0.0f);
  EXPECT_THROW(node->enqueue_edge_value(edge_value{dets}, 0),
               std::out_of_range);
}

// (EnqueueEdgeValueWrongSlotThrows above supersedes this; removed.)

TEST(DecodingTaskGraph, EnqueueEdgeValueResetClearsLastLo) {
  // Fire once, reset, and verify the cached streaming output is invalidated.
  const auto inp = make_rep3_inputs();
  auto d_mat = make_csr(inp.d, "D");
  const auto n_dets = static_cast<uint32_t>(d_mat.num_rows());
  const auto n_cols = static_cast<uint32_t>(d_mat.num_cols());
  auto node = make_rep3_window_node(inp, {1u, 1u, 0u, 0u}, n_cols);

  detection_events zero_dets;
  zero_dets.bits.assign(n_dets, 0.0f);
  EXPECT_TRUE(node->enqueue_edge_value(edge_value{zero_dets}, 0));
  EXPECT_NE(node->streaming_output(0), nullptr);

  node->reset_decoder();
  EXPECT_EQ(node->streaming_output(0), nullptr);

  // After reset the node is ready to fire again from scratch.
  EXPECT_TRUE(node->enqueue_edge_value(edge_value{zero_dets}, 0));
  EXPECT_NE(node->streaming_output(0), nullptr);
}

// A `rounds`-round window over identity D/O/H with a real decoder, so a
// measurement byte IS a detector bit and the same data can be delivered either
// as raw bytes or as a routed detection_events value. Same shape as
// make_guard_probe below, but input-sensitive: all_faults_decoder ignores its
// syndrome, which would make the equivalence test below vacuous.
static std::unique_ptr<window_node> make_identity_window(window_spec ws,
                                                         uint32_t rounds) {
  std::vector<std::vector<uint32_t>> identity(rounds);
  for (uint32_t i = 0; i < rounds; ++i)
    identity[i].push_back(i);
  auto h = sparse_binary_matrix::from_nested_csr(rounds, rounds, identity);
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(rounds, rounds, identity));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(rounds, rounds, identity));
  return std::make_unique<window_node>(
      d, o, get_decoder("single_error_lut", h, {}), /*spr=*/1, ws);
}

// Returns a fixed observable frame and declares it, standing in for any inner
// decoder that predicts observables rather than fault mechanisms.
class obs_projecting_decoder final : public decoder {
public:
  obs_projecting_decoder(sparse_binary_matrix h, std::vector<float_t> obs,
                         bool declare_obs)
      : decoder(make_inputs(std::move(h), obs.size()),
                declare_obs ? decode_result_type::observables
                            : decode_result_type::errors),
        obs_(std::move(obs)) {}
  decoder_result decode(const std::vector<float_t> &) override {
    return decoder_result{true, obs_};
  }

private:
  static decoder_init make_inputs(sparse_binary_matrix h,
                                  std::size_t num_observables) {
    const auto num_errors = h.num_cols();
    std::vector<std::vector<uint32_t>> empty_rows(num_observables);
    auto o = sparse_binary_matrix::from_nested_csr(num_observables, num_errors,
                                                   empty_rows);
    return decoder_init(std::move(h), std::move(o));
  }

  std::vector<float_t> obs_;
};

// One-observable window whose inner decoder projects to observables itself.
static std::unique_ptr<window_node>
make_obs_window(std::vector<float_t> obs, bool declare_obs, uint32_t rounds) {
  std::vector<std::vector<uint32_t>> identity(rounds);
  for (uint32_t i = 0; i < rounds; ++i)
    identity[i].push_back(i);
  auto h = sparse_binary_matrix::from_nested_csr(rounds, rounds, identity);
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(rounds, rounds, identity));
  // O has `rounds` fault columns but one observable row, so an observable
  // frame and a fault-column vector are different lengths -- which is what
  // makes the decode_to_errs half of the test below fail loudly.
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(
          1, rounds, std::vector<std::vector<uint32_t>>{{0}}));
  // The decoder keeps H's fault-column block_size and only switches its result
  // vector to observables. That is the convention in which block_size does NOT
  // coincide with the observable count; a fixture built the other way -- one
  // that rebuilds block_size as the observable count -- would hide the
  // difference and let a block_size-based check look correct.
  return std::make_unique<window_node>(
      d, o,
      std::make_unique<obs_projecting_decoder>(h, std::move(obs), declare_obs),
      /*spr=*/1, window_spec{rounds, rounds, 0, 0});
}

TEST(DecodingTaskGraph, WindowNodeUsesAnObservableProjectingDecoderDirectly) {
  // An observable-projecting decoder returns observables, not fault columns.
  // The node takes that frame as it stands; indexing it by O's column ids
  // would throw require_width and make such a decoder unusable as an inner
  // decoder.
  auto node = make_obs_window({1.0f}, /*declare_obs=*/true, 3);
  const uint8_t zeros[3] = {0, 0, 0};
  ASSERT_TRUE(node->enqueue(zeros, 3, round_type::BULK));
  EXPECT_EQ(node->streaming_output(0)->get<logical_outcome>().bits,
            (std::vector<uint8_t>{1}));

  // Not vacuous: the identical decoder that does NOT declare decode_to_obs is
  // still rejected, so it is the declaration doing the work here, not a
  // widened width check.
  auto liar = make_obs_window({1.0f}, /*declare_obs=*/false, 3);
  EXPECT_THROW(liar->enqueue(zeros, 3, round_type::BULK), std::out_of_range);
}

TEST(DecodingTaskGraph, ObservableProjectingDecoderWorksOnOneShotPaths) {
  // The one-shot paths have to read an observable frame the same way the
  // streaming fire does.  Projecting it through O unconditionally would make
  // `run()` throw for any node whose O has more fault columns than the decoder
  // has observables -- which is every observable-projecting node.
  auto node = make_obs_window({1.0f}, /*declare_obs=*/true, 3);
  ASSERT_GT(node->o().num_cols(), 1u) << "O must be wider than the frame, or "
                                         "this test cannot see the bug";

  measurement_results m;
  m.bits.assign(3, 0);
  std::vector<edge_value> in{edge_value{m}}, out(1);
  node->execute_indexed(in, out);
  EXPECT_EQ(out[0].get<logical_outcome>().bits, (std::vector<uint8_t>{1}));

  EXPECT_EQ(node->run_async(m).get().bits, (std::vector<uint8_t>{1}));
}

// ---------------------------------------------------------------------------
// predecoder_node
// ---------------------------------------------------------------------------

// Returns a caller-chosen error pattern. all_faults_decoder is the
// "everything" special case; this is the general one, and it is what makes the
// residual hand-computable.
class fixed_error_decoder final : public decoder {
public:
  fixed_error_decoder(
      sparse_binary_matrix h, std::vector<float_t> e, bool converged,
      std::optional<std::vector<float_t>> residual = std::nullopt,
      decoder_output_request output = {decode_result_type::errors})
      : decoder(decoder_init(std::move(h)), output), e_(std::move(e)),
        converged_(converged), residual_(std::move(residual)) {}
  decoder_result decode(const std::vector<float_t> &) override {
    decoder_result result{converged_, e_};
    if (residual_) {
      cudaqx::heterogeneous_map auxiliary;
      auxiliary.insert("residual_detectors", *residual_);
      result.opt_results = std::move(auxiliary);
    }
    return result;
  }

private:
  std::vector<float_t> e_;
  bool converged_;
  std::optional<std::vector<float_t>> residual_;
};

// A 4-detector, 5-fault chain: detector i is faults {i, i+1}. Only fault 0
// flips the observable, so O and H disagree about which columns matter --
// which is the point, since the residual uses H and the commit uses O.
static std::shared_ptr<const sparse_binary_matrix> chain_h() {
  return std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}));
}

using residual_src = predecoder_node::residual_source;

static std::unique_ptr<predecoder_node> make_chain_predecoder(
    std::vector<float_t> e, bool converged = true,
    residual_src src = residual_src::parity_check,
    std::optional<std::vector<float_t>> residual = std::nullopt) {
  auto h = chain_h();
  auto d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 5, {{0}}));
  auto dec = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}),
      std::move(e), converged, std::move(residual),
      src == residual_src::model
          ? decoder_output_request{decode_result_type::observables,
                                   decoder_auxiliary_result_type::
                                       residual_detectors}
          : decoder_output_request{decode_result_type::errors});
  return std::make_unique<predecoder_node>(src, d, o, h, std::move(dec),
                                           /*spr=*/4, /*rounds=*/1);
}

TEST(DecodingTaskGraph, PredecoderDeclaresPreLAndResidualPorts) {
  auto node = make_chain_predecoder({0, 0, 0, 0, 0});
  EXPECT_EQ(node->inputs(), (std::vector<std::string>{"input"}));
  // The order IS the slot numbering, so pin the index and not just the names:
  // `{from: pre.residual}` resolves to slot 1 through exactly this.
  ASSERT_EQ(node->outputs(), (std::vector<std::string>{"pre_L", "residual"}));
  EXPECT_EQ(node->output_type(0), edge_value_type("logical_outcome"));
  EXPECT_EQ(node->output_type(1), edge_value_type("detection_events"));
  EXPECT_FALSE(node->output_type(2).has_value());
  EXPECT_TRUE(
      node->accepts_input_type(0, edge_value_type("measurement_results")));
  EXPECT_TRUE(node->supports_raw_streaming_input(0));
  EXPECT_FALSE(node->supports_raw_streaming_input(1));
  // Raw-fed: an edge into it is a build-time error.
  EXPECT_FALSE(node->supports_streaming_input(0));
}

TEST(DecodingTaskGraph, PredecoderResidualIsTheUnexplainedSyndrome) {
  // The crux. Predicting fault 0 explains detector 0 (H's column 0 is {D0}),
  // so that detector leaves the syndrome and the rest stays for whoever
  // decodes the residual. The commit is O*e, which is a different matrix.
  auto node = make_chain_predecoder({1, 0, 0, 0, 0});
  const uint8_t msyn[4] = {1, 0, 0, 1}; // s = D0, D3
  ASSERT_TRUE(node->enqueue(msyn, 4, round_type::BULK));

  // s XOR H*e = (1,0,0,1) XOR (1,0,0,0) = (0,0,0,1)
  EXPECT_EQ(node->streaming_output(1)->get<detection_events>().bits,
            (std::vector<float_t>{0, 0, 0, 1}));
  EXPECT_EQ(node->streaming_output(0)->get<logical_outcome>().bits,
            (std::vector<uint8_t>{1}));

  // Break-it: this input is chosen so H*e is neither 0 nor s, and O*e is
  // nonzero, so each of `residual = s`, `residual = H*e` and `residual = 0`
  // fails, as does a commit hardwired to zero.
  ASSERT_NE((std::vector<float_t>{0, 0, 0, 1}),
            (std::vector<float_t>{1, 0, 0, 1}));
}

TEST(DecodingTaskGraph, PredecoderAbstainsWhenTheDecodeDoesNotConverge) {
  // A non-zero prediction that did not converge must not be committed: the
  // whole syndrome goes downstream untouched instead. The LUT decoders return
  // zeros here, but a BP backend may return a best-effort vector, and
  // committing that would leave a correction nothing undoes.
  auto node = make_chain_predecoder({1, 1, 1, 1, 1}, /*converged=*/false);
  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(node->enqueue(msyn, 4, round_type::BULK));

  EXPECT_EQ(node->streaming_output(1)->get<detection_events>().bits,
            (std::vector<float_t>{1, 0, 0, 1})); // s, unchanged
  EXPECT_EQ(node->streaming_output(0)->get<logical_outcome>().bits,
            (std::vector<uint8_t>{0}));
  EXPECT_EQ(node->corrections()[0], 0u);
  // xor_logical ANDs the flags, so forwarding the inner decoder's false here
  // would mark every routine abstention as a failed decode.
  EXPECT_TRUE(node->streaming_output(0)->get<logical_outcome>().converged);
}

TEST(DecodingTaskGraph, PredecoderOutputsAreNullBeforeFireAndAfterReset) {
  auto node = make_chain_predecoder({1, 0, 0, 0, 0});
  EXPECT_EQ(node->streaming_output(0), nullptr);
  EXPECT_EQ(node->streaming_output(1), nullptr);

  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(node->enqueue(msyn, 4, round_type::BULK));
  EXPECT_NE(node->streaming_output(0), nullptr);
  EXPECT_NE(node->streaming_output(1), nullptr);

  node->reset_decoder();
  EXPECT_EQ(node->streaming_output(0), nullptr);
  EXPECT_EQ(node->streaming_output(1), nullptr);
}

TEST(DecodingTaskGraph, PredecoderResidualIsHardValued) {
  // detection_events carries no prob/LLR tag, so the port contract is hard
  // 0/1: a soft rule would be silently wrong under the other convention.
  auto node = make_chain_predecoder({1, 0, 0, 0, 0});
  const uint8_t msyn[4] = {1, 1, 0, 1};
  ASSERT_TRUE(node->enqueue(msyn, 4, round_type::BULK));
  for (auto v : node->streaming_output(1)->get<detection_events>().bits)
    EXPECT_TRUE(v == 0.0f || v == 1.0f) << v;
}

TEST(DecodingTaskGraph, PredecoderBatchPathFillsBothSlots) {
  auto node = make_chain_predecoder({1, 0, 0, 0, 0});
  measurement_results m;
  m.bits = {1, 0, 0, 1};
  std::vector<edge_value> in{edge_value{m}}, out(2);
  node->execute_indexed(in, out);
  EXPECT_EQ(out[0].get<logical_outcome>().bits, (std::vector<uint8_t>{1}));
  EXPECT_EQ(out[1].get<detection_events>().bits,
            (std::vector<float_t>{0, 0, 0, 1}));
}

TEST(DecodingTaskGraph, PredecoderKindIsVisibleAndBuildable) {
  const auto kinds = registered_task_graph_node_kinds();
  cudaqx::heterogeneous_map model_params;
  cudaqx::heterogeneous_map parity_params;
  const task_graph_node_kind_spec *found =
      find_task_graph_node_kind("PreDecoder");
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  std::unique_ptr<decoding_task_graph> graph;

  EXPECT_NE(std::find(kinds.begin(), kinds.end(), "predecoder"), kinds.end());
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name, "predecoder");
  model_params.insert("residual_from", std::string("model"));
  parity_params.insert("residual_from", std::string("parity_check"));
  const auto model_output = found->inner_decoder_output(model_params);
  EXPECT_EQ(model_output.primary, decode_result_type::observables);
  EXPECT_TRUE(
      model_output.requests(decoder_auxiliary_result_type::residual_detectors));
  const auto parity_output = found->inner_decoder_output(parity_params);
  EXPECT_EQ(parity_output.primary, decode_result_type::errors);
  EXPECT_FALSE(parity_output.has_auxiliary_outputs());
  EXPECT_TRUE(found->inner_decoder_takes_observables(model_params));
  EXPECT_FALSE(found->inner_decoder_takes_observables(parity_params));
  // Negative lookup, so the alias assertion cannot be passing because every
  // name resolves.
  EXPECT_EQ(find_task_graph_node_kind("predecoders"), nullptr);

  node.id = "pre";
  node.kind = "predecoder";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  node.node_params.insert("residual_from", std::string("parity_check"));
  nodes.push_back(std::move(node));
  graph = decoding_task_graph::from_topology(std::move(nodes));
  EXPECT_GT(graph->get_num_msyn_per_decode(), 0u);
  EXPECT_GT(graph->get_num_observables(), 0u);
}

TEST(DecodingTaskGraph, OutputDefaultsToTheObservableModelPolicy) {
  task_graph_node_kind_spec with_observables;
  task_graph_node_kind_spec without_observables;
  cudaqx::heterogeneous_map params;

  without_observables.inner_decoder_observable_policy =
      [](const cudaqx::heterogeneous_map &) { return false; };

  EXPECT_EQ(with_observables.inner_decoder_output(params).primary,
            decode_result_type::observables);
  EXPECT_EQ(without_observables.inner_decoder_output(params).primary,
            decode_result_type::errors);
  EXPECT_FALSE(
      with_observables.inner_decoder_output(params).has_auxiliary_outputs());
  EXPECT_FALSE(
      without_observables.inner_decoder_output(params).has_auxiliary_outputs());
}

TEST(DecodingTaskGraph, PredecoderParamsAreSchemaChecked) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto *schema =
      cfg::find_decoder_schema(task_graph_node_schema_name("predecoder"));
  ASSERT_NE(schema, nullptr);

  std::vector<std::string> keys;
  for (const auto &p : schema->params)
    keys.push_back(p.key);
  std::sort(keys.begin(), keys.end());
  // Exactly the keys it accepts. A pre-decoder covers its whole DEM and empties
  // on every fire, so it declares no step_size and no guards -- naming one is
  // then an unknown-key error rather than a special case in a validator.
  EXPECT_EQ(keys, (std::vector<std::string>{"num_syndromes_per_round",
                                            "residual_from", "window_size"}));

  cudaqx::heterogeneous_map good;
  good.insert("window_size", std::size_t{5});
  good.insert("residual_from", std::string("parity_check"));
  EXPECT_NO_THROW(cfg::validate_custom_args(
      task_graph_node_schema_name("predecoder"), good));

  cudaqx::heterogeneous_map bad;
  bad.insert("window_size", std::size_t{0});
  EXPECT_THROW(
      cfg::validate_custom_args(task_graph_node_schema_name("predecoder"), bad),
      std::runtime_error);
}

TEST(DecodingTaskGraph, PredecoderStreamingPointersAreStableAcrossFires) {
  // Both slots are refilled in place, so the addresses hold while the contents
  // change. Asserting only the addresses would pass for a node that never
  // refills; asserting only the contents would pass for one that reallocates
  // every fire and cannot promise validity until the next mutating call.
  auto node = make_chain_predecoder({1, 0, 0, 0, 0});
  const uint8_t first[4] = {1, 0, 0, 1};
  ASSERT_TRUE(node->enqueue(first, 4, round_type::BULK));
  const edge_value *lo0 = node->streaming_output(0);
  const edge_value *res0 = node->streaming_output(1);
  const auto residual_first = res0->get<detection_events>().bits;

  const uint8_t second[4] = {0, 1, 1, 0};
  ASSERT_TRUE(node->enqueue(second, 4, round_type::BULK));
  EXPECT_EQ(node->streaming_output(0), lo0);
  EXPECT_EQ(node->streaming_output(1), res0);
  // ...and the residual actually tracked the new shot, so the pointers are not
  // stable merely because nothing was written.
  EXPECT_NE(res0->get<detection_events>().bits, residual_first);
}

TEST(DecodingTaskGraph, PredecoderRejectsAnObservableProjectingInnerDecoder) {
  // The residual is `s XOR H*e` over fault columns; an observable frame is a
  // different space, and many error patterns share one observable outcome, so
  // the residual is undetermined rather than merely awkward.
  auto h = sparse_binary_matrix::from_nested_csr(
      4, 5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  task_graph_node_config node;
  node.id = "pre";
  node.kind = "predecoder";
  node.dem = rep5_spec(5);
  node.decoder_instance = std::make_unique<obs_projecting_decoder>(
      h, std::vector<float_t>{1.0f}, /*declare_obs=*/true);
  node.node_params.insert("residual_from", std::string("parity_check"));
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  try {
    decoding_task_graph::from_topology(std::move(nodes));
    FAIL() << "expected an observable-projecting decoder to be rejected";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("observables"), std::string::npos)
        << e.what();
  }
}

// Records what model inputs a build path handed an inner decoder.
struct probe_params_record {
  bool had_O = false;
  bool had_error_rate_vec = false;
  decoder_output_request output{decode_result_type::errors};
};
static std::vector<probe_params_record> probe_records;

class params_probe_decoder : public decoder {
public:
  params_probe_decoder(decoder_init inputs, decoder_output_request output,
                       const cudaqx::heterogeneous_map &params)
      : decoder(inputs, output) {
    (void)params;
    probe_records.push_back(
        {inputs.has_observable_model(), !inputs.error_rates().empty(), output});
  }
  decoder_result decode(const std::vector<float_t> &) override {
    decoder_result r;
    r.converged = true;
    r.result.assign(block_size, 0.0f);
    return r;
  }
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      params_probe_decoder,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decoder_output_request> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<params_probe_decoder>(
            std::move(inputs),
            output.value_or(decoder_output_request{decode_result_type::errors}),
            params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(params_probe_decoder)

// Refuse a flattened model so a native-source handoff regression fails at
// construction, even when both representations give identical finite results.
class chunk_source_probe_decoder : public decoder {
public:
  chunk_source_probe_decoder(decoder_init inputs, decoder_output_request output,
                             const cudaqx::heterogeneous_map &params)
      : decoder(inputs, output) {
    if (!inputs.has_dem_chunks() || inputs.has_matrices())
      throw std::invalid_argument("native chunk recipe was flattened");
    if (inputs.dem_chunks().num_rounds !=
        params.get<std::size_t>("expected_rounds"))
      throw std::invalid_argument("native recipe lost effective window size");
    if (inputs.has_observable_model() !=
        (output.primary == decode_result_type::observables))
      throw std::invalid_argument("native recipe lost output policy");
  }
  decoder_result decode(const std::vector<float_t> &) override {
    const auto width = get_result_type() == decode_result_type::observables
                           ? get_num_observables()
                           : get_block_size();
    return decoder_result{true, std::vector<float_t>(width, 0)};
  }
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      chunk_source_probe_decoder,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decoder_output_request> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<chunk_source_probe_decoder>(
            std::move(inputs),
            output.value_or(decoder_output_request{decode_result_type::errors}),
            params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(chunk_source_probe_decoder)

TEST(DecodingTaskGraph, NativeChunkConsumersKeepRecipeAndOutputPolicy) {
  register_decoder_native_model_source("chunk_source_probe_decoder",
                                       decoder_model_source::dem_chunks);
  for (const auto kind : {"streaming", "sliding_window"}) {
    task_graph_node_config node;
    node.id = "native";
    node.kind = kind;
    node.dem = rep5_spec(5);
    node.decoder_type = "chunk_source_probe_decoder";
    node.params = {{"expected_rounds", std::size_t{5}}};
    if (node.kind == "streaming")
      node.node_params = {{"num_syndromes_per_round", std::size_t{4}}};
    else {
      node.dem = rep5_spec(7);
      node.node_params = {{"window_size", std::size_t{5}},
                          {"step_size", std::size_t{3}},
                          {"start_guard", std::size_t{1}},
                          {"end_guard", std::size_t{1}}};
    }
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(std::move(nodes), {});
    EXPECT_EQ(graph->run(measurement_results{std::vector<uint8_t>(20)}).bits,
              std::vector<uint8_t>{0});
  }
}

TEST(DecodingTaskGraph, NativeStreamingKeepsTheObservableProjection) {
  register_decoder_native_model_source("chunk_source_probe_decoder",
                                       decoder_model_source::dem_chunks);
  for (const std::size_t rounds : {3u, 4u, 7u, 1000u}) {
    const auto spec = rep5_spec(rounds);
    // Offline oracle only: graph construction must use the compact recipe.
    const auto chunks = dem_chunks_from_spec(spec);
    const auto expected_o = dem_chunks_to_o_sparse(chunks);
    const auto expected_d = dem_chunks_to_d_sparse(chunks);
    task_graph_node_config node;
    node.id = "native";
    node.kind = "streaming";
    node.dem = spec;
    node.decoder_type = "chunk_source_probe_decoder";
    node.params = {{"expected_rounds", rounds}};
    node.node_params = {{"num_syndromes_per_round", std::size_t{4}}};
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(std::move(nodes), {});
    EXPECT_EQ(graph->o_matrix().to_nested_csr(), expected_o);
    std::size_t observable_columns = 0;
    for (const auto &row : expected_o)
      for (const auto column : row)
        observable_columns =
            std::max(observable_columns, std::size_t{column} + 1);
    EXPECT_EQ(graph->o_matrix().num_cols(), observable_columns);
    EXPECT_EQ(graph->d_matrix().to_nested_csr(), expected_d);
    EXPECT_EQ(
        graph->run(measurement_results{std::vector<uint8_t>(4 * rounds)}).bits,
        std::vector<uint8_t>{0});
  }
}

static const bool probe_schema_registered = [] {
  decoding::config::register_decoder_schema({"params_probe_decoder", {}});
  return true;
}();

// A predecoder over the rep-5 DEM with a reporting inner decoder.
static std::unique_ptr<decoding_task_graph>
predecoder_probe_graph(cudaqx::heterogeneous_map inner_params) {
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config pre;
  pre.id = "pre";
  pre.kind = "predecoder";
  pre.dem = rep5_spec(5);
  pre.decoder_type = "params_probe_decoder";
  pre.params = std::move(inner_params);
  pre.node_params.insert("window_size", std::size_t{5});
  pre.node_params.insert("residual_from", std::string("parity_check"));
  nodes.push_back(std::move(pre));
  return decoding_task_graph::from_topology(std::move(nodes));
}

TEST(DecodingTaskGraph, PredecoderInnerNeverReceivesTheObservableModel) {
  // A window-kind inner gets the "O" default. Without this half, a predecoder
  // inner seeing no O could just as well be a DEM that carries none.
  probe_records.clear();
  {
    std::vector<task_graph_node_config> nodes;
    auto node = rep5_streaming_node();
    node.decoder_type = "params_probe_decoder";
    nodes.push_back(std::move(node));
    (void)decoding_task_graph::from_topology(std::move(nodes));
  }
  ASSERT_EQ(probe_records.size(), 1u);
  EXPECT_TRUE(probe_records[0].had_O);
  EXPECT_TRUE(probe_records[0].had_error_rate_vec);

  // A parity-check predecoder over the same DEM sees no O; the error rates
  // still reach it.
  probe_records.clear();
  (void)predecoder_probe_graph({});
  ASSERT_EQ(probe_records.size(), 1u);
  EXPECT_FALSE(probe_records[0].had_O);
  EXPECT_TRUE(probe_records[0].had_error_rate_vec);
}

TEST(DecodingTaskGraph, FlatConfigModelPredecoderReceivesEntryO) {
  namespace cfg = cudaq::qec::decoding::config;
  const std::string yaml = R"(
decoders:
  - id: 0
    block_size: 5
    syndrome_size: 4
    H_sparse: [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1, 2, -1, 3, -1]
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          residual_from: model
          inner: {type: params_probe_decoder}
      edges:
        - {from: measurements, to: pre}
        - {from: pre.pre_L, to: output}
)";

  probe_records.clear();
  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();

  ASSERT_EQ(probe_records.size(), 1u);
  EXPECT_TRUE(probe_records[0].had_O);
  EXPECT_EQ(probe_records[0].output.primary, decode_result_type::observables);
  EXPECT_TRUE(probe_records[0].output.requests(
      decoder_auxiliary_result_type::residual_detectors));
}

TEST(DecodingTaskGraph, EntryFlatPredecoderRunsWithPyMatching) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace host = cudaq::qec::decoding::host;
  const std::string yaml = R"(
decoders:
  - id: 0
    block_size: 2
    syndrome_size: 2
    H_sparse: [0, -1, 1, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1]
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          residual_from: parity_check
          inner: {type: pymatching}
      edges:
        - {from: measurements, to: pre}
        - {from: pre.pre_L, to: output}
)";
  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  auto graph = host::build_task_graph(parsed.decoders[0]);
  measurement_results shot;
  shot.bits = {1, 0};

  EXPECT_NE(graph->inner_decoder(), nullptr);
  EXPECT_FALSE(graph->supports_graph_dispatch());
  EXPECT_NO_THROW(graph->release_graph(nullptr));
  const auto outcome = graph->run(shot);
  EXPECT_TRUE(outcome.converged);
  EXPECT_EQ(outcome.bits, (std::vector<uint8_t>{1}));
  EXPECT_THROW((void)graph->run_async(shot), std::logic_error);
}

TEST(DecodingTaskGraph, NodeFlatPredecoderRunsWithPyMatching) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace host = cudaq::qec::decoding::host;
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          residual_from: parity_check
          inner: {type: pymatching}
          block_size: 2
          syndrome_size: 2
          H_sparse: [0, -1, 1, -1]
          O_sparse: [0, -1]
          D_sparse: [0, -1, 1, -1]
      edges:
        - {from: measurements, to: pre}
        - {from: pre.pre_L, to: output}
)";
  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  auto graph = host::build_task_graph(parsed.decoders[0]);
  measurement_results shot;
  shot.bits = {1, 0};

  const auto outcome = graph->run(shot);
  EXPECT_TRUE(outcome.converged);
  EXPECT_EQ(outcome.bits, (std::vector<uint8_t>{1}));
}

TEST(DecodingTaskGraph, PredecoderBuildsItsInnerOnRawDemText) {
  namespace cfg = cudaq::qec::decoding::config;

  // A predecoder reads an error pattern, so its stim_dem is the node's whole
  // model: H, O and the error rates come from the text, in one fault space, and
  // only D stays the node's. A decode_to_obs kind reads the same key as its
  // inner decoder's model alone, and states its own O.
  const std::string dem = "error(0.125) D0 L0\nerror(0.125) D1 L1\n"
                          "error(0.125) D2\nerror(0.125) D3\n";
  const auto quoted = [](const std::string &text) {
    std::string out;
    for (char c : text)
      out += (c == '\n') ? std::string("\\n") : std::string(1, c);
    return "\"" + out + "\"";
  };
  const auto doc = [&](const char *kind, const char *extra) {
    return R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: )" +
           std::string(kind) + R"(
          inner: {type: params_probe_decoder}
          stim_dem: )" +
           quoted(dem) + R"(
          D_sparse: [0, -1, 1, -1, 2, -1, 3, -1]
)" + extra +
           R"(      edges:
        - {from: measurements, to: pre}
        - {from: pre.pre_L, to: output}
)";
  };

  probe_records.clear();
  auto parsed = cfg::multi_decoder_config::from_yaml_str(
      doc("predecoder", "          residual_from: parity_check\n"));
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();

  // The inner is constructed on the text's H without O, which the node applies
  // itself; the error rates still reach it.
  ASSERT_EQ(probe_records.size(), 1u);
  EXPECT_FALSE(probe_records[0].had_O);
  EXPECT_TRUE(probe_records[0].had_error_rate_vec);

  // Model residuals add an auxiliary output, but the primary result remains
  // observable and is therefore valid with a stim_dem-owned fault space.
  probe_records.clear();
  parsed = cfg::multi_decoder_config::from_yaml_str(
      doc("predecoder", "          residual_from: model\n"
                        "          O_sparse: [0, -1, 1, -1]\n"));
  ASSERT_EQ(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();

  ASSERT_EQ(probe_records.size(), 1u);
  EXPECT_TRUE(probe_records[0].had_O);
  EXPECT_TRUE(probe_records[0].had_error_rate_vec);
  EXPECT_EQ(probe_records[0].output.primary, decode_result_type::observables);
  EXPECT_TRUE(probe_records[0].output.requests(
      decoder_auxiliary_result_type::residual_detectors));
}

TEST(DecodingTaskGraph, StimDemStillNeedsAnObservableProjectingInner) {
  namespace cfg = cudaq::qec::decoding::config;

  // The other direction. A kind taking the DEM's O reads stim_dem as its inner
  // decoder's model alone, so an error pattern from it is indexed in the text's
  // fault columns rather than the node's, and is refused.
  const std::string yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: solve
          type: single_window
          inner: {type: params_probe_decoder}
          stim_dem: "error(0.125) D0 L0\nerror(0.125) D1 L1\n"
          D_sparse: [0, -1, 1, -1]
          O_sparse: [0, -1, 1, -1]
      edges:
        - {from: solve, to: output}
)";

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_NE(cfg::configure_decoders(parsed), 0);
  cfg::finalize_decoders();
}

TEST(DecodingTaskGraph, PredecoderRejectsACallerSuppliedObservableModel) {
  // Skipping the default is not enough; a hand-written O would flip the inner
  // just the same.
  cudaqx::heterogeneous_map inner;
  inner.insert("O", 1);
  try {
    (void)predecoder_probe_graph(std::move(inner));
    FAIL() << "expected the caller-supplied O to be rejected";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("must not be given one"),
              std::string::npos)
        << e.what();
  }
}

// ---------------------------------------------------------------------------
// The residual topology, driven end to end
// ---------------------------------------------------------------------------

namespace {

/// The rep-5 model a residual topology is built on, plus the pieces a test
/// needs to author a shot for it: H to pick faults from, O to know the truth,
/// and D to turn a chosen syndrome back into measurement bytes.
struct rep5_model {
  detector_error_model closed;
  std::vector<std::vector<uint32_t>> d;
  std::size_t n_det = 0, n_mech = 0, dets_per_round = 0;

  static rep5_model build(uint64_t rounds = 5) {
    const auto spec = rep5_spec(rounds);
    const auto chunks = dem_chunks_from_spec(spec);
    rep5_model m;
    m.closed =
        dem_close_all(chunks, seam_name::next_round, seam_name::prev_round);
    m.d = dem_chunks_to_d_sparse(chunks, seam_name::next_round,
                                 seam_name::prev_round);
    m.n_det = m.closed.num_detectors();
    m.n_mech = m.closed.num_error_mechanisms();
    m.dets_per_round = m.n_det / rounds;
    return m;
  }

  /// The syndrome a set of faults produces: the GF(2) sum of their H columns.
  std::vector<uint8_t>
  syndrome_of(const std::vector<std::size_t> &faults) const {
    std::vector<uint8_t> s(n_det, 0);
    for (auto j : faults)
      for (std::size_t i = 0; i < n_det; ++i)
        s[i] ^= closed.detector_error_matrix.at({i, j});
    return s;
  }

  /// Whether those faults flip the observable.
  uint8_t truth_of(const std::vector<std::size_t> &faults) const {
    uint8_t t = 0;
    for (auto j : faults)
      t ^= closed.observables_flips_matrix.at({0, j});
    return t;
  }

  /// Measurement bytes whose D-apply is `s`. Detector r*d+k is the XOR of
  /// measurements (r-1)*d+k and r*d+k (round 0 is the measurement itself), so
  /// inverting is a prefix XOR down the round axis.
  std::vector<uint8_t> measurements_for(const std::vector<uint8_t> &s) const {
    std::vector<uint8_t> m(n_det, 0);
    for (std::size_t k = 0; k < dets_per_round; ++k) {
      uint8_t running = 0;
      for (std::size_t r = 0; r * dets_per_round + k < n_det; ++r) {
        running ^= s[r * dets_per_round + k];
        m[r * dets_per_round + k] = running;
      }
    }
    return m;
  }
};

task_graph_node_config residual_node(std::string id, std::string kind,
                                     std::string decoder, int lut_depth = 0) {
  task_graph_node_config n;
  n.id = std::move(id);
  n.kind = std::move(kind);
  n.dem = rep5_spec(5);
  n.decoder_type = std::move(decoder);
  if (lut_depth)
    n.params.insert("lut_error_depth", lut_depth);
  if (n.kind == "predecoder")
    n.node_params.insert("residual_from", std::string("parity_check"));
  return n;
}

/// pre --residual--> global, both committing into one XOR.
std::unique_ptr<decoding_task_graph>
build_residual_graph(task_graph_options options = {}) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));
  nodes.push_back(
      residual_node("global", "single_window", "multi_error_lut", 2));
  task_graph_node_config comb;
  comb.id = "combine";
  comb.kind = "xor";
  nodes.push_back(std::move(comb));
  return decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "pre", "input"},
       {"pre", "residual", "global", "input"},
       {"pre", "pre_L", "combine", "a"},
       {"global", "output", "combine", "b"},
       {"combine", "output", "output", "input"}},
      std::move(options));
}

/// build_residual_graph() with `global`'s inner decoder swapped for a counting
/// one, so a test can see how many residuals actually reached it.
std::unique_ptr<decoding_task_graph>
build_counted_residual_graph(rotating_fault_decoder **probe) {
  const auto model = rep5_model::build();
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));

  task_graph_node_config global;
  global.id = "global";
  global.kind = "single_window";
  global.dem = rep5_spec(5);
  auto dec = std::make_unique<rotating_fault_decoder>(
      sparse_binary_matrix(model.closed.detector_error_matrix), model.n_mech);
  *probe = dec.get();
  global.decoder_instance = std::move(dec);
  nodes.push_back(std::move(global));

  task_graph_node_config comb;
  comb.id = "combine";
  comb.kind = "xor";
  nodes.push_back(std::move(comb));
  return decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "pre", "input"},
                         {"pre", "residual", "global", "input"},
                         {"pre", "pre_L", "combine", "a"},
                         {"global", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
}

/// The predecoder alone, so a shot can be attributed: whatever the full graph
/// gets that this does not came from the residual edge.
std::unique_ptr<decoding_task_graph> build_pre_only_graph() {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));
  return decoding_task_graph::from_topology(std::move(nodes));
}

uint8_t stream_shot(decoding_task_graph &g, const std::vector<uint8_t> &m) {
  g.reset_decoder();
  const std::size_t spr = g.get_num_msyn_per_round();
  for (std::size_t off = 0; off + spr <= m.size(); off += spr)
    g.enqueue_syndrome(m.data() + off, spr);
  return g.get_obs_corrections()[0] & 1u;
}

} // namespace

TEST(DecodingTaskGraph, ResidualTopologySplitsWorkBetweenPreAndGlobal) {
  const auto model = rep5_model::build();
  auto graph = build_residual_graph();
  auto pre_only = build_pre_only_graph();

  // Byte counts come from the graph under test, not from a lookalike.
  ASSERT_EQ(graph->get_num_msyn_per_decode(), model.n_det);
  ASSERT_EQ(graph->get_num_observables(), 1u);

  // A single fault the depth-1 LUT resolves, and a pair it cannot: for the
  // pair, no single H column carries that syndrome, so `pre` must abstain and
  // only the depth-2 `global` can explain it. Searched rather than hardcoded,
  // and asserted found, so a DEM change fails loudly instead of silently
  // testing nothing.
  const auto column_matches = [&](const std::vector<uint8_t> &s) {
    for (std::size_t j = 0; j < model.n_mech; ++j) {
      bool same = true;
      for (std::size_t i = 0; i < model.n_det && same; ++i)
        same = model.closed.detector_error_matrix.at({i, j}) == s[i];
      if (same)
        return true;
    }
    return false;
  };

  std::vector<std::size_t> single, pair;
  for (std::size_t j = 0; j < model.n_mech && single.empty(); ++j)
    if (model.truth_of({j}) == 1)
      single = {j};
  for (std::size_t j = 0; j < model.n_mech && pair.empty(); ++j)
    for (std::size_t k = j + 1; k < model.n_mech; ++k)
      if (model.truth_of({j, k}) == 1 &&
          !column_matches(model.syndrome_of({j, k}))) {
        pair = {j, k};
        break;
      }
  ASSERT_FALSE(single.empty()) << "no observable-flipping single fault";
  ASSERT_FALSE(pair.empty()) << "no two-fault syndrome outside the depth-1 LUT";

  for (const auto *shot : {&single, &pair}) {
    const auto s = model.syndrome_of(*shot);
    const auto m = model.measurements_for(s);
    // Self-check the inversion: D applied to these bytes must give back `s`.
    for (std::size_t i = 0; i < model.n_det; ++i) {
      uint8_t parity = 0;
      for (auto c : model.d[i])
        parity ^= m[c];
      ASSERT_EQ(parity, s[i]) << "detector " << i;
    }

    const uint8_t truth = model.truth_of(*shot);
    ASSERT_EQ(truth, 1u);
    EXPECT_EQ(stream_shot(*graph, m), truth)
        << (shot == &single ? "single-fault shot" : "two-fault shot");

    // run() the same DAG on the same shot. This is the only path that
    // executes xor_node, since streaming combines through
    // get_obs_corrections() instead.
    measurement_results mr;
    mr.bits = m;
    EXPECT_EQ(graph->run(mr).bits[0] & 1u, truth);
  }

  // Attribution, which is what makes this test about routing rather than about
  // the decoders. The predecoder alone gets the single-fault shot right and
  // the two-fault shot wrong, so the full graph's correct answer on the latter
  // can only have come through the residual edge into `global`.
  EXPECT_EQ(
      stream_shot(*pre_only, model.measurements_for(model.syndrome_of(single))),
      1u);
  EXPECT_EQ(
      stream_shot(*pre_only, model.measurements_for(model.syndrome_of(pair))),
      0u)
      << "the predecoder was expected to abstain on the two-fault shot; if it "
         "resolves it, the test no longer distinguishes the two routes";
}

TEST(DecodingTaskGraph, ChunkedPredecoderChainRoutesEveryResidual) {
  // The predecoder rewrites one residual buffer per fire, so metering from
  // inside enqueue() instead of from the sweep would hand `global` only the
  // last residual of a multi-fire call.  Counting `global`'s decodes says that
  // directly, rather than hoping a lost residual disturbs one output bit.
  rotating_fault_decoder *probe = nullptr;
  auto graph = build_counted_residual_graph(&probe);
  const std::size_t capacity = graph->get_num_msyn_per_decode();
  const std::size_t windows = 3;

  std::vector<uint8_t> m(capacity * windows, 0);
  for (std::size_t i = 0; i < m.size(); i += 3)
    m[i] = 1;

  const auto residuals_for = [&](std::size_t chunk) {
    graph->reset_decoder();
    for (std::size_t off = 0; off < m.size(); off += chunk)
      graph->enqueue_syndrome(m.data() + off, std::min(chunk, m.size() - off));
    return probe->calls();
  };

  ASSERT_EQ(residuals_for(capacity), windows) << "one window per call";
  for (std::size_t chunk = 1; chunk <= m.size(); ++chunk)
    EXPECT_EQ(residuals_for(chunk), windows) << "chunk size " << chunk;
}

/// The same split as build_residual_graph(), published as two outputs instead
/// of folded into one.
std::unique_ptr<decoding_task_graph> build_two_sink_residual_graph() {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));
  nodes.push_back(
      residual_node("global", "single_window", "multi_error_lut", 2));
  return decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "pre", "input"},
                         {"pre", "residual", "global", "input"},
                         {"pre", "pre_L", "output", "pre"},
                         {"global", "output", "output", "global"}});
}

TEST(DecodingTaskGraph, ASyndromePortCannotBeAGraphOutput) {
  // An output is a logical outcome. Publishing a residual there would have
  // run_all() reading detection_events out of the variant, so the topology is
  // refused instead. A batch-only fold exposes no streaming capability and
  // stays publishable -- that is how the xor graphs reach their output.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));
  try {
    (void)decoding_task_graph::from_topology(
        std::move(nodes), {{"measurements", "output", "pre", "input"},
                           {"pre", "residual", "output", "input"}});
    FAIL() << "a detection_events port reached the sink";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("residual"), std::string::npos)
        << e.what();
  }
}

TEST(DecodingTaskGraph, ChainedFoldsStillCarryEveryCorrection) {
  // Two folds deep, which only from_topology() can build -- combine() refuses a
  // multi-node graph. Neither xor fires while streaming, so all three windows
  // have to reach the output *through* both of them.
  const auto model = rep5_model::build();
  const auto build = [] {
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(residual_node("w1", "single_window", "single_error_lut"));
    nodes.push_back(residual_node("w2", "single_window", "multi_error_lut", 2));
    nodes.push_back(residual_node("w3", "single_window", "multi_error_lut", 3));
    for (const char *id : {"c1", "c2"}) {
      task_graph_node_config x;
      x.id = id;
      x.kind = "xor";
      nodes.push_back(std::move(x));
    }
    return decoding_task_graph::from_topology(
        std::move(nodes), {{"measurements", "output", "w1", "input"},
                           {"measurements", "output", "w2", "input"},
                           {"measurements", "output", "w3", "input"},
                           {"w1", "output", "c1", "a"},
                           {"w2", "output", "c1", "b"},
                           {"c1", "output", "c2", "a"},
                           {"w3", "output", "c2", "b"},
                           {"c2", "output", "output", "input"}});
  };
  auto batch = build();
  auto stream = build();

  std::mt19937 rng(23);
  std::bernoulli_distribution flip(0.08);
  std::size_t nonzero = 0;
  constexpr int shots = 300;
  for (int shot = 0; shot < shots; ++shot) {
    std::vector<std::size_t> faults;
    for (std::size_t j = 0; j < model.n_mech; ++j)
      if (flip(rng))
        faults.push_back(j);
    const auto m = model.measurements_for(model.syndrome_of(faults));

    measurement_results mr;
    mr.bits = m;
    batch->reset_decoder();
    const auto lo = batch->run(mr);
    ASSERT_EQ(lo.bits.size(), 1u);
    EXPECT_EQ(stream_shot(*stream, m), lo.bits[0] & 1u) << "shot " << shot;
    nonzero += (lo.bits[0] & 1u) != 0;
  }
  EXPECT_GT(nonzero, 0u);
}

TEST(DecodingTaskGraph, ResidualEdgeDoesNotRouteCorrectionsDownstream) {
  // A residual edge carries detection events for `global` to decode, not
  // corrections to fold into whatever `global` feeds. So `pre` belongs to the
  // output its pre_L reaches and to no other, and the two paths must agree on
  // both outputs.
  const auto model = rep5_model::build();
  auto batch = build_two_sink_residual_graph();
  auto stream = build_two_sink_residual_graph();

  ASSERT_EQ(batch->output_names().size(), 2u);
  EXPECT_EQ(batch->output_names()[0], "pre");
  EXPECT_EQ(batch->output_names()[1], "global");
  ASSERT_EQ(stream->get_num_observables(), 2u);

  std::mt19937 rng(11);
  std::bernoulli_distribution flip(0.08);
  std::size_t global_nonzero = 0, pre_nonzero = 0;
  constexpr int shots = 400;
  for (int shot = 0; shot < shots; ++shot) {
    std::vector<std::size_t> faults;
    for (std::size_t j = 0; j < model.n_mech; ++j)
      if (flip(rng))
        faults.push_back(j);
    const auto m = model.measurements_for(model.syndrome_of(faults));

    measurement_results mr;
    mr.bits = m;
    batch->reset_decoder();
    const auto all = batch->run_all(mr);
    ASSERT_EQ(all.size(), 2u);

    stream->reset_decoder();
    const std::size_t spr = stream->get_num_msyn_per_round();
    for (std::size_t off = 0; off + spr <= m.size(); off += spr)
      stream->enqueue_syndrome(m.data() + off, spr);
    const uint8_t *c = stream->get_obs_corrections();

    EXPECT_EQ(c[0], all[0].bits[0] & 1u) << "output 'pre', shot " << shot;
    EXPECT_EQ(c[1], all[1].bits[0] & 1u) << "output 'global', shot " << shot;
    pre_nonzero += (all[0].bits[0] & 1u) != 0;
    global_nonzero += (all[1].bits[0] & 1u) != 0;
  }
  // Both halves have to move, or the outputs could not be confused.
  EXPECT_GT(pre_nonzero, 0u);
  EXPECT_GT(global_nonzero, 0u);
}

TEST(DecodingTaskGraph, StreamingElidesTheXorAndStillMatchesRun) {
  // Streaming never fires `combine`: both halves reach one output, so their XOR
  // into its segment stands in for the fold, which run() performs for real. The
  // two paths must agree bit for bit. Only a graph carrying an elided xor tests
  // that -- a single-window graph has none, and the split tests below do the
  // fold in the test rather than through the graph.
  const auto model = rep5_model::build();
  auto batch = build_residual_graph();
  auto stream = build_residual_graph();

  std::mt19937 rng(7);
  std::bernoulli_distribution flip(0.08);
  std::size_t nonzero = 0;
  constexpr int shots = 500;
  for (int shot = 0; shot < shots; ++shot) {
    std::vector<std::size_t> faults;
    for (std::size_t j = 0; j < model.n_mech; ++j)
      if (flip(rng))
        faults.push_back(j);
    const auto m = model.measurements_for(model.syndrome_of(faults));

    measurement_results mr;
    mr.bits = m;
    batch->reset_decoder();
    const auto lo = batch->run(mr);
    ASSERT_EQ(lo.bits.size(), 1u);
    EXPECT_EQ(stream_shot(*stream, m), lo.bits[0] & 1u) << "shot " << shot;
    nonzero += (lo.bits[0] & 1u) != 0;
  }
  // Agreeing only on zeros would prove nothing about the fold.
  EXPECT_GT(nonzero, 0u);
}

TEST(DecodingTaskGraph, PredecoderPlusGlobalMatchesMonolithicDecodeAccuracy) {
  // The realistic counterpart to the exact test below: real LUT decoders on
  // the rep-5 model, where explanations are degenerate and per-shot equality
  // is not available. `global` and `mono` run the *same* decoder, so any gap
  // in the rates is the split rather than decoder quality.
  const auto model = rep5_model::build();
  auto split = build_residual_graph();

  std::vector<task_graph_node_config> mono_nodes;
  mono_nodes.push_back(
      residual_node("mono", "single_window", "multi_error_lut", 2));
  auto mono = decoding_task_graph::from_topology(std::move(mono_nodes));

  std::mt19937 rng(99);
  std::bernoulli_distribution flip(0.02);
  constexpr int shots = 3000;
  std::size_t split_wrong = 0, mono_wrong = 0, low_weight = 0,
              low_weight_split_wrong = 0;
  std::size_t pre_committed = 0, residual_nonzero = 0;

  auto pre_only = build_pre_only_graph();
  for (int shot = 0; shot < shots; ++shot) {
    std::vector<std::size_t> faults;
    for (std::size_t j = 0; j < model.n_mech; ++j)
      if (flip(rng))
        faults.push_back(j);
    const auto s = model.syndrome_of(faults);
    const auto m = model.measurements_for(s);
    const uint8_t truth = model.truth_of(faults);

    const uint8_t split_bit = stream_shot(*split, m);
    const uint8_t mono_bit = stream_shot(*mono, m);
    const uint8_t pre_bit = stream_shot(*pre_only, m);

    split_wrong += (split_bit != truth);
    mono_wrong += (mono_bit != truth);
    if (pre_bit)
      ++pre_committed;
    // `pre` abstained, so everything it saw went downstream.
    if (!pre_bit &&
        std::any_of(s.begin(), s.end(), [](uint8_t b) { return b != 0; }))
      ++residual_nonzero;
    // Compare the split against the monolithic decode, not against the truth:
    // this DEM has duplicate columns, so some weight-1 syndromes are genuinely
    // ambiguous and *both* decoders "fail" them identically. Blaming the split
    // for that would be blaming it for the model.
    if (faults.size() <= 1) {
      ++low_weight;
      low_weight_split_wrong += (split_bit != mono_bit);
    }
  }

  // On a weight-0/1 shot the two must agree bit for bit: whatever the
  // monolithic decoder concludes, the split has to reach the same conclusion.
  // This is the tier that catches a misrouted or double-counted commit, which
  // an aggregate rate can hide inside its tolerance.
  ASSERT_GT(low_weight, 0u);
  EXPECT_EQ(low_weight_split_wrong, 0u)
      << low_weight_split_wrong << "/" << low_weight
      << " weight-0/1 shots where the split disagreed with the monolithic "
         "decode; a correction is being lost or double-counted";

  const double split_ler = static_cast<double>(split_wrong) / shots;
  const double mono_ler = static_cast<double>(mono_wrong) / shots;
  EXPECT_LT(mono_ler, 0.3) << "sanity: the monolithic decode is near chance";
  EXPECT_NEAR(split_ler, mono_ler, 0.03)
      << "split=" << split_ler << " monolithic=" << mono_ler;

  // Not vacuous: both halves must be load-bearing over the run.
  EXPECT_GT(pre_committed, 0u) << "the pre-decoder never committed anything";
  EXPECT_GT(residual_nonzero, 0u)
      << "the pre-decoder never deferred a syndrome to global";
}

namespace {

/// Solves `H e = s` exactly for the unit lower-bidiagonal chain below, by
/// forward substitution: s[i] = e[i] XOR e[i-1], so e[i] = s[i] XOR e[i-1].
/// Complete by construction -- every syndrome has exactly one explanation --
/// which is what lets the split be checked for equality rather than for a rate.
class exact_chain_decoder final : public decoder {
public:
  explicit exact_chain_decoder(sparse_binary_matrix h)
      : decoder(decoder_init(std::move(h))) {}
  decoder_result decode(const std::vector<float_t> &s) override {
    std::vector<float_t> e(block_size, 0.0f);
    uint8_t run = 0;
    for (std::size_t i = 0; i < block_size; ++i) {
      run ^= static_cast<uint8_t>(convert_soft_to_hard(s[i]));
      e[i] = static_cast<float_t>(run);
    }
    return decoder_result{true, std::move(e)};
  }
};

/// An oracle pre-decoder: predicts the true error, but only on the columns it
/// owns. Standing in for "a cheap decoder that resolves the easy faults" keeps
/// the experiment about the split rather than about decoder quality.
class oracle_subset_decoder final : public decoder {
public:
  oracle_subset_decoder(sparse_binary_matrix h, std::vector<uint8_t> owns)
      : decoder(decoder_init(std::move(h))), owns_(std::move(owns)),
        truth_(owns_.size(), 0) {}
  void set_truth(const std::vector<uint8_t> &e) { truth_ = e; }
  decoder_result decode(const std::vector<float_t> &) override {
    std::vector<float_t> e(block_size, 0.0f);
    for (std::size_t i = 0; i < block_size; ++i)
      if (owns_[i] && truth_[i])
        e[i] = 1.0f;
    return decoder_result{true, std::move(e)};
  }

private:
  std::vector<uint8_t> owns_, truth_;
};

} // namespace

TEST(DecodingTaskGraph,
     PredecoderPlusGlobalExactlyReproducesTheMonolithicDecode) {
  // The strongest available statement: with a decoder that is complete on this
  // model, `pre` and `global` together must reproduce the monolithic answer on
  // *every* shot, not merely at the same rate. H is unit lower bidiagonal, so
  // its columns are independent and each syndrome has one explanation; any
  // disagreement is therefore the split, never a tie broken differently.
  constexpr std::size_t R = 10;
  constexpr int shots = 2000;

  std::vector<std::vector<uint32_t>> h_rows(R);
  for (std::size_t i = 0; i < R; ++i) {
    h_rows[i] = (i == 0) ? std::vector<uint32_t>{0}
                         : std::vector<uint32_t>{static_cast<uint32_t>(i - 1),
                                                 static_cast<uint32_t>(i)};
  }
  auto h_mat = sparse_binary_matrix::from_nested_csr(R, R, h_rows);
  auto h = std::make_shared<const sparse_binary_matrix>(h_mat);
  // D is the identity, so a measurement byte is a detector bit and a shot can
  // be handed to the graph as the syndrome itself.
  std::vector<std::vector<uint32_t>> ident(R);
  for (std::size_t i = 0; i < R; ++i)
    ident[i].push_back(static_cast<uint32_t>(i));
  auto d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(R, R, ident));
  // One persistent observable: the parity of every fault, so both halves of
  // the split can move it.
  std::vector<uint32_t> all(R);
  for (std::size_t i = 0; i < R; ++i)
    all[i] = static_cast<uint32_t>(i);
  auto o = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, R, {all}));

  // `pre` owns the even columns; whatever is left is `global`'s.
  std::vector<uint8_t> owns(R, 0);
  for (std::size_t i = 0; i < R; i += 2)
    owns[i] = 1;
  auto oracle = std::make_unique<oracle_subset_decoder>(h_mat, owns);
  auto *oracle_raw = oracle.get();

  predecoder_node pre(residual_src::parity_check, d, o, h, std::move(oracle),
                      /*spr=*/R, /*rounds=*/1);
  window_node global(d, o, std::make_unique<exact_chain_decoder>(h_mat),
                     /*spr=*/R, window_spec{1, 1, 0, 0});
  window_node mono(d, o, std::make_unique<exact_chain_decoder>(h_mat),
                   /*spr=*/R, window_spec{1, 1, 0, 0});

  std::mt19937 rng(1234);
  std::bernoulli_distribution flip(0.15);
  std::size_t pre_committed = 0, residual_changed = 0, residual_nonzero = 0;
  std::size_t pre_alone_wrong = 0, global_alone_wrong = 0;

  for (int shot = 0; shot < shots; ++shot) {
    std::vector<uint8_t> e_true(R, 0);
    for (auto &b : e_true)
      b = flip(rng);
    // s = H * e_true, and O is the parity of every column.
    std::vector<uint8_t> s(R, 0);
    uint8_t truth = 0;
    for (std::size_t j = 0; j < R; ++j)
      truth ^= e_true[j];
    for (std::size_t i = 0; i < R; ++i)
      for (auto c : h_rows[i])
        s[i] ^= e_true[c];

    oracle_raw->set_truth(e_true);
    pre.reset_decoder();
    global.reset_decoder();
    mono.reset_decoder();

    ASSERT_TRUE(pre.enqueue(s.data(), R, round_type::BULK)) << "shot " << shot;
    const auto &residual =
        pre.streaming_output(1)->get<detection_events>().bits;
    const uint8_t pre_bit = pre.corrections()[0] & 1u;

    if (pre_bit)
      ++pre_committed;
    bool differs = false, nonzero = false;
    for (std::size_t i = 0; i < R; ++i) {
      differs |= (residual[i] != static_cast<float_t>(s[i]));
      nonzero |= (residual[i] != 0.0f);
    }
    residual_changed += differs;
    residual_nonzero += nonzero;

    ASSERT_TRUE(global.enqueue_edge_value(*pre.streaming_output(1), 0))
        << "shot " << shot;
    const uint8_t global_bit = global.corrections()[0] & 1u;

    ASSERT_TRUE(mono.enqueue(s.data(), R, round_type::BULK)) << "shot " << shot;
    const uint8_t mono_bit = mono.corrections()[0] & 1u;

    ASSERT_EQ(mono_bit, truth) << "shot " << shot
                               << ": the complete decoder "
                                  "should be exact on an invertible H";
    EXPECT_EQ(pre_bit ^ global_bit, mono_bit) << "shot " << shot;
    pre_alone_wrong += (pre_bit != mono_bit);
    global_alone_wrong += (global_bit != mono_bit);
  }

  // Not vacuous: each half has to be doing work, and the residual has to be
  // both changed by `pre` and still carrying something for `global`. Without
  // these a pre-decoder that explained nothing, or everything, would pass.
  EXPECT_GT(pre_committed, shots / 5u);
  EXPECT_GT(residual_changed, shots / 5u);
  EXPECT_GT(residual_nonzero, shots / 5u);
  EXPECT_GT(pre_alone_wrong, 0u) << "global never contributed";
  EXPECT_GT(global_alone_wrong, 0u) << "pre never contributed";
}

TEST(DecodingTaskGraph, PredecoderConstructorEnforcesItsOwnInvariants) {
  // The factory is not the only way in: a caller can construct the node
  // directly, so the invariants the residual depends on belong here too.
  auto d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 5, {{0}}));

  // An observable frame cannot say which detectors it explained, so there is
  // no residual to compute -- rejected here, not only in the factory.
  auto obs = std::make_unique<obs_projecting_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 1, {{0}, {0}, {0}, {0}}),
      std::vector<float_t>{1.0f}, /*declare_obs=*/true);
  auto o_one = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 1, {{0}}));
  auto h_one = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 1, {{0}, {0}, {0}, {0}}));
  EXPECT_THROW(predecoder_node(residual_src::parity_check, d, o_one, h_one,
                               std::move(obs), 4, 1),
               std::invalid_argument);

  // A CSC H is the same matrix, but csr_rows would walk its columns as if they
  // were rows. Taking it by value and normalizing means a caller cannot get a
  // silently wrong residual out of a correctly-built matrix.
  auto csc_h = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csc(
          4, 5, {{0}, {0, 1}, {1, 2}, {2, 3}, {3}}));
  ASSERT_EQ(csc_h->layout(), sparse_binary_matrix_layout::csc);
  auto dec = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}),
      std::vector<float_t>{1, 0, 0, 0, 0}, true);
  predecoder_node node(residual_src::parity_check, d, o, csc_h, std::move(dec),
                       4, 1);
  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(node.enqueue(msyn, 4, round_type::BULK));
  // Same answer the CSR-built node gives in
  // PredecoderResidualIsTheUnexplainedSyndrome: s XOR H*e0 = (0,0,0,1).
  EXPECT_EQ(node.streaming_output(1)->get<detection_events>().bits,
            (std::vector<float_t>{0, 0, 0, 1}));
}

TEST(DecodingTaskGraph,
     PredecoderModelRejectsObservablesWithoutResidualContract) {
  auto d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 5, {{0}}));
  auto observables_only = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}),
      std::vector<float_t>{1}, true, std::nullopt,
      decoder_output_request{decode_result_type::observables});

  EXPECT_THROW(predecoder_node(residual_src::model, d, o, chain_h(),
                               std::move(observables_only), 4, 1),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, PredecoderRejectsADecoderThatDoesNotMatchItsDem) {
  // The residual is `s XOR H*e` with H from the node's DEM, but `e` comes from
  // the inner decoder. A caller-supplied decoder built on a different matrix
  // therefore indexes H by columns that mean something else: with mismatched
  // dimensions that is an out-of-range read, and with matching dimensions it
  // is a silently wrong residual. Reject at construction, where the shapes are
  // still visible, rather than at the first decode.
  auto d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 5, {{0}}));
  auto h = chain_h(); // 4 detectors x 5 mechanisms

  // Wrong syndrome_size: this decoder reads 3 detectors, H produces 4.
  auto narrow = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(3, 5, {{0}, {1}, {2}}),
      std::vector<float_t>{1, 0, 0, 0, 0}, true);
  EXPECT_THROW(predecoder_node(residual_src::parity_check, d, o, h,
                               std::move(narrow), 4, 1),
               std::invalid_argument);

  // Wrong block_size: this decoder predicts 4 mechanisms, H has 5 columns, so
  // H's last column would index past the end of the result.
  auto short_e = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}),
      std::vector<float_t>{1, 0, 0, 0}, true);
  EXPECT_THROW(predecoder_node(residual_src::parity_check, d, o, h,
                               std::move(short_e), 4, 1),
               std::invalid_argument);

  // Not vacuous: the matching decoder still builds.
  auto ok = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}),
      std::vector<float_t>{1, 0, 0, 0, 0}, true);
  EXPECT_NO_THROW(predecoder_node(residual_src::parity_check, d, o, h,
                                  std::move(ok), 4, 1));
}

// The flat form with its H, which is the third of the DEM's matrices and the
// one a residual needs. `chain_h()` is 4 detectors x 5 mechanisms.
// An empty `residual_from` leaves the key off entirely, which is how the
// missing-key case is expressed. `e` is the inner decoder's primary result:
// an error pattern under parity_check or pre_L under model. Model residuals
// arrive separately through decoder_result::opt_results.
static task_graph_node_config flat_predecoder_node(
    bool with_h, const std::string &residual_from = "parity_check",
    std::vector<float_t> e = {1, 0, 0, 0, 0},
    std::optional<std::vector<float_t>> residual = std::nullopt) {
  task_graph_node_config node;
  node.id = "pre";
  node.kind = "predecoder";
  node.d_sparse = {{0}, {1}, {2}, {3}};
  node.o_sparse = {{0}};
  if (with_h)
    node.h_sparse = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
  node.decoder_instance = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}),
      std::move(e), true, std::move(residual),
      residual_from == "model"
          ? decoder_output_request{decode_result_type::observables,
                                   decoder_auxiliary_result_type::
                                       residual_detectors}
          : decoder_output_request{decode_result_type::errors});
  node.node_params.insert("num_syndromes_per_round", std::size_t{4});
  if (!residual_from.empty())
    node.node_params.insert("residual_from", residual_from);
  return node;
}

static std::unique_ptr<decoding_task_graph> build_flat_predecoder(
    bool with_h, const std::string &residual_from = "parity_check",
    std::vector<float_t> e = {1, 0, 0, 0, 0},
    std::optional<std::vector<float_t>> residual = std::nullopt) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_predecoder_node(with_h, residual_from, std::move(e),
                                       std::move(residual)));
  return decoding_task_graph::from_topology(std::move(nodes));
}

TEST(DecodingTaskGraph, PredecoderBuildsFromFlatMatricesWithAnH) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_predecoder_node(/*with_h=*/true));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));

  // And the residual it produces is the same `s XOR H*e` the chunk form gives:
  // predicting fault 0 explains detector 0, so s = (1,0,0,1) leaves (0,0,0,1).
  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(graph->enqueue_syndrome(msyn, 4));
  EXPECT_EQ(graph->get_obs_corrections()[0] & 1u, 1u);
}

TEST(DecodingTaskGraph, PredecoderRejectsFlatMatricesWithoutAnH) {
  // Not vacuous alongside the test above: the only difference is h_sparse, so
  // this cannot be passing because the flat form is rejected outright.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_predecoder_node(/*with_h=*/false));
  try {
    decoding_task_graph::from_topology(std::move(nodes));
    FAIL() << "expected the flat form without an H to be rejected";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("h_sparse"), std::string::npos)
        << e.what();
  }
}

TEST(DecodingTaskGraph, PredecoderRequiresADeclaredResidualSource) {
  // No default, and nothing to infer one from: an inner decoder built without
  // O reports decode_to_errs under either contract, so its result type cannot
  // say which it holds. A mis-declared model must fail the build, not the
  // first decode.
  try {
    build_flat_predecoder(/*with_h=*/true, /*residual_from=*/"");
    FAIL() << "expected a predecoder with no residual_from to be rejected";
  } catch (const std::exception &e) {
    EXPECT_NE(std::string(e.what()).find("residual_from"), std::string::npos)
        << e.what();
  }

  // Not vacuous: the same node builds once the key is there.
  EXPECT_NO_THROW(build_flat_predecoder(/*with_h=*/true, "parity_check"));

  // A near-miss spelling is an error, not a silent fallback to either mode.
  EXPECT_THROW(build_flat_predecoder(/*with_h=*/true, "parity"),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, PredecoderModelReadsNamedResidualResult) {
  // Both derivations read pre_L from column 0 with this O, so only the
  // residual can tell them apart.
  const uint8_t msyn[4] = {1, 0, 0, 1};
  const std::vector<float_t> error{1, 0, 1, 0, 0};
  const std::vector<float_t> model_residual{0, 1, 0, 0};

  auto supplied =
      make_chain_predecoder({1}, true, residual_src::model, model_residual);
  ASSERT_TRUE(supplied->enqueue(msyn, 4, round_type::BULK));
  EXPECT_EQ(supplied->streaming_output(1)->get<detection_events>().bits,
            model_residual);
  EXPECT_EQ(supplied->corrections()[0] & 1u, 1u);

  // The A/B: dets = (1,0,0,1) and H*e = (1,1,1,0) for the same vector read as
  // an error pattern, so parity_check leaves (0,1,1,1) -- a different residual
  // from the same inputs, which is why the derivation has to be declared.
  auto derived = make_chain_predecoder(error, true, residual_src::parity_check);
  ASSERT_TRUE(derived->enqueue(msyn, 4, round_type::BULK));
  EXPECT_EQ(derived->streaming_output(1)->get<detection_events>().bits,
            (std::vector<float_t>{0, 1, 1, 1}));
}

TEST(DecodingTaskGraph, PredecoderModelIgnoresTheDemsH) {
  // H comes from the DEM, so `model` cannot refuse one -- every chunk-described
  // node carries it. Unused is the only option left, and an H too narrow for
  // the inner's result shows it: `model` builds and decodes anyway.
  auto with_narrow_h = [](const std::string &residual_from) {
    auto node = residual_from == "model"
                    ? flat_predecoder_node(/*with_h=*/true, residual_from, {1},
                                           std::vector<float_t>{0, 1, 0, 0})
                    : flat_predecoder_node(/*with_h=*/true, residual_from,
                                           {1, 0, 1, 0, 0});
    node.h_sparse = {{0}, {0}, {0}, {0}}; // 4 rows, one fault column
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    return decoding_task_graph::from_topology(std::move(nodes));
  };

  auto graph = with_narrow_h("model");
  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(graph->enqueue_syndrome(msyn, 4));
  EXPECT_EQ(graph->get_obs_corrections()[0] & 1u, 1u);

  // Not vacuous: parity_check does read that H, and rejects it for exactly the
  // mismatch model is entitled to ignore.
  EXPECT_THROW(with_narrow_h("parity_check"), std::invalid_argument);
}

TEST(DecodingTaskGraph, ModelHoldsNoHAtAll) {
  // Ignoring the DEM's H is one thing; holding none is what a supplied decoder
  // and no h_sparse reach, and every shape check has to skip it rather than
  // read through it.
  auto d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 5, {{0}}));
  const auto inner = sparse_binary_matrix::from_nested_csr(
      4, 5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  predecoder_node node(
      residual_src::model, d, o, nullptr,
      std::make_unique<fixed_error_decoder>(
          inner, std::vector<float_t>{1}, true,
          std::vector<float_t>{0, 1, 0, 0},
          decoder_output_request{
              decode_result_type::observables,
              decoder_auxiliary_result_type::residual_detectors}),
      /*spr=*/4,
      /*rounds=*/1);
  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(node.enqueue(msyn, 4, round_type::BULK));
  EXPECT_EQ(node.corrections()[0] & 1u, 1u);

  // parity_check has nothing to push a prediction through without one, so the
  // same absence is refused there.
  EXPECT_THROW(
      predecoder_node(residual_src::parity_check, d, o, nullptr,
                      std::make_unique<fixed_error_decoder>(
                          inner, std::vector<float_t>{1, 0, 1, 0, 0}, true),
                      4, 1),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, PredecoderModelRejectsAnUndeclaredWidth) {
  // Both the primary observable result and named residual must exactly match
  // O and D respectively.
  auto missing = build_flat_predecoder(/*with_h=*/true, "model", {1});
  const uint8_t msyn[4] = {1, 0, 0, 1};
  EXPECT_THROW(missing->enqueue_syndrome(msyn, 4), std::out_of_range);

  auto narrow = build_flat_predecoder(
      /*with_h=*/true, "model", {1}, std::vector<float_t>{0, 0, 0});
  EXPECT_THROW(narrow->enqueue_syndrome(msyn, 4), std::out_of_range);

  auto wide = build_flat_predecoder(
      /*with_h=*/true, "model", {1, 0}, std::vector<float_t>{0, 0, 0, 0});
  EXPECT_THROW(wide->enqueue_syndrome(msyn, 4), std::out_of_range);
}

TEST(DecodingTaskGraph, FlatHMustAgreeWithD) {
  // Two independent sources of H in the flat form -- the node's h_sparse and
  // whatever the supplied decoder was built on -- so the shapes are checked
  // against D and against each other rather than assumed.
  auto node = flat_predecoder_node(/*with_h=*/true);
  node.h_sparse = {{0, 1}, {1, 2}}; // 2 detector rows, D produces 4
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  EXPECT_THROW(decoding_task_graph::from_topology(std::move(nodes)),
               std::invalid_argument);
}

TEST(DecodingTaskGraph,
     ObservableProjectingDecoderMayKeepFaultColumnBlockSize) {
  // Observable-projecting decoders in the tree follow two conventions: some
  // rebuild block_size as the observable count, others keep it at H's
  // fault-column count and switch only the *result vector* to observables. A
  // node must accept both, so block_size is not a usable stand-in for the
  // observable count at construction time -- only the result width is, and
  // that is not known until the first decode.
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 5, {{0}}));
  // block_size 5 (fault columns), one observable emitted: the shape in which
  // the two counts disagree.
  auto dec = std::make_unique<obs_projecting_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 5,
                                            {{0, 1}, {1, 2}, {2, 3}, {3, 4}}),
      std::vector<float_t>{1.0f}, /*declare_obs=*/true);
  ASSERT_EQ(dec->get_block_size(), 5u);
  ASSERT_NE(dec->get_block_size(), o->num_rows());

  window_node node(d, o, std::move(dec), /*spr=*/4, window_spec{1, 1, 0, 0});
  const uint8_t msyn[4] = {1, 0, 0, 1};
  ASSERT_TRUE(node.enqueue(msyn, 4, round_type::BULK));
  EXPECT_EQ(node.streaming_output(0)->get<logical_outcome>().bits,
            (std::vector<uint8_t>{1}));
}

TEST(DecodingTaskGraph, WindowNodeRejectsADecoderThatDoesNotMatchItsDem) {
  // window_node hands the decoder exactly D's detector rows on both paths, so
  // a caller-supplied decoder built on some other DEM would read past the end
  // of that buffer -- and the inner decoder is entitled not to notice (the
  // LUT's length assert compiles out in release).
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 4, {{0}}));
  const window_spec ws{4, 4, 0, 0};

  auto too_many = std::make_unique<obs_projecting_decoder>(
      sparse_binary_matrix::from_nested_csr(6, 4,
                                            {{0}, {1}, {2}, {3}, {0}, {1}}),
      std::vector<float_t>(4, 0.0f), /*declare_obs=*/false);
  EXPECT_THROW(window_node(d, o, std::move(too_many), 1, ws),
               std::invalid_argument);

  // And an error pattern narrower than the columns O indexes.
  auto too_narrow = std::make_unique<obs_projecting_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 2, {{0}, {1}, {0}, {1}}),
      std::vector<float_t>(2, 0.0f), /*declare_obs=*/false);
  auto wide_o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, 4, {{3}}));
  EXPECT_THROW(window_node(d, wide_o, std::move(too_narrow), 1, ws),
               std::invalid_argument);

  // Not vacuous: the matching decoder still builds.
  auto ok = std::make_unique<obs_projecting_decoder>(
      sparse_binary_matrix::from_nested_csr(4, 4, {{0}, {1}, {2}, {3}}),
      std::vector<float_t>(4, 0.0f), /*declare_obs=*/false);
  EXPECT_NO_THROW(window_node(d, o, std::move(ok), 1, ws));
}

TEST(DecodingTaskGraph, ObservableProjectingDecoderCannotBeGuarded) {
  // Guards mask fault columns; an observable frame has none to mask.
  std::vector<std::vector<uint32_t>> identity(3);
  for (uint32_t i = 0; i < 3; ++i)
    identity[i].push_back(i);
  auto h = sparse_binary_matrix::from_nested_csr(3, 3, identity);
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(3, 3, identity));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(3, 3, identity));
  EXPECT_THROW(window_node(d, o,
                           std::make_unique<obs_projecting_decoder>(
                               h, std::vector<float_t>{1.0f}, true),
                           1, window_spec{3, 1, 1, 1}, {0, 1, 2, 3}),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, WindowNodeStillRefusesAnOversizedChunk) {
  // The graph's metering never sends one of these, but a node still fires at
  // most once per call and cannot honour a chunk bigger than its buffer.
  // Refusing beats truncating: the accumulated bytes survive.
  auto node = make_identity_window({/*size=*/3, /*stride=*/3, 0, 0}, 3);
  const std::size_t capacity = node->num_msyn_per_decode();
  ASSERT_EQ(*node->raw_chunk_limit(), capacity);

  const std::vector<uint8_t> too_big(capacity + 1, 0);
  EXPECT_FALSE(node->enqueue(too_big.data(), too_big.size(), round_type::BULK));
  EXPECT_EQ(*node->raw_chunk_limit(), capacity) << "buffer was written anyway";

  const std::vector<uint8_t> exact(capacity, 0);
  EXPECT_TRUE(node->enqueue(exact.data(), capacity, round_type::BULK));
}

TEST(DecodingTaskGraph, EdgeFedWindowDecodesAWholeWindowPerValue) {
  // A routed value is one WHOLE window, decoded on arrival -- not one round
  // accumulated `window_size` times. Break-it, verified: restoring the old
  // `det_buf_.assign(ws_.size * d_->num_rows(), ...)` sizing aborts this test
  // on lut.cpp's `syndrome_str.length() == syndrome_size` assertion, because
  // the node hands the decoder a buffer ws_.size times too long. That assert
  // compiles out in release, where the lookup silently never matches and the
  // node abstains on every syndrome instead.
  auto node = make_identity_window({/*size=*/3, /*stride=*/3, 0, 0}, 3);
  detection_events dets;
  dets.bits.assign(3, 0.0f);
  dets.bits[1] = 1.0f; // a weight-1 syndrome the LUT resolves

  EXPECT_TRUE(node->enqueue_edge_value(edge_value{dets}, 0));
  const auto &lo = node->streaming_output(0)->get<logical_outcome>();
  EXPECT_EQ(lo.bits, (std::vector<uint8_t>{0, 1, 0}));
}

TEST(DecodingTaskGraph, EdgeFedWindowStreamMatchesTheRawByteStream) {
  // The two entry points into the same window must agree. An equivalence, so
  // it assumes nothing about the buffer layout -- which is what makes it a
  // regression test for the det_buf_ sizing rather than a restatement of it.
  constexpr uint32_t rounds = 3;
  auto raw = make_identity_window({rounds, rounds, 0, 0}, rounds);
  auto edged = make_identity_window({rounds, rounds, 0, 0}, rounds);

  for (uint32_t shot = 0; shot < 4; ++shot) {
    std::vector<uint8_t> bytes(rounds, 0);
    bytes[shot % rounds] = 1;
    detection_events dets;
    dets.bits.assign(bytes.begin(), bytes.end());

    ASSERT_TRUE(raw->enqueue(bytes.data(), bytes.size(), round_type::BULK))
        << "shot " << shot;
    ASSERT_TRUE(edged->enqueue_edge_value(edge_value{dets}, 0))
        << "shot " << shot;
    EXPECT_EQ(raw->streaming_output(0)->get<logical_outcome>().bits,
              edged->streaming_output(0)->get<logical_outcome>().bits)
        << "shot " << shot;
  }
  EXPECT_EQ(
      std::vector<uint8_t>(raw->corrections(), raw->corrections() + rounds),
      std::vector<uint8_t>(edged->corrections(),
                           edged->corrections() + rounds));
}

// Emits detection_events on its output slot. No shipped kind does yet, so
// this is the only way to build a `measurements -> producer -> window`
// topology and
// exercise the streaming-edge validation compile_wires performs.
class detector_source_node : public task_graph_node,
                             public streaming_task_graph_node {
public:
  detector_source_node(uint32_t spr, uint32_t n_dets, uint32_t n_obs,
                       window_spec schedule)
      : spr_(spr), schedule_(schedule), corr_(n_obs, 0) {
    val_ = edge_value{detection_events{}};
    val_.get<detection_events>().bits.assign(n_dets, 0.0f);
  }
  streaming_task_graph_node *streaming() override { return this; }
  bool enqueue(const uint8_t *, std::size_t len, round_type) override {
    bytes_in_ += len;
    if (bytes_in_ < num_msyn_per_decode())
      return false;
    bytes_in_ -= static_cast<std::size_t>(spr_) * schedule_.stride;
    return fired_ = true;
  }
  const uint8_t *corrections() const override { return corr_.data(); }
  void clear_corrections() override {}
  void reset_streaming_state() override {
    bytes_in_ = 0;
    fired_ = false;
  }
  std::size_t num_msyn_per_decode() const override {
    return static_cast<std::size_t>(spr_) * schedule_.size;
  }
  std::size_t num_msyn_per_round() const override { return spr_; }
  std::size_t num_observables() const override { return corr_.size(); }
  std::optional<window_spec> streaming_schedule() const override {
    return schedule_;
  }
  std::optional<std::size_t> raw_chunk_limit() const override {
    return num_msyn_per_decode() - bytes_in_;
  }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("detection_events");
    return std::nullopt;
  }
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    if (slot != 0)
      return std::nullopt;
    return val_.get<detection_events>().bits.size();
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == edge_value_type("measurement_results");
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return (slot == 0 && fired_) ? &val_ : nullptr;
  }
  void execute_indexed(const std::vector<edge_value> &,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = val_;
  }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }

private:
  uint32_t spr_;
  window_spec schedule_;
  std::vector<uint8_t> corr_;
  edge_value val_;
  std::size_t bytes_in_ = 0;
  bool fired_ = false;
};

static const bool detector_source_registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "test_detector_source";
  spec.requires_decoder = false;
  decoding::config::decoder_schema schema;
  schema.params = window_kind_common_params();
  schema.params.push_back({"step_size", decoding::config::param_kind::uint64});
  schema.params.push_back(
      {"start_guard", decoding::config::param_kind::uint64});
  schema.params.push_back({"end_guard", decoding::config::param_kind::uint64});
  register_node_kind(
      std::move(spec), std::move(schema), [](node_build_context &ctx) {
        const auto &params = *ctx.node_params;
        const auto size = static_cast<uint32_t>(
            params.get<std::size_t>("window_size", ctx.dem_rounds));
        const auto stride =
            static_cast<uint32_t>(params.get<std::size_t>("step_size", size));
        const auto start_guard =
            static_cast<uint32_t>(params.get<std::size_t>("start_guard", 0));
        const auto end_guard =
            static_cast<uint32_t>(params.get<std::size_t>("end_guard", 0));
        return std::unique_ptr<task_graph_node>(new detector_source_node(
            ctx.spr, ctx.d->num_rows(), ctx.o->num_rows(),
            {size, stride, start_guard, end_guard}));
      });
  return true;
}();

// measurements -> producer -> window, the shape a residual topology has.
// `step_size`
// decides whether the consumer is legal, so the two cases differ in one
// parameter and nothing else.
static std::unique_ptr<decoding_task_graph>
build_edge_fed_window_graph(std::size_t consumer_step) {
  task_graph_node_config src;
  src.id = "src";
  src.kind = "test_detector_source";
  src.dem = rep5_streaming_spec();
  src.node_params.insert("window_size", std::size_t{3});

  task_graph_node_config win;
  win.id = "win";
  win.kind = "sliding_window";
  win.dem = rep5_streaming_spec();
  win.decoder_type = "single_error_lut";
  win.node_params.insert("window_size", std::size_t{3});
  win.node_params.insert("step_size", consumer_step);

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(src));
  nodes.push_back(std::move(win));
  return decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "src", "input"},
                         {"src", "output", "win", "input"},
                         {"win", "output", "output", "input"}});
}

// Takes a logical_outcome and re-emits it. No shipped kind consumes one while
// streaming, so this is the only way to build a three-hop streaming chain and
// check that the middle node's output is delivered onward.
class relay_node : public task_graph_node, public streaming_task_graph_node {
public:
  relay_node(std::size_t n_obs, std::size_t schedule_factor,
             bool declares_schedule)
      : corr_(n_obs, 0), schedule_factor_(schedule_factor),
        declares_schedule_(declares_schedule) {
    val_ = edge_value{logical_outcome{}};
    val_.get<logical_outcome>().bits.assign(n_obs, 0);
  }
  streaming_task_graph_node *streaming() override { return this; }
  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    return false;
  }
  bool enqueue_edge_value(const edge_value &v, std::size_t) override {
    ++received_;
    if ((received_ - 1) % schedule_factor_ != 0)
      return false;
    const auto &lo = v.get<logical_outcome>();
    val_.get<logical_outcome>() = lo;
    for (std::size_t i = 0; i < corr_.size() && i < lo.bits.size(); ++i)
      corr_[i] ^= lo.bits[i];
    output_valid_ = true;
    ++emitted_;
    return true;
  }
  const uint8_t *corrections() const override { return corr_.data(); }
  void clear_corrections() override {
    std::fill(corr_.begin(), corr_.end(), 0);
  }
  void reset_streaming_state() override {
    received_ = 0;
    emitted_ = 0;
    output_valid_ = false;
    std::fill(corr_.begin(), corr_.end(), 0);
  }
  // This node can only be driven by an edge. Reporting a raw-byte capacity
  // would make an unwired relay look reachable to compile_wires().
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return corr_.size(); }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("logical_outcome");
    return std::nullopt;
  }
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    return slot == 0 ? std::optional<std::size_t>{corr_.size()} : std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == edge_value_type("logical_outcome");
  }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override {
    if (!declares_schedule_ || output_slot != 0 || inputs.size() != 1 ||
        inputs[0].slot != 0)
      return std::nullopt;
    auto schedule = inputs[0].schedule;
    schedule.stride *= schedule_factor_;
    return schedule;
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return (slot == 0 && output_valid_) ? &val_ : nullptr;
  }
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = inputs[0];
  }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::size_t received() const { return received_; }
  std::size_t emitted() const { return emitted_; }

private:
  std::vector<uint8_t> corr_;
  edge_value val_;
  std::size_t schedule_factor_;
  bool declares_schedule_;
  std::size_t received_ = 0;
  std::size_t emitted_ = 0;
  bool output_valid_ = false;
};

static relay_node *last_relay = nullptr;

static const bool relay_registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "test_relay";
  spec.requires_decoder = false;
  decoding::config::decoder_schema schema;
  schema.params = window_kind_common_params();
  schema.params.push_back(
      {"schedule_factor", decoding::config::param_kind::uint64});
  schema.params.push_back(
      {"declare_schedule", decoding::config::param_kind::boolean});
  register_node_kind(
      std::move(spec), std::move(schema), [](node_build_context &ctx) {
        const auto &params = *ctx.node_params;
        const auto factor = params.get<std::size_t>("schedule_factor", 1);
        if (factor == 0)
          throw std::invalid_argument(
              "test_relay: schedule_factor must be positive");
        auto node = new relay_node(ctx.o->num_rows(), factor,
                                   params.get<bool>("declare_schedule", true));
        last_relay = node;
        return std::unique_ptr<task_graph_node>(node);
      });
  return true;
}();

class routing_probe_node : public task_graph_node,
                           public streaming_task_graph_node {
public:
  explicit routing_probe_node(std::string mode) : mode_(std::move(mode)) {}

  streaming_task_graph_node *streaming() override { return this; }
  bool enqueue(const uint8_t *, std::size_t len, round_type) override {
    if (mode_ == "throwing")
      throw std::runtime_error("test streaming failure");
    if (mode_ == "guarded_noop_tail") {
      bytes_in_ += len;
      if (bytes_in_ < 3)
        return false;
      bytes_in_ = 0;
      return true;
    }
    return false;
  }
  const uint8_t *corrections() const override { return &correction_; }
  void clear_corrections() override { correction_ = 0; }
  void reset_streaming_state() override {
    correction_ = 0;
    bytes_in_ = 0;
  }
  std::size_t num_msyn_per_decode() const override {
    return mode_ == "guarded_noop_tail" ? 3 : 1;
  }
  std::size_t num_observables() const override { return 1; }
  std::optional<window_spec> streaming_schedule() const override {
    if (mode_ == "guarded_noop_tail")
      return window_spec{3, 2, 0, 1};
    return streaming_task_graph_node::streaming_schedule();
  }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return mode_ == "multi_raw" ? slot < 2
                                : (mode_ == "mixed" || mode_ == "throwing" ||
                                   mode_ == "guarded_noop_tail") &&
                                      slot == 0;
  }
  bool supports_streaming_input(std::size_t slot) const override {
    return (mode_ == "mixed" && slot == 1) || (mode_ == "edge" && slot == 0);
  }
  bool enqueue_edge_value(const edge_value &, std::size_t) override {
    return false;
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("logical_outcome");
    return std::nullopt;
  }
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    return slot == 0 ? std::optional<std::size_t>{1} : std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    if (mode_ == "multi_raw")
      return slot < 2 && type == edge_value_type("measurement_results");
    if (mode_ == "mixed")
      return (slot == 0 && type == edge_value_type("measurement_results")) ||
             (slot == 1 && type == edge_value_type("detection_events"));
    if (mode_ == "edge")
      return slot == 0 && type == edge_value_type("logical_outcome");
    return (mode_ == "no_raw_support" || mode_ == "throwing" ||
            mode_ == "guarded_noop_tail") &&
           slot == 0 && type == edge_value_type("measurement_results");
  }
  void execute_indexed(const std::vector<edge_value> &,
                       std::vector<edge_value> &outputs) const override {
    logical_outcome result;
    result.bits = {0};
    result.converged = true;
    outputs[0] = std::move(result);
  }
  std::vector<std::string> inputs() const override {
    if (mode_ == "multi_raw")
      return {"a", "b"};
    if (mode_ == "mixed")
      return {"raw", "edge"};
    if (mode_ == "no_inputs")
      return {};
    return {"input"};
  }
  std::vector<std::string> outputs() const override { return {"output"}; }

private:
  std::string mode_;
  std::size_t bytes_in_ = 0;
  uint8_t correction_ = 0;
};

static const bool routing_probe_registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "test_routing_probe";
  spec.requires_decoder = false;
  spec.requires_dem = false;
  decoding::config::decoder_schema schema;
  schema.params.push_back(
      {"mode", decoding::config::param_kind::string, /*required=*/true});
  register_node_kind(std::move(spec), std::move(schema),
                     [](node_build_context &ctx) {
                       return std::make_unique<routing_probe_node>(
                           ctx.node_params->get<std::string>("mode"));
                     });
  return true;
}();

static task_graph_node_config routing_probe_config(std::string id,
                                                   std::string mode) {
  task_graph_node_config node;
  node.id = std::move(id);
  node.kind = "test_routing_probe";
  node.node_params.insert("mode", std::move(mode));
  return node;
}

class ordered_stream_node : public task_graph_node,
                            public streaming_task_graph_node {
public:
  ordered_stream_node(std::string mode, uint8_t value)
      : mode_(std::move(mode)), value_(value) {
    logical_outcome output;
    output.bits = {value};
    output.converged = true;
    value_out_ = edge_value{std::move(output)};
  }

  streaming_task_graph_node *streaming() override { return this; }
  bool enqueue(const uint8_t *syndrome, std::size_t len, round_type) override {
    if (is_source() && syndrome != nullptr && len != 0)
      value_out_.get<logical_outcome>().bits[0] = syndrome[0] ^ value_;
    if (mode_ == "tail_source" || mode_ == "throwing_tail_source") {
      tail_pending_ = true;
      return false;
    }
    valid_ = true;
    return true;
  }
  bool flush() override {
    if ((mode_ != "tail_source" && mode_ != "throwing_tail_source") ||
        !tail_pending_)
      return false;
    if (throw_on_flush_)
      throw std::runtime_error("ordered stream flush failure");
    tail_pending_ = false;
    valid_ = true;
    return true;
  }
  bool enqueue_edge_value(const edge_value &value, std::size_t slot) override {
    if (mode_ == "join" || mode_ == "throwing_join") {
      std::lock_guard<std::mutex> lock(observed_mutex_);
      observed_slots_.push_back(slot);
      observed_deliveries_.emplace_back(slot,
                                        value.get<logical_outcome>().bits[0]);
    }
    if (mode_ == "throwing_join")
      throw std::runtime_error("ordered stream failure " +
                               std::to_string(slot));
    value_out_.get<logical_outcome>() = value.get<logical_outcome>();
    correction_ ^= value.get<logical_outcome>().bits[0];
    valid_ = true;
    return true;
  }
  const uint8_t *corrections() const override { return &correction_; }
  void clear_corrections() override { correction_ = 0; }
  void reset_streaming_state() override {
    correction_ = 0;
    valid_ = false;
    tail_pending_ = false;
    throw_on_flush_ = false;
  }
  std::size_t num_msyn_per_decode() const override {
    return is_source() ? 1 : 0;
  }
  std::size_t num_msyn_per_round() const override { return 1; }
  std::size_t num_observables() const override { return 1; }
  std::optional<std::size_t> raw_chunk_limit() const override {
    return is_source() ? std::optional<std::size_t>{1} : std::nullopt;
  }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return is_source() && slot == 0;
  }
  bool supports_streaming_input(std::size_t slot) const override {
    return mode_ == "pass"
               ? slot == 0
               : (mode_ == "join" || mode_ == "throwing_join") && slot < 2;
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override {
    if (is_source())
      return streaming_task_graph_node::derive_streaming_schedule(output_slot,
                                                                  inputs);
    const auto expected_inputs = mode_ == "pass" ? 1u : 2u;
    if (output_slot != 0 || inputs.size() != expected_inputs ||
        (inputs.size() == 2 && !(inputs[0].schedule == inputs[1].schedule)))
      return std::nullopt;
    return inputs[0].schedule;
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("logical_outcome");
    return std::nullopt;
  }
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    return slot == 0 ? std::optional<std::size_t>{1} : std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    if (is_source())
      return slot == 0 && type == edge_value_type("measurement_results");
    return slot < (mode_ == "pass" ? 1u : 2u) &&
           type == edge_value_type("logical_outcome");
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return slot == 0 && valid_ ? &value_out_ : nullptr;
  }
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    if (is_source())
      outputs[0] = value_out_;
    else
      outputs[0] = inputs[value_ % inputs.size()];
  }
  std::vector<std::string> inputs() const override {
    if (is_source() || mode_ == "pass")
      return {"input"};
    return {"a", "b"};
  }
  std::vector<std::string> outputs() const override { return {"output"}; }

  static void clear_observed() {
    std::lock_guard<std::mutex> lock(observed_mutex_);
    observed_slots_.clear();
    observed_deliveries_.clear();
  }
  static std::vector<std::size_t> observed() {
    std::lock_guard<std::mutex> lock(observed_mutex_);
    return observed_slots_;
  }
  static std::vector<std::pair<std::size_t, uint8_t>> observed_deliveries() {
    std::lock_guard<std::mutex> lock(observed_mutex_);
    return observed_deliveries_;
  }

private:
  bool is_source() const {
    return mode_ == "source" || mode_ == "tail_source" ||
           mode_ == "throwing_tail_source";
  }

  std::string mode_;
  uint8_t value_;
  uint8_t correction_ = 0;
  bool valid_ = false;
  bool tail_pending_ = false;
  bool throw_on_flush_ = mode_ == "throwing_tail_source";
  edge_value value_out_;
  inline static std::mutex observed_mutex_;
  inline static std::vector<std::size_t> observed_slots_;
  inline static std::vector<std::pair<std::size_t, uint8_t>>
      observed_deliveries_;
};

static const bool ordered_stream_registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "test_ordered_stream";
  spec.requires_decoder = false;
  spec.requires_dem = false;
  decoding::config::decoder_schema schema;
  schema.params.push_back(
      {"mode", decoding::config::param_kind::string, /*required=*/true});
  schema.params.push_back({"value", decoding::config::param_kind::uint64});
  register_node_kind(std::move(spec), std::move(schema),
                     [](node_build_context &ctx) {
                       return std::make_unique<ordered_stream_node>(
                           ctx.node_params->get<std::string>("mode"),
                           static_cast<uint8_t>(
                               ctx.node_params->get<std::size_t>("value", 0)));
                     });
  return true;
}();

class lockstep_probe {
public:
  void block() {
    std::unique_lock<std::mutex> lock(mutex_);
    slow_entered_ = true;
    changed_.notify_all();
    release_.wait(lock, [&] { return released_; });
  }

  void advance() {
    std::lock_guard<std::mutex> lock(mutex_);
    downstream_entered_ = true;
    changed_.notify_all();
  }

  bool wait_for_slow() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(2),
                             [&] { return slow_entered_; });
  }

  bool wait_for_downstream() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(2),
                             [&] { return downstream_entered_; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    release_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::condition_variable release_;
  bool slow_entered_ = false;
  bool downstream_entered_ = false;
  bool released_ = false;
};

class lockstep_stream_node : public task_graph_node,
                             public streaming_task_graph_node {
public:
  lockstep_stream_node(std::string mode, std::shared_ptr<lockstep_probe> probe)
      : mode_(std::move(mode)), probe_(std::move(probe)) {
    logical_outcome output;
    output.bits = {0};
    output_ = edge_value{std::move(output)};
  }

  streaming_task_graph_node *streaming() override { return this; }

  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    if (mode_ == "slow_raw")
      probe_->block();
    if (mode_ == "slow_flush" || mode_ == "fast_flush") {
      tail_pending_ = true;
      return false;
    }
    valid_ = true;
    return true;
  }

  bool enqueue_edge_value(const edge_value &, std::size_t) override {
    if (mode_ == "slow_edge")
      probe_->block();
    if (mode_ == "observer")
      probe_->advance();
    valid_ = true;
    return true;
  }

  bool flush() override {
    if (!tail_pending_)
      return false;
    if (mode_ == "slow_flush")
      probe_->block();
    tail_pending_ = false;
    valid_ = true;
    return true;
  }

  const uint8_t *corrections() const override { return &correction_; }
  void clear_corrections() override {}
  void reset_streaming_state() override {
    valid_ = false;
    tail_pending_ = false;
  }
  std::size_t num_msyn_per_decode() const override {
    return is_source() ? 1 : 0;
  }
  std::size_t num_msyn_per_round() const override { return 1; }
  std::size_t num_observables() const override { return 1; }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return is_source() && slot == 0;
  }
  bool supports_streaming_input(std::size_t slot) const override {
    return !is_source() && slot == 0;
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override {
    if (output_slot != 0)
      return std::nullopt;
    if (is_source())
      return window_spec{1, 1, 0, 0};
    if (inputs.size() != 1)
      return std::nullopt;
    return inputs.front().schedule;
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("logical_outcome");
    return std::nullopt;
  }
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    return slot == 0 ? std::optional<std::size_t>{1} : std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 &&
           type == edge_value_type(is_source() ? "measurement_results"
                                               : "logical_outcome");
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return slot == 0 && valid_ ? &output_ : nullptr;
  }
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = is_source() ? output_ : inputs[0];
  }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }

private:
  bool is_source() const {
    return mode_ == "slow_raw" || mode_ == "fast_raw" ||
           mode_ == "slow_flush" || mode_ == "fast_flush";
  }

  std::string mode_;
  std::shared_ptr<lockstep_probe> probe_;
  uint8_t correction_ = 0;
  bool valid_ = false;
  bool tail_pending_ = false;
  edge_value output_;
};

static std::shared_ptr<lockstep_probe> active_lockstep_probe;

static const bool lockstep_stream_registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "test_lockstep_stream";
  spec.requires_decoder = false;
  spec.requires_dem = false;
  decoding::config::decoder_schema schema;
  schema.params.push_back(
      {"mode", decoding::config::param_kind::string, /*required=*/true});
  register_node_kind(
      std::move(spec), std::move(schema), [](node_build_context &ctx) {
        return std::make_unique<lockstep_stream_node>(
            ctx.node_params->get<std::string>("mode"), active_lockstep_probe);
      });
  return true;
}();

static task_graph_node_config lockstep_node(std::string id, std::string mode) {
  task_graph_node_config config;
  config.id = std::move(id);
  config.kind = "test_lockstep_stream";
  config.node_params.insert("mode", std::move(mode));
  return config;
}

static void expect_fast_branch_advances(decoding_task_graph &graph,
                                        lockstep_probe &probe,
                                        round_type type = round_type::BULK) {
  const uint8_t zero = 0;
  auto enqueue = std::async(std::launch::async, [&] {
    return graph.enqueue_syndrome(&zero, 1, type);
  });
  const bool slow_entered = probe.wait_for_slow();
  const bool downstream_entered = slow_entered && probe.wait_for_downstream();
  probe.release();

  EXPECT_TRUE(slow_entered);
  EXPECT_TRUE(downstream_entered);
  EXPECT_NO_THROW((void)enqueue.get());
}

TEST(DecodingTaskGraph, RawStreamingHasNoGlobalPhaseBarrier) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "event-driven graph execution needs two CPU threads";
  ASSERT_TRUE(lockstep_stream_registered);
  active_lockstep_probe = std::make_shared<lockstep_probe>();

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(lockstep_node("slow", "slow_raw"));
  nodes.push_back(lockstep_node("fast", "fast_raw"));
  nodes.push_back(lockstep_node("observer", "observer"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "slow", "input"},
                         {"measurements", "output", "fast", "input"},
                         {"fast", "output", "observer", "input"},
                         {"slow", "output", "output", "slow"},
                         {"observer", "output", "output", "fast"}});

  expect_fast_branch_advances(*graph, *active_lockstep_probe);
}

TEST(DecodingTaskGraph, RoutedStreamingHasNoGlobalPhaseBarrier) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "event-driven graph execution needs two CPU threads";
  ASSERT_TRUE(lockstep_stream_registered);
  active_lockstep_probe = std::make_shared<lockstep_probe>();

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(lockstep_node("source", "fast_raw"));
  nodes.push_back(lockstep_node("slow", "slow_edge"));
  nodes.push_back(lockstep_node("fast", "fast_edge"));
  nodes.push_back(lockstep_node("observer", "observer"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"source", "output", "slow", "input"},
                         {"source", "output", "fast", "input"},
                         {"fast", "output", "observer", "input"},
                         {"slow", "output", "output", "slow"},
                         {"observer", "output", "output", "fast"}});

  expect_fast_branch_advances(*graph, *active_lockstep_probe);
}

TEST(DecodingTaskGraph, StreamingFlushHasNoGlobalPhaseBarrier) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "event-driven graph execution needs two CPU threads";
  ASSERT_TRUE(lockstep_stream_registered);
  active_lockstep_probe = std::make_shared<lockstep_probe>();

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(lockstep_node("slow", "slow_flush"));
  nodes.push_back(lockstep_node("fast", "fast_flush"));
  nodes.push_back(lockstep_node("observer", "observer"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "slow", "input"},
                         {"measurements", "output", "fast", "input"},
                         {"fast", "output", "observer", "input"},
                         {"slow", "output", "output", "slow"},
                         {"observer", "output", "output", "fast"}});

  expect_fast_branch_advances(*graph, *active_lockstep_probe,
                              round_type::FINAL);
}

TEST(DecodingTaskGraph, LargeRawInputReusesBoundedContinuationSlot) {
  ASSERT_TRUE(ordered_stream_registered);
  task_graph_node_config source;
  source.id = "source";
  source.kind = "test_ordered_stream";
  source.node_params.insert("mode", std::string{"source"});
  source.node_params.insert("value", std::size_t{0});
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(source));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"source", "output", "output", "result"}});
  const std::vector<uint8_t> syndromes(256, 0);

  EXPECT_TRUE(graph->enqueue_syndrome(syndromes.data(), syndromes.size()));
  EXPECT_EQ(graph->streaming_result().generation, syndromes.size());
}

TEST(DecodingTaskGraph, AMiddleNodesOutputIsRoutedOnward) {
  // measurements -> src -> win -> relay. Only `src` is fed raw, so a sweep
  // that routes just its raw-fed nodes fires `win` and stops: the relay never
  // sees the outcome. Routing has to follow the cascade however deep it runs.
  ASSERT_TRUE(detector_source_registered && relay_registered);
  last_relay = nullptr;

  task_graph_node_config src;
  src.id = "src";
  src.kind = "test_detector_source";
  src.dem = rep5_streaming_spec();
  src.node_params.insert("window_size", std::size_t{3});

  task_graph_node_config win;
  win.id = "win";
  win.kind = "sliding_window";
  win.dem = rep5_streaming_spec();
  win.decoder_type = "single_error_lut";
  win.node_params.insert("window_size", std::size_t{3});
  win.node_params.insert("step_size", std::size_t{3});

  task_graph_node_config relay;
  relay.id = "relay";
  relay.kind = "test_relay";
  relay.dem = rep5_streaming_spec();
  relay.node_params.insert("window_size", std::size_t{3});

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(src));
  nodes.push_back(std::move(win));
  nodes.push_back(std::move(relay));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "src", "input"},
                         {"src", "output", "win", "input"},
                         {"win", "output", "relay", "input"},
                         {"relay", "output", "output", "input"}});
  ASSERT_NE(last_relay, nullptr);

  const std::size_t n = graph->get_num_msyn_per_decode();
  std::vector<uint8_t> bytes(n, 0);
  for (int i = 0; i < 4; ++i)
    graph->enqueue_syndrome(bytes.data(), n);

  EXPECT_GT(last_relay->received(), 0u)
      << "the window fired but its outcome never reached the relay";
  graph->reset_decoder();
  EXPECT_EQ(last_relay->received(), 0u);
}

TEST(DecodingTaskGraph, StreamingRelayDoesNotDoubleCountItsProducer) {
  ASSERT_TRUE(relay_registered);
  last_relay = nullptr;

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("win"));
  task_graph_node_config relay;
  relay.id = "relay";
  relay.kind = "test_relay";
  relay.dem = rep5_streaming_spec();
  relay.node_params.insert("window_size", std::size_t{3});
  nodes.push_back(std::move(relay));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "win", "input"},
                         {"win", "output", "relay", "input"},
                         {"relay", "output", "output", "input"}});

  const auto model = rep5_model::build();
  std::vector<uint8_t> measurements;
  for (std::size_t fault = 0; fault < model.n_mech; ++fault) {
    auto candidate = model.measurements_for(model.syndrome_of({fault}));
    measurement_results shot;
    shot.bits = candidate;
    if (graph->run(shot).bits[0] & 1u) {
      measurements = std::move(candidate);
      break;
    }
  }
  ASSERT_FALSE(measurements.empty());
  EXPECT_EQ(stream_shot(*graph, measurements), 1u);
}

static void expect_streaming_conflict(decoding_task_graph &graph,
                                      const std::string &diagnostic_fragment) {
  EXPECT_FALSE(graph.supports_streaming());
  EXPECT_NE(graph.streaming_incompatibility_reason().find(diagnostic_fragment),
            std::string::npos)
      << graph.streaming_incompatibility_reason();

  std::vector<uint8_t> bytes(graph.get_num_msyn_per_decode(), 0);
  try {
    (void)graph.enqueue_syndrome(bytes.data(), bytes.size());
    FAIL() << "streaming execution accepted an incompatible graph";
  } catch (const std::logic_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find(diagnostic_fragment), std::string::npos) << msg;
    EXPECT_NE(msg.find("decodes only in batch"), std::string::npos) << msg;
  }
  EXPECT_THROW((void)graph.get_obs_corrections(), std::logic_error);
  EXPECT_THROW((void)graph.streaming_result(), std::logic_error);
}

static void expect_batch_valid_streaming_conflict(
    std::unique_ptr<decoding_task_graph> graph,
    const std::string &diagnostic_fragment) {
  ASSERT_NE(graph, nullptr);
  // This is intentionally only a construction/execution smoke check. Tests
  // making a claim about batch semantics must also use a complete,
  // discriminating shot and compare against an independent expected result.
  measurement_results shot;
  shot.bits.assign(graph->get_num_msyn_per_decode(), 0);
  EXPECT_NO_THROW((void)graph->run_all(shot));
  expect_streaming_conflict(*graph, diagnostic_fragment);
}

static void expect_streaming_works(std::unique_ptr<decoding_task_graph> graph) {
  ASSERT_NE(graph, nullptr);
  EXPECT_TRUE(graph->supports_streaming())
      << graph->streaming_incompatibility_reason();
  EXPECT_TRUE(graph->streaming_incompatibility_reason().empty());

  std::vector<uint8_t> bytes(graph->get_num_msyn_per_decode(), 0);
  bool fired = false;
  ASSERT_NO_THROW(fired = graph->enqueue_syndrome(bytes.data(), bytes.size()));
  EXPECT_TRUE(fired);
  EXPECT_NO_THROW((void)graph->get_obs_corrections());

  std::vector<uint8_t> round(graph->get_num_msyn_per_round(), 0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NO_THROW((void)graph->enqueue_syndrome(round.data(), round.size()));
    EXPECT_NO_THROW((void)graph->get_obs_corrections());
  }
}

TEST(DecodingTaskGraph, RawInputRequiresAnExplicitStreamingCapability) {
  ASSERT_TRUE(routing_probe_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(routing_probe_config("probe", "no_raw_support"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "probe", "input"},
                         {"probe", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph),
                                        "cannot consume raw measurements");
}

TEST(DecodingTaskGraph, MultipleRawInputsAreBatchOnly) {
  ASSERT_TRUE(routing_probe_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(routing_probe_config("probe", "multi_raw"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "probe", "a"},
                         {"measurements", "output", "probe", "b"},
                         {"probe", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph),
                                        "multiple raw measurement inputs");
}

TEST(DecodingTaskGraph, MixedRawAndEdgeInputsAreBatchOnly) {
  ASSERT_TRUE(detector_source_registered && routing_probe_registered);
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config source;
  source.id = "source";
  source.kind = "test_detector_source";
  source.dem = rep5_streaming_spec();
  source.node_params.insert("window_size", std::size_t{5});
  nodes.push_back(std::move(source));
  nodes.push_back(routing_probe_config("probe", "mixed"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"measurements", "output", "probe", "raw"},
                         {"source", "output", "probe", "edge"},
                         {"probe", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(
      std::move(graph), "mixes raw measurement and streaming-edge inputs");
}

TEST(DecodingTaskGraph, StreamingNodeWithNoInputIsBatchOnly) {
  ASSERT_TRUE(routing_probe_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(routing_probe_config("probe", "no_inputs"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"probe", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph),
                                        "has no streaming input");
}

TEST(DecodingTaskGraph, EdgeFedSlidingConsumerIsBatchOnly) {
  // A slid window's leading detector band is measured against the initial
  // state and cannot be rebuilt from a routed value. compile_wires asks the
  // consumer to vet the edge for streaming, while batch execution still routes
  // the complete detection_events value through execute_indexed().
  ASSERT_TRUE(detector_source_registered);
  expect_batch_valid_streaming_conflict(
      build_edge_fed_window_graph(/*consumer_step=*/1), "cannot slide");

  // Positive control: the same topology whose consumer does not slide streams,
  // so it is the stride being refused and not the wiring.
  auto compatible = build_edge_fed_window_graph(/*consumer_step=*/3);
  std::vector<uint8_t> bytes(compatible->get_num_msyn_per_decode(), 0);
  EXPECT_NO_THROW(
      (void)compatible->enqueue_syndrome(bytes.data(), bytes.size()));
}

TEST(DecodingTaskGraph, WindowNodeOneShotExecutionAcceptsDetectionEvents) {
  // The one-shot path takes either form, matching the two streaming entry
  // points; before the fix it did std::get<measurement_results> and threw
  // std::bad_variant_access on a residual edge.
  auto node = make_identity_window({3, 3, 0, 0}, 3);
  detection_events dets;
  dets.bits.assign(3, 0.0f);
  dets.bits[2] = 1.0f;
  std::vector<edge_value> in{edge_value{dets}}, out(1);

  node->execute_indexed(in, out);
  EXPECT_EQ(out[0].get<logical_outcome>().bits,
            (std::vector<uint8_t>{0, 0, 1}));

  dets.bits.assign(4, 0.0f); // wrong detector count
  in[0] = edge_value{dets};
  EXPECT_THROW(node->execute_indexed(in, out), std::out_of_range);
}

TEST(DecodingTaskGraph, CombineStreamingBroadcastUnchangedAfterRouting) {
  // combine() is a broadcast topology: each node receives raw syndrome bytes
  // independently (no streaming edges between them). Verify that the new
  // enqueue_syndrome topological walk produces the same result as before —
  // both nodes fire and corrections XOR to zero for zero syndromes.
  const auto spec = rep5_spec(5);
  auto left = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));

  const std::size_t n = combined->get_num_msyn_per_decode();
  std::vector<uint8_t> zeros(n, 0);
  EXPECT_TRUE(combined->enqueue_syndrome(zeros.data(), n));

  for (std::size_t i = 0; i < combined->get_num_observables(); ++i)
    EXPECT_EQ(combined->get_obs_corrections()[i], 0u);
}

// ---------------------------------------------------------------------------
// Guarded commit
// ---------------------------------------------------------------------------

// Predicts every fault on every decode, so what reaches the corrections is
// decided entirely by the window's commit range rather than by the decoder.
class all_faults_decoder final : public decoder {
public:
  // Only H is taken: this decoder ignores its input, and the observable model
  // is applied by the window_node rather than by the decoder.
  all_faults_decoder(sparse_binary_matrix h, std::size_t width)
      : decoder(decoder_init(std::move(h))), width_(width) {}

  decoder_result decode(const std::vector<float_t> &) override {
    return decoder_result{true, std::vector<float_t>(width_, 1.0f)};
  }

private:
  std::size_t width_;
};

// One-column-per-round window over `rounds` rounds, with one detector and one
// observable per round, so a committed round shows up as exactly one set bit.
static std::unique_ptr<window_node> make_guard_probe(window_spec ws,
                                                     uint32_t rounds) {
  std::vector<std::vector<uint32_t>> identity(rounds);
  for (uint32_t i = 0; i < rounds; ++i)
    identity[i].push_back(i);

  auto h = sparse_binary_matrix::from_nested_csr(rounds, rounds, identity);
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(rounds, rounds, identity));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(rounds, rounds, identity));
  auto dec = std::make_unique<all_faults_decoder>(h, rounds);

  // One fault column per round, which is what a guard boundary is expressed in.
  std::vector<std::size_t> fault_offsets(rounds + 1);
  for (uint32_t i = 0; i <= rounds; ++i)
    fault_offsets[i] = i;

  return std::make_unique<window_node>(d, o, std::move(dec), /*spr=*/1, ws,
                                       std::move(fault_offsets));
}

// Identity D/H with every fault projected to one logical bit. A single
// syndrome byte therefore follows one round through overlapping windows and
// exposes duplicate commits as XOR cancellation.
static std::unique_ptr<window_node>
make_guarded_identity_probe(window_spec ws) {
  std::vector<std::vector<uint32_t>> identity(ws.size);
  std::vector<uint32_t> logical_row(ws.size);
  std::vector<std::size_t> fault_offsets(ws.size + 1);
  for (uint32_t i = 0; i < ws.size; ++i) {
    identity[i].push_back(i);
    logical_row[i] = i;
    fault_offsets[i] = i;
  }
  fault_offsets[ws.size] = ws.size;

  auto h = sparse_binary_matrix::from_nested_csr(ws.size, ws.size, identity);
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(ws.size, ws.size, identity));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, ws.size, {logical_row}));
  return std::make_unique<window_node>(d, o,
                                       get_decoder("single_error_lut", h, {}),
                                       /*spr=*/1, ws, std::move(fault_offsets));
}

// The bits one fire committed.
static std::vector<uint8_t> fire_once(window_node &node) {
  const uint8_t zero = 0;
  const bool fired = node.enqueue(&zero, 1, round_type::BULK);
  if (!fired)
    return {};
  return node.streaming_output(0)->get<logical_outcome>().bits;
}

// The bits one flush committed, or {} if it committed nothing.
static std::vector<uint8_t> flush_once(window_node &node) {
  if (!node.flush())
    return {};
  return node.streaming_output(0)->get<logical_outcome>().bits;
}

TEST(DecodingTaskGraph, UnguardedWindowCommitsEveryRound) {
  auto node = make_guard_probe({/*size=*/3, /*stride=*/1, 0, 0}, 3);

  EXPECT_TRUE(fire_once(*node).empty());
  EXPECT_TRUE(fire_once(*node).empty());
  EXPECT_EQ(fire_once(*node), (std::vector<uint8_t>{1, 1, 1}));
  EXPECT_EQ(fire_once(*node), (std::vector<uint8_t>{1, 1, 1}));
}

TEST(DecodingTaskGraph, GuardsWithholdBoundaryRoundsFromTheCommit) {
  auto node = make_guard_probe(
      {/*size=*/3, /*stride=*/1, /*start_guard=*/1, /*end_guard=*/1}, 3);

  fire_once(*node);
  fire_once(*node);
  // Nothing covered the leading rounds of the first window, so it commits them
  // and withholds only its trailing guard.
  EXPECT_EQ(fire_once(*node), (std::vector<uint8_t>{1, 1, 0}));
  // From then on the previous fire has committed round 0 of this window.
  EXPECT_EQ(fire_once(*node), (std::vector<uint8_t>{0, 1, 0}));
  EXPECT_EQ(fire_once(*node), (std::vector<uint8_t>{0, 1, 0}));
}

// ---------------------------------------------------------------------------
// End-of-stream flush
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, ResetWarnsWhenATailIsStillUncommitted) {
  // Losing the tail is silent for every caller that never flushes -- which is
  // every caller over the wire today -- so reset_streaming_state(), reached
  // here through reset_decoder(), says so. The warning is the only signal;
  // assert the state that triggers it is reachable and flushing first clears
  // it.
  auto node = make_guard_probe({4, 2, 1, 1}, 4);
  const uint8_t zero = 0;
  for (int i = 0; i < 4; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  node->reset_decoder(); // warns: a tail was withheld and never committed

  for (int i = 0; i < 4; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  ASSERT_TRUE(node->flush());
  node->reset_decoder(); // silent: nothing was left behind

  // Reset is still usable afterwards either way.
  EXPECT_NO_THROW(node->enqueue(&zero, 1, round_type::BULK));
}

TEST(DecodingTaskGraph, FlushCommitsExactlyTheWithheldTail) {
  // Flush commits the complement of what the last fire did, so the two cover
  // the window exactly once. A size-3 window needs three enqueues to fire at
  // all, and the first fire keeps its leading guard.
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  fire_once(*node);
  fire_once(*node);
  ASSERT_EQ(fire_once(*node), (std::vector<uint8_t>{1, 1, 0}));
  ASSERT_EQ(fire_once(*node), (std::vector<uint8_t>{0, 1, 0}));

  // Break-it: mask to [0, hi_col_) instead of [hi_col_, end) and this returns
  // the fire's own bits, double-committing the middle round.
  EXPECT_EQ(flush_once(*node), (std::vector<uint8_t>{0, 0, 1}));
}

// A guarded sliding_window graph, the only shape that withholds a tail.
static std::unique_ptr<decoding_task_graph>
make_guarded_graph(task_graph_options options = {},
                   bool arbitrary_length = false) {
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "w";
  node.kind = "sliding_window";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  node.node_params.insert("window_size", std::size_t{5});
  node.node_params.insert("step_size", std::size_t{3});
  node.node_params.insert("start_guard", std::size_t{1});
  node.node_params.insert("end_guard", std::size_t{1});
  if (arbitrary_length)
    node.node_params.insert("arbitrary_length", true);
  nodes.push_back(std::move(node));
  return decoding_task_graph::from_topology(std::move(nodes), {},
                                            std::move(options));
}

TEST(DecodingTaskGraph, AFinalRoundCommitsTheTailOnAFullWindow) {
  // The FINAL round also FIRES here, which is what makes this the test that
  // catches `fired || flush_sweep()`: the short-circuit skips the flush
  // exactly in this case, and the returned bool is true either way.  So the
  // assertion has to be on the corrections, not the return value.
  auto graph = make_guarded_graph();
  const std::size_t n = graph->get_num_msyn_per_decode();
  const std::size_t per_round = graph->get_num_msyn_per_round();
  const std::size_t obs = graph->get_num_observables();
  std::vector<uint8_t> shot(n, 0);
  shot[n - per_round] = 1; // lands in the end_guard

  ASSERT_TRUE(
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  const std::vector<uint8_t> tagged(graph->get_obs_corrections(),
                                    graph->get_obs_corrections() + obs);

  // Same shot without the tag commits strictly less: the tail stays withheld.
  auto plain = make_guarded_graph();
  ASSERT_TRUE(plain->enqueue_syndrome(shot.data(), shot.size()));
  const std::vector<uint8_t> untagged(plain->get_obs_corrections(),
                                      plain->get_obs_corrections() + obs);
  // Break-it: `return fired || impl_->flush_sweep();`
  EXPECT_NE(tagged, untagged) << "FINAL committed nothing the plain round did "
                                 "not -- the flush was skipped";
}

// Release contract: chunk/window geometry is chosen before runtime, but the
// control system must not constrain the shot length to a stride multiple.
TEST(DecodingTaskGraph, ArbitraryLengthFinalMatchesFiniteShotOracle) {
  for (const auto async : {false, true})
    for (const auto rounds :
         {std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{6},
          std::size_t{7}, std::size_t{8}, std::size_t{1000}, std::size_t{1001},
          std::size_t{1002}, std::size_t{1003}}) {
      SCOPED_TRACE(::testing::Message()
                   << "rounds=" << rounds << " async=" << async);
      auto graph = make_guarded_graph(
          {.async_streaming = async, .async_queue_capacity = 2}, true);
      auto oracle =
          decoding_task_graph::from_spec(rep5_spec(rounds), "single_error_lut");
      const auto width = graph->get_num_msyn_per_round();
      std::vector<uint8_t> shot(rounds * width);
      shot[(rounds - 1) * width] = 1;
      const auto expected = oracle->run(measurement_results{shot});
      ASSERT_EQ(expected.bits[0], 1);
      // Fixed three-round deliveries in the middle, then whatever remains.
      const auto chunk_bytes = 3 * width;
      for (int epoch = 0; epoch < 2; ++epoch) {
        std::size_t offset = 0;
        while (shot.size() - offset > chunk_bytes) {
          graph->enqueue_syndrome(shot.data() + offset, chunk_bytes);
          offset += chunk_bytes;
        }
        graph->enqueue_syndrome(shot.data() + offset, shot.size() - offset,
                                round_type::FINAL);
        EXPECT_EQ(graph->get_obs_corrections()[0], expected.bits[0]);
        const auto result = graph->streaming_result();
        EXPECT_EQ(result.status, task_graph_streaming_status::complete);
        ASSERT_EQ(result.outputs.size(), 1);
        EXPECT_EQ(result.outputs[0].coverage_end_round, rounds);
        graph->reset_decoder();
      }
    }
}

static dem_chunks_spec arbitrary_memory_spec(std::size_t rounds,
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
  }
  return spec;
}

static std::unique_ptr<decoding_task_graph> make_arbitrary_memory_graph(
    const std::string &name = "single_error_lut",
    task_graph_options options = {},
    const cudaqx::heterogeneous_map &params = {}, bool explicit_d = false,
    bool arbitrary_length = true, window_spec schedule = {5, 3, 1, 1},
    std::size_t observable = 0) {
  task_graph_node_config node;
  node.id = "memory";
  node.kind = "sliding_window";
  node.dem = arbitrary_memory_spec(schedule.size, explicit_d, observable);
  node.decoder_type = name;
  node.params = params;
  node.node_params = {{"window_size", std::size_t{schedule.size}},
                      {"step_size", std::size_t{schedule.stride}},
                      {"start_guard", std::size_t{schedule.start_guard}},
                      {"end_guard", std::size_t{schedule.end_guard}},
                      {"arbitrary_length", arbitrary_length}};
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  return decoding_task_graph::from_topology(std::move(nodes), {}, options);
}

TEST(ArbitraryLengthMemory, PyMatchingEveryFaultAndByteFragmentMatchesOracle) {
  const cudaqx::heterogeneous_map params{
      {"merge_strategy", std::string{"independent"}}};
  for (const auto explicit_d : {false, true})
    for (const auto rounds :
         {std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{6},
          std::size_t{7}, std::size_t{17}}) {
      auto graph =
          make_arbitrary_memory_graph("pymatching", {}, params, explicit_d);
      auto oracle = decoding_task_graph::from_spec(
          arbitrary_memory_spec(rounds, explicit_d), "pymatching", params);
      for (std::size_t fault = 0; fault < rounds; ++fault) {
        std::vector<uint8_t> shot(rounds * 4);
        shot[fault * 4] = 1;
        const auto expected = oracle->run(measurement_results{shot});
        ASSERT_EQ(expected.bits[0], 1);
        for (std::size_t i = 0; i < shot.size(); ++i)
          graph->enqueue_syndrome(shot.data() + i, 1,
                                  i + 1 == shot.size() ? round_type::FINAL
                                                       : round_type::BULK);
        EXPECT_EQ(graph->get_obs_corrections()[0], expected.bits[0]);
        EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round,
                  rounds);
        graph->clear_corrections();
      }
    }
}

static dem_chunks_spec multi_terminal_memory_spec(std::size_t rounds,
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
  }
  return spec;
}

TEST(ArbitraryLengthMemory, DistinctTerminalPhasesMatchClosedOracle) {
  const cudaqx::heterogeneous_map params{
      {"merge_strategy", std::string{"independent"}}};
  std::size_t nonzero = 0;
  for (const auto tail_count : {std::size_t{2}, std::size_t{3}})
    for (const bool asynchronous : {false, true}) {
      task_graph_node_config node;
      node.id = "memory";
      node.kind = "sliding_window";
      constexpr std::size_t window = 9, guard = 3;
      node.dem = multi_terminal_memory_spec(window, tail_count);
      node.decoder_type = "pymatching";
      node.params = params;
      node.node_params = {{"window_size", window},
                          {"step_size", window - 2 * guard},
                          {"start_guard", guard},
                          {"end_guard", guard},
                          {"arbitrary_length", true}};
      std::vector<task_graph_node_config> nodes;
      nodes.push_back(std::move(node));
      auto graph = decoding_task_graph::from_topology(
          std::move(nodes), {},
          {.async_streaming = asynchronous, .async_queue_capacity = 1});
      for (const auto rounds :
           {tail_count + 2, std::size_t{9}, std::size_t{10}, std::size_t{11},
            std::size_t{12}, std::size_t{13}, std::size_t{14}, std::size_t{15},
            std::size_t{17}, std::size_t{65}}) {
        auto oracle = decoding_task_graph::from_spec(
            multi_terminal_memory_spec(rounds, tail_count), "pymatching",
            params);
        for (std::size_t fault = 0; fault < rounds; ++fault) {
          std::vector<uint8_t> shot(rounds * 4);
          shot[fault * 4 + (fault + 1 == rounds ? 0 : 1)] = 1;
          const auto expected = oracle->run(measurement_results{shot});
          nonzero += expected.bits[0] != 0;
          // Byte fragmentation also splits the preterminal records.
          for (std::size_t i = 0; i < shot.size(); ++i)
            graph->enqueue_syndrome(shot.data() + i, 1,
                                    i + 1 == shot.size() ? round_type::FINAL
                                                         : round_type::BULK);
          EXPECT_EQ(graph->get_obs_corrections()[0], expected.bits[0]);
          EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round,
                    rounds);
          graph->reset_decoder();
        }
      }
    }
  EXPECT_GT(nonzero, 0u);
}

TEST(ArbitraryLengthMemory, MultipleTerminalPhasesRequireEnoughLeadingContext) {
  task_graph_node_config node;
  node.id = "memory";
  node.kind = "sliding_window";
  node.dem = multi_terminal_memory_spec(9, 3);
  node.decoder_type = "single_error_lut";
  node.node_params = {{"window_size", std::size_t{9}},
                      {"step_size", std::size_t{7}},
                      {"start_guard", std::size_t{1}},
                      {"end_guard", std::size_t{1}},
                      {"arbitrary_length", true}};
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  EXPECT_THROW(decoding_task_graph::from_topology(std::move(nodes), {}),
               std::invalid_argument);
}

TEST(ArbitraryLengthMemory, TerminalLookaheadHasFixedStorageAndFireDelay) {
  constexpr std::size_t window = 30, guard = 3, width = 4, rounds = 1003;
  constexpr std::size_t stride = window - 2 * guard;
  for (const auto tail_count : {std::size_t{2}, std::size_t{3}}) {
    const auto spec = multi_terminal_memory_spec(window, tail_count);
    const auto chunks = dem_chunks_from_spec(spec);
    const auto closed =
        dem_close_all(chunks, spec.seam.from_seam, spec.seam.to_seam);
    const auto rows = dem_chunks_to_d_sparse(spec);
    node_build_context ctx;
    ctx.memory_spec = std::make_shared<const dem_chunks_spec>(spec);
    ctx.d = std::make_shared<const sparse_binary_matrix>(
        sparse_binary_matrix::from_nested_csr(rows.size(), window * width,
                                              rows));
    ctx.o = std::make_shared<const sparse_binary_matrix>(
        closed.observables_flips_matrix, sparse_binary_matrix_layout::csr);
    ctx.decoder_type = "single_error_lut";
    ctx.dec = get_decoder(ctx.decoder_type,
                          decoder_init(closed.detector_error_matrix,
                                       std::nullopt, closed.error_rates),
                          decoder_output_request{decode_result_type::errors});
    ctx.spr = width;
    memory_window_node node(ctx, {window, stride, guard, guard});
    // The documented eager-model formula includes the base decoder.
    const auto prefix = spec.repeating_rounds().prefix;
    const auto expected_models = prefix / stride + prefix + 4 + window +
                                 tail_count - spec.minimum_rounds() + 1;
    ASSERT_EQ(node.prepared_decoder_count(), expected_models);
    ASSERT_EQ(node.retained_round_capacity(), window + tail_count + 1);
    ASSERT_EQ(node.streaming_schedule()->size, window + tail_count);
    ASSERT_EQ(node.streaming_schedule()->core(), stride);
    std::array<uint8_t, width> record{};
    for (std::size_t r = 0; r < rounds; ++r) {
      const bool terminal = r + 1 == rounds;
      record[0] = terminal ? 1 : 0;
      const bool fired =
          node.enqueue(record.data(), record.size(),
                       terminal ? round_type::FINAL : round_type::BULK);
      const bool expected_fire = !terminal && r + 1 >= window + tail_count &&
                                 (r + 1 - window - tail_count) % stride == 0;
      EXPECT_EQ(fired, expected_fire);
      EXPECT_EQ(node.prepared_decoder_count(), expected_models);
      EXPECT_EQ(node.retained_round_capacity(), window + tail_count + 1);
    }
    ASSERT_TRUE(node.flush());
    ASSERT_NE(node.streaming_output(0), nullptr);
    EXPECT_EQ(node.streaming_output(0)->get<logical_outcome>().bits[0], 1);
  }
}

TEST(ArbitraryLengthMemory, PartialRoundFinalFailsAndResetRecovers) {
  for (const auto async : {false, true}) {
    auto graph = make_arbitrary_memory_graph(
        "single_error_lut",
        {.async_streaming = async, .async_queue_capacity = 1});
    std::vector<uint8_t> partial(13);
    EXPECT_THROW(graph->enqueue_syndrome(partial.data(), partial.size(),
                                         round_type::FINAL),
                 std::invalid_argument);
    EXPECT_EQ(graph->streaming_result().status,
              task_graph_streaming_status::failed);
    EXPECT_THROW(graph->enqueue_syndrome(partial.data(), 1), std::exception);
    graph->reset_decoder();
    std::vector<uint8_t> shot(24);
    shot[20] = 1;
    graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL);
    EXPECT_EQ(graph->get_obs_corrections()[0], 1);
    EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round, 6);
  }
}

TEST(ArbitraryLengthMemory, SparseCarryMatchesMaskedRowProjection) {
  constexpr std::size_t detectors = 19, faults = 31;
  std::mt19937 random(8129);
  std::vector<std::vector<uint32_t>> rows(detectors);
  for (auto &row : rows)
    for (std::size_t col = 0; col < faults; ++col)
      if (random() % 5 == 0)
        row.push_back(col);
  const auto h = sparse_binary_matrix::from_nested_csr(detectors, faults, rows);
  const auto columns = h.to_csc();
  constexpr std::array<float_t, 5> estimates{0, 0.2, 0.49, 0.5, 1};
  for (std::size_t trial = 0; trial < 1000; ++trial) {
    auto begin = random() % (faults + 1);
    auto end = random() % (faults + 1);
    if (begin > end)
      std::swap(begin, end);
    std::vector<float_t> prediction(faults), masked(faults);
    for (std::size_t col = 0; col < faults; ++col) {
      prediction[col] = estimates[random() % estimates.size()];
      if (begin <= col && col < end)
        masked[col] = prediction[col];
    }
    std::vector<uint8_t> expected(detectors), actual(detectors);
    for (auto &bit : expected)
      bit = random() % 2;
    actual = expected; // Existing carry must be XORed, not replaced.
    project_into(csr_rows{h}, masked.data(), expected.data());
    scatter_fault_syndrome(columns, prediction.data(), begin, end,
                           [&](auto row) { actual[row] ^= uint8_t{1}; });
    EXPECT_EQ(actual, expected) << "trial " << trial;
  }
}
TEST(ArbitraryLengthMemory, MultipleFaultShotsMatchOracleAcrossFinalAndReset) {
  // This is execution equivalence on a fixed model, not production LER
  // qualification. Exercise more than isolated faults before skipping the
  // terminal carry update, including the next shot's reset/clear lifecycle.
  const cudaqx::heterogeneous_map params{
      {"merge_strategy", std::string{"independent"}}};
  std::mt19937 random(9173);
  std::bernoulli_distribution fault(0.05);
  std::size_t zero_outcomes = 0, nonzero_outcomes = 0;
  for (const bool explicit_d : {false, true})
    for (const bool asynchronous : {false, true})
      for (const auto rounds : {std::size_t{3}, std::size_t{17},
                                std::size_t{31}, std::size_t{1003}}) {
        SCOPED_TRACE(rounds);
        task_graph_options options;
        options.async_streaming = asynchronous;
        options.async_queue_capacity = 2;
        auto graph = make_arbitrary_memory_graph("pymatching", options, params,
                                                 explicit_d);
        auto oracle = decoding_task_graph::from_spec(
            arbitrary_memory_spec(rounds, explicit_d), "pymatching", params);
        for (std::size_t shot_index = 0; shot_index < 16; ++shot_index) {
          std::vector<uint8_t> shot(rounds * 4);
          for (auto &bit : shot)
            bit = fault(random);
          const auto expected = oracle->run(measurement_results{shot});
          expected.bits[0] ? ++nonzero_outcomes : ++zero_outcomes;
          std::size_t offset = 0;
          constexpr std::size_t chunk_bytes = 50 * 4;
          while (shot.size() - offset > chunk_bytes) {
            graph->enqueue_syndrome(shot.data() + offset, chunk_bytes);
            offset += chunk_bytes;
          }
          graph->enqueue_syndrome(shot.data() + offset, shot.size() - offset,
                                  round_type::FINAL);
          EXPECT_EQ(graph->get_obs_corrections()[0], expected.bits[0]);
          EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round,
                    rounds);
          if (shot_index % 2)
            graph->clear_corrections();
          else
            graph->reset_decoder();
        }
      }
  EXPECT_GT(zero_outcomes, 0u);
  EXPECT_GT(nonzero_outcomes, 0u);
}

TEST(ArbitraryLengthMemory, LongStreamUsesFixedWindowsAndSupportsClearEpochs) {
  constexpr std::size_t rounds = 1000003;
  auto graph = make_arbitrary_memory_graph(
      "single_error_lut", {.async_streaming = true, .async_queue_capacity = 2});
  EXPECT_EQ(graph->get_num_msyn_per_decode(), 20);
  const std::array<uint8_t, 12> chunk{};
  for (int epoch = 0; epoch < 2; ++epoch) {
    std::size_t delivered = 0;
    while (rounds - delivered > 3) {
      graph->enqueue_syndrome(chunk.data(), chunk.size());
      delivered += 3;
    }
    std::vector<uint8_t> tail((rounds - delivered) * 4);
    tail[tail.size() - 4] = 1;
    graph->enqueue_syndrome(tail.data(), tail.size(), round_type::FINAL);
    EXPECT_EQ(graph->get_obs_corrections()[0], 1);
    EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round, rounds);
    graph->clear_corrections();
  }
}

TEST(ArbitraryLengthMemory, GpuDecoderWhenRequested) {
  const auto *plugins =
      std::getenv("CUDAQ_QEC_ARBITRARY_LENGTH_GPU_PLUGIN_DIR");
  if (!plugins)
    GTEST_SKIP() << "Set CUDAQ_QEC_ARBITRARY_LENGTH_GPU_PLUGIN_DIR on "
                    "compatible hardware";
  load_plugins(plugins, PluginType::DECODER);
  const cudaqx::heterogeneous_map params{{"use_sparsity", true},
                                         {"bp_method", 1},
                                         {"max_iterations", 50},
                                         {"clip_value", 5.0},
                                         {"repeatable", true}};
  auto graph = make_arbitrary_memory_graph("nv-qldpc-decoder", {}, params);
  for (const auto rounds :
       {std::size_t{3}, std::size_t{5}, std::size_t{6}, std::size_t{1000},
        std::size_t{1001}, std::size_t{1002}, std::size_t{1003}}) {
    std::vector<uint8_t> shot(rounds * 4);
    shot[(rounds - 1) * 4] = 1;
    graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL);
    EXPECT_EQ(graph->get_obs_corrections()[0], 1);
    EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round, rounds);
    graph->reset_decoder();
  }
}

TEST(ArbitraryLengthMemory, FiniteShotRunUsesTheConfiguredClosedWindow) {
  auto graph = make_arbitrary_memory_graph();
  auto oracle = decoding_task_graph::from_spec(arbitrary_memory_spec(5),
                                               "single_error_lut");
  std::vector<uint8_t> shot(20);
  shot[16] = 1;
  const auto expected = oracle->run(measurement_results{shot});
  ASSERT_EQ(expected.bits[0], 1);
  EXPECT_EQ(graph->get_num_msyn_per_decode(), shot.size());
  EXPECT_EQ(graph->run(measurement_results{shot}).bits, expected.bits);
}

TEST(ArbitraryLengthMemory, PreparedModelAndRingCountsStayFixed) {
  constexpr std::size_t window = 30, rounds = 1003, width = 4;
  auto spec = arbitrary_memory_spec(window);
  const auto chunks = dem_chunks_from_spec(spec);
  const auto closed =
      dem_close_all(chunks, spec.seam.from_seam, spec.seam.to_seam);
  const auto rows = dem_chunks_to_d_sparse(spec);
  node_build_context ctx;
  ctx.memory_spec = std::make_shared<const dem_chunks_spec>(spec);
  ctx.d = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(rows.size(), window * width, rows));
  ctx.o = std::make_shared<const sparse_binary_matrix>(
      closed.observables_flips_matrix, sparse_binary_matrix_layout::csr);
  ctx.decoder_type = "single_error_lut";
  ctx.dec = get_decoder(ctx.decoder_type,
                        decoder_init(closed.detector_error_matrix, std::nullopt,
                                     closed.error_rates),
                        decoder_output_request{decode_result_type::errors});
  ctx.spr = width;
  memory_window_node node(ctx, {window, 28, 0, 2});
  ASSERT_EQ(node.prepared_decoder_count(), 33);
  ASSERT_EQ(node.retained_round_capacity(), window + 1);
  std::array<uint8_t, width> record{};
  for (std::size_t r = 0; r < rounds; ++r) {
    record[0] = r + 1 == rounds ? 1 : 0;
    node.enqueue(record.data(), record.size(),
                 r + 1 == rounds ? round_type::FINAL : round_type::BULK);
    EXPECT_EQ(node.prepared_decoder_count(), 33);
    EXPECT_EQ(node.retained_round_capacity(), window + 1);
  }
  ASSERT_TRUE(node.flush());
  EXPECT_EQ(node.corrections()[0], 1);
  EXPECT_EQ(node.streaming_coverage()->end_round, rounds);
}

TEST(ArbitraryLengthMemory, UnequalWidthRecipeIsRejected) {
  task_graph_node_config node;
  node.id = "unsupported-width";
  node.kind = "sliding_window";
  node.dem = arbitrary_memory_spec(5, true);
  node.dem->phases.back().spec.num_measurements = 5;
  node.decoder_type = "single_error_lut";
  node.node_params = {{"window_size", std::size_t{5}},
                      {"step_size", std::size_t{3}},
                      {"guard_rounds", std::size_t{1}},
                      {"arbitrary_length", true}};
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  EXPECT_THROW(decoding_task_graph::from_topology(std::move(nodes), {}),
               std::invalid_argument);
}

TEST(ArbitraryLengthMemory, RepeatedFinalDoesNotChangeCorrectionsOrCoverage) {
  auto graph = make_arbitrary_memory_graph();
  std::vector<uint8_t> shot(24);
  shot[20] = 1;
  graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL);
  const auto before = graph->streaming_result();
  ASSERT_EQ(before.outputs[0].coverage_end_round, 6);
  ASSERT_EQ(graph->get_obs_corrections()[0], 1);
  graph->enqueue_syndrome(shot.data() + 20, 4, round_type::FINAL);
  const auto after = graph->streaming_result();
  EXPECT_EQ(after.outputs[0].generation, before.outputs[0].generation);
  EXPECT_EQ(after.outputs[0].coverage_begin_round,
            before.outputs[0].coverage_begin_round);
  EXPECT_EQ(after.outputs[0].coverage_end_round,
            before.outputs[0].coverage_end_round);
  EXPECT_EQ(graph->get_obs_corrections()[0], 1);
}

TEST(ArbitraryLengthMemory, AsymmetricGuardsMatchEachNonzeroOracleFault) {
  const cudaqx::heterogeneous_map params{
      {"merge_strategy", std::string{"independent"}}};
  constexpr std::size_t rounds = 17;
  auto oracle = decoding_task_graph::from_spec(arbitrary_memory_spec(rounds),
                                               "pymatching", params);
  for (const auto schedule :
       {window_spec{5, 3, 0, 2}, window_spec{5, 2, 2, 1}}) {
    auto graph = make_arbitrary_memory_graph("pymatching", {}, params, false,
                                             true, schedule);
    for (std::size_t fault = 0; fault < rounds; ++fault) {
      std::vector<uint8_t> shot(rounds * 4);
      shot[fault * 4] = 1;
      const auto expected = oracle->run(measurement_results{shot});
      ASSERT_EQ(expected.bits[0], 1);
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL);
      EXPECT_EQ(graph->get_obs_corrections()[0], expected.bits[0]);
      EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round,
                rounds);
      graph->reset_decoder();
    }
  }
}

TEST(ArbitraryLengthMemory, FiftyRoundChunksAcceptAllTerminalRemainders) {
  auto graph = make_arbitrary_memory_graph("single_error_lut", {}, {}, false,
                                           true, {50, 40, 5, 5});
  const std::array<uint8_t, 200> chunk{};
  for (const auto rounds : {std::size_t{1000}, std::size_t{1001},
                            std::size_t{1002}, std::size_t{1003}}) {
    std::size_t delivered = 0;
    while (rounds - delivered > 50) {
      graph->enqueue_syndrome(chunk.data(), chunk.size());
      delivered += 50;
    }
    std::vector<uint8_t> tail((rounds - delivered) * 4);
    tail[tail.size() - 4] = 1;
    graph->enqueue_syndrome(tail.data(), tail.size(), round_type::FINAL);
    EXPECT_EQ(graph->get_obs_corrections()[0], 1);
    EXPECT_EQ(graph->streaming_result().outputs[0].coverage_end_round, rounds);
    graph->reset_decoder();
  }
}

TEST(ArbitraryLengthMemory, CombinedParallelBranchesPublishExactFinalCoverage) {
  const cudaqx::heterogeneous_map params{
      {"merge_strategy", std::string{"independent"}}};
  for (const auto async : {false, true}) {
    auto graph = decoding_task_graph::combine(
        make_arbitrary_memory_graph("pymatching", {}, params),
        make_arbitrary_memory_graph("single_error_lut", {}, {}, false, true,
                                    {5, 3, 1, 1}, 1),
        {.async_streaming = async, .async_queue_capacity = 2});
    for (const auto rounds :
         {std::size_t{6}, std::size_t{17}, std::size_t{1003}}) {
      std::vector<uint8_t> shot(rounds * 4);
      shot[(rounds - 1) * 4] = 1;
      // Channel zero flips only the left branch's observable, so dropped
      // tails cannot pass by cancelling identical branch results.
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL);
      EXPECT_EQ(graph->get_obs_corrections()[0], 1);
      const auto result = graph->streaming_result();
      EXPECT_EQ(result.status, task_graph_streaming_status::complete);
      EXPECT_EQ(result.outputs[0].coverage_end_round, rounds);
      graph->clear_corrections();
    }
  }
}

TEST(ArbitraryLengthMemory, ReportsTerminalCostWhenRequested) {
  if (!std::getenv("CUDAQ_QEC_ARBITRARY_LENGTH_BENCHMARK"))
    GTEST_SKIP()
        << "Set CUDAQ_QEC_ARBITRARY_LENGTH_BENCHMARK to measure FINAL cost";
  const auto *plugins =
      std::getenv("CUDAQ_QEC_ARBITRARY_LENGTH_GPU_PLUGIN_DIR");
  const std::string name = plugins ? "nv-qldpc-decoder" : "single_error_lut";
  cudaqx::heterogeneous_map params;
  if (plugins) {
    load_plugins(plugins, PluginType::DECODER);
    params = {{"use_sparsity", true},
              {"bp_method", 1},
              {"max_iterations", 50},
              {"clip_value", 5.0},
              {"repeatable", true}};
  }
  const auto *window_override =
      std::getenv("CUDAQ_QEC_ARBITRARY_LENGTH_BENCHMARK_WINDOW");
  const std::size_t window = window_override ? std::stoul(window_override) : 5;
  ASSERT_GE(window, 3u);
  // Benchmark bounds only: eager short-shot preparation is quadratic in W.
  constexpr std::size_t maximum_benchmark_window = 64;
  ASSERT_LE(window, maximum_benchmark_window);
  const window_spec schedule{static_cast<uint32_t>(window),
                             static_cast<uint32_t>(window - 2), 1, 1};
  // Aligned, so both paths decode the same shot with four regular strides.
  const std::size_t rounds = window + 4 * schedule.stride;
  constexpr std::size_t trials = 5, shots_per_trial = 100, warmup = 20;
  std::array<std::unique_ptr<decoding_task_graph>, 2> graphs;
  for (std::size_t mode = 0; mode < graphs.size(); ++mode)
    graphs[mode] = make_arbitrary_memory_graph(name, {}, params, false,
                                               mode != 0, schedule);
  std::vector<uint8_t> shot(rounds * 4);
  shot[(rounds - 1) * 4] = 1;
  const auto run = [&](std::size_t mode) {
    auto &graph = graphs[mode];
    graph->enqueue_syndrome(shot.data(), shot.size() - 4);
    const auto start = std::chrono::steady_clock::now();
    graph->enqueue_syndrome(shot.data() + shot.size() - 4, 4,
                            round_type::FINAL);
    const auto stop = std::chrono::steady_clock::now();
    EXPECT_EQ(graph->get_obs_corrections()[0], 1);
    graph->reset_decoder();
    return std::chrono::duration<double, std::micro>(stop - start).count();
  };
  for (std::size_t i = 0; i < warmup; ++i)
    for (std::size_t mode = 0; mode < graphs.size(); ++mode)
      run(mode);
  std::array<std::vector<double>, 2> means;
  // Alternate which mode runs first to avoid systematically measuring one
  // path with a different GPU clock, warm-up state or host scheduling phase.
  for (std::size_t trial = 0; trial < trials; ++trial)
    for (std::size_t position = 0; position < graphs.size(); ++position) {
      const auto mode = (trial + position) % graphs.size();
      double total = 0;
      for (std::size_t i = 0; i < shots_per_trial; ++i)
        total += run(mode);
      means[mode].push_back(total / shots_per_trial);
    }
  for (std::size_t mode = 0; mode < graphs.size(); ++mode) {
    std::sort(means[mode].begin(), means[mode].end());
    std::cout << "[memory FINAL benchmark] decoder=" << name
              << " arbitrary_length=" << mode << " rounds=" << rounds
              << " window=" << window << " stride=" << schedule.stride
              << " median_mean_final_us=" << means[mode][trials / 2] << '\n';
  }
}

TEST(DecodingTaskGraph, ABulkRoundDoesNotCommitTheTail) {
  // Not vacuous: the difference above is the tag, not the enqueue.
  auto tagged = make_guarded_graph();
  auto plain = make_guarded_graph();
  const std::size_t n = tagged->get_num_msyn_per_decode();
  const std::size_t obs = tagged->get_num_observables();
  std::vector<uint8_t> shot(n, 0);
  shot[n - tagged->get_num_msyn_per_round()] = 1;

  tagged->enqueue_syndrome(shot.data(), shot.size(), round_type::BULK);
  plain->enqueue_syndrome(shot.data(), shot.size());
  EXPECT_EQ(std::vector<uint8_t>(tagged->get_obs_corrections(),
                                 tagged->get_obs_corrections() + obs),
            std::vector<uint8_t>(plain->get_obs_corrections(),
                                 plain->get_obs_corrections() + obs));
}

// Same window size as make_guarded_graph(), but no guards or sliding stride.
static std::unique_ptr<decoding_task_graph> make_unguarded_graph() {
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "w";
  node.kind = "single_window";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  node.node_params.insert("window_size", std::size_t{5});
  nodes.push_back(std::move(node));
  return decoding_task_graph::from_topology(std::move(nodes));
}

TEST(DecodingTaskGraph, CombineDefersDifferentGuardsToStreaming) {
  // Equal buffer sizes cannot make these branches an aligned XOR: one commits
  // its whole window immediately, while the other commits a guarded core and
  // withholds its tail until FINAL. Their complete batch outcomes still XOR.
  auto graph = decoding_task_graph::combine(make_unguarded_graph(),
                                            make_guarded_graph());
  expect_batch_valid_streaming_conflict(std::move(graph), "start_guard");
}

TEST(DecodingTaskGraph, RoundsAfterAFinalAreDroppedNotThrown) {
  auto graph = make_guarded_graph();
  const std::size_t n = graph->get_num_msyn_per_decode();
  const std::size_t obs = graph->get_num_observables();
  std::vector<uint8_t> shot(n, 0);
  graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL);
  const std::vector<uint8_t> after(graph->get_obs_corrections(),
                                   graph->get_obs_corrections() + obs);

  // Enqueue is fire-and-forget on the wire, so an exception would surface as
  // INTERNAL_ERROR at the next get_corrections rather than as a rejection.
  // Break-it: drop the finalized check, or make it throw.
  EXPECT_NO_THROW({
    EXPECT_FALSE(
        graph->enqueue_syndrome(shot.data(), shot.size(), round_type::BULK));
    EXPECT_FALSE(
        graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  });
  EXPECT_EQ(std::vector<uint8_t>(graph->get_obs_corrections(),
                                 graph->get_obs_corrections() + obs),
            after);

  // The untagged overload is refused too: resuming through it corrupts just
  // as much.  Break-it: check the latch only in the tagged overload.
  EXPECT_FALSE(graph->enqueue_syndrome(shot.data(), shot.size()));
  const auto &still_readable = graph->streaming_result();
  EXPECT_FALSE(still_readable.operation_completed);
  EXPECT_GT(still_readable.generation, 0u);
  EXPECT_EQ(still_readable.status, task_graph_streaming_status::complete);

  // clear_corrections() consumes the accumulator and re-opens the stream, but
  // keeps overlap from the last window. One stride must therefore fire.
  graph->clear_corrections();
  const std::size_t per_round = graph->get_num_msyn_per_round();
  std::vector<uint8_t> stride(3 * per_round, 0);
  EXPECT_TRUE(graph->enqueue_syndrome(stride.data(), stride.size()))
      << "the graph stayed closed after clear_corrections()";
}

TEST(DecodingTaskGraph, GraphFlushCommitsTheTailThroughThePublicApi) {
  // The node-level tests above call window_node::flush() on a bare node, which
  // exercises none of what the graph adds: the in-flight guard, the sweep over
  // streaming nodes, and reading the result back through get_obs_corrections().
  // A graph that never called node->flush() would pass all of them.
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "w";
  node.kind = "sliding_window";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  node.node_params.insert("window_size", std::size_t{5});
  node.node_params.insert("step_size", std::size_t{3});
  node.node_params.insert("start_guard", std::size_t{1});
  node.node_params.insert("end_guard", std::size_t{1});
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));

  // A fault in the last round of the window: it lands in the end_guard, so the
  // fire withholds it and only a FINAL round commits it.
  const std::size_t n = graph->get_num_msyn_per_decode();
  const std::size_t per_round = graph->get_num_msyn_per_round();
  const std::size_t obs = graph->get_num_observables();
  std::vector<uint8_t> shot(n, 0);
  shot[n - per_round] = 1;

  // Break-it: make the graph's FINAL path skip its nodes -- every node-level
  // test still passes.
  ASSERT_TRUE(
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  const std::vector<uint8_t> after(graph->get_obs_corrections(),
                                   graph->get_obs_corrections() + obs);
  EXPECT_NE(after, std::vector<uint8_t>(obs, 0))
      << "the withheld tail never reached the graph's corrections buffer";

  // The shot is closed; a repeat is dropped rather than double-committed.
  EXPECT_FALSE(
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  const std::vector<uint8_t> again(graph->get_obs_corrections(),
                                   graph->get_obs_corrections() + obs);
  EXPECT_EQ(again, after);
}

TEST(DecodingTaskGraph, StartGuardOnlyWindowHasNoTailToFlush) {
  // {start_guard=1, end_guard=0} is the shape the accuracy test tiles with:
  // hi_col_ is set (a guard exists) but equals the result width, so the last
  // fire already committed through the end. flush must report no commit and
  // must not overwrite that fire's streaming output with zeros.
  auto node = make_guard_probe({4, 3, 1, 0}, 4);
  const uint8_t zero = 0;
  for (int i = 0; i < 4; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  const std::vector<uint8_t> last_fire =
      node->streaming_output(0)->get<logical_outcome>().bits;
  const std::vector<uint8_t> before(node->corrections(),
                                    node->corrections() + 4);

  EXPECT_FALSE(node->flush()) << "nothing was withheld";
  EXPECT_EQ(node->streaming_output(0)->get<logical_outcome>().bits, last_fire)
      << "flush clobbered the last fire's output with zeros";
  const std::vector<uint8_t> after(node->corrections(),
                                   node->corrections() + 4);
  EXPECT_EQ(after, before);
}

TEST(DecodingTaskGraph, FlushIsIdempotent) {
  // A second flush would XOR the tail back out, silently restoring the
  // pre-flush answer. Break-it: drop the flushed_ latch.
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  fire_once(*node);
  fire_once(*node);
  fire_once(*node);
  ASSERT_TRUE(node->flush());

  std::vector<uint8_t> after_first(node->corrections(),
                                   node->corrections() + 3);
  EXPECT_FALSE(node->flush());
  EXPECT_FALSE(node->flush());
  const std::vector<uint8_t> after_more(node->corrections(),
                                        node->corrections() + 3);
  EXPECT_EQ(after_more, after_first);
}

TEST(DecodingTaskGraph, EveryRoundIsCommittedExactlyOnceAcrossFlush) {
  // The point of the change, read off accumulated corrections: every round's
  // fault is present, so a round committed once reads 1, twice reads 0.
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  for (int i = 0; i < 3; ++i)
    fire_once(*node);
  node->flush();
  const std::vector<uint8_t> corr(node->corrections(), node->corrections() + 3);
  EXPECT_EQ(corr, (std::vector<uint8_t>{1, 1, 1}))
      << "a 0 means a round was committed twice (or never)";
}

TEST(DecodingTaskGraph, FlushDropsRoundsThatNeverFilledAWindow) {
  // Warn and drop rather than pad a short window into a fabricated decode.
  // No one-line Break-it: the node never fires, so nothing is ever cached.
  // What this pins is that flush does not fabricate.
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  const uint8_t zero = 0;
  ASSERT_FALSE(
      node->enqueue(&zero, 1, round_type::BULK)); // buffered, never fired
  EXPECT_FALSE(node->flush());
  const std::vector<uint8_t> corr(node->corrections(), node->corrections() + 3);
  EXPECT_EQ(corr, (std::vector<uint8_t>{0, 0, 0}));
}

TEST(DecodingTaskGraph, EnqueueAfterFlushIsRejected) {
  // flush() ends the stream. Resuming would double-commit: the rounds flush
  // committed are exactly the ones the next window's core commits, and corr_
  // XORs, so they would cancel to zero with no error. Verified: streaming
  // through a mid-stream flush diverges from flushing only at the end.
  auto node = make_guard_probe({4, 2, 1, 1}, 4);
  const uint8_t zero = 0;
  for (int i = 0; i < 4; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  ASSERT_TRUE(node->flush());

  // Break-it: drop reject_after_flush() from enqueue().
  EXPECT_THROW(node->enqueue(&zero, 1, round_type::BULK), std::logic_error);
  // Even the chunk that would have been dropped anyway: the violation is
  // resuming at all, not the size of what resumed.
  const std::vector<uint8_t> too_big(9, 0);
  EXPECT_THROW(node->enqueue(too_big.data(), too_big.size(), round_type::BULK),
               std::logic_error);

  // clear_corrections() is the documented between-shots call, and the one the
  // wire path uses -- get_corrections(reset=true) clears rather than resetting.
  // It empties the accumulator the hazard lives in, so it re-opens the node.
  // Break-it: clear flushed_ only in reset_streaming_state().
  node->clear_corrections();
  EXPECT_NO_THROW(node->enqueue(&zero, 1, round_type::BULK));

  // reset_decoder() also works, and additionally drops the buffered rounds.
  for (int i = 0; i < 3; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  ASSERT_TRUE(node->flush());
  node->reset_decoder();
  EXPECT_NO_THROW(node->enqueue(&zero, 1, round_type::BULK));
}

TEST(DecodingTaskGraph, TheWireShotLoopSurvivesAFlush) {
  // The realtime loop verbatim: stream, flush at end of shot, read, clear,
  // keep streaming -- no reset_decoder() anywhere, because
  // get_corrections(reset=true) clears rather than resetting. This must not
  // throw and must keep producing results.
  //
  // Note clear_corrections() zeroes the XOR accumulator, not the window
  // buffer, so the window keeps sliding across the boundary; making shots
  // independent needs reset_decoder(). That is pre-existing streaming
  // behaviour, not something flush changed.
  auto node = make_guard_probe({4, 2, 1, 1}, 4);
  const uint8_t zero = 0;
  for (int i = 0; i < 4; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  ASSERT_TRUE(node->flush());
  node->clear_corrections();

  bool fired_again = false;
  for (int i = 0; i < 4; ++i)
    fired_again |= node->enqueue(&zero, 1, round_type::BULK);
  EXPECT_TRUE(fired_again) << "the node stopped decoding after a flush";
  EXPECT_TRUE(node->flush()) << "the next shot has its own tail to commit";
}

TEST(DecodingTaskGraph, CorrectionsAreUnchangedByARejectedResume) {
  // The rejection must leave the flushed result intact -- a caller that
  // catches the exception still has a correct shot to read.
  auto node = make_guard_probe({4, 2, 1, 1}, 4);
  const uint8_t zero = 0;
  for (int i = 0; i < 4; ++i)
    node->enqueue(&zero, 1, round_type::BULK);
  ASSERT_TRUE(node->flush());
  const std::vector<uint8_t> after_flush(node->corrections(),
                                         node->corrections() + 4);

  EXPECT_THROW(node->enqueue(&zero, 1, round_type::BULK), std::logic_error);
  const std::vector<uint8_t> now(node->corrections(), node->corrections() + 4);
  EXPECT_EQ(now, after_flush);
}

TEST(DecodingTaskGraph, MidStreamFlushCannotCorruptBecauseItIsTerminal) {
  // The property the rejection exists for, stated as an equivalence: a node
  // flushed only at the end and a node someone tried to flush mid-stream must
  // not silently disagree. Before flush() was terminal these produced 1001 and
  // 1000 -- the tail XORed itself out.
  auto at_end = make_guard_probe({4, 2, 1, 1}, 4);
  auto mid = make_guard_probe({4, 2, 1, 1}, 4);
  const uint8_t zero = 0;
  for (int i = 0; i < 6; ++i)
    at_end->enqueue(&zero, 1, round_type::BULK);
  at_end->flush();

  for (int i = 0; i < 4; ++i)
    mid->enqueue(&zero, 1, round_type::BULK);
  mid->flush();
  EXPECT_THROW(mid->enqueue(&zero, 1, round_type::BULK), std::logic_error)
      << "a mid-stream flush must be refused, not silently corrupt";
}

TEST(DecodingTaskGraph, ResetDropsAnUnflushedTail) {
  // A shot that ends unflushed must not leak its tail into the next.
  // Break-it needs both gone: last_result_valid_ = false in
  // reset_streaming_state() (called by reset_decoder()), and the
  // fires_since_reset_ == 0 return in flush().
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  for (int i = 0; i < 3; ++i)
    fire_once(*node); // fires once, withholds a tail, never flushed
  node->reset_decoder();

  EXPECT_FALSE(node->flush());
  const std::vector<uint8_t> corr(node->corrections(), node->corrections() + 3);
  EXPECT_EQ(corr, (std::vector<uint8_t>{0, 0, 0}));
}

TEST(DecodingTaskGraph, APartialWindowSurvivesFlushUntilReset) {
  // FINAL commits the withheld tail; it does not drain the node.  So an
  // incomplete window is reported but kept, and clear_corrections() -- which
  // zeroes the accumulator, not the buffer -- leaves it there.  A pre-decoder
  // empties on every fire, so leftover bytes are never legitimate overlap:
  // the next shot's first rounds would complete THIS shot's window.
  //
  // reset_decoder() is the only reopen that makes shots independent.
  auto node = make_chain_predecoder({0, 0, 0, 0, 0});
  const std::size_t n = node->num_msyn_per_decode();
  std::vector<uint8_t> partial(n - 1, 0);
  ASSERT_FALSE(node->enqueue(partial.data(), partial.size(), round_type::BULK));
  EXPECT_FALSE(node->flush());
  node->clear_corrections();

  // One byte completes the carried-over window: the buffer survived.
  const uint8_t one = 0;
  EXPECT_TRUE(node->enqueue(&one, 1, round_type::BULK))
      << "the partial window did not survive, so the warning is wrong";

  // After a reset it does not: the same byte only starts a new window.
  node->reset_decoder();
  ASSERT_FALSE(node->enqueue(partial.data(), partial.size(), round_type::BULK));
  node->flush();
  node->reset_decoder();
  // Break-it: have reset_decoder() keep bytes_in_.
  EXPECT_FALSE(node->enqueue(&one, 1, round_type::BULK))
      << "a byte completed a window that reset_decoder() should have dropped";
}

TEST(DecodingTaskGraph, PredecoderFlushCommitsNothingAndDoesNotClose) {
  // A pre-decoder empties its buffer on every fire and carries no guard, so it
  // never withholds a tail.  It overrides flush() anyway so "nothing withheld"
  // is a decision rather than an inherited default, and so a partial window at
  // end of stream is named rather than vanishing.
  auto node = make_chain_predecoder({0, 0, 0, 0, 0});
  const std::size_t n = node->num_msyn_per_decode();
  std::vector<uint8_t> partial(n - 1, 0);
  ASSERT_FALSE(node->enqueue(partial.data(), partial.size(), round_type::BULK));

  // Break-it: drop the override and inherit
  // streaming_task_graph_node::flush(). This still returns false, so the
  // assertion below is about the warning path -- what the override buys is
  // that a dropped partial window is reported.
  EXPECT_FALSE(node->flush());

  // Nothing was committed, so the node stays usable.
  EXPECT_NO_THROW((void)node->enqueue(partial.data(), 1, round_type::BULK));
}

TEST(DecodingTaskGraph, AFlushThatCommitsNothingDoesNotCloseTheStream) {
  // The latch guards a *committed* tail against being XOR-cancelled by a later
  // overlapping fire. A flush that commits nothing has no such tail, so
  // latching would reject the next enqueue for a hazard that does not exist --
  // and every unguarded window commits nothing, which is every in-tree config.
  // Break-it: latch flushed_ at the top of flush() instead of after the
  // early returns.
  const uint8_t zero = 0;
  auto unguarded = make_guard_probe({3, 1, 0, 0}, 3);
  for (int i = 0; i < 3; ++i)
    unguarded->enqueue(&zero, 1, round_type::BULK);
  ASSERT_FALSE(unguarded->flush()) << "nothing was withheld";
  EXPECT_NO_THROW(unguarded->enqueue(&zero, 1, round_type::BULK));

  // Same for a guarded window that never filled one: nothing decoded, so
  // nothing committed, so nothing to protect.
  auto never_fired = make_guard_probe({3, 1, 1, 1}, 3);
  ASSERT_FALSE(never_fired->enqueue(&zero, 1, round_type::BULK));
  ASSERT_FALSE(never_fired->flush());
  EXPECT_NO_THROW(never_fired->enqueue(&zero, 1, round_type::BULK));
}

TEST(DecodingTaskGraph, UnguardedWindowHasNothingToFlush) {
  // Nothing withheld without guards, so flush is a no-op, not a second commit.
  // Break-it: snapshot unconditionally in fire() (drop its hi_col_ test).
  auto node = make_guard_probe({3, 1, 0, 0}, 3);
  for (int i = 0; i < 3; ++i)
    fire_once(*node);
  EXPECT_FALSE(node->flush());
  const std::vector<uint8_t> corr(node->corrections(), node->corrections() + 3);
  EXPECT_EQ(corr, (std::vector<uint8_t>{1, 1, 1}));
}

TEST(DecodingTaskGraph, GuardedOneShotExecutionCommitsTheWholeShot) {
  // A one-shot execution has only one window, so neither guard masks and the
  // whole decode lands. Streaming is where a guard defers instead;
  // GuardedCorrectionsAccumulateOneRoundPerFire covers that.
  //
  // Break-it: pass hi_col_ instead of max() in execute_indexed or run_async.
  // Column 2 is the end guard, dropped on one-shot execution with no way back.
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  measurement_results measurements;
  measurements.bits.assign(3, 0);
  std::vector<edge_value> inputs{measurements};
  std::vector<edge_value> outputs(1);

  node->execute_indexed(inputs, outputs);
  EXPECT_EQ(outputs[0].get<logical_outcome>().bits,
            (std::vector<uint8_t>{1, 1, 1}));
  EXPECT_EQ(node->run_async(measurements).get().bits,
            (std::vector<uint8_t>{1, 1, 1}));
}

TEST(DecodingTaskGraph, OneShotExecutionIgnoresGuards) {
  // An equivalence rather than a literal, so it encodes the rule: a guard
  // splits a stream, the one-shot path does not split, so guarding must not
  // change what one-shot execution commits.
  //
  // Break-it: restore either mask in execute_indexed.
  auto guarded = make_guard_probe({3, 1, 1, 1}, 3);
  auto unguarded = make_guard_probe({3, 3, 0, 0}, 3);
  measurement_results measurements;
  measurements.bits.assign(3, 0);
  std::vector<edge_value> inputs{measurements};
  std::vector<edge_value> guarded_out(1), unguarded_out(1);

  guarded->execute_indexed(inputs, guarded_out);
  unguarded->execute_indexed(inputs, unguarded_out);
  EXPECT_EQ(guarded_out[0].get<logical_outcome>().bits,
            unguarded_out[0].get<logical_outcome>().bits);
  // Not vacuous: the shared value is the full commit, not two empty vectors.
  EXPECT_EQ(unguarded_out[0].get<logical_outcome>().bits,
            (std::vector<uint8_t>{1, 1, 1}));
}

TEST(DecodingTaskGraph, GuardedCorrectionsAccumulateOneRoundPerFire) {
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  // Three rounds fill the window and fire it; the fourth slides it and fires
  // again.
  for (int i = 0; i < 4; ++i)
    fire_once(*node);

  // The first fire committed the window's rounds 0 and 1; the second committed
  // its round 1 again, which cancels, leaving the parity per column.
  const auto *corr = node->corrections();
  EXPECT_EQ(corr[0], 1u);
  EXPECT_EQ(corr[1], 0u);
  EXPECT_EQ(corr[2], 0u);
}

TEST(DecodingTaskGraph, ResetRestoresTheLeadingGuardCommit) {
  auto node = make_guard_probe({3, 1, 1, 1}, 3);
  for (int i = 0; i < 4; ++i)
    fire_once(*node);
  ASSERT_EQ(fire_once(*node), (std::vector<uint8_t>{0, 1, 0}));

  // A reset starts a new shot, whose first window again has nothing before it.
  node->reset_decoder();
  fire_once(*node);
  fire_once(*node);
  EXPECT_EQ(fire_once(*node), (std::vector<uint8_t>{1, 1, 0}));
}

TEST(DecodingTaskGraph, GuardsRequireOverlappingWindows) {
  // stride == size means no other window covers the guarded rounds.
  EXPECT_THROW(make_guard_probe({3, 3, 0, 1}, 3), std::invalid_argument);
  EXPECT_THROW(make_guard_probe({3, 1, 2, 1}, 3), std::invalid_argument);
}

TEST(DecodingTaskGraph, GuardedWindowsMustTileTheStreamExactlyOnce) {
  EXPECT_THROW(make_guarded_identity_probe({4, 2, 0, 1}),
               std::invalid_argument);

  auto tiled = make_guarded_identity_probe({4, 3, 0, 1});
  const std::vector<uint8_t> rounds{0, 0, 1, 0, 0, 0, 0};
  for (const uint8_t round : rounds)
    tiled->enqueue(&round, 1, round_type::BULK);
  EXPECT_TRUE(tiled->flush());
  EXPECT_EQ(tiled->corrections()[0], 1u);
}

TEST(DecodingTaskGraph, WindowStrideMustNotExceedSize) {
  EXPECT_THROW(make_guard_probe({3, 4, 0, 0}, 3), std::invalid_argument);
}

TEST(DecodingTaskGraph, GuardsRequirePerRoundFaultOffsets) {
  // Guards are given in rounds and committed in columns, so a window with no
  // per-round column layout cannot honor them.
  auto identity = std::vector<std::vector<uint32_t>>{{0}, {1}, {2}};
  auto h = sparse_binary_matrix::from_nested_csr(3, 3, identity);
  auto d = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(3, 3, identity));
  auto o = std::make_shared<sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(3, 3, identity));
  auto dec = std::make_unique<all_faults_decoder>(h, 3);

  EXPECT_THROW(window_node(d, o, std::move(dec), 1, {3, 1, 0, 1},
                           /*fault_offsets=*/{}),
               std::invalid_argument);
}

// Same one-fault-per-round toy model as make_guard_probe (D=H=O=identity), but
// driven by real Bernoulli-sampled fault patterns through a genuine decoder
// (single_error_lut) rather than a single fixed all-ones probe. Both windows
// run the identical decoder over the identical error model and differ only in
// how they're windowed, so their logical error rates should agree -- a
// commit-range or seam bug would move rounds between windows or double-commit
// an overlap, which shows up as a guarded LER that diverges from the
// monolithic one, not as a coincidentally-matching self-consistent result
// (the risk with the existing fixed all-ones/all-zeros guard tests above).
TEST(DecodingTaskGraph, GuardedSlidingWindowMatchesMonolithicDecodeAccuracy) {
  constexpr uint32_t num_rounds = 10;
  constexpr std::size_t num_shots = 5000;
  // size=4, start_guard=1, end_guard=0, stride=size-start_guard-end_guard=3:
  // the first fire keeps its whole leading guard (nothing precedes it) and
  // commits 4 rounds; every fire after that commits exactly `stride` = 3
  // rounds. 4 + 3 + 3 == num_rounds (three fires), so they exactly tile all
  // 10 rounds with no gap and no double-commit -- the property a mismatched
  // stride/guard/window_size would break.
  constexpr window_spec guarded_ws{/*size=*/4, /*stride=*/3,
                                   /*start_guard=*/1, /*end_guard=*/0};

  // One persistent logical bit, the XOR of every round's fault -- like a
  // repetition-code logical qubit tracked across a whole shot -- rather than
  // one observable per round: a window_node's corrections accumulate by
  // XORing each fire's committed contribution into the *same* observable
  // slots, so the slot count has to be the thing that persists across fires
  // (few, fixed), not the round count (see make_guard_probe above, whose O is
  // per-round only because its tests read one fire's output at a time, never
  // the fully-accumulated result).
  //
  // A window's D/H/O are local to its own byte buffer, not the whole
  // experiment: the mono window's identity H is num_rounds x num_rounds, but
  // the guarded window's is only guarded_ws.size x guarded_ws.size, and each
  // fire's O sums its own local committed columns into that one slot.
  auto identity_of = [](uint32_t n) {
    std::vector<std::vector<uint32_t>> rows(n);
    for (uint32_t i = 0; i < n; ++i)
      rows[i].push_back(i);
    return rows;
  };
  auto xor_all_of = [](uint32_t n) {
    std::vector<uint32_t> cols(n);
    for (uint32_t i = 0; i < n; ++i)
      cols[i] = i;
    return std::vector<std::vector<uint32_t>>{std::move(cols)};
  };
  auto h_mono = sparse_binary_matrix::from_nested_csr(num_rounds, num_rounds,
                                                      identity_of(num_rounds));
  auto d_mono = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(num_rounds, num_rounds,
                                            identity_of(num_rounds)));
  auto o_mono = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, num_rounds,
                                            xor_all_of(num_rounds)));
  auto h_guard = sparse_binary_matrix::from_nested_csr(
      guarded_ws.size, guarded_ws.size, identity_of(guarded_ws.size));
  auto d_guard = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(guarded_ws.size, guarded_ws.size,
                                            identity_of(guarded_ws.size)));
  auto o_guard = std::make_shared<const sparse_binary_matrix>(
      sparse_binary_matrix::from_nested_csr(1, guarded_ws.size,
                                            xor_all_of(guarded_ws.size)));

  std::vector<std::size_t> fault_offsets(guarded_ws.size + 1);
  for (uint32_t i = 0; i <= guarded_ws.size; ++i)
    fault_offsets[i] = i;

  auto mono = std::make_unique<window_node>(
      d_mono, o_mono, get_decoder("single_error_lut", h_mono, {}), /*spr=*/1,
      window_spec{num_rounds, num_rounds, 0, 0});
  auto guarded = std::make_unique<window_node>(
      d_guard, o_guard, get_decoder("single_error_lut", h_guard, {}),
      /*spr=*/1, guarded_ws, fault_offsets);

  std::vector<uint8_t> h_flat(static_cast<std::size_t>(num_rounds) * num_rounds,
                              0);
  for (uint32_t i = 0; i < num_rounds; ++i)
    h_flat[i * num_rounds + i] = 1;
  cudaqx::tensor<uint8_t> h_tensor({num_rounds, num_rounds});
  h_tensor.copy(h_flat.data(), h_tensor.shape());
  // Low enough that most windows see at most one fault (single_error_lut's
  // solvable case), so both paths spend most shots decoding correctly and the
  // comparison is not dominated by decoder-inherent ambiguity.
  std::vector<double> error_rates(num_rounds, 0.02);
  auto [checks, errors] = dem_sampler::cpu::sample_dem(
      h_tensor, num_shots, error_rates, /*seed=*/42);
  (void)errors;

  std::size_t mono_mismatches = 0, guarded_mismatches = 0;
  // On a weight-0/1 shot, single_error_lut is exact (H is identity, so a lone
  // fault has an unambiguous syndrome): mono always decodes it exactly, and
  // guarded must too if its tiling has no gap or double-commit. This is what
  // catches a one-round gap or off-by-one guard boundary -- averaged over the
  // full weight distribution such a bug is only a ~p shift in guarded_ler
  // (p = 0.02 here), inside the tolerance below, but a strict per-shot check
  // on isolated faults is not (verified with a deliberately introduced
  // one-round gap: this check fails while the LER-band comparison passes).
  std::size_t low_weight_shots = 0, low_weight_guarded_mismatches = 0;
  for (std::size_t shot = 0; shot < num_shots; ++shot) {
    std::vector<uint8_t> true_bits(num_rounds);
    uint8_t true_parity = 0;
    unsigned weight = 0;
    for (uint32_t r = 0; r < num_rounds; ++r) {
      true_bits[r] = checks.at({shot, r});
      true_parity ^= true_bits[r];
      weight += true_bits[r];
    }
    const bool low_weight = weight <= 1;

    ASSERT_TRUE(
        mono->enqueue(true_bits.data(), true_bits.size(), round_type::BULK));
    const uint8_t mono_bit = mono->corrections()[0];
    if (mono_bit != true_parity)
      ++mono_mismatches;
    ASSERT_TRUE(!low_weight || mono_bit == true_parity)
        << "shot " << shot << " weight " << weight
        << ": single_error_lut should be exact on <=1 fault";
    mono->reset_decoder();

    for (uint32_t r = 0; r < num_rounds; ++r)
      guarded->enqueue(&true_bits[r], 1, round_type::BULK);
    const uint8_t guarded_bit = guarded->corrections()[0];
    if (guarded_bit != true_parity)
      ++guarded_mismatches;
    if (low_weight) {
      ++low_weight_shots;
      if (guarded_bit != true_parity)
        ++low_weight_guarded_mismatches;
    }
    guarded->reset_decoder();
  }

  ASSERT_GT(low_weight_shots, 0u);
  EXPECT_EQ(low_weight_guarded_mismatches, 0u)
      << low_weight_guarded_mismatches << "/" << low_weight_shots
      << " weight-0/1 shots mismatched -- some round is not committed "
         "exactly once";

  const double mono_ler = static_cast<double>(mono_mismatches) / num_shots;
  const double guarded_ler =
      static_cast<double>(guarded_mismatches) / num_shots;

  // Aggregate check across the whole weight distribution, complementing the
  // strict weight-0/1 check above: both paths share a decoder and error
  // model, so a large gap here is windowing/guard/seam logic, not decoder
  // quality. Tolerance is generous to stay non-flaky while still catching a
  // broken commit range (verified: a double-commit drives guarded_ler far
  // outside it).
  EXPECT_LT(mono_ler, 0.2) << "sanity: decode should not be near chance";
  EXPECT_NEAR(guarded_ler, mono_ler, 0.03)
      << "guarded=" << guarded_ler << " monolithic=" << mono_ler;
}

// Same guard config as GuardsWithholdBoundaryRoundsFromTheCommit, but driven
// for thousands of rounds in one continuous shot instead of a handful: after
// the first couple of fires, every subsequent fire commits the identical
// steady-state pattern {0,1,0} forever. A regression that corrupts state on a
// long run -- an index that overflows, a memmove miscalculation that only
// shows up once the buffer has slid many times -- would drift away from this
// pattern rather than hold it for 20000 rounds.
TEST(DecodingTaskGraph, GuardedWindowStaysAtSteadyStateOverThousandsOfRounds) {
  auto node = make_guard_probe(
      {/*size=*/3, /*stride=*/1, /*start_guard=*/1, /*end_guard=*/1}, 3);

  constexpr int num_rounds = 20000;
  for (int i = 0; i < num_rounds; ++i) {
    const auto committed = fire_once(*node);
    if (i >= 3) // past the two-fire startup transient (see the test above)
      ASSERT_EQ(committed, (std::vector<uint8_t>{0, 1, 0})) << "round " << i;
  }
}

TEST(DecodingTaskGraph, FlatGraphAcceptsNoObservableRows) {
  const auto inputs = make_rep3_inputs();
  const cudaqx::heterogeneous_map params;
  auto dec = get_decoder("single_error_lut",
                         inputs.closed.detector_error_matrix, params);
  decoding_task_graph graph(std::move(dec), inputs.d, {});

  EXPECT_EQ(graph.get_num_observables(), 0u);
}

// ---------------------------------------------------------------------------
// from_topology
// ---------------------------------------------------------------------------

// The node config from_spec builds for a streaming spec, spelled out.
static task_graph_node_config rep5_streaming_node(std::string id) {
  task_graph_node_config node;
  node.id = std::move(id);
  node.kind = "sliding_window";
  node.dem = rep5_streaming_spec();
  node.decoder_type = "single_error_lut";
  node.node_params.insert("step_size", std::size_t{1});
  return node;
}

// A five-round sliding node with an explicit firing and commit schedule.
static task_graph_node_config scheduled_rep5_node(std::string id,
                                                  std::size_t step_size,
                                                  std::size_t start_guard = 0,
                                                  std::size_t end_guard = 0) {
  auto node = rep5_streaming_node(std::move(id));
  node.node_params.insert("window_size", std::size_t{5});
  node.node_params.insert("step_size", step_size);
  if (start_guard > 0)
    node.node_params.insert("start_guard", start_guard);
  if (end_guard > 0)
    node.node_params.insert("end_guard", end_guard);
  return node;
}

TEST(DecodingTaskGraph, FoldedWindowsDeferUnequalStridesToStreaming) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("every_round", 1));
  nodes.push_back(scheduled_rep5_node("every_other_round", 2));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "every_round", "input"},
       {"measurements", "output", "every_other_round", "input"},
       {"every_round", "output", "combine", "a"},
       {"every_other_round", "output", "combine", "b"},
       {"combine", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "step_size");
}

TEST(DecodingTaskGraph, FoldedWindowsAcceptTheSameGuardedSchedule) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("left", 2, 1, 2));
  nodes.push_back(scheduled_rep5_node("right", 2, 1, 2));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "combine", "a"},
                         {"right", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  expect_streaming_works(std::move(graph));
}

TEST(DecodingTaskGraph, FoldedWindowsDeferDifferentCommitRegionsToStreaming) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("left", 2, 1, 2));
  nodes.push_back(scheduled_rep5_node("right", 2, 2, 1));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "combine", "a"},
                         {"right", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "start_guard");
}

TEST(DecodingTaskGraph, SeparateOutputsMayUseUnequalStrides) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("every_round", 1));
  nodes.push_back(scheduled_rep5_node("every_other_round", 2));

  EXPECT_NO_THROW((void)decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "every_round", "input"},
       {"measurements", "output", "every_other_round", "input"},
       {"every_round", "output", "output", "fast"},
       {"every_other_round", "output", "output", "slow"}}));
}

static task_graph_node_config
scheduled_detector_source(std::string id, std::size_t step_size,
                          std::size_t start_guard = 0,
                          std::size_t end_guard = 0) {
  task_graph_node_config node;
  node.id = std::move(id);
  node.kind = "test_detector_source";
  node.dem = rep5_streaming_spec();
  node.node_params.insert("window_size", std::size_t{5});
  node.node_params.insert("step_size", step_size);
  node.node_params.insert("start_guard", start_guard);
  node.node_params.insert("end_guard", end_guard);
  return node;
}

static task_graph_node_config scheduled_relay(std::string id,
                                              std::size_t schedule_factor = 1,
                                              bool declare_schedule = true) {
  task_graph_node_config node;
  node.id = std::move(id);
  node.kind = "test_relay";
  node.dem = rep5_streaming_spec();
  node.node_params.insert("window_size", std::size_t{5});
  node.node_params.insert("schedule_factor", schedule_factor);
  node.node_params.insert("declare_schedule", declare_schedule);
  return node;
}

TEST(DecodingTaskGraph, CorrectionRelaysDeferUnequalStridesToStreaming) {
  ASSERT_TRUE(relay_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("every_round", 1));
  nodes.push_back(scheduled_rep5_node("every_other_round", 2));
  nodes.push_back(scheduled_relay("fast_relay"));
  nodes.push_back(scheduled_relay("slow_relay"));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "every_round", "input"},
       {"measurements", "output", "every_other_round", "input"},
       {"every_round", "output", "fast_relay", "input"},
       {"every_other_round", "output", "slow_relay", "input"},
       {"fast_relay", "output", "combine", "a"},
       {"slow_relay", "output", "combine", "b"},
       {"combine", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "slow_relay.output");
}

TEST(DecodingTaskGraph, EdgeFedFoldDefersUnequalUpstreamStridesToStreaming) {
  ASSERT_TRUE(detector_source_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_detector_source("fast_producer", 1));
  nodes.push_back(scheduled_detector_source("slow_producer", 2));
  nodes.push_back(scheduled_rep5_node("fast_consumer", 5));
  nodes.push_back(scheduled_rep5_node("slow_consumer", 5));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "fast_producer", "input"},
                         {"measurements", "output", "slow_producer", "input"},
                         {"fast_producer", "output", "fast_consumer", "input"},
                         {"slow_producer", "output", "slow_consumer", "input"},
                         {"fast_consumer", "output", "combine", "a"},
                         {"slow_consumer", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph),
                                        "slow_consumer.output");
}

TEST(DecodingTaskGraph, EdgeFedFoldAcceptsAlignedUpstreamSchedules) {
  ASSERT_TRUE(detector_source_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_detector_source("left_producer", 2));
  nodes.push_back(scheduled_detector_source("right_producer", 2));
  nodes.push_back(scheduled_rep5_node("left_consumer", 5));
  nodes.push_back(scheduled_rep5_node("right_consumer", 5));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "left_producer", "input"},
       {"measurements", "output", "right_producer", "input"},
       {"left_producer", "output", "left_consumer", "input"},
       {"right_producer", "output", "right_consumer", "input"},
       {"left_consumer", "output", "combine", "a"},
       {"right_consumer", "output", "combine", "b"},
       {"combine", "output", "output", "input"}});
  expect_streaming_works(std::move(graph));
}

TEST(DecodingTaskGraph, EdgeFedWindowUsesItsOwnCommitGuards) {
  ASSERT_TRUE(detector_source_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_detector_source("guarded_producer", 2, 1, 2));
  nodes.push_back(scheduled_rep5_node("guarded_branch", 2, 1, 2));
  nodes.push_back(scheduled_rep5_node("edge_consumer", 5));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "guarded_producer", "input"},
       {"measurements", "output", "guarded_branch", "input"},
       {"guarded_producer", "output", "edge_consumer", "input"},
       {"edge_consumer", "output", "combine", "a"},
       {"guarded_branch", "output", "combine", "b"},
       {"combine", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "start_guard");
}

TEST(DecodingTaskGraph, EdgeFedSeparateOutputsMayUseUnequalUpstreamStrides) {
  ASSERT_TRUE(detector_source_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_detector_source("fast_producer", 1));
  nodes.push_back(scheduled_detector_source("slow_producer", 2));
  nodes.push_back(scheduled_rep5_node("fast_consumer", 5));
  nodes.push_back(scheduled_rep5_node("slow_consumer", 5));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "fast_producer", "input"},
                         {"measurements", "output", "slow_producer", "input"},
                         {"fast_producer", "output", "fast_consumer", "input"},
                         {"slow_producer", "output", "slow_consumer", "input"},
                         {"fast_consumer", "output", "output", "fast"},
                         {"slow_consumer", "output", "output", "slow"}});
  expect_streaming_works(std::move(graph));
}

TEST(DecodingTaskGraph, EdgeTransformCanAlignDifferentProducerSchedules) {
  ASSERT_TRUE(relay_registered);
  last_relay = nullptr;
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("fast", 1));
  nodes.push_back(scheduled_rep5_node("slow", 2));
  nodes.push_back(scheduled_relay("decimate_fast", 2));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "fast", "input"},
                         {"measurements", "output", "slow", "input"},
                         {"fast", "output", "decimate_fast", "input"},
                         {"decimate_fast", "output", "combine", "a"},
                         {"slow", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  ASSERT_NE(last_relay, nullptr);
  ASSERT_TRUE(graph->supports_streaming())
      << graph->streaming_incompatibility_reason();

  std::vector<uint8_t> first(graph->get_num_msyn_per_decode(), 0);
  std::vector<uint8_t> round(graph->get_num_msyn_per_round(), 0);
  EXPECT_TRUE(graph->enqueue_syndrome(first.data(), first.size()));
  EXPECT_EQ(last_relay->received(), 1u);
  EXPECT_EQ(last_relay->emitted(), 1u);
  EXPECT_EQ(graph->streaming_result().generation, 1u);
  EXPECT_NO_THROW((void)graph->enqueue_syndrome(round.data(), round.size()));
  EXPECT_EQ(last_relay->received(), 2u);
  EXPECT_EQ(last_relay->emitted(), 1u);
  // The fast producer advanced, but its relay and the slow contributor did
  // not. The one output remains behind their shared publication barrier.
  EXPECT_EQ(graph->streaming_result().generation, 1u);
  EXPECT_NO_THROW((void)graph->enqueue_syndrome(round.data(), round.size()));
  EXPECT_EQ(last_relay->received(), 3u);
  EXPECT_EQ(last_relay->emitted(), 2u);
  EXPECT_EQ(graph->streaming_result().generation, 2u);
}

TEST(DecodingTaskGraph, EdgeTransformMisalignmentIsStreamingOnly) {
  ASSERT_TRUE(relay_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("left", 1));
  nodes.push_back(scheduled_rep5_node("right", 1));
  nodes.push_back(scheduled_relay("decimate_left", 2));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "decimate_left", "input"},
                         {"decimate_left", "output", "combine", "a"},
                         {"right", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "step_size");
}

TEST(DecodingTaskGraph, UnknownEffectiveScheduleIsStreamingOnly) {
  ASSERT_TRUE(relay_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(scheduled_rep5_node("source", 1));
  nodes.push_back(scheduled_relay("unknown", 1, false));
  nodes.push_back(scheduled_relay("downstream"));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"source", "output", "unknown", "input"},
                         {"unknown", "output", "downstream", "input"},
                         {"downstream", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "unknown.output");
}

TEST(DecodingTaskGraph, EdgeFedDetectorShapeMismatchRejectsEveryMode) {
  ASSERT_TRUE(detector_source_registered);
  auto source = scheduled_detector_source("three_round_source", 3);
  source.node_params.insert("window_size", std::size_t{3});
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(source));
  nodes.push_back(scheduled_rep5_node("five_round_consumer", 5));

  try {
    (void)decoding_task_graph::from_topology(
        std::move(nodes),
        {{"measurements", "output", "three_round_source", "input"},
         {"three_round_source", "output", "five_round_consumer", "input"},
         {"five_round_consumer", "output", "output", "input"}});
    FAIL() << "a detector-frame shape mismatch built as a batch graph";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("12 values"), std::string::npos) << msg;
    EXPECT_NE(msg.find("D produces 20"), std::string::npos) << msg;
  }
}

TEST(DecodingTaskGraph, CombineDefersUnequalStridesToStreaming) {
  std::vector<task_graph_node_config> left_nodes;
  std::vector<task_graph_node_config> right_nodes;
  left_nodes.push_back(scheduled_rep5_node("left", 1));
  right_nodes.push_back(scheduled_rep5_node("right", 2));
  auto left = decoding_task_graph::from_topology(std::move(left_nodes));
  auto right = decoding_task_graph::from_topology(std::move(right_nodes));

  auto graph = decoding_task_graph::combine(std::move(left), std::move(right));
  expect_batch_valid_streaming_conflict(std::move(graph), "step_size");
}

TEST(DecodingTaskGraph, FromTopologyMatchesFromSpecForOneNode) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node());
  auto topology = decoding_task_graph::from_topology(std::move(nodes));
  auto spec_built =
      decoding_task_graph::from_spec(rep5_streaming_spec(), "single_error_lut");

  EXPECT_EQ(topology->get_num_msyn_per_decode(),
            spec_built->get_num_msyn_per_decode());
  EXPECT_EQ(topology->get_num_msyn_per_round(),
            spec_built->get_num_msyn_per_round());
  EXPECT_EQ(topology->get_num_observables(), spec_built->get_num_observables());

  measurement_results m;
  m.bits.assign(topology->get_num_msyn_per_decode(), 0);
  m.bits[0] = 1;
  EXPECT_EQ(topology->run(m).bits, spec_built->run(m).bits);
}

TEST(DecodingTaskGraph, FromTopologyMatchesCombine) {
  // combine()'s topology, written out: two windows over the same DEM feeding
  // an xor node.
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config left;
  left.id = "left";
  left.kind = "single_window";
  left.dem = rep5_spec(5);
  left.decoder_type = "single_error_lut";
  auto right = task_graph_node_config{};
  right.id = "right";
  right.kind = "single_window";
  right.dem = rep5_spec(5);
  right.decoder_type = "single_error_lut";
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(left));
  nodes.push_back(std::move(right));
  nodes.push_back(std::move(combiner));

  const std::vector<task_graph_edge> edges{
      {"measurements", "output", "left", "input"},
      {"measurements", "output", "right", "input"},
      {"left", "output", "combine", "a"},
      {"right", "output", "combine", "b"},
      {"combine", "output", "output", "input"}};

  auto topology = decoding_task_graph::from_topology(std::move(nodes), edges);
  auto combined = decoding_task_graph::combine(
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut"),
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut"));

  EXPECT_EQ(topology->get_num_msyn_per_decode(),
            combined->get_num_msyn_per_decode());

  measurement_results m;
  m.bits.assign(topology->get_num_msyn_per_decode(), 0);
  m.bits[1] = 1;
  EXPECT_EQ(topology->run(m).bits, combined->run(m).bits);
}

TEST(DecodingTaskGraph, FromTopologyReservesSyntheticEndpointNames) {
  for (const char *id : {"measurements", "dets", "output"}) {
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(rep5_streaming_node(id));
    EXPECT_THROW(decoding_task_graph::from_topology(std::move(nodes)),
                 std::invalid_argument);
  }
}

TEST(DecodingTaskGraph, FromTopologyAcceptsLegacyDetsSource) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("node"));
  EXPECT_NO_THROW((void)decoding_task_graph::from_topology(
      std::move(nodes), {{"dets", "output", "node", "input"},
                         {"node", "output", "output", "input"}}));
}

TEST(DecodingTaskGraph, FromTopologyRejectsRawGraphOutput) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));

  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "a", "input"},
                             {"measurements", "output", "output", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologyRequiresEveryInput) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));

  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "a", "input"},
                             {"a", "output", "combine", "a"},
                             {"combine", "output", "output", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologyRejectsMultipleInputProducers) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));
  nodes.push_back(rep5_streaming_node("b"));
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));

  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "a", "input"},
                             {"measurements", "output", "b", "input"},
                             {"a", "output", "combine", "a"},
                             {"b", "output", "combine", "a"},
                             {"a", "output", "combine", "b"},
                             {"combine", "output", "output", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologyRequiresGraphOutput) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));

  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "a", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologySupportsMultipleGraphOutputs) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));
  nodes.push_back(rep5_streaming_node("b"));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "a", "input"},
                         {"measurements", "output", "b", "input"},
                         {"a", "output", "output", "left"},
                         {"b", "output", "output", "right"}});
  EXPECT_EQ(graph->output_names(), (std::vector<std::string>{"left", "right"}));
}

TEST(DecodingTaskGraph, FromTopologyRequiresLogicalGraphOutput) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));

  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "pre", "input"},
                             {"pre", "residual", "output", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologyRejectsEdgeTypeMismatch) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(residual_node("pre", "predecoder", "single_error_lut"));
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));

  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "pre", "input"},
                             {"pre", "residual", "combine", "a"},
                             {"pre", "residual", "combine", "b"},
                             {"combine", "output", "output", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, StreamingRepeatedSourceThroughXorCancels) {
  const auto model = rep5_model::build();
  std::vector<std::size_t> fault;
  for (std::size_t fault_idx = 0; fault_idx < model.n_mech; ++fault_idx)
    if (model.truth_of({fault_idx}) == 1) {
      fault = {fault_idx};
      break;
    }
  ASSERT_FALSE(fault.empty());
  const auto measurements = model.measurements_for(model.syndrome_of(fault));

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));

  auto repeated = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "a", "input"},
                         {"a", "output", "combine", "a"},
                         {"a", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  std::vector<task_graph_node_config> source_nodes;
  source_nodes.push_back(rep5_streaming_node("source"));
  auto source = decoding_task_graph::from_topology(std::move(source_nodes));

  measurement_results shot;
  shot.bits = measurements;
  ASSERT_EQ(source->run(shot).bits[0] & 1u, 1u);
  EXPECT_EQ(repeated->run(shot).bits[0] & 1u, 0u);
  EXPECT_EQ(stream_shot(*repeated, measurements), 0u);
}

// Same 4-detector / 5-fault bit-flip chain as the Python
// test_multi_node_task_graph_serves_a_shot_via_configure_decoders. Weight-2
// syndrome {1,0,0,1} is uniquely faults {0,4} among weight <= 2 (the chain's
// all-ones kernel would pair it with a weight-3 complement). Batch run() is
// the path that actually calls xor_node::execute_indexed; the Python kernel
// uses enqueue/get_corrections, which XOR-folds streaming window nodes and
// never enters the xor node (it is one-shot-only).
TEST(DecodingTaskGraph, XorCombineRunDisagreesOnUniqueTwoFaultSyndrome) {
  auto h = sparse_binary_matrix::from_nested_csr(
      4, 5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  const std::vector<std::vector<uint32_t>> d{{0}, {1}, {2}, {3}};
  const std::vector<std::vector<uint32_t>> o{{0}};

  auto make_window = [&](const char *id, const char *decoder,
                         cudaqx::heterogeneous_map params = {}) {
    task_graph_node_config n;
    n.id = id;
    n.kind = "single_window";
    n.d_sparse = d;
    n.o_sparse = o;
    n.decoder_instance = get_decoder(decoder, h, std::move(params));
    return n;
  };

  cudaqx::heterogeneous_map z_params;
  z_params.insert("lut_error_depth", 2);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(make_window("x", "single_error_lut"));
  nodes.push_back(make_window("z", "multi_error_lut", std::move(z_params)));
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));

  const std::vector<task_graph_edge> edges{
      {"measurements", "output", "x", "input"},
      {"measurements", "output", "z", "input"},
      {"x", "output", "combine", "a"},
      {"z", "output", "combine", "b"},
      {"combine", "output", "output", "input"}};
  auto graph = decoding_task_graph::from_topology(std::move(nodes), edges);

  auto run_bits = [&](std::vector<uint8_t> bits) {
    measurement_results m;
    m.bits = std::move(bits);
    return graph->run(m).bits;
  };

  // Zero: both branches agree at 0, xor_node must emit 0.
  EXPECT_EQ(run_bits({0, 0, 0, 0}), (std::vector<uint8_t>{0}));
  // Fault 0 alone: both LUTs report observable 1, xor_node must cancel to 0.
  EXPECT_EQ(run_bits({1, 0, 0, 0}), (std::vector<uint8_t>{0}));
  // Unique {0,4}: depth-1 stays 0, depth-2 reports 1, xor_node must emit 1.
  // A broken execute_indexed (constant 0, or forwarding one port) fails here
  // together with the case above.
  EXPECT_EQ(run_bits({1, 0, 0, 1}), (std::vector<uint8_t>{1}));
}

TEST(DecodingTaskGraph, FromTopologyRejectsAnEmptyGraph) {
  EXPECT_THROW(decoding_task_graph::from_topology({}), std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologyRejectsDuplicateNodeIds) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("same"));
  nodes.push_back(rep5_streaming_node("same"));
  EXPECT_THROW(
      decoding_task_graph::from_topology(
          std::move(nodes), {{"measurements", "output", "same", "input"},
                             {"same", "output", "output", "input"}}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, FromTopologyInfersEdgesOnlyForOneNode) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(rep5_streaming_node("a"));
  nodes.push_back(rep5_streaming_node("b"));
  try {
    decoding_task_graph::from_topology(std::move(nodes));
    FAIL() << "expected a two-node graph with no edges to be rejected";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("edges are required"),
              std::string::npos)
        << e.what();
  }
}

TEST(DecodingTaskGraph, FromTopologyRejectsAnUnregisteredKind) {
  std::vector<task_graph_node_config> nodes;
  auto node = rep5_streaming_node();
  node.kind = "trapezoid_window";
  nodes.push_back(std::move(node));
  EXPECT_THROW(decoding_task_graph::from_topology(std::move(nodes)),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, XorNodeTakesNeitherDecoderNorDem) {
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  combiner.dem = rep5_spec(5);
  nodes.push_back(std::move(combiner));
  EXPECT_THROW(decoding_task_graph::from_topology(std::move(nodes)),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, GuardedNodeBuildsThroughTopologyAndStreams) {
  std::vector<task_graph_node_config> nodes;
  auto node = rep5_streaming_node();
  node.node_params.insert("window_size", std::size_t{4});
  node.node_params.insert("step_size", std::size_t{3});
  node.node_params.insert("end_guard", std::size_t{1});
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));

  const std::size_t per_round = graph->get_num_msyn_per_round();
  ASSERT_EQ(graph->get_num_msyn_per_decode(), 4 * per_round);
  std::vector<uint8_t> round(per_round, 0);
  for (int i = 0; i < 3; ++i)
    EXPECT_FALSE(graph->enqueue_syndrome(round.data(), per_round));
  EXPECT_TRUE(graph->enqueue_syndrome(round.data(), per_round));
  for (std::size_t i = 0; i < graph->get_num_observables(); ++i)
    EXPECT_EQ(graph->get_obs_corrections()[i], 0u);
}

TEST(DecodingTaskGraph, SingleWindowTakesNoStepSizeOrGuards) {
  // The kind's schema does not declare them, so a stride over flat-free DEM
  // still covers the whole window and a guard never reaches the node.
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "node0";
  node.kind = "single_window";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));
  EXPECT_EQ(graph->get_num_msyn_per_decode(),
            5 * graph->get_num_msyn_per_round());
}

// ---------------------------------------------------------------------------
// Node-kind registry extensibility
// ---------------------------------------------------------------------------

// A kind registered from this test file: if it parses, validates, builds, and
// shows up in the exported schema without touching a shipped file, the registry
// is open in the way the design claims.
static const bool test_kind_registered = [] {
  decoding::config::register_decoder_schema({"test_only_nested", {}});

  task_graph_node_kind_spec spec;
  spec.name = "test_only_window";
  spec.aliases = {"TestOnlyWindow"};
  decoding::config::decoder_schema schema;
  schema.params = window_kind_common_params();
  schema.params.push_back({"required_knob",
                           decoding::config::param_kind::uint64,
                           /*required=*/true});
  schema.params.push_back(
      {"unused_knob", decoding::config::param_kind::uint64});
  schema.params.push_back(
      {"nested_kind", decoding::config::param_kind::string});
  schema.params.push_back({"nested_params",
                           decoding::config::param_kind::discriminated, false,
                           "", "nested_kind", /*materialize_empty=*/true});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    if (p.contains("unused_knob") && p.get<std::size_t>("unused_knob") > 7)
      throw std::runtime_error("test_only_window: unused_knob must be <= 7");
  };
  register_node_kind(
      std::move(spec), std::move(schema), [](node_build_context &ctx) {
        if (ctx.node_params->contains("nested_kind") &&
            !ctx.node_params->contains("nested_params"))
          throw std::logic_error(
              "test_only_window: defaults were not materialized");
        return make_window_node(ctx, /*sliding=*/false);
      });
  return true;
}();

TEST(DecodingTaskGraph, RegisteredKindIsVisibleAndBuildable) {
  ASSERT_TRUE(test_kind_registered);
  const auto kinds = registered_task_graph_node_kinds();
  EXPECT_NE(std::find(kinds.begin(), kinds.end(), "test_only_window"),
            kinds.end());
  // An alias resolves to the canonical name, which is what emission writes.
  const auto *found = find_task_graph_node_kind("TestOnlyWindow");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name, "test_only_window");

  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "custom";
  node.kind = "test_only_window";
  node.dem = rep5_spec(5);
  node.decoder_type = "single_error_lut";
  node.node_params.insert("required_knob", std::size_t{1});
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));
  EXPECT_GT(graph->get_num_msyn_per_decode(), 0u);
}

TEST(DecodingTaskGraph, RegisteredKindParamsAreSchemaChecked) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto *schema =
      cfg::find_decoder_schema(task_graph_node_schema_name("test_only_window"));
  ASSERT_NE(schema, nullptr);

  cudaqx::heterogeneous_map good;
  good.insert("required_knob", std::size_t{1});
  good.insert("unused_knob", std::size_t{3});
  EXPECT_NO_THROW(cfg::validate_custom_args(*schema, good, "test"));

  cudaqx::heterogeneous_map bad;
  bad.insert("required_knob", std::size_t{1});
  bad.insert("unused_knob", std::size_t{9});
  EXPECT_THROW(cfg::validate_custom_args(*schema, bad, "test"),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// Open edge-value registry
// ---------------------------------------------------------------------------

/// A payload no built-in node knows about, registered from this TU exactly
/// the way a plugin's .cpp would register one.
struct test_dummy_value {
  std::vector<float> x;
};

/// Deliberately never registered.
struct test_unregistered_value {
  int v = 0;
};

CUDAQ_REGISTER_EDGE_VALUE_TYPE(test_dummy_value, "test_dummy_value")
CUDAQ_REGISTER_EDGE_VALUE_TYPE(test_dummy_value, "test_dummy_value_alias")
static const bool dummy_value_registered = true;

/// Single-port base for the kinds below: no decoder, no DEM.  Streaming is
/// added per node through the capability mixin.
class one_shot_test_node : public task_graph_node,
                           public streaming_task_graph_node {
public:
  streaming_task_graph_node *streaming() override { return this; }
  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    return false;
  }
  const uint8_t *corrections() const override { return &corr_; }
  void clear_corrections() override { corr_ = 0; }
  void reset_streaming_state() override { corr_ = 0; }
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return 1; }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }

protected:
  uint8_t corr_ = 0;
};

/// measurement_results in, test_dummy_value out (the bits, widened).  While
/// streaming, each raw byte fires and goes out on the streaming edge.
class dummy_source_node : public one_shot_test_node {
public:
  dummy_source_node() : val_(test_dummy_value{}) {}
  bool enqueue(const uint8_t *msyn, std::size_t, round_type) override {
    val_.get<test_dummy_value>().x.assign(1, static_cast<float>(msyn[0]));
    return true;
  }
  std::size_t num_msyn_per_decode() const override { return 1; }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return slot == 0 ? &val_ : nullptr;
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("test_dummy_value");
    return std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == edge_value_type("measurement_results");
  }
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    const auto &m = inputs[0].get<measurement_results>();
    test_dummy_value d;
    d.x.assign(m.bits.begin(), m.bits.end());
    outputs[0] = std::move(d);
  }

private:
  edge_value val_;
};

/// test_dummy_value in, logical_outcome out: one bit, the payload's parity.
/// While streaming, each routed payload's parity folds into the correction.
class dummy_sink_node : public one_shot_test_node {
public:
  bool supports_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override {
    if (output_slot != 0 || inputs.size() != 1)
      return std::nullopt;
    return inputs[0].schedule;
  }
  bool enqueue_edge_value(const edge_value &v, std::size_t) override {
    for (float f : v.get<test_dummy_value>().x)
      corr_ ^= static_cast<uint8_t>(f != 0.0f);
    return true;
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("logical_outcome");
    return std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == edge_value_type("test_dummy_value");
  }
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    const auto &d = inputs[0].get<test_dummy_value>();
    logical_outcome lo;
    lo.converged = true;
    lo.bits.assign(1, 0);
    for (float v : d.x)
      lo.bits[0] ^= static_cast<uint8_t>(v != 0.0f);
    outputs[0] = std::move(lo);
  }
};

/// Names an output type nothing registered.
class unregistered_source_node : public dummy_source_node {
public:
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type::of<test_unregistered_value>();
    return std::nullopt;
  }
};

/// Accepts only a type nothing registered.
class unregistered_sink_node : public dummy_sink_node {
public:
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == edge_value_type::of<test_unregistered_value>();
  }
};

/// Declares its input through a second name for test_dummy_value.
class aliased_sink_node : public dummy_sink_node {
public:
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == edge_value_type("test_dummy_value_alias");
  }
};

static void register_one_shot_test_kind(const char *name,
                                        node_factory factory) {
  task_graph_node_kind_spec spec;
  spec.name = name;
  spec.requires_decoder = false;
  spec.requires_dem = false;
  register_node_kind(std::move(spec), {}, std::move(factory));
}

static const bool dummy_kinds_registered = [] {
  register_one_shot_test_kind("test_dummy_source", [](node_build_context &) {
    return std::unique_ptr<task_graph_node>(new dummy_source_node());
  });
  register_one_shot_test_kind("test_dummy_sink", [](node_build_context &) {
    return std::unique_ptr<task_graph_node>(new dummy_sink_node());
  });
  register_one_shot_test_kind(
      "test_unregistered_source", [](node_build_context &) {
        return std::unique_ptr<task_graph_node>(new unregistered_source_node());
      });
  register_one_shot_test_kind(
      "test_unregistered_sink", [](node_build_context &) {
        return std::unique_ptr<task_graph_node>(new unregistered_sink_node());
      });
  register_one_shot_test_kind(
      "test_dummy_sink_aliased", [](node_build_context &) {
        return std::unique_ptr<task_graph_node>(new aliased_sink_node());
      });
  return true;
}();

static task_graph_node_config decoderless_node(const char *id,
                                               const char *kind) {
  task_graph_node_config cfg;
  cfg.id = id;
  cfg.kind = kind;
  return cfg;
}

TEST(DecodingTaskGraph, RegisteredEdgeValueTypeWiresAndRuns) {
  ASSERT_TRUE(dummy_value_registered && dummy_kinds_registered);
  const auto ids = registered_edge_value_types();
  EXPECT_NE(
      std::find(ids.begin(), ids.end(), edge_value_type("test_dummy_value")),
      ids.end());
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(decoderless_node("src", "test_dummy_source"));
  nodes.push_back(decoderless_node("sink", "test_dummy_sink"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"dets", "output", "src", "input"},
                         {"src", "output", "sink", "input"},
                         {"sink", "output", "output", "input"}});
  measurement_results m;
  m.bits = {1, 0, 1, 1};
  const auto lo = graph->run(m);
  ASSERT_EQ(lo.bits.size(), 1u);
  EXPECT_EQ(lo.bits[0], 1);
  // A second shot reuses the same buffers, so the in-place assignment paths
  // run too.
  m.bits = {1, 1, 0, 0};
  EXPECT_EQ(graph->run(m).bits[0], 0);
}

TEST(DecodingTaskGraph, RegisteredEdgeValueTypeFlowsOnAStreamingEdge) {
  ASSERT_TRUE(dummy_kinds_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(decoderless_node("src", "test_dummy_source"));
  nodes.push_back(decoderless_node("sink", "test_dummy_sink"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"dets", "output", "src", "input"},
                         {"src", "output", "sink", "input"},
                         {"sink", "output", "output", "input"}});
  const uint8_t one = 1, zero = 0;
  ASSERT_TRUE(graph->enqueue_syndrome(&one, 1));
  EXPECT_EQ(graph->get_obs_corrections()[0], 1);
  ASSERT_TRUE(graph->enqueue_syndrome(&zero, 1));
  EXPECT_EQ(graph->get_obs_corrections()[0], 1);
  ASSERT_TRUE(graph->enqueue_syndrome(&one, 1));
  EXPECT_EQ(graph->get_obs_corrections()[0], 0);
}

TEST(DecodingTaskGraph, EdgeValueRegistryRejectsBadNames) {
  ASSERT_TRUE(dummy_value_registered);
  EXPECT_THROW(detail::edge_value_registration(
                   "", std::in_place_type<test_unregistered_value>),
               std::invalid_argument);
  // A second module may register the same binding; the name lives until both
  // have released it.
  {
    detail::edge_value_registration again("test_dummy_value",
                                          std::in_place_type<test_dummy_value>);
  }
  EXPECT_EQ(edge_value_type("test_dummy_value"),
            edge_value_type::of<test_dummy_value>());
  // A name is taken whatever payload asks for it.
  EXPECT_THROW(
      detail::edge_value_registration(
          "test_dummy_value", std::in_place_type<test_unregistered_value>),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, EdgeValueAliasesNameOneTypeAndWireTogether) {
  ASSERT_TRUE(dummy_value_registered && dummy_kinds_registered);
  EXPECT_EQ(edge_value_type("test_dummy_value_alias"),
            edge_value_type("test_dummy_value"));
  EXPECT_EQ(edge_value_type("test_dummy_value_alias"),
            edge_value_type::of<test_dummy_value>());
  EXPECT_EQ(edge_value{test_dummy_value{}}.type(),
            edge_value_type("test_dummy_value_alias"));

  // A source declaring the type by one name feeds a sink declaring it by the
  // other; the engine compares types, never names.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(decoderless_node("src", "test_dummy_source"));
  nodes.push_back(decoderless_node("sink", "test_dummy_sink_aliased"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"dets", "output", "src", "input"},
                         {"src", "output", "sink", "input"},
                         {"sink", "output", "output", "input"}});
  measurement_results m;
  m.bits = {1, 0, 1, 1};
  EXPECT_EQ(graph->run(m).bits[0], 1);
}

TEST(DecodingTaskGraph, UnregisteredEdgeValueTypeCanBeRegisteredAgain) {
  struct payload {
    int v = 0;
  };
  const auto type = edge_value_type::of<payload>();
  const auto is_registered = [&] {
    const auto ids = registered_edge_value_types();
    return std::find(ids.begin(), ids.end(), type) != ids.end();
  };

  EXPECT_FALSE(is_registered());
  {
    detail::edge_value_registration reg("test_reregistered_value",
                                        std::in_place_type<payload>);
    EXPECT_TRUE(is_registered());
    EXPECT_EQ(edge_value{payload{}}.type(), type);
  }
  EXPECT_FALSE(is_registered());
  EXPECT_THROW((void)edge_value_type("test_reregistered_value"),
               std::invalid_argument);
  EXPECT_THROW(edge_value{payload{}}, std::invalid_argument);
  {
    detail::edge_value_registration reg("test_reregistered_value",
                                        std::in_place_type<payload>);
    EXPECT_TRUE(is_registered());
  }
  EXPECT_FALSE(is_registered());
}

TEST(DecodingTaskGraph, EdgeValueTypeStaysRegisteredWhileAnyNameRemains) {
  struct payload {
    int v = 0;
  };
  std::optional<detail::edge_value_registration> first(
      std::in_place, "test_alias_first", std::in_place_type<payload>);
  std::optional<detail::edge_value_registration> second(
      std::in_place, "test_alias_second", std::in_place_type<payload>);
  first.reset();
  EXPECT_EQ(edge_value{payload{}}.type(), edge_value_type::of<payload>());
  EXPECT_EQ(edge_value_type("test_alias_second"),
            edge_value_type::of<payload>());
  EXPECT_THROW((void)edge_value_type("test_alias_first"),
               std::invalid_argument);
  second.reset();
  EXPECT_THROW(edge_value{payload{}}, std::invalid_argument);
}

TEST(DecodingTaskGraph, RegisteredEdgeValueTypeMismatchIsRejected) {
  ASSERT_TRUE(dummy_kinds_registered);
  // A logical_outcome port into a port that takes only test_dummy_value fails
  // the same way any other port mismatch does, naming both ends.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(decoderless_node("src", "test_dummy_source"));
  nodes.push_back(decoderless_node("sink", "test_dummy_sink"));
  nodes.push_back(decoderless_node("sink2", "test_dummy_sink"));
  try {
    decoding_task_graph::from_topology(
        std::move(nodes), {{"dets", "output", "src", "input"},
                           {"src", "output", "sink", "input"},
                           {"sink", "output", "sink2", "input"},
                           {"sink2", "output", "output", "input"}});
    FAIL() << "expected the logical_outcome -> test_dummy_value wire to fail";
  } catch (const std::invalid_argument &e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("edge type mismatch"), std::string::npos) << what;
    EXPECT_NE(what.find("'sink2'"), std::string::npos) << what;
  }
}

TEST(DecodingTaskGraph, UnregisteredEdgeValueTypeIsRefusedAtBuild) {
  ASSERT_TRUE(dummy_kinds_registered);
  {
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(decoderless_node("src", "test_unregistered_source"));
    nodes.push_back(decoderless_node("sink", "test_dummy_sink"));
    try {
      decoding_task_graph::from_topology(
          std::move(nodes), {{"dets", "output", "src", "input"},
                             {"src", "output", "sink", "input"},
                             {"sink", "output", "output", "input"}});
      FAIL() << "expected an unregistered output type to be refused";
    } catch (const std::invalid_argument &e) {
      const std::string what = e.what();
      EXPECT_NE(what.find("unregistered edge-value type"), std::string::npos)
          << what;
      EXPECT_NE(what.find("'src'"), std::string::npos) << what;
    }
  }
  {
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(decoderless_node("src", "test_dummy_source"));
    nodes.push_back(decoderless_node("sink", "test_unregistered_sink"));
    try {
      decoding_task_graph::from_topology(
          std::move(nodes), {{"dets", "output", "src", "input"},
                             {"src", "output", "sink", "input"},
                             {"sink", "output", "output", "input"}});
      FAIL() << "expected an input accepting no registered type to be refused";
    } catch (const std::invalid_argument &e) {
      const std::string what = e.what();
      EXPECT_NE(what.find("accepts no registered edge-value type"),
                std::string::npos)
          << what;
      EXPECT_NE(what.find("'sink'"), std::string::npos) << what;
    }
  }
  // Building a value of an unregistered type fails the same way rather than
  // as a bad cast, and so does looking a name up that nothing registered.
  EXPECT_THROW((void)edge_value(test_unregistered_value{}),
               std::invalid_argument);
  EXPECT_THROW((void)edge_value_type("test_unregistered_value"),
               std::invalid_argument);
}

TEST(DecodingTaskGraph, EdgeValueAssignsInPlaceWhenTypesMatch) {
  edge_value v{logical_outcome{}};
  EXPECT_EQ(v.type(), edge_value_type("logical_outcome"));
  auto &lo = v.get<logical_outcome>();
  lo.bits.assign(4, 1);
  const auto *payload = lo.bits.data();

  // Same T on both sides: the value is copied into the payload that is
  // already there, so its storage stays put.
  logical_outcome other;
  other.bits.assign(4, 0);
  v = other;
  EXPECT_EQ(v.get<logical_outcome>().bits, other.bits);
  EXPECT_EQ(v.get<logical_outcome>().bits.data(), payload);
  edge_value w{logical_outcome{}};
  w.get<logical_outcome>().bits.assign(4, 1);
  v = w;
  EXPECT_EQ(v.get<logical_outcome>().bits, w.get<logical_outcome>().bits);
  EXPECT_EQ(v.get<logical_outcome>().bits.data(), payload);
  // A move steals the other payload's storage instead of copying it.
  const auto *stolen = w.get<logical_outcome>().bits.data();
  v = std::move(w);
  EXPECT_EQ(v.get<logical_outcome>().bits.data(), stolen);

  // Moving from a value empties it; it can be assigned to and from again
  // without tripping the in-place path on a payload that is no longer there.
  edge_value moved{std::move(v)};
  EXPECT_FALSE(v.type().has_value());
  EXPECT_EQ(v.get_if<logical_outcome>(), nullptr);
  EXPECT_NO_THROW(v = moved);
  EXPECT_EQ(v.get<logical_outcome>().bits, moved.get<logical_outcome>().bits);
  edge_value sink;
  sink = std::move(moved);
  EXPECT_FALSE(moved.type().has_value());
  EXPECT_EQ(sink.type(), edge_value_type("logical_outcome"));
  // A same-type move assignment moves the payload's contents and leaves the
  // source holding a valid, moved-from T.
  v = std::move(sink);
  EXPECT_TRUE(sink.type().has_value());
  // Assigning an empty value, copied or moved, empties the target.
  v = moved;
  EXPECT_FALSE(v.type().has_value());
  sink = std::move(moved);
  EXPECT_FALSE(sink.type().has_value());

  // Asking for the wrong type throws, and an empty value holds nothing.
  EXPECT_THROW(v.get<detection_events>(), std::invalid_argument);
  edge_value empty;
  EXPECT_FALSE(empty.type().has_value());
  EXPECT_THROW(empty.get<logical_outcome>(), std::invalid_argument);

  // A different T replaces the payload.
  detection_events dets;
  dets.bits.assign(2, 1.0f);
  v = dets;
  EXPECT_NE(v.get_if<detection_events>(), nullptr);
  EXPECT_EQ(v.get_if<logical_outcome>(), nullptr);
}

TEST(DecodingTaskGraph, FromTopologyAppliesNodeParameterSchema) {
  const auto make_node = [](cudaqx::heterogeneous_map params) {
    task_graph_node_config node;
    node.id = "custom";
    node.kind = "test_only_window";
    node.dem = rep5_spec(5);
    node.decoder_type = "single_error_lut";
    node.node_params = std::move(params);
    return node;
  };
  const auto build = [&](cudaqx::heterogeneous_map params) {
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(make_node(std::move(params)));
    return decoding_task_graph::from_topology(std::move(nodes));
  };

  cudaqx::heterogeneous_map unknown;
  unknown.insert("required_knob", std::size_t{1});
  unknown.insert("step_szie", std::size_t{1});
  EXPECT_THROW(build(std::move(unknown)), std::runtime_error);

  EXPECT_THROW(build({}), std::runtime_error);

  cudaqx::heterogeneous_map wrong_type;
  wrong_type.insert("required_knob", std::size_t{1});
  wrong_type.insert("unused_knob", std::string{"three"});
  EXPECT_THROW(build(std::move(wrong_type)), std::runtime_error);

  cudaqx::heterogeneous_map cross_field;
  cross_field.insert("required_knob", std::size_t{1});
  cross_field.insert("unused_knob", std::size_t{9});
  EXPECT_THROW(build(std::move(cross_field)), std::runtime_error);

  cudaqx::heterogeneous_map defaults;
  defaults.insert("required_knob", std::size_t{1});
  defaults.insert("nested_kind", std::string{"test_only_nested"});
  EXPECT_NO_THROW(build(std::move(defaults)));
}

// ---------------------------------------------------------------------------
// Async one-shot interface
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, RunAsyncOutlivesItsArgument) {
  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");

  auto m = std::make_unique<measurement_results>();
  m->bits.assign(graph->get_num_msyn_per_decode(), 0);
  const auto expected = graph->run(*m);

  auto future = graph->run_async(*m);
  m.reset(); // the async task must not depend on the caller's shot
  const auto actual = future.get();

  EXPECT_EQ(actual.converged, expected.converged);
  EXPECT_EQ(actual.bits, expected.bits);
}

// ---------------------------------------------------------------------------
// cuda_device_id(): what decode_target::pin_device() reads (decode_target.h).
// ---------------------------------------------------------------------------

namespace {
// Reports whatever device it's told, without touching CUDA -- a real pin goes
// through decoder::get()'s "cuda_device_id" param, which needs a physical GPU.
class pinned_decoder final : public decoder {
public:
  pinned_decoder(sparse_binary_matrix h, int device)
      : decoder(decoder_init(std::move(h))) {
    cuda_device_id_ = device;
  }
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }
};

// Claims graph dispatch unconditionally, like nv-qldpc-decoder does -- used to
// prove window_node::supports_graph_dispatch() gates on ws_ (stride/guards)
// itself, not just the inner decoder's own answer.
class graph_capable_decoder final : public decoder {
public:
  explicit graph_capable_decoder(sparse_binary_matrix h)
      : decoder(decoder_init(std::move(h))) {}
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }
  bool supports_graph_dispatch() const override { return true; }
};

// Blocks inside decode() until released, so a second caller genuinely overlaps
// the first rather than merely appearing to. Every tripwire test below needs
// real overlap: the contract is about concurrency, and a test that only poked
// the counter would pass against a broken guard.
class gated_decoder final : public decoder {
public:
  explicit gated_decoder(sparse_binary_matrix h)
      : decoder(decoder_init(std::move(h))) {}
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (!entered_) {
        entered_ = true;
        entered_cv_.notify_all();
      }
      // Bounded, so a broken guard produces a red test rather than a hung one:
      // if the tripwire stops firing, the second caller reaches decode() and
      // would otherwise block here forever with nobody left to release it.
      released_cv_.wait_for(lk, std::chrono::seconds(10),
                            [&] { return released_; });
    }
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }
  /// Blocks until some thread is inside decode().
  void wait_until_entered() {
    std::unique_lock<std::mutex> lk(mu_);
    ASSERT_TRUE(entered_cv_.wait_for(lk, std::chrono::seconds(10), [&] {
      return entered_;
    })) << "no thread ever entered decode()";
  }
  void release() {
    std::lock_guard<std::mutex> lk(mu_);
    released_ = true;
    released_cv_.notify_all();
  }
  /// Opens the gate up front, for the cases that need a decode to complete
  /// without a second thread driving it.
  void open() { release(); }

private:
  std::mutex mu_;
  std::condition_variable entered_cv_, released_cv_;
  bool entered_ = false, released_ = false;
};

/// Coordinates two independent decoders so a test can prove they execute at
/// the same time without relying on elapsed-time comparisons.
class parallel_decode_probe {
public:
  void enter() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++entered_;
    entered_cv_.notify_all();
  }

  void enter_and_wait_for_peers(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    ++entered_;
    entered_cv_.notify_all();
    entered_cv_.wait_for(lock, std::chrono::seconds(2),
                         [&] { return entered_ >= count; });
  }

  void enter_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++entered_;
    entered_cv_.notify_all();
    release_cv_.wait_for(lock, std::chrono::seconds(10),
                         [&] { return released_; });
  } // end - enter_and_wait()

  bool wait_for_entered(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, std::chrono::seconds(2),
                                [&] { return entered_ >= count; });
  } // end - wait_for_entered()

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    release_cv_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  std::size_t entered_ = 0;
  bool released_ = false;
};

class async_stream_probe {
public:
  explicit async_stream_probe(bool blocked = false) : blocked_(blocked) {}

  void record_and_wait(const std::vector<float_t> &syndrome) {
    std::unique_lock<std::mutex> lock(mutex_);
    syndromes_.push_back(syndrome);
    entered_cv_.notify_all();
    if (blocked_)
      release_cv_.wait_for(lock, std::chrono::seconds(10),
                           [&] { return released_; });
  }

  bool wait_for_count(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, std::chrono::seconds(2),
                                [&] { return syndromes_.size() >= count; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    release_cv_.notify_all();
  }

  std::vector<std::vector<float_t>> syndromes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return syndromes_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  std::vector<std::vector<float_t>> syndromes_;
  bool blocked_ = false;
  bool released_ = false;
};

/// Decoder whose decode call rendezvous with sibling graph branches.
class parallel_probe_decoder final : public decoder {
public:
  parallel_probe_decoder(sparse_binary_matrix h,
                         std::shared_ptr<parallel_decode_probe> probe,
                         bool should_throw = false,
                         bool recover_after_reset = false,
                         std::size_t output_width = 0,
                         bool wait_for_peer_before_throw = false)
      : decoder(decoder_init(std::move(h))), probe_(std::move(probe)),
        should_throw_(should_throw), recover_after_reset_(recover_after_reset),
        wait_for_peer_before_throw_(wait_for_peer_before_throw) {
    output_width_ = output_width;
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    std::shared_ptr<parallel_decode_probe> probe = probe_;
    if (should_throw_) {
      if (wait_for_peer_before_throw_)
        probe->enter_and_wait_for_peers(2);
      else
        probe->enter();
      throw std::runtime_error("parallel probe failure");
    }
    probe->enter_and_wait();
    return decoder_result{
        true, std::vector<float_t>(
                  output_width_ ? output_width_ : syndrome.size(), 0.0f)};
  } // end - decode()

  void reset_decoder() override {
    if (recover_after_reset_)
      should_throw_ = false;
  }

private:
  std::shared_ptr<parallel_decode_probe> probe_;
  bool should_throw_ = false;
  bool recover_after_reset_ = false;
  std::size_t output_width_ = 0;
  bool wait_for_peer_before_throw_ = false;
};

class async_stream_probe_decoder final : public decoder {
public:
  async_stream_probe_decoder(sparse_binary_matrix h,
                             std::shared_ptr<async_stream_probe> probe)
      : decoder(decoder_init(std::move(h))), probe_(std::move(probe)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    probe_->record_and_wait(syndrome);
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }

private:
  std::shared_ptr<async_stream_probe> probe_;
};

class coordinator_cpu_probe_decoder final : public decoder {
public:
  coordinator_cpu_probe_decoder(sparse_binary_matrix h,
                                std::shared_ptr<std::atomic<int>> observed)
      : decoder(decoder_init(std::move(h))), observed_(std::move(observed)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
#if defined(__linux__)
    observed_->store(sched_getcpu(), std::memory_order_release);
#endif
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }

private:
  std::shared_ptr<std::atomic<int>> observed_;
};

class prefix_wait_probe {
public:
  void decode() {
    std::unique_lock<std::mutex> lock(mutex_);
    const std::size_t record = ++entered_;
    entered_cv_.notify_all();
    if (record == 1)
      first_cv_.wait(lock, [&] { return release_first_; });
    else if (record == 3)
      later_cv_.wait(lock, [&] { return release_later_; });
  }

  bool wait_for_entered(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, std::chrono::seconds(2),
                                [&] { return entered_ >= count; });
  }

  void release_first() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_first_ = true;
    first_cv_.notify_all();
  }

  void release_later() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_later_ = true;
    later_cv_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable entered_cv_, first_cv_, later_cv_;
  std::size_t entered_ = 0;
  bool release_first_ = false;
  bool release_later_ = false;
};

class prefix_wait_decoder final : public decoder {
public:
  prefix_wait_decoder(sparse_binary_matrix h,
                      std::shared_ptr<prefix_wait_probe> probe)
      : decoder(decoder_init(std::move(h))), probe_(std::move(probe)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    probe_->decode();
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }

private:
  std::shared_ptr<prefix_wait_probe> probe_;
};

template <typename Predicate>
task_graph_streaming_result
wait_for_streaming_result(decoding_task_graph &graph, Predicate ready) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  do {
    task_graph_streaming_result snapshot = graph.streaming_result();
    if (ready(snapshot))
      return snapshot;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);

  ADD_FAILURE() << "timed out waiting for asynchronous streaming result";
  return graph.streaming_result();
}

/// Records which worker executes each node in a width-one dependency chain.
class decode_thread_probe {
public:
  void record() {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_ids_.push_back(std::this_thread::get_id());
  }

  bool exactly_two_on_one_thread() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return thread_ids_.size() == 2 && thread_ids_[0] == thread_ids_[1];
  } // end - exactly_two_on_one_thread()

private:
  mutable std::mutex mutex_;
  std::vector<std::thread::id> thread_ids_;
};

/// Decoder used to observe serial execution without changing node results.
class thread_recording_decoder final : public decoder {
public:
  thread_recording_decoder(sparse_binary_matrix h,
                           std::shared_ptr<decode_thread_probe> probe)
      : decoder(decoder_init(std::move(h))), probe_(std::move(probe)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    probe_->record();
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }

private:
  std::shared_ptr<decode_thread_probe> probe_;
};

/// Records the syndrome observed by every consumer of a fanned-out edge.
class fanout_probe {
public:
  void record(const std::vector<float_t> &syndrome) {
    std::lock_guard<std::mutex> lock(mutex_);
    syndromes_.push_back(syndrome);
  }

  bool observed_twice(const std::vector<float_t> &expected) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return syndromes_.size() == 2 &&
           std::all_of(
               syndromes_.begin(), syndromes_.end(),
               [&](const auto &syndrome) { return syndrome == expected; });
  }

private:
  mutable std::mutex mutex_;
  std::vector<std::vector<float_t>> syndromes_;
};

class fanout_recording_decoder final : public decoder {
public:
  fanout_recording_decoder(sparse_binary_matrix h,
                           std::shared_ptr<fanout_probe> probe)
      : decoder(decoder_init(std::move(h))), probe_(std::move(probe)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    probe_->record(syndrome);
    return decoder_result{true, std::vector<float_t>(syndrome.size(), 0.0f)};
  }

private:
  std::shared_ptr<fanout_probe> probe_;
};

// Returns the graph and a borrowed pointer to its gate.
std::pair<std::unique_ptr<decoding_task_graph>, gated_decoder *>
make_gated_graph() {
  auto h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto dec = std::make_unique<gated_decoder>(h);
  auto *gate = dec.get();
  auto g = std::make_unique<decoding_task_graph>(
      std::move(dec), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}});
  return {std::move(g), gate};
}

std::unique_ptr<decoding_task_graph> make_single_node_graph(int device) {
  auto h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto dec = std::make_unique<pinned_decoder>(h, device);
  return std::make_unique<decoding_task_graph>(
      std::move(dec), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}});
}

std::unique_ptr<decoding_task_graph> make_parallel_probe_graph(
    const std::shared_ptr<parallel_decode_probe> &probe,
    bool should_throw = false, bool recover_after_reset = false,
    task_graph_options options = {}, bool wait_for_peer_before_throw = false) {
  sparse_binary_matrix h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto dec = std::make_unique<parallel_probe_decoder>(
      std::move(h), probe, should_throw, recover_after_reset,
      /*output_width=*/0, wait_for_peer_before_throw);
  return std::make_unique<decoding_task_graph>(
      std::move(dec), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}}, std::move(options));
} // end - make_parallel_probe_graph()

std::unique_ptr<decoding_task_graph>
make_async_stream_probe_graph(const std::shared_ptr<async_stream_probe> &probe,
                              task_graph_options options = {}) {
  sparse_binary_matrix h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto dec = std::make_unique<async_stream_probe_decoder>(std::move(h), probe);
  return std::make_unique<decoding_task_graph>(
      std::move(dec), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}}, std::move(options));
}
} // namespace

TEST(DecodingTaskGraph, AsyncStreamingCopiesInputAndAppliesBackpressure) {
  auto probe = std::make_shared<async_stream_probe>(/*blocked=*/true);
  auto graph = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 2});
  uint8_t first = 0;
  uint8_t second = 1;
  uint8_t third = 0;

  EXPECT_FALSE(graph->enqueue_syndrome(&first, 1));
  ASSERT_TRUE(probe->wait_for_count(1));
  EXPECT_FALSE(graph->enqueue_syndrome(&second, 1));
  second = 0;

  auto final = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(&third, 1, round_type::FINAL);
  });
  EXPECT_EQ(final.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  probe->release();
  EXPECT_TRUE(final.get());
  EXPECT_EQ(probe->syndromes(),
            (std::vector<std::vector<float_t>>{{0}, {1}, {0}}));
} // end - AsyncStreamingCopiesInputAndAppliesBackpressure

TEST(DecodingTaskGraph, AsyncStreamingPinsItsCoordinatorWhenRequested) {
#if !defined(__linux__)
  GTEST_SKIP() << "coordinator CPU affinity is Linux-only";
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed),
            0);
  int requested_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &allowed)) {
      requested_cpu = cpu;
      break;
    }
  ASSERT_GE(requested_cpu, 0);

  auto observed = std::make_shared<std::atomic<int>>(-1);
  auto h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto decoder =
      std::make_unique<coordinator_cpu_probe_decoder>(std::move(h), observed);
  auto graph = std::make_unique<decoding_task_graph>(
      std::move(decoder), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}},
      task_graph_options{.async_streaming = true,
                         .async_queue_capacity = 1,
                         .async_coordinator_cpu = requested_cpu,
                         .async_spin_pause_iters = 0});
  const uint8_t zero = 0;

  ASSERT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));
  EXPECT_EQ(observed->load(std::memory_order_acquire), requested_cpu);
#endif
}

TEST(DecodingTaskGraph, AsyncStreamingRejectsCoordinatorCpuBelowSentinel) {
  auto probe = std::make_shared<async_stream_probe>();
  EXPECT_THROW(
      (void)make_async_stream_probe_graph(probe, {.async_streaming = true,
                                                  .async_queue_capacity = 1,
                                                  .async_coordinator_cpu = -2}),
      std::invalid_argument);
}

TEST(DecodingTaskGraph, AsyncStreamingRejectsCoordinatorCpuBeyondCpuSet) {
#if !defined(__linux__)
  GTEST_SKIP() << "CPU_SETSIZE is a Linux affinity limit";
#else
  auto probe = std::make_shared<async_stream_probe>();
  EXPECT_THROW((void)make_async_stream_probe_graph(
                   probe, {.async_streaming = true,
                           .async_queue_capacity = 1,
                           .async_coordinator_cpu = CPU_SETSIZE}),
               std::out_of_range);
#endif
}

TEST(DecodingTaskGraph,
     AsyncStreamingRollsBackAfterCoordinatorAffinityFailure) {
#if !defined(__linux__)
  GTEST_SKIP() << "pthread coordinator affinity is Linux-only";
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed),
            0);
  int unavailable_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (!CPU_ISSET(cpu, &allowed)) {
      unavailable_cpu = cpu;
      break;
    }
  if (unavailable_cpu < 0)
    GTEST_SKIP() << "all representable CPUs are available to this process";

  auto probe = std::make_shared<async_stream_probe>();
  EXPECT_THROW((void)make_async_stream_probe_graph(
                   probe, {.async_streaming = true,
                           .async_queue_capacity = 1,
                           .async_coordinator_cpu = unavailable_cpu}),
               std::runtime_error);

  // A failed preparation must join its coordinator rather than leaking state
  // that interferes with a later graph.
  auto recovered = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 1});
  const uint8_t zero = 0;
  EXPECT_TRUE(recovered->enqueue_syndrome(&zero, 1, round_type::FINAL));
#endif
}

TEST(DecodingTaskGraph, AsyncStreamingRejectsAffinityOffLinux) {
#if defined(__linux__)
  GTEST_SKIP() << "non-Linux affinity rejection is compiled on other hosts";
#else
  auto probe = std::make_shared<async_stream_probe>();
  EXPECT_THROW(
      (void)make_async_stream_probe_graph(probe, {.async_streaming = true,
                                                  .async_queue_capacity = 1,
                                                  .async_coordinator_cpu = 0}),
      std::runtime_error);
#endif
}

TEST(DecodingTaskGraph, AsyncStreamingPublishesStableMidShotResults) {
  auto synchronous =
      decoding_task_graph::from_spec(rep5_streaming_spec(), "single_error_lut");
  auto asynchronous = decoding_task_graph::from_spec(
      rep5_streaming_spec(), "single_error_lut", {},
      {.async_streaming = true, .async_queue_capacity = 1});
  const std::size_t syndrome_size = synchronous->get_num_msyn_per_decode();
  std::vector<uint8_t> syndrome(syndrome_size, uint8_t{0});
  syndrome[0] = 1;
  ASSERT_GT(syndrome.size(), asynchronous->get_num_msyn_per_round());

  ASSERT_TRUE(synchronous->enqueue_syndrome(syndrome.data(), syndrome.size()));
  const task_graph_streaming_result expected = synchronous->streaming_result();
  ASSERT_NE(
      std::count(expected.values.begin(), expected.values.end(), uint8_t{0}),
      static_cast<long>(expected.values.size()));

  EXPECT_FALSE(
      asynchronous->enqueue_syndrome(syndrome.data(), syndrome.size()));
  const auto snapshot = wait_for_streaming_result(
      *asynchronous, [&](const task_graph_streaming_result &result) {
        return result.result_revision >= expected.result_revision;
      });
  EXPECT_EQ(snapshot.generation, expected.generation);
  EXPECT_EQ(snapshot.result_revision, expected.result_revision);
  EXPECT_EQ(snapshot.status, expected.status);
  EXPECT_EQ(snapshot.values, expected.values);
  const std::vector<uint8_t> corrections(
      asynchronous->get_obs_corrections(),
      asynchronous->get_obs_corrections() +
          asynchronous->get_num_observables());
  EXPECT_EQ(corrections, snapshot.values);

  const std::vector<uint8_t> last_round(asynchronous->get_num_msyn_per_round(),
                                        uint8_t{0});
  EXPECT_NO_THROW((void)asynchronous->enqueue_syndrome(
      last_round.data(), last_round.size(), round_type::FINAL));
  const auto final = asynchronous->streaming_result();
  EXPECT_EQ(final.status, task_graph_streaming_status::complete);
  EXPECT_EQ(final.values, snapshot.values);
  asynchronous->clear_corrections();
  EXPECT_EQ(asynchronous->streaming_result().status,
            task_graph_streaming_status::pending);
} // end - AsyncStreamingPublishesStableMidShotResults

TEST(DecodingTaskGraph, AsyncStreamingClearsCompletedOperationPulse) {
  auto graph = decoding_task_graph::from_spec(
      rep5_spec(5), "single_error_lut", {},
      {.async_streaming = true, .async_queue_capacity = 2});
  std::vector<uint8_t> window(graph->get_num_msyn_per_decode(), 0);
  std::vector<uint8_t> partial(graph->get_num_msyn_per_round(), 0);

  EXPECT_FALSE(graph->enqueue_syndrome(window.data(), window.size()));
  const auto completed = graph->streaming_result();
  ASSERT_TRUE(completed.operation_completed);
  ASSERT_EQ(completed.completed_operations, 1u);

  EXPECT_FALSE(graph->enqueue_syndrome(partial.data(), partial.size()));
  const auto pending = graph->streaming_result();
  EXPECT_FALSE(pending.operation_completed);
  EXPECT_EQ(pending.completed_operations, 1u);

  graph->reset_decoder();
}

TEST(DecodingTaskGraph, AsyncClearMatchesGuardedMidShotConsume) {
  const auto build = [](task_graph_options options = {}) {
    task_graph_node_config node;
    node.id = "w";
    node.kind = "sliding_window";
    node.dem = rep5_spec(5);
    node.decoder_type = "single_error_lut";
    node.node_params.insert("window_size", std::size_t{5});
    node.node_params.insert("step_size", std::size_t{3});
    node.node_params.insert("start_guard", std::size_t{1});
    node.node_params.insert("end_guard", std::size_t{1});
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    return decoding_task_graph::from_topology(std::move(nodes), {},
                                              std::move(options));
  };

  auto synchronous = build();
  auto asynchronous =
      build({.async_streaming = true, .async_queue_capacity = 1});
  const auto expect_progress_equal =
      [](const task_graph_streaming_result &actual,
         const task_graph_streaming_result &expected) {
        ASSERT_EQ(actual.outputs.size(), expected.outputs.size());
        for (std::size_t i = 0; i < actual.outputs.size(); ++i) {
          EXPECT_EQ(actual.outputs[i].generation,
                    expected.outputs[i].generation);
          EXPECT_EQ(actual.outputs[i].coverage_begin_round,
                    expected.outputs[i].coverage_begin_round);
          EXPECT_EQ(actual.outputs[i].coverage_end_round,
                    expected.outputs[i].coverage_end_round);
          EXPECT_EQ(actual.outputs[i].status, expected.outputs[i].status);
        }
      };
  const std::size_t window = synchronous->get_num_msyn_per_decode();
  const std::size_t per_round = synchronous->get_num_msyn_per_round();
  std::vector<uint8_t> first_window(window, uint8_t{0});
  first_window[window - per_round] = 1; // withheld by the end guard

  ASSERT_TRUE(
      synchronous->enqueue_syndrome(first_window.data(), first_window.size()));
  EXPECT_FALSE(
      asynchronous->enqueue_syndrome(first_window.data(), first_window.size()));

  // clear_corrections() is a consume-and-continue operation. The async graph
  // must drain this enqueue without publishing the guarded tail as FINAL would.
  synchronous->clear_corrections();
  asynchronous->clear_corrections();
  const auto sync_consumed = synchronous->streaming_result();
  const auto async_consumed = asynchronous->streaming_result();
  EXPECT_EQ(async_consumed.generation, sync_consumed.generation);
  expect_progress_equal(async_consumed, sync_consumed);
  EXPECT_EQ(async_consumed.values, sync_consumed.values);

  // Preserve the same overlap and commit the same next core and final tail.
  std::vector<uint8_t> stride(3 * per_round, uint8_t{0});
  ASSERT_TRUE(synchronous->enqueue_syndrome(stride.data(), stride.size()));
  EXPECT_FALSE(asynchronous->enqueue_syndrome(stride.data(), stride.size()));
  const std::vector<uint8_t> last_round(per_round, uint8_t{0});
  ASSERT_TRUE(synchronous->enqueue_syndrome(
      last_round.data(), last_round.size(), round_type::FINAL));
  ASSERT_TRUE(asynchronous->enqueue_syndrome(
      last_round.data(), last_round.size(), round_type::FINAL));
  EXPECT_EQ(asynchronous->streaming_result().generation,
            synchronous->streaming_result().generation);
  expect_progress_equal(asynchronous->streaming_result(),
                        synchronous->streaming_result());
  EXPECT_EQ(asynchronous->streaming_result().values,
            synchronous->streaming_result().values);
} // end - AsyncClearMatchesGuardedMidShotConsume

TEST(DecodingTaskGraph, AsyncStreamingDefersWorkerFailureUntilExistingApiCall) {
  auto probe = std::make_shared<parallel_decode_probe>();
  auto graph = make_parallel_probe_graph(
      probe, /*should_throw=*/true,
      /*recover_after_reset=*/true,
      {.async_streaming = true, .async_queue_capacity = 1});
  uint8_t syndrome = 0;

  EXPECT_FALSE(graph->enqueue_syndrome(&syndrome, 1));
  ASSERT_TRUE(probe->wait_for_entered(1));

  const auto snapshot = wait_for_streaming_result(
      *graph, [](const task_graph_streaming_result &result) {
        return result.status == task_graph_streaming_status::failed;
      });
  EXPECT_EQ(snapshot.status, task_graph_streaming_status::failed);
  EXPECT_FALSE(snapshot.operation_completed);
  EXPECT_FALSE(snapshot.error.empty());
  ASSERT_FALSE(snapshot.outputs.empty());
  for (const auto &output : snapshot.outputs)
    EXPECT_EQ(output.status, task_graph_streaming_status::failed);
  EXPECT_EQ(std::vector<uint8_t>(graph->get_obs_corrections(),
                                 graph->get_obs_corrections() +
                                     graph->get_num_observables()),
            snapshot.values);
  EXPECT_THROW(graph->enqueue_syndrome(&syndrome, 1), std::runtime_error);

  EXPECT_NO_THROW(graph->reset_decoder());
  const auto reset = graph->streaming_result();
  EXPECT_EQ(reset.status, task_graph_streaming_status::pending);
  EXPECT_GT(reset.shot_epoch, snapshot.shot_epoch);

  probe->release();
  EXPECT_TRUE(graph->enqueue_syndrome(&syndrome, 1, round_type::FINAL));
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
} // end - AsyncStreamingDefersWorkerFailureUntilExistingApiCall

TEST(DecodingTaskGraph, AsyncStreamingPreservesFifoAcrossRingWraparound) {
  auto probe = std::make_shared<async_stream_probe>();
  auto graph = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 2});

  for (std::size_t i = 0; i < 19; ++i) {
    uint8_t syndrome = static_cast<uint8_t>(i % 2);
    EXPECT_FALSE(graph->enqueue_syndrome(&syndrome, 1));
  }
  uint8_t final_syndrome = 1;
  EXPECT_TRUE(graph->enqueue_syndrome(&final_syndrome, 1, round_type::FINAL));

  const auto observed = probe->syndromes();
  ASSERT_EQ(observed.size(), 20);
  for (std::size_t i = 0; i < observed.size(); ++i)
    EXPECT_EQ(observed[i], (std::vector<float_t>{static_cast<float_t>(i % 2)}));
} // end - AsyncStreamingPreservesFifoAcrossRingWraparound

TEST(DecodingTaskGraph, AsyncStreamingResultWaitsOnlyForAcceptedPrefix) {
  auto probe = std::make_shared<prefix_wait_probe>();
  auto h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto decoder = std::make_unique<prefix_wait_decoder>(std::move(h), probe);
  auto graph = std::make_unique<decoding_task_graph>(
      std::move(decoder), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}},
      task_graph_options{.async_streaming = true, .async_queue_capacity = 4});
  const uint8_t zero = 0;

  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));
  ASSERT_TRUE(probe->wait_for_entered(1));
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));

  auto prefix = std::async(std::launch::async, [&] {
    return task_graph_streaming_result(graph->streaming_result());
  });
  // Give streaming_result() time to capture the two-record submitted prefix
  // while record one remains blocked, then accept a later record.
  EXPECT_EQ(prefix.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));

  probe->release_first();
  const bool third_entered = probe->wait_for_entered(3);
  if (!third_entered)
    probe->release_later();
  ASSERT_TRUE(third_entered);
  const auto prefix_status = prefix.wait_for(std::chrono::seconds(1));
  if (prefix_status != std::future_status::ready)
    probe->release_later();
  ASSERT_EQ(prefix_status, std::future_status::ready);
  const auto snapshot = prefix.get();
  EXPECT_EQ(snapshot.completed_operations, 2u);

  probe->release_later();
  EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));
}

TEST(DecodingTaskGraph, AsyncCorrectionsWaitOnlyForAcceptedPrefix) {
  auto probe = std::make_shared<prefix_wait_probe>();
  auto h = sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  auto decoder = std::make_unique<prefix_wait_decoder>(std::move(h), probe);
  auto graph = std::make_unique<decoding_task_graph>(
      std::move(decoder), std::vector<std::vector<uint32_t>>{{0}},
      std::vector<std::vector<uint32_t>>{{0}},
      task_graph_options{.async_streaming = true, .async_queue_capacity = 4});
  const uint8_t zero = 0;

  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));
  ASSERT_TRUE(probe->wait_for_entered(1));
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));

  auto prefix = std::async(std::launch::async, [&] {
    const auto *corrections = graph->get_obs_corrections();
    return std::vector<uint8_t>(corrections,
                                corrections + graph->get_num_observables());
  });
  // Capture two accepted records, then prove a third record accepted while
  // this accessor waits does not extend its synchronization boundary.
  EXPECT_EQ(prefix.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));

  probe->release_first();
  const bool third_entered = probe->wait_for_entered(3);
  if (!third_entered)
    probe->release_later();
  ASSERT_TRUE(third_entered);
  const auto prefix_status = prefix.wait_for(std::chrono::seconds(1));
  if (prefix_status != std::future_status::ready)
    probe->release_later();
  ASSERT_EQ(prefix_status, std::future_status::ready);
  EXPECT_EQ(prefix.get(), (std::vector<uint8_t>{0}));

  probe->release_later();
  EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));
}

TEST(DecodingTaskGraph, AsyncStreamingReusesPreparedCoordinatorAcrossShots) {
  auto probe = std::make_shared<async_stream_probe>();
  auto graph = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 1});

  for (std::size_t shot = 0; shot < 100; ++shot) {
    uint8_t syndrome = static_cast<uint8_t>(shot % 2);
    EXPECT_TRUE(graph->enqueue_syndrome(&syndrome, 1, round_type::FINAL));
    graph->clear_corrections();
  }

  const auto observed = probe->syndromes();
  ASSERT_EQ(observed.size(), 100);
  for (std::size_t shot = 0; shot < observed.size(); ++shot)
    EXPECT_EQ(observed[shot],
              (std::vector<float_t>{static_cast<float_t>(shot % 2)}));
} // end - AsyncStreamingReusesPreparedCoordinatorAcrossShots

TEST(DecodingTaskGraph,
     AsyncStreamingFinalAndImplicitDrainMatchAcrossCapacities) {
  const uint8_t zero = 0;
  for (const std::size_t capacity : {1u, 2u, 7u}) {
    auto probe = std::make_shared<async_stream_probe>();
    auto graph =
        make_async_stream_probe_graph(probe, {.async_streaming = true,
                                              .async_queue_capacity = capacity,
                                              .async_spin_pause_iters = 32});

    // clear_corrections() closes an active session without manufacturing a
    // FINAL flush, then the next shot can start on the same coordinator.
    EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));
    graph->clear_corrections();
    EXPECT_EQ(graph->streaming_result().status,
              task_graph_streaming_status::pending);

    for (std::size_t shot = 0; shot < 50; ++shot) {
      EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));
      const auto result = graph->streaming_result();
      EXPECT_EQ(result.status, task_graph_streaming_status::complete);
      EXPECT_TRUE(result.operation_completed);
      graph->clear_corrections();
    }
    EXPECT_EQ(probe->syndromes().size(), 51u);
  }
}

TEST(DecodingTaskGraph, AsyncStreamingMatchesSynchronousGraphKinds) {
  const auto compare = [](std::unique_ptr<decoding_task_graph> synchronous,
                          std::unique_ptr<decoding_task_graph> asynchronous) {
    std::vector<uint8_t> syndrome(synchronous->get_num_msyn_per_decode(),
                                  uint8_t{0});
    for (std::size_t i = 0; i < syndrome.size(); ++i)
      syndrome[i] = static_cast<uint8_t>(i % 2);

    (void)synchronous->enqueue_syndrome(syndrome.data(), syndrome.size(),
                                        round_type::FINAL);
    const std::vector<uint8_t> expected(synchronous->get_obs_corrections(),
                                        synchronous->get_obs_corrections() +
                                            synchronous->get_num_observables());

    EXPECT_NO_THROW((void)asynchronous->enqueue_syndrome(
        syndrome.data(), syndrome.size(), round_type::FINAL));
    const std::vector<uint8_t> actual(asynchronous->get_obs_corrections(),
                                      asynchronous->get_obs_corrections() +
                                          asynchronous->get_num_observables());
    EXPECT_EQ(actual, expected);
  };

  compare(decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut"),
          decoding_task_graph::from_spec(
              rep5_spec(5), "single_error_lut", {},
              {.async_streaming = true, .async_queue_capacity = 2}));
  compare(
      make_guarded_graph(),
      make_guarded_graph({.async_streaming = true, .async_queue_capacity = 2}));

  auto sync_left =
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto sync_right =
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto async_left =
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  auto async_right =
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  compare(
      decoding_task_graph::combine(std::move(sync_left), std::move(sync_right)),
      decoding_task_graph::combine(
          std::move(async_left), std::move(async_right),
          {.async_streaming = true, .async_queue_capacity = 2}));

  compare(build_residual_graph(),
          build_residual_graph(
              {.async_streaming = true, .async_queue_capacity = 2}));
} // end - AsyncStreamingMatchesSynchronousGraphKinds

TEST(DecodingTaskGraph, StreamingExecutesIndependentBranchesConcurrently) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  auto probe = std::make_shared<parallel_decode_probe>();
  auto left = make_parallel_probe_graph(probe);
  auto right = make_parallel_probe_graph(probe);
  auto graph = decoding_task_graph::combine(std::move(left), std::move(right));
  uint8_t syndrome = 0;

  auto enqueue = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(&syndrome, 1, round_type::FINAL);
  });
  const bool both_entered = probe->wait_for_entered(2);
  probe->release();

  EXPECT_TRUE(both_entered);
  EXPECT_TRUE(enqueue.get());
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
}

TEST(DecodingTaskGraph, AsyncStreamingExecutesIndependentBranchesConcurrently) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  auto probe = std::make_shared<parallel_decode_probe>();
  auto left = make_parallel_probe_graph(probe);
  auto right = make_parallel_probe_graph(probe);
  auto graph = decoding_task_graph::combine(
      std::move(left), std::move(right),
      {.async_streaming = true, .async_queue_capacity = 2});
  uint8_t syndrome = 0;

  auto enqueue = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(&syndrome, 1, round_type::FINAL);
  });
  const bool both_entered = probe->wait_for_entered(2);
  probe->release();
  EXPECT_TRUE(enqueue.get());
  const auto result = graph->streaming_result();

  EXPECT_TRUE(both_entered);
  EXPECT_EQ(result.status, task_graph_streaming_status::complete);
  EXPECT_EQ(result.generation, 1u);
}

TEST(DecodingTaskGraph, AsyncFinalReportsFlushFailureAndResetRecovers) {
  ASSERT_TRUE(ordered_stream_registered);
  const auto node = [](std::string id, std::string mode) {
    task_graph_node_config config;
    config.id = std::move(id);
    config.kind = "test_ordered_stream";
    config.node_params.insert("mode", std::move(mode));
    config.node_params.insert("value", std::size_t{0});
    return config;
  };

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("healthy", "tail_source"));
  nodes.push_back(node("failing", "throwing_tail_source"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "healthy", "input"},
       {"measurements", "output", "failing", "input"},
       {"healthy", "output", "output", "healthy"},
       {"failing", "output", "output", "failing"}},
      {.async_streaming = true, .async_queue_capacity = 2});
  const uint8_t zero = 0;

  EXPECT_THROW((void)graph->enqueue_syndrome(&zero, 1, round_type::FINAL),
               std::runtime_error);
  const auto failed = graph->streaming_result();
  EXPECT_EQ(failed.status, task_graph_streaming_status::failed);
  EXPECT_NE(failed.error.find("ordered stream flush failure"),
            std::string::npos);

  EXPECT_NO_THROW(graph->reset_decoder());
  EXPECT_GT(graph->streaming_result().shot_epoch, failed.shot_epoch);
  EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
}

TEST(DecodingTaskGraph, AsyncResetCancelsQueuedRecordsAndStartsCleanShot) {
  auto probe = std::make_shared<async_stream_probe>(/*blocked=*/true);
  auto graph = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 3});
  const auto first_epoch = graph->streaming_result().shot_epoch;
  uint8_t first = 0;
  uint8_t queued_one = 1;
  uint8_t queued_two = 0;

  EXPECT_FALSE(graph->enqueue_syndrome(&first, 1));
  ASSERT_TRUE(probe->wait_for_count(1));
  EXPECT_FALSE(graph->enqueue_syndrome(&queued_one, 1));
  EXPECT_FALSE(graph->enqueue_syndrome(&queued_two, 1));

  auto reset = std::async(std::launch::async, [&] { graph->reset_decoder(); });
  EXPECT_EQ(reset.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  probe->release();
  EXPECT_NO_THROW(reset.get());

  ASSERT_EQ(probe->syndromes().size(), 1u);
  EXPECT_GT(graph->streaming_result().shot_epoch, first_epoch);
  uint8_t next_shot = 1;
  EXPECT_TRUE(graph->enqueue_syndrome(&next_shot, 1, round_type::FINAL));
  ASSERT_EQ(probe->syndromes().size(), 2u);
  EXPECT_EQ(probe->syndromes().back(), (std::vector<float_t>{1.0f}));
}

TEST(DecodingTaskGraph,
     AsyncParallelRoutingPreservesOrderAcrossQueueWraparound) {
  ASSERT_TRUE(ordered_stream_registered);
  ordered_stream_node::clear_observed();
  const auto node = [](std::string id, std::string mode,
                       std::size_t value = 0) {
    task_graph_node_config config;
    config.id = std::move(id);
    config.kind = "test_ordered_stream";
    config.node_params.insert("mode", std::move(mode));
    config.node_params.insert("value", value);
    return config;
  };

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("left", "source", 0));
  nodes.push_back(node("right", "source", 1));
  nodes.push_back(node("join", "join"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "left", "input"},
       {"measurements", "output", "right", "input"},
       {"left", "output", "join", "a"},
       {"right", "output", "join", "b"},
       {"join", "output", "output", "input"}},
      {.async_streaming = true, .async_queue_capacity = 2});

  std::vector<std::pair<std::size_t, uint8_t>> expected;
  for (std::size_t record = 0; record < 20; ++record) {
    const uint8_t syndrome = static_cast<uint8_t>(record % 2);
    const auto type = record == 19 ? round_type::FINAL : round_type::BULK;
    (void)graph->enqueue_syndrome(&syndrome, 1, type);
    expected.emplace_back(0, syndrome);
    expected.emplace_back(1, syndrome ^ 1u);
  }

  EXPECT_EQ(ordered_stream_node::observed_deliveries(), expected);
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
}

TEST(DecodingTaskGraph, AsyncStreamingRejectsOverlappingProducers) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "overlapping producers need at least two CPU threads";

  constexpr std::size_t num_threads = 8;
  constexpr std::size_t iterations = 500;
  auto probe = std::make_shared<async_stream_probe>();
  auto graph = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 64});
  const uint8_t warmup = 0;
  EXPECT_FALSE(graph->enqueue_syndrome(&warmup, 1));
  (void)graph->streaming_result();

  std::barrier rendezvous(static_cast<std::ptrdiff_t>(num_threads));
  std::atomic<std::size_t> rejected = 0;
  std::atomic<std::size_t> unexpected = 0;
  std::vector<std::thread> producers;
  producers.reserve(num_threads);
  for (std::size_t thread = 0; thread < num_threads; ++thread) {
    producers.emplace_back([&] {
      const uint8_t syndrome = 0;
      for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        rendezvous.arrive_and_wait();
        try {
          (void)graph->enqueue_syndrome(&syndrome, 1);
        } catch (const std::logic_error &error) {
          if (std::string(error.what()).find("concurrent producers") !=
              std::string::npos)
            rejected.fetch_add(1, std::memory_order_relaxed);
          else
            unexpected.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto &producer : producers)
    producer.join();

  EXPECT_GT(rejected.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(unexpected.load(std::memory_order_relaxed), 0u);
  const uint8_t final = 0;
  EXPECT_TRUE(graph->enqueue_syndrome(&final, 1, round_type::FINAL));
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
}

static std::unique_ptr<decoding_task_graph>
make_parallel_fanout_graph(const std::shared_ptr<parallel_decode_probe> &probe,
                           bool left_should_throw = false,
                           bool recover_after_reset = false) {
  const auto model = rep5_model::build();
  task_graph_node_config source;
  source.id = "source";
  source.kind = "test_detector_source";
  source.dem = rep5_streaming_spec();
  source.node_params.insert("window_size", std::size_t{5});

  auto make_consumer = [&](std::string id, bool should_throw) {
    auto node = rep5_streaming_node(std::move(id));
    node.node_params.insert("window_size", std::size_t{5});
    node.node_params.insert("step_size", std::size_t{5});
    node.decoder_type.clear();
    node.decoder_instance = std::make_unique<parallel_probe_decoder>(
        sparse_binary_matrix(model.closed.detector_error_matrix), probe,
        should_throw, recover_after_reset, model.n_mech,
        /*wait_for_peer_before_throw=*/should_throw);
    return node;
  };

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(source));
  nodes.push_back(make_consumer("left", left_should_throw));
  nodes.push_back(make_consumer("right", /*should_throw=*/false));
  return decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"source", "output", "left", "input"},
                         {"source", "output", "right", "input"},
                         {"left", "output", "output", "left"},
                         {"right", "output", "output", "right"}});
}

TEST(DecodingTaskGraph, StreamingFanoutExecutesConsumersConcurrently) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  ASSERT_TRUE(detector_source_registered);
  auto probe = std::make_shared<parallel_decode_probe>();
  auto graph = make_parallel_fanout_graph(probe);
  std::vector<uint8_t> syndrome(graph->get_num_msyn_per_decode(), 0);

  auto enqueue = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(syndrome.data(), syndrome.size(),
                                   round_type::FINAL);
  });
  const bool both_entered = probe->wait_for_entered(2);
  probe->release();

  EXPECT_TRUE(both_entered);
  EXPECT_TRUE(enqueue.get());
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_EQ(result.outputs[0].generation, 1u);
  EXPECT_EQ(result.outputs[1].generation, 1u);
}

TEST(DecodingTaskGraph, ParallelStreamingRouteJoinsBeforeRethrowing) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  ASSERT_TRUE(detector_source_registered);
  auto probe = std::make_shared<parallel_decode_probe>();
  auto graph = make_parallel_fanout_graph(probe, /*left_should_throw=*/true,
                                          /*recover_after_reset=*/true);
  std::vector<uint8_t> syndrome(graph->get_num_msyn_per_decode(), 0);

  auto enqueue = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(syndrome.data(), syndrome.size());
  });
  const bool both_entered = probe->wait_for_entered(2);
  if (both_entered)
    EXPECT_EQ(enqueue.wait_for(std::chrono::milliseconds(20)),
              std::future_status::timeout);
  probe->release();

  EXPECT_TRUE(both_entered);
  EXPECT_THROW((void)enqueue.get(), std::runtime_error);
  graph->reset_decoder();
  EXPECT_TRUE(graph->enqueue_syndrome(syndrome.data(), syndrome.size()));
}

TEST(DecodingTaskGraph, ParallelStreamingFanInIsDeliveredInWireOrder) {
  ASSERT_TRUE(ordered_stream_registered);
  ordered_stream_node::clear_observed();

  const auto node = [](std::string id, std::string mode,
                       std::size_t value = 0) {
    task_graph_node_config config;
    config.id = std::move(id);
    config.kind = "test_ordered_stream";
    config.node_params.insert("mode", std::move(mode));
    config.node_params.insert("value", value);
    return config;
  };
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("left", "source", 0));
  nodes.push_back(node("right", "source", 1));
  nodes.push_back(node("join", "join"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "join", "a"},
                         {"right", "output", "join", "b"},
                         {"join", "output", "output", "input"}});
  uint8_t syndrome[2] = {0, 0};

  EXPECT_TRUE(graph->enqueue_syndrome(syndrome, 2));
  EXPECT_EQ(ordered_stream_node::observed(),
            (std::vector<std::size_t>{0, 0, 1, 1}));
  EXPECT_EQ(graph->streaming_result().generation, 4u);
}

TEST(DecodingTaskGraph, ParallelStreamingPreservesDepthFirstReconvergence) {
  ASSERT_TRUE(ordered_stream_registered);
  ordered_stream_node::clear_observed();

  const auto node = [](std::string id, std::string mode,
                       std::size_t value = 0) {
    task_graph_node_config config;
    config.id = std::move(id);
    config.kind = "test_ordered_stream";
    config.node_params.insert("mode", std::move(mode));
    config.node_params.insert("value", value);
    return config;
  };
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("source", "source"));
  nodes.push_back(node("pass", "pass"));
  nodes.push_back(node("join", "join"));
  nodes.push_back(node("downstream", "pass"));
  nodes.push_back(node("independent", "source"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"source", "output", "pass", "input"},
                         {"pass", "output", "join", "a"},
                         {"source", "output", "join", "b"},
                         {"join", "output", "downstream", "input"},
                         {"measurements", "output", "independent", "input"},
                         {"downstream", "output", "output", "joined"},
                         {"independent", "output", "output", "independent"}});
  uint8_t syndrome = 0;

  EXPECT_TRUE(graph->enqueue_syndrome(&syndrome, 1));
  EXPECT_EQ(ordered_stream_node::observed(), (std::vector<std::size_t>{0, 1}));
}

TEST(DecodingTaskGraph, ParallelStreamingConnectedFlushPreservesSerialOrder) {
  ASSERT_TRUE(ordered_stream_registered);
  ordered_stream_node::clear_observed();

  const auto node = [](std::string id, std::string mode,
                       std::size_t value = 0) {
    task_graph_node_config config;
    config.id = std::move(id);
    config.kind = "test_ordered_stream";
    config.node_params.insert("mode", std::move(mode));
    config.node_params.insert("value", value);
    return config;
  };
  // The topological order is A, C, B, J. A's flush must route through C into
  // J before B's flush reaches J directly.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("a", "tail_source", 0));
  nodes.push_back(node("c", "pass"));
  nodes.push_back(node("b", "tail_source", 1));
  nodes.push_back(node("join", "join"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "a", "input"},
                         {"a", "output", "c", "input"},
                         {"c", "output", "join", "a"},
                         {"measurements", "output", "b", "input"},
                         {"b", "output", "join", "b"},
                         {"join", "output", "output", "input"}});
  uint8_t syndrome = 0;

  EXPECT_TRUE(graph->enqueue_syndrome(&syndrome, 1, round_type::FINAL));
  EXPECT_EQ(ordered_stream_node::observed(), (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
}

TEST(DecodingTaskGraph, SerialStreamingAcceptsNullZeroLengthInput) {
  auto graph = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");

  EXPECT_FALSE(graph->enqueue_syndrome(nullptr, 0));
  EXPECT_EQ(graph->streaming_result().generation, 0u);
}

TEST(DecodingTaskGraph, ParallelStreamingAcceptsNullZeroLengthInput) {
  auto graph = decoding_task_graph::combine(
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut"),
      decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut"));

  EXPECT_FALSE(graph->enqueue_syndrome(nullptr, 0));
  EXPECT_EQ(graph->streaming_result().generation, 0u);
}

TEST(DecodingTaskGraph, ParallelStreamingFlushMatchesSerialBranches) {
  auto left_reference = make_guarded_graph();
  auto right_reference = make_guarded_graph();
  auto graph =
      decoding_task_graph::combine(make_guarded_graph(), make_guarded_graph());
  std::vector<uint8_t> syndrome(graph->get_num_msyn_per_decode(), 0);
  syndrome[syndrome.size() - graph->get_num_msyn_per_round()] = 1;

  ASSERT_TRUE(left_reference->enqueue_syndrome(syndrome.data(), syndrome.size(),
                                               round_type::FINAL));
  ASSERT_TRUE(right_reference->enqueue_syndrome(
      syndrome.data(), syndrome.size(), round_type::FINAL));
  ASSERT_TRUE(graph->enqueue_syndrome(syndrome.data(), syndrome.size(),
                                      round_type::FINAL));

  const auto obs = graph->get_num_observables();
  std::vector<uint8_t> expected(obs, 0);
  for (std::size_t i = 0; i < obs; ++i)
    expected[i] = left_reference->get_obs_corrections()[i] ^
                  right_reference->get_obs_corrections()[i];
  EXPECT_EQ(std::vector<uint8_t>(graph->get_obs_corrections(),
                                 graph->get_obs_corrections() + obs),
            expected);
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::complete);
}

TEST(DecodingTaskGraph, ParallelStreamingJoinsActiveWorkBeforeRethrowing) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  auto probe = std::make_shared<parallel_decode_probe>();
  auto slow = make_parallel_probe_graph(probe);
  auto failing = make_parallel_probe_graph(probe, /*should_throw=*/true,
                                           /*recover_after_reset=*/true, {},
                                           /*wait_for_peer_before_throw=*/true);
  auto graph =
      decoding_task_graph::combine(std::move(failing), std::move(slow));
  uint8_t syndrome = 0;

  auto enqueue = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(&syndrome, 1);
  });
  const bool both_entered = probe->wait_for_entered(2);
  if (both_entered)
    EXPECT_EQ(enqueue.wait_for(std::chrono::milliseconds(20)),
              std::future_status::timeout);
  probe->release();

  EXPECT_TRUE(both_entered);
  EXPECT_THROW((void)enqueue.get(), std::runtime_error);
  graph->reset_decoder();
  EXPECT_TRUE(graph->enqueue_syndrome(&syndrome, 1));
}

TEST(DecodingTaskGraph, ParallelStreamingDoesNotStartQueuedWorkAfterFailure) {
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  ASSERT_TRUE(lockstep_stream_registered && ordered_stream_registered);
  ordered_stream_node::clear_observed();
  active_lockstep_probe = std::make_shared<lockstep_probe>();

  task_graph_node_config source;
  source.id = "source";
  source.kind = "test_ordered_stream";
  source.node_params.insert("mode", std::string{"source"});
  source.node_params.insert("value", std::size_t{0});
  task_graph_node_config join;
  join.id = "join";
  join.kind = "test_ordered_stream";
  join.node_params.insert("mode", std::string{"throwing_join"});
  join.node_params.insert("value", std::size_t{0});
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(lockstep_node("slow", "slow_raw"));
  nodes.push_back(std::move(source));
  nodes.push_back(std::move(join));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "slow", "input"},
                         {"measurements", "output", "source", "input"},
                         {"source", "output", "join", "a"},
                         {"source", "output", "join", "b"},
                         {"slow", "output", "output", "slow"},
                         {"join", "output", "output", "failed"}});
  uint8_t syndrome = 0;

  auto enqueue = std::async(std::launch::async, [&] {
    return graph->enqueue_syndrome(&syndrome, 1);
  });
  const bool slow_entered = active_lockstep_probe->wait_for_slow();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (slow_entered && ordered_stream_node::observed().empty() &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const auto observed_before_release = ordered_stream_node::observed();
  active_lockstep_probe->release();
  EXPECT_TRUE(slow_entered);
  EXPECT_EQ(observed_before_release, (std::vector<std::size_t>{0}));
  try {
    (void)enqueue.get();
    FAIL() << "expected the first queued delivery to fail";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(), "ordered stream failure 0");
  }
  EXPECT_EQ(ordered_stream_node::observed(), (std::vector<std::size_t>{0}));
}

TEST(DecodingTaskGraph, AsyncStreamingDestructorJoinsPendingDecode) {
  auto probe = std::make_shared<async_stream_probe>(/*blocked=*/true);
  auto graph = make_async_stream_probe_graph(
      probe, {.async_streaming = true, .async_queue_capacity = 1});
  uint8_t syndrome = 0;

  EXPECT_FALSE(graph->enqueue_syndrome(&syndrome, 1));
  ASSERT_TRUE(probe->wait_for_count(1));

  auto destroy =
      std::async(std::launch::async,
                 [owned = std::move(graph)]() mutable { owned.reset(); });
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  probe->release();
  EXPECT_NO_THROW(destroy.get());
} // end - AsyncStreamingDestructorJoinsPendingDecode

TEST(DecodingTaskGraph, CudaDeviceIdIsNulloptWhenNoNodeIsPinned) {
  auto g = make_single_node_graph(/*device=*/-1);
  EXPECT_FALSE(g->cuda_device_id().has_value());
}

TEST(DecodingTaskGraph, CudaDeviceIdReflectsThePinnedInnerDecoder) {
  auto g = make_single_node_graph(/*device=*/2);
  ASSERT_TRUE(g->cuda_device_id().has_value());
  EXPECT_EQ(*g->cuda_device_id(), 2);
}

TEST(DecodingTaskGraph, CudaDeviceIdAgreesWhenCombinedNodesShareADevice) {
  auto left = make_single_node_graph(4);
  auto right = make_single_node_graph(4);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  ASSERT_TRUE(combined->cuda_device_id().has_value());
  EXPECT_EQ(*combined->cuda_device_id(), 4);
}

// One thread can't be pinned to two devices for one shot, so this must fail
// loudly rather than silently pick one and risk decoding on the wrong GPU.
TEST(DecodingTaskGraph, CudaDeviceIdThrowsWhenCombinedNodesDisagree) {
  auto left = make_single_node_graph(3);
  auto right = make_single_node_graph(5);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  const auto error_from = [](auto &&operation) {
    try {
      operation();
    } catch (const std::runtime_error &error) {
      return std::string(error.what());
    }
    return std::string{};
  };
  const auto public_error =
      error_from([&] { (void)combined->cuda_device_id(); });
  uint8_t syndrome = 0;
  const auto streaming_error =
      error_from([&] { (void)combined->enqueue_syndrome(&syndrome, 1); });

  EXPECT_FALSE(public_error.empty());
  EXPECT_EQ(streaming_error, public_error);
  EXPECT_NE(public_error.find("3 vs 5"), std::string::npos);
}

// decode_target::pin_device() itself is covered end to end by the
// SessionRegistry-driven tests in test_decoding_server_core.cpp (this binary
// doesn't link libcudart, so a pin_device() call can't link here).

// Only proves the CudaDeviceGuard wrapper didn't change run()'s result --
// can't observe the actual pin without a real GPU.
TEST(DecodingTaskGraph, CombinedRunAsyncMatchesRunOnTheCallingThread) {
  const auto spec = rep5_spec(5);
  auto left = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto right = decoding_task_graph::from_spec(spec, "single_error_lut");
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));

  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);
  const auto expected = combined->run(m);

  auto future = combined->run_async(m);
  const auto actual = future.get();

  EXPECT_EQ(actual.converged, expected.converged);
  EXPECT_EQ(actual.bits, expected.bits);
}

TEST(DecodingTaskGraph, CombinedRunAsyncExecutesReadyBranchesConcurrently) {
  auto probe = std::make_shared<parallel_decode_probe>();
  auto left = make_parallel_probe_graph(probe);
  auto right = make_parallel_probe_graph(probe);
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  measurement_results measurements;
  measurements.bits.assign(combined->get_num_msyn_per_decode(), 0);

  auto result = combined->run_async(measurements);
  const bool both_entered = probe->wait_for_entered(2);
  probe->release();

  EXPECT_TRUE(both_entered);
  EXPECT_EQ(result.get().bits, (std::vector<uint8_t>{0}));
} // end - CombinedRunAsyncExecutesReadyBranchesConcurrently

TEST(DecodingTaskGraph, CombinedRunAsyncJoinsActiveWorkBeforeRethrowing) {
  auto probe = std::make_shared<parallel_decode_probe>();
  auto slow = make_parallel_probe_graph(probe);
  auto failing = make_parallel_probe_graph(probe, /*should_throw=*/true);
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "parallel graph execution needs at least two CPU threads";
  auto combined =
      decoding_task_graph::combine(std::move(slow), std::move(failing));
  measurement_results measurements;
  measurements.bits.assign(combined->get_num_msyn_per_decode(), 0);

  auto result = combined->run_async(measurements);
  const bool both_entered = probe->wait_for_entered(2);
  if (!both_entered)
    probe->release();
  ASSERT_TRUE(both_entered);
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  probe->release();

  EXPECT_THROW((void)result.get(), std::runtime_error);
  EXPECT_THROW((void)combined->run(measurements), std::runtime_error);
} // end - CombinedRunAsyncJoinsActiveWorkBeforeRethrowing

TEST(DecodingTaskGraph, RunAsyncKeepsWidthOneMultiNodeGraphSerial) {
  auto probe = std::make_shared<decode_thread_probe>();
  task_graph_node_config predecoder;
  task_graph_node_config global;
  std::vector<task_graph_node_config> nodes;
  std::vector<task_graph_edge> edges;
  measurement_results measurements;
  std::unique_ptr<decoding_task_graph> graph;

  predecoder.id = "pre";
  predecoder.kind = "predecoder";
  predecoder.d_sparse = {{0}};
  predecoder.o_sparse = {{0}};
  predecoder.h_sparse = {{0}};
  predecoder.node_params.insert("residual_from", std::string("parity_check"));
  predecoder.decoder_instance = std::make_unique<thread_recording_decoder>(
      sparse_binary_matrix::from_nested_csr(1, 1, {{0}}), probe);
  nodes.push_back(std::move(predecoder));

  global.id = "global";
  global.kind = "single_window";
  global.d_sparse = {{0}};
  global.o_sparse = {{0}};
  global.decoder_instance = std::make_unique<thread_recording_decoder>(
      sparse_binary_matrix::from_nested_csr(1, 1, {{0}}), probe);
  nodes.push_back(std::move(global));

  edges = {{"measurements", "output", "pre", "input"},
           {"pre", "residual", "global", "input"},
           {"global", "output", "output", "input"}};
  graph =
      decoding_task_graph::from_topology(std::move(nodes), std::move(edges));
  measurements.bits = {0};

  EXPECT_EQ(graph->run_async(measurements).get().bits,
            (std::vector<uint8_t>{0}));
  EXPECT_TRUE(probe->exactly_two_on_one_thread());
} // end - RunAsyncKeepsWidthOneMultiNodeGraphSerial

TEST(DecodingTaskGraph, RunAsyncPublishesOneOutputToTwoDependents) {
  auto probe = std::make_shared<fanout_probe>();
  const sparse_binary_matrix identity =
      sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  task_graph_node_config producer;
  task_graph_node_config left;
  task_graph_node_config right;
  task_graph_node_config combine;
  std::vector<task_graph_node_config> nodes;
  std::vector<task_graph_edge> edges;
  measurement_results measurements;

  producer.id = "producer";
  producer.kind = "predecoder";
  producer.d_sparse = {{0}};
  producer.o_sparse = {{0}};
  producer.h_sparse = {{0}};
  producer.node_params.insert("residual_from", std::string("parity_check"));
  producer.decoder_instance =
      std::make_unique<fixed_error_decoder>(identity, std::vector<float_t>{0},
                                            /*converged=*/true);
  nodes.push_back(std::move(producer));

  left.id = "left";
  left.kind = "single_window";
  left.d_sparse = {{0}};
  left.o_sparse = {{0}};
  left.decoder_instance =
      std::make_unique<fanout_recording_decoder>(identity, probe);
  nodes.push_back(std::move(left));

  right.id = "right";
  right.kind = "single_window";
  right.d_sparse = {{0}};
  right.o_sparse = {{0}};
  right.decoder_instance =
      std::make_unique<fanout_recording_decoder>(identity, probe);
  nodes.push_back(std::move(right));

  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  edges = {{"measurements", "output", "producer", "input"},
           {"producer", "residual", "left", "input"},
           {"producer", "residual", "right", "input"},
           {"left", "output", "combine", "a"},
           {"right", "output", "combine", "b"},
           {"combine", "output", "output", "input"}};
  auto graph =
      decoding_task_graph::from_topology(std::move(nodes), std::move(edges));
  measurements.bits = {1};

  EXPECT_EQ(graph->run_async(measurements).get().bits,
            (std::vector<uint8_t>{0}));
  EXPECT_TRUE(probe->observed_twice(std::vector<float_t>{1}));
} // end - RunAsyncPublishesOneOutputToTwoDependents

TEST(DecodingTaskGraph, RunAsyncSizesPoolForCrossLevelFrontier) {
  auto probe = std::make_shared<parallel_decode_probe>();
  const sparse_binary_matrix identity =
      sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
  task_graph_node_config producer;
  task_graph_node_config source;
  task_graph_node_config left;
  task_graph_node_config right;
  task_graph_node_config combine_children;
  task_graph_node_config combine_all;
  std::vector<task_graph_node_config> nodes;
  std::vector<task_graph_edge> edges;
  measurement_results measurements;

  if (std::thread::hardware_concurrency() < 3)
    GTEST_SKIP() << "cross-level frontier test needs three CPU threads";

  producer.id = "producer";
  producer.kind = "predecoder";
  producer.d_sparse = {{0}};
  producer.o_sparse = {{0}};
  producer.h_sparse = {{0}};
  producer.node_params.insert("residual_from", std::string("parity_check"));
  producer.decoder_instance =
      std::make_unique<fixed_error_decoder>(identity, std::vector<float_t>{0},
                                            /*converged=*/true);
  nodes.push_back(std::move(producer));

  source.id = "source";
  source.kind = "single_window";
  source.d_sparse = {{0}};
  source.o_sparse = {{0}};
  source.decoder_instance =
      std::make_unique<parallel_probe_decoder>(identity, probe);
  nodes.push_back(std::move(source));

  left.id = "left";
  left.kind = "single_window";
  left.d_sparse = {{0}};
  left.o_sparse = {{0}};
  left.decoder_instance =
      std::make_unique<parallel_probe_decoder>(identity, probe);
  nodes.push_back(std::move(left));

  right.id = "right";
  right.kind = "single_window";
  right.d_sparse = {{0}};
  right.o_sparse = {{0}};
  right.decoder_instance =
      std::make_unique<parallel_probe_decoder>(identity, probe);
  nodes.push_back(std::move(right));

  combine_children.id = "combine_children";
  combine_children.kind = "xor";
  nodes.push_back(std::move(combine_children));
  combine_all.id = "combine_all";
  combine_all.kind = "xor";
  nodes.push_back(std::move(combine_all));

  edges = {{"measurements", "output", "producer", "input"},
           {"measurements", "output", "source", "input"},
           {"producer", "residual", "left", "input"},
           {"producer", "residual", "right", "input"},
           {"left", "output", "combine_children", "a"},
           {"right", "output", "combine_children", "b"},
           {"combine_children", "output", "combine_all", "a"},
           {"source", "output", "combine_all", "b"},
           {"combine_all", "output", "output", "input"}};
  auto graph =
      decoding_task_graph::from_topology(std::move(nodes), std::move(edges));
  measurements.bits = {1};

  auto result = graph->run_async(measurements);
  const bool full_frontier_entered = probe->wait_for_entered(3);
  probe->release();

  EXPECT_TRUE(full_frontier_entered);
  EXPECT_EQ(result.get().bits, (std::vector<uint8_t>{0}));
} // end - RunAsyncSizesPoolForCrossLevelFrontier

// Must surface from the call itself, not deferred into the future, where a
// caller that never inspects the exception would miss it.
TEST(DecodingTaskGraph, CombinedRunAsyncThrowsSynchronouslyWhenNodesDisagree) {
  auto left = make_single_node_graph(3);
  auto right = make_single_node_graph(5);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));

  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);
  EXPECT_THROW(combined->run_async(m), std::runtime_error);
}

// ---------------------------------------------------------------------------
// DEVICE-mode graph dispatch: supports_graph_dispatch() / capture_graph() /
// release_graph(). Every case below is checked before window_node ever calls
// into hardware_guards.h (either the node doesn't support graph dispatch, or
// the graph is multi-node), so none of this touches CUDA -- safe on a
// GPU-less machine and in a binary that doesn't link libcudart.
// ---------------------------------------------------------------------------

TEST(DecodingTaskGraph, SupportsGraphDispatchIsFalseForAnOrdinaryDecoder) {
  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");
  EXPECT_FALSE(graph->supports_graph_dispatch());
}

// capture_graph() delegates the whole window to the inner decoder's own
// captured graph, which fires once per full window and resets to empty
// afterward -- it has no notion of sliding (retaining size - stride rounds)
// or of masking guard columns out of the commit. A sliding window with a
// graph-capable inner decoder must still report false, or DEVICE dispatch
// would silently fire every full window instead of every stride.
TEST(DecodingTaskGraph, SupportsGraphDispatchIsFalseForASlidingWindow) {
  task_graph_node_config node;
  node.id = "node0";
  node.kind = "sliding_window";
  node.dem = rep5_streaming_spec();
  // Shaped like the DEM this node builds, so the decoder is one the node could
  // actually drive rather than a 1x1 placeholder that only survives because
  // nothing calls decode() here.
  node.decoder_instance =
      std::make_unique<graph_capable_decoder>(rep5_streaming_h());
  node.node_params.insert("step_size", std::size_t{1});

  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));

  // Not vacuous: the inner decoder really does claim graph dispatch: it's
  // window_node's own stride check that must override it.
  ASSERT_TRUE(graph->inner_decoder()->supports_graph_dispatch());
  EXPECT_FALSE(graph->supports_graph_dispatch());
}

TEST(DecodingTaskGraph, SupportsGraphDispatchIsFalseForACombinedGraph) {
  // pinned_decoder never overrides supports_graph_dispatch() (defaults to
  // false), but the point here is the node-count check itself: a combined
  // graph reports false regardless of what its branches could otherwise do.
  auto left = make_single_node_graph(-1);
  auto right = make_single_node_graph(-1);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  EXPECT_FALSE(combined->supports_graph_dispatch());
}

TEST(DecodingTaskGraph, CaptureGraphThrowsWhenTheNodeDoesNotSupportIt) {
  const auto spec = rep5_spec(5);
  auto graph = decoding_task_graph::from_spec(spec, "single_error_lut");
  ASSERT_FALSE(graph->supports_graph_dispatch());
  EXPECT_THROW(graph->capture_graph(), std::logic_error);
}

TEST(DecodingTaskGraph, CaptureGraphThrowsForAMultiNodeGraph) {
  auto left = make_single_node_graph(-1);
  auto right = make_single_node_graph(-1);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  EXPECT_THROW(combined->capture_graph(), std::logic_error);
}

TEST(DecodingTaskGraph, ReleaseGraphThrowsForAMultiNodeGraph) {
  auto left = make_single_node_graph(-1);
  auto right = make_single_node_graph(-1);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  EXPECT_THROW(combined->release_graph(nullptr), std::logic_error);
}

} // namespace

// ---------------------------------------------------------------------------
// Threading-contract tripwire
// ---------------------------------------------------------------------------
//
// Before the tripwire, two overlapping run_async calls on one multi-node graph
// aborted the process on a corrupted heap ("malloc(): unaligned tcache chunk")
// on every run -- not a wrong answer a test could assert on, so the guard has
// to convert the race into a throw at the call that would have raced.

TEST(DecodingTaskGraph, TripwireRejectsConcurrentRun) {
  auto [g, gate] = make_gated_graph();
  measurement_results m;
  m.bits.assign(1, 0);

  std::thread first([&] { (void)g->run(m); });
  gate->wait_until_entered();
  // Break-it: drop the guard from run() and this second call returns normally.
  EXPECT_THROW(g->run(m), std::logic_error);
  gate->release();
  first.join();
}

TEST(DecodingTaskGraph, TripwireRejectsRunWhileAStreamingDecodeIsInFlight) {
  auto [g, gate] = make_gated_graph();
  const uint8_t byte = 0;
  measurement_results m;
  m.bits.assign(1, 0);

  std::thread first([&] { (void)g->enqueue_syndrome(&byte, 1); });
  gate->wait_until_entered();
  // The concurrent half of "streaming and one-shot are mutually exclusive".
  // Break-it: unarm enqueue_syndrome.
  EXPECT_THROW(g->run(m), std::logic_error);
  gate->release();
  first.join();
}

TEST(DecodingTaskGraph,
     TripwireRejectsASecondRunAsyncBeforeTheFirstIsCollected) {
  auto [g, gate] = make_gated_graph();
  measurement_results m;
  m.bits.assign(1, 0);

  auto f1 = g->run_async(m);
  gate->wait_until_entered();
  // The case the whole design exists for: decode_async() launches eagerly, so
  // the token has to ride the future rather than the call frame.
  // Break-it: guard run_async with a plain call-scoped in_flight_guard.
  EXPECT_THROW(g->run_async(m), std::logic_error);
  gate->release();
  (void)f1.get();
}

TEST(DecodingTaskGraph, TripwireClearsAfterACollectedRunAsync) {
  auto [g, gate] = make_gated_graph();
  measurement_results m;
  m.bits.assign(1, 0);
  gate->open();

  auto f = g->run_async(m);
  (void)f.get();
  // f is deliberately still alive: collecting, not destroying, is what frees
  // the graph, so a caller that keeps its futures around is not locked out.
  // Break-it: hold the token past get() (e.g. copy the shared_ptr into the
  // future's own state). Note the explicit token->release() in
  // pending_async_run is NOT what this pins -- libstdc++ destroys the deferred
  // functor inside get(), which releases the token anyway; the explicit call
  // is there because that destruction timing is not guaranteed.
  EXPECT_NO_THROW((void)g->run(m));
}

TEST(DecodingTaskGraph, TripwireClearsAfterADiscardedRunAsyncFuture) {
  auto [g, gate] = make_gated_graph();
  measurement_results m;
  m.bits.assign(1, 0);
  gate->open();

  {
    auto f = g->run_async(m);
  } // never collected
  // Break-it: remove ~in_flight_guard's release, or reorder
  // pending_async_run's members so the token outlives the joining future.
  EXPECT_NO_THROW((void)g->run(m));
}

// The single-node and multi-node run_async paths release the token by
// different mechanisms -- pending_async_run holds it until the future is
// collected or destroyed, scoped_release frees it when the worker body ends --
// so the single-node tests above cannot stand in for these two. Dropping the
// lambda's `token` capture passes every one of them while reintroducing the
// original heap corruption.

TEST(DecodingTaskGraph, TripwireRejectsOverlappingRunAsyncOnAMultiNodeGraph) {
  auto [ga, gate_a] = make_gated_graph();
  auto [gb, gate_b] = make_gated_graph();
  auto combined = decoding_task_graph::combine(std::move(ga), std::move(gb));
  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);

  // The motivating failure verbatim: two overlapping run_async calls on a
  // combined graph, which aborted on a corrupted heap before the tripwire.
  auto f1 = combined->run_async(m);
  gate_a->wait_until_entered(); // worker is inside run_dag, not merely queued
  // Break-it, verified: drop `token` from the worker lambda's capture list (and
  // the scoped_release that needs it) -- the local shared_ptr then dies when
  // run_async returns, freeing the graph while run_dag is still walking it.
  EXPECT_THROW((void)combined->run_async(m), std::logic_error);
  gate_a->release();
  gate_b->release();
  (void)f1.get();
}

TEST(DecodingTaskGraph, TripwireClearsWhenAMultiNodeRunAsyncBodyEnds) {
  auto [ga, gate_a] = make_gated_graph();
  auto [gb, gate_b] = make_gated_graph();
  gate_a->open();
  gate_b->open();
  auto combined = decoding_task_graph::combine(std::move(ga), std::move(gb));
  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);

  auto f = combined->run_async(m);
  f.wait(); // body done, future deliberately not collected
  // Multi-node frees the graph when the worker body ends rather than when the
  // caller collects, so an uncollected future must not lock the graph out.
  // Break-it: delete scoped_release and let the captured token die with the
  // shared state instead.
  EXPECT_NO_THROW((void)combined->run(m));
  (void)f.get();
}

TEST(DecodingTaskGraph, TripwireRejectsATaggedEnqueueWhileADecodeIsInFlight) {
  // The tagged overload runs both sweeps, so it must arm the same guard the
  // untagged one does -- and arm it once, not once per sweep.
  // Break-it: drop the in_flight_guard from the tagged enqueue_syndrome.
  auto [g, gate] = make_gated_graph();
  const uint8_t byte = 0;
  std::thread first([&] { (void)g->enqueue_syndrome(&byte, 1); });
  gate->wait_until_entered();
  EXPECT_THROW((void)g->enqueue_syndrome(&byte, 1, round_type::FINAL),
               std::logic_error);
  gate->release();
  first.join();
}

TEST(DecodingTaskGraph, TripwireAllowsSequentialStreamThenOneShot) {
  // The sequential half of rule (c) is deliberately NOT enforced: three
  // in-tree callers do exactly this, two of them shipped documentation.
  // Break-it: turn the tripwire into a streaming/one-shot mode latch.
  auto graph =
      decoding_task_graph::from_spec(rep5_streaming_spec(), "single_error_lut");
  const auto n = graph->get_num_msyn_per_decode();
  std::vector<uint8_t> bytes(n, 0);
  graph->reset_decoder();
  EXPECT_TRUE(graph->enqueue_syndrome(bytes.data(), bytes.size()));
  (void)graph->get_obs_corrections();

  measurement_results m;
  m.bits.assign(n, 0);
  EXPECT_NO_THROW((void)graph->run(m));
}

TEST(DecodingTaskGraph, TripwireClearsWhenTheGuardedCallThrows) {
  // RAII, not a manual decrement at the end of the happy path -- otherwise one
  // rejected call poisons the graph forever.
  auto g = make_single_node_graph(/*device=*/-1);
  measurement_results too_short, ok;
  too_short.bits.assign(0, 0);
  ok.bits.assign(1, 0);

  EXPECT_ANY_THROW((void)g->run(too_short));
  // Break-it, verified: empty ~in_flight_guard() plus an explicit
  // guard.release() on run()'s success path -- correct when nothing throws,
  // leaks the counter when something does. Both EXPECT_NO_THROWs below go red.
  EXPECT_NO_THROW((void)g->run(ok));

  auto left = make_single_node_graph(3);
  auto right = make_single_node_graph(5);
  auto combined =
      decoding_task_graph::combine(std::move(left), std::move(right));
  measurement_results m;
  m.bits.assign(combined->get_num_msyn_per_decode(), 0);
  // Throws from cuda_device_id() AFTER the token was taken.
  EXPECT_THROW((void)combined->run_async(m), std::runtime_error);
  EXPECT_NO_THROW((void)combined->run(m));
}

// ---------------------------------------------------------------------------
// Multi-output graphs
// ---------------------------------------------------------------------------

// A window over `t` measurements, one per round, whose observable model the
// caller chooses -- so `t` is the streaming window size and `o_rows` alone
// decides the output, all_faults_decoder predicting every fault.
static task_graph_node_config
flat_probe_node(std::string id, std::size_t t,
                std::vector<std::vector<uint32_t>> o_rows) {
  std::vector<std::vector<uint32_t>> identity(t);
  for (std::size_t i = 0; i < t; ++i)
    identity[i].push_back(static_cast<uint32_t>(i));

  task_graph_node_config n;
  n.id = std::move(id);
  n.kind = "single_window";
  n.d_sparse = identity;
  n.o_sparse = std::move(o_rows);
  n.decoder_instance = std::make_unique<all_faults_decoder>(
      sparse_binary_matrix::from_nested_csr(t, t, identity), t);
  return n;
}

static task_graph_node_config
flat_probe_node(std::string id, std::vector<std::vector<uint32_t>> o_rows) {
  return flat_probe_node(std::move(id), 3, std::move(o_rows));
}

static task_graph_node_config flat_fixed_result_node(std::string id,
                                                     bool converged) {
  constexpr std::size_t t = 3;
  std::vector<std::vector<uint32_t>> identity(t);
  for (std::size_t i = 0; i < t; ++i)
    identity[i].push_back(static_cast<uint32_t>(i));

  task_graph_node_config n;
  n.id = std::move(id);
  n.kind = "single_window";
  n.d_sparse = identity;
  n.o_sparse = {{0}};
  n.decoder_instance = std::make_unique<fixed_error_decoder>(
      sparse_binary_matrix::from_nested_csr(t, t, identity),
      std::vector<float_t>{1, 0, 0}, converged);
  return n;
}

// A deterministic batch-only node. It deliberately ignores its inputs so a
// topology that cannot be lowered to per-node streaming remains a valid batch
// program, which is the contract streaming_conflict exists to preserve.
class batch_constant_node final : public task_graph_node {
public:
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return edge_value_type("logical_outcome");
    return std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot < 2 && type == edge_value_type("logical_outcome");
  }
  void execute_indexed(const std::vector<edge_value> &,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = logical_outcome{{1}, true};
  }
  std::vector<std::string> inputs() const override { return {"a", "b"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
};

TEST(DecodingTaskGraph, BatchOnlyNodeExposesNoOptionalCapabilities) {
  batch_constant_node node;
  EXPECT_EQ(node.streaming(), nullptr);
  EXPECT_EQ(node.decoder_backed(), nullptr);
}

static const bool batch_constant_registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "test_batch_constant";
  spec.requires_decoder = false;
  spec.requires_dem = false;
  register_node_kind(std::move(spec), {}, [](node_build_context &) {
    return std::make_unique<batch_constant_node>();
  });
  return true;
}();

TEST(DecodingTaskGraph, InputFromNonStreamingNodeIsBatchOnly) {
  ASSERT_TRUE(batch_constant_registered && routing_probe_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("left", {{0}}));
  nodes.push_back(flat_probe_node("right", {{1}}));
  task_graph_node_config batch;
  batch.id = "batch";
  batch.kind = "test_batch_constant";
  nodes.push_back(std::move(batch));
  nodes.push_back(routing_probe_config("probe", "edge"));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "batch", "a"},
                         {"right", "output", "batch", "b"},
                         {"batch", "output", "probe", "input"},
                         {"probe", "output", "output", "input"}});
  expect_batch_valid_streaming_conflict(std::move(graph),
                                        "input from a non-streaming node");
}

static task_graph_node_config detector_source_config(std::string id) {
  task_graph_node_config src;
  src.id = std::move(id);
  src.kind = "test_detector_source";
  src.dem = rep5_streaming_spec();
  src.node_params.insert("window_size", std::size_t{3});
  return src;
}

static task_graph_node_config relay_config(std::string id) {
  task_graph_node_config relay;
  relay.id = std::move(id);
  relay.kind = "test_relay";
  relay.dem = rep5_streaming_spec();
  relay.node_params.insert("window_size", std::size_t{3});
  return relay;
}

TEST(DecodingTaskGraph, CompileWiresRejectsDuplicateStreamingTargets) {
  ASSERT_TRUE(detector_source_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(detector_source_config("left"));
  nodes.push_back(detector_source_config("right"));
  auto consumer = rep5_streaming_node("consumer");
  consumer.node_params.insert("window_size", std::size_t{3});
  consumer.node_params.insert("step_size", std::size_t{3});
  nodes.push_back(std::move(consumer));

  try {
    (void)decoding_task_graph::from_topology(
        std::move(nodes), {{"measurements", "output", "left", "input"},
                           {"measurements", "output", "right", "input"},
                           {"left", "output", "consumer", "input"},
                           {"right", "output", "consumer", "input"},
                           {"consumer", "output", "output", "input"}});
    FAIL() << "two streaming producers reached one input slot";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "input 'input' of node 'consumer' has multiple "
                  "producers"),
              std::string::npos)
        << e.what();
  }
}

TEST(DecodingTaskGraph, CompileWiresRejectsEdgeTypeMismatch) {
  ASSERT_TRUE(detector_source_registered && relay_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(detector_source_config("source"));
  nodes.push_back(relay_config("relay"));

  try {
    (void)decoding_task_graph::from_topology(
        std::move(nodes), {{"measurements", "output", "source", "input"},
                           {"source", "output", "relay", "input"},
                           {"relay", "output", "output", "input"}});
    FAIL() << "detection_events reached a logical_outcome input";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("edge type mismatch"), std::string::npos) << msg;
    EXPECT_NE(msg.find("node 'relay' input 'input' does not accept"),
              std::string::npos)
        << msg;
    EXPECT_NE(msg.find("node 'source' output 'output' produces"),
              std::string::npos)
        << msg;
  }
}

TEST(DecodingTaskGraph, CompileWiresRejectsAnUnproducedInput) {
  ASSERT_TRUE(relay_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(relay_config("orphan"));

  try {
    (void)decoding_task_graph::from_topology(
        std::move(nodes), {{"orphan", "output", "output", "input"}});
    FAIL() << "an edge-only node had no producer";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("input 'input' of node 'orphan' has no producer"),
              std::string::npos)
        << msg;
  }
}

TEST(DecodingTaskGraph,
     ObservableWidthConflictKeepsBatchAndDefersStreamingRefusal) {
  ASSERT_TRUE(batch_constant_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("one_bit", {{0}}));
  nodes.push_back(flat_probe_node("two_bits", {{0}, {1}}));
  task_graph_node_config fold;
  fold.id = "fold";
  fold.kind = "test_batch_constant";
  nodes.push_back(std::move(fold));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "one_bit", "input"},
                         {"measurements", "output", "two_bits", "input"},
                         {"one_bit", "output", "fold", "a"},
                         {"two_bits", "output", "fold", "b"},
                         {"fold", "output", "output", "L"}});

  measurement_results shot;
  shot.bits.assign(3, 0);
  EXPECT_EQ(graph->run(shot).bits, (std::vector<uint8_t>{1}));

  const uint8_t zero = 0;
  try {
    (void)graph->enqueue_syndrome(&zero, 1);
    FAIL() << "a conflicting corrections layout entered streaming";
  } catch (const std::logic_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disagree on num_observables for output 'L'"),
              std::string::npos)
        << msg;
    EXPECT_NE(msg.find("decodes only in batch"), std::string::npos) << msg;
  }
}

TEST(DecodingTaskGraph, NonXorBatchFoldDefersStreamingRefusal) {
  ASSERT_TRUE(batch_constant_registered);
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("streaming", {{0}}));
  task_graph_node_config batch;
  batch.id = "batch";
  batch.kind = "test_batch_constant";
  nodes.push_back(std::move(batch));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "streaming", "input"},
                         {"streaming", "output", "output", "live"},
                         {"streaming", "output", "batch", "a"},
                         {"streaming", "output", "batch", "b"},
                         {"batch", "output", "output", "batch_only"}});

  measurement_results shot;
  shot.bits.assign(3, 0);
  const auto outcomes = graph->run_all(shot);
  ASSERT_EQ(outcomes.size(), 2u);
  EXPECT_EQ(outcomes[0].bits, (std::vector<uint8_t>{1}));
  EXPECT_EQ(outcomes[1].bits, (std::vector<uint8_t>{1}));

  const uint8_t zero = 0;
  try {
    (void)graph->enqueue_syndrome(&zero, 1);
    FAIL() << "a non-XOR batch fold entered streaming";
  } catch (const std::logic_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("one-shot-only node 'batch'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("decodes only in batch"), std::string::npos) << msg;
  }
}

TEST(DecodingTaskGraph, RunPropagatesFailedInnerDecoderConvergence) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_fixed_result_node("failed", false));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));

  measurement_results shot;
  shot.bits.assign(3, 0);
  const auto outcome = graph->run(shot);
  EXPECT_EQ(outcome.bits, (std::vector<uint8_t>{1}));
  EXPECT_FALSE(outcome.converged);
}

TEST(DecodingTaskGraph, XorRunPropagatesFailedBranchConvergence) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_fixed_result_node("failed", false));
  nodes.push_back(flat_fixed_result_node("succeeded", true));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "failed", "input"},
                         {"measurements", "output", "succeeded", "input"},
                         {"failed", "output", "combine", "a"},
                         {"succeeded", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});

  measurement_results shot;
  shot.bits.assign(3, 0);
  const auto outcome = graph->run(shot);
  EXPECT_EQ(outcome.bits, (std::vector<uint8_t>{0}));
  EXPECT_FALSE(outcome.converged);
}

static std::unique_ptr<decoding_task_graph> multi_output_probe_graph() {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("left", {{0}}));
  nodes.push_back(flat_probe_node("right", {{0}, {1, 2}}));
  return decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "output", "L_left"},
                         {"right", "output", "output", "L_right"}});
}

TEST(DecodingTaskGraph, MultiOutputRunAllReturnsOnePerSinkEdge) {
  // Two windows each wired straight to output.  run() must refuse to truncate
  // the program rather than answer with the first sink; run_all() returns both,
  // named by their sink edges.
  auto graph = multi_output_probe_graph();

  EXPECT_EQ(graph->output_names(),
            (std::vector<std::string>{"L_left", "L_right"}));

  measurement_results m;
  m.bits.assign(3, 0);
  EXPECT_THROW((void)graph->run(m), std::logic_error);

  const auto outcomes = graph->run_all(m);
  ASSERT_EQ(outcomes.size(), 2u);
  EXPECT_EQ(outcomes[0].bits, (std::vector<uint8_t>{1}));
  EXPECT_EQ(outcomes[1].bits, (std::vector<uint8_t>{1, 0}));
}

TEST(DecodingTaskGraph, SingleNodeMultiOutputRunAsyncRejectsSynchronously) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("only", {{0}}));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "only", "input"},
                         {"only", "output", "output", "left"},
                         {"only", "output", "output", "right"}});
  measurement_results m;
  m.bits.assign(3, 0);

  try {
    (void)graph->run_async(m);
    FAIL() << "run_async returned only one of two single-node outputs";
  } catch (const std::logic_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("2 edges reach output; run_async() returns a single "
                       "outcome"),
              std::string::npos)
        << msg;
  }

  // Validation runs before the in-flight token is acquired, so rejection must
  // not lock out the API that can represent both outputs.
  EXPECT_NO_THROW((void)graph->run_all(m));
}

TEST(DecodingTaskGraph, MultiOutputRunBatchRejectsRatherThanDroppingOutputs) {
  auto graph = multi_output_probe_graph();
  std::vector<measurement_results> shots(2);
  for (auto &shot : shots)
    shot.bits.assign(3, 0);

  try {
    (void)graph->run_batch(shots);
    FAIL() << "run_batch returned only one of two graph outputs";
  } catch (const std::logic_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("2 edges reach output; run_batch() returns a single "
                       "outcome"),
              std::string::npos)
        << msg;
    EXPECT_NE(msg.find("Call run_all() instead"), std::string::npos) << msg;
  }
}

TEST(DecodingTaskGraph, StreamingRoutesCorrectionsPerOutput) {
  // Two windows of *different* observable widths, each on its own output.
  // Asserts two things that were both broken: that the graph builds at all
  // (compile_wires demanded one shared num_observables), and that the
  // corrections are the outputs concatenated rather than folded together.
  //
  // Break-it: fold the segments into one buffer.  The widths differ, so a fold
  // cannot even be expressed -- hence the separate aggregate test below.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("left", {{0}}));          // 1 bit
  nodes.push_back(flat_probe_node("right", {{0}, {1, 2}})); // 2 bits
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "output", "L_left"},
                         {"right", "output", "output", "L_right"}});

  EXPECT_EQ(graph->get_num_observables(), 3u);

  measurement_results m;
  m.bits.assign(3, 0);
  const auto outcomes = graph->run_all(m);
  ASSERT_EQ(outcomes.size(), 2u);

  const uint8_t zero = 0;
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));
  EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1));
  const uint8_t *corr = graph->get_obs_corrections();
  ASSERT_NE(corr, nullptr);
  // Streaming and batch agree, which is what makes the layout a contract.
  EXPECT_EQ(corr[0], outcomes[0].bits[0]);
  EXPECT_EQ(corr[1], outcomes[1].bits[0]);
  EXPECT_EQ(corr[2], outcomes[1].bits[1]);
}

TEST(DecodingTaskGraph, StreamingSingleOutputAggregateUnchanged) {
  // combine()'s shape: two windows folded by an xor_node into ONE output.  The
  // xor_node is batch-only, so nothing fires it while streaming; both windows
  // XOR into the single output's segment instead.  The case per-output routing
  // must not regress.
  //
  // Break-it: give each streaming node its own segment unconditionally.  The
  // buffer widens to 2 and the fold is lost.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("left", {{0}}));     // 1
  nodes.push_back(flat_probe_node("right", {{1, 2}})); // 1 ^ 1 = 0
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "combine", "a"},
                         {"right", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});

  EXPECT_EQ(graph->get_num_observables(), 1u);

  measurement_results m;
  m.bits.assign(3, 0);
  const auto expected = graph->run(m);
  ASSERT_EQ(expected.bits.size(), 1u);

  const uint8_t zero = 0;
  graph->enqueue_syndrome(&zero, 1);
  graph->enqueue_syndrome(&zero, 1);
  EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1));
  EXPECT_EQ(graph->get_obs_corrections()[0], expected.bits[0]);
}

// ---------------------------------------------------------------------------
// Heterogeneous streaming windows
// ---------------------------------------------------------------------------

// A 3-round window and a 6-round one reading the same stream, on their own
// outputs.  Each fire flips its output's bit, so the parity of a segment is
// the parity of that node's fire count: `fast` twice, `slow` once.
static std::vector<task_graph_node_config> mixed_size_probes() {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("fast", 3, {{0}}));
  nodes.push_back(flat_probe_node("slow", 6, {{0}}));
  return nodes;
}

TEST(DecodingTaskGraph, StreamingResultDescribesPackedMultiOutputValues) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("left", {{0}, {1, 2}}));
  nodes.push_back(flat_probe_node("right", {{0}, {1}, {2}}));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "left", "input"},
                         {"measurements", "output", "right", "input"},
                         {"left", "output", "output", "L_left"},
                         {"right", "output", "output", "L_right"}});

  const auto &initial = graph->streaming_result();
  ASSERT_EQ(initial.outputs.size(), 2u);
  EXPECT_GT(initial.shot_epoch, 0u);
  EXPECT_EQ(initial.result_revision, 0u);
  EXPECT_EQ(initial.generation, 0u);
  EXPECT_EQ(initial.status, task_graph_streaming_status::pending);
  EXPECT_FALSE(initial.operation_completed);
  EXPECT_TRUE(initial.error.empty());
  ASSERT_EQ(initial.values.size(), 5u);
  EXPECT_EQ(initial.values, std::vector<uint8_t>(5, 0));
  EXPECT_EQ(initial.outputs[0].name, "L_left");
  EXPECT_EQ(initial.outputs[0].value_offset, 0u);
  EXPECT_EQ(initial.outputs[0].value_width, 2u);
  EXPECT_EQ(initial.outputs[1].name, "L_right");
  EXPECT_EQ(initial.outputs[1].value_offset, 2u);
  EXPECT_EQ(initial.outputs[1].value_width, 3u);
  for (const auto &output : initial.outputs) {
    EXPECT_EQ(output.generation, 0u);
    EXPECT_EQ(output.coverage_begin_round, 0u);
    EXPECT_EQ(output.coverage_end_round, 0u);
    EXPECT_EQ(output.status, task_graph_streaming_status::pending);
  }
  EXPECT_EQ(graph->streaming_result().result_revision, 0u);
  const uint8_t zero = 0;
  EXPECT_FALSE(graph->enqueue_syndrome(&zero, 0));
  EXPECT_EQ(graph->streaming_result().result_revision, 0u);

  std::vector<uint8_t> shot(3, 0);
  ASSERT_TRUE(graph->enqueue_syndrome(shot.data(), shot.size()));
  const auto &published = graph->streaming_result();
  EXPECT_EQ(published.result_revision, 1u);
  EXPECT_EQ(published.generation, 1u);
  EXPECT_EQ(published.values, (std::vector<uint8_t>{1, 0, 1, 1, 1}));
}

TEST(DecodingTaskGraph, OrdinaryGuardedFiresReportCommittedCoverage) {
  auto graph = make_guarded_graph();
  std::vector<uint8_t> first(graph->get_num_msyn_per_decode(), 0);
  ASSERT_TRUE(graph->enqueue_syndrome(first.data(), first.size()));
  const auto &first_result = graph->streaming_result();
  ASSERT_EQ(first_result.outputs.size(), 1u);
  EXPECT_EQ(first_result.outputs[0].generation, 1u);
  EXPECT_EQ(first_result.outputs[0].coverage_begin_round, 0u);
  EXPECT_EQ(first_result.outputs[0].coverage_end_round, 4u);

  std::vector<uint8_t> stride(3 * graph->get_num_msyn_per_round(), 0);
  ASSERT_TRUE(graph->enqueue_syndrome(stride.data(), stride.size()));
  const auto &second_result = graph->streaming_result();
  EXPECT_EQ(second_result.outputs[0].generation, 2u);
  EXPECT_EQ(second_result.outputs[0].coverage_begin_round, 4u);
  EXPECT_EQ(second_result.outputs[0].coverage_end_round, 7u);
}

TEST(DecodingTaskGraph, StreamingResultTracksIndependentOutputs) {
  auto graph = decoding_task_graph::from_topology(
      mixed_size_probes(), {{"measurements", "output", "fast", "input"},
                            {"measurements", "output", "slow", "input"},
                            {"fast", "output", "output", "L_fast"},
                            {"slow", "output", "output", "L_slow"}});
  const uint8_t zero = 0;

  for (int i = 0; i < 3; ++i)
    graph->enqueue_syndrome(&zero, 1);
  const auto &half = graph->streaming_result();
  ASSERT_EQ(half.outputs.size(), 2u);
  EXPECT_EQ(half.outputs[0].name, "L_fast");
  EXPECT_EQ(half.outputs[0].status, task_graph_streaming_status::available);
  EXPECT_EQ(half.outputs[0].generation, 1u);
  EXPECT_EQ(half.outputs[0].coverage_begin_round, 0u);
  EXPECT_EQ(half.outputs[0].coverage_end_round, 3u);
  EXPECT_EQ(half.outputs[1].name, "L_slow");
  EXPECT_EQ(half.outputs[1].status, task_graph_streaming_status::pending);
  EXPECT_EQ(half.generation, 0u);
  EXPECT_EQ(half.result_revision, 1u);
  EXPECT_FALSE(half.operation_completed);
  ASSERT_EQ(half.values.size(), 2u);
  EXPECT_EQ(half.values[0], 1u);
  EXPECT_EQ(half.values[1], 0u);

  for (int i = 0; i < 3; ++i)
    graph->enqueue_syndrome(&zero, 1);
  // A held C++ result is a live view. Metadata and values must advance
  // together even before streaming_result() is called again.
  EXPECT_EQ(half.outputs[0].generation, 2u);
  EXPECT_EQ(half.outputs[1].generation, 1u);
  EXPECT_EQ(half.generation, 1u);
  EXPECT_EQ(half.values, (std::vector<uint8_t>{0, 1}));
  const auto &complete_generation = graph->streaming_result();
  EXPECT_EQ(complete_generation.outputs[0].generation, 2u);
  EXPECT_EQ(complete_generation.outputs[0].coverage_begin_round, 3u);
  EXPECT_EQ(complete_generation.outputs[0].coverage_end_round, 6u);
  EXPECT_EQ(complete_generation.outputs[1].generation, 1u);
  EXPECT_EQ(complete_generation.outputs[1].coverage_begin_round, 0u);
  EXPECT_EQ(complete_generation.outputs[1].coverage_end_round, 6u);
  EXPECT_EQ(complete_generation.generation, 1u);
  EXPECT_EQ(complete_generation.result_revision, 2u);
  EXPECT_TRUE(complete_generation.operation_completed);
  EXPECT_EQ(complete_generation.status, task_graph_streaming_status::available);
}

TEST(DecodingTaskGraph,
     AsyncStreamingResultTracksIndependentOutputsAndFinalState) {
  auto graph = decoding_task_graph::from_topology(
      mixed_size_probes(),
      {{"measurements", "output", "fast", "input"},
       {"measurements", "output", "slow", "input"},
       {"fast", "output", "output", "L_fast"},
       {"slow", "output", "output", "L_slow"}},
      {.async_streaming = true, .async_queue_capacity = 6});

  const uint8_t zero = 0;
  for (int i = 0; i < 3; ++i)
    EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));

  const auto snapshot = wait_for_streaming_result(
      *graph, [](const task_graph_streaming_result &result) {
        return result.result_revision >= 1;
      });
  ASSERT_EQ(snapshot.outputs.size(), 2u);
  EXPECT_EQ(snapshot.outputs[0].name, "L_fast");
  EXPECT_EQ(snapshot.outputs[0].generation, 1u);
  EXPECT_EQ(snapshot.outputs[0].status, task_graph_streaming_status::available);
  EXPECT_EQ(snapshot.outputs[1].name, "L_slow");
  EXPECT_EQ(snapshot.outputs[1].generation, 0u);
  EXPECT_EQ(snapshot.outputs[1].status, task_graph_streaming_status::pending);
  EXPECT_EQ(snapshot.generation, 0u);
  EXPECT_EQ(snapshot.status, task_graph_streaming_status::pending);
  EXPECT_EQ(snapshot.result_revision, 1u);
  EXPECT_FALSE(snapshot.operation_completed);
  EXPECT_EQ(snapshot.values, (std::vector<uint8_t>{1, 0}));

  for (int i = 0; i < 2; ++i)
    EXPECT_FALSE(graph->enqueue_syndrome(&zero, 1));
  EXPECT_TRUE(graph->enqueue_syndrome(&zero, 1, round_type::FINAL));

  const task_graph_streaming_result finished = graph->streaming_result();
  EXPECT_EQ(finished.outputs[0].generation, 2u);
  EXPECT_EQ(finished.outputs[0].status, task_graph_streaming_status::complete);
  EXPECT_EQ(finished.outputs[1].generation, 1u);
  EXPECT_EQ(finished.outputs[1].status, task_graph_streaming_status::complete);
  EXPECT_EQ(finished.generation, 1u);
  EXPECT_EQ(finished.status, task_graph_streaming_status::complete);
  EXPECT_EQ(finished.result_revision, 3u);
  EXPECT_TRUE(finished.operation_completed);
  EXPECT_EQ(finished.values, (std::vector<uint8_t>{0, 1}));
  EXPECT_EQ(finished.shot_epoch, snapshot.shot_epoch);
  ASSERT_EQ(finished.outputs.size(), snapshot.outputs.size());
  for (std::size_t i = 0; i < finished.outputs.size(); ++i)
    EXPECT_EQ(finished.outputs[i].name, snapshot.outputs[i].name);
}

TEST(DecodingTaskGraph, StreamingResultPublishesAParityCancelledOutput) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("source", 3, {{0}}));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "source", "input"},
                         {"source", "output", "combine", "a"},
                         {"source", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});
  std::vector<uint8_t> shot(3, 0);

  ASSERT_TRUE(graph->enqueue_syndrome(shot.data(), shot.size()));
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_EQ(result.outputs[0].generation, 1u);
  EXPECT_EQ(result.outputs[0].status, task_graph_streaming_status::available);
  EXPECT_EQ(result.generation, 1u);
  ASSERT_EQ(result.values.size(), 1u);
  EXPECT_EQ(result.values[0], 0u);
}

TEST(DecodingTaskGraph,
     ParityCancelledContributorDoesNotConstrainScheduleAlignment) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("cancelled", 3, {{0}}));
  nodes.push_back(flat_probe_node("live", 6, {{0}}));
  task_graph_node_config cancel;
  cancel.id = "cancel";
  cancel.kind = "xor";
  nodes.push_back(std::move(cancel));
  task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";
  nodes.push_back(std::move(combine));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "cancelled", "input"},
                         {"measurements", "output", "live", "input"},
                         {"cancelled", "output", "cancel", "a"},
                         {"cancelled", "output", "cancel", "b"},
                         {"cancel", "output", "combine", "a"},
                         {"live", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});

  ASSERT_TRUE(graph->supports_streaming())
      << graph->streaming_incompatibility_reason();
  std::vector<uint8_t> shot(6, 0);
  ASSERT_TRUE(graph->enqueue_syndrome(shot.data(), shot.size()));
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_EQ(result.outputs[0].generation, 1u);
  EXPECT_EQ(result.outputs[0].coverage_begin_round, 0u);
  EXPECT_EQ(result.outputs[0].coverage_end_round, 6u);
  EXPECT_EQ(result.status, task_graph_streaming_status::available);
  EXPECT_EQ(result.values, (std::vector<uint8_t>{1}));
}

TEST(DecodingTaskGraph, ClearConsumesResultAndResetStartsANewEpoch) {
  auto graph = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  std::vector<uint8_t> shot(graph->get_num_msyn_per_decode(), 0);

  EXPECT_TRUE(
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  const auto &final = graph->streaming_result();
  ASSERT_EQ(final.outputs.size(), 1u);
  EXPECT_EQ(final.outputs[0].status, task_graph_streaming_status::complete);
  EXPECT_EQ(final.status, task_graph_streaming_status::complete);
  EXPECT_EQ(final.outputs[0].generation, 1u);
  EXPECT_EQ(final.outputs[0].coverage_begin_round, 0u);
  EXPECT_EQ(final.outputs[0].coverage_end_round, 5u);
  EXPECT_EQ(final.values[0], 0u);

  const auto epoch = final.shot_epoch;
  const auto final_revision = final.result_revision;
  graph->clear_corrections();
  const auto &cleared = graph->streaming_result();
  EXPECT_EQ(cleared.shot_epoch, epoch);
  EXPECT_EQ(cleared.result_revision, final_revision + 1);
  EXPECT_EQ(cleared.status, task_graph_streaming_status::pending);
  EXPECT_EQ(cleared.outputs[0].generation, 1u);
  EXPECT_EQ(cleared.outputs[0].status, task_graph_streaming_status::pending);
  EXPECT_EQ(cleared.values[0], 0u);

  graph->reset_decoder();
  const auto &reset = graph->streaming_result();
  EXPECT_EQ(reset.shot_epoch, epoch + 1);
  EXPECT_EQ(reset.result_revision, 0u);
  EXPECT_EQ(reset.generation, 0u);
  EXPECT_EQ(reset.outputs[0].generation, 0u);
}

TEST(DecodingTaskGraph, StreamingResultLatchesOperationalFailure) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(routing_probe_config("probe", "throwing"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "probe", "input"},
                         {"probe", "output", "output", "input"}});
  const uint8_t zero = 0;

  EXPECT_THROW(graph->enqueue_syndrome(&zero, 1), std::runtime_error);
  const auto &failed = graph->streaming_result();
  EXPECT_EQ(failed.status, task_graph_streaming_status::failed);
  EXPECT_FALSE(failed.operation_completed);
  EXPECT_NE(failed.error.find("test streaming failure"), std::string::npos);
  EXPECT_EQ(failed.result_revision, 1u);
  ASSERT_EQ(failed.outputs.size(), 1u);
  EXPECT_EQ(failed.outputs[0].status, task_graph_streaming_status::failed);
  EXPECT_THROW(graph->enqueue_syndrome(&zero, 1), std::logic_error);
  EXPECT_EQ(graph->streaming_result().status,
            task_graph_streaming_status::failed);

  const auto failed_epoch = failed.shot_epoch;
  const auto failed_revision = failed.result_revision;
  graph->clear_corrections();
  const auto &consumed_failure = graph->streaming_result();
  EXPECT_EQ(consumed_failure.status, task_graph_streaming_status::failed);
  EXPECT_EQ(consumed_failure.result_revision, failed_revision + 1);
  EXPECT_FALSE(consumed_failure.error.empty());

  graph->reset_decoder();
  const auto &reset = graph->streaming_result();
  EXPECT_EQ(reset.shot_epoch, failed_epoch + 1);
  EXPECT_EQ(reset.result_revision, 0u);
  EXPECT_EQ(reset.status, task_graph_streaming_status::pending);
  EXPECT_TRUE(reset.error.empty());
  EXPECT_EQ(reset.outputs[0].status, task_graph_streaming_status::pending);
}

TEST(DecodingTaskGraph, FinalOnTheLastRoundPublishesACompleteGeneration) {
  auto graph = decoding_task_graph::from_spec(rep5_spec(5), "single_error_lut");
  std::vector<uint8_t> shot(graph->get_num_msyn_per_decode(), 0);
  EXPECT_FALSE(graph->enqueue_syndrome(shot.data(), shot.size() - 1));

  EXPECT_TRUE(graph->enqueue_syndrome(&shot.back(), 1, round_type::FINAL));
  const auto &result = graph->streaming_result();
  EXPECT_EQ(result.generation, 1u);
  EXPECT_TRUE(result.operation_completed);
  EXPECT_EQ(result.status, task_graph_streaming_status::complete);
}

TEST(DecodingTaskGraph, FinalKeepsNeverPublishedOutputsPending) {
  auto graph = decoding_task_graph::from_topology(
      mixed_size_probes(), {{"measurements", "output", "fast", "input"},
                            {"measurements", "output", "slow", "input"},
                            {"fast", "output", "output", "L_fast"},
                            {"slow", "output", "output", "L_slow"}});
  std::vector<uint8_t> first_window(3, 0);
  EXPECT_TRUE(graph->enqueue_syndrome(first_window.data(), first_window.size(),
                                      round_type::FINAL));
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_EQ(result.outputs[0].status, task_graph_streaming_status::complete);
  EXPECT_EQ(result.outputs[1].status, task_graph_streaming_status::pending);
  EXPECT_EQ(result.outputs[1].generation, 0u);
  EXPECT_EQ(result.status, task_graph_streaming_status::pending);
  EXPECT_EQ(result.generation, 0u);
  EXPECT_FALSE(result.operation_completed);
}

TEST(DecodingTaskGraph, FinalCoverageUsesTheLastGuardedWindowTail) {
  auto graph = make_guarded_graph();
  std::vector<uint8_t> first(graph->get_num_msyn_per_decode(), 0);
  ASSERT_TRUE(graph->enqueue_syndrome(first.data(), first.size()));
  std::vector<uint8_t> stride(3 * graph->get_num_msyn_per_round(), 0);

  ASSERT_TRUE(
      graph->enqueue_syndrome(stride.data(), stride.size(), round_type::FINAL));
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_EQ(result.outputs[0].generation, 3u);
  EXPECT_EQ(result.outputs[0].coverage_begin_round, 7u);
  EXPECT_EQ(result.outputs[0].coverage_end_round, 8u);
  EXPECT_EQ(result.outputs[0].status, task_graph_streaming_status::complete);

  graph->clear_corrections();
  ASSERT_TRUE(graph->enqueue_syndrome(stride.data(), stride.size()));
  const auto &resumed = graph->streaming_result();
  EXPECT_EQ(resumed.outputs[0].generation, 4u);
  EXPECT_EQ(resumed.outputs[0].coverage_begin_round, 7u);
  EXPECT_EQ(resumed.outputs[0].coverage_end_round, 10u);
}

TEST(DecodingTaskGraph, EmptyGuardedTailStillAdvancesPublication) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(routing_probe_config("probe", "guarded_noop_tail"));
  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "probe", "input"},
                         {"probe", "output", "output", "input"}});
  std::vector<uint8_t> shot(3, 0);

  ASSERT_TRUE(
      graph->enqueue_syndrome(shot.data(), shot.size(), round_type::FINAL));
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_EQ(result.outputs[0].generation, 2u);
  EXPECT_EQ(result.result_revision, 2u);
  EXPECT_EQ(result.outputs[0].coverage_begin_round, 2u);
  EXPECT_EQ(result.outputs[0].coverage_end_round, 3u);
  EXPECT_EQ(result.outputs[0].status, task_graph_streaming_status::complete);
  EXPECT_EQ(result.values[0], 0u);
}

TEST(DecodingTaskGraph, HeterogeneousWindowsFireOnTheirOwnSchedules) {
  // compile_wires() made every raw-input node agree on capacity, so this graph
  // did not build at all.
  //
  // Break-it: restore the agreement throw and from_topology() rejects this.
  auto graph = decoding_task_graph::from_topology(
      mixed_size_probes(), {{"measurements", "output", "fast", "input"},
                            {"measurements", "output", "slow", "input"},
                            {"fast", "output", "output", "L_fast"},
                            {"slow", "output", "output", "L_slow"}});

  // The largest window: what the slowest node needs before its first fire.
  EXPECT_EQ(graph->get_num_msyn_per_decode(), 6u);
  EXPECT_EQ(graph->get_num_msyn_per_round(), 1u);

  const uint8_t zero = 0;
  std::vector<bool> fired;
  for (int i = 0; i < 6; ++i)
    fired.push_back(graph->enqueue_syndrome(&zero, 1));
  EXPECT_EQ(fired, (std::vector<bool>{false, false, true, false, false, true}));

  const uint8_t *corr = graph->get_obs_corrections();
  ASSERT_NE(corr, nullptr);
  EXPECT_EQ(corr[0], 0); // fast fired twice
  EXPECT_EQ(corr[1], 1); // slow fired once
}

TEST(DecodingTaskGraph, OneMeteredCallDrivesBothWindowSchedules) {
  // The whole stream in a single call has to land the same fires as one byte at
  // a time, which is step 5a's metering holding up when the nodes it meters to
  // have different room.
  auto graph = decoding_task_graph::from_topology(
      mixed_size_probes(), {{"measurements", "output", "fast", "input"},
                            {"measurements", "output", "slow", "input"},
                            {"fast", "output", "output", "L_fast"},
                            {"slow", "output", "output", "L_slow"}});

  const std::vector<uint8_t> whole(6, 0);
  EXPECT_TRUE(graph->enqueue_syndrome(whole.data(), whole.size()));
  const uint8_t *corr = graph->get_obs_corrections();
  ASSERT_NE(corr, nullptr);
  EXPECT_EQ(corr[0], 0);
  EXPECT_EQ(corr[1], 1);
  const auto &result = graph->streaming_result();
  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_EQ(result.outputs[0].generation, 2u);
  EXPECT_EQ(result.outputs[0].coverage_begin_round, 3u);
  EXPECT_EQ(result.outputs[0].coverage_end_round, 6u);
  EXPECT_EQ(result.outputs[1].generation, 1u);
  EXPECT_EQ(result.outputs[1].coverage_begin_round, 0u);
  EXPECT_EQ(result.outputs[1].coverage_end_round, 6u);
  EXPECT_EQ(result.generation, 1u);
}

TEST(DecodingTaskGraph, HeterogeneousRealDemsStreamRatherThanDefer) {
  // Two real DEMs of different round counts on their own outputs.  This is the
  // shape that used to construct and then refuse to stream -- window sizes
  // disagreed -- so the graph was a batch program only.  It now streams, each
  // window metered to its own room.  The flat fixtures above pin the schedule;
  // this pins that a DEM-driven graph reaches the same place.
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config a;
  a.id = "a";
  a.kind = "single_window";
  a.dem = rep5_spec(5);
  a.decoder_type = "single_error_lut";
  task_graph_node_config b;
  b.id = "b";
  b.kind = "single_window";
  b.dem = rep5_spec(7);
  b.decoder_type = "single_error_lut";
  nodes.push_back(std::move(a));
  nodes.push_back(std::move(b));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "a", "input"},
                         {"measurements", "output", "b", "input"},
                         {"a", "output", "output", "La"},
                         {"b", "output", "output", "Lb"}});

  // The larger DEM's window, so feeding this many bytes fires both.
  const auto n = graph->get_num_msyn_per_decode();
  ASSERT_GT(n, 0u);
  const std::vector<uint8_t> shot(n, 0);
  EXPECT_NO_THROW(EXPECT_TRUE(graph->enqueue_syndrome(shot.data(), n)));
  EXPECT_NE(graph->get_obs_corrections(), nullptr);
}

TEST(DecodingTaskGraph, ASolvedOutputIsReadableBeforeTheShotEnds) {
  // get_obs_corrections() is a current-value read: an output is published as
  // soon as its node fires, not at end of shot.  Nothing asserted that, and it
  // is the whole of what a latency-sensitive caller polling mid-stream gets.
  auto graph = decoding_task_graph::from_topology(
      mixed_size_probes(), {{"measurements", "output", "fast", "input"},
                            {"measurements", "output", "slow", "input"},
                            {"fast", "output", "output", "L_fast"},
                            {"slow", "output", "output", "L_slow"}});

  const uint8_t zero = 0;
  for (int i = 0; i < 3; ++i)
    graph->enqueue_syndrome(&zero, 1);

  // `fast` has fired once, `slow` not at all -- and an output whose node has
  // not fired reads as zero rather than as garbage.
  const uint8_t *mid = graph->get_obs_corrections();
  ASSERT_NE(mid, nullptr);
  EXPECT_EQ(mid[0], 1);
  EXPECT_EQ(mid[1], 0);

  // Reading twice answers twice, so polling is free of side effects.
  //
  // Break-it: drop the std::fill in get_obs_corrections() and the second read
  // XORs each segment back to zero.
  const uint8_t *again = graph->get_obs_corrections();
  EXPECT_EQ(again[0], 1);
  EXPECT_EQ(again[1], 0);

  for (int i = 0; i < 3; ++i)
    graph->enqueue_syndrome(&zero, 1);
  const uint8_t *end = graph->get_obs_corrections();
  EXPECT_EQ(end[0], 0); // fast fired twice
  EXPECT_EQ(end[1], 1); // slow fired once
}

TEST(DecodingTaskGraph, FoldedWindowSizeMismatchIsStreamingOnly) {
  // The same two sizes, but folded by a batch-only xor into ONE output.
  // Nothing fires the xor while streaming, so the windows XOR into its segment
  // instead, and that only equals the fold while both sit at the same point in
  // the stream -- which windows of different sizes never do. Batch execution
  // still computes both complete branches before executing the xor.
  auto nodes = mixed_size_probes();
  task_graph_node_config combiner;
  combiner.id = "combine";
  combiner.kind = "xor";
  nodes.push_back(std::move(combiner));

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "fast", "input"},
                         {"measurements", "output", "slow", "input"},
                         {"fast", "output", "combine", "a"},
                         {"slow", "output", "combine", "b"},
                         {"combine", "output", "output", "input"}});

  std::vector<task_graph_node_config> fast_nodes;
  fast_nodes.push_back(flat_probe_node("fast_reference", 3, {{0}}));
  auto fast_reference =
      decoding_task_graph::from_topology(std::move(fast_nodes));
  std::vector<task_graph_node_config> slow_nodes;
  slow_nodes.push_back(flat_probe_node("slow_reference", 6, {{0}}));
  auto slow_reference =
      decoding_task_graph::from_topology(std::move(slow_nodes));
  measurement_results shot;
  shot.bits = {1, 0, 1, 0, 1, 0};
  const auto fast = fast_reference->run(shot);
  const auto slow = slow_reference->run(shot);
  ASSERT_EQ(fast.bits, (std::vector<uint8_t>{1}));
  ASSERT_EQ(slow.bits, (std::vector<uint8_t>{1}));
  const auto expected = xor_logical(fast, slow);
  const auto actual = graph->run(shot);
  EXPECT_EQ(actual.bits, expected.bits);
  EXPECT_EQ(actual.converged, expected.converged);

  expect_batch_valid_streaming_conflict(std::move(graph), "window_size");
}

TEST(DecodingTaskGraph, RoundSizeMismatchIsStreamingOnly) {
  // Round size is the one thing metering cannot paper over: there is a single
  // byte stream, and get_num_msyn_per_round() would answer for one node only.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(flat_probe_node("ones", 6, {{0}}));
  nodes.push_back(flat_probe_node("twos", 6, {{0}}));
  nodes[1].node_params.insert("num_syndromes_per_round", std::size_t{2});

  auto graph = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "ones", "input"},
                         {"measurements", "output", "twos", "input"},
                         {"ones", "output", "output", "L_ones"},
                         {"twos", "output", "output", "L_twos"}});
  expect_batch_valid_streaming_conflict(std::move(graph), "num_msyn_per_round");
}

} // namespace cudaq::qec
