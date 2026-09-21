/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <atomic>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>

TEST(PyMatchingDecoder, checkRegularEdges) {
  using cudaq::qec::float_t;

  std::size_t block_size = 2;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  // clang-format off
  std::vector<uint8_t> H_vec = {1, 0,
                                1, 1,
                                0, 1};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);

  // Activate error in column 0 and verify that the error is detected.
  std::vector<float_t> syndrome = {1, 1, 0};
  auto result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);

  // Activate error in column 1 and verify that the error is detected.
  syndrome = {0, 1, 1};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 1.0);

  // Activate errors in columns 0 and 1 and verify that the errors are detected.
  syndrome = {1, 0, 1};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 1.0);
}

TEST(PyMatchingDecoder, checkBoundaryEdges) {
  using cudaq::qec::float_t;

  std::size_t block_size = 3;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  // clang-format off
  std::vector<uint8_t> H_vec = {1, 0, 0,
                                0, 1, 0,
                                0, 0, 1};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);

  // Activate error in column 0 and verify that the error is detected.
  std::vector<float_t> syndrome = {1, 0, 0};
  auto result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);

  // Activate error in column 1 and verify that the error is detected.
  syndrome = {0, 1, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 1.0);
  EXPECT_EQ(result.result[2], 0.0);

  // Activate error in column 2 and verify that the error is detected.
  syndrome = {0, 0, 1};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 1.0);

  syndrome = {0.5, 0, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);
}

TEST(PyMatchingDecoder, rejectsDuplicateSparseInput) {
  using index_type = cudaq::qec::sparse_binary_matrix::index_type;

  // Duplicate sparse entries are not allowed: callers that want GF(2)
  // duplicate-collapse semantics must canonicalize before constructing the
  // decoder.
  std::vector<std::vector<index_type>> nested = {{0, 0, 0}, {0, 1}, {1}};
  auto H = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      /*num_rows=*/2, /*num_cols=*/3, nested);

  cudaqx::heterogeneous_map custom_args;
  EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", H, custom_args),
               std::invalid_argument);
}

TEST(PyMatchingDecoder, preservesCallerColumnOrderUnderNonCanonicalOrdering) {
  using cudaq::qec::float_t;

  std::size_t block_size = 4;
  std::size_t syndrome_size = 4;
  cudaqx::heterogeneous_map custom_args;

  // Column c is a boundary edge at row: col0->0, col1->1, col2->3, col3->2.
  // Topological order by row content is [0,1,3,2], i.e. columns 2 and 3 swap.
  // clang-format off
  std::vector<uint8_t> H_vec = {1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 0, 1,
                                0, 0, 1, 0};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);

  // Detector at row 3 is only touched by column 2. If columns were reordered to
  // [0,1,3,2], this would wrongly light up column 3.
  std::vector<float_t> syndrome = {0, 0, 0, 1};
  auto result = d->decode(syndrome);
  ASSERT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 4u);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 1.0);
  EXPECT_EQ(result.result[3], 0.0);

  // Detector at row 2 is only touched by column 3.
  syndrome = {0, 0, 1, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);
  EXPECT_EQ(result.result[3], 1.0);

  // Columns 0 and 1 are already in canonical position; verify they're
  // unaffected.
  syndrome = {1, 0, 0, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);
  EXPECT_EQ(result.result[3], 0.0);
}

TEST(PyMatchingDecoder, AcceptsAllMergeStrategiesAndRejectsUnknown) {
  cudaqx::tensor<uint8_t> H;
  std::vector<uint8_t> H_vec = {1};
  H.copy(H_vec.data(), {1, 1});

  for (const std::string &strategy :
       {"disallow", "independent", "smallest_weight", "keep_original",
        "replace"}) {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", strategy);
    std::unique_ptr<cudaq::qec::decoder> d;
    if (strategy == "disallow") {
      d = cudaq::qec::decoder::get("pymatching", H, params);
    } else {
      auto sparse_H = cudaq::qec::sparse_binary_matrix(H);
      auto O = cudaq::qec::sparse_binary_matrix::from_nested_csr(1, 1, {{0}});
      d = cudaq::qec::decoder::get(
          "pymatching",
          cudaq::qec::decoder_init(std::move(sparse_H), std::move(O)),
          cudaq::qec::decode_result_type::observables, params);
    }
    ASSERT_NE(d, nullptr) << strategy;
    auto result = d->decode(std::vector<cudaq::qec::float_t>{1.0});
    ASSERT_TRUE(result.converged) << strategy;
    ASSERT_EQ(result.result.size(), 1u) << strategy;
  }

  cudaqx::heterogeneous_map params;
  params.insert("merge_strategy", std::string("not_a_strategy"));
  EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", H, params),
               std::runtime_error);
}

// Parallel columns share one matching edge, so an error frame has to name one
// of them. The column named is the one whose parameters the graph actually
// holds after the merge: KEEP_ORIGINAL and INDEPENDENT retain the first
// column's observables, REPLACE adopts the last, SMALLEST_WEIGHT adopts
// whichever weight is smaller. Baseline main named the last column for every
// strategy, which contradicts the retained edge for the first two.
TEST(PyMatchingDecoder,
     ConstructionInputErrorOutputTracksMergedParallelEdgeColumn) {
  cudaqx::tensor<uint8_t> H;
  const std::vector<uint8_t> H_vec = {1, 1};
  H.copy(H_vec.data(), {1, 2});

  auto decode_with = [&](const std::string &strategy) {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", strategy);
    auto O = cudaq::qec::sparse_binary_matrix::from_csr(
        0, 2, std::vector<std::uint32_t>{0}, {});
    auto inputs = cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H),
                                           std::move(O), {0.1, 0.2});
    auto decoder = cudaq::qec::decoder::get(
        "pymatching", std::move(inputs), cudaq::qec::decode_result_type::errors,
        params);
    return decoder->decode(std::vector<cudaq::qec::float_t>{1.0}).result;
  };

  EXPECT_EQ(decode_with("keep_original"),
            (std::vector<cudaq::qec::float_t>{1.0, 0.0}));
  EXPECT_EQ(decode_with("independent"),
            (std::vector<cudaq::qec::float_t>{1.0, 0.0}));
  EXPECT_EQ(decode_with("replace"),
            (std::vector<cudaq::qec::float_t>{0.0, 1.0}));
  EXPECT_EQ(decode_with("smallest_weight"),
            (std::vector<cudaq::qec::float_t>{0.0, 1.0}));
  EXPECT_THROW((void)decode_with("disallow"), std::invalid_argument);
}

TEST(PyMatchingDecoder, RejectsObservableMatrixWithWrongBlockSize) {
  cudaqx::tensor<uint8_t> H;
  std::vector<uint8_t> H_vec = {1, 0, 0, 1};
  H.copy(H_vec.data(), {2, 2});

  cudaqx::tensor<uint8_t> O({1, 3});
  O.at({0, 0}) = 1;
  EXPECT_THROW(
      (void)cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H),
                                     cudaq::qec::sparse_binary_matrix(O)),
      std::invalid_argument);
}

TEST(PyMatchingDecoder, ObservableModelDoesNotSelectOutputBasis) {
  cudaqx::tensor<uint8_t> H;
  const std::vector<uint8_t> H_vec = {1, 0, 0, 1};
  H.copy(H_vec.data(), {2, 2});

  cudaqx::tensor<uint8_t> O;
  const std::vector<uint8_t> O_vec = {1, 1};
  O.copy(O_vec.data(), {1, 2});

  auto inputs = cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H),
                                         cudaq::qec::sparse_binary_matrix(O),
                                         std::vector<double>{0.1, 0.1});

  auto error_decoder = cudaq::qec::decoder::get("pymatching", inputs);
  EXPECT_EQ(error_decoder->get_result_type(),
            cudaq::qec::decode_result_type::errors);
  const auto error_result = error_decoder->decode({1.0, 0.0});
  ASSERT_TRUE(error_result.converged);
  EXPECT_EQ(error_result.result, (std::vector<cudaq::qec::float_t>{1.0, 0.0}));

  auto observable_decoder = cudaq::qec::decoder::get(
      "pymatching", inputs, cudaq::qec::decode_result_type::observables);
  EXPECT_EQ(observable_decoder->get_result_type(),
            cudaq::qec::decode_result_type::observables);
  const auto observable_result = observable_decoder->decode({1.0, 0.0});
  ASSERT_TRUE(observable_result.converged);
  EXPECT_EQ(observable_result.result, (std::vector<cudaq::qec::float_t>{1.0}));

  EXPECT_THROW(
      (void)cudaq::qec::decoder::get(
          "pymatching", inputs,
          cudaq::qec::decode_result_type::observables_and_residual_detectors),
      std::invalid_argument);
}

TEST(PyMatchingDecoder, ObservableModelDoesNotChangeErrorMergeDefault) {
  cudaqx::tensor<uint8_t> H;
  const std::vector<uint8_t> H_vec = {1, 1};
  H.copy(H_vec.data(), {1, 2});

  cudaqx::tensor<uint8_t> O;
  const std::vector<uint8_t> O_vec = {1, 0};
  O.copy(O_vec.data(), {1, 2});
  auto inputs = cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H),
                                         cudaq::qec::sparse_binary_matrix(O),
                                         std::vector<double>{0.1, 0.2});

  // Parallel edges remain disallowed for error output even when O is model
  // data. Only an explicit observable-output request selects the independent
  // merge default used for detector-error models.
  EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", inputs),
               std::invalid_argument);
  EXPECT_THROW(
      (void)cudaq::qec::decoder::get("pymatching", inputs,
                                     cudaq::qec::decode_result_type::errors),
      std::invalid_argument);
  EXPECT_NO_THROW((void)cudaq::qec::decoder::get(
      "pymatching", inputs, cudaq::qec::decode_result_type::observables));
}

// Regression test: when two H columns share the same edge (parallel columns),
// edge2col_idx must record the column that the graph actually retains after the
// merge, not always the last one seen. Under KEEP_ORIGINAL / INDEPENDENT the
// first column wins; under SMALLEST_WEIGHT the lower-weight column wins; under
// REPLACE the last column wins.
TEST(PyMatchingDecoder, ErrorOutputTracksMergedParallelEdgeColumn) {
  using cudaq::qec::float_t;

  // Cols 0 and 1 are parallel — both connect detectors {row0, row1}.
  // Col 2 connects {row1, row2}.
  // clang-format off
  std::vector<uint8_t> H_vec = {1, 1, 0,
                                1, 1, 1,
                                0, 0, 1};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {3, 3});

  // KEEP_ORIGINAL retains the first column. INDEPENDENT's decoded edge
  // representation cannot distinguish parallel caller columns, so CUDA-QX
  // deterministically attributes it to the first column. Both name col 0.
  for (const std::string &strategy : {"keep_original", "independent"}) {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", strategy);
    auto d = cudaq::qec::decoder::get("pymatching", H, params);
    ASSERT_NE(d, nullptr) << strategy;

    std::vector<float_t> syndrome = {1.0, 1.0, 0.0};
    auto result = d->decode(syndrome);
    ASSERT_TRUE(result.converged) << strategy;
    ASSERT_EQ(result.result.size(), 3u) << strategy;
    EXPECT_EQ(result.result[0], 1.0) << strategy << ": col 0 must be flagged";
    EXPECT_EQ(result.result[1], 0.0)
        << strategy << ": col 1 must not be flagged";
    EXPECT_EQ(result.result[2], 0.0) << strategy;
  }

  // DISALLOW rejects a second parallel edge during graph construction.
  {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", std::string("disallow"));
    EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", H, params),
                 std::invalid_argument);
  }

  // Under SMALLEST_WEIGHT the lower-weight parallel column wins. The higher
  // error rate for col 1 gives it the lower matching weight.
  {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", std::string("smallest_weight"));
    auto inputs = cudaq::qec::decoder_init(
        cudaq::qec::sparse_binary_matrix(H), std::nullopt,
        std::vector<double>{0.01, 0.1, 0.01});
    auto d = cudaq::qec::decoder::get("pymatching", std::move(inputs),
                                      cudaq::qec::decode_result_type::errors,
                                      params);
    ASSERT_NE(d, nullptr);

    std::vector<float_t> syndrome = {1.0, 1.0, 0.0};
    auto result = d->decode(syndrome);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.result.size(), 3u);
    EXPECT_EQ(result.result[0], 0.0)
        << "col 0 must not be flagged under smallest_weight";
    EXPECT_EQ(result.result[1], 1.0)
        << "col 1 must be flagged under smallest_weight";
    EXPECT_EQ(result.result[2], 0.0);
  }

  // With equal weights, SMALLEST_WEIGHT keeps the existing edge, so the first
  // parallel column remains the result column.
  {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", std::string("smallest_weight"));
    auto d = cudaq::qec::decoder::get("pymatching", H, params);
    ASSERT_NE(d, nullptr);

    std::vector<float_t> syndrome = {1.0, 1.0, 0.0};
    auto result = d->decode(syndrome);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.result.size(), 3u);
    EXPECT_EQ(result.result[0], 1.0)
        << "col 0 must be flagged on a smallest_weight tie";
    EXPECT_EQ(result.result[1], 0.0)
        << "col 1 must not be flagged on a smallest_weight tie";
    EXPECT_EQ(result.result[2], 0.0);
  }

  // Under REPLACE the graph adopts the last column's edge — result must name
  // col 1.
  {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", std::string("replace"));
    auto d = cudaq::qec::decoder::get("pymatching", H, params);
    ASSERT_NE(d, nullptr);

    std::vector<float_t> syndrome = {1.0, 1.0, 0.0};
    auto result = d->decode(syndrome);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.result.size(), 3u);
    EXPECT_EQ(result.result[0], 0.0)
        << "col 0 must not be flagged under replace";
    EXPECT_EQ(result.result[1], 1.0) << "col 1 must be flagged under replace";
    EXPECT_EQ(result.result[2], 0.0);
  }
}

TEST(PyMatchingDecoder, DecodesHighObservableIndicesAcrossPaths) {
  using cudaq::qec::float_t;

  // Exercise the packed-mask path at bit 32 and retain the vector-path
  // coverage at 64 observables; each identity edge flips only its matching bit.
  for (const std::size_t num_observables : {33u, 64u}) {
    cudaqx::tensor<uint8_t> H({num_observables, num_observables});
    cudaqx::tensor<uint8_t> O({num_observables, num_observables});
    for (std::size_t i = 0; i < num_observables; ++i) {
      H.at({i, i}) = 1;
      O.at({i, i}) = 1;
    }

    auto d = cudaq::qec::decoder::get(
        "pymatching",
        cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H),
                                 cudaq::qec::sparse_binary_matrix(O)),
        cudaq::qec::decode_result_type::observables);
    // ASSERT: valid graph-like identity matrices must construct a decoder.
    ASSERT_NE(d, nullptr);

    // Decode several times on the same instance. At 64 observables PyMatching
    // writes the prediction through a caller-supplied buffer that it XORs into
    // rather than assigns, and that buffer is reused across calls, so a single
    // decode would still pass if it were not cleared per call: the flips would
    // instead leak into the following decode. Firing a different detector each
    // time, and revisiting one at the end, exercises that.
    for (const std::size_t fired : {num_observables - 1, std::size_t{0},
                                    num_observables / 2, num_observables - 1}) {
      std::vector<float_t> syndrome(num_observables, 0.0);
      syndrome[fired] = 1.0;
      auto result = d->decode(syndrome);
      // ASSERT: both observable decoding paths must successfully converge.
      ASSERT_TRUE(result.converged) << "num_observables=" << num_observables;
      // ASSERT: observable-aware decoding returns one result per O row.
      ASSERT_EQ(result.result.size(), num_observables);
      // ASSERT: the fired detector flips its own observable and nothing else,
      // so the high bit must not alias another bit through a narrow mask and no
      // flip may survive from an earlier decode.
      for (std::size_t i = 0; i < num_observables; ++i)
        EXPECT_EQ(result.result[i], i == fired ? 1.0 : 0.0)
            << "num_observables=" << num_observables << ", fired=" << fired
            << ", index=" << i;
    }
  }
}

// Hammer one decoder from two threads: every call must either return the right
// answer or be rejected, never return a corrupted one. Removing the guard fails
// this loudly, aborting on an out-of-bounds index or an escaped
// std::invalid_argument from inside PyMatching.
//
// Whether a run actually overlaps is up to the scheduler, so no rejection count
// is asserted. That the guard is released is covered by the sequential tests
// above, which decode repeatedly on one instance.
TEST(PyMatchingDecoder, RejectsOverlappingDecodeRatherThanCorrupting) {
  using cudaq::qec::float_t;

  constexpr std::size_t num_nodes = 40;
  cudaqx::tensor<uint8_t> H({num_nodes, num_nodes});
  for (std::size_t i = 0; i < num_nodes; ++i)
    H.at({i, i}) = 1;

  cudaqx::heterogeneous_map params;
  auto d = cudaq::qec::decoder::get("pymatching", H, params);
  ASSERT_NE(d, nullptr);

  std::atomic<int> rejected{0};
  std::atomic<int> corrupted{0};
  std::atomic<int> completed{0};
  // A decode runs in well under a microsecond, so without this barrier the
  // first thread finishes its loop before the second is even scheduled.
  std::atomic<bool> go{false};
  constexpr int iterations = 20000;

  // Each identity column is a boundary edge, so firing detector `fired` must
  // predict exactly error `fired`.
  auto hammer = [&](std::size_t fired) {
    std::vector<float_t> syndrome(num_nodes, 0.0);
    syndrome[fired] = 1.0;
    while (!go.load(std::memory_order_acquire))
      ;
    for (int iter = 0; iter < iterations; ++iter) {
      try {
        auto result = d->decode(syndrome);
        bool ok = result.converged && result.result.size() == num_nodes;
        for (std::size_t i = 0; ok && i < num_nodes; ++i)
          ok = result.result[i] == (i == fired ? 1.0 : 0.0);
        if (!ok)
          corrupted.fetch_add(1);
        completed.fetch_add(1);
      } catch (const std::runtime_error &) {
        rejected.fetch_add(1);
      }
    }
  };

  std::thread t0(hammer, 0);
  std::thread t1(hammer, num_nodes - 1);
  go.store(true, std::memory_order_release);
  t0.join();
  t1.join();

  EXPECT_EQ(corrupted.load(), 0);
  // No call failed in some third way.
  EXPECT_EQ(completed.load() + rejected.load(), 2 * iterations);
  // Rejecting everything would mean the guard is never released.
  EXPECT_GT(completed.load(), 0);
}
