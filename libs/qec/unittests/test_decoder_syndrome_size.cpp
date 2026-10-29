/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Every decoder must reject a syndrome whose length it does not accept with
// std::invalid_argument, through every entry point, instead of reading out of
// bounds (https://github.com/NVIDIA/cudaq-qec/issues/884).

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_init.h"
#include "cudaq/qec/pcm_utils.h"
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cudaq::qec::float_t;

// | 1 1 0 |
// | 0 1 1 |
constexpr std::size_t kSyndromeSize = 2;
constexpr std::size_t kBlockSize = 3;

cudaqx::tensor<uint8_t> small_H() {
  std::vector<uint8_t> H_vec = {1, 1, 0, 0, 1, 1};
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {kSyndromeSize, kBlockSize});
  return H;
}

class FixedWidthDecoder : public ::testing::TestWithParam<std::string> {
protected:
  std::unique_ptr<cudaq::qec::decoder> make() {
    cudaqx::heterogeneous_map params;
    return cudaq::qec::decoder::get(GetParam(), small_H(), params);
  }
};

const std::vector<std::size_t> kBadLengths = {0, 1, kSyndromeSize + 1};

TEST_P(FixedWidthDecoder, DecodeRejectsWrongLength) {
  auto d = make();
  for (auto n : kBadLengths)
    EXPECT_THROW(d->decode(std::vector<float_t>(n, 0.0)), std::invalid_argument)
        << "length " << n;
}

TEST_P(FixedWidthDecoder, DecodeRejectsWrongLengthWithErrorsSet) {
  auto d = make();
  EXPECT_THROW(d->decode(std::vector<float_t>{1.0}), std::invalid_argument);
  EXPECT_THROW(d->decode(std::vector<float_t>{0.0, 0.0, 1.0}),
               std::invalid_argument);
}

TEST_P(FixedWidthDecoder, TensorDecodeRejectsWrongLength) {
  auto d = make();
  std::vector<uint8_t> bits = {1};
  cudaqx::tensor<uint8_t> t;
  t.copy(bits.data(), {bits.size()});
  EXPECT_THROW(d->decode(t), std::invalid_argument);
}

TEST_P(FixedWidthDecoder, DecodeBatchRejectsWrongLengthInAnyRow) {
  auto d = make();
  const std::vector<float_t> good(kSyndromeSize, 0.0);
  for (auto n : kBadLengths) {
    const std::vector<float_t> bad(n, 0.0);
    EXPECT_THROW(d->decode_batch({bad}), std::invalid_argument)
        << "length " << n;
    EXPECT_THROW(d->decode_batch({good, bad}), std::invalid_argument)
        << "length " << n;
  }
}

TEST_P(FixedWidthDecoder, DecodeAsyncRejectsWrongLength) {
  auto d = make();
  auto fut = d->decode_async(std::vector<float_t>{1.0});
  EXPECT_THROW(fut.get(), std::invalid_argument);
}

TEST_P(FixedWidthDecoder, AcceptsCorrectLengthAfterRejection) {
  auto d = make();
  EXPECT_THROW(d->decode(std::vector<float_t>{1.0}), std::invalid_argument);
  auto r = d->decode(std::vector<float_t>(kSyndromeSize, 0.0));
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(d->decode_batch({}).size(), 0u);
}

INSTANTIATE_TEST_SUITE_P(Decoders, FixedWidthDecoder,
                         ::testing::Values("single_error_lut",
                                           "multi_error_lut",
                                           "single_error_lut_example",
                                           "pymatching"));

cudaq::qec::decoder_init
sliding_window_inputs(const cudaqx::tensor<uint8_t> &pcm) {
  auto H = cudaq::qec::sparse_binary_matrix(pcm);
  auto O = cudaq::qec::sparse_binary_matrix::from_csr(0, H.num_cols(), {0}, {});
  return cudaq::qec::decoder_init(std::move(H), std::move(O),
                                  std::vector<double>(pcm.shape()[1], 0.1));
}

std::unique_ptr<cudaq::qec::decoder>
make_sliding_window(std::size_t &num_syndromes, std::size_t &per_round) {
  const std::size_t n_rounds = 4;
  const std::size_t n_errs_per_round = 3;
  per_round = 2;
  auto pcm =
      cudaq::qec::generate_random_pcm(n_rounds, n_errs_per_round, per_round,
                                      /*weight=*/1, std::mt19937_64(884));
  pcm = cudaq::qec::sort_pcm_columns(pcm, per_round);
  num_syndromes = pcm.shape()[0];

  cudaqx::heterogeneous_map params;
  params.insert("window_size", std::size_t{2});
  params.insert("step_size", std::size_t{1});
  params.insert("num_syndromes_per_round", per_round);
  params.insert("inner_decoder_name", std::string("single_error_lut"));
  params.insert("inner_decoder_params", cudaqx::heterogeneous_map{});
  return cudaq::qec::decoder::get("sliding_window", sliding_window_inputs(pcm),
                                  params);
}

TEST(SlidingWindowSyndromeSize, WholeBlockBatchRejectsShortLaterRow) {
  std::size_t num_syndromes = 0, per_round = 0;
  auto d = make_sliding_window(num_syndromes, per_round);
  const std::vector<float_t> full(num_syndromes, 0.0);
  const std::vector<float_t> short_row(num_syndromes - 1, 0.0);
  EXPECT_THROW(d->decode_batch({full, short_row}), std::invalid_argument);
  auto results = d->decode_batch({full, full});
  EXPECT_EQ(results.size(), 2u);
}

TEST(SlidingWindowSyndromeSize, OtherLengthNamesBothValidWidths) {
  std::size_t num_syndromes = 0, per_round = 0;
  auto d = make_sliding_window(num_syndromes, per_round);
  try {
    d->decode(std::vector<float_t>(num_syndromes - 1, 0.0));
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("full syndrome (" + std::to_string(num_syndromes) + ")"),
              std::string::npos)
        << msg;
    EXPECT_NE(msg.find("round 0 (" + std::to_string(per_round) + ")"),
              std::string::npos)
        << msg;
  }
}

TEST(SlidingWindowSyndromeSize, PerRoundInputStillAccepted) {
  std::size_t num_syndromes = 0, per_round = 0;
  auto d = make_sliding_window(num_syndromes, per_round);
  const std::vector<float_t> round(per_round, 0.0);
  EXPECT_NO_THROW(d->decode(round));
}

} // namespace
