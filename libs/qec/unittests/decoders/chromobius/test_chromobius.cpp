/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"

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
