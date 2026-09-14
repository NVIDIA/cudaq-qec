/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {
const char *getVersion() { return "test"; }
const char *getFullRepositoryVersion() { return "test"; }
} // namespace cudaq::qec

namespace {

std::vector<int> g_sets;
int g_current = 1;

class throwing_ctor_decoder : public cudaq::qec::decoder {
public:
  throwing_ctor_decoder(const cudaq::qec::sparse_binary_matrix &H,
                        const cudaqx::heterogeneous_map &)
      : decoder(H) {
    throw std::runtime_error("ctor-fail");
  }
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {};
  }
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      throwing_ctor_decoder, static std::unique_ptr<cudaq::qec::decoder> create(
                                 const cudaq::qec::decoder_init &init,
                                 const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<throwing_ctor_decoder>(init,
                                                                   params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(throwing_ctor_decoder)

} // namespace

extern "C" {

cudaError_t cudaGetDevice(int *device) {
  if (device)
    *device = g_current;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  g_sets.push_back(device);
  g_current = device;
  return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count) {
  if (count)
    *count = 2;
  return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t) { return "cuda-rollback-stub"; }

} // extern "C"

// Construction selects device 0 then the throwing ctor rolls back to 1.
TEST(DecoderCudaRollback, FailedCtorRestoresPreviousDevice) {
  g_sets.clear();
  g_current = 1;
  cudaqx::tensor<uint8_t> H({std::size_t{1}, std::size_t{1}});
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 0);
  EXPECT_THROW(cudaq::qec::decoder::get("throwing_ctor_decoder", H, params),
               std::runtime_error);
  ASSERT_EQ(g_sets.size(), 2u);
  EXPECT_EQ(g_sets[0], 0);
  EXPECT_EQ(g_sets[1], 1);
  EXPECT_EQ(g_current, 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
