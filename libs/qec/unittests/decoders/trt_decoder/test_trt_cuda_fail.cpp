/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_cc_test_helpers.h"
#include "cudaq/qec/decoder.h"

#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
cudaError_t __real_cudaMemset(void *dev, int v, size_t n);
cudaError_t __real_cudaStreamBeginCapture(cudaStream_t s,
                                          cudaStreamCaptureMode m);
cudaError_t __real_cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *g);
cudaError_t __real_cudaGraphInstantiate(cudaGraphExec_t *e, cudaGraph_t g,
                                        unsigned long long f);
cudaError_t __real_cudaGraphDestroy(cudaGraph_t g);
cudaError_t __real_cudaGraphExecDestroy(cudaGraphExec_t e);
}

namespace {
const char *g_mode = "";
int g_destroy_graph = 0;
int g_destroy_exec = 0;
cudaGraph_t g_last_graph = nullptr;
cudaGraphExec_t g_fake_exec =
    reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(0xF1));

cudaqx::tensor<uint8_t> make_h() {
  cudaqx::tensor<uint8_t> H({std::size_t{3}, std::size_t{3}});
  for (std::size_t i = 0; i < 3; ++i)
    H.at({i, i}) = 1;
  return H;
}

int run_mode(const char *mode) {
  g_mode = mode;
  g_destroy_graph = 0;
  g_destroy_exec = 0;
#ifndef TRT_TEST_UINT8_ONNX_PATH
  return 2;
#else
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string(TRT_TEST_UINT8_ONNX_PATH));
  std::unique_ptr<cudaq::qec::decoder> dec;
  try {
    dec = cudaq::qec::decoder::get("trt_decoder", make_h(), params);
  } catch (const std::exception &) {
    return 2;
  }
  auto r = dec->decode({0.0, 1.0, 1.0});
  if (!r.converged || r.result.size() != 3)
    return 2;
  if (std::strcmp(mode, "instantiate_throw") == 0 &&
      (g_destroy_graph < 1 || g_destroy_exec < 1))
    return 2;
  return 0;
#endif
}
} // namespace

extern "C" {

cudaError_t __wrap_cudaMemset(void *dev, int v, size_t n) {
  if (std::strcmp(g_mode, "memset") == 0)
    return cudaErrorUnknown;
  return __real_cudaMemset(dev, v, n);
}
cudaError_t __wrap_cudaStreamBeginCapture(cudaStream_t s,
                                          cudaStreamCaptureMode m) {
  if (std::strcmp(g_mode, "begin") == 0)
    return cudaErrorUnknown;
  return __real_cudaStreamBeginCapture(s, m);
}
cudaError_t __wrap_cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *g) {
  if (std::strcmp(g_mode, "end") == 0) {
    cudaGraph_t tmp = nullptr;
    cudaError_t e = __real_cudaStreamEndCapture(s, &tmp);
    if (e == cudaSuccess && tmp)
      __real_cudaGraphDestroy(tmp);
    if (g)
      *g = nullptr;
    return cudaErrorUnknown;
  }
  return __real_cudaStreamEndCapture(s, g);
}
cudaError_t __wrap_cudaGraphInstantiate(cudaGraphExec_t *e, cudaGraph_t g,
                                        unsigned long long f) {
  if (std::strcmp(g_mode, "instantiate") == 0)
    return cudaErrorUnknown;
  if (std::strcmp(g_mode, "instantiate_throw") == 0) {
    if (e)
      *e = g_fake_exec;
    g_last_graph = g;
    throw std::runtime_error("instantiate-boom");
  }
  return __real_cudaGraphInstantiate(e, g, f);
}
cudaError_t __wrap_cudaGraphDestroy(cudaGraph_t g) {
  ++g_destroy_graph;
  if (g == nullptr)
    return cudaSuccess;
  return __real_cudaGraphDestroy(g);
}
cudaError_t __wrap_cudaGraphExecDestroy(cudaGraphExec_t e) {
  ++g_destroy_exec;
  if (e == g_fake_exec)
    return cudaSuccess;
  return __real_cudaGraphExecDestroy(e);
}

} // extern "C"

TEST(TrtCudaFail, HelperModes) {
  EXPECT_EQ(qec_cc::exec_self("memset"), 0);
  EXPECT_EQ(qec_cc::exec_self("begin"), 0);
  EXPECT_EQ(qec_cc::exec_self("end"), 0);
  EXPECT_EQ(qec_cc::exec_self("instantiate"), 0);
  EXPECT_EQ(qec_cc::exec_self("instantiate_throw"), 0);
}

int main(int argc, char **argv) {
  if (const char *helper = qec_cc::helper_name(argc, argv))
    return run_mode(helper);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
