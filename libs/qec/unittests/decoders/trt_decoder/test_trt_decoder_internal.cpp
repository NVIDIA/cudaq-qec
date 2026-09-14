/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

#include "cudaq/qec/logger.h"

#include "../../../lib/decoders/plugins/trt_decoder/trt_decoder.cpp"

namespace {
std::vector<cudaGraph_t> g_graphs;
std::vector<cudaGraphExec_t> g_execs;
} // namespace

extern "C" {
cudaError_t __real_cudaGraphDestroy(cudaGraph_t);
cudaError_t __real_cudaGraphExecDestroy(cudaGraphExec_t);

cudaError_t __wrap_cudaGraphDestroy(cudaGraph_t g) {
  g_graphs.push_back(g);
  return cudaSuccess;
}
cudaError_t __wrap_cudaGraphExecDestroy(cudaGraphExec_t ge) {
  g_execs.push_back(ge);
  return cudaSuccess;
}
}

namespace {

TEST(TrtDecoderInternal, TrimFilenameSlashAndPlain) {
  std::string slash = "dir/sub/file.cpp";
  trim_filename(slash);
  EXPECT_EQ(slash, "file.cpp");
  std::string plain = "plain.cpp";
  trim_filename(plain);
  EXPECT_EQ(plain, "plain.cpp");
}

TEST(TrtDecoderInternal, TensorRtLoggerSeverities) {
  Logger log;
  const auto prev = cudaq::qec::detail::get_log_level();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  log.log(nvinfer1::ILogger::Severity::kWARNING,
          "logger passed into x differs from one already registered");
  log.log(nvinfer1::ILogger::Severity::kWARNING, "real-warning");
  log.log(nvinfer1::ILogger::Severity::kERROR, "real-error");
  cudaq::qec::detail::flush_logs();
  const std::string out = testing::internal::GetCapturedStdout();
  const std::string err = testing::internal::GetCapturedStderr();
  cudaq::qec::detail::set_log_level(prev);
  EXPECT_EQ(out.find("already registered"), std::string::npos);
  EXPECT_NE(out.find("real-warning"), std::string::npos);
  EXPECT_NE(err.find("real-error"), std::string::npos);
}

} // namespace

namespace cudaq::qec {
namespace {

TEST(TrtDecoderInternal, CudaGraphExecutorMoveAndSelfAssign) {
  g_graphs.clear();
  g_execs.clear();
  const auto g1 = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(0x11));
  const auto e1 =
      reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(0x12));
  const auto g2 = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(0x21));
  const auto e2 =
      reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(0x22));
  {
    CudaGraphExecutor a(g1, e1);
    a = std::move(a);
    EXPECT_TRUE(g_graphs.empty());
    EXPECT_TRUE(g_execs.empty());
    CudaGraphExecutor b(g2, e2);
    b = std::move(a);
    ASSERT_EQ(g_graphs.size(), 1u);
    ASSERT_EQ(g_execs.size(), 1u);
    EXPECT_EQ(g_graphs[0], g2);
    EXPECT_EQ(g_execs[0], e2);
  }
  ASSERT_EQ(g_graphs.size(), 2u);
  ASSERT_EQ(g_execs.size(), 2u);
  EXPECT_EQ(g_graphs[1], g1);
  EXPECT_EQ(g_execs[1], e1);
}

TEST(TrtDecoderInternal, DataTypeSizeAndNameCoverEnums) {
  const nvinfer1::DataType types[] = {
      nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF,
      nvinfer1::DataType::kINT8,  nvinfer1::DataType::kINT32,
      nvinfer1::DataType::kBOOL,  nvinfer1::DataType::kUINT8,
      nvinfer1::DataType::kFP8,   nvinfer1::DataType::kBF16,
      nvinfer1::DataType::kINT64};
  for (auto t : types) {
    EXPECT_GT(dataTypeSize(t), 0u);
    EXPECT_STRNE(dataTypeName(t), "unknown");
  }
  auto bogus = static_cast<nvinfer1::DataType>(12345);
  EXPECT_EQ(dataTypeSize(bogus), sizeof(float));
  EXPECT_STREQ(dataTypeName(bogus), "unknown");
}

} // namespace
} // namespace cudaq::qec

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
