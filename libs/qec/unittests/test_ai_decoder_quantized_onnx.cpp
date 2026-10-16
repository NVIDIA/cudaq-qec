/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_ASSERT_OK(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

using namespace cudaq::qec::realtime::experimental;
namespace rt_sdk = cudaq::realtime;

static constexpr size_t kNumElements = 8;
static constexpr size_t kPayloadBytes = kNumElements * sizeof(float);
static constexpr size_t kSlotSize = CUDAQ_RPC_HEADER_SIZE + kPayloadBytes;

const std::vector<float> kInputs = {-4.0f, -2.0f, -1.0f, 0.0f,
                                    1.0f,  2.0f,  3.0f,  4.0f};

bool isGpuAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

bool isFp8HardwareAvailable() {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess)
    return false;

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess)
    return false;
  return prop.major >= 9;
}

std::string build_output_larger_engine() {
  auto engine_path = std::filesystem::path(::testing::TempDir()) /
                     "ai_decoder_output_larger_than_input.engine";

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(ai_decoder_service::gLogger));
  if (!builder)
    throw std::runtime_error("Failed to create TensorRT builder");

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0));
  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!network || !config)
    throw std::runtime_error("Failed to create TensorRT network/config");

  nvinfer1::Dims input_dims{};
  input_dims.nbDims = 1;
  input_dims.d[0] = static_cast<int64_t>(kNumElements);
  auto *input =
      network->addInput("input", nvinfer1::DataType::kFLOAT, input_dims);
  if (!input)
    throw std::runtime_error("Failed to add TensorRT input");

  // Build a valid asymmetric model: output is [0.0, input...], so the
  // output payload is exactly one float larger than the input payload.
  float zero = 0.0f;
  nvinfer1::Weights zero_weights{nvinfer1::DataType::kFLOAT, &zero, 1};
  nvinfer1::Dims zero_dims{};
  zero_dims.nbDims = 1;
  zero_dims.d[0] = 1;
  auto *constant = network->addConstant(zero_dims, zero_weights);
  if (!constant)
    throw std::runtime_error("Failed to add TensorRT constant");

  nvinfer1::ITensor *concat_inputs[] = {constant->getOutput(0), input};
  auto *concat = network->addConcatenation(concat_inputs, 2);
  if (!concat)
    throw std::runtime_error("Failed to add TensorRT concatenation");
  concat->setAxis(0);
  concat->getOutput(0)->setName("output");
  network->markOutput(*concat->getOutput(0));

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan)
    throw std::runtime_error("Failed to build asymmetric TensorRT engine");

  std::ofstream out(engine_path, std::ios::binary);
  if (!out.good())
    throw std::runtime_error("Failed to open temporary engine path");
  out.write(static_cast<const char *>(plan->data()), plan->size());
  return engine_path.string();
}

void write_rpc_slot(uint8_t *slot_host, const std::vector<float> &input) {
  std::memset(slot_host, 0, kSlotSize);
  rt_sdk::RPCHeader hdr{};
  hdr.magic = rt_sdk::RPC_MAGIC_REQUEST;
  hdr.arg_len = static_cast<uint32_t>(input.size() * sizeof(float));
  std::memcpy(slot_host, &hdr, sizeof(hdr));
  std::memcpy(slot_host + CUDAQ_RPC_HEADER_SIZE, input.data(),
              input.size() * sizeof(float));
}

class AiDecoderQuantizedOnnxSmokeTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!isGpuAvailable())
      GTEST_SKIP() << "No GPU available, skipping TensorRT smoke test";

    CUDA_ASSERT_OK(cudaSetDevice(0));
    CUDA_ASSERT_OK(cudaHostAlloc(reinterpret_cast<void **>(&slot_host_),
                                 kSlotSize, cudaHostAllocMapped));
    CUDA_ASSERT_OK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&slot_dev_), slot_host_, 0));
    CUDA_ASSERT_OK(cudaHostAlloc(reinterpret_cast<void **>(&mailbox_host_),
                                 sizeof(void *), cudaHostAllocMapped));
    CUDA_ASSERT_OK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&mailbox_dev_), mailbox_host_, 0));
    mailbox_host_[0] = slot_dev_;
    CUDA_ASSERT_OK(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_)
      cudaStreamDestroy(stream_);
    if (mailbox_host_)
      cudaFreeHost(mailbox_host_);
    if (slot_host_)
      cudaFreeHost(slot_host_);
  }

  void run_service(const std::string &onnx_path,
                   const onnx_quant_info &expected_info,
                   std::vector<float> &output) {
    ai_decoder_service service(onnx_path,
                               reinterpret_cast<void **>(mailbox_dev_), "",
                               network_typing_override::automatic);

    const auto &build_info = service.get_quant_info();
    EXPECT_EQ(build_info.has_fp8, expected_info.has_fp8);
    EXPECT_EQ(build_info.has_int8, expected_info.has_int8);
    EXPECT_TRUE(build_info.requires_strongly_typed());
    EXPECT_EQ(service.get_input_num_elements(), kNumElements);
    EXPECT_EQ(service.get_output_num_elements(), kNumElements);
    EXPECT_EQ(service.get_input_size(), kPayloadBytes);
    EXPECT_EQ(service.get_output_size(), kPayloadBytes);

    service.capture_graph(stream_);
    ASSERT_NE(service.get_executable_graph(), nullptr);

    write_rpc_slot(slot_host_, kInputs);
    CUDA_ASSERT_OK(cudaGraphLaunch(service.get_executable_graph(), stream_));
    CUDA_ASSERT_OK(cudaStreamSynchronize(stream_));

    auto *response = reinterpret_cast<rt_sdk::RPCResponse *>(slot_host_);
    EXPECT_EQ(response->magic, rt_sdk::RPC_MAGIC_RESPONSE);
    EXPECT_EQ(response->status, 0u);
    EXPECT_EQ(response->result_len, kPayloadBytes);

    output.resize(kNumElements);
    std::memcpy(output.data(), slot_host_ + CUDAQ_RPC_HEADER_SIZE,
                kPayloadBytes);
  }

  uint8_t *slot_host_ = nullptr;
  uint8_t *slot_dev_ = nullptr;
  void **mailbox_host_ = nullptr;
  void **mailbox_dev_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

void expect_identity_qdq(const std::vector<float> &actual, float tolerance) {
  ASSERT_EQ(actual.size(), kInputs.size());
  for (size_t i = 0; i < kInputs.size(); ++i) {
    EXPECT_NEAR(actual[i], kInputs[i], tolerance)
        << "Mismatch at element " << i;
  }
}

TEST_F(AiDecoderQuantizedOnnxSmokeTest, Int8QdqRunsWithExpectedNumerics) {
  onnx_quant_info expected{};
  expected.has_int8 = true;

  auto info = inspect_onnx(INT8_QDQ_ONNX_PATH);
  ASSERT_TRUE(info.has_int8);
  EXPECT_FALSE(info.has_fp8);
  EXPECT_TRUE(info.requires_strongly_typed());

  std::vector<float> output;
  run_service(INT8_QDQ_ONNX_PATH, expected, output);
  expect_identity_qdq(output, 0.0f);
}

TEST_F(AiDecoderQuantizedOnnxSmokeTest, Fp8QdqRunsWithExpectedNumerics) {
  onnx_quant_info expected{};
  expected.has_fp8 = true;

  auto info = inspect_onnx(FP8_QDQ_ONNX_PATH);
  ASSERT_TRUE(info.has_fp8);
  EXPECT_FALSE(info.has_int8);
  EXPECT_TRUE(info.requires_strongly_typed());

  if (!isFp8HardwareAvailable())
    GTEST_SKIP() << "FP8 Q/DQ requires FP8-capable GPU hardware";

  std::vector<float> output;
  run_service(FP8_QDQ_ONNX_PATH, expected, output);
  expect_identity_qdq(output, 1.0e-3f);
}

TEST_F(AiDecoderQuantizedOnnxSmokeTest, RejectsOversizedGatewayOutput) {
  constexpr size_t kOutputBytes = (kNumElements + 1) * sizeof(float);

  auto engine_path = build_output_larger_engine();
  ai_decoder_service service(engine_path,
                             reinterpret_cast<void **>(mailbox_dev_), "",
                             network_typing_override::automatic);
  ASSERT_EQ(service.get_input_size(), kPayloadBytes);
  ASSERT_EQ(service.get_output_size(), kOutputBytes);

  // By default the service assumes an RPC slot sized from the request payload.
  // A larger output must be rejected before graph capture can write past it.
  EXPECT_THROW(service.capture_graph(stream_), std::length_error);
}

TEST_F(AiDecoderQuantizedOnnxSmokeTest, WritesGatewayOutputWithinSlot) {
  constexpr size_t kOutputBytes = (kNumElements + 1) * sizeof(float);
  constexpr size_t kLargeSlotSize = CUDAQ_RPC_HEADER_SIZE + kOutputBytes;

  uint8_t *slots_host = nullptr;
  uint8_t *slots_dev = nullptr;
  CUDA_ASSERT_OK(cudaHostAlloc(reinterpret_cast<void **>(&slots_host),
                               2 * kLargeSlotSize, cudaHostAllocMapped));
  auto slots_cleanup = std::unique_ptr<uint8_t, decltype(&cudaFreeHost)>(
      slots_host, cudaFreeHost);
  CUDA_ASSERT_OK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&slots_dev),
                                          slots_host, 0));

  write_rpc_slot(slots_host, kInputs);
  uint8_t *adjacent_slot = slots_host + kLargeSlotSize;
  std::memset(adjacent_slot, 0xA5, kLargeSlotSize);
  mailbox_host_[0] = slots_dev;

  auto engine_path = build_output_larger_engine();
  ai_decoder_service service(
      engine_path, reinterpret_cast<void **>(mailbox_dev_), "",
      network_typing_override::automatic, kLargeSlotSize);
  ASSERT_EQ(service.get_input_size(), kPayloadBytes);
  ASSERT_EQ(service.get_output_size(), kOutputBytes);

  service.capture_graph(stream_);
  ASSERT_NE(service.get_executable_graph(), nullptr);
  CUDA_ASSERT_OK(cudaGraphLaunch(service.get_executable_graph(), stream_));
  CUDA_ASSERT_OK(cudaStreamSynchronize(stream_));

  auto *response = reinterpret_cast<rt_sdk::RPCResponse *>(slots_host);
  EXPECT_EQ(response->magic, rt_sdk::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(response->status, 0u);
  EXPECT_EQ(response->result_len, kOutputBytes);

  const auto *output =
      reinterpret_cast<const float *>(slots_host + CUDAQ_RPC_HEADER_SIZE);
  EXPECT_FLOAT_EQ(output[0], 0.0f);
  for (size_t i = 0; i < kInputs.size(); ++i)
    EXPECT_FLOAT_EQ(output[i + 1], kInputs[i]);

  // ASSERT that the larger, explicitly declared slot contains the whole
  // response and that the next slot remains untouched.
  for (size_t i = 0; i < kLargeSlotSize; ++i)
    EXPECT_EQ(adjacent_slot[i], 0xA5) << "gateway_output_kernel overwrote byte "
                                      << i << " of the adjacent RPC slot";
}

} // namespace
