/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <NvOnnxParser.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define DECODER_CUDA_CHECK(call)                                               \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("CUDA Error in ai_decoder_service: ") +                  \
          cudaGetErrorString(err));                                            \
    }                                                                          \
  } while (0)

namespace cudaq::qec::realtime::experimental {

// =============================================================================
// Gateway Kernels
// =============================================================================

__global__ void gateway_input_kernel(void **mailbox_slot_ptr,
                                     void *trt_fixed_input,
                                     size_t copy_size_bytes,
                                     size_t payload_capacity_bytes) {
  void *ring_buffer_data = *mailbox_slot_ptr;
  if (ring_buffer_data == nullptr)
    return;

  const char *src =
      (const char *)ring_buffer_data + sizeof(cudaq::realtime::RPCHeader);
  char *dst = (char *)trt_fixed_input;
  size_t safe_copy_size = copy_size_bytes < payload_capacity_bytes
                              ? copy_size_bytes
                              : payload_capacity_bytes;

  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < safe_copy_size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }
}

__global__ void gateway_output_kernel(void **mailbox_slot_ptr,
                                      const void *trt_fixed_output,
                                      size_t result_size_bytes,
                                      size_t payload_capacity_bytes) {
  void *ring_buffer_data = *mailbox_slot_ptr;
  if (ring_buffer_data == nullptr)
    return;

  char *dst = (char *)ring_buffer_data + sizeof(cudaq::realtime::RPCHeader);
  const char *src = (const char *)trt_fixed_output;
  size_t safe_result_size = result_size_bytes < payload_capacity_bytes
                                ? result_size_bytes
                                : payload_capacity_bytes;

  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < safe_result_size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }

  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto *header = (const cudaq::realtime::RPCHeader *)ring_buffer_data;
    uint32_t rid = header->request_id;
    uint64_t pts = header->ptp_timestamp;

    auto *response = (cudaq::realtime::RPCResponse *)ring_buffer_data;
    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = (safe_result_size == result_size_bytes) ? 0 : 1;
    response->result_len = static_cast<uint32_t>(safe_result_size);
    response->request_id = rid;
    response->ptp_timestamp = pts;
    __threadfence_system();
  }
}

// =============================================================================
// Helpers
// =============================================================================

// Size of a TRT DataType in bits.  Sub-byte dtypes (INT4, FP4) are
// returned as 4; callers that need a byte count should ceil-divide by 8.
static int trt_dtype_bits(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 32;
  case nvinfer1::DataType::kHALF:
    return 16;
  case nvinfer1::DataType::kBF16:
    return 16;
  case nvinfer1::DataType::kFP8:
    return 8;
  case nvinfer1::DataType::kINT8:
    return 8;
  case nvinfer1::DataType::kUINT8:
    return 8;
  case nvinfer1::DataType::kINT32:
    return 32;
  case nvinfer1::DataType::kINT64:
    return 64;
  case nvinfer1::DataType::kBOOL:
    return 8;
  case nvinfer1::DataType::kINT4:
    return 4;
#if NV_TENSORRT_VERSION >= NV_TENSORRT_VERSION_INT(10, 8, 0)
  case nvinfer1::DataType::kFP4:
    return 4;
#endif
  default:
    return 32;
  }
}

// Packed storage size in bytes for @p volume elements of @p dtype.
// Handles sub-byte dtypes correctly (volume*bits rounded up to a byte).
static size_t trt_tensor_bytes(nvinfer1::DataType dtype, size_t volume) {
  return (volume * static_cast<size_t>(trt_dtype_bits(dtype)) + 7) / 8;
}

static size_t tensor_volume(const nvinfer1::Dims &d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i)
    v *= (d.d[i] > 0) ? static_cast<size_t>(d.d[i]) : 1;
  return v;
}

// =============================================================================
// Class Implementation
// =============================================================================

ai_decoder_service::Logger ai_decoder_service::gLogger;

void ai_decoder_service::Logger::log(Severity severity,
                                     const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::printf("[TensorRT] %s\n", msg);
  }
}

ai_decoder_service::ai_decoder_service(const std::string &model_path,
                                       void **device_mailbox_slot,
                                       const std::string &engine_save_path,
                                       network_typing_override typing_override,
                                       size_t rpc_slot_size_bytes)
    : device_mailbox_slot_(device_mailbox_slot),
      rpc_slot_size_bytes_(rpc_slot_size_bytes) {
  std::string ext = model_path.substr(model_path.find_last_of('.'));
  if (ext == ".onnx") {
    build_engine_from_onnx(model_path, engine_save_path, typing_override);
  } else {
    load_engine(model_path);
  }
  setup_bindings();
  set_default_rpc_slot_size();
  allocate_resources();
}

ai_decoder_service::ai_decoder_service(void **device_mailbox_slot,
                                       size_t input_bytes, size_t output_bytes,
                                       size_t rpc_slot_size_bytes)
    : device_mailbox_slot_(device_mailbox_slot), input_size_(input_bytes),
      output_size_(output_bytes), rpc_slot_size_bytes_(rpc_slot_size_bytes) {
  set_default_rpc_slot_size();
  allocate_resources();
}

std::unique_ptr<ai_decoder_service>
ai_decoder_service::create_passthrough(void **device_mailbox_slot,
                                       size_t input_bytes, size_t output_bytes,
                                       size_t rpc_slot_size_bytes) {
  return std::unique_ptr<ai_decoder_service>(new ai_decoder_service(
      device_mailbox_slot, input_bytes, output_bytes, rpc_slot_size_bytes));
}

ai_decoder_service::~ai_decoder_service() {
  if (graph_exec_)
    cudaGraphExecDestroy(graph_exec_);
  if (d_trt_input_)
    cudaFree(d_trt_input_);
  if (d_trt_output_)
    cudaFree(d_trt_output_);
  for (auto *buf : d_aux_buffers_)
    cudaFree(buf);
}

void ai_decoder_service::load_engine(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("Error opening engine file: " + path);

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  context_.reset(engine_->createExecutionContext());
}

// Scan a parsed TRT network for quantization signals.  Populates the
// returned @c onnx_quant_info based on (1) the dtypes of every
// QuantizeLayer/DequantizeLayer output tensor and (2) the declared
// dtypes of the network's primary input tensors.
static onnx_quant_info
scan_network_for_quant_info(const nvinfer1::INetworkDefinition &network) {
  onnx_quant_info info{};

  for (int i = 0; i < network.getNbLayers(); ++i) {
    const auto *layer = network.getLayer(i);
    auto lt = layer->getType();
    if (lt != nvinfer1::LayerType::kQUANTIZE &&
        lt != nvinfer1::LayerType::kDEQUANTIZE)
      continue;
    for (int o = 0; o < layer->getNbOutputs(); ++o) {
      auto dt = layer->getOutput(o)->getType();
      switch (dt) {
      case nvinfer1::DataType::kFP8:
        info.has_fp8 = true;
        break;
#if NV_TENSORRT_VERSION >= NV_TENSORRT_VERSION_INT(10, 8, 0)
      case nvinfer1::DataType::kFP4:
        info.has_fp4 = true;
        break;
#endif
      case nvinfer1::DataType::kINT8:
      case nvinfer1::DataType::kINT4:
        info.has_int8 = true;
        break;
      default:
        break;
      }
    }
  }

  for (int i = 0; i < network.getNbInputs(); ++i) {
    auto dt = network.getInput(i)->getType();
    if (dt == nvinfer1::DataType::kBF16)
      info.has_explicit_bf16 = true;
    else if (dt == nvinfer1::DataType::kHALF)
      info.has_explicit_fp16 = true;
  }

  return info;
}

onnx_quant_info inspect_onnx(const std::string &onnx_path) {
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(ai_decoder_service::gLogger));
  if (!builder)
    return {};
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0));
  if (!network)
    return {};
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, ai_decoder_service::gLogger));
  if (!parser || !parser->parseFromFile(
                     onnx_path.c_str(),
                     static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
    return {};
  }
  return scan_network_for_quant_info(*network);
}

network_typing_override parse_network_typing(const std::string &value) {
  if (value == "auto" || value == "automatic")
    return network_typing_override::automatic;
  if (value == "weak" || value == "weakly_typed" || value == "weakly-typed")
    return network_typing_override::weakly_typed;
  if (value == "strong" || value == "strongly_typed" ||
      value == "strongly-typed")
    return network_typing_override::strongly_typed;
  throw std::invalid_argument("Invalid network-typing mode: '" + value +
                              "' (expected auto|weak|strong)");
}

void ai_decoder_service::build_engine_from_onnx(
    const std::string &onnx_path, const std::string &engine_save_path,
    network_typing_override typing_override) {
  runtime_.reset(nvinfer1::createInferRuntime(gLogger));

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger));

  // ---------------------------------------------------------------------
  // Resolve network typing mode.  Auto-detect inspects the ONNX: any
  // quantized op (FP8 / NVFP4 / INT8) or explicit BF16 IO forces
  // strongly-typed; otherwise fall back to weakly-typed + FP16 hint.
  // ---------------------------------------------------------------------
  bool strongly_typed = false;
  switch (typing_override) {
  case network_typing_override::automatic:
    quant_info_ = inspect_onnx(onnx_path);
    strongly_typed = quant_info_.requires_strongly_typed();
    std::printf("[TensorRT] Auto-detected ONNX quant signals: "
                "fp8=%d fp4=%d int8=%d bf16=%d fp16=%d -> %s\n",
                quant_info_.has_fp8, quant_info_.has_fp4, quant_info_.has_int8,
                quant_info_.has_explicit_bf16, quant_info_.has_explicit_fp16,
                strongly_typed ? "strongly-typed" : "weakly-typed");
    break;
  case network_typing_override::weakly_typed:
    strongly_typed = false;
    std::printf("[TensorRT] Network typing forced to weakly-typed\n");
    break;
  case network_typing_override::strongly_typed:
    strongly_typed = true;
    std::printf("[TensorRT] Network typing forced to strongly-typed\n");
    break;
  }

  using NetFlag = nvinfer1::NetworkDefinitionCreationFlag;
  uint32_t net_flags = 0;
  if (strongly_typed)
    net_flags |= 1U << static_cast<uint32_t>(NetFlag::kSTRONGLY_TYPED);

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(net_flags));
  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  // Precision hints only apply to weakly-typed networks.  A strongly-typed
  // network ignores BuilderFlag::kFP16/kBF16/kFP8/kINT8 and builds per
  // the dtypes declared in the ONNX.
  if (!strongly_typed) {
    // platformHasFastFp16 and BuilderFlag::kFP16 were deprecated in TRT
    // 10.x and removed in TRT 11.x in favor of strongly-typed networks.
    // For TRT < 11, the hint is still useful for unquantized FP32/FP16 ONNX.
#if NV_TENSORRT_MAJOR < 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (builder->platformHasFastFp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
      std::printf("[TensorRT] FP16 precision enabled (weakly-typed).\n");
    } else {
      std::printf("[TensorRT] Warning: Platform does not support fast FP16. "
                  "Using FP32.\n");
    }
#pragma GCC diagnostic pop
#endif
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger));

  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX file: " + onnx_path);
  }

  // If we got here via weakly_typed override (not automatic), populate
  // quant_info_ post-parse so get_quant_info() still reflects reality.
  if (typing_override != network_typing_override::automatic)
    quant_info_ = scan_network_for_quant_info(*network);

  bool has_dynamic = false;
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto *input = network->getInput(i);
    auto dims = input->getDimensions();
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] <= 0) {
        has_dynamic = true;
        break;
      }
    }
    if (has_dynamic)
      break;
  }

  if (has_dynamic) {
    auto *profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto *input = network->getInput(i);
      auto dims = input->getDimensions();
      nvinfer1::Dims fixed = dims;
      for (int d = 0; d < fixed.nbDims; ++d) {
        if (fixed.d[d] <= 0)
          fixed.d[d] = 1;
      }
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kMIN, fixed);
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kOPT, fixed);
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kMAX, fixed);
      std::printf("[TensorRT] Set dynamic input \"%s\" to batch=1\n",
                  input->getName());
    }
    config->addOptimizationProfile(profile);
  }

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan)
    throw std::runtime_error("Failed to build TRT engine from ONNX");

  if (!engine_save_path.empty()) {
    std::ofstream out(engine_save_path, std::ios::binary);
    if (out.good()) {
      out.write(static_cast<const char *>(plan->data()), plan->size());
      std::printf("[TensorRT] Saved engine to: %s\n", engine_save_path.c_str());
    } else {
      std::fprintf(stderr, "[TensorRT] Warning: could not save engine to %s\n",
                   engine_save_path.c_str());
    }
  }

  engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_)
    throw std::runtime_error("Failed to deserialize built engine");

  context_.reset(engine_->createExecutionContext());

  std::printf("[TensorRT] Built engine from ONNX: %s\n", onnx_path.c_str());
}

void ai_decoder_service::setup_bindings() {
  int num_io = engine_->getNbIOTensors();
  bool found_input = false;
  bool found_output = false;

  for (int i = 0; i < num_io; ++i) {
    const char *name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    auto dims = engine_->getTensorShape(name);
    auto dtype = engine_->getTensorDataType(name);
    size_t volume = tensor_volume(dims);
    size_t size_bytes = trt_tensor_bytes(dtype, volume);

    bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);

    std::printf("[TensorRT] Binding %d: \"%s\" %s, dtype=%d, elem_bits=%d, "
                "volume=%zu, %zu bytes\n",
                i, name, is_input ? "INPUT" : "OUTPUT", static_cast<int>(dtype),
                trt_dtype_bits(dtype), volume, size_bytes);

    tensor_binding binding{name, nullptr, size_bytes, is_input};

    if (is_input && !found_input) {
      input_size_ = size_bytes;
      input_num_elements_ = volume;
      found_input = true;
    } else if (!is_input && !found_output) {
      output_size_ = size_bytes;
      output_num_elements_ = volume;
      found_output = true;
    }

    all_bindings_.push_back(std::move(binding));
  }
}

void ai_decoder_service::allocate_resources() {
  if (all_bindings_.empty()) {
    // Passthrough path (no TRT bindings)
    if (cudaMalloc(&d_trt_input_, input_size_) != cudaSuccess)
      throw std::runtime_error("Failed to allocate TRT Input");
    if (cudaMalloc(&d_trt_output_, output_size_) != cudaSuccess)
      throw std::runtime_error("Failed to allocate TRT Output");
    return;
  }

  bool assigned_input = false;
  bool assigned_output = false;

  for (auto &b : all_bindings_) {
    void *buf = nullptr;
    if (cudaMalloc(&buf, b.size_bytes) != cudaSuccess)
      throw std::runtime_error("Failed to allocate buffer for " + b.name);
    cudaMemset(buf, 0, b.size_bytes);
    b.d_buffer = buf;

    if (b.is_input && !assigned_input) {
      d_trt_input_ = buf;
      assigned_input = true;
    } else if (!b.is_input && !assigned_output) {
      d_trt_output_ = buf;
      assigned_output = true;
    } else {
      d_aux_buffers_.push_back(buf);
    }
  }
}

void ai_decoder_service::set_rpc_slot_size_bytes(size_t rpc_slot_size_bytes) {
  rpc_slot_size_bytes_ = rpc_slot_size_bytes;
  set_default_rpc_slot_size();
}

void ai_decoder_service::set_default_rpc_slot_size() {
  if (rpc_slot_size_bytes_ == 0)
    rpc_slot_size_bytes_ = sizeof(cudaq::realtime::RPCHeader) + input_size_;
}

void ai_decoder_service::validate_rpc_slot_size() const {
  const size_t request_frame_size =
      sizeof(cudaq::realtime::RPCHeader) + input_size_;
  const size_t response_frame_size =
      sizeof(cudaq::realtime::RPCResponse) + output_size_;

  if (output_size_ > std::numeric_limits<uint32_t>::max())
    throw std::length_error(
        "ai_decoder_service output exceeds RPCResponse::result_len capacity");

  if (rpc_slot_size_bytes_ < request_frame_size) {
    throw std::length_error(
        "ai_decoder_service RPC slot is too small for the input payload");
  }
  if (rpc_slot_size_bytes_ < response_frame_size) {
    throw std::length_error(
        "ai_decoder_service RPC slot is too small for the output payload");
  }
}

void ai_decoder_service::capture_graph(cudaStream_t stream) {
  validate_rpc_slot_size();

  for (auto &b : all_bindings_) {
    context_->setTensorAddress(b.name.c_str(), b.d_buffer);
  }

  if (!context_->enqueueV3(stream))
    throw std::runtime_error(
        "TRT enqueueV3 warmup failed in ai_decoder_service");
  DECODER_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph;
  DECODER_CUDA_CHECK(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  const size_t input_payload_capacity =
      rpc_slot_size_bytes_ - sizeof(cudaq::realtime::RPCHeader);
  const size_t output_payload_capacity =
      rpc_slot_size_bytes_ - sizeof(cudaq::realtime::RPCResponse);

  gateway_input_kernel<<<1, 128, 0, stream>>>(
      device_mailbox_slot_, d_trt_input_, input_size_, input_payload_capacity);
  if (!context_->enqueueV3(stream))
    throw std::runtime_error(
        "TRT enqueueV3 failed during graph capture in ai_decoder_service");
  gateway_output_kernel<<<1, 128, 0, stream>>>(device_mailbox_slot_,
                                               d_trt_output_, output_size_,
                                               output_payload_capacity);

  DECODER_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaError_t inst_err = cudaGraphInstantiateWithFlags(
      &graph_exec_, graph, cudaGraphInstantiateFlagDeviceLaunch);
  if (inst_err != cudaSuccess) {
    cudaGraphDestroy(graph);
    throw std::runtime_error(
        std::string(
            "cudaGraphInstantiateWithFlags failed in ai_decoder_service: ") +
        cudaGetErrorString(inst_err));
  }

  DECODER_CUDA_CHECK(cudaGraphUpload(graph_exec_, stream));
  cudaGraphDestroy(graph);
  DECODER_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace cudaq::qec::realtime::experimental
