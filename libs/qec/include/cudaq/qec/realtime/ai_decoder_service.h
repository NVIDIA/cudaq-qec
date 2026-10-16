/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// TensorRT 10.12+ headers emit deprecation warnings for internal symbols
// (IPluginV2, legacy calibrator enums, IAlgorithmSelector, etc.) that are
// scheduled for removal in a future release.  These warnings originate inside
// the TensorRT headers themselves and cannot be fixed on our side.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "NvInfer.h"
#pragma GCC diagnostic pop

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::realtime::experimental {

/// @brief Override the network-typing mode used when compiling an ONNX.
/// @details In TensorRT, the @em network-typing choice is made at
/// @c createNetworkV2() time and cannot be changed later:
/// - @c weakly_typed : TRT's optimizer selects per-layer precision
///   guided by @c BuilderFlag hints (e.g. @c kFP16). Appropriate for
///   plain FP32/FP16 ONNX where we want TRT to optimize.
/// - @c strongly_typed : TRT honors the dtypes declared in the ONNX
///   verbatim, including every Q/DQ node. Required for quantized
///   ONNX (FP8, NVFP4, INT8) and for explicit mixed-precision ONNX.
/// - @c automatic : inspect the ONNX and pick the correct mode.
enum class network_typing_override {
  automatic,
  weakly_typed,
  strongly_typed,
};

/// @brief Quantization signals discovered in an ONNX model.
/// @details Populated by @c inspect_onnx. A @c true field means the
/// ONNX contains at least one Q/DQ or explicit-dtype tensor of that
/// kind; mixed-precision models will have multiple fields @c true.
struct onnx_quant_info {
  bool has_fp8 = false;           ///< FP8 Q/DQ present.
  bool has_fp4 = false;           ///< NVFP4 Q/DQ present.
  bool has_int8 = false;          ///< INT8 Q/DQ present.
  bool has_explicit_bf16 = false; ///< BF16 tensors declared in ONNX.
  bool has_explicit_fp16 = false; ///< FP16 tensors declared in ONNX.

  /// @brief Whether TRT must be configured with a strongly-typed network
  /// to build a correct engine from this ONNX.
  bool requires_strongly_typed() const {
    return has_fp8 || has_fp4 || has_int8 || has_explicit_bf16;
  }
};

/// @brief Inspect an ONNX model and report the quantization/precision
/// signals TRT will see during engine build.
/// @details Performs a one-off TRT parse into a throwaway weakly-typed
/// network and walks the resulting layer list plus input tensor types.
/// Returns all-@c false on parse failure rather than throwing; callers
/// that care can check the file themselves or pass an explicit override.
onnx_quant_info inspect_onnx(const std::string &onnx_path);

/// @brief Convert a string CLI value to @c network_typing_override.
/// @returns @c network_typing_override::automatic for @c "auto" /
/// @c "automatic", @c weakly_typed for @c "weak", @c strongly_typed
/// for @c "strong".  Throws @c std::invalid_argument otherwise.
network_typing_override parse_network_typing(const std::string &value);

class ai_decoder_service {
public:
  class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
  } static gLogger;

  /// @brief Constructor. Accepts a serialized TRT engine (.engine/.plan) or
  ///        an ONNX model (.onnx) which will be compiled to a TRT engine.
  /// @param model_path Path to the model file
  /// @param device_mailbox_slot Pointer to the specific slot in the global
  /// mailbox bank
  /// @param engine_save_path If non-empty and model_path is .onnx, save the
  ///        built engine to this path for fast reloading on subsequent runs
  /// @param typing_override Override the network-typing mode used when
  ///        compiling an ONNX. @c automatic (default) inspects the ONNX
  ///        and picks strongly-typed for any quantized (FP8/NVFP4/INT8)
  ///        or explicitly-BF16 model, weakly-typed otherwise.
  /// @param rpc_slot_size_bytes Total mapped RPC slot size, including the
  ///        RPC header/response. Zero (default) assumes the slot is sized
  ///        for the request frame (@c RPCHeader + input bytes). Graph
  ///        capture rejects models whose response frame would not fit.
  ai_decoder_service(const std::string &model_path, void **device_mailbox_slot,
                     const std::string &engine_save_path = "",
                     network_typing_override typing_override =
                         network_typing_override::automatic,
                     size_t rpc_slot_size_bytes = 0);

  /// Create a passthrough (identity copy) instance for testing without TRT.
  static std::unique_ptr<ai_decoder_service>
  create_passthrough(void **device_mailbox_slot,
                     size_t input_bytes = 1600 * sizeof(float),
                     size_t output_bytes = 1600 * sizeof(float),
                     size_t rpc_slot_size_bytes = 0);

  virtual ~ai_decoder_service();

  virtual void capture_graph(cudaStream_t stream);

  cudaGraphExec_t get_executable_graph() const { return graph_exec_; }

  /// @brief Size of the primary input tensor in bytes (payload from RPC)
  size_t get_input_size() const { return input_size_; }

  /// @brief Size of the primary output tensor in bytes (forwarded to CPU)
  size_t get_output_size() const { return output_size_; }

  /// @brief Set the total RPC slot size, including the RPC header/response.
  ///
  /// If left as zero, the service assumes slots are sized for the input
  /// request frame (RPCHeader + input bytes). Graph capture rejects models
  /// whose response frame would not fit.
  void set_rpc_slot_size_bytes(size_t rpc_slot_size_bytes);

  /// @brief Total RPC slot size in bytes, including the RPC header/response.
  size_t get_rpc_slot_size_bytes() const { return rpc_slot_size_bytes_; }

  /// @brief Logical element count (detector count) of the primary input
  /// tensor.  Computed from the engine's tensor volume; independent of
  /// the IO dtype so FP16 / INT32 / FP8 inputs all report the same
  /// detector count for a given model.  Zero for passthrough instances.
  size_t get_input_num_elements() const { return input_num_elements_; }

  /// @brief Logical element count of the primary output tensor.
  size_t get_output_num_elements() const { return output_num_elements_; }

  /// @brief Quantization / precision signals discovered at build time.
  /// @details All-@c false for engines loaded from a pre-built plan
  /// (the plan is opaque to us), or for passthrough instances.
  const onnx_quant_info &get_quant_info() const { return quant_info_; }

  void *get_trt_input_ptr() const { return d_trt_input_; }

protected:
  /// Passthrough constructor (no TRT, identity copy kernel only).
  ai_decoder_service(void **device_mailbox_slot, size_t input_bytes,
                     size_t output_bytes, size_t rpc_slot_size_bytes = 0);

  void load_engine(const std::string &path);
  void build_engine_from_onnx(const std::string &onnx_path,
                              const std::string &engine_save_path,
                              network_typing_override typing_override);
  void setup_bindings();
  void allocate_resources();
  void set_default_rpc_slot_size();
  void validate_rpc_slot_size() const;

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  cudaGraphExec_t graph_exec_ = nullptr;

  void **device_mailbox_slot_;
  void *d_trt_input_ = nullptr;  // Primary input buffer
  void *d_trt_output_ = nullptr; // Primary output buffer (residual_detectors)
  std::vector<void *> d_aux_buffers_; // Additional I/O buffers TRT needs

  struct tensor_binding {
    std::string name;
    void *d_buffer = nullptr;
    size_t size_bytes = 0;
    bool is_input = false;
  };
  std::vector<tensor_binding> all_bindings_;

  size_t input_size_ = 0;
  size_t output_size_ = 0;
  size_t input_num_elements_ = 0;
  size_t output_num_elements_ = 0;
  size_t rpc_slot_size_bytes_ = 0;

  onnx_quant_info quant_info_;
};

} // namespace cudaq::qec::realtime::experimental
