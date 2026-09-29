/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "cudaq/qec/realtime/autonomous_decoder.cuh"
#include "cudaq/qec/realtime/decoder_context.h"

// cudaq_function_entry_t for function table initialization
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

namespace cudaq::qec::realtime {

//==============================================================================
// Mock Decoder Class (CRTP)
//==============================================================================

/// @brief Mock decoder for CI testing, inheriting from autonomous_decoder.
///
/// This decoder uses a lookup table to map input measurements to expected
/// corrections. It demonstrates the autonomous_decoder CRTP pattern and
/// provides a testable implementation for verifying the real-time dispatch
/// infrastructure.
class mock_decoder : public autonomous_decoder<mock_decoder> {
public:
  /// @brief Constructor taking a reference to the decoder context.
  ///
  /// @param ctx Reference to mock_decoder_context with lookup tables
  __device__ __host__ explicit mock_decoder(mock_decoder_context &ctx)
      : ctx_(ctx) {}

  /// @brief Core decode implementation (required by CRTP base).
  ///
  /// Searches the lookup table for matching measurements and returns the
  /// corresponding corrections. This is called by the base class's decode()
  /// method.
  ///
  /// @param measurements Input raw measurements from quantum circuit
  /// @param corrections Output buffer for computed corrections
  /// @param num_measurements Number of input measurements
  /// @param num_observables Number of observable corrections to output
  __device__ void decode_impl(const uint8_t *__restrict__ measurements,
                              uint8_t *__restrict__ corrections,
                              std::size_t num_measurements,
                              std::size_t num_observables);

  /// @brief Get the decoder context.
  /// @return Reference to the mock_decoder_context
  __device__ __host__ mock_decoder_context &context() { return ctx_; }

  /// @brief Get the decoder context (const version).
  __device__ __host__ const mock_decoder_context &context() const {
    return ctx_;
  }

private:
  mock_decoder_context &ctx_; ///< Reference to decoder context
};

//==============================================================================
// Global Device Decoder Instance
//==============================================================================

/// @brief Global device pointer to mock decoder instance.
/// Must be set via set_mock_decoder() before kernel launch.
extern __device__ mock_decoder *g_mock_decoder;

/// @brief Set the mock decoder instance from host.
/// @param decoder Device pointer to mock_decoder instance
inline void set_mock_decoder(mock_decoder *decoder) {
  cudaMemcpyToSymbol(g_mock_decoder, &decoder, sizeof(mock_decoder *));
}

//==============================================================================
// DeviceRPCFunction-Compatible Wrapper
//==============================================================================

/// @brief RPC-compatible wrapper for the mock decoder.
///
/// This function matches the DeviceRPCFunction signature expected by the
/// cuda-quantum dispatch kernel.  The handler reads measurements from the
/// input buffer and writes corrections directly to the output buffer.
///
/// @param input  Input measurements (RX buffer, read-only)
/// @param output Output corrections (TX buffer, write-only)
/// @param arg_len Length of input measurement data
/// @param max_result_len Maximum space available for results
/// @param result_len Output: actual result length written
/// @return 0 on success, non-zero on error
__device__ int mock_decode_rpc(const void *input, void *output,
                               std::uint32_t arg_len,
                               std::uint32_t max_result_len,
                               std::uint32_t *result_len);

/// @brief Get device function pointer for mock_decode_rpc.
/// @return Device function pointer to mock_decode_rpc
__device__ auto get_mock_decode_rpc_ptr();

//==============================================================================
// Graph-Compatible RPC Handler (for CUDAQ_DISPATCH_GRAPH_LAUNCH)
//==============================================================================

/// @brief Graph kernel using GraphIOContext for input/output separation.
///
/// The dispatch kernel fills a GraphIOContext with pointers to the RX slot
/// (input), TX slot (output), and the TX flag before each fire-and-forget
/// graph launch.  This kernel reads measurements from the RX slot, decodes,
/// writes the RPCResponse + corrections to the TX slot, and signals
/// completion by setting the TX flag.
///
/// @param io_ctx Device pointer to GraphIOContext (filled by dispatch kernel)
__global__ void
mock_decode_graph_kernel(cudaq::realtime::GraphIOContext *io_ctx);

//==============================================================================
// Mock Decoder Context GPU Setup
//==============================================================================

/// @brief GPU resources for mock decoder context.
///
/// Holds device pointers allocated during setup. Call cleanup() to free.
struct MockDecoderGpuResources {
  uint8_t *d_lookup_measurements = nullptr;
  uint8_t *d_lookup_corrections = nullptr;
  mock_decoder_context *d_ctx = nullptr;
  mock_decoder *d_decoder = nullptr;

  void cleanup() {
    if (d_lookup_measurements)
      cudaFree(d_lookup_measurements);
    if (d_lookup_corrections)
      cudaFree(d_lookup_corrections);
    if (d_decoder)
      cudaFree(d_decoder);
    if (d_ctx)
      cudaFree(d_ctx);
    d_lookup_measurements = nullptr;
    d_lookup_corrections = nullptr;
    d_ctx = nullptr;
    d_decoder = nullptr;
  }
};

/// @brief Build lookup tables and upload mock decoder context to the GPU.
///
/// After this call, the global device context is set and the dispatch kernel
/// can invoke mock_decode_rpc.
///
/// @param measurements Flat array of measurements (num_entries * syndrome_size)
/// @param corrections  Array of expected corrections (num_entries)
/// @param num_entries  Number of lookup entries
/// @param syndrome_size Number of measurements per entry
/// @param[out] resources Device pointers (caller must call cleanup())
/// @return cudaSuccess on success
cudaError_t setup_mock_decoder_on_gpu(const uint8_t *measurements,
                                      const uint8_t *corrections,
                                      std::size_t num_entries,
                                      std::size_t syndrome_size,
                                      MockDecoderGpuResources &resources);

//==============================================================================
// Function Table Initialization
//==============================================================================

/// @brief Function ID for mock decoder (FNV-1a hash of "mock_decode").
constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("mock_decode");

/// @brief Device kernel to initialize a cudaq function table entry for the
///        mock decoder RPC handler.
///
/// Must be called as: init_mock_decode_function_table<<<1,1>>>(d_entries);
/// @param entries Device pointer to pre-allocated cudaq_function_entry_t array
__global__ void
init_mock_decode_function_table(cudaq_function_entry_t *entries);

/// @brief Host-callable wrapper that launches init_mock_decode_function_table
///        and synchronizes.
///
/// This allows callers compiled by a C++ compiler (not nvcc) to set up the
/// function table without needing CUDA kernel launch syntax.
/// @param d_entries Device pointer to pre-allocated cudaq_function_entry_t
void setup_mock_decode_function_table(cudaq_function_entry_t *d_entries);

} // namespace cudaq::qec::realtime
