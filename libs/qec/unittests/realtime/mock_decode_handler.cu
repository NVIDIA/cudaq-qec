/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <new>

#include "mock_decode_handler.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

namespace cudaq::qec::realtime {

//==============================================================================
// Global Device Decoder Instance
//==============================================================================

__device__ mock_decoder *g_mock_decoder = nullptr;

//==============================================================================
// Mock Decoder Class Implementation
//==============================================================================

__device__ void mock_decoder::decode_impl(
    const uint8_t *__restrict__ measurements, uint8_t *__restrict__ corrections,
    std::size_t num_measurements, std::size_t num_observables) {
  // Thread 0 does the lookup (simple single-threaded implementation for CI)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Default: if no match found, output zeros
    for (std::size_t i = 0; i < num_observables; ++i) {
      corrections[i] = 0;
    }

    // Search for matching measurements in lookup table
    for (std::size_t entry = 0; entry < ctx_.num_lookup_entries; ++entry) {
      const uint8_t *entry_measurements =
          ctx_.lookup_measurements + entry * num_measurements;

      // Check if this entry matches the input
      bool match = true;
      for (std::size_t i = 0; i < num_measurements; ++i) {
        if (entry_measurements[i] != measurements[i]) {
          match = false;
          break;
        }
      }

      if (match) {
        // Found match - copy expected corrections to output
        const uint8_t *entry_corrections =
            ctx_.lookup_corrections + entry * num_observables;
        for (std::size_t i = 0; i < num_observables; ++i) {
          corrections[i] = entry_corrections[i];
        }
        break;
      }
    }
  }

  // Single-threaded handler; no block-wide sync required.
}

//==============================================================================
// DeviceRPCFunction-Compatible Wrapper
//==============================================================================

/// Upper bound on the request bytes mock_decode_rpc stages locally so that it
/// never reads the request after writing the response (see below).
constexpr std::size_t kMockMaxMeasurementBytes = 1024;

__device__ int mock_decode_rpc(const void *rx_slot, void *tx_slot,
                               std::size_t slot_size) {
  using cudaq::realtime::RPCHeader;
  using cudaq::realtime::RPCResponse;

  // Two-pointer ABI (same form as a HOST_CALL handler, plus a returned
  // status): the request (RPCHeader + measurements) is in rx_slot and the
  // corrections go right after the RPCResponse in tx_slot.  The handler owns
  // response->result_len; the dispatch kernel fills the other header fields
  // and forces result_len to 0 on a non-zero status.
  if (slot_size < sizeof(RPCResponse))
    return -1;
  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  const uint8_t *measurements = reinterpret_cast<const uint8_t *>(header + 1);
  auto *response = static_cast<RPCResponse *>(tx_slot);
  uint8_t *corrections = reinterpret_cast<uint8_t *>(response + 1);
  const std::size_t max_result_len = slot_size - sizeof(RPCResponse);

  if (g_mock_decoder != nullptr) {
    const auto &ctx = g_mock_decoder->context();
    if (ctx.num_observables > max_result_len)
      return -2; // Error: result does not fit in the TX slot
    if (ctx.num_measurements > kMockMaxMeasurementBytes)
      return -3; // Error: request larger than the local staging buffer

    // On an in-place ring tx_slot aliases rx_slot, so the measurement bytes
    // and the correction bytes share an address.  decode() zeroes the
    // corrections before it compares the measurements, so the ABI's
    // read-the-request-before-writing-the-response rule means staging the
    // measurements locally first.
    uint8_t local_measurements[kMockMaxMeasurementBytes];
    for (std::size_t i = 0; i < ctx.num_measurements; ++i)
      local_measurements[i] = measurements[i];

    g_mock_decoder->decode(local_measurements, corrections,
                           ctx.num_measurements, ctx.num_observables);

    response->result_len = static_cast<std::uint32_t>(ctx.num_observables);
    return 0; // Success
  } else {
    // Error: decoder not set
    return -1;
  }
}

__device__ auto get_mock_decode_rpc_ptr() { return &mock_decode_rpc; }

//==============================================================================
// Graph-Compatible RPC Handler (for CUDAQ_DISPATCH_GRAPH_LAUNCH)
//==============================================================================

__global__ void
mock_decode_graph_kernel(cudaq::realtime::GraphIOContext *io_ctx) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (io_ctx == nullptr || io_ctx->rx_slot == nullptr)
      return;

    // Parse RPC header from RX slot (input)
    auto *header = static_cast<cudaq::realtime::RPCHeader *>(io_ctx->rx_slot);
    uint8_t *measurements = reinterpret_cast<uint8_t *>(header + 1);

    // TX slot for response (output)
    auto *response =
        reinterpret_cast<cudaq::realtime::RPCResponse *>(io_ctx->tx_slot);
    uint8_t *corrections =
        io_ctx->tx_slot + sizeof(cudaq::realtime::RPCResponse);

    if (g_mock_decoder != nullptr) {
      const auto &ctx = g_mock_decoder->context();
      g_mock_decoder->decode(measurements, corrections, ctx.num_measurements,
                             ctx.num_observables);

      response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
      response->status = 0;
      response->result_len = static_cast<std::uint32_t>(ctx.num_observables);
      response->request_id = header->request_id;
      response->ptp_timestamp = header->ptp_timestamp;
    } else {
      response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
      response->status = -1;
      response->result_len = 0;
      response->request_id = header->request_id;
      response->ptp_timestamp = header->ptp_timestamp;
    }

    // Signal completion: write tx_flag so the host/emulator knows the
    // response is ready.  Must fence before the flag write.
    __threadfence_system();
    if (io_ctx->tx_flag != nullptr)
      *(io_ctx->tx_flag) = io_ctx->tx_flag_value;
  }
}

//==============================================================================
// Mock Decoder Context GPU Setup
//==============================================================================

/// @brief Kernel to initialize mock_decoder via placement new on device.
__global__ void init_mock_decoder_on_gpu(mock_decoder *d_decoder,
                                         mock_decoder_context *d_ctx) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (d_decoder) mock_decoder(*d_ctx);
  }
}

cudaError_t setup_mock_decoder_on_gpu(const uint8_t *measurements,
                                      const uint8_t *corrections,
                                      std::size_t num_entries,
                                      std::size_t syndrome_size,
                                      MockDecoderGpuResources &resources) {
  cudaError_t err;

  std::size_t meas_size = num_entries * syndrome_size;

  // Allocate and copy lookup measurements
  err = cudaMalloc(&resources.d_lookup_measurements, meas_size);
  if (err != cudaSuccess)
    return err;
  err = cudaMemcpy(resources.d_lookup_measurements, measurements, meas_size,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    return err;

  // Allocate and copy lookup corrections
  err = cudaMalloc(&resources.d_lookup_corrections, num_entries);
  if (err != cudaSuccess)
    return err;
  err = cudaMemcpy(resources.d_lookup_corrections, corrections, num_entries,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    return err;

  // Build and upload context
  mock_decoder_context ctx;
  ctx.num_measurements = syndrome_size;
  ctx.num_observables = 1;
  ctx.lookup_measurements = resources.d_lookup_measurements;
  ctx.lookup_corrections = resources.d_lookup_corrections;
  ctx.num_lookup_entries = num_entries;

  err = cudaMalloc(&resources.d_ctx, sizeof(mock_decoder_context));
  if (err != cudaSuccess)
    return err;
  err = cudaMemcpy(resources.d_ctx, &ctx, sizeof(ctx), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    return err;

  // Allocate and initialize CRTP decoder instance on device
  err = cudaMalloc(&resources.d_decoder, sizeof(mock_decoder));
  if (err != cudaSuccess)
    return err;
  init_mock_decoder_on_gpu<<<1, 1>>>(resources.d_decoder, resources.d_ctx);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    return err;

  // Set global device decoder pointer (calls cudaMemcpyToSymbol)
  set_mock_decoder(resources.d_decoder);

  return cudaSuccess;
}

//==============================================================================
// Function Table Initialization
//==============================================================================

__global__ void
init_mock_decode_function_table(cudaq_function_entry_t *entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void *>(&mock_decode_rpc);
    entries[0].function_id = MOCK_DECODE_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    // Schema: 1 bit-packed argument (128 bits = 16 bytes), 1 uint8 result
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.reserved = 0;
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_BIT_PACKED;
    entries[0].schema.args[0].reserved[0] = 0;
    entries[0].schema.args[0].reserved[1] = 0;
    entries[0].schema.args[0].reserved[2] = 0;
    entries[0].schema.args[0].size_bytes = 16;    // 128 bits
    entries[0].schema.args[0].num_elements = 128; // 128 bits
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 1;
    entries[0].schema.results[0].num_elements = 1;
  }
}

void setup_mock_decode_function_table(cudaq_function_entry_t *d_entries) {
  init_mock_decode_function_table<<<1, 1>>>(d_entries);
  cudaDeviceSynchronize();
}

} // namespace cudaq::qec::realtime
