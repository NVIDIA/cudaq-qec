/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cstddef>
#include <cstdint>

namespace cudaq::qec::decoding::rpc {

// The QEC multi-handler accumulator pattern is fixed by
// proposals/decoder_server_runtime.md.  All producers and consumers must use
// these canonical function IDs and payload shapes.
constexpr std::uint32_t kEnqueueSyndromesFunctionId =
    cudaq::realtime::fnv1a_hash("enqueue_syndromes");
constexpr std::uint32_t kGetCorrectionsFunctionId =
    cudaq::realtime::fnv1a_hash("get_corrections");
constexpr std::uint32_t kResetDecoderFunctionId =
    cudaq::realtime::fnv1a_hash("reset_decoder");

static_assert(kEnqueueSyndromesFunctionId == 0x7ED8BE82u,
              "enqueue_syndromes function_id must match "
              "decoder_server_runtime.md");
static_assert(kGetCorrectionsFunctionId == 0x882D5BA1u,
              "get_corrections function_id must match "
              "decoder_server_runtime.md");
static_assert(kResetDecoderFunctionId == 0x977A59CFu,
              "reset_decoder function_id must match "
              "decoder_server_runtime.md");

// Wire framing is cudaq::realtime::RPCHeader / RPCResponse (24 bytes each,
// packed).  Pin the sizes here so a cudaq-side layout change cannot silently
// desynchronize the decoder wire format from decoder_server_runtime.md.
static_assert(sizeof(cudaq::realtime::RPCHeader) == 24,
              "cudaq::realtime::RPCHeader must be 24 bytes");
static_assert(sizeof(cudaq::realtime::RPCResponse) == 24,
              "cudaq::realtime::RPCResponse must be 24 bytes");

// Status codes carried in cudaq::realtime::RPCResponse::status.
enum class RpcStatus : std::int32_t {
  OK = 0,
  INVALID_DECODER = 1,
  BAD_REQUEST = 2,
  INTERNAL_ERROR = 3,
  NOT_READY = 4,
  BUSY = 5,
  SYNDROMES_DROPPED = 6,
};

// Hard cap for one enqueue_syndromes request. Enforced at both transport and
// session boundaries so alternate transports cannot bypass allocation and
// packed-length validation.
inline constexpr std::uint64_t kMaxSyndromeBits = 1u << 20; // 1 M bits

struct __attribute__((packed)) EnqueueRequestPayload {
  std::int64_t decoder_id;          ///< arg0
  std::int64_t counter;             ///< arg1
  std::int64_t syndrome_mapping_id; ///< arg2
  std::int64_t num_syndromes;       ///< arg3 (# syndrome bits following)
  // Trailing: ceil(num_syndromes/8) bit-packed bytes (LSB-first), no pad.
};
static_assert(sizeof(EnqueueRequestPayload) == 32,
              "EnqueueRequestPayload must be exactly 32 bytes per "
              "decoder_server_runtime.md#enqueue_syndromes");

struct __attribute__((packed)) GetCorrectionsRequestPayload {
  std::int64_t decoder_id;  ///< arg0
  std::int64_t return_size; ///< arg1 (# correction bits to fetch; the
                            ///<       cc.device_call lowering serializes the
                            ///<       OUT std::vector<bool> length here)
  std::uint8_t reset;       ///< arg2 (0 = keep state, 1 = reset after read;
                            ///<       trailing bool, no padding)
};
static_assert(sizeof(GetCorrectionsRequestPayload) == 17,
              "GetCorrectionsRequestPayload must be exactly 17 bytes per "
              "decoder_server_runtime.md#get_corrections");

struct __attribute__((packed)) ResetRequestPayload {
  std::int64_t decoder_id; ///< arg0
};
static_assert(sizeof(ResetRequestPayload) == 8,
              "ResetRequestPayload must be exactly 8 bytes per "
              "decoder_server_runtime.md#reset_decoder");

#ifdef __CUDACC__
#define CUDAQX_RPC_HD __host__ __device__
#else
#define CUDAQX_RPC_HD
#endif

CUDAQX_RPC_HD constexpr std::size_t bit_packed_bytes(std::size_t num_bits) {
  return (num_bits + 7) / 8;
}

CUDAQX_RPC_HD constexpr std::size_t align_to_8(std::size_t bytes) {
  return (bytes + 7) & ~static_cast<std::size_t>(7);
}

#undef CUDAQX_RPC_HD

// Maximum number of decoders the realtime dispatch can register at once.  GPU
// dispatch implementations use it to size their device-side state tables; host
// session code uses it for early range checks without depending on private GPU
// dispatch headers.
inline constexpr std::size_t kMaxDispatchedDecoders = 32;

} // namespace cudaq::qec::decoding::rpc
