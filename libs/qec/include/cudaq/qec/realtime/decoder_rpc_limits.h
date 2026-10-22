/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Size limits shared by the realtime decoding host code and the GPU dispatch
// wire format. This header must stay free of CUDA and cuda-quantum realtime
// includes: it is consumed by the config parser in cudaq-qec-realtime-decoding,
// which is built (and shipped in the Python wheel) in environments where the
// cuda-quantum realtime headers do not exist. Anything that needs the RPC wire
// types belongs in decoder_rpc_wire_format.h instead.

#include <cstddef>
#include <cstdint>

namespace cudaq::qec::decoding::rpc {

// Hard cap for one enqueue_syndromes request. Enforced at both transport and
// session boundaries so alternate transports cannot bypass allocation and
// packed-length validation.
inline constexpr std::uint64_t kMaxSyndromeBits = 1u << 20; // 1 M bits

// Maximum number of decoders the realtime dispatch can register at once.  GPU
// dispatch implementations use it to size their device-side state tables; host
// session code uses it for early range checks without depending on private GPU
// dispatch headers.
inline constexpr std::size_t kMaxDispatchedDecoders = 32;

// syndrome_mapping ids are bounded to [0, kMaxSyndromeMappingRows), like
// kMaxDispatchedDecoders: GPU dispatch sizes its device table with it, and host
// code fills that table without private GPU dispatch headers.
//
// Bounding ids rather than row count keeps the dispatch modes agreeing --
// validation refuses an out-of-range document under HOST too -- and makes the
// device table a dense array indexed by id.
inline constexpr std::size_t kMaxSyndromeMappingRows = 16;

} // namespace cudaq::qec::decoding::rpc
