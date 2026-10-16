/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Backend-neutral device-side wrappers for the QEC realtime decoding API.
//
// These mirror the simulation backend's __qpu__ wrappers, but the cudaq::
// device_call targets use the *generic* callee names defined by
// cudaq-spec/proposals/decoder_server_runtime.md -- enqueue_syndromes /
// get_corrections / reset_decoder -- rather than the simulation backend's
// simulation_* / C++-mangled symbols. The realtime function_id is
// fnv1a_32(<callee name>), so these names are what the host-dispatch service
// (decoding_server_cqr.cpp) registers.
//
// The wrappers also perform the spec's on-wire rewrite themselves (until the
// device_call lowering can): syndrome/correction bits cross the wire
// bit-packed (LSB-first) in a std::vector<uint8_t> placed last, with the
// explicit num_syndromes / return_size scalars of the spec's
// enqueue_syndromes / get_corrections wire layouts. The stdvec<u8>
// serialization's uint64 byte-count prefix is the spec's ARRAY_UINT8 prefix.
//
// This translation unit is compiled with -frealtime-lowering (see the
// cudaq-qec-realtime-decoding-simulation-cqr library), which rewrites each
// device_call to the realtime frame ABI and dispatches it by function_id
// through the host-dispatch channel. The callee names below are only hashed
// into the function_id; the symbols themselves are defined in
// simulation_cqr_host.cpp with HIDDEN visibility (purely to satisfy the link),
// so the very generic names are never exported and cannot collide with other
// libraries.

#include <cstdint>
#include <vector>

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"

// device_call targets, in the spec's on-wire argument order (variable-length
// byte vector last). Named to match the spec function_ids. Defined (hidden) in
// simulation_cqr_host.cpp.
extern "C" {
void enqueue_syndromes(std::uint64_t decoder_id, std::uint64_t counter,
                       std::uint64_t syndrome_mapping_id,
                       const std::vector<bool> &syndrome_bits);

void get_corrections(std::uint64_t decoder_id, std::vector<bool> &corrections,
                     bool reset);

void reset_decoder(std::uint64_t decoder_id);
}

namespace cudaq::qec::decoding {

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {
  // No syndrome mapping table yet; id 0 is the identity mapping.
  constexpr std::uint64_t kSyndromeMappingId = 0;
  // Pass the syndrome bits as a std::vector<bool>; the realtime device_call
  // lowering bit-packs it LSB-first into a CUDAQ_TYPE_BIT_PACKED payload (cudaq
  // PR 4816) and serializes its length as an element-count prefix.  That prefix
  // IS num_syndromes, so no separate num_syndromes argument is passed.
  std::size_t num_bits = syndromes.size();
  std::vector<bool> bits(num_bits);
  for (std::size_t i = 0; i < num_bits; ++i)
    bits[i] = syndromes[i];
  // device_id == decoder_id: each decoder gets its own device_call session,
  // i.e. its own ring buffer and its own dispatcher (the runtime keys
  // sessions, rings, and channels by device id).  The payload still carries
  // decoder_id as a cross-check for fan-in topologies.
  cudaq::device_call(decoder_id, ::enqueue_syndromes, decoder_id, tag,
                     kSyndromeMappingId, bits);
}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {
  constexpr std::uint64_t kSyndromeMappingId = 0;
  // syndromes is already a std::vector<bool>; the lowering bit-packs it and
  // serializes its length as the element-count prefix (== num_syndromes).
  cudaq::device_call(decoder_id, ::enqueue_syndromes, decoder_id, tag,
                     kSyndromeMappingId, syndromes);
}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  // The OUT corrections vector is a std::vector<bool>: the realtime lowering
  // serializes its length as an element (bit) count and unpacks the BIT_PACKED
  // response straight back into it (cudaq PR 4816).  That length IS the return
  // size, so no separate return_size argument is passed (matches
  // simulation_device.cpp).  reset is a trailing bool, so the request payload
  // is 17 bytes with no trailing padding.
  std::vector<bool> result(return_size);
  cudaq::device_call(decoder_id, ::get_corrections, decoder_id, result, reset);
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {
  cudaq::device_call(decoder_id, ::reset_decoder, decoder_id);
}

} // namespace cudaq::qec::decoding
