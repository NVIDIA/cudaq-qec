/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Definitions for the generic device_call targets referenced by
// simulation_cqr_device.cpp. These exist for two reasons:
//   1. To satisfy the link: the -frealtime-lowering'd device stubs still emit
//   an
//      undefined reference to the callee symbol that the final link must
//      resolve.
//   2. To service the direct/emulate channel, where the device_call resolves
//      straight to these (the host-dispatch channel instead routes by
//      function_id to the service in decoding_server_cqr.cpp and never calls
//      these).
//
// The wire arguments are the spec's on-wire form (bit-packed byte vector
// last, explicit bit-count scalars); these trampolines unpack to the
// byte-per-bool buffers the host:: decoding API takes, mirroring what the
// host-dispatch service does on the server side.
//
// They are given HIDDEN visibility: the device stubs resolve them within this
// library, but the deliberately generic names (enqueue_syndromes, ...) are NOT
// exported, so they cannot collide with symbols in other libraries.

#include "../realtime_decoding.h"

#include <cstdint>
#include <vector>

namespace {
// nvq++ lowers a std::vector<uint8_t> (or std::vector<bool>) device_call
// argument to a {pointer, length} span; the trampolines must accept that
// layout (same trick as the simulation backend's boolean_span).
struct byte_span {
  std::uint8_t *buffer;
  std::uint64_t length;
};
} // namespace

extern "C" __attribute__((visibility("hidden"))) void
enqueue_syndromes(std::uint64_t decoder_id, std::uint64_t counter,
                  std::uint64_t syndrome_mapping_id, byte_span syndrome_bits) {
  // No syndrome mapping table yet: syndrome_mapping_id 0 is the identity
  // mapping. syndrome_bits is a std::vector<bool> span: `length` is the logical
  // bit count (== num_syndromes) and `buffer` points at the LSB-first
  // bit-packed bytes.
  (void)syndrome_mapping_id;
  const std::uint64_t num_syndromes = syndrome_bits.length;
  std::vector<std::uint8_t> bits(num_syndromes);
  for (std::uint64_t i = 0; i < num_syndromes; ++i)
    bits[i] = (syndrome_bits.buffer[i / 8] >> (i % 8)) & 1; // LSB-first
  cudaq::qec::decoding::host::enqueue_syndromes(decoder_id, bits.data(),
                                                num_syndromes, counter);
}

extern "C" __attribute__((visibility("hidden"))) void
get_corrections(std::uint64_t decoder_id, byte_span corrections, bool reset) {
  // corrections is a std::vector<bool> span: `length` is the logical bit count
  // (the return size) and `buffer` points at the LSB-first bit-packed bytes.
  const std::uint64_t return_size = corrections.length;
  std::vector<std::uint8_t> bits(return_size);
  cudaq::qec::decoding::host::get_corrections(decoder_id, bits.data(),
                                              return_size, reset);
  std::uint64_t num_bytes = (return_size + 7) / 8;
  for (std::uint64_t byte = 0; byte < num_bytes; ++byte) {
    std::uint8_t value = 0;
    for (std::uint64_t bit = 0; bit < 8; ++bit) {
      std::uint64_t index = byte * 8 + bit;
      if (index < return_size && bits[index])
        value |= static_cast<std::uint8_t>(1u << bit); // LSB-first
    }
    corrections.buffer[byte] = value;
  }
}

extern "C" __attribute__((visibility("hidden"))) void
reset_decoder(std::uint64_t decoder_id) {
  cudaq::qec::decoding::host::reset_decoder(decoder_id);
}
