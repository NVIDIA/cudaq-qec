/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simulation_host.h"
#include "../realtime_decoding.h"
#include <cstdint>

namespace cudaq::qec::decoding::simulation {

// WARNING: This is playing with very dangerous fire!
// Here, we must lie about the signature of the function. nvq++ will convert
// vectors to spans. If the corresponding implementation does not implement this
// exact signature (other than converting vectors to spans), there will be no
// errors or warnings, other than bad crashes.
extern "C" __attribute__((visibility("default"))) void
simulation_enqueue_syndromes(std::uint64_t decoder_id, boolean_span syndromes,
                             std::uint64_t tag) {
  cudaq::qec::decoding::host::enqueue_syndromes(decoder_id, syndromes.buffer,
                                                syndromes.length, tag);
}

extern "C" __attribute__((visibility("default"))) void
simulation_get_corrections(std::uint64_t decoder_id, boolean_span corrections,
                           bool reset) {
  cudaq::qec::decoding::host::get_corrections(decoder_id, corrections.buffer,
                                              corrections.length, reset);
}

__attribute__((visibility("default"))) void
enqueue_syndromes(std::uint64_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag) {
  cudaq::qec::decoding::host::enqueue_syndromes(decoder_id, syndromes,
                                                syndrome_length, tag);
}

__attribute__((visibility("default"))) void
get_corrections(std::uint64_t decoder_id, uint8_t *corrections,
                std::uint64_t correction_length, bool reset) {
  cudaq::qec::decoding::host::get_corrections(decoder_id, corrections,
                                              correction_length, reset);
}

__attribute__((visibility("default"))) void
reset_decoder(std::uint64_t decoder_id) {
  cudaq::qec::decoding::host::reset_decoder(decoder_id);
}

} // namespace cudaq::qec::decoding::simulation
