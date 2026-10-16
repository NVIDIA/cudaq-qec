/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <vector>

#include "cudaq.h"
#include "simulation_host.h"
#include "cudaq/qec/realtime/decoding.h"

// WARNING: This is playing with very dangerous fire!
// Here, we must lie about the signature of the function. nvq++ will convert
// vectors to spans. If the corresponding implementation does not implement this
// exact signature (other than converting vectors to spans), there will be no
// errors or warnings, other than bad crashes.
extern "C" __attribute__((visibility("default"))) void
simulation_enqueue_syndromes(std::uint64_t decoder_id,
                             const std::vector<bool> &syndromes,
                             std::uint64_t tag);

extern "C" __attribute__((visibility("default"))) void
simulation_get_corrections(std::uint64_t decoder_id,
                           std::vector<bool> &corrections, bool reset);

namespace cudaq::qec::decoding {

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {
  // The host trampoline takes a bit-typed span (see WARNING above), so
  // discriminate the handles before crossing the boundary.
  cudaq::device_call(simulation_enqueue_syndromes, decoder_id,
                     cudaq::to_bools(syndromes), tag);
}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {
  cudaq::device_call(simulation_enqueue_syndromes, decoder_id, syndromes, tag);
}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  std::vector<bool> result(return_size);
  cudaq::device_call(simulation_get_corrections, decoder_id, result, reset);
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {
  cudaq::device_call(simulation::reset_decoder, decoder_id);
}

} // namespace cudaq::qec::decoding
