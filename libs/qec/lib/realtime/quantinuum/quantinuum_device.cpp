/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <vector>

// We don't want all of CUDA-Q here...
#include "quantinuum_decoding.h"
#include "cudaq/driver/device.h"
#include "cudaq/qis/qubit_qis.h"

namespace cudaq::qec::decoding {

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {
  uint64_t syndrome_size = syndromes.size();
  // Discriminate first, then bit-pack into the `uint64` wire format.
  uint64_t syndrome = cudaq::to_integer(cudaq::to_bools(syndromes));
  cudaq::device_call(enqueue_syndromes_ui64, decoder_id, syndrome_size,
                     syndrome, tag);
}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {
  uint64_t syndrome_size = syndromes.size();
  uint64_t syndrome = cudaq::to_integer(syndromes);
  cudaq::device_call(enqueue_syndromes_ui64, decoder_id, syndrome_size,
                     syndrome, tag);
}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  std::vector<bool> result(return_size);
  auto ic = cudaq::device_call(get_corrections_ui64, decoder_id, return_size,
                               static_cast<uint64_t>(reset));
  for (std::size_t i = 0; i < return_size; i++)
    result[i] = (ic >> i) & 1;
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {
  cudaq::device_call(reset_decoder_ui64, decoder_id);
}

} // namespace cudaq::qec::decoding
