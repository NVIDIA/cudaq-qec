/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qis/qubit_qis.h"

// Define the CUDA-Q QEC Realtime Decoding API
//
// These functions enable CUDA-Q quantum kernel code to
// offload decoding work to our QEC decoders in real time
// (within qubit coherence times)
//
// The design here is as follows: We declare but do not
// implement the API. Then we allow users to specify concrete
// implementations of the API via the target specification passed to
// nvq++.

namespace cudaq::qec::decoding {
// CUDA-Q QEC Realtime Decoding API (declarations)

/// @brief Enqueue syndromes for decoding.
/// @param decoder_id The ID of the decoder to use.
/// @param syndromes The syndromes to enqueue.
/// @param tag The tag to use for the syndrome (currently useful for logging
/// only)
__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag = 0);

/// @brief Test-only entry point for enqueueing pre-discriminated syndrome
/// bits (e.g., raw syndromes replayed from a saved file). Production code
/// must use `enqueue_syndromes`. This entry point exists because raw bits
/// read from disk have no measurement-event identity to preserve, so the
/// `vector<measure_result>` API is the wrong shape for them.
/// @param decoder_id The ID of the decoder to use.
/// @param syndromes Pre-discriminated syndrome bits.
/// @param tag The tag to use for the syndrome (currently useful for logging
/// only)
__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag = 0);

/// @brief Get the corrections for a given decoder.
/// @param decoder_id The ID of the decoder to use.
/// @param return_size The number of bits to return (in bits). This is expected
/// to match the number of observables in the decoder.
/// @param reset Whether to reset the decoder corrections after retrieving them.
/// @return The corrections (detected bit flips) for the given decoder, based on
/// all of the decoded syndromes since the last time any corrections were reset.
__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset = false);

/// @brief Reset the decoder. This clears any queued syndromes and resets any
/// corrections back to 0.
/// @param decoder_id The ID of the decoder to reset.
__qpu__ void reset_decoder(std::uint64_t decoder_id);
} // namespace cudaq::qec::decoding
