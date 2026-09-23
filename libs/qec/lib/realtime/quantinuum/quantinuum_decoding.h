/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

// Note the "__attribute__((visibility("default")))" is needed for local tests
// that compile and run with "--target quantinuum --emulate". They are not
// technically needed for a deployed server library.

extern "C" {
/// @brief Enqueue a syndrome for decoding.
/// @param decoder_id The ID of the decoder to use.
/// @param syndrome_size The size of the syndrome (in bits). This must be <= 64.
/// @param syndrome The bit-packed syndrome to enqueue. The least significant
/// bit (i.e. syndrome & 1) is the first bit of the syndrome. The last valid bit
/// is `syndrome_size - 1` (i.e. syndrome & (1 << (syndrome_size - 1)).
/// @param tag The tag to use for the syndrome (logging only)
__attribute__((visibility("default"))) void
enqueue_syndromes_ui64(uint64_t decoder_id, uint64_t syndrome_size,
                       uint64_t syndrome, uint64_t tag);

// Private variant for alternative testing (closer to in-system).
__attribute__((visibility("default"))) void
enqueue_syndromes_ui64_private(uint64_t decoder_id, uint64_t syndrome_size,
                               uint64_t syndrome, uint64_t tag);

/// @brief Get the corrections for a given decoder.
/// @param decoder_id The ID of the decoder to use.
/// @param return_size The number of bits to return (in bits). This must be
/// <= 64. This is expected to match the number of observables in the decoder.
/// The least significant bit (i.e. return_value & 1) is the first bit of the
/// corrections. The last valid bit is `return_size - 1`.
/// @param reset Whether to reset the decoder corrections after retrieving them.
/// @return The corrections (detected bit flips) for the given decoder, based on
/// all of the decoded syndromes since the last time any corrections were reset.
__attribute__((visibility("default"))) std::uint64_t
get_corrections_ui64(uint64_t decoder_id, uint64_t return_size, uint64_t reset);

// Private variant for alternative testing (closer to in-system).
__attribute__((visibility("default"))) std::uint64_t
get_corrections_ui64_private(uint64_t decoder_id, uint64_t return_size,
                             uint64_t reset);

/// @brief Reset the decoder. This clears any queued syndromes and resets any
/// corrections back to 0.
/// @param decoder_id The ID of the decoder to reset.
__attribute__((visibility("default"))) void
reset_decoder_ui64(uint64_t decoder_id);

// Private variant for alternative testing (closer to in-system).
__attribute__((visibility("default"))) void
reset_decoder_ui64_private(uint64_t decoder_id);
}
