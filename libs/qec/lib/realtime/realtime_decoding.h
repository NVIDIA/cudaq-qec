/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cstdint>
#include <filesystem>
#include <memory>

// Note: none of these are intended to be user-facing functions.

namespace cudaq::qec::decoding::host {

__attribute__((visibility("default"))) void
enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag);

__attribute__((visibility("default"))) cudaqx::heterogeneous_map
prepare_decoder_params(
    const cudaq::qec::decoding::config::decoder_config &decoder_config);

/// Resolve a decoder configuration's model into construction inputs.
///
/// Selects the one authoritative model source, reads and parses a raw Stim DEM
/// when `stim_dem_path` is set, builds the canonical measurement-to-detector
/// map, and validates dimensions and any supplied assertions. Performs no
/// side effects: it allocates no decoder, touches no process state, and can be
/// called for every entry of a configuration before any of them is applied.
///
/// @param base_dir Directory a relative `stim_dem_path` resolves against. The
/// configuration file's parent directory for a file-based configuration, or
/// the process working directory for a programmatic or raw-string one.
/// @throws std::runtime_error on any resolution or validation failure.
__attribute__((visibility("default"))) cudaq::qec::decoder_init
resolve_decoder_init(
    const cudaq::qec::decoding::config::decoder_config &decoder_config,
    const std::filesystem::path &base_dir);

/// Construct and initialize one decoder for realtime use from already-resolved
/// inputs. The returned decoder is fully configured with its ID and D matrix,
/// but is not installed in a process-global registry or attached to a worker
/// thread.
///
/// @throws std::invalid_argument if the decoder ID cannot be represented.
/// @throws std::runtime_error if decoder construction/initialization fails.
__attribute__((visibility("default"))) std::unique_ptr<cudaq::qec::decoder>
create_realtime_decoder(
    const cudaq::qec::decoding::config::decoder_config &decoder_config,
    cudaq::qec::decoder_init inputs);

__attribute__((visibility("default"))) void
get_corrections(std::size_t decoder_id, uint8_t *corrections,
                std::uint64_t correction_length, bool reset);

__attribute__((visibility("default"))) void
reset_decoder(std::size_t decoder_id);

/// Apply a configuration: resolve every entry's model, then construct and
/// install the decoders.
/// @param base_dir Directory relative model paths resolve against.
int configure_decoders(
    cudaq::qec::decoding::config::multi_decoder_config &config,
    const std::filesystem::path &base_dir);
int configure_decoders_from_file(const char *config_file);
int configure_decoders_from_str(const char *config_str);
void finalize_decoders();

/// @brief Set a callback to capture syndrome data as it's enqueued.
/// Used by --save_syndrome feature to record syndromes to file.
/// @param callback Function pointer that receives packed syndrome bytes.
///                 Set to nullptr to disable capture.
__attribute__((visibility("default"))) void
_set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t));

/// @brief The currently registered syndrome-capture callback (nullptr if
/// none). Served decode paths that bypass host::enqueue_syndromes (the
/// decoding-server service) use this to keep --save_syndrome working.
__attribute__((visibility("default"))) void (*_get_syndrome_capture_callback())(
    const uint8_t *, size_t);

} // namespace cudaq::qec::decoding::host
