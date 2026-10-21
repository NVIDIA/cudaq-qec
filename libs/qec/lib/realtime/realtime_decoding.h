/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoding_config.h"

namespace cudaq::qec {
class decoding_task_graph;
} // namespace cudaq::qec
#include <cstdint>
#include <filesystem>
#include <memory>

// Note: none of these are intended to be user-facing functions.

namespace cudaq::qec::decoding::host {

__attribute__((visibility("default"))) void
enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag);

/// Same, naming the syndrome_mapping row this round belongs to, as the __qpu__
/// signature and the wire both do.  `tag` says which round, the id says what
/// kind.  A shot ends by naming a row declared `type: final` -- the shape a
/// guarded window reads to commit its withheld tail.
///
/// The id travels verbatim; an undeclared one throws, as the dispatcher
/// BAD_REQUESTs it.  The parameter is an id rather than a shape because rows
/// can share a shape: only an id identifies one.
__attribute__((visibility("default"))) void
enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag,
                  std::uint64_t syndrome_mapping_id);

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
    cudaq::qec::decoder_init inputs,
    cudaq::qec::decoder_output_request output =
        cudaq::qec::decoder_output_request{
            cudaq::qec::decode_result_type::observables});

/// Build the decoding_task_graph a `task_graph` block describes, in either the
/// single-node or the `nodes`/`edges` form. The counterpart to
/// create_realtime_decoder(): `decoder_config.task_graph` selects which of the
/// two applies to an entry.
///
/// @throws std::runtime_error if the block names an unsupported node kind or
/// leaves the graph without the matrices it needs.
__attribute__((visibility("default")))
std::unique_ptr<cudaq::qec::decoding_task_graph>
build_task_graph(
    const cudaq::qec::decoding::config::decoder_config &decoder_config,
    const std::filesystem::path &base_dir = std::filesystem::current_path());

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

/// @brief Register a decoding_task_graph at the given decoder slot.
///
/// Resizes `g_task_graphs` and `g_decoders` to accommodate `id` if needed,
/// places `graph` at `g_task_graphs[id]`, and nulls `g_decoders[id]` so the
/// task-graph dispatch path is taken for that slot.  Must not be called while
/// a realtime session is active.  For tests and integration code that build a
/// graph in C++; a `task_graph` block in the configuration reaches the same
/// slot through `configure_decoders`. The graph must support streaming.
///
/// @throws std::invalid_argument if graph is null or cannot stream.
__attribute__((visibility("default"))) void
register_task_graph(std::size_t id,
                    std::unique_ptr<cudaq::qec::decoding_task_graph> graph);

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
