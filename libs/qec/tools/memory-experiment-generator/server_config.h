/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "model_provider.h"
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::qec::memory_experiment {

/// Public parallel-window configuration. Decoder arguments are CLI
/// path=value leaves, typed by the decoder's registered parameter schema.
struct server_config_options {
  std::string decoder;
  std::uint64_t commit_rounds = 0;
  std::uint64_t buffer_rounds = 0;
  std::optional<std::uint64_t> gap_rounds;
  std::vector<std::string> decoder_args;
  /// Pool workers per independent graph, excluding its coordinator. Unset
  /// leaves execution policy to the framework; this is not a shared budget.
  std::optional<std::uint64_t> workers_per_graph;
};

/// Checked total across independent patches, excluding coordinator threads.
/// An unset per-graph count leaves the total unspecified. Validate before
/// circuit preparation or allocating per-patch configuration entries.
std::optional<std::uint64_t>
configured_worker_total(const server_config_options &options,
                        std::uint64_t logical_qubits);

/// Check the linked public configuration capability, geometry and decoder
/// arguments before circuit preparation or output publication. This does not
/// construct a decoder or claim the parallel-window executor is available.
void validate_server_config_options(const server_config_options &options);

/// Serialize and reparse a complete public server configuration. Independent
/// patches reference one named model and explicitly request corrections, not
/// corrected raw logical values. No graph is constructed or executed.
std::string emit_server_config(const prepared_model &model,
                               const server_config_options &options,
                               std::uint64_t logical_qubits);

/// Check the emitted topology, independent patch identities and shared model
/// through the public parser, with the expected per-graph worker policy.
/// Does not construct or execute the task graph.
void validate_server_config(
    std::string_view yaml, const dem_chunks_spec &expected_model,
    std::uint64_t logical_qubits,
    std::optional<std::uint64_t> workers_per_graph = {});

/// Validate the model and serialize before publishing a configuration file.
/// Stage in the destination directory, then publish the complete file without
/// replacing any existing path. The parent must exist. Failures remove only
/// this call's temporary file; no output directory or unrelated file is erased.
void write_server_config(const preparation_request &request,
                         const prepared_model &model,
                         const server_config_options &options,
                         const std::filesystem::path &config_file);

} // namespace cudaq::qec::memory_experiment
