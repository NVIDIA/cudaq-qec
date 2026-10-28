/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "model_provider.h"
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::memory_experiment {

/// Public parallel-window configuration. Argument leaves are typed by the
/// registered decoder and pattern schemas.
struct server_config_options {
  std::string decoder;
  std::uint64_t commit_rounds = 0;
  std::uint64_t buffer_rounds = 0;
  std::optional<std::uint64_t> gap_rounds;
  std::vector<std::string> decoder_args;
  /// Pattern PATH=VALUE leaves override the profile's geometry and output
  /// defaults. The CLI rejects duplicate dedicated geometry options.
  std::vector<std::string> pattern_args;
  /// Pool workers per independent graph, excluding its coordinator. Unset
  /// leaves execution policy to the framework; this is not a shared budget.
  /// Sets the node's max_in_flight to the same count for each A/B layer.
  std::optional<std::uint64_t> workers_per_graph;
};

/// Select decomposition for a verified Surface Code model profile. Matching
/// uses graphlike columns; BP and LUT retain joint hyperedges. Other model
/// profiles need separate characterization and are not emitted by this tool.
bool requires_graphlike_memory_model(const server_config_options &options);

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
/// patches reference one named model and share a schema-validated pattern
/// recipe. Output defaults to corrections. No graph is constructed or executed.
std::string emit_server_config(const prepared_model &model,
                               const server_config_options &options,
                               std::uint64_t logical_qubits);

/// Validate the model and serialize before creating the destination
/// exclusively. Existing paths are never replaced, and the parent must exist.
/// Write failures remove only the file created by this call. Visibility is not
/// atomic: readers may observe an incomplete write, and interruption can leave
/// a partial file.
void write_server_config(const preparation_request &request,
                         const prepared_model &model,
                         const server_config_options &options,
                         const std::filesystem::path &config_file);

} // namespace cudaq::qec::memory_experiment
