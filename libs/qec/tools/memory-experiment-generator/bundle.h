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

namespace cudaq::qec::memory_experiment {

struct bundle_options {
  std::uint64_t batch_rounds = 1;
  std::uint64_t replay_shots = 0;
  std::uint64_t seed = 1;
};

/// Publish model.yaml and manifest.yaml in a new directory, with optional raw
/// replay files. Never overwrites an existing directory. No runnable server
/// configuration is emitted.
void write_model_bundle(const preparation_request &request,
                        const prepared_model &model,
                        const bundle_options &options,
                        const std::filesystem::path &output);

/// Check the versioned manifest, file integrity, bounded phase-model topology,
/// raw-record layout and readout map. Does not construct or execute a graph.
void validate_model_bundle(const std::filesystem::path &directory);

} // namespace cudaq::qec::memory_experiment
