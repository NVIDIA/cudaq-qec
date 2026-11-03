/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "model_provider.h"
#include <cstdint>
#include <functional>
#include <string>

namespace cudaq::qec::memory_experiment {

/// Qualification-only sample; not part of the installed generator interface.
struct raw_replay_shot {
  std::uint64_t patch;
  std::uint64_t shot;
  std::uint64_t seed;
  /// Chronological extraction records followed by final data-qubit readout.
  std::string measurements;
  /// Final readout multiplied by final_m2o, not a decoded correction.
  std::string raw_logical;
};

/// Validate the deterministic seed range without preparing a circuit. Zero
/// shots is a no-op; otherwise patch count and base seed must be positive.
void validate_raw_replay_options(std::uint64_t patches, std::uint64_t shots,
                                 std::uint64_t base_seed);

/// Sample the existing compiled memory kernel once per complete shot and pass
/// each result to consume without retaining the complete replay. Code geometry
/// is prepared once per invocation; memory scales with one shot, not the number
/// of shots or independent patches. The test helper prepares code schedules
/// for each shot without characterizing a circuit or constructing a DEM.
/// References passed to consume remain valid only for that callback.
///
/// Seeds are base_seed + patch * shots_per_patch + shot. A positive base is
/// required because CUDA-Q seed zero requests unseeded sampling. The range and
/// model geometry are validated before any sampling. Zero shots is a no-op,
/// including when consume is empty. Sampling calls through this helper
/// serialize access to CUDA-Q's global RNG and noise state. Each shot replaces
/// any existing noise model and clears it on success or failure; callers must
/// not change global noise or the seed concurrently through other APIs. No
/// decoding, expected A/B corrections, or accuracy evaluation is performed. The
/// active backend must support this circuit's noise model.
void sample_raw_replay(
    const preparation_request &request, const prepared_model &model,
    std::uint64_t shots_per_patch, std::uint64_t base_seed,
    const std::function<void(const raw_replay_shot &)> &consume);

} // namespace cudaq::qec::memory_experiment
