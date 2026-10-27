// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace cudaq::qec::hsb_playback {

// Final software-emulator snapshot, excluding any finite ILA preflight.
struct LoopStats {
  std::uint64_t frames = 0;
  std::uint64_t responses = 0;
  std::uint32_t errors = 0;
  std::uint32_t timeouts = 0;
  std::uint32_t response_failures = 0;
};

struct MeasurementRunSummary {
  std::uint64_t playback_frames = 0;
  std::uint64_t response_frames = 0;
  std::uint64_t completed_measurements = 0;
  std::uint64_t partial_measurement_frames = 0;
  bool all_responses_received = false;
  std::uint32_t transport_errors = 0;
  std::uint32_t transport_timeouts = 0;
  std::uint32_t response_failures = 0;
  std::optional<std::uint64_t> requested_measurements;
};

inline MeasurementRunSummary make_measurement_run_summary(
    const LoopStats &stats, std::uint64_t frames_per_measurement,
    std::optional<std::uint64_t> requested_measurements = std::nullopt) {
  if (frames_per_measurement == 0)
    throw std::invalid_argument("Loop frames per measurement must be positive");
  // A timed stop may leave a prefix of the next shot. Count only complete
  // acknowledged groups. Division avoids overflowing a minimum*frames product.
  return {.playback_frames = stats.frames,
          .response_frames = stats.responses,
          .completed_measurements = stats.responses / frames_per_measurement,
          .partial_measurement_frames =
              stats.responses % frames_per_measurement,
          .all_responses_received = stats.frames == stats.responses,
          .transport_errors = stats.errors,
          .transport_timeouts = stats.timeouts,
          .response_failures = stats.response_failures,
          .requested_measurements = requested_measurements};
}

inline bool measurement_run_passed(const MeasurementRunSummary &summary) {
  return summary.completed_measurements != 0 &&
         summary.all_responses_received && summary.transport_errors == 0 &&
         summary.transport_timeouts == 0 && summary.response_failures == 0 &&
         (!summary.requested_measurements ||
          summary.completed_measurements >= *summary.requested_measurements);
}

} // namespace cudaq::qec::hsb_playback
