// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hsb_playback_loop.h"

#include <gtest/gtest.h>
#include <limits>

using namespace cudaq::qec::hsb_playback;

TEST(HSBPlaybackLoop, CountsCompletedMeasurements) {
  const auto summary = make_measurement_run_summary({3000, 3000}, 3, 1000);
  EXPECT_EQ(summary.playback_frames, 3000U);
  EXPECT_EQ(summary.response_frames, 3000U);
  EXPECT_EQ(summary.completed_measurements, 1000U);
  EXPECT_EQ(summary.partial_measurement_frames, 0U);
  EXPECT_TRUE(summary.all_responses_received);
  EXPECT_TRUE(measurement_run_passed(summary));
}

TEST(HSBPlaybackLoop, PartialTailDoesNotCountTowardMinimum) {
  const auto summary = make_measurement_run_summary({3002, 3002}, 3, 1000);
  EXPECT_EQ(summary.completed_measurements, 1000U);
  EXPECT_EQ(summary.partial_measurement_frames, 2U);
  EXPECT_TRUE(measurement_run_passed(summary));
  EXPECT_FALSE(measurement_run_passed(
      make_measurement_run_summary({3002, 3002}, 3, 1001)));
}

TEST(HSBPlaybackLoop, EmptyOrPartialOnlyRunFails) {
  EXPECT_FALSE(measurement_run_passed(make_measurement_run_summary({0, 0}, 3)));
  EXPECT_FALSE(measurement_run_passed(make_measurement_run_summary({2, 2}, 3)));
}

TEST(HSBPlaybackLoop, RequiresOneResponsePerFrame) {
  for (const auto responses : {std::uint64_t{2999}, std::uint64_t{3001}}) {
    const auto summary =
        make_measurement_run_summary({3000, responses}, 3, 999);
    EXPECT_FALSE(summary.all_responses_received);
    EXPECT_FALSE(measurement_run_passed(summary));
  }
}

TEST(HSBPlaybackLoop, TransportOrRPCFailuresFailAcceptance) {
  EXPECT_FALSE(measurement_run_passed(
      make_measurement_run_summary({3000, 3000, 1, 0, 0}, 3, 1000)));
  EXPECT_FALSE(measurement_run_passed(
      make_measurement_run_summary({3000, 3000, 0, 1, 0}, 3, 1000)));
  EXPECT_FALSE(measurement_run_passed(
      make_measurement_run_summary({3000, 3000, 0, 0, 1}, 3, 1000)));
}

TEST(HSBPlaybackLoop, RejectsZeroFramesPerMeasurement) {
  EXPECT_THROW(make_measurement_run_summary({3000, 3000}, 0),
               std::invalid_argument);
}

TEST(HSBPlaybackLoop, UnrequestedMinimumStillRequiresCompletedMeasurement) {
  EXPECT_TRUE(measurement_run_passed(make_measurement_run_summary({3, 3}, 3)));
  EXPECT_FALSE(measurement_run_passed(make_measurement_run_summary({3, 3}, 4)));
}

TEST(HSBPlaybackLoop, UsesFullWidthCountersAndMinimum) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  const auto summary =
      make_measurement_run_summary({maximum, maximum}, 2, maximum / 2);
  EXPECT_EQ(summary.completed_measurements, maximum / 2);
  EXPECT_EQ(summary.partial_measurement_frames, 1U);
  EXPECT_TRUE(measurement_run_passed(summary));
  EXPECT_FALSE(measurement_run_passed(
      make_measurement_run_summary({maximum, maximum}, 2, maximum)));
  EXPECT_TRUE(measurement_run_passed(
      make_measurement_run_summary({maximum, maximum}, 1, maximum)));
}

TEST(HSBPlaybackLoop, FullWidthDivisorDoesNotOverflow) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  const auto complete =
      make_measurement_run_summary({maximum, maximum}, maximum, 1);
  EXPECT_EQ(complete.completed_measurements, 1U);
  EXPECT_EQ(complete.partial_measurement_frames, 0U);
  EXPECT_TRUE(measurement_run_passed(complete));
  const auto partial =
      make_measurement_run_summary({maximum - 1, maximum - 1}, maximum, 1);
  EXPECT_EQ(partial.completed_measurements, 0U);
  EXPECT_EQ(partial.partial_measurement_frames, maximum - 1);
  EXPECT_FALSE(measurement_run_passed(partial));
}
