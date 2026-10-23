/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experiment_spec.h"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace cudaq::qec::memory_experiment {
namespace {

constexpr auto max_count = std::numeric_limits<std::uint64_t>::max();

TEST(MemoryExperimentSpec, AcceptsBothBasesAndIndependentPatches) {
  for (auto basis : {memory_basis::x, memory_basis::z})
    for (std::uint64_t distance : {3, 5, 11}) {
      experiment_spec spec{"surface_code", distance, 26, 4, basis};
      EXPECT_NO_THROW(validate_experiment_spec(spec));
      spec.rounds = 1;
      EXPECT_NO_THROW(validate_experiment_spec(spec));
      spec.rounds = max_count - 1;
      EXPECT_NO_THROW(validate_experiment_spec(spec));
    }
}

TEST(MemoryExperimentSpec, RejectsUnsupportedExperimentInputs) {
  const experiment_spec valid{"surface_code", 3, 26, 1, memory_basis::z};
  for (std::uint64_t distance : {0, 1, 2, 4, 10}) {
    auto spec = valid;
    spec.distance = distance;
    EXPECT_THROW(validate_experiment_spec(spec), std::invalid_argument);
  }
  auto spec = valid;
  spec.code_family = "color_code";
  EXPECT_THROW(validate_experiment_spec(spec), std::invalid_argument);
  spec = valid;
  spec.rounds = 0;
  EXPECT_THROW(validate_experiment_spec(spec), std::invalid_argument);
  spec = valid;
  spec.logical_qubits = 0;
  EXPECT_THROW(validate_experiment_spec(spec), std::invalid_argument);
  spec = valid;
  spec.basis = static_cast<memory_basis>(2);
  EXPECT_THROW(validate_experiment_spec(spec), std::invalid_argument);
  spec = valid;
  spec.rounds = max_count;
  EXPECT_THROW(validate_experiment_spec(spec), std::overflow_error);
}

TEST(MemoryRecordLayout, ReferenceLayoutPreservesTerminalReadoutAndLookback) {
  const memory_record_layout layout(26, 8, 9);
  EXPECT_EQ(layout.extraction_rounds(), 26);
  EXPECT_EQ(layout.input_rounds(), 27);
  EXPECT_EQ(layout.raw_measurements(), 217);
  const auto delivery = layout.batches(3);
  EXPECT_EQ(delivery.regular_batches, 8);
  EXPECT_EQ(delivery.regular_batch_measurements, 24);
  EXPECT_EQ(delivery.terminal_extraction_rounds, 2);
  EXPECT_EQ(delivery.terminal_batch_measurements, 25);
  // These raw-round ranges come from the reference D mapping, not a scheduler.
  const auto b0 = layout.measurements_for_rounds(2, 12);
  EXPECT_EQ(b0.begin, 16);
  EXPECT_EQ(b0.end, 96);
  EXPECT_EQ(b0.size(), 80);
  const auto a2 = layout.measurements_for_rounds(20, 27);
  EXPECT_EQ(a2.begin, 160);
  EXPECT_EQ(a2.end, 217);
  EXPECT_EQ(a2.size(), 57);
}

TEST(MemoryRecordLayout, HalfOpenRangesIncludeEmptyAndReadoutOnly) {
  const memory_record_layout layout(26, 8, 9);
  EXPECT_EQ(layout.measurements_for_rounds(0, 27).size(), 217);
  EXPECT_EQ(layout.measurements_for_rounds(0, 26).size(), 208);
  const auto final = layout.measurements_for_rounds(26, 27);
  EXPECT_EQ(final.begin, 208);
  EXPECT_EQ(final.end, 217);
  for (std::uint64_t round : {0, 13, 26, 27}) {
    const auto empty = layout.measurements_for_rounds(round, round);
    EXPECT_EQ(empty.begin, empty.end);
  }
  EXPECT_EQ(layout.measurements_for_rounds(27, 27).begin, 217);
  EXPECT_THROW(layout.measurements_for_rounds(1, 0), std::out_of_range);
  EXPECT_THROW(layout.measurements_for_rounds(0, 28), std::out_of_range);
  EXPECT_THROW(layout.measurements_for_rounds(28, 28), std::out_of_range);
  EXPECT_THROW(layout.measurements_for_rounds(max_count, max_count),
               std::out_of_range);
}

TEST(MemoryRecordLayout, RejectsZeroCountsAndOverflowBeforeUsingOffsets) {
  EXPECT_THROW(memory_record_layout(0, 8, 9), std::invalid_argument);
  EXPECT_THROW(memory_record_layout(26, 0, 9), std::invalid_argument);
  EXPECT_THROW(memory_record_layout(26, 8, 0), std::invalid_argument);
  EXPECT_THROW(memory_record_layout(max_count, 1, 1), std::overflow_error);
  EXPECT_THROW(memory_record_layout(max_count / 2 + 1, 2, 1),
               std::overflow_error);
  EXPECT_THROW(memory_record_layout(max_count / 2, 2, 2), std::overflow_error);
  const memory_record_layout layout(26, 8, 9);
  EXPECT_THROW(layout.batches(0), std::invalid_argument);
}

TEST(MemoryRecordLayout, ExactUint64BoundaryRemainsRepresentable) {
  const memory_record_layout layout(max_count / 2, 2, 1);
  EXPECT_EQ(layout.raw_measurements(), max_count);
  EXPECT_EQ(layout.measurements_for_rounds(0, layout.input_rounds()).end,
            max_count);
  const auto final = layout.measurements_for_rounds(layout.extraction_rounds(),
                                                    layout.input_rounds());
  EXPECT_EQ(final.begin, max_count - 1);
  EXPECT_EQ(final.size(), 1);
  const auto delivery = layout.batches(max_count / 2);
  EXPECT_EQ(delivery.regular_batches, 1);
  EXPECT_EQ(delivery.regular_batch_measurements, max_count - 1);
  EXPECT_EQ(delivery.terminal_batch_measurements, 1);
}

TEST(MemoryRecordLayout, MaximumInputRoundCountHasValidEndpoint) {
  const memory_record_layout layout(max_count - 1, 1, 1);
  EXPECT_EQ(layout.input_rounds(), max_count);
  EXPECT_EQ(layout.raw_measurements(), max_count);
  EXPECT_EQ(layout.measurements_for_rounds(max_count, max_count).begin,
            max_count);
  EXPECT_EQ(layout.measurements_for_rounds(max_count - 1, max_count).size(), 1);
}

TEST(MemoryRecordLayout, AlignedBatchesKeepReadoutInItsOwnTerminalBatch) {
  const memory_record_layout layout(24, 8, 9);
  const auto delivery = layout.batches(3);
  EXPECT_EQ(delivery.regular_batches, 8);
  EXPECT_EQ(delivery.regular_batch_measurements, 24);
  EXPECT_EQ(delivery.terminal_extraction_rounds, 0);
  EXPECT_EQ(delivery.terminal_batch_measurements, 9);
}

TEST(MemoryRecordLayout, OversizedBatchDoesNotMultiplyAnUnusedWidth) {
  const memory_record_layout layout(26, 8, 9);
  for (auto batch : {std::uint64_t{27}, max_count}) {
    const auto delivery = layout.batches(batch);
    EXPECT_EQ(delivery.regular_batches, 0);
    EXPECT_EQ(delivery.regular_batch_measurements, 0);
    EXPECT_EQ(delivery.terminal_extraction_rounds, 26);
    EXPECT_EQ(delivery.terminal_batch_measurements, 217);
  }
}

TEST(MemoryRecordLayout, HugeHorizonNeedsNoPerRoundMaterialization) {
  const auto rounds = max_count / 8 - 1;
  const memory_record_layout layout(rounds, 8, 9);
  EXPECT_EQ(layout.raw_measurements(), rounds * 8 + 9);
  const auto delivery = layout.batches(max_count);
  EXPECT_EQ(delivery.regular_batches, 0);
  EXPECT_EQ(delivery.terminal_extraction_rounds, rounds);
  EXPECT_EQ(delivery.terminal_batch_measurements, layout.raw_measurements());
  EXPECT_EQ(layout.measurements_for_rounds(rounds, rounds + 1).size(), 9);
}

TEST(MemoryRecordLayout, BatchingConservesRecordsWithNarrowAndWideReadouts) {
  for (std::uint64_t rounds : {1, 26, 99, 1000})
    for (auto widths : {std::array<std::uint64_t, 2>{8, 9}, {24, 9}, {1, 32}}) {
      const memory_record_layout layout(rounds, widths[0], widths[1]);
      for (std::uint64_t batch : {1, 2, 3, 5, 1001}) {
        const auto delivery = layout.batches(batch);
        EXPECT_EQ(delivery.regular_batches * batch +
                      delivery.terminal_extraction_rounds,
                  rounds);
        EXPECT_EQ(delivery.regular_batches *
                          delivery.regular_batch_measurements +
                      delivery.terminal_batch_measurements,
                  layout.raw_measurements());
        EXPECT_EQ(layout.measurements_for_rounds(rounds, rounds + 1).size(),
                  widths[1]);
      }
    }
}

} // namespace
} // namespace cudaq::qec::memory_experiment
