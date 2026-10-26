/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "experiment_spec.h"
#include <cstdint>
#include <stdexcept>

namespace cudaq::qec::memory_experiment {

/// A half-open range of chronological raw-measurement bit positions.
struct measurement_range {
  std::uint64_t begin;
  std::uint64_t end;

  std::uint64_t size() const { return end - begin; }
};

/// A replay's transport batching, per independent patch. Only the terminal
/// batch contains final data readout, after any remaining extraction rounds.
struct delivery_layout {
  std::uint64_t regular_batches;
  /// Zero when no regular batch exists.
  std::uint64_t regular_batch_measurements;
  std::uint64_t terminal_extraction_rounds;
  std::uint64_t terminal_batch_measurements;
};

/// Qualification-only ranges and batches over the production record counts.
/// These describe raw-record delivery, not a detector map or A/B scheduler.
class replay_record_layout {
public:
  replay_record_layout(std::uint64_t extraction_rounds,
                       std::uint64_t regular_measurements,
                       std::uint64_t final_measurements)
      : records_(extraction_rounds, regular_measurements, final_measurements),
        regular_measurements_(regular_measurements) {}

  std::uint64_t extraction_rounds() const {
    return records_.extraction_rounds();
  }
  std::uint64_t input_rounds() const { return records_.input_rounds(); }
  std::uint64_t raw_measurements() const { return records_.raw_measurements(); }

  /// Map [first, past_last) raw rounds, including the distinct final readout.
  /// Empty ranges are legal; reversed and out-of-horizon ranges are rejected.
  measurement_range measurements_for_rounds(std::uint64_t first,
                                            std::uint64_t past_last) const {
    if (first > past_last || past_last > input_rounds())
      throw std::out_of_range(
          "memory experiment: raw-round range is outside the experiment");
    return {measurement_offset(first), measurement_offset(past_last)};
  }

  /// Describe batches without materializing a duration-dependent schedule.
  delivery_layout batches(std::uint64_t extraction_rounds_per_batch) const {
    if (extraction_rounds_per_batch == 0)
      throw std::invalid_argument(
          "memory experiment: extraction rounds per batch must be positive");
    const auto regular_batches =
        extraction_rounds() / extraction_rounds_per_batch;
    // Do not multiply an unused oversized batch width: it need not be
    // representable even when the shot's checked total is representable.
    const auto batch_measurements =
        regular_batches == 0
            ? 0
            : extraction_rounds_per_batch * regular_measurements_;
    return {regular_batches, batch_measurements,
            extraction_rounds() % extraction_rounds_per_batch,
            raw_measurements() - regular_batches * batch_measurements};
  }

private:
  std::uint64_t measurement_offset(std::uint64_t round) const {
    // The production constructor bounds in-range offsets by the total count.
    return round == input_rounds() ? raw_measurements()
                                   : round * regular_measurements_;
  }

  memory_record_layout records_;
  std::uint64_t regular_measurements_;
};

} // namespace cudaq::qec::memory_experiment
