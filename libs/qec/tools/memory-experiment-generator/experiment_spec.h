/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace cudaq::qec::memory_experiment {

enum class memory_basis { x, z };

/// Inputs shared by the model provider and graph emitter. Rounds count
/// stabilizer-extraction cycles, excluding terminal data-qubit readout.
/// Multiple logical qubits denote independent patches, not an interacting code.
struct experiment_spec {
  std::string code_family = "surface_code";
  std::uint64_t distance = 0;
  std::uint64_t rounds = 0;
  std::uint64_t logical_qubits = 1;
  memory_basis basis = memory_basis::z;
};

/// Validate the initially supported experiment family. This does not assert
/// that a decoder or the target framework supports its geometry or duration.
/// Unsupported inputs are rejected without rounding the requested experiment.
void validate_experiment_spec(const experiment_spec &spec);

/// Checked raw-record counts for a memory experiment with fixed-width
/// extraction rounds and a possibly different terminal data-readout width.
/// Widths must come from the model provider's record convention, not a detector
/// count.
///
/// This is a record layout, not a measurement-to-detector map or an A/B
/// scheduler. It neither infers detector counts nor adds lookback records.
/// Storage and construction are constant in the requested experiment duration.
class memory_record_layout {
public:
  memory_record_layout(std::uint64_t extraction_rounds,
                       std::uint64_t regular_measurements,
                       std::uint64_t final_measurements);

  std::uint64_t extraction_rounds() const { return extraction_rounds_; }
  /// Includes the final readout record, not an extra extraction cycle.
  std::uint64_t input_rounds() const { return extraction_rounds_ + 1; }
  std::uint64_t raw_measurements() const { return raw_measurements_; }

private:
  std::uint64_t extraction_rounds_;
  std::uint64_t raw_measurements_;
};

} // namespace cudaq::qec::memory_experiment
