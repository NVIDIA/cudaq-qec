/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experiment_spec.h"

#include <limits>
#include <stdexcept>

namespace cudaq::qec::memory_experiment {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::overflow_error("memory experiment: record count exceeds uint64");
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right)
    throw std::overflow_error("memory experiment: record count exceeds uint64");
  return left * right;
}

void validate_rounds(std::uint64_t rounds) {
  if (rounds == 0)
    throw std::invalid_argument(
        "memory experiment: rounds must be a positive extraction-cycle count");
  checked_add(rounds, 1);
}

} // namespace

void validate_experiment_spec(const experiment_spec &spec) {
  if (spec.code_family != "surface_code")
    throw std::invalid_argument(
        "memory experiment: initially supported code_family is surface_code");
  if (spec.distance < 3 || spec.distance % 2 == 0)
    throw std::invalid_argument(
        "memory experiment: surface_code distance must be odd and at least 3");
  validate_rounds(spec.rounds);
  if (spec.logical_qubits == 0)
    throw std::invalid_argument("memory experiment: logical_qubits must be "
                                "positive (independent patches)");
  if (spec.basis != memory_basis::x && spec.basis != memory_basis::z)
    throw std::invalid_argument("memory experiment: basis must be X or Z");
}

memory_record_layout::memory_record_layout(std::uint64_t extraction_rounds,
                                           std::uint64_t regular_measurements,
                                           std::uint64_t final_measurements)
    : extraction_rounds_(extraction_rounds), raw_measurements_(0) {
  validate_rounds(extraction_rounds);
  if (regular_measurements == 0 || final_measurements == 0)
    throw std::invalid_argument("memory experiment: extraction and readout "
                                "record widths must be positive");
  raw_measurements_ =
      checked_add(checked_multiply(extraction_rounds, regular_measurements),
                  final_measurements);
}

} // namespace cudaq::qec::memory_experiment
