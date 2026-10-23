/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cudaq/qec/extended_dem.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cudaq::qec {

struct decoder_inputs;

/// Compare validated phase probabilities using the relative tolerance used to
/// identify a stationary memory-circuit bulk. No absolute tolerance is added,
/// so zero and arbitrarily small positive probabilities remain distinct.
inline bool memory_dem_probabilities_equivalent(double left, double right) {
  constexpr double relative_tolerance = 1e-9;
  return std::abs(left - right) <= relative_tolerance * std::max(left, right);
}

/// @brief Split a characterized memory circuit into reusable DEM phases.
///
/// The input records contain @p extraction_rounds groups of
/// @p measurements_per_round bits followed by @p final_measurements data
/// readout bits. Each detector is assigned to the round containing its latest
/// measurement. Its D row may read that round and its immediate predecessor
/// only. Detector numbering must already be round-major. Each independent
/// fault must touch a detector and span at most two consecutive detector
/// rounds. Up to 64 observables are supported.
///
/// Consecutive phases matching the middle round are collapsed into one bulk
/// phase with a self-loop. At least two matching rounds are required as
/// evidence of repetition. Prefix and suffix transients are retained
/// separately. Rates must agree within a relative tolerance of 1e-9 for
/// phases to be considered equal. Faults with identical detector and
/// observable signatures are merged using independent XOR probabilities.
///
/// The returned spec carries the original horizon: extraction_rounds + 1,
/// including the distinct final readout round. A caller may replace
/// num_rounds to repeat the validated bulk phase over another horizon, subject
/// to the spec's minimum_rounds(). Characterizing a bounded representative
/// circuit and then retargeting the result avoids building a long dense DEM.
/// This structural comparison is not a proof of repetition for arbitrary
/// circuit families; callers must establish that their circuit repeats.
///
/// The input matrices and measurement maps are validated before indexing.
/// Measurement-map rows must be sorted and unique. Shared error_ids are
/// rejected: dem_chunks does not preserve cross-column exclusivity.
/// The raw measurement-to-observable map is validated but is not included in
/// the returned spec, whose O maps faults, not raw measurements.
///
/// @throws std::invalid_argument for invalid dimensions, mappings,
/// probabilities, locality, missing repetition, unsupported correlated
/// columns, or a phase with no faults. Zero-fault phases require separate
/// support in extended_dem and are not fabricated by adding dummy faults.
/// @throws std::overflow_error when a dimension exceeds the index format.
dem_chunks_spec memory_dem_chunks_from_decoder_inputs(
    const decoder_inputs &inputs, std::size_t extraction_rounds,
    std::size_t measurements_per_round, std::size_t final_measurements);

} // namespace cudaq::qec
