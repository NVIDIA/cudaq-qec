/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Memory-circuit-specific DEM chunk utilities.
//
// The generic open-boundary DEM infrastructure (extended_dem, dem_stitch,
// dem_close, dem_chunk_spec, dem_chunks_spec, dem_chunk_from_spec,
// dem_chunks_from_spec, ...) lives in extended_dem.h and is circuit-agnostic.
//
// This file provides memory-circuit phase splitting and measurement-map
// utilities. For any other circuit, the D matrix must come from analysis
// rather than the memory experiment's consecutive-round convention.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

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

/// @brief Extract the D_sparse measurement-to-detector map from T DEM chunks.
///
/// D_sparse[det_id] lists the raw per-round measurement bit positions (within
/// a flat T*d buffer laid out as round-0 bits 0..d-1, round-1 bits d..2d-1,
/// ...) that XOR-combine to produce detector det_id:
///   - Detector k       (r=0): bits {k}               (vs zero initial state)
///   - Detector r*d+k (r>0): bits {(r-1)*d+k, r*d+k} (syndrome difference)
///
/// This encodes the memory-experiment detector convention: consecutive syndrome
/// rounds are XOR-differenced. For a non-memory circuit whose detectors are
/// not pairs of adjacent rounds, construct D_sparse from circuit analysis
/// instead of using this function.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @param from_seam  Seam contracting forward (default: seam_name::next_round)
/// @param to_seam    Seam contracting backward (default: seam_name::prev_round)
/// @throws std::invalid_argument on empty sequence, width or tag mismatch, or
///         non-whole-round interior row counts.
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const std::vector<extended_dem> &dem_chunks,
                       seam_id from_seam = seam_name::next_round,
                       seam_id to_seam = seam_name::prev_round);

/// @brief D_sparse for the closed model of a dem_chunks_spec.
///
/// When the phases carry D_sparse rows (see dem_chunk_spec::D_sparse), each
/// chunk's measurements follow the previous chunk's in the stream and every
/// chunk's rows are rebased onto it, so rounds may differ in width and
/// detectors need not be adjacent-round differences; each round's size is
/// dem_chunks_spec::measurements_of() its phase. Otherwise this is the
/// memory-experiment convention above, validated on a compact chain without
/// expanding the full-shot fault model.
/// The returned map still contains one row per configured detector, so its
/// storage grows with the configured round count. Compact fault-model
/// preparation does not make the returned D map constant-size.
///
/// @throws std::invalid_argument if the spec does not expand, a chunk's rows
///   do not number its detectors, or an index lies outside the previous and
///   current rounds' measurements.
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const dem_chunks_spec &spec);

/// @brief Raw measurements per shot the spec's rounds declare.
///
/// The sum of dem_chunks_spec::measurements_of() over phase_sequence(), which
/// is the width of the measurement stream a decoder is fed and so the column
/// count of the D built from dem_chunks_to_d_sparse(). It is not recoverable
/// from those rows: a round's trailing measurements may be referenced by no
/// detector, and the largest referenced index then falls short of the stream.
///
/// @return The declared total, or nullopt when the spec supplies no D_sparse
///   rows or any phase lacks a count -- in which case the memory convention of
///   dem_chunks_to_d_sparse() fixes the width itself and every measurement it
///   lays out is referenced.
/// @throws std::invalid_argument if the spec does not expand.
std::optional<std::uint64_t>
dem_chunks_measurement_count(const dem_chunks_spec &spec);

} // namespace cudaq::qec
