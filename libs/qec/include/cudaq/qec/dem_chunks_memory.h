/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Memory-circuit-specific DEM chunk utilities.
//
// The generic open-boundary DEM infrastructure (extended_dem, dem_stitch,
// dem_close, dem_chunk_spec, dem_chunks_spec, dem_chunk_from_spec,
// dem_chunks_from_spec, ...) lives in extended_dem.h and is circuit-agnostic.
//
// This file provides only dem_chunks_to_d_sparse: the XOR-of-consecutive-
// rounds detector map that encodes the memory-experiment detector convention.
// For any other circuit, the D matrix must come from circuit analysis.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace cudaq::qec {

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
/// memory-experiment convention above, over dem_chunks_from_spec(spec).
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
