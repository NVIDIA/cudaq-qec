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
// This file provides only dem_chunks_to_d_sparse: the XOR-of-consecutive-
// rounds detector map that encodes the memory-experiment detector convention.
// For any other circuit, the D matrix must come from circuit analysis.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <cstdint>
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

} // namespace cudaq::qec
