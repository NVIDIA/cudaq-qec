/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "cudaq/qec/decoder.h"

namespace cudaq::qec::realtime {

/// @brief Floating-point type for soft syndrome values.
/// Matches cudaq::qec::float_t for compatibility.
using float_t = cudaq::qec::float_t;

//==============================================================================
// DEVICE FUNCTIONS FOR REALTIME DECODER PIPELINE
//==============================================================================

/// @brief Preprocess a single detector by XORing measurements via D matrix.
///
/// For detector at row `detector_idx`:
///   1. XOR measurements at column indices specified by D matrix row
///   2. Convert result to soft probability (0 -> 0.0, 1 -> 1.0)
///
/// @param detector_idx Index of the detector to compute
/// @param measurements Input raw measurement buffer
/// @param D_row_ptr CSR row pointers for D matrix
/// @param D_col_idx CSR column indices for D matrix
/// @param soft_syndrome Output soft syndrome array
/// @param num_detectors Total number of detectors
__device__ void preprocess_detector(std::size_t detector_idx,
                                    const uint8_t *__restrict__ measurements,
                                    const uint32_t *__restrict__ D_row_ptr,
                                    const uint32_t *__restrict__ D_col_idx,
                                    float_t *__restrict__ soft_syndrome,
                                    std::size_t num_detectors);

/// @brief Postprocess a single observable by XORing thresholded soft decisions
/// via O matrix.
///
/// For observable at row `observable_idx`:
///   1. Read soft decisions at column positions from O matrix row
///   2. Threshold each value (>= thresh -> 1, else 0)
///   3. XOR the hard decisions to compute this window's correction
///   4. XOR-accumulate that correction into corrections[observable_idx]
///
/// XOR-accumulate semantics mirror the legacy CPU contract where every
/// full-window decode XORs that window's observable bits into a per-shot
/// corrections buffer. Callers are responsible for zeroing `corrections` at
/// shot boundaries.
///
/// @param observable_idx Index of the observable to compute
/// @param soft_decisions Input soft decisions from decoder
/// @param O_row_ptr CSR row pointers for O matrix
/// @param O_col_idx CSR column indices for O matrix
/// @param corrections In/out per-shot accumulator
/// @param num_observables Total number of observables
/// @param thresh Soft-to-hard threshold (default 0.5)
__device__ void postprocess_observable(
    std::size_t observable_idx, const float_t *__restrict__ soft_decisions,
    const uint32_t *__restrict__ O_row_ptr,
    const uint32_t *__restrict__ O_col_idx, uint8_t *__restrict__ corrections,
    std::size_t num_observables, float_t thresh = 0.5);

/// @brief Preprocess all detectors using grid-stride loop.
///
/// All threads in the grid participate, each handling a subset of detectors.
///
/// @param measurements Input raw measurement buffer
/// @param D_row_ptr CSR row pointers for D matrix (size: num_detectors + 1)
/// @param D_col_idx CSR column indices for D matrix
/// @param soft_syndrome Output soft syndrome values
/// @param num_detectors Number of detectors (rows in D matrix)
__device__ void preprocess_all(const uint8_t *__restrict__ measurements,
                               const uint32_t *__restrict__ D_row_ptr,
                               const uint32_t *__restrict__ D_col_idx,
                               float_t *__restrict__ soft_syndrome,
                               std::size_t num_detectors);

/// @brief Postprocess all observables using grid-stride loop.
///
/// All threads in the grid participate, each handling a subset of observables.
/// Calls postprocess_observable per observable with XOR-accumulate semantics.
///
/// @param soft_decisions Input soft decoder output
/// @param O_row_ptr CSR row pointers for O matrix (size: num_observables + 1)
/// @param O_col_idx CSR column indices for O matrix
/// @param corrections In/out per-shot accumulator
/// @param num_observables Number of observables (rows in O matrix)
/// @param thresh Threshold for soft-to-hard conversion (default 0.5)
__device__ void postprocess_all(const float_t *__restrict__ soft_decisions,
                                const uint32_t *__restrict__ O_row_ptr,
                                const uint32_t *__restrict__ O_col_idx,
                                uint8_t *__restrict__ corrections,
                                std::size_t num_observables,
                                float_t thresh = 0.5);

} // namespace cudaq::qec::realtime
