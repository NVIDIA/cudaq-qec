/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace cudaq::qec::dem_sampler {

// CPU implementation of DEM sampling, using std::bernoulli_distribution and a
// tensor dot product. The sample_dem overloads below are the public API.
namespace cpu {

/// @brief Sample measurements from a check matrix (CPU, per-mechanism probs)
/// @param check_matrix Binary matrix [num_checks × num_error_mechanisms]
/// @param numShots Number of measurement shots
/// @param error_probabilities Per-error-mechanism probabilities
/// @return (checks [numShots × num_checks], errors [numShots × num_mechanisms])
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_dem(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
           const std::vector<double> &error_probabilities);

/// @brief Sample measurements from a check matrix with seed (CPU)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_dem(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
           const std::vector<double> &error_probabilities, unsigned seed);

} // namespace cpu

// GPU implementation of DEM sampling via the cuStabilizer C API.
namespace gpu {

/// @brief GPU DEM sampling with caller-provided device pointers.
///
/// The function composes (all GPU memory is allocated and freed internally
/// per call):
///   1. pack_check_matrix_rowwise (dense uint8 → bitpacked uint32)
///   2. custabilizerSampleProbArraySparseCompute (sparse Bernoulli sampling)
///   3. custabilizerGF2SparseDenseMatrixMultiply (syndrome = errors * H^T)
///   4. unpack_syndromes_gpu (bitpacked uint32 → uint8)
///   5. csr_to_dense_fused (CSR errors → dense uint8)
///
/// @param d_check_matrix Device pointer [num_checks × num_error_mechanisms]
/// @param num_checks Number of checks (rows of H)
/// @param num_error_mechanisms Number of error mechanisms (columns of H)
/// @param d_error_probabilities Device pointer [num_error_mechanisms]
/// @param num_shots Number of samples
/// @param seed RNG seed
/// @param d_checks_out Device pointer [num_shots × num_checks] (OUTPUT)
/// @param d_errors_out Device pointer [num_shots × num_error_mechanisms] (OUT)
/// @param stream_handle Optional CUDA stream handle (uintptr_t cast), 0 for
///        default stream
/// @return true on success, false if cuStabilizer is unavailable
bool sample_dem(const uint8_t *d_check_matrix, size_t num_checks,
                size_t num_error_mechanisms,
                const double *d_error_probabilities, size_t num_shots,
                unsigned seed, uint8_t *d_checks_out, uint8_t *d_errors_out,
                std::uintptr_t stream_handle = 0);

} // namespace gpu

} // namespace cudaq::qec::dem_sampler
