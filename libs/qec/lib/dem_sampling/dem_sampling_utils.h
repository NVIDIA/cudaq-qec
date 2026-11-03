/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cudaq::qec::dem_sampling_utils {

inline __host__ __device__ uint64_t bitpack_words(uint64_t num_bits) {
  return (num_bits + 31) / 32;
}

/// Pack [M x N] uint8 matrix row-wise into [M x words] uint32, where
/// words = ceil(N/32). Bit i in word w encodes column w*32+i.
void pack_check_matrix_rowwise(const uint8_t *d_check_matrix,
                               uint32_t *d_check_packed, uint64_t M, uint64_t N,
                               cudaStream_t stream = 0);

/// Pack H^T directly from H without an explicit transpose buffer.
/// Input H is [num_checks x num_error_mechanisms] uint8.
/// Output packed H^T is [num_error_mechanisms x ceil(num_checks/32)] uint32.
void pack_check_matrix_transposed_rowwise(const uint8_t *d_check_matrix,
                                          uint32_t *d_check_t_packed,
                                          uint64_t num_checks,
                                          uint64_t num_error_mechanisms,
                                          cudaStream_t stream = 0);

/// Unpack [S x words] uint32 syndromes into [S x M] uint8.
void unpack_syndromes_gpu(const uint32_t *d_packed, uint8_t *d_unpacked,
                          uint64_t num_shots, uint64_t num_checks,
                          cudaStream_t stream = 0);

/// Convert CSR errors (uint64_t offsets/indices from cuStabilizer) to
/// dense [num_shots x num_error_mechanisms] uint8, fusing zeroing with
/// scatter. The warp-cooperative kernel avoids a separate cudaMemset.
void csr_to_dense_fused(const uint64_t *d_row_offsets,
                        const uint64_t *d_col_indices, uint64_t num_shots,
                        uint64_t num_error_mechanisms, uint8_t *d_errors_out,
                        cudaStream_t stream = 0);

} // namespace cudaq::qec::dem_sampling_utils
