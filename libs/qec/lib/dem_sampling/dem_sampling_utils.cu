/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dem_sampling_utils.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cudaq::qec::dem_sampling_utils {

namespace {

constexpr uint64_t kMaxGridDimX = static_cast<uint64_t>(INT32_MAX);
constexpr uint64_t kMaxGridDimYZ = 65535u;

void check_kernel_launch(const char *kernel_name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(kernel_name) +
                             " launch: " + cudaGetErrorString(err));
}

void check_grid_dims(uint64_t grid_x, uint64_t grid_y,
                     const char *kernel_name) {
  if (grid_x > kMaxGridDimX || grid_y > kMaxGridDimYZ)
    throw std::runtime_error(std::string(kernel_name) + ": grid dimensions (" +
                             std::to_string(grid_x) + ", " +
                             std::to_string(grid_y) + ") exceed CUDA limits (" +
                             std::to_string(kMaxGridDimX) + ", " +
                             std::to_string(kMaxGridDimYZ) + ")");
}

} // namespace

// ── Row-wise packing: [M × N] uint8 → [M × words] uint32 ──────────────────
__global__ void
pack_check_matrix_rowwise_kernel(const uint8_t *__restrict__ d_check_matrix,
                                 uint32_t *__restrict__ d_check_packed,
                                 uint32_t M, uint32_t N,
                                 uint32_t errors_words) {
  const uint32_t row = blockIdx.x;
  const uint32_t word_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (row >= M || word_idx >= errors_words)
    return;

  const uint8_t *row_ptr = d_check_matrix + static_cast<uint64_t>(row) * N;
  const uint32_t bit_start = word_idx * 32;

  uint32_t packed = 0;
#pragma unroll
  for (uint32_t bit = 0; bit < 32; bit++) {
    const uint32_t col = bit_start + bit;
    if (col < N) {
      packed |= (static_cast<uint32_t>(__ldg(&row_ptr[col]) & 1u) << bit);
    }
  }

  d_check_packed[static_cast<uint64_t>(row) * errors_words + word_idx] = packed;
}

void pack_check_matrix_rowwise(const uint8_t *d_check_matrix,
                               uint32_t *d_check_packed, uint64_t M, uint64_t N,
                               cudaStream_t stream) {
  if (M == 0 || N == 0)
    return;

  uint64_t words = bitpack_words(N);
  dim3 block(256);
  uint64_t grid_x = M;
  uint64_t grid_y = (words + block.x - 1) / block.x;
  check_grid_dims(grid_x, grid_y, "pack_check_matrix_rowwise");
  dim3 grid(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y));

  pack_check_matrix_rowwise_kernel<<<grid, block, 0, stream>>>(
      d_check_matrix, d_check_packed, static_cast<uint32_t>(M),
      static_cast<uint32_t>(N), static_cast<uint32_t>(words));
  check_kernel_launch("pack_check_matrix_rowwise");
}

// ── Direct transpose+pack: H [checks x errors] → packed H^T [errors x words]
__global__ void pack_check_matrix_transposed_rowwise_kernel(
    const uint8_t *__restrict__ d_check_matrix,
    uint32_t *__restrict__ d_check_t_packed, uint32_t num_checks,
    uint32_t num_error_mechanisms, uint32_t checks_words) {
  const uint32_t error_mech = blockIdx.x;
  const uint32_t word_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (error_mech >= num_error_mechanisms || word_idx >= checks_words)
    return;

  const uint32_t bit_start = word_idx * 32;
  uint32_t packed = 0;
#pragma unroll
  for (uint32_t bit = 0; bit < 32; ++bit) {
    const uint32_t check = bit_start + bit;
    if (check < num_checks) {
      const uint64_t idx =
          static_cast<uint64_t>(check) * num_error_mechanisms + error_mech;
      packed |=
          (static_cast<uint32_t>(__ldg(&d_check_matrix[idx]) & 1u) << bit);
    }
  }

  d_check_t_packed[static_cast<uint64_t>(error_mech) * checks_words +
                   word_idx] = packed;
}

void pack_check_matrix_transposed_rowwise(const uint8_t *d_check_matrix,
                                          uint32_t *d_check_t_packed,
                                          uint64_t num_checks,
                                          uint64_t num_error_mechanisms,
                                          cudaStream_t stream) {
  if (num_checks == 0 || num_error_mechanisms == 0)
    return;

  const uint64_t checks_words = bitpack_words(num_checks);
  dim3 block(256);
  uint64_t grid_x = num_error_mechanisms;
  uint64_t grid_y = (checks_words + block.x - 1) / block.x;
  check_grid_dims(grid_x, grid_y, "pack_check_matrix_transposed_rowwise");
  dim3 grid(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y));

  pack_check_matrix_transposed_rowwise_kernel<<<grid, block, 0, stream>>>(
      d_check_matrix, d_check_t_packed, static_cast<uint32_t>(num_checks),
      static_cast<uint32_t>(num_error_mechanisms),
      static_cast<uint32_t>(checks_words));
  check_kernel_launch("pack_check_matrix_transposed_rowwise");
}

// ── Syndrome unpacking: [S × words] uint32 → [S × M] uint8 ────────────────
__global__ void unpack_syndromes_kernel(const uint32_t *__restrict__ d_packed,
                                        uint8_t *__restrict__ d_unpacked,
                                        uint64_t S, uint64_t M) {
  const uint64_t shot = blockIdx.x;
  const uint64_t check =
      static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;

  if (check >= M)
    return;

  const uint64_t words_per_row = (M + 31) >> 5;
  const uint32_t word_idx = static_cast<uint32_t>(check >> 5);
  const uint32_t bit_pos = static_cast<uint32_t>(check & 31);

  const uint32_t word = __ldg(&d_packed[shot * words_per_row + word_idx]);
  d_unpacked[shot * M + check] = static_cast<uint8_t>((word >> bit_pos) & 1u);
}

void unpack_syndromes_gpu(const uint32_t *d_packed, uint8_t *d_unpacked,
                          uint64_t num_shots, uint64_t num_checks,
                          cudaStream_t stream) {
  if (num_shots == 0 || num_checks == 0)
    return;

  const int threads = 256;
  uint64_t grid_x = num_shots;
  uint64_t grid_y = (num_checks + threads - 1) / threads;
  check_grid_dims(grid_x, grid_y, "unpack_syndromes");
  dim3 grid(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y));

  unpack_syndromes_kernel<<<grid, threads, 0, stream>>>(d_packed, d_unpacked,
                                                        num_shots, num_checks);
  check_kernel_launch("unpack_syndromes");
}

// ── CSR → dense (uint64_t indices, warp-cooperative fused zeroing) ─────────
__global__ void
csr_to_dense_fused_kernel(const uint64_t *__restrict__ d_row_offsets,
                          const uint64_t *__restrict__ d_col_indices,
                          uint64_t num_shots, uint64_t num_error_mechanisms,
                          uint8_t *__restrict__ d_errors_out) {
  const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const uint32_t lane = threadIdx.x % 32;

  if (warp_id >= num_shots)
    return;

  const uint64_t shot = warp_id;
  uint8_t *shot_out = d_errors_out + shot * num_error_mechanisms;

  for (uint64_t i = lane; i < num_error_mechanisms; i += 32)
    shot_out[i] = 0;

  __syncwarp();

  const uint64_t start = d_row_offsets[shot];
  const uint64_t end = d_row_offsets[shot + 1];
  const uint64_t num_errors = end - start;

  for (uint64_t e = lane; e < num_errors; e += 32) {
    uint64_t idx = d_col_indices[start + e];
    if (idx < num_error_mechanisms) {
      shot_out[idx] = 1;
    }
  }
}

void csr_to_dense_fused(const uint64_t *d_row_offsets,
                        const uint64_t *d_col_indices, uint64_t num_shots,
                        uint64_t num_error_mechanisms, uint8_t *d_errors_out,
                        cudaStream_t stream) {
  if (num_shots == 0 || num_error_mechanisms == 0)
    return;

  const int warps_per_block = 8;
  const int threads_per_block = warps_per_block * 32;
  uint64_t num_blocks_u64 = (num_shots + warps_per_block - 1) / warps_per_block;
  check_grid_dims(num_blocks_u64, 1, "csr_to_dense_fused");
  const int num_blocks = static_cast<int>(num_blocks_u64);

  csr_to_dense_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_row_offsets, d_col_indices, num_shots, num_error_mechanisms,
      d_errors_out);
  check_kernel_launch("csr_to_dense_fused");
}

} // namespace cudaq::qec::dem_sampling_utils
