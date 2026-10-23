/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/realtime/gpu_kernels.cuh"

namespace cudaq::qec::realtime {

__device__ void preprocess_detector(std::size_t detector_idx,
                                    const uint8_t *__restrict__ measurements,
                                    const uint32_t *__restrict__ D_row_ptr,
                                    const uint32_t *__restrict__ D_col_idx,
                                    float_t *__restrict__ soft_syndrome,
                                    std::size_t num_detectors) {

  if (detector_idx >= num_detectors)
    return;

  uint8_t detector = 0;
  const uint32_t row_start = D_row_ptr[detector_idx];
  const uint32_t row_end = D_row_ptr[detector_idx + 1];

  for (uint32_t j = row_start; j < row_end; ++j) {
    const uint32_t col = D_col_idx[j];
    detector ^= measurements[col];
  }

  soft_syndrome[detector_idx] = static_cast<float_t>(detector);
}

__device__ void postprocess_observable(
    std::size_t observable_idx, const float_t *__restrict__ soft_decisions,
    const uint32_t *__restrict__ O_row_ptr,
    const uint32_t *__restrict__ O_col_idx, uint8_t *__restrict__ corrections,
    std::size_t num_observables, float_t thresh) {

  if (observable_idx >= num_observables)
    return;

  uint8_t result = 0;
  const uint32_t row_start = O_row_ptr[observable_idx];
  const uint32_t row_end = O_row_ptr[observable_idx + 1];

  for (uint32_t j = row_start; j < row_end; ++j) {
    const uint32_t col = O_col_idx[j];
    if (soft_decisions[col] >= thresh) {
      result ^= 1;
    }
  }

  corrections[observable_idx] ^= result;
}

__device__ void preprocess_all(const uint8_t *__restrict__ measurements,
                               const uint32_t *__restrict__ D_row_ptr,
                               const uint32_t *__restrict__ D_col_idx,
                               float_t *__restrict__ soft_syndrome,
                               std::size_t num_detectors) {

  const std::size_t grid_size =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t global_idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (std::size_t idx = global_idx; idx < num_detectors; idx += grid_size) {
    preprocess_detector(idx, measurements, D_row_ptr, D_col_idx, soft_syndrome,
                        num_detectors);
  }
}

__device__ void postprocess_all(const float_t *__restrict__ soft_decisions,
                                const uint32_t *__restrict__ O_row_ptr,
                                const uint32_t *__restrict__ O_col_idx,
                                uint8_t *__restrict__ corrections,
                                std::size_t num_observables, float_t thresh) {

  const std::size_t grid_size =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t global_idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (std::size_t idx = global_idx; idx < num_observables; idx += grid_size) {
    postprocess_observable(idx, soft_decisions, O_row_ptr, O_col_idx,
                           corrections, num_observables, thresh);
  }
}

} // namespace cudaq::qec::realtime
