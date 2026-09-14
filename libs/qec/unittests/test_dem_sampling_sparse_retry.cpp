/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/dem_sampling.h"

#include <atomic>
#include <cuda_runtime.h>
#include <custabilizer.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
std::atomic<int> g_compute_calls{0};
}

extern "C" {
custabilizerStatus_t __real_custabilizerSampleProbArraySparseCompute(
    custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs,
    const double *probs, uint64_t seed, uint64_t *nnz, uint64_t *columnIndices,
    uint64_t *rowOffsets, void *workspace, size_t workspaceSize,
    cudaStream_t stream);

custabilizerStatus_t __wrap_custabilizerSampleProbArraySparseCompute(
    custabilizerHandle_t handle, int64_t numSamples, int64_t numProbs,
    const double *probs, uint64_t seed, uint64_t *nnz, uint64_t *columnIndices,
    uint64_t *rowOffsets, void *workspace, size_t workspaceSize,
    cudaStream_t stream) {
  const int n = ++g_compute_calls;
  if (n == 1) {
    *nnz = *nnz + 1;
    return CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE;
  }
  return __real_custabilizerSampleProbArraySparseCompute(
      handle, numSamples, numProbs, probs, seed, nnz, columnIndices, rowOffsets,
      workspace, workspaceSize, stream);
}
}

TEST(DemSamplingSparseRetry, FirstComputeRetryThenMatchesAllOnes) {
  int gpu = 0;
  ASSERT_EQ(cudaGetDeviceCount(&gpu), cudaSuccess);
  ASSERT_GT(gpu, 0);
  const size_t n_checks = 3, n_err = 4, n_shots = 5;
  std::vector<uint8_t> H = {1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1};
  std::vector<double> probs(n_err, 1.0);
  uint8_t *d_H = nullptr, *d_checks = nullptr, *d_errors = nullptr;
  double *d_probs = nullptr;
  ASSERT_EQ(cudaMalloc(&d_H, H.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_probs, probs.size() * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_checks, n_shots * n_checks), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_errors, n_shots * n_err), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_H, H.data(), H.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_probs, probs.data(), probs.size() * sizeof(double),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  g_compute_calls.store(0);
  ASSERT_TRUE(cudaq::qec::dem_sampler::gpu::sample_dem(
      d_H, n_checks, n_err, d_probs, n_shots, 7, d_checks, d_errors));
  EXPECT_EQ(g_compute_calls.load(), 2);
  std::vector<uint8_t> checks(n_shots * n_checks), errors(n_shots * n_err);
  ASSERT_EQ(cudaMemcpy(checks.data(), d_checks, checks.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(errors.data(), d_errors, errors.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (auto e : errors)
    EXPECT_EQ(e, 1);
  for (size_t s = 0; s < n_shots; ++s) {
    EXPECT_EQ(checks[s * n_checks + 0], 0);
    EXPECT_EQ(checks[s * n_checks + 1], 0);
    EXPECT_EQ(checks[s * n_checks + 2], 1);
  }
  cudaFree(d_H);
  cudaFree(d_probs);
  cudaFree(d_checks);
  cudaFree(d_errors);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
