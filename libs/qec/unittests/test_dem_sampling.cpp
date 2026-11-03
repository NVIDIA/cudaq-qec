/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/experiments.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

namespace {

/// Reference syndrome: checks = errors * H^T mod 2  (single shot, flat layout)
std::vector<uint8_t> compute_syndrome(const std::vector<uint8_t> &H,
                                      const std::vector<uint8_t> &errors,
                                      size_t num_checks, size_t num_errors) {
  std::vector<uint8_t> syndrome(num_checks, 0);
  for (size_t col = 0; col < num_errors; col++) {
    if (errors[col]) {
      for (size_t row = 0; row < num_checks; row++) {
        syndrome[row] ^= H[row * num_errors + col];
      }
    }
  }
  return syndrome;
}

/// Build a cudaqx tensor from a flat uint8 vector.
cudaqx::tensor<uint8_t> make_tensor(const std::vector<uint8_t> &data,
                                    size_t rows, size_t cols) {
  cudaqx::tensor<uint8_t> t({rows, cols});
  t.copy(data.data(), t.shape());
  return t;
}

} // namespace

// =============================================================================
// CPU tests
// =============================================================================

TEST(DemSamplingCPU, AllZeroProbabilities) {
  //   H = | 1 0 1 0 |     probs = [0, 0, 0, 0]
  //       | 0 1 1 0 |
  //       | 0 0 0 1 |
  // No errors should ever fire, so errors=0 and checks=0.
  auto H = make_tensor({1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1}, 3, 4);
  std::vector<double> probs(4, 0.0);

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 10, probs, 42);

  EXPECT_EQ(checks.shape()[0], 10u);
  EXPECT_EQ(checks.shape()[1], 3u);
  EXPECT_EQ(errors.shape()[0], 10u);
  EXPECT_EQ(errors.shape()[1], 4u);

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 3; j++)
      EXPECT_EQ(checks.at({i, j}), 0);
    for (size_t j = 0; j < 4; j++)
      EXPECT_EQ(errors.at({i, j}), 0);
  }
}

TEST(DemSamplingCPU, AllOneProbabilities) {
  //   H = | 1 0 1 0 |     probs = [1, 1, 1, 1]
  //       | 1 1 0 1 |
  //       | 0 1 1 0 |
  // Every error fires. Syndrome for all-ones error = each row summed mod 2.
  //   row 0: 1+0+1+0 = 2 mod 2 = 0
  //   row 1: 1+1+0+1 = 3 mod 2 = 1
  //   row 2: 0+1+1+0 = 2 mod 2 = 0
  std::vector<uint8_t> H_data = {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0};
  auto H = make_tensor(H_data, 3, 4);
  std::vector<double> probs(4, 1.0);

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 5, probs, 42);

  for (size_t shot = 0; shot < 5; shot++) {
    for (size_t e = 0; e < 4; e++)
      EXPECT_EQ(errors.at({shot, e}), 1);
    EXPECT_EQ(checks.at({shot, 0}), 0);
    EXPECT_EQ(checks.at({shot, 1}), 1);
    EXPECT_EQ(checks.at({shot, 2}), 0);
  }
}

TEST(DemSamplingCPU, MixedDeterministicProbs) {
  //   H = | 1 0 1 |     probs = [1, 0, 1]
  //       | 0 1 1 |
  // Errors = [1, 0, 1] every shot.
  //   check 0: 1*1 + 0*0 + 1*1 = 2 mod 2 = 0
  //   check 1: 0*1 + 1*0 + 1*1 = 1 mod 2 = 1
  auto H = make_tensor({1, 0, 1, 0, 1, 1}, 2, 3);
  std::vector<double> probs = {1.0, 0.0, 1.0};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 8, probs, 99);

  for (size_t shot = 0; shot < 8; shot++) {
    EXPECT_EQ(errors.at({shot, 0}), 1);
    EXPECT_EQ(errors.at({shot, 1}), 0);
    EXPECT_EQ(errors.at({shot, 2}), 1);
    EXPECT_EQ(checks.at({shot, 0}), 0);
    EXPECT_EQ(checks.at({shot, 1}), 1);
  }
}

TEST(DemSamplingCPU, IdentityMatrix) {
  // H = I_5 with p=1: syndromes must equal errors (all ones).
  auto H = make_tensor({1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
                       5, 5);
  std::vector<double> probs(5, 1.0);

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 3, probs, 7);

  for (size_t shot = 0; shot < 3; shot++)
    for (size_t j = 0; j < 5; j++) {
      EXPECT_EQ(errors.at({shot, j}), 1);
      EXPECT_EQ(checks.at({shot, j}), 1);
    }
}

TEST(DemSamplingCPU, AllOnesMatrixEvenColumns) {
  // H = 3x4 all-ones.  p = [1,1,1,1].
  // Every error fires: 4 ones per row, 4 mod 2 = 0.  All syndromes = 0.
  std::vector<uint8_t> H_data(3 * 4, 1);
  auto H = make_tensor(H_data, 3, 4);
  std::vector<double> probs(4, 1.0);

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 4, probs, 0);

  for (size_t shot = 0; shot < 4; shot++)
    for (size_t c = 0; c < 3; c++)
      EXPECT_EQ(checks.at({shot, c}), 0) << "Even column count => syndrome 0";
}

TEST(DemSamplingCPU, AllOnesMatrixOddColumns) {
  // H = 3x3 all-ones.  p = [1,1,1].
  // Every error fires: 3 ones per row, 3 mod 2 = 1.  All syndromes = 1.
  std::vector<uint8_t> H_data(3 * 3, 1);
  auto H = make_tensor(H_data, 3, 3);
  std::vector<double> probs(3, 1.0);

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 4, probs, 0);

  for (size_t shot = 0; shot < 4; shot++)
    for (size_t c = 0; c < 3; c++)
      EXPECT_EQ(checks.at({shot, c}), 1) << "Odd column count => syndrome 1";
}

TEST(DemSamplingCPU, SingleColumnMatrix) {
  //  H = | 1 |   p = [1.0]
  //      | 0 |
  //      | 1 |
  // The single error always fires. Syndrome = column = [1, 0, 1].
  auto H = make_tensor({1, 0, 1}, 3, 1);
  std::vector<double> probs = {1.0};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 6, probs, 0);

  for (size_t shot = 0; shot < 6; shot++) {
    EXPECT_EQ(errors.at({shot, 0}), 1);
    EXPECT_EQ(checks.at({shot, 0}), 1);
    EXPECT_EQ(checks.at({shot, 1}), 0);
    EXPECT_EQ(checks.at({shot, 2}), 1);
  }
}

TEST(DemSamplingCPU, SingleRowMatrix) {
  //  H = | 1 1 0 1 0 |   p = [1, 0, 1, 0, 1]
  // Errors = [1, 0, 1, 0, 1].
  // Single check: 1*1 + 1*0 + 0*1 + 1*0 + 0*1 = 1 mod 2 = 1.
  auto H = make_tensor({1, 1, 0, 1, 0}, 1, 5);
  std::vector<double> probs = {1.0, 0.0, 1.0, 0.0, 1.0};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 4, probs, 0);

  for (size_t shot = 0; shot < 4; shot++) {
    EXPECT_EQ(errors.at({shot, 0}), 1);
    EXPECT_EQ(errors.at({shot, 1}), 0);
    EXPECT_EQ(errors.at({shot, 2}), 1);
    EXPECT_EQ(errors.at({shot, 3}), 0);
    EXPECT_EQ(errors.at({shot, 4}), 1);
    EXPECT_EQ(checks.at({shot, 0}), 1);
  }
}

TEST(DemSamplingCPU, SingleShot) {
  auto H = make_tensor({1, 1, 0, 0, 1, 1}, 2, 3);
  std::vector<double> probs = {1.0, 1.0, 0.0};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 1, probs, 0);

  EXPECT_EQ(errors.shape()[0], 1u);
  EXPECT_EQ(errors.at({0, 0}), 1);
  EXPECT_EQ(errors.at({0, 1}), 1);
  EXPECT_EQ(errors.at({0, 2}), 0);
  // check 0: 1*1 + 1*1 + 0*0 = 2 mod 2 = 0
  // check 1: 0*1 + 1*1 + 1*0 = 1 mod 2 = 1
  EXPECT_EQ(checks.at({0, 0}), 0);
  EXPECT_EQ(checks.at({0, 1}), 1);
}

TEST(DemSamplingCPU, RejectOutOfRangeProbabilities) {
  auto H = make_tensor({1, 0, 1, 0, 1, 1}, 2, 3);

  std::vector<double> probs_negative = {0.1, -0.2, 0.3};
  EXPECT_THROW(
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 16, probs_negative, 1),
      std::invalid_argument);

  std::vector<double> probs_above_one = {0.1, 1.2, 0.3};
  EXPECT_THROW(
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 16, probs_above_one, 1),
      std::invalid_argument);

  std::vector<double> probs_nan = {
      0.1, std::numeric_limits<double>::quiet_NaN(), 0.3};
  EXPECT_THROW(cudaq::qec::dem_sampler::cpu::sample_dem(H, 16, probs_nan, 1),
               std::invalid_argument);
}

TEST(DemSamplingCPU, SeedReproducibility) {
  auto H = make_tensor({1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1}, 3, 5);
  std::vector<double> probs = {0.1, 0.3, 0.5, 0.7, 0.9};

  auto [c1, e1] = cudaq::qec::dem_sampler::cpu::sample_dem(H, 100, probs, 42);
  auto [c2, e2] = cudaq::qec::dem_sampler::cpu::sample_dem(H, 100, probs, 42);

  for (size_t i = 0; i < 100; i++) {
    for (size_t j = 0; j < 3; j++)
      EXPECT_EQ(c1.at({i, j}), c2.at({i, j}));
    for (size_t j = 0; j < 5; j++)
      EXPECT_EQ(e1.at({i, j}), e2.at({i, j}));
  }
}

TEST(DemSamplingCPU, SyndromeConsistency) {
  // Use a fixed, known H and verify syndrome = errors * H^T mod 2 per shot.
  std::vector<uint8_t> H_data = {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                                 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
                                 0, 1, 1, 1, 0, 0, 0, 0, 1, 1};
  const size_t num_checks = 4, num_errors = 8, num_shots = 200;
  auto H = make_tensor(H_data, num_checks, num_errors);
  std::vector<double> probs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, num_shots, probs, 42);

  for (size_t shot = 0; shot < num_shots; shot++) {
    std::vector<uint8_t> shot_errors(num_errors);
    for (size_t e = 0; e < num_errors; e++)
      shot_errors[e] = errors.at({shot, e});

    auto expected =
        compute_syndrome(H_data, shot_errors, num_checks, num_errors);
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(checks.at({shot, c}), expected[c])
          << "Shot " << shot << " check " << c;
  }
}

TEST(DemSamplingCPU, RepetitionCodeParity) {
  // Repetition code: H = [[1,1,0,0], [0,1,1,0], [0,0,1,1]]
  // Single error on qubit 1 (index 1): flips checks 0 and 1.
  auto H = make_tensor({1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1}, 3, 4);
  std::vector<double> probs = {0.0, 1.0, 0.0, 0.0};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 3, probs, 0);

  for (size_t shot = 0; shot < 3; shot++) {
    EXPECT_EQ(errors.at({shot, 0}), 0);
    EXPECT_EQ(errors.at({shot, 1}), 1);
    EXPECT_EQ(errors.at({shot, 2}), 0);
    EXPECT_EQ(errors.at({shot, 3}), 0);
    EXPECT_EQ(checks.at({shot, 0}), 1);
    EXPECT_EQ(checks.at({shot, 1}), 1);
    EXPECT_EQ(checks.at({shot, 2}), 0);
  }
}

TEST(DemSamplingCPU, BackwardsCompatibility) {
  auto H = make_tensor({1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1}, 3, 5);
  std::vector<double> probs = {0.1, 0.2, 0.15, 0.05, 0.25};
  unsigned seed = 42;

  auto [old_checks, old_errors] = cudaq::qec::dem_sampling(H, 100, probs, seed);
  auto [new_checks, new_errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 100, probs, seed);

  for (size_t i = 0; i < 100; i++) {
    for (size_t j = 0; j < 3; j++)
      EXPECT_EQ(old_checks.at({i, j}), new_checks.at({i, j}));
    for (size_t j = 0; j < 5; j++)
      EXPECT_EQ(old_errors.at({i, j}), new_errors.at({i, j}));
  }
}

TEST(DemSamplingCPU, ZeroShots) {
  auto H = make_tensor({1, 0, 1, 0, 1, 1}, 2, 3);
  std::vector<double> probs = {0.5, 0.5, 0.5};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 0, probs, 42);

  EXPECT_EQ(checks.shape()[0], 0u);
  EXPECT_EQ(checks.shape()[1], 2u);
  EXPECT_EQ(errors.shape()[0], 0u);
  EXPECT_EQ(errors.shape()[1], 3u);
}

TEST(DemSamplingCPU, NonBinaryCheckMatrixMasked) {
  // H entries > 1 should be treated as H & 1 (matching GPU behavior).
  // H = [[2, 3], [1, 0]]  ->  binarized [[0, 1], [1, 0]]
  // probs = [1, 1]  ->  errors = [1, 1]
  // syndromes with binarized H: [0^1, 1^0] = [1, 1]
  auto H = make_tensor({2, 3, 1, 0}, 2, 2);
  std::vector<double> probs = {1.0, 1.0};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 4, probs, 0);

  for (size_t shot = 0; shot < 4; shot++) {
    EXPECT_EQ(errors.at({shot, 0}), 1);
    EXPECT_EQ(errors.at({shot, 1}), 1);
    EXPECT_EQ(checks.at({shot, 0}), 1)
        << "Binarized row 0: [0,1], sum = 1 mod 2 = 1";
    EXPECT_EQ(checks.at({shot, 1}), 1)
        << "Binarized row 1: [1,0], sum = 1 mod 2 = 1";
  }
}

TEST(DemSamplingCPU, SeedlessPathRuns) {
  auto H = make_tensor({1, 0, 1, 0, 1, 1}, 2, 3);
  std::vector<double> probs = {0.5, 0.5, 0.5};

  auto [checks, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, 10, probs);

  EXPECT_EQ(checks.shape()[0], 10u);
  EXPECT_EQ(errors.shape()[0], 10u);
  for (size_t shot = 0; shot < 10; shot++)
    for (size_t c = 0; c < 2; c++)
      EXPECT_TRUE(checks.at({shot, c}) == 0 || checks.at({shot, c}) == 1);
}

TEST(DemSamplingCPU, PublicSeedlessWrapperRuns) {
  // The public wrapper without a seed delegates to the CPU sampler and should
  // preserve the expected tensor shapes.
  auto H = make_tensor({1, 0, 1, 0, 1, 1}, 2, 3);
  std::vector<double> probs = {0.0, 1.0, 0.0};

  auto [checks, errors] = cudaq::qec::dem_sampling(H, 5, probs);

  EXPECT_EQ(checks.shape()[0], 5u);
  EXPECT_EQ(checks.shape()[1], 2u);
  EXPECT_EQ(errors.shape()[0], 5u);
  EXPECT_EQ(errors.shape()[1], 3u);
  for (size_t shot = 0; shot < 5; shot++) {
    EXPECT_EQ(errors.at({shot, 0}), 0);
    EXPECT_EQ(errors.at({shot, 1}), 1);
    EXPECT_EQ(errors.at({shot, 2}), 0);
    EXPECT_EQ(checks.at({shot, 0}), 0);
    EXPECT_EQ(checks.at({shot, 1}), 1);
  }
}

// =============================================================================
// GPU tests
// =============================================================================

class DemSamplingGPU : public ::testing::Test {
protected:
  static bool has_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }
  void SetUp() override {
    if (!has_gpu())
      GTEST_SKIP() << "No GPU available";
  }

  /// RAII wrapper for a set of GPU DEM sampling buffers.
  struct GpuBuffers {
    uint8_t *d_H = nullptr, *d_checks = nullptr, *d_errors = nullptr;
    double *d_probs = nullptr;
    size_t num_checks, num_errors, num_shots;

    GpuBuffers(const std::vector<uint8_t> &H, const std::vector<double> &probs,
               size_t checks, size_t errors, size_t shots)
        : num_checks(checks), num_errors(errors), num_shots(shots) {
      EXPECT_EQ(cudaMalloc(&d_H, checks * errors), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&d_probs, errors * sizeof(double)), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&d_checks, shots * checks), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&d_errors, shots * errors), cudaSuccess);
      EXPECT_EQ(
          cudaMemcpy(d_H, H.data(), checks * errors, cudaMemcpyHostToDevice),
          cudaSuccess);
      EXPECT_EQ(cudaMemcpy(d_probs, probs.data(), errors * sizeof(double),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
    }

    ~GpuBuffers() {
      cudaFree(d_H);
      cudaFree(d_probs);
      cudaFree(d_checks);
      cudaFree(d_errors);
    }

    bool run(unsigned seed) {
      return cudaq::qec::dem_sampler::gpu::sample_dem(
          d_H, num_checks, num_errors, d_probs, num_shots, seed, d_checks,
          d_errors);
    }

    std::vector<uint8_t> get_checks() {
      std::vector<uint8_t> out(num_shots * num_checks);
      EXPECT_EQ(
          cudaMemcpy(out.data(), d_checks, out.size(), cudaMemcpyDeviceToHost),
          cudaSuccess);
      return out;
    }

    std::vector<uint8_t> get_errors() {
      std::vector<uint8_t> out(num_shots * num_errors);
      EXPECT_EQ(
          cudaMemcpy(out.data(), d_errors, out.size(), cudaMemcpyDeviceToHost),
          cudaSuccess);
      return out;
    }

    GpuBuffers(const GpuBuffers &) = delete;
    GpuBuffers &operator=(const GpuBuffers &) = delete;
  };
};

TEST_F(DemSamplingGPU, AllZeroProbabilities) {
  const size_t num_checks = 3, num_errors = 5, num_shots = 10;
  std::vector<uint8_t> H = {1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1};
  std::vector<double> probs(num_errors, 0.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(42));

  for (auto v : buf.get_checks())
    EXPECT_EQ(v, 0);
  for (auto v : buf.get_errors())
    EXPECT_EQ(v, 0);
}

TEST_F(DemSamplingGPU, SparseCapacityClampsToShotCount) {
  // With zero probabilities and many shots, the initial sparse capacity
  // estimate is below num_shots and must be raised before cuStabilizer runs.
  const size_t num_checks = 2, num_errors = 3, num_shots = 2048;
  std::vector<uint8_t> H = {1, 0, 1, 0, 1, 1};
  std::vector<double> probs(num_errors, 0.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(42));

  for (auto v : buf.get_checks())
    EXPECT_EQ(v, 0);
  for (auto v : buf.get_errors())
    EXPECT_EQ(v, 0);
}

TEST_F(DemSamplingGPU, AllOneProbabilities) {
  //   H = | 1 0 1 0 |   probs = [1,1,1,1]
  //       | 1 1 0 1 |
  //       | 0 1 1 0 |
  // Expected syndromes: [0, 1, 0] (same as CPU test).
  const size_t num_checks = 3, num_errors = 4, num_shots = 5;
  std::vector<uint8_t> H = {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0};
  std::vector<double> probs(num_errors, 1.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(42));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    for (size_t e = 0; e < num_errors; e++)
      EXPECT_EQ(h_errors[shot * num_errors + e], 1);
    EXPECT_EQ(h_checks[shot * num_checks + 0], 0);
    EXPECT_EQ(h_checks[shot * num_checks + 1], 1);
    EXPECT_EQ(h_checks[shot * num_checks + 2], 0);
  }
}

TEST_F(DemSamplingGPU, MixedDeterministicProbs) {
  //   H = | 1 0 1 |   probs = [1, 0, 1]
  //       | 0 1 1 |
  // Expected: errors = [1,0,1], checks = [0, 1]
  const size_t num_checks = 2, num_errors = 3, num_shots = 8;
  std::vector<uint8_t> H = {1, 0, 1, 0, 1, 1};
  std::vector<double> probs = {1.0, 0.0, 1.0};

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(99));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    EXPECT_EQ(h_errors[shot * num_errors + 0], 1);
    EXPECT_EQ(h_errors[shot * num_errors + 1], 0);
    EXPECT_EQ(h_errors[shot * num_errors + 2], 1);
    EXPECT_EQ(h_checks[shot * num_checks + 0], 0);
    EXPECT_EQ(h_checks[shot * num_checks + 1], 1);
  }
}

TEST_F(DemSamplingGPU, IdentityMatrix) {
  const size_t N = 5, num_shots = 3;
  std::vector<uint8_t> H(N * N, 0);
  for (size_t i = 0; i < N; i++)
    H[i * N + i] = 1;
  std::vector<double> probs(N, 1.0);

  GpuBuffers buf(H, probs, N, N, num_shots);
  ASSERT_TRUE(buf.run(7));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++)
    for (size_t j = 0; j < N; j++) {
      EXPECT_EQ(h_errors[shot * N + j], 1);
      EXPECT_EQ(h_checks[shot * N + j], 1);
    }
}

TEST_F(DemSamplingGPU, AllOnesMatrixEvenColumns) {
  // H = 3x4 all-ones, p=1. Even column count => all syndromes = 0.
  const size_t num_checks = 3, num_errors = 4, num_shots = 4;
  std::vector<uint8_t> H(num_checks * num_errors, 1);
  std::vector<double> probs(num_errors, 1.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(0));

  for (auto v : buf.get_checks())
    EXPECT_EQ(v, 0) << "Even column count => syndrome 0";
}

TEST_F(DemSamplingGPU, AllOnesMatrixOddColumns) {
  // H = 3x3 all-ones, p=1. Odd column count => all syndromes = 1.
  const size_t N = 3, num_shots = 4;
  std::vector<uint8_t> H(N * N, 1);
  std::vector<double> probs(N, 1.0);

  GpuBuffers buf(H, probs, N, N, num_shots);
  ASSERT_TRUE(buf.run(0));

  for (auto v : buf.get_checks())
    EXPECT_EQ(v, 1) << "Odd column count => syndrome 1";
}

TEST_F(DemSamplingGPU, RepetitionCodeParity) {
  // H = [[1,1,0,0],[0,1,1,0],[0,0,1,1]], single error on qubit 1.
  const size_t num_checks = 3, num_errors = 4, num_shots = 3;
  std::vector<uint8_t> H = {1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1};
  std::vector<double> probs = {0.0, 1.0, 0.0, 0.0};

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(0));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    EXPECT_EQ(h_errors[shot * num_errors + 0], 0);
    EXPECT_EQ(h_errors[shot * num_errors + 1], 1);
    EXPECT_EQ(h_errors[shot * num_errors + 2], 0);
    EXPECT_EQ(h_errors[shot * num_errors + 3], 0);
    EXPECT_EQ(h_checks[shot * num_checks + 0], 1);
    EXPECT_EQ(h_checks[shot * num_checks + 1], 1);
    EXPECT_EQ(h_checks[shot * num_checks + 2], 0);
  }
}

TEST_F(DemSamplingGPU, SyndromeConsistency) {
  // Fixed H, verify syndrome = errors * H^T mod 2 for every shot.
  const size_t num_checks = 4, num_errors = 8, num_shots = 100;
  std::vector<uint8_t> H = {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                            0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<double> probs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(42));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    std::vector<uint8_t> shot_err(h_errors.begin() + shot * num_errors,
                                  h_errors.begin() + (shot + 1) * num_errors);
    auto expected = compute_syndrome(H, shot_err, num_checks, num_errors);
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(h_checks[shot * num_checks + c], expected[c])
          << "Shot " << shot << " check " << c;
  }
}

TEST_F(DemSamplingGPU, SeedReproducibility) {
  const size_t num_checks = 3, num_errors = 5, num_shots = 100;
  std::vector<uint8_t> H = {1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1};
  std::vector<double> probs = {0.1, 0.3, 0.5, 0.7, 0.9};

  GpuBuffers buf1(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf1.run(42));
  auto checks1 = buf1.get_checks();
  auto errors1 = buf1.get_errors();

  GpuBuffers buf2(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf2.run(42));
  auto checks2 = buf2.get_checks();
  auto errors2 = buf2.get_errors();

  EXPECT_EQ(checks1, checks2) << "Same seed must produce identical syndromes";
  EXPECT_EQ(errors1, errors2) << "Same seed must produce identical errors";
}

TEST_F(DemSamplingGPU, CpuGpuCrossValidation) {
  // Use deterministic probs (0 and 1) so CPU and GPU must match exactly.
  const size_t num_checks = 3, num_errors = 6, num_shots = 20;
  std::vector<uint8_t> H_data = {1, 0, 1, 0, 0, 1, 0, 1, 0,
                                 1, 0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<double> probs = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};

  auto H_tensor = make_tensor(H_data, num_checks, num_errors);
  auto [cpu_checks, cpu_errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H_tensor, num_shots, probs, 42);

  GpuBuffers buf(H_data, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(42));
  auto gpu_checks = buf.get_checks();
  auto gpu_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    for (size_t e = 0; e < num_errors; e++)
      EXPECT_EQ(gpu_errors[shot * num_errors + e], cpu_errors.at({shot, e}))
          << "Shot " << shot << " error " << e;
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(gpu_checks[shot * num_checks + c], cpu_checks.at({shot, c}))
          << "Shot " << shot << " check " << c;
  }
}

TEST_F(DemSamplingGPU, BinaryOutputOnly) {
  const size_t num_checks = 3, num_errors = 5, num_shots = 200;
  std::vector<uint8_t> H = {1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1};
  std::vector<double> probs = {0.2, 0.4, 0.6, 0.8, 0.5};

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(123));

  for (auto v : buf.get_checks())
    EXPECT_TRUE(v == 0 || v == 1)
        << "Syndrome value must be 0 or 1, got " << (int)v;
  for (auto v : buf.get_errors())
    EXPECT_TRUE(v == 0 || v == 1)
        << "Error value must be 0 or 1, got " << (int)v;
}

TEST_F(DemSamplingGPU, BitpackBoundary32Columns) {
  // 32 columns lands exactly on a uint32 word boundary — no padding bits.
  const size_t num_checks = 4, num_errors = 32, num_shots = 50;
  std::vector<uint8_t> H(num_checks * num_errors, 0);
  for (size_t i = 0; i < num_checks; i++)
    for (size_t j = 0; j < num_errors; j++)
      H[i * num_errors + j] = ((i + j) % 3 == 0) ? 1 : 0;
  std::vector<double> probs(num_errors, 1.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(77));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t e = 0; e < num_errors; e++)
    EXPECT_EQ(h_errors[e], 1) << "p=1 => all errors fire";

  for (size_t shot = 0; shot < num_shots; shot++) {
    std::vector<uint8_t> shot_err(h_errors.begin() + shot * num_errors,
                                  h_errors.begin() + (shot + 1) * num_errors);
    auto expected = compute_syndrome(H, shot_err, num_checks, num_errors);
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(h_checks[shot * num_checks + c], expected[c])
          << "Shot " << shot << " check " << c;
  }
}

TEST_F(DemSamplingGPU, BitpackBoundary33Columns) {
  // 33 columns = 1 word + 1 bit, tests partial-word handling.
  const size_t num_checks = 3, num_errors = 33, num_shots = 50;
  std::vector<uint8_t> H(num_checks * num_errors, 0);
  for (size_t i = 0; i < num_checks; i++)
    for (size_t j = 0; j < num_errors; j++)
      H[i * num_errors + j] = ((i + j) % 2 == 0) ? 1 : 0;
  std::vector<double> probs(num_errors, 1.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(88));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    std::vector<uint8_t> shot_err(h_errors.begin() + shot * num_errors,
                                  h_errors.begin() + (shot + 1) * num_errors);
    auto expected = compute_syndrome(H, shot_err, num_checks, num_errors);
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(h_checks[shot * num_checks + c], expected[c])
          << "Shot " << shot << " check " << c;
  }
}

TEST_F(DemSamplingGPU, BitpackBoundary64Columns) {
  // 64 columns = exactly 2 uint32 words.
  const size_t num_checks = 4, num_errors = 64, num_shots = 40;
  std::vector<uint8_t> H(num_checks * num_errors, 0);
  for (size_t i = 0; i < num_checks; i++)
    for (size_t j = 0; j < num_errors; j++)
      H[i * num_errors + j] = ((i * 7 + j * 3) % 5 == 0) ? 1 : 0;
  std::vector<double> probs(num_errors, 1.0);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(99));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    std::vector<uint8_t> shot_err(h_errors.begin() + shot * num_errors,
                                  h_errors.begin() + (shot + 1) * num_errors);
    auto expected = compute_syndrome(H, shot_err, num_checks, num_errors);
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(h_checks[shot * num_checks + c], expected[c])
          << "Shot " << shot << " check " << c;
  }
}

TEST_F(DemSamplingGPU, LargeScaleSyndromeConsistency) {
  // Larger problem: 20 checks, 100 errors, 1000 shots.
  const size_t num_checks = 20, num_errors = 100, num_shots = 1000;

  std::mt19937 gen(12345);
  std::bernoulli_distribution dist(0.3);
  std::vector<uint8_t> H(num_checks * num_errors);
  for (auto &v : H)
    v = dist(gen) ? 1 : 0;

  std::uniform_real_distribution<double> pdist(0.01, 0.5);
  std::vector<double> probs(num_errors);
  for (auto &p : probs)
    p = pdist(gen);

  GpuBuffers buf(H, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(42));

  auto h_checks = buf.get_checks();
  auto h_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    std::vector<uint8_t> shot_err(h_errors.begin() + shot * num_errors,
                                  h_errors.begin() + (shot + 1) * num_errors);
    auto expected = compute_syndrome(H, shot_err, num_checks, num_errors);
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(h_checks[shot * num_checks + c], expected[c])
          << "Shot " << shot << " check " << c;
  }
}

TEST_F(DemSamplingGPU, ZeroShots) {
  const size_t num_checks = 3, num_errors = 5;
  std::vector<uint8_t> H = {1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1};
  std::vector<double> probs(num_errors, 0.5);

  // num_shots = 0 should return true immediately with no work
  ASSERT_TRUE(cudaq::qec::dem_sampler::gpu::sample_dem(
      nullptr, num_checks, num_errors, nullptr, 0, 42, nullptr, nullptr));
}

TEST_F(DemSamplingGPU, NonBinaryCheckMatrixCpuGpuMatch) {
  // H with entries > 1: both paths should agree after binarization.
  const size_t num_checks = 2, num_errors = 2, num_shots = 4;
  std::vector<uint8_t> H_data = {2, 3, 1, 0};
  std::vector<double> probs = {1.0, 1.0};

  auto H_tensor = make_tensor(H_data, num_checks, num_errors);
  auto [cpu_checks, cpu_errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H_tensor, num_shots, probs, 0);

  GpuBuffers buf(H_data, probs, num_checks, num_errors, num_shots);
  ASSERT_TRUE(buf.run(0));
  auto gpu_checks = buf.get_checks();
  auto gpu_errors = buf.get_errors();

  for (size_t shot = 0; shot < num_shots; shot++) {
    for (size_t e = 0; e < num_errors; e++)
      EXPECT_EQ(gpu_errors[shot * num_errors + e], cpu_errors.at({shot, e}))
          << "Shot " << shot << " error " << e;
    for (size_t c = 0; c < num_checks; c++)
      EXPECT_EQ(gpu_checks[shot * num_checks + c], cpu_checks.at({shot, c}))
          << "Shot " << shot << " check " << c;
  }
}
