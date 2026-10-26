/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "cudaq/qec/realtime/gpu_kernels.cuh"
#include "cudaq/qec/realtime/sparse_to_csr.h"

// Use explicit namespace to avoid conflict with ::float_t from math.h
using realtime_float_t = cudaq::qec::realtime::float_t;

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

// Test kernel wrapper for preprocess_all
__global__ void test_preprocess_kernel(const uint8_t *measurements,
                                       const uint32_t *D_row_ptr,
                                       const uint32_t *D_col_idx,
                                       realtime_float_t *soft_syndrome,
                                       std::size_t num_detectors) {
  cudaq::qec::realtime::preprocess_all(measurements, D_row_ptr, D_col_idx,
                                       soft_syndrome, num_detectors);
}

// Test kernel wrapper for postprocess_all
__global__ void test_postprocess_kernel(const realtime_float_t *soft_decisions,
                                        const uint32_t *O_row_ptr,
                                        const uint32_t *O_col_idx,
                                        uint8_t *corrections,
                                        std::size_t num_observables) {
  cudaq::qec::realtime::postprocess_all(soft_decisions, O_row_ptr, O_col_idx,
                                        corrections, num_observables);
}

// Simple YAML-like parser for sparse vectors (to avoid including
// decoding_config.h in CUDA)
std::vector<int64_t> parse_sparse_vector(const std::string &content,
                                         const std::string &field_name) {
  std::vector<int64_t> result;

  // Find the field
  std::size_t pos = content.find(field_name + ":");
  if (pos == std::string::npos)
    return result;

  // Find the opening bracket
  pos = content.find('[', pos);
  if (pos == std::string::npos)
    return result;

  // Find the closing bracket
  std::size_t end_pos = content.find(']', pos);
  if (end_pos == std::string::npos)
    return result;

  // Extract and parse the values
  std::string values_str = content.substr(pos + 1, end_pos - pos - 1);
  std::istringstream iss(values_str);
  std::string token;

  while (std::getline(iss, token, ',')) {
    // Trim whitespace
    token.erase(0, token.find_first_not_of(" \t\n\r"));
    token.erase(token.find_last_not_of(" \t\n\r") + 1);
    if (!token.empty()) {
      try {
        result.push_back(std::stoll(token));
      } catch (...) {
        // Skip invalid values
      }
    }
  }

  return result;
}

// Parse a scalar value from YAML-like content
uint64_t parse_scalar(const std::string &content,
                      const std::string &field_name) {
  std::size_t pos = content.find(field_name + ":");
  if (pos == std::string::npos)
    return 0;

  pos = content.find(':', pos);
  if (pos == std::string::npos)
    return 0;

  std::size_t end_pos = content.find_first_of("\n[", pos + 1);
  if (end_pos == std::string::npos)
    end_pos = content.length();

  std::string value_str = content.substr(pos + 1, end_pos - pos - 1);
  // Trim whitespace
  value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
  value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);

  try {
    return std::stoull(value_str);
  } catch (...) {
    return 0;
  }
}

// Structure to hold syndrome data
struct SyndromeEntry {
  std::vector<uint8_t> measurements;
  uint8_t correction;
};

// Load syndrome data from file
std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                          std::size_t syndrome_size) {
  std::vector<SyndromeEntry> entries;
  std::ifstream file(path);
  if (!file.good())
    return entries;

  std::string line;
  SyndromeEntry current_entry;
  bool reading_shot = false;

  while (std::getline(file, line)) {
    if (line.find("SHOT_START") == 0) {
      if (reading_shot && !current_entry.measurements.empty()) {
        entries.push_back(current_entry);
      }
      current_entry = SyndromeEntry{};
      reading_shot = true;
    } else if (line.find("ROUND_START") == 0) {
      // Per-callback round marker introduced for the per-round dispatch
      // demo.  This loader concatenates per-shot bits and ignores round
      // boundaries -- skip ROUND_START silently.
      continue;
    } else if (reading_shot) {
      // Trim whitespace
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;

      try {
        int bit = std::stoi(line);
        // First syndrome_size entries are measurements, then comes the
        // correction
        if (current_entry.measurements.size() < syndrome_size) {
          current_entry.measurements.push_back(static_cast<uint8_t>(bit));
        } else {
          current_entry.correction = static_cast<uint8_t>(bit);
        }
      } catch (...) {
        // Skip non-numeric lines
      }
    }
  }

  // Don't forget the last entry
  if (reading_shot && !current_entry.measurements.empty()) {
    entries.push_back(current_entry);
  }

  return entries;
}

// Helper function to check if a GPU is available
bool isGpuAvailable() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return (err == cudaSuccess && deviceCount > 0);
}

} // namespace

class GpuKernelsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Skip all tests if no GPU is available
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping GPU kernel tests";
    }
    // Get the path to the test data directory
    data_dir_ = std::string(TEST_DATA_DIR);

    // Load config file
    std::ifstream config_file(data_dir_ + "/config_multi_err_lut.yml");
    ASSERT_TRUE(config_file.good()) << "Could not open config file";
    std::string config_content((std::istreambuf_iterator<char>(config_file)),
                               std::istreambuf_iterator<char>());

    // Parse the config
    syndrome_size_ = parse_scalar(config_content, "syndrome_size");
    block_size_ = parse_scalar(config_content, "block_size");
    ASSERT_GT(syndrome_size_, 0u) << "Failed to parse syndrome_size";
    ASSERT_GT(block_size_, 0u) << "Failed to parse block_size";

    // Parse sparse matrices
    auto D_sparse = parse_sparse_vector(config_content, "D_sparse");
    auto O_sparse = parse_sparse_vector(config_content, "O_sparse");
    ASSERT_FALSE(D_sparse.empty()) << "Failed to parse D_sparse";
    ASSERT_FALSE(O_sparse.empty()) << "Failed to parse O_sparse";

    // Convert to CSR
    num_detectors_ = cudaq::qec::realtime::sparse_vec_to_csr(
        D_sparse, D_row_ptr_, D_col_idx_);
    num_observables_ = cudaq::qec::realtime::sparse_vec_to_csr(
        O_sparse, O_row_ptr_, O_col_idx_);

    ASSERT_GT(num_detectors_, 0u) << "No detectors found";
    ASSERT_GT(num_observables_, 0u) << "No observables found";

    // Load syndrome test data
    syndromes_ = load_syndromes(data_dir_ + "/syndromes_multi_err_lut.txt",
                                syndrome_size_);
    ASSERT_FALSE(syndromes_.empty()) << "No syndrome data loaded";
  }

  std::string data_dir_;

  std::vector<uint32_t> D_row_ptr_;
  std::vector<uint32_t> D_col_idx_;
  std::vector<uint32_t> O_row_ptr_;
  std::vector<uint32_t> O_col_idx_;

  std::size_t num_detectors_ = 0;
  std::size_t num_observables_ = 0;
  std::size_t syndrome_size_ = 0;
  std::size_t block_size_ = 0;

  std::vector<SyndromeEntry> syndromes_;
};

TEST_F(GpuKernelsTest, PreprocessDetectorBasic) {
  // Test preprocessing with the first syndrome entry
  ASSERT_FALSE(syndromes_.empty());
  const auto &entry = syndromes_[0];

  // Allocate device memory
  uint8_t *d_measurements = nullptr;
  uint32_t *d_D_row_ptr = nullptr;
  uint32_t *d_D_col_idx = nullptr;
  realtime_float_t *d_soft_syndrome = nullptr;

  CUDA_CHECK(cudaMalloc(&d_measurements, entry.measurements.size()));
  CUDA_CHECK(cudaMalloc(&d_D_row_ptr, D_row_ptr_.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_D_col_idx, D_col_idx_.size() * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_soft_syndrome, num_detectors_ * sizeof(realtime_float_t)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_measurements, entry.measurements.data(),
                        entry.measurements.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_row_ptr, D_row_ptr_.data(),
                        D_row_ptr_.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_col_idx, D_col_idx_.data(),
                        D_col_idx_.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  test_preprocess_kernel<<<1, 256>>>(d_measurements, d_D_row_ptr, d_D_col_idx,
                                     d_soft_syndrome, num_detectors_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  std::vector<realtime_float_t> soft_syndrome(num_detectors_);
  CUDA_CHECK(cudaMemcpy(soft_syndrome.data(), d_soft_syndrome,
                        num_detectors_ * sizeof(realtime_float_t),
                        cudaMemcpyDeviceToHost));

  // Verify: soft syndrome values should be 0.0 or 1.0
  for (std::size_t i = 0; i < num_detectors_; ++i) {
    EXPECT_TRUE(soft_syndrome[i] == 0.0 || soft_syndrome[i] == 1.0)
        << "Detector " << i << " has invalid value: " << soft_syndrome[i];
  }

  // Cleanup
  cudaFree(d_measurements);
  cudaFree(d_D_row_ptr);
  cudaFree(d_D_col_idx);
  cudaFree(d_soft_syndrome);
}

TEST_F(GpuKernelsTest, PostprocessObservableBasic) {
  // Create some test soft decisions (all zeros should produce zero corrections)
  std::vector<realtime_float_t> soft_decisions(block_size_, 0.0);

  // Allocate device memory
  realtime_float_t *d_soft_decisions = nullptr;
  uint32_t *d_O_row_ptr = nullptr;
  uint32_t *d_O_col_idx = nullptr;
  uint8_t *d_corrections = nullptr;

  CUDA_CHECK(cudaMalloc(&d_soft_decisions,
                        soft_decisions.size() * sizeof(realtime_float_t)));
  CUDA_CHECK(cudaMalloc(&d_O_row_ptr, O_row_ptr_.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_O_col_idx, O_col_idx_.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_corrections, num_observables_));
  CUDA_CHECK(cudaMemset(d_corrections, 0, num_observables_));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_soft_decisions, soft_decisions.data(),
                        soft_decisions.size() * sizeof(realtime_float_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_O_row_ptr, O_row_ptr_.data(),
                        O_row_ptr_.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_O_col_idx, O_col_idx_.data(),
                        O_col_idx_.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  test_postprocess_kernel<<<1, 256>>>(d_soft_decisions, d_O_row_ptr,
                                      d_O_col_idx, d_corrections,
                                      num_observables_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  std::vector<uint8_t> corrections(num_observables_);
  CUDA_CHECK(cudaMemcpy(corrections.data(), d_corrections, num_observables_,
                        cudaMemcpyDeviceToHost));

  // All zeros input should produce all zeros output
  for (std::size_t i = 0; i < num_observables_; ++i) {
    EXPECT_EQ(corrections[i], 0) << "Observable " << i << " should be 0";
  }

  // Cleanup
  cudaFree(d_soft_decisions);
  cudaFree(d_O_row_ptr);
  cudaFree(d_O_col_idx);
  cudaFree(d_corrections);
}

TEST_F(GpuKernelsTest, PreprocessMultipleShots) {
  // Test preprocessing with multiple syndrome entries
  const std::size_t num_test_shots =
      std::min(syndromes_.size(), static_cast<std::size_t>(10));

  for (std::size_t shot = 0; shot < num_test_shots; ++shot) {
    const auto &entry = syndromes_[shot];

    // Allocate device memory
    uint8_t *d_measurements = nullptr;
    uint32_t *d_D_row_ptr = nullptr;
    uint32_t *d_D_col_idx = nullptr;
    realtime_float_t *d_soft_syndrome = nullptr;

    CUDA_CHECK(cudaMalloc(&d_measurements, entry.measurements.size()));
    CUDA_CHECK(cudaMalloc(&d_D_row_ptr, D_row_ptr_.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_D_col_idx, D_col_idx_.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_soft_syndrome,
                          num_detectors_ * sizeof(realtime_float_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_measurements, entry.measurements.data(),
                          entry.measurements.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D_row_ptr, D_row_ptr_.data(),
                          D_row_ptr_.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D_col_idx, D_col_idx_.data(),
                          D_col_idx_.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Launch kernel
    test_preprocess_kernel<<<1, 256>>>(d_measurements, d_D_row_ptr, d_D_col_idx,
                                       d_soft_syndrome, num_detectors_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<realtime_float_t> soft_syndrome(num_detectors_);
    CUDA_CHECK(cudaMemcpy(soft_syndrome.data(), d_soft_syndrome,
                          num_detectors_ * sizeof(realtime_float_t),
                          cudaMemcpyDeviceToHost));

    // Verify: soft syndrome values should be 0.0 or 1.0
    for (std::size_t i = 0; i < num_detectors_; ++i) {
      EXPECT_TRUE(soft_syndrome[i] == 0.0 || soft_syndrome[i] == 1.0)
          << "Shot " << shot << ", detector " << i
          << " has invalid value: " << soft_syndrome[i];
    }

    // Cleanup
    cudaFree(d_measurements);
    cudaFree(d_D_row_ptr);
    cudaFree(d_D_col_idx);
    cudaFree(d_soft_syndrome);
  }
}
