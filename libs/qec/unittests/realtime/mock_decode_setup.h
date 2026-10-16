/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file mock_decode_setup.h
/// @brief Host-side helpers for mock decoder setup.
///
/// This header-only file provides shared utilities used by both the
/// test_realtime_decoding test and the gpu_roce_mock_decoder_bridge tool:
///   - Config YAML scalar parsing
///   - Syndrome file loading
///   - Mock decoder context GPU setup
///   - Dispatch kernel launch wrapper
///
/// All functions are plain C++ (no __device__ code) and can be compiled
/// by any C++17 compiler without nvcc.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "mock_decode_handler.cuh"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

namespace cudaq::qec::realtime {

//==============================================================================
// Config YAML Parsing
//==============================================================================

/// @brief Parse a scalar integer value from simple YAML content.
/// @param content Full YAML file content as a string
/// @param field_name YAML key name (e.g., "syndrome_size")
/// @return Parsed value, or 0 on failure
inline uint64_t parse_scalar(const std::string &content,
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
  value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
  value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);

  try {
    return std::stoull(value_str);
  } catch (...) {
    return 0;
  }
}

//==============================================================================
// Syndrome Loading
//==============================================================================

/// @brief A single syndrome entry: measurements and expected correction.
struct SyndromeEntry {
  std::vector<uint8_t> measurements;
  uint8_t expected_correction;
};

/// @brief Load syndrome entries from a text file.
///
/// File format uses SHOT_START / CORRECTIONS_START / CORRECTIONS_END markers.
///
/// @param path Path to syndrome text file
/// @param syndrome_size Expected number of measurements per shot
/// @return Vector of syndrome entries (empty on failure)
inline std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                                 std::size_t syndrome_size) {
  std::vector<SyndromeEntry> entries;
  std::ifstream file(path);
  if (!file.good())
    return entries;

  std::string line;
  std::vector<uint8_t> current_shot;
  std::vector<std::vector<uint8_t>> shots;
  std::vector<uint8_t> corrections;
  bool reading_shot = false;
  bool reading_corrections = false;

  while (std::getline(file, line)) {
    if (line.find("SHOT_START") == 0) {
      if (reading_shot && !current_shot.empty()) {
        shots.push_back(current_shot);
      }
      current_shot.clear();
      reading_shot = true;
      reading_corrections = false;
      continue;
    }
    if (line == "CORRECTIONS_START") {
      if (reading_shot && !current_shot.empty()) {
        shots.push_back(current_shot);
      }
      current_shot.clear();
      reading_shot = false;
      reading_corrections = true;
      continue;
    }
    if (line == "CORRECTIONS_END") {
      break;
    }
    if (line.find("NUM_DATA") == 0 || line.find("NUM_LOGICAL") == 0) {
      continue;
    } else if (line.find("ROUND_START") == 0) {
      continue;
    } else if (reading_shot) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        current_shot.push_back(static_cast<uint8_t>(bit));
      } catch (...) {
      }
    } else if (reading_corrections) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        corrections.push_back(static_cast<uint8_t>(bit));
      } catch (...) {
      }
    }
  }

  if (reading_shot && !current_shot.empty()) {
    shots.push_back(current_shot);
  }

  for (std::size_t i = 0; i < shots.size(); ++i) {
    if (shots[i].size() < syndrome_size) {
      shots[i].resize(syndrome_size, 0);
    } else if (shots[i].size() > syndrome_size) {
      shots[i].resize(syndrome_size);
    }
    SyndromeEntry entry{};
    entry.measurements = std::move(shots[i]);
    entry.expected_correction = (i < corrections.size()) ? corrections[i] : 0;
    entries.push_back(std::move(entry));
  }

  return entries;
}

//==============================================================================
// Mock Decoder Context Setup (convenience wrapper)
//==============================================================================

// Forward-declared from mock_decode_handler.cuh -- the actual function is
// compiled by nvcc and lives in the test or mock library.
// We only need the struct and function signature here; callers that include
// mock_decode_handler.cuh get the full declaration.

/// @brief Convenience helper: flatten SyndromeEntry vector and call the library
///        setup_mock_decoder_on_gpu() that lives in the nvcc-compiled library.
///
/// This function can be called from plain C++ code; the underlying
/// cudaMemcpyToSymbol call happens inside the library.
///
/// @param syndromes Loaded syndrome entries
/// @param syndrome_size Number of measurements per entry
/// @param[out] resources Device pointers (caller must call cleanup())
/// @return cudaSuccess on success
inline cudaError_t setup_mock_decoder_from_syndromes(
    const std::vector<SyndromeEntry> &syndromes, std::size_t syndrome_size,
    cudaq::qec::realtime::MockDecoderGpuResources &resources) {

  std::size_t num_entries = syndromes.size();
  std::vector<uint8_t> measurements(num_entries * syndrome_size);
  std::vector<uint8_t> corrections(num_entries);

  for (std::size_t i = 0; i < num_entries; ++i) {
    std::size_t copy_size =
        std::min(syndromes[i].measurements.size(), syndrome_size);
    for (std::size_t j = 0; j < copy_size; ++j)
      measurements[i * syndrome_size + j] = syndromes[i].measurements[j];
    corrections[i] = syndromes[i].expected_correction;
  }

  return cudaq::qec::realtime::setup_mock_decoder_on_gpu(
      measurements.data(), corrections.data(), num_entries, syndrome_size,
      resources);
}

//==============================================================================
// Dispatch Kernel Launch Wrapper
//==============================================================================

/// @brief C-compatible launch wrapper matching cudaq_dispatch_launch_fn_t.
///
/// Delegates to cudaq_launch_dispatch_kernel_regular from libcudaq-realtime.
inline void mock_decode_launch_dispatch_kernel(
    volatile std::uint64_t *rx_flags, volatile std::uint64_t *tx_flags,
    std::uint8_t *rx_data, std::uint8_t *tx_data, std::size_t rx_stride_sz,
    std::size_t tx_stride_sz, cudaq_function_entry_t *function_table,
    std::size_t func_count, volatile int *shutdown_flag, std::uint64_t *stats,
    std::size_t num_slots, std::uint32_t num_blocks,
    std::uint32_t threads_per_block, cudaStream_t stream) {
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, rx_data, tx_data, rx_stride_sz, tx_stride_sz,
      function_table, func_count, shutdown_flag, stats, num_slots, num_blocks,
      threads_per_block, stream);
}

} // namespace cudaq::qec::realtime
