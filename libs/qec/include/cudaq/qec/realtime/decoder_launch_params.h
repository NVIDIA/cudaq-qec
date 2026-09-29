/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace cudaq::qec::realtime {

/// @brief Configuration for launching a decoder kernel.
///
/// This struct holds the launch parameters required for the dispatch kernel,
/// including dynamic shared memory size and grid/block dimensions.
/// Different decoders may require different amounts of shared memory
struct decoder_launch_config {
  /// Dynamic shared memory size in bytes (0 for no dynamic shared memory)
  std::size_t shared_memory_bytes = 0;

  /// Grid dimensions for kernel launch
  struct {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
  } grid_size;

  /// Block dimensions for kernel launch
  struct {
    std::uint32_t x = 256;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
  } block_size;
};

/// @brief Interface for decoder launch parameter providers.
///
/// Each decoder implementation should provide a class that implements
/// this interface to specify the required launch configuration based
/// on the problem parameters.
///
/// Example implementations:
/// - mock_decoder_launch_params (in unittests/realtime): Returns minimal
/// defaults for testing
/// - bp_decoder_launch_params: Calculates shared memory for Relay BP
/// - tensorrt_decoder_launch_params: Configuration for TensorRT graph launch
class i_decoder_launch_params {
public:
  virtual ~i_decoder_launch_params() = default;

  /// @brief Get the launch configuration for this decoder.
  ///
  /// @param num_shots Number of shots to process
  /// @param syndrome_size Size of syndrome data per shot
  /// @param block_size Number of error mechanisms
  /// @param num_detectors Number of detectors
  /// @param num_observables Number of observables
  /// @return decoder_launch_config with computed launch parameters
  virtual decoder_launch_config
  get_launch_config(std::size_t num_shots, std::size_t syndrome_size,
                    std::size_t block_size, std::size_t num_detectors,
                    std::size_t num_observables) const = 0;
};

} // namespace cudaq::qec::realtime
