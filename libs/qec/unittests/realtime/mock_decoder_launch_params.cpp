/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mock_decoder_launch_params.h"

namespace cudaq::qec::realtime {

decoder_launch_config mock_decoder_launch_params::get_launch_config(
    std::size_t num_shots, std::size_t syndrome_size, std::size_t block_size,
    std::size_t num_detectors, std::size_t num_observables) const {

  decoder_launch_config config;

  // Mock decoder doesn't need dynamic shared memory
  config.shared_memory_bytes = 0;

  // Use a single block with default thread count for mock decoder
  // This is sufficient since mock decoder just does table lookups
  config.grid_size.x = 1;
  config.grid_size.y = 1;
  config.grid_size.z = 1;

  config.block_size.x = 256;
  config.block_size.y = 1;
  config.block_size.z = 1;

  return config;
}

} // namespace cudaq::qec::realtime
