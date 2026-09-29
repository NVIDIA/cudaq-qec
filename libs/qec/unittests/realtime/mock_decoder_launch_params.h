/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qec/realtime/decoder_launch_params.h"

namespace cudaq::qec::realtime {

/// @brief Launch parameter provider for the mock decoder.
///
/// The mock decoder is used for CI testing and doesn't require any
/// dynamic shared memory. It returns minimal default configurations.
class mock_decoder_launch_params : public i_decoder_launch_params {
public:
  /// @brief Get the launch configuration for the mock decoder.
  ///
  /// Returns minimal defaults since the mock decoder simply does
  /// a table lookup and doesn't need special GPU resources.
  ///
  /// @return decoder_launch_config with shared_memory_bytes = 0
  decoder_launch_config
  get_launch_config(std::size_t num_shots, std::size_t syndrome_size,
                    std::size_t block_size, std::size_t num_detectors,
                    std::size_t num_observables) const override;
};

} // namespace cudaq::qec::realtime
