/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: parameter checks shared by the windowed-decoding node
// kinds, decode_window and the parallel_window pattern, which pass both
// parameters through to the layers they describe.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <cstddef>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

/// Throws std::runtime_error, naming @p kind, unless verify_residual, when
/// set, is false, fail (or true), or report.
inline void check_verify_residual(const cudaqx::heterogeneous_map &params,
                                  const std::string &kind) {
  if (!params.contains("verify_residual"))
    return;
  const auto value = params.get<std::string>("verify_residual");
  if (value != "false" && value != "true" && value != "fail" &&
      value != "report")
    throw std::runtime_error(kind +
                             ": verify_residual must be false, fail (or true), "
                             "or report; got '" +
                             value + "'");
}

/// Throws std::runtime_error, naming @p kind, if max_in_flight is set to 0.
inline void check_max_in_flight(const cudaqx::heterogeneous_map &params,
                                const std::string &kind) {
  if (params.contains("max_in_flight") &&
      params.get<std::size_t>("max_in_flight") == 0)
    throw std::runtime_error(kind + ": max_in_flight must be at least 1");
}

} // namespace cudaq::qec
