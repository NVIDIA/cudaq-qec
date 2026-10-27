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
/// set, is none, fail or report. No value is a YAML boolean, so YAML tools
/// read each as the string the schema declares.
inline void check_verify_residual(const cudaqx::heterogeneous_map &params,
                                  const std::string &kind) {
  if (!params.contains("verify_residual"))
    return;
  const auto value = params.get<std::string>("verify_residual");
  if (value != "none" && value != "fail" && value != "report")
    throw std::runtime_error(kind +
                             ": verify_residual must be none, fail or report; "
                             "got '" +
                             value + "'");
}

} // namespace cudaq::qec
