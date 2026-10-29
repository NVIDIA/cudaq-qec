/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: verify_residual's modes and their spellings, shared by
// the windowed-decoding node kinds, decode_window and the parallel_window
// pattern, which passes the parameter through to the layers it describes,
// and by what builds a decode_window core from a node's parameters. Their
// minimums are declared with the kinds.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <cstddef>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

/// What a fire does with a solution that does not satisfy its window's
/// detectors, H e != d: the decode_window node's verify_residual.
enum class verify_residual_mode {
  none,   ///< Nothing: its record says the window was not checked
  fail,   ///< The fire throws, naming the window and the first such row
  report, ///< Its record says the window was checked and does not hold
};

/// The verify_residual @p params set, none when they set none. No spelling
/// is a YAML boolean, so YAML tools read each as the string the schema
/// declares. A kind's parameter check calls it for its error alone.
/// @throws std::runtime_error, naming @p kind, unless verify_residual is
/// none, fail or report.
inline verify_residual_mode
verify_residual_of(const cudaqx::heterogeneous_map &params,
                   const std::string &kind) {
  if (!params.contains("verify_residual"))
    return verify_residual_mode::none;
  const auto value = params.get<std::string>("verify_residual");
  if (value == "none")
    return verify_residual_mode::none;
  if (value == "fail")
    return verify_residual_mode::fail;
  if (value == "report")
    return verify_residual_mode::report;
  throw std::runtime_error(kind +
                           ": verify_residual must be none, fail or report; "
                           "got '" +
                           value + "'");
}

} // namespace cudaq::qec
