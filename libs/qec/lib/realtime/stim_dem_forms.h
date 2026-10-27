/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the rules a decoder's or task-graph node's Stim DEM keys
// share, so the parser and the host builder, which checks a configuration
// built in code again, name them alike. Never included from a public header.

#pragma once

#include <stdexcept>
#include <string>

namespace cudaq::qec::decoding::config {

/// The key a Stim DEM is given under: stim_dem_path when @p stim_dem_path
/// names a file, else stim_dem.
inline const char *stim_dem_key(const std::string &stim_dem_path) {
  return stim_dem_path.empty() ? "stim_dem" : "stim_dem_path";
}

/// The error for a decoder or node, named by @p where, that sets both
/// stim_dem and stim_dem_path.
inline std::runtime_error both_stim_forms(const std::string &where) {
  return std::runtime_error(where +
                            " sets both stim_dem and stim_dem_path; the model "
                            "comes in exactly one form, inline or by file");
}

} // namespace cudaq::qec::decoding::config
