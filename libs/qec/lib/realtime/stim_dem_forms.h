/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the rules a decoder's or task-graph node's Stim DEM keys
// share, so the parser and the host builder, which checks a configuration
// built in code again, name them alike. Never included from a public header.

#pragma once

#include "cudaq/qec/realtime/decoding_config.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// The whitespace that tells a data_artifacts name, which holds none, from
/// Stim model text, which always holds some: Stim separates targets with
/// spaces or tabs and instructions with line breaks. Other whitespace, such
/// as Unicode's, may appear in a name.
inline constexpr const char *name_breaks = " \t\r\n";

/// Checks data_artifacts entries each on its own and against the others, as
/// parsing does. @p list is sorted by name.
/// @throws std::runtime_error naming the first bad entry.
void validate_data_artifacts(const std::vector<data_artifact> &list);

/// Resolves a stim_dem that names an entry of @p artifacts, sorted by name, as
/// parsing does: a value with no whitespace, which Stim model text always
/// has, moves to @p artifact, and @p text or @p path becomes a copy of the
/// entry's text or path. @p where names the decoder or node in errors.
/// @throws std::runtime_error if no Stim DEM entry has that name, or if the
/// entry's file meets a @p path of the decoder's or node's own.
void resolve_stim_dem(const std::vector<data_artifact> &artifacts,
                      std::string &text, std::string &path,
                      std::string &artifact, const std::string &where);

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
