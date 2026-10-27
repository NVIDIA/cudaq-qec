/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: YAML for one window request, the shape of an entry of an
// ingest node's `windows` map. Never included from a public header.

#pragma once

#include "cudaq/qec/window_request.h"
#include <string>

namespace cudaq::qec {

/// @brief Parse one window request.
///
/// Every round range is written [begin, end] or {rounds: [begin, end]}; the
/// map form reserves `occurrences` and `block`. `ports` names only `input`
/// and `output`. The window keys `responsibility` and `soundness_class` are
/// reserved. A send's `rounds` is one range, or a list of ranges in either
/// form. `period`, `zones` and `ports` are required.
/// @throws std::invalid_argument for malformed YAML, a missing required key,
/// an unknown key, a reserved key or port, a malformed range, or a request
/// check_window_request() rejects.
__attribute__((visibility("default"))) experimental::window_request
window_request_from_yaml_str(const std::string &text);

/// @brief Emit a window request as YAML that window_request_from_yaml_str()
/// reads back to an equal request. A send's rounds are written [begin, end]
/// when there is one range, and as a list of ranges when there are several.
__attribute__((visibility("default"))) std::string
window_request_to_yaml_str(const experimental::window_request &request);

} // namespace cudaq::qec
