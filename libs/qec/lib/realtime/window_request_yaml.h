/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Internal header: YAML for one window request, the shape of an entry of an
// ingest node's `windows` map. Never included from a public header.

#pragma once

#include "../decoding_task/decoding_task_decoding_window.h"
#include <string>

namespace cudaq::qec {

/// @brief Parse one window request.
///
/// Every round range is written [begin, end] or {rounds: [begin, end]}; the
/// map form reserves `occurrences` and `block`. `ports` names only `input`
/// and `output`. The window keys `responsibility` and `soundness_class` are
/// reserved. A send's `rounds` is one range.
/// @throws std::invalid_argument naming the operation for malformed YAML,
/// an unknown key, a reserved key or port, or a malformed range.
__attribute__((visibility("default"))) window_request
window_request_from_yaml_str(const std::string &text);

/// @brief Emit a window request as YAML that window_request_from_yaml_str()
/// reads back to an equal request.
/// @throws std::invalid_argument if a send has more than one round range,
/// which the YAML form cannot write.
__attribute__((visibility("default"))) std::string
window_request_to_yaml_str(const window_request &request);

} // namespace cudaq::qec
