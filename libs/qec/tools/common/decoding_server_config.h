/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdlib>
#include <filesystem>

namespace cudaq::qec::tools {

inline constexpr auto decoder_config_environment = "CUDAQ_QEC_DECODER_CONFIG";
inline constexpr auto default_server_config_filename = "server.yaml";

/// Shared by the configuration generator and standalone server. Deployments
/// may reuse the service's existing environment setting; otherwise the file
/// lives in the working directory, including the release image's WORKDIR.
inline std::filesystem::path default_server_config_path() {
  if (const auto *path = std::getenv(decoder_config_environment); path && *path)
    return std::filesystem::absolute(path);
  return std::filesystem::current_path() / default_server_config_filename;
}

} // namespace cudaq::qec::tools
