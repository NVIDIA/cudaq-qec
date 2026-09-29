/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlfcn.h>
#include <map>
#include <memory>
#include <string>

namespace cudaq::qec {

/// @brief Enum to define different types of plugins
enum class PluginType {
  DECODER, // Decoder plugins
  CODE     // QEC codes plugins
           // Add other plugin types here as needed
};

struct PluginDeleter // deleter
{
  void operator()(void *h) const {
    if (h)
      dlclose(h);
  };
};

/// @brief A struct to store plugin handle with its type
struct PluginHandle {
  // Pointer to the shared library handle. This is the result of dlopen()
  // function.
  std::unique_ptr<void, PluginDeleter> handle;
  // Type of the plugin (e.g., decoder, code, etc)
  PluginType type;
};

/// @brief Function to load plugins from a directory based on type
/// @param plugin_dir The directory where the plugins are located
/// @param type The type of plugins to load. Only plugins of this type will be
/// loaded.
void load_plugins(const std::string &plugin_dir, PluginType type);

/// @brief Function to clean up loaded plugins of a specific type
/// @param type The type of plugins to clean up. Only plugins of this type will
/// be cleaned up.
void cleanup_plugins(PluginType type);

} // namespace cudaq::qec
