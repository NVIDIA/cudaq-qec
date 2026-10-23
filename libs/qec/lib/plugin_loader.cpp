/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/plugin_loader.h"
#include "decoding_task/decoding_task_node_kind_registry.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/task_graph_plugin.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace cudaq::qec {

namespace {
std::recursive_mutex &plugin_handles_mutex() {
  static auto *mutex = new std::recursive_mutex();
  return *mutex;
}

void *owned_plugin_symbol(void *handle, const fs::path &path,
                          const char *name) {
  dlerror();
  void *symbol = dlsym(handle, name);
  const char *error = dlerror();
  Dl_info information{};
  std::error_code filesystem_error;
  if (error || !symbol || !dladdr(symbol, &information) ||
      !information.dli_fname ||
      !fs::equivalent(path, information.dli_fname, filesystem_error) ||
      filesystem_error)
    throw std::runtime_error(std::string("missing module-owned entry point: ") +
                             name);
  return symbol;
}

bool load_task_graph_plugin(const fs::path &path, PluginHandle &module) {
  // The caller owns module before this transaction is constructed. Failed
  // registrations are therefore erased while the image is still mapped.
  node_kind_plugin_registration registration;
  module.handle.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
  if (!module.handle) {
    CUDA_QEC_WARN("Failed to load task-graph plugin {}: {}", path.string(),
                  dlerror());
    return false;
  }
  try {
    if (registration.initialization_rejected())
      throw std::runtime_error(
          "static-initializer registration is unsupported; rebuild with the "
          "explicit task-graph plugin interface");
    const auto describe =
        reinterpret_cast<cudaq_qec_task_graph_plugin_describe_function>(
            owned_plugin_symbol(module.handle.get(), path,
                                CUDAQ_QEC_TASK_GRAPH_PLUGIN_DESCRIBE_SYMBOL));
    const auto register_plugin =
        reinterpret_cast<cudaq_qec_task_graph_plugin_register_function>(
            owned_plugin_symbol(module.handle.get(), path,
                                CUDAQ_QEC_TASK_GRAPH_PLUGIN_REGISTER_SYMBOL));
    const auto *descriptor = describe();
    const auto expected = make_task_graph_plugin_descriptor("host", "host");
    if (!descriptor || descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->protocol_version != expected.protocol_version ||
        descriptor->framework_abi_version != expected.framework_abi_version ||
        descriptor->glibcxx_cxx11_abi != expected.glibcxx_cxx11_abi ||
        descriptor->cxx_abi_version != expected.cxx_abi_version ||
        descriptor->reserved != 0 ||
        descriptor->node_build_context_size !=
            expected.node_build_context_size ||
        descriptor->node_kind_spec_size != expected.node_kind_spec_size ||
        descriptor->task_graph_node_size != expected.task_graph_node_size ||
        !descriptor->framework_commit || !descriptor->framework_commit[0] ||
        !descriptor->build_id || !descriptor->build_id[0])
      throw std::runtime_error("task-graph plugin ABI/layout mismatch; rebuild "
                               "against this framework");

    registration.begin_registration();
    constexpr std::size_t error_capacity = 512;
    std::array<char, error_capacity> error{};
    const int status = register_plugin(error.data(), error.size());
    error.back() = '\0';
    if (status != 0 || !registration.ready_to_commit())
      throw std::runtime_error(std::string("explicit registration failed: ") +
                               error.data());
    registration.commit();
    CUDA_QEC_INFO("Loaded task-graph plugin {} (ABI {}, build {})",
                  path.string(), descriptor->framework_abi_version,
                  descriptor->build_id);
    return true;
  } catch (const std::exception &error) {
    CUDA_QEC_WARN("Refusing task-graph plugin {}: {}", path.string(),
                  error.what());
  } catch (...) {
    CUDA_QEC_WARN(
        "Refusing task-graph plugin {}: entry point threw an exception",
        path.string());
  }
  return false;
}
} // namespace

static std::map<std::string, PluginHandle> &get_plugin_handles() {
  // Formerly this code was like this:
  // static std::map<std::string, PluginHandle> plugin_handles;
  // return plugin_handles;
  // But that created a double free error when the program exited.
  // This was because the static destructor for the map was called first,
  // and then the cleanup_plugins function was called which tried to access the
  // already destroyed map. This new approach does create a small memory leak,
  // but it prevents the double free error.
  static auto *plugin_handles = new std::map<std::string, PluginHandle>();
  return *plugin_handles; // Dereference pointer to return reference
}

// Function to load plugins from a directory based on their type
void load_plugins(const std::string &plugin_dir, PluginType type) {
  std::lock_guard<std::recursive_mutex> lock(plugin_handles_mutex());
  if (!fs::exists(plugin_dir)) {
    std::cerr << "WARNING: Plugin directory does not exist: " << plugin_dir
              << std::endl;
    return;
  }
  std::vector<fs::path> plugins;
  for (const auto &entry : fs::directory_iterator(plugin_dir))
    if (entry.path().extension() == ".so")
      plugins.push_back(entry.path());
  std::sort(plugins.begin(), plugins.end());
  for (const auto &path : plugins) {
    const auto key = path.string();
    if (get_plugin_handles().count(key) != 0)
      continue;
    if (type == PluginType::TASK_GRAPH_NODE) {
      // Allocate the handle entry before invoking plugin code. On refusal the
      // registration transaction rolls back first; erase then calls dlclose.
      auto [entry, inserted] = get_plugin_handles().emplace(
          key, PluginHandle{std::unique_ptr<void, PluginDeleter>{}, type});
      if (!load_task_graph_plugin(path, entry->second))
        get_plugin_handles().erase(entry);
      continue;
    }
    void *raw_handle = dlopen(path.c_str(), RTLD_NOW);
    if (raw_handle) {
      get_plugin_handles().emplace(
          key, PluginHandle{std::unique_ptr<void, PluginDeleter>(
                                raw_handle, PluginDeleter()),
                            type});
      CUDA_QEC_INFO("Successfully loaded plugin: {}", key);
    } else {
      CUDA_QEC_INFO("Failed to load plugin {} Error: {}. Continuing anyway.",
                    key, dlerror());
    }
  }
}

// Function to clean up the plugin handles
void cleanup_plugins(PluginType type) {
  std::lock_guard<std::recursive_mutex> lock(plugin_handles_mutex());
  if (type == PluginType::TASK_GRAPH_NODE)
    return;
  auto &handles = get_plugin_handles();
  auto it = handles.begin();
  while (it != handles.end()) {
    if (it->second.type == type) {
      it = handles.erase(it); // dlclose is handled by the custom deleter
    } else {
      ++it;
    }
  }
}

} // namespace cudaq::qec
