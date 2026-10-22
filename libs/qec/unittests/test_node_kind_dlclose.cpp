/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verifies task-graph node registries are safe across a real plugin unload.

#include "decoding_task/decoding_task_node_kind_registry.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/decoding_task_graph.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cudaq::qec {

namespace {

constexpr const char *plugin_kind = "node_kind_dlclose_plugin";
constexpr const char *plugin_alias = "node_kind_dlclose_shared_alias";
constexpr const char *host_kind = "node_kind_dlclose_host";

/// @brief Determine whether a canonical node-kind name is registered.
/// @param name Canonical name to find.
/// @return True when the canonical name appears in registry diagnostics.
bool registered_kind_contains(const std::string &name) {
  const std::vector<std::string> names = registered_task_graph_node_kinds();
  return std::find(names.begin(), names.end(), name) != names.end();
}

/// @brief Register a still-loaded kind that takes over the plugin's alias.
void register_alias_owner() {
  task_graph_node_kind_spec spec;
  spec.name = host_kind;
  spec.aliases = {plugin_alias};
  spec.requires_decoder = false;
  spec.requires_dem = false;
  register_task_graph_node_kind(std::move(spec), {});
}

/// @brief Verify plugin-owned callbacks are erased during a real unload.
TEST(NodeKindDlclose, UnregistersPluginCallbacksBeforeUnload) {
  void *handle = dlopen(NODE_KIND_DLCLOSE_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL);
  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  std::unique_ptr<decoding_task_graph> graph;
  measurement_results shot;
  const task_graph_node_kind_spec *alias_owner = nullptr;

  ASSERT_NE(handle, nullptr) << dlerror();
  ASSERT_TRUE(registered_kind_contains(plugin_kind));
  ASSERT_NE(find_task_graph_node_kind(plugin_kind), nullptr);
  ASSERT_NE(find_node_factory(plugin_kind), nullptr);
  ASSERT_NE(decoding::config::find_decoder_schema(
                task_graph_node_schema_name(plugin_kind)),
            nullptr);

  node.id = "plugin_node";
  node.kind = plugin_kind;
  nodes.push_back(std::move(node));
  graph = decoding_task_graph::from_topology(std::move(nodes));
  shot.bits = {1};
  EXPECT_EQ(graph->run(shot).bits, (std::vector<uint8_t>{1}));
  graph.reset();

  register_alias_owner();
  alias_owner = find_task_graph_node_kind(plugin_alias);
  ASSERT_NE(alias_owner, nullptr);
  ASSERT_EQ(alias_owner->name, host_kind);

  ASSERT_EQ(dlclose(handle), 0) << dlerror();
  EXPECT_EQ(find_task_graph_node_kind(plugin_kind), nullptr);
  EXPECT_EQ(find_node_factory(plugin_kind), nullptr);
  EXPECT_FALSE(registered_kind_contains(plugin_kind));
  EXPECT_EQ(decoding::config::find_decoder_schema(
                task_graph_node_schema_name(plugin_kind)),
            nullptr);

  alias_owner = find_task_graph_node_kind(plugin_alias);
  ASSERT_NE(alias_owner, nullptr);
  EXPECT_EQ(alias_owner->name, host_kind);
} // end - NodeKindDlclose.UnregistersPluginCallbacksBeforeUnload

/// @brief Verify dlclose waits for an active plugin metadata callback.
TEST(NodeKindDlclose, UnloadWaitsForActivePolicyCallback) {
  using void_function = void (*)();
  using bool_function = bool (*)();

  void *handle = dlopen(NODE_KIND_DLCLOSE_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL);
  void_function block_policy = nullptr;
  bool_function policy_entered = nullptr;
  void_function release_policy = nullptr;
  std::future<void> build_future;
  std::promise<void> unload_started;
  std::future<void> unload_started_future = unload_started.get_future();
  std::future<int> unload_future;
  const auto policy_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);

  ASSERT_NE(handle, nullptr) << dlerror();
  block_policy = reinterpret_cast<void_function>(
      dlsym(handle, "node_kind_dlclose_block_policy"));
  policy_entered = reinterpret_cast<bool_function>(
      dlsym(handle, "node_kind_dlclose_policy_entered"));
  release_policy = reinterpret_cast<void_function>(
      dlsym(handle, "node_kind_dlclose_release_policy"));
  ASSERT_NE(block_policy, nullptr);
  ASSERT_NE(policy_entered, nullptr);
  ASSERT_NE(release_policy, nullptr);

  block_policy();
  build_future = std::async(std::launch::async, [] {
    // Keep the image mapped until the graph's plugin-owned node is destroyed.
    // from_topology takes this recursive lock again while it invokes callbacks.
    auto kind_lock = lock_node_kind_registry();
    std::vector<task_graph_node_config> nodes;
    task_graph_node_config node;
    node.id = "plugin_node";
    node.kind = plugin_kind;
    nodes.push_back(std::move(node));
    auto graph = decoding_task_graph::from_topology(std::move(nodes));
    (void)graph;
  });

  while (!policy_entered() &&
         std::chrono::steady_clock::now() < policy_deadline)
    std::this_thread::yield();
  if (!policy_entered())
    release_policy();
  ASSERT_TRUE(policy_entered());

  unload_future = std::async(std::launch::async, [&] {
    unload_started.set_value();
    return dlclose(handle);
  });
  unload_started_future.wait();
  EXPECT_EQ(unload_future.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);

  release_policy();
  EXPECT_NO_THROW(build_future.get());
  EXPECT_EQ(unload_future.get(), 0);
  EXPECT_EQ(find_task_graph_node_kind(plugin_kind), nullptr);
  EXPECT_EQ(find_node_factory(plugin_kind), nullptr);
} // end - NodeKindDlclose.UnloadWaitsForActivePolicyCallback

/// @brief Verify unregister waits until an active factory call has returned.
TEST(NodeKindDlclose, UnregisterWaitsForActiveFactory) {
  std::atomic<bool> factory_entered = false;
  std::atomic<bool> release_factory = false;
  task_graph_node_kind_spec spec;
  node_kind_registration registration;
  std::future<void> build_future;
  std::promise<void> reset_started;
  std::future<void> reset_started_future = reset_started.get_future();
  std::future<void> reset_future;
  const auto factory_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);

  spec.name = "node_kind_concurrent_unregister";
  spec.requires_decoder = false;
  spec.requires_dem = false;
  registration = register_scoped_node_kind(
      std::move(spec), {},
      [&](node_build_context &) -> std::unique_ptr<task_graph_node> {
        factory_entered.store(true, std::memory_order_release);
        while (!release_factory.load(std::memory_order_acquire))
          std::this_thread::yield();
        return nullptr;
      });

  build_future = std::async(std::launch::async, [] {
    std::vector<task_graph_node_config> nodes;
    task_graph_node_config node;
    node.id = "concurrent_node";
    node.kind = "node_kind_concurrent_unregister";
    nodes.push_back(std::move(node));
    (void)decoding_task_graph::from_topology(std::move(nodes));
  });

  while (!factory_entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < factory_deadline)
    std::this_thread::yield();
  if (!factory_entered.load(std::memory_order_acquire))
    release_factory.store(true, std::memory_order_release);
  ASSERT_TRUE(factory_entered.load(std::memory_order_acquire));

  reset_future = std::async(std::launch::async, [&] {
    reset_started.set_value();
    registration.reset();
  });
  reset_started_future.wait();
  EXPECT_EQ(reset_future.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);

  release_factory.store(true, std::memory_order_release);
  EXPECT_THROW(build_future.get(), std::logic_error);
  EXPECT_NO_THROW(reset_future.get());
  EXPECT_EQ(find_task_graph_node_kind("node_kind_concurrent_unregister"),
            nullptr);
  EXPECT_EQ(find_node_factory("node_kind_concurrent_unregister"), nullptr);
} // end - NodeKindDlclose.UnregisterWaitsForActiveFactory

/// @brief Verify an old scoped token cannot erase a replacement registration.
TEST(NodeKindDlclose, StaleTokenPreservesReplacement) {
  task_graph_node_kind_spec first_spec;
  task_graph_node_kind_spec second_spec;
  node_kind_registration first_registration;
  node_kind_registration second_registration;
  const std::string kind_name = "node_kind_scoped_replacement";
  const std::string retired_alias = "node_kind_retired_alias";
  const auto empty_factory =
      [](node_build_context &) -> std::unique_ptr<task_graph_node> {
    return nullptr;
  };

  first_spec.name = kind_name;
  first_spec.aliases = {retired_alias};
  first_spec.requires_decoder = false;
  first_spec.requires_dem = false;
  first_registration =
      register_scoped_node_kind(std::move(first_spec), {}, empty_factory);

  second_spec.name = kind_name;
  second_spec.requires_decoder = false;
  second_spec.requires_dem = false;
  second_registration =
      register_scoped_node_kind(std::move(second_spec), {}, empty_factory);

  first_registration.reset();
  EXPECT_NE(find_task_graph_node_kind(kind_name), nullptr);
  EXPECT_NE(find_node_factory(kind_name), nullptr);
  EXPECT_NE(decoding::config::find_decoder_schema(
                task_graph_node_schema_name(kind_name)),
            nullptr);
  ASSERT_NE(find_task_graph_node_kind(retired_alias), nullptr);
  EXPECT_EQ(find_task_graph_node_kind(retired_alias)->name, kind_name);

  second_registration.reset();
  EXPECT_EQ(find_task_graph_node_kind(kind_name), nullptr);
  EXPECT_EQ(find_node_factory(kind_name), nullptr);
  EXPECT_EQ(find_task_graph_node_kind(retired_alias), nullptr);
  EXPECT_EQ(decoding::config::find_decoder_schema(
                task_graph_node_schema_name(kind_name)),
            nullptr);
} // end - NodeKindDlclose.StaleTokenPreservesReplacement

} // namespace

} // namespace cudaq::qec
