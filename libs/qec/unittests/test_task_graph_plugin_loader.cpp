/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "decoding_task/decoding_task_node_kind_registry.h"
#include "cudaq/qec/plugin_loader.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

std::map<std::string, unsigned> &registration_calls() {
  static std::map<std::string, unsigned> calls;
  return calls;
}

} // namespace

// The fixtures call this host-owned observer only from their register gateway.
// A mismatch rejected before that gateway must leave its counter at zero.
extern "C" __attribute__((visibility("default"))) void
task_graph_plugin_test_register_called(const char *name) {
  ++registration_calls()[name];
}

namespace cudaq::qec {
namespace {

namespace fs = std::filesystem;

constexpr const char *host_kind = "task_graph_plugin_loader_host";
constexpr const char *host_alias = "task_graph_plugin_loader_host_alias";

std::string kind_name(const std::string &fixture) {
  return "task_graph_plugin_loader_" + fixture;
}

fs::path fixture_directory(const std::string &fixture) {
  return fs::path(TASK_GRAPH_PLUGIN_FIXTURE_ROOT) / fixture;
}

fs::path module_path(const std::string &fixture) {
  return fixture_directory(fixture) / "libtask_graph_plugin_fixture.so";
}

void expect_absent(const std::string &fixture) {
  const auto kind = kind_name(fixture);
  EXPECT_EQ(find_task_graph_node_kind(kind), nullptr);
  EXPECT_EQ(find_node_factory(kind), nullptr);
  EXPECT_EQ(
      decoding::config::find_decoder_schema(task_graph_node_schema_name(kind)),
      nullptr);
}

node_kind_registration register_host_kind() {
  task_graph_node_kind_spec spec;
  spec.name = host_kind;
  spec.aliases = {host_alias};
  spec.requires_decoder = false;
  spec.requires_dem = false;
  return register_scoped_node_kind(
      std::move(spec), {},
      [](node_build_context &) -> std::unique_ptr<task_graph_node> {
        return nullptr;
      });
}

class RejectedTaskGraphPlugin : public ::testing::TestWithParam<const char *> {
};

TEST_P(RejectedTaskGraphPlugin, NeverCallsRegisterOrLeavesKinds) {
  const std::string fixture = GetParam();
  ASSERT_TRUE(fs::is_regular_file(module_path(fixture)));
  expect_absent(fixture);
  EXPECT_EQ(registration_calls()[fixture], 0u);

  EXPECT_NO_THROW(load_plugins(fixture_directory(fixture).string(),
                               PluginType::TASK_GRAPH_NODE));

  EXPECT_EQ(registration_calls()[fixture], 0u);
  expect_absent(fixture);
  // Fixtures are non-GNU-unique and are not pinned after a refused load.
  void *handle = dlopen(module_path(fixture).c_str(), RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(handle, nullptr);
  if (handle)
    dlclose(handle);
}

INSTANTIATE_TEST_SUITE_P(
    Gateway, RejectedTaskGraphPlugin,
    ::testing::Values("framework_mismatch", "protocol_mismatch",
                      "descriptor_size_mismatch", "context_size_mismatch",
                      "kind_size_mismatch", "node_size_mismatch",
                      "compiler_abi_mismatch", "string_abi_mismatch",
                      "missing_describe", "missing_register",
                      "static_registration", "borrowed_entrypoint"));

TEST(TaskGraphPluginLoader, RollsBackPartialRegistrationBeforeUnloading) {
  const std::string fixture = "partial_failure";
  ASSERT_TRUE(fs::is_regular_file(module_path(fixture)));
  expect_absent(fixture);

  EXPECT_NO_THROW(load_plugins(fixture_directory(fixture).string(),
                               PluginType::TASK_GRAPH_NODE));

  EXPECT_EQ(registration_calls()[fixture], 1u);
  expect_absent(fixture);
  void *handle = dlopen(module_path(fixture).c_str(), RTLD_NOW | RTLD_NOLOAD);
  EXPECT_EQ(handle, nullptr);
  if (handle)
    dlclose(handle);
}

class CollidingTaskGraphPlugin : public ::testing::TestWithParam<const char *> {
};

TEST_P(CollidingTaskGraphPlugin, PreservesHostKindAndAliasAndRollsBack) {
  const std::string fixture = GetParam();
  auto host_registration = register_host_kind();
  const auto *host_factory = find_node_factory(host_kind);
  ASSERT_NE(host_factory, nullptr);
  ASSERT_TRUE(fs::is_regular_file(module_path(fixture)));

  EXPECT_NO_THROW(load_plugins(fixture_directory(fixture).string(),
                               PluginType::TASK_GRAPH_NODE));

  EXPECT_EQ(registration_calls()[fixture], 1u);
  expect_absent(fixture);
  EXPECT_EQ(find_task_graph_node_kind(kind_name(fixture) + "_collision"),
            nullptr);
  ASSERT_NE(find_task_graph_node_kind(host_kind), nullptr);
  ASSERT_NE(find_task_graph_node_kind(host_alias), nullptr);
  EXPECT_EQ(find_task_graph_node_kind(host_alias)->name, host_kind);
  EXPECT_EQ(find_node_factory(host_kind), host_factory);
  EXPECT_EQ(find_node_factory(host_alias), host_factory);
  EXPECT_NE(decoding::config::find_decoder_schema(
                task_graph_node_schema_name(host_kind)),
            nullptr);
}

INSTANTIATE_TEST_SUITE_P(Gateway, CollidingTaskGraphPlugin,
                         ::testing::Values("collision_kind",
                                           "collision_alias"));

TEST(TaskGraphPluginLoader, CompatibleModuleStaysUsableAfterCleanup) {
  const std::string fixture = "compatible";
  ASSERT_TRUE(fs::is_regular_file(module_path(fixture)));
  EXPECT_EQ(cudaq_qec_task_graph_host_abi_version(),
            CUDAQ_QEC_TASK_GRAPH_FRAMEWORK_ABI_VERSION);

  load_plugins(fixture_directory(fixture).string(),
               PluginType::TASK_GRAPH_NODE);
  ASSERT_EQ(registration_calls()[fixture], 1u);
  ASSERT_NE(find_task_graph_node_kind(kind_name(fixture)), nullptr);
  ASSERT_NE(find_node_factory(kind_name(fixture)), nullptr);
  cleanup_plugins(PluginType::TASK_GRAPH_NODE);

  void *handle = dlopen(module_path(fixture).c_str(), RTLD_NOW | RTLD_NOLOAD);
  ASSERT_NE(handle, nullptr) << dlerror();
  EXPECT_EQ(dlclose(handle), 0);

  std::vector<task_graph_node_config> nodes;
  task_graph_node_config node;
  node.id = "loaded_plugin";
  node.kind = kind_name(fixture);
  nodes.push_back(std::move(node));
  auto graph = decoding_task_graph::from_topology(std::move(nodes));
  for (uint8_t bit : {uint8_t{0}, uint8_t{1}}) {
    measurement_results shot;
    shot.bits = {bit};
    EXPECT_EQ(graph->run(shot).bits, (std::vector<uint8_t>{bit}));
  }
  load_plugins(fixture_directory(fixture).string(),
               PluginType::TASK_GRAPH_NODE);
  EXPECT_EQ(registration_calls()[fixture], 1u);
}

TEST(TaskGraphPluginLoader, LegacyDecoderLoadDoesNotRequireTaskGraphGateway) {
  const std::string fixture = "legacy_decoder";
  ASSERT_TRUE(fs::is_regular_file(module_path(fixture)));
  expect_absent(fixture);

  load_plugins(fixture_directory(fixture).string(), PluginType::DECODER);

  ASSERT_NE(find_task_graph_node_kind(kind_name(fixture)), nullptr);
  EXPECT_EQ(registration_calls()[fixture], 0u);
  cleanup_plugins(PluginType::DECODER);
  expect_absent(fixture);
}

} // namespace
} // namespace cudaq::qec
