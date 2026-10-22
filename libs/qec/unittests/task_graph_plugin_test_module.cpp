/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Each fixture is built into its own directory, outside automatic plugin paths.
#include "decoding_task/decoding_task_node_kind_registry.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>

extern "C" void task_graph_plugin_test_register_called(const char *name);

#if defined(TASK_GRAPH_PLUGIN_TEST_BORROWED_ENTRYPOINT)

extern "C" void task_graph_plugin_test_dependency_anchor();

namespace {
// Retain the dependency even with a linker configured with --as-needed. Its
// entry points must not satisfy the wrapper module's own gateway contract.
const bool dependency_loaded = [] {
  task_graph_plugin_test_dependency_anchor();
  return true;
}();
} // namespace

#else

namespace cudaq::qec {
namespace {

constexpr const char *fixture_name = TASK_GRAPH_PLUGIN_TEST_CASE;
constexpr const char *fixture_commit =
    "0123456789abcdef0123456789abcdef01234567";
constexpr const char *host_kind = "task_graph_plugin_loader_host";
constexpr const char *host_alias = "task_graph_plugin_loader_host_alias";
constexpr int registration_failure = 17;

std::string registered_name() {
  return std::string("task_graph_plugin_loader_") + fixture_name;
}

class fixture_node final : public task_graph_node {
public:
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    const auto &bits = inputs.at(0).get<measurement_results>().bits;
    outputs.at(0) = logical_outcome{
        {static_cast<uint8_t>(bits.empty() ? 0 : bits[0] & 1u)}, true};
  }

  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }

  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    return slot == 0 ? std::optional{logical_outcome_type} : std::nullopt;
  }

  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == measurement_results_type;
  }

  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    return slot == 0 ? std::optional<std::size_t>{1} : std::nullopt;
  }
};

task_graph_node_kind_spec make_spec(std::string name) {
  task_graph_node_kind_spec spec;
  spec.name = std::move(name);
  spec.requires_decoder = false;
  spec.requires_dem = false;
  return spec;
}

void register_fixture(std::string name) {
  register_node_kind(make_spec(std::move(name)), {}, [](node_build_context &) {
    return std::make_unique<fixture_node>();
  });
}

#if defined(TASK_GRAPH_PLUGIN_TEST_STATIC_REGISTRATION)
const bool constructor_registration = [] {
  // Older constructors can catch a registration error. The loader must still
  // remember the attempted mutation and reject the module before its gateway.
  try {
    register_fixture(registered_name());
  } catch (const std::exception &) {
  }
  return true;
}();
#elif defined(TASK_GRAPH_PLUGIN_TEST_LEGACY_DECODER)
const node_kind_registration constructor_registration =
    register_scoped_node_kind(
        make_spec(registered_name()), {},
        [](node_build_context &) { return std::make_unique<fixture_node>(); });
#endif

cudaq_qec_task_graph_plugin_descriptor make_descriptor() {
  auto descriptor =
      make_task_graph_plugin_descriptor(fixture_commit, fixture_name);
#if defined(TASK_GRAPH_PLUGIN_TEST_FRAMEWORK_MISMATCH)
  ++descriptor.framework_abi_version;
#elif defined(TASK_GRAPH_PLUGIN_TEST_PROTOCOL_MISMATCH)
  ++descriptor.protocol_version;
#elif defined(TASK_GRAPH_PLUGIN_TEST_DESCRIPTOR_SIZE_MISMATCH)
  --descriptor.struct_size;
#elif defined(TASK_GRAPH_PLUGIN_TEST_CONTEXT_SIZE_MISMATCH)
  ++descriptor.node_build_context_size;
#elif defined(TASK_GRAPH_PLUGIN_TEST_KIND_SIZE_MISMATCH)
  ++descriptor.node_kind_spec_size;
#elif defined(TASK_GRAPH_PLUGIN_TEST_NODE_SIZE_MISMATCH)
  ++descriptor.task_graph_node_size;
#elif defined(TASK_GRAPH_PLUGIN_TEST_COMPILER_ABI_MISMATCH)
  ++descriptor.cxx_abi_version;
#elif defined(TASK_GRAPH_PLUGIN_TEST_STRING_ABI_MISMATCH)
  descriptor.glibcxx_cxx11_abi ^= 1u;
#endif
  return descriptor;
}

} // namespace
} // namespace cudaq::qec

#if !defined(TASK_GRAPH_PLUGIN_TEST_MISSING_DESCRIBE) &&                       \
    !defined(TASK_GRAPH_PLUGIN_TEST_LEGACY_DECODER)
extern "C" __attribute__((visibility("default")))
const cudaq_qec_task_graph_plugin_descriptor *
cudaq_qec_task_graph_plugin_describe() {
  static const auto descriptor = cudaq::qec::make_descriptor();
  return &descriptor;
}
#endif

#if !defined(TASK_GRAPH_PLUGIN_TEST_MISSING_REGISTER) &&                       \
    !defined(TASK_GRAPH_PLUGIN_TEST_LEGACY_DECODER)
extern "C" __attribute__((visibility("default"))) int
cudaq_qec_task_graph_plugin_register(char *error, size_t capacity) {
  using namespace cudaq::qec;
  task_graph_plugin_test_register_called(fixture_name);
  try {
    register_fixture(registered_name());
#if defined(TASK_GRAPH_PLUGIN_TEST_PARTIAL_FAILURE)
    if (capacity)
      std::snprintf(error, capacity, "Deliberate partial registration failure");
    return registration_failure;
#elif defined(TASK_GRAPH_PLUGIN_TEST_COLLISION_KIND)
    register_fixture(host_kind);
#elif defined(TASK_GRAPH_PLUGIN_TEST_COLLISION_ALIAS)
    auto spec = make_spec(registered_name() + "_collision");
    spec.aliases = {host_alias};
    register_node_kind(std::move(spec), {}, [](node_build_context &) {
      return std::make_unique<fixture_node>();
    });
#endif
    return 0;
  } catch (const std::exception &exception) {
    if (capacity)
      std::snprintf(error, capacity, "%s", exception.what());
    return registration_failure;
  } catch (...) {
    if (capacity)
      std::snprintf(error, capacity, "Unknown fixture registration failure");
    return registration_failure;
  }
}
#endif

extern "C" __attribute__((visibility("default"))) void
task_graph_plugin_test_dependency_anchor() {}

#endif
