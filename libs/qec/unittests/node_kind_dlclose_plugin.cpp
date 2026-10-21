/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test plugin that registers a task-graph node kind with unload cleanup.

#include "decoding_task/decoding_task_node_kind_registry.h"

#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace cudaq::qec {

namespace {

std::atomic<bool> block_policy = false;
std::atomic<bool> policy_entered = false;
std::atomic<bool> release_policy = false;

/// @brief Exercise a plugin-owned metadata callback during graph construction.
/// @return True; the test kind does not build an inner decoder.
bool inner_decoder_takes_observables(const cudaqx::heterogeneous_map &) {
  if (block_policy.load(std::memory_order_acquire)) {
    policy_entered.store(true, std::memory_order_release);
    while (!release_policy.load(std::memory_order_acquire))
      std::this_thread::yield();
  }
  return true;
}

class dlclose_test_node final : public task_graph_node {
public:
  /// @brief Forward the first measurement bit as a logical outcome.
  /// @param inputs Values delivered to the node's input slots.
  /// @param outputs Storage for values emitted from the output slots.
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    const auto &measurements = inputs[0].get<measurement_results>();
    const uint8_t bit =
        measurements.bits.empty() ? 0 : measurements.bits[0] & 1u;
    outputs[0] = logical_outcome{{bit}, true};
  }

  /// @brief Declare the node's single input slot.
  /// @return Input slot names in index order.
  std::vector<std::string> inputs() const override { return {"input"}; }

  /// @brief Declare the node's single output slot.
  /// @return Output slot names in index order.
  std::vector<std::string> outputs() const override { return {"output"}; }

  /// @brief Describe the value emitted by the output slot.
  /// @param slot Output slot index.
  /// @return Logical-outcome type for slot zero, otherwise none.
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return logical_outcome_type;
    return std::nullopt;
  }

  /// @brief Report whether an input slot accepts the supplied value type.
  /// @param slot Input slot index.
  /// @param type Candidate edge value type.
  /// @return True for measurements delivered to slot zero.
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == measurement_results_type;
  }

  /// @brief Give the statically known logical-output width.
  /// @param slot Output slot index.
  /// @return One bit for slot zero, otherwise no known size.
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    return slot == 0 ? std::optional<std::size_t>{1} : std::nullopt;
  }
};

/// @brief Register the test node and return its unload cleanup token.
/// @return Registration token held until the plugin is unloaded.
node_kind_registration register_dlclose_test_node() {
  task_graph_node_kind_spec spec;
  decoding::config::decoder_schema schema;
  spec.name = "node_kind_dlclose_plugin";
  spec.aliases = {"node_kind_dlclose_shared_alias"};
  spec.requires_decoder = false;
  spec.requires_dem = false;
  spec.inner_decoder_observable_policy = inner_decoder_takes_observables;
  schema.validate = [](const cudaqx::heterogeneous_map &) {};
  return register_scoped_node_kind(
      std::move(spec), std::move(schema),
      [](node_build_context &) -> std::unique_ptr<task_graph_node> {
        return std::make_unique<dlclose_test_node>();
      });
}

const node_kind_registration registration = register_dlclose_test_node();

} // namespace

/// @brief Configure the policy callback to block until explicitly released.
extern "C" __attribute__((visibility("default"))) void
node_kind_dlclose_block_policy() {
  policy_entered.store(false, std::memory_order_release);
  release_policy.store(false, std::memory_order_release);
  block_policy.store(true, std::memory_order_release);
}

/// @brief Report whether graph construction entered the policy callback.
/// @return True after the blocked callback begins.
extern "C" __attribute__((visibility("default"))) bool
node_kind_dlclose_policy_entered() {
  return policy_entered.load(std::memory_order_acquire);
}

/// @brief Allow the blocked policy callback to return.
extern "C" __attribute__((visibility("default"))) void
node_kind_dlclose_release_policy() {
  release_policy.store(true, std::memory_order_release);
}

} // namespace cudaq::qec
