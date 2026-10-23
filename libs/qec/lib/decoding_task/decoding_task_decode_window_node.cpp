/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `decode_window` node kind: decodes whichever window its `input` view
// carries, applies the deltas and residuals on its `receive_<i>` inputs, and
// publishes its frame update and one output per send. The kind is registered
// so configurations that use it parse and validate; the node itself is not
// available yet.

#include "decoding_task_node_kind_registry.h"
#include "decoding_task_window_kind_params.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {
namespace {

/// num_receives creates the ports receive_0 ... receive_{n-1};
/// max_in_flight bounds the windows decoding at once.
decoding::config::decoder_schema decode_window_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"num_receives", param_kind::uint64});
  schema.params.push_back({"max_in_flight", param_kind::uint64});
  schema.params.push_back({"verify_residual", param_kind::string});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    check_max_in_flight(p, "decode_window");
    check_verify_residual(p, "decode_window");
  };
  return schema;
}

// Its window comes from its input, so it reads no DEM; it wraps the inner
// decoder its `inner:` block names.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "decode_window";
  spec.requires_decoder = true;
  spec.requires_dem = false;
  register_node_kind(
      std::move(spec), decode_window_params(),
      [](node_build_context &ctx) -> std::unique_ptr<task_graph_node> {
        throw std::invalid_argument(
            "task graph node '" + ctx.id +
            "': the 'decode_window' kind is not available yet");
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
