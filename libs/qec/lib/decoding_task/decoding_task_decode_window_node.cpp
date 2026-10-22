/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The `decode_window` node kind: decodes whichever window its `input` view
// carries, applies the deltas and residuals on its `receive_<i>` inputs, and
// publishes its frame update and one output per send. The kind is registered
// so configurations that use it parse and validate; the node itself is not
// available yet.

#include "decoding_task_node_kind_registry.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {
namespace {

/// Accepted values of verify_residual: off (false), fail the node on a
/// solution that does not satisfy its detectors (fail, or true), or complete
/// the shot and report it open (report).
bool is_verify_residual(const std::string &value) {
  return value == "false" || value == "true" || value == "fail" ||
         value == "report";
}

/// num_receives creates the ports receive_0 ... receive_{n-1};
/// max_in_flight bounds the windows decoding at once.
decoding::config::decoder_schema decode_window_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"num_receives", param_kind::uint64});
  schema.params.push_back({"max_in_flight", param_kind::uint64});
  schema.params.push_back({"verify_residual", param_kind::string});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    if (p.contains("max_in_flight") && p.get<std::size_t>("max_in_flight") == 0)
      throw std::runtime_error("decode_window: max_in_flight must be at "
                               "least 1");
    if (p.contains("verify_residual") &&
        !is_verify_residual(p.get<std::string>("verify_residual")))
      throw std::runtime_error(
          "decode_window: verify_residual must be false, fail (or true), or "
          "report; got '" +
          p.get<std::string>("verify_residual") + "'");
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
