/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `parallel_window` pattern: written as one node, wired measurements ->
// <id> -> output, it stands for the parallel-window graph of an ingest node,
// two decode_window layers A and B, and their xor. It is registered so
// configurations that use it parse and validate; expanding it is not
// available yet.

#include "decoding_task_node_kind_registry.h"
#include "decoding_task_window_kind_params.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {
namespace {

/// Throws std::runtime_error unless string parameter @p key, when set, is
/// one of @p first or @p second.
void check_choice(const cudaqx::heterogeneous_map &p, const std::string &key,
                  const std::string &first, const std::string &second) {
  if (!p.contains(key))
    return;
  const auto value = p.get<std::string>(key);
  if (value != first && value != second)
    throw std::runtime_error("parallel_window: " + key + " must be " + first +
                             " or " + second + "; got '" + value + "'");
}

/// A's commit and buffer zones are required; B's gap defaults to
/// commit_rounds + 2 buffer_rounds. The rest pass through to the layers.
decoding::config::decoder_schema parallel_window_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"commit_rounds", param_kind::uint64, true});
  schema.params.push_back({"buffer_rounds", param_kind::uint64, true});
  schema.params.push_back({"gap_rounds", param_kind::uint64});
  schema.params.push_back({"boundary_prior", param_kind::f64});
  schema.params.push_back({"output", param_kind::string});
  schema.params.push_back({"input_boundary", param_kind::string});
  schema.params.push_back({"merge_identical", param_kind::boolean});
  schema.params.push_back({"conflicting_parallel", param_kind::string});
  schema.params.push_back({"verify_residual", param_kind::string});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    check_choice(p, "output", "corrected_logical", "corrections");
    check_choice(p, "input_boundary", "columns", "truncate");
    check_choice(p, "conflicting_parallel", "reject", "most_likely");
    if (p.contains("boundary_prior")) {
      const auto prior = p.get<double>("boundary_prior");
      if (!(prior > 0.0 && prior < 1.0))
        throw std::runtime_error("parallel_window: boundary_prior must be "
                                 "strictly between 0 and 1");
      if (p.contains("input_boundary") &&
          p.get<std::string>("input_boundary") == "truncate")
        throw std::runtime_error("parallel_window: boundary_prior is for "
                                 "input_boundary columns; truncate keeps each "
                                 "crossing fault's own prior");
    }
    check_verify_residual(p, "parallel_window");
  };
  return schema;
}

// It reads the template and names the layers' inner decoder.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "parallel_window";
  spec.requires_decoder = true;
  spec.requires_dem = true;
  spec.parameter_minimums = {
      {"commit_rounds", 1}, {"buffer_rounds", 1}, {"gap_rounds", 1}};
  register_node_kind(
      std::move(spec), parallel_window_params(),
      [](node_build_context &ctx) -> std::unique_ptr<task_graph_node> {
        throw std::invalid_argument(
            "task graph node '" + ctx.id +
            "': the 'parallel_window' pattern is not available yet");
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
