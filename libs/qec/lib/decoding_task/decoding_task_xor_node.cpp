/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decoding_task_xor_node.h"
#include "decoding_task_graph_impl.h"
#include "decoding_task_node_kind_registry.h"

#include <stdexcept>

namespace cudaq::qec {

// The ports combine() and existing configurations name.
xor_node::xor_node() : inputs_{"a", "b"} {}

// Ports in_0 ... in_{N-1}, so N need not be known to name them.
xor_node::xor_node(std::size_t num_inputs) {
  if (num_inputs < 2 || num_inputs > max_inputs)
    throw std::invalid_argument("xor: num_inputs must be from 2 to " +
                                std::to_string(max_inputs) + ", not " +
                                std::to_string(num_inputs));
  for (std::size_t i = 0; i < num_inputs; ++i)
    inputs_.push_back("in_" + std::to_string(i));
}

// Out-of-line destructor anchors the vtable to this translation unit.
xor_node::~xor_node() = default;

namespace {

/// The kind's one parameter: how many inputs, when not the two of a and b.
/// Its bounds are declared with the kind.
decoding::config::decoder_schema xor_params() {
  decoding::config::decoder_schema schema;
  schema.params.push_back({"num_inputs", decoding::config::param_kind::uint64});
  return schema;
}

/// The `xor` node kind, which is what makes combine()'s topology expressible
/// from YAML. It wraps no decoder and needs no DEM, so it declares neither.
/// Without num_inputs its ports stay a and b, so existing configurations are
/// unchanged.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "xor";
  spec.aliases = {"XOR", "Xor"};
  spec.requires_decoder = false;
  spec.requires_dem = false;
  spec.parameter_minimums = {{"num_inputs", 2}};
  spec.parameter_maximums = {{"num_inputs", xor_node::max_inputs}};
  register_node_kind(std::move(spec), xor_params(),
                     [](node_build_context &ctx) {
                       const auto *p = ctx.node_params;
                       if (p && p->contains("num_inputs"))
                         return std::unique_ptr<task_graph_node>(
                             new xor_node(p->get<std::size_t>("num_inputs")));
                       return std::unique_ptr<task_graph_node>(new xor_node());
                     });
  return true;
}();

} // namespace

// ---------------------------------------------------------------------------
// decoding_task_graph::combine (two window_nodes XORed together)
// ---------------------------------------------------------------------------

std::unique_ptr<decoding_task_graph>
decoding_task_graph::combine(std::unique_ptr<decoding_task_graph> left,
                             std::unique_ptr<decoding_task_graph> right,
                             task_graph_options options) {
  if (!left || !right)
    throw std::invalid_argument("combine: graphs must not be null");
  if (left->impl_->nodes.size() != 1 || right->impl_->nodes.size() != 1)
    throw std::invalid_argument(
        "combine: each input graph must have exactly one node");

  if (left->get_num_observables() != right->get_num_observables())
    throw std::invalid_argument(
        "combine: graphs must have the same number of observables to XOR");

  auto i = std::make_unique<impl>();
  i->d = left->impl_->d;
  // o_matrix() is not meaningful on a combined graph; leave i->o unset.

  i->nodes.push_back(std::move(left->impl_->nodes[0]));
  i->nodes.push_back(std::move(right->impl_->nodes[0]));
  i->nodes.push_back(std::make_unique<xor_node>());

  i->node_ids = {"left", "right", "xor"};
  i->edges = {
      {"measurements", "output", "left", "input"},
      {"measurements", "output", "right", "input"},
      {"left", "output", "xor", "a"},
      {"right", "output", "xor", "b"},
      {"xor", "output", "output", "input"},
  };
  return assemble(std::move(i), options);
}

} // namespace cudaq::qec
