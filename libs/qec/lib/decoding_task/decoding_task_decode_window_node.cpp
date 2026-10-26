/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `decode_window` node and its kind: decodes whichever window its
// `input` carries, applies the deltas and residuals on its `receive_<i>`
// inputs, and publishes its frame update and one output per send. Its kind
// builds it from the store, template and layer a windowed topology
// resolves, with the inner decoder its `inner:` block names.

#include "decoding_task_decode_window_node.h"
#include "decoding_task_node_kind_registry.h"
#include "decoding_task_window_kind_params.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// One slot: an executor runs copies of the node for fires at once. The
// frame update has a bit per observable of the template.
decode_window_node::decode_window_node(
    const data_artifact_store &store,
    std::shared_ptr<const experiment_template> context, window_request request,
    const std::string &decoder_name, const cudaqx::heterogeneous_map &params,
    std::size_t num_receives, verify_residual_mode verify)
    : core_(store, context, request, decoder_name, params, 1, verify),
      receives_(num_receives), received_(num_receives, nullptr),
      frame_(logical_outcome{}), corrections_(context->num_observables(), 0) {
  inputs_.push_back("input");
  for (std::size_t i = 0; i < num_receives; ++i)
    inputs_.push_back("receive_" + std::to_string(i));
  outputs_.push_back("output");
  for (const auto &send : request.sends) {
    outputs_.push_back(send.id);
    sends_.emplace_back(window_send_payload{});
  }
  frame_.get<logical_outcome>().bits = corrections_;
} // end - decode_window_node()

// Inputs are checked before the core fires, which checks the window and
// every received row before it touches its slot. An executor and ingest
// may number instances apart from windows, but by one offset a shot.
// Publications are copies, so an executor may keep them after the next
// fire; corrections accumulate every fire's frame update by XOR.
bool decode_window_node::fire(std::int64_t k,
                              const std::vector<const edge_value *> &inputs) {
  published_ = false;
  if (inputs.size() != inputs_.size())
    throw std::invalid_argument("decode_window: instance " + std::to_string(k) +
                                " has " + std::to_string(inputs.size()) +
                                " inputs, not the node's " +
                                std::to_string(inputs_.size()));
  if (!inputs[0])
    throw std::invalid_argument("decode_window: instance " + std::to_string(k) +
                                " has no window on its input");
  const auto &window = inputs[0]->get<window_rows>();
  for (std::size_t i = 0; i < receives_.size(); ++i) {
    received_[i] = nullptr;
    if (!inputs[1 + i])
      continue;
    receives_[i] = receive_of(inputs[1 + i]->get<window_send_payload>());
    received_[i] = &receives_[i];
  }
  if (offset_ && window.k - k != *offset_)
    throw std::invalid_argument(
        "decode_window: instance " + std::to_string(k) + " carries window " +
        std::to_string(window.k) + ", but this shot's instances are " +
        std::to_string(*offset_) + " below their windows");
  const auto &fired = core_.fire(0, window, received_);
  auto &frame = frame_.get<logical_outcome>();
  frame.bits = fired.solution->outputs.frame;
  frame.converged = fired.solution->converged;
  for (std::size_t i = 0; i < corrections_.size(); ++i)
    corrections_[i] ^= frame.bits[i] & 1;
  offset_ = window.k - k;
  for (std::size_t s = 0; s < sends_.size(); ++s)
    copy_send(fired.sends[s], sends_[s].get<window_send_payload>());
  commit_ = {static_cast<std::uint64_t>(fired.commit.begin),
             static_cast<std::uint64_t>(fired.commit.end)};
  index_ = k;
  published_ = true;
  return true;
} // end - fire()

std::optional<std::int64_t> decode_window_node::published_index() const {
  return published_ ? std::optional<std::int64_t>(index_) : std::nullopt;
}

// Windows are decoded as their instances' inputs join, never per shot.
void decode_window_node::execute_indexed(const std::vector<edge_value> &,
                                         std::vector<edge_value> &) const {
  throw std::logic_error("decode_window: the node joins its inputs by index; "
                         "it has no one-shot path");
}

// The frame update, then a send per port.
std::optional<edge_value_type>
decode_window_node::output_type(std::size_t slot) const {
  if (slot == 0)
    return logical_outcome_type;
  if (slot < outputs_.size())
    return edge_value_type::of<window_send_payload>();
  return std::nullopt;
}

// A window on `input`, a send on each receive.
bool decode_window_node::accepts_input_type(std::size_t slot,
                                            edge_value_type type) const {
  if (slot == 0)
    return type == edge_value_type::of<window_rows>();
  return slot < inputs_.size() &&
         type == edge_value_type::of<window_send_payload>();
}

bool decode_window_node::enqueue(const uint8_t *, std::size_t, round_type) {
  throw std::logic_error(
      "decode_window: the node takes windows, not raw measurements");
}

// Nothing stands once a fire has failed, or before the first.
const edge_value *decode_window_node::streaming_output(std::size_t slot) const {
  if (!published_ || slot >= outputs_.size())
    return nullptr;
  return slot == 0 ? &frame_ : &sends_[slot - 1];
}

std::optional<streaming_task_graph_node::committed_coverage>
decode_window_node::streaming_coverage() const {
  return published_ ? std::optional<committed_coverage>(commit_) : std::nullopt;
}

void decode_window_node::clear_corrections() {
  std::fill(corrections_.begin(), corrections_.end(), 0);
}

// A new shot: no slot holds a fire, nothing is published, and instances
// may be numbered afresh.
void decode_window_node::reset_streaming_state() {
  core_.reset();
  published_ = false;
  offset_.reset();
  clear_corrections();
}

namespace {

/// num_receives creates the ports receive_0 ... receive_{n-1}. The windows
/// decoding at once are bounded by the node's max_in_flight.
decoding::config::decoder_schema decode_window_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"num_receives", param_kind::uint64});
  schema.params.push_back({"verify_residual", param_kind::string});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    verify_residual_of(p, "decode_window");
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
        if (!ctx.store || !ctx.context || !ctx.layer)
          throw std::logic_error("task graph node '" + ctx.id +
                                 "': a decode_window node is built only "
                                 "within a windowed topology");
        return std::make_unique<decode_window_node>(
            *ctx.store, ctx.context, *ctx.layer, ctx.decoder_type,
            ctx.decoder_params,
            ctx.node_params->get<std::size_t>("num_receives", 0),
            verify_residual_of(*ctx.node_params, "decode_window"));
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
