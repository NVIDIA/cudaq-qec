/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `ingest` node and its kind: the only raw-fed node of a windowed graph.
// It forms detectors from raw rounds and publishes each layer's windows, one
// output port per entry of its `windows` map, and with `read_logical` the
// raw logical value on `logical`. The kind is registered so configurations
// that use it parse and validate; building it is not available yet.

#include "decoding_task_ingest_node.h"
#include "decoding_task_node_kind_registry.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

namespace {

// The ingest core's layers: a node copies each view as it publishes it and
// releases it at once, so one view per layer is ever out.
std::vector<ingest_layer>
core_layers(const std::vector<experimental::ingest_window> &windows) {
  std::vector<ingest_layer> layers;
  for (const auto &window : windows)
    layers.push_back({window.request, 1});
  return layers;
}

} // namespace

// The logical value has one bit per row of the last phase's m2o_sparse.
ingest_node::ingest_node(
    std::shared_ptr<const experiment_template> context,
    const std::vector<experimental::ingest_window> &windows, bool read_logical)
    : core_(context, core_layers(windows), read_logical),
      layers_(windows.size()), read_logical_(read_logical) {
  for (const auto &window : windows)
    outputs_.push_back(window.layer);
  if (read_logical) {
    outputs_.push_back("logical");
    corrections_.assign(
        static_cast<std::size_t>(std::count(context->readout().begin(),
                                            context->readout().end(), -1)),
        0);
  }
  logical_ = logical_outcome{corrections_, true};
} // end - ingest_node()

// One-shot paths stream the shot through the graph instead.
void ingest_node::execute_indexed(const std::vector<edge_value> &,
                                  std::vector<edge_value> &) const {
  throw std::logic_error("ingest: the node streams; it has no one-shot path");
}

// A window per layer, then the logical value.
std::optional<edge_value_type>
ingest_node::output_type(std::size_t slot) const {
  if (slot < layers_)
    return edge_value_type::of<window_rows>();
  if (slot == layers_ && read_logical_)
    return logical_outcome_type;
  return std::nullopt;
}

// Raw measurements, on `input` alone.
bool ingest_node::accepts_input_type(std::size_t slot,
                                     edge_value_type type) const {
  return slot == 0 && type == measurement_results_type;
}

// Every index the chunk readies is copied out of the ring as it is polled,
// so no view stays out and the ring always has room for a record.
bool ingest_node::enqueue(const uint8_t *msyn, std::size_t len,
                          round_type type) {
  check_handed_over("enqueue()");
  clear_publications();
  if (core_.enqueue(msyn, len, type == round_type::FINAL) ==
      ingest_status::deferred)
    throw std::logic_error("ingest: the ring has no room for a record, "
                           "though no view is out");
  collect();
  current_ = filled_ > 0 ? 0 : none;
  return filled_ > 0;
}

// The logical value is ready once FINAL is taken; it is published alone.
bool ingest_node::flush() {
  check_handed_over("flush()");
  clear_publications();
  if (!read_logical_ || !core_.shot_rounds() || logical_published_)
    return false;
  std::copy(core_.logical().begin(), core_.logical().end(),
            corrections_.begin());
  logical_.get<logical_outcome>().bits = corrections_;
  logical_current_ = true;
  logical_published_ = true;
  return true;
}

// The logical value is never followed by another publication.
bool ingest_node::publish_next() {
  if (!logical_current_ && current_ != none && current_ + 1 < filled_) {
    ++current_;
    return true;
  }
  clear_publications();
  return false;
}

std::optional<std::int64_t> ingest_node::published_index() const {
  if (logical_current_ || current_ == none)
    return std::nullopt;
  return ready_[current_].index;
}

// Null for a layer without a window at the current index, and for every
// port no publication is current on.
const edge_value *ingest_node::streaming_output(std::size_t slot) const {
  if (slot < layers_ && !logical_current_ && current_ != none &&
      ready_[current_].present[slot])
    return &ready_[current_].windows[slot];
  if (slot == layers_ && logical_current_)
    return &logical_;
  return nullptr;
}

std::optional<streaming_task_graph_node::committed_coverage>
ingest_node::streaming_coverage() const {
  if (!logical_current_)
    return std::nullopt;
  return committed_coverage{0, whole_shot};
}

void ingest_node::clear_corrections() {
  std::fill(corrections_.begin(), corrections_.end(), 0);
}

// No view is out, so the core resets at any point of a shot.
void ingest_node::reset_streaming_state() {
  core_.reset();
  clear_publications();
  logical_published_ = false;
  clear_corrections();
}

// A publication's buffers are kept for the next shot, so once every index a
// record can ready has been seen, collecting allocates nothing.
void ingest_node::collect() {
  while (const auto *p = core_.poll()) {
    if (filled_ == ready_.size())
      ready_.push_back({0, std::vector<edge_value>(layers_, window_rows{}),
                        std::vector<std::uint8_t>(layers_, 0)});
    auto &entry = ready_[filled_++];
    entry.index = p->k - core_.first_index();
    for (std::size_t l = 0; l < layers_; ++l) {
      const auto *view = p->views[l];
      entry.present[l] = view != nullptr;
      if (!view)
        continue;
      core_.copy_window(*view, entry.windows[l].get<window_rows>());
      core_.release(*view);
    }
  } // end - while(p)
}

// The current publication counts as handed over; those after it wait.
void ingest_node::check_handed_over(const std::string &call) const {
  if (logical_current_ || current_ == none || current_ + 1 >= filled_)
    return;
  throw std::logic_error("ingest: " + call + " while " +
                         std::to_string(filled_ - current_ - 1) +
                         " publications wait to be handed over; "
                         "publish_next() hands each over");
}

void ingest_node::clear_publications() {
  filled_ = 0;
  current_ = none;
  logical_current_ = false;
}

namespace {

/// The kind's own parameter; its windows are a node field, not a parameter.
decoding::config::decoder_schema ingest_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"read_logical", param_kind::boolean});
  return schema;
}

// It reads its template from the DEM and takes windows; it wraps no decoder.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "ingest";
  spec.requires_decoder = false;
  spec.requires_dem = true;
  spec.takes_windows = true;
  register_node_kind(
      std::move(spec), ingest_params(),
      [](node_build_context &ctx) -> std::unique_ptr<task_graph_node> {
        throw std::invalid_argument(
            "task graph node '" + ctx.id +
            "': the 'ingest' kind is not available yet");
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
