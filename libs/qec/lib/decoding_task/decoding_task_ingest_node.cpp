/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `ingest` node and its kind: the raw-fed node that cuts a windowed
// graph's shot into windows. It forms detectors from raw rounds and publishes
// each layer's windows, one output port per entry of its `windows` map. Its
// kind builds it from the template and layers a windowed topology resolves.

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

// One output port per layer, named as its window entry is.
ingest_node::ingest_node(
    std::shared_ptr<const experiment_template> context,
    const std::vector<experimental::ingest_window> &windows)
    : core_(context, core_layers(windows)), layers_(windows.size()) {
  for (const auto &window : windows)
    outputs_.push_back(window.layer);
  // Every publication a record can ready, each window buffer at its layer's
  // most rows, so no shot allocates, FINAL's burst included.
  ready_.resize(core_.max_ready_per_record());
  for (auto &entry : ready_) {
    entry.present.assign(layers_, 0);
    for (std::size_t l = 0; l < layers_; ++l) {
      window_rows rows;
      rows.detectors.reserve(core_.max_window_rows(l));
      entry.windows.emplace_back(std::move(rows));
    }
  }
} // end - ingest_node()

// One-shot paths stream the shot through the graph instead.
void ingest_node::execute_indexed(const std::vector<edge_value> &,
                                  std::vector<edge_value> &) const {
  throw std::logic_error("ingest: the node streams; it has no one-shot path");
}

// A window per layer.
std::optional<edge_value_type>
ingest_node::output_type(std::size_t slot) const {
  if (slot < layers_)
    return edge_value_type::of<window_rows>();
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

// FINAL readied every window, so the shot's end publishes nothing.
bool ingest_node::flush() {
  check_handed_over("flush()");
  clear_publications();
  flushed_ = core_.shot_rounds().has_value();
  return false;
}

// The last publication stays current, since a graph that reads outputs in
// place reads them after this returns false; enqueue() and flush() clear it.
bool ingest_node::publish_next() {
  if (current_ != none && current_ + 1 < filled_) {
    ++current_;
    return true;
  }
  return false;
}

std::optional<std::int64_t>
ingest_node::published_index(std::size_t slot) const {
  if (slot >= layers_ || current_ == none)
    return std::nullopt;
  return ready_[current_].index;
}

// Null for a layer without a window at the current index, and while no
// publication is current.
const edge_value *ingest_node::streaming_output(std::size_t slot) const {
  if (slot < layers_ && current_ != none && ready_[current_].present[slot])
    return &ready_[current_].windows[slot];
  return nullptr;
}

// Clearing the corrections of a flushed shot is how a caller starts the
// next one, as it is for every streaming node.
void ingest_node::clear_corrections() {
  if (flushed_)
    reset_streaming_state();
}

// No view is out, so the core resets at any point of a shot.
void ingest_node::reset_streaming_state() {
  core_.reset();
  clear_publications();
  flushed_ = false;
}

// The publications are sized at construction for the most a record can
// ready; growing past them would mean that bound is wrong.
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
void ingest_node::check_handed_over(const char *call) const {
  if (current_ == none || current_ + 1 >= filled_)
    return;
  throw std::logic_error(std::string("ingest: ") + call + " while " +
                         std::to_string(filled_ - current_ - 1) +
                         " publications wait to be handed over; "
                         "publish_next() hands each over");
}

void ingest_node::clear_publications() {
  filled_ = 0;
  current_ = none;
}

namespace {

// It reads its template from the DEM and takes windows, a node field, not a
// parameter; it takes no parameters and wraps no decoder.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "ingest";
  spec.requires_decoder = false;
  spec.requires_dem = true;
  spec.takes_windows = true;
  spec.takes_template = true;
  register_node_kind(
      std::move(spec), {},
      [](node_build_context &ctx) -> std::unique_ptr<task_graph_node> {
        if (!ctx.context)
          throw std::logic_error("task graph node '" + ctx.id +
                                 "': an ingest node is built only within a "
                                 "windowed topology");
        return std::make_unique<ingest_node>(ctx.context, ctx.windows);
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
