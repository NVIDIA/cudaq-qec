/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the window schedule of a shot and commitment validation.
// A layer is the windows one decode_window node decodes, from one ingest
// window port; edges between decode_window nodes carry a layer's sends to
// other layers' receive inputs, with their index maps.
// The build fills both from the topology, calls validate_commitment(), and
// reports schedule_windows() through describe_schedule(). Never included
// from a public header.

#pragma once

#include "decoding_task_decoding_window.h"
#include "decoding_task_experiment_template.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/decoding_task_graph_config.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudaq::qec {

/// The windows one decode_window node decodes: the request of the ingest
/// window port that feeds it.
struct window_layer {
  std::string node;       ///< The decode_window node's id
  std::string port;       ///< The ingest output port: the layer's name
  window_request request; ///< That port's entry of ingest.windows
};

/// An input of a scheduled window: the producer instance an edge maps it
/// to. Field names follow task_graph_edge.
struct window_dependency {
  std::string from_node;
  std::string from_output;
  std::int64_t k = 0; ///< The producer instance
  std::string to_input;
  bool exists = false; ///< An absent producer gives an empty input
  bool operator==(const window_dependency &) const = default;
};

/// One window instance of a shot: what it decodes and what it waits for.
struct scheduled_window {
  std::string node;
  std::int64_t k = 0;
  variant_key variant;
  round_range rounds;
  round_range commit;
  std::vector<round_range> buffers;
  window_port input;
  window_port output;
  std::vector<window_send> sends;
  std::vector<window_dependency> inputs;
};

/// Every window of a shot, by layer in the order given, then by k.
struct window_schedule {
  std::size_t num_rounds = 0;
  std::vector<scheduled_window> windows;
};

/// The windows of a @p num_rounds round shot and their dependencies.
/// @throws std::invalid_argument naming the operation if a request is
/// invalid, a node is repeated, or an edge names an unknown node or send or
/// reaches further than window_round_limit instances.
window_schedule
schedule_windows(std::shared_ptr<const experiment_template> context,
                 const std::vector<window_layer> &layers,
                 const std::vector<task_graph_edge> &edges,
                 std::size_t num_rounds);

/// The shot lengths validate_commitment() checks: [min_rounds(), end).
/// They cover every prefix and terminal variant of every layer, two bulk
/// periods, and the periods the farthest of @p edges reaches back.
/// @throws std::invalid_argument naming the operation if a request is
/// invalid, the layers' periods differ, or an edge reaches further than
/// window_round_limit instances.
std::size_t commitment_probe_rounds(const experiment_template &context,
                                    const std::vector<window_layer> &layers,
                                    const std::vector<task_graph_edge> &edges);

/// Check, for every probe shot length, that every detector row has exactly
/// one finalizer, that every fault touching it is the finalizer's or
/// committed elsewhere and delivered to the finalizer by a send covering
/// the row, and that every detectable fault is committed exactly once
/// (F1-F4); that sends match receives, none twice; and that no commit
/// column has an ambiguous parallel column.
/// @throws std::invalid_argument naming the check, and the row, column,
/// window or edge that fails it.
void validate_commitment(std::shared_ptr<const experiment_template> context,
                         const std::vector<window_layer> &layers,
                         const std::vector<task_graph_edge> &edges);

/// Check a topology's windowed decoding before any node is built, and do
/// nothing for a topology with no ingest or decode_window node. Every ingest
/// node is fed by measurements alone and has a template, with a readout map
/// when it reads the logical value. Every decode_window node is fed by one
/// window port of an ingest node, which gives its layer, and receives only
/// from decode_window nodes, on ports below its num_receives, from its own
/// or earlier instances. Two ingest nodes are two shots: no send joins their
/// windows, and ones with different templates reach no common node. Each
/// ingest node's layers and sends then pass validate_commitment().
/// @throws std::invalid_argument naming the node or edge, or as
/// experiment_template and validate_commitment() do.
void validate_window_topology(const std::vector<task_graph_node_config> &nodes,
                              const std::vector<task_graph_edge> &edges);

} // namespace cudaq::qec
