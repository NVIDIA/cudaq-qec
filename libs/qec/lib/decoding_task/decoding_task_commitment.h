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
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cudaq::qec {

class data_artifact_store;

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
  bool operator==(const scheduled_window &) const = default;
};

/// The decoders a node builds for one variant of its layer, and the sha256
/// the store records the variant under.
struct variant_decoders {
  std::string node;
  variant_key variant;
  std::size_t decoders = 0;
  std::string sha256;
  bool operator==(const variant_decoders &) const = default;
};

/// Every window of a shot, by layer in the order given, then by k, and, by
/// node and variant, the decoders each node builds for the variants the
/// shot uses. Only describe_schedule() lists the decoders.
struct window_schedule {
  std::size_t num_rounds = 0;
  std::vector<scheduled_window> windows;
  std::vector<variant_decoders> decoders;
  /// By ingest node: the sha256 the store records its template under.
  std::map<std::string, std::string> templates;
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
/// @param store Where the layers' windows are built and read, once each, so
/// the graph's nodes read the very windows checked; null to build them for
/// the check alone. @p context must be the store's own template.
/// @throws std::invalid_argument naming the check, and the row, column,
/// window or edge that fails it.
void validate_commitment(std::shared_ptr<const experiment_template> context,
                         const std::vector<window_layer> &layers,
                         const std::vector<task_graph_edge> &edges,
                         data_artifact_store *store = nullptr);

/// Check a topology's windowed decoding before any node is built, and do
/// nothing for a topology with no ingest or decode_window node. Every ingest
/// node is fed by measurements alone and has a template, with a readout map
/// when it reads the logical value. Every decode_window node is fed by one
/// window port of an ingest node, which gives its layer, and receives only
/// from decode_window nodes, on ports below its num_receives, from its own
/// or earlier instances; the receives from the same instance close no
/// cycle, in which a window would wait on itself. Two ingest nodes are two
/// shots: no send joins their windows, and ones with different templates
/// reach no common node. Each ingest node's layers and sends then pass
/// validate_commitment().
/// Each ingest node's template is registered in @p store, and its layers'
/// windows are built there as validate_commitment() checks them.
/// @return Each ingest node's layers, by node id: the decode_window nodes
/// reading it, in id order. Empty for a topology with neither kind.
/// @throws std::invalid_argument naming the node or edge, or as
/// experiment_template and validate_commitment() do.
std::map<std::string, std::vector<window_layer>>
validate_window_topology(const std::vector<task_graph_node_config> &nodes,
                         const std::vector<task_graph_edge> &edges,
                         data_artifact_store &store);

/// The windows a @p num_rounds round shot of a topology decodes, as
/// from_topology() would build it with @p options: every ingest node's
/// windows, scheduled apart, and the decoders each node builds per variant
/// the shot uses: one per copy for the bulk, as an asynchronous graph runs
/// max_in_flight copies, and one for each prefix or terminal variant, which
/// the copies share; with the sha256 of each variant and template, as the
/// store records them. Lowers patterns and validates the windows as
/// from_topology() does, but builds no node. Internal until cudaq-edem fixes
/// what a window is; empty for a topology without windows.
/// @throws std::invalid_argument as from_topology() does for the layout, or
/// if a template admits no shot of @p num_rounds rounds.
window_schedule describe_schedule(std::vector<task_graph_node_config> nodes,
                                  std::vector<task_graph_edge> edges,
                                  const task_graph_options &options,
                                  std::size_t num_rounds);

} // namespace cudaq::qec
