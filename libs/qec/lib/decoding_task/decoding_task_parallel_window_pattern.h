/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the `parallel_window` pattern's expansion. Written as one
// node P, wired measurements -> P -> output, the pattern stands for the
// parallel-window graph: P.ingest with layers A and B, the decode_window
// nodes P.A and P.B, and their xor P.sum. The graph builder lowers it before
// any node is checked or built, as it lowers ensembles, so the expanded
// graph is validated as a hand-written one is. Never included from a public
// header.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "decoding_task_experiment_template.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/window_request.h"
#include <utility>
#include <vector>

namespace cudaq::qec {

/// The pattern's layers: A commits commit_rounds between buffer_rounds on
/// each side, with an open input and forward and backward sends; B commits
/// the gap_rounds before A's commit zone, closed at both ports. Both share
/// the column options. A's open input with boundary columns takes
/// boundary_prior, or else the largest prior of a fault that crosses into
/// the next round.
/// @param params The pattern's parameters, checked against its schema.
/// @param context The template the windows are built on.
/// @throws std::invalid_argument if A's input needs a prior and no fault
/// crosses into the next round with a prior strictly between 0 and 1.
std::pair<experimental::window_request, experimental::window_request>
parallel_window_requests(const cudaqx::heterogeneous_map &params,
                         const experiment_template &context);

/// Whether @p cfg is a `parallel_window` pattern.
bool is_parallel_window(const task_graph_node_config &cfg);

/// Lower pattern @p cfg, of id P, into P.ingest, P.A, P.B and P.sum,
/// appended to @p out in that order, and their edges, appended to
/// @p edges. The edge into P's input feeds P.ingest instead, and the edges
/// from P's output leave P.sum. With `output: corrected_logical`, the
/// default, P.ingest reads the logical value into P.sum's third input. The
/// inner decoder, max_in_flight and verify_residual go to both layers.
/// @throws std::invalid_argument naming the pattern if it has no id, its
/// parameters are invalid, it has no dem_chunks template, it is not fed by
/// exactly one edge into 'input', an edge names another port, or as
/// parallel_window_requests() does.
void lower_parallel_window(task_graph_node_config cfg,
                           std::vector<task_graph_node_config> &out,
                           std::vector<task_graph_edge> &edges);

} // namespace cudaq::qec
