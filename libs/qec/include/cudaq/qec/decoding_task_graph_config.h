/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Plain configuration types shared by decoding_task_graph.h and the realtime
// decoder configuration (realtime/decoding_config.h), kept apart so a decoder
// configuration does not pull in the task graph itself.

#pragma once

#include <cstddef>
#include <string>

namespace cudaq::qec {

/// A directed edge of a task graph, as data: node ids and port names.
///
/// Two endpoints are implicit rather than nodes: `measurements` as the source
/// of the raw shot, and `output` as the sink. Port names default so the common
/// case stays terse; a node declaring several ports (an `xor` node's `a` and
/// `b`) names them.
struct task_graph_edge {
  std::string from_node;              ///< node id, or "measurements"
  std::string from_output = "output"; ///< named output on from_node
  std::string to_node;                ///< node id, or "output"
  std::string to_input = "input";     ///< named input on to_node

  bool operator==(const task_graph_edge &) const = default;
};

/// Construction-time execution policy for a decoding task graph.
struct task_graph_options {
  /// Process streaming records on the graph's persistent coordinator thread.
  bool async_streaming = false;
  /// Maximum public enqueue operations queued or executing. Must be nonzero
  /// when asynchronous.
  std::size_t async_queue_capacity = 8;
  /// CPU reserved for the asynchronous coordinator, or -1 to inherit the
  /// constructing thread's affinity. This pins only the coordinator; decoder
  /// pool workers retain their separately configured affinity.
  int async_coordinator_cpu = -1;
  /// Pause iterations before an active coordinator parks for more input.
  /// Zero disables spinning.
  std::size_t async_spin_pause_iters = 0;
  bool operator==(const task_graph_options &) const = default;
};

} // namespace cudaq::qec
