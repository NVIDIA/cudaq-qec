/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Plain configuration types shared by decoding_task_graph.h and the realtime
// decoder configuration (realtime/decoding_config.h), kept apart so a decoder
// configuration does not pull in the task graph itself.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
  /// Consumer instance k reads producer instance k + index_offset, between
  /// nodes that join their inputs by index. Zero or negative.
  std::int64_t index_offset = 0;

  bool operator==(const task_graph_edge &) const = default;
};

/// Construction-time execution policy for a decoding task graph.
struct task_graph_options {
  /// Process streaming records on the graph's persistent coordinator thread.
  /// Records then overlap: a later record runs ahead on nodes that finished
  /// the earlier ones, up to 16 records in flight, or the queue's capacity if
  /// smaller.
  bool async_streaming = false;
  /// Maximum public enqueue operations queued or executing: how far the
  /// producer may run ahead of the graph. Must be nonzero when asynchronous.
  /// A windowed graph admits queued rounds together, so a deeper queue lets
  /// it catch up in larger batches, but a graph that falls behind returns
  /// FINAL's result only after draining it. 32 keeps that backlog to tens of
  /// microseconds while still batching; each slot holds one enqueue.
  std::size_t async_queue_capacity = 32;
  /// CPU reserved for the asynchronous coordinator, or -1 to inherit the
  /// constructing thread's affinity. This pins only the coordinator; decoder
  /// pool workers retain their separately configured affinity.
  int async_coordinator_cpu = -1;
  /// Pause iterations before an active coordinator parks for more input.
  /// Zero disables spinning.
  std::size_t async_spin_pause_iters = 0;
  /// Pool threads that run node fires in an asynchronous graph, not counting
  /// the coordinator. Unset sizes the pool from the graph: its parallel
  /// width, including every fire pool's copies, or, with an indexed node,
  /// every copy plus the worker its fires leave free, capped by hardware
  /// concurrency. Must be at least one when set. Synchronous graphs ignore
  /// it.
  std::optional<std::size_t> async_num_workers;
  /// CPUs those threads are pinned to, round-robin (Linux only). Empty
  /// inherits the process's affinity. A CPU listed twice or shared with
  /// `async_coordinator_cpu` is an error; one shared with another decoder's
  /// asynchronous graph is a warning. Synchronous graphs ignore it.
  std::vector<int> async_worker_cpus;
  bool operator==(const task_graph_options &) const = default;
};

} // namespace cudaq::qec
