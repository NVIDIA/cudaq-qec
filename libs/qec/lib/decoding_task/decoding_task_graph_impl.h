/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: complete definition of decoding_task_graph::impl.
//
// Include this in any translation unit that needs to construct or inspect impl
// directly — topology factories (from_spec, combine, future node factories)
// and the graph mechanics (.cpp).  Never include from a public header.

#pragma once
#include "decoding_task_graph_node.h"
#include "thread_pool.h"
#include "cudaq/qec/decoding_task_graph.h"
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::qec {

struct decoding_task_graph::impl {
  impl();

  enum class async_enqueue_status { accepted, queue_full, failed, closed };

  struct compiled_streaming_node {
    int node_index;
    streaming_task_graph_node *capability;
  };

  // Nodes in topological execution order.
  std::vector<std::unique_ptr<task_graph_node>> nodes;

  // --- Build-time only (cleared after compile_wires()) ---
  std::vector<std::string> node_ids; // parallel to nodes[]
  std::vector<task_graph_edge> edges;

  // --- Pre-computed by compile_wires() ---

  // wires_to[i] = wires whose to_node == i.
  // wires_to.back() = wires to the output sink (to_node == -1).
  std::vector<std::vector<compiled_wire>> wires_to; // size = nodes.size() + 1

  // Pre-allocated output buffers; node_outputs[i] pre-sized to
  // outputs().size().
  mutable std::vector<std::vector<edge_value>> node_outputs;

  std::vector<std::size_t> node_input_counts;

  // Parallel one-shot execution metadata. predecessor_counts counts incoming
  // node edges (raw measurement edges do not count), and dependents is the
  // matching outgoing adjacency. source_nodes have no node predecessors.
  std::vector<int32_t> predecessor_counts;
  std::vector<std::vector<std::size_t>> dependents;
  std::vector<std::size_t> source_nodes;

  // One input staging buffer per node. The serial walk reuses these in
  // topological order; the event runtime lets independent nodes use them
  // concurrently.
  mutable std::vector<std::vector<edge_value>> parallel_node_inputs;

  // Persistent worker count is bounded by the DAG's maximum antichain width
  // (the largest possible dynamic ready/running frontier) and hardware
  // concurrency. The coordinating run_async worker is worker zero.
  std::unique_ptr<detail::thread_pool> parallel_pool;

  // Shared, preallocated event runtime used by one-shot and streaming walks.
  // Its complete definition stays in the executor translation unit.
  struct event_runtime;
  mutable std::unique_ptr<event_runtime> runtime;

  // The raw shot as an edge value; pre-built so run_dag() refills it in place.
  mutable edge_value meas_val;

  // One name per output, parallel to wires_to.back(): the sink edge's
  // to_input, or its producer's name when left at the default.
  std::vector<std::string> output_names;

  // Nodes that opted into per-round streaming, with capability lookup cached
  // once at graph compilation.
  std::vector<compiled_streaming_node> streaming_nodes;
  // Direct node-index lookup into streaming_nodes, or -1 for DAG-only nodes.
  // compile_wires() must reject every out-of-range topology index before the
  // streaming hot path uses this table without repeated bounds checks.
  std::vector<int> streaming_by_node;

  // Streaming edges indexed by producer node. Incoming compilation reuses
  // wires_to; this outgoing adjacency is retained for hot-path routing.
  std::vector<std::vector<compiled_wire>> streaming_wires_from;

  // Edges leaving a node's implicit `done` output (present when an edge names
  // it), published once the node has completed.
  std::vector<std::vector<compiled_wire>> completion_wires_from;
  std::vector<std::size_t> streaming_wires_to_count;
  mutable std::vector<std::size_t> streaming_deliveries_seen;

  // streaming_node_has_raw_input[i] is true when streaming_nodes[i]
  // receives raw syndrome bytes (no streaming predecessor feeds it via
  // enqueue_edge_value).  Populated by compile_wires().
  std::vector<bool> streaming_node_has_raw_input;

  // Streaming-only dependency metadata, indexed by streaming node index (not
  // topology node index). The reachability rows order overlapping causal
  // branches without imposing a barrier on independent branches.
  std::vector<std::vector<std::size_t>> streaming_dependents;
  std::vector<int32_t> streaming_predecessor_counts;
  std::vector<std::vector<uint8_t>> streaming_reachability;
  std::vector<std::vector<uint8_t>> streaming_overlap;
  std::size_t streaming_parallel_width = 1;

  // Sizes cached by compile_wires() so the accessors do not have to nominate a
  // representative node.  n_msyn is the largest window among the raw-input
  // streaming nodes, n_round the round size they all agreed on, and n_obs the
  // outputs' total width.
  std::size_t n_msyn = 0, n_round = 0, n_obs = 0;

  // Offsets into streaming_result_state.values, parallel to streaming_nodes:
  // one per output the node XOR-accumulates into. Empty when
  // streaming_conflict is set.
  std::vector<std::vector<std::size_t>> streaming_node_output_offsets;

  // Contributors and effective schedule for each named graph output.
  std::vector<std::vector<std::size_t>> output_streaming_nodes;
  std::vector<window_spec> output_streaming_schedules;

  // Graph-owned publication counters and persistent result storage. All are
  // pre-sized during compilation; streaming operations perform no allocation.
  std::vector<std::uint64_t> streaming_node_generations;
  std::vector<std::uint64_t> streaming_output_fire_ordinals;
  mutable task_graph_streaming_result streaming_result_state;

  // First deterministic streaming-only incompatibility. All-mode DAG errors
  // still throw from compile_wires(); this leaves run()/run_all() available.
  std::string streaming_conflict;

  // Queue, lifecycle, and reader-publication state shared between the async
  // producer, coordinator, and result accessors.
  struct async_streaming_state {
    struct slot {
      std::vector<uint8_t> syndrome;
      std::size_t len = 0;
      std::optional<round_type> type; ///< Absent when the caller gave none.
    };

    // Sequence counters are monotonic across sessions; submitted-completed is
    // the bounded outstanding count.
    std::vector<slot> slots;
    std::size_t capacity = 0;
    std::size_t spin_pause_iters = 0;
    bool enabled = false;
    std::thread thread;
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> wake_epoch{0};
    std::atomic<bool> active{false};
    std::atomic<bool> accepting{false};
    std::atomic<bool> failed{false};
    std::atomic<bool> finish_requested{false};
    std::atomic<bool> flush_requested{false};
    std::atomic<bool> cancel_requested{false};
    std::atomic<bool> shutdown{false};
    std::atomic<bool> producer_call_active{false};
    std::atomic<bool> producer_active{false};
    std::atomic<std::uint32_t> producer_waiters{0};
    std::atomic<std::uint32_t> completion_waiters{0};
    std::atomic<bool> coordinator_waiting{false};
    uint64_t finish_target = 0;
    int cuda_device_id = -1;
    bool needs_reset = false;
    bool operation_completed = false;
    bool result_changed = false;
    mutable std::mutex snapshot_mutex;
    task_graph_streaming_result result_snapshot;
    mutable task_graph_streaming_result read_snapshot;
    bool snapshot_valid = false;
    std::exception_ptr error;
  };
  async_streaming_state async_streaming;

  // D matrix: shared with the primary window_node; used only by d_matrix().
  std::shared_ptr<const sparse_binary_matrix> d;
  // O matrix: shared with the primary window_node; unset for combined graphs.
  std::shared_ptr<const sparse_binary_matrix> o;

  // Threading-contract tripwire: how many graph bodies are executing on this
  // instance right now.  A counter and not a mutex on purpose -- a lock would
  // silently serialise a caller that broke the contract instead of telling it.
  // See in_flight_guard in decoding_task_graph.cpp.
  std::atomic<uint32_t> in_flight{0};

  // Resolve string edges to integer-indexed compiled_wires; clear build-time
  // fields on return.  Defined in decoding_task_graph.cpp.
  void compile_wires();

  /// Record the first streaming-only incompatibility found during compilation.
  void record_streaming_conflict(std::string msg);

  /// Reject a streaming API before it reads or mutates streaming state.
  void require_streaming_support(const char *who) const;

  // Serial full-shot walk used by synchronous APIs: route the raw shot to each
  // "measurements" source edge and execute in topological order.
  void walk_dag(const measurement_results &m);

  // walk_dag() plus the single-output sink read, without the in-flight
  // tripwire. run() and run_batch() each arm the guard once and then call this,
  // so arming it here too would self-trip.
  /// `who` names the public entry point, so a multi-output graph's refusal
  /// points at the call the caller actually made.
  logical_outcome run_dag(const measurement_results &m, const char *who);

  /// Dependency-driven run_async() walk using the persistent graph pool.
  logical_outcome run_dag_parallel(const measurement_results &m,
                                   const char *who, int cuda_device_id);
  /// Resolve the common CUDA device for internal pool workers.
  int resolve_cuda_device_id() const;

  /// Reject an API that returns one logical outcome when the graph has any
  /// other sink cardinality. Kept separate so run_async() can validate before
  /// choosing its single-node fast path.
  void require_single_output(const char *who) const;

  /// Record one node publication and update graph-output barriers afterward.
  void record_streaming_publication(int node_idx);
  void refresh_streaming_progress(bool final);
  void consume_streaming_result();
  void reset_streaming_progress();
  void fail_streaming_progress(const std::string &message);
  void compose_streaming_values() const;

  // The two streaming event walks, without the in-flight tripwire: the tagged
  // enqueue runs both in one call, and arming the guard twice would self-trip.
  // `type` is absent for an untagged record, which nodes receive untagged.
  bool enqueue_sweep(const uint8_t *syndrome, std::size_t len,
                     std::optional<round_type> type, int cuda_device_id);
  bool flush_sweep(int cuda_device_id);
  /// Allocate the bounded event store after graph compilation.
  void prepare_event_runtime(std::size_t capacity);
  /// Process one streaming record and update public result progress.
  bool process_streaming_record(const uint8_t *syndrome, std::size_t len,
                                std::optional<round_type> type);
  /// Flush an asynchronously finished stream and update public result progress.
  bool finish_streaming_record();

  /// Publish a stable snapshot from the coordinator between record executions.
  void publish_async_stream_snapshot();

  /// Persistent asynchronous stream coordinator entry point.
  void async_stream_loop() noexcept;
  /// Record the first background streaming failure and close acceptance.
  void store_async_stream_error(std::exception_ptr error) noexcept;
  /// Reflect a background exception in result and control-plane state.
  void handle_async_stream_error(std::exception_ptr error) noexcept;
  /// Rethrow the first background streaming failure, if any.
  void rethrow_async_stream_error();
  /// Invalidate the reader-facing result after cancellation or graph reuse.
  void invalidate_async_stream_snapshot();
  /// Wake the coordinator after publishing input or a lifecycle request.
  void wake_async_stream() noexcept;
  /// Wake a parked coordinator after publishing one queue record.
  void wake_async_stream_for_record() noexcept;
  /// Stop and join the prepared coordinator during destruction.
  void shutdown_async_streaming() noexcept;
  /// Allocate the bounded queue and launch the parked coordinator.
  void prepare_async_streaming(std::size_t queue_capacity,
                               std::size_t initial_payload_bytes,
                               int coordinator_cpu,
                               std::size_t spin_pause_iters);
  /// Acquire graph exclusivity and begin accepting one shot's records.
  void begin_async_streaming(int cuda_device_id);
  /// Try to copy one public enqueue operation into the bounded queue.
  async_enqueue_status
  try_enqueue_syndrome_async(const uint8_t *syndrome, std::size_t len,
                             std::optional<round_type> type);
  /// Wait until one bounded queue slot becomes available.
  void wait_for_async_enqueue_slot();
  /// Refresh and return the stable reader-owned result snapshot.
  const task_graph_streaming_result &async_streaming_result() const;
  /// Copy and enqueue one public streaming input operation.
  bool enqueue_syndrome_async(const uint8_t *syndrome, std::size_t len,
                              std::optional<round_type> type,
                              int cuda_device_id);
  /// Wait until all records submitted before this call have completed.
  void synchronize_async_streaming();
  /// Drain accepted input without flushing, then release graph exclusivity.
  void drain_async_streaming();
  /// Drain accepted input, flush once, and release graph exclusivity.
  void finish_async_streaming();
  /// Discard queued input and release graph exclusivity for reset.
  void cancel_async_streaming();

  // Set by a FINAL round; cleared by clear_corrections() and reset_decoder().
  // While set, further rounds are warned about and dropped -- the rounds a
  // flush committed are the ones the next window's core would commit, and
  // corrections accumulate by XOR, so resuming would cancel them to zero.
  bool finalized = false;

  ~impl();
};

} // namespace cudaq::qec
