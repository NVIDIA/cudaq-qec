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
#include <utility>
#include <vector>

namespace cudaq::qec {

/// Combine closures: not_checked outranks open, which outranks closed, so a
/// shot is closed only when every window was checked and held. None, for
/// a node that decodes no window, changes nothing.
inline std::optional<task_graph_closure>
combine_closure(std::optional<task_graph_closure> a,
                std::optional<task_graph_closure> b) {
  const auto rank = [](task_graph_closure c) {
    return c == task_graph_closure::closed ? 0
           : c == task_graph_closure::open ? 1
                                           : 2;
  };
  if (!a || !b)
    return a ? a : b;
  return rank(*a) >= rank(*b) ? a : b;
}

struct decoding_task_graph::impl {
  impl();

  enum class async_enqueue_status { accepted, queue_full, failed, closed };

  struct compiled_streaming_node {
    int node_index;
    streaming_task_graph_node *capability; ///< same as replicas[0]
    /// Every copy of the node; more than one makes it a fire pool.
    std::vector<streaming_task_graph_node *> replicas;
    /// The copies' index-join capability, parallel to replicas; null when
    /// the node joins by record.
    std::vector<indexed_join_node *> indexed;
  };

  // Nodes in topological execution order.
  std::vector<std::unique_ptr<task_graph_node>> nodes;
  // Extra copies of nodes[n] for a fire pool (max_in_flight - 1 of them).
  std::vector<std::vector<std::unique_ptr<task_graph_node>>> node_replicas;

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
  // concurrency. The caller is worker zero; an asynchronous graph adds one
  // thread so its dispatching coordinator never does node work.
  std::size_t pool_width = 1;
  std::vector<std::int32_t> pool_worker_cpus;
  std::unique_ptr<detail::thread_pool> parallel_pool;

  // Most bytes per record context: one chunk of the graph's single raw-input
  // node, when it meters its own. Longer tagged records run as several;
  // async graphs join queued rounds up to it. Zero: records taken as given.
  std::size_t record_bytes = 0;

  // Preallocated record contexts and the scheduler that runs them. Their
  // complete definitions stay in the executor translation unit.
  struct record_context;
  struct record_scheduler;
  mutable std::unique_ptr<record_scheduler> scheduler;

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
  // All streaming wires, and those carrying a value (each has a buffer id).
  std::size_t streaming_wire_count = 0;
  std::size_t streaming_value_wire_count = 0;

  // streaming_node_has_raw_input[i] is true when streaming_nodes[i]
  // receives raw syndrome bytes (no streaming predecessor feeds it via
  // enqueue_edge_value).  Populated by compile_wires().
  std::vector<bool> streaming_node_has_raw_input;

  // Streaming-only dependency metadata, indexed by streaming node index (not
  // topology node index). The reachability rows order overlapping causal
  // branches without imposing a barrier on independent branches.
  std::vector<std::vector<std::size_t>> streaming_dependents;
  std::vector<std::vector<std::size_t>> streaming_producers;
  // By streaming node: the other nodes that reach it through a loop-carried
  // wire, which the dependency graph above leaves out.
  std::vector<std::vector<std::size_t>> streaming_loop_producers;
  std::vector<int32_t> streaming_predecessor_counts;
  std::vector<std::vector<uint8_t>> streaming_reachability;
  /// How two streaming nodes' branches meet: not at all; only at
  /// rendezvous descendants, which take inputs in any order; or at one that
  /// needs ordered delivery.
  enum class streaming_overlap_kind : uint8_t { disjoint, rendezvous, ordered };
  std::vector<std::vector<streaming_overlap_kind>> streaming_overlap;
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

  // A staged output folds exact commit intervals in round order: a delta
  // waits in `pending` until the frontier reaches its first round. The
  // contributors' deltas never go through latest_node_corrections. Its
  // intervals are preallocated, and `pending` and `spare` index them, so
  // staging allocates nothing.
  struct output_stager {
    struct interval {
      std::uint64_t begin = 0;
      std::uint64_t end = 0;
      std::vector<uint8_t> bits;
    };
    bool enabled = false;
    std::uint64_t frontier = 0;
    std::uint64_t generation = 0;
    std::vector<uint8_t> bits;  ///< the frame at the frontier
    std::vector<uint8_t> whole; ///< whole-shot deltas held until FINAL
    bool has_whole = false;
    std::vector<interval> intervals;  ///< capacity of them, bits sized
    std::vector<std::size_t> pending; ///< waiting, sorted by begin round
    std::vector<std::size_t> spare;   ///< free intervals
    std::size_t capacity = 0;         ///< fixed at build; overflow is a bug
    /// Return every waiting interval to the spares.
    void clear_pending() {
      spare.insert(spare.end(), pending.begin(), pending.end());
      pending.clear();
    }
  };
  std::vector<output_stager> output_stagers;
  std::vector<uint8_t> streaming_node_staged; // by streaming node
  std::vector<std::vector<std::size_t>> streaming_node_outputs; // sinks fed
  void stage_delta(std::size_t k,
                   streaming_task_graph_node::committed_coverage coverage,
                   const std::vector<uint8_t> &bits);
  void fold_whole_shot(output_stager &stager);

  // Graph-owned publication counters and persistent result storage. All are
  // pre-sized during compilation; streaming operations perform no allocation.
  std::vector<std::uint64_t> streaming_node_generations;
  std::vector<std::uint64_t> streaming_output_fire_ordinals;
  mutable task_graph_streaming_result streaming_result_state;
  // Node state as of the latest retired record, indexed by streaming node.
  // The progress model reads these rather than the nodes, which may already
  // be serving a later record.
  std::vector<std::vector<uint8_t>> latest_node_corrections;
  std::vector<std::optional<streaming_task_graph_node::committed_coverage>>
      latest_node_coverage;
  // Each node's publications' closures this shot, combined; none before its
  // first window, or for a node that decodes none.
  std::vector<std::optional<task_graph_closure>> latest_node_closure;
  // Each node's publications this shot: whether every one's decoder
  // converged; none before its first window.
  std::vector<std::optional<bool>> latest_node_converged;
  // A node has no one-shot path, so one-shot calls run through the stream.
  bool one_shot_streams = false;
  // A record was accepted since the last reset, or since a completed shot
  // was consumed, and its shot has not failed: one-shot calls through the
  // stream would discard it. Atomic, as an overlapping enqueue sets it
  // before it is rejected, and an asynchronous failure clears it.
  std::atomic<bool> shot_open{false};
  // A one-shot call through the stream is running: held for the whole call,
  // as each step it takes guards only itself.
  std::atomic<bool> one_shot_running{false};
  // reset_decoder() and clear_corrections() calls, each of which ends a
  // shot: a one-shot call compares it to tell another call's from its own.
  std::atomic<std::uint64_t> shot_resets{0};
  /// Whether every window reaching output @p k converged this shot.
  bool output_converged(std::size_t k) const;
  // Progress captured when the record being retired began.
  std::uint64_t retiring_generation = 0;
  std::uint64_t retiring_revision = 0;

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
    /// Admit queued rounds together: record_bytes is nonzero.
    bool coalesce = false;
    int cuda_device_id = -1;
    bool needs_reset = false;
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

  void refresh_streaming_progress(bool final);
  void consume_streaming_result();
  void reset_streaming_progress();
  void fail_streaming_progress(const std::string &message);
  /// Zero every value the result is composed from and its coverage, and mark
  /// it pending, so nothing of the previous shot stays readable.
  void hide_streaming_result();
  void compose_streaming_values() const;
  void clear_latest_node_corrections();

  // The two streaming event walks, without the in-flight tripwire: the tagged
  // enqueue runs both in one call, and arming the guard twice would self-trip.
  // `type` is absent for an untagged record, which nodes receive untagged.
  bool enqueue_sweep(const uint8_t *syndrome, std::size_t len,
                     std::optional<round_type> type, int cuda_device_id);
  bool flush_sweep(int cuda_device_id);
  /// One record through one context; enqueue_sweep() cuts long ones.
  bool sweep_record(const uint8_t *syndrome, std::size_t len,
                    std::optional<round_type> type, int cuda_device_id);
  /// Allocate the worker pool and record contexts after graph compilation.
  void prepare_event_runtime(std::size_t context_count, bool dispatcher_thread);
  /// Drop every record, join entry and replica lease, as a new shot needs.
  void discard_records();
  /// Fold a finished record's captured node state into the graph-level view,
  /// one public operation at a time.
  /// @return Operations before the last that advanced the result, and the
  /// generation the last started from.
  std::pair<std::uint64_t, std::uint64_t>
  apply_record_context(record_context &c);
  /// The mid-shot generation refresh_streaming_progress() would publish.
  std::uint64_t pending_generation() const;
  /// Retire the oldest asynchronous record and publish its progress.
  void retire_record_context(record_context &c);
  /// Worker zero of an asynchronous session: admit and retire records.
  void dispatch_async_session();
  /// Run one asynchronous session on the pool until finish, cancel, or failure.
  void run_async_session();
  /// Process one streaming record and update public result progress.
  bool process_streaming_record(const uint8_t *syndrome, std::size_t len,
                                std::optional<round_type> type);
  /// Either public enqueue_syndrome(): opens the shot, unless the record is
  /// rejected, and runs it synchronously or asynchronously.
  bool enqueue_record(const uint8_t *syndrome, std::size_t len,
                      std::optional<round_type> type, int cuda_device_id);
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
  /// Spin up to `pause_iters`; true when the epoch moved past `observed`.
  bool spin_for_async_wake(uint64_t observed,
                           std::size_t pause_iters) const noexcept;
  /// Park the coordinator until the wake epoch moves or `work_ready()` holds.
  /// The waiting flag is advertised before the check, so a producer either
  /// sees it and wakes us or we see its record.
  template <typename WorkReady>
  void wait_for_async_wake(uint64_t &observed_epoch, WorkReady &&work_ready,
                           std::size_t pause_iters) {
    auto &state = async_streaming;
    if (state.active.load(std::memory_order_acquire) &&
        spin_for_async_wake(observed_epoch, pause_iters)) {
      observed_epoch = state.wake_epoch.load(std::memory_order_acquire);
      return;
    }
    state.coordinator_waiting.exchange(true, std::memory_order_acq_rel);
    if (!work_ready() &&
        state.wake_epoch.load(std::memory_order_acquire) == observed_epoch)
      state.wake_epoch.wait(observed_epoch, std::memory_order_acquire);
    state.coordinator_waiting.store(false, std::memory_order_release);
    observed_epoch = state.wake_epoch.load(std::memory_order_acquire);
  }
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
  /// Drain accepted input and flush once; return when the result is
  /// published. A failed or cancelled session first releases the graph,
  /// and a failure rethrows.
  void finish_async_streaming();
  /// Wait for a finished session to release the graph.
  void settle_async_streaming();
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
