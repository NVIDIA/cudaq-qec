/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
///
/// Shared event-driven execution for one-shot and streaming task graphs.
/// Every graph walk runs inside a record context, which owns the event store
/// and every value the record moves between nodes. The scheduler lets several
/// streaming records execute at once while each node still sees them in order.

#include "decoding_task_graph_impl.h"
#include "../hardware_guards.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace cudaq::qec {

namespace {

constexpr std::size_t no_event = std::numeric_limits<std::size_t>::max();
constexpr std::int64_t no_index = std::numeric_limits<std::int64_t>::min();

// Hand-offs between the dispatcher and workers usually complete within a few
// microseconds, far less than a park and wake. Spin that long first while a
// session has records in flight; idle sessions park at once.
constexpr std::size_t kHandoffSpinPauseIters = 4096;

enum class graph_event_kind : uint8_t {
  one_shot,
  streaming_raw,
  streaming_edge,
  streaming_flush,
  streaming_flush_release,
  streaming_indexed_fire
};

struct graph_event {
  graph_event_kind kind = graph_event_kind::one_shot;
  std::size_t node = 0;
  compiled_wire wire{};
  std::size_t offset = 0;
  std::size_t root = no_event;
  std::int64_t index =
      no_index; ///< publication index, when the producer has one
  std::size_t entry = no_event; ///< join-table entry an indexed fire consumes
  bool absent = false; ///< the producer's instance `index` never exists
};

} // namespace

// Everything one record moves between nodes or reads back from a node after a
// fire lives here, so concurrently executing records never share data.
struct decoding_task_graph::impl::record_context {
  enum class phase : uint8_t { idle, one_shot, enqueue, flush };

  struct slot {
    graph_event event;
    graph_event continuation;
    std::size_t pool_slot = no_event; ///< replica a running delivery holds
    bool releases_pool = false;       ///< last delivery: hand the replica back
    edge_value value;                 ///< an indexed delivery's own copy
    bool has_value = false;
    std::size_t parent = no_event;
    std::size_t blocked_by = no_event;
    std::size_t first_child = no_event;
    std::size_t last_child = no_event;
    std::size_t previous_sibling = no_event;
    std::size_t next_sibling = no_event;
    std::size_t first_blocked_waiter = no_event;
    std::size_t last_blocked_waiter = no_event;
    std::size_t next_blocked_waiter = no_event;
    std::size_t previous_ready = no_event;
    std::size_t next_ready = no_event;
    std::size_t pending_children = 0;
    std::size_t sequence = 0;
    bool occupied = false;
    bool queued = false;
    bool running = false;
    bool self_done = false;
    bool wait_for_siblings = false;
    bool has_continuation = false;
  };

  std::vector<slot> slots;
  std::vector<std::size_t> free_slots;
  std::vector<int32_t> dependency_remaining;
  std::vector<std::size_t> root_nodes;
  std::vector<uint8_t> root_done;
  // Edge deliveries a pooled node received in this record.
  std::vector<std::size_t> deliveries_seen;
  // Producer outputs copied at publication, indexed by compiled_wire::id.
  std::vector<edge_value> wire_values;
  std::vector<uint8_t> wire_valid;
  // Node state captured at publication, indexed by streaming node.
  std::vector<std::vector<uint8_t>> node_corrections;
  std::vector<std::optional<streaming_task_graph_node::committed_coverage>>
      node_coverage;
  std::vector<uint8_t> node_snapshotted;
  // Set when node_corrections holds a delta to XOR in rather than a copy.
  std::vector<uint8_t> node_delta;
  // Deltas of staged nodes, each with the commit interval it covers, in the
  // order their fires finished; folded in round order when the record retires.
  struct staged_delta {
    streaming_task_graph_node::committed_coverage coverage;
    std::vector<uint8_t> bits;
  };
  std::vector<std::vector<staged_delta>> staged;
  // Replica of each pooled node leased to this record, or no_event.
  std::vector<std::size_t> pool_replica;
  std::vector<std::uint32_t> node_published;
  // Release accounting, indexed by streaming node: live events targeting
  // the node and whether this record has handed the node on.
  std::vector<std::uint32_t> node_live;
  std::vector<uint8_t> node_released;
  std::size_t occupied = 0;
  std::size_t next_sequence = 0;
  std::size_t ready_head = no_event;
  std::size_t ready_tail = no_event;
  std::atomic<bool> fired{false};
  bool complete = false;
  bool gated = false;
  bool in_use = false;
  uint64_t sequence = 0;
  phase kind = phase::idle;
  // Record payload.
  const uint8_t *syndrome = nullptr;
  std::size_t len = 0;
  std::optional<round_type> type;
  const edge_value *measurement = nullptr;
  // Asynchronous record bookkeeping: which public operation this context
  // belongs to and whether retiring it completes that operation.
  uint64_t ring_sequence = 0;
  bool closes_record = false;
  bool finish_flush = false;
};

// One bounded event store per record context drives every execution mode.
// Event slots remain alive until their causal subtree drains, so a later
// reconvergent branch can wait for an earlier branch without delaying work in
// an unrelated component. Across contexts, a node runs records in sequence
// order: a context releases a node once it can create no more events for it,
// which hands the node's turn to the next record.
struct decoding_task_graph::impl::record_scheduler {
  using context = record_context;
  using phase = record_context::phase;
  using slot = record_context::slot;
  struct join_entry;

  record_scheduler(impl &owner, std::size_t context_count,
                   std::size_t event_capacity)
      : owner(owner), contexts(context_count), node_busy(owner.nodes.size(), 0),
        node_turn(owner.streaming_nodes.size(), 0),
        pool_free(owner.streaming_nodes.size()),
        replica_busy(owner.streaming_nodes.size()),
        joins(owner.streaming_nodes.size()),
        join_offsets(owner.streaming_nodes.size()),
        join_required(owner.streaming_nodes.size()),
        dropped(owner.streaming_nodes.size()),
        wire_last(owner.streaming_value_wire_count, no_index),
        fire_inputs(owner.streaming_nodes.size()),
        has_waiting_workers(owner.parallel_pool->num_threads() > 1) {
    const auto N = owner.nodes.size();
    const auto M = owner.streaming_nodes.size();
    // Join leads, in topological order: how far past the newest published
    // index an indexed node can fire (f) or have an entry created (e).
    std::vector<std::int64_t> fire_lead(M, 0), entry_lead(M, 0);
    for (std::size_t si = 0; si < M; ++si) {
      const auto copies = owner.streaming_nodes[si].replicas.size();
      const bool indexed = this->indexed(si);
      if (copies <= 1 && !indexed)
        continue;
      copy_outputs = true;
      for (std::size_t r = copies; r-- > 0;)
        pool_free[si].push_back(r);
      replica_busy[si].assign(copies, 0);
      if (!indexed)
        continue;
      const auto node_idx = owner.streaming_nodes[si].node_index;
      const auto inputs = owner.node_input_counts[node_idx];
      join_offsets[si].assign(inputs, 0);
      join_required[si].assign(inputs, 0);
      for (std::size_t i = 0; i < inputs; ++i)
        join_required[si][i] =
            owner.streaming_nodes[si].indexed.front()->requires_input(i);
      std::int64_t f = std::numeric_limits<std::int64_t>::max(), e = 0;
      for (const auto &wire : owner.wires_to[node_idx]) {
        join_offsets[si][static_cast<std::size_t>(wire.to_slot)] =
            wire.index_offset;
        // A loop-carried input publishes instance k + offset ahead of its
        // consumer's k, from the consumer's own fires; it does not bound how
        // far the consumer fires.
        if (wire.loop_carried())
          e = std::max(e, -wire.index_offset);
      }
      for (const auto producer : owner.streaming_producers[si]) {
        const auto from = owner.streaming_nodes[producer].node_index;
        for (const auto *wires : {&owner.streaming_wires_from[from],
                                  &owner.completion_wires_from[from]})
          for (const auto &wire : *wires)
            if (wire.to_node == node_idx && !wire.loop_carried()) {
              const auto lead = fire_lead[producer] - wire.index_offset;
              f = std::min(f, lead);
              e = std::max(e, lead);
            }
      }
      fire_lead[si] = f == std::numeric_limits<std::int64_t>::max() ? 0 : f;
      entry_lead[si] = e;
      // Entries exist only for published values or in-flight producer
      // outputs: the records in flight bound how far publication runs ahead.
      const auto capacity =
          context_count + static_cast<std::size_t>(e) + copies;
      joins[si].clear();
      for (std::size_t i = 0; i < capacity; ++i)
        add_join_entry(si);
      fire_inputs[si].assign(copies,
                             std::vector<const edge_value *>(
                                 owner.node_input_counts[node_idx], nullptr));
    }
    for (std::size_t k = 0; k < owner.output_stagers.size(); ++k) {
      auto &stager = owner.output_stagers[k];
      if (!stager.enabled)
        continue;
      // Every pending interval is a published delta of a record that has
      // not retired, or of an entry its join table held for one.
      std::size_t capacity = 1;
      for (const auto si : owner.output_streaming_nodes[k])
        capacity += context_count + (this->indexed(si) ? joins[si].size() : 0) +
                    owner.streaming_nodes[si].replicas.size();
      stager.capacity = capacity;
      stager.pending.reserve(capacity);
    }
    if (context_count > 1)
      copy_outputs = true;
    node_deliveries_seen.assign(N, 0);
    for (auto &c : contexts) {
      c.slots.resize(event_capacity);
      c.free_slots.reserve(event_capacity);
      c.root_nodes.reserve(M);
      c.root_done.reserve(M);
      c.deliveries_seen.assign(N, 0);
      c.wire_values.resize(owner.streaming_value_wire_count);
      c.wire_valid.assign(owner.streaming_value_wire_count, 0);
      c.node_corrections.resize(M);
      for (std::size_t si = 0; si < M; ++si)
        c.node_corrections[si].assign(
            owner.streaming_nodes[si].capability->num_observables(), 0);
      c.node_coverage.assign(M, std::nullopt);
      c.node_snapshotted.assign(M, 0);
      c.node_delta.assign(M, 0);
      c.staged.assign(M, {});
      c.pool_replica.assign(M, no_event);
      c.node_published.assign(M, 0);
      c.node_live.assign(M, 0);
      c.node_released.assign(M, 0);
      for (std::size_t i = event_capacity; i-- > 0;)
        c.free_slots.push_back(i);
    }
    active.reserve(context_count);
  }

  // ----- Context lifecycle -------------------------------------------------

  context *try_acquire_locked() {
    for (auto &c : contexts)
      if (!c.in_use) {
        c.in_use = true;
        reset(c);
        return &c;
      }
    return nullptr;
  }

  context &acquire() {
    auto lock = lock_runtime();
    auto *c = try_acquire_locked();
    if (!c)
      throw std::logic_error("decoding_task_graph: no free record context");
    return *c;
  }

  void reset(context &c) {
    // Successful walks retire and clear every occupied slot. Rebuild the free
    // list only after failure leaves queued events behind; the common path
    // therefore does no work proportional to the conservative capacity.
    if (c.occupied != 0 || c.free_slots.size() != c.slots.size()) {
      c.free_slots.clear();
      for (std::size_t i = c.slots.size(); i-- > 0;) {
        c.slots[i] = {};
        c.free_slots.push_back(i);
      }
    }
    c.root_nodes.clear();
    c.root_done.clear();
    std::fill(c.deliveries_seen.begin(), c.deliveries_seen.end(), 0);
    std::fill(c.wire_valid.begin(), c.wire_valid.end(), 0);
    std::fill(c.node_snapshotted.begin(), c.node_snapshotted.end(), 0);
    std::fill(c.node_delta.begin(), c.node_delta.end(), 0);
    for (auto &deltas : c.staged)
      deltas.clear();
    for (std::size_t si = 0; si < c.node_corrections.size(); ++si)
      if (pooled(si) || indexed(si))
        std::fill(c.node_corrections[si].begin(), c.node_corrections[si].end(),
                  uint8_t{0});
    std::fill(c.pool_replica.begin(), c.pool_replica.end(), no_event);
    std::fill(c.node_published.begin(), c.node_published.end(), 0);
    std::fill(c.node_live.begin(), c.node_live.end(), 0);
    std::fill(c.node_released.begin(), c.node_released.end(), 0);
    c.occupied = 0;
    c.next_sequence = 0;
    c.ready_head = no_event;
    c.ready_tail = no_event;
    c.fired.store(false, std::memory_order_relaxed);
    c.complete = false;
    c.gated = false;
    c.sequence = 0;
    c.kind = phase::idle;
    c.syndrome = nullptr;
    c.len = 0;
    c.type.reset();
    c.measurement = nullptr;
    c.ring_sequence = 0;
    c.closes_record = false;
    c.finish_flush = false;
  }

  void release_locked(context &c) {
    active.erase(std::remove(active.begin(), active.end(), &c), active.end());
    c.in_use = false;
    c.kind = phase::idle;
  }

  void release(context &c) {
    auto lock = lock_runtime();
    release_locked(c);
  }

  // Drop every context after a cancelled or failed session.
  void discard_all() {
    auto lock = lock_runtime();
    active.clear();
    for (auto &c : contexts) {
      c.in_use = false;
      reset(c);
    }
    std::fill(node_busy.begin(), node_busy.end(), uint8_t{0});
    for (std::size_t si = 0; si < pool_free.size(); ++si)
      if (pooled(si) || indexed(si)) {
        pool_free[si].clear();
        for (std::size_t r = owner.streaming_nodes[si].replicas.size();
             r-- > 0;)
          pool_free[si].push_back(r);
        std::fill(replica_busy[si].begin(), replica_busy[si].end(), 0);
      }
    for (auto &table : joins)
      for (auto &entry : table)
        entry.used = false;
    for (auto &instances : dropped)
      instances.clear();
    std::fill(wire_last.begin(), wire_last.end(), no_index);
    std::fill(node_deliveries_seen.begin(), node_deliveries_seen.end(), 0);
    stop = false;
    error = nullptr;
  }

  // Begin a streaming session: records are numbered from zero and every node
  // waits for record zero first.
  void begin_session() {
    auto lock = lock_runtime();
    std::fill(node_turn.begin(), node_turn.end(), uint64_t{0});
    next_record = 0;
    session_done = false;
    stop = false;
    error = nullptr;
  }

  // ----- Event population --------------------------------------------------

  void push_raw_roots(context &c) {
    auto lock = lock_runtime();
    for (std::size_t node = 0; node < owner.streaming_nodes.size(); ++node)
      if (owner.streaming_node_has_raw_input[node])
        c.root_nodes.push_back(node);
    c.root_done.resize(c.root_nodes.size());
    std::fill(c.root_done.begin(), c.root_done.end(), uint8_t{0});
    for (std::size_t root = 0; root < c.root_nodes.size(); ++root)
      push_locked(
          c,
          {graph_event_kind::streaming_raw, c.root_nodes[root], {}, 0, root});
  }

  void configure_dependencies(context &c, const std::vector<int32_t> &counts) {
    auto lock = lock_runtime();
    c.dependency_remaining = counts;
  }

  void configure_flush(context &c, const std::vector<int32_t> &counts) {
    auto lock = lock_runtime();
    c.dependency_remaining = counts;
    for (std::size_t node = 0; node < counts.size(); ++node)
      if (counts[node] == 0)
        c.root_nodes.push_back(node);
    c.root_done.resize(c.root_nodes.size());
    std::fill(c.root_done.begin(), c.root_done.end(), uint8_t{0});
    for (std::size_t root = 0; root < c.root_nodes.size(); ++root)
      push_locked(
          c,
          {graph_event_kind::streaming_flush, c.root_nodes[root], {}, 0, root});
  }

  // Make a populated context eligible for execution. Nodes this record never
  // touches are handed on immediately so later records need not wait.
  void activate(context &c, bool gated) {
    auto lock = lock_runtime();
    c.gated = gated;
    if (gated) {
      c.sequence = next_record++;
      for (std::size_t si = 0; si < owner.streaming_nodes.size(); ++si)
        try_release_locked(c, si);
    }
    active.push_back(&c);
    if (c.occupied == 0)
      mark_complete_locked(c);
    notify_workers();
  }

  std::size_t push(context &c, graph_event event, std::size_t parent = no_event,
                   std::size_t blocked_by = no_event,
                   bool wait_for_siblings = false) {
    auto lock = lock_runtime();
    const auto id =
        push_locked(c, event, parent, blocked_by, wait_for_siblings);
    notify_workers();
    return id;
  }

  void release_dependencies(context &c, const std::vector<std::size_t> &nodes,
                            graph_event_kind kind) {
    auto lock = lock_runtime();
    if (stop)
      return;
    bool made_ready = false;
    for (const auto node : nodes) {
      if (node >= c.dependency_remaining.size() ||
          c.dependency_remaining[node] <= 0) {
        throw std::logic_error(
            "decoding_task_graph: event dependency underflow");
      }
      if (--c.dependency_remaining[node] == 0) {
        push_locked(c, {kind, node});
        made_ready = true;
      }
    }
    if (made_ready)
      notify_workers();
  }

  void set_continuation(context &c, std::size_t id, graph_event event) {
    auto lock = lock_runtime();
    if (stop || id >= c.slots.size() || !c.slots[id].occupied)
      return;
    if (c.slots[id].has_continuation) {
      throw std::logic_error(
          "decoding_task_graph: event already has a continuation");
    }
    c.slots[id].continuation = event;
    c.slots[id].has_continuation = true;
  }

  void require_dependencies_complete(context &c, const char *message) {
    auto lock = lock_runtime();
    if (std::any_of(c.dependency_remaining.begin(),
                    c.dependency_remaining.end(),
                    [](int32_t count) { return count != 0; }))
      throw std::logic_error(message);
  }

  // ----- Streaming publication --------------------------------------------

  // Capture what a node exposes after it ran, while this record still holds
  // the node. Retirement folds the capture into the graph-level result in
  // record order, so a node already serving a later record cannot leak into
  // an earlier record's publication.
  void snapshot_node(context &c, std::size_t si,
                     const streaming_task_graph_node *capability = nullptr) {
    if (!capability)
      capability = owner.streaming_nodes[si].capability;
    if (staged(si)) {
      // Each publication is one delta over one commit interval; the record
      // hands them to the output stagers when it retires.
      const auto coverage = capability->streaming_coverage();
      if (!coverage)
        return; // nothing to fold from this publication
      const auto n = capability->num_observables();
      const uint8_t *delta = capability->corrections();
      {
        auto lock = lock_runtime();
        auto &entry = c.staged[si].emplace_back();
        entry.coverage = *coverage;
        if (delta)
          entry.bits.assign(delta, delta + n);
        else
          entry.bits.assign(n, 0);
        c.node_snapshotted[si] = 1;
      }
      // A replica is cleared before each fire; a record-ordered node is
      // cleared here so its next publication is a delta as well.
      if (!pooled(si) && !indexed(si))
        owner.streaming_nodes[si].capability->clear_corrections();
      return;
    }
    if (pooled(si) || indexed(si)) {
      // The replica's corrections were cleared before this delivery, so they
      // hold only this fire's delta. Deliveries of one record to a pooled node
      // may overlap on different replicas, so fold under the lock.
      auto lock = lock_runtime();
      auto &dst = c.node_corrections[si];
      if (const uint8_t *delta = capability->corrections())
        for (std::size_t i = 0; i < dst.size(); ++i)
          dst[i] ^= delta[i];
      c.node_delta[si] = 1;
      c.node_snapshotted[si] = 1;
      return;
    }
    auto &dst = c.node_corrections[si];
    if (!dst.empty())
      if (const uint8_t *src = capability->corrections())
        std::copy(src, src + dst.size(), dst.begin());
    c.node_coverage[si] = capability->streaming_coverage();
    c.node_snapshotted[si] = 1;
  }

  // A fire leaves a publication current; publish it and each further one
  // the node makes current. Only the first may carry the fire's own index.
  bool publish_all(context &c, std::size_t parent, std::size_t root,
                   std::size_t producer,
                   streaming_task_graph_node *capability = nullptr,
                   std::int64_t index = no_index) {
    if (!capability)
      capability = owner.streaming_nodes[producer].capability;
    bool repeat = false;
    do {
      publish_streaming(c, parent, root, producer, capability,
                        repeat ? no_index : index, repeat);
      repeat = true;
    } while (capability->publish_next());
    return true;
  }

  bool publish_streaming(context &c, std::size_t parent, std::size_t root,
                         std::size_t producer,
                         const streaming_task_graph_node *capability = nullptr,
                         std::int64_t index = no_index, bool repeat = false) {
    const auto node_idx = owner.streaming_nodes[producer].node_index;
    if (!capability)
      capability = owner.streaming_nodes[producer].capability;
    // A record-ordered consumer reads the wire's one buffer when its delivery
    // runs, so a second publication before then would replace the first.
    if (repeat)
      for (const auto &wire : owner.streaming_wires_from[node_idx])
        if (!indexed(static_cast<std::size_t>(
                owner.streaming_by_node[wire.to_node])) &&
            capability->streaming_output(wire.from_slot))
          throw std::logic_error(
              "decoding_task_graph: node " + std::to_string(node_idx) +
              " published twice in one delivery on output " +
              std::to_string(wire.from_slot) +
              ", whose consumer joins its "
              "inputs by record; only an indexed consumer takes repeat "
              "publications");
    snapshot_node(c, producer, capability);
    {
      auto lock = lock_runtime();
      ++c.node_published[producer];
    }
    // Copy outputs now: a later record may run the producer before the
    // consumer's delivery event executes. With one context none can, so the
    // consumer reads the output in place instead; a graph with a fire pool
    // always copies. The copy is made under the lock that publishes it:
    // copies of a pooled indexed node may fire for several indices of one
    // record at once, and share the context's one value per wire.
    const auto &wires = owner.streaming_wires_from[node_idx];
    {
      auto lock = lock_runtime();
      if (copy_outputs)
        for (const auto &wire : wires) {
          const auto *value = capability->streaming_output(wire.from_slot);
          const auto wire_id = static_cast<std::size_t>(wire.id);
          if (value)
            c.wire_values[wire_id] = *value;
          c.wire_valid[wire_id] = value != nullptr;
        }
      // Each port carries its own index, unless the fire named one for all
      // of them. A null port says nothing to an indexed consumer this time.
      for (const auto &wire : wires) {
        const auto consumer =
            static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
        if (indexed(consumer) && !capability->streaming_output(wire.from_slot))
          continue;
        const auto wire_index =
            index != no_index ? index
                              : capability->published_index(wire.from_slot)
                                    .value_or(no_index);
        publish_wire_locked(c, parent, root, wire, wire_index);
      }
      publish_wires_locked(c, parent, root,
                           owner.completion_wires_from[node_idx], index);
    }
    notify_workers();
    return true;
  }

  void publish_wires(context &c, std::size_t parent, std::size_t root,
                     const std::vector<compiled_wire> &wires,
                     std::int64_t index = no_index) {
    if (wires.empty())
      return;
    auto lock = lock_runtime();
    publish_wires_locked(c, parent, root, wires, index);
    notify_workers();
  }

  void publish_wires_locked(context &c, std::size_t parent, std::size_t root,
                            const std::vector<compiled_wire> &wires,
                            std::int64_t index, bool absent = false) {
    for (const auto &wire : wires)
      publish_wire_locked(c, parent, root, wire, index, absent);
  }

  // With `absent`, the publication says instance `index` of the producer
  // never exists: it carries no value and goes only to indexed consumers,
  // which file the absence under the instances reading that index.
  void publish_wire_locked(context &c, std::size_t parent, std::size_t root,
                           const compiled_wire &wire, std::int64_t index,
                           bool absent = false) {
    {
      const auto consumer =
          static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
      if (absent && !indexed(consumer))
        return;
      // A record-ordered producer publishes its indices in order, so the
      // ones between its last publication and this one never come: announce
      // them, from the first the consumer reads, before the value itself.
      if (!absent && index != no_index && indexed(consumer) && wire.id >= 0) {
        const auto producer =
            static_cast<std::size_t>(owner.streaming_by_node[wire.from_node]);
        auto &last = wire_last[static_cast<std::size_t>(wire.id)];
        if (!pooled(producer) && !indexed(producer)) {
          const auto from = last == no_index ? wire.index_offset : last + 1;
          for (auto skipped = from; skipped < index; ++skipped)
            publish_wire_locked(c, parent, root, wire, skipped,
                                /*absent=*/true);
          last = index;
        }
      }
      std::size_t blocker = no_event;
      std::size_t blocker_sequence = 0;
      // Direct children are stored in declared wire order. A later branch
      // waits only when its transitive descendants intersect an earlier
      // branch; disjoint fan-out remains immediately runnable. An indexed
      // consumer accepts its inputs in any order, so it never waits.
      for (std::size_t earlier = c.slots[parent].first_child;
           earlier != no_event && !indexed(consumer);
           earlier = c.slots[earlier].next_sibling) {
        const auto &candidate = c.slots[earlier];
        if (candidate.event.kind != graph_event_kind::streaming_edge)
          continue;
        if (branches_overlap(candidate.event.node, consumer) &&
            (blocker == no_event || candidate.sequence > blocker_sequence)) {
          blocker = earlier;
          blocker_sequence = candidate.sequence;
        }
      }
      graph_event event{graph_event_kind::streaming_edge, consumer, wire, 0,
                        root};
      event.index = index;
      event.absent = absent;
      const auto id = push_locked(c, event, parent, blocker);
      // Several instances may publish through one wire before any delivery
      // runs, so an indexed delivery takes the record's copy of the value.
      // The swap hands the wire a buffer of the same type for the next one.
      const auto wire_id = static_cast<std::size_t>(wire.id);
      if (id != no_event && !absent && indexed(consumer) && wire.id >= 0 &&
          c.wire_valid[wire_id]) {
        std::swap(c.slots[id].value, c.wire_values[wire_id]);
        c.slots[id].has_value = true;
        c.wire_valid[wire_id] = 0;
      }
    }
  }

  const edge_value *delivered_value(const context &c,
                                    const graph_event &event) const {
    const auto from = static_cast<std::size_t>(event.wire.from_node);
    const auto from_slot = static_cast<std::size_t>(event.wire.from_slot);
    // The implicit `done` output is the constant completion marker.
    const auto *marker = &owner.node_outputs[from][from_slot];
    if (marker->get_if<completion>())
      return marker;
    if (!copy_outputs)
      return owner.streaming_nodes[owner.streaming_by_node[from]]
          .capability->streaming_output(from_slot);
    if (event.wire.id >= 0 &&
        c.wire_valid[static_cast<std::size_t>(event.wire.id)])
      return &c.wire_values[static_cast<std::size_t>(event.wire.id)];
    return nullptr;
  }

  // File one delivery under the consumer instance it addresses. The entry is
  // complete once every input is present or known to be absent; the fire it
  // queues belongs to the record whose delivery completed it.
  void deliver_indexed(context &c, std::size_t id, const graph_event &event) {
    const auto si = event.node;
    const auto &node = owner.streaming_nodes[si];
    if (event.index == no_index)
      throw std::logic_error(
          "decoding_task_graph: node " + std::to_string(node.node_index) +
          " joins its inputs by index but received an unindexed value from "
          "node " +
          std::to_string(event.wire.from_node));
    const auto k = event.index - event.wire.index_offset;
    const auto slot = static_cast<std::size_t>(event.wire.to_slot);
    // Any copy is made before taking the lock; the entry takes the
    // delivery's value by swap below.
    auto &own = c.slots[id];
    if (!event.absent && !own.has_value)
      if (const auto *value = delivered_value(c, event)) {
        own.value = *value;
        own.has_value = true;
      }
    auto lock = lock_runtime();
    if (k < 0 || is_dropped(si, k))
      return; // instances are numbered from 0, and a dropped one stays so
    auto &table = joins[si];
    std::size_t index = table.size();
    for (std::size_t i = 0; i < table.size() && index == table.size(); ++i)
      if (table[i].used && table[i].k == k)
        index = i;
    const bool fresh = index == table.size();
    if (fresh) {
      for (std::size_t i = 0; i < table.size() && index == table.size(); ++i)
        if (!table[i].used)
          index = i;
      // The preallocation covers one publication per producer per record;
      // a record carrying several rounds can publish more, so grow instead
      // of failing. A deque keeps entries in use where they are.
      if (index == table.size())
        add_join_entry(si);
    }
    auto *entry = &table[index];
    if (fresh) {
      entry->used = true;
      entry->k = k;
      entry->pending = 0;
      for (std::size_t i = 0; i < entry->state.size(); ++i) {
        // A producer index below 0 never publishes.
        const bool absent = k + join_offsets[si][i] < 0;
        if (absent && join_required[si][i]) {
          entry->used = false;
          drop_instance_locked(c, id, si, k, event.root);
          return; // an instance without a required input does not exist
        }
        entry->state[i] = absent ? join_entry::absent : join_entry::missing;
        entry->pending += !absent;
      }
    }
    if (entry->state[slot] == join_entry::present)
      throw std::logic_error("decoding_task_graph: node " +
                             std::to_string(node.node_index) + " input " +
                             std::to_string(slot) + " of instance " +
                             std::to_string(k) + " was delivered twice");
    if (entry->state[slot] == join_entry::missing)
      --entry->pending;
    if (own.has_value) {
      std::swap(entry->inputs[slot], own.value);
      own.has_value = false;
      entry->state[slot] = join_entry::present;
    } else {
      entry->state[slot] = join_entry::absent;
      if (join_required[si][slot]) {
        entry->used = false;
        drop_instance_locked(c, id, si, k, event.root);
        return;
      }
    }
    if (entry->pending != 0)
      return;
    queue_indexed_fire_locked(c, id, si, index, event.root);
    notify_workers();
  }

  bool is_dropped(std::size_t si, std::int64_t k) const {
    const auto &instances = dropped[si];
    return std::find(instances.begin(), instances.end(), k) != instances.end();
  }

  // Instance `k` of node `si` never exists: tell its indexed consumers once,
  // so the instances reading it stop waiting for it.
  void drop_instance_locked(context &c, std::size_t parent, std::size_t si,
                            std::int64_t k, std::size_t root) {
    if (is_dropped(si, k))
      return;
    dropped[si].push_back(k);
    const auto node_idx = owner.streaming_nodes[si].node_index;
    publish_wires_locked(c, parent, root, owner.streaming_wires_from[node_idx],
                         k, /*absent=*/true);
    notify_workers();
  }

  void queue_indexed_fire_locked(context &c, std::size_t parent, std::size_t si,
                                 std::size_t entry, std::size_t root) {
    graph_event fire{graph_event_kind::streaming_indexed_fire, si, {}, 0, root};
    fire.index = joins[si][entry].k;
    fire.entry = entry;
    push_locked(c, fire, parent);
  }

  // The shot has ended and every producer of the node has finished, so no
  // pending input will arrive: instances with their required inputs fire
  // with the rest absent, and the others never existed.
  // The table may have outgrown the event store, so stop once half of it is
  // queued and report that a later flush has to continue; true when so.
  bool resolve_joins_locked(context &c, std::size_t parent, std::size_t si,
                            std::size_t root) {
    auto &table = joins[si];
    for (std::size_t index = 0; index < table.size(); ++index) {
      auto &entry = table[index];
      if (!entry.used || entry.pending == 0)
        continue;
      if (c.free_slots.size() <= c.slots.size() / 2)
        return true;
      bool exists = true;
      for (std::size_t i = 0; i < entry.state.size(); ++i)
        if (entry.state[i] == join_entry::missing && join_required[si][i])
          exists = false;
      if (!exists) {
        entry.used = false;
        continue;
      }
      for (auto &state : entry.state)
        if (state == join_entry::missing)
          state = join_entry::absent;
      entry.pending = 0;
      queue_indexed_fire_locked(c, parent, si, index, root);
    }
    return false;
  }

  // Whether no event of any record other than `self`'s flush still targets
  // the pooled or indexed node: its producers' deliveries have all been filed
  // or run and its own fires have all finished.
  bool quiescent_locked(const context &self, std::size_t si) const {
    for (const auto busy : replica_busy[si])
      if (busy)
        return false;
    for (const auto *c : active) {
      auto live = c->node_live[si];
      if (c == &self)
        --live; // the flush event asking
      if (live != 0)
        return false;
    }
    return true;
  }

  // Run instance k on the replica claimed at dispatch, with the entry's
  // inputs, then free the entry.
  bool execute_indexed_fire(context &c, std::size_t id,
                            const graph_event &event) {
    const auto si = event.node;
    const auto &node = owner.streaming_nodes[si];
    const auto replica = c.slots[id].pool_slot;
    join_entry *entry_ptr;
    {
      // Another record's delivery may be growing the deque.
      auto lock = lock_runtime();
      entry_ptr = &joins[si][event.entry];
    }
    auto &entry = *entry_ptr;
    auto &inputs = fire_inputs[si][replica];
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] =
          entry.state[i] == join_entry::present ? &entry.inputs[i] : nullptr;
    auto *capability = node.replicas[replica];
    capability->clear_corrections(); // its corrections become this delta
    const bool fired = node.indexed[replica]->fire(event.index, inputs);
    {
      auto lock = lock_runtime();
      entry.used = false;
    }
    if (fired)
      (void)publish_all(c, id, event.root, si, capability, event.index);
    else
      publish_wires(c, id, event.root,
                    owner.completion_wires_from[node.node_index], event.index);
    return fired;
  }

  bool deliver_streaming_edge(context &c, std::size_t id,
                              const graph_event &event) {
    if (indexed(event.node)) {
      deliver_indexed(c, id, event);
      return false;
    }
    const auto consumer =
        static_cast<std::size_t>(owner.streaming_nodes[event.node].node_index);
    const bool pool = pooled(event.node);
    // A pooled consumer is the replica this record leased at its first
    // delivery. Clearing first makes its corrections this fire's delta.
    auto *consumer_node = pool ? owner.streaming_nodes[event.node]
                                     .replicas[c.pool_replica[event.node]]
                               : owner.streaming_nodes[event.node].capability;
    if (pool)
      consumer_node->clear_corrections();
    const auto *value = delivered_value(c, event);
    const bool fired =
        value && consumer_node->enqueue_edge_value(*value, event.wire.to_slot);
    if (fired)
      (void)publish_all(c, id, event.root, event.node, consumer_node);
    // An edge-fed node completes the window after its last delivery. A
    // pooled node counts per record; a record-ordered one counts across
    // records, as its producers may fire in different ones.
    auto &seen =
        pool ? c.deliveries_seen[consumer] : node_deliveries_seen[consumer];
    if (++seen == owner.streaming_wires_to_count[consumer]) {
      seen = 0;
      if (pool)
        c.slots[id].releases_pool = true;
      if (!fired)
        publish_wires(c, id, event.root, owner.completion_wires_from[consumer]);
    }
    return fired;
  }

  // ----- Event handlers ----------------------------------------------------

  void execute(context &c, std::size_t id, const graph_event &event) {
    switch (event.kind) {
    case graph_event_kind::one_shot:
      execute_one_shot(c, event);
      return;
    case graph_event_kind::streaming_raw:
      execute_raw(c, id, event);
      return;
    case graph_event_kind::streaming_edge:
      if (deliver_streaming_edge(c, id, event))
        c.fired.store(true, std::memory_order_relaxed);
      return;
    case graph_event_kind::streaming_flush:
      execute_flush(c, id, event);
      return;
    case graph_event_kind::streaming_flush_release:
      release_dependencies(c, owner.streaming_dependents[event.node],
                           graph_event_kind::streaming_flush);
      return;
    case graph_event_kind::streaming_indexed_fire:
      if (execute_indexed_fire(c, id, event))
        c.fired.store(true, std::memory_order_relaxed);
      return;
    }
    throw std::logic_error("decoding_task_graph: unexpected graph event");
  }

  // The final predecessor publishes a dependent while holding the runtime
  // lock, so all predecessor output writes happen before the dependent is
  // dispatched.
  void execute_one_shot(context &c, const graph_event &event) {
    const auto node = event.node;
    auto &inputs = owner.parallel_node_inputs[node];
    for (const auto &wire : owner.wires_to[node])
      inputs[static_cast<std::size_t>(wire.to_slot)] =
          wire.from_node < 0
              ? *c.measurement
              : owner.node_outputs[static_cast<std::size_t>(wire.from_node)]
                                  [static_cast<std::size_t>(wire.from_slot)];
    owner.nodes[node]->execute_indexed(inputs, owner.node_outputs[node]);
    release_dependencies(c, owner.dependents[node], graph_event_kind::one_shot);
  }

  void execute_raw(context &c, std::size_t id, const graph_event &event) {
    auto &node = *owner.streaming_nodes[event.node].capability;
    const auto limit =
        c.type ? node.raw_chunk_limit() : node.untagged_chunk_limit();
    const auto take =
        limit ? std::min(*limit, c.len - event.offset) : c.len - event.offset;
    const auto *chunk = c.syndrome ? c.syndrome + event.offset : nullptr;
    const auto next = event.offset + take;
    const auto chunk_type = c.type == round_type::FINAL && next == c.len
                                ? round_type::FINAL
                                : round_type::BULK;
    const bool fired = c.type ? node.enqueue(chunk, take, chunk_type)
                              : node.enqueue_untagged(chunk, take);
    if (fired)
      c.fired.store(publish_all(c, id, event.root, event.node),
                    std::memory_order_relaxed);
    else if (chunk_type == round_type::FINAL && !staged(event.node))
      snapshot_node(c, event.node); // a FINAL round may commit coverage
    if (next < c.len && take != 0)
      set_continuation(
          c, id,
          {graph_event_kind::streaming_raw, event.node, {}, next, event.root});
  }

  void execute_flush(context &c, std::size_t id, const graph_event &event) {
    if (indexed(event.node)) {
      bool more;
      {
        auto lock = lock_runtime();
        more = resolve_joins_locked(c, id, event.node, event.root);
      }
      if (more) { // continue once the queued fires have drained
        push(c, event, id, no_event, /*wait_for_siblings=*/true);
        return;
      }
      // Every instance of the shot is resolved; the next shot's numbering
      // starts over.
      auto lock = lock_runtime();
      dropped[event.node].clear();
    }
    if (owner.streaming_nodes[event.node].capability->flush())
      c.fired.store(publish_all(c, id, event.root, event.node),
                    std::memory_order_relaxed);
    else if (!pooled(event.node) && !indexed(event.node) && !staged(event.node))
      snapshot_node(c, event.node); // a pooled copy holds only its last delta
    {
      // The flush was the shot's last publication on the node's wires; the
      // next shot's indices start a new run.
      auto lock = lock_runtime();
      const auto node_idx = owner.streaming_nodes[event.node].node_index;
      for (const auto &wire : owner.streaming_wires_from[node_idx])
        if (wire.id >= 0)
          wire_last[static_cast<std::size_t>(wire.id)] = no_index;
    }
    push(c, {graph_event_kind::streaming_flush_release, event.node}, id,
         no_event, /*wait_for_siblings=*/true);
  }

  // ----- Worker loops ------------------------------------------------------

  // Run one context to completion with every pool worker, the caller
  // included. Used by the synchronous streaming and one-shot paths.
  void run_single(context &c, int cuda_device_id) {
    owner.parallel_pool->run_workers([&](std::size_t /*worker_id*/) {
      execute_loop(cuda_device_id, [&] { return c.occupied == 0; });
    });
    std::exception_ptr first_error;
    {
      auto lock = lock_runtime();
      first_error = error;
    }
    if (first_error)
      std::rethrow_exception(first_error);
  }

  // Run an asynchronous session: worker zero only admits and retires records,
  // every other worker executes node events until the dispatcher ends the
  // session.
  void run_session(int cuda_device_id) {
    spin_while_active = true;
    owner.parallel_pool->run_workers([&](std::size_t worker_id) {
      if (worker_id == 0)
        owner.dispatch_async_session();
      else
        execute_loop(cuda_device_id, [&] { return session_done; });
    });
    spin_while_active = false;
  }

  template <typename Until>
  void execute_loop(int cuda_device_id, Until &&until) {
    context *active_context = nullptr;
    std::size_t active_id = no_event;
    try {
      detail_affinity::CudaDeviceGuard device_guard(cuda_device_id);
      while (true) {
        context *c = nullptr;
        std::size_t id = no_event;
        graph_event event;
        {
          auto lock = lock_runtime();
          while (id == no_event) {
            if (stop) {
              if (running == 0)
                return;
              wake.wait(lock);
              continue;
            }
            if (until())
              return;
            std::tie(c, id) = select_ready_locked();
            if (id == no_event) {
              if (!has_waiting_workers)
                throw std::logic_error(
                    "decoding_task_graph: event queue stalled");
              if (!spin_for_ready(lock))
                wake.wait(lock);
              continue;
            }
          }
          auto &selected = c->slots[id];
          selected.running = true;
          selected.queued = false;
          ++running;
          claim_locked(*c, selected);
          event = selected.event;
        }

        active_context = c;
        active_id = id;
        execute(*c, id, event);
        finish(*c, id);
        active_context = nullptr;
        active_id = no_event;
      }
    } catch (...) {
      fail(active_context, active_id, std::current_exception());
    }
  }

  // ----- Dispatcher support ------------------------------------------------

  std::unique_lock<std::mutex> lock_runtime() {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (has_waiting_workers)
      lock.lock();
    return lock;
  }

  // Oldest active context, if it has finished executing.
  context *oldest_complete_locked() {
    if (active.empty() || !active.front()->complete)
      return nullptr;
    return active.front();
  }

  void request_stop_locked() {
    stop = true;
    notify_workers();
  }

  void wait_until_idle_locked(std::unique_lock<std::mutex> &lock) {
    while (running != 0)
      wake.wait(lock);
  }

  void end_session_locked() {
    session_done = true;
    notify_workers();
  }

  void notify_workers() {
    if (!has_waiting_workers)
      return;
    ready_generation.fetch_add(1, std::memory_order_release);
    wake.notify_all();
  }

private:
  // Spin briefly for a scheduler change before parking. Returns true when one
  // happened, so the caller re-selects instead of waiting.
  bool spin_for_ready(std::unique_lock<std::mutex> &lock) {
    if (!spin_while_active)
      return false;
    const auto seen = ready_generation.load(std::memory_order_acquire);
    lock.unlock();
    bool changed = false;
    for (std::size_t spin = 0; spin < kHandoffSpinPauseIters && !changed;
         ++spin) {
      changed = ready_generation.load(std::memory_order_acquire) != seen;
      if (!changed)
        CUDAQ_REALTIME_CPU_RELAX();
    }
    lock.lock();
    return changed || ready_generation.load(std::memory_order_acquire) != seen;
  }

  std::size_t push_locked(context &c, graph_event event,
                          std::size_t parent = no_event,
                          std::size_t blocked_by = no_event,
                          bool wait_for_siblings = false) {
    if (stop)
      return no_event;
    if (parent != no_event &&
        (parent >= c.slots.size() || !c.slots[parent].occupied)) {
      throw std::logic_error(
          "decoding_task_graph: event parent is no longer active");
    }
    if (blocked_by != no_event &&
        (blocked_by >= c.slots.size() || !c.slots[blocked_by].occupied)) {
      throw std::logic_error(
          "decoding_task_graph: event blocker is no longer active");
    }
    if (c.free_slots.empty())
      throw std::logic_error(
          "decoding_task_graph: bounded event store overflow");
    const auto id = c.free_slots.back();
    c.free_slots.pop_back();
    auto &created = c.slots[id];
    created = {};
    created.event = event;
    created.parent = parent;
    created.blocked_by = blocked_by;
    created.sequence = c.next_sequence++;
    created.wait_for_siblings = wait_for_siblings;
    created.occupied = true;
    ++c.occupied;
    const auto target = release_target(event);
    if (target != no_event)
      ++c.node_live[target];
    if (parent != no_event) {
      auto &ancestor = c.slots[parent];
      created.previous_sibling = ancestor.last_child;
      if (ancestor.last_child == no_event)
        ancestor.first_child = id;
      else
        c.slots[ancestor.last_child].next_sibling = id;
      ancestor.last_child = id;
      ++ancestor.pending_children;
    }
    if (blocked_by != no_event) {
      auto &blocker = c.slots[blocked_by];
      if (blocker.last_blocked_waiter == no_event)
        blocker.first_blocked_waiter = id;
      else
        c.slots[blocker.last_blocked_waiter].next_blocked_waiter = id;
      blocker.last_blocked_waiter = id;
    }
    if (blocked_by == no_event && (!wait_for_siblings || parent == no_event))
      enqueue_locked(c, id);
    return id;
  }

  void enqueue_locked(context &c, std::size_t id) {
    auto &candidate = c.slots[id];
    if (!candidate.occupied || candidate.queued || candidate.running)
      return;
    candidate.queued = true;
    candidate.previous_ready = c.ready_tail;
    candidate.next_ready = no_event;
    if (c.ready_tail == no_event)
      c.ready_head = id;
    else
      c.slots[c.ready_tail].next_ready = id;
    c.ready_tail = id;
  }

  void remove_ready_locked(context &c, std::size_t id) {
    auto &candidate = c.slots[id];
    if (candidate.previous_ready == no_event)
      c.ready_head = candidate.next_ready;
    else
      c.slots[candidate.previous_ready].next_ready = candidate.next_ready;
    if (candidate.next_ready == no_event)
      c.ready_tail = candidate.previous_ready;
    else
      c.slots[candidate.next_ready].previous_ready = candidate.previous_ready;
    candidate.previous_ready = no_event;
    candidate.next_ready = no_event;
  }

  // Topology node whose state the event mutates; it may not run concurrently
  // with another event on the same node.
  std::size_t serialized_node(const graph_event &event) const {
    if (event.kind == graph_event_kind::streaming_flush_release ||
        event.kind == graph_event_kind::streaming_indexed_fire)
      return no_event;
    if (event.kind == graph_event_kind::one_shot)
      return event.node;
    if (event.node >= owner.streaming_nodes.size())
      return no_event;
    // A delivery to a pooled node holds one replica, not the node; one to an
    // indexed node only files a value under the lock.
    if (event.kind == graph_event_kind::streaming_edge &&
        (pooled(event.node) || indexed(event.node)))
      return no_event;
    return static_cast<std::size_t>(
        owner.streaming_nodes[event.node].node_index);
  }

  bool pooled(std::size_t si) const {
    return owner.streaming_nodes[si].replicas.size() > 1;
  }
  bool indexed(std::size_t si) const {
    return !owner.streaming_nodes[si].indexed.empty();
  }
  // A staged node's deltas arrive only with its publications, so a FINAL
  // that does not fire has nothing to snapshot.
  bool staged(std::size_t si) const {
    return si < owner.streaming_node_staged.size() &&
           owner.streaming_node_staged[si];
  }

  // Events whose target keeps no state between fires need no record order.
  bool targets_pool(const graph_event &event) const {
    return event.kind != graph_event_kind::one_shot &&
           event.node < owner.streaming_nodes.size() &&
           (pooled(event.node) || indexed(event.node));
  }

  bool is_pool_delivery(const graph_event &event) const {
    return event.kind == graph_event_kind::streaming_edge &&
           targets_pool(event) && !indexed(event.node);
  }

  // Mark what a dispatched event holds: its node, or one replica of a pooled
  // node. A record leases a replica at its first delivery and keeps it until
  // its last, so a multi-input consumer sees all of a record's inputs on one
  // copy.
  void claim_locked(context &c, slot &selected) {
    if (selected.event.kind == graph_event_kind::streaming_indexed_fire) {
      // Leased per fire, not per record: fires of one node overlap freely.
      const auto si = selected.event.node;
      selected.pool_slot = pool_free[si].back();
      pool_free[si].pop_back();
      replica_busy[si][selected.pool_slot] = 1;
      selected.releases_pool = true;
      return;
    }
    if (is_pool_delivery(selected.event)) {
      const auto si = selected.event.node;
      if (c.pool_replica[si] == no_event) {
        c.pool_replica[si] = pool_free[si].back();
        pool_free[si].pop_back();
      }
      selected.pool_slot = c.pool_replica[si];
      replica_busy[si][selected.pool_slot] = 1;
      return;
    }
    const auto node = serialized_node(selected.event);
    if (node != no_event)
      node_busy[node] = 1;
  }

  void unclaim_locked(context &c, slot &done) {
    if (done.pool_slot != no_event) {
      const auto si = done.event.node;
      replica_busy[si][done.pool_slot] = 0;
      if (done.releases_pool) {
        pool_free[si].push_back(done.pool_slot);
        if (c.pool_replica[si] == done.pool_slot)
          c.pool_replica[si] = no_event;
        done.releases_pool = false;
      }
      done.pool_slot = no_event;
      return;
    }
    const auto node = serialized_node(done.event);
    if (node != no_event)
      node_busy[node] = 0;
  }

  // Streaming node the event counts against for release. Unlike
  // serialized_node this includes flush releases, which create flush events
  // for dependents and so must finish before their node is handed on.
  std::size_t release_target(const graph_event &event) const {
    if (event.kind == graph_event_kind::one_shot ||
        event.node >= owner.streaming_nodes.size())
      return no_event;
    return event.node;
  }

  bool roots_overlap(const context &c, std::size_t earlier,
                     std::size_t later) const {
    if (earlier >= c.root_nodes.size() || later >= c.root_nodes.size())
      return false;
    return branches_overlap(c.root_nodes[earlier], c.root_nodes[later]);
  }

  bool branches_overlap(std::size_t left, std::size_t right) const {
    return owner.streaming_overlap[left][right] != 0;
  }

  bool root_is_ready(const context &c, const graph_event &event,
                     const slot &candidate) const {
    if (event.root == no_event || candidate.parent == no_event)
      return true;
    for (std::size_t root = 0; root < event.root; ++root)
      if (!c.root_done[root] && roots_overlap(c, root, event.root))
        return false;
    return true;
  }

  std::pair<context *, std::size_t> select_ready_locked() {
    for (auto *c : active) {
      const auto id = select_ready_locked(*c);
      if (id != no_event)
        return {c, id};
    }
    return {nullptr, no_event};
  }

  std::size_t select_ready_locked(context &c) {
    for (std::size_t id = c.ready_head; id != no_event;
         id = c.slots[id].next_ready) {
      const auto &candidate = c.slots[id];
      const auto node = serialized_node(candidate.event);
      if (node != no_event && node_busy[node])
        continue;
      if (is_pool_delivery(candidate.event)) {
        // Needs this record's replica to be idle, or a free one to lease.
        const auto si = candidate.event.node;
        const auto leased = c.pool_replica[si];
        if (leased == no_event ? pool_free[si].empty()
                               : replica_busy[si][leased] != 0)
          continue;
      }
      if (candidate.event.kind == graph_event_kind::streaming_indexed_fire &&
          pool_free[candidate.event.node].empty())
        continue;
      // Resolving an indexed node's pending instances needs every delivery
      // to it, from any record, filed first.
      if (candidate.event.kind == graph_event_kind::streaming_flush &&
          targets_pool(candidate.event) &&
          !quiescent_locked(c, candidate.event.node))
        continue;
      // A pooled node keeps no state, so its records need no order.
      if (c.gated && !targets_pool(candidate.event)) {
        const auto target = release_target(candidate.event);
        if (target != no_event && node_turn[target] != c.sequence)
          continue;
      }
      if (!root_is_ready(c, candidate.event, candidate))
        continue;
      remove_ready_locked(c, id);
      return id;
    }
    return no_event;
  }

  // Hand node `si` to the next record once this record can create no more
  // events for it: every producer has been handed on and nothing targeting
  // the node is queued or running.
  void try_release_locked(context &c, std::size_t si) {
    if (!c.gated || c.node_released[si] || c.node_live[si] != 0 ||
        node_turn[si] != c.sequence)
      return;
    for (const auto producer : owner.streaming_producers[si])
      if (!c.node_released[producer])
        return;
    c.node_released[si] = 1;
    node_turn[si] = c.sequence + 1;
    for (const auto dependent : owner.streaming_dependents[si])
      try_release_locked(c, dependent);
    // The next record may have been waiting only for its turn on this node.
    for (auto *next : active)
      if (next->sequence == c.sequence + 1 && next->gated)
        try_release_locked(*next, si);
  }

  void mark_complete_locked(context &c) {
    // A producer that skipped this record leaves a pooled consumer's lease
    // held and its inputs partial; hand the replica back clean.
    for (std::size_t si = 0; si < c.pool_replica.size(); ++si)
      if (c.pool_replica[si] != no_event) {
        const auto replica = c.pool_replica[si];
        owner.streaming_nodes[si].replicas[replica]->reset_streaming_state();
        pool_free[si].push_back(replica);
        c.pool_replica[si] = no_event;
      }
    c.complete = true;
    if (c.gated)
      owner.wake_async_stream();
  }

  void finish(context &c, std::size_t id) {
    auto lock = lock_runtime();
    if (stop) {
      if (id < c.slots.size() && c.slots[id].running) {
        c.slots[id].running = false;
        unclaim_locked(c, c.slots[id]);
      }
      if (running != 0)
        --running;
      notify_workers();
      return;
    }
    auto &completed = c.slots[id];
    unclaim_locked(c, completed);
    completed.running = false;
    completed.self_done = true;
    if (running != 0)
      --running;
    if (!completed.has_continuation) {
      const auto target = release_target(completed.event);
      if (target != no_event && --c.node_live[target] == 0)
        try_release_locked(c, target);
    }
    if (completed.pending_children == 0)
      complete_or_continue_locked(c, id);
    else if (completed.pending_children == 1)
      enqueue_sibling_waiter_locked(c, id);
    notify_workers();
  }

  void complete_or_continue_locked(context &c, std::size_t id) {
    if (!c.slots[id].has_continuation) {
      retire_locked(c, id);
      return;
    }
    auto &continued = c.slots[id];
    continued.event = continued.continuation;
    continued.continuation = {};
    continued.has_continuation = false;
    continued.blocked_by = no_event;
    continued.sequence = c.next_sequence++;
    continued.queued = false;
    continued.running = false;
    continued.self_done = false;
    continued.wait_for_siblings = false;
    enqueue_locked(c, id);
  }

  void retire_locked(context &c, std::size_t id) {
    auto &completed = c.slots[id];
    const auto parent = completed.parent;
    const auto root = completed.event.root;

    // Release deterministic branch successors before this slot is recycled.
    std::size_t waiting = completed.first_blocked_waiter;
    while (waiting != no_event) {
      const auto next = c.slots[waiting].next_blocked_waiter;
      c.slots[waiting].blocked_by = no_event;
      c.slots[waiting].next_blocked_waiter = no_event;
      if (!c.slots[waiting].wait_for_siblings)
        enqueue_locked(c, waiting);
      waiting = next;
    }

    if (parent != no_event) {
      auto &ancestor = c.slots[parent];
      if (completed.previous_sibling == no_event)
        ancestor.first_child = completed.next_sibling;
      else
        c.slots[completed.previous_sibling].next_sibling =
            completed.next_sibling;
      if (completed.next_sibling == no_event)
        ancestor.last_child = completed.previous_sibling;
      else
        c.slots[completed.next_sibling].previous_sibling =
            completed.previous_sibling;
    }
    auto value = std::move(completed.value); // keep the slot's typed buffer
    completed = {};
    completed.value = std::move(value);
    c.free_slots.push_back(id);
    if (c.occupied != 0)
      --c.occupied;
    if (parent == no_event) {
      if (root != no_event && root < c.root_done.size())
        c.root_done[root] = 1;
      if (c.occupied == 0)
        mark_complete_locked(c);
      notify_workers();
      return;
    }

    auto &ancestor = c.slots[parent];
    if (ancestor.pending_children == 0)
      throw std::logic_error(
          "decoding_task_graph: event child counter underflow");
    --ancestor.pending_children;
    if (ancestor.self_done && ancestor.pending_children == 1)
      enqueue_sibling_waiter_locked(c, parent);
    if (ancestor.self_done && ancestor.pending_children == 0)
      complete_or_continue_locked(c, parent);
  }

  void enqueue_sibling_waiter_locked(context &c, std::size_t parent) {
    for (std::size_t waiting = c.slots[parent].first_child; waiting != no_event;
         waiting = c.slots[waiting].next_sibling)
      if (c.slots[waiting].wait_for_siblings &&
          c.slots[waiting].blocked_by == no_event)
        enqueue_locked(c, waiting);
  }

  void fail(context *c, std::size_t id, std::exception_ptr failure) noexcept {
    auto lock = lock_runtime();
    if (!error)
      error = std::move(failure);
    stop = true;
    if (c && id < c->slots.size() && c->slots[id].running) {
      c->slots[id].running = false;
      unclaim_locked(*c, c->slots[id]);
    }
    if (running != 0)
      --running;
    notify_workers();
    if (c && c->gated)
      owner.wake_async_stream();
  }

public:
  impl &owner;
  std::vector<context> contexts;
  std::vector<context *> active;   // oldest record first
  std::vector<uint8_t> node_busy;  // by topology node
  std::vector<uint64_t> node_turn; // by streaming node: record it serves next
  // By streaming node: free replicas, and which replicas are running.
  std::vector<std::vector<std::size_t>> pool_free;
  std::vector<std::vector<uint8_t>> replica_busy;
  // Executor-held joins of indexed nodes: one table per streaming node,
  // entries keyed by instance, inputs copied in as they arrive.
  struct join_entry {
    enum : uint8_t { missing = 0, present = 1, absent = 2 };
    std::int64_t k = 0;
    bool used = false;
    std::size_t pending = 0;
    std::vector<edge_value> inputs;
    std::vector<uint8_t> state;
  };
  std::vector<std::deque<join_entry>> joins;
  // Edge deliveries each record-ordered topology node has received since
  // its last completion; cleared with the session.
  std::vector<std::size_t> node_deliveries_seen;
  // By streaming node and input slot: the edge's index offset, and whether
  // an instance exists only if that input is delivered.
  std::vector<std::vector<std::int64_t>> join_offsets;
  std::vector<std::vector<uint8_t>> join_required;
  // Instances found never to exist before FINAL, by streaming node; each is
  // announced to the consumers once and never created afterwards. Cleared
  // when the node's FINAL flush has resolved the shot's instances.
  std::vector<std::vector<std::int64_t>> dropped;
  // The last index a record-ordered producer published on each value wire,
  // by compiled_wire::id: its indices never decrease, so the ones it skipped
  // are announced as absent when the next one is published. Cleared by the
  // producer's FINAL flush, after which a shot's numbering starts over.
  std::vector<std::int64_t> wire_last;

  join_entry &add_join_entry(std::size_t si) {
    const auto inputs =
        owner.node_input_counts[owner.streaming_nodes[si].node_index];
    auto &entry = joins[si].emplace_back();
    entry.inputs.resize(inputs);
    entry.state.assign(inputs, 0);
    return entry;
  }
  // Per-replica scratch for the input pointers handed to fire().
  std::vector<std::vector<std::vector<const edge_value *>>> fire_inputs;
  // Whether publication copies outputs into the record (several contexts,
  // or a fire pool).
  bool copy_outputs = false;
  const bool has_waiting_workers;
  std::mutex mutex;
  std::condition_variable wake;
  std::atomic<uint64_t> ready_generation{0};
  std::size_t running = 0;
  uint64_t next_record = 0;
  bool stop = false;
  bool session_done = false;
  // Workers spin before parking only inside an asynchronous session, where
  // the dispatcher hands them records continuously.
  bool spin_while_active = false;
  std::exception_ptr error;
};

// Construct empty graph state before the private scheduler type is complete.
decoding_task_graph::impl::impl() = default;

void decoding_task_graph::impl::discard_records() {
  if (scheduler)
    scheduler->discard_all();
}

void decoding_task_graph::impl::prepare_event_runtime(std::size_t context_count,
                                                      bool dispatcher_thread) {
  // Width-one graphs still use the same event loop, with the caller as the
  // sole worker, so execution semantics do not depend on pool width. An
  // asynchronous dispatcher does no node work, so it gets a thread of its own.
  scheduler.reset();
  parallel_pool = std::make_unique<detail::thread_pool>(
      pool_width + (dispatcher_thread ? 1 : 0), pool_worker_cpus,
      detail::thread_pool::AffinityPolicy::RoundRobin);
  // Causal ancestors remain resident until their descendants drain. This
  // linear multiple covers all simultaneously live frontier, publication,
  // continuation, and flush descriptors without hot-path allocation.
  const auto event_capacity =
      std::max<std::size_t>(64, 16 * (nodes.size() + streaming_wire_count + 1));
  scheduler =
      std::make_unique<record_scheduler>(*this, context_count, event_capacity);
}

// Keep both the persistent event workers and asynchronous coordinator inside
// the graph lifetime.
decoding_task_graph::impl::~impl() { shutdown_async_streaming(); }

// Resolve device affinity once per streaming sweep so every participating pool
// worker enters the same CUDA context before invoking a decoder-backed node.
int decoding_task_graph::impl::resolve_cuda_device_id() const {
  int resolved = -1;
  for (const auto &node : nodes) {
    const auto *decoder_backed = node->decoder_backed();
    if (!decoder_backed)
      continue;
    const auto id = decoder_backed->cuda_device_id();
    if (!id)
      continue;
    if (resolved >= 0 && resolved != *id)
      throw std::runtime_error(
          "decoding_task_graph::cuda_device_id: nodes disagree on CUDA "
          "device (" +
          std::to_string(resolved) + " vs " + std::to_string(*id) +
          "); a graph with GPU-affine nodes on different devices cannot be "
          "pinned to one device for a single decode");
    resolved = *id;
  }
  return resolved;
}

// Fold one retired record's captured node state into the graph-level view the
// progress model reads. Generations advance here, in record order.
void decoding_task_graph::impl::apply_record_context(record_context &c) {
  for (std::size_t si = 0; si < streaming_nodes.size(); ++si) {
    if (si < streaming_node_staged.size() && streaming_node_staged[si]) {
      for (const auto &delta : c.staged[si])
        for (const auto k : streaming_node_outputs[si])
          stage_delta(k, delta.coverage, delta.bits);
      c.staged[si].clear();
    } else if (c.node_snapshotted[si]) {
      auto &latest = latest_node_corrections[si];
      const auto &captured = c.node_corrections[si];
      if (c.node_delta[si]) {
        for (std::size_t i = 0; i < latest.size(); ++i)
          latest[i] ^= captured[i];
        std::fill(c.node_corrections[si].begin(), c.node_corrections[si].end(),
                  uint8_t{0});
      } else {
        std::copy(captured.begin(), captured.end(), latest.begin());
      }
      latest_node_coverage[si] = c.node_coverage[si];
    }
    streaming_node_generations[si] += c.node_published[si];
  }
}

bool decoding_task_graph::impl::enqueue_sweep(const uint8_t *syndrome,
                                              std::size_t len,
                                              std::optional<round_type> type,
                                              int cuda_device_id) {
  require_streaming_support("enqueue_syndrome");
  auto &c = scheduler->acquire();
  c.kind = record_context::phase::enqueue;
  c.syndrome = syndrome;
  c.len = len;
  c.type = type;
  scheduler->push_raw_roots(c);
  scheduler->activate(c, /*gated=*/false);
  try {
    scheduler->run_single(c, cuda_device_id);
  } catch (...) {
    scheduler->discard_all();
    throw;
  }
  apply_record_context(c);
  const bool fired = c.fired.load(std::memory_order_relaxed);
  scheduler->release(c);
  return fired;
}

bool decoding_task_graph::impl::flush_sweep(int cuda_device_id) {
  auto &c = scheduler->acquire();
  c.kind = record_context::phase::flush;
  scheduler->configure_flush(c, streaming_predecessor_counts);
  scheduler->activate(c, /*gated=*/false);
  try {
    scheduler->run_single(c, cuda_device_id);
    scheduler->require_dependencies_complete(
        c, "decoding_task_graph: streaming flush dependency queue stalled");
  } catch (...) {
    scheduler->discard_all();
    throw;
  }
  apply_record_context(c);
  const bool fired = c.fired.load(std::memory_order_relaxed);
  scheduler->release(c);
  return fired;
}

// Execute one graph shot through the same event loop used by streaming.
logical_outcome decoding_task_graph::impl::run_dag_parallel(
    const measurement_results &measurements, const char *who,
    int cuda_device_id) {
  const std::size_t node_count = nodes.size();
  edge_value measurement_value{measurements};
  const std::vector<compiled_wire> *sink_wires = nullptr;

  require_single_output(who);
  sink_wires = &wires_to.back();
  if (parallel_pool == nullptr || scheduler == nullptr)
    throw std::logic_error(
        "decoding_task_graph: parallel runtime was not compiled");
  if (predecessor_counts.size() != node_count ||
      dependents.size() != node_count || source_nodes.empty()) {
    throw std::logic_error(
        "decoding_task_graph: malformed compiled dependency graph");
  }

  auto &c = scheduler->acquire();
  c.kind = record_context::phase::one_shot;
  c.measurement = &measurement_value;
  scheduler->configure_dependencies(c, predecessor_counts);
  for (const std::size_t node : source_nodes)
    scheduler->push(c, {graph_event_kind::one_shot, node});
  scheduler->activate(c, /*gated=*/false);
  try {
    scheduler->run_single(c, cuda_device_id);
    scheduler->require_dependencies_complete(
        c, "decoding_task_graph: one-shot dependency queue stalled");
  } catch (...) {
    scheduler->discard_all();
    throw;
  }
  scheduler->release(c);

  const compiled_wire &sink_wire = sink_wires->front();
  return node_outputs[static_cast<std::size_t>(sink_wire.from_node)]
                     [static_cast<std::size_t>(sink_wire.from_slot)]
                         .get<logical_outcome>();
} // end - decoding_task_graph::impl::run_dag_parallel()

// ----- Asynchronous session ------------------------------------------------

// Retire the oldest record context in sequence order: fold its captures into
// the graph-level result and, when it ends a public operation, publish the
// same progress model the synchronous path produces.
void decoding_task_graph::impl::retire_record_context(record_context &c) {
  auto &state = async_streaming;
  auto &result = streaming_result_state;
  const bool begins_operation =
      c.kind == record_context::phase::enqueue || c.finish_flush;
  if (begins_operation) {
    result.operation_completed = false;
    retiring_generation = result.generation;
    retiring_revision = result.result_revision;
  }
  apply_record_context(c);
  const bool flush = c.kind == record_context::phase::flush;
  if (flush)
    finalized = true;
  refresh_streaming_progress(flush);
  if (!c.closes_record && !c.finish_flush) {
    compose_streaming_values();
    return;
  }
  result.operation_completed =
      result.generation > retiring_generation ||
      (flush && result.generation > 0 &&
       result.status == task_graph_streaming_status::complete);
  compose_streaming_values();
  if (c.finish_flush) {
    publish_async_stream_snapshot();
    return;
  }
  if (result.operation_completed)
    ++result.completed_operations;
  // Most public operations only fill a window and leave every observable and
  // status unchanged. Avoid serializing with readers and copying an identical
  // snapshot for those no-op operations. FINAL still publishes its terminal
  // status even for an empty shot.
  const bool operation_pulse_changed =
      state.result_snapshot.operation_completed != result.operation_completed;
  if (result.result_revision != retiring_revision || operation_pulse_changed ||
      flush)
    publish_async_stream_snapshot();
  const auto completed = c.ring_sequence + 1;
  state.completed.store(completed, std::memory_order_release);
  if (state.completion_waiters.load(std::memory_order_acquire) != 0)
    state.completed.notify_all();
}

void decoding_task_graph::impl::run_async_session() {
  scheduler->begin_session();
  scheduler->run_session(async_streaming.cuda_device_id);
}

// Worker zero of an asynchronous session. Admits ring records into free
// contexts as soon as they are published, retires finished contexts in order,
// and ends the session on finish, cancel, or failure. It performs no node
// work itself, so admission latency is independent of decoder cost.
void decoding_task_graph::impl::dispatch_async_session() {
  auto &state = async_streaming;
  auto &S = *scheduler;
  uint64_t admitted = state.completed.load(std::memory_order_acquire);
  uint64_t observed_epoch = state.wake_epoch.load(std::memory_order_acquire);
  bool flush_admitted = false;
  bool final_pending = false;
  std::exception_ptr failure;

  const auto fail_session = [&](std::exception_ptr error) {
    if (!failure)
      failure = std::move(error);
    auto lock = S.lock_runtime();
    S.request_stop_locked();
  };

  while (true) {
    bool progressed = false;
    {
      auto lock = S.lock_runtime();
      if (S.stop) {
        S.wait_until_idle_locked(lock);
        if (!failure)
          failure = S.error;
        S.end_session_locked();
        lock.unlock();
        if (failure)
          handle_async_stream_error(failure);
        S.discard_all();
        return;
      }
    }
    if (state.cancel_requested.load(std::memory_order_acquire)) {
      auto lock = S.lock_runtime();
      S.request_stop_locked();
      continue;
    }

    // Retire finished records, oldest first.
    while (true) {
      record_context *c = nullptr;
      {
        auto lock = S.lock_runtime();
        c = S.oldest_complete_locked();
      }
      if (!c)
        break;
      try {
        retire_record_context(*c);
      } catch (...) {
        fail_session(std::current_exception());
        break;
      }
      S.release(*c);
      progressed = true;
    }
    if (failure)
      continue;

    // Admit published records.
    while (admitted < state.submitted.load(std::memory_order_acquire)) {
      auto lock = S.lock_runtime();
      auto *c = S.try_acquire_locked();
      if (!c)
        break;
      auto &slot = state.slots[admitted % state.capacity];
      c->kind = record_context::phase::enqueue;
      c->syndrome = slot.syndrome.data();
      c->len = slot.len;
      c->type = slot.type;
      c->ring_sequence = admitted;
      c->closes_record = slot.type != round_type::FINAL;
      if (slot.type == round_type::FINAL)
        final_pending = true;
      lock.unlock();
      try {
        S.push_raw_roots(*c);
      } catch (...) {
        fail_session(std::current_exception());
        break;
      }
      S.activate(*c, /*gated=*/true);
      ++admitted;
      progressed = true;
    }
    if (failure)
      continue;

    // A FINAL record's flush follows it as its own gated context; the record
    // completes when the flush retires. A finish without FINAL flushes once
    // every record has retired.
    const bool finishing =
        state.finish_requested.load(std::memory_order_acquire) &&
        admitted >= state.finish_target;
    bool wants_flush = false;
    if (final_pending && !flush_admitted)
      wants_flush = true;
    else if (finishing && !flush_admitted && !finalized &&
             state.flush_requested.load(std::memory_order_relaxed)) {
      auto lock = S.lock_runtime();
      wants_flush = S.active.empty();
    }
    if (wants_flush) {
      auto lock = S.lock_runtime();
      if (auto *c = S.try_acquire_locked()) {
        c->kind = record_context::phase::flush;
        c->ring_sequence = admitted - 1;
        c->closes_record = final_pending;
        c->finish_flush = !final_pending;
        lock.unlock();
        try {
          S.configure_flush(*c, streaming_predecessor_counts);
        } catch (...) {
          fail_session(std::current_exception());
          continue;
        }
        S.activate(*c, /*gated=*/true);
        flush_admitted = true;
        progressed = true;
      }
    }

    if (finishing) {
      auto lock = S.lock_runtime();
      const bool flush_settled =
          flush_admitted || finalized ||
          !state.flush_requested.load(std::memory_order_relaxed);
      if (S.active.empty() && flush_settled) {
        S.end_session_locked();
        return;
      }
    }

    if (progressed)
      continue;
    bool in_flight = false;
    {
      auto lock = S.lock_runtime();
      in_flight = !S.active.empty();
    }
    wait_for_async_wake(
        observed_epoch,
        [&] {
          // Unadmitted records are work only if a context can take them;
          // otherwise a completion, which bumps the epoch, must come first.
          auto lock = S.lock_runtime();
          if (S.stop || S.oldest_complete_locked() != nullptr)
            return true;
          return admitted < state.submitted.load(std::memory_order_acquire) &&
                 std::any_of(
                     S.contexts.begin(), S.contexts.end(),
                     [](const auto &c) { return !c.in_use; });
        },
        in_flight ? std::max(state.spin_pause_iters, kHandoffSpinPauseIters)
                  : state.spin_pause_iters);
  }
}

} // namespace cudaq::qec
