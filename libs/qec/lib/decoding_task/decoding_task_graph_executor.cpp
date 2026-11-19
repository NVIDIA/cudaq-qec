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
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/nvtx_helpers.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <functional>
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

// Cap on record_bytes. A windowed graph's ingest chunk is one first-layer
// period, 3d rounds of about d^2 measurements: 8 KB at d = 11, 126 KB at
// d = 35. 128 KB covers that and bounds each context's batch buffer.
constexpr std::size_t max_record_bytes = std::size_t{128} << 10;

// Flush stage tags stored in graph_event::offset. Zero begins normal flush.
constexpr std::size_t flush_resolved = 1;

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
  std::int64_t next_skipped = no_index; ///< first missing index before index
  std::size_t entry = no_event; ///< join-table entry an indexed fire consumes
  bool absent = false;      ///< the producer's instance `index` never exists
  bool empty_value = false; ///< this value port was empty at publication
  /// Queue sequence of the public operation it follows from: its latest
  /// cause's.
  std::uint64_t operation = 0;
};

// Stored with the completed join until its feedback fire can take a slot.
struct deferred_fire {
  graph_event event;
  deferred_fire *next = nullptr;
};

} // namespace

// Everything one record moves between nodes or reads back from a node after a
// fire lives here, so concurrently executing records never share data.
struct decoding_task_graph::impl::record_context {
  enum class phase : uint8_t { idle, one_shot, enqueue, flush };

  struct pending_delivery {
    graph_event event;
    std::size_t next = no_event;
    edge_value *buffer = nullptr;
  };

  struct slot {
    graph_event event;
    graph_event continuation;
    std::size_t pool_slot = no_event; ///< replica a running delivery holds
    bool releases_pool = false;       ///< last delivery: hand the replica back
    edge_value *buffer = nullptr;     ///< its buffer in the wire's pool, if any
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
    // Pending deliveries borrow this slot while its saved continuation
    // and feedback boundary still belong to the original event.
    bool replaying_delivery = false;
    bool fire_boundary = false;
    // A feedback fire can reuse this slot after its current subtree drains.
    deferred_fire *first_feedback = nullptr;
    deferred_fire *last_feedback = nullptr;
    std::size_t first_delivery = no_event;
    std::size_t last_child_delivery = no_event;
    std::size_t reserved_slots = 0; ///< reservation across every reuse
    // This slot uses its own or an ancestor's reservation.
    bool covered = false;
  };

  std::vector<slot> slots;
  std::vector<std::size_t> free_slots;
  // Excess deliveries reuse their parent's slot.
  // Recycle their descriptors as the queues drain and keep the storage for
  // the next record.
  std::vector<pending_delivery> pending_deliveries;
  std::size_t free_delivery = no_event;
  std::vector<int32_t> dependency_remaining;
  std::vector<std::size_t> root_nodes;
  std::vector<uint8_t> root_done;
  // Per root: its raw or flush event and the fires queued without a parent
  // on its behalf still to finish; the root is done at zero.
  std::vector<std::uint32_t> root_open;
  // Edge deliveries a pooled node received in this record.
  std::vector<std::size_t> deliveries_seen;
  // By wire: deliveries keep their own typed buffers until they retire.
  // A buffer's address stays stable while other deliveries grow the pool.
  struct wire_buffers {
    std::vector<std::unique_ptr<edge_value>> values;
    std::vector<edge_value *> free;
  };
  std::vector<wire_buffers> wire_pool;
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
    std::uint32_t operation = 0; ///< of the record's operations
  };
  // In use: the first staged_count[si] of staged[si]; the rest keep their
  // buffers for the record's next publications.
  std::vector<std::vector<staged_delta>> staged;
  std::vector<std::size_t> staged_count;
  // Retirement heap: at most one entry per node, reserved at construction.
  std::vector<std::size_t> staged_nodes;
  // Replica of each pooled node leased to this record, or no_event.
  std::vector<std::size_t> pool_replica;
  // Publications by operation and streaming node: [operation * M + node].
  std::vector<std::uint32_t> node_published;
  // Each node's publications' closures in this record, combined.
  std::vector<std::optional<task_graph_closure>> node_closure;
  // And whether each one's decoder converged.
  std::vector<std::optional<bool>> node_converged;
  // Release accounting, indexed by streaming node: live events targeting
  // the node and whether this record has handed the node on.
  std::vector<std::uint32_t> node_live;
  // Of those, the indexed fires.
  std::vector<std::uint32_t> node_fires_live;
  std::vector<uint8_t> node_released;
  std::size_t occupied = 0;
  std::size_t next_sequence = 0;
  std::size_t ready_head = no_event;
  std::size_t ready_tail = no_event;
  std::atomic<bool> fired{false};
  // Set by the worker that finishes the context, under the runtime lock;
  // the async coordinator reads it without.
  std::atomic<bool> complete{false};
  // Whether no raw node's chunk limit has cut the record so far, though
  // rounds admitted together still go in one chunk each (see fire_parent());
  // atomic, as a raw event clears it outside the lock.
  std::atomic<bool> one_chunk{true};
  // Slots reserved by this record's fires queued without a parent, including
  // the feedback fires that can reuse their slots.
  std::size_t detached_slots = 0;
  // Reservations of all fire subtrees, and live slots outside those trees.
  std::size_t reserved_slots = 0;
  std::size_t unreserved_slots = 0;
  bool gated = false;
  bool in_use = false;
  uint64_t sequence = 0;
  phase kind = phase::idle;
  // Record payload: all or part of a queue slot's, or this buffer's, which
  // holds rounds admitted together.
  std::vector<uint8_t> batch;
  const uint8_t *syndrome = nullptr;
  std::size_t len = 0;
  std::optional<round_type> type;
  const edge_value *measurement = nullptr;
  // Asynchronous record bookkeeping: which public operation this context
  // belongs to and whether retiring it completes that operation.
  uint64_t ring_sequence = 0;
  // Public operations this record holds (rounds admitted together) and
  // where each one's bytes end.
  std::size_t operations = 1;
  std::vector<std::size_t> operation_ends;
  // Queue sequence of its first operation; ring_sequence is its last.
  std::uint64_t first_operation = 0;
  // Whether this record starts its public operation: only the first of a
  // record cut into several does.
  bool opens_record = true;
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
  struct feedback_flush;
  // Index and topological node order put a producer before its consumer.
  using feedback_join = std::tuple<std::int64_t, std::size_t, std::size_t>;
  enum class drop_chain : uint8_t { unseen, finite, cyclic };

  record_scheduler(impl &owner, std::size_t context_count,
                   std::size_t event_capacity)
      : owner(owner), contexts(context_count), node_busy(owner.nodes.size(), 0),
        node_turn(owner.streaming_nodes.size(), 0),
        pool_free(owner.streaming_nodes.size()),
        replica_busy(owner.streaming_nodes.size()),
        joins(owner.streaming_nodes.size()),
        join_live(owner.streaming_nodes.size()),
        join_scan(owner.streaming_nodes.size()),
        self_fed(owner.streaming_nodes.size(), 0),
        fire_slots(owner.streaming_nodes.size(), 0),
        reaches_ordered(owner.streaming_nodes.size(), 0),
        join_grew(owner.streaming_nodes.size(), join_growth::none),
        join_home(owner.streaming_nodes.size(), 1),
        join_displaced(owner.streaming_nodes.size(), 0),
        join_offsets(owner.streaming_nodes.size()),
        join_required(owner.streaming_nodes.size()),
        required_drop_chain(owner.streaming_nodes.size(), drop_chain::unseen),
        defer_drop_notifications(owner.streaming_nodes.size(), 0),
        defer_publications(owner.streaming_nodes.size(), 0),
        copy_publications(owner.streaming_nodes.size(), 0),
        dropped(owner.streaming_nodes.size()),
        wire_last(owner.streaming_value_wire_count, no_index),
        fire_inputs(owner.streaming_nodes.size()),
        fire_entries(owner.streaming_nodes.size()),
        event_labels(owner.streaming_nodes.size()),
        one_shot_labels(owner.nodes.size()),
        has_waiting_workers(owner.parallel_pool->num_threads() > 1) {
    // NVTX range names: an event's kind and its node. A one-shot event names
    // a topology node, which has an id only if it also streams.
    static constexpr const char *kind_names[event_kinds] = {
        "one-shot", "ingest", "deliver", "flush", "flush release", "fire"};
    for (std::size_t n = 0; n < owner.nodes.size(); ++n)
      one_shot_labels[n] = "one-shot node " + std::to_string(n);
    for (std::size_t si = 0; si < owner.streaming_nodes.size(); ++si) {
      const auto &id = owner.streaming_nodes[si].id;
      for (std::size_t k = 0; k < event_kinds; ++k)
        event_labels[si][k] = std::string(kind_names[k]) + " " + id;
      if (owner.streaming_nodes[si].node_index >= 0 &&
          static_cast<std::size_t>(owner.streaming_nodes[si].node_index) <
              owner.nodes.size())
        one_shot_labels[owner.streaming_nodes[si].node_index] =
            "one-shot " + id;
    }
    const auto N = owner.nodes.size();
    admitting.reserve(context_count);
    const auto M = owner.streaming_nodes.size();
    // Join leads, in topological order: how far past the newest published
    // index an indexed node can fire (f) or have an entry created (e). And
    // the most each node publishes per delivery: its own, or, fed by edges,
    // as many as one of its producers' deliveries brings.
    std::vector<std::int64_t> fire_lead(M, 0), entry_lead(M, 0),
        min_required_offset(M, 0);
    std::vector<std::size_t> publications(M, 1);
    // Completion edges don't order sibling branches. A node downstream of
    // one can run twice before its first output is delivered, even when only
    // one record is active and each fire publishes once.
    for (const auto &wires : owner.completion_wires_from)
      for (const auto &wire : wires)
        copy_publications[owner.streaming_by_node[wire.to_node]] = 1;
    bool copy_outputs = context_count > 1;
    for (std::size_t si = 0; si < M; ++si) {
      const auto node_idx = owner.streaming_nodes[si].node_index;
      const auto *capability = owner.streaming_nodes[si].capability;
      publications[si] = capability->max_publications_per_delivery();
      // Overlapping branches can make a rendezvous node publish twice
      // before the first delivery reads its output.
      copy_publications[si] |=
          publications[si] > 1 || capability->supports_streaming_rendezvous();
      for (const auto producer : owner.streaming_producers[si]) {
        publications[si] = std::max(publications[si], publications[producer]);
        copy_publications[si] |= copy_publications[producer];
      }
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
      const auto inputs = owner.node_input_counts[node_idx];
      join_offsets[si].assign(inputs, 0);
      join_required[si].assign(inputs, 0);
      for (std::size_t i = 0; i < inputs; ++i)
        join_required[si][i] =
            owner.streaming_nodes[si].indexed.front()->requires_input(i);
      std::int64_t f = std::numeric_limits<std::int64_t>::max(), e = 0;
      for (const auto &wire : owner.wires_to[node_idx]) {
        const auto slot = static_cast<std::size_t>(wire.to_slot);
        join_offsets[si][slot] = wire.index_offset;
        if (join_required[si][slot])
          min_required_offset[si] =
              std::min(min_required_offset[si], wire.index_offset);
        self_fed[si] = self_fed[si] || wire.loop_carried();
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
      // outputs: the records in flight, each a delivery publishing as many
      // as the node's producers can, bound how far publication runs ahead.
      std::size_t inflow = 1;
      for (const auto producer : owner.streaming_producers[si])
        inflow = std::max(inflow, publications[producer]);
      const auto capacity =
          context_count * inflow + static_cast<std::size_t>(e) + copies;
      joins[si].clear();
      join_live[si].clear();
      for (std::size_t i = 0; i < capacity; ++i)
        add_join_entry(si);
      join_home[si] = capacity;
      join_displaced[si] = 0;
      // Instances live no further apart than the table holds; twice that
      // keeps every drop a delivery could still address.
      dropped[si].ring.assign(2 * capacity, no_index);
      fire_entries[si].assign(copies, nullptr);
      fire_inputs[si].assign(copies,
                             std::vector<const edge_value *>(
                                 owner.node_input_counts[node_idx], nullptr));
    }
    // An absence that reaches a cycle through required inputs propagates
    // forever. Every lap advances the index, since feedback offsets are
    // strictly negative. Cache this reachability before recycling slots
    // for feedback notifications.
    const auto reaches_required_cycle = [&](auto &&visit,
                                            std::size_t si) -> bool {
      auto &chain = required_drop_chain[si];
      if (chain != drop_chain::unseen)
        return chain == drop_chain::cyclic;
      chain = drop_chain::cyclic; // a recursive visit closes the cycle
      const auto node_idx = owner.streaming_nodes[si].node_index;
      for (const auto &wire : owner.streaming_wires_from[node_idx]) {
        if (wire.to_node < 0)
          continue;
        const auto consumer = owner.streaming_by_node[wire.to_node];
        if (consumer >= 0 && indexed(consumer) &&
            join_required[consumer][wire.to_slot] && visit(visit, consumer))
          return true;
      }
      chain = drop_chain::finite;
      return false;
    };
    for (std::size_t si = M; si-- > 0;)
      if (indexed(si))
        reaches_required_cycle(reaches_required_cycle, si);

    configure_feedback_flush();

    // Bound the events one delivery holds until it drains. A feedback fire
    // gets its own reservation or waits for the current subtree to drain.
    // A dropped join instead sends its absence immediately on its value
    // wires to indexed consumers, so those deliveries use this reservation.
    // Cap the estimates at the store's capacity to keep expensive work
    // attached.
    const auto add_slots = [event_capacity](std::size_t a, std::size_t b) {
      return a + std::min(b, event_capacity - a);
    };
    const auto multiply_slots = [event_capacity](std::size_t count,
                                                 std::size_t slots) {
      if (slots != 0 && count > event_capacity / slots)
        return event_capacity;
      return count * slots;
    };
    std::vector<std::optional<std::size_t>> delivery_slots(M), drop_slots(M);
    const auto count_slots = [&](auto &&count, std::size_t si,
                                 bool dropped_join) -> std::size_t {
      auto &bound = dropped_join ? drop_slots[si] : delivery_slots[si];
      if (bound)
        return *bound;
      // If a notification path revisits a bound still being computed, the
      // topology provides no finite bound for this cascade.
      bound = event_capacity;
      const auto downstream_slots =
          [&](const std::vector<compiled_wire> &wires) {
            std::size_t slots = 0;
            for (const auto &wire : wires) {
              if (wire.to_node < 0 || owner.streaming_by_node[wire.to_node] < 0)
                continue;
              const auto consumer = static_cast<std::size_t>(
                  owner.streaming_by_node[wire.to_node]);
              const bool indexed_consumer = this->indexed(consumer);
              if (dropped_join && !indexed_consumer)
                continue;
              std::size_t delivery = 1;
              if (!wire.loop_carried()) {
                delivery = count(count, consumer, /*dropped_join=*/false);
              } else if (indexed_consumer &&
                         (join_required[consumer][wire.to_slot] ||
                          wire.index_offset > min_required_offset[consumer])) {
                // An absence on a required input drops the join. A value can
                // also drop it if another required input names a negative
                // index.
                delivery =
                    add_slots(1, count(count, consumer, /*dropped_join=*/true));
              }
              // A record-ordered producer's value may wait in its slot
              // while one skipped-index notification runs underneath it.
              if (indexed_consumer && wire.id >= 0 && !indexed(si) &&
                  !pooled(si))
                delivery = add_slots(1, delivery);
              slots = add_slots(slots, delivery);
            }
            return slots;
          };
      const auto node_idx = owner.streaming_nodes[si].node_index;
      const auto value_slots =
          downstream_slots(owner.streaming_wires_from[node_idx]);
      if (dropped_join) {
        defer_drop_notifications[si] = value_slots == event_capacity;
        bound = value_slots;
        return *bound;
      }
      const auto publication_count =
          owner.streaming_nodes[si].capability->max_publications_per_delivery();
      const auto completion_slots =
          downstream_slots(owner.completion_wires_from[node_idx]);
      // An indexed fire sends one completion for its instance, even if it
      // publishes several values or none.
      const bool indexed_producer = this->indexed(si);
      const auto published = multiply_slots(
          publication_count, indexed_producer
                                 ? value_slots
                                 : add_slots(value_slots, completion_slots));
      auto outgoing = indexed_producer ? add_slots(published, completion_slots)
                                       : std::max(published, completion_slots);
      defer_publications[si] = outgoing == event_capacity;
      if (indexed_producer) {
        fire_slots[si] = add_slots(1, outgoing);
        outgoing =
            std::max(fire_slots[si], count(count, si, /*dropped_join=*/true));
      }
      bound = add_slots(1, outgoing);
      return *bound;
    };
    for (std::size_t si = M; si-- > 0;)
      count_slots(count_slots, si, /*dropped_join=*/false);
    // A detached fire reserves enough for any subtree that can reuse its slot
    // through feedback. Also propagate whether an indexed fire can reach a
    // node that joins its inputs by record, which precludes detaching.
    // Loop-carried wires lead to earlier nodes, so repeat until nothing
    // changes.
    for (bool changed = true; changed;) {
      changed = false;
      for (std::size_t si = M; si-- > 0;) {
        if (!this->indexed(si))
          continue;
        const auto node_idx = owner.streaming_nodes[si].node_index;
        auto slots = fire_slots[si];
        auto ordered = reaches_ordered[si];
        for (const auto *wires : {&owner.streaming_wires_from[node_idx],
                                  &owner.completion_wires_from[node_idx]})
          for (const auto &wire : *wires) {
            if (wire.to_node < 0 || owner.streaming_by_node[wire.to_node] < 0)
              continue;
            const auto consumer =
                static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
            slots = std::max(slots, fire_slots[consumer]);
            if (!this->indexed(consumer) || reaches_ordered[consumer])
              ordered = 1;
          }
        changed = changed || slots != fire_slots[si] ||
                  ordered != reaches_ordered[si];
        fire_slots[si] = slots;
        reaches_ordered[si] = ordered;
      } // end - for(si)
    } // end - for(changed)
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
      stager.intervals.assign(
          capacity, {0, 0, std::vector<uint8_t>(stager.bits.size(), 0)});
      stager.pending.clear();
      stager.pending.reserve(capacity);
      stager.spare.resize(capacity);
      for (std::size_t i = 0; i < capacity; ++i)
        stager.spare[i] = i;
    }
    if (copy_outputs)
      std::fill(copy_publications.begin(), copy_publications.end(), 1);
    node_deliveries_seen.assign(N, 0);
    for (auto &c : contexts) {
      c.slots.resize(event_capacity);
      c.free_slots.reserve(event_capacity);
      c.root_nodes.reserve(M);
      c.root_done.reserve(M);
      c.root_open.reserve(M);
      c.deliveries_seen.assign(N, 0);
      c.wire_pool.resize(owner.streaming_value_wire_count);
      c.node_corrections.resize(M);
      for (std::size_t si = 0; si < M; ++si)
        c.node_corrections[si].assign(
            owner.streaming_nodes[si].capability->num_observables(), 0);
      c.node_coverage.assign(M, std::nullopt);
      c.node_snapshotted.assign(M, 0);
      c.node_delta.assign(M, 0);
      c.staged.assign(M, {});
      c.staged_count.assign(M, 0);
      c.staged_nodes.reserve(M);
      c.pool_replica.assign(M, no_event);
      c.node_published.assign(M, 0);
      c.node_closure.assign(M, std::nullopt);
      c.node_converged.assign(M, std::nullopt);
      c.node_live.assign(M, 0);
      c.node_fires_live.assign(M, 0);
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

  // A free context for the async coordinator to fill before activating it.
  // The coordinator alone acquires, activates and releases contexts, and
  // workers look only at active ones, so filling it needs no lock.
  context *try_acquire_for_admission() { return try_acquire_locked(); }

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
      for (auto &pool : c.wire_pool) {
        pool.free.clear();
        for (const auto &buffer : pool.values)
          pool.free.push_back(buffer.get());
      }
    }
    c.root_nodes.clear();
    c.root_done.clear();
    c.root_open.clear();
    std::fill(c.deliveries_seen.begin(), c.deliveries_seen.end(), 0);
    std::fill(c.node_snapshotted.begin(), c.node_snapshotted.end(), 0);
    std::fill(c.node_delta.begin(), c.node_delta.end(), 0);
    std::fill(c.staged_count.begin(), c.staged_count.end(), 0);
    for (std::size_t si = 0; si < c.node_corrections.size(); ++si)
      if (pooled(si) || indexed(si))
        std::fill(c.node_corrections[si].begin(), c.node_corrections[si].end(),
                  uint8_t{0});
    std::fill(c.pool_replica.begin(), c.pool_replica.end(), no_event);
    std::fill(c.node_published.begin(), c.node_published.end(), 0);
    std::fill(c.node_closure.begin(), c.node_closure.end(), std::nullopt);
    std::fill(c.node_converged.begin(), c.node_converged.end(), std::nullopt);
    std::fill(c.node_live.begin(), c.node_live.end(), 0);
    std::fill(c.node_fires_live.begin(), c.node_fires_live.end(), 0);
    std::fill(c.node_released.begin(), c.node_released.end(), 0);
    c.occupied = 0;
    c.next_sequence = 0;
    c.ready_head = no_event;
    c.ready_tail = no_event;
    c.fired.store(false, std::memory_order_relaxed);
    c.complete.store(false, std::memory_order_relaxed);
    c.one_chunk.store(true, std::memory_order_relaxed);
    c.detached_slots = 0;
    c.reserved_slots = 0;
    c.unreserved_slots = 0;
    c.pending_deliveries.clear();
    c.free_delivery = no_event;
    c.gated = false;
    c.sequence = 0;
    c.kind = phase::idle;
    c.syndrome = nullptr;
    c.len = 0;
    c.type.reset();
    c.measurement = nullptr;
    c.ring_sequence = 0;
    c.operations = 1;
    c.first_operation = 0;
    c.opens_record = true;
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
      for (auto &entry : table) {
        entry.used = false;
        entry.displaced = false;
      }
    for (auto &live : join_live)
      live.clear();
    for (auto &flush : feedback_flushes) {
      flush.pending.clear();
      flush.resolving = false;
    }
    running_fires = 0;
    std::fill(join_displaced.begin(), join_displaced.end(), 0);
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
    push_raw_roots_locked(c);
  }

  void push_raw_roots_locked(context &c) {
    for (std::size_t node = 0; node < owner.streaming_nodes.size(); ++node)
      if (owner.streaming_node_has_raw_input[node])
        c.root_nodes.push_back(node);
    c.root_done.resize(c.root_nodes.size());
    std::fill(c.root_done.begin(), c.root_done.end(), uint8_t{0});
    c.root_open.assign(c.root_nodes.size(), 1);
    for (std::size_t root = 0; root < c.root_nodes.size(); ++root)
      push_locked(
          c,
          {graph_event_kind::streaming_raw, c.root_nodes[root], {}, 0, root});
  }

  void configure_dependencies(context &c, const std::vector<int32_t> &counts) {
    auto lock = lock_runtime();
    c.dependency_remaining = counts;
  }

  // Start FINAL at the roots of the graph's flush dependencies.
  void configure_flush(context &c) {
    auto lock = lock_runtime();
    c.dependency_remaining = flush_predecessor_counts;
    for (std::size_t node = 0; node < flush_predecessor_counts.size(); ++node)
      if (flush_predecessor_counts[node] == 0)
        c.root_nodes.push_back(node);
    c.root_done.resize(c.root_nodes.size());
    std::fill(c.root_done.begin(), c.root_done.end(), uint8_t{0});
    c.root_open.assign(c.root_nodes.size(), 1);
    for (std::size_t root = 0; root < c.root_nodes.size(); ++root)
      push_locked(
          c,
          {graph_event_kind::streaming_flush, c.root_nodes[root], {}, 0, root});
  }

  // Make a populated context eligible for execution. Nodes this record never
  // touches are handed on immediately so later records need not wait.
  void activate(context &c, bool gated) {
    auto lock = lock_runtime();
    activate_locked(c, gated);
    notify_workers();
  }

  // activate() under the caller's lock; the caller wakes the workers.
  void activate_locked(context &c, bool gated) {
    c.gated = gated;
    if (gated) {
      c.sequence = next_record++;
      for (std::size_t si = 0; si < owner.streaming_nodes.size(); ++si)
        try_release_locked(c, si);
    }
    active.push_back(&c);
    if (c.occupied == 0)
      mark_complete_locked(c);
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
    // It follows from what the event it continues follows from.
    event.operation = std::max(event.operation, c.slots[id].event.operation);
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

  // The record holding @p operation: @p c, or a later, unretired one, found
  // by scanning the active records.
  context &holder_locked(context &c, std::uint64_t operation) {
    if (operation <= c.ring_sequence)
      return c;
    for (auto *later : active)
      if (later->first_operation <= operation &&
          operation <= later->ring_sequence)
        return *later;
    throw std::logic_error(
        "decoding_task_graph: no active record holds operation " +
        std::to_string(operation));
  }

  // Capture what a node exposes after it ran, while this record still holds
  // the node. Retirement folds the capture into the graph-level result in
  // record order, so a node already serving a later record cannot leak into
  // an earlier record's publication. Used for a FINAL record and a flush,
  // which each hold one operation.
  void snapshot_node(context &c, std::size_t si) {
    // Copies of a pooled or indexed node may publish into one record at
    // once, as may a staged node's.
    std::unique_lock<std::mutex> lock;
    if (staged(si) || pooled(si) || indexed(si))
      lock = lock_runtime();
    snapshot_node_locked(c, si, *owner.streaming_nodes[si].capability, 0);
  }

  // snapshot_node() of @p capability, a copy of node si, under the lock
  // where it needs one.
  void snapshot_node_locked(context &c, std::size_t si,
                            const streaming_task_graph_node &capability,
                            std::uint32_t operation) {
    if (staged(si)) {
      // Each publication is one delta over one commit interval; the record
      // hands them to the output stagers when it retires.
      const auto coverage = capability.streaming_coverage();
      if (!coverage)
        return; // nothing to fold from this publication
      const auto n = capability.num_observables();
      const uint8_t *delta = capability.corrections();
      auto &deltas = c.staged[si];
      auto &used = c.staged_count[si];
      if (used == deltas.size())
        deltas.emplace_back(); // only while the record warms up
      auto &entry = deltas[used++];
      entry.coverage = *coverage;
      entry.operation = operation;
      if (delta)
        entry.bits.assign(delta, delta + n);
      else
        entry.bits.assign(n, 0);
      c.node_snapshotted[si] = 1;
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
      auto &dst = c.node_corrections[si];
      if (const uint8_t *delta = capability.corrections())
        for (std::size_t i = 0; i < dst.size(); ++i)
          dst[i] ^= delta[i];
      c.node_delta[si] = 1;
      c.node_snapshotted[si] = 1;
      return;
    }
    auto &dst = c.node_corrections[si];
    if (!dst.empty())
      if (const uint8_t *src = capability.corrections())
        std::copy(src, src + dst.size(), dst.begin());
    c.node_coverage[si] = capability.streaming_coverage();
    c.node_snapshotted[si] = 1;
  }

  // A fire leaves a publication current; publish it and each further one
  // the node makes current. Only the first may carry the fire's own index.
  // Workers are woken once, for all of them.
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
    const auto node_idx = owner.streaming_nodes[producer].node_index;
    const auto &completions = owner.completion_wires_from[node_idx];
    if (indexed(producer) && index != no_index && !completions.empty())
      publish_wires(c, parent, root, completions, index);
    else
      notify_workers();
    return true;
  }

  // One publication, in one critical section: its capture, its closure and
  // convergence, its outputs' copies and its deliveries. Wakes no worker.
  void publish_streaming(context &c, std::size_t parent, std::size_t root,
                         std::size_t producer,
                         const streaming_task_graph_node *capability = nullptr,
                         std::int64_t index = no_index, bool repeat = false) {
    const auto node_idx = owner.streaming_nodes[producer].node_index;
    if (!capability)
      capability = owner.streaming_nodes[producer].capability;
    const auto &wires = owner.streaming_wires_from[node_idx];
    const auto &completions = owner.completion_wires_from[node_idx];
    const auto anchor = feedback_anchor[producer];
    auto lock = lock_runtime();
    // Captured in the record holding the operation it follows from, as a
    // synchronous graph would.
    const auto &event = c.slots[parent].event;
    const auto operation = event.operation;
    auto &home = holder_locked(c, operation);
    const auto local =
        static_cast<std::uint32_t>(operation - home.first_operation);
    snapshot_node_locked(home, producer, *capability, local);
    ++home.node_published[local * owner.streaming_nodes.size() + producer];
    home.node_closure[producer] = combine_closure(
        home.node_closure[producer], capability->publication_closure());
    if (const auto converged = capability->publication_converged())
      home.node_converged[producer] =
          home.node_converged[producer].value_or(true) && *converged;
    // Each port carries its own index, unless the fire named one for all
    // of them. A null port says nothing to an indexed consumer this time.
    for (const auto &wire : wires) {
      const auto consumer =
          static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
      const auto *value = capability->streaming_output(wire.from_slot);
      if (repeat && value && !indexed(consumer))
        throw std::logic_error(
            "decoding_task_graph: node " + std::to_string(node_idx) +
            " published twice in one delivery on output " +
            std::to_string(wire.from_slot) +
            ", whose consumer joins its inputs by record; only an indexed "
            "consumer takes repeat publications");
      if (!value && indexed(consumer))
        continue;
      const bool internal_feedback =
          anchor != no_event && anchor == feedback_anchor[consumer];
      if (internal_feedback && event.kind == graph_event_kind::streaming_flush)
        throw std::logic_error("decoding_task_graph: node '" +
                               owner.streaming_nodes[producer].id +
                               "' flush published inside a feedback cycle");
      const auto wire_index =
          index != no_index
              ? index
              : capability->published_index(wire.from_slot).value_or(no_index);
      // A repeat publication can name its index independently. Internal
      // feedback uses the firing instance's index so FINAL can settle
      // dependencies in instance order.
      if (internal_feedback && wire_index != event.index)
        throw std::logic_error(
            "decoding_task_graph: node '" + owner.streaming_nodes[producer].id +
            "' feedback index " + std::to_string(wire_index) +
            " differs from firing index " + std::to_string(event.index));
      publish_wire_locked(c, parent, root, wire, wire_index, value);
    }
    if (!indexed(producer)) {
      publish_wires_locked(c, parent, root, completions, index);
    } else if (event.kind == graph_event_kind::streaming_flush) {
      // A callback publication completes ordered consumers. Indexed
      // consumers get a completion only from the fire that names its index.
      for (const auto &wire : completions)
        if (!indexed(owner.streaming_by_node[wire.to_node]))
          publish_wire_locked(c, parent, root, wire, no_index, nullptr);
    }
  } // end - publish_streaming()

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
      publish_wire_locked(c, parent, root, wire, index, nullptr, absent);
  }

  // With `absent`, the publication says instance `index` of the producer
  // never exists: it carries no value and goes only to indexed consumers,
  // which file the absence under the instances reading that index.
  void publish_wire_locked(context &c, std::size_t parent, std::size_t root,
                           const compiled_wire &wire, std::int64_t index,
                           const edge_value *value, bool absent = false) {
    {
      const auto consumer =
          static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
      if (stop || (absent && !indexed(consumer)))
        return;
      const auto source =
          static_cast<std::size_t>(owner.streaming_by_node[wire.from_node]);
      // Put a record-ordered producer's skipped range on the value delivery.
      // The delivery announces one absence at a time before filing its value.
      graph_event event{graph_event_kind::streaming_edge, consumer, wire, 0,
                        root};
      event.index = index;
      event.absent = absent;
      event.empty_value = wire.id >= 0 && !value;
      if (!absent && index != no_index && indexed(consumer) && wire.id >= 0) {
        auto &last = wire_last[static_cast<std::size_t>(wire.id)];
        if (!pooled(source) && !indexed(source)) {
          if (last == no_index ? index > 0 : last < index - 1)
            event.next_skipped = last == no_index ? 0 : last + 1;
          last = index;
        }
      }
      if (absent && wire.loop_carried()) {
        // Carry the original operation into the earlier ancestor's subtree.
        // Each lap can then retire its deliveries while the ancestor still
        // holds the notification's root and any reservation.
        event.operation = c.slots[parent].event.operation;
        parent = feedback_notification_parent(c, parent);
      }
      // Leave room for descendants by replaying excess deliveries in
      // publication order once the producer's current subtree drains.
      // An indexed flush can send a completion per publication, beyond
      // the single completion budgeted for an indexed fire.
      auto &producer = c.slots[parent];
      const bool defer =
          !absent &&
          (defer_publications[source] || producer.first_delivery != no_event ||
           c.free_slots.size() <= c.slots.size() / 2 ||
           (producer.event.kind == graph_event_kind::streaming_flush &&
            indexed(source)));
      std::size_t blocker = no_event;
      std::size_t blocker_sequence = 0;
      // Direct children are stored in declared wire order. A later branch
      // waits only when its transitive descendants intersect an earlier
      // branch; disjoint fan-out remains immediately runnable. An indexed
      // consumer accepts its inputs in any order, so it never waits.
      for (std::size_t earlier = c.slots[parent].first_child;
           earlier != no_event && !defer && !indexed(consumer);
           earlier = c.slots[earlier].next_sibling) {
        const auto &candidate = c.slots[earlier];
        if (candidate.event.kind != graph_event_kind::streaming_edge)
          continue;
        if (branches_ordered(candidate.event.node, consumer) &&
            (blocker == no_event || candidate.sequence > blocker_sequence)) {
          blocker = earlier;
          blocker_sequence = candidate.sequence;
        }
      }
      // Later publications or records may replace the producer's value
      // before this delivery runs. The simple single-publication walk can
      // read it in place.
      edge_value *buffer = nullptr;
      if (value && copy_publications[source]) {
        auto &pool = c.wire_pool[static_cast<std::size_t>(wire.id)];
        // A new buffer only while the pool warms up.
        if (pool.free.empty()) {
          pool.values.push_back(std::make_unique<edge_value>());
          pool.free.reserve(pool.values.capacity());
          pool.free.push_back(pool.values.back().get());
        }
        buffer = pool.free.back();
        *buffer = *value;
        pool.free.pop_back();
      }
      if (defer) {
        event.operation = producer.event.operation;
        defer_delivery_locked(c, producer, event, buffer);
      } else {
        const auto id = push_locked(c, event, parent, blocker);
        c.slots[id].buffer = buffer;
      }
    }
  } // end - publish_wire_locked()

  // Read this delivery's snapshot or the producer output held for it.
  const edge_value *delivered_value(const graph_event &event,
                                    const edge_value *buffer) const {
    if (buffer || event.empty_value || event.absent)
      return buffer;
    const auto from = static_cast<std::size_t>(event.wire.from_node);
    const auto from_slot = static_cast<std::size_t>(event.wire.from_slot);
    // The implicit `done` output is the constant completion marker.
    if (event.wire.id < 0)
      return &owner.node_outputs[from][from_slot];
    return owner.streaming_nodes[owner.streaming_by_node[from]]
        .capability->streaming_output(from_slot);
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
    auto *buffer = c.slots[id].buffer;
    const auto *value = delivered_value(event, buffer);
    auto lock = lock_runtime();
    if (k < 0 || is_dropped(si, k))
      return; // instances are numbered from 0, and a dropped one stays so
    auto &table = joins[si];
    std::size_t index = find_join_locked(si, k);
    const bool fresh = index == table.size();
    if (fresh) {
      index = join_home_of(si, k);
      if (table[index].used) {
        // Taken by an instance as far away as the preallocation: rare, so
        // any free entry will do, and lookups scan while one is displaced.
        index = table.size();
        for (std::size_t i = 0; i < table.size() && index == table.size(); ++i)
          if (!table[i].used)
            index = i;
      }
      // The preallocation covers the most a record can publish; should it
      // fall short, grow rather than fail. The node's FINAL flush says so,
      // once, outside the lock. A deque keeps entries in use where they are.
      if (index == table.size()) {
        if (join_grew[si] == join_growth::none)
          join_grew[si] = join_growth::unreported;
        add_join_entry(si);
        // Live instances may now lie as far apart as the table holds, and
        // the dropped ring covers twice that.
        dropped[si].grow(2 * table.size());
      }
    }
    auto *entry = &table[index];
    if (fresh) {
      entry->used = true;
      entry->live_at = join_live[si].size();
      join_live[si].push_back(index);
      entry->displaced = index != join_home_of(si, k);
      join_displaced[si] += entry->displaced;
      entry->k = k;
      entry->pending = 0;
      entry->operation = 0;
      for (std::size_t i = 0; i < entry->state.size(); ++i) {
        // A producer index below 0 never publishes.
        const bool absent = k + join_offsets[si][i] < 0;
        if (absent && join_required[si][i]) {
          release_join_locked(si, *entry);
          drop_instance_locked(c, id, si, k, event.root);
          return; // an instance without a required input does not exist
        }
        entry->state[i] = absent ? join_entry::absent : join_entry::missing;
        entry->pending += !absent;
      }
    }
    entry->operation = std::max(entry->operation, event.operation);
    if (entry->state[slot] == join_entry::present)
      throw std::logic_error("decoding_task_graph: node " +
                             std::to_string(node.node_index) + " input " +
                             std::to_string(slot) + " of instance " +
                             std::to_string(k) + " was delivered twice");
    // An input already known absent changes nothing when told so again, and
    // the delivery that completed the instance has queued its one fire;
    // queuing another would fire it twice. A value for it is a producer
    // publishing an index it said never comes.
    if (entry->state[slot] == join_entry::absent) {
      if (value)
        throw std::logic_error(
            "decoding_task_graph: node " + std::to_string(node.node_index) +
            " input " + std::to_string(slot) + " of instance " +
            std::to_string(k) + " was delivered after it was known absent");
      if (!fresh || entry->pending != 0)
        return;
    } else {
      --entry->pending;
    }
    if (value) {
      if (buffer)
        std::swap(entry->inputs[slot], *buffer);
      else
        entry->inputs[slot] = *value;
      entry->state[slot] = join_entry::present;
    } else {
      entry->state[slot] = join_entry::absent;
      if (join_required[si][slot]) {
        release_join_locked(si, *entry);
        drop_instance_locked(c, id, si, k, event.root);
        return;
      }
    }
    if (entry->pending != 0) {
      const auto anchor = feedback_anchor[si];
      if (fresh && c.kind == phase::flush && anchor != no_event) {
        auto &group = feedback_flushes[anchor];
        if (group.resolving) {
          group.pending.emplace_back(k, si, index);
          std::push_heap(group.pending.begin(), group.pending.end(),
                         std::greater<>{});
        }
      }
      return;
    }
    queue_indexed_fire_locked(c, fire_parent(c, id), si, index, event.root);
    notify_workers();
  }

  bool is_dropped(std::size_t si, std::int64_t k) const {
    return dropped[si].contains(k);
  }

  // Instance k's place in node si's join table when it is not displaced.
  std::size_t join_home_of(std::size_t si, std::int64_t k) const {
    return static_cast<std::size_t>(k) % join_home[si];
  }

  // The entry of instance k (k >= 0) of node si, or the table's size if it
  // has none. Live instances lie closer than the preallocation, so each
  // sits at its home; only while one is displaced does a lookup scan.
  std::size_t find_join_locked(std::size_t si, std::int64_t k) const {
    const auto &table = joins[si];
    const auto home = join_home_of(si, k);
    if (table[home].used && table[home].k == k)
      return home;
    if (join_displaced[si] == 0)
      return table.size();
    for (const auto i : join_live[si])
      if (table[i].displaced && table[i].k == k)
        return i;
    return table.size();
  }

  // Free a join entry of node si, swapping the last used entry into its
  // place in join_live. Freeing a free one does nothing.
  void release_join_locked(std::size_t si, join_entry &entry) {
    if (!entry.used)
      return;
    auto &live = join_live[si];
    const auto moved = live.back();
    live[entry.live_at] = moved;
    joins[si][moved].live_at = entry.live_at;
    live.pop_back();
    join_displaced[si] -= entry.displaced;
    entry.displaced = false;
    entry.used = false;
  }

  // Instance `k` of node `si` never exists: tell its indexed consumers once,
  // so the instances reading it stop waiting for it.
  void drop_instance_locked(context &c, std::size_t parent, std::size_t si,
                            std::int64_t k, std::size_t root) {
    if (is_dropped(si, k))
      return;
    if (required_drop_chain[si] == drop_chain::cyclic)
      throw std::logic_error(
          "decoding_task_graph: missing input propagates around a cycle of "
          "required inputs");
    dropped[si].add(k);
    const auto node_idx = owner.streaming_nodes[si].node_index;
    const auto &wires = owner.streaming_wires_from[node_idx];
    auto &delivery = c.slots[parent];
    if (!defer_drop_notifications[si] && delivery.first_delivery == no_event &&
        wires.size() <= c.free_slots.size()) {
      publish_wires_locked(c, parent, root, wires, k, /*absent=*/true);
      notify_workers();
      return;
    }

    // A cascade whose estimate filled the store may branch on every lap.
    // File its absences here and deliver them one at a time in this slot,
    // keeping the parent and reservation until the whole queue drains.
    for (const auto &wire : wires) {
      const auto consumer =
          static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
      if (!indexed(consumer))
        continue;
      graph_event event{graph_event_kind::streaming_edge, consumer, wire, 0,
                        root};
      event.index = k;
      event.absent = true;
      event.operation = delivery.event.operation;
      defer_delivery_locked(c, delivery, event);
    }
  }

  // Schedule a delivery to borrow its producer's slot after children drain.
  void defer_delivery_locked(context &c, slot &producer,
                             const graph_event &event,
                             edge_value *buffer = nullptr) {
    const context::pending_delivery item{event, no_event, buffer};
    auto next = c.free_delivery;
    if (next == no_event) {
      next = c.pending_deliveries.size();
      c.pending_deliveries.push_back(item);
    } else {
      auto &pending = c.pending_deliveries[next];
      c.free_delivery = pending.next;
      pending = item;
    }
    // A replayed delivery may defer children of its own. They must drain
    // before the siblings still queued in this slot.
    if (producer.last_child_delivery == no_event) {
      c.pending_deliveries[next].next = producer.first_delivery;
      producer.first_delivery = next;
    } else {
      auto &last_child = c.pending_deliveries[producer.last_child_delivery];
      c.pending_deliveries[next].next = last_child.next;
      last_child.next = next;
    }
    producer.last_child_delivery = next;
    ++c.node_live[event.node];
  }

  // An absence on a loop edge keeps its enclosing indexed subtree alive.
  // Attach it directly there so each lap releases the intervening slots.
  // If the subtree owns a reservation, keep the delivery inside it.
  std::size_t feedback_notification_parent(const context &c,
                                           std::size_t parent) const {
    auto boundary = parent;
    for (auto ancestor = parent; ancestor != no_event;
         ancestor = c.slots[ancestor].parent) {
      const auto &candidate = c.slots[ancestor];
      const auto &event = candidate.event;
      if (candidate.fire_boundary ||
          (event.kind == graph_event_kind::streaming_edge &&
           indexed(event.node)))
        boundary = ancestor;
      if (candidate.reserved_slots != 0)
        break;
    }
    return boundary;
  }

  // Space left after accounting for work already queued or reserved. A
  // fire starting in an existing unreserved slot can reuse that slot.
  std::size_t available_fire_slots(const context &c,
                                   std::size_t reusable = 0) const {
    const auto committed = c.reserved_slots + c.unreserved_slots - reusable;
    return c.slots.size() - std::min(c.slots.size(), committed);
  }

  // Charge a detached fire and its descendants to one slot reservation.
  void reserve_fire_slots_locked(context &c, slot &fire, std::size_t slots) {
    fire.covered = true;
    fire.reserved_slots = slots;
    --c.unreserved_slots;
    c.reserved_slots += slots;
  }

  // The parent of a fire that delivery @p id completes in record @p c, of
  // the node it delivers to. Its join entry holds copies of its inputs, so
  // in an enqueue record no chunk limit cut, rounds admitted together
  // included, it needs none and ingest goes on while it decodes, as long as
  // the record's fires without a parent, each with the slots it holds until
  // its deliveries drain, fit in half the event store, and none of its
  // deliveries can reach a node that joins by record. A feedback cycle
  // keeps its fires attached so each lap drains before the record advances.
  std::size_t fire_parent(const context &c, std::size_t id) const {
    const auto si = c.slots[id].event.node;
    const bool one_chunk = owner.record_bytes != 0 &&
                           c.len <= owner.record_bytes &&
                           c.one_chunk.load(std::memory_order_relaxed);
    const auto budget = c.slots.size() / 2;
    const bool room = c.detached_slots <= budget &&
                      fire_slots[si] <= budget - c.detached_slots &&
                      fire_slots[si] <= available_fire_slots(c);
    const bool detach = c.kind == phase::enqueue && one_chunk && room &&
                        !reaches_ordered[si] && feedback_anchor[si] == no_event;
    return detach ? no_event : id;
  }

  // The fire of join entry @p entry of node @p si, on behalf of @p root.
  graph_event indexed_fire_event(std::size_t si, std::size_t entry,
                                 std::size_t root) const {
    graph_event fire{graph_event_kind::streaming_indexed_fire, si, {}, 0, root};
    fire.index = joins[si][entry].k;
    fire.entry = entry;
    fire.operation = joins[si][entry].operation;
    return fire;
  }

  // Queue the fire of join entry @p entry of node @p si in record @p c, as a
  // child of @p parent, or of none. A flush resolving joins passes itself,
  // so it continues only once the fires it queued have drained the store.
  // If a loop-carried delivery cannot detach its fire, let the outermost
  // fire's subtree drain before reusing that fire's slot. This releases all
  // intermediate fires and deliveries from a lap of the feedback loop,
  // while the slot's parent still waits for the feedback to finish.
  void queue_indexed_fire_locked(context &c, std::size_t parent, std::size_t si,
                                 std::size_t entry, std::size_t root) {
    const auto fire = indexed_fire_event(si, entry, root);
    if (parent != no_event &&
        c.slots[parent].event.kind == graph_event_kind::streaming_edge &&
        c.slots[parent].event.wire.loop_carried()) {
      auto boundary = c.slots[parent].fire_boundary ? parent : no_event;
      for (auto ancestor = c.slots[parent].parent; ancestor != no_event;
           ancestor = c.slots[ancestor].parent)
        if (c.slots[ancestor].fire_boundary)
          boundary = ancestor;
      if (boundary != no_event) {
        defer_fire_locked(c, c.slots[boundary], fire);
        return;
      }
    }
    const auto id = push_locked(c, fire, parent);
    if (id == no_event || parent != no_event)
      return;
    // Keep the reservation and root open until the slot's last fire drains.
    reserve_fire_slots_locked(c, c.slots[id], fire_slots[si]);
    c.detached_slots += fire_slots[si];
    if (root < c.root_open.size())
      ++c.root_open[root];
  }

  // Keep deferred fires in their live join entries until the shared slot
  // can run them. Count each fire now so flush waits for it too.
  void defer_fire_locked(context &c, slot &boundary, const graph_event &fire) {
    auto &deferred = joins[fire.node][fire.entry].feedback;
    deferred = {fire};
    if (boundary.last_feedback)
      boundary.last_feedback->next = &deferred;
    else
      boundary.first_feedback = &deferred;
    boundary.last_feedback = &deferred;
    ++c.node_live[fire.node];
    ++c.node_fires_live[fire.node];
  }

  // The shot has ended and every producer of the node has finished, so no
  // pending input will arrive: instances with their required inputs fire
  // with the rest absent, and the others never existed.
  // The table may have outgrown the event store. Keep space for one fire's
  // subtree as well as the queued descriptors, and report when a later
  // pass must continue.
  bool resolve_joins_locked(context &c, std::size_t parent, std::size_t si,
                            std::size_t root) {
    auto &table = joins[si];
    // The used entries in table order, the order a full scan took them;
    // releasing one below reorders join_live, not this copy.
    auto &scan = join_scan[si];
    scan.assign(join_live[si].begin(), join_live[si].end());
    std::sort(scan.begin(), scan.end());
    bool queued = false;
    for (const auto index : scan) {
      auto &entry = table[index];
      if (!entry.used || entry.pending == 0)
        continue;
      // Each descriptor uses a slot until it runs. Leave enough room for
      // an already queued fire, which can reuse its own slot to dispatch.
      // Start with at least one even if the estimate is saturated.
      if (queued && (c.free_slots.size() <= c.slots.size() / 2 ||
                     available_fire_slots(c) < fire_slots[si]))
        return true;
      if (c.free_slots.empty())
        return true;
      queued |= resolve_join_locked(c, parent, si, index, root);
    }
    return false;
  }

  // Close missing inputs and queue one fire if all required values arrived.
  bool resolve_join_locked(context &c, std::size_t parent, std::size_t si,
                           std::size_t index, std::size_t root) {
    auto &entry = joins[si][index];
    for (std::size_t i = 0; i < entry.state.size(); ++i)
      if (entry.state[i] == join_entry::missing && join_required[si][i]) {
        release_join_locked(si, entry);
        return false;
      }
    const auto *joiner = owner.streaming_nodes[si].indexed.front();
    for (std::size_t i = 0; i < entry.state.size(); ++i)
      if (entry.state[i] == join_entry::missing &&
          joiner->input_guaranteed(i, entry.k))
        throw std::runtime_error(
            "decoding_task_graph: node " +
            std::to_string(owner.streaming_nodes[si].node_index) + " input " +
            std::to_string(i) + " of instance " + std::to_string(entry.k) +
            " never arrived, though its producer instance exists");
    for (auto &state : entry.state)
      if (state == join_entry::missing)
        state = join_entry::absent;
    entry.pending = 0;
    queue_indexed_fire_locked(c, parent, si, index, root);
    return true;
  }

  // Outside producers have finished. Every internal dependency runs from
  // a lower index, or from an earlier node at the same index. Resolve joins
  // in that order and drain each fire before considering its consumers.
  bool resolve_feedback_joins_locked(context &c, std::size_t parent,
                                     const graph_event &event) {
    auto &group = feedback_flushes[event.node];
    auto &pending = group.pending;
    if (!group.resolving) {
      pending.clear();
      for (const auto node : group.members)
        for (const auto index : join_live[node]) {
          const auto &entry = joins[node][index];
          if (entry.pending != 0)
            pending.emplace_back(entry.k, node, index);
        }
      std::make_heap(pending.begin(), pending.end(), std::greater<>{});
      group.resolving = true;
    } // end - if(!group.resolving)

    while (!pending.empty()) {
      const auto [k, node, index] = pending.front();
      const auto &entry = joins[node][index];
      const bool live = entry.used && entry.k == k && entry.pending != 0;
      if (live && c.free_slots.empty())
        return true;
      std::pop_heap(pending.begin(), pending.end(), std::greater<>{});
      pending.pop_back();
      if (live && resolve_join_locked(c, parent, node, index, event.root))
        return true;
    } // end - while(!pending.empty())
    group.resolving = false;
    return false;
  } // end - resolve_feedback_joins_locked()

  // Whether an indexed node's flush may resolve its instances: no delivery
  // to it is in flight. Its own fires may still run unless a loop-carried
  // edge feeds it back; then, and for the flush's second pass, this is
  // flush_ready_locked().
  bool flush_may_start_locked(const context &self,
                              const graph_event &flush) const {
    const auto si = flush.node;
    const auto anchor = feedback_anchor[si];
    if (anchor == si)
      return std::all_of(feedback_flushes[si].members.begin(),
                         feedback_flushes[si].members.end(),
                         [&](std::size_t node) {
                           return quiescent_locked(self, node, node == si);
                         });
    if (anchor != no_event)
      return quiescent_locked(self, si);
    if (!indexed(si) || self_fed[si] || flush.offset == flush_resolved)
      return flush_ready_locked(self, si);
    // Live events may be only fires, plus the asking flush in `self`;
    // compared, not subtracted, so no count can wrap.
    for (const auto *c : active) {
      const std::uint32_t allowed =
          c->node_fires_live[si] + (c == &self ? 1u : 0u);
      if (c->node_live[si] != allowed)
        return false;
    }
    return true;
  }

  // Whether no context other than `self` has an event targeting node si.
  bool others_quiet_locked(const context &self, std::size_t si) const {
    for (const auto *c : active)
      if (c != &self && c->node_live[si] != 0)
        return false;
    return true;
  }

  // Whether no event of any record other than `self`'s flush still targets
  // the pooled or indexed node: its producers' deliveries have all been filed
  // or run and its own fires have all finished. With @p asking, one of
  // `self`'s events is the flush of node `si` asking.
  bool quiescent_locked(const context &self, std::size_t si,
                        bool asking = true) const {
    for (const auto busy : replica_busy[si])
      if (busy)
        return false;
    for (const auto *c : active) {
      auto live = c->node_live[si];
      if (asking && c == &self)
        --live; // the flush event asking
      if (live != 0)
        return false;
    }
    return true;
  }

  // Whether the flush of pooled or indexed node `si` may resolve it: the
  // node is quiescent, and so is every node feeding it through a
  // loop-carried wire, whose fires may yet publish into its pending
  // instances. Such a producer is later in node order, so it flushes after
  // `si`; waiting for its flush would never end.
  bool flush_ready_locked(const context &self, std::size_t si) const {
    if (!quiescent_locked(self, si))
      return false;
    for (const auto producer : owner.streaming_loop_producers[si])
      if (!quiescent_locked(self, producer, /*asking=*/false))
        return false;
    return true;
  }

  // Run instance k on the replica claimed at dispatch, with the entry's
  // inputs, then free the entry.
  bool execute_indexed_fire(context &c, std::size_t id,
                            const graph_event &event) {
    const auto si = event.node;
    const auto &node = owner.streaming_nodes[si];
    const auto replica = c.slots[id].pool_slot;
    auto &entry = *fire_entries[si][replica];
    auto &inputs = fire_inputs[si][replica];
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] =
          entry.state[i] == join_entry::present ? &entry.inputs[i] : nullptr;
    auto *capability = node.replicas[replica];
    capability->clear_corrections(); // its corrections become this delta
    // The entry is freed once: by finish_locked() when this returns, or by
    // fail() if the fire or its publication throws.
    const bool fired = node.indexed[replica]->fire(event.index, inputs);
    // A failure the node reported takes the path of a thrown one.
    if (const auto *failure = capability->fire_failure())
      throw std::runtime_error(*failure);
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
    const auto *value = delivered_value(event, c.slots[id].buffer);
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
      if (event.next_skipped != no_index) {
        execute_skipped_index(c, id, event);
        return;
      }
      if (deliver_streaming_edge(c, id, event))
        c.fired.store(true, std::memory_order_relaxed);
      return;
    case graph_event_kind::streaming_flush:
      execute_flush(c, id, event);
      return;
    case graph_event_kind::streaming_flush_release:
      release_dependencies(c, flush_dependents[event.node],
                           graph_event_kind::streaming_flush);
      return;
    case graph_event_kind::streaming_indexed_fire:
      if (execute_indexed_fire(c, id, event))
        c.fired.store(true, std::memory_order_relaxed);
      return;
    }
    throw std::logic_error("decoding_task_graph: unexpected graph event");
  }

  // Drain one skipped index before continuing the original value delivery.
  void execute_skipped_index(context &c, std::size_t id,
                             const graph_event &event) {
    auto missing = event;
    missing.index = event.next_skipped;
    missing.next_skipped = no_index;
    missing.absent = true;
    push(c, missing, id);
  }

  // The final predecessor publishes a dependent while holding the runtime
  // lock, so all predecessor output writes happen before the dependent is
  // dispatched.
  void execute_one_shot(context &c, const graph_event &event) {
    const auto node = event.node;
    auto &inputs = owner.parallel_node_inputs[node];
    // A node reaching only the logical output still releases its dependents,
    // which are skipped too, so the walk completes.
    if (!owner.one_shot_skips[node]) {
      for (const auto &wire : owner.wires_to[node])
        inputs[static_cast<std::size_t>(wire.to_slot)] =
            wire.from_node < 0
                ? *c.measurement
                : owner.node_outputs[static_cast<std::size_t>(wire.from_node)]
                                    [static_cast<std::size_t>(wire.from_slot)];
      owner.nodes[node]->execute_indexed(inputs, owner.node_outputs[node]);
    }
    release_dependencies(c, owner.dependents[node], graph_event_kind::one_shot);
  }

  void execute_raw(context &c, std::size_t id, const graph_event &event) {
    auto &node = *owner.streaming_nodes[event.node].capability;
    const auto node_index = owner.streaming_nodes[event.node].node_index;
    const bool no_deliveries = owner.streaming_wires_from[node_index].empty() &&
                               owner.completion_wires_from[node_index].empty();
    auto offset = event.offset;
    auto operation = event.operation;
    while (true) {
      const auto end = c.operations > 1
                           ? c.operation_ends[operation - c.first_operation]
                           : c.len;
      const auto limit =
          c.type ? node.raw_chunk_limit() : node.untagged_chunk_limit();
      const auto take = limit ? std::min(*limit, end - offset) : end - offset;
      // Another chunk follows. Checked every chunk, as a limit can shrink
      // below record_bytes as the node fills.
      if (limit && *limit < c.len - offset)
        c.one_chunk.store(false, std::memory_order_relaxed);
      const auto *chunk = c.syndrome ? c.syndrome + offset : nullptr;
      const auto next = offset + take;
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
      if (next < c.len && take != 0) {
        // Recheck capacity after each chunk and let its causal deliveries
        // finish before another operation can reuse this context's publication
        // values.
        if (next == end)
          ++operation;
        if (c.operations > 1 && (!fired || no_deliveries)) {
          if (stop.load(std::memory_order_acquire))
            return;
          // No deliveries means no children can read this running event's
          // operation tag. Captured results remain owned by this context.
          offset = next;
          c.slots[id].event.operation = operation;
          continue;
        }
        auto continuation = event;
        continuation.offset = next;
        continuation.operation = operation;
        set_continuation(c, id, continuation);
      }
      return;
    }
  }

  // Publish any data the node commits when its last input is known.
  void flush_node(context &c, std::size_t id, const graph_event &event) {
    if (owner.streaming_nodes[event.node].capability->flush())
      c.fired.store(publish_all(c, id, event.root, event.node),
                    std::memory_order_relaxed);
    else if (!pooled(event.node) && !indexed(event.node) && !staged(event.node))
      snapshot_node(c, event.node); // a pooled copy holds only its last delta
  }

  // Drain the node or feedback group before running the node's callback.
  void execute_flush(context &c, std::size_t id, const graph_event &event) {
    if (indexed(event.node)) {
      bool more;
      {
        auto lock = lock_runtime();
        more = feedback_anchor[event.node] == event.node
                   ? resolve_feedback_joins_locked(c, id, event)
                   : resolve_joins_locked(c, id, event.node, event.root);
      }
      if (more) { // reuse this slot once the queued fires have drained
        set_continuation(c, id, event);
        return;
      }
      // Resolved while earlier records' fires may still run (see
      // flush_may_start_locked): end the node's shot once they finish.
      if (event.offset != flush_resolved) {
        bool settled = true;
        {
          auto lock = lock_runtime();
          settled = others_quiet_locked(c, event.node);
        }
        if (!settled) {
          auto resolved = event;
          resolved.offset = flush_resolved;
          set_continuation(c, id, resolved);
          return;
        }
      }
      // Every instance of the shot is resolved; the next shot's numbering
      // starts over.
      bool grew = false;
      {
        auto lock = lock_runtime();
        dropped[event.node].clear();
        grew = join_grew[event.node] == join_growth::unreported;
        if (grew)
          join_grew[event.node] = join_growth::reported;
      }
      if (grew)
        CUDA_QEC_WARN("decoding_task_graph: the join table of node {} grew "
                      "past its {} preallocated entries",
                      owner.streaming_nodes[event.node].node_index,
                      join_home[event.node]);
    } // end - if(indexed(event.node))
    flush_node(c, id, event);
    {
      // The flush was the shot's last publication on the node's wires; the
      // next shot's indices start a new run.
      auto lock = lock_runtime();
      const auto node_idx = owner.streaming_nodes[event.node].node_index;
      for (const auto &wire : owner.streaming_wires_from[node_idx])
        if (wire.id >= 0)
          wire_last[static_cast<std::size_t>(wire.id)] = no_index;
    }
    // Release dependents after both queued and deferred deliveries drain.
    set_continuation(c, id,
                     {graph_event_kind::streaming_flush_release,
                      event.node,
                      {},
                      0,
                      event.root});
  } // end - execute_flush()

  // ----- Worker loops ------------------------------------------------------

  // Run one context to completion with every pool worker, the caller
  // included. Used by the synchronous streaming and one-shot paths.
  void run_single(context &c, int cuda_device_id) {
    fire_cap = fire_cap_for(owner.parallel_pool->num_threads());
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
    // Worker zero dispatches; the rest execute events.
    fire_cap = fire_cap_for(owner.parallel_pool->num_threads() - 1);
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
    // The event this worker ran last, finished under the next lock it takes.
    context *done_context = nullptr;
    std::size_t done_id = no_event;
    try {
      detail_affinity::CudaDeviceGuard device_guard(cuda_device_id);
      while (true) {
        context *c = nullptr;
        std::size_t id = no_event;
        graph_event event;
        {
          auto lock = lock_runtime();
          if (done_context) {
            finish_locked(*done_context, done_id);
            done_context = nullptr;
          }
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
        {
          NVTX_RANGE_INDEXED(event_label(event), event.index);
          execute(*c, id, event);
        }
        done_context = c;
        done_id = id;
        active_context = nullptr;
        active_id = no_event;
      }
    } catch (...) {
      fail(active_context, active_id, std::current_exception());
    }
  }

  // ----- Dispatcher support ------------------------------------------------

  // Fires @p executing workers may run at once: all but one, at least one.
  static std::size_t fire_cap_for(std::size_t executing) {
    return executing > 1 ? executing - 1 : 1;
  }

  std::unique_lock<std::mutex> lock_runtime() {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (has_waiting_workers)
      lock.lock();
    return lock;
  }

  // Only the async coordinator admits, activates and releases contexts, so it
  // reads `active` and `in_use` without the lock; workers read them under it.

  // How many of the oldest active contexts have finished executing, in order.
  // Coordinator only.
  std::size_t complete_prefix() const {
    std::size_t n = 0;
    while (n < active.size() &&
           active[n]->complete.load(std::memory_order_acquire))
      ++n;
    return n;
  }

  // Release the @p n oldest active contexts, which have retired.
  void release_prefix_locked(std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      active[i]->in_use = false;
      active[i]->kind = phase::idle;
    }
    active.erase(active.begin(),
                 active.begin() + static_cast<std::ptrdiff_t>(n));
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
  // Group feedback cycles and order FINAL after their outside producers.
  void configure_feedback_flush() {
    const auto size = owner.streaming_nodes.size();
    flush_dependents = owner.streaming_dependents;
    flush_producers = owner.streaming_producers;
    flush_predecessor_counts = owner.streaming_predecessor_counts;
    feedback_anchor.assign(size, no_event);

    std::vector<std::pair<std::size_t, std::size_t>> backward;
    for (std::size_t target = 0; target < size; ++target) {
      const auto node = owner.streaming_nodes[target].node_index;
      for (const auto &wire : owner.wires_to[node])
        if (wire.loop_carried() && owner.streaming_by_node[wire.from_node] >= 0)
          backward.emplace_back(owner.streaming_by_node[wire.from_node],
                                target);
    }
    if (backward.empty())
      return;

    // Find the strongly connected components using the full streaming graph.
    auto dependents = owner.streaming_dependents;
    auto producers = owner.streaming_producers;
    for (const auto &[source, target] : backward) {
      dependents[source].push_back(target);
      producers[target].push_back(source);
    }
    std::vector<uint8_t> visited(size, 0);
    std::vector<std::size_t> order;
    const auto visit = [&](auto &&self, std::size_t node) -> void {
      if (visited[node])
        return;
      visited[node] = 1;
      for (const auto next : dependents[node])
        self(self, next);
      order.push_back(node);
    };
    for (std::size_t node = 0; node < size; ++node)
      visit(visit, node);

    std::vector<std::size_t> component(size, no_event);
    const auto collect = [&](auto &&self, std::size_t node, std::size_t leader,
                             std::vector<std::size_t> &members) -> void {
      if (component[node] != no_event)
        return;
      component[node] = leader;
      members.push_back(node);
      for (const auto producer : producers[node])
        self(self, producer, leader, members);
    };
    const auto add_dependency = [&](std::size_t source, std::size_t target) {
      flush_dependents[source].push_back(target);
      flush_producers[target].push_back(source);
      ++flush_predecessor_counts[target];
    };
    feedback_flushes.resize(size);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      if (component[*it] != no_event)
        continue;
      std::vector<std::size_t> members;
      collect(collect, *it, *it, members);
      const bool cycle = members.size() > 1 ||
                         std::find(producers[*it].begin(), producers[*it].end(),
                                   *it) != producers[*it].end();
      if (!cycle)
        continue;
      for (const auto member : members) {
        const auto &id = owner.streaming_nodes[member].id;
        if (!indexed(member))
          owner.record_streaming_conflict("record-ordered node '" + id +
                                          "' in a feedback cycle");
        else if (std::none_of(join_required[member].begin(),
                              join_required[member].end(),
                              [](bool required) { return required; }))
          owner.record_streaming_conflict("indexed node '" + id +
                                          "' in a feedback cycle needs a "
                                          "required input");
      }

      // One flush resolves the group's instances before its other node
      // flushes can run. Wait for every producer outside the group first.
      const auto anchor = *std::min_element(members.begin(), members.end());
      auto &group = feedback_flushes[anchor];
      for (const auto member : members) {
        feedback_anchor[member] = anchor;
        if (member != anchor)
          add_dependency(anchor, member);
        for (const auto producer : producers[member])
          if (component[producer] != component[member]) {
            group.producers.push_back(producer);
            add_dependency(producer, anchor);
          }
      }
      std::sort(group.producers.begin(), group.producers.end());
      group.producers.erase(
          std::unique(group.producers.begin(), group.producers.end()),
          group.producers.end());
      std::sort(members.begin(), members.end());
      std::size_t capacity = 0;
      for (const auto member : members)
        capacity += joins[member].size();
      group.pending.reserve(capacity);
      group.members = std::move(members);
    }
    // A backward edge between components can wait for its producer's full
    // flush, including publications it makes while resolving its joins.
    for (const auto &[source, target] : backward)
      if (component[source] != component[target])
        add_dependency(source, target);

    record_dependents = std::move(dependents);
    record_producers = std::move(producers);
  }

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
    created.fire_boundary =
        event.kind == graph_event_kind::streaming_indexed_fire;
    // At least its record's first operation, and its parent's.
    created.event.operation = std::max(event.operation, c.first_operation);
    if (parent != no_event)
      created.event.operation =
          std::max(created.event.operation, c.slots[parent].event.operation);
    created.parent = parent;
    created.blocked_by = blocked_by;
    created.sequence = c.next_sequence++;
    created.wait_for_siblings = wait_for_siblings;
    created.occupied = true;
    created.covered = parent != no_event && c.slots[parent].covered;
    c.unreserved_slots += !created.covered;
    ++c.occupied;
    const auto target = release_target(event);
    if (target != no_event) {
      ++c.node_live[target];
      c.node_fires_live[target] +=
          event.kind == graph_event_kind::streaming_indexed_fire;
    }
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
      if (!selected.covered)
        reserve_fire_slots_locked(c, selected, fire_slots[si]);
      selected.pool_slot = pool_free[si].back();
      pool_free[si].pop_back();
      replica_busy[si][selected.pool_slot] = 1;
      selected.releases_pool = true;
      fire_entries[si][selected.pool_slot] = &joins[si][selected.event.entry];
      ++running_fires;
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
    if (done.event.kind == graph_event_kind::streaming_indexed_fire &&
        running_fires != 0)
      --running_fires;
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

  // Branches that meet at all, which orders unrelated raw roots.
  bool branches_overlap(std::size_t left, std::size_t right) const {
    return owner.streaming_overlap[left][right] !=
           impl::streaming_overlap_kind::disjoint;
  }
  // Branches that meet at a descendant needing ordered delivery, which
  // orders sibling deliveries.
  bool branches_ordered(std::size_t left, std::size_t right) const {
    return owner.streaming_overlap[left][right] ==
           impl::streaming_overlap_kind::ordered;
  }

  bool root_is_ready(const context &c, const graph_event &event,
                     const slot &candidate) const {
    // A replayed delivery is logically a child even in a root's slot.
    if (event.root == no_event ||
        (candidate.parent == no_event && !candidate.replaying_delivery))
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
      if (candidate.event.kind == graph_event_kind::streaming_indexed_fire) {
        const auto si = candidate.event.node;
        // Leave a worker for other events. An uncovered fire needs space for
        // its subtree alongside existing reservations. Let it start if
        // there are no other reservations, even if its estimate saturated.
        if (pool_free[si].empty() || running_fires >= fire_cap ||
            (!candidate.covered && c.reserved_slots != 0 &&
             fire_slots[si] > available_fire_slots(c, 1)))
          continue;
      }
      // Resolving an indexed node's pending instances needs every delivery
      // to it, from any record and any loop-carried producer, filed first.
      if (candidate.event.kind == graph_event_kind::streaming_flush &&
          targets_pool(candidate.event) &&
          !flush_may_start_locked(c, candidate.event))
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
    const bool flushing = c.kind == phase::flush;
    const auto &dependents =
        flushing ? flush_dependents
                 : (record_dependents.empty() ? owner.streaming_dependents
                                              : record_dependents);
    const auto anchor = feedback_anchor[si];
    if (anchor != no_event) {
      const auto &group = feedback_flushes[anchor];
      // One member can generate new work for another over a backward edge.
      // Advance every member's turn together once all such work has drained.
      for (const auto member : group.members)
        if (c.node_live[member] != 0 || node_turn[member] != c.sequence)
          return;
      for (const auto producer : group.producers)
        if (!c.node_released[producer])
          return;
      for (const auto member : group.members) {
        c.node_released[member] = 1;
        node_turn[member] = c.sequence + 1;
      }
      for (const auto member : group.members)
        advance_record_dependents_locked(c, member, dependents[member]);
      return;
    }
    const auto &producers =
        flushing ? flush_producers
                 : (record_producers.empty() ? owner.streaming_producers
                                             : record_producers);
    for (const auto producer : producers[si])
      if (!c.node_released[producer])
        return;
    c.node_released[si] = 1;
    node_turn[si] = c.sequence + 1;
    advance_record_dependents_locked(c, si, dependents[si]);
  }

  // Check which dependents and following records can take the next turn.
  void
  advance_record_dependents_locked(context &c, std::size_t si,
                                   const std::vector<std::size_t> &dependents) {
    for (const auto dependent : dependents)
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
    c.complete.store(true, std::memory_order_release);
    if (c.gated)
      owner.wake_async_stream();
  }

  // Finish event @p id, which ran to completion, freeing a fire's join
  // entry. Called under the lock its worker takes to select the next event.
  void finish_locked(context &c, std::size_t id) {
    if (id < c.slots.size() && c.slots[id].running &&
        c.slots[id].event.kind == graph_event_kind::streaming_indexed_fire) {
      const auto si = c.slots[id].event.node;
      release_join_locked(si, *fire_entries[si][c.slots[id].pool_slot]);
    }
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
    if (completed.event.next_skipped == no_index &&
        (!completed.has_continuation || completed.replaying_delivery)) {
      const auto target = release_target(completed.event);
      if (target != no_event)
        c.node_fires_live[target] -=
            completed.event.kind == graph_event_kind::streaming_indexed_fire;
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
    auto &continued = c.slots[id];
    if (continued.event.next_skipped != no_index) {
      // The value's buffer stays here until the entire gap has drained.
      if (++continued.event.next_skipped == continued.event.index)
        continued.event.next_skipped = no_index;
    } else {
      release_wire_buffer(c, continued);
      continued.replaying_delivery = false;
      if (continued.first_delivery != no_event) {
        const auto first = continued.first_delivery;
        auto &pending = c.pending_deliveries[first];
        continued.event = pending.event;
        continued.buffer = pending.buffer;
        continued.replaying_delivery = true;
        continued.first_delivery = pending.next;
        continued.last_child_delivery = no_event;
        pending.next = c.free_delivery;
        c.free_delivery = first;
      } else if (const auto *feedback = continued.first_feedback) {
        continued.event = feedback->event;
        continued.first_feedback = feedback->next;
        if (!continued.first_feedback)
          continued.last_feedback = nullptr;
      } else if (continued.has_continuation) {
        continued.event = continued.continuation;
        continued.continuation = {};
        continued.has_continuation = false;
      } else {
        retire_locked(c, id);
        return;
      }
    }
    continued.blocked_by = no_event;
    continued.sequence = c.next_sequence++;
    continued.queued = false;
    continued.running = false;
    continued.self_done = false;
    continued.wait_for_siblings = false;
    enqueue_locked(c, id);
  }

  // Return a delivery's snapshot after it and its descendants drain.
  void release_wire_buffer(context &c, slot &completed) {
    if (!completed.buffer)
      return;
    auto &pool = c.wire_pool[static_cast<std::size_t>(completed.event.wire.id)];
    pool.free.push_back(completed.buffer);
    completed.buffer = nullptr;
  }

  void retire_locked(context &c, std::size_t id) {
    auto &completed = c.slots[id];
    const auto parent = completed.parent;
    const auto root = completed.event.root;
    const auto reserved_slots = completed.reserved_slots;

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
    c.reserved_slots -= reserved_slots;
    c.unreserved_slots -= !completed.covered;
    completed = {};
    c.free_slots.push_back(id);
    if (c.occupied != 0)
      --c.occupied;
    if (parent == no_event) {
      c.detached_slots -= reserved_slots;
      // Done once its own event and every fire it queued without a parent
      // have finished, so later roots stay behind all of its work.
      if (root != no_event && root < c.root_open.size() &&
          c.root_open[root] != 0 && --c.root_open[root] == 0)
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
      auto &failed = c->slots[id];
      // A fire's entry, which finish_locked() would have freed.
      if (failed.event.kind == graph_event_kind::streaming_indexed_fire)
        release_join_locked(failed.event.node,
                            *fire_entries[failed.event.node][failed.pool_slot]);
      failed.running = false;
      unclaim_locked(*c, failed);
    }
    if (running != 0)
      --running;
    notify_workers();
    // Wake a parked coordinator to end the session, whether or not the
    // failed worker held a record.
    owner.wake_async_stream();
  }

public:
  impl &owner;
  std::vector<context> contexts;
  // Contexts the coordinator has filled and is about to activate together;
  // reserved for all of them.
  std::vector<context *> admitting;
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
    bool displaced = false;  ///< Not at its home, k modulo the preallocation
    std::size_t live_at = 0; ///< its place in join_live, while used
    std::size_t pending = 0;
    std::vector<edge_value> inputs;
    std::vector<uint8_t> state;
    std::uint64_t operation = 0; ///< the latest of its inputs'
    deferred_fire feedback;
  };
  std::vector<std::deque<join_entry>> joins;
  // By streaming node: its used entries' table indices, unordered, so a
  // flush need not scan the whole table; and the flush's scratch to sort
  // them in. Both are reserved with the table.
  std::vector<std::vector<std::size_t>> join_live;
  std::vector<std::vector<std::size_t>> join_scan;
  std::vector<std::vector<std::size_t>> record_producers;
  std::vector<std::vector<std::size_t>> record_dependents;
  std::vector<std::vector<std::size_t>> flush_producers;
  std::vector<std::vector<std::size_t>> flush_dependents;
  std::vector<int32_t> flush_predecessor_counts;
  struct feedback_flush {
    std::vector<std::size_t> members;
    std::vector<std::size_t> producers;
    std::vector<feedback_join> pending;
    bool resolving = false;
  };
  // A feedback group's first node owns its flush work and heap.
  std::vector<std::size_t> feedback_anchor;
  std::vector<feedback_flush> feedback_flushes;
  // By streaming node: whether a loop-carried edge feeds it, so its own
  // fires may still deliver to it.
  std::vector<uint8_t> self_fed;
  // By indexed node: the largest subtree its slot may run, including
  // feedback, capped at the event store capacity.
  std::vector<std::size_t> fire_slots;
  // By streaming node: for an indexed one, whether its fires can deliver to
  // a node that joins its inputs by record (see fire_parent()).
  std::vector<uint8_t> reaches_ordered;
  // By streaming node: whether its join table grew, and whether that was
  // reported, which happens once.
  enum class join_growth : uint8_t { none, unreported, reported };
  std::vector<join_growth> join_grew;
  // By streaming node: the preallocated entries, which instance k's home
  // (k modulo it) indexes, and the used entries not at their home.
  std::vector<std::size_t> join_home;
  std::vector<std::size_t> join_displaced;
  // Edge deliveries each record-ordered topology node has received since
  // its last completion; cleared with the session.
  std::vector<std::size_t> node_deliveries_seen;
  // By streaming node and input slot: the edge's index offset, and whether
  // an instance exists only if that input is delivered.
  std::vector<std::vector<std::int64_t>> join_offsets;
  std::vector<std::vector<uint8_t>> join_required;
  std::vector<drop_chain> required_drop_chain;
  // Dropped instances whose notification subtree can fill the event store.
  std::vector<uint8_t> defer_drop_notifications;
  // Publications whose delivery subtrees can fill the event store.
  std::vector<uint8_t> defer_publications;
  // Producers whose next publication might replace a value still in flight.
  std::vector<uint8_t> copy_publications;
  // The instances found never to exist before FINAL, kept in a ring sized
  // at build so a long shot does not grow it: each is announced to the
  // consumers once and never created afterwards. Instance k sits at k
  // modulo the ring's size, so a drop overwrites only one at least that
  // much older, which receives no delivery. Cleared when the node's FINAL
  // flush has resolved the shot's instances.
  struct dropped_instances {
    std::vector<std::int64_t> ring; ///< no_index where empty
    bool contains(std::int64_t k) const {
      return k >= 0 && !ring.empty() &&
             ring[static_cast<std::size_t>(k) % ring.size()] == k;
    }
    void add(std::int64_t k) {
      if (k >= 0 && !ring.empty())
        ring[static_cast<std::size_t>(k) % ring.size()] = k;
    }
    void clear() { std::fill(ring.begin(), ring.end(), no_index); }
    /// Hold @p size instances, keeping the drops held now. Two may share a
    /// slot of the new size: re-added oldest first, the newer one keeps it.
    void grow(std::size_t size) {
      std::vector<std::int64_t> held;
      for (const auto k : ring)
        if (k != no_index)
          held.push_back(k);
      std::sort(held.begin(), held.end());
      ring.assign(size, no_index);
      for (const auto k : held)
        add(k);
    }
  };
  std::vector<dropped_instances> dropped;
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
    join_live[si].reserve(joins[si].size());
    join_scan[si].reserve(joins[si].size());
    return entry;
  }
  // Per-replica scratch for the input pointers handed to fire().
  std::vector<std::vector<std::vector<const edge_value *>>> fire_inputs;
  // Per streaming node and copy: the join entry its running fire reads, found
  // at claim. A growing deque keeps entries in place, so no lock is needed.
  std::vector<std::vector<join_entry *>> fire_entries;
  // NVTX range names, by streaming node and event kind, and of one-shot
  // events by topology node.
  // One name per graph_event_kind, the last being an indexed fire.
  static constexpr std::size_t event_kinds = 6;
  static_assert(
      static_cast<std::size_t>(graph_event_kind::streaming_indexed_fire) + 1 ==
          event_kinds,
      "kind_names lists every graph_event_kind, in order");
  std::vector<std::array<std::string, event_kinds>> event_labels;
  std::vector<std::string> one_shot_labels;
  // The NVTX range name of @p event.
  const char *event_label(const graph_event &event) const {
    if (event.kind == graph_event_kind::one_shot)
      return one_shot_labels[event.node].c_str();
    return event_labels[event.node][static_cast<std::size_t>(event.kind)]
        .c_str();
  }
  const bool has_waiting_workers;
  std::mutex mutex;
  std::condition_variable wake;
  // Idle workers spin on this, so it keeps a cache line of its own, apart
  // from the mutex every worker takes and the state written under it.
  alignas(64) std::atomic<uint64_t> ready_generation{0};
  alignas(64) std::size_t running = 0;
  // Indexed fires running, and the most that may: one fewer than the
  // executing workers, so other events never queue behind decodes.
  std::size_t running_fires = 0;
  std::size_t fire_cap = 1;
  uint64_t next_record = 0;
  // Written under the runtime lock; the async coordinator reads it without.
  std::atomic<bool> stop{false};
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
  // Records hold one chunk of a single metering raw-input node (see
  // record_bytes): one delivery, as the per-record join tables assume.
  auto &state = async_streaming;
  std::size_t raw_inputs = 0;
  std::size_t chunk = 0;
  bool metered = true;
  // A reads_logical() node takes each record whole, so it sets no chunk size.
  for (std::size_t si = 0; si < streaming_nodes.size(); ++si)
    if (streaming_node_has_raw_input[si] &&
        !nodes[streaming_nodes[si].node_index]->reads_logical()) {
      const auto limit = streaming_nodes[si].capability->raw_chunk_limit();
      ++raw_inputs;
      if (!limit)
        metered = false;
      else
        chunk = *limit;
    }
  record_bytes =
      raw_inputs == 1 && metered ? std::min(chunk, max_record_bytes) : 0;
  state.coalesce = dispatcher_thread && record_bytes != 0;
  if (state.coalesce)
    for (auto &c : scheduler->contexts) {
      c.batch.resize(record_bytes);
      c.operation_ends.resize(state.capacity);
      c.node_published.assign(state.capacity * streaming_nodes.size(), 0);
    }
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
// progress model reads. Generations advance here, in record order, then
// operation order.
std::pair<std::uint64_t, std::uint64_t>
decoding_task_graph::impl::apply_record_context(record_context &c) {
  // One record can finish arbitrarily many interleaved layer instances.
  // Folding by node would fill the bounded stager with A intervals before
  // its B intervals close the gaps. Sort each capture in reverse order of
  // operation, then round, and merge the nodes' tails, reusing their buffers
  // and a node-sized heap.
  const auto M = streaming_nodes.size();
  const auto key = [](const record_context::staged_delta &delta) {
    return std::pair{delta.operation, delta.coverage.begin_round};
  };
  auto &pending = c.staged_nodes;
  const std::vector<uint8_t> no_flips; // a cancelled path's delta
  std::uint64_t advanced = 0;
  std::uint64_t before = streaming_result_state.generation;
  pending.clear();
  for (std::size_t si = 0; si < c.staged_count.size(); ++si) {
    const auto count = c.staged_count[si];
    if (count == 0)
      continue;
    auto &deltas = c.staged[si];
    std::sort(deltas.begin(), deltas.begin() + count,
              [&](const auto &left, const auto &right) {
                return key(left) > key(right);
              });
    pending.push_back(si);
  }
  const auto later = [&](std::size_t left, std::size_t right) {
    return key(c.staged[left][c.staged_count[left] - 1]) >
           key(c.staged[right][c.staged_count[right] - 1]);
  };
  std::make_heap(pending.begin(), pending.end(), later);
  // Operations fold one at a time, counting each but the last that advances
  // the generation; coverage and values publish once, after the last.
  for (std::uint32_t operation = 0; operation < c.operations; ++operation) {
    const bool last = operation + 1 == c.operations;
    while (
        !pending.empty() &&
        (last || c.staged[pending.front()][c.staged_count[pending.front()] - 1]
                         .operation == operation)) {
      std::pop_heap(pending.begin(), pending.end(), later);
      const auto si = pending.back();
      const auto &delta = c.staged[si][--c.staged_count[si]];
      for (const auto k : streaming_node_outputs[si])
        stage_delta(k, delta.coverage, delta.bits);
      for (const auto k : streaming_node_cancelled_outputs[si])
        stage_delta(k, delta.coverage, no_flips);
      if (c.staged_count[si] != 0)
        std::push_heap(pending.begin(), pending.end(), later);
      else
        pending.pop_back();
    }
    for (std::size_t si = 0; si < M; ++si) {
      auto &published = c.node_published[operation * M + si];
      streaming_node_generations[si] += published;
      published = 0;
    }
    if (last)
      break;
    const auto now = pending_generation();
    advanced += now > before;
    before = now;
  } // end - for(operation)
  for (std::size_t si = 0; si < streaming_nodes.size(); ++si) {
    const bool staged =
        si < streaming_node_staged.size() && streaming_node_staged[si];
    if (!staged && c.node_snapshotted[si]) {
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
    latest_node_closure[si] =
        combine_closure(latest_node_closure[si], c.node_closure[si]);
    c.node_closure[si].reset();
    if (c.node_converged[si])
      latest_node_converged[si] =
          latest_node_converged[si].value_or(true) && *c.node_converged[si];
    c.node_converged[si].reset();
  }
  return {advanced, before};
} // end - apply_record_context()

// The least correction output generation: a staged output's stager's,
// another's slowest contributor's. The logical output is not counted.
std::uint64_t decoding_task_graph::impl::pending_generation() const {
  const auto &outputs = streaming_result_state.outputs;
  std::uint64_t generation =
      outputs.empty() ? 0 : std::numeric_limits<std::uint64_t>::max();
  for (std::size_t k = 0; k < num_correction_outputs; ++k) {
    std::uint64_t output = outputs[k].generation;
    if (k < output_stagers.size() && output_stagers[k].enabled) {
      output = std::max(output, output_stagers[k].generation);
    } else if (!output_streaming_nodes[k].empty()) {
      std::uint64_t slowest = std::numeric_limits<std::uint64_t>::max();
      for (const auto si : output_streaming_nodes[k])
        slowest = std::min(slowest, streaming_node_generations[si]);
      output = std::max(output, slowest);
    }
    generation = std::min(generation, output);
  }
  return generation;
}

bool decoding_task_graph::impl::enqueue_sweep(const uint8_t *syndrome,
                                              std::size_t len,
                                              std::optional<round_type> type,
                                              int cuda_device_id) {
  require_streaming_support("enqueue_syndrome");
  // A tagged record longer than record_bytes runs as several, each retired
  // before the next: BULK, then the record's own type for the last. Cuts
  // may fall mid-round: a node that meters its own chunks takes a record
  // cut anywhere (raw_chunk_limit()).
  if (record_bytes != 0 && type && len > record_bytes) {
    bool fired = false;
    for (std::size_t at = 0; at < len; at += record_bytes) {
      const auto piece = std::min(record_bytes, len - at);
      const auto piece_type = at + piece == len ? type : round_type::BULK;
      fired = sweep_record(syndrome + at, piece, piece_type, cuda_device_id) ||
              fired;
    }
    return fired;
  }
  return sweep_record(syndrome, len, type, cuda_device_id);
}

// Run one record through a context of its own and fold it in.
bool decoding_task_graph::impl::sweep_record(const uint8_t *syndrome,
                                             std::size_t len,
                                             std::optional<round_type> type,
                                             int cuda_device_id) {
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
  scheduler->configure_flush(c);
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
      (c.kind == record_context::phase::enqueue && c.opens_record) ||
      c.finish_flush;
  if (begins_operation) {
    result.operation_completed = false;
    retiring_generation = result.generation;
    retiring_revision = result.result_revision;
  }
  // The last of several operations is judged below, from where the others
  // left the generation.
  const auto [earlier_completed, last_from] = apply_record_context(c);
  if (c.operations > 1)
    retiring_generation = last_from;
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
  result.completed_operations +=
      earlier_completed + (result.operation_completed ? 1 : 0);
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
  // Bytes of the record at `admitted` already admitted, while it is cut.
  std::size_t cut_at = 0;
  std::exception_ptr failure;

  const auto fail_session = [&](std::exception_ptr error) {
    if (!failure)
      failure = std::move(error);
    auto lock = S.lock_runtime();
    S.request_stop_locked();
  };

  while (true) {
    bool progressed = false;
    if (S.stop.load(std::memory_order_acquire)) {
      auto lock = S.lock_runtime();
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
    if (state.cancel_requested.load(std::memory_order_acquire)) {
      auto lock = S.lock_runtime();
      S.request_stop_locked();
      continue;
    }

    // Retire finished records, oldest first, then release them together.
    // Marked in NVTX only when there are some, not on every idle pass.
    if (const auto complete = S.complete_prefix(); complete != 0) {
      NVTX_RANGE("task graph retire");
      std::size_t retired = 0;
      for (; retired < complete; ++retired) {
        try {
          retire_record_context(*S.active[retired]);
        } catch (...) {
          fail_session(std::current_exception());
          break;
        }
      }
      if (retired != 0) {
        auto lock = S.lock_runtime();
        S.release_prefix_locked(retired);
        progressed = true;
      }
    }
    if (failure)
      continue;

    // Admit what free contexts allow: fill them without the lock, then
    // activate them under one lock and wake the workers once.
    if (admitted < state.submitted.load(std::memory_order_acquire)) {
      NVTX_RANGE("task graph admit");
      std::exception_ptr admit_error;
      const auto submitted = state.submitted.load(std::memory_order_acquire);
      S.admitting.clear();
      while (admitted < submitted) {
        auto *c = S.try_acquire_for_admission();
        if (!c)
          break;
        auto &slot = state.slots[admitted % state.capacity];
        // A tagged record longer than record_bytes is admitted as several
        // pieces of its slot: BULK but the last, the first opening its
        // operation and the last closing it. Cuts may fall mid-round, as
        // in enqueue_sweep().
        if (record_bytes != 0 && slot.type &&
            (cut_at != 0 || slot.len > record_bytes)) {
          const auto piece = std::min(record_bytes, slot.len - cut_at);
          const bool last = cut_at + piece == slot.len;
          c->kind = record_context::phase::enqueue;
          c->syndrome = slot.syndrome.data() + cut_at;
          c->len = piece;
          c->type = last ? slot.type : round_type::BULK;
          c->opens_record = cut_at == 0;
          c->first_operation = admitted;
          c->ring_sequence = admitted;
          c->closes_record = last && slot.type != round_type::FINAL;
          if (last && slot.type == round_type::FINAL)
            final_pending = true;
          S.admitting.push_back(c);
          cut_at = last ? 0 : cut_at + piece;
          admitted += last ? 1 : 0;
          continue;
        }
        // BULK rounds queued behind this one join its record. FINAL stays a
        // record of its own: its flush counts the shot's last operation.
        std::size_t take = 1;
        std::size_t bytes = slot.len;
        if (state.coalesce && slot.type == round_type::BULK && slot.len != 0)
          while (admitted + take < submitted) {
            const auto &next = state.slots[(admitted + take) % state.capacity];
            if (next.type != round_type::BULK || next.len == 0 ||
                bytes + next.len > c->batch.size())
              break;
            bytes += next.len;
            ++take;
          }
        const auto &last = state.slots[(admitted + take - 1) % state.capacity];
        c->kind = record_context::phase::enqueue;
        c->syndrome = slot.syndrome.data();
        c->len = slot.len;
        if (take > 1) {
          std::size_t at = 0;
          for (std::size_t r = 0; r < take; ++r) {
            const auto &queued = state.slots[(admitted + r) % state.capacity];
            std::memcpy(c->batch.data() + at, queued.syndrome.data(),
                        queued.len);
            at += queued.len;
            c->operation_ends[r] = at;
          }
          c->syndrome = c->batch.data();
          c->len = bytes;
        }
        c->type = last.type;
        c->operations = take;
        c->first_operation = admitted;
        c->ring_sequence = admitted + take - 1;
        c->closes_record = last.type != round_type::FINAL;
        if (last.type == round_type::FINAL)
          final_pending = true;
        S.admitting.push_back(c);
        admitted += take;
      }
      if (!S.admitting.empty()) {
        auto lock = S.lock_runtime();
        std::size_t activated = 0;
        for (; activated < S.admitting.size(); ++activated) {
          auto &c = *S.admitting[activated];
          try {
            S.push_raw_roots_locked(c);
          } catch (...) {
            admit_error = std::current_exception();
            break;
          }
          S.activate_locked(c, /*gated=*/true);
          progressed = true;
        }
        // Release any contexts a failure left unactivated.
        for (std::size_t r = activated; r < S.admitting.size(); ++r)
          S.release_locked(*S.admitting[r]);
        S.notify_workers();
        lock.unlock();
      }
      if (admit_error)
        fail_session(admit_error);
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
        c->first_operation = admitted - 1;
        c->ring_sequence = admitted - 1;
        c->closes_record = final_pending;
        c->finish_flush = !final_pending;
        lock.unlock();
        try {
          S.configure_flush(*c);
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
    const bool in_flight = !S.active.empty();
    wait_for_async_wake(
        observed_epoch,
        [&] {
          // Unadmitted records are work only if a context can take them;
          // otherwise a completion, which bumps the epoch, must come first.
          if (S.stop.load(std::memory_order_acquire) ||
              S.complete_prefix() != 0)
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
