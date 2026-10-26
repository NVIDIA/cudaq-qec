/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
///
/// Shared event-driven execution for one-shot and streaming task graphs.

#include "decoding_task_graph_impl.h"
#include "../hardware_guards.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace cudaq::qec {

namespace {

constexpr std::size_t no_event = std::numeric_limits<std::size_t>::max();

enum class graph_event_kind : uint8_t {
  one_shot,
  streaming_raw,
  streaming_edge,
  streaming_flush,
  streaming_flush_release
};

struct graph_event {
  graph_event_kind kind = graph_event_kind::one_shot;
  std::size_t node = 0;
  compiled_wire wire{};
  std::size_t offset = 0;
  std::size_t root = no_event;
};

} // namespace

// One bounded store drives every graph execution mode. Event slots remain
// alive until their causal subtree drains; this provides publication leases
// and lets a later reconvergent branch wait for an earlier branch without
// delaying work in an unrelated component.
struct decoding_task_graph::impl::event_runtime {
  struct slot {
    graph_event event;
    graph_event continuation;
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
    bool holds_node_lease = false;
    bool has_continuation = false;
  };

  explicit event_runtime(impl &owner, std::size_t capacity)
      : owner(owner), slots(capacity), node_busy(owner.nodes.size(), 0),
        has_waiting_workers(owner.parallel_pool->num_threads() > 1) {
    free_slots.reserve(capacity);
    root_nodes.reserve(owner.streaming_nodes.size());
    root_done.reserve(owner.streaming_nodes.size());
    for (std::size_t i = capacity; i-- > 0;)
      free_slots.push_back(i);
  }

  void reset() {
    auto lock = lock_runtime();
    // Successful sweeps retire and clear every occupied slot. Rebuild the
    // free list only after failure leaves queued events behind; the common
    // path therefore does no work proportional to the conservative capacity.
    if (occupied != 0 || running != 0 || free_slots.size() != slots.size()) {
      free_slots.clear();
      for (std::size_t i = slots.size(); i-- > 0;) {
        slots[i] = {};
        free_slots.push_back(i);
      }
      std::fill(node_busy.begin(), node_busy.end(), uint8_t{0});
    }
    root_nodes.clear();
    root_done.clear();
    occupied = 0;
    running = 0;
    next_sequence = 0;
    ready_head = no_event;
    ready_tail = no_event;
    stop = false;
    error = nullptr;
  }

  void push_raw_roots() {
    auto lock = lock_runtime();
    for (std::size_t node = 0; node < owner.streaming_nodes.size(); ++node)
      if (owner.streaming_node_has_raw_input[node])
        root_nodes.push_back(node);
    root_done.resize(root_nodes.size());
    std::fill(root_done.begin(), root_done.end(), uint8_t{0});
    for (std::size_t root = 0; root < root_nodes.size(); ++root)
      push_locked(
          {graph_event_kind::streaming_raw, root_nodes[root], {}, 0, root});
  }

  void configure_dependencies(const std::vector<int32_t> &counts) {
    auto lock = lock_runtime();
    dependency_remaining = counts;
  }

  void configure_flush(const std::vector<int32_t> &counts) {
    auto lock = lock_runtime();
    dependency_remaining = counts;
    for (std::size_t node = 0; node < counts.size(); ++node)
      if (counts[node] == 0)
        root_nodes.push_back(node);
    root_done.resize(root_nodes.size());
    std::fill(root_done.begin(), root_done.end(), uint8_t{0});
    for (std::size_t root = 0; root < root_nodes.size(); ++root)
      push_locked(
          {graph_event_kind::streaming_flush, root_nodes[root], {}, 0, root});
  }

  std::size_t push(graph_event event, std::size_t parent = no_event,
                   std::size_t blocked_by = no_event,
                   bool wait_for_siblings = false) {
    auto lock = lock_runtime();
    const auto id = push_locked(event, parent, blocked_by, wait_for_siblings);
    notify_workers();
    return id;
  }

  void release_dependencies(const std::vector<std::size_t> &nodes,
                            graph_event_kind kind) {
    auto lock = lock_runtime();
    if (stop)
      return;
    bool made_ready = false;
    for (const auto node : nodes) {
      if (node >= dependency_remaining.size() ||
          dependency_remaining[node] <= 0) {
        throw std::logic_error(
            "decoding_task_graph: event dependency underflow");
      }
      if (--dependency_remaining[node] == 0) {
        push_locked({kind, node});
        made_ready = true;
      }
    }
    if (made_ready)
      notify_workers();
  }

  void set_continuation(std::size_t id, graph_event event) {
    auto lock = lock_runtime();
    if (stop || id >= slots.size() || !slots[id].occupied)
      return;
    if (slots[id].has_continuation) {
      throw std::logic_error(
          "decoding_task_graph: event already has a continuation");
    }
    slots[id].continuation = event;
    slots[id].has_continuation = true;
  }

  void require_dependencies_complete(const char *message) {
    auto lock = lock_runtime();
    if (std::any_of(dependency_remaining.begin(), dependency_remaining.end(),
                    [](int32_t count) { return count != 0; }))
      throw std::logic_error(message);
  }

  bool publish_streaming(std::size_t parent, std::size_t root,
                         std::size_t producer) {
    const auto node_idx = owner.streaming_nodes[producer].node_index;
    {
      auto lock = lock_runtime();
      owner.record_streaming_publication(node_idx);
    }
    publish_wires(parent, root, owner.streaming_wires_from[node_idx]);
    publish_wires(parent, root, owner.completion_wires_from[node_idx]);
    return true;
  }

  void publish_wires(std::size_t parent, std::size_t root,
                     const std::vector<compiled_wire> &wires) {
    auto lock = lock_runtime();
    slots[parent].holds_node_lease =
        slots[parent].holds_node_lease || !wires.empty();
    for (const auto &wire : wires) {
      const auto consumer =
          static_cast<std::size_t>(owner.streaming_by_node[wire.to_node]);
      std::size_t blocker = no_event;
      std::size_t blocker_sequence = 0;
      // Direct children are stored in declared wire order. A later branch
      // waits only when its transitive descendants intersect an earlier
      // branch; disjoint fan-out remains immediately runnable.
      for (std::size_t earlier = slots[parent].first_child; earlier != no_event;
           earlier = slots[earlier].next_sibling) {
        const auto &candidate = slots[earlier];
        if (candidate.event.kind != graph_event_kind::streaming_edge)
          continue;
        if (branches_overlap(candidate.event.node, consumer) &&
            (blocker == no_event || candidate.sequence > blocker_sequence)) {
          blocker = earlier;
          blocker_sequence = candidate.sequence;
        }
      }
      push_locked({graph_event_kind::streaming_edge, consumer, wire, 0, root},
                  parent, blocker);
    }
    if (!wires.empty())
      notify_workers();
  }

  bool deliver_streaming_edge(std::size_t id, const graph_event &event) {
    const auto producer =
        static_cast<std::size_t>(owner.streaming_by_node[event.wire.from_node]);
    auto *producer_node = owner.streaming_nodes[producer].capability;
    auto *consumer_node = owner.streaming_nodes[event.node].capability;
    const auto from = static_cast<std::size_t>(event.wire.from_node);
    const auto from_slot = static_cast<std::size_t>(event.wire.from_slot);
    // The implicit `done` output is the constant completion marker.
    const auto *marker = &owner.node_outputs[from][from_slot];
    const auto *value = marker->get_if<completion>()
                            ? marker
                            : producer_node->streaming_output(from_slot);
    const bool fired =
        value && consumer_node->enqueue_edge_value(*value, event.wire.to_slot);
    if (fired)
      (void)publish_streaming(id, event.root, event.node);
    // An edge-fed node completes the window after its last delivery.
    const auto consumer =
        static_cast<std::size_t>(owner.streaming_nodes[event.node].node_index);
    if (++owner.streaming_deliveries_seen[consumer] ==
        owner.streaming_wires_to_count[consumer]) {
      owner.streaming_deliveries_seen[consumer] = 0;
      if (!fired)
        publish_wires(id, event.root, owner.completion_wires_from[consumer]);
    }
    return fired;
  }

  template <typename Handler>
  void run(int cuda_device_id, Handler &&handler) {
    owner.parallel_pool->run_workers([&](std::size_t /*worker_id*/) {
      std::size_t active_id = no_event;
      try {
        detail_affinity::CudaDeviceGuard device_guard(cuda_device_id);
        while (true) {
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
              if (occupied == 0)
                return;
              id = select_ready_locked();
              if (id == no_event) {
                if (!has_waiting_workers)
                  throw std::logic_error(
                      "decoding_task_graph: event queue stalled");
                wake.wait(lock);
                continue;
              }
            }
            auto &selected = slots[id];
            selected.running = true;
            selected.queued = false;
            ++running;
            const auto node = serialized_node(selected.event);
            if (node != no_event)
              node_busy[node] = 1;
            event = selected.event;
          }

          active_id = id;
          handler(*this, id, event);
          finish(id);
          active_id = no_event;
        }
      } catch (...) {
        fail(active_id, std::current_exception());
      }
    });

    std::exception_ptr first_error;
    {
      auto lock = lock_runtime();
      first_error = error;
    }
    if (first_error)
      std::rethrow_exception(first_error);
  }

private:
  std::unique_lock<std::mutex> lock_runtime() {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (has_waiting_workers)
      lock.lock();
    return lock;
  }

  std::size_t push_locked(graph_event event, std::size_t parent = no_event,
                          std::size_t blocked_by = no_event,
                          bool wait_for_siblings = false) {
    if (stop)
      return no_event;
    if (parent != no_event &&
        (parent >= slots.size() || !slots[parent].occupied)) {
      throw std::logic_error(
          "decoding_task_graph: event parent is no longer active");
    }
    if (blocked_by != no_event &&
        (blocked_by >= slots.size() || !slots[blocked_by].occupied)) {
      throw std::logic_error(
          "decoding_task_graph: event blocker is no longer active");
    }
    if (free_slots.empty())
      throw std::logic_error(
          "decoding_task_graph: bounded event store overflow");
    const auto id = free_slots.back();
    free_slots.pop_back();
    auto &created = slots[id];
    created = {};
    created.event = event;
    created.parent = parent;
    created.blocked_by = blocked_by;
    created.sequence = next_sequence++;
    created.wait_for_siblings = wait_for_siblings;
    created.occupied = true;
    ++occupied;
    if (parent != no_event) {
      auto &ancestor = slots[parent];
      created.previous_sibling = ancestor.last_child;
      if (ancestor.last_child == no_event)
        ancestor.first_child = id;
      else
        slots[ancestor.last_child].next_sibling = id;
      ancestor.last_child = id;
      ++ancestor.pending_children;
    }
    if (blocked_by != no_event) {
      auto &blocker = slots[blocked_by];
      if (blocker.last_blocked_waiter == no_event)
        blocker.first_blocked_waiter = id;
      else
        slots[blocker.last_blocked_waiter].next_blocked_waiter = id;
      blocker.last_blocked_waiter = id;
    }
    if (blocked_by == no_event && (!wait_for_siblings || parent == no_event))
      enqueue_locked(id);
    return id;
  }

  void enqueue_locked(std::size_t id) {
    auto &candidate = slots[id];
    if (!candidate.occupied || candidate.queued || candidate.running)
      return;
    candidate.queued = true;
    candidate.previous_ready = ready_tail;
    candidate.next_ready = no_event;
    if (ready_tail == no_event)
      ready_head = id;
    else
      slots[ready_tail].next_ready = id;
    ready_tail = id;
  }

  void notify_workers() {
    if (has_waiting_workers)
      wake.notify_all();
  }

  void remove_ready_locked(std::size_t id) {
    auto &candidate = slots[id];
    if (candidate.previous_ready == no_event)
      ready_head = candidate.next_ready;
    else
      slots[candidate.previous_ready].next_ready = candidate.next_ready;
    if (candidate.next_ready == no_event)
      ready_tail = candidate.previous_ready;
    else
      slots[candidate.next_ready].previous_ready = candidate.previous_ready;
    candidate.previous_ready = no_event;
    candidate.next_ready = no_event;
  }

  std::size_t serialized_node(const graph_event &event) const {
    if (event.kind == graph_event_kind::streaming_flush_release)
      return no_event;
    if (event.kind == graph_event_kind::one_shot)
      return event.node;
    if (event.node >= owner.streaming_nodes.size())
      return no_event;
    return static_cast<std::size_t>(
        owner.streaming_nodes[event.node].node_index);
  }

  bool roots_overlap(std::size_t earlier, std::size_t later) const {
    if (earlier >= root_nodes.size() || later >= root_nodes.size())
      return false;
    return branches_overlap(root_nodes[earlier], root_nodes[later]);
  }

  bool branches_overlap(std::size_t left, std::size_t right) const {
    return owner.streaming_overlap[left][right] != 0;
  }

  bool root_is_ready(const graph_event &event, const slot &candidate) const {
    if (event.root == no_event || candidate.parent == no_event)
      return true;
    for (std::size_t root = 0; root < event.root; ++root)
      if (!root_done[root] && roots_overlap(root, event.root))
        return false;
    return true;
  }

  std::size_t select_ready_locked() {
    for (std::size_t id = ready_head; id != no_event;
         id = slots[id].next_ready) {
      const auto &candidate = slots[id];
      const auto node = serialized_node(candidate.event);
      const bool inherits_lease =
          node != no_event && node_busy[node] && candidate.parent != no_event &&
          slots[candidate.parent].holds_node_lease &&
          candidate.wait_for_siblings &&
          serialized_node(slots[candidate.parent].event) == node;
      const bool owns_lease =
          node != no_event && node_busy[node] && candidate.holds_node_lease;
      if ((node != no_event && node_busy[node] && !inherits_lease &&
           !owns_lease) ||
          !root_is_ready(candidate.event, candidate))
        continue;
      if (inherits_lease)
        slots[candidate.parent].holds_node_lease = false;
      remove_ready_locked(id);
      return id;
    }
    return no_event;
  }

  void finish(std::size_t id) {
    auto lock = lock_runtime();
    if (stop) {
      if (id < slots.size() && slots[id].running) {
        slots[id].running = false;
        const auto node = serialized_node(slots[id].event);
        if (node != no_event)
          node_busy[node] = 0;
      }
      if (running != 0)
        --running;
      notify_workers();
      return;
    }
    auto &completed = slots[id];
    const auto node = serialized_node(completed.event);
    if (node != no_event && !completed.holds_node_lease)
      node_busy[node] = 0;
    completed.running = false;
    completed.self_done = true;
    if (running != 0)
      --running;
    if (completed.pending_children == 0)
      complete_or_continue_locked(id);
    else if (completed.pending_children == 1)
      enqueue_sibling_waiter_locked(id);
    notify_workers();
  }

  void complete_or_continue_locked(std::size_t id) {
    if (!slots[id].has_continuation) {
      retire_locked(id);
      return;
    }
    auto &continued = slots[id];
    continued.event = continued.continuation;
    continued.continuation = {};
    continued.has_continuation = false;
    continued.blocked_by = no_event;
    continued.sequence = next_sequence++;
    continued.queued = false;
    continued.running = false;
    continued.self_done = false;
    continued.wait_for_siblings = false;
    enqueue_locked(id);
  }

  void retire_locked(std::size_t id) {
    auto &completed = slots[id];
    const auto parent = completed.parent;
    const auto root = completed.event.root;
    const auto node = serialized_node(completed.event);

    // Release deterministic branch successors before this slot is recycled.
    std::size_t waiting = completed.first_blocked_waiter;
    while (waiting != no_event) {
      const auto next = slots[waiting].next_blocked_waiter;
      slots[waiting].blocked_by = no_event;
      slots[waiting].next_blocked_waiter = no_event;
      if (!slots[waiting].wait_for_siblings)
        enqueue_locked(waiting);
      waiting = next;
    }

    if (completed.holds_node_lease && node != no_event)
      node_busy[node] = 0;
    if (parent != no_event) {
      auto &ancestor = slots[parent];
      if (completed.previous_sibling == no_event)
        ancestor.first_child = completed.next_sibling;
      else
        slots[completed.previous_sibling].next_sibling = completed.next_sibling;
      if (completed.next_sibling == no_event)
        ancestor.last_child = completed.previous_sibling;
      else
        slots[completed.next_sibling].previous_sibling =
            completed.previous_sibling;
    }
    completed = {};
    free_slots.push_back(id);
    if (occupied != 0)
      --occupied;
    if (parent == no_event) {
      if (root != no_event && root < root_done.size())
        root_done[root] = 1;
      notify_workers();
      return;
    }

    auto &ancestor = slots[parent];
    if (ancestor.pending_children == 0)
      throw std::logic_error(
          "decoding_task_graph: event child counter underflow");
    --ancestor.pending_children;
    if (ancestor.self_done && ancestor.pending_children == 1)
      enqueue_sibling_waiter_locked(parent);
    if (ancestor.self_done && ancestor.pending_children == 0)
      complete_or_continue_locked(parent);
  }

  void enqueue_sibling_waiter_locked(std::size_t parent) {
    for (std::size_t waiting = slots[parent].first_child; waiting != no_event;
         waiting = slots[waiting].next_sibling)
      if (slots[waiting].wait_for_siblings &&
          slots[waiting].blocked_by == no_event) {
        if (serialized_node(slots[waiting].event) == no_event &&
            slots[parent].holds_node_lease) {
          const auto node = serialized_node(slots[parent].event);
          if (node != no_event)
            node_busy[node] = 0;
          slots[parent].holds_node_lease = false;
        }
        enqueue_locked(waiting);
      }
  }

  void fail(std::size_t id, std::exception_ptr failure) noexcept {
    auto lock = lock_runtime();
    if (!error)
      error = std::move(failure);
    stop = true;
    if (id < slots.size() && slots[id].running) {
      slots[id].running = false;
      const auto node = serialized_node(slots[id].event);
      if (node != no_event)
        node_busy[node] = 0;
    }
    if (running != 0)
      --running;
    notify_workers();
  }

  impl &owner;
  std::vector<slot> slots;
  std::vector<std::size_t> free_slots;
  std::vector<uint8_t> node_busy;
  std::vector<int32_t> dependency_remaining;
  std::vector<std::size_t> root_nodes;
  std::vector<uint8_t> root_done;
  const bool has_waiting_workers;
  std::mutex mutex;
  std::condition_variable wake;
  std::size_t occupied = 0;
  std::size_t running = 0;
  std::size_t next_sequence = 0;
  std::size_t ready_head = no_event;
  std::size_t ready_tail = no_event;
  bool stop = false;
  std::exception_ptr error;
};

// Construct empty graph state before the private event runtime is complete.
decoding_task_graph::impl::impl() = default;

void decoding_task_graph::impl::prepare_event_runtime(std::size_t capacity) {
  runtime = std::make_unique<event_runtime>(*this, capacity);
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

bool decoding_task_graph::impl::enqueue_sweep(const uint8_t *syndrome,
                                              std::size_t len,
                                              std::optional<round_type> type,
                                              int cuda_device_id) {
  require_streaming_support("enqueue_syndrome");
  runtime->reset();

  runtime->push_raw_roots();

  std::atomic<bool> fired{false};
  runtime->run(cuda_device_id, [&](event_runtime &loop, std::size_t id,
                                   const graph_event &event) {
    if (event.kind == graph_event_kind::streaming_raw) {
      auto &node = *streaming_nodes[event.node].capability;
      const auto limit =
          type ? node.raw_chunk_limit() : node.untagged_chunk_limit();
      const auto take =
          limit ? std::min(*limit, len - event.offset) : len - event.offset;
      const auto *chunk = syndrome ? syndrome + event.offset : nullptr;
      const auto next = event.offset + take;
      const auto chunk_type = type == round_type::FINAL && next == len
                                  ? round_type::FINAL
                                  : round_type::BULK;
      if (type ? node.enqueue(chunk, take, chunk_type)
               : node.enqueue_untagged(chunk, take))
        fired.store(loop.publish_streaming(id, event.root, event.node),
                    std::memory_order_relaxed);
      if (next < len && take != 0)
        loop.set_continuation(id, {graph_event_kind::streaming_raw,
                                   event.node,
                                   {},
                                   next,
                                   event.root});
      return;
    }

    if (event.kind != graph_event_kind::streaming_edge)
      throw std::logic_error("decoding_task_graph: unexpected enqueue event");
    if (loop.deliver_streaming_edge(id, event))
      fired.store(true, std::memory_order_relaxed);
  });
  return fired.load(std::memory_order_relaxed);
}

bool decoding_task_graph::impl::flush_sweep(int cuda_device_id) {
  runtime->reset();
  runtime->configure_flush(streaming_predecessor_counts);

  std::atomic<bool> fired{false};
  runtime->run(cuda_device_id, [&](event_runtime &loop, std::size_t id,
                                   const graph_event &event) {
    if (event.kind == graph_event_kind::streaming_edge) {
      if (loop.deliver_streaming_edge(id, event))
        fired.store(true, std::memory_order_relaxed);
      return;
    }
    if (event.kind == graph_event_kind::streaming_flush_release) {
      loop.release_dependencies(streaming_dependents[event.node],
                                graph_event_kind::streaming_flush);
      return;
    }
    if (event.kind != graph_event_kind::streaming_flush)
      throw std::logic_error("decoding_task_graph: unexpected flush event");

    if (streaming_nodes[event.node].capability->flush())
      fired.store(loop.publish_streaming(id, event.root, event.node),
                  std::memory_order_relaxed);
    loop.push({graph_event_kind::streaming_flush_release, event.node}, id,
              no_event, /*wait_for_siblings=*/true);
  });
  runtime->require_dependencies_complete(
      "decoding_task_graph: streaming flush dependency queue stalled");
  return fired.load(std::memory_order_relaxed);
}

// Execute one graph shot through the same event loop used by streaming. The
// final predecessor publishes a dependent while holding the runtime lock, so
// all predecessor output writes happen before the dependent is dispatched.
logical_outcome decoding_task_graph::impl::run_dag_parallel(
    const measurement_results &measurements, const char *who,
    int cuda_device_id) {
  const std::size_t node_count = nodes.size();
  edge_value measurement_value{measurements};
  const std::vector<compiled_wire> *sink_wires = nullptr;

  require_single_output(who);
  sink_wires = &wires_to.back();
  if (parallel_pool == nullptr || runtime == nullptr)
    throw std::logic_error(
        "decoding_task_graph: parallel runtime was not compiled");
  if (predecessor_counts.size() != node_count ||
      dependents.size() != node_count || source_nodes.empty()) {
    throw std::logic_error(
        "decoding_task_graph: malformed compiled dependency graph");
  }

  runtime->reset();
  runtime->configure_dependencies(predecessor_counts);
  for (const std::size_t node : source_nodes)
    runtime->push({graph_event_kind::one_shot, node});

  runtime->run(cuda_device_id, [&](event_runtime &loop, std::size_t /*id*/,
                                   const graph_event &event) {
    const auto node = event.node;
    auto &inputs = parallel_node_inputs[node];
    for (const auto &wire : wires_to[node])
      inputs[static_cast<std::size_t>(wire.to_slot)] =
          wire.from_node < 0
              ? measurement_value
              : node_outputs[static_cast<std::size_t>(wire.from_node)]
                            [static_cast<std::size_t>(wire.from_slot)];
    nodes[node]->execute_indexed(inputs, node_outputs[node]);
    loop.release_dependencies(dependents[node], graph_event_kind::one_shot);
  });
  runtime->require_dependencies_complete(
      "decoding_task_graph: one-shot dependency queue stalled");

  const compiled_wire &sink_wire = sink_wires->front();
  return node_outputs[static_cast<std::size_t>(sink_wire.from_node)]
                     [static_cast<std::size_t>(sink_wire.from_slot)]
                         .get<logical_outcome>();
} // end - decoding_task_graph::impl::run_dag_parallel()

} // namespace cudaq::qec
