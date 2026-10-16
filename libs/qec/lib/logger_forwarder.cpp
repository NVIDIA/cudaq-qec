/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "logger_forwarder.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

/// @file
/// @brief Internal forwarder implementation for QEC logger.
/// @details
/// Implements a bounded lock-free ring buffer (multi-producer, single-consumer)
/// used when asynchronous forwarding is enabled. Producers enqueue fixed-size
/// records without heap allocation; a dedicated worker thread converts them to
/// user-facing records and invokes the configured callback.

namespace cudaq::qec::detail::forwarder_internal {
namespace {

class async_forwarder {
public:
  async_forwarder() = default;
  ~async_forwarder() { clear(); }

  void set(forwarder_config config) {
    clear();
    config_ = std::move(config);
    if (!config_.callback)
      return;
    if (config_.queue_capacity == 0)
      config_.queue_capacity = 1;
    if (config_.message_capacity == 0)
      config_.message_capacity = 1;
    config_.message_capacity = std::min(
        config_.message_capacity, realtime_forwarder_max_message_capacity);
    message_capacity_.store(config_.message_capacity,
                            std::memory_order_relaxed);

    capacity_ = round_up_to_pow2(config_.queue_capacity);
    ring_ = std::make_unique<ring_slot[]>(capacity_);
    for (std::size_t i = 0; i < capacity_; ++i)
      ring_[i].sequence.store(i, std::memory_order_relaxed);

    mask_ = capacity_ - 1;
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
    stop_.store(false, std::memory_order_relaxed);
    enabled_.store(true, std::memory_order_release);
    worker_ = std::thread([this] { run_worker(); });
  }

  // Stop producer admission first, then wait for in-flight producers before
  // tearing down the ring storage to avoid use-after-free races.
  void clear() {
    enabled_.store(false, std::memory_order_release);
    while (active_producers_.load(std::memory_order_acquire) != 0)
      std::this_thread::yield();
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (worker_.joinable())
      worker_.join();
    ring_.reset();
    capacity_ = 0;
    mask_ = 0;
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
    stop_.store(false, std::memory_order_relaxed);
    message_capacity_.store(realtime_forwarder_default_message_capacity,
                            std::memory_order_relaxed);
    config_ = {};
  }

  bool is_enabled() const { return enabled_.load(std::memory_order_relaxed); }

  forwarder_stats stats() const {
    forwarder_stats stats;
    stats.enqueued_records = enqueued_records_.load(std::memory_order_relaxed);
    stats.dropped_records = dropped_records_.load(std::memory_order_relaxed);
    stats.truncated_records =
        truncated_records_.load(std::memory_order_relaxed);
    stats.forward_failures = forward_failures_.load(std::memory_order_relaxed);
    return stats;
  }

  void reset_stats() {
    enqueued_records_.store(0, std::memory_order_relaxed);
    dropped_records_.store(0, std::memory_order_relaxed);
    truncated_records_.store(0, std::memory_order_relaxed);
    forward_failures_.store(0, std::memory_order_relaxed);
    drop_warning_pending_.store(false, std::memory_order_relaxed);
    drop_warning_emitted_.store(false, std::memory_order_relaxed);
  }

  std::size_t message_capacity() const {
    return message_capacity_.load(std::memory_order_relaxed);
  }

  void note_truncation() {
    truncated_records_.fetch_add(1, std::memory_order_relaxed);
  }

  void flush() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    cv_.wait(lock, [&] {
      return ((!enabled_.load()) ||
              (is_queue_empty() &&
               in_flight_callbacks_.load(std::memory_order_acquire) == 0));
    });
  }

  void enqueue(queued_log_record record) {
    if (!is_enabled())
      return;
    producer_guard guard(active_producers_);
    if (!is_enabled() || !config_.callback || !ring_)
      return;

    while (true) {
      if (try_enqueue_one(record)) {
        enqueued_records_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_one();
        return;
      }

      // The queue is full. drop_oldest evicts the oldest record to admit the
      // incoming one; the evicted record is the genuinely lost message, so it
      // is the only thing counted as a drop here. If the eviction fails because
      // a consumer already drained the slot, the queue is no longer full, so we
      // simply retry the enqueue without counting a drop (the incoming record
      // is not lost). This keeps dropped_records equal to the number of records
      // actually discarded, and ensures the drop_newest fallthrough below
      // counts only the incoming record it drops.
      if (config_.drop_policy == forward_drop_policy::drop_oldest) {
        queued_log_record ignored;
        if (try_dequeue_one(ignored))
          notify_drop();
        continue;
      }

      // drop_newest: the incoming record is the one we lose.
      notify_drop();
      return;
    }
  }

private:
  struct ring_slot {
    std::atomic<std::size_t> sequence{0};
    queued_log_record record;
  };

  struct producer_guard {
    explicit producer_guard(std::atomic<std::uint64_t> &counter_ref)
        : counter(counter_ref) {
      counter.fetch_add(1, std::memory_order_acq_rel);
    }
    ~producer_guard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
    std::atomic<std::uint64_t> &counter;
  };

  static std::size_t round_up_to_pow2(std::size_t value) {
    // Ring indexing uses `pos & (capacity - 1)`, so capacity must remain a
    // power-of-two for correct slot mapping.
    std::size_t rounded = 2;
    while (rounded < value)
      rounded <<= 1;
    return rounded;
  }

  bool try_enqueue_one(queued_log_record &record) {
    std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    while (true) {
      ring_slot &slot = ring_[pos & mask_];
      const std::size_t sequence =
          slot.sequence.load(std::memory_order_acquire);
      const std::intptr_t dif = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos);

      if (dif == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed))
          break;
        continue;
      }
      if (dif < 0)
        return false;
      pos = enqueue_pos_.load(std::memory_order_relaxed);
    }

    ring_slot &slot = ring_[pos & mask_];
    slot.record = std::move(record);
    slot.sequence.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool try_dequeue_one(queued_log_record &record) {
    std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    while (true) {
      ring_slot &slot = ring_[pos & mask_];
      const std::size_t sequence =
          slot.sequence.load(std::memory_order_acquire);
      const std::intptr_t dif = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos + 1);

      if (dif == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed))
          break;
        continue;
      }
      if (dif < 0)
        return false;
      pos = dequeue_pos_.load(std::memory_order_relaxed);
    }

    ring_slot &slot = ring_[pos & mask_];
    record = std::move(slot.record);
    slot.sequence.store(pos + capacity_, std::memory_order_release);
    return true;
  }

  bool is_queue_empty() const {
    return enqueue_pos_.load(std::memory_order_acquire) ==
           dequeue_pos_.load(std::memory_order_acquire);
  }

  void notify_drop() {
    dropped_records_.fetch_add(1, std::memory_order_relaxed);
    drop_warning_pending_.store(true, std::memory_order_relaxed);
    cv_.notify_one();
  }

  void emit_drop_warning_if_pending() {
    if (!drop_warning_pending_.exchange(false, std::memory_order_relaxed))
      return;
    bool expected = false;
    if (!drop_warning_emitted_.compare_exchange_strong(
            expected, true, std::memory_order_relaxed))
      return;
    std::fputs("[cudaq::qec::logger] forwarder dropped log records "
               "(queue full); increase forwarder_config::queue_capacity or "
               "reduce callback latency.\n",
               stderr);
  }

  // Single consumer loop that drains the ring and calls the user callback.
  // Drop warnings are emitted from this worker to keep producer hot paths free
  // of blocking I/O syscalls.
  void run_worker() {
    while (true) {
      emit_drop_warning_if_pending();
      // Mark callback slot in-flight before dequeue attempt so flush() cannot
      // observe (queue empty && inFlight == 0) in the tiny window between a
      // successful dequeue and callback start.
      in_flight_callbacks_.fetch_add(1, std::memory_order_acq_rel);
      queued_log_record queued_record;
      if (!try_dequeue_one(queued_record)) {
        in_flight_callbacks_.fetch_sub(1, std::memory_order_acq_rel);
        // Publish the in-flight/queue state under wait_mutex_ before notifying
        // so flush() cannot miss this wakeup in the window between checking its
        // predicate and blocking on cv_. See notify after callback below.
        notify_flush_waiters();
        std::unique_lock<std::mutex> lock(wait_mutex_);
        cv_.wait(lock, [&] {
          return stop_.load(std::memory_order_acquire) || !is_queue_empty();
        });
        if (stop_.load(std::memory_order_acquire) && is_queue_empty())
          return;
        continue;
      }

      forwarded_log_record record;
      record.level = queued_record.level;
      record.timestamp_ns = queued_record.timestamp_ns;
      record.file_name.assign(queued_record.file_name.data(),
                              queued_record.file_name_len);
      record.line_no = queued_record.line_no;
      record.message.assign(queued_record.message.data(),
                            queued_record.message_len);
      record.thread_id = queued_record.thread_id;

      try {
        // `record` is a per-iteration local discarded after this call, so it
        // is safe to hand ownership to the callback via rvalue. A consumer may
        // move out of it (e.g. to re-queue) instead of copying its strings.
        if (config_.callback)
          config_.callback(std::move(record));
      } catch (...) {
        forward_failures_.fetch_add(1, std::memory_order_relaxed);
      }
      in_flight_callbacks_.fetch_sub(1, std::memory_order_acq_rel);

      emit_drop_warning_if_pending();
      notify_flush_waiters();
    }
  }

  // Wake flush() waiters without losing the wakeup. The worker mutates the
  // state flush() inspects (in_flight_callbacks_ and the ring positions) via
  // atomics that are not held under wait_mutex_. Briefly taking wait_mutex_
  // before notifying serializes against a flush() thread that has evaluated its
  // predicate as false but has not yet blocked on cv_: that thread holds
  // wait_mutex_ across the predicate check and atomically releases it inside
  // cv_.wait(), so this notify either happens before the check (observed via
  // the acquire load) or after the waiter is parked (delivered by notify).
  void notify_flush_waiters() {
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
    }
    cv_.notify_all();
  }

  std::condition_variable_any cv_;
  mutable std::mutex wait_mutex_;
  forwarder_config config_;
  std::unique_ptr<ring_slot[]> ring_;
  std::size_t capacity_ = 0;
  std::size_t mask_ = 0;
  std::atomic<std::size_t> enqueue_pos_{0};
  std::atomic<std::size_t> dequeue_pos_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
  std::atomic<bool> enabled_{false};
  std::atomic<std::uint64_t> enqueued_records_{0};
  std::atomic<std::uint64_t> dropped_records_{0};
  std::atomic<std::uint64_t> truncated_records_{0};
  std::atomic<std::uint64_t> forward_failures_{0};
  std::atomic<std::uint64_t> in_flight_callbacks_{0};
  std::atomic<std::uint64_t> active_producers_{0};
  std::atomic<bool> drop_warning_pending_{false};
  std::atomic<bool> drop_warning_emitted_{false};
  std::atomic<std::size_t> message_capacity_{
      realtime_forwarder_default_message_capacity};
};

async_forwarder &forwarder() {
  static async_forwarder instance;
  return instance;
}

} // namespace

void set(forwarder_config config) { forwarder().set(std::move(config)); }
void clear() { forwarder().clear(); }
bool is_enabled() { return forwarder().is_enabled(); }
void reset_stats() { forwarder().reset_stats(); }
forwarder_stats stats() { return forwarder().stats(); }
void flush() { forwarder().flush(); }
void enqueue(queued_log_record record) {
  forwarder().enqueue(std::move(record));
}
std::size_t message_capacity() { return forwarder().message_capacity(); }
void note_truncation() { forwarder().note_truncation(); }

} // namespace cudaq::qec::detail::forwarder_internal
