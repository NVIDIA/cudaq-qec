/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file
///
/// Bounded asynchronous input processing for streaming decoding task graphs.

#include "decoding_task_graph_impl.h"
#include "../hardware_guards.h"
#include "cudaq/qec/logger.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <stdexcept>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CUDAQ_QEC_ASYNC_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CUDAQ_QEC_ASYNC_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
#define CUDAQ_QEC_ASYNC_CPU_RELAX() ((void)0)
#endif

namespace cudaq::qec {

namespace {

constexpr int kAsyncFinishSpinPauseIters = 4096;

// Keep an active coordinator hot for a bounded interval. Waiting
// sessions still park, so an idle prepared graph consumes no CPU.
bool spin_for_async_wake(std::atomic<uint64_t> &wake_epoch,
                         uint64_t observed_epoch,
                         std::size_t pause_iters) noexcept {
  for (std::size_t spin = 0; spin < pause_iters; ++spin) {
    if (wake_epoch.load(std::memory_order_acquire) != observed_epoch)
      return true;
    CUDAQ_QEC_ASYNC_CPU_RELAX();
  }
  return false;
}

// Release the single-producer publication guard and wake a lifecycle operation
// that is waiting for an in-progress enqueue to finish.
void release_producer(std::atomic<bool> &producer_active,
                      std::atomic<std::uint32_t> &producer_waiters) noexcept {
  producer_active.store(false, std::memory_order_release);
  if (producer_waiters.load(std::memory_order_acquire) != 0)
    producer_active.notify_all();
}

// Advertise before checking the producer flag, so a producer either observes
// the waiter and notifies or completes before the waiter attempts to park.
void wait_for_producer(std::atomic<bool> &producer_active,
                       std::atomic<std::uint32_t> &producer_waiters) noexcept {
  producer_waiters.fetch_add(1, std::memory_order_acq_rel);
  while (producer_active.load(std::memory_order_acquire))
    producer_active.wait(true, std::memory_order_acquire);
  producer_waiters.fetch_sub(1, std::memory_order_acq_rel);
}

void notify_completion_waiters(
    std::atomic<uint64_t> &completed,
    std::atomic<std::uint32_t> &completion_waiters) noexcept {
  if (completion_waiters.load(std::memory_order_acquire) != 0)
    completed.notify_all();
}

void wait_for_completion_change(std::atomic<uint64_t> &completed,
                                std::atomic<std::uint32_t> &completion_waiters,
                                uint64_t observed) noexcept {
  completion_waiters.fetch_add(1, std::memory_order_acq_rel);
  if (completed.load(std::memory_order_acquire) == observed)
    completed.wait(observed, std::memory_order_acquire);
  completion_waiters.fetch_sub(1, std::memory_order_acq_rel);
}

// Final-round decode commonly finishes within a few microseconds. Spin briefly
// before entering the kernel-backed atomic wait so scheduler park/wake latency
// does not dominate that short tail; long decoder work still parks normally.
void wait_until_inactive(std::atomic<bool> &active,
                         bool spin_first = true) noexcept {
  const int pause_iters = spin_first ? kAsyncFinishSpinPauseIters : 0;
  for (int spin = 0; spin < pause_iters; ++spin) {
    if (!active.load(std::memory_order_acquire))
      return;
    CUDAQ_QEC_ASYNC_CPU_RELAX();
  }
  while (active.load(std::memory_order_acquire))
    active.wait(true, std::memory_order_acquire);
}

} // namespace

// Advance a monotonic event counter before notifying the coordinator. Unlike a
// notification alone, the counter prevents an event from being missed if it
// arrives just before the coordinator parks.
void decoding_task_graph::impl::wake_async_stream() noexcept {
  auto &state = async_streaming;
  state.wake_epoch.fetch_add(1, std::memory_order_release);
  if (state.coordinator_waiting.load(std::memory_order_acquire))
    state.wake_epoch.notify_one();
}

void decoding_task_graph::impl::wake_async_stream_for_record() noexcept {
  auto &state = async_streaming;
  const bool waiting =
      state.coordinator_waiting.exchange(false, std::memory_order_acq_rel);
  if (!waiting && state.spin_pause_iters == 0)
    return;
  state.wake_epoch.fetch_add(1, std::memory_order_release);
  if (waiting)
    state.wake_epoch.notify_one();
}

// Latch the first worker failure, close input, and retire every sequence that
// was accepted before publication stopped. This guarantees drain waiters make
// progress even though the failed record did not complete normally. The
// coordinator is the error slot's sole writer; the failed release store
// publishes the exception_ptr to control-thread acquire loads without a lock.
void decoding_task_graph::impl::store_async_stream_error(
    std::exception_ptr error) noexcept {
  auto &state = async_streaming;
  if (state.error == nullptr)
    state.error = std::move(error);
  state.needs_reset = true;
  state.failed.store(true, std::memory_order_release);
  state.accepting.store(false, std::memory_order_release);

  // An enqueue that passed its initial acceptance check may still be
  // publishing. Wait until its release store before discarding the outstanding
  // range, so every accepted sequence is covered by completed.
  wait_for_producer(state.producer_active, state.producer_waiters);
  state.completed.store(state.submitted.load(std::memory_order_acquire),
                        std::memory_order_release);
  notify_completion_waiters(state.completed, state.completion_waiters);
}

// Keep result snapshots and the control-plane exception channel consistent for
// every coordinator failure, including failures before graph processing starts
// (for example, while selecting the CUDA device).
void decoding_task_graph::impl::handle_async_stream_error(
    std::exception_ptr error) noexcept {
  try {
    if (error != nullptr) {
      try {
        std::rethrow_exception(error);
      } catch (const std::exception &exception) {
        fail_streaming_progress(exception.what());
      } catch (...) {
        fail_streaming_progress("unknown asynchronous streaming failure");
      }
    } else {
      fail_streaming_progress("unknown asynchronous streaming failure");
    }
    publish_async_stream_snapshot();
  } catch (...) {
    // The exception_ptr channel remains authoritative if materializing failure
    // metadata itself cannot complete.
  }
  store_async_stream_error(std::move(error));
}

// Keep the successful path lock-free. The acquire load pairs with the
// coordinator's failed release store, making its exception_ptr write visible.
void decoding_task_graph::impl::rethrow_async_stream_error() {
  const auto &state = async_streaming;
  if (!state.failed.load(std::memory_order_acquire))
    return;
  if (state.error != nullptr)
    std::rethrow_exception(state.error);
}

// Copy the coordinator-owned coherent result into its reader-facing snapshot.
// Publication occurs only between records and excludes concurrent readers.
void decoding_task_graph::impl::publish_async_stream_snapshot() {
  auto &state = async_streaming;
  std::lock_guard<std::mutex> lock(state.snapshot_mutex);
  const auto &source = streaming_result_state;
  auto &snapshot = state.result_snapshot;
  if (source.outputs.size() != snapshot.outputs.size() ||
      source.values.size() != snapshot.values.size())
    throw std::logic_error(
        "asynchronous streaming result shape changed after preparation");
  snapshot.shot_epoch = source.shot_epoch;
  snapshot.result_revision = source.result_revision;
  snapshot.completed_operations = source.completed_operations;
  snapshot.generation = source.generation;
  snapshot.status = source.status;
  snapshot.operation_completed = source.operation_completed;
  snapshot.error = source.error;
  std::copy(source.outputs.begin(), source.outputs.end(),
            snapshot.outputs.begin());
  std::copy(source.values.begin(), source.values.end(),
            snapshot.values.begin());
  state.snapshot_valid = true;
}

// Exclude readers while retiring the last publication from a cancelled or
// reused graph. A successfully finished session deliberately keeps its final
// snapshot available until one of these invalidating operations.
void decoding_task_graph::impl::invalidate_async_stream_snapshot() {
  auto &state = async_streaming;
  std::lock_guard<std::mutex> lock(state.snapshot_mutex);
  state.snapshot_valid = false;
}

// Consume accepted records in sequence order on the graph's sole streaming
// state-mutator thread. Lifecycle requests are handled only between node calls:
// cancellation cannot interrupt a decoder, while finish drains and optionally
// flushes.
void decoding_task_graph::impl::async_stream_loop() noexcept {
  auto &state = async_streaming;
  uint64_t observed_epoch = 0;

  // Release graph exclusivity before publishing inactive, so a caller waking
  // from active.wait() may immediately use a synchronous graph API.
  const auto deactivate = [this, &state] {
    state.accepting.store(false, std::memory_order_release);
    state.flush_requested.store(false, std::memory_order_relaxed);
    state.cancel_requested.store(false, std::memory_order_relaxed);
    in_flight.fetch_sub(1, std::memory_order_acq_rel);
    state.active.store(false, std::memory_order_release);
    state.active.notify_all();
  };

  while (true) {
    if (state.active.load(std::memory_order_acquire) &&
        spin_for_async_wake(state.wake_epoch, observed_epoch,
                            state.spin_pause_iters)) {
      observed_epoch = state.wake_epoch.load(std::memory_order_acquire);
    } else {
      // The exchange acquires the latest producer publication before checking
      // the queue. A producer that observes this advertisement clears it and
      // performs the one required wake.
      state.coordinator_waiting.exchange(true, std::memory_order_acq_rel);
      const bool work_ready =
          state.active.load(std::memory_order_acquire) &&
          state.completed.load(std::memory_order_acquire) <
              state.submitted.load(std::memory_order_acquire);
      if (!work_ready &&
          state.wake_epoch.load(std::memory_order_acquire) == observed_epoch)
        state.wake_epoch.wait(observed_epoch, std::memory_order_acquire);
      state.coordinator_waiting.store(false, std::memory_order_release);
      observed_epoch = state.wake_epoch.load(std::memory_order_acquire);
    }
    if (state.shutdown.load(std::memory_order_acquire))
      return;
    if (!state.active.load(std::memory_order_acquire))
      continue;

    if (state.cancel_requested.load(std::memory_order_acquire)) {
      state.needs_reset = true;
      state.completed.store(state.submitted.load(std::memory_order_acquire),
                            std::memory_order_release);
      notify_completion_waiters(state.completed, state.completion_waiters);
      deactivate();
      continue;
    }

    try {
      uint64_t completed = state.completed.load(std::memory_order_acquire);
      while (completed < state.submitted.load(std::memory_order_acquire)) {
        if (state.cancel_requested.load(std::memory_order_acquire))
          break;
        auto &slot = state.slots[completed % state.capacity];
        detail_affinity::CudaDeviceGuard device_guard(state.cuda_device_id);
        const auto prior_revision = streaming_result_state.result_revision;
        (void)process_streaming_record(slot.syndrome.data(), slot.len,
                                       slot.type);
        state.result_changed |=
            streaming_result_state.result_revision != prior_revision;
        state.operation_completed = streaming_result_state.operation_completed;
        if (state.operation_completed)
          ++streaming_result_state.completed_operations;
        // Most public operations only fill a window and leave every observable
        // and status unchanged. Avoid serializing with readers and copying an
        // identical snapshot for those no-op operations. FINAL still publishes
        // its terminal status even for an empty shot.
        const bool operation_pulse_changed =
            state.result_snapshot.operation_completed !=
            streaming_result_state.operation_completed;
        if (state.result_changed || operation_pulse_changed ||
            slot.type == round_type::FINAL)
          publish_async_stream_snapshot();
        state.operation_completed = false;
        state.result_changed = false;
        ++completed;
        state.completed.store(completed, std::memory_order_release);
        notify_completion_waiters(state.completed, state.completion_waiters);
      }
    } catch (...) {
      handle_async_stream_error(std::current_exception());
    }

    if (state.cancel_requested.load(std::memory_order_acquire)) {
      state.needs_reset = true;
      state.completed.store(state.submitted.load(std::memory_order_acquire),
                            std::memory_order_release);
      notify_completion_waiters(state.completed, state.completion_waiters);
      deactivate();
      continue;
    }

    if (state.finish_requested.load(std::memory_order_acquire) &&
        state.completed.load(std::memory_order_acquire) >=
            state.finish_target) {
      if (state.flush_requested.load(std::memory_order_relaxed) &&
          !state.failed.load(std::memory_order_acquire) && !finalized) {
        try {
          detail_affinity::CudaDeviceGuard device_guard(state.cuda_device_id);
          (void)finish_streaming_record();
          publish_async_stream_snapshot();
        } catch (...) {
          handle_async_stream_error(std::current_exception());
        }
      }
      deactivate();
    }
  }
}

// Cancel an active session if necessary, then publish shutdown and join the
// persistent coordinator. Decoder work already executing is allowed to return.
void decoding_task_graph::impl::shutdown_async_streaming() noexcept {
  auto &state = async_streaming;
  if (!state.thread.joinable())
    return;
  if (state.active.load(std::memory_order_acquire)) {
    state.accepting.store(false, std::memory_order_release);
    state.cancel_requested.store(true, std::memory_order_release);
    wake_async_stream();
    wait_until_inactive(state.active);
  }
  state.shutdown.store(true, std::memory_order_release);
  wake_async_stream();
  state.thread.join();
}

// Allocate every ring slot with room for the common per-round payload and
// launch the parked coordinator outside the real-time enqueue path. An
// oversized public operation grows one free slot rather than changing the
// operation's decoder-visible boundary. The coordinator reports the result of
// its own first-action affinity setup before preparation returns. A failed
// launch or affinity request rolls the prepared state back so preparation may
// be retried.
void decoding_task_graph::impl::prepare_async_streaming(
    std::size_t queue_capacity, std::size_t initial_payload_bytes,
    int coordinator_cpu, std::size_t spin_pause_iters) {
  auto &g = *this;
  auto &state = g.async_streaming;
  g.require_streaming_support("prepare_async_streaming");
  if (queue_capacity == 0 || initial_payload_bytes == 0) {
    throw std::invalid_argument(
        "prepare_async_streaming: queue capacities must be nonzero");
  }
  if (coordinator_cpu < -1)
    throw std::invalid_argument(
        "prepare_async_streaming: coordinator CPU must be -1 or nonnegative");
#if !defined(__linux__)
  if (coordinator_cpu >= 0)
    throw std::runtime_error(
        "prepare_async_streaming: coordinator affinity requires Linux");
#else
  if (coordinator_cpu >= CPU_SETSIZE)
    throw std::out_of_range(
        "prepare_async_streaming: coordinator CPU exceeds CPU_SETSIZE");
#endif
  if (state.thread.joinable())
    throw std::logic_error("prepare_async_streaming: already prepared");
  if (g.in_flight.load(std::memory_order_acquire) != 0)
    throw std::logic_error(
        "prepare_async_streaming: another graph operation is in flight");

  state.capacity = queue_capacity;
  state.spin_pause_iters = spin_pause_iters;
  state.slots.resize(queue_capacity);
  state.result_snapshot = g.streaming_result_state;
  state.read_snapshot = g.streaming_result_state;
  for (auto &slot : state.slots)
    slot.syndrome.resize(initial_payload_bytes);
  try {
    std::promise<int> affinity_result;
    auto affinity_ready = affinity_result.get_future();
    state.thread =
        std::thread([impl = &g, coordinator_cpu,
                     affinity_result = std::move(affinity_result)]() mutable {
          int error = 0;
#if defined(__linux__)
          if (coordinator_cpu >= 0) {
            cpu_set_t cpu_set;
            CPU_ZERO(&cpu_set);
            CPU_SET(coordinator_cpu, &cpu_set);
            error = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set),
                                           &cpu_set);
          }
#endif
          affinity_result.set_value(error);
          if (error == 0)
            impl->async_stream_loop();
        });
    const int error = affinity_ready.get();
    if (error != 0)
      throw std::runtime_error(
          "prepare_async_streaming: failed to pin coordinator to CPU " +
          std::to_string(coordinator_cpu) + ": " + std::strerror(error));
  } catch (...) {
    if (state.thread.joinable()) {
      state.shutdown.store(true, std::memory_order_release);
      g.wake_async_stream();
      state.thread.join();
      state.shutdown.store(false, std::memory_order_relaxed);
    }
    state.slots.clear();
    state.result_snapshot = {};
    state.capacity = 0;
    throw;
  }
}

// Acquire the graph's existing single-flight token for the complete session,
// resolve CUDA affinity synchronously, clear prior status, and wake the
// prepared coordinator. Finish or cancel releases the token.
void decoding_task_graph::impl::begin_async_streaming(int cuda_device_id) {
  auto &g = *this;
  auto &state = g.async_streaming;
  uint32_t expected = 0;

  g.require_streaming_support("begin_async_streaming");
  if (!state.thread.joinable())
    throw std::logic_error("begin_async_streaming: graph is not prepared");
  if (state.active.load(std::memory_order_acquire))
    throw std::logic_error("begin_async_streaming: session already active");
  if (g.finalized)
    throw std::logic_error(
        "begin_async_streaming: clear or reset the finalized graph first");
  if (state.needs_reset)
    throw std::logic_error(
        "begin_async_streaming: reset the cancelled or failed graph first");
  if (!g.in_flight.compare_exchange_strong(
          expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    throw std::logic_error(
        "begin_async_streaming: another graph operation is in flight");
  }

  try {
    state.error = nullptr;
    g.invalidate_async_stream_snapshot();
    state.cuda_device_id = cuda_device_id;
    state.failed.store(false, std::memory_order_relaxed);
    state.finish_requested.store(false, std::memory_order_relaxed);
    state.flush_requested.store(false, std::memory_order_relaxed);
    state.cancel_requested.store(false, std::memory_order_relaxed);
    state.operation_completed = false;
    state.result_changed = false;
    state.active.store(true, std::memory_order_release);
    state.accepting.store(true, std::memory_order_release);
    g.wake_async_stream();
  } catch (...) {
    g.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    throw;
  }
}

// Reserve one bounded SPSC sequence without waiting, copy its payload, and
// publish it with release ordering. The completed counter prevents slot reuse
// while the coordinator still reads or processes that sequence.
decoding_task_graph::impl::async_enqueue_status
decoding_task_graph::impl::try_enqueue_syndrome_async(
    const uint8_t *syndrome, std::size_t len, std::optional<round_type> type) {
  auto &g = *this;
  auto &state = g.async_streaming;
  uint64_t submitted = 0;
  uint64_t completed = 0;

  if (syndrome == nullptr && len != 0)
    throw std::invalid_argument(
        "try_enqueue_syndrome_async: null input with nonzero length");
  if (state.producer_active.exchange(true, std::memory_order_acq_rel)) {
    throw std::logic_error(
        "try_enqueue_syndrome_async: concurrent producers are unsupported");
  }

  if (!state.active.load(std::memory_order_acquire) ||
      !state.accepting.load(std::memory_order_acquire)) {
    const auto status = state.failed.load(std::memory_order_acquire)
                            ? async_enqueue_status::failed
                            : async_enqueue_status::closed;
    release_producer(state.producer_active, state.producer_waiters);
    return status;
  }

  submitted = state.submitted.load(std::memory_order_relaxed);
  completed = state.completed.load(std::memory_order_acquire);
  if (submitted - completed >= state.capacity) {
    release_producer(state.producer_active, state.producer_waiters);
    return async_enqueue_status::queue_full;
  }

  auto &slot = state.slots[submitted % state.capacity];
  try {
    if (slot.syndrome.size() < len)
      slot.syndrome.resize(len);
  } catch (...) {
    release_producer(state.producer_active, state.producer_waiters);
    throw;
  }
  if (len != 0)
    std::memcpy(slot.syndrome.data(), syndrome, len);
  slot.len = len;
  slot.type = type;
  if (type == round_type::FINAL) {
    // Arm teardown before publishing FINAL. The coordinator's acquire load of
    // submitted then observes both the record and its finish target, allowing
    // decode, snapshot publication, and deactivation in one wake.
    state.finish_target = submitted + 1;
    state.flush_requested.store(true, std::memory_order_relaxed);
    state.finish_requested.store(true, std::memory_order_release);
    state.accepting.store(false, std::memory_order_release);
  }
  state.submitted.store(submitted + 1, std::memory_order_release);
  release_producer(state.producer_active, state.producer_waiters);
  g.wake_async_stream_for_record();
  return async_enqueue_status::accepted;
}

// Park on the completion sequence only while the bounded ring is full. Every
// retired record notifies this counter, so the caller resumes after one slot
// opens instead of waiting for the entire submitted prefix as drain does.
void decoding_task_graph::impl::wait_for_async_enqueue_slot() {
  auto &g = *this;
  auto &state = g.async_streaming;
  while (true) {
    g.rethrow_async_stream_error();
    if (!state.active.load(std::memory_order_acquire) ||
        !state.accepting.load(std::memory_order_acquire)) {
      throw std::logic_error(
          "wait_for_async_enqueue_slot: no accepting session is active");
    }

    const uint64_t submitted = state.submitted.load(std::memory_order_acquire);
    const uint64_t completed = state.completed.load(std::memory_order_acquire);
    if (submitted - completed < state.capacity)
      return;
    wait_for_completion_change(state.completed, state.completion_waiters,
                               completed);
  }
}

// Existing result accessors require a coherent stable reference rather than a
// nonblocking try. Copy the latest publication into storage the coordinator
// never mutates, then return that storage to the graph facade.
const task_graph_streaming_result &
decoding_task_graph::impl::async_streaming_result() const {
  auto &state = async_streaming;
  std::lock_guard<std::mutex> lock(state.snapshot_mutex);
  if (state.snapshot_valid)
    state.read_snapshot = state.result_snapshot;
  return state.read_snapshot;
}

// Lazily begin a shot, apply bounded-queue backpressure, and preserve FINAL's
// synchronous end-of-shot boundary. BULK work returns before decoding; callers
// observe completion through streaming_result().
bool decoding_task_graph::impl::enqueue_syndrome_async(
    const uint8_t *syndrome, std::size_t len, std::optional<round_type> type,
    int cuda_device_id) {
  auto &state = async_streaming;
  if (syndrome == nullptr && len != 0)
    throw std::invalid_argument(
        "enqueue_syndrome: null input with nonzero length");

  // Reject overlap for the whole public call, including lazy startup and queue
  // backpressure, rather than only for the short ring-publication critical
  // section. This makes the documented single-producer contract independent of
  // scheduling and payload-copy duration.
  if (state.producer_call_active.exchange(true, std::memory_order_acq_rel))
    throw std::logic_error(
        "enqueue_syndrome: concurrent producers are unsupported");
  const auto release_call = [&state] {
    state.producer_call_active.store(false, std::memory_order_release);
  };

  try {
    if (finalized)
      CUDA_QEC_WARN("decoding_task_graph: dropping a round after the shot was "
                    "closed by a FINAL round; read the corrections and call "
                    "clear_corrections() (or reset_decoder()) to start the "
                    "next shot");
    if (finalized) {
      release_call();
      return false;
    }
    if (!state.active.load(std::memory_order_acquire))
      begin_async_streaming(cuda_device_id);

    while (true) {
      switch (try_enqueue_syndrome_async(syndrome, len, type)) {
      case async_enqueue_status::accepted:
        break;
      case async_enqueue_status::queue_full:
        wait_for_async_enqueue_slot();
        continue;
      case async_enqueue_status::failed:
        rethrow_async_stream_error();
        throw std::runtime_error("asynchronous task graph streaming failed");
      case async_enqueue_status::closed:
        throw std::logic_error(
            "enqueue_syndrome: asynchronous stream stopped while accepting "
            "one input operation");
      }
      break;
    }

    if (type != round_type::FINAL) {
      release_call();
      return false;
    }
    finish_async_streaming();
    const bool completed = async_streaming_result().operation_completed;
    release_call();
    return completed;
  } catch (...) {
    release_call();
    throw;
  }
}

// Wait for the accepted prefix visible at entry without closing the session or
// flushing guarded tails. Errors remain represented in the result snapshot.
void decoding_task_graph::impl::synchronize_async_streaming() {
  auto &g = *this;
  auto &state = g.async_streaming;
  wait_for_producer(state.producer_active, state.producer_waiters);
  const uint64_t target = state.submitted.load(std::memory_order_acquire);
  g.wake_async_stream();
  uint64_t completed = state.completed.load(std::memory_order_acquire);
  while (completed < target) {
    wait_for_completion_change(state.completed, state.completion_waiters,
                               completed);
    completed = state.completed.load(std::memory_order_acquire);
  }
}

// Stop publication, include any enqueue already inside its producer critical
// section, and drain through that final sequence without applying end-of-shot
// semantics.
void decoding_task_graph::impl::drain_async_streaming() {
  auto &g = *this;
  auto &state = g.async_streaming;
  if (!state.active.load(std::memory_order_acquire))
    throw std::logic_error("drain_async_streaming: no active session");

  state.accepting.store(false, std::memory_order_release);
  wait_for_producer(state.producer_active, state.producer_waiters);
  state.finish_target = state.submitted.load(std::memory_order_acquire);
  state.flush_requested.store(false, std::memory_order_relaxed);
  state.finish_requested.store(true, std::memory_order_release);
  g.wake_async_stream();
  wait_until_inactive(state.active);
  g.rethrow_async_stream_error();
}

// Drain accepted input, then apply end-of-shot semantics before releasing graph
// exclusivity.
void decoding_task_graph::impl::finish_async_streaming() {
  auto &g = *this;
  auto &state = g.async_streaming;
  // FINAL arms finish before publishing its queue record. Keep the general
  // path for callers that close an active session without a FINAL record.
  if (!state.finish_requested.load(std::memory_order_acquire)) {
    if (!state.active.load(std::memory_order_acquire))
      throw std::logic_error("finish_async_streaming: no active session");
    state.accepting.store(false, std::memory_order_release);
    wait_for_producer(state.producer_active, state.producer_waiters);
    state.finish_target = state.submitted.load(std::memory_order_acquire);
    state.flush_requested.store(true, std::memory_order_relaxed);
    state.finish_requested.store(true, std::memory_order_release);
    g.wake_async_stream();
  }
  if (state.active.load(std::memory_order_acquire)) {
    const auto completed = state.completed.load(std::memory_order_acquire);
    const bool short_tail = completed >= state.finish_target ||
                            state.finish_target - completed <= 1;
    wait_until_inactive(state.active, short_tail);
  }
  g.rethrow_async_stream_error();
}

// Stop publication and ask the coordinator to discard queued sequences after
// any currently executing decoder returns. Clear the latched exception for
// explicit reset-based recovery, while retaining the needs-reset tripwire.
void decoding_task_graph::impl::cancel_async_streaming() {
  auto &g = *this;
  auto &state = g.async_streaming;
  if (!state.active.load(std::memory_order_acquire))
    throw std::logic_error("cancel_async_streaming: no active session");

  state.accepting.store(false, std::memory_order_release);
  wait_for_producer(state.producer_active, state.producer_waiters);
  state.cancel_requested.store(true, std::memory_order_release);
  g.wake_async_stream();
  wait_until_inactive(state.active);
  state.error = nullptr;
  state.failed.store(false, std::memory_order_release);
  g.invalidate_async_stream_snapshot();
}

} // namespace cudaq::qec
