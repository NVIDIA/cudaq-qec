/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
/// @brief Persistent fork-join thread pool for QEC dependency executors.

#include "thread_pool.h"
#include "cudaq/qec/realtime/nvtx_helpers.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#define CUDAQ_QEC_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CUDAQ_QEC_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
#define CUDAQ_QEC_CPU_RELAX() ((void)0)
#endif

namespace cudaq::qec::detail {
namespace {

constexpr int kMainWaitSpinPauseIters = 4096;

// Reject affinity masks that cannot be represented by the platform API before
// any worker threads are started.
void validate_affinity_mask(const thread_pool::AffinityMask &affinity_mask) {
  for (std::int32_t cpu : affinity_mask) {
    if (cpu < 0) {
      throw std::invalid_argument(
          "thread_pool: affinity mask contains a negative CPU id");
    }
#if defined(__linux__)
    if (cpu >= CPU_SETSIZE) {
      throw std::out_of_range(
          "thread_pool: affinity CPU id exceeds CPU_SETSIZE");
    }
#endif
  }
}

// Apply the configured affinity policy to one owned worker. The caller thread
// is intentionally not modified because it is managed by the pool's caller.
void apply_worker_affinity(std::thread &worker, std::size_t worker_id,
                           const thread_pool::AffinityMask &affinity_mask,
                           thread_pool::AffinityPolicy affinity_policy) {
#if defined(__linux__)
  cpu_set_t cpu_set;
  std::size_t cpu_index = 0;
  std::int32_t cpu = 0;
  int error = 0;

  if (affinity_mask.empty())
    return;
  CPU_ZERO(&cpu_set);
  if (affinity_policy == thread_pool::AffinityPolicy::RoundRobin) {
    cpu_index = (worker_id - 1) % affinity_mask.size();
    cpu = affinity_mask[cpu_index];
    CPU_SET(static_cast<int>(cpu), &cpu_set);
  } else {
    for (std::int32_t allowed_cpu : affinity_mask)
      CPU_SET(static_cast<int>(allowed_cpu), &cpu_set);
  }
  error =
      pthread_setaffinity_np(worker.native_handle(), sizeof(cpu_set), &cpu_set);
  if (error != 0) {
    throw std::runtime_error("thread_pool: failed to set worker affinity: " +
                             std::string(std::strerror(error)));
  }
#else
  (void)worker;
  (void)worker_id;
  (void)affinity_policy;
  if (!affinity_mask.empty()) {
    throw std::runtime_error(
        "thread_pool: worker affinity is only supported on Linux");
  }
#endif
}

} // namespace

// Resolve the requested worker count, prepare optional tracing labels, and
// start the owned workers numbered [1, n_threads_). Worker 0 is the caller.
// If startup or affinity setup fails, stop and join every worker already
// created before propagating the exception.
thread_pool::thread_pool(std::size_t num_threads, AffinityMask affinity_mask,
                         AffinityPolicy affinity_policy)
    : n_threads_(num_threads), affinity_mask_(std::move(affinity_mask)),
      affinity_policy_(affinity_policy), phase_(0), remaining_workers_(0),
      shutdown_(false), current_trampoline_(nullptr), current_ctx_(nullptr),
      current_n_tasks_(0), phase_error_seen_(false), phase_error_(nullptr) {
  validate_affinity_mask(affinity_mask_);
  if (n_threads_ == 0) {
    n_threads_ = std::thread::hardware_concurrency();
    if (n_threads_ == 0)
      n_threads_ = 1;
  }
  if (affinity_policy_ == AffinityPolicy::RoundRobin &&
      !affinity_mask_.empty() && n_threads_ > affinity_mask_.size() + 1) {
    n_threads_ = affinity_mask_.size() + 1;
  }
#ifdef ENABLE_NVTX
  worker_idle_labels_.reserve(n_threads_);
  for (std::size_t worker = 0; worker < n_threads_; ++worker) {
    worker_idle_labels_.push_back("idle::worker_loop_" +
                                  std::to_string(worker));
  }
#endif
  if (n_threads_ <= 1)
    return;

  workers_.reserve(n_threads_ - 1);
  try {
    for (std::size_t worker = 1; worker < n_threads_; ++worker) {
      workers_.emplace_back([this, worker] { worker_loop(worker); });
      apply_worker_affinity(workers_.back(), worker, affinity_mask_,
                            affinity_policy_);
    }
  } catch (...) {
    stop_and_join_workers();
    throw;
  }
}

// Wake and join all owned workers before destroying their shared phase state.
thread_pool::~thread_pool() { stop_and_join_workers(); }

// Publish shutdown before advancing the phase counter. The phase notification
// wakes workers parked in atomic::wait so each can observe shutdown and exit.
void thread_pool::stop_and_join_workers() {
  if (workers_.empty())
    return;
  shutdown_.store(true, std::memory_order_release);
  phase_.fetch_add(1, std::memory_order_acq_rel);
  phase_.notify_all();
  for (auto &worker : workers_) {
    if (worker.joinable())
      worker.join();
  }
  workers_.clear();
}

// Publish one fork-join phase and execute worker 0's static stride inline.
// The release increment of phase_ makes the callback and context visible to
// workers that acquire the new phase. Every participant decrements
// remaining_workers_ exactly once, including after an exception, so the caller
// can safely join the whole phase before rethrowing its first error.
void thread_pool::run_phase(std::size_t num_tasks, Trampoline trampoline,
                            void *ctx) {
  {
    std::lock_guard<std::mutex> lock(phase_error_mutex_);
    phase_error_ = nullptr;
  }
  phase_error_seen_.store(false, std::memory_order_relaxed);
  current_trampoline_ = trampoline;
  current_ctx_ = ctx;
  current_n_tasks_ = num_tasks;
  remaining_workers_.store(n_threads_, std::memory_order_relaxed);
  phase_.fetch_add(1, std::memory_order_release);
  phase_.notify_all();

  try {
    for (std::size_t task = 0; task < num_tasks; task += n_threads_)
      trampoline(ctx, task, /*worker_id=*/0);
  } catch (...) {
    store_phase_error(std::current_exception());
  }
  remaining_workers_.fetch_sub(1, std::memory_order_acq_rel);

  int spins = 0;
  while (remaining_workers_.load(std::memory_order_acquire) > 0) {
    if (spins < kMainWaitSpinPauseIters) {
      ++spins;
      CUDAQ_QEC_CPU_RELAX();
    } else {
      std::this_thread::yield();
    }
  }
  rethrow_phase_error();
}

// Park until a new phase is published, snapshot its immutable work descriptor,
// and execute this worker's static stride. The acquire phase load pairs with
// run_phase's release publication; the final decrement publishes completion.
void thread_pool::worker_loop(std::size_t worker_id) {
  std::uint64_t my_phase = 0;
  while (true) {
    std::uint64_t current_phase = my_phase;
    {
      NVTX_RANGE(worker_idle_labels_[worker_id].c_str());
      // atomic::wait is specified to return only after the value differs from
      // its argument, even if the underlying wait wakes spuriously. Keep the
      // predicate explicit here as a defensive phase-lifecycle invariant.
      while (current_phase == my_phase &&
             !shutdown_.load(std::memory_order_acquire)) {
        phase_.wait(my_phase, std::memory_order_acquire);
        current_phase = phase_.load(std::memory_order_acquire);
      }
    }
    if (shutdown_.load(std::memory_order_acquire))
      return;
    my_phase = current_phase;

    Trampoline trampoline = current_trampoline_;
    void *ctx = current_ctx_;
    const std::size_t num_tasks = current_n_tasks_;
    const std::size_t stride = n_threads_;
    try {
      for (std::size_t task = worker_id; task < num_tasks; task += stride)
        trampoline(ctx, task, worker_id);
    } catch (...) {
      store_phase_error(std::current_exception());
    }
    remaining_workers_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

// Preserve only the first phase failure. The release flag provides a lock-free
// fast-path check while the mutex protects exception_ptr itself.
void thread_pool::store_phase_error(std::exception_ptr error) {
  {
    std::lock_guard<std::mutex> lock(phase_error_mutex_);
    if (phase_error_ == nullptr)
      phase_error_ = error;
  }
  phase_error_seen_.store(true, std::memory_order_release);
}

// Rethrow the stored phase failure after run_phase has joined every worker.
void thread_pool::rethrow_phase_error() {
  if (!phase_error_seen_.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::mutex> lock(phase_error_mutex_);
  if (phase_error_ != nullptr)
    std::rethrow_exception(phase_error_);
}

} // namespace cudaq::qec::detail
