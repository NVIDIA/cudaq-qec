/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
///
/// Persistent fork-join thread pool for QEC dependency executors.
/// This QEC-local implementation deliberately exposes only the phase
/// operations used by decoding task graphs.

#ifndef CUDAQ_QEC_DETAIL_THREAD_POOL_H
#define CUDAQ_QEC_DETAIL_THREAD_POOL_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#ifdef ENABLE_NVTX
#include <string>
#endif
#include <thread>
#include <type_traits>
#include <vector>

namespace cudaq::qec::detail {

/// Persistent fork-join pool. The calling thread participates as worker 0;
/// owned worker threads are numbered `[1, num_threads)`. Idle workers park on
/// an atomic phase wait between calls. Calls that start work on the same pool
/// must not overlap; external synchronization is required if multiple callers
/// share one instance.
class thread_pool {
public:
  /// Raw callback receiving context, task index, and worker id.
  using Trampoline = void (*)(void *ctx, std::size_t task, std::size_t worker);
  /// List of CPU ids that owned worker threads may run on (Linux only).
  using AffinityMask = std::vector<std::int32_t>;

  /// How an affinity mask is mapped onto owned worker threads.
  enum class AffinityPolicy : std::uint8_t {
    /// Each worker may run on any CPU in the mask.
    Float,
    /// Each worker is pinned to one CPU chosen round-robin from the mask.
    RoundRobin,
  };

  /// @brief Construct a persistent fork-join pool.
  ///
  /// The total worker count includes the calling thread as worker 0. A zero
  /// count resolves to hardware concurrency, falling back to one. No owned
  /// thread is created when the resolved count is one.
  ///
  /// @param num_threads Total worker count, including the calling thread.
  /// @param affinity_mask CPUs available to owned workers; empty means no
  /// affinity constraint.
  /// @param affinity_policy How to assign the affinity mask to owned workers.
  /// @throws std::invalid_argument If the mask contains a negative CPU id.
  /// @throws std::out_of_range If a Linux CPU id exceeds `CPU_SETSIZE`.
  /// @throws std::runtime_error If affinity setup fails.
  /// @throws std::system_error If worker creation fails.
  explicit thread_pool(std::size_t num_threads = 0,
                       AffinityMask affinity_mask = {},
                       AffinityPolicy affinity_policy = AffinityPolicy::Float);

  /// @brief Stop and join all owned workers.
  ~thread_pool();

  thread_pool(const thread_pool &) = delete;
  thread_pool &operator=(const thread_pool &) = delete;

  /// @brief Return the total worker count, including the calling thread.
  std::size_t num_threads() const { return n_threads_; }

  /// @brief Return the configured CPU affinity mask.
  const AffinityMask &affinity_mask() const { return affinity_mask_; }

  /// @brief Return the configured affinity assignment policy.
  AffinityPolicy affinity_policy() const { return affinity_policy_; }

  /// @brief Run a statically scheduled fork-join phase.
  ///
  /// Runs every task in `[0, num_tasks)` exactly once with
  /// `worker_id = task_index % num_threads` and blocks until all workers
  /// finish. The callable may execute concurrently and must remain alive until
  /// this function returns.
  ///
  /// @tparam Fn Callable accepting `(std::size_t task, std::size_t worker)`.
  /// @param num_tasks Number of task indices to execute.
  /// @param fn Task callable.
  /// @throws Any exception raised by a task, after all workers join the phase.
  template <typename Fn>
  void parallel_for(std::size_t num_tasks, Fn &&fn) {
    if (num_tasks == 0)
      return;
    if (n_threads_ <= 1) {
      for (std::size_t task = 0; task < num_tasks; ++task)
        fn(task, /*worker_id=*/0);
      return;
    }
    auto trampoline = +[](void *ctx, std::size_t task, std::size_t worker) {
      (*static_cast<std::remove_reference_t<Fn> *>(ctx))(task, worker);
    };
    run_phase(num_tasks, trampoline, &fn);
  }

  /// @brief Run one callback per worker.
  ///
  /// Worker 0 executes on the calling thread. This is intended for callbacks
  /// that implement their own dynamic work loop, such as the task-graph ready
  /// queue.
  ///
  /// @tparam Fn Callable accepting `(std::size_t worker)`.
  /// @param fn Worker callable.
  /// @throws Any exception raised by a worker, after all workers join the
  /// phase.
  template <typename Fn>
  void run_workers(Fn &&fn) {
    if (n_threads_ <= 1) {
      fn(/*worker_id=*/0);
      return;
    }
    auto trampoline = +[](void *ctx, std::size_t /*task*/, std::size_t worker) {
      (*static_cast<std::remove_reference_t<Fn> *>(ctx))(worker);
    };
    run_phase(n_threads_, trampoline, &fn);
  }

private:
  void run_phase(std::size_t num_tasks, Trampoline trampoline, void *ctx);
  void worker_loop(std::size_t worker_id);
  void store_phase_error(std::exception_ptr error);
  void rethrow_phase_error();
  void stop_and_join_workers();

  std::size_t n_threads_;
  std::vector<std::thread> workers_;
  AffinityMask affinity_mask_;
  AffinityPolicy affinity_policy_;

#ifdef ENABLE_NVTX
  std::vector<std::string> worker_idle_labels_;
#endif

  alignas(64) std::atomic<std::uint64_t> phase_;
  alignas(64) std::atomic<std::uint64_t> remaining_workers_;
  alignas(64) std::atomic<bool> shutdown_;

  Trampoline current_trampoline_;
  void *current_ctx_;
  std::size_t current_n_tasks_;

  alignas(64) std::atomic<bool> phase_error_seen_;
  std::mutex phase_error_mutex_;
  std::exception_ptr phase_error_;
};

} // namespace cudaq::qec::detail

#endif // CUDAQ_QEC_DETAIL_THREAD_POOL_H
