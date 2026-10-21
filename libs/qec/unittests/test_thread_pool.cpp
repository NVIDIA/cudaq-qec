/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file
///
/// Unit tests for the shared persistent QEC thread pool.

#include "../lib/decoding_task/thread_pool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>

namespace cudaq::qec::detail {
namespace {

TEST(QecThreadPool, ParallelForExecutesEveryTaskAndCanBeReused) {
  thread_pool pool(/*num_threads=*/4);
  std::array<std::atomic<std::size_t>, 32> executions{};
  std::array<std::atomic<std::size_t>, 32> workers{};

  pool.parallel_for(32, [&](std::size_t task, std::size_t worker) {
    executions[task].fetch_add(1, std::memory_order_relaxed);
    workers[task].store(worker, std::memory_order_relaxed);
  });
  for (std::size_t task = 0; task < executions.size(); ++task) {
    EXPECT_EQ(executions[task].load(std::memory_order_relaxed), 1);
    EXPECT_EQ(workers[task].load(std::memory_order_relaxed), task % 4);
  }

  pool.parallel_for(7, [&](std::size_t task, std::size_t worker) {
    executions[task].fetch_add(1, std::memory_order_relaxed);
    workers[task].store(worker, std::memory_order_relaxed);
  });
  for (std::size_t task = 0; task < executions.size(); ++task) {
    EXPECT_EQ(executions[task].load(std::memory_order_relaxed),
              task < 7 ? 2 : 1);
    EXPECT_EQ(workers[task].load(std::memory_order_relaxed), task % 4);
  }
} // end - ParallelForExecutesEveryTaskAndCanBeReused

TEST(QecThreadPool, SingleThreadRunsEveryTaskInline) {
  thread_pool pool(/*num_threads=*/1);
  std::array<std::size_t, 5> executions{};
  const std::thread::id caller = std::this_thread::get_id();

  pool.parallel_for(5, [&](std::size_t task, std::size_t worker) {
    ++executions[task];
    EXPECT_EQ(worker, 0);
    EXPECT_EQ(std::this_thread::get_id(), caller);
  });

  EXPECT_EQ(executions, (std::array<std::size_t, 5>{1, 1, 1, 1, 1}));
} // end - SingleThreadRunsEveryTaskInline

TEST(QecThreadPool, ParkedWorkersWakeForALaterPhase) {
  thread_pool pool(/*num_threads=*/4);
  std::atomic<std::size_t> workers{0};

  // Give every owned worker time to enter the between-phase atomic wait, then
  // exercise enough wake/park transitions to catch a missed notification.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (std::size_t phase = 0; phase < 100; ++phase) {
    pool.run_workers([&](std::size_t /*worker*/) {
      workers.fetch_add(1, std::memory_order_relaxed);
    });
  }

  EXPECT_EQ(workers.load(std::memory_order_relaxed), 400);
} // end - ParkedWorkersWakeForALaterPhase

TEST(QecThreadPool, WorkerExceptionIsJoinedAndRethrownToCaller) {
  thread_pool pool(/*num_threads=*/4);
  std::atomic<std::size_t> completed{0};
  std::atomic<std::size_t> throwing_worker{4};

  EXPECT_THROW(pool.parallel_for(
                   8,
                   [&](std::size_t task, std::size_t worker) {
                     if (task == 0) {
                       throwing_worker.store(worker, std::memory_order_relaxed);
                       throw std::runtime_error("worker failure");
                     }
                     completed.fetch_add(1, std::memory_order_relaxed);
                   }),
               std::runtime_error);
  EXPECT_EQ(throwing_worker.load(std::memory_order_relaxed), 0);

  completed.store(0, std::memory_order_relaxed);
  EXPECT_NO_THROW(
      pool.parallel_for(8, [&](std::size_t /*task*/, std::size_t /*worker*/) {
        completed.fetch_add(1, std::memory_order_relaxed);
      }));
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 8);
} // end - WorkerExceptionIsJoinedAndRethrownToCaller

TEST(QecThreadPool, RunWorkersPropagatesErrorsAndRemainsReusable) {
  thread_pool pool(/*num_threads=*/4);
  std::atomic<std::size_t> workers{0};

  EXPECT_THROW(pool.run_workers([&](std::size_t worker) {
    if (worker == 1)
      throw std::runtime_error("worker callback failure");
    workers.fetch_add(1, std::memory_order_relaxed);
  }),
               std::runtime_error);

  workers.store(0, std::memory_order_relaxed);
  EXPECT_NO_THROW(pool.run_workers([&](std::size_t /*worker*/) {
    workers.fetch_add(1, std::memory_order_relaxed);
  }));
  EXPECT_EQ(workers.load(std::memory_order_relaxed), 4);
} // end - RunWorkersPropagatesErrorsAndRemainsReusable

} // namespace
} // namespace cudaq::qec::detail
