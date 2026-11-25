/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
///
/// Unit tests for the shared persistent QEC thread pool, and the idle waits
/// its workers use: a spin bounded by time, and the gate that sends one idle
/// worker at a time to search for work after a scheduler change.

#include "../lib/decoding_task/thread_pool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>

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

using std::chrono::steady_clock;

// A wait that must time out: long enough to measure, short enough to keep
// the suite fast.
constexpr std::chrono::milliseconds short_budget{20};
// A wait that must end by an event, never by its budget.
constexpr std::chrono::seconds long_budget{30};

TEST(IdleSpin, SpinForReturnsOnceDone) {
  int calls = 0;
  const auto start = steady_clock::now();

  EXPECT_TRUE(spin_for(long_budget, [&] { return ++calls == 100; }));
  EXPECT_EQ(calls, 100);
  EXPECT_LT(steady_clock::now() - start, long_budget);
}

TEST(IdleSpin, SpinForGivesUpAfterItsBudget) {
  const auto start = steady_clock::now();

  EXPECT_FALSE(spin_for(short_budget, [] { return false; }));
  EXPECT_GE(steady_clock::now() - start, short_budget);
}

TEST(IdleSpin, SpinForChecksOnceWithoutABudget) {
  int calls = 0;

  EXPECT_FALSE(spin_for(std::chrono::nanoseconds{0}, [&] {
    ++calls;
    return false;
  }));
  EXPECT_EQ(calls, 1);
}

TEST(IdleSearchGate, WaitTimesOutWithoutAChange) {
  idle_search_gate gate;
  const auto seen = gate.generation();
  const auto start = steady_clock::now();

  EXPECT_EQ(gate.wait(seen, short_budget), idle_wake::timeout);
  EXPECT_GE(steady_clock::now() - start, short_budget);
}

TEST(IdleSearchGate, AChangeSendsAWaiterToSearch) {
  idle_search_gate gate;
  const auto seen = gate.generation();

  gate.publish();
  EXPECT_NE(gate.generation(), seen);
  EXPECT_EQ(gate.wait(seen, long_budget), idle_wake::search);
  gate.end_search();
}

TEST(IdleSearchGate, OnlyOneWaiterSearchesAtATime) {
  idle_search_gate gate;
  const auto seen = gate.generation();

  gate.publish();
  ASSERT_EQ(gate.wait(seen, long_budget), idle_wake::search);
  // The change is not yet searched, but the gate is held.
  EXPECT_EQ(gate.wait(seen, short_budget), idle_wake::timeout);
  gate.end_search();
  // Released without a search: the change still needs one.
  EXPECT_EQ(gate.wait(seen, long_budget), idle_wake::search);
  gate.end_search();
}

TEST(IdleSearchGate, ASearchCoversEveryEarlierChange) {
  idle_search_gate gate;
  const auto seen = gate.generation();

  gate.publish();
  gate.publish();
  ASSERT_EQ(gate.wait(seen, long_budget), idle_wake::search);
  gate.begin_search();
  gate.end_search();
  EXPECT_EQ(gate.wait(seen, short_budget), idle_wake::timeout);

  // A change after the search still needs one.
  gate.publish();
  EXPECT_EQ(gate.wait(seen, long_budget), idle_wake::search);
  gate.end_search();
}

TEST(IdleSearchGate, InterruptReleasesEveryWaiter) {
  constexpr int waiters = 4;
  idle_search_gate gate;
  const auto seen = gate.generation();
  std::atomic<int> interrupted{0};
  std::vector<std::thread> threads;

  for (int i = 0; i < waiters; ++i)
    threads.emplace_back([&] {
      if (gate.wait(seen, long_budget) == idle_wake::interrupted)
        interrupted.fetch_add(1, std::memory_order_relaxed);
    });
  gate.interrupt();
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(interrupted.load(), waiters);

  gate.clear_interrupt();
  EXPECT_EQ(gate.wait(seen, short_budget), idle_wake::timeout);
}

// Many spinning workers and one change: one searches, and its search covers
// the change for the rest, which keep spinning until interrupted.
TEST(IdleSearchGate, ManyWaitersSendOneSearchPerChange) {
  constexpr int waiters = 6;
  idle_search_gate gate;
  const auto seen = gate.generation();
  std::atomic<int> searches{0};
  std::atomic<int> interrupted{0};
  std::vector<std::thread> threads;

  for (int i = 0; i < waiters; ++i)
    threads.emplace_back([&] {
      const auto wake = gate.wait(seen, long_budget);
      if (wake == idle_wake::search) {
        gate.begin_search();
        searches.fetch_add(1, std::memory_order_relaxed);
        gate.end_search();
        return;
      }
      if (wake == idle_wake::interrupted)
        interrupted.fetch_add(1, std::memory_order_relaxed);
    });
  gate.publish();
  const auto deadline = steady_clock::now() + long_budget;
  while (searches.load() == 0 && steady_clock::now() < deadline)
    std::this_thread::yield();
  // Give the other waiters time to search the change again, as they would
  // if the search did not cover it.
  std::this_thread::sleep_for(short_budget);
  gate.interrupt();
  for (auto &thread : threads)
    thread.join();

  EXPECT_EQ(searches.load(), 1);
  EXPECT_EQ(interrupted.load(), waiters - 1);
}

} // namespace
} // namespace cudaq::qec::detail
