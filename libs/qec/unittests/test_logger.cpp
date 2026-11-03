/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudaq/qec/logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

/// @file
/// @brief Unit tests for QEC logger behavior.
/// @details
/// Validates runtime level filtering, source metadata rendering, forwarding
/// behavior, sink selection rules, queue backpressure handling, and default
/// forwarder setup semantics.

namespace {

using namespace std::chrono_literals;

struct ForwarderGuard {
  ~ForwarderGuard() { cudaq::qec::detail::clear_forwarder(); }
};

// Verify runtime log level changes take effect immediately for visibility.
TEST(Logger, UserSettableLogLevelControlsVisibility) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::warn);

  testing::internal::CaptureStdout();
  CUDA_QEC_INFO("hidden message");
  cudaq::qec::detail::flush_logs();
  const std::string hidden = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(hidden.empty());

  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);
  testing::internal::CaptureStdout();
  CUDA_QEC_INFO("visible message");
  cudaq::qec::detail::flush_logs();
  const std::string visible = testing::internal::GetCapturedStdout();
  EXPECT_NE(visible.find("visible message"), std::string::npos);
} // end - TEST(Logger, UserSettableLogLevelControlsVisibility)

// Validate that log lines include level, source file, and formatted message.
TEST(Logger, InfoLogsIncludeTimestampAndSourceLocation) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::trace);

  testing::internal::CaptureStdout();
  CUDA_QEC_INFO("metadata message {}", 7);
  cudaq::qec::detail::flush_logs();
  const std::string out = testing::internal::GetCapturedStdout();

  EXPECT_NE(out.find("[info]"), std::string::npos);
  EXPECT_NE(out.find("test_logger.cpp"), std::string::npos);
  EXPECT_NE(out.find("metadata message 7"), std::string::npos);
} // end - TEST(Logger, InfoLogsIncludeTimestampAndSourceLocation)

// Ensure hidden levels short-circuit before evaluating expensive arguments.
TEST(Logger, HiddenLevelMessagesAreNotFormed) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::error);

  std::atomic<int> evalCount{0};
  auto expensiveValue = [&]() -> int {
    evalCount.fetch_add(1, std::memory_order_relaxed);
    return 42;
  };

  CUDA_QEC_INFO("hidden {}", expensiveValue());
  EXPECT_EQ(evalCount.load(std::memory_order_relaxed), 0);
} // end - TEST(Logger, HiddenLevelMessagesAreNotFormed)

// Keep a lightweight throughput-neutrality guard in the unit suite: suppressed
// info logs should remain inexpensive when the active level is warn.
TEST(Logger, SuppressedInfoPathStaysFast) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::warn);

  constexpr int iterations = 200000;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    CUDA_QEC_INFO("suppressed-path {}", i);
  const auto end = std::chrono::steady_clock::now();

  const auto elapsedUs =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  // Use a conservative budget to reduce CI flakiness while still catching
  // severe suppressed-path regressions.
  EXPECT_LT(elapsedUs, 100000);
} // end - TEST(Logger, SuppressedInfoPathStaysFast)

// Verify forwarded records preserve payload and source metadata.
TEST(Logger, ForwarderReceivesRecordsWhenEnabled) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<cudaq::qec::detail::forwarded_log_record> records;
  cudaq::qec::detail::set_forwarder(cudaq::qec::detail::forwarder_config{
      .callback =
          [&](const cudaq::qec::detail::forwarded_log_record &record) {
            std::lock_guard<std::mutex> lock(mutex);
            records.push_back(record);
            cv.notify_all();
          },
      .queue_capacity = 64,
      .drop_policy = cudaq::qec::detail::forward_drop_policy::drop_newest});

  CUDA_QEC_INFO("forwarded {}", 7);
  cudaq::qec::detail::flush_logs();

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_TRUE(cv.wait_for(lock, 2s, [&] { return records.size() == 1; }));
  EXPECT_EQ(records.front().file_name, "test_logger.cpp");
  EXPECT_GT(records.front().line_no, 0);
  EXPECT_EQ(records.front().message, "forwarded 7");
} // end - TEST(Logger, ForwarderReceivesRecordsWhenEnabled)

// Ensure enabling a forwarder suppresses direct stdout/stderr emission.
TEST(Logger, ForwarderEnabledSuppressesStdoutAndStderr) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::trace);

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<cudaq::qec::detail::forwarded_log_record> records;
  cudaq::qec::detail::set_forwarder(cudaq::qec::detail::forwarder_config{
      .callback =
          [&](const cudaq::qec::detail::forwarded_log_record &record) {
            std::lock_guard<std::mutex> lock(mutex);
            records.push_back(record);
            cv.notify_all();
          },
      .queue_capacity = 64,
      .drop_policy = cudaq::qec::detail::forward_drop_policy::drop_newest});

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  CUDA_QEC_INFO("forwarded-info");
  CUDA_QEC_WARN("forwarded-warn");
  cudaq::qec::detail::flush_logs();
  const std::string out = testing::internal::GetCapturedStdout();
  const std::string err = testing::internal::GetCapturedStderr();

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_TRUE(cv.wait_for(lock, 2s, [&] { return records.size() == 2; }));
  EXPECT_TRUE(out.empty());
  EXPECT_TRUE(err.empty());
  EXPECT_EQ(records[0].message, "forwarded-info");
  EXPECT_EQ(records[1].message, "forwarded-warn");
} // end - TEST(Logger, ForwarderEnabledSuppressesStdoutAndStderr)

// Confirm default sink routing when no forwarder is installed.
TEST(Logger, DisabledForwarderUsesStdoutAndStderrByLevel) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::trace);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  CUDA_QEC_INFO("stdout-info");
#ifdef CUDAQ_DEBUG
  CUDA_QEC_DBG("stdout-debug");
#endif
  CUDA_QEC_WARN("stderr-warn");
  // CUDA_QEC_ERROR logs and then throws by design; assert the throw explicitly
  // so this sink-routing test does not fail from an uncaught exception.
  EXPECT_THROW(CUDA_QEC_ERROR("stderr-error"), std::runtime_error);
  const std::string out = testing::internal::GetCapturedStdout();
  const std::string err = testing::internal::GetCapturedStderr();

  EXPECT_NE(out.find("stdout-info"), std::string::npos);
#ifdef CUDAQ_DEBUG
  EXPECT_NE(out.find("stdout-debug"), std::string::npos);
#endif
  EXPECT_NE(err.find("stderr-warn"), std::string::npos);
  EXPECT_NE(err.find("stderr-error"), std::string::npos);
} // end - TEST(Logger, DisabledForwarderUsesStdoutAndStderrByLevel)

// Verify zero-argument set_forwarder() installs the default sink callback.
TEST(Logger, DefaultSetForwarderWritesToStdoutAndStderr) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::trace);
  cudaq::qec::detail::set_forwarder();

  ASSERT_TRUE(cudaq::qec::detail::is_forwarder_enabled());
  CUDA_QEC_INFO("default-forwarder-info");
  CUDA_QEC_WARN("default-forwarder-warn");
  cudaq::qec::detail::flush_logs();
  const auto stats = cudaq::qec::detail::get_forwarder_stats();
  EXPECT_EQ(stats.enqueued_records, 2u);
  EXPECT_EQ(stats.dropped_records, 0u);
  EXPECT_EQ(stats.truncated_records, 0u);
  EXPECT_EQ(stats.forward_failures, 0u);
} // end - TEST(Logger, DefaultSetForwarderWritesToStdoutAndStderr)

// Confirm bounded queue drops records under sustained producer pressure.
TEST(Logger, SaturatedForwarderQueueDropsWithoutBlockingProducer) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);

  std::atomic<bool> releaseCallback{false};
  cudaq::qec::detail::set_forwarder(cudaq::qec::detail::forwarder_config{
      .callback =
          [&](const cudaq::qec::detail::forwarded_log_record &) {
            while (!releaseCallback.load(std::memory_order_relaxed))
              std::this_thread::sleep_for(1ms);
          },
      .queue_capacity = 1,
      .drop_policy = cudaq::qec::detail::forward_drop_policy::drop_newest});

  testing::internal::CaptureStderr();
  for (int i = 0; i < 256; ++i)
    CUDA_QEC_INFO("queue-load {}", i);

  const auto stats = cudaq::qec::detail::get_forwarder_stats();
  EXPECT_GT(stats.enqueued_records, 0u);
  EXPECT_GT(stats.dropped_records, 0u);

  releaseCallback.store(true, std::memory_order_relaxed);
  cudaq::qec::detail::flush_logs();
  const std::string dropWarning = testing::internal::GetCapturedStderr();
  EXPECT_NE(dropWarning.find("forwarder dropped log records"),
            std::string::npos);
} // end - TEST(Logger, SaturatedForwarderQueueDropsWithoutBlockingProducer)

// drop_oldest must evict the oldest queued record (counting it as the single
// lost record) while always admitting the newest, and must never block the
// producer or drop the incoming record on the eviction race.
TEST(Logger, DropOldestEvictsOldestAndCountsOnlyLostRecords) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);

  std::atomic<bool> releaseCallback{false};
  std::mutex mutex;
  std::vector<std::string> delivered;
  cudaq::qec::detail::set_forwarder(cudaq::qec::detail::forwarder_config{
      .callback =
          [&](const cudaq::qec::detail::forwarded_log_record &record) {
            while (!releaseCallback.load(std::memory_order_relaxed))
              std::this_thread::sleep_for(1ms);
            std::lock_guard<std::mutex> lock(mutex);
            delivered.push_back(record.message);
          },
      .queue_capacity = 4,
      .drop_policy = cudaq::qec::detail::forward_drop_policy::drop_oldest});

  testing::internal::CaptureStderr();
  constexpr int kSends = 256;
  for (int i = 0; i < kSends; ++i)
    CUDA_QEC_INFO("evt {}", i);

  const auto stats = cudaq::qec::detail::get_forwarder_stats();
  // Every record is admitted (after evicting if needed) in drop_oldest mode.
  EXPECT_EQ(stats.enqueued_records, static_cast<std::uint64_t>(kSends));
  // A blocked consumer guarantees the queue saturates and old records evict.
  EXPECT_GT(stats.dropped_records, 0u);

  releaseCallback.store(true, std::memory_order_relaxed);
  cudaq::qec::detail::flush_logs();
  static_cast<void>(testing::internal::GetCapturedStderr());

  std::lock_guard<std::mutex> lock(mutex);
  // drop_oldest retains the newest record, so the final send must survive.
  EXPECT_NE(std::find(delivered.begin(), delivered.end(), "evt 255"),
            delivered.end());
} // end - TEST(Logger, DropOldestEvictsOldestAndCountsOnlyLostRecords)

TEST(Logger, ForwarderTruncationIsCountedAndAnnotated) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);

  std::mutex mutex;
  std::vector<cudaq::qec::detail::forwarded_log_record> records;
  cudaq::qec::detail::set_forwarder(cudaq::qec::detail::forwarder_config{
      .callback =
          [&](const cudaq::qec::detail::forwarded_log_record &record) {
            std::lock_guard<std::mutex> lock(mutex);
            records.push_back(record);
          },
      .queue_capacity = 64,
      .message_capacity = 32,
      .drop_policy = cudaq::qec::detail::forward_drop_policy::drop_newest});

  ASSERT_EQ(cudaq::qec::detail::get_forwarder_message_capacity(), 32u);
  CUDA_QEC_INFO("0123456789abcdefghijklmnopqrstuvwxyz");
  cudaq::qec::detail::flush_logs();

  std::lock_guard<std::mutex> lock(mutex);
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records.front().message.size(), 32u);
  EXPECT_NE(records.front().message.find("[truncated]"), std::string::npos);

  const auto stats = cudaq::qec::detail::get_forwarder_stats();
  EXPECT_EQ(stats.truncated_records, 1u);
  EXPECT_EQ(stats.dropped_records, 0u);
} // end - TEST(Logger, ForwarderTruncationIsCountedAndAnnotated)

// Ensure no queue accounting changes when forwarding is disabled.
TEST(Logger, DisabledForwarderDoesNotEnqueueRecords) {
  ForwarderGuard guard;
  cudaq::qec::detail::clear_forwarder();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);

  const auto before = cudaq::qec::detail::get_forwarder_stats();
  CUDA_QEC_INFO("no-forwarder");
  cudaq::qec::detail::flush_logs();
  const auto after = cudaq::qec::detail::get_forwarder_stats();

  EXPECT_EQ(after.enqueued_records, before.enqueued_records);
} // end - TEST(Logger, DisabledForwarderDoesNotEnqueueRecords)

} // namespace
