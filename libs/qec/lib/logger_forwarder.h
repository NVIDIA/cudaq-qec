/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cudaq/qec/logger.h"
#include <array>
#include <cstdint>
#include <thread>

/// @file
/// @brief Internal asynchronous forwarder API for the QEC logger.
/// @details
/// This header is intentionally private to the logger implementation. It
/// exposes the lock-free queue payload type and a minimal control surface used
/// by `logger.cpp` to enqueue preformatted records and query forwarding stats.

namespace cudaq::qec::detail::forwarder_internal {

/// @brief Fixed-size queue record used by the forwarder ring buffer.
/// @details Uses bounded arrays so producer-side enqueue can avoid heap
/// allocation when forwarding is enabled.
struct queued_log_record {
  log_level level = log_level::info;
  std::uint64_t timestamp_ns = 0;
  int line_no = 0;
  std::thread::id thread_id;
  // The character buffers are intentionally left uninitialized: only the first
  // `file_name_len`/`message_len` bytes are ever written (and read back), so
  // zero-filling the full capacity on every record would waste cycles on the
  // producer hot path.
  std::array<char, 128> file_name;
  std::size_t file_name_len = 0;
  std::array<char, realtime_forwarder_max_message_capacity> message;
  std::size_t message_len = 0;
};

/// @brief Install/replace forwarder configuration and start worker.
void set(forwarder_config config);
/// @brief Disable forwarding and stop worker thread.
void clear();
/// @brief Return true when forwarding is active.
bool is_enabled();
/// @brief Reset forwarding counters for a fresh configuration epoch.
void reset_stats();
/// @brief Return current forwarding counters.
forwarder_stats stats();
/// @brief Block until queue drains (or forwarding is disabled).
void flush();
/// @brief Enqueue one fixed-size record on producer path.
void enqueue(queued_log_record record);
/// @brief Return active producer-side forwarded message capacity.
std::size_t message_capacity();
/// @brief Increment truncation counter (format/packing helper hook).
void note_truncation();

} // namespace cudaq::qec::detail::forwarder_internal
