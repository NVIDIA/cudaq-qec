/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/logger.h"
#include "logger_forwarder.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>

/// @file
/// @brief Core QEC logger front-end.
/// @details
/// Owns level parsing, timestamp/source formatting, and synchronous sink
/// output. When forwarding is enabled, this file packs records into bounded
/// buffers and delegates asynchronous delivery to `logger_forwarder.cpp`.

namespace cudaq::qec::detail {
namespace {

using clock = std::chrono::system_clock;

std::atomic<log_level> g_log_level{log_level::warn};
std::once_flag g_log_level_init_flag;

// Convert a log level token to lowercase for robust parsing.
std::string to_lower(std::string value) {
  for (auto &ch : value)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

// Parse CUDAQ_LOG_LEVEL text into internal enum.
std::optional<log_level> parse_log_level(const std::string_view level) {
  const std::string lower = to_lower(std::string(level));
  if (lower == "trace")
    return log_level::trace;
  if (lower == "debug")
    return log_level::debug;
  if (lower == "info")
    return log_level::info;
  if (lower == "warn" || lower == "warning")
    return log_level::warn;
  if (lower == "error")
    return log_level::error;
  return std::nullopt;
}

// Convert internal level enum to stable output label.
const char *log_level_name(const log_level level) {
  switch (level) {
  case log_level::trace:
    return "trace";
  case log_level::debug:
    return "debug";
  case log_level::info:
    return "info";
  case log_level::warn:
    return "warn";
  case log_level::error:
    return "error";
  }
  return "info";
}

// Return current wall-clock timestamp in nanoseconds.
std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          clock::now().time_since_epoch())
          .count());
}

// Render local-time timestamp with microsecond precision.
std::string format_timestamp(const clock::time_point tp) {
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(tp);
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          tp.time_since_epoch() - seconds.time_since_epoch())
                          .count();

  const std::time_t t = clock::to_time_t(tp);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.' << std::setw(6)
      << std::setfill('0') << micros;
  return out.str();
}

// Build one final text log line with timestamp/source metadata. The
// `[file:line]` segment is omitted when no filename is supplied (empty or
// null), allowing source-free log lines via the `log()` helper.
std::string compose_line(const log_level level, const std::string_view message,
                         const char *file_name, const int line_no,
                         const clock::time_point ts) {
  std::ostringstream out;
  out << '[' << format_timestamp(ts) << "] [" << log_level_name(level) << "] ";
  if (file_name != nullptr && file_name[0] != '\0')
    out << '[' << path_to_file_name(file_name) << ':' << line_no << "] ";
  out << message;
  return out.str();
}

// Lightweight filename extraction view to avoid allocations.
std::string_view path_to_file_name_view(const std::string_view full_file_path) {
  const auto pos = full_file_path.find_last_of("/\\");
  if (pos == std::string_view::npos)
    return full_file_path;
  return full_file_path.substr(pos + 1);
}

template <std::size_t N>
std::size_t copy_to_fixed(std::array<char, N> &dest, const std::string_view src,
                          std::size_t cap, bool *truncated = nullptr) {
  static_assert(N > 0);
  cap = std::min(cap, N - 1);
  const std::size_t copied = std::min(src.size(), cap);
  if (copied > 0)
    std::memcpy(dest.data(), src.data(), copied);
  dest[copied] = '\0';
  if (truncated)
    *truncated = src.size() > copied;
  return copied;
}

// Lazily initialize runtime log level from CUDAQ_LOG_LEVEL once.
void initialize_log_level_from_env() {
  std::call_once(g_log_level_init_flag, [] {
    if (const char *env = std::getenv("CUDAQ_LOG_LEVEL")) {
      if (const auto parsed = parse_log_level(env))
        g_log_level.store(*parsed, std::memory_order_relaxed);
    }
  });
}

void emit(const log_level level, const std::string_view raw_message,
          const char *file_name, const int line_no) {
  // Forwarder mode uses fixed-size packing to keep producer-side allocation
  // predictable and bounded.
  if (forwarder_internal::is_enabled()) {
    forwarder_internal::queued_log_record record;
    const std::size_t message_cap = forwarder_internal::message_capacity();
    record.level = level;
    record.timestamp_ns = now_ns();
    record.file_name_len =
        copy_to_fixed(record.file_name, path_to_file_name_view(file_name),
                      record.file_name.size() - 1);
    record.line_no = line_no;
    bool truncated = false;
    record.message_len =
        copy_to_fixed(record.message, raw_message, message_cap, &truncated);
    if (truncated) {
      forwarder_internal::note_truncation();
      if (record.message_len >= realtime_truncation_suffix.size()) {
        const std::size_t start =
            record.message_len - realtime_truncation_suffix.size();
        std::copy(realtime_truncation_suffix.begin(),
                  realtime_truncation_suffix.end(),
                  record.message.begin() + start);
      }
    }
    record.thread_id = std::this_thread::get_id();
    forwarder_internal::enqueue(std::move(record));
    return;
  }

  const auto ts = clock::now();
  const std::string full_line =
      compose_line(level, raw_message, file_name, line_no, ts);
  FILE *stream =
      (level == log_level::warn || level == log_level::error) ? stderr : stdout;
  std::fputs(full_line.c_str(), stream);
  std::fputc('\n', stream);
}

} // namespace

// Compare candidate level against current runtime threshold.
bool should_log(const log_level level) {
  initialize_log_level_from_env();
  return static_cast<int>(level) >=
         static_cast<int>(g_log_level.load(std::memory_order_relaxed));
}

// Public API: install/replace asynchronous forwarding callback.
void set_forwarder(forwarder_config config) {
  forwarder_internal::set(std::move(config));
  forwarder_internal::reset_stats();
}

// Public API: enable forwarding with default stdout/stderr callback.
void set_forwarder() {
  set_forwarder(forwarder_config{
      .callback =
          [](forwarded_log_record &&record) {
            const auto tp = clock::time_point(std::chrono::nanoseconds(
                static_cast<std::int64_t>(record.timestamp_ns)));
            const std::string full_line =
                compose_line(record.level, record.message,
                             record.file_name.c_str(), record.line_no, tp);
            FILE *stream = (record.level == log_level::warn ||
                            record.level == log_level::error)
                               ? stderr
                               : stdout;
            std::fputs(full_line.c_str(), stream);
            std::fputc('\n', stream);
          },
      .queue_capacity = forwarder_config{}.queue_capacity,
      .drop_policy = forwarder_config{}.drop_policy});
}

// Public API: disable asynchronous forwarding.
void clear_forwarder() { forwarder_internal::clear(); }
// Public API: report whether forwarding is active.
bool is_forwarder_enabled() { return forwarder_internal::is_enabled(); }
std::size_t get_forwarder_message_capacity() {
  return forwarder_internal::message_capacity();
}
void record_forwarder_message_truncation() {
  forwarder_internal::note_truncation();
}
// Public API: return forwarding counters snapshot.
forwarder_stats get_forwarder_stats() { return forwarder_internal::stats(); }

// Public API: direct sinks for already-formatted messages.
void trace(const std::string_view msg) {
  emit(log_level::trace, msg, "<unknown>", 0);
}
void info(const std::string_view msg) {
  emit(log_level::info, msg, "<unknown>", 0);
}
void debug(const std::string_view msg) {
  emit(log_level::debug, msg, "<unknown>", 0);
}
void warn(const std::string_view msg) {
  emit(log_level::warn, msg, "<unknown>", 0);
}
void error(const std::string_view msg) {
  emit(log_level::error, msg, "<unknown>", 0);
}

// Strip directory prefix to keep compact source metadata output.
std::string path_to_file_name(const std::string_view full_file_path) {
  const auto pos = full_file_path.find_last_of("/\\");
  if (pos == std::string_view::npos)
    return std::string(full_file_path);
  return std::string(full_file_path.substr(pos + 1));
}

// Override runtime logging threshold (primarily used in tests).
void set_log_level(const log_level level) {
  g_log_level.store(level, std::memory_order_relaxed);
}

// Return current runtime logging threshold.
log_level get_log_level() {
  return g_log_level.load(std::memory_order_relaxed);
}

// Flush primary sinks and wait for forwarded queue drain.
void flush_logs() {
  std::fflush(stdout);
  std::fflush(stderr);
  forwarder_internal::flush();
}

// Entry points used by templated header helpers after formatting.
void log_message_formatted(log_level level, std::string formatted_message,
                           const char *file_name, int line_no) {
  emit(level, formatted_message, file_name, line_no);
}

void log_message_buffer(log_level level, const char *formatted_message,
                        std::size_t message_len, const char *file_name,
                        int line_no) {
  emit(level, std::string_view(formatted_message, message_len), file_name,
       line_no);
}

void log_with_timestamp_formatted(std::string formatted_message,
                                  const char *file_name, int line_no) {
  emit(log_level::info, formatted_message, file_name, line_no);
}

void log_with_timestamp_buffer(const char *formatted_message,
                               std::size_t message_len, const char *file_name,
                               int line_no) {
  emit(log_level::info, std::string_view(formatted_message, message_len),
       file_name, line_no);
}

} // namespace cudaq::qec::detail
