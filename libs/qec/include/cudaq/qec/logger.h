/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

/// @file
/// @brief QEC logging interface used by runtime and plugin code.
/// @details
/// Declares log levels, forwarding configuration, and lightweight templated
/// formatting helpers used by logging macros (`CUDA_QEC_INFO`, `CUDA_QEC_WARN`,
/// `CUDA_QEC_ERROR`, `CUDA_QEC_DBG`) and direct
/// `cudaq::qec::info/warn/error/debug` calls. When forwarding is enabled, the
/// producer path uses bounded fixed-size buffers, supports configurable message
/// capacity, and tracks truncation and drop statistics.

namespace cudaq::qec {

// Keep all spdlog headers hidden in the implementation file.
namespace detail {
// This enum must match spdlog::level enums. This is checked via static_assert
// in logger.cpp.
/// @brief Severity levels supported by the QEC logger.
enum class log_level { trace, debug, info, warn, error };
/// @brief Default forwarded message payload capacity in bytes.
inline constexpr std::size_t realtime_forwarder_default_message_capacity = 512;
/// @brief Upper bound for forwarded message payload capacity in bytes.
inline constexpr std::size_t realtime_forwarder_max_message_capacity = 4096;
/// @brief Suffix appended to forwarded messages when truncation occurs.
inline constexpr std::string_view realtime_truncation_suffix =
    " ...[truncated]";

/// @brief Log payload forwarded to an optional background callback.
struct forwarded_log_record {
  log_level level;
  std::uint64_t timestamp_ns;
  std::string file_name;
  int line_no;
  std::string message;
  std::thread::id thread_id;
};

/// @brief Queue backpressure policy for forwarded records.
enum class forward_drop_policy { drop_newest, drop_oldest };

/// @brief Runtime configuration for asynchronous forwarding.
struct forwarder_config {
  /// @brief Callback executed on the forwarder worker thread.
  /// @details The record is passed as an rvalue: the worker owns a fresh,
  /// per-record instance that is discarded after the callback returns, so a
  /// callback may move out of it (e.g. `std::move(record)` to re-queue it) to
  /// avoid copying the heap-allocated `file_name`/`message` strings. Read-only
  /// callbacks taking `const ForwardedLogRecord &` remain valid.
  std::function<void(forwarded_log_record &&)> callback;
  /// @brief Bounded queue capacity (records), clamped to at least 1.
  std::size_t queue_capacity = 1024;
  /// @brief Max forwarded message bytes copied on producer path.
  /// @details Values are clamped to [1,
  /// realtime_forwarder_max_message_capacity].
  std::size_t message_capacity = realtime_forwarder_default_message_capacity;
  /// @brief Overflow policy when the forwarder queue is saturated.
  forward_drop_policy drop_policy = forward_drop_policy::drop_newest;
};

/// @brief Runtime counters for asynchronous forwarding behavior.
struct forwarder_stats {
  /// @brief Records successfully enqueued to the forwarder queue.
  std::uint64_t enqueued_records = 0;
  /// @brief Records dropped due to queue saturation/overflow policy.
  std::uint64_t dropped_records = 0;
  /// @brief Records whose message payload was truncated to fit capacity.
  std::uint64_t truncated_records = 0;
  /// @brief Callback invocations that threw exceptions.
  std::uint64_t forward_failures = 0;
};

/// @brief Return true if the given level is currently enabled.
bool should_log(const log_level level);
/// @brief Install or replace the optional async forwarding callback.
void set_forwarder(forwarder_config config);
/// @brief Enable forwarding with a default worker callback to stdout/stderr.
void set_forwarder();
/// @brief Disable forwarding and tear down the forwarding worker.
void clear_forwarder();
/// @brief Return true when forwarding is enabled.
bool is_forwarder_enabled();
/// @brief Return active forwarded-message capacity in bytes.
std::size_t get_forwarder_message_capacity();
/// @brief Return a snapshot of forwarding counters.
forwarder_stats get_forwarder_stats();
/// @brief Internal helper to record forwarder truncation events.
/// @details Primarily used by templated formatting helpers.
void record_forwarder_message_truncation();
/// @brief Emit a preformatted trace message.
void trace(const std::string_view msg);
/// @brief Emit a preformatted info message.
void info(const std::string_view msg);
/// @brief Emit a preformatted debug message.
void debug(const std::string_view msg);
/// @brief Emit a preformatted warning message.
void warn(const std::string_view msg);
/// @brief Emit a preformatted error message.
void error(const std::string_view msg);
/// @brief Extract filename from a path-like string.
std::string path_to_file_name(const std::string_view full_file_path);

// Test/debug helpers. Production callers configure the level via the
// CUDAQ_LOG_LEVEL environment variable.
/// @brief Override log level at runtime.
void set_log_level(log_level level);
/// @brief Return the current runtime log level.
log_level get_log_level();

// Flushes any buffered log output. Useful in tests that need to inspect
// captured stdout immediately after emitting an info/debug message
// (initializeLogger only enables flush_on(warn)).
/// @brief Flush logger sinks and pending forwarded records.
void flush_logs();

/// @brief Emit a formatted message with file and line metadata.
void log_message_formatted(log_level level, std::string formatted_message,
                           const char *file_name, int line_no);
/// @brief Emit a preformatted character buffer with explicit length.
/// @details The implementation must consume/copy the buffer during the call.
void log_message_buffer(log_level level, const char *formatted_message,
                        std::size_t message_len, const char *file_name,
                        int line_no);
/// @brief Emit a formatted timestamped message with file and line metadata.
void log_with_timestamp_formatted(std::string formatted_message,
                                  const char *file_name, int line_no);
/// @brief Emit a preformatted timestamped buffer with explicit length.
/// @details The implementation must consume/copy the buffer during the call.
void log_with_timestamp_buffer(const char *formatted_message,
                               std::size_t message_len, const char *file_name,
                               int line_no);

/// @brief Format a message and arguments using `fmt`.
template <typename... Args>
std::string format_message(const std::string_view message, Args &&...args) {
  return fmt::vformat(message, fmt::make_format_args(args...));
}

/// @brief Format and emit a message for the provided log level.
/// @details If the forwarder is enabled, this path uses bounded formatting into
/// fixed storage and may truncate with `realtime_truncation_suffix`.
template <typename... Args>
void log_message(log_level level, const std::string_view message,
                 const char *file_name, int line_no, Args &&...args) {
  if (!should_log(level))
    return;
  if (is_forwarder_enabled()) {
    // Intentionally left uninitialized: `fmt` writes `used` bytes and we
    // explicitly NUL-terminate at `buffer[used]`, so zero-filling the full
    // capacity on every call would be wasted work on the hot path.
    std::array<char, realtime_forwarder_max_message_capacity> buffer;
    const std::size_t cap =
        std::min(get_forwarder_message_capacity(), buffer.size() - 1);
    auto result = fmt::vformat_to_n(buffer.data(), cap, message,
                                    fmt::make_format_args(args...));
    auto used = std::min<std::size_t>(result.size, cap);
    if (result.size > cap) {
      record_forwarder_message_truncation();
      if (used >= realtime_truncation_suffix.size()) {
        const std::size_t start = used - realtime_truncation_suffix.size();
        std::copy(realtime_truncation_suffix.begin(),
                  realtime_truncation_suffix.end(), buffer.begin() + start);
      }
    }
    buffer[used] = '\0';
    log_message_buffer(level, buffer.data(), used, file_name, line_no);
    return;
  }
  log_message_formatted(level,
                        format_message(message, std::forward<Args>(args)...),
                        file_name, line_no);
}
} // namespace detail

/// These types seek to enable automated injection of the source location of the
/// `cudaq::qec::info()` or `debug()` call. The actual formatting is out-of-line
/// in logger.cpp so callers do not need to parse `fmt` or `chrono` headers.
#define CUDA_QEC_LOGGER_DEDUCTION_STRUCT(NAME)                                 \
  template <typename... Args>                                                  \
  struct NAME {                                                                \
    NAME(const std::string_view message, Args &&...args,                       \
         const char *file_name = __builtin_FILE(),                             \
         int line_no = __builtin_LINE()) {                                     \
      if (detail::should_log(detail::log_level::NAME))                         \
        detail::log_message(detail::log_level::NAME, message, file_name,       \
                            line_no, std::forward<Args>(args)...);             \
    }                                                                          \
  };                                                                           \
  template <typename... Args>                                                  \
  NAME(const std::string_view, Args &&...) -> NAME<Args...>;

CUDA_QEC_LOGGER_DEDUCTION_STRUCT(info);
CUDA_QEC_LOGGER_DEDUCTION_STRUCT(warn);
CUDA_QEC_LOGGER_DEDUCTION_STRUCT(error);

#ifdef CUDAQ_DEBUG
CUDA_QEC_LOGGER_DEDUCTION_STRUCT(debug);
#else
// Remove cudaq::debug log messages from Release binaries.
template <typename... Args>
void debug(const std::string_view, Args &&...) {}
#endif

/// @brief Log a message with timestamp but without file/line metadata.
/// @details This helper always emits regardless of log level. The
/// `[file:line]` segment is intentionally omitted from the output; use the
/// `CUDA_QEC_*` macros instead if source location is required.
template <typename... Args>
void log(const std::string_view message, Args &&...args) {
  if (detail::is_forwarder_enabled()) {
    // Intentionally left uninitialized; see log_message() for rationale.
    std::array<char, detail::realtime_forwarder_max_message_capacity> buffer;
    const std::size_t cap =
        std::min(detail::get_forwarder_message_capacity(), buffer.size() - 1);
    auto result = fmt::vformat_to_n(buffer.data(), cap, message,
                                    fmt::make_format_args(args...));
    auto used = std::min<std::size_t>(result.size, cap);
    if (result.size > cap) {
      detail::record_forwarder_message_truncation();
      if (used >= detail::realtime_truncation_suffix.size()) {
        const std::size_t start =
            used - detail::realtime_truncation_suffix.size();
        std::copy(detail::realtime_truncation_suffix.begin(),
                  detail::realtime_truncation_suffix.end(),
                  buffer.begin() + start);
      }
    }
    buffer[used] = '\0';
    // Pass an empty filename so compose_line omits the [file:line] segment.
    detail::log_with_timestamp_buffer(buffer.data(), used, "", 0);
    return;
  }
  detail::log_with_timestamp_formatted(
      detail::format_message(message, std::forward<Args>(args)...), "", 0);
}

} // namespace cudaq::qec

// The following macros avoid the unnecessary processing cost of argument
// evaluation and string formation until after the log level check is done.
#define CUDA_QEC_LOG_IMPL(LEVEL, msg, ...)                                     \
  do {                                                                         \
    if (::cudaq::qec::detail::should_log(                                      \
            ::cudaq::qec::detail::log_level::LEVEL)) {                         \
      ::cudaq::qec::detail::log_message(                                       \
          ::cudaq::qec::detail::log_level::LEVEL, msg, __FILE__,               \
          __LINE__ __VA_OPT__(, ) __VA_ARGS__);                                \
    }                                                                          \
  } while (false)

#define CUDA_QEC_ERROR_IMPL(msg, ...)                                          \
  do {                                                                         \
    ::cudaq::qec::detail::log_message(::cudaq::qec::detail::log_level::error,  \
                                      msg, __FILE__,                           \
                                      __LINE__ __VA_OPT__(, ) __VA_ARGS__);    \
    throw std::runtime_error(                                                  \
        ::cudaq::qec::detail::format_message(msg __VA_OPT__(, ) __VA_ARGS__)); \
  } while (false)

#define CUDA_QEC_ERROR(...) CUDA_QEC_ERROR_IMPL(__VA_ARGS__)
#define CUDA_QEC_WARN(...) CUDA_QEC_LOG_IMPL(warn, __VA_ARGS__)
#define CUDA_QEC_INFO(...) CUDA_QEC_LOG_IMPL(info, __VA_ARGS__)

#ifdef CUDAQ_DEBUG
#define CUDA_QEC_DBG(...) CUDA_QEC_LOG_IMPL(debug, __VA_ARGS__)
#else
#define CUDA_QEC_DBG(...)
#endif
