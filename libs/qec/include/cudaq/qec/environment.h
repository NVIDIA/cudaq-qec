/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file environment.h
/// @brief Lightweight, dependency-free environment-variable parsing helpers.

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>

namespace cudaq::qec {

/// @brief Whether an environment variable is set (regardless of its value).
inline bool is_env_set(const char *name) {
  return std::getenv(name) != nullptr;
}

/// @brief Read an environment variable as a string.
/// @return The value if set (possibly empty), @p default_value otherwise.
inline std::string get_env_string(const char *name,
                                  std::string default_value = {}) {
  const char *value = std::getenv(name);
  if (!value)
    return default_value;
  return std::string(value);
}

/// @brief Interpret an environment variable as a boolean.
///
/// Returns @p default_value when unset. "1", "true", "yes", and "on" (case
/// insensitive) are treated as true; any other value is false.
inline bool get_env_bool(const char *name, bool default_value = false) {
  const char *value = std::getenv(name);
  if (!value)
    return default_value;
  std::string lowered(value);
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on";
}

/// @brief Interpret an environment variable as a signed integer (base 10).
///
/// Returns @p default_value when unset, empty, or not a valid `int` (this
/// includes trailing garbage and values outside the `int` range).
inline int get_env_int(const char *name, int default_value = 0) {
  const char *value = std::getenv(name);
  if (!value || *value == '\0')
    return default_value;
  errno = 0;
  char *end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max())
    return default_value;
  return static_cast<int>(parsed);
}

/// @brief Interpret an environment variable as an unsigned integer (base 10).
///
/// Returns @p default_value when unset, empty, or not a valid `unsigned int`.
inline unsigned int get_env_uint(const char *name,
                                 unsigned int default_value = 0) {
  const char *value = std::getenv(name);
  if (!value || *value == '\0')
    return default_value;
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed > std::numeric_limits<unsigned int>::max())
    return default_value;
  return static_cast<unsigned int>(parsed);
}

/// @brief Interpret an environment variable as a floating-point value.
///
/// Returns @p default_value when unset, empty, or not a valid number.
inline double get_env_double(const char *name, double default_value = 0.0) {
  const char *value = std::getenv(name);
  if (!value || *value == '\0')
    return default_value;
  errno = 0;
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || errno == ERANGE)
    return default_value;
  return parsed;
}

} // namespace cudaq::qec
