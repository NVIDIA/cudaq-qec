/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Internal header: the type-erased value that flows along a task-graph edge.
// Payload types are registered through decoding_task_graph.h.

#pragma once
#include "cudaq/qec/decoding_task_graph.h"
#include <any>
#include <optional>
#include <type_traits>
#include <utility>

namespace cudaq::qec {

struct edge_value_descriptor; // registry entry; decoding_task_edge_value.cpp

/// Type-erased value on a task-graph edge; always holds a registered type.
///
/// std::any::operator= constructs a new heap object on every assignment, even
/// when both sides already hold the same T, so operator= assigns into the
/// existing payload whenever the held types match.  A value is empty exactly
/// when desc_ is null; moving from a value leaves it empty.
class __attribute__((visibility("default"))) edge_value {
public:
  /// Holds nothing; type() is nullopt and get() throws.
  edge_value() = default;
  edge_value(const edge_value &) = default;
  edge_value(edge_value &&other) noexcept
      : desc_(std::exchange(other.desc_, nullptr)),
        value_(std::move(other.value_)) {
    other.value_.reset();
  }

  /// A value of the registered type T.  Throws std::invalid_argument when T
  /// is unregistered.
  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, edge_value>>>
  edge_value(T &&v)
      : desc_(descriptor_for(edge_value_type::of<std::decay_t<T>>())),
        value_(std::forward<T>(v)) {}

  edge_value &operator=(const edge_value &other);
  edge_value &operator=(edge_value &&other);

  /// Refills the payload in place when it already holds a T, otherwise
  /// replaces it with a value of the registered type T.
  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, edge_value>>>
  edge_value &operator=(T &&v) {
    using U = std::decay_t<T>;
    if (auto *held = std::any_cast<U>(&value_)) {
      *held = std::forward<T>(v);
      return *this;
    }
    desc_ = descriptor_for(edge_value_type::of<U>());
    value_ = std::forward<T>(v);
    return *this;
  }

  /// Id of the held type, or nullopt when empty.
  std::optional<edge_value_type> type() const;

  template <typename T>
  const T *get_if() const noexcept {
    return std::any_cast<T>(&value_);
  }
  template <typename T>
  T *get_if() noexcept {
    return std::any_cast<T>(&value_);
  }

  /// The held T.  Throws std::invalid_argument when the value is empty or
  /// holds something else.  The non-const overload is the in-place refill
  /// path described above.
  template <typename T>
  const T &get() const {
    if (const auto *held = std::any_cast<T>(&value_))
      return *held;
    throw_type_mismatch();
  }
  template <typename T>
  T &get() {
    if (auto *held = std::any_cast<T>(&value_))
      return *held;
    throw_type_mismatch();
  }

private:
  static const edge_value_descriptor *descriptor_for(edge_value_type type);
  [[noreturn]] void throw_type_mismatch() const;

  const edge_value_descriptor *desc_ = nullptr;
  std::any value_;
};

} // namespace cudaq::qec
