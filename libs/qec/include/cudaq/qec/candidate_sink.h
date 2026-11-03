/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file candidate_sink.h
/// @brief `candidate_sink`: a multi-producer queue of candidate logical
/// outcomes. Producers push costed candidates and may read the cheapest so
/// far.

#include "cudaq/qec/decoding_task_types.h"
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

/// A candidate answer, the id of the node that produced it and, when the
/// producer can rank its own answer, a cost (lower is better).
struct candidate {
  logical_outcome result;
  std::string producer;
  std::optional<double> cost;
};

/// @brief Shared candidate queue. Copies share one storage.
///
/// `push()` is thread-safe. It allocates a node, takes a mutex when the
/// candidate may be the cheapest so far, and then runs the observer on the
/// pushing thread.
class candidate_sink {
public:
  using observer = std::function<void(const candidate &)>;

  candidate_sink() : s_(std::make_shared<state>()) {}

  // --- producers -----------------------------------------------------------

  /// @brief Queue a candidate and show it to the round's observer. A costed
  /// candidate becomes the best so far only when strictly cheaper, so ties
  /// keep the earlier one; uncosted candidates never do.
  void push(logical_outcome solution, std::string producer_id = {},
            std::optional<double> cost = std::nullopt) const {
    auto *n = new node{{std::move(solution), std::move(producer_id), cost},
                       s_->head.load(std::memory_order_relaxed)};
    while (!s_->head.compare_exchange_weak(
        n->next, n, std::memory_order_release, std::memory_order_relaxed)) {
    }
    if (cost && *cost < s_->best_cost.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(s_->best_mutex);
      if (!s_->best || *cost < *s_->best->cost) {
        s_->best = n->c;
        s_->best_cost.store(*cost, std::memory_order_release);
      }
    }
    if (s_->on_push)
      s_->on_push(n->c);
  }

  /// @brief The cheapest costed candidate pushed this round, or `std::nullopt`.
  std::optional<candidate> best_so_far() const {
    std::lock_guard<std::mutex> lock(s_->best_mutex);
    return s_->best;
  }

  // --- round owner ---------------------------------------------------------

  /// @brief Start a round: drop queued candidates and the best so far, and
  /// install the observer every push of the round is shown to.
  void reset(observer on_push = {}) const {
    free(s_->head.exchange(nullptr, std::memory_order_acq_rel));
    {
      std::lock_guard<std::mutex> lock(s_->best_mutex);
      s_->best.reset();
      s_->best_cost.store(std::numeric_limits<double>::infinity(),
                          std::memory_order_release);
    }
    s_->on_push = std::move(on_push);
  }

  /// @brief Take every queued candidate, in push order.
  std::vector<candidate> drain() const {
    std::vector<candidate> out;
    auto *n = s_->head.exchange(nullptr, std::memory_order_acq_rel);
    for (auto *p = n; p; p = p->next)
      out.push_back(p->c);
    free(n);
    return {out.rbegin(), out.rend()};
  }

private:
  struct node {
    candidate c;
    node *next;
  };
  struct state {
    std::atomic<node *> head{nullptr};
    std::atomic<double> best_cost{std::numeric_limits<double>::infinity()};
    std::mutex best_mutex;
    std::optional<candidate> best;
    observer on_push;
    ~state() { candidate_sink::free(head.load()); }
  };
  static void free(node *n) {
    while (n)
      delete std::exchange(n, n->next);
  }

  std::shared_ptr<state> s_;
};

} // namespace cudaq::qec
