/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The built-in selection strategies and the name registry an ensemble resolves
// `join_strategy` against. Each strategy's parameter schema is registered in
// the decoder-schema registry under `selection_strategy_schema_name()`, which
// is what lets the fan-out declare `strategy_params` as a section discriminated
// by `strategy` with that prefix.

#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/selection_strategy.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace cudaq::qec {

namespace {

struct registration {
  selection_strategy_factory factory;
  std::uint64_t id;
};
std::map<std::string, registration> &registry() {
  static auto *r = new std::map<std::string, registration>();
  return *r;
}
std::mutex &registry_mutex() {
  static auto *m = new std::mutex();
  return *m;
}

/// Commits to the first converged candidate and ignores the rest.
class first_strategy : public selection_strategy {
public:
  void begin(cancellation_source, candidate_sink) override { done_.clear(); }
  std::optional<logical_outcome> offer(const candidate &c) override {
    if (!c.result.converged || done_.test_and_set())
      return std::nullopt;
    return c.result;
  }

private:
  std::atomic_flag done_ = ATOMIC_FLAG_INIT;
};

/// One thread serving every armed deadline in the process: at each due time
/// it runs the given action, under its lock, so `disarm()` returns only once
/// a running action is over. Shared by all `deadline` strategy instances and
/// started when the first one is built, so no run pays for a thread.
class deadline_timer {
public:
  using clock = std::chrono::steady_clock;
  using handle = std::uint64_t;

  ~deadline_timer() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable())
      thread_.join();
  }

  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_.joinable())
      thread_ = std::thread([this] { serve(); });
  }

  handle arm(clock::time_point at, std::function<void()> action) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto id = next_++;
    due_.emplace(at, entry{id, std::move(action)});
    cv_.notify_all();
    return id;
  }

  void disarm(handle id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = due_.begin(); it != due_.end(); ++it)
      if (it->second.id == id) {
        due_.erase(it);
        return;
      }
  }

private:
  struct entry {
    handle id;
    std::function<void()> action;
  };

  void serve() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (due_.empty()) {
        cv_.wait(lock);
        continue;
      }
      const auto at = due_.begin()->first;
      if (cv_.wait_until(lock, at) == std::cv_status::timeout ||
          clock::now() >= at) {
        while (!due_.empty() && due_.begin()->first <= clock::now()) {
          auto e = due_.begin()->second;
          due_.erase(due_.begin());
          e.action();
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::multimap<clock::time_point, entry> due_;
  handle next_ = 1;
  bool stop_ = false;
  std::thread thread_;
};

/// Decides at `deadline_us` after `begin()`, from the timer, or at `finish()`
/// if every member is done sooner: the cheapest costed candidate the sink
/// holds, falling back to the first one seen when none carries a cost. With
/// `soft_stop_before_us`, asks the members to wind down that long before the
/// deadline. Once decided it looks at no further candidate, so a late push
/// enters the sink but not the choice.
class deadline_strategy : public selection_strategy {
public:
  deadline_strategy(std::shared_ptr<deadline_timer> timer,
                    std::chrono::microseconds deadline,
                    std::chrono::microseconds soft_before)
      : timer_(std::move(timer)), deadline_(deadline),
        soft_before_(soft_before) {
    timer_->start();
  }
  ~deadline_strategy() override { disarm(); }

  void begin(cancellation_source cancel, candidate_sink sink) override {
    disarm();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sink_ = std::move(sink);
      first_.reset();
      decided_ = false;
      final_.reset();
    }
    const auto hard_at = deadline_timer::clock::now() + deadline_;
    if (soft_before_.count() != 0)
      soft_ = timer_->arm(hard_at - soft_before_,
                          [cancel]() mutable { cancel.request_soft_stop(); });
    hard_ = timer_->arm(hard_at, [this, cancel]() mutable {
      cancel.request_hard_stop();
      decide();
    });
  }
  std::optional<logical_outcome> offer(const candidate &c) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!decided_ && !first_)
      first_ = c.result;
    return std::nullopt;
  }
  std::optional<logical_outcome> finish() override {
    disarm();
    std::lock_guard<std::mutex> lock(mutex_);
    return decided_ ? final_ : choice();
  }

private:
  /// The choice as of now; under `mutex_`.
  std::optional<logical_outcome> choice() const {
    if (const auto best = sink_.best_so_far())
      return best->result;
    return first_;
  }
  /// Fix the choice at the deadline, on the timer thread.
  void decide() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!decided_) {
      decided_ = true;
      final_ = choice();
    }
  }
  void disarm() {
    if (hard_)
      timer_->disarm(*std::exchange(hard_, std::nullopt));
    if (soft_)
      timer_->disarm(*std::exchange(soft_, std::nullopt));
  }

  std::shared_ptr<deadline_timer> timer_;
  std::chrono::microseconds deadline_, soft_before_;
  std::optional<deadline_timer::handle> soft_, hard_;
  std::mutex mutex_;
  candidate_sink sink_;
  std::optional<logical_outcome> first_, final_;
  bool decided_ = false;
};

const auto registered = [] {
  using decoding::config::param_kind;
  std::vector<selection_strategy_registration> held;
  held.push_back(register_selection_strategy(
      "first", {}, [](const cudaqx::heterogeneous_map &) {
        return std::make_unique<first_strategy>();
      }));

  decoding::config::decoder_schema deadline;
  deadline.params = {{"deadline_us", param_kind::uint64, /*required=*/true},
                     {"soft_stop_before_us", param_kind::uint64}};
  deadline.validate = [](const cudaqx::heterogeneous_map &p) {
    if (p.get<std::size_t>("soft_stop_before_us", 0) >
        p.get<std::size_t>("deadline_us"))
      throw std::runtime_error(
          "deadline: soft_stop_before_us must not exceed deadline_us");
  };
  // Intentionally leaked with the registry: its thread must outlive any
  // strategy instance that armed it.
  auto *timer = new std::shared_ptr<deadline_timer>(new deadline_timer());
  held.push_back(register_selection_strategy(
      "deadline", std::move(deadline),
      [timer](const cudaqx::heterogeneous_map &p) {
        return std::make_unique<deadline_strategy>(
            *timer,
            std::chrono::microseconds(p.get<std::size_t>("deadline_us")),
            std::chrono::microseconds(
                p.get<std::size_t>("soft_stop_before_us", 0)));
      }));
  return held;
}();

} // namespace

selection_strategy_registration
register_selection_strategy(std::string name,
                            decoding::config::decoder_schema params,
                            selection_strategy_factory factory) {
  params.name = selection_strategy_schema_name(name);
  decoding::config::register_decoder_schema(std::move(params));
  static std::uint64_t next_id = 0;
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto id = ++next_id;
  registry()[name] = {std::move(factory), id};
  return {std::move(name), id};
}

selection_strategy_registration::~selection_strategy_registration() {
  if (name_.empty())
    return;
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto it = registry().find(name_);
  if (it != registry().end() && it->second.id == id_)
    registry().erase(it);
}

std::unique_ptr<selection_strategy>
make_selection_strategy(const std::string &name,
                        const cudaqx::heterogeneous_map &params) {
  selection_strategy_factory factory;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = registry().find(name);
    if (it == registry().end())
      throw std::invalid_argument("selection strategy '" + name +
                                  "' is not registered");
    factory = it->second.factory;
  }
  return factory(params);
}

std::vector<std::string> registered_selection_strategies() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  std::vector<std::string> names;
  for (const auto &[name, entry] : registry())
    names.push_back(name);
  return names;
}

} // namespace cudaq::qec
