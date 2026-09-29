/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file test_session_fakes.h
/// @brief `blocking_session`: test-double base for a backend whose answer is
/// already at hand -- implement `send_sync` and get a worker thread that
/// reports each reply to the run() collector for free. Lives in the test
/// tree on purpose: a real backend's reply genuinely arrives later, so it
/// must implement send()/event_done() itself to avoid blocking the timing
/// thread -- no shipping backend is built on this.

#include "session.h"

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::qec::playback {

class blocking_session : public session {
public:
  virtual RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                              std::size_t &reply_len) = 0;

  void start(run_ctx &collector) override {
    collector_ = &collector;
    worker_ = std::thread([this] { drain(); });
  }

  void send(const frame &f, tag t) override {
    if (f.size < sizeof(cudaq::realtime::RPCHeader))
      throw std::invalid_argument(
          "blocking_session: frame is smaller than RPCHeader");
    entry e;
    e.kind = entry::kRequest;
    e.t = t;
    e.bytes.assign(f.bytes, f.bytes + f.size);
    push(std::move(e));
  }

  void event_done(std::uint32_t event, std::uint32_t issued, std::int32_t term,
                  bool has_term) override {
    entry e;
    e.kind = entry::kEventDone;
    e.event = event;
    e.issued = issued;
    e.term = term;
    e.has_term = has_term;
    push(std::move(e));
  }

  void stop(std::chrono::nanoseconds) override {
    if (!worker_.joinable())
      return;
    entry e;
    e.kind = entry::kStop;
    push(std::move(e));
    worker_.join();
  }

protected:
  /// Runs on the worker thread, after send_sync() answers a request and
  /// before that reply is delivered -- the hook a fake that wants a delayed
  /// or reordered reply overrides, since sleeping here never blocks send().
  virtual void before_reply() {}

private:
  /// Nothing here knows which RPC a frame holds (that is the point of the
  /// session interface), so a reply lands in a buffer big enough for any of
  /// them and is trimmed to what send_sync() actually wrote.
  static constexpr std::size_t kHeldReplyBytes = 4096;

  struct entry {
    enum kind_t { kRequest, kEventDone, kStop } kind = kStop;
    tag t{};
    std::vector<std::uint8_t> bytes;
    std::uint32_t event = 0, issued = 0;
    std::int32_t term = 0;
    bool has_term = false;
  };

  static std::uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
  }

  void push(entry e) {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push_back(std::move(e));
    cv_.notify_one();
  }

  void drain() {
    for (;;) {
      entry e;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        e = std::move(queue_.front());
        queue_.pop_front();
      }
      switch (e.kind) {
      case entry::kRequest: {
        std::vector<std::uint8_t> reply(kHeldReplyBytes);
        std::size_t len = 0;
        const auto status =
            send_sync({e.bytes.data(), e.bytes.size()}, reply, len);
        before_reply();
        handle_reply(*collector_, e.t, status, reply.data(), len, now_ns());
        break;
      }
      case entry::kEventDone:
        handle_event_done(*collector_, e.event, e.issued, e.term, e.has_term);
        break;
      case entry::kStop:
        return;
      }
    }
  }

  run_ctx *collector_ = nullptr;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<entry> queue_;
  std::thread worker_;
};

} // namespace cudaq::qec::playback
