/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "logger_forwarder.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

namespace {

struct gate {
  std::mutex mu;
  std::condition_variable cv;
  bool arrived = false;
  bool release = false;
  void wait_arrived() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return arrived; });
  }
  void arrive_and_wait() {
    {
      std::lock_guard<std::mutex> lk(mu);
      arrived = true;
    }
    cv.notify_all();
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return release; });
  }
  void go() {
    {
      std::lock_guard<std::mutex> lk(mu);
      release = true;
    }
    cv.notify_all();
  }
};

gate *g_guard_gate = nullptr;
gate *g_enq_pos_gate = nullptr;
gate *g_cas_gate = nullptr;
gate *g_deq_gate = nullptr;
std::atomic<int> g_enq_pos_hits{0};
std::atomic<int> g_cas_hits{0};
std::atomic<int> g_deq_hits{0};

cudaq::qec::detail::forwarder_internal::queued_log_record make_rec(char c) {
  cudaq::qec::detail::forwarder_internal::queued_log_record r;
  r.message_len = 1;
  r.message[0] = c;
  return r;
}

} // namespace

extern "C" {

void qec_cc_logger_hook_after_producer_guard() {
  if (g_guard_gate)
    g_guard_gate->arrive_and_wait();
}

void qec_cc_logger_hook_after_load_enqueue_pos() {
  if (!g_enq_pos_gate)
    return;
  if (g_enq_pos_hits.fetch_add(1) == 0)
    g_enq_pos_gate->arrive_and_wait();
}

void qec_cc_logger_hook_after_enqueue_cas_ready() {
  if (!g_cas_gate)
    return;
  if (g_cas_hits.fetch_add(1) == 0)
    g_cas_gate->arrive_and_wait();
}

void qec_cc_logger_hook_after_load_dequeue_pos() {
  if (!g_deq_gate)
    return;
  // Skip the worker's startup empty-dequeue; pause the next load.
  if (g_deq_hits.fetch_add(1) == 1)
    g_deq_gate->arrive_and_wait();
}

} // extern "C"

namespace {

// clear() waits for in-flight producers; the paused producer then sees the
// second disabled return after the guard.
TEST(LoggerForwarderRaces, ClearWaitsForActiveProducer) {
  using namespace cudaq::qec::detail;
  gate g;
  g_guard_gate = &g;
  forwarder_internal::set(forwarder_config{
      .callback = [](forwarded_log_record &&) {}, .queue_capacity = 4});
  std::thread producer([] { forwarder_internal::enqueue(make_rec('a')); });
  g.wait_arrived();
  std::thread clearer([] { forwarder_internal::clear(); });
  g.go();
  producer.join();
  clearer.join();
  g_guard_gate = nullptr;
  EXPECT_FALSE(forwarder_internal::is_enabled());
}

// Producer A reloads enqueue_pos after B advances it (positive sequence
// difference).
TEST(LoggerForwarderRaces, EnqueueReloadsAfterPeerAdvance) {
  using namespace cudaq::qec::detail;
  gate g;
  g_enq_pos_hits.store(0);
  g_enq_pos_gate = &g;
  std::atomic<int> n{0};
  forwarder_internal::set(forwarder_config{
      .callback = [&](forwarded_log_record &&) { n.fetch_add(1); },
      .queue_capacity = 8});
  std::thread a([] { forwarder_internal::enqueue(make_rec('a')); });
  g.wait_arrived();
  forwarder_internal::enqueue(make_rec('b'));
  g.go();
  a.join();
  forwarder_internal::flush();
  forwarder_internal::clear();
  g_enq_pos_gate = nullptr;
  EXPECT_EQ(n.load(), 2);
}

// Producer A retries CAS after B claims the same slot.
TEST(LoggerForwarderRaces, EnqueueCasRetry) {
  using namespace cudaq::qec::detail;
  gate g;
  g_cas_hits.store(0);
  g_cas_gate = &g;
  std::atomic<int> n{0};
  forwarder_internal::set(forwarder_config{
      .callback = [&](forwarded_log_record &&) { n.fetch_add(1); },
      .queue_capacity = 8});
  std::thread a([] { forwarder_internal::enqueue(make_rec('a')); });
  g.wait_arrived();
  forwarder_internal::enqueue(make_rec('b'));
  g.go();
  a.join();
  forwarder_internal::flush();
  forwarder_internal::clear();
  g_cas_gate = nullptr;
  EXPECT_GE(g_cas_hits.load(), 2);
  EXPECT_EQ(n.load(), 2);
}

// Worker reloads dequeue_pos after drop_oldest evicts the slot it had loaded.
TEST(LoggerForwarderRaces, WorkerReloadsAfterDropOldest) {
  using namespace cudaq::qec::detail;
  gate g;
  g_deq_hits.store(0);
  g_deq_gate = &g;
  std::atomic<int> n{0};
  forwarder_internal::set(forwarder_config{
      .callback = [&](forwarded_log_record &&) { n.fetch_add(1); },
      .queue_capacity = 2,
      .drop_policy = forward_drop_policy::drop_oldest});
  forwarder_internal::enqueue(make_rec('a'));
  g.wait_arrived();
  forwarder_internal::enqueue(make_rec('b'));
  forwarder_internal::enqueue(make_rec('c'));
  g.go();
  forwarder_internal::flush();
  forwarder_internal::clear();
  g_deq_gate = nullptr;
  EXPECT_GE(g_deq_hits.load(), 2);
  EXPECT_GE(n.load(), 1);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
