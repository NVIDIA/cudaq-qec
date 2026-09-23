/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// Tests the session backends: session.h's ordering contract (submission
/// order is delivery order; a waiting caller holds back nobody), the `null`
/// jitter floor, and the UDP backend (timeouts, max_frame_bytes
/// trustworthiness, plan()'s oversized-enqueue rejection).

#include "emulator.h"
#include "session.h"
#include "syndrome_source.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace cudaq::qec::playback;

// ─── SessionOrdering ────────────────────────────────────────────────────────
//
// Without both halves of the ordering contract a schedule can't place a read
// relative to its syndromes without stalling for a whole decode, and a late
// read is worse than useless (DecodingSession drops an unread result once
// the next round lands). The recording server here only tests the client's
// transport and dispatch order, not decoder semantics.

namespace {

using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

constexpr std::chrono::milliseconds kDecodeTime{120};

/// A loopback UDP socket that logs the function_id of every frame it
/// receives, in arrival order, and answers reads after `decode_time` --
/// long enough that a client which waits for one instead of moving on shows
/// up as a reordering, not as a scheduling wobble.
class recording_server {
public:
  explicit recording_server(std::chrono::milliseconds decode_time = kDecodeTime)
      : decode_time_(decode_time) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr *>(&addr), &len);
    endpoint_ = "127.0.0.1:" + std::to_string(ntohs(addr.sin_port));

    timeval tv{};
    tv.tv_usec = 50'000;
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    thread_ = std::thread([this] { serve(); });
  }

  ~recording_server() {
    stop_.store(true);
    if (thread_.joinable())
      thread_.join();
    ::close(fd_);
  }

  const std::string &endpoint() const { return endpoint_; }

  std::vector<std::uint32_t> arrivals() const {
    std::lock_guard<std::mutex> lock(mu_);
    return arrivals_;
  }

private:
  void serve() {
    std::vector<std::uint8_t> buf(65536);
    while (!stop_.load()) {
      sockaddr_in from{};
      socklen_t from_len = sizeof(from);
      const ssize_t n =
          ::recvfrom(fd_, buf.data(), buf.size(), 0,
                     reinterpret_cast<sockaddr *>(&from), &from_len);
      if (n < static_cast<ssize_t>(sizeof(RPCHeader)))
        continue;
      RPCHeader hdr{};
      std::memcpy(&hdr, buf.data(), sizeof(hdr));
      {
        std::lock_guard<std::mutex> lock(mu_);
        arrivals_.push_back(hdr.function_id);
      }
      // Answering inline is what makes this a faithful stand-in: the real
      // server runs one dispatcher thread per ring, so a slow decode delays
      // everything behind it on that ring and nothing else. Only
      // get_corrections is deliberately slow; enqueue acks immediately, same
      // as the real server's ingestion-only ack.
      const bool is_read = hdr.function_id == kGetCorrectionsFunctionId;
      if (is_read)
        std::this_thread::sleep_for(decode_time_);
      std::vector<std::uint8_t> out(sizeof(RPCResponse) + (is_read ? 1 : 0), 0);
      RPCResponse resp{};
      resp.magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
      resp.status = 0;
      resp.result_len = is_read ? 1 : 0;
      resp.request_id = hdr.request_id;
      std::memcpy(out.data(), &resp, sizeof(resp));
      ::sendto(fd_, out.data(), out.size(), 0,
               reinterpret_cast<sockaddr *>(&from), from_len);
    }
  }

  int fd_ = -1;
  std::string endpoint_;
  std::chrono::milliseconds decode_time_;
  mutable std::mutex mu_;
  std::vector<std::uint32_t> arrivals_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

} // namespace

TEST(SessionOrdering, AWaitingCallerDoesNotHoldBackTheNextSend) {
  // If send() collected the reply itself instead of just publishing the
  // frame, syndromes meant to follow this read would go out while it was
  // still in flight, and the decoder would see them in the wrong order.
  recording_server server;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, server.endpoint()}},
      /*timeout_ms=*/5000);
  std::unordered_map<std::uint64_t, session *> router{
      {0, sessions[0].second.get()}};

  auto sched = parse("0 get_corrections return_size=1\n"
                     "+0 enqueue source=0b1\n",
                     {0}, 1000);
  auto result = run(plan(sched, router, {}, {}));

  ASSERT_EQ(result.records.size(), 2u);
  const auto &read = result.records[0];
  const auto &enqueue = result.records[1];
  EXPECT_LT(enqueue.call_ns - read.call_ns,
            static_cast<std::uint64_t>(kDecodeTime.count()) * 1'000'000 / 2)
      << "dispatch waited for the read's reply instead of moving straight to "
         "the next event";
  EXPECT_GE(read.return_ns - read.call_ns,
            static_cast<std::uint64_t>(kDecodeTime.count()) * 1'000'000)
      << "the record should still span the whole read";
  EXPECT_EQ(read.status, static_cast<std::int32_t>(RpcStatus::OK));

  EXPECT_EQ(server.arrivals(),
            (std::vector<std::uint32_t>{kGetCorrectionsFunctionId,
                                        kEnqueueSyndromesFunctionId}));
}

TEST(SessionOrdering, ConcurrentSubmissionsEachGetTheirOwnReply) {
  // Two outstanding reads at once must both work: matching replies by
  // request_id, not by arrival order, is what makes that safe. Both dispatch
  // well before either's reply lands (a 120ms decode vs. an immediate
  // `+0`), so both are genuinely in flight together.
  recording_server server;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, server.endpoint()}},
      /*timeout_ms=*/5000);
  std::unordered_map<std::uint64_t, session *> router{
      {0, sessions[0].second.get()}};

  auto sched = parse("0 get_corrections return_size=1\n"
                     "+0 get_corrections return_size=1\n",
                     {0}, 1000);
  auto result = run(plan(sched, router, {}, {}));

  ASSERT_EQ(result.records.size(), 2u);
  EXPECT_EQ(result.records[0].status, static_cast<std::int32_t>(RpcStatus::OK));
  EXPECT_EQ(result.records[1].status, static_cast<std::int32_t>(RpcStatus::OK));
  EXPECT_EQ(server.arrivals().size(), 2u);
}

TEST(SessionOrdering, AnUnblockingReadKeepsItsPlaceInTheSyndromeStream) {
  // Schedule level: a `signal=` read sits on the timeline that owns the
  // syndromes, between the rounds it must separate, and hands its answer
  // back through a signal. What the decoder sees must be enqueue, read,
  // enqueue -- and the timeline must not have stopped for the decode.
  recording_server server;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, server.endpoint()}},
      /*timeout_ms=*/5000);
  std::unordered_map<std::uint64_t, session *> router{
      {0, sessions[0].second.get()}};
  static_source src(std::vector<std::vector<std::uint8_t>>(8, {1}));

  auto sched = parse("0 stream source=0 rounds=1\n"
                     "+0 get_corrections return_size=1 signal=shot\n"
                     "+0 stream source=0 rounds=1\n",
                     {0}, 1000);
  auto result = run(plan(sched, router, {{0, &src}}, {}));

  EXPECT_EQ(server.arrivals(),
            (std::vector<std::uint32_t>{kEnqueueSyndromesFunctionId,
                                        kGetCorrectionsFunctionId,
                                        kEnqueueSyndromesFunctionId}));

  const auto &read = result.records[1];
  const auto &next_round = result.records[2];
  EXPECT_LT(next_round.call_ns - read.call_ns,
            static_cast<std::uint64_t>(kDecodeTime.count()) * 1'000'000 / 2)
      << "the timeline stopped to wait for the answer instead of moving on";
  EXPECT_TRUE(read.read_completed); // filled in by the session's own worker
  EXPECT_EQ(read.correction_count, 1u);
  EXPECT_GE(read.return_ns - read.call_ns,
            static_cast<std::uint64_t>(kDecodeTime.count()) * 1'000'000)
      << "the record should still span the whole read";
}

TEST(SessionOrdering, ParserAndPlanRejectAnUncollectableRead) {
  EXPECT_THROW(parse("0 get_corrections return_size=1 signal=\n", {0}, 1000),
               std::invalid_argument); // `signal=` with nothing to raise
  EXPECT_THROW(
      parse("0 get_corrections return_size=1 signal=a signal=b\n", {0}, 1000),
      std::invalid_argument); // one read, one answer, one signal
  EXPECT_NO_THROW(
      parse("0 get_corrections return_size=1 signal=done\n", {0}, 1000));

  // A signal an unblocking read raises counts as raised for the schedule's
  // wait graph: a later `until=` has something that does eventually bring
  // it up.
  auto null = make_null_session();
  std::unordered_map<std::uint64_t, session *> router{{0, null.get()}};
  static_source src(std::vector<std::vector<std::uint8_t>>(4, {1}));
  EXPECT_NO_THROW(plan(parse("0 get_corrections return_size=1 signal=done\n"
                             "+0 stream source=0 until=done\n",
                             {0}, 1000),
                       router, {{0, &src}}, {}));
}

// ─── NullBackendAdvanced ────────────────────────────────────────────────────
//
// Adversarial tests for the `null` backend (jitter floor): max_frame_bytes=0
// (unbounded), discards every frame via an atomic checksum. Probes the
// edges -- empty/huge frames, reply-buffer bounds, repeated calls -- since a
// backend whose job is to cost nothing is the likeliest to skip a bounds check.

TEST(NullBackendAdvanced,
     FramesOfEverySizeAreDiscardedWithoutCrashingOrReplying) {
  // 0 bits and the odd widths either side of a power of two are the edges
  // that would show up first if the per-byte checksum loop, or the reply's
  // bit-packing, ran off a partial trailing byte. Driven through
  // get_corrections' return_size, since that is what actually varies the
  // request/reply size null_session sees on a real run.
  for (std::uint32_t bits : {0u, 1u, 3u, 7u, 63u, 65u, 4095u, 4097u, 8192u}) {
    SCOPED_TRACE(bits);
    auto s = make_null_session();
    std::unordered_map<std::uint64_t, session *> router{{0, s.get()}};
    auto sched =
        parse("0 get_corrections return_size=" + std::to_string(bits) + "\n",
              {0}, 1000);
    run_result result;
    EXPECT_NO_THROW(result = run(plan(sched, router, {}, {})));

    ASSERT_EQ(result.records.size(), 1u);
    EXPECT_EQ(result.records[0].status,
              static_cast<std::int32_t>(RpcStatus::OK))
        << "null always answers OK";
    EXPECT_TRUE(result.records[0].read_completed);
    EXPECT_EQ(result.records[0].correction_count, bits);
  }
}

TEST(NullBackendAdvanced, MaxFrameBytesIsUnboundedAndStaysThatWayUnderLoad) {
  // 0 = unbounded, per session.h's comment. A long run of varying-width
  // enqueues must not disturb it: nothing mutable is shared between
  // requests except the internal checksum accumulator.
  auto s = make_null_session();
  EXPECT_EQ(s->max_frame_bytes, 0u);
  std::unordered_map<std::uint64_t, session *> router{{0, s.get()}};
  std::string text;
  for (int i = 0; i < 500; ++i) {
    const auto width = static_cast<std::size_t>(1 + (i % 37));
    text += std::to_string(i) + " enqueue source=0b" + std::string(width, '1') +
            "\n";
  }
  auto sched = parse(text, {0}, 1000);
  auto result = run(plan(sched, router, {}, {}));

  ASSERT_EQ(result.records.size(), 500u);
  for (const auto &r : result.records)
    EXPECT_EQ(r.status, static_cast<std::int32_t>(stream_terminate::OK));
  EXPECT_EQ(s->max_frame_bytes, 0u);
}

// ─── UdpBackendAdvanced ─────────────────────────────────────────────────────
//
// Adversarial tests for the UDP backend ("Replies are matched by
// request_id, never by arrival order... This is not optional"): a
// genuinely-closed-port timeout, max_frame_bytes trustworthiness, and
// plan()'s oversized-enqueue rejection trusting that number.

namespace {

/// Binds and immediately closes a loopback UDP socket to obtain a port
/// number that is (at the moment of the check) genuinely unbound -- more
/// reliable than guessing an arbitrary high port.
std::string closed_loopback_endpoint() {
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len);
  auto port = ntohs(addr.sin_port);
  ::close(fd);
  return "127.0.0.1:" + std::to_string(port);
}

} // namespace

TEST(UdpBackendAdvanced,
     AwaitOnAGenuinelyClosedPortTimesOutBoundedAndReturnsAnError) {
  auto endpoint = closed_loopback_endpoint();
  constexpr std::uint32_t kTimeoutMs = 80;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}},
      kTimeoutMs);
  ASSERT_EQ(sessions.size(), 1u);
  std::unordered_map<std::uint64_t, session *> router{
      {0, sessions[0].second.get()}};

  auto sched = parse("0 reset\n", {0}, 1000);
  auto t0 = std::chrono::steady_clock::now();
  auto result = run(plan(sched, router, {}, {}));
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

  // session.cpp: a socket that has stayed silent for longer than the
  // timeout is synthesized as RpcStatus::INTERNAL_ERROR -- there is no
  // dedicated "local timeout" status. Must NOT be OK, and must return in
  // bounded time (well under the 1s default drain, so it is genuinely the
  // per-request sweep that resolved this, not stop()'s fallback), not hang.
  ASSERT_EQ(result.records.size(), 1u);
  EXPECT_EQ(result.records[0].status,
            static_cast<std::int32_t>(RpcStatus::INTERNAL_ERROR));
  EXPECT_LT(elapsed_ms, static_cast<long long>(kTimeoutMs) + 500);
}

TEST(UdpBackendAdvanced,
     TheReceiverThreadSurvivesATransientErrorAndKeepsListening) {
  auto endpoint = closed_loopback_endpoint();
  constexpr std::uint32_t kTimeoutMs = 150;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}},
      kTimeoutMs);
  ASSERT_EQ(sessions.size(), 1u);
  std::unordered_map<std::uint64_t, session *> router{
      {0, sessions[0].second.get()}};

  const auto port = endpoint.substr(endpoint.rfind(':') + 1);
  // Binds the real listener well after the first reset has already timed
  // out against the closed port, so the one receiver thread the session
  // starts for this whole run has to shake off that transient error and
  // still answer the second reset -- not get restarted in between.
  std::thread late_server([port] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<std::uint16_t>(std::stoi(port)));
    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
      ::close(fd);
      return;
    }
    std::vector<std::uint8_t> buf(256);
    sockaddr_in from{};
    socklen_t from_len = sizeof(from);
    const ssize_t n =
        ::recvfrom(fd, buf.data(), buf.size(), 0,
                   reinterpret_cast<sockaddr *>(&from), &from_len);
    if (n >= static_cast<ssize_t>(sizeof(RPCHeader))) {
      RPCHeader hdr{};
      std::memcpy(&hdr, buf.data(), sizeof(hdr));
      RPCResponse resp{};
      resp.magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
      resp.status = 0;
      resp.result_len = 0;
      resp.request_id = hdr.request_id;
      ::sendto(fd, &resp, sizeof(resp), 0, reinterpret_cast<sockaddr *>(&from),
               from_len);
    }
    ::close(fd);
  });

  // event 0 times out (~150ms) against the closed port; event 1, 500ms
  // later, dispatches well after late_server has bound and is listening.
  auto sched = parse("0 reset\n"
                     "500 reset\n",
                     {0}, 1'000'000);
  auto result = run(plan(sched, router, {}, {}));
  late_server.join();

  ASSERT_EQ(result.records.size(), 2u);
  EXPECT_EQ(result.records[0].status,
            static_cast<std::int32_t>(RpcStatus::INTERNAL_ERROR));
  EXPECT_EQ(result.records[1].status, static_cast<std::int32_t>(RpcStatus::OK));
}

TEST(UdpBackendAdvanced,
     MaxFrameBytesIsTheMaxUdpDatagramPayloadNotZeroOrUnbounded) {
  // No listener needed: connect() on a UDP socket just records the default
  // destination, it doesn't require a live peer.
  auto endpoint = closed_loopback_endpoint();
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}}, 50);
  ASSERT_EQ(sessions.size(), 1u);

  // session_udp.cpp: kMaxDatagram = 65535 - 8 (UDP header) - 20 (IPv4
  // header) = 65507.
  EXPECT_EQ(sessions[0].second->max_frame_bytes, 65507u);
  EXPECT_NE(sessions[0].second->max_frame_bytes, 0u)
      << "0 would mean unbounded, which UDP "
         "is not";
}

TEST(UdpBackendAdvanced,
     PlanTimeCheckTrustsMaxFrameBytesAndRejectsAnOversizedEnqueue) {
  // Really about proving max_frame_bytes is trustworthy, since plan() trusts
  // it wholesale to keep oversized frames off the wire. Build an enqueue
  // frame (24B header + 32B payload + packed bits) exceeding kMaxDatagram
  // (65507) by construction: n > 523608 bits.
  auto endpoint = closed_loopback_endpoint();
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}}, 50);
  ASSERT_EQ(sessions.size(), 1u);
  ASSERT_EQ(sessions[0].second->max_frame_bytes, 65507u);

  constexpr std::size_t kBits = 523'616; // 56 + 523616/8 = 65508 > 65507
  std::string sched_text =
      "0 enqueue source=0b" + std::string(kBits, '1') + "\n";
  auto sched = parse(sched_text, {0}, 1000);
  std::unordered_map<std::uint64_t, session *> router;
  router[0] = sessions[0].second.get();
  EXPECT_THROW(plan(sched, router, {}), std::invalid_argument);
}

TEST(UdpBackendAdvanced, SubmitRejectsAFrameShorterThanAnRPCHeader) {
  auto endpoint = closed_loopback_endpoint();
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}}, 50);
  ASSERT_EQ(sessions.size(), 1u);
  auto &sess = sessions[0].second;

  std::vector<std::uint8_t> short_frame(sizeof(RPCHeader) - 1, 0);
  frame f{short_frame.data(), short_frame.size()};
  EXPECT_THROW(sess->send(f, {}), std::invalid_argument);
}
