/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file session.cpp
/// @brief All session backends: `null` (discard every frame), inproc
/// (dispatch to a decoder's own DecodingSession), UDP and CPU RoCE (a
/// decoding server). Concrete classes are anonymous-namespace private; the
/// factories in session.h are the only public surface.

#include "session.h"
#include "RpcSlot.h"
#include "SessionRegistry.h"

#ifdef CUDAQ_QEC_PLAYBACK_CPU_ROCE
#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#endif

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace cudaq::qec::playback {

namespace {

std::uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

} // namespace

// ─── dispatch_rpc / reply_capacity_for ─────────────────────────────────────
//
// Shared by the null and inproc backends: parses a raw request frame and
// (for inproc) runs it against a DecodingSession, or (for null) just says how
// big a reply it would have produced.

namespace {

using cudaq::qec::decoding_server::DecodingSession;
using cudaq::qec::decoding_server::SessionRegistry;

/// Result of dispatching one raw request frame -- every RPC gets a reply.
struct DispatchResult {
  cudaq::qec::decoding::rpc::RpcStatus status =
      cudaq::qec::decoding::rpc::RpcStatus::BAD_REQUEST;
  std::size_t reply_len = 0;
};

/// Parses \p bytes as one of the three decoder RPCs and calls the matching
/// core on \p dec, copying (for get_corrections) the still-bit-packed
/// result into \p reply (capacity \p reply_capacity).
DispatchResult dispatch_rpc(DecodingSession &dec, const std::uint8_t *bytes,
                            std::size_t size, std::uint8_t *reply,
                            std::size_t reply_capacity) {
  namespace slot = cudaq::qec::decoding_server::slot;
  using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
  using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
  using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
  using cudaq::qec::decoding::rpc::RpcStatus;
  using cudaq::realtime::RPCHeader;

  if (size < sizeof(RPCHeader))
    return {RpcStatus::BAD_REQUEST, 0};
  const auto *header = reinterpret_cast<const RPCHeader *>(bytes);

  if (header->function_id == kEnqueueSyndromesFunctionId) {
    // Mirrors DecodingSession::handle_enqueue: reject anything but the
    // identity syndrome mapping, and ack OK even on an internal decode
    // failure -- that surfaces at the next get_corrections instead.
    slot::EnqueueView view;
    if (!slot::parse_enqueue(bytes, size, view))
      return {RpcStatus::BAD_REQUEST, 0};
    if (view.syndrome_mapping_id != 0)
      return {RpcStatus::BAD_REQUEST, 0};
    dec.enqueue_core(view);
    return {RpcStatus::OK, 0};
  }

  if (header->function_id == kGetCorrectionsFunctionId) {
    slot::GetCorrectionsView view;
    if (!slot::parse_get_corrections(bytes, size, view))
      return {RpcStatus::BAD_REQUEST, 0};
    std::size_t reply_len = 0;
    auto status = dec.get_corrections_core(view.return_size, view.reset, reply,
                                           reply_capacity, reply_len);
    return {status, reply_len};
  }

  if (header->function_id == kResetDecoderFunctionId) {
    slot::ResetView view;
    if (!slot::parse_reset(bytes, size, view))
      return {RpcStatus::BAD_REQUEST, 0};
    return {dec.reset_core(), 0};
  }

  return {RpcStatus::BAD_REQUEST, 0};
}

/// How big a reply this frame can produce, so a submitted request can carry
/// its own result buffer.
std::size_t reply_capacity_for(const std::uint8_t *bytes, std::size_t size) {
  namespace slot = cudaq::qec::decoding_server::slot;
  using cudaq::realtime::RPCHeader;
  if (size < sizeof(RPCHeader) ||
      reinterpret_cast<const RPCHeader *>(bytes)->function_id !=
          cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId)
    return 0;
  slot::GetCorrectionsView view;
  if (!slot::parse_get_corrections(bytes, size, view))
    return 0;
  return cudaq::qec::decoding::rpc::bit_packed_bytes(
      static_cast<std::size_t>(std::max<std::int64_t>(view.return_size, 0)));
}

} // namespace

// ─── null_session ──────────────────────────────────────────────────────────
//
// The `null` backend. It discards every frame, but touches every byte of it
// through an atomic checksum so the optimizer cannot prove the frame is dead
// and elide the serialization work the real backends would also have to pay
// for. Completes synchronously inside send(), on the timing thread: there is
// no decoder and no payload, so there is nothing to hand off to a worker.

namespace {

class null_session : public session {
public:
  void start(run_ctx &collector) override { collector_ = &collector; }

  void send(const frame &f, tag t) override {
    checksum(f);
    scratch_.assign(reply_capacity_for(f.bytes, f.size), 0);
    handle_reply(*collector_, t, RpcStatus::OK, scratch_.data(),
                 scratch_.size(), now_ns());
  }

  void event_done(std::uint32_t event, std::uint32_t issued, std::int32_t term,
                  bool has_term) override {
    handle_event_done(*collector_, event, issued, term, has_term);
  }

  void stop(std::chrono::nanoseconds) override {}

private:
  // Folds every byte of the frame into an atomic accumulator so the compiler
  // cannot elide the build/serialize step even though the result is unused.
  void checksum(const frame &f) {
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < f.size; ++i)
      acc ^= (std::uint64_t(f.bytes[i]) << (8 * (i & 7)));
    checksum_.fetch_xor(acc, std::memory_order_relaxed);
  }

  std::atomic<std::uint64_t> checksum_{0};
  run_ctx *collector_ = nullptr;
  std::vector<std::uint8_t> scratch_;
};

} // namespace

std::unique_ptr<session> make_null_session() {
  return std::make_unique<null_session>();
}

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_null_sessions(const std::vector<std::uint64_t> &decoder_ids) {
  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(decoder_ids.size());
  for (auto id : decoder_ids)
    out.emplace_back(id, make_null_session());
  return out;
}

void adopt_sessions(
    std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> sessions,
    std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> &owned,
    std::unordered_map<std::uint64_t, session *> &router) {
  for (auto &[id, s] : sessions) {
    if (!router.emplace(id, s.get()).second)
      throw std::invalid_argument("decoder_id " + std::to_string(id) +
                                  " is named by more than one backend");
    owned.emplace_back(id, std::move(s));
  }
}

// ─── inproc_session ────────────────────────────────────────────────────────
//
// One session per decoder, dispatching straight to that decoder's own
// DecodingSession payload-level cores (skipping shared-memory rings and CUDA
// graph dispatch). The timing thread publishes requests into a single-
// producer/single-consumer ring; one worker thread both runs the decoder and
// reports the reply straight to handle_reply() -- no reader thread, no
// request_id lookup.

namespace {

/// Single-producer/single-consumer ring, timing thread -> worker. Each side
/// keeps a cached copy of the other's index and only re-reads the atomic
/// when that cache says empty/full, so the hot path never touches the other
/// core's cache line.
class request_ring {
public:
  static constexpr std::size_t kCapacity = 4096; // comfortably above any
                                                 // pipeline depth in use
  static constexpr std::size_t kInlineBytes = 256;

  struct entry {
    enum kind_t : std::uint8_t { kRequest, kEventDone, kStop } kind = kStop;
    tag t{};
    std::uint32_t frame_len = 0;
    std::array<std::uint8_t, kInlineBytes> inline_bytes{};
    std::vector<std::uint8_t> overflow_bytes; // used when frame_len is large
    std::uint32_t event = 0, issued = 0;
    std::int32_t term = 0;
    bool has_term = false;

    const std::uint8_t *bytes() const {
      return frame_len <= kInlineBytes ? inline_bytes.data()
                                       : overflow_bytes.data();
    }
  };

  request_ring() : buf_(kCapacity) {}

  // Producer (timing thread) only: reserve the next slot, fill it in place,
  // then publish(). Spins if the ring is full.
  entry &reserve() {
    while (tail_local_ - cached_head_ >= kCapacity) {
      cached_head_ = head_.load(std::memory_order_acquire);
      if (tail_local_ - cached_head_ >= kCapacity)
        std::this_thread::yield();
    }
    return buf_[tail_local_ & kMask];
  }
  void publish() {
    tail_.store(++tail_local_, std::memory_order_release);
    if (parked_.load(std::memory_order_seq_cst)) {
      std::lock_guard<std::mutex> lock(park_mu_);
      park_cv_.notify_one();
    }
  }

  // Consumer (worker thread) only.
  entry *try_pop() {
    if (head_local_ == cached_tail_) {
      cached_tail_ = tail_.load(std::memory_order_acquire);
      if (head_local_ == cached_tail_)
        return nullptr;
    }
    return &buf_[head_local_ & kMask];
  }
  void pop_done() { head_.store(++head_local_, std::memory_order_release); }

  // Idle-park protocol: the worker sets `parked_`, re-checks the ring, then
  // waits; publish() checks `parked_` after publishing, closing the lost-
  // wake window at the cost of one seq_cst load per push.
  std::atomic<bool> parked_{false};
  std::mutex park_mu_;
  std::condition_variable park_cv_;

private:
  static constexpr std::size_t kMask = kCapacity - 1;
  static_assert((kCapacity & kMask) == 0, "kCapacity must be a power of two");

  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
  std::size_t tail_local_ = 0, cached_head_ = 0; // producer-owned
  std::size_t head_local_ = 0, cached_tail_ = 0; // consumer-owned
  std::vector<entry> buf_;
};

/// How long the worker spins on an empty ring before parking on a condvar.
constexpr auto kIdleSpin = std::chrono::microseconds(50);

class inproc_session : public session {
public:
  inproc_session(std::shared_ptr<SessionRegistry> registry,
                 std::uint64_t decoder_id)
      : registry_(std::move(registry)) {
    dec_ = registry_->find(decoder_id);
    assert(dec_ && "make_inproc_sessions() must only construct a session "
                   "for a decoder_id actually present in the config");
  }

  ~inproc_session() override {
    if (worker_.joinable()) {
      ring_.reserve().kind = request_ring::entry::kStop;
      ring_.publish();
      worker_.join();
    }
  }

  void start(run_ctx &collector) override {
    collector_ = &collector;
    worker_ = std::thread([this] { drain(); });
  }

  void send(const frame &f, tag t) override {
    if (f.size < sizeof(cudaq::realtime::RPCHeader))
      throw std::invalid_argument(
          "inproc_session: frame is smaller than RPCHeader");
    auto &e = ring_.reserve();
    e.kind = request_ring::entry::kRequest;
    e.t = t;
    e.frame_len = static_cast<std::uint32_t>(f.size);
    if (f.size <= e.inline_bytes.size())
      std::memcpy(e.inline_bytes.data(), f.bytes, f.size);
    else
      e.overflow_bytes.assign(f.bytes, f.bytes + f.size);
    ring_.publish();
  }

  void event_done(std::uint32_t event, std::uint32_t issued, std::int32_t term,
                  bool has_term) override {
    auto &e = ring_.reserve();
    e.kind = request_ring::entry::kEventDone;
    e.event = event;
    e.issued = issued;
    e.term = term;
    e.has_term = has_term;
    ring_.publish();
  }

  // Pushes the stop marker and joins; the worker drains everything ahead of
  // it first, so `drain` only bounds a decoder that hangs -- which this
  // backend, dispatching in-process code the caller controls, does not.
  void stop(std::chrono::nanoseconds) override {
    if (!worker_.joinable())
      return;
    ring_.reserve().kind = request_ring::entry::kStop;
    ring_.publish();
    worker_.join();
  }

private:
  void drain() {
    using clock = std::chrono::steady_clock;
    bool idling = false;
    clock::time_point idle_since;
    for (;;) {
      auto *e = ring_.try_pop();
      if (!e) {
        if (!idling) {
          idling = true;
          idle_since = clock::now();
        }
        if (clock::now() - idle_since < kIdleSpin) {
          std::this_thread::yield();
          continue;
        }
        ring_.parked_.store(true, std::memory_order_seq_cst);
        e = ring_.try_pop();
        if (!e) {
          std::unique_lock<std::mutex> lock(ring_.park_mu_);
          ring_.park_cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
        ring_.parked_.store(false, std::memory_order_relaxed);
        if (!e)
          continue;
      }
      idling = false;

      switch (e->kind) {
      case request_ring::entry::kRequest: {
        const tag t = e->t;
        const auto *bytes = e->bytes();
        const auto len = e->frame_len;
        reply_scratch_.resize(reply_capacity_for(bytes, len));
        const auto result = dispatch_rpc(
            *dec_, bytes, len, reply_scratch_.data(), reply_scratch_.size());
        const auto ret_ns = now_ns();
        ring_.pop_done();
        handle_reply(*collector_, t, result.status, reply_scratch_.data(),
                     result.reply_len, ret_ns);
        break;
      }
      case request_ring::entry::kEventDone: {
        const auto event = e->event, issued = e->issued;
        const auto term = e->term;
        const auto has_term = e->has_term;
        ring_.pop_done();
        handle_event_done(*collector_, event, issued, term, has_term);
        break;
      }
      case request_ring::entry::kStop:
        ring_.pop_done();
        return;
      }
    }
  }

  std::shared_ptr<SessionRegistry> registry_; // keeps every decoder alive
  DecodingSession *dec_ = nullptr;
  request_ring ring_;
  run_ctx *collector_ = nullptr;
  std::vector<std::uint8_t> reply_scratch_; // worker-owned
  std::thread worker_;
};

} // namespace

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_inproc_sessions(
    const cudaq::qec::decoding::config::multi_decoder_config &config) {
  auto registry = std::make_shared<SessionRegistry>();
  registry->load_from_config(config, "playback-emulator");

  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(config.decoders.size());
  for (const auto &d : config.decoders) {
    const auto id = static_cast<std::uint64_t>(d.id);
    out.emplace_back(id, std::make_unique<inproc_session>(registry, id));
  }
  return out;
}

// ─── udp_session ───────────────────────────────────────────────────────────
//
// Connected UDP socket(s) to a decoding server speaking decoder_rpc_wire_
// format.h. Pure transport: this file looks no further than the generic
// RPCHeader/RPCResponse framing and has no notion of which RPC a frame
// holds. The receiver thread matches datagrams to the tag they were sent
// under, sweeps timed-out requests, and delivers `event_done` notices --
// the only thread that ever reports into the collector.

namespace {

using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

// A UDP datagram is at most 65507 bytes of payload (65535 - 8-byte UDP
// header - 20-byte IPv4 header); size scratch buffers to that.
constexpr std::size_t kMaxDatagram = 65507;

// How often the receiver wakes on its own (via SO_RCVTIMEO) to sweep expired
// requests and drain `event_done` notices, independent of any one request's
// own timeout.
constexpr std::uint32_t kReceiveTickMs = 1;

/// Splits "host:port" on the LAST ':'
void split_endpoint(const std::string &endpoint, std::string &host,
                    std::string &port) {
  auto pos = endpoint.rfind(':');
  if (pos == std::string::npos)
    throw std::runtime_error("playback endpoint missing ':port': " + endpoint);
  host = endpoint.substr(0, pos);
  port = endpoint.substr(pos + 1);
}

int make_connected_udp_socket(const std::string &endpoint) {
  std::string host, port;
  split_endpoint(endpoint, host, port);

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo *res = nullptr;
  int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
  if (rc != 0 || !res)
    throw std::runtime_error("playback UDP: failed to resolve endpoint '" +
                             endpoint + "': " + gai_strerror(rc));

  int fd = -1;
  for (addrinfo *p = res; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0)
      continue;
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0)
      break;
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0)
    throw std::runtime_error("playback UDP: failed to connect to '" + endpoint +
                             "'");

  int rcvbuf = 1 << 20; // generous SO_RCVBUF
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

  timeval tv{};
  tv.tv_usec = static_cast<long>(kReceiveTickMs * 1000);
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  return fd;
}

/// One connected UDP socket talks to one decoder on the decoding server.
/// Exactly one thread publishes on a session (see session.h).
class udp_session : public session {
public:
  udp_session(int fd, std::uint32_t timeout_ms)
      : fd_(fd), timeout_(std::chrono::milliseconds(timeout_ms)) {
    max_frame_bytes = static_cast<std::uint32_t>(kMaxDatagram);
  }

  ~udp_session() override {
    if (receiver_.joinable()) {
      stop_.store(true, std::memory_order_release);
      if (fd_ >= 0)
        ::shutdown(fd_, SHUT_RDWR); // unblock a receiver parked in recv()
      receiver_.join();
    }
    if (fd_ >= 0)
      ::close(fd_);
  }

  void start(run_ctx &collector) override {
    collector_ = &collector;
    receiver_ = std::thread([this] { receive_loop(); });
  }

  void send(const frame &f, tag t) override {
    if (f.size < sizeof(RPCHeader))
      throw std::invalid_argument(
          "udp_session::send: frame is smaller than RPCHeader");
    const std::uint32_t request_id =
        reinterpret_cast<const RPCHeader *>(f.bytes)->request_id;
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    {
      std::lock_guard<std::mutex> lock(mu_);
      pending_[request_id] = {t, deadline};
    }
    if (::send(fd_, f.bytes, f.size, 0) < 0) {
      // Nothing else will ever complete this request, since it never left
      // the socket for the receiver to eventually time out on its own.
      bool erased;
      {
        std::lock_guard<std::mutex> lock(mu_);
        erased = pending_.erase(request_id) != 0;
      }
      if (erased)
        complete(t, RpcStatus::INTERNAL_ERROR, nullptr, 0);
    }
  }

  void event_done(std::uint32_t event, std::uint32_t issued, std::int32_t term,
                  bool has_term) override {
    std::lock_guard<std::mutex> lock(notices_mu_);
    notices_.push_back({event, issued, term, has_term});
  }

  /// Waits up to `drain` for `pending_` to empty (the receiver sweeps it),
  /// force-completes whatever is left as INTERNAL_ERROR, then stops the
  /// receiver.
  void stop(std::chrono::nanoseconds drain) override {
    const auto deadline = std::chrono::steady_clock::now() + drain;
    while (!pending_empty() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

    std::vector<std::pair<std::uint32_t, pending>> leftover;
    {
      std::lock_guard<std::mutex> lock(mu_);
      leftover.assign(pending_.begin(), pending_.end());
      pending_.clear();
    }
    for (auto &[id, pe] : leftover)
      complete(pe.t, RpcStatus::INTERNAL_ERROR, nullptr, 0);

    if (receiver_.joinable()) {
      stop_.store(true, std::memory_order_release);
      ::shutdown(fd_, SHUT_RDWR);
      receiver_.join();
    }
    drain_notices();
  }

private:
  /// One outstanding request: where its reply must be tagged, and when it
  /// gives up waiting. Only the receiver thread ever reads or writes an
  /// entry once it exists.
  struct pending {
    tag t;
    std::chrono::steady_clock::time_point deadline;
  };
  struct notice {
    std::uint32_t event, issued;
    std::int32_t term;
    bool has_term;
  };

  bool pending_empty() {
    std::lock_guard<std::mutex> lock(mu_);
    return pending_.empty();
  }

  void complete(tag t, RpcStatus status, const std::uint8_t *body,
                std::size_t len) {
    handle_reply(*collector_, t, status, body, len, now_ns());
  }

  void drain_notices() {
    std::vector<notice> batch;
    {
      std::lock_guard<std::mutex> lock(notices_mu_);
      batch.swap(notices_);
    }
    for (auto &n : batch)
      handle_event_done(*collector_, n.event, n.issued, n.term, n.has_term);
  }

  /// No dedicated "local timeout" status exists in RpcStatus; a client-side
  /// synthesized timeout/hard-error is reported as INTERNAL_ERROR. Each
  /// request's own deadline (send + timeout_) governs it, not overall socket
  /// silence -- unrelated traffic must never mask one dead request forever.
  void sweep_stale() {
    std::vector<std::pair<std::uint32_t, pending>> stale;
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (auto it = pending_.begin(); it != pending_.end();) {
        if (now < it->second.deadline) {
          ++it;
          continue;
        }
        stale.emplace_back(it->first, it->second);
        it = pending_.erase(it);
      }
    }
    for (auto &[id, pe] : stale)
      complete(pe.t, RpcStatus::INTERNAL_ERROR, nullptr, 0);
  }

  void receive_loop() {
    std::vector<std::uint8_t> scratch(kMaxDatagram);
    while (!stop_.load(std::memory_order_acquire)) {
      // Runs every SO_RCVTIMEO tick (kReceiveTickMs) as well as after every
      // datagram, so a notice or an expired request is never held up by a
      // quiet -- or a noisy -- socket.
      sweep_stale();
      drain_notices();
      const ssize_t n = ::recv(fd_, scratch.data(), scratch.size(), 0);
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
          continue; // SO_RCVTIMEO tick: just re-check stop_
        if (stop_.load(std::memory_order_acquire))
          return; // intentional shutdown
        // A connected UDP socket also surfaces async ICMP errors here (e.g.
        // ECONNREFUSED while the server is briefly down); that is not a
        // reason to stop listening.
        continue;
      }
      if (static_cast<std::size_t>(n) < sizeof(RPCResponse))
        continue; // short/garbage datagram
      RPCResponse resp;
      std::memcpy(&resp, scratch.data(), sizeof(RPCResponse));
      if (resp.magic != RPC_MAGIC_RESPONSE)
        continue; // garbage/truncated datagram

      tag t;
      bool found;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pending_.find(resp.request_id);
        found = it != pending_.end();
        if (found) {
          t = it->second.t;
          pending_.erase(it);
        }
      }
      if (!found)
        continue; // stale reply: nobody is waiting on this id any more
      const std::size_t avail =
          static_cast<std::size_t>(n) - sizeof(RPCResponse);
      complete(t, static_cast<RpcStatus>(resp.status),
               scratch.data() + sizeof(RPCResponse),
               std::min<std::size_t>(resp.result_len, avail));
    }
  }

  int fd_ = -1;
  std::chrono::milliseconds timeout_;
  std::mutex mu_;
  std::unordered_map<std::uint32_t, pending> pending_;
  std::mutex notices_mu_;
  std::vector<notice> notices_;
  std::atomic<bool> stop_{false};
  run_ctx *collector_ = nullptr;
  std::thread receiver_;
};

} // namespace

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_udp_sessions(
    const std::unordered_map<std::uint64_t, std::string> &endpoints,
    std::uint32_t timeout_ms) {
  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(endpoints.size());
  for (const auto &[id, endpoint] : endpoints) {
    int fd = make_connected_udp_socket(endpoint);
    out.emplace_back(id, std::make_unique<udp_session>(fd, timeout_ms));
  }
  return out;
}

// ─── cpu_roce_session ────────────────────────────────────────────────────────
//
// CPU RoCE (libibverbs) client to a `decoding_server --transport=cpu_roce`
// speaking the CUDA-Q CpuRoceTransceiver ring protocol: a request is
// RDMA-written into the server's rx slot of the same index and the reply is
// Sent back into ours, in strict slot order (at most `num_slots` in flight).

namespace {

void validate(const cpu_roce_options &opts) {
  if (opts.num_slots == 0 || (opts.num_slots & (opts.num_slots - 1)) != 0)
    throw std::invalid_argument(
        "cpu_roce num_slots must be a non-zero power of two");
  if (opts.slot_size < sizeof(RPCHeader) ||
      opts.slot_size < sizeof(RPCResponse))
    throw std::invalid_argument(
        "cpu_roce slot_size is too small to hold an RPC header/response");
  if (opts.device.empty() || opts.local_ip.empty())
    throw std::invalid_argument("cpu_roce requires a device and a local_ip");
}

#ifdef CUDAQ_QEC_PLAYBACK_CPU_ROCE

inline void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

/// Byte-for-byte the service end's rendezvous struct (network order).
struct roce_rendezvous {
  std::uint32_t qp_number = 0, rkey = 0, roce_ipv4 = 0;
};

[[noreturn]] void roce_fail(const std::string &what) {
  throw std::runtime_error("playback cpu_roce: " + what);
}

/// TCP connect to the rendezvous, retrying until `deadline`; the socket
/// timeouts bound the swap that follows too.
int connect_rendezvous(const std::string &host, const std::string &port,
                       std::chrono::steady_clock::time_point deadline) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
    roce_fail("failed to resolve rendezvous '" + host + ":" + port + "'");
  for (;;) {
    const long ms =
        std::max<long>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
                              deadline - std::chrono::steady_clock::now())
                              .count());
    const timeval tv{ms / 1000, (ms % 1000) * 1000};
    for (addrinfo *p = res; p; p = p->ai_next) {
      const int fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
      if (fd < 0)
        continue;
      ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
      if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
        freeaddrinfo(res);
        return fd;
      }
      ::close(fd);
    }
    if (std::chrono::steady_clock::now() > deadline) {
      freeaddrinfo(res);
      roce_fail("could not reach rendezvous '" + host + ":" + port + "'");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

/// setup() a transceiver, swap QP/rkey/IP with the service over TCP (client
/// writes first, mirroring the server), then connect(). Throws on failure.
cpu_roce_transceiver_t connect_roce(const std::string &endpoint,
                                    const cpu_roce_options &opts) {
  std::string host, port;
  split_endpoint(endpoint, host, port);
  in_addr local{};
  if (inet_pton(AF_INET, opts.local_ip.c_str(), &local) != 1)
    roce_fail("invalid local_ip '" + opts.local_ip + "'");

  cpu_roce_transceiver_t x = cpu_roce_create_transceiver(
      opts.device.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/0u, opts.slot_size,
      opts.slot_size, opts.num_slots, /*peer_ip=*/"0.0.0.0", 0, 0, 0, 0,
      CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM, 0, 0);
  if (!x)
    roce_fail("transceiver create failed (device='" + opts.device + "')");
  cpu_roce_set_local_ip(x, opts.local_ip.c_str());
  try {
    if (!cpu_roce_setup(x))
      roce_fail("transceiver setup() failed (device='" + opts.device +
                "', local_ip='" + opts.local_ip + "')");
    const int fd = connect_rendezvous(
        host, port,
        std::chrono::steady_clock::now() +
            std::chrono::milliseconds(opts.connect_timeout_ms));
    const roce_rendezvous self{htonl(cpu_roce_get_qp_number(x)),
                               htonl(cpu_roce_get_rkey(x)), local.s_addr};
    roce_rendezvous peer{};
    const bool ok = ::send(fd, &self, sizeof self, 0) == sizeof self &&
                    ::recv(fd, &peer, sizeof peer, MSG_WAITALL) == sizeof peer;
    ::close(fd);
    if (!ok)
      roce_fail("rendezvous exchange failed");
    char peer_ip[INET_ADDRSTRLEN] = {0};
    const in_addr pa{peer.roce_ipv4};
    if (!inet_ntop(AF_INET, &pa, peer_ip, sizeof peer_ip))
      roce_fail("bad peer RoCE IPv4 in rendezvous");
    if (!cpu_roce_connect(x, ntohl(peer.qp_number), peer_ip, ntohl(peer.rkey)))
      roce_fail("transceiver connect() failed");
  } catch (...) {
    cpu_roce_destroy_transceiver(x);
    throw;
  }
  return x;
}

class cpu_roce_session : public session {
public:
  cpu_roce_session(cpu_roce_transceiver_t xcvr, std::uint32_t num_slots,
                   std::uint32_t slot_size, std::chrono::milliseconds timeout)
      : xcvr_(xcvr), slot_mask_(num_slots - 1),
        stride_(cpu_roce_get_page_size(xcvr)), timeout_(timeout),
        tx_data_(
            static_cast<std::uint8_t *>(cpu_roce_get_tx_ring_data_addr(xcvr))),
        tx_flags_(cpu_roce_get_tx_ring_flag_addr(xcvr)),
        rx_data_(
            static_cast<std::uint8_t *>(cpu_roce_get_rx_ring_data_addr(xcvr))),
        rx_flags_(cpu_roce_get_rx_ring_flag_addr(xcvr)),
        max_inflight_(num_slots > 1 ? num_slots - 1 : 1),
        slot_owner_(num_slots, kNoOwner), reply_scratch_(stride_) {
    max_frame_bytes = slot_size;
  }

  ~cpu_roce_session() override {
    stop(std::chrono::nanoseconds(0));
    cpu_roce_destroy_transceiver(xcvr_);
  }

  void start(run_ctx &collector) override {
    collector_ = &collector;
    // The transceiver's busy-polling RX/TX threads run from start() to stop().
    monitor_ = std::thread([this] {
      monitor_entered_.store(true, std::memory_order_release);
      cpu_roce_blocking_monitor(xcvr_);
    });
    worker_ = std::thread([this] { worker_loop(); });
  }

  void send(const frame &f, tag t) override {
    if (f.size < sizeof(RPCHeader))
      throw std::invalid_argument(
          "cpu_roce_session::send: frame is smaller than RPCHeader");
    if (f.size > stride_)
      throw std::invalid_argument(
          "cpu_roce_session::send: frame exceeds the slot size");
    auto &e = ring_.reserve();
    e.kind = request_ring::entry::kRequest;
    e.t = t;
    e.frame_len = static_cast<std::uint32_t>(f.size);
    if (f.size <= e.inline_bytes.size())
      std::memcpy(e.inline_bytes.data(), f.bytes, f.size);
    else
      e.overflow_bytes.assign(f.bytes, f.bytes + f.size);
    ring_.publish();
  }

  void event_done(std::uint32_t event, std::uint32_t issued, std::int32_t term,
                  bool has_term) override {
    auto &e = ring_.reserve();
    e.kind = request_ring::entry::kEventDone;
    e.event = event;
    e.issued = issued;
    e.term = term;
    e.has_term = has_term;
    ring_.publish();
  }

  /// Idempotent; the destructor calls it with a zero drain.
  void stop(std::chrono::nanoseconds drain) override {
    if (worker_.joinable()) {
      giveup_ns_.store((clock::now() + drain).time_since_epoch().count(),
                       std::memory_order_release);
      ring_.reserve().kind = request_ring::entry::kStop;
      ring_.publish();
      worker_.join();
    }
    // cpu_roce_close() must not race a cpu_roce_blocking_monitor() that has
    // not spawned its threads yet (they would outlive the ring memory).
    while (monitor_.joinable() &&
           !monitor_entered_.load(std::memory_order_acquire))
      std::this_thread::yield();
    cpu_roce_close(xcvr_);
    if (monitor_.joinable())
      monitor_.join();
  }

private:
  using clock = std::chrono::steady_clock;
  static constexpr std::uint32_t kNoOwner = 0xFFFFFFFFu;

  /// One outstanding request. Worker-thread-only.
  struct pending {
    tag t;
    clock::time_point deadline;
    std::uint32_t slot;
  };

  clock::time_point giveup() const {
    return clock::time_point{
        clock::duration(giveup_ns_.load(std::memory_order_acquire))};
  }

  void complete(tag t, RpcStatus status, const std::uint8_t *body,
                std::size_t len) {
    handle_reply(*collector_, t, status, body, len, now_ns());
  }

  /// Publish the head request into the next TX slot; false (it stays queued)
  /// while that slot's last request is unanswered or still being shipped.
  bool try_publish(const request_ring::entry &e) {
    const std::uint32_t slot = rr_;
    if (pending_.size() >= max_inflight_ || slot_owner_[slot] != kNoOwner ||
        __atomic_load_n(&tx_flags_[slot], __ATOMIC_ACQUIRE) != 0)
      return false;
    const std::uint32_t rid =
        reinterpret_cast<const RPCHeader *>(e.bytes())->request_id;
    std::uint8_t *dst = tx_data_ + static_cast<std::size_t>(slot) * stride_;
    std::memset(dst, 0, stride_); // the whole stride is shipped
    std::memcpy(dst, e.bytes(), e.frame_len);
    pending_.emplace(rid, pending{e.t, clock::now() + timeout_, slot});
    slot_owner_[slot] = rid;
    __atomic_store_n(&tx_flags_[slot], reinterpret_cast<std::uint64_t>(dst),
                     __ATOMIC_RELEASE);
    rr_ = (rr_ + 1) & slot_mask_;
    return true;
  }

  /// Collect the reply at the RX cursor, if any. Replies land in arrival
  /// order (Sends consume our recv WQEs FIFO), which drifts from TX slot
  /// order once a request has timed out and its slot was reused, so the TX
  /// slot is released by the matched request_id, never by RX position.
  bool poll_reply() {
    const std::uint32_t slot = cursor_;
    if (__atomic_load_n(&rx_flags_[slot], __ATOMIC_ACQUIRE) == 0)
      return false;
    std::memcpy(reply_scratch_.data(),
                rx_data_ + static_cast<std::size_t>(slot) * stride_, stride_);
    __atomic_store_n(&rx_flags_[slot], 0ull, __ATOMIC_RELEASE); // re-arm
    cursor_ = (cursor_ + 1) & slot_mask_;

    const auto *resp =
        reinterpret_cast<const RPCResponse *>(reply_scratch_.data());
    auto it = resp->magic == RPC_MAGIC_RESPONSE
                  ? pending_.find(resp->request_id)
                  : pending_.end();
    if (it == pending_.end())
      return true; // garbage, or a late reply whose request already timed out
    const pending p = it->second;
    pending_.erase(it);
    if (slot_owner_[p.slot] == resp->request_id)
      slot_owner_[p.slot] = kNoOwner;
    complete(
        p.t, static_cast<RpcStatus>(resp->status),
        reply_scratch_.data() + sizeof(RPCResponse),
        std::min<std::size_t>(resp->result_len, stride_ - sizeof(RPCResponse)));
    return true;
  }

  /// Time out requests past their deadline, freeing their slot. As with udp,
  /// a client-side timeout is reported INTERNAL_ERROR.
  bool sweep_stale() {
    const auto now = clock::now();
    bool any = false;
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (now < it->second.deadline) {
        ++it;
        continue;
      }
      if (slot_owner_[it->second.slot] == it->first)
        slot_owner_[it->second.slot] = kNoOwner;
      const tag t = it->second.t;
      it = pending_.erase(it);
      complete(t, RpcStatus::INTERNAL_ERROR, nullptr, 0);
      any = true;
    }
    return any;
  }

  void worker_loop() {
    bool idling = false;
    clock::time_point idle_since;
    for (;;) {
      bool did_work = poll_reply();
      did_work |= sweep_stale();

      if (auto *e = ring_.try_pop()) {
        switch (e->kind) {
        case request_ring::entry::kRequest:
          if (try_publish(*e)) {
            ring_.pop_done();
            did_work = true;
          } else if (clock::now() >= giveup()) {
            // stop()'s drain elapsed with no free slot: fail it, reach kStop.
            const tag t = e->t;
            ring_.pop_done();
            complete(t, RpcStatus::INTERNAL_ERROR, nullptr, 0);
            did_work = true;
          } // else the ring is full: keep the head queued, service replies
          break;
        case request_ring::entry::kEventDone: {
          const auto ev = e->event, is = e->issued;
          const auto tm = e->term;
          const auto ht = e->has_term;
          ring_.pop_done();
          handle_event_done(*collector_, ev, is, tm, ht);
          did_work = true;
          break;
        }
        case request_ring::entry::kStop:
          // Honor outstanding replies up to the drain deadline, fail the rest.
          ring_.pop_done();
          while (!pending_.empty() && clock::now() < giveup())
            if (!poll_reply()) {
              sweep_stale();
              cpu_pause();
            }
          for (auto &[rid, p] : pending_)
            complete(p.t, RpcStatus::INTERNAL_ERROR, nullptr, 0);
          return;
        }
      }

      if (did_work) {
        idling = false;
        continue;
      }
      if (!idling) {
        idling = true;
        idle_since = clock::now();
      }
      if (clock::now() - idle_since < kIdleSpin)
        cpu_pause();
      else
        std::this_thread::yield();
    }
  }

  cpu_roce_transceiver_t xcvr_;
  const std::uint32_t slot_mask_;
  const std::size_t stride_;
  const std::chrono::milliseconds timeout_;
  // Ring memory, owned by the transceiver.
  std::uint8_t *const tx_data_;
  std::uint64_t *const tx_flags_;
  std::uint8_t *const rx_data_;
  std::uint64_t *const rx_flags_;

  // Worker-thread-only state.
  std::uint32_t rr_ = 0;     // next TX slot to publish into
  std::uint32_t cursor_ = 0; // next RX slot to collect from
  const std::uint32_t max_inflight_;
  std::unordered_map<std::uint32_t, pending> pending_;
  std::vector<std::uint32_t> slot_owner_; // request_id per slot, or kNoOwner
  std::vector<std::uint8_t> reply_scratch_;

  std::atomic<std::int64_t> giveup_ns_{
      std::numeric_limits<std::int64_t>::max()};
  std::atomic<bool> monitor_entered_{false};
  request_ring ring_; // timing thread -> worker
  run_ctx *collector_ = nullptr;
  std::thread worker_;
  std::thread monitor_; // runs cpu_roce_blocking_monitor
};

#endif // CUDAQ_QEC_PLAYBACK_CPU_ROCE

} // namespace

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_cpu_roce_sessions(
    const std::unordered_map<std::uint64_t, std::string> &endpoints,
    const cpu_roce_options &opts, std::uint32_t timeout_ms) {
  validate(opts);
#ifndef CUDAQ_QEC_PLAYBACK_CPU_ROCE
  throw std::runtime_error(
      "playback emulator was built without CPU RoCE support "
      "(cudaq-realtime-cpu-roce-transport / libibverbs not found at build "
      "time)");
#else
  // Connect concurrently: the server accepts its rings one at a time, so a
  // serial loop in the wrong order would deadlock.
  std::vector<std::pair<std::uint64_t, std::string>> eps(endpoints.begin(),
                                                         endpoints.end());
  std::vector<std::future<cpu_roce_transceiver_t>> connecting;
  for (const auto &[id, ep] : eps)
    connecting.push_back(
        std::async(std::launch::async, connect_roce, ep, std::cref(opts)));
  std::vector<cpu_roce_transceiver_t> xcvrs;
  std::exception_ptr err;
  for (auto &f : connecting)
    try {
      xcvrs.push_back(f.get());
    } catch (...) {
      if (!err)
        err = std::current_exception();
    }
  if (err) {
    for (auto x : xcvrs)
      cpu_roce_destroy_transceiver(x);
    std::rethrow_exception(err);
  }

  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  for (std::size_t i = 0; i < eps.size(); ++i)
    out.emplace_back(eps[i].first, std::make_unique<cpu_roce_session>(
                                       xcvrs[i], opts.num_slots, opts.slot_size,
                                       std::chrono::milliseconds(timeout_ms)));
  return out;
#endif
}

} // namespace cudaq::qec::playback
