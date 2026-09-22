/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file emulator.cpp
/// @brief plan() and run(): pre-serialize every static frame before t0,
/// validate every frame size it can before t0, then run one timing thread that
/// does nothing between its deadlines but wait and dispatch.

#include "emulator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace cudaq::qec::playback {

using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPCHeader;
namespace wire = cudaq::qec::decoding::rpc;

namespace {

// -- Timing core: sleep off the bulk of a wait, then spin
// the last `kSpinSlackNs` to the clock-read floor.

std::uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

constexpr std::uint64_t kMaxNapNs = 1'000'000; // cap on one nanosleep step

void sleep_until(std::uint64_t target_ns) {
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(target_ns / 1'000'000'000ull);
  ts.tv_nsec = static_cast<long>(target_ns % 1'000'000'000ull);
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

/// How much of a wait `wait_until` spins rather than sleeps: a little above
/// typical clock_nanosleep overshoot. Fixed, not measured/configurable --
/// a startup calibration risks sampling one bad preemption and spinning on
/// every later deadline for the rest of the run.
constexpr std::uint64_t kSpinSlackNs = 200'000;

/// The floor between rounds of an unpaced (every==0) stream waiting on
/// `until=`. Without this, nothing bounds how far ahead of a slow session
/// worker the send loop can get -- a bare yield is only a scheduling hint,
/// not real backpressure.
constexpr std::uint64_t kUnpacedStreamFloorNs = 50'000;

/// Saturating add. A deadline far enough out to overflow the clock is
/// clamped to "never"
inline std::uint64_t add_sat(std::uint64_t a, std::uint64_t b) {
  std::uint64_t sum = 0;
  return __builtin_add_overflow(a, b, &sum) ? ~std::uint64_t(0) : sum;
}

/// Block until CLOCK_MONOTONIC reaches `deadline_ns` (an absolute
/// CLOCK_MONOTONIC timestamp). Sleeps in bounded naps down to
/// `deadline_ns - kSpinSlackNs`, then spins a plain clock read to the deadline
void wait_until(std::uint64_t deadline_ns) {
  const std::uint64_t spin_from =
      deadline_ns > kSpinSlackNs ? deadline_ns - kSpinSlackNs : 0;
  for (std::uint64_t t = now_ns(); t < spin_from; t = now_ns())
    sleep_until(t + std::min(spin_from - t, kMaxNapNs));
  while (now_ns() < deadline_ns) {
  }
}

// -- Frame construction / parsing helpers. --

void append_bytes(std::vector<std::uint8_t> &buf, const void *p,
                  std::size_t n) {
  const std::size_t off = buf.size();
  buf.resize(off + n);
  std::memcpy(buf.data() + off, p, n);
}

/// Pack one-byte-per-bit (0x00/0x01) values from `bits` into LSB-first bytes,
/// appended to `buf`.
void append_packed_bits(std::vector<std::uint8_t> &buf,
                        const std::uint8_t *bits, std::size_t n) {
  const std::size_t off = buf.size();
  buf.resize(off + wire::bit_packed_bytes(n), 0);
  for (std::size_t i = 0; i < n; ++i)
    if (bits[i])
      buf[off + i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
}

/// Unpack LSB-first bit-packed bytes into one-byte-per-bit values, appended
/// to `out`.
void append_unpacked_bits(std::vector<std::uint8_t> &out,
                          const std::uint8_t *packed, std::size_t n_bits) {
  for (std::size_t i = 0; i < n_bits; ++i)
    out.push_back((packed[i / 8] >> (i % 8)) & 1u);
}

/// Build one wire frame: RPCHeader + a fixed payload struct, plus optional
/// trailing bit-packed syndrome bytes. The single point
/// where every operation's frame shape is assembled.
std::vector<std::uint8_t>
build_frame(std::uint32_t function_id, std::uint32_t request_id,
            const void *payload, std::size_t payload_len,
            const std::uint8_t *bits = nullptr, std::size_t n_bits = 0) {
  const std::size_t trailing = bits ? wire::bit_packed_bytes(n_bits) : 0;
  RPCHeader h{};
  h.magic = RPC_MAGIC_REQUEST;
  h.function_id = function_id;
  h.arg_len = static_cast<std::uint32_t>(payload_len + trailing);
  h.request_id = request_id;
  h.ptp_timestamp = 0;

  std::vector<std::uint8_t> buf;
  append_bytes(buf, &h, sizeof(h));
  append_bytes(buf, payload, payload_len);
  if (bits)
    append_packed_bits(buf, bits, n_bits);
  return buf;
}

/// The sendable bytes of a round plan()'s already serialized.
frame frame_of(const run_plan &plan, const round_plan &rp) {
  return {plan.frame_arena.data() + rp.frame_offset, rp.frame_len};
}

/// Overwrites a frame's request_id in place, after it was built with a
/// placeholder.
void set_request_id(std::uint8_t *frame_bytes, std::uint32_t rid) {
  reinterpret_cast<RPCHeader *>(frame_bytes)->request_id = rid;
}

std::vector<std::uint8_t> build_reset_frame(std::uint64_t decoder_id,
                                            std::uint32_t rid) {
  wire::ResetRequestPayload p{static_cast<std::int64_t>(decoder_id)};
  return build_frame(wire::kResetDecoderFunctionId, rid, &p, sizeof(p));
}

std::vector<std::uint8_t> build_enqueue_frame(std::uint64_t decoder_id,
                                              std::uint32_t rid,
                                              const std::uint8_t *bits,
                                              std::size_t n_bits) {
  wire::EnqueueRequestPayload p{static_cast<std::int64_t>(decoder_id),
                                /*counter=*/0,
                                /*syndrome_mapping_id=*/0,
                                static_cast<std::int64_t>(n_bits)};
  return build_frame(wire::kEnqueueSyndromesFunctionId, rid, &p, sizeof(p),
                     bits, n_bits);
}

std::vector<std::uint8_t> build_get_corrections_frame(std::uint64_t decoder_id,
                                                      std::int64_t return_size,
                                                      std::uint32_t rid) {
  // reset=1 always: a playback read consumes the shot it reports on.
  wire::GetCorrectionsRequestPayload p{static_cast<std::int64_t>(decoder_id),
                                       return_size,
                                       /*reset=*/1};
  return build_frame(wire::kGetCorrectionsFunctionId, rid, &p, sizeof(p));
}

/// The width a returning RPC requests
std::uint32_t return_size_for(const event &e) {
  return std::max(e.return_size, e.expected_count);
}

/// A session that returns OK but hands back fewer bytes than `return_size`
/// bits requires is indistinguishable from "the correction happens to be
/// all-zero" once those bytes are naively unpacked from a zero-initialized
/// buffer
RpcStatus reject_truncated_reply(RpcStatus status, std::size_t reply_len,
                                 std::uint32_t return_size) {
  if (status == RpcStatus::OK &&
      reply_len < wire::bit_packed_bytes(return_size))
    return RpcStatus::INTERNAL_ERROR;
  return status;
}

bool mismatches_expected(const schedule &sched, const event &e,
                         const std::uint8_t *bits, std::size_t n) {
  if (e.expected_count == 0)
    return false; // nothing to compare against
  return n != e.expected_count ||
         !std::equal(bits, bits + n,
                     sched.expected_arena.begin() + e.expected_offset);
}

/// Reject a schedule whose `until=`/`after=` can never come up before it is
/// asked for.
void check_signal_order(const schedule &sched) {
  std::vector<bool> raised(sched.signal_names.size(), false);
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    const auto &e = sched.events[i];
    if (e.op == operation::stream && e.until_signal_id != kNoSignal &&
        !raised[e.until_signal_id])
      throw std::invalid_argument("event " + std::to_string(i) +
                                  " streams until signal '" +
                                  sched.signal_names[e.until_signal_id] +
                                  "', which no earlier 'signal=' event raises");
    if (e.after_signal_id != kNoSignal && !raised[e.after_signal_id])
      throw std::invalid_argument("event " + std::to_string(i) +
                                  " dispatches after signal '" +
                                  sched.signal_names[e.after_signal_id] +
                                  "', which no earlier 'signal=' event raises");
    if (e.signal_id != kNoSignal)
      raised[e.signal_id] = true;
  }
}

} // namespace

std::shared_ptr<run_plan>
plan(const schedule &sched_in,
     const std::unordered_map<std::uint64_t, session *> &router,
     const std::unordered_map<std::uint32_t, syndrome_source *> &sources,
     const run_params &params) {
  auto impl = std::make_shared<run_plan>();
  impl->sched = sched_in;
  impl->router = router;
  impl->sources = sources;
  impl->params = params;
  auto &sched = impl->sched;

  check_signal_order(sched);

  // -- Every event must have somewhere to send to and something to draw from
  for (auto &e : sched.events) {
    if (!router.contains(e.decoder_id))
      throw std::invalid_argument("no session routes decoder_id=" +
                                  std::to_string(e.decoder_id));

    // Literal `source=0b<bits>` events carry their own round and name no
    // source, so only the ones that do have anything to look up.
    if (e.source_id != kNoSource &&
        (e.op == operation::stream || e.op == operation::enqueue_data) &&
        !sources.contains(e.source_id))
      throw std::invalid_argument(
          "no syndrome_source registered for source_id=" +
          std::to_string(e.source_id));
  }

  // -- Draw and serialize everything whose bytes are known before t0, in
  // file order, so a source is consumed in the order the schedule sends it.
  // A source stops being pre-drawable once an event on it decides its round
  // count at run time (`until=`); every later event on it must then draw live.
  impl->event_plans.resize(sched.events.size());
  std::unordered_map<std::uint32_t, bool> predrawable;

  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    auto &e = sched.events[i];
    auto &ep = impl->event_plans[i];
    const auto max_frame_bytes = router.at(e.decoder_id)->max_frame_bytes;

    auto place = [&](std::vector<std::uint8_t> bytes, std::uint32_t bits_offset,
                     std::uint32_t bits_count) {
      if (max_frame_bytes != 0 && bytes.size() > max_frame_bytes)
        throw std::invalid_argument(
            "event " + std::to_string(i) +
            " (decoder_id=" + std::to_string(e.decoder_id) + ") builds a " +
            std::to_string(bytes.size()) +
            "-byte frame, exceeding the session's max_frame_bytes=" +
            std::to_string(max_frame_bytes));
      round_plan rp;
      rp.frame_offset = static_cast<std::uint32_t>(impl->frame_arena.size());
      rp.frame_len = static_cast<std::uint32_t>(bytes.size());
      rp.bits_offset = bits_offset;
      rp.bits_count = bits_count;
      impl->frame_arena.insert(impl->frame_arena.end(), bytes.begin(),
                               bytes.end());
      ep.push_back(rp);
    };

    switch (e.op) {
    case operation::reset:
      place(build_reset_frame(e.decoder_id, /*rid=*/0), 0, 0);
      break;
    case operation::get_corrections:
      place(build_get_corrections_frame(e.decoder_id, return_size_for(e),
                                        /*rid=*/0),
            0, 0);
      break;
    case operation::stream:
    case operation::enqueue_data: {
      if (e.source_id == kNoSource) {
        // Literal `source=0b...`: the same round, repeated. The parser has
        // already pinned the count, so every frame is known now.
        for (std::uint32_t r = 0; r < e.stream_min_rounds; ++r)
          place(build_enqueue_frame(e.decoder_id, /*rid=*/0,
                                    sched.syndrome_arena.data() +
                                        e.syndrome_offset,
                                    e.syndrome_count),
                e.syndrome_offset, e.syndrome_count);
        break;
      }

      auto &src = *sources.at(e.source_id);
      auto &ok = predrawable.try_emplace(e.source_id, true).first->second;
      // `enqueue_data` is always exactly one readout; a stream's count is
      // known only when nothing at run time can shorten it.
      const bool fixed =
          e.op == operation::enqueue_data || e.until_signal_id == kNoSignal;
      if (!fixed || src.is_streamed() || !ok) {
        ok = false; // and so is everything after it on this source
        break;
      }

      const std::uint32_t want =
          e.op == operation::enqueue_data ? 1u : e.stream_min_rounds;
      for (std::uint32_t r = 0; r < want; ++r) {
        auto round = e.op == operation::enqueue_data ? src.read_data()
                                                     : src.next_round();
        if (round.empty())
          break; // dry: dispatch reports SOURCE_EXHAUSTED for what is missing
        const auto off =
            static_cast<std::uint32_t>(sched.syndrome_arena.size());
        const auto len = static_cast<std::uint32_t>(round.size());
        sched.syndrome_arena.insert(sched.syndrome_arena.end(), round.begin(),
                                    round.end());
        if (r == 0) {
          e.syndrome_offset = off;
          e.syndrome_count = len;
        }
        place(build_enqueue_frame(e.decoder_id, /*rid=*/0,
                                  sched.syndrome_arena.data() + off, len),
              off, len);
      }
      break;
    }
    }
  }

  // -- No session may serve two decoder_ids
  {
    std::unordered_map<session *, std::uint64_t> owner;
    for (const auto &e : sched.events) {
      auto [it, inserted] =
          owner.emplace(router.at(e.decoder_id), e.decoder_id);
      if (!inserted && it->second != e.decoder_id)
        throw std::invalid_argument(
            "decoder_id=" + std::to_string(e.decoder_id) +
            " and decoder_id=" + std::to_string(it->second) +
            " share one session instance -- each decoder_id must have its "
            "own");
    }
  }

  // -- Upper bound on how many requests run() can issue, so it can size its
  // per-request logs once and append to them without a lock. reset/
  // get_corrections issue exactly one; a pre-built stream issues exactly as
  // many as it was given; anything else is bounded by stream_max_rounds
  // (the parser always sets this, defaulting an unbounded `until=` stream to
  // kDefaultUntilMaxRounds).
  std::uint64_t total_requests = 0;
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    const auto &e = sched.events[i];
    switch (e.op) {
    case operation::reset:
    case operation::get_corrections:
      total_requests += 1;
      break;
    case operation::stream:
    case operation::enqueue_data:
      total_requests += impl->event_plans[i].empty()
                            ? e.stream_max_rounds
                            : impl->event_plans[i].size();
      break;
    }
  }
  // n_requests (and every log it indexes) is uint32_t, so a schedule whose
  // worst-case request count doesn't fit would silently wrap and undersize
  // the logs -- reject it here instead.
  if (total_requests > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument(
        "schedule's worst-case request count (" +
        std::to_string(total_requests) +
        ") exceeds what a single run can "
        "index (" +
        std::to_string(std::numeric_limits<std::uint32_t>::max()) + ")");
  impl->max_requests = static_cast<std::uint32_t>(total_requests);

  return impl;
}

/// Trivial test-and-set spinlock. Guards `correction_log` and `warnings`,
/// which more than one session's worker may append to concurrently for a
/// multi-decoder schedule; hold times are sub-microsecond, so this costs
/// nothing in the common (uncontended, single-decoder) case.
class spinlock {
public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire))
      std::this_thread::yield();
  }
  void unlock() { flag_.clear(std::memory_order_release); }

private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

/// Per-event bookkeeping for the reply/event_done callbacks: touched only by
/// the one session worker that owns the event's decoder_id, so it needs no
/// synchronization of its own.
struct collector_entry {
  std::uint32_t collected = 0;
  std::uint32_t expected = 0; // 0 until on_event_done sets it
  bool has_term = false;
  std::int32_t term = 0;
  std::uint64_t last_return_ns = 0;
  RpcStatus last_status = RpcStatus::OK;
  bool any_error = false;
  bool completed = false;
};

/// Everything the timing thread and the session workers share, built on
/// run()'s stack.
struct run_state {
  std::uint32_t next_request_id = 1; // timing thread only
  std::uint32_t n_requests = 0;      // timing thread only; indexes the logs
  std::atomic<bool> aborted{false};

  spinlock logs_lock; // guards correction_log and warnings only

  /// One flag per signal name in the schedule, raised by a session worker
  /// once a `signal=` event's reply/acks are all collected, and read at a
  /// stream's round boundaries (`until=`) or before an `after=` dispatch.
  std::vector<std::atomic<bool>> signals;

  /// One entry per event, indexed the same way as `run_result::records`.
  std::vector<collector_entry> collectors;

  bool signal_raised(std::uint32_t id) const {
    return signals[id].load(std::memory_order_acquire);
  }
  void raise_signal(std::uint32_t id) {
    signals[id].store(true, std::memory_order_release);
  }
};

/// The four things every step of a run needs
struct run_ctx {
  run_plan &plan;
  run_result &result;
  run_state &st;
  std::uint64_t t0;

  const event &ev(std::uint32_t i) const { return plan.sched.events[i]; }
  record &rec(std::uint32_t i) const { return result.records[i]; }
};

/// Appends one line to `result.warnings`.
void warn(const run_ctx &c, const std::string &message) {
  std::lock_guard<spinlock> lock(c.st.logs_lock);
  c.result.warnings.push_back(message);
}

/// Describes an event the way a warning should name it.
std::string event_label(const run_ctx &c, std::uint32_t i) {
  return "event " + std::to_string(i) + " (" + to_string(c.ev(i).op) +
         ", decoder_id=" + std::to_string(c.ev(i).decoder_id) + ")";
}

/// Flips `aborted` from false to true; returns whether this call did it, so
/// concurrent failures racing the same run (e.g. several already-dispatched
/// requests all timing out against one dead endpoint) log only one warning.
bool try_abort(run_state &st) {
  bool expected = false;
  return st.aborted.compare_exchange_strong(expected, true,
                                            std::memory_order_relaxed);
}

/// Records the warning and flips `aborted` so the timing thread stops
/// dispatching further events.
void abort_on_hard_error(const run_ctx &c, RpcStatus status, std::uint32_t i) {
  if (status == RpcStatus::OK || status == RpcStatus::NOT_READY)
    return;
  if (try_abort(c.st))
    warn(c, event_label(c, i) + " returned status " +
                std::to_string(static_cast<int>(status)) +
                "; aborting the run");
}

/// One request's slot in the per-request logs.
struct issued_request {
  std::uint32_t rid;
  std::uint32_t log_index;
};

/// Take the next request_id and open its per-request log slots (pre-sized to
/// `run_plan::max_requests`, so this is a plain indexed write -- the timing
/// thread is their only writer). For a stream round, also records the bits
/// it is about to send. Every RPC the run puts on the wire goes through here.
issued_request begin_request(const run_ctx &c, record &rec,
                             const std::uint8_t *bits = nullptr,
                             std::size_t n_bits = 0, bool first_round = false) {
  const std::uint32_t rid = c.st.next_request_id++;
  const std::uint32_t idx = c.st.n_requests++;
  if (rec.request_id_count == 0)
    rec.request_id_offset = idx;
  c.result.request_id_log[idx] = rid;
  ++rec.request_id_count;
  if (bits) {
    if (first_round)
      rec.syndrome_offset =
          static_cast<std::uint32_t>(c.result.syndrome_log.size());
    c.result.syndrome_log.insert(c.result.syndrome_log.end(), bits,
                                 bits + n_bits);
    rec.syndrome_count += static_cast<std::uint32_t>(n_bits);
  }
  return {rid, idx};
}

/// Timing-thread-only element write, right before the frame is sent on the wire
void stamp_dispatch(const run_ctx &c, std::uint32_t log_index) {
  c.result.request_dispatch_ns_log[log_index] = now_ns() - c.t0;
}

/// Settles one event's record once its reply count has reached what
/// on_event_done said to expect. Idempotent, though by construction each
/// caller only reaches this once, right when collected == expected.
void settle(const run_ctx &c, std::uint32_t event_index, collector_entry &e) {
  if (e.completed)
    return;
  e.completed = true;
  record &rec = c.rec(event_index);
  rec.return_ns = e.collected ? e.last_return_ns : now_ns() - c.t0;
  rec.status =
      e.has_term
          ? (e.any_error ? static_cast<std::int32_t>(stream_terminate::ERROR)
                         : e.term)
          : static_cast<std::int32_t>(e.last_status);
  const auto signal_id = c.ev(event_index).signal_id;
  if (signal_id != kNoSignal)
    c.st.raise_signal(signal_id);
}

/// Declared in session.h; called directly by whichever thread the owning
/// session completes replies on. Stamps the per-request log, folds the
/// reply into the event's record (unpacking a get_corrections payload), and
/// settles the event once every reply it is owed has arrived.
void handle_reply(run_ctx &c, tag t, RpcStatus status,
                  const std::uint8_t *reply, std::size_t reply_len,
                  std::uint64_t return_ns) {
  const std::uint32_t i = t.event;
  const event &e = c.ev(i);
  RpcStatus final_status = status;

  if (e.op == operation::get_corrections) {
    const std::uint32_t return_size = return_size_for(e);
    final_status = reject_truncated_reply(status, reply_len, return_size);
    if (final_status == RpcStatus::OK) {
      record &rec = c.rec(i);
      rec.read_completed = true;
      rec.correction_count = return_size;
      // Unpack and compare in a local buffer: another decoder's thread may
      // append to correction_log (reallocating it) at any time, so the
      // shared vector can only be touched under logs_lock
      std::vector<std::uint8_t> unpacked;
      unpacked.reserve(return_size);
      append_unpacked_bits(unpacked, reply, return_size);
      rec.correction_mismatch =
          mismatches_expected(c.plan.sched, e, unpacked.data(), return_size);
      {
        std::lock_guard<spinlock> lock(c.st.logs_lock);
        rec.correction_offset =
            static_cast<std::uint32_t>(c.result.correction_log.size());
        c.result.correction_log.insert(c.result.correction_log.end(),
                                       unpacked.begin(), unpacked.end());
      }
    }
  }

  c.result.request_return_ns_log[t.log_index] = return_ns - c.t0;
  c.result.request_status_log[t.log_index] =
      static_cast<std::int32_t>(final_status);

  switch (e.op) {
  case operation::reset:
  case operation::get_corrections:
    abort_on_hard_error(c, final_status, i);
    break;
  case operation::stream:
  case operation::enqueue_data:
    if (c.plan.params.collect_enqueue_acks)
      abort_on_hard_error(c, final_status, i);
    break;
  }

  collector_entry &col = c.st.collectors[i];
  col.last_return_ns = std::max(col.last_return_ns, return_ns - c.t0);
  col.last_status = final_status;
  col.any_error |=
      final_status != RpcStatus::OK && final_status != RpcStatus::NOT_READY;
  ++col.collected;
  if (col.expected != 0 && col.collected == col.expected)
    settle(c, i, col);
}

/// Declared in session.h; called directly by the owning session. The event
/// named by `event` has issued its last request.
void handle_event_done(run_ctx &c, std::uint32_t event, std::uint32_t issued,
                       std::int32_t term, bool has_term) {
  collector_entry &col = c.st.collectors[event];
  col.expected = issued;
  col.has_term = has_term;
  col.term = term;
  if (col.collected == col.expected)
    settle(c, event, col);
}

/// One `stream` or `enqueue_data` event: draw a round, send it, decide
/// whether to send another, until a terminal status is reached; `rec` and
/// `result` are updated in place. Both ops share this loop since they are the
/// same wire operation; `enqueue_data` just pulls a data readout with 1/1
/// bounds.
void run_stream(const run_ctx &c, std::uint32_t i,
                std::uint64_t deadline_abs_ns, session &s) {
  auto &plan = c.plan;
  auto &st = c.st;
  const event &e = c.ev(i);
  const event_plan &ep = c.plan.event_plans[i];
  record &rec = c.rec(i);
  // Either every round was drawn and serialized by plan(), or none was and
  // this loop draws them as it goes -- plan() never leaves half a stream
  // pre-built, so there is no third case to reconcile here.
  const bool prebuilt = !ep.empty();
  syndrome_source *source = prebuilt ? nullptr : plan.sources.at(e.source_id);

  std::uint64_t next = deadline_abs_ns;
  std::uint32_t rounds = 0;
  stream_terminate term = stream_terminate::OK;
  for (;;) {
    if (st.aborted.load(std::memory_order_relaxed)) {
      term = stream_terminate::ERROR;
      break;
    }
    if (rounds >= e.stream_max_rounds) {
      term = stream_terminate::EXHAUSTED_ROUNDS;
      break;
    }

    // Pre-built rounds run out only when the source went dry while plan()
    // was drawing them, which is the same exhaustion a run-time draw sees.
    if (prebuilt && rounds >= ep.size()) {
      term = stream_terminate::SOURCE_EXHAUSTED;
      break;
    }
    std::vector<std::uint8_t> drawn;
    std::vector<std::uint8_t> built;
    if (!prebuilt) {
      try {
        drawn = e.op == operation::enqueue_data ? source->read_data()
                                                : source->next_round();
      } catch (const std::exception &ex) {
        term = stream_terminate::ERROR;
        if (try_abort(st))
          warn(c, event_label(c, i) + " threw while drawing from its source: " +
                      ex.what() + "; aborting the run");
        break;
      }
      if (drawn.empty()) {
        term = stream_terminate::SOURCE_EXHAUSTED;
        break;
      }
      built = build_enqueue_frame(e.decoder_id, 0, drawn.data(), drawn.size());
      if (s.max_frame_bytes != 0 && built.size() > s.max_frame_bytes) {
        term = stream_terminate::ERROR;
        if (try_abort(st))
          warn(c, event_label(c, i) +
                      " drew a round exceeding the session's max_frame_bytes; "
                      "aborting the run");
        break;
      }
    }
    if (e.stream_every_ticks > 0) {
      wait_until(next);
      next += e.stream_every_ticks * plan.sched.tick_ns;
    }

    // Where this round's bits live and how its frame is produced are the
    // only things the two modes disagree on.
    const std::uint8_t *bits =
        prebuilt ? plan.sched.syndrome_arena.data() + ep[rounds].bits_offset
                 : drawn.data();
    const std::size_t n_bits = prebuilt ? ep[rounds].bits_count : drawn.size();
    const auto [rid, log_index] =
        begin_request(c, rec, bits, n_bits, rounds == 0);
    const tag t{i, log_index};
    stamp_dispatch(c, log_index);
    if (prebuilt) {
      const round_plan &rp = ep[rounds];
      set_request_id(plan.frame_arena.data() + rp.frame_offset, rid);
      s.send(frame_of(plan, rp), t);
    } else {
      set_request_id(built.data(), rid);
      s.send({built.data(), built.size()}, t);
    }
    ++rounds;

    if (rounds < e.stream_min_rounds)
      continue; // below the floor: nothing can stop this stream yet

    if (e.until_signal_id == kNoSignal || st.signal_raised(e.until_signal_id))
      break;

    // Unpaced (every==0): this is the loop's only yield point, and the only
    // thing capping how far ahead of the worker it can get. Paced (every>0)
    // doesn't need this -- wait_until() above already spaced the rounds out.
    if (e.stream_every_ticks == 0)
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(kUnpacedStreamFloorNs));
  }

  rec.rounds_streamed = rounds;
  s.event_done(i, rounds, static_cast<std::int32_t>(term), /*has_term=*/true);
}

/// One event, dispatched on run()'s single timing thread. `prev_return_ns`
/// is when the previous event finished being dispatched here. Every op
/// sends and returns without waiting -- the session's own worker processes
/// the reply/acks and owns writing `rec.return_ns`/`rec.status`.
void dispatch_event(const run_ctx &c, std::uint32_t i, session &s,
                    std::uint64_t &prev_return_ns) {
  auto &plan = c.plan;
  auto &st = c.st;
  const auto &e = c.ev(i);
  const auto &ep = plan.event_plans[i];
  auto &rec = c.rec(i);
  const std::uint64_t t0 = c.t0;

  const std::uint64_t deadline_ns =
      e.trig == trigger::tick ? e.deadline_ns
                              : add_sat(prev_return_ns, e.deadline_ns);
  rec.deadline_ns = deadline_ns;

  wait_until(add_sat(t0, deadline_ns));
  if (e.after_signal_id != kNoSignal) {
    // Busy-spin for the first 2ms (the common case: the signal is already
    // raised or arrives almost immediately), then fall back to yielding
    // between checks -- never sleep_for, which would blow well past a
    // signal that arrives right after a check.
    const auto spin_deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    while (!st.signal_raised(e.after_signal_id)) {
      if (st.aborted.load(std::memory_order_relaxed))
        return; // aborted while waiting on `after=`; never dispatched
      if (std::chrono::steady_clock::now() >= spin_deadline)
        std::this_thread::yield();
    }
  }
  rec.call_ns = now_ns() - t0;
  rec.dispatched = true;

  switch (e.op) {
  case operation::reset:
  case operation::get_corrections: {
    const auto &rp = ep[0];
    const auto [rid, log_index] = begin_request(c, rec);
    set_request_id(plan.frame_arena.data() + rp.frame_offset, rid);
    stamp_dispatch(c, log_index);
    s.send(frame_of(plan, rp), {i, log_index});
    s.event_done(i, /*issued=*/1, /*term=*/0, /*has_term=*/false);
    break;
  }
  case operation::stream:
  case operation::enqueue_data:
    // Same wire operation, same loop -- see run_stream.
    run_stream(c, i, add_sat(t0, deadline_ns), s);
    break;
  }

  prev_return_ns = now_ns() - t0; // the timeline is free either way
}

run_result run(std::shared_ptr<run_plan> p) {
  auto &plan = *p;
  auto &sched = plan.sched;

  run_result result;
  result.records.resize(sched.events.size());
  // Every event gets an identified slot up front, dispatched or not
  // record::dispatched is how a caller tells "ran" from
  // "pre-empted by the abort."
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    auto &rec = result.records[i];
    rec.event_index = static_cast<std::uint32_t>(i);
    rec.decoder_id = sched.events[i].decoder_id;
    rec.op = sched.events[i].op;
  }
  result.syndrome_log.reserve(sched.syndrome_arena.size());
  result.correction_log.reserve(sched.expected_arena.size());
  // Sized once, before t0, to plan()'s upper bound: the timing thread then
  // appends request_id/dispatch_ns by index, and each session's worker
  // writes its own request's return_ns/status by index, with no lock and no
  // reallocation racing either side.
  result.request_id_log.resize(plan.max_requests);
  result.request_dispatch_ns_log.resize(plan.max_requests, 0);
  result.request_return_ns_log.resize(plan.max_requests, 0);
  result.request_status_log.resize(plan.max_requests, kNoStatus);

  const std::uint64_t t0 = now_ns() + plan.params.lead_in_ns;

  run_state st;
  st.signals = std::vector<std::atomic<bool>>(sched.signal_names.size());
  st.collectors.resize(sched.events.size());

  run_ctx c{plan, result, st, t0};

  // One session per decoder present in the schedule; start every worker
  // during the lead-in, so thread creation never eats into the timed run.
  std::vector<std::uint64_t> decoder_ids;
  for (const auto &e : sched.events) {
    if (std::find(decoder_ids.begin(), decoder_ids.end(), e.decoder_id) !=
        decoder_ids.end())
      continue;
    decoder_ids.push_back(e.decoder_id);
    plan.router.at(e.decoder_id)->start(c);
  }

  wait_until(t0);

  // The whole dispatch model: one thread, schedule order.
  std::uint64_t prev_return_ns = 0;
  for (std::size_t i = 0;
       i < sched.events.size() && !st.aborted.load(std::memory_order_relaxed);
       ++i)
    dispatch_event(c, static_cast<std::uint32_t>(i),
                   *plan.router.at(sched.events[i].decoder_id), prev_return_ns);

  // Dispatch is done, so no more requests can be issued; stop every session
  // (each drains what is still in flight through on_reply) before anybody
  // looks at the records they own.
  const std::chrono::nanoseconds ack_drain_timeout(
      plan.params.ack_drain_timeout_ns);
  for (auto decoder_id : decoder_ids)
    plan.router.at(decoder_id)->stop(ack_drain_timeout);

  // A dispatched event whose collector never settled -- only reachable if a
  // session failed to deliver every reply/event_done it owed -- keeps its
  // default record, same as an event the abort pre-empted before dispatch.
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    if (!result.records[i].dispatched)
      continue;
    const collector_entry &col = st.collectors[i];
    if (col.completed)
      continue;
    if (col.collected < col.expected)
      warn(c, event_label(c, static_cast<std::uint32_t>(i)) + ": " +
                  std::to_string(col.expected - col.collected) +
                  " request(s) never got a reply; giving up on them");
  }

  result.request_id_log.resize(st.n_requests);
  result.request_dispatch_ns_log.resize(st.n_requests);
  result.request_return_ns_log.resize(st.n_requests);
  result.request_status_log.resize(st.n_requests);

  result.t0_ns = t0;
  result.tick_ns = sched.tick_ns;
  return result;
}

// -- Downstream analysis

namespace {

// syndrome_log/correction_log are one byte (0x00/0x01) per BIT --
// render each one as a '0'/'1' character, in log order.
std::string bits_to_string(const std::uint8_t *bits, std::size_t count) {
  std::string out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    out.push_back(bits[i] ? '1' : '0');
  return out;
}

/// Bounds-checks (offset, count) against `arena` and returns how many bits
/// are actually safe to read
std::pair<const std::uint8_t *, std::size_t>
safe_bit_span(const std::vector<std::uint8_t> &arena, std::uint32_t offset,
              std::uint32_t count) {
  if (arena.empty() || offset >= arena.size())
    return {nullptr, 0};
  return {arena.data() + offset,
          std::min<std::size_t>(count, arena.size() - offset)};
}

/// One record's slice of a per-request log as a single space-separated cell:
/// variable-length like the bit columns, and free of commas so it needs no
/// CSV quoting. Bounds-checked the same way, so a record pointing past the
/// log renders empty rather than reading off the end.
template <typename T>
std::string join_log(const std::vector<T> &log, std::uint32_t offset,
                     std::uint32_t count) {
  if (log.empty() || offset >= log.size())
    return {};
  const std::size_t n = std::min<std::size_t>(count, log.size() - offset);
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0)
      out.push_back(' ');
    out += std::to_string(log[offset + i]);
  }
  return out;
}

} // namespace

void write_csv(const run_result &result, std::ostream &out) {
  out << "event_index,decoder_id,op,deadline_ns,call_ns,return_ns,"
         "status,rounds_streamed,read_completed,"
         "syndrome_bits,correction_bits,correction_mismatch,request_ids,"
         "dispatched,request_dispatch_ns,request_return_ns\n";
  for (const auto &r : result.records) {
    const auto [syndrome_bits, syndrome_n] =
        safe_bit_span(result.syndrome_log, r.syndrome_offset, r.syndrome_count);
    const auto [correction_bits, correction_n] = safe_bit_span(
        result.correction_log, r.correction_offset, r.correction_count);
    out << r.event_index << ',' << r.decoder_id << ',' << to_string(r.op) << ','
        << r.deadline_ns << ',' << r.call_ns << ',' << r.return_ns << ','
        << r.status << ',' << r.rounds_streamed << ','
        << (r.read_completed ? 1 : 0) << ','
        << bits_to_string(syndrome_bits, syndrome_n) << ','
        << bits_to_string(correction_bits, correction_n) << ','
        << (r.correction_mismatch ? 1 : 0) << ','
        << join_log(result.request_id_log, r.request_id_offset,
                    r.request_id_count)
        << ',' << (r.dispatched ? 1 : 0) << ','
        << join_log(result.request_dispatch_ns_log, r.request_id_offset,
                    r.request_id_count)
        << ','
        << join_log(result.request_return_ns_log, r.request_id_offset,
                    r.request_id_count)
        << '\n';
  }
}

std::string write_csv(const run_result &result) {
  std::ostringstream oss;
  write_csv(result, oss);
  return oss.str();
}

} // namespace cudaq::qec::playback
