/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Tests the emulator core: plan()/run() orchestration and everything that
/// rides on it -- CSV serialization, capability validation, the timing loop,
/// signal-driven unblocking, request_id bookkeeping, hard-error abort
/// semantics, and the truncated-reply failure mode.

#include "RpcSlot.h"
#include "emulator.h"
#include "session.h"
#include "syndrome_source.h"
#include "test_session_fakes.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <gtest/gtest.h>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_set>

using namespace cudaq::qec::playback;

// ─── Analyzer ───────────────────────────────────────────────────────────────
//
// Tests that write_csv() is a plain, lossless serialization of a run_result

namespace {
run_result make_sample_result() {
  run_result r;
  record rec;
  rec.event_index = 0;
  rec.decoder_id = 3;
  rec.op = operation::get_corrections;
  rec.deadline_ns = 1000;
  rec.call_ns = 1200;
  rec.return_ns = 1500;
  rec.status = 0;
  rec.rounds_streamed = 0;
  rec.read_completed = true;
  rec.correction_mismatch = false;
  rec.request_id_offset = 0;
  rec.request_id_count = 1;
  r.request_id_log = {7};
  r.request_dispatch_ns_log.push_back(1100);
  r.request_return_ns_log.push_back(1500);
  r.request_status_log.push_back(0);
  r.records.push_back(rec);
  return r;
}

std::vector<std::string> lines_of(const std::string &csv) {
  std::vector<std::string> lines;
  std::istringstream iss(csv);
  for (std::string line; std::getline(iss, line);)
    lines.push_back(line);
  return lines;
}

std::vector<std::string> columns_of(const std::string &line) {
  std::vector<std::string> cols;
  std::istringstream iss(line);
  for (std::string cur; std::getline(iss, cur, ',');)
    cols.push_back(cur);
  return cols;
}
} // namespace

TEST(Analyzer, AnEmptyResultProducesTheHeaderAndNothingElse) {
  auto csv = write_csv(run_result{});
  ASSERT_FALSE(csv.empty());
  EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), 1);
  const auto header = columns_of(lines_of(csv)[0]);
  for (const char *col : {"event_index", "decoder_id", "op", "deadline_ns",
                          "call_ns", "return_ns", "status", "rounds_streamed",
                          "read_completed", "syndrome_bits", "correction_bits",
                          "correction_mismatch", "request_ids", "dispatched",
                          "request_dispatch_ns", "request_return_ns"})
    EXPECT_NE(std::find(header.begin(), header.end(), col), header.end())
        << "missing column " << col;
}

TEST(Analyzer, RecordFieldsSurviveIntoTheCsvRow) {
  auto result = make_sample_result();
  const auto csv = write_csv(result);
  const auto lines = lines_of(csv);
  ASSERT_EQ(lines.size(), 2u);

  const auto header = columns_of(lines[0]);
  const auto row = columns_of(lines[1]);
  ASSERT_EQ(row.size(), header.size());
  const auto column = [&](const char *name) {
    const auto it = std::find(header.begin(), header.end(), name);
    EXPECT_NE(it, header.end()) << name;
    return row[static_cast<std::size_t>(std::distance(header.begin(), it))];
  };
  EXPECT_EQ(column("decoder_id"), "3");
  EXPECT_EQ(column("op"), "get_corrections");
  EXPECT_EQ(column("request_ids"), "7");

  // The ostream overload has to agree with the string one exactly, or a
  // report written to a file would differ from the same report in memory.
  std::ostringstream oss;
  write_csv(result, oss);
  EXPECT_EQ(oss.str(), csv);
}

// ─── AnalyzerEdgeCases ────────────────────────────────────────────────────
//
// Adversarial tests for write_csv(), focused on the bounds-checking
// `safe_bit_span` helper in lib/emulator.cpp and on exact
// output for every enum value, boolean combination, and
// bit-string encoding.

namespace {

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> cols;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      cols.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  cols.push_back(cur);
  return cols;
}

std::vector<std::string> csv_lines(const std::string &csv) {
  std::vector<std::string> lines;
  std::istringstream iss(csv);
  std::string line;
  while (std::getline(iss, line))
    lines.push_back(line);
  return lines;
}

constexpr std::size_t kNumColumns = 16;

// Columns these tests assert on, by position in write_csv()'s header.
constexpr std::size_t kColOp = 2;
constexpr std::size_t kColStatus = 6;
constexpr std::size_t kColReadCompleted = 8;
constexpr std::size_t kColSyndromeBits = 9;
constexpr std::size_t kColCorrectionBits = 10;
constexpr std::size_t kColMismatch = 11;
constexpr std::size_t kColRequestIds = 12;

/// The data columns of the one row `r` produces.
std::vector<std::string> only_row(const run_result &r) {
  auto lines = csv_lines(write_csv(r));
  EXPECT_EQ(lines.size(), 2u);
  return split_csv_line(lines[1]);
}

} // namespace

// -- malformed / out-of-range record bounds -------------------------------

TEST(AnalyzerEdgeCases,
     EveryOutOfRangeBitSpanClampsInsteadOfReadingPastTheArena) {
  // safe_bit_span's guard is `offset >= arena.size()`, so an offset one past
  // the last valid index has to clamp to nothing while one *at* the last
  // index still reads it -- the off-by-one either way is a read past the end
  // of the arena, which is the whole reason the helper exists.
  struct {
    std::vector<std::uint8_t> log;
    std::uint32_t offset, count;
    const char *expected;
    const char *why;
  } cases[] = {
      {{0x01, 0x00}, 1000, 8, "", "offset far past a short arena"},
      {{}, 5, 16, "", "any offset into an empty arena"},
      {{1, 0, 1, 1}, 4, 8, "", "offset exactly at the end"},
      {{1, 0, 1, 1}, 3, 8, "1", "offset at the last index, count clamped to 1"},
      {{1, 1}, 999999, 0, "", "zero count, offset out of range"},
      {{1, 0, 1, 1}, 1, 0, "", "zero count, offset in range"},
  };
  // Both bit-string columns index their own arena through the same helper,
  // so each case is run through both rather than trusting one to stand for
  // the other.
  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    run_result syn;
    record srec;
    srec.syndrome_offset = c.offset;
    srec.syndrome_count = c.count;
    syn.syndrome_log = c.log;
    syn.records.push_back(srec);
    EXPECT_EQ(only_row(syn)[kColSyndromeBits], c.expected) << "syndrome_bits";

    run_result cor;
    record crec;
    crec.correction_offset = c.offset;
    crec.correction_count = c.count;
    cor.correction_log = c.log;
    cor.records.push_back(crec);
    EXPECT_EQ(only_row(cor)[kColCorrectionBits], c.expected)
        << "correction_bits";
  }
}

TEST(AnalyzerEdgeCases,
     RequestIdSliceRunningPastTheLogClampsInsteadOfReadingOOB) {
  // The request_ids column indexes its own log the same way the bit-string
  // columns index theirs, so it needs the same guard: a count reaching past the
  // end renders what is actually there, and an offset past the end renders
  // nothing.
  run_result r;
  record over_long, past_end;
  over_long.request_id_offset = 1;
  over_long.request_id_count = 9; // only 2 ids left after the offset
  past_end.request_id_offset = 3; // == log.size()
  past_end.request_id_count = 4;
  r.request_id_log = {11, 12, 13};
  r.records = {over_long, past_end};
  auto lines = csv_lines(write_csv(r));
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(split_csv_line(lines[1])[kColRequestIds], "12 13");
  EXPECT_EQ(split_csv_line(lines[2])[kColRequestIds], "");
}

TEST(AnalyzerEdgeCases, SyndromeAndCorrectionBitsAreIndependentInTheSameRow) {
  run_result r;
  record rec;
  rec.syndrome_offset = 0;
  rec.syndrome_count = 4;
  rec.correction_offset = 2; // deliberately non-zero, mid-arena
  rec.correction_count = 4;
  r.syndrome_log = {1, 1, 0, 0};
  r.correction_log = {0, 0, 1, 1, 0, 1}; // bits [2..5] = 1,1,0,1
  r.records.push_back(rec);
  auto cols = only_row(r);
  EXPECT_EQ(cols[kColSyndromeBits], "1100");
  EXPECT_EQ(cols[kColCorrectionBits], "1101");
}

// -- enum coverage ----------------------------------------------------------

TEST(AnalyzerEdgeCases, EveryEnumValueSurvivesIntoItsColumn) {
  // `op` is stringified through to_string(); `status` is not -- it is a raw
  // numeric column holding either an RpcStatus or a stream_terminate (the
  // two ranges are disjoint, see types.h), so what matters is that the exact
  // number survives.
  struct {
    operation op;
    const char *op_name;
    std::int32_t status;
  } cases[] = {
      {operation::reset, "reset", 0},                       // RpcStatus::OK
      {operation::get_corrections, "get_corrections", 2},   // BAD_REQUEST
      {operation::get_corrections, "get_corrections", 4},   // NOT_READY
      {operation::enqueue_data, "enqueue_data", kNoStatus}, // never dispatched
      {operation::stream, "stream",
       static_cast<std::int32_t>(stream_terminate::OK)},
      {operation::stream, "stream",
       static_cast<std::int32_t>(stream_terminate::SOURCE_EXHAUSTED)},
      {operation::stream, "stream",
       static_cast<std::int32_t>(stream_terminate::EXHAUSTED_ROUNDS)},
      {operation::stream, "stream",
       static_cast<std::int32_t>(stream_terminate::ERROR)},
  };
  // The four stream_terminate values must stay disjoint from RpcStatus, or
  // a reader cannot tell which enum a status column holds.
  EXPECT_EQ(static_cast<std::int32_t>(stream_terminate::OK), 100);
  EXPECT_EQ(static_cast<std::int32_t>(stream_terminate::ERROR), 103);

  run_result r;
  for (const auto &c : cases) {
    record rec;
    rec.op = c.op;
    rec.status = c.status;
    r.records.push_back(rec);
  }
  auto lines = csv_lines(write_csv(r));
  ASSERT_EQ(lines.size(), std::size(cases) + 1);
  for (std::size_t i = 0; i < std::size(cases); ++i) {
    auto cols = split_csv_line(lines[i + 1]);
    EXPECT_EQ(cols[kColOp], cases[i].op_name) << "row " << i;
    EXPECT_EQ(cols[kColStatus], std::to_string(cases[i].status)) << "row " << i;
  }
}

// -- boolean 0/1 encoding ----------------------------------------------------

TEST(AnalyzerEdgeCases,
     ReadCompletedAndCorrectionMismatchEncodeExactlyAsZeroOrOne) {
  run_result r;
  for (bool read_completed : {false, true})
    for (bool mismatch : {false, true}) {
      record rec;
      rec.read_completed = read_completed;
      rec.correction_mismatch = mismatch;
      r.records.push_back(rec);
    }
  auto lines = csv_lines(write_csv(r));
  ASSERT_EQ(lines.size(), 5u);
  const std::pair<const char *, const char *> expected[] = {
      {"0", "0"}, {"0", "1"}, {"1", "0"}, {"1", "1"}};
  for (std::size_t i = 0; i < 4; ++i) {
    auto cols = split_csv_line(lines[i + 1]);
    EXPECT_EQ(cols[kColReadCompleted], expected[i].first) << "row " << i;
    EXPECT_EQ(cols[kColMismatch], expected[i].second) << "row " << i;
  }
}

// -- bit-string encoding ------------------------------------------------------

TEST(AnalyzerEdgeCases, BitStringEncodingIsExactAndKeepsBitOrder) {
  // One '0'/'1' character per bit, in log order -- no packing, so no
  // boundary to get wrong, but order still has to survive.
  const std::vector<std::uint8_t> alternating = {1, 0, 1, 0, 1, 0, 1, 0};
  const std::vector<std::uint8_t> sparse_nine = {1, 0, 0, 0, 0, 0, 0, 0, 1};
  struct {
    std::vector<std::uint8_t> bits;
    const char *expected;
  } cases[] = {
      {{}, ""},
      {{1}, "1"},
      {std::vector<std::uint8_t>(7, 1), "1111111"},
      {std::vector<std::uint8_t>(8, 1), "11111111"},
      {alternating, "10101010"},
      {sparse_nine, "100000001"},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE("count=" + std::to_string(c.bits.size()));
    run_result r;
    record rec;
    rec.syndrome_count = static_cast<std::uint32_t>(c.bits.size());
    r.syndrome_log = c.bits;
    r.records.push_back(rec);
    EXPECT_EQ(only_row(r)[kColSyndromeBits], c.expected);
  }
}

// -- maximally-degenerate result ---------------------------------------------

TEST(AnalyzerEdgeCases, ADefaultResultWithWarningsProducesOneWellFormedRow) {
  // Warnings are not part of the CSV schema: they travel on run_result for a
  // caller to report, and must not leak into a row or shift the columns.
  run_result r;
  r.warnings = {"foo", "bar"};
  r.records.push_back(record{}); // every field at its default
  auto csv = write_csv(r);
  EXPECT_EQ(csv.find("foo"), std::string::npos);
  EXPECT_EQ(csv.find("bar"), std::string::npos);
  auto cols = only_row(r);
  ASSERT_EQ(cols.size(), kNumColumns);
  EXPECT_EQ(cols[kColOp], "reset"); // operation{} == reset
  EXPECT_EQ(cols[kColSyndromeBits], "");
  EXPECT_EQ(cols[kColCorrectionBits], "");
  EXPECT_EQ(cols[kColRequestIds], "");
}

// -- scale: many records, no crash, no accidental quadratic blowup ----------

TEST(AnalyzerEdgeCases, ManyRecordsProduceExactlyOneRowEachAndFinishQuickly) {
  run_result r;
  constexpr int kCount = 750;
  r.syndrome_log.assign(64, 1);
  r.correction_log.assign(64, 1);
  const operation ops[] = {operation::reset, operation::stream,
                           operation::get_corrections, operation::enqueue_data};
  for (int i = 0; i < kCount; ++i) {
    record rec;
    rec.event_index = static_cast<std::uint32_t>(i);
    rec.op = ops[i % 4];
    rec.syndrome_count = 32;
    rec.correction_count = 32;
    r.records.push_back(rec);
  }
  auto start = std::chrono::steady_clock::now();
  auto csv = write_csv(r);
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::seconds(2));
  EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'),
            static_cast<long>(kCount + 1));
}

// ─── Capabilities ───────────────────────────────────────────────────────────
//
// Tests that anything plan() can prove will not work is a startup error,
// thrown before t0, rather than a runtime surprise: a frame no session can
// carry, a source nothing registered, or two decoders sharing one session.

namespace {
struct FakeSession : blocking_session {
  RpcStatus send_sync(const frame &, std::span<std::uint8_t>,
                      std::size_t &reply_len) override {
    reply_len = 0;
    return RpcStatus::OK;
  }
};

/// A syndrome_source whose rounds grow: round i is (i+1) bits wide, all
/// zero. Used to force a streamed round past a frame-size limit at run
/// time, since that width is unknowable at plan() time for a JIT source.
struct GrowingSource : syndrome_source {
  uint32_t next_width = 1;
  std::vector<uint8_t> next_round() override {
    return std::vector<uint8_t>(next_width++, 0);
  }
  bool is_streamed() const override { return true; }
};
} // namespace

TEST(Capabilities, PlanRefusesEverythingItCanProveWillNotWork) {
  // Two decoder_ids sharing a session instance would interleave frames for
  // different decoders on one socket and match each other's replies, so it
  // is refused
  FakeSession small, unbounded, shared;
  small.max_frame_bytes = 16; // the header alone is 24 bytes

  struct {
    const char *text;
    std::vector<std::uint64_t> ids;
    std::unordered_map<std::uint64_t, session *> router;
    const char *why;
  } cases[] = {
      {"0 reset\n", {0}, {{0, &small}}, "no frame fits the session's limit"},
      {"0 enqueue source=7\n",
       {0},
       {{0, &unbounded}},
       "no source registered for an enqueue"},
      {"0 stream source=7 rounds=3\n",
       {0},
       {{0, &unbounded}},
       "no source registered for a stream"},
      {"0 reset\n0 reset session=1\n",
       {0, 1},
       {{0, &shared}, {1, &shared}},
       "two decoders with events on one session"},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    auto sched = parse(c.text, c.ids, 1000);
    EXPECT_THROW(plan(sched, c.router, {}), std::invalid_argument);
  }
}

TEST(Capabilities, PlanAcceptsWhatItCannotProveWrong) {
  FakeSession a, b;
  a.max_frame_bytes = 0; // unbounded

  // Within every limit.
  auto sized = parse("0 enqueue source=0b1010\n", {0}, 1000);
  std::unordered_map<std::uint64_t, session *> one{{0, &a}};
  EXPECT_NO_THROW(plan(sized, one, {}));

  // Two decoders, each with its own session.
  auto two_decoders = parse("0 reset\n0 reset session=1\n", {0, 1}, 1000);
  std::unordered_map<std::uint64_t, session *> distinct{{0, &a}, {1, &b}};
  EXPECT_NO_THROW(plan(two_decoders, distinct, {}));

  // Sharing is refused only when both decoder_ids have events: one that is
  // routed but never named by a schedule line sends nothing and can't
  // collide with anybody.
  auto one_decoder = parse("0 reset\n", {0, 1}, 1000);
  std::unordered_map<std::uint64_t, session *> shared{{0, &a}, {1, &a}};
  EXPECT_NO_THROW(plan(one_decoder, shared, {}));
}

TEST(Capabilities, AStreamedRoundTooBigForItsSessionFailsLoudlyAtRunTime) {
  // A JIT source's round width is not knowable until it is asked, so this
  // one cannot be caught in plan() -- but it must still terminate with
  // ERROR rather than silently drop the oversized round.
  FakeSession s;
  // An enqueue frame is 24B header + 32B payload + packed bits, so a limit
  // below 56 bytes rejects even round 1's near-empty frame.
  s.max_frame_bytes = 40;
  GrowingSource src;
  auto sched = parse("0 stream source=0 every=0 rounds=100\n", {0}, 1000);
  std::unordered_map<std::uint64_t, session *> router{{0, &s}};
  auto result = run(plan(sched, router, {{0, &src}}));

  ASSERT_EQ(result.records.size(), 1u);
  EXPECT_EQ(result.records[0].status,
            static_cast<std::int32_t>(stream_terminate::ERROR));
  EXPECT_EQ(result.records[0].rounds_streamed, 0u);
}

// ─── Timing ─────────────────────────────────────────────────────────────────
//
// Tests the timing core, mostly against the `null` backend (the jitter
// floor). No decoder involved -- this is purely about whether run() hits
// deadlines, honours the "never rewrite the schedule" overrun policy, and
// keeps its timestamps sane at the edges of lead-in and tick width.

namespace {

std::uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

/// A session whose first call blocks the *caller* for `delay` before
/// answering OK, simulating a dispatch so slow it overruns the next event's
/// deadline -- a session's `send` is only ever supposed to publish and
/// return, but nothing stops a misbehaving one from blocking anyway, and the
/// timing thread must still record the truth when that happens.
struct SlowFirstResetSession : session {
  explicit SlowFirstResetSession(std::chrono::milliseconds delay)
      : delay_(delay) {}

  void start(run_ctx &collector) override { collector_ = &collector; }

  void send(const frame &, tag t) override {
    if (!fired_once_) {
      fired_once_ = true;
      std::this_thread::sleep_for(delay_);
    }
    handle_reply(*collector_, t, RpcStatus::OK, nullptr, 0, now_ns());
  }

  void event_done(std::uint32_t event, std::uint32_t issued, std::int32_t term,
                  bool has_term) override {
    handle_event_done(*collector_, event, issued, term, has_term);
  }

  void stop(std::chrono::nanoseconds) override {}

  std::chrono::milliseconds delay_;
  bool fired_once_ = false;
  run_ctx *collector_ = nullptr;
};

run_result run_text(const std::string &text, std::uint64_t tick_ns, session &s,
                    const run_params &params = {}) {
  std::unordered_map<std::uint64_t, session *> router{{0, &s}};
  return run(plan(parse(text, {0}, tick_ns), router, {}, params));
}

/// call_ns - deadline_ns, which is what write_csv reports as lateness.
std::int64_t lateness_of(const record &r) {
  return static_cast<std::int64_t>(r.call_ns) -
         static_cast<std::int64_t>(r.deadline_ns);
}

} // namespace

TEST(Timing, EveryEventFiresInOrderAtOrAfterItsOwnDeadlineWithoutDrifting) {
  // Four schedule shapes against the same rules. The 10us spacing is the
  // one that stresses wait_until's spin path, and the mixed-op run is the
  // one where a per-op cost difference could show up as accumulating drift.
  std::string tight, mixed;
  constexpr int kTight = 50, kMixed = 100;
  for (int i = 0; i < kTight; ++i)
    tight += std::to_string(i) + " reset\n";
  for (int i = 0; i < kMixed; ++i) {
    const int m = i % 4;
    const char *op = m == 0 ? "reset" : m == 3 ? "get_corrections" : "enqueue";
    const char *bits = (m == 1 || m == 2) ? " source=0b10" : "";
    mixed += std::to_string(i) + " " + op + bits + "\n";
  }
  struct {
    std::string text;
    std::uint64_t tick_ns;
    std::size_t count;
    const char *why;
  } cases[] = {
      {"0 reset\n", 1'000'000, 1, "a single event, at deadline zero"},
      {"0 reset\n1 reset\n2 reset\n", 1'000'000, 3, "1ms apart"},
      {tight, 10'000, kTight, "50 events 10us apart"},
      {mixed, 100'000, kMixed, "100 mixed-operation events 100us apart"},
  };

  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    auto s = make_null_session();
    auto result = run_text(c.text, c.tick_ns, *s);
    ASSERT_EQ(result.records.size(), c.count);
    EXPECT_EQ(result.tick_ns, c.tick_ns);
    EXPECT_GT(result.t0_ns, 0u);

    for (std::size_t i = 0; i < result.records.size(); ++i) {
      const auto &r = result.records[i];
      EXPECT_EQ(r.event_index, i);
      // The recorded deadline is exactly what the schedule said, however
      // late dispatch actually ran: run() never rewrites the schedule.
      EXPECT_EQ(r.deadline_ns, i * c.tick_ns) << "event " << i;
      EXPECT_GE(r.call_ns, r.deadline_ns) << "event " << i << " fired early";
      EXPECT_GE(r.return_ns, r.call_ns) << "event " << i;
      if (i > 0) {
        EXPECT_GE(r.call_ns, result.records[i - 1].call_ns) << "event " << i;
        // The null backend completes every reply synchronously, inline on
        // the timing thread, so return_ns is ordered the same way call_ns is.
        EXPECT_GE(r.return_ns, result.records[i - 1].return_ns)
            << "event " << i;
      }
    }
    // No cumulative drift: the last event's lateness must not have ballooned
    // relative to the first. The null backend is fast, so a failure here
    // means the loop is systematically falling behind, not merely noisy.
    EXPECT_LT(lateness_of(result.records.back()) -
                  lateness_of(result.records.front()),
              50'000'000)
        << "50ms+ of drift accumulated over " << c.count << " events";
  }
}

TEST(Timing, AnOverrunIsRecordedRatherThanAbsorbedOrRealigned) {
  // The first reset's send() blocks for 30ms against a schedule whose
  // second event is due at 5ms. Three things have to hold at once: the
  // second event's recorded deadline is still the schedule's 5ms (not
  // shifted to "the first event's dispatch + 5ms"), the overrun is visible
  // as lateness rather than being swallowed, and the second event fires as
  // soon as the first event's send() returns instead of waiting out another
  // tick.
  SlowFirstResetSession s(std::chrono::milliseconds(30));
  auto result = run_text("0 reset\n5 reset\n", 1'000'000, s);

  ASSERT_EQ(result.records.size(), 2u);
  const auto &first = result.records[0];
  const auto &second = result.records[1];

  EXPECT_EQ(second.deadline_ns, 5'000'000u);
  // One serial timing thread, so event 1 cannot be dispatched until event
  // 0's send() (blocked here) returns -- which is exactly what drags its
  // call_ns past its own 5ms deadline.
  EXPECT_GE(second.call_ns, first.call_ns + 29'000'000u);
  EXPECT_GT(lateness_of(second), 20'000'000)
      << "expected event 0's ~30ms delay to show up as event 1's lateness";
  EXPECT_LT(second.call_ns - first.call_ns, 35'000'000u)
      << "event 1 should fire as soon as event 0's send() returns";
}

TEST(Timing, ARelativeDeadlineResolvesAgainstThePreviousDispatchOrT0) {
  // "+2" is 2 ticks after event 0's send() actually returned on the timing
  // thread, not after the tick its file position would imply, and not the
  // instant its (independently, asynchronously stamped) return_ns lands.
  // Unlike an absolute deadline placed at a guessed tick, it therefore
  // tracks an overrun instead of being blown past the instant it is reached.
  SlowFirstResetSession s(std::chrono::milliseconds(30));
  auto result = run_text("0 reset\n+2 reset\n", 1'000'000, s);
  ASSERT_EQ(result.records.size(), 2u);
  const auto &first = result.records[0];
  const auto &second = result.records[1];
  EXPECT_GE(second.deadline_ns, first.call_ns + 29'000'000u);
  EXPECT_LE(second.deadline_ns, first.call_ns + 35'000'000u);
  EXPECT_GE(second.call_ns, second.deadline_ns);
  EXPECT_LT(second.call_ns - second.deadline_ns, 5'000'000u);

  // With nothing in front of it, a delta resolves against t0 instead.
  auto null = make_null_session();
  auto first_line = run_text("+3 reset\n", 1'000'000, *null);
  ASSERT_EQ(first_line.records.size(), 1u);
  EXPECT_EQ(first_line.records[0].deadline_ns, 3'000'000u);
}

TEST(Timing, LeadInAtEitherExtremeKeepsTimestampsRelativeToT0AndSane) {
  // call_ns/return_ns are unsigned and relative to t0, so a lead-in bug
  // shows up as a huge wrapped value rather than a small negative one.
  for (std::uint64_t lead_in_ns :
       {std::uint64_t{0}, std::uint64_t{100'000'000}}) {
    SCOPED_TRACE(lead_in_ns);
    auto s = make_null_session();
    run_params params;
    params.lead_in_ns = lead_in_ns;
    auto result = run_text("0 reset\n1 reset\n", 1'000'000, *s, params);
    ASSERT_EQ(result.records.size(), 2u);
    for (const auto &r : result.records) {
      EXPECT_LT(r.call_ns, 1'000'000'000u) << "looks like unsigned underflow";
      EXPECT_LT(r.return_ns, 1'000'000'000u) << "looks like unsigned underflow";
      EXPECT_GE(r.call_ns, r.deadline_ns);
    }
  }
}

TEST(Timing, TwoIndependentRunsFromTheSameScheduleDoNotLeakState) {
  auto s1 = make_null_session();
  auto s2 = make_null_session();
  const std::string text = "0 reset\n1 reset\n2 reset\n";
  auto result1 = run_text(text, 1'000'000, *s1);
  auto result2 = run_text(text, 1'000'000, *s2);

  ASSERT_EQ(result1.records.size(), 3u);
  ASSERT_EQ(result2.records.size(), 3u);
  // The deadline structure comes purely from the shared schedule, so it is
  // identical; t0 is sampled per run() call, so it must not be.
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_EQ(result1.records[i].deadline_ns, result2.records[i].deadline_ns);
  EXPECT_NE(result1.t0_ns, result2.t0_ns);
}

TEST(Timing, ARunAgainstTheNullBackendStillProducesAFullReport) {
  auto s = make_null_session();
  auto result = run_text("0 reset\n1 enqueue source=0b1010\n", 1000, *s);
  ASSERT_EQ(result.records.size(), 2u);
  EXPECT_EQ(result.records[0].status, static_cast<std::int32_t>(RpcStatus::OK));
  // `enqueue` is the one-round spelling of `stream`, so it reports a
  // stream_terminate rather than the wire status.
  EXPECT_EQ(result.records[1].status,
            static_cast<std::int32_t>(stream_terminate::OK));
  EXPECT_EQ(result.records[1].rounds_streamed, 1u);

  auto csv = write_csv(result);
  EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), 3); // header + 2 rows
}

TEST(Timing, ANullBackendGetCorrectionsWithRealWidthStillReportsOk) {
  // null_session zero-fills the whole reply span and reports that as
  // reply_len, so a real return_size is never mistaken for a truncated one.
  auto s = make_null_session();
  auto result = run_text("0 get_corrections return_size=8\n", 1000, *s);
  ASSERT_EQ(result.records.size(), 1u);
  EXPECT_EQ(result.records[0].status, static_cast<std::int32_t>(RpcStatus::OK));
  EXPECT_TRUE(result.records[0].read_completed);
  EXPECT_EQ(result.records[0].correction_count, 8u);
}

// ─── Signals ────────────────────────────────────────────────────────────────
//
// Signals: the schedule-level way to say "keep streaming until that round
// trip lands". Covers parser accept/reject, plan()'s ordering rule, the
// two-part run-time stop rule (a rounds floor AND an arrival, ceiling as
// backstop), and that neither reader op ever blocks the timeline.

namespace {

using cudaq::qec::decoding_server::slot::parse_get_corrections;

const std::vector<std::uint64_t> kTwoDecoders = {0, 1};

/// Counts enqueues and answers get_corrections OK, with the answer landing
/// `delay` after it was asked for. The wait belongs in `before_reply` rather
/// than `send_sync`: `send` runs on the timing thread, so sleeping there
/// would stall the very stream whose overlap with the decode is under test.
class DelayedCountingSession : public blocking_session {
public:
  explicit DelayedCountingSession(std::chrono::milliseconds delay = {})
      : delay_(delay) {}

  RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                      std::size_t &reply_len) override {
    cudaq::qec::decoding_server::slot::GetCorrectionsView view;
    if (!parse_get_corrections(f.bytes, f.size, view)) {
      cudaq::qec::decoding_server::slot::ResetView rv;
      if (!cudaq::qec::decoding_server::slot::parse_reset(f.bytes, f.size, rv))
        enqueues_.fetch_add(1, std::memory_order_relaxed);
      return RpcStatus::OK;
    }
    std::fill(reply.begin(), reply.end(), 0x00u);
    reply_len = reply.size();
    return RpcStatus::OK;
  }

  std::atomic<int> enqueues_{0};

protected:
  void before_reply() override {
    if (delay_.count() > 0)
      std::this_thread::sleep_for(delay_);
  }

private:
  std::chrono::milliseconds delay_;
};

/// A source deep enough that no test here runs it dry.
std::unique_ptr<static_source> deep_source(std::size_t rounds = 4000) {
  return std::make_unique<static_source>(std::vector<std::vector<std::uint8_t>>(
      rounds, std::vector<std::uint8_t>{1}));
}

run_result run_multi_text(const std::string &text,
                          std::unordered_map<std::uint64_t, session *> router,
                          syndrome_source &src, std::uint64_t tick_ns = 1000) {
  auto sched = parse(text, kTwoDecoders, tick_ns);
  return run(plan(sched, router, {{0, &src}}, {}));
}

} // namespace

// ---------------------------------------------------------------------------
// Signals: parsing
// ---------------------------------------------------------------------------

TEST(Signals, ASignalNameIsInternedOnceAtWhicheverEndNamesIt) {
  // `signal=` and `until=` are the two ends of one signal, so both intern
  // into the same table and a name used at both ends is one entry, not two.
  auto sched = parse("0 reset signal=ready\n"
                     "0 get_corrections return_size=8 signal=go\n"
                     "0 stream session=1 source=0 min_rounds=2 until=go\n",
                     kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), 3u);
  ASSERT_EQ(sched.signal_names.size(), 2u); // "ready" and "go", not four
  EXPECT_EQ(sched.signal_names[sched.events[0].signal_id], "ready");
  EXPECT_EQ(sched.signal_names[sched.events[1].signal_id], "go");

  const auto &stream = sched.events[2];
  EXPECT_EQ(stream.op, operation::stream);
  EXPECT_EQ(sched.signal_names[stream.until_signal_id], "go");
  EXPECT_EQ(stream.stream_min_rounds, 2u);
  EXPECT_EQ(stream.stream_max_rounds, 1000u); // bounded even though it waits
}

TEST(Signals, StreamRoundsIsShorthandForEqualMinAndMaxAndWaitsOnNothing) {
  auto sched = parse("0 stream source=0 rounds=3\n", kTwoDecoders, 1000);
  const auto &e = sched.events[0];
  EXPECT_EQ(e.stream_min_rounds, 3u);
  EXPECT_EQ(e.stream_max_rounds, 3u);
  EXPECT_EQ(e.until_signal_id, kNoSignal);
  EXPECT_TRUE(sched.signal_names.empty());
}

TEST(Signals, MalformedSignalSpellingsAreRejected) {
  const char *bad[] = {
      // A ceiling above the floor with nothing to stop the stream early is a
      // round count nothing can ever reach.
      "0 stream source=0 max_rounds=9\n",
      // An empty name at either end of the signal.
      "0 stream source=0 until=\n",
      "0 reset signal=\n",
      // `signal=` is the only operand a reset takes.
      "0 reset source=0\n",
      // The trigger column is only ever a tick or a '+N'/'-' offset, never a
      // signal name.
      "0 get_corrections return_size=8 signal=go\ngo reset session=1\n",
      // A floor above the ceiling.
      "0 get_corrections return_size=8 signal=go\n"
      "0 stream session=1 source=0 min_rounds=9 max_rounds=2 until=go\n",
  };
  for (const char *text : bad) {
    SCOPED_TRACE(text);
    EXPECT_THROW(parse(text, kTwoDecoders, 1000), std::invalid_argument);
  }
}

// ---------------------------------------------------------------------------
// Signals: the ordering rule plan() enforces
// ---------------------------------------------------------------------------

TEST(Signals, AWaitOnlyPlansIfSomeEarlierLineRaisesThatSignal) {
  // One timeline, so a signal raised by a line that has not been dispatched
  // yet cannot come up in time: the stream provably runs to its ceiling.
  DelayedCountingSession s0, s1;
  std::unordered_map<std::uint64_t, session *> router{{0, &s0}, {1, &s1}};
  const char *raiser = "0 get_corrections return_size=8 signal=go\n";
  const char *waiter =
      "0 stream session=1 source=0 min_rounds=1 max_rounds=8 until=go\n";

  for (const std::string &text : {std::string(waiter),             // nobody
                                  std::string(waiter) + raiser}) { // too late
    SCOPED_TRACE(text);
    auto src = deep_source();
    auto sched = parse(text, kTwoDecoders, 1000);
    EXPECT_THROW(plan(sched, router, {{0, src.get()}}, {}),
                 std::invalid_argument);
  }

  auto src = deep_source();
  auto ok = parse(std::string(raiser) + waiter, kTwoDecoders, 1000);
  EXPECT_NO_THROW(plan(ok, router, {{0, src.get()}}, {}));
}

// ---------------------------------------------------------------------------
// Signals: the two-part stop rule at run time
// ---------------------------------------------------------------------------

TEST(Signals, AStreamKeepsGoingPastItsFloorUntilTheSignalArrives) {
  // The read sits behind a 40 ms decode, so the stream -- whose floor is one
  // round at 100 us apiece -- has to send well past that floor before the
  // signal comes up, and must stop soon after rather than at its ceiling.
  DelayedCountingSession s0(std::chrono::milliseconds(40)), s1;
  auto src = deep_source();
  auto result =
      run_multi_text("0 get_corrections return_size=8 signal=go\n"
                     "0 stream session=1 source=0 every=1 min_rounds=1 "
                     "max_rounds=2000 until=go\n",
                     {{0, &s0}, {1, &s1}}, *src, /*tick_ns=*/100'000);
  const auto &r = result.records[1];
  EXPECT_EQ(r.status, static_cast<std::int32_t>(stream_terminate::OK));
  EXPECT_GT(r.rounds_streamed, 1u);    // past the floor, waiting
  EXPECT_LT(r.rounds_streamed, 2000u); // and stopped on the signal, not the cap
}

TEST(Signals, AStreamWhoseSignalArrivesTooLateExhaustsItsRoundsAndSaysSo) {
  // The read is slower than the stream's whole round budget, so the ceiling
  // is what ends it. Bounded and loud, never hung.
  DelayedCountingSession s0(std::chrono::milliseconds(500)), s1;
  auto src = deep_source();
  auto result =
      run_multi_text("0 get_corrections return_size=8 signal=go\n"
                     "0 stream session=1 source=0 every=0 min_rounds=1 "
                     "max_rounds=4 until=go\n",
                     {{0, &s0}, {1, &s1}}, *src);
  const auto &r = result.records[1];
  EXPECT_EQ(r.status,
            static_cast<std::int32_t>(stream_terminate::EXHAUSTED_ROUNDS));
  EXPECT_EQ(r.rounds_streamed, 4u);
  EXPECT_EQ(s1.enqueues_.load(), 4);
}

// ---------------------------------------------------------------------------
// Signals: `reset` never blocks the timeline; `signal=` only gates the flag
// ---------------------------------------------------------------------------

TEST(Signals, AResetNeverBlocksTheTimelineRegardlessOfSignal) {
  // The same 40 ms round trip either way. `reset` is always collected off
  // the timing thread, so the next line starts immediately whether or not
  // `signal=` is given -- while the record still spans the whole
  // call-to-answer, stamped by the reader rather than the timing thread.
  DelayedCountingSession unsignaled{std::chrono::milliseconds(40)};
  auto unsignaled_src = deep_source();
  auto a = run_multi_text("0 reset\n"
                          "- stream source=0 rounds=1\n",
                          {{0, &unsignaled}}, *unsignaled_src);
  EXPECT_LT(a.records[1].call_ns - a.records[0].call_ns, 20'000'000u);
  EXPECT_GE(a.records[0].return_ns - a.records[0].call_ns, 40'000'000u);
  EXPECT_EQ(a.records[0].status, static_cast<std::int32_t>(RpcStatus::OK));

  DelayedCountingSession signaled{std::chrono::milliseconds(40)};
  auto signaled_src = deep_source();
  auto b = run_multi_text("0 reset signal=ready\n"
                          "- stream source=0 rounds=1\n",
                          {{0, &signaled}}, *signaled_src);
  EXPECT_LT(b.records[1].call_ns - b.records[0].call_ns, 20'000'000u);
  EXPECT_GE(b.records[0].return_ns - b.records[0].call_ns, 40'000'000u);
  EXPECT_EQ(b.records[0].status, static_cast<std::int32_t>(RpcStatus::OK));
}

TEST(Signals, AStreamCanWaitOnAResetsSignal) {
  DelayedCountingSession s0{std::chrono::milliseconds(2)};
  auto src = deep_source();
  auto result = run_multi_text("0 reset signal=ready\n"
                               "- stream source=0 every=0 min_rounds=1 "
                               "max_rounds=4000 until=ready\n",
                               {{0, &s0}}, *src);
  const auto &r = result.records[1];
  EXPECT_EQ(r.status, static_cast<std::int32_t>(stream_terminate::OK));
  EXPECT_GT(r.rounds_streamed, 1u); // it kept sending while the reset flew
}

// ─── RequestIds ─────────────────────────────────────────────────────────────
//
// A record names every request_id its event sent, as a slice of
// request_id_log (one entry per op, one per round for a stream), for
// correlating a report row against a decoding server's own log. Must be
// complete, honest, and unambiguous.

namespace {

const std::vector<std::uint64_t> kOneDecoder = {0};

/// Answers everything OK and counts what was sent, so a test can check the
/// recorded ids against the number of frames that actually went out.
struct CountingSession : blocking_session {
  std::atomic<int> sends{0};

  RpcStatus send_sync(const frame &, std::span<std::uint8_t> reply,
                      std::size_t &reply_len) override {
    sends.fetch_add(1, std::memory_order_relaxed);
    std::fill(reply.begin(), reply.end(), 0x00u);
    reply_len = reply.size();
    return RpcStatus::OK;
  }
};

std::unique_ptr<static_source> source_of(std::size_t rounds) {
  return std::make_unique<static_source>(std::vector<std::vector<std::uint8_t>>(
      rounds, std::vector<std::uint8_t>{1}));
}

/// The ids one record claims to have sent.
std::vector<std::uint32_t> ids_of(const run_result &r, std::size_t i) {
  const auto &rec = r.records[i];
  return {r.request_id_log.begin() + rec.request_id_offset,
          r.request_id_log.begin() + rec.request_id_offset +
              rec.request_id_count};
}

} // namespace

TEST(RequestIds, EveryOpRecordsExactlyTheIdsItActuallySent) {
  // One id per RPC: one per round for a stream, one for every other op, and
  // none at all for a stream that never got a round out. A stream is the
  // only op that can send more than one, and the only one whose count is
  // decided at run time rather than by the schedule.
  const struct {
    const char *text;
    std::size_t rounds_available;
    std::uint32_t expected_ids;
    const char *why;
  } cases[] = {
      {"0 reset\n", 4, 1, "a reset sends one"},
      {"0 enqueue source=0\n", 4, 1, "an enqueue sends one"},
      {"0 get_corrections return_size=1\n", 4, 1, "a read sends one"},
      {"0 stream source=0 rounds=5\n", 16, 5, "one per round"},
      // Fewer rounds available than asked for: the count follows what was
      // sent, not what the schedule wanted.
      {"0 stream source=0 rounds=9\n", 2, 2, "cut short by a dry source"},
      // Nothing went out at all, so the record must claim nothing -- a count
      // of 0 is the only way to say that, since id 0 is never issued.
      {"0 stream source=0 rounds=3\n", 0, 0, "source dry before round 0"},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    auto src = source_of(c.rounds_available);
    CountingSession s;
    std::unordered_map<std::uint64_t, session *> router{{0, &s}};
    auto sched = parse(c.text, kOneDecoder, 1000);
    auto result = run(plan(sched, router, {{0, src.get()}}, {}));

    ASSERT_EQ(result.records.size(), 1u);
    EXPECT_EQ(result.records[0].request_id_count, c.expected_ids);
    EXPECT_EQ(s.sends.load(), static_cast<int>(c.expected_ids));
    // Ids are issued from 1 upward in send order, so a single-event
    // schedule's slice is exactly 1..N.
    std::vector<std::uint32_t> expected(c.expected_ids);
    std::iota(expected.begin(), expected.end(), 1u);
    EXPECT_EQ(ids_of(result, 0), expected);
    EXPECT_EQ(result.request_id_log, expected);
  }
}

TEST(RequestIds, AnUndispatchedEventRecordsNoIdsAndTheLogStopsAtTheAbort) {
  // A hard error leaves the later records present but default, so their
  // slices must be empty rather than pointing at somebody else's ids. The
  // abort is collected asynchronously on the session's own worker, so it is
  // not guaranteed to land before the very next dispatch -- a long tail of
  // events, spaced far apart relative to that async completion, turns that
  // race into a formality (see AbortOnHardError for the same pattern).
  struct FailOnSecond : blocking_session {
    int calls = 0;
    RpcStatus send_sync(const frame &, std::span<std::uint8_t>,
                        std::size_t &reply_len) override {
      reply_len = 0;
      return ++calls == 2 ? RpcStatus::BAD_REQUEST : RpcStatus::OK;
    }
  } s;
  std::unordered_map<std::uint64_t, session *> router{{0, &s}};
  std::string text = "0 reset\n1 reset\n";
  for (int tick = 2; tick <= 1000; ++tick)
    text += std::to_string(tick) + " reset\n";
  auto sched = parse(text, kOneDecoder, 500'000);
  auto result = run(plan(sched, router, {}));

  ASSERT_EQ(result.records.size(), 1001u);
  EXPECT_EQ(result.records[0].request_id_count, 1u);
  EXPECT_EQ(result.records[1].request_id_count, 1u); // it was sent, then failed
  ASSERT_FALSE(result.records.back().dispatched)
      << "the abort never caught up with 999 trailing events";
  for (const auto &rec : result.records)
    if (!rec.dispatched)
      EXPECT_EQ(rec.request_id_count, 0u);
  // The log holds exactly the ids every dispatched record actually sent.
  std::uint32_t total = 0;
  for (const auto &rec : result.records)
    total += rec.request_id_count;
  EXPECT_EQ(result.request_id_log.size(), total);
}

TEST(RequestIds, TheLogIsStrictlyIncreasingAndEveryIdIsUsedOnce) {
  // The property the (offset, count) slicing rests on: ids are handed out in
  // issue order by one thread, so a record's ids are a contiguous run and no
  // two records can name the same id.
  auto src = source_of(64);
  CountingSession s;
  std::unordered_map<std::uint64_t, session *> router{{0, &s}};
  auto sched = parse("0 reset\n"
                     "1 stream source=0 rounds=6\n"
                     "2 get_corrections return_size=1\n"
                     "3 stream source=0 rounds=4\n"
                     "4 get_corrections return_size=1\n",
                     kOneDecoder, 1000);
  auto result = run(plan(sched, router, {{0, src.get()}}, {}));

  const auto &log = result.request_id_log;
  ASSERT_EQ(log.size(), 1u + 6u + 1u + 4u + 1u);
  for (std::size_t i = 1; i < log.size(); ++i)
    EXPECT_LT(log[i - 1], log[i]) << "at " << i;
  EXPECT_EQ(std::unordered_set<std::uint32_t>(log.begin(), log.end()).size(),
            log.size());

  // Each record's slice covers its own contiguous stretch, and together they
  // account for the whole log with nothing left over.
  std::uint32_t next = 0;
  for (const auto &rec : result.records) {
    EXPECT_EQ(rec.request_id_offset, next);
    next += rec.request_id_count;
  }
  EXPECT_EQ(next, log.size());
}

// ─── AbortOnHardError ───────────────────────────────────────────────────────
//
// Any status other than OK or NOT_READY is a hard error that aborts the run
// (a decoder stuck on INVALID_DECODER won't self-correct). result.records
// is never truncated; `record::dispatched`, not vector length, distinguishes
// "ran" from "the abort pre-empted it."

namespace {

/// A session whose Nth send_sync call (1-indexed) answers `status`;
/// everything else answers OK. Lets a test pick exactly which event in a
/// multi-event schedule goes bad.
struct FailOnCallSession : blocking_session {
  int fail_on_call = -1;
  RpcStatus fail_status = RpcStatus::BAD_REQUEST;
  int calls = 0;

  RpcStatus send_sync(const frame &, std::span<std::uint8_t>,
                      std::size_t &reply_len) override {
    ++calls;
    reply_len = 0;
    return calls == fail_on_call ? fail_status : RpcStatus::OK;
  }
};

} // namespace

TEST(AbortOnHardError, AHardErrorEventuallyStopsTheRun) {
  // `reset` and `get_corrections` are both always collected asynchronously,
  // on the session's own worker, so the abort is not guaranteed to land
  // before the very next dispatch -- it lands whenever that worker gets to
  // it, which races the timing thread racing ahead through cheap, instantly
  // answered events. Spacing the 1000 trailing events 500us apart (versus a
  // sub-microsecond async completion even under load) turns that race into
  // a formality.
  const struct {
    const char *op;
    RpcStatus status;
    const char *why;
  } cases[] = {
      {"reset", RpcStatus::BAD_REQUEST, "on a reset"},
      {"get_corrections", RpcStatus::INVALID_DECODER, "on a get_corrections"},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    FailOnCallSession s;
    s.fail_on_call = 2; // the second event
    s.fail_status = c.status;
    std::string text = std::string("0 ") + c.op + "\n";
    for (int tick = 1; tick <= 1000; ++tick)
      text += std::to_string(tick) + " " + c.op + "\n";
    auto sched = parse(text, {0}, 500'000);
    std::unordered_map<std::uint64_t, session *> router{{0, &s}};
    auto result = run(plan(sched, router, {}));

    ASSERT_EQ(result.records.size(), 1001u);
    EXPECT_TRUE(result.records[0].dispatched);
    EXPECT_EQ(result.records[0].status,
              static_cast<std::int32_t>(RpcStatus::OK));
    EXPECT_TRUE(result.records[1].dispatched);
    EXPECT_EQ(result.records[1].status, static_cast<std::int32_t>(c.status));
    // records is never truncated, so `dispatched` is what tells "ran" from
    // "the abort pre-empted it" -- some tail of the 999 events after the
    // failure must have been pre-empted, even though exactly where that
    // tail starts is a race the reader is not guaranteed to win
    // immediately.
    EXPECT_FALSE(result.records.back().dispatched)
        << "the abort never caught up with 1000 trailing events";
  }
}

TEST(AbortOnHardError, NotReadyNeverAbortsTheRun) {
  // Never answers anything but NOT_READY -- if NOT_READY were (wrongly)
  // treated as a hard error, this would abort after the first event.
  struct AlwaysNotReadySession : blocking_session {
    RpcStatus send_sync(const frame &, std::span<std::uint8_t>,
                        std::size_t &reply_len) override {
      reply_len = 0;
      return RpcStatus::NOT_READY;
    }
  } s;
  auto sched = parse("0 get_corrections\n1 get_corrections\n", {0}, 1000);
  std::unordered_map<std::uint64_t, session *> router;
  router[0] = &s;
  auto p = plan(sched, router, {});
  auto result = run(std::move(p));

  // Both events were dispatched; NOT_READY never aborts.
  ASSERT_EQ(result.records.size(), 2u);
  EXPECT_EQ(result.records[0].status,
            static_cast<int32_t>(RpcStatus::NOT_READY));
  EXPECT_EQ(result.records[1].status,
            static_cast<int32_t>(RpcStatus::NOT_READY));
}

// ─── MultiDecoderAdversarial ────────────────────────────────────────────────
//
// One timeline, several decoders (`session=N` picks which). Routing is
// plan()'s business (Capabilities, above); here: both decoders share the one
// timeline, so a hard error on either stops the whole run.

namespace {

using cudaq::qec::decoding_server::slot::parse_reset;

/// A session dedicated to one decoder_id (per the one-session-per-decoder
/// rule plan() enforces) that answers OK, having first checked that what
/// arrived really was a reset.
class ResetOkSession : public blocking_session {
public:
  RpcStatus send_sync(const frame &f, std::span<std::uint8_t>,
                      std::size_t &reply_len) override {
    reply_len = 0;
    ResetView rv;
    return parse_reset(f.bytes, f.size, rv) ? RpcStatus::OK
                                            : RpcStatus::BAD_REQUEST;
  }

private:
  using ResetView = cudaq::qec::decoding_server::slot::ResetView;
};

} // namespace

// -- A hard error on one decoder's event aborts the whole run: the events
// behind it, whichever decoder they address, must not dispatch.
// record::dispatched is how a caller tells "ran" from "pre-empted by the
// abort." --
TEST(MultiDecoderAdversarial,
     HardErrorOnOneDecoderStopsEveryOtherDecodersEvents) {
  struct ImmediateBadRequestSession : blocking_session {
    RpcStatus send_sync(const frame &, std::span<std::uint8_t>,
                        std::size_t &reply_len) override {
      reply_len = 0;
      return RpcStatus::BAD_REQUEST;
    }
  } bad;
  ResetOkSession ok;

  // Decoder 0 fails at tick 0; decoder 1's long tail of events sits behind
  // it on the one timeline. `reset` is always collected off the timing
  // thread, so the abort is not guaranteed to land before the very next
  // dispatch -- spacing the tail's 1000 ticks 500us apart (500ms total)
  // gives decoder 0's async failure, which lands in well under a
  // millisecond even under load, a wide margin to turn that race into a
  // formality (see AbortOnHardError for the same pattern).
  std::string text = "0 reset\n";
  for (int tick = 1; tick <= 1000; ++tick)
    text += std::to_string(tick) + " reset session=1\n";
  auto sched = parse(text, {0, 1}, 500'000);
  std::unordered_map<std::uint64_t, session *> router;
  router[0] = &bad;
  router[1] = &ok;
  auto p = plan(sched, router, {});
  auto result = run(std::move(p));

  ASSERT_EQ(result.records.size(), 1001u);
  const auto &d0 = result.records[0];
  EXPECT_EQ(d0.decoder_id, 0u);
  EXPECT_TRUE(d0.dispatched);
  EXPECT_EQ(d0.status, static_cast<std::int32_t>(RpcStatus::BAD_REQUEST));

  EXPECT_FALSE(result.records.back().dispatched)
      << "decoder 1's events sit behind decoder 0's failure on the one "
         "timeline, so the abort must eventually have pre-empted them";
}

// ─── TruncatedReply ─────────────────────────────────────────────────────────
//
// A session that answers OK but hands back fewer bytes than return_size:
// naively unpacking is indistinguishable from an all-zero correction. The
// emulator must demote such a truncated OK to INTERNAL_ERROR rather than
// trust bits that were never written.

namespace {

/// Answers OK to get_corrections but only ever writes `bytes_written` bytes
/// into the reply buffer and reports that as reply_len, regardless of how
/// many bytes the caller actually asked for.
struct TruncatingSession : blocking_session {
  std::size_t bytes_written = 0;

  RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                      std::size_t &reply_len) override {
    using cudaq::qec::decoding_server::slot::parse_get_corrections;
    cudaq::qec::decoding_server::slot::GetCorrectionsView view;
    if (!parse_get_corrections(f.bytes, f.size, view))
      return RpcStatus::BAD_REQUEST;
    const std::size_t n = std::min(bytes_written, reply.size());
    for (std::size_t i = 0; i < n; ++i)
      reply[i] = 0xFF;
    reply_len = n;
    return RpcStatus::OK;
  }
};

} // namespace

TEST(TruncatedReply, AShortReplyIsRejectedAndOnlyAnExactlyFullOneIsTrusted) {
  struct {
    const char *expected_bits; // width the schedule asks back
    std::size_t bytes_written; // what the session actually writes
    bool accepted;
    const char *why;
  } cases[] = {
      // The classic failure: nothing was written, and the zero-initialized
      // buffer would read back as a perfectly plausible all-zero correction.
      {"00000000", 0, false, "8 bits requested, 0 bytes written"},
      {"0000000000000000", 1, false, "16 bits requested, only 1 byte written"},
      {"00000000", 1, true, "8 bits requested, ceil(8/8) = 1 byte written"},
      // return_size 0 means "request nothing", and bit_packed_bytes(0) == 0,
      // so an empty reply here is genuinely complete rather than truncated.
      {"", 0, true, "nothing requested, nothing written"},
  };

  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    TruncatingSession s;
    s.bytes_written = c.bytes_written;
    auto sched = parse(
        std::string("0 get_corrections ") + c.expected_bits + "\n", {0}, 1000);
    std::unordered_map<std::uint64_t, session *> router{{0, &s}};
    auto result = run(plan(sched, router, {}));

    ASSERT_EQ(result.records.size(), 1u);
    const auto &rec = result.records[0];
    EXPECT_EQ(rec.status,
              static_cast<std::int32_t>(
                  c.accepted ? RpcStatus::OK : RpcStatus::INTERNAL_ERROR));
    EXPECT_EQ(rec.read_completed, c.accepted);
    if (!c.accepted)
      EXPECT_EQ(rec.correction_count, 0u)
          << "a rejected reply must contribute no correction bits";
  }
}
