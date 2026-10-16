/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Tests the input format and schedule data model: strict
/// parsing, arena packing, and tick -> deadline_ns resolution.

#include "emulator.h"

#include <gtest/gtest.h>

using namespace cudaq::qec::playback;

namespace {
const std::vector<std::uint64_t> kTwoDecoders = {0, 1};

/// parse() reports errors as std::invalid_argument
/// with the offending line number embedded in the
/// message ("playback schedule, line N: ...")
void expect_error_at_line(const std::string &text,
                          const std::vector<std::uint64_t> &known_decoder_ids,
                          std::uint64_t tick_ns, std::size_t line) {
  try {
    parse(text, known_decoder_ids, tick_ns);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    const std::string expect = "line " + std::to_string(line) + ":";
    EXPECT_NE(std::string(e.what()).find(expect), std::string::npos)
        << "expected \"" << expect << "\" in: " << e.what();
  }
}

/// One rejected line, with a note naming the rule that turns it down.
struct bad_line {
  const char *text;
  const char *why;
};

void expect_all_rejected(const std::vector<bad_line> &cases,
                         const std::vector<std::uint64_t> &ids = kTwoDecoders,
                         std::uint64_t tick_ns = 1000) {
  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    expect_error_at_line(c.text, ids, tick_ns, 1);
  }
}
} // namespace

// ---------------------------------------------------------------------------
// Baseline: file shape
// ---------------------------------------------------------------------------

TEST(Parser, CommentsAndBlankLinesAreSkippedButStillCountTowardsLineNumbers) {
  // Skipped lines still have to advance the counter, or every error message
  // in a commented schedule points at the wrong place.
  const std::string text = R"(# header
   # indented comment

0 reset
)";
  auto sched = parse(text, kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), 1u);
  EXPECT_EQ(sched.events[0].op, operation::reset);
  EXPECT_EQ(sched.events[0].decoder_id, 0u);
  EXPECT_EQ(sched.events[0].deadline_ns, 0u);

  expect_error_at_line(text + "0 reset session=7\n", kTwoDecoders, 1000, 5);
}

TEST(Parser, EventsKeepFileOrderIncludingTiesOnOneTick) {
  // parse() never reorders: the input is required to be in dispatch order
  // and comes out that way. Worth pinning down for same-tick lines (a common
  // case: several decoders' events on one tick), which are the only place
  // anything could have been tempted to reorder.
  const std::string text = R"(
1 reset
1 reset session=1
3 reset session=1
)";
  auto sched = parse(text, kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), 3u);
  EXPECT_LE(sched.events[0].deadline_ns, sched.events[1].deadline_ns);
  EXPECT_LE(sched.events[1].deadline_ns, sched.events[2].deadline_ns);
  EXPECT_EQ(sched.events[0].decoder_id, 0u);
  EXPECT_EQ(sched.events[1].decoder_id, 1u); // tie at tick 1, file order kept
}

TEST(Parser, DecoderInfoCapturesKnownDecoderIds) {
  auto sched = parse("0 reset\n0 reset session=1\n", kTwoDecoders, 1000);
  ASSERT_EQ(sched.decoders.size(), 2u);
}

// ---------------------------------------------------------------------------
// Baseline: operands land in the right field and arena
// ---------------------------------------------------------------------------

TEST(Parser, EachOpsOperandsLandInTheRightFieldAndArena) {
  const std::string text = R"(
1 enqueue source=0b010110
1 enqueue source=3
2 get_corrections 10
3 stream session=1 source=2 every=4 max_rounds=32 until=c0
)";
  auto sched = parse(text, kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), 4u);

  // Literal bits: no source id, and the bits land in the syndrome arena.
  const auto &literal = sched.events[0];
  EXPECT_EQ(literal.op, operation::stream); // `enqueue` is a one-round stream
  ASSERT_EQ(literal.syndrome_count, 6u);
  ASSERT_LE(literal.syndrome_offset + literal.syndrome_count,
            sched.syndrome_arena.size());
  const std::vector<std::uint8_t> bits = {0, 1, 0, 1, 1, 0};
  for (std::size_t i = 0; i < bits.size(); ++i)
    EXPECT_EQ(sched.syndrome_arena[literal.syndrome_offset + i], bits[i]);

  // A source id instead: nothing is drawn until run time, so no arena bits.
  EXPECT_EQ(sched.events[1].source_id, 3u);
  EXPECT_EQ(sched.events[1].syndrome_count, 0u);

  // A read's expected bits go to the *other* arena.
  const auto &read = sched.events[2];
  EXPECT_EQ(read.op, operation::get_corrections);
  ASSERT_EQ(read.expected_count, 2u);
  const std::vector<std::uint8_t> expected = {1, 0};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(sched.expected_arena[read.expected_offset + i], expected[i]);

  const auto &stream = sched.events[3];
  EXPECT_EQ(stream.op, operation::stream);
  EXPECT_EQ(stream.decoder_id, 1u);
  EXPECT_EQ(stream.source_id, 2u);
  EXPECT_EQ(stream.stream_every_ticks, 4u);
  EXPECT_EQ(stream.stream_max_rounds, 32u);
  // A signal to wait on is what makes this stream not fixed-round.
  ASSERT_NE(stream.until_signal_id, kNoSignal);
  EXPECT_EQ(sched.signal_names[stream.until_signal_id], "c0");
}

TEST(Parser, StreamPacingDefaultsToOneTickAndOnlyEveryZeroIsUnpaced) {
  auto paced = parse("0 stream source=0 until=c0\n", kTwoDecoders, 1000);
  EXPECT_EQ(paced.events[0].stream_every_ticks, 1u);
  EXPECT_GT(paced.events[0].stream_max_rounds, 0u); // bounded, always

  auto unpaced =
      parse("0 stream source=0 every=0 until=c0\n", kTwoDecoders, 1000);
  EXPECT_EQ(unpaced.events[0].stream_every_ticks, 0u);
}

// ---------------------------------------------------------------------------
// Baseline: triggers
// ---------------------------------------------------------------------------

TEST(Parser, AnAbsoluteTriggerIsATickScaledByTheScheduleTickWidth) {
  auto sched = parse("5 reset\n", kTwoDecoders, 2000);
  ASSERT_EQ(sched.events.size(), 1u);
  EXPECT_EQ(sched.events[0].trig, trigger::tick);
  EXPECT_EQ(sched.events[0].deadline_ns, 10000u);
}

TEST(Parser, ARelativeTriggerHoldsADeltaAndKeepsItsPlaceInFileOrder) {
  // A relative line's deadline_ns is a delta, not an absolute offset, so it
  // is routinely *smaller* than the deadline of the line above it. File
  // order alone places it; nothing compares the two numbers.
  auto sched = parse("10 reset\n+2 enqueue source=0\n", kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), 2u);
  EXPECT_EQ(sched.events[0].trig, trigger::tick);
  EXPECT_EQ(sched.events[0].deadline_ns, 10000u);
  EXPECT_EQ(sched.events[1].trig, trigger::delta);
  EXPECT_EQ(sched.events[1].deadline_ns, 2000u); // smaller, and still last

  // `-` is the transcript spelling of `+0`: one line straight after the
  // one above, with nothing to wait for. It is a trigger, not a signal name.
  auto dash = parse("0 reset\n- reset\n", kTwoDecoders, 1000);
  auto plus_zero = parse("0 reset\n+0 reset\n", kTwoDecoders, 1000);
  ASSERT_EQ(dash.events.size(), 2u);
  EXPECT_EQ(dash.events[1].trig, trigger::delta);
  EXPECT_EQ(dash.events[1].deadline_ns, 0u);
  EXPECT_EQ(dash.events[1].trig, plus_zero.events[1].trig);
  EXPECT_EQ(dash.events[1].deadline_ns, plus_zero.events[1].deadline_ns);
  EXPECT_TRUE(dash.signal_names.empty());
}

TEST(Parser, AbsoluteTicksMustBeNonDecreasingAndDeltasAreExempt) {
  EXPECT_NO_THROW(parse("0 reset\n0 reset\n1 reset\n", kTwoDecoders, 1000));
  expect_error_at_line("2 reset\n1 reset\n", kTwoDecoders, 1000, 2);
  EXPECT_NO_THROW(parse("100 reset\n+1 reset\n", kTwoDecoders, 1000));
}

TEST(Parser, OnceATriggerGoesRelativeEveryLaterTriggerMustAlsoBeRelative) {
  expect_error_at_line("5 reset\n+100 reset\n4 reset\n", kTwoDecoders, 1000, 3);
  expect_error_at_line("100 reset\n+1 reset\n200 reset\n", kTwoDecoders, 1000,
                       3);
}

// ---------------------------------------------------------------------------
// Baseline: rejections
// ---------------------------------------------------------------------------

TEST(Parser, LinesThatAreNotAValidScheduleAreRejectedNamingTheLine) {
  struct {
    const char *text;
    std::uint64_t tick_ns;
    const char *why;
  } cases[] = {
      {"0 frobnicate\n", 1000, "no such operation"},
      {"0 reset session=7\n", 1000, "decoder_id not in the config"},
      {"+ reset\n", 1000, "'+' with no digits after it"},
      {"0 enqueue source=0b01201\n", 1000, "'2' is not a bit"},
      // source_id and max_rounds are uint32_t; one past the top must be an
      // error rather than a silent truncation naming a different source.
      {"0 enqueue source=4294967296\n", 1000, "source id past uint32"},
      {"0 stream source=4294967296 until=c0\n", 1000, "source id past uint32"},
      {"0 stream source=0 max_rounds=4294967296 until=c0\n", 1000,
       "max_rounds past uint32"},
      // A tick is uint64_t, but tick * tick_ns still has to fit in one.
      {"18446744073709551615 reset\n", 1'000'000, "tick * tick_ns overflows"},
      {"+18446744073709551615 reset\n", 1'000'000, "same, for a delta"},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.why);
    expect_error_at_line(c.text, kTwoDecoders, c.tick_ns, 1);
  }

  // A name where a trigger belongs, on a line that is not the first, so the
  // reported line number is shown to track.
  expect_error_at_line("0 reset\nlater reset\n", kTwoDecoders, 1000, 2);
}

// ---------------------------------------------------------------------------
// Adversarial: tokenizing -- whitespace and comments
// ---------------------------------------------------------------------------

TEST(ParserAdversarial, EveryWhitespaceFormTokenizesTheSameWay) {
  // tokenize() splits on runs of whitespace via istringstream::operator>>, so
  // neither kind nor count of separators matters. '\r' is the interesting
  // case: getline() splits only on '\n', leaving one on a CRLF line's end,
  // harmless only because the "C" locale treats it as whitespace too.
  for (const char *text :
       {"1 reset\n", "1\treset\n", "1    reset\n", "1 reset   \n",
        "  1 reset\n", "1\t \treset\t\n", "1\r reset\n", "1 reset\r\n"}) {
    SCOPED_TRACE(text);
    auto sched = parse(text, kTwoDecoders, 1000);
    ASSERT_EQ(sched.events.size(), 1u);
    EXPECT_EQ(sched.events[0].op, operation::reset);
    EXPECT_EQ(sched.events[0].decoder_id, 0u);
    EXPECT_EQ(sched.events[0].deadline_ns, 1000u);
  }
}

TEST(ParserAdversarial, AHashStartsACommentWhereverItAppears) {
  // '#' truncates the raw line before tokenization, so it needs no
  // surrounding space and can cut a token in half: "0b010#111" is the bit
  // string "010" plus a comment, not a malformed one.
  for (const char *text : {
           "0 enqueue source=0b010\n",            // no comment: the baseline
           "0 enqueue source=0b010#111\n",        // cuts a bit string in half
           "0 enqueue source=0b010#\n",           // nothing after the hash
           "0 enqueue source=0b010 # until=c0\n", // drops a later operand
       }) {
    SCOPED_TRACE(text);
    auto sched = parse(text, kTwoDecoders, 1000);
    ASSERT_EQ(sched.events.size(), 1u);
    const auto &e = sched.events[0];
    ASSERT_EQ(e.syndrome_count, 3u);
    const std::vector<std::uint8_t> expected = {0, 1, 0};
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(sched.syndrome_arena[e.syndrome_offset + i], expected[i]);
  }
}

TEST(ParserAdversarial, TextWithNothingToRunProducesZeroEventsNotAnError) {
  for (const char *text :
       {"", "\n", "   \t   \n", "#\n", "# header\n\n   # indented\n#\n"}) {
    SCOPED_TRACE(text);
    EXPECT_EQ(parse(text, kTwoDecoders, 1000).events.size(), 0u);
  }
}

// ---------------------------------------------------------------------------
// Adversarial: operands
// ---------------------------------------------------------------------------

TEST(ParserAdversarial, MalformedOperandsAreAllRejectedNamingTheLine) {
  expect_all_rejected({
      // Operands are split into key=value pairs once, before any op looks at
      // them, so "said twice" and "not mine" are caught in one place for
      // every key rather than depending on which op happened to check.
      {"0 stream source=1 source=9 until=c0\n", "same key twice"},
      {"0 enqueue source=0b010 source=1\n", "same key twice: bits and an id"},
      {"0 get_corrections 010 101\n", "two operands with no key"},
      {"0 stream source=0 bogus=1 until=c0\n", "unknown key"},
      {"0 reset extra_token\n", "reset takes no operand without a key"},
      // `stream` has no wall-clock budget: it runs on max_rounds and source
      // exhaustion alone; `timeout=` is not a recognized operand.
      {"0 stream source=0 timeout=5ms until=c0\n", "unknown operand: timeout="},
      {"0 stream until=c0\n", "stream requires source="},
      {"0 enqueue\n", "enqueue requires source="},
      {"0 enqueue source=0b\n", "0b prefix with no bits"},
      {"0 enqueue source=0b234567890\n", "non-binary digits"},
      {"0 stream source=0 min_rounds=1 max_rounds=8\n",
       "min_rounds and max_rounds differ with no until="},
  });
}

TEST(ParserAdversarial, OperandsOutOfOrderAreAccepted) {
  const std::string text = "0 stream max_rounds=7 every=3 source=2 until=c0\n";
  auto sched = parse(text, kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), 1u);
  const auto &e = sched.events[0];
  EXPECT_EQ(e.source_id, 2u);
  EXPECT_EQ(e.stream_every_ticks, 3u);
  EXPECT_EQ(e.stream_max_rounds, 7u);
}

// ---------------------------------------------------------------------------
// Adversarial: numeric extremes -- each operand parses at its own width
// ---------------------------------------------------------------------------

TEST(ParserAdversarial, EachOperandAcceptsTheFullRangeOfItsOwnWidth) {
  // every_ticks and decoder_id are uint64_t; max_rounds is uint32_t. The
  // widths are independent, so each is pushed to its own top separately.
  auto paced = parse("0 stream source=0 every=18446744073709551615 until=c0\n",
                     kTwoDecoders, /*tick_ns=*/1);
  EXPECT_EQ(paced.events[0].stream_every_ticks, 18446744073709551615ull);

  for (std::uint32_t max_rounds : {1u, 4294967295u}) {
    SCOPED_TRACE(max_rounds);
    auto s =
        parse("0 stream source=0 max_rounds=" + std::to_string(max_rounds) +
                  " until=c0\n",
              kTwoDecoders, 1000);
    EXPECT_EQ(s.events[0].stream_max_rounds, max_rounds);
  }

  constexpr std::uint64_t kBigId = 18446744073709551615ull; // UINT64_MAX
  auto big = parse("0 reset session=18446744073709551615\n", {kBigId}, 1000);
  ASSERT_EQ(big.events.size(), 1u);
  EXPECT_EQ(big.events[0].decoder_id, kBigId);
}

TEST(ParserAdversarial, ArithmeticThatWouldOverflowIsRejectedNotWrapped) {
  // Unguarded, either product wraps to a deadline in the past: the event
  // fires immediately and the stream runs flat out -- the opposite of the
  // pacing that was asked for, and silent about it.
  expect_all_rejected(
      {{"0 stream source=0 every=18446744073709551615 until=c0\n",
        "every= * tick_ns overflows"}},
      kTwoDecoders, 1000);
  expect_all_rejected({{"2 reset\n", "tick * tick_ns overflows"}}, kTwoDecoders,
                      18446744073709551615ull);
}

TEST(ParserAdversarial, ATickNsOfZeroCollapsesEveryDeadlineToZero) {
  // tick * 0 == 0: no overflow, every event just fires at t0. Semantically
  // odd but not a crash -- this documents the current behaviour.
  for (const char *text : {"0 reset\n", "500 reset\n"}) {
    SCOPED_TRACE(text);
    auto sched = parse(text, kTwoDecoders, /*tick_ns=*/0);
    ASSERT_EQ(sched.events.size(), 1u);
    EXPECT_EQ(sched.events[0].deadline_ns, 0u);
  }
}

// ---------------------------------------------------------------------------
// Adversarial: decoder_id membership
// ---------------------------------------------------------------------------

TEST(ParserAdversarial, DecoderIdZeroIsRejectedWhenAbsentFromTheKnownSet) {
  // Nothing is special about 0: it is the default only because most
  // schedules have one decoder, and it still has to be in the config.
  expect_error_at_line("0 reset\n", /*ids=*/{1, 2}, 1000, 1);
}

TEST(ParserAdversarial, DuplicateKnownDecoderIdsArePreservedInDecoderInfo) {
  auto sched = parse("0 reset session=5\n", /*ids=*/{5, 5}, 1000);
  ASSERT_EQ(sched.decoders.size(), 2u); // parser doesn't dedup its bookkeeping
  EXPECT_EQ(sched.decoders[0], 5u);
  EXPECT_EQ(sched.decoders[1], 5u);
  ASSERT_EQ(sched.events.size(), 1u);
  EXPECT_EQ(sched.events[0].decoder_id, 5u); // membership check still works
}

// ---------------------------------------------------------------------------
// Adversarial: arena packing, from one bit up to scale
// ---------------------------------------------------------------------------

TEST(ParserAdversarial, EveryEventsArenaSliceMatchesItsOwnBitPattern) {
  // Each event's (offset, count) has to address its own bits and nobody
  // else's, at one bit and at 500. The `get_corrections` line at the end
  // lands in the other arena, so the two are filled and indexed separately.
  std::vector<std::string> patterns = {"0", "111", "0101", "000111000", "1"};
  std::string long_bits;
  for (int i = 0; i < 500; ++i)
    long_bits.push_back((i % 3 == 0) ? '1' : '0');
  patterns.push_back(long_bits);

  std::string text;
  for (std::size_t i = 0; i < patterns.size(); ++i)
    text += std::to_string(i) + " enqueue source=0b" + patterns[i] + "\n";
  text +=
      std::to_string(patterns.size()) + " get_corrections " + long_bits + "\n";

  auto sched = parse(text, kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), patterns.size() + 1);
  for (std::size_t i = 0; i < patterns.size(); ++i) {
    const auto &e = sched.events[i];
    ASSERT_EQ(e.syndrome_count, patterns[i].size()) << "pattern " << i;
    for (std::size_t b = 0; b < patterns[i].size(); ++b)
      EXPECT_EQ(sched.syndrome_arena[e.syndrome_offset + b],
                patterns[i][b] == '1' ? 1u : 0u)
          << "pattern " << i << " bit " << b;
  }
  const auto &read = sched.events.back();
  ASSERT_EQ(read.expected_count, long_bits.size());
  for (std::size_t b = 0; b < long_bits.size(); ++b)
    EXPECT_EQ(sched.expected_arena[read.expected_offset + b],
              long_bits[b] == '1' ? 1u : 0u)
        << "expected bit " << b;
}

TEST(ParserAdversarial, LargeScheduleParsesCorrectlyAndStaysInOrder) {
  // Half the lines repeat the tick before them, so this covers ties as well
  // as scale: file order has to survive both.
  constexpr int kLines = 5000;
  std::string text;
  text.reserve(kLines * 12);
  for (int i = 0; i < kLines; ++i) {
    text += std::to_string(i / 2) + " reset";
    text += (i % 2 == 0) ? "\n" : " session=1\n";
  }
  auto sched = parse(text, kTwoDecoders, 1000);
  ASSERT_EQ(sched.events.size(), static_cast<std::size_t>(kLines));
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    EXPECT_EQ(sched.events[i].deadline_ns, (i / 2) * 1000u) << "event " << i;
    EXPECT_EQ(sched.events[i].decoder_id, i % 2 == 0 ? 0u : 1u)
        << "event " << i;
  }
}
