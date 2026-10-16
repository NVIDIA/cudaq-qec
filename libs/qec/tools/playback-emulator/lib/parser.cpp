/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file parser.cpp
/// @brief parse(): line-oriented, `<trigger> <op> [key=value...]`. Unknown
/// op, malformed bits, or an unlisted `session=` is a parse error. Signal
/// names are interned into `schedule::signal_names`; events carry indices only.

#include "emulator.h"

#include <algorithm>
#include <charconv>
#include <sstream>
#include <type_traits>
#include <unordered_set>

namespace cudaq::qec::playback {

namespace {

/// Splits on runs of whitespace; '#' starts a comment that runs to EOL.
std::vector<std::string> tokenize(const std::string &line) {
  std::string trimmed = line;
  if (auto hash = trimmed.find('#'); hash != std::string::npos)
    trimmed.resize(hash);
  std::vector<std::string> tokens;
  std::istringstream iss(trimmed);
  std::string tok;
  while (iss >> tok)
    tokens.push_back(tok);
  return tokens;
}

bool is_digits(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isdigit(c);
  });
}

/// Parses `s` as a base-10 `T` (T is uint32_t or uint64_t), rejecting
/// anything non-digit or too wide for T
template <typename T>
T parse_uint(const std::string &s, const char *what) {
  static_assert(std::is_same_v<T, std::uint32_t> ||
                std::is_same_v<T, std::uint64_t>);
  if (!is_digits(s))
    throw std::invalid_argument(std::string(what) + " '" + s +
                                "' is not a valid integer");
  T v = 0;
  const auto end = s.data() + s.size();
  const auto res = std::from_chars(s.data(), end, v);
  if (res.ec != std::errc() || res.ptr != end)
    throw std::invalid_argument(std::string(what) + " '" + s +
                                "' does not fit in " +
                                (sizeof(T) == 4 ? "32" : "64") + " bits");
  return v;
}

std::uint32_t parse_u32(const std::string &s, const char *what) {
  return parse_uint<std::uint32_t>(s, what);
}

std::uint64_t parse_u64(const std::string &s, const char *what) {
  return parse_uint<std::uint64_t>(s, what);
}

/// Bit-string token: only '0'/'1' characters. Appends one byte (0x00/0x01)
/// per character to `arena`
void append_bits(std::vector<std::uint8_t> &arena, const std::string &s,
                 const char *what) {
  for (char c : s) {
    if (c != '0' && c != '1')
      throw std::invalid_argument(std::string("malformed ") + what +
                                  " bit string '" + s +
                                  "': expected only '0'/'1'");
    arena.push_back(static_cast<std::uint8_t>(c - '0'));
  }
}

/// One line's operands, split once into `key=value` pairs and the at most one
/// token written without a keyword. Every accessor marks what it read, so
/// whatever is left at the end is an operand the op does not take, and is
/// rejected.
class operands {
public:
  operands(std::vector<std::string>::const_iterator first,
           std::vector<std::string>::const_iterator last) {
    for (auto it = first; it != last; ++it) {
      const auto eq = it->find('=');
      if (eq == std::string::npos) {
        if (!bare_.empty())
          throw std::invalid_argument("takes at most one operand written "
                                      "without a 'key=', got '" +
                                      bare_ + "' and '" + *it + "'");
        bare_ = *it;
        continue;
      }
      const std::string key = it->substr(0, eq);
      if (std::any_of(kv_.begin(), kv_.end(),
                      [&](const entry &kv) { return kv.key == key; }))
        throw std::invalid_argument("'" + key + "=' given more than once");
      kv_.push_back({key, it->substr(eq + 1), false});
    }
  }

  /// `key`'s value, or nullptr if the line did not carry it.
  const std::string *value(std::string_view key) {
    for (auto &kv : kv_)
      if (kv.key == key) {
        kv.read = true;
        return &kv.value;
      }
    return nullptr;
  }

  bool has(std::string_view key) { return value(key) != nullptr; }

  std::uint64_t u64(std::string_view key, std::uint64_t fallback) {
    const auto *v = value(key);
    return v ? parse_u64(*v, std::string(key).c_str()) : fallback;
  }

  std::uint32_t u32(std::string_view key, std::uint32_t fallback) {
    const auto *v = value(key);
    return v ? parse_u32(*v, std::string(key).c_str()) : fallback;
  }

  /// `key`'s value as a name, or an empty string when the line omits it. A
  /// name with nothing in it is rejected here rather than by each caller,
  /// since it never means anything anywhere.
  std::string name(std::string_view key) {
    const auto *v = value(key);
    if (!v)
      return {};
    if (v->empty())
      throw std::invalid_argument("'" + std::string(key) + "=' needs a name");
    return *v;
  }

  /// The one operand written without a keyword, empty if there was none.
  const std::string &bare() {
    bare_read_ = true;
    return bare_;
  }

  /// Whatever nobody asked for. Reported together so a line with two stray
  /// operands takes one run to fix rather than two.
  void reject_unread(const std::string &op_name) const {
    std::string stray;
    for (const auto &kv : kv_)
      if (!kv.read)
        stray += (stray.empty() ? "" : ", ") + kv.key + "=";
    if (!bare_.empty() && !bare_read_)
      stray += (stray.empty() ? "" : ", ") + bare_;
    if (!stray.empty())
      throw std::invalid_argument("'" + op_name + "' does not take " + stray);
  }

private:
  struct entry {
    std::string key, value;
    bool read;
  };
  std::vector<entry> kv_;
  std::string bare_;
  bool bare_read_ = false;
};

/// Index of `name` in `names`, appending it if this is its first mention.
/// Names are interned once here so nothing downstream of parse() ever
/// compares or hashes a string.
std::uint32_t intern(std::vector<std::string> &names, const std::string &name) {
  for (std::size_t i = 0; i < names.size(); ++i)
    if (names[i] == name)
      return static_cast<std::uint32_t>(i);
  names.push_back(name);
  return static_cast<std::uint32_t>(names.size() - 1);
}

/// The default round cap for a stream that waits on a signal. A stream is
/// bounded no matter what, so a schedule whose signal never arrives fails
/// loudly with EXHAUSTED_ROUNDS instead of hanging.
constexpr std::uint32_t kDefaultUntilMaxRounds = 1000;

/// The trigger: the one operand every line must carry, and the only one that
/// is neither a keyword nor an operand of the op. `-` is `+0`, "go as soon as
/// the timeline gets here", which is what a line says when it simply follows
/// the one above.
void read_trigger(const std::string &tok, std::uint64_t tick_ns, event &e,
                  std::uint64_t &last_tick, bool &have_last_tick,
                  bool &saw_delta) {
  const bool relative = !tok.empty() && (tok[0] == '+' || tok == "-");
  const std::string number = tok == "-" ? "0" : relative ? tok.substr(1) : tok;
  if (!is_digits(number))
    throw std::invalid_argument(
        "trigger '" + tok +
        "' is neither a tick nor a '+N'/'-' offset from the previous line");

  e.trig = relative ? trigger::delta : trigger::tick;
  const std::uint64_t tick = parse_u64(number, "trigger tick");
  if (relative) {
    saw_delta = true;
  } else {
    // A delta's duration is only known at run time, so nothing here can
    // place a later absolute tick relative to it. Left alone, such a tick
    // silently resolves into the past and dispatches immediately.
    if (saw_delta)
      throw std::invalid_argument(
          "tick " + std::to_string(tick) +
          " follows a '+N'/'-' offset, whose duration is only known at run "
          "time; once a schedule goes relative it must stay relative");
    if (have_last_tick && tick < last_tick)
      throw std::invalid_argument(
          "a schedule must be written in dispatch order, but tick " +
          std::to_string(tick) + " follows tick " + std::to_string(last_tick) +
          "; ticks must be non-decreasing");
    last_tick = tick;
    have_last_tick = true;
  }
  if (tick != 0 && __builtin_mul_overflow(tick, tick_ns, &e.deadline_ns))
    throw std::invalid_argument("tick " + tok +
                                " * tick_ns overflows a 64-bit nanosecond "
                                "offset");
}

/// `session=N`, defaulting to decoder 0 -- which is every schedule that has
/// only one decoder to talk to.
std::uint64_t read_session(operands &ops,
                           const std::unordered_set<std::uint64_t> &known) {
  const std::uint64_t decoder_id = ops.u64("session", 0);
  if (!known.contains(decoder_id))
    throw std::invalid_argument("session=" + std::to_string(decoder_id) +
                                " is not present in the decoder config");
  return decoder_id;
}

/// The signal named by `key`, interned, or kNoSignal when the line omits it.
/// `signal=` raises a signal once an event's reply/acks are all collected;
/// `after=` blocks that event's own dispatch until a signal is raised;
/// `until=` is a stream waiting on one to stop early.
std::uint32_t read_signal(operands &ops, const char *key,
                          std::vector<std::string> &names) {
  const std::string name = ops.name(key);
  return name.empty() ? kNoSignal : intern(names, name);
}

/// `source=N`, which a stream or a shot boundary cannot do without.
/// Marks a `source=` value as literal syndrome bits rather than a source id.
constexpr std::string_view kLiteralPrefix = "0b";

/// `source=` says where an op's syndrome bits come from: a plain integer
/// names a registered syndrome_source; `0b<bits>` carries one round inline
/// and lands in the syndrome arena, leaving `source_id` at kNoSource -- the
/// flag every later stage reads to tell the two apart.
void read_source(operands &ops, const std::string &op_name, schedule &sched,
                 event &e) {
  const std::string value = ops.name("source");
  if (value.empty())
    throw std::invalid_argument("'" + op_name +
                                "' requires 'source=N' or 'source=0b<bits>'");

  if (value.compare(0, kLiteralPrefix.size(), kLiteralPrefix) != 0) {
    e.source_id = parse_u32(value, "source");
    // kNoSource is the "no source at all" sentinel, so it cannot also name
    // one; without this the id silently reads back as a literal.
    if (e.source_id == kNoSource)
      throw std::invalid_argument("source=" + value +
                                  " is reserved and cannot name a source");
    return;
  }

  const std::string bits = value.substr(kLiteralPrefix.size());
  if (bits.empty())
    throw std::invalid_argument("'" + op_name +
                                "' source=0b needs at least one bit");
  e.source_id = kNoSource;
  e.syndrome_offset = static_cast<std::uint32_t>(sched.syndrome_arena.size());
  append_bits(sched.syndrome_arena, bits, "source");
  e.syndrome_count = static_cast<std::uint32_t>(bits.size());
}

/// The bare 0/1 token, appended to `arena`. Which arena it lands in is the
/// caller's business
void read_bits(operands &ops, std::vector<std::uint8_t> &arena,
               std::uint32_t &offset, std::uint32_t &count, const char *what) {
  const std::string &bits = ops.bare();
  if (bits.empty())
    return;
  offset = static_cast<std::uint32_t>(arena.size());
  append_bits(arena, bits, what);
  count = static_cast<std::uint32_t>(bits.size());
}

/// Two ways of saying how wide a correction is, so they had better agree.
void check_return_width(const event &e, const std::string &op_name) {
  if (e.return_size != 0 && e.expected_count != 0 &&
      e.return_size != e.expected_count)
    throw std::invalid_argument(
        "'" + op_name + "' return_size=" + std::to_string(e.return_size) +
        " does not match the width of the expected bits (" +
        std::to_string(e.expected_count) + ")");
}

/// `rounds=` / `min_rounds=` / `max_rounds=`, and the three ways they can
/// contradict each other or the stream's stop condition.
void read_round_bounds(operands &ops, event &e) {
  bool have_min = false, have_max = false;
  const auto *rounds = ops.value("rounds");
  if (rounds) {
    e.stream_min_rounds = e.stream_max_rounds = parse_u32(*rounds, "rounds");
    have_min = have_max = true;
  }
  if (const auto *min = ops.value("min_rounds")) {
    if (rounds)
      throw std::invalid_argument(
          "'stream' cannot combine 'rounds=' with 'min_rounds='");
    e.stream_min_rounds = parse_u32(*min, "min_rounds");
    have_min = true;
  }
  if (const auto *max = ops.value("max_rounds")) {
    if (rounds)
      throw std::invalid_argument(
          "'stream' cannot combine 'rounds=' with 'max_rounds='");
    e.stream_max_rounds = parse_u32(*max, "max_rounds");
    have_max = true;
  }

  // A stream with nothing to wait for has no arrival to stop it early, so a
  // ceiling above its floor would be a round count nothing can ever reach.
  if (e.until_signal_id == kNoSignal) {
    if (have_max && !have_min)
      throw std::invalid_argument(
          "'stream' with 'max_rounds=' but no 'until=' has nothing to stop it "
          "early -- use 'rounds=' for a fixed count");
    if (have_min && have_max && e.stream_min_rounds != e.stream_max_rounds)
      throw std::invalid_argument(
          "'stream' with 'min_rounds=' and 'max_rounds=' but no 'until=' has "
          "nothing to stop it early before max_rounds.");
    if (!have_max)
      e.stream_max_rounds = e.stream_min_rounds;
  } else if (!have_max) {
    e.stream_max_rounds = kDefaultUntilMaxRounds;
  }

  if (e.stream_min_rounds > e.stream_max_rounds)
    throw std::invalid_argument(
        "'stream' min_rounds=" + std::to_string(e.stream_min_rounds) +
        " exceeds max_rounds=" + std::to_string(e.stream_max_rounds));
  if (e.stream_max_rounds == 0)
    throw std::invalid_argument("'stream' must send at least one round");
}

} // namespace

schedule parse(std::string_view text,
               const std::vector<std::uint64_t> &known_decoder_ids,
               std::uint64_t tick_ns) {
  schedule sched;
  sched.tick_ns = tick_ns;

  std::unordered_set<std::uint64_t> known(known_decoder_ids.begin(),
                                          known_decoder_ids.end());
  for (auto id : known_decoder_ids)
    sched.decoders.push_back(id);

  std::uint64_t last_tick = 0;
  bool have_last_tick = false;
  bool saw_delta = false;

  std::istringstream stream{std::string(text)};
  std::string raw_line;
  std::size_t line_number = 0;
  while (std::getline(stream, raw_line)) {
    ++line_number;
    auto tokens = tokenize(raw_line);
    if (tokens.empty())
      continue; // blank or comment-only line

    try {
      if (tokens.size() < 2)
        throw std::invalid_argument(
            "expected '<trigger> <op> [key=value...]', got '" + raw_line + "'");

      event e;
      const std::string &op_name = tokens[1];
      operands ops(tokens.begin() + 2, tokens.end());

      read_trigger(tokens[0], tick_ns, e, last_tick, have_last_tick, saw_delta);
      e.decoder_id = read_session(ops, known);
      // Every op accepts `signal=`/`after=` -- see read_signal().
      e.signal_id = read_signal(ops, "signal", sched.signal_names);
      e.after_signal_id = read_signal(ops, "after", sched.signal_names);

      if (op_name == "reset") {
        e.op = operation::reset;
      } else if (op_name == "enqueue") {
        // The one-round spelling of `stream`, and it reads `source=` exactly
        // the same way.
        e.op = operation::stream;
        e.stream_min_rounds = e.stream_max_rounds = 1;
        read_source(ops, op_name, sched, e);
      } else if (op_name == "enqueue_data") {
        // Wire-identical to a one-round `enqueue`, and reads `source=` the
        // same way; all that differs is which end of the source it pulls.
        e.op = operation::enqueue_data;
        read_source(ops, op_name, sched, e);
      } else if (op_name == "get_corrections") {
        e.op = operation::get_corrections;
        e.return_size = ops.u32("return_size", 0);
        read_bits(ops, sched.expected_arena, e.expected_offset,
                  e.expected_count, "get_corrections expected");
        check_return_width(e, op_name);
      } else if (op_name == "stream") {
        e.op = operation::stream;
        read_source(ops, op_name, sched, e);
        e.stream_every_ticks = ops.u64("every", 1);
        std::uint64_t every_ns = 0;
        if (e.stream_every_ticks != 0 &&
            __builtin_mul_overflow(e.stream_every_ticks, tick_ns, &every_ns))
          throw std::invalid_argument(
              "'every=' * tick_ns overflows a 64-bit nanosecond offset");
        e.until_signal_id = read_signal(ops, "until", sched.signal_names);
        read_round_bounds(ops, e);
        if (e.source_id == kNoSource) {
          if (e.until_signal_id != kNoSignal)
            throw std::invalid_argument(
                "'stream' with literal syndromes cannot wait on 'until=': "
                "its round count is fixed when the schedule is written");
          if (e.stream_min_rounds != e.stream_max_rounds)
            throw std::invalid_argument(
                "'stream' with literal syndromes needs a fixed 'rounds=N', "
                "not a min/max range");
        }
        if (!ops.bare().empty())
          throw std::invalid_argument(
              "'stream' does not take a bit string: its syndromes come from "
              "'source=N' (use 'enqueue source=0b<bits>' to send literal "
              "ones), and an expected correction belongs on the "
              "'get_corrections' that reads it");
      } else {
        throw std::invalid_argument("unknown operation '" + op_name + "'");
      }

      ops.reject_unread(op_name);
      sched.events.push_back(e);
    } catch (const std::invalid_argument &ex) {
      throw std::invalid_argument("playback schedule, line " +
                                  std::to_string(line_number) + ": " +
                                  ex.what());
    }
  }

  return sched;
}

} // namespace cudaq::qec::playback
