/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file types.h
/// @brief Playback emulator data model: `schedule` (parsed input), `record`
/// (per-event output), and `run_result` (what `run()` returns) -- flat,
/// allocation-free on the hot path.

#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::qec::playback {

/// Sentinel: no syndrome source attached to this event.
inline constexpr std::uint32_t kNoSource = ~std::uint32_t(0);

/// Sentinel for `record::status`: this event never dispatched, so it has no
/// outcome at all. Disjoint from both status spaces below (RpcStatus is
/// 0..6, stream_terminate 100..103), so it cannot be read as success.
inline constexpr std::int32_t kNoStatus = -1;

/// Sentinel: this event neither raises nor waits on a signal.
inline constexpr std::uint32_t kNoSignal = ~std::uint32_t(0);

/// The four operations a schedule line can name. `stream` sends
/// `stream_min_rounds` and returns, or continues to `stream_max_rounds`
/// unless `until=` arrives first; `enqueue` is its one-round spelling.
/// `enqueue_data` is wire-identical but marks a shot boundary, pulling a
/// source's terminal readout instead of another stabilizer round.
enum class operation : std::uint8_t {
  reset,
  stream,
  enqueue_data,
  get_corrections,
};

/// What decides when an event dispatches: the one operand every schedule line
/// must carry, written first, with no keyword. `tick` is a plain number, an
/// offset from t0. `delta` is `+N`, an offset from the completion of the
/// previous event.
enum class trigger : std::uint8_t {
  tick,
  delta,
};

inline const char *to_string(operation op) {
  switch (op) {
  case operation::reset:
    return "reset";
  case operation::stream:
    return "stream";
  case operation::enqueue_data:
    return "enqueue_data";
  case operation::get_corrections:
    return "get_corrections";
  }
  return "unknown";
}

/// Final outcome of one record: the wire `RpcStatus` for `reset`/
/// `enqueue_data`/`get_corrections`, or one of these four termination
/// reasons for `stream`. Disjoint by value (RpcStatus 0..6, this 100..103),
/// so the value alone says which; `kNoStatus` (-1) means never dispatched.
enum class stream_terminate : std::int32_t {
  OK = 100,
  SOURCE_EXHAUSTED = 101,
  EXHAUSTED_ROUNDS = 102,
  ERROR = 103,
};

inline const char *to_string(stream_terminate t) {
  switch (t) {
  case stream_terminate::OK:
    return "OK";
  case stream_terminate::SOURCE_EXHAUSTED:
    return "SOURCE_EXHAUSTED";
  case stream_terminate::EXHAUSTED_ROUNDS:
    return "EXHAUSTED_ROUNDS";
  case stream_terminate::ERROR:
    return "ERROR";
  }
  return "unknown";
}

/// One line of the parsed playback schedule. Data and timing only.
struct event {
  trigger trig = trigger::tick;
  std::uint64_t deadline_ns = 0; // an offset from t0 for `tick`, or from the
                                 // previous event's completion for `delta`
  std::uint64_t decoder_id = 0;  // which session to send to (`session=N`)
  operation op = operation::reset;
  // syndrome source for stream/enqueue_data; kNoSource otherwise.
  std::uint32_t source_id = kNoSource;
  std::uint32_t syndrome_offset = 0; // into schedule::syndrome_arena
  std::uint32_t syndrome_count = 0;
  std::uint32_t expected_offset = 0; // into schedule::expected_arena
  std::uint32_t expected_count = 0;
  std::uint32_t return_size = 0; // correction-bit width, overriding
                                 // expected_count when larger

  /// The signal this event raises once its reply/acks are all collected, an
  /// index into `schedule::signal_names`, set by `signal=NAME`.
  std::uint32_t signal_id = kNoSignal;
  /// The signal this event's dispatch waits on in addition to its own
  /// deadline, set by `after=NAME`. kNoSignal means dispatch on deadline alone.
  std::uint32_t after_signal_id = kNoSignal;

  // -- stream only; meaningless (left default) for every other op --
  /// Which signal ends this stream (`until=NAME`), or kNoSignal for a
  /// fixed-round stream.
  std::uint32_t until_signal_id = kNoSignal;
  /// Pacing, in ticks. 0 means unpaced (rounds fire as fast as the decoder
  /// accepts them); the parser defaults this to 1 tick (paced) when `every=`
  /// is omitted, and only sets it to 0 on an explicit `every=0`
  std::uint64_t stream_every_ticks = 1;
  /// The two halves of the stop rule.
  std::uint32_t stream_min_rounds = 1;
  std::uint32_t stream_max_rounds = 1;
};

/// The parsed input. Bits live in shared arenas; events hold
/// (offset, count) into them.
struct schedule {
  std::vector<event> events;                // in file order
  std::vector<std::uint8_t> syndrome_arena; // one byte per bit (0x00/0x01)
  std::vector<std::uint8_t> expected_arena; // one byte per bit
  std::vector<std::uint64_t> decoders;      // known decoder_ids
  // Interned once at parse: events carry indices into this
  std::vector<std::string> signal_names;
  std::uint64_t tick_ns = 1000; // wall-clock duration of one tick
};

/// One record per event, in event order, preallocated before t0
struct record {
  // -- identity --
  std::uint32_t event_index = 0;
  std::uint64_t decoder_id = 0;
  operation op = operation::reset;
  // True once the dispatch loop actually reached this event. False means a
  // hard error aborted the run first
  bool dispatched = false;

  // -- timing (ns, relative to t0) --
  std::uint64_t deadline_ns = 0; // where it was supposed to fire
  std::uint64_t call_ns = 0;     // when the dispatch actually began
  // When this event's last reply/ack landed: the one request's reply for
  // reset/get_corrections, or the max over a stream/enqueue_data's rounds.
  std::uint64_t return_ns = 0;

  // -- outcome --
  std::int32_t status = kNoStatus;   // RpcStatus, or stream_terminate (above)
  std::uint32_t rounds_streamed = 0; // stream only
  bool read_completed = false;       // a correction was actually consumed

  // -- data (offsets into the run's arenas) --
  std::uint32_t syndrome_offset = 0, syndrome_count = 0;     // what was SENT
  std::uint32_t correction_offset = 0, correction_count = 0; // what came BACK
  bool correction_mismatch = false; // vs. the event's expected bits

  // -- cross-referencing --
  // Every request_id this event put on the wire, in send order, as a slice of
  // run_result::request_id_log. One per RPC, so one per round for a stream and
  // exactly one for everything else. The same (offset, count) also slices
  // request_dispatch_ns_log/request_return_ns_log/request_status_log, one
  // entry per request in the same order.
  std::uint32_t request_id_offset = 0, request_id_count = 0;
};

/// What `run()` returns: the per-event records, the arenas they index into,
/// and the run-level numbers a report needs to interpret them without
/// re-running.
struct run_result {
  std::vector<record> records;
  std::vector<std::uint8_t> syndrome_log; // arena the records index into
  std::vector<std::uint8_t>
      correction_log; // arena the corrections are stored in
  std::vector<std::uint32_t> request_id_log; // Every request_id the run issued,
                                             // in the order it issued them.
  // Per-request timing/outcome, parallel to request_id_log entry-for-entry.
  // dispatch_ns is stamped by the timing thread when the request was put on
  // the wire; return_ns and status are stamped by the session's own worker
  // when that request's own reply landed (0/kNoStatus if never collected).
  std::vector<std::uint64_t> request_dispatch_ns_log;
  std::vector<std::uint64_t> request_return_ns_log;
  std::vector<std::int32_t> request_status_log;
  std::vector<std::string> warnings;
  std::uint64_t t0_ns = 0; // CLOCK_MONOTONIC value run() aligned to
  std::uint64_t tick_ns = 0;
};

} // namespace cudaq::qec::playback
