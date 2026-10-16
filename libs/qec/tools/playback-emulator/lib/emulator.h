/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file emulator.h
/// @brief Top-level entry points: parse() -> plan() -> run()
/// Callable identically from the CLI tool and the Python binding.

#include "session.h"
#include "syndrome_source.h"
#include "types.h"

#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cudaq::qec::playback {

/// Parse a line-oriented playback description into a `schedule`. A line is
/// `<trigger> <op> [key=value...]`: the trigger is a tick (`deadline_ns =
/// tick*tick_ns`), a run-time-resolved `+N` delta, or `-` for `+0`; `session=N`
/// (default 0) must be in `known_decoder_ids`. Throws std::invalid_argument.
schedule parse(std::string_view text,
               const std::vector<std::uint64_t> &known_decoder_ids,
               std::uint64_t tick_ns);

struct run_params {
  std::uint64_t lead_in_ns = 20'000'000; // 20 ms before t0
  // How long close() waits, after dispatch ends, for acks still in flight
  // before giving up on them and recording INTERNAL_ERROR.
  std::uint64_t ack_drain_timeout_ns = 1'000'000'000;
  // A bad stream/enqueue_data ack aborts the run, same as every other op,
  // unless the peer is known not to ack enqueue faithfully.
  bool collect_enqueue_acks = true;
};

/// One pre-serialized round: where its frame sits in the run_plan's frame
/// arena, and where the syndrome bits that frame carries sit in the
/// schedule's. `bits_count` is 0 for the ops that send no syndromes.
struct round_plan {
  std::uint32_t frame_offset = 0;
  std::uint32_t frame_len = 0;
  std::uint32_t bits_offset = 0;
  std::uint32_t bits_count = 0;
};

/// Per-event plan-time state: every frame whose bytes are known before t0.
/// `reset`/`get_corrections` have exactly one. A `stream` has one per round
/// only when its round count is fixed and its source is pre-drawable;
/// otherwise (streamed source, or `until=`) it has none and builds live.
using event_plan = std::vector<round_plan>;

/// Pre-planned, immediately-runnable schedule: frames
/// pre-serialized into one contiguous buffer, frame sizes validated,
/// records and log arenas pre-faulted at final size. Built by plan(),
/// consumed by run().
struct run_plan {
  schedule sched;
  std::vector<std::uint8_t> frame_arena;
  std::vector<event_plan> event_plans;
  std::unordered_map<std::uint64_t, session *> router;
  std::unordered_map<std::uint32_t, syndrome_source *> sources;
  run_params params;
  // Upper bound on the number of requests run() can issue, so its per-request
  // logs can be sized once, before t0, and appended to lock-free.
  std::uint32_t max_requests = 0;
};

/// Validate `sched` against `router`'s session frame limits and pre-build
/// everything run()'s timing loop must not do on the hot path. `router` maps
/// decoder_id -> session, `sources` maps source_id -> syndrome_source. Caller
/// keeps both alive for the returned run_plan's lifetime.
std::shared_ptr<run_plan>
plan(const schedule &sched,
     const std::unordered_map<std::uint64_t, session *> &router,
     const std::unordered_map<std::uint32_t, syndrome_source *> &sources,
     const run_params &params = {});

/// Run the plan on one timing thread, in schedule order: wait_until(t0+
/// deadline), send, record -- every RPC sends and returns without waiting,
/// while each session's own worker processes replies and raises `signal=`.
/// A hard error aborts without truncating `result.records`.
run_result run(std::shared_ptr<run_plan> plan);

/// Downstream analysis writes CSV. One row per record: identity, timings,
/// derived lateness/latency, status, rounds streamed, and the
/// syndrome/correction bits resolved from the logs (one column each, rendered
/// as a '0'/'1' string in log order).
void write_csv(const run_result &result, std::ostream &out);
std::string write_csv(const run_result &result);

} // namespace cudaq::qec::playback
