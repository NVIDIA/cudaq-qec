/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file session.h
/// @brief The `session` interface and its backend factories. A session
/// carries an RPC frame (RPCHeader + payload, + bit-packed syndrome bytes
/// for `enqueue`) to a decoder and brings a reply back. Concrete classes are
/// private to their .cpp files; only the factories below are public.

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::playback {

using cudaq::qec::decoding::rpc::RpcStatus;

inline const char *to_string(RpcStatus status) {
  switch (status) {
  case RpcStatus::OK:
    return "OK";
  case RpcStatus::INVALID_DECODER:
    return "INVALID_DECODER";
  case RpcStatus::BAD_REQUEST:
    return "BAD_REQUEST";
  case RpcStatus::INTERNAL_ERROR:
    return "INTERNAL_ERROR";
  case RpcStatus::NOT_READY:
    return "NOT_READY";
  case RpcStatus::BUSY:
    return "BUSY";
  case RpcStatus::SYNDROMES_DROPPED:
    return "SYNDROMES_DROPPED";
  }
  return "unknown";
}

/// A pre-serialized RPC request: RPCHeader + payload (+ trailing bit-packed
/// syndrome bytes for `enqueue`). Non-owning -- the bytes live in the
/// run_plan's frame arena, built once before t0.
struct frame {
  const std::uint8_t *bytes = nullptr;
  std::size_t size = 0;
};

/// Identifies a reply without going through a request_id lookup: an index
/// into `schedule::events` and into `run_result::request_*_log`.
struct tag {
  std::uint32_t event = 0;
  std::uint32_t log_index = 0;
};

/// Owns every request's outcome for one run() call: emulator.cpp defines it
/// and is the only thing that ever constructs one. Opaque here -- a session
/// only ever hands one back to handle_reply()/handle_event_done() below, the
/// one collector every session backend reports to.
struct run_ctx;

/// `run()`'s reply/event_done handlers (emulator.cpp), called directly by a
/// session's worker thread, never concurrently with each other for the same
/// session -- see session::start. `return_ns` is a CLOCK_MONOTONIC absolute
/// timestamp; `reply`/`reply_len` are only valid for the duration of the call.
void handle_reply(run_ctx &collector, tag t, RpcStatus status,
                  const std::uint8_t *reply, std::size_t reply_len,
                  std::uint64_t return_ns);
void handle_event_done(run_ctx &collector, std::uint32_t event,
                       std::uint32_t issued, std::int32_t term, bool has_term);

/// Anything that can carry an RPC frame to a decoder and bring a reply back.
/// The timing thread calls `send`/`event_done`; a session's own worker
/// produces and processes every reply, reporting it to `collector`.
/// LIFETIME: no thread may be inside any method when destroyed.
class session {
public:
  virtual ~session() = default;

  /// Records `collector` and starts the session's worker. Called once,
  /// before t0; `collector` outlives every session used in the same run().
  virtual void start(run_ctx &collector) = 0;

  /// Publish a frame. Returns as soon as it is queued/sent -- never waits on
  /// the decoder. Timing thread only. `f.bytes` may be freed once this
  /// returns.
  virtual void send(const frame &f, tag t) = 0;

  /// The event named by `event` has issued its last request (`issued`
  /// requests total). `term`/`has_term` carry a stream's termination reason,
  /// for handle_event_done() to forward. Timing thread only.
  virtual void event_done(std::uint32_t event, std::uint32_t issued,
                          std::int32_t term, bool has_term) = 0;

  /// Waits up to `drain` for outstanding replies (reporting the rest via
  /// handle_reply() with INTERNAL_ERROR), then stops the worker. No call into
  /// the collector happens after this returns.
  virtual void stop(std::chrono::nanoseconds drain) = 0;

  /// The largest frame this session can carry, or 0 for unbounded. Set once
  /// at construction and validated against the schedule before t0, so a
  /// session that cannot carry what the schedule asks for is a startup error
  /// rather than a runtime surprise.
  std::uint32_t max_frame_bytes = 0;
};

// -- Backend factories. The concrete session classes live in anonymous
// namespaces in their own .cpp files

/// Discards everything, but still builds and serializes a full frame per
/// event and checksums it so the compiler cannot elide the work.
std::unique_ptr<session> make_null_session();

/// One `make_null_session()` per decoder_id -- each decoder dispatches on
/// its own thread, so sharing one instance across decoder_ids is never
/// safe. Same shape as make_inproc_sessions()/make_udp_sessions() below.
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_null_sessions(const std::vector<std::uint64_t> &decoder_ids);

/// Realizes the decoders named in `config` in this process, one session per
/// decoder, each dispatching directly to that decoder's own DecodingSession
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_inproc_sessions(
    const cudaq::qec::decoding::config::multi_decoder_config &config);

/// Connected UDP client session(s) to a decoding server speaking the wire
/// format in decoder_rpc_wire_format.h. `endpoints` maps decoder_id ->
/// "host:port"; one session per decoder_id
/// `timeout_ms` bounds how long any one request waits for its own reply.
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_udp_sessions(
    const std::unordered_map<std::uint64_t, std::string> &endpoints,
    std::uint32_t timeout_ms = 200);

/// Points `router[id]` at each session's owning pointer, for a
/// make_*_sessions() result the caller is keeping alive elsewhere. The
/// shared last step of adopting any backend's sessions into a run.
void route_sessions(
    const std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
        &sessions,
    std::unordered_map<std::uint64_t, session *> &router);

} // namespace cudaq::qec::playback
