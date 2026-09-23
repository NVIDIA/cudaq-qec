/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingSession.h"
#include "HopStats.h"
#include "../../hardware_guards.h"
#include "../realtime_decoding.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::rpc::bit_packed_bytes;
using cudaq::qec::decoding::rpc::RpcStatus;
using cudaq::realtime::RPCHeader;

// Busy high-water mark across all sessions (bumped while a session executes
// a request inline on a dispatcher thread).
static std::atomic<uint64_t> g_busy_sessions{0};
static std::atomic<uint64_t> g_max_busy_sessions{0};

uint64_t max_concurrent_busy_sessions() {
  return g_max_busy_sessions.load(std::memory_order_relaxed);
}

namespace {
/// RAII busy accounting for the inline handlers: concurrency evidence for
/// the multi-logical-qubit tests.
struct BusyScope {
  BusyScope() {
    const uint64_t busy =
        g_busy_sessions.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t observed = g_max_busy_sessions.load(std::memory_order_relaxed);
    while (busy > observed && !g_max_busy_sessions.compare_exchange_weak(
                                  observed, busy, std::memory_order_relaxed))
      ;
  }
  ~BusyScope() { g_busy_sessions.fetch_sub(1, std::memory_order_relaxed); }
};
} // namespace

/// Debug-only single-caller tripwire (see debug_in_flight in the header):
/// a scoped guard that asserts no two handlers execute on one session
/// concurrently.  Scoped rather than thread-affine on purpose — a dispatcher
/// thread may legitimately be replaced across a channel restart; only
/// OVERLAP indicates the invalid two-rings-one-decoder topology.  The entire
/// check compiles out in release builds — an invalid-configuration guard
/// must not add hot-path latency.
struct SingleCallerGuard {
#ifndef NDEBUG
  explicit SingleCallerGuard(DecodingSession &s) : session(s) {
    const uint32_t prior =
        session.debug_in_flight.fetch_add(1, std::memory_order_acq_rel);
    assert(prior == 0 &&
           "DecodingSession accessed from multiple threads concurrently: "
           "invalid topology (two rings feeding one decoder?)");
    (void)prior;
  }
  ~SingleCallerGuard() {
    session.debug_in_flight.fetch_sub(1, std::memory_order_acq_rel);
  }
  DecodingSession &session;
#else
  explicit SingleCallerGuard(DecodingSession &) {}
#endif
  SingleCallerGuard(const SingleCallerGuard &) = delete;
  SingleCallerGuard &operator=(const SingleCallerGuard &) = delete;
};

std::unique_ptr<DecodingSession>
DecodingSession::create(std::unique_ptr<cudaq::qec::decoder> decoder,
                        bool capture_graph) {
  if (!decoder)
    throw std::invalid_argument("DecodingSession requires a decoder");

  // An unhonorable CUDA pin must fail server bring-up, not the first RPC:
  // the guard range-checks cuda_device_id and throws (restoring the previous
  // device on exit).  No-op for unpinned decoders (< 0).
  {
    cudaq::qec::detail_affinity::CudaDeviceGuard probe(
        decoder->get_cuda_device_id());
  }

  auto s = std::make_unique<DecodingSession>();
  s->dec = std::move(decoder);

  if (capture_graph && s->dec->supports_graph_dispatch()) {
    // Reserve SMs so the cooperative decode graph can become co-resident
    // with everything else occupying the GPU when it is fired device-side:
    // the persistent dispatch graph itself (1 block) plus any transport
    // kernels.  A cooperative grid sized for ALL SMs deadlocks at
    // grid.sync() the moment anything else is resident -- the launch
    // silently queues forever.  Overridable for rigs with more coresident
    // kernels (e.g. GpuRoceTransceiver RX/TX) via
    // QEC_DEVICE_GRAPH_RESERVED_SMS.
    int reserved_sms = 1;
    if (const char *env = std::getenv("QEC_DEVICE_GRAPH_RESERVED_SMS")) {
      char *end = nullptr;
      long v = std::strtol(env, &end, 10);
      // A malformed, zero, or negative override would reinstate the
      // reserve-all-SMs behavior this fix exists to prevent (atoi silently
      // yields 0 for junk), so accept only a fully-parsed value >= 1 and keep
      // the safe floor otherwise.
      if (end != env && *end == '\0' && v >= 1)
        reserved_sms = static_cast<int>(v);
      else
        cudaq::qec::warn("QEC_DEVICE_GRAPH_RESERVED_SMS='{}' is not a positive "
                         "integer; keeping reserved_sms=1",
                         env);
    }
    void *gr = cudaq::qec::detail_affinity::capture_graph_pinned(*s->dec,
                                                                 reserved_sms);
    s->graph_resources =
        GraphResourcesPtr(gr, GraphResourcesDeleter{s->dec.get()});
  }

  return s;
}

// ---------------------------------------------------------------------------
// Payload-level cores
// ---------------------------------------------------------------------------

void DecodingSession::enqueue_core(const slot::EnqueueView &req) {
  // Once processing has failed, accepting more rounds would make the shot's
  // measurement history unknowable. Only a full reset can establish a clean
  // epoch again.
  if (shot_state == ShotState::failed)
    return;

  // TODO: add byte-packed compat path once compiler lowering PR lands.
  // Unpack bit-packed syndromes to byte-per-bit for the decoder.
  unpack_scratch_.resize(static_cast<size_t>(req.num_syndromes));
  for (uint64_t i = 0; i < req.num_syndromes; ++i)
    unpack_scratch_[i] = (req.packed_bits[i / 8] >> (i % 8)) & 1u;

  try {
    // Any accepted input after a completed decode starts a new volume; the old
    // correction vector must not be reported as the result of that volume.
    shot_state = ShotState::collecting;

    const size_t expected_syndromes = dec->get_num_msyn_per_decode();
    if (accepted_syndromes > expected_syndromes ||
        unpack_scratch_.size() > expected_syndromes - accepted_syndromes)
      throw std::invalid_argument(
          "Syndrome volume exceeds decoder measurement capacity");

    accepted_syndromes += unpack_scratch_.size();
    // Host-decoder path.  On the device_graph path, the CUDAQ device-graph
    // scheduler (cudaq_create_dispatch_graph_regular) handles
    // RX→dispatch→decode→TX entirely on the GPU; this code is never reached
    // for device_graph sessions.
    const bool did_decode =
        dec->enqueue_syndrome(unpack_scratch_.data(), unpack_scratch_.size());

    if (did_decode) {
      ++decode_count;
      accepted_syndromes = 0;
      shot_state = ShotState::result_ready;
    }
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::enqueue_core: {}", e.what());
    ++error_count;
    // Fire-and-forget: no response carries this failure, so latch it and
    // surface it until the client establishes a clean epoch with reset.
    shot_state = ShotState::failed;
  }
}

RpcStatus DecodingSession::get_corrections_core(int64_t return_size_arg,
                                                bool reset, uint8_t *out,
                                                std::size_t out_capacity,
                                                std::size_t &out_len) {
  out_len = 0;

  // Spec validation: return_size (the OUT std::vector<bool> length) must be
  // positive.
  if (return_size_arg <= 0) {
    ++error_count;
    return RpcStatus::BAD_REQUEST;
  }

  // Surface a sticky deferred enqueue failure from this shot. Reporting it
  // does not make partially accumulated decoder state safe to reuse.
  if (shot_state == ShotState::failed)
    return RpcStatus::INTERNAL_ERROR;

  try {
    const auto return_size = static_cast<size_t>(return_size_arg);
    if (return_size != dec->get_num_observables()) {
      ++error_count;
      return RpcStatus::BAD_REQUEST;
    }
    if (shot_state != ShotState::result_ready)
      return RpcStatus::NOT_READY;
    const uint8_t *corrections = dec->get_obs_corrections();
    if (!corrections) {
      shot_state = ShotState::failed;
      return RpcStatus::INTERNAL_ERROR;
    }
    // result_len = ceil(R/8) exactly per decoder_server_runtime.md spec.
    // The spec forbids trailing padding in the wire result_len; if a transport
    // layer needs 8-byte alignment, it must add padding in its own framing.
    const size_t result_len = bit_packed_bytes(return_size);
    // Truncating would advertise bytes that were never written, so the client
    // would read stale memory as correction bits.  Fail the RPC explicitly
    // (the pre-decoding-server code returned result-buffer-too-small here).
    if (!out || result_len > out_capacity)
      return RpcStatus::INTERNAL_ERROR;
    // get_obs_corrections() returns byte-per-bit; pack into the wire format.
    std::memset(out, 0, result_len);
    for (size_t i = 0; i < return_size; ++i) {
      if (corrections[i] & 1u)
        out[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    if (reset) {
      // clear_corrections (not a full reset_decoder): matches the host-path
      // semantics of get_corrections(reset=true).  Runs BEFORE the OK is
      // reported: `out` already owns a copy of the correction bits, and a
      // throw here must produce the single INTERNAL_ERROR status below, not
      // a second response after an already-delivered OK.
      dec->clear_corrections();
      shot_state = ShotState::collecting;
    }
    out_len = result_len;
    return RpcStatus::OK;
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::get_corrections_core: {}", e.what());
    ++error_count;
    shot_state = ShotState::failed;
    return RpcStatus::INTERNAL_ERROR;
  }
}

RpcStatus DecodingSession::reset_core() {
  try {
    dec->reset_decoder();
    accepted_syndromes = 0;
    shot_state = ShotState::collecting;
    return RpcStatus::OK;
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::reset_core: {}", e.what());
    ++error_count;
    shot_state = ShotState::failed;
    return RpcStatus::INTERNAL_ERROR;
  }
}

// ---------------------------------------------------------------------------
// Inline HOST_CALL path — one thread, zero copies, response written in place
// ---------------------------------------------------------------------------

/// --save_syndrome capture, same semantics as the legacy path's
/// capture_syndromes() in host::enqueue_syndromes: record what is submitted
/// for decode, repacked MSB-first — byte-identical to the legacy
/// saved-syndrome format (the wire carries LSB-first bits).
static void capture_syndromes(const slot::EnqueueView &req) {
  auto callback = cudaq::qec::decoding::host::_get_syndrome_capture_callback();
  if (!callback)
    return;
  std::vector<uint8_t> packed(req.byte_count, 0);
  for (uint64_t i = 0; i < req.num_syndromes; ++i)
    if ((req.packed_bits[i / 8] >> (i % 8)) & 1u)
      packed[i / 8] |= static_cast<uint8_t>(1u << (7 - (i % 8)));
  callback(packed.data(), packed.size());
}

void DecodingSession::handle_enqueue(const void *rx_slot, void *tx_slot,
                                     std::size_t slot_size) noexcept {
  SingleCallerGuard serial_guard(*this);
  ++enqueue_count;
  hopstats::StageScope stats(
      cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId);

  // Mirrors host::enqueue_syndromes (realtime_decoding.cpp): validate, then
  // pin, then capture, then hand the decoder the bits — on this thread.
  slot::EnqueueView req;
  if (!slot::parse_enqueue(rx_slot, slot_size, req)) {
    // The caller (dispatch_rpc) validated the header, so echoing it is safe.
    const auto *hdr = static_cast<const RPCHeader *>(rx_slot);
    ++error_count;
    shot_state = ShotState::failed;
    slot::write_response(tx_slot, hdr->request_id, hdr->ptp_timestamp,
                         RpcStatus::BAD_REQUEST);
    return;
  }
  const uint32_t rid = req.header->request_id;
  const uint64_t ptp = req.header->ptp_timestamp;

  // The wire carries a syndrome mapping id (decoder_server_runtime.md), but
  // the only mapping this server has ever honored is the identity mapping
  // (id 0).  Reject anything else rather than silently decoding as if the
  // mapping were identity; non-identity mappings re-enter via the
  // identity-aware decoder API when they become configurable.
  if (req.syndrome_mapping_id != 0) {
    cudaq::qec::error(
        "DecodingSession::handle_enqueue: unknown syndrome_mapping_id {}",
        req.syndrome_mapping_id);
    ++error_count;
    shot_state = ShotState::failed;
    slot::write_response(tx_slot, rid, ptp, RpcStatus::BAD_REQUEST);
    return;
  }
  stats.parsed(rid);

  {
    BusyScope busy;
    try {
      cudaq::qec::detail_affinity::pin_decode_device_cached(*dec);
      capture_syndromes(req);
      enqueue_core(req);
    } catch (const std::exception &e) {
      // enqueue_core contains its own try/catch; this guards the pin and the
      // capture hook.  Fire-and-forget contract: latch, respond OK below,
      // surface at the next get_corrections.
      cudaq::qec::error("DecodingSession::handle_enqueue: {}", e.what());
      ++error_count;
      shot_state = ShotState::failed;
    } catch (...) {
      cudaq::qec::error("DecodingSession::handle_enqueue: non-std exception");
      ++error_count;
      shot_state = ShotState::failed;
    }
  }
  stats.decoded();

  // Single response write, at the tail: the tx flag publishes only when this
  // handler returns, so an earlier write would not reach the client sooner.
  // OK certifies ingestion (deferred-error contract unchanged); a decode
  // failure latched above surfaces at the next get_corrections.
  slot::write_response(tx_slot, rid, ptp, RpcStatus::OK);
}

void DecodingSession::handle_get_corrections(const void *rx_slot, void *tx_slot,
                                             std::size_t slot_size) noexcept {
  SingleCallerGuard serial_guard(*this);
  ++get_corrections_count;
  hopstats::StageScope stats(
      cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId);

  slot::GetCorrectionsView req;
  if (!slot::parse_get_corrections(rx_slot, slot_size, req)) {
    const auto *hdr = static_cast<const RPCHeader *>(rx_slot);
    ++error_count;
    slot::write_response(tx_slot, hdr->request_id, hdr->ptp_timestamp,
                         RpcStatus::BAD_REQUEST);
    return;
  }
  const uint32_t rid = req.header->request_id;
  const uint64_t ptp = req.header->ptp_timestamp;
  stats.parsed(rid);

  BusyScope busy;
  slot::ResultWriter writer(tx_slot, slot_size);
  std::size_t out_len = 0;
  RpcStatus status;
  try {
    cudaq::qec::detail_affinity::pin_decode_device_cached(*dec);
    // The corrections pack straight into the tx slot's payload area; the
    // header (and the magic, last) follow in commit().  result-too-large is
    // detected by the core against the slot capacity — no truncation.
    const std::size_t out_capacity = writer.payload_capacity();
    status = get_corrections_core(req.return_size, req.reset,
                                  writer.payload(out_capacity), out_capacity,
                                  out_len);
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::handle_get_corrections: {}", e.what());
    ++error_count;
    shot_state = ShotState::failed;
    status = RpcStatus::INTERNAL_ERROR;
  } catch (...) {
    cudaq::qec::error(
        "DecodingSession::handle_get_corrections: non-std exception");
    ++error_count;
    shot_state = ShotState::failed;
    status = RpcStatus::INTERNAL_ERROR;
  }
  stats.decoded();
  writer.commit(status, rid, ptp, status == RpcStatus::OK ? out_len : 0);
}

void DecodingSession::handle_reset(const void *rx_slot, void *tx_slot,
                                   std::size_t slot_size) noexcept {
  SingleCallerGuard serial_guard(*this);
  ++reset_count;
  hopstats::StageScope stats(
      cudaq::qec::decoding::rpc::kResetDecoderFunctionId);

  slot::ResetView req;
  if (!slot::parse_reset(rx_slot, slot_size, req)) {
    const auto *hdr = static_cast<const RPCHeader *>(rx_slot);
    ++error_count;
    slot::write_response(tx_slot, hdr->request_id, hdr->ptp_timestamp,
                         RpcStatus::BAD_REQUEST);
    return;
  }
  stats.parsed(req.header->request_id);

  BusyScope busy;
  RpcStatus status;
  try {
    cudaq::qec::detail_affinity::pin_decode_device_cached(*dec);
    status = reset_core();
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::handle_reset: {}", e.what());
    ++error_count;
    shot_state = ShotState::failed;
    status = RpcStatus::INTERNAL_ERROR;
  } catch (...) {
    cudaq::qec::error("DecodingSession::handle_reset: non-std exception");
    ++error_count;
    shot_state = ShotState::failed;
    status = RpcStatus::INTERNAL_ERROR;
  }
  stats.decoded();
  slot::write_response(tx_slot, req.header->request_id,
                       req.header->ptp_timestamp, status);
}

} // namespace cudaq::qec::decoding_server
