/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DecodingSession.h"
#include "HopStats.h"
#include "../../hardware_guards.h"
#include "../decode_target.h"
#include "../realtime_decoding.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::rpc::bit_packed_bytes;
using cudaq::qec::decoding::rpc::RpcStatus;

namespace {
/// The session's decode endpoint, whichever kind it holds.  Rebuilt per use
/// rather than stored, to keep the decode_target detail (and the CUDA affinity
/// guards it includes) out of DecodingSession.h.  It is two pointers.
cudaq::qec::detail_dispatch::decode_target target_of(const DecodingSession &s) {
  if (s.graph)
    return cudaq::qec::detail_dispatch::decode_target(s.graph.get());
  return cudaq::qec::detail_dispatch::decode_target(s.dec.get());
}

/// Reserve SMs so the cooperative decode graph can become co-resident with
/// everything else occupying the GPU when it is fired device-side: the
/// persistent dispatch graph itself (1 block) plus any transport kernels. A
/// cooperative grid sized for ALL SMs deadlocks at grid.sync() the moment
/// anything else is resident -- the launch silently queues forever.
/// Overridable for rigs with more coresident kernels (e.g. GpuRoceTransceiver
/// RX/TX) via QEC_DEVICE_GRAPH_RESERVED_SMS. Shared by both create()
/// overloads below, since a task graph's captured graph needs the same
/// reservation a plain decoder's does.
int reserved_sms_from_env() {
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
  return reserved_sms;
}

using set_syndrome_mapping_fn =
    void (*)(const cudaq::qec::decoding::rpc::DeviceSyndromeMappingTable *);

/// Push the document's rows to the DEVICE-mode enqueue handler, which
/// validates every request's syndrome_mapping_id against them.
///
/// The handler's default -- nothing installed -- services id 0 alone, so a
/// document that declares only the pass-through row needs no install and this
/// returns without resolving anything.  A document that declares more must not
/// silently fall back to that default: refusing under DEVICE an id that HOST
/// accepts is the divergence this install closes.
///
/// The caller must have made the session's device current: the installer runs a
/// kernel, and the table it writes is a __device__ global, so it exists once
/// per GPU rather than once per process.
void install_device_syndrome_mapping(const SyndromeMappingTable &rows) {
  const auto table = device_syndrome_mapping_table(rows);
  if (!needs_device_syndrome_mapping_install(table))
    return;

  auto *fn = reinterpret_cast<set_syndrome_mapping_fn>(::dlsym(
      RTLD_DEFAULT, "cudaqx_qec_realtime_dispatch_set_syndrome_mapping"));
  if (!fn)
    throw std::runtime_error(
        "DecodingSession::create: a device_graph decoder whose configuration "
        "declares syndrome_mapping rows requires "
        "cudaqx_qec_realtime_dispatch_set_syndrome_mapping, which was not "
        "found via dlsym(RTLD_DEFAULT, ...).  The binary must absorb "
        "libcudaq-qec-realtime-cudevice.a and link with --export-dynamic.");

  fn(&table);
}
} // namespace
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

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
                        bool want_device_graph,
                        SyndromeMappingTable mapping_table) {
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
  // An empty map is not the identity row: it would reject id 0, which is the
  // only id anything sends today.
  if (!mapping_table.empty())
    s->mapping_table = std::move(mapping_table);
  s->dec = std::move(decoder);

  if (want_device_graph && s->dec->supports_graph_dispatch()) {
    void *gr = cudaq::qec::detail_affinity::capture_graph_pinned(
        *s->dec, reserved_sms_from_env());
    s->graph_resources =
        GraphResourcesPtr(gr, GraphResourcesDeleter{s->dec.get(), nullptr});
    // capture_graph_pinned() restores the previous device, so pin again for
    // the install kernel.
    cudaq::qec::detail_affinity::CudaDeviceGuard pin(
        s->dec->get_cuda_device_id());
    install_device_syndrome_mapping(s->mapping_table);
  }

  return s;
}

std::unique_ptr<DecodingSession>
DecodingSession::create(std::unique_ptr<cudaq::qec::decoding_task_graph> graph,
                        bool want_device_graph,
                        SyndromeMappingTable mapping_table) {
  if (!graph)
    throw std::invalid_argument("DecodingSession requires a task graph");
  // Every RPC drives the graph through its streaming API, so a graph that
  // decodes only full shots can never be served.  Fail bring-up with the
  // graph's own reason rather than streaming_result()'s logic_error below.
  if (!graph->supports_streaming())
    throw std::invalid_argument(
        "DecodingSession: task graph cannot run on the realtime streaming "
        "path: " +
        graph->streaming_incompatibility_reason());

  // Same fail-fast contract as the decoder path above: cuda_device_id()
  // throws on a node disagreement, and the guard range-checks whatever
  // device it resolved (no-op if unpinned).
  {
    cudaq::qec::detail_affinity::CudaDeviceGuard probe(
        graph->cuda_device_id().value_or(-1));
  }

  auto s = std::make_unique<DecodingSession>();
  if (!mapping_table.empty())
    s->mapping_table = std::move(mapping_table);

  if (want_device_graph && graph->supports_graph_dispatch()) {
    // graph->capture_graph() already pins internally (its single node
    // delegates to capture_graph_pinned() on its own inner decoder -- see
    // window_node::capture_graph()), so this needs no separate pin call the
    // way the decoder branch above does.
    void *gr = graph->capture_graph(reserved_sms_from_env());
    s->graph_resources =
        GraphResourcesPtr(gr, GraphResourcesDeleter{nullptr, graph.get()});
    // The capture restored the previous device, so pin again for the install
    // kernel.
    cudaq::qec::detail_affinity::CudaDeviceGuard pin(
        graph->cuda_device_id().value_or(-1));
    install_device_syndrome_mapping(s->mapping_table);
  }

  s->graph_result_ = graph->streaming_result();
  s->graph = std::move(graph);
  return s;
}

// ---------------------------------------------------------------------------
// Payload-level cores
// ---------------------------------------------------------------------------

void DecodingSession::enqueue_core(const slot::EnqueueView &req,
                                   round_type type) {
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

    const auto target = target_of(*this);
    // The volume bounds a call, not the round: both kinds of target accept any
    // chunking of it.  A graph's num_msyn_per_round() is the chunk a caller
    // should prefer, not a limit -- a window with room takes whatever fits.
    //
    // Only a decoder is budgeted.  A graph meters a chunk across its nodes and
    // fires as their windows fill, so it cannot run past a volume without a
    // decode resetting the count, and budgeting it would refuse the very
    // chunking the metering exists to accept.
    if (const auto budget = target.chunk_limit();
        budget && (accepted_syndromes > *budget ||
                   unpack_scratch_.size() > *budget - accepted_syndromes))
      throw std::invalid_argument(
          "Syndrome volume exceeds decoder measurement capacity");

    accepted_syndromes += unpack_scratch_.size();
    // Host-decoder path.  On the device_graph path, the CUDAQ device-graph
    // scheduler (cudaq_create_dispatch_graph_regular) handles
    // RX→dispatch→decode→TX entirely on the GPU; this code is never reached
    // for device_graph sessions.
    const bool did_decode = target.enqueue_syndrome(
        unpack_scratch_.data(), unpack_scratch_.size(), type);
    // Async BULK enqueue returns false after queueing. Do not call the blocking
    // streaming_result() accessor here; get_corrections owns that completion
    // boundary. Synchronous graphs can only advance when enqueue returned true.
    if (!target.is_graph() || did_decode || type == round_type::FINAL) {
      const auto update =
          target.capture_result_update(did_decode, graph_result_);
      if (update.operation_completed) {
        shot_state = ShotState::result_ready;
        decode_count.fetch_add(update.completed_results,
                               std::memory_order_relaxed);
        accepted_syndromes = 0;
      }
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
    const auto target = target_of(*this);
    const auto return_size = static_cast<size_t>(return_size_arg);
    if (return_size != target.num_observables()) {
      ++error_count;
      return RpcStatus::BAD_REQUEST;
    }
    // Only an aggregate completed since the last snapshot makes a graph ready:
    // once enqueue has started a new volume, the snapshot is the previous
    // shot's result, which a decoder session would report as NOT_READY.
    if (target.is_graph()) {
      const auto update = target.capture_result_update(false, graph_result_);
      if (update.operation_completed) {
        shot_state = ShotState::result_ready;
        decode_count.fetch_add(update.completed_results,
                               std::memory_order_relaxed);
        accepted_syndromes = 0;
      }
    }
    if (shot_state != ShotState::result_ready)
      return RpcStatus::NOT_READY;
    const uint8_t *corrections =
        target.published_corrections(graph_result_.values.data());
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
      target.clear_corrections();
      graph_result_ =
          graph ? graph->streaming_result() : task_graph_streaming_result{};
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
    target_of(*this).reset();
    accepted_syndromes = 0;
    graph_result_ =
        graph ? graph->streaming_result() : task_graph_streaming_result{};
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

  // Resolve the round's shape before anything touches decoder state.  Kept
  // here rather than in enqueue_core: that returns void, so a failure there
  // could only surface as INTERNAL_ERROR at the next get_corrections, and a
  // client that gets a synchronous rejection today would be told its bad
  // enqueue succeeded.
  const auto mapping_it = mapping_table.find(req.syndrome_mapping_id);
  if (mapping_it == mapping_table.end()) {
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
      target_of(*this).pin_device_cached();
      capture_syndromes(req);
      enqueue_core(req, mapping_it->second.type);
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
    target_of(*this).pin_device_cached();
    // The corrections pack straight into the tx slot's payload area; the
    // header (and the magic, last) follow in commit().  result-too-large is
    // detected by the core against the slot capacity — no truncation.
    status = get_corrections_core(req.return_size, req.reset,
                                  static_cast<uint8_t *>(tx_slot) +
                                      sizeof(RPCResponse),
                                  slot_size - sizeof(RPCResponse), out_len);
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
    target_of(*this).pin_device_cached();
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
