/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "RpcSlot.h"
#include "cudaq/qec/decoder.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace cudaq::qec::decoding_server {

/// RAII wrapper: calls decoder::release_decode_graph() on destruction.
struct GraphResourcesDeleter {
  cudaq::qec::decoder *owner = nullptr;
  void operator()(void *p) const noexcept {
    if (p && owner)
      owner->release_decode_graph(p);
  }
};
using GraphResourcesPtr = std::unique_ptr<void, GraphResourcesDeleter>;

/// One decoder plus the server-side state and handlers needed to answer its
/// RPCs on whatever thread they arrive.
///
/// There is no thread, no queue, and no lock here: requests execute INLINE on
/// the transport dispatcher thread that delivered them (handle_*), exactly
/// like the legacy direct path (host::enqueue_syndromes).  Sessions are
/// single-threaded by topology — one ring per decoder, and a shared ring
/// serves its decoders from one dispatcher thread — enforced by a debug-only
/// owner assert (owner_thread below).
struct DecodingSession {
  enum class ShotState { collecting, result_ready, failed };

  // -- Decoder and GPU resources --
  std::unique_ptr<cudaq::qec::decoder> dec;
  GraphResourcesPtr graph_resources;

  // Session-owned state for the current shot. result_ready means a decode
  // call completed; it is deliberately independent of decoder_result::
  // converged.
  ShotState shot_state = ShotState::collecting;
  size_t accepted_syndromes = 0;

  // Per-session metrics (atomics: read cross-thread by the stats hooks).
  std::atomic<uint64_t> enqueue_count{0};
  std::atomic<uint64_t> decode_count{0};
  std::atomic<uint64_t> get_corrections_count{0};
  std::atomic<uint64_t> reset_count{0};
  std::atomic<uint64_t> error_count{0};
  std::atomic<uint64_t> busy_count{0};

  /// Single-caller tripwire, armed in debug builds only (zero release-build
  /// cost — the checks compile out; the member stays so the struct layout is
  /// NDEBUG-independent).  Counts handlers currently executing on this
  /// session: a second concurrent entry means two rings are feeding one
  /// decoder — an invalid configuration — and asserts loudly instead of
  /// racing session state silently.  Deliberately NOT thread-identity based:
  /// serial handoff between dispatcher threads (e.g. a channel restart after
  /// a timeout) is legal.
  std::atomic<uint32_t> debug_in_flight{0};

  DecodingSession() = default;
  DecodingSession(const DecodingSession &) = delete;
  DecodingSession &operator=(const DecodingSession &) = delete;
  DecodingSession(DecodingSession &&) = delete;
  DecodingSession &operator=(DecodingSession &&) = delete;

  /// Construct a session around an already configured decoder and, when
  /// `capture_graph` is set and the decoder supports graph dispatch, capture
  /// its device graph resources.  Host-dispatch sessions pass `false`: they
  /// are served inline on the dispatcher thread and never launch the graph,
  /// so a decoder whose graph capture cannot be honored in this process
  /// (e.g. a prebuilt plugin whose device-side handlers are not linked into
  /// this binary) still serves host dispatch.  Probes the decoder's CUDA
  /// device pin so an unhonorable pin fails server bring-up, not the first
  /// RPC.
  static std::unique_ptr<DecodingSession>
  create(std::unique_ptr<cudaq::qec::decoder> decoder,
         bool capture_graph = true);

  // -- Inline HOST_CALL path (CUDAQ dispatcher thread) --
  //
  // Serve one RPC entirely on the calling thread: parse rx_slot in place,
  // run the decoder, write the RPCResponse into tx_slot (magic release-
  // stored last, so the CUDAQ dispatcher can publish the tx flag the moment
  // the handler returns).  Mirrors the legacy direct path
  // (host::enqueue_syndromes) step for step: validate, capture, pin,
  // decoder call.  Never throws; never blocks on another thread.  A decode
  // triggered by the volume-completing enqueue runs HERE and blocks only
  // this session's ring.

  void handle_enqueue(const void *rx_slot, void *tx_slot,
                      std::size_t slot_size) noexcept;
  void handle_get_corrections(const void *rx_slot, void *tx_slot,
                              std::size_t slot_size) noexcept;
  void handle_reset(const void *rx_slot, void *tx_slot,
                    std::size_t slot_size) noexcept;

  // -- Payload-level cores (the single implementation of the decoder-facing
  //    logic; handle_* parse/validate the slot and delegate here) --

  /// Core of enqueue_syndromes.  Fire-and-forget contract: never produces a
  /// response; failures latch shot_state = failed and surface as
  /// INTERNAL_ERROR at this decoder's next get_corrections.  The caller
  /// validates payload bounds and the syndrome mapping id.
  void enqueue_core(const slot::EnqueueView &req);

  /// Core of get_corrections.  On OK, packs ceil(return_size/8) LSB-first
  /// correction bytes into \p out (capacity \p out_capacity) and sets
  /// \p out_len; a result too large for \p out fails with INTERNAL_ERROR
  /// (truncation would advertise bytes that were never written).
  cudaq::qec::decoding::rpc::RpcStatus
  get_corrections_core(int64_t return_size, bool reset, uint8_t *out,
                       std::size_t out_capacity, std::size_t &out_len);

  /// Core of reset_decoder.
  cudaq::qec::decoding::rpc::RpcStatus reset_core();

private:
  /// Reused byte-per-bit unpack buffer: no per-call allocation once grown.
  std::vector<uint8_t> unpack_scratch_;
};

/// High-water mark of sessions simultaneously executing requests across all
/// sessions in this process (concurrency evidence for multi-logical-qubit
/// tests and server stats).
uint64_t max_concurrent_busy_sessions();

} // namespace cudaq::qec::decoding_server
