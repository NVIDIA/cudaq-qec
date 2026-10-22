/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "RpcSlot.h"
#include "SyndromeMappingTable.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoding_task_graph.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace cudaq::qec::decoding_server {

/// RAII wrapper: calls decoder::release_decode_graph() (decoder session) or
/// decoding_task_graph::release_graph() (task graph session) on destruction.
/// Exactly one owner is set, matching whichever create() overload built the
/// session -- same one-of-two-pointers shape as decode_target.
struct GraphResourcesDeleter {
  cudaq::qec::decoder *decoder_owner = nullptr;
  cudaq::qec::decoding_task_graph *graph_owner = nullptr;
  void operator()(void *p) const noexcept {
    if (!p)
      return;
    if (decoder_owner)
      decoder_owner->release_decode_graph(p);
    else if (graph_owner)
      graph_owner->release_graph(p);
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

  // -- Decode endpoint and GPU resources --
  //
  // Exactly one of `dec` / `graph` is set; the handlers reach whichever it is
  // through a decode_target (including the CUDA device pin -- a task graph's
  // inner decoder can still be GPU-affine, so the pin is not `dec`-only).  A
  // graph session leaves `dec` and `graph_resources` null, so anything that
  // wants the decoder itself for something a decode_target does not expose
  // (e.g. a captured graph) must check `dec`.
  std::unique_ptr<cudaq::qec::decoder> dec;
  std::unique_ptr<cudaq::qec::decoding_task_graph> graph;
  GraphResourcesPtr graph_resources;

  // Wire-compatibility state for the current shot. result_ready is set only
  // by a completed operation -- a decoder's enqueue returning true, or a
  // graph's aggregate completing, which freezes it into graph_result_ -- and
  // any later enqueue returns to collecting. It remains independent of
  // decoder_result::converged.
  ShotState shot_state = ShotState::collecting;
  size_t accepted_syndromes = 0;

  // id -> round shape.  Never empty: create() installs the pass-through BULK
  // row under id 0 when a configuration declares none, so a default-built
  // session accepts the id every existing client sends.
  SyndromeMappingTable mapping_table = default_syndrome_mapping_table();

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

  /// Construct a session around an already configured decoder.  Probes the
  /// decoder's CUDA device pin so an unhonorable pin fails server bring-up,
  /// not the first RPC.
  ///
  /// @param want_device_graph  Capture graph resources when the decoder
  ///        supports it.  Only device_graph dispatch should ask for this: a
  ///        decoder capable of graph dispatch in principle (e.g.
  ///        nv-qldpc-decoder unconditionally claims it) may still lack the
  ///        device-only settings capture needs (relay-BP composition, an O
  ///        matrix, no OSD) when it's actually configured for HOST dispatch
  ///        -- capturing anyway would fail session creation for a config that
  ///        never asked for DEVICE mode.  Host-dispatch sessions pass
  ///        `false`: they are served inline on the dispatcher thread and
  ///        never launch the graph, so a decoder whose graph capture cannot
  ///        be honored in this process (e.g. a prebuilt plugin whose
  ///        device-side handlers are not linked into this binary) still
  ///        serves host dispatch.  Defaults to true so a caller with no
  ///        dispatch concept of its own (tests, programmatic use) keeps the
  ///        original eager-capture-when-possible behavior.
  static std::unique_ptr<DecodingSession>
  create(std::unique_ptr<cudaq::qec::decoder> decoder,
         bool want_device_graph = true,
         SyndromeMappingTable mapping_table = default_syndrome_mapping_table());

  /// Same, for an already built task graph.  No graph capture of its own
  /// beyond the same want_device_graph rule above: a graph runs on the
  /// calling thread. Still probes cuda_device_id(), since an inner decoder
  /// can be GPU-affine.
  static std::unique_ptr<DecodingSession>
  create(std::unique_ptr<cudaq::qec::decoding_task_graph> graph,
         bool want_device_graph = true,
         SyndromeMappingTable mapping_table = default_syndrome_mapping_table());

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
  void enqueue_core(const slot::EnqueueView &req, round_type type);

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
  /// Last aggregate graph publication retained for correction polling.
  cudaq::qec::task_graph_streaming_result graph_result_;
};

/// High-water mark of sessions simultaneously executing requests across all
/// sessions in this process (concurrency evidence for multi-logical-qubit
/// tests and server stats).
uint64_t max_concurrent_busy_sessions();

} // namespace cudaq::qec::decoding_server
