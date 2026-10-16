/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// cudaq-realtime (cqr) DeviceCallService plugin for the decoding server.
///
/// The plugin registers the three default-route RPCs (enqueue_syndromes /
/// get_corrections / reset_decoder) as CUDAQ_DISPATCH_HOST_CALL entries.
/// Each request executes ENTIRELY on the CUDAQ dispatcher thread that
/// delivered it: dispatch_rpc resolves the DecodingSession by the payload's
/// decoder_id and calls its handle_* method, which parses the rx slot in
/// place, runs the decoder inline, and writes the RPCResponse straight into
/// the tx slot.  There are no worker threads, queues, or copies — the same
/// shape as the GPU device-graph dispatch path and the legacy direct path
/// (host::enqueue_syndromes).  Decoder parallelism comes from the transport:
/// one ring per decoder means one dispatcher thread per decoder.
///
/// The decoder configuration comes from, in priority order:
///   1. the CUDAQ_QEC_DECODER_CONFIG env var (path to a multi_decoder_config
///      YAML) -- the standalone-server path;
///   2. the last multi_decoder_config passed to
///      cudaq::qec::decoding::config::configure_decoders() in this process --
///      the in-process (host_dispatch) application path.

#include "DecodingSession.h"
#include "HopStats.h"
#include "RpcSlot.h"
#include "SessionRegistry.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <iostream>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C" void cudaqx_qec_decoding_server_shutdown();
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
using cudaq::qec::decoding::rpc::RpcStatus;
using cudaq::qec::decoding_server::DecodingSession;
using cudaq::qec::decoding_server::SessionRegistry;
namespace slot = cudaq::qec::decoding_server::slot;

// The registry of decoder sessions this plugin serves.  Requests execute
// INLINE on the CUDAQ dispatcher thread that delivered them (no worker
// threads, no queues, no transceiver): dispatch_rpc resolves the session by
// the payload's decoder_id and calls its handle_* method, which parses the
// rx slot in place and writes the response into the tx slot.
//
// g_registry is published (release) only after the registry is fully
// constructed; dispatch_rpc treats null as "not serving".
static std::unique_ptr<SessionRegistry> g_registry_owner;
static std::atomic<SessionRegistry *> g_registry{nullptr};
static std::once_flag g_init_flag;

// Counts requests dispatched through this service (test hook).
static std::atomic<uint64_t> g_service_dispatch_count{0};

static void init_server() {
  auto reg = std::make_unique<SessionRegistry>();

  if (const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
      cfg && cfg[0] != '\0') {
    reg->load_from_config(std::string(cfg));
  } else if (const auto config = cudaq::qec::decoding::config::
                 last_configured_multi_decoder_config()) {
    reg->load_from_config(*config, "configure_decoders()");
  } else {
    throw std::runtime_error(
        "decoding-server config not found: set CUDAQ_QEC_DECODER_CONFIG to a "
        "multi_decoder_config YAML path, or call "
        "cudaq::qec::decoding::config::configure_decoders() before realtime "
        "initialization");
  }
  g_registry_owner = std::move(reg);
  g_registry.store(g_registry_owner.get(), std::memory_order_release);
  // In-process applications never call the explicit shutdown hook the server
  // uses; tear down at exit() so session/decoder resources are released and
  // the stats report still prints.
  std::atexit([] { cudaqx_qec_decoding_server_shutdown(); });
}

// ---------------------------------------------------------------------------
// CUDAQ handler functions — inline dispatch to DecodingSession::handle_*
// ---------------------------------------------------------------------------

// Write an error RPCResponse into tx_slot, echoing the request identity from
// rx_slot (handler-level failures must not propagate into the transport
// dispatcher loop).  Guards the slots, then delegates to slot::write_response.
constexpr int32_t kStatusHandlerException =
    static_cast<int32_t>(cudaq::qec::decoding::rpc::RpcStatus::INTERNAL_ERROR);

static void write_error_response(const void *rx_slot, void *tx_slot,
                                 std::size_t slot_size, int32_t status) {
  if (!tx_slot || !rx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
    return;
  const auto *req = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  slot::write_response(tx_slot, req->request_id, req->ptp_timestamp,
                       static_cast<RpcStatus>(status));
}

// The registry is constructed lazily on the first RPC (the in-process
// application path configures decoders AFTER the realtime channel — and
// with it this dispatch session — is created); the server path instead
// initializes eagerly at session creation via CUDAQ_QEC_DECODER_CONFIG so
// slow decoder construction happens before its READY line.
//
// From here down the request runs entirely on the calling CUDAQ dispatcher
// thread: resolve the session by the payload's decoder_id (the handler ABI
// carries no context pointer, and a shared ring may serve several decoders),
// then hand the slot pair to the session, which parses in place and writes
// the response in place.
static void dispatch_rpc(const void *rx_slot, void *tx_slot,
                         std::size_t slot_size, uint32_t function_id) {
  g_service_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  try {
    std::call_once(g_init_flag, init_server);
    // g_registry is null if init_server failed or after shutdown().
    // g_init_flag is not resettable, so call_once won't retry after shutdown.
    auto *registry = g_registry.load(std::memory_order_acquire);
    if (!registry) {
      write_error_response(rx_slot, tx_slot, slot_size,
                           kStatusHandlerException);
      return;
    }
    // Thread naming + optional QEC_PIN_DISPATCHER affinity, once per thread.
    cudaq::qec::decoding_server::hopstats::on_dispatcher_thread();
    // An unreadable slot cannot even echo a request_id; leave it to the
    // transport (same contract as before this path existed).
    if (!rx_slot || !tx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
      return;
    const auto *hdr = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (hdr->magic != cudaq::realtime::RPC_MAGIC_REQUEST) {
      write_error_response(rx_slot, tx_slot, slot_size,
                           static_cast<int32_t>(RpcStatus::BAD_REQUEST));
      return;
    }
    uint64_t decoder_id = 0;
    if (!slot::peek_decoder_id(rx_slot, slot_size, decoder_id)) {
      write_error_response(rx_slot, tx_slot, slot_size,
                           static_cast<int32_t>(RpcStatus::BAD_REQUEST));
      return;
    }
    DecodingSession *session = registry->find(decoder_id);
    if (!session) {
      write_error_response(rx_slot, tx_slot, slot_size,
                           static_cast<int32_t>(RpcStatus::INVALID_DECODER));
      return;
    }
    if (function_id == kEnqueueSyndromesFunctionId)
      session->handle_enqueue(rx_slot, tx_slot, slot_size);
    else if (function_id == kGetCorrectionsFunctionId)
      session->handle_get_corrections(rx_slot, tx_slot, slot_size);
    else if (function_id == kResetDecoderFunctionId)
      session->handle_reset(rx_slot, tx_slot, slot_size);
    else
      write_error_response(rx_slot, tx_slot, slot_size,
                           static_cast<int32_t>(RpcStatus::BAD_REQUEST));
  } catch (const std::exception &e) {
    // Log via the non-throwing cudaq::qec::error() free function, NOT the
    // CUDA_QEC_ERROR macro: the macro throws, and an exception escaping this
    // handler into the transport dispatcher loop would terminate the process
    // instead of returning the error response written below.
    cudaq::qec::error("decoding-server RPC failed: {}", e.what());
    write_error_response(rx_slot, tx_slot, slot_size, kStatusHandlerException);
  } catch (...) {
    write_error_response(rx_slot, tx_slot, slot_size, kStatusHandlerException);
  }
}

void enqueue_syndromes_host(const void *rx_slot, void *tx_slot,
                            std::size_t slot_size) {
  dispatch_rpc(rx_slot, tx_slot, slot_size, kEnqueueSyndromesFunctionId);
}

void get_corrections_host(const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  dispatch_rpc(rx_slot, tx_slot, slot_size, kGetCorrectionsFunctionId);
}

void reset_decoder_host(const void *rx_slot, void *tx_slot,
                        std::size_t slot_size) {
  dispatch_rpc(rx_slot, tx_slot, slot_size, kResetDecoderFunctionId);
}

// ---------------------------------------------------------------------------
// DeviceCallService plugin
// ---------------------------------------------------------------------------

// The schema entries below register under the SAME function IDs dispatch_rpc
// routes on: the kXFunctionId constants from decoder_rpc_wire_format.h, which
// derives them from the RPC names via cudaq::realtime::fnv1a_hash so a rename
// cannot silently desynchronize registration from routing.

constexpr int32_t kHostDispatchDeviceId = 0;
constexpr uint8_t kNoResults = 0;
constexpr uint8_t kSingleResult = 1;
constexpr uint8_t kScalarU8Size = sizeof(uint8_t);
constexpr uint8_t kScalarU64Size = sizeof(uint64_t);

// Wire argument order per decoder_server_runtime.md: fixed-size scalars
// first, the variable-length bit-packed byte array last.
constexpr std::uint8_t kEnqueueDecoderIdArg = 0;
constexpr std::uint8_t kEnqueueCounterArg = 1;
constexpr std::uint8_t kEnqueueMappingIdArg = 2;
constexpr std::uint8_t kEnqueueSyndromeBitsArg = 3;
constexpr std::uint8_t kEnqueueArgCount = 4;

constexpr std::uint8_t kGetCorrectionsDecoderIdArg = 0;
constexpr std::uint8_t kGetCorrectionsReturnSizeArg = 1;
constexpr std::uint8_t kGetCorrectionsResetArg = 2;
constexpr std::uint8_t kGetCorrectionsArgCount = 3;

constexpr std::uint8_t kResetDecoderIdArg = 0;
constexpr std::uint8_t kResetDecoderArgCount = 1;

constexpr std::uint8_t kCorrectionsResult = 0;

enum DeviceCallEntryIndex : std::size_t {
  kEnqueueSyndromesEntry,
  kGetCorrectionsEntry,
  kResetDecoderEntry,
  kDeviceCallEntryCount
};

static void set_u64(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_INT64;
  d.size_bytes = kScalarU64Size;
  d.num_elements = 1;
}

static void set_u8(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_UINT8;
  d.size_bytes = kScalarU8Size;
  d.num_elements = 1;
}

// Syndrome/correction bits cross the wire bit-packed (LSB-first), so the
// argument type is CUDAQ_TYPE_BIT_PACKED -- matching the realtime device_call
// lowering for std::vector<bool> (cudaq PR 4816) -- rather than the old
// CUDAQ_TYPE_ARRAY_UINT8 stand-in used before that lowering existed.
static void set_bit_packed(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_BIT_PACKED;
}

static void configure_entry(cudaq_function_entry_t &e, uint32_t fn_id,
                            cudaq_host_rpc_fn_t handler, uint8_t num_args,
                            uint8_t num_results) {
  e = {};
  e.handler.host_fn = handler;
  e.function_id = fn_id;
  e.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  e.schema.num_args = num_args;
  e.schema.num_results = num_results;
}

static std::array<cudaq_function_entry_t, kDeviceCallEntryCount>
make_entries() {
  std::array<cudaq_function_entry_t, kDeviceCallEntryCount> entries{};

  // enqueue_syndromes: 4-arg spec format per decoder_server_runtime.md.
  // decoder_id, counter, syndrome_mapping_id (scalars) + syndrome_bits
  // (bit_packed: element-count prefix == num_syndromes, then LSB-first bits).
  auto &eq = entries[kEnqueueSyndromesEntry];
  configure_entry(eq, kEnqueueSyndromesFunctionId, enqueue_syndromes_host,
                  kEnqueueArgCount, kNoResults);
  set_u64(eq.schema.args[kEnqueueDecoderIdArg]);
  set_u64(eq.schema.args[kEnqueueCounterArg]);
  set_u64(eq.schema.args[kEnqueueMappingIdArg]);
  set_bit_packed(eq.schema.args[kEnqueueSyndromeBitsArg]);

  // get_corrections: 3-arg spec format per decoder_server_runtime.md.
  // decoder_id (scalar) + corrections (OUT std::vector<bool>: the request
  // carries its length as return_size) + reset (scalar).
  auto &gc = entries[kGetCorrectionsEntry];
  configure_entry(gc, kGetCorrectionsFunctionId, get_corrections_host,
                  kGetCorrectionsArgCount, kSingleResult);
  set_u64(gc.schema.args[kGetCorrectionsDecoderIdArg]);
  set_u64(gc.schema.args[kGetCorrectionsReturnSizeArg]);
  set_u8(gc.schema.args[kGetCorrectionsResetArg]);
  set_bit_packed(gc.schema.results[kCorrectionsResult]);

  auto &rd = entries[kResetDecoderEntry];
  configure_entry(rd, kResetDecoderFunctionId, reset_decoder_host,
                  kResetDecoderArgCount, kNoResults);
  set_u64(rd.schema.args[kResetDecoderIdArg]);

  return entries;
}

} // namespace

/// The HOST_CALL function table this service serves: the three default-route
/// RPCs (enqueue_syndromes / get_corrections / reset_decoder) registered as
/// CUDAQ_DISPATCH_HOST_CALL entries.  `count` receives the entry count.
/// Returns nullptr (count 0) when the decoder sessions could not be built.
///
/// This is the single construction path for the table, reached two ways: the
/// standalone decoding_server process calls it directly, and the in-process
/// DeviceCallService shim (decoding_server_cqr_device_call_service.cpp) wraps
/// it in the virtual interface CUDA-Q discovers.  It deliberately names no
/// CUDA-Q type -- only cudaq-realtime's cudaq_function_entry_t -- so a server
/// built without a CUDA-Q install can reach it.
extern "C" __attribute__((visibility("default"))) const cudaq_function_entry_t *
cudaqx_qec_decoding_server_host_call_table(std::uint32_t *count) {
  // Process-lifetime storage: the dispatcher reads these entries in place for
  // as long as the channel is up.
  static std::array<cudaq_function_entry_t, kDeviceCallEntryCount> entries =
      make_entries();

  // Server path: the config path is in the environment, so build the decoder
  // sessions NOW (before the server's READY line).  The in-process
  // application path has not called configure_decoders() yet at this point;
  // it initializes lazily on the first RPC (see dispatch_rpc).
  if (const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
      cfg && cfg[0] != '\0') {
    try {
      std::call_once(g_init_flag, init_server);
    } catch (const std::exception &e) {
      cudaq::qec::error(
          "decoding-server init failed (CUDAQ_QEC_DECODER_CONFIG={}): {}", cfg,
          e.what());
      if (count)
        *count = 0;
      return nullptr;
    }
  }

  if (count)
    *count = static_cast<std::uint32_t>(entries.size());
  return entries.data();
}

extern "C" __attribute__((visibility("default"))) uint64_t
cudaqx_qec_device_call_dispatch_count() {
  return g_service_dispatch_count.load(std::memory_order_relaxed);
}

/// High-water mark of DecodingSessions simultaneously executing requests on
/// their dispatcher threads -- the server's concurrency evidence for
/// multi-logical-qubit tests.
extern "C" __attribute__((visibility("default"))) uint64_t
cudaqx_qec_decoding_server_max_concurrent() {
  return cudaq::qec::decoding_server::max_concurrent_busy_sessions();
}

/// Opaque graph resources (decoder::capture_decode_graph()) of one decoder
/// hosted by this service, or nullptr.  The decoding_server process uses
/// this to wire a device-graph ring consumer to a decoder whose sessions
/// live behind this plugin.
extern "C" __attribute__((visibility("default"))) void *
cudaqx_qec_decoding_server_graph_resources(uint64_t decoder_id) {
  auto *registry = g_registry.load(std::memory_order_acquire);
  if (!registry)
    return nullptr;
  auto *session = registry->find(decoder_id);
  if (!session || !session->graph_resources)
    return nullptr;
  return session->graph_resources.get();
}

/// Per-decoder session counters (decodes/enqueues/...), one stdout line per
/// decoder. Test/diagnostic evidence; callers gate on the
/// QEC_DECODING_SERVER_STATS environment variable.
extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_decoding_server_print_stats() {
  if (auto *registry = g_registry.load(std::memory_order_acquire)) {
    for (const auto &[id, session] : registry->sessions()) {
      std::cout << "QEC_DECODING_SERVER_DECODER_STATS id=" << id
                << " decodes=" << session->decode_count.load()
                << " enqueues=" << session->enqueue_count.load()
                << " corrections=" << session->get_corrections_count.load()
                << " resets=" << session->reset_count.load()
                << " errors=" << session->error_count.load() << std::endl;
    }
  }
  cudaq::qec::decoding_server::hopstats::report();
}

/// Tear down the decoder sessions.  Callers (the decoding_server process,
/// the atexit hook) invoke this only after the CUDAQ dispatcher threads have
/// stopped delivering requests — nothing can be mid-handle_* here.
extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_decoding_server_shutdown() {
  if (g_registry_owner) {
    g_registry.store(nullptr, std::memory_order_release);
    g_registry_owner.reset();
  }
  // Latency-probe report (QEC_DECODING_SERVER_HOP_STATS); prints once, after
  // all dispatcher threads have quiesced.
  cudaq::qec::decoding_server::hopstats::report();
}
