/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "DecodingServer.h"
#include "DecodingSession.h"
#include "RpcSlot.h"
#include "../lib/hardware_guards.h"
#include "../lib/realtime/realtime_decoding.h"

#include "SessionRegistry.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace cudaq::qec::decoding_server;
using namespace cudaq::qec::decoding::rpc;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

class ControlledDecoder final : public cudaq::qec::decoder {
public:
  // device: sets cuda_device_id_ directly, skipping the real pin
  // decoder::get() would need a physical GPU for. Default -1 = unpinned.
  explicit ControlledDecoder(int device = -1)
      : decoder(
            cudaq::qec::decoder_init(
                /*H=*/cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1},
                                                                 {0}),
                /*O=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}),
                /*error_rates=*/{},
                // One detector is the parity of two incoming measurement
                // bits, so a decode completes only after two one-bit enqueue
                // calls.
                /*D=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 2, {0, 2},
                                                           {0, 1})),
            cudaq::qec::decode_result_type::errors) {
    cuda_device_id_ = device;
  }

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    if (throw_on_decode)
      throw std::runtime_error("controlled decoder failure");
    if (decode_delay.count() != 0)
      std::this_thread::sleep_for(decode_delay);
    cudaq::qec::decoder_result result;
    result.converged = converged;
    result.result = {syndrome.at(0)};
    return result;
  }

  bool converged = false;
  bool throw_on_decode = false;
  std::chrono::milliseconds decode_delay{0};
};

// Claims graph dispatch and returns a heap-allocated (but otherwise inert --
// no real cudaGraphExec_t) graph_resources, so DecodingSession::create()'s
// capture path can be exercised without a real decode graph to capture.
// capture_graph_pinned() (called via window_node::capture_graph()) still
// does one real cudaSetDevice(0) internally, so this needs a real GPU --
// any CUDA-capable one, not the compute-capability-9.0 floor DEVICE-mode's
// device-graph scheduler needs.
static cudaq::qec::decoder_init graph_stub_inputs(bool with_observables) {
  auto h = cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0});
  auto d = cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0});
  std::optional<cudaq::qec::sparse_binary_matrix> o;
  if (with_observables)
    o = cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0});
  return cudaq::qec::decoder_init(std::move(h), std::move(o), {}, std::move(d));
}

class GraphCapableStubDecoder final : public cudaq::qec::decoder {
public:
  GraphCapableStubDecoder()
      : decoder(graph_stub_inputs(/*with_observables=*/true),
                cudaq::qec::decode_result_type::observables) {}
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {};
  }
  bool supports_graph_dispatch() const override { return true; }
  void *capture_decode_graph(int) override {
    return new cudaq::qec::realtime::graph_resources{};
  }
  void release_decode_graph(void *p) override {
    delete static_cast<cudaq::qec::realtime::graph_resources *>(p);
  }
};

// Same as GraphCapableStubDecoder, but its immutable inputs omit O.
// capture_decode_graph() fails the way a real device-only decoder (e.g.
// nv-qldpc-decoder's validate_device_call_config()) does when O is missing.
// The session must fall back instead of mutating the decoder during capture.
class GraphCapableNoOStubDecoder final : public cudaq::qec::decoder {
public:
  GraphCapableNoOStubDecoder()
      : decoder(graph_stub_inputs(/*with_observables=*/false)) {}
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {};
  }
  bool supports_graph_dispatch() const override { return true; }
  void *capture_decode_graph(int) override {
    if (get_num_observables() == 0)
      return nullptr;
    return new cudaq::qec::realtime::graph_resources{};
  }
  void release_decode_graph(void *p) override {
    delete static_cast<cudaq::qec::realtime::graph_resources *>(p);
  }
};

std::pair<std::unique_ptr<DecodingSession>, ControlledDecoder *>
make_session() {
  auto decoder = std::make_unique<ControlledDecoder>();
  auto *raw_decoder = decoder.get();
  return {DecodingSession::create(std::move(decoder)), raw_decoder};
}

/// The same ControlledDecoder behind the same D/O, wrapped in a one-node graph,
/// so a session built from this must answer the RPCs identically to
/// make_session()'s.
///
/// D's two columns and the default num_syndromes_per_round of 1 make it a
/// two-round window over a two-measurement volume, so its round is smaller than
/// its volume -- the case that tells the two accessors apart.
std::pair<std::unique_ptr<DecodingSession>, ControlledDecoder *>
make_graph_session(bool async_streaming = false) {
  auto decoder = std::make_unique<ControlledDecoder>();
  auto *raw_decoder = decoder.get();

  cudaq::qec::task_graph_node_config node;
  node.id = "window";
  node.kind = "single_window";
  node.d_sparse = {{0, 1}};
  node.o_sparse = {{0}};
  node.decoder_instance = std::move(decoder);

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  cudaq::qec::task_graph_options options;
  options.async_streaming = async_streaming;
  return {
      DecodingSession::create(cudaq::qec::decoding_task_graph::from_topology(
          std::move(nodes), {}, options)),
      raw_decoder};
}

std::unique_ptr<DecodingSession>
make_independent_output_graph_session(bool async_streaming = false) {
  cudaq::qec::task_graph_node_config fast;
  fast.id = "fast";
  fast.kind = "single_window";
  fast.d_sparse = {{0}};
  fast.o_sparse = {{0}};
  fast.decoder_instance = std::make_unique<ControlledDecoder>();

  cudaq::qec::task_graph_node_config slow;
  slow.id = "slow";
  slow.kind = "single_window";
  slow.d_sparse = {{0, 1}};
  slow.o_sparse = {{0}};
  slow.decoder_instance = std::make_unique<ControlledDecoder>();

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(fast));
  nodes.push_back(std::move(slow));
  std::vector<cudaq::qec::task_graph_edge> edges = {
      {"measurements", "output", "fast", "input"},
      {"measurements", "output", "slow", "input"},
      {"fast", "output", "output", "fast"},
      {"slow", "output", "output", "slow"},
  };
  cudaq::qec::task_graph_options options;
  options.async_streaming = async_streaming;
  auto graph = cudaq::qec::decoding_task_graph::from_topology(
      std::move(nodes), std::move(edges), options);
  return DecodingSession::create(std::move(graph),
                                 /*want_device_graph=*/false);
}

// ---------------------------------------------------------------------------
// CQR slot builders — the wire format as the CUDAQ dispatcher hands it to
// the HOST_CALL handlers (RPCHeader + payload in one slot).
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_cqr_slot(uint32_t function_id, uint32_t request_id,
                                   const std::vector<uint8_t> &payload = {}) {
  std::vector<uint8_t> slot(sizeof(RPCHeader) + payload.size());
  RPCHeader header{};
  header.magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header.function_id = function_id;
  header.arg_len = static_cast<uint32_t>(payload.size());
  header.request_id = request_id;
  std::memcpy(slot.data(), &header, sizeof(header));
  if (!payload.empty())
    std::memcpy(slot.data() + sizeof(header), payload.data(), payload.size());
  return slot;
}

// CQR enqueue wire payload: [u64 decoder_id][u64 counter][u64 mapping_id]
// [u64 num_syndromes][bit-packed bytes, LSB-first].
std::vector<uint8_t> make_enqueue_payload(uint64_t counter,
                                          const std::vector<uint8_t> &bits,
                                          uint64_t mapping_id = 0,
                                          uint64_t decoder_id = 0) {
  std::vector<uint8_t> payload(4 * sizeof(uint64_t) +
                               bit_packed_bytes(bits.size()));
  const std::array<uint64_t, 4> fields = {decoder_id, counter, mapping_id,
                                          static_cast<uint64_t>(bits.size())};
  std::memcpy(payload.data(), fields.data(), sizeof(fields));
  for (std::size_t i = 0; i < bits.size(); ++i)
    if (bits[i] & 1u)
      payload[sizeof(fields) + i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  return payload;
}

std::vector<uint8_t> make_get_corrections_payload(int64_t return_size,
                                                  bool reset,
                                                  int64_t decoder_id = 0) {
  GetCorrectionsRequestPayload req{};
  req.decoder_id = decoder_id;
  req.return_size = return_size;
  req.reset = reset ? 1 : 0;
  std::vector<uint8_t> payload(sizeof(req));
  std::memcpy(payload.data(), &req, sizeof(req));
  return payload;
}

std::vector<uint8_t> make_reset_payload(int64_t decoder_id = 0) {
  ResetRequestPayload req{};
  req.decoder_id = decoder_id;
  std::vector<uint8_t> payload(sizeof(req));
  std::memcpy(payload.data(), &req, sizeof(req));
  return payload;
}

void expect_tx_status(const std::vector<uint8_t> &tx, RpcStatus status,
                      uint32_t request_id) {
  ASSERT_GE(tx.size(), sizeof(RPCResponse));
  const auto *response = reinterpret_cast<const RPCResponse *>(tx.data());
  EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(response->status, static_cast<int32_t>(status));
  EXPECT_EQ(response->request_id, request_id);
}

// ---------------------------------------------------------------------------
// Slot parsing (RpcSlot.h): the advertised payload must be physically
// present in the supplied slot.
// ---------------------------------------------------------------------------

TEST(RpcSlotParse, RejectsPayloadBeyondReportedSlot) {
  auto rx = make_cqr_slot(kEnqueueSyndromesFunctionId, 17,
                          make_enqueue_payload(7, {1}, 0, 3));
  // Keep accessible, valid-looking payload bytes beyond the reported slot.
  // A parser that trusts arg_len instead of slot_size will incorrectly accept
  // this request even without a sanitizer detecting the contract violation.
  slot::EnqueueView view;
  EXPECT_FALSE(slot::parse_enqueue(rx.data(), sizeof(RPCHeader), view));
}

TEST(RpcSlotParse, AcceptsAnExactlySizedEnqueueRequestPayload) {
  auto rx = make_cqr_slot(kEnqueueSyndromesFunctionId, 23,
                          make_enqueue_payload(7, {1}, 0, 3));
  slot::EnqueueView view;
  ASSERT_TRUE(slot::parse_enqueue(rx.data(), rx.size(), view));
  EXPECT_EQ(view.decoder_id, 3u);
  EXPECT_EQ(view.counter, 7u);
  EXPECT_EQ(view.syndrome_mapping_id, 0u);
  EXPECT_EQ(view.num_syndromes, 1u);
  ASSERT_EQ(view.byte_count, 1u);
  EXPECT_EQ(view.packed_bits[0], 1u);
}

TEST(RpcSlotParse, PeeksTheDecoderIdFromEveryRequestKind) {
  uint64_t id = 0;
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                          make_enqueue_payload(0, {1}, 0, /*decoder_id=*/5));
  ASSERT_TRUE(slot::peek_decoder_id(eq.data(), eq.size(), id));
  EXPECT_EQ(id, 5u);
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 2,
                          make_get_corrections_payload(1, false, 6));
  ASSERT_TRUE(slot::peek_decoder_id(gc.data(), gc.size(), id));
  EXPECT_EQ(id, 6u);
  auto rst = make_cqr_slot(kResetDecoderFunctionId, 3, make_reset_payload(7));
  ASSERT_TRUE(slot::peek_decoder_id(rst.data(), rst.size(), id));
  EXPECT_EQ(id, 7u);
  // Empty payload has no decoder_id to peek.
  auto bare = make_cqr_slot(kResetDecoderFunctionId, 4);
  EXPECT_FALSE(slot::peek_decoder_id(bare.data(), bare.size(), id));
}

// ---------------------------------------------------------------------------
// Device resolution + pin probes
// ---------------------------------------------------------------------------

TEST(ResolveDecodeDevice, UnpinnedDefaultsToZero) {
  EXPECT_EQ(cudaq::qec::decoding_server::resolve_decode_device(-1), 0);
}

TEST(ResolveDecodeDevice, PinSelectsDevice) {
  EXPECT_EQ(cudaq::qec::decoding_server::resolve_decode_device(3), 3);
}

TEST(SetCudaDeviceForDecode, UnpinnedIsNoOp) {
  // -1 = unpinned: must never touch the device or throw, even on a machine
  // with no CUDA devices at all.
  EXPECT_NO_THROW(cudaq::qec::detail_affinity::set_cuda_device_for_decode(-1));
}

TEST(SetCudaDeviceForDecode, ImpossibleDeviceThrows) {
  // An id beyond the device count fails cudaSetDevice on any machine,
  // including GPU-less CI.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    count = 0;
  EXPECT_THROW(
      cudaq::qec::detail_affinity::set_cuda_device_for_decode(count + 7),
      std::runtime_error);
}

/// cuda_device_id_ is protected: setting an impossible id directly bypasses
/// decoder::get()'s construction-time range check, the only front door --
/// which is exactly what makes create()'s pin-probe failure path injectable.
class MispinnedDecoder final : public cudaq::qec::decoder {
public:
  MispinnedDecoder()
      : decoder(
            cudaq::qec::decoder_init(
                /*H=*/cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1},
                                                                 {0}),
                /*O=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}),
                /*error_rates=*/{},
                // One detector is the parity of two incoming measurement
                // bits, so a decode completes only after two one-bit
                // enqueue calls.
                /*D=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 2, {0, 2},
                                                           {0, 1})),
            cudaq::qec::decode_result_type::errors) {
    cuda_device_id_ = 1 << 20;
  }
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {};
  }
};

TEST(DecodingSessionCreate, UnhonorablePinFailsCreate) {
  // The contract under test: a session whose decoder cannot pin must fail at
  // server bring-up (create()), never at the first RPC.  This is the test
  // that fails if the create()-time probe ever reverts to log-and-continue.
  EXPECT_THROW(DecodingSession::create(std::make_unique<MispinnedDecoder>()),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// Inline HOST_CALL path (DecodingSession::handle_*): the request is served
// entirely on the calling thread — rx parsed in place, decoder run inline,
// response written straight into the tx slot (magic release-stored last).
// ---------------------------------------------------------------------------

TEST(DecodingSessionInline, ServesAFullShotInPlaceOnTheCallingThread) {
  auto [session, decoder] = make_session();
  ASSERT_FALSE(decoder->converged);

  std::vector<uint8_t> tx(64, 0xEE);

  // Not ready before any decode.
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 7,
                          make_get_corrections_payload(1, false));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);

  // Two one-bit rounds complete the volume; each enqueue completes its slot.
  auto eq0 = make_cqr_slot(kEnqueueSyndromesFunctionId, 8,
                           make_enqueue_payload(0, {1}));
  session->handle_enqueue(eq0.data(), tx.data(), eq0.size());
  expect_tx_status(tx, RpcStatus::OK, 8);

  auto eq1 = make_cqr_slot(kEnqueueSyndromesFunctionId, 9,
                           make_enqueue_payload(1, {0}));
  session->handle_enqueue(eq1.data(), tx.data(), eq1.size());
  expect_tx_status(tx, RpcStatus::OK, 9);

  // A completed decode is ready even when the algorithm reports that it did
  // not converge. Readiness and convergence are different contracts.
  // Corrections are packed directly into the tx slot payload area.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 7);
  const auto *response = reinterpret_cast<const RPCResponse *>(tx.data());
  ASSERT_EQ(response->result_len, 1u);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
  EXPECT_EQ(session->decode_count.load(), 1u);

  // Accepting part of the next volume makes the previous result stale.
  auto eq2 = make_cqr_slot(kEnqueueSyndromesFunctionId, 10,
                           make_enqueue_payload(2, {0}));
  session->handle_enqueue(eq2.data(), tx.data(), eq2.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);

  // get_corrections(reset=true) clears the result for the next volume.
  auto eq3 = make_cqr_slot(kEnqueueSyndromesFunctionId, 11,
                           make_enqueue_payload(3, {0}));
  session->handle_enqueue(eq3.data(), tx.data(), eq3.size());
  auto gc_reset = make_cqr_slot(kGetCorrectionsFunctionId, 12,
                                make_get_corrections_payload(1, true));
  session->handle_get_corrections(gc_reset.data(), tx.data(), gc_reset.size());
  expect_tx_status(tx, RpcStatus::OK, 12);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);
}

// The same shot as the decoder session above, through a graph: same statuses,
// same packed correction, same decode count.  Any difference would be the
// session layer routing one kind of target differently from the other.
TEST(DecodingSessionInline, ServesAFullShotThroughATaskGraph) {
  auto [session, decoder] = make_graph_session();
  ASSERT_EQ(session->dec, nullptr) << "a graph session owns no decoder";
  ASSERT_NE(session->graph, nullptr);

  std::vector<uint8_t> tx(64, 0xEE);

  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 7,
                          make_get_corrections_payload(1, false));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);

  auto eq0 = make_cqr_slot(kEnqueueSyndromesFunctionId, 8,
                           make_enqueue_payload(0, {1}));
  session->handle_enqueue(eq0.data(), tx.data(), eq0.size());
  expect_tx_status(tx, RpcStatus::OK, 8);

  // One round in, the window is not full, so nothing has been decoded.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);

  auto eq1 = make_cqr_slot(kEnqueueSyndromesFunctionId, 9,
                           make_enqueue_payload(1, {0}));
  session->handle_enqueue(eq1.data(), tx.data(), eq1.size());
  expect_tx_status(tx, RpcStatus::OK, 9);

  // D = {{0, 1}} makes the detector the parity of the two rounds, so {1, 0}
  // lights it and the observable follows -- the decoder session's bit.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 7);
  const auto *response = reinterpret_cast<const RPCResponse *>(tx.data());
  ASSERT_EQ(response->result_len, 1u);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
  EXPECT_EQ(session->decode_count.load(), 1u);
  EXPECT_FALSE(decoder->converged) << "readiness is not convergence";

  // Accepting part of the next volume makes the previous result stale, even
  // though the graph's frozen snapshot still holds it.
  auto eq2 = make_cqr_slot(kEnqueueSyndromesFunctionId, 13,
                           make_enqueue_payload(2, {0}));
  session->handle_enqueue(eq2.data(), tx.data(), eq2.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);

  // Completing that volume makes it ready again; reset=true consumes it.
  auto eq3 = make_cqr_slot(kEnqueueSyndromesFunctionId, 11,
                           make_enqueue_payload(3, {0}));
  session->handle_enqueue(eq3.data(), tx.data(), eq3.size());
  auto gc_reset = make_cqr_slot(kGetCorrectionsFunctionId, 12,
                                make_get_corrections_payload(1, true));
  session->handle_get_corrections(gc_reset.data(), tx.data(), gc_reset.size());
  expect_tx_status(tx, RpcStatus::OK, 12);
  EXPECT_EQ(session->decode_count.load(), 2u);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);

  // reset_decoder returns the graph to a clean epoch.
  auto rst = make_cqr_slot(kResetDecoderFunctionId, 10, make_reset_payload());
  session->handle_reset(rst.data(), tx.data(), rst.size());
  expect_tx_status(tx, RpcStatus::OK, 10);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 7);
}

TEST(DecodingSessionInline, GraphOwnsAggregateOutputReadiness) {
  auto session = make_independent_output_graph_session();
  std::vector<uint8_t> tx(64, 0);
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(2, false));

  auto first = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                             make_enqueue_payload(0, {1}));
  session->handle_enqueue(first.data(), tx.data(), first.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  EXPECT_EQ(session->graph->streaming_result().outputs[0].generation, 1u);
  EXPECT_EQ(session->graph->streaming_result().outputs[1].generation, 0u);
  EXPECT_EQ(session->decode_count.load(), 0u);
  expect_tx_status(tx, RpcStatus::NOT_READY, 3);

  auto second = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                              make_enqueue_payload(1, {0}));
  session->handle_enqueue(second.data(), tx.data(), second.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  EXPECT_EQ(session->graph->streaming_result().generation, 1u);
  EXPECT_EQ(session->decode_count.load(), 1u);
  expect_tx_status(tx, RpcStatus::OK, 3);
  const auto *aggregate_response =
      reinterpret_cast<const RPCResponse *>(tx.data());
  ASSERT_EQ(aggregate_response->result_len, 1u);
  const uint8_t aggregate_payload = tx[sizeof(RPCResponse)] & 0x3u;
  EXPECT_EQ(aggregate_payload, 0x3u);

  // Consuming through the existing RPC clears availability without resetting
  // independent-output generations or the graph's streaming overlap.
  auto gc_consume = make_cqr_slot(kGetCorrectionsFunctionId, 5,
                                  make_get_corrections_payload(2, true));
  session->handle_get_corrections(gc_consume.data(), tx.data(),
                                  gc_consume.size());
  expect_tx_status(tx, RpcStatus::OK, 5);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 0x3u, aggregate_payload);
  EXPECT_EQ(session->graph->streaming_result().generation, 1u);
  EXPECT_EQ(session->graph->streaming_result().status,
            cudaq::qec::task_graph_streaming_status::pending);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 3);

  // Fast advances alone on round 3, so the aggregate generation does not. The
  // live value changes, but a partial aggregate is not a result.
  auto third = make_cqr_slot(kEnqueueSyndromesFunctionId, 4,
                             make_enqueue_payload(2, {1}));
  session->handle_enqueue(third.data(), tx.data(), third.size());
  const auto &live_result = session->graph->streaming_result();
  EXPECT_FALSE(live_result.operation_completed);
  EXPECT_EQ(live_result.values, (std::vector<uint8_t>{1, 0}));
  EXPECT_EQ(session->decode_count.load(), 1u);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 3);
}

// Without a consume, the frozen aggregate is still the previous shot's: the
// next round must make it stale, as it would a decoder's result.
TEST(DecodingSessionInline, NewRoundMakesUnconsumedGraphAggregateStale) {
  auto session = make_independent_output_graph_session();
  std::vector<uint8_t> tx(64, 0);
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(2, false));

  auto first = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                             make_enqueue_payload(0, {1}));
  auto second = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                              make_enqueue_payload(1, {0}));
  session->handle_enqueue(first.data(), tx.data(), first.size());
  session->handle_enqueue(second.data(), tx.data(), second.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
  // Re-reading without new input is still the same ready result.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);

  auto third = make_cqr_slot(kEnqueueSyndromesFunctionId, 4,
                             make_enqueue_payload(2, {1}));
  session->handle_enqueue(third.data(), tx.data(), third.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 3);
  EXPECT_EQ(session->decode_count.load(), 1u);
}

TEST(DecodingSessionInline, AsyncGraphFreezesIndependentAggregateOutput) {
  auto session =
      make_independent_output_graph_session(/*async_streaming=*/true);
  std::vector<uint8_t> tx(64, 0);
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(2, false));

  auto first = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                             make_enqueue_payload(0, {1}));
  auto second = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                              make_enqueue_payload(1, {0}));
  session->handle_enqueue(first.data(), tx.data(), first.size());
  session->handle_enqueue(second.data(), tx.data(), second.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
  const uint8_t aggregate_payload = tx[sizeof(RPCResponse)] & 0x3u;
  EXPECT_EQ(aggregate_payload, 0x3u);
  EXPECT_EQ(session->decode_count.load(), 1u);

  // Re-reading before any new round serves the same frozen aggregate.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 0x3u, aggregate_payload);

  // Only the fast output advances. The live graph value changes, but the new
  // round has started a volume whose aggregate is incomplete.
  auto third = make_cqr_slot(kEnqueueSyndromesFunctionId, 4,
                             make_enqueue_payload(2, {1}));
  session->handle_enqueue(third.data(), tx.data(), third.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 3);
  EXPECT_EQ(session->graph->streaming_result().values,
            (std::vector<uint8_t>{0, 1}));
  EXPECT_EQ(session->decode_count.load(), 1u);
}

// Chunking is the caller's choice for either kind of target, so a client that
// sends a whole syndrome per call -- what the wire protocol's users do -- must
// get the same service from a window as from the decoder.  A graph's smaller
// num_msyn_per_round() is the chunk a caller should prefer, and enqueue_core
// must not bound the call by it.
TEST(DecodingSessionInline, AWholeVolumeInOneEnqueueServesEitherTarget) {
  std::vector<uint8_t> tx(64, 0);
  auto whole_volume = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                                    make_enqueue_payload(0, {1, 0}));
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 2,
                          make_get_corrections_payload(1, false));

  auto serves_a_shot_in_one_call = [&](auto make) {
    auto [session, decoder] = make();
    (void)decoder;
    session->handle_enqueue(whole_volume.data(), tx.data(),
                            whole_volume.size());
    expect_tx_status(tx, RpcStatus::OK, 1);
    session->handle_get_corrections(gc.data(), tx.data(), gc.size());
    expect_tx_status(tx, RpcStatus::OK, 2);
    EXPECT_EQ(session->decode_count.load(), 1u);
    EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
  };

  serves_a_shot_in_one_call(make_session);
  // Not vacuous: the window's round really is smaller than the call.
  {
    auto [probe, decoder] = make_graph_session();
    (void)decoder;
    ASSERT_LT(probe->graph->get_num_msyn_per_round(),
              probe->graph->get_num_msyn_per_decode());
  }
  serves_a_shot_in_one_call([] { return make_graph_session(); });
}

TEST(DecodingSessionInline,
     AsyncGraphUsesExistingRpcAndSplitsWholeVolumeInternally) {
  auto [session, decoder] = make_graph_session(/*async_streaming=*/true);
  decoder->decode_delay = std::chrono::milliseconds(10);
  std::vector<uint8_t> tx(64, 0);
  auto whole_volume = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                                    make_enqueue_payload(0, {1, 0}));
  auto next_partial = make_cqr_slot(kEnqueueSyndromesFunctionId, 3,
                                    make_enqueue_payload(0, {0}));
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 2,
                          make_get_corrections_payload(1, false));

  session->handle_enqueue(whole_volume.data(), tx.data(), whole_volume.size());
  expect_tx_status(tx, RpcStatus::OK, 1);
  session->handle_enqueue(next_partial.data(), tx.data(), next_partial.size());
  expect_tx_status(tx, RpcStatus::OK, 3);

  // The existing RPC is not a poll: one get waits through all preceding
  // enqueues and observes the latest completed graph snapshot.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 2);
  EXPECT_EQ(session->decode_count.load(), 1u);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
}

TEST(DecodingSessionInline, AsyncFinalPublishesBeforeTheEnqueueRpcReturns) {
  auto [session, decoder] = make_graph_session(/*async_streaming=*/true);
  (void)decoder;
  session->mapping_table = {
      {0, syndrome_mapping_row{cudaq::qec::round_type::BULK, {{}}}},
      {1, syndrome_mapping_row{cudaq::qec::round_type::FINAL, {{}}}}};
  std::vector<uint8_t> tx(64, 0);
  auto bulk = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                            make_enqueue_payload(0, {1}, /*mapping_id=*/0));
  auto final = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                             make_enqueue_payload(1, {0}, /*mapping_id=*/1));
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(1, false));

  session->handle_enqueue(bulk.data(), tx.data(), bulk.size());
  expect_tx_status(tx, RpcStatus::OK, 1);
  session->handle_enqueue(final.data(), tx.data(), final.size());
  expect_tx_status(tx, RpcStatus::OK, 2);

  const auto &result = session->graph->streaming_result();
  EXPECT_EQ(result.status, cudaq::qec::task_graph_streaming_status::complete);
  EXPECT_EQ(session->decode_count.load(), 1u);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
}

TEST(DecodingSessionInline, ArbitraryLengthFinalCommitsEveryWireRound) {
  using namespace cudaq::qec;
  for (const bool asynchronous : {false, true}) {
    dem_chunks_spec spec;
    spec.num_rounds = 5;
    spec.measurements_per_round = 1;
    spec.seam = {seam_name::next_round, seam_name::prev_round};
    spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                        {phase_name::dem_bulk, phase_name::dem_bulk},
                        {phase_name::dem_bulk, phase_name::dem_final}};
    for (const auto id :
         {phase_name::dem_init, phase_name::dem_bulk, phase_name::dem_final}) {
      dem_chunk_spec phase;
      phase.num_faults = 1;
      phase.H_sparse = {0, -1};
      phase.O_sparse = {0, -1};
      phase.error_rates = {0.01};
      phase.D_sparse = id == phase_name::dem_init
                           ? std::vector<int64_t>{0, -1}
                           : std::vector<int64_t>{0, 1, -1};
      phase.num_measurements = 1;
      spec.phases.push_back({id, std::move(phase)});
    }
    task_graph_node_config node;
    node.id = "phase-memory";
    node.kind = "sliding_window";
    node.dem = spec;
    node.decoder_type = "single_error_lut";
    node.node_params = {{"window_size", std::size_t{5}},
                        {"step_size", std::size_t{3}},
                        {"guard_rounds", std::size_t{1}},
                        {"arbitrary_length", true}};
    std::vector<task_graph_node_config> nodes;
    nodes.push_back(std::move(node));
    task_graph_options options;
    options.async_streaming = asynchronous;
    options.async_queue_capacity = 2;
    auto session = DecodingSession::create(
        decoding_task_graph::from_topology(std::move(nodes), {}, options),
        false,
        {{0, syndrome_mapping_row{round_type::BULK, {{}}}},
         {1, syndrome_mapping_row{round_type::FINAL, {{}}}}});
    std::vector<uint8_t> tx(64);
    for (const std::uint64_t total : {1000u, 1001u, 1002u, 1003u}) {
      for (std::uint64_t r = 0; r + 1 < total; ++r) {
        auto request = make_cqr_slot(kEnqueueSyndromesFunctionId, r + 1,
                                     make_enqueue_payload(r, {uint8_t{0}}));
        session->handle_enqueue(request.data(), tx.data(), request.size());
        expect_tx_status(tx, RpcStatus::OK, r + 1);
      }
      auto final = make_cqr_slot(kEnqueueSyndromesFunctionId, total,
                                 make_enqueue_payload(total - 1, {1}, 1));
      session->handle_enqueue(final.data(), tx.data(), final.size());
      expect_tx_status(tx, RpcStatus::OK, total);
      EXPECT_EQ(
          session->graph->streaming_result().outputs.front().coverage_end_round,
          total);
      auto read = make_cqr_slot(kGetCorrectionsFunctionId, total + 1,
                                make_get_corrections_payload(1, true));
      session->handle_get_corrections(read.data(), tx.data(), read.size());
      expect_tx_status(tx, RpcStatus::OK, total + 1);
      EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
    }
  }
}

TEST(DecodingSessionInline, AsyncGraphCountsEveryCompletedOperationInABurst) {
  auto [session, decoder] = make_graph_session(/*async_streaming=*/true);
  decoder->decode_delay = std::chrono::milliseconds(5);
  std::vector<uint8_t> tx(64, 0);
  auto first = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                             make_enqueue_payload(0, {1, 0}));
  auto second = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                              make_enqueue_payload(2, {0, 0}));
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(1, false));

  session->handle_enqueue(first.data(), tx.data(), first.size());
  session->handle_enqueue(second.data(), tx.data(), second.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());

  expect_tx_status(tx, RpcStatus::OK, 3);
  EXPECT_EQ(session->decode_count.load(), 2u);
  EXPECT_EQ(session->graph->streaming_result().generation, 2u);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 1u);
}

TEST(DecodingSessionCreate, NullTaskGraphFailsCreate) {
  EXPECT_THROW(DecodingSession::create(
                   std::unique_ptr<cudaq::qec::decoding_task_graph>{}),
               std::invalid_argument);
}

// Every RPC goes through the graph's streaming API, so a graph that decodes
// only full shots must be refused at create() with its incompatibility reason.
TEST(DecodingSessionCreate, FullShotOnlyTaskGraphFailsCreate) {
  cudaq::qec::task_graph_node_config fast;
  fast.id = "fast";
  fast.kind = "single_window";
  fast.d_sparse = {{0}};
  fast.o_sparse = {{0}};
  fast.decoder_instance = std::make_unique<ControlledDecoder>();

  cudaq::qec::task_graph_node_config slow;
  slow.id = "slow";
  slow.kind = "single_window";
  slow.d_sparse = {{0, 1}};
  slow.o_sparse = {{0}};
  slow.decoder_instance = std::make_unique<ControlledDecoder>();

  cudaq::qec::task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(fast));
  nodes.push_back(std::move(slow));
  nodes.push_back(std::move(combine));
  std::vector<cudaq::qec::task_graph_edge> edges = {
      {"measurements", "output", "fast", "input"},
      {"measurements", "output", "slow", "input"},
      {"fast", "output", "combine", "a"},
      {"slow", "output", "combine", "b"},
      {"combine", "output", "output", "input"},
  };
  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes),
                                                              std::move(edges));
  ASSERT_FALSE(graph->supports_streaming())
      << "not vacuous: the graph must be full-shot only";
  const std::string reason = graph->streaming_incompatibility_reason();

  try {
    (void)DecodingSession::create(std::move(graph),
                                  /*want_device_graph=*/false);
    FAIL() << "create() accepted a full-shot-only task graph";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(reason), std::string::npos)
        << e.what();
  }
}

// Nodes disagreeing on device can never be served, so this must fail at
// create(), same as an unhonorable decoder pin, not on the first RPC.
TEST(DecodingSessionCreate, GraphWithDisagreeingDevicePinsFailsCreate) {
  cudaq::qec::task_graph_node_config x;
  x.id = "x";
  x.kind = "single_window";
  x.d_sparse = {{0}};
  x.o_sparse = {{0}};
  x.decoder_instance = std::make_unique<ControlledDecoder>(/*device=*/3);

  cudaq::qec::task_graph_node_config z;
  z.id = "z";
  z.kind = "single_window";
  z.d_sparse = {{0}};
  z.o_sparse = {{0}};
  z.decoder_instance = std::make_unique<ControlledDecoder>(/*device=*/5);

  cudaq::qec::task_graph_node_config combine;
  combine.id = "combine";
  combine.kind = "xor";

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(x));
  nodes.push_back(std::move(z));
  nodes.push_back(std::move(combine));

  std::vector<cudaq::qec::task_graph_edge> edges = {
      {"measurements", "output", "x", "input"},
      {"measurements", "output", "z", "input"},
      {"x", "output", "combine", "a"},
      {"z", "output", "combine", "b"},
      {"combine", "output", "output", "input"},
  };

  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes),
                                                              std::move(edges));
  EXPECT_THROW(DecodingSession::create(std::move(graph)), std::runtime_error);
}

static int cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    return 0;
  return count;
}

// A single-node graph whose inner decoder supports graph dispatch must
// capture just like a plain decoder does -- this is what lets SessionRegistry
// accept `task_graph` + `dispatch: device_graph` for it instead of rejecting
// the combination outright.
TEST(DecodingSessionCreate, GraphDispatchCapableGraphPopulatesGraphResources) {
  // Capture pins device 0 (see graph_stub_inputs).
  if (cuda_device_count() < 1)
    GTEST_SKIP() << "needs >= 1 CUDA device";
  cudaq::qec::task_graph_node_config node;
  node.id = "window";
  node.kind = "single_window";
  node.d_sparse = {{0}};
  node.o_sparse = {{0}};
  node.decoder_instance = std::make_unique<GraphCapableStubDecoder>();

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes));
  ASSERT_TRUE(graph->supports_graph_dispatch());

  auto session = DecodingSession::create(std::move(graph));
  EXPECT_NE(session->graph_resources, nullptr);
}

// An immutable caller-provided decoder cannot acquire the graph's O at capture
// time. Such an instance must remain uncaptured instead of silently changing
// its construction model.
TEST(DecodingSessionCreate, CaptureGraphDoesNotMutateTheDecodersModel) {
  // Capture pins device 0 before the stub declines (see graph_stub_inputs).
  if (cuda_device_count() < 1)
    GTEST_SKIP() << "needs >= 1 CUDA device";
  cudaq::qec::task_graph_node_config node;
  node.id = "window";
  node.kind = "single_window";
  node.d_sparse = {{0}};
  node.o_sparse = {{0}};
  node.decoder_instance = std::make_unique<GraphCapableNoOStubDecoder>();

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes));
  ASSERT_TRUE(graph->supports_graph_dispatch());

  auto session = DecodingSession::create(std::move(graph));
  EXPECT_EQ(session->graph_resources, nullptr);
}

// A decoder that claims graph dispatch in principle (like nv-qldpc-decoder,
// unconditionally) must not be captured for a HOST-dispatch session: it may
// lack the device-only settings capture needs (relay-BP composition, no
// OSD), and capturing anyway would fail session creation for a config that
// never asked for DEVICE mode. want_device_graph=false is what
// SessionRegistry passes for dispatch: host (or the default).
TEST(DecodingSessionCreate, HostDispatchSkipsCaptureEvenWhenSupported) {
  ASSERT_TRUE(GraphCapableStubDecoder{}.supports_graph_dispatch())
      << "not vacuous: the decoder really does claim graph dispatch";
  auto session = DecodingSession::create(
      std::make_unique<GraphCapableStubDecoder>(), /*want_device_graph=*/false);
  EXPECT_EQ(session->graph_resources, nullptr);
}

TEST(DecodingSessionCreate, HostDispatchSkipsCaptureForATaskGraphToo) {
  cudaq::qec::task_graph_node_config node;
  node.id = "window";
  node.kind = "single_window";
  node.d_sparse = {{0}};
  node.o_sparse = {{0}};
  node.decoder_instance = std::make_unique<GraphCapableStubDecoder>();

  std::vector<cudaq::qec::task_graph_node_config> nodes;
  nodes.push_back(std::move(node));
  auto graph = cudaq::qec::decoding_task_graph::from_topology(std::move(nodes));
  ASSERT_TRUE(graph->supports_graph_dispatch())
      << "not vacuous: the graph really does support graph dispatch";

  auto session =
      DecodingSession::create(std::move(graph), /*want_device_graph=*/false);
  EXPECT_EQ(session->graph_resources, nullptr);
}

TEST(DecodingSessionInline, EnqueueRespondsOkAndDefersDecodeErrors) {
  auto [session, decoder] = make_session();
  decoder->throw_on_decode = true;
  std::vector<uint8_t> tx(64, 0);

  // Fire-and-forget contract: the decode failure does NOT fail the enqueue
  // response; it latches and surfaces at the next get_corrections.
  auto eq0 = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                           make_enqueue_payload(0, {1}));
  session->handle_enqueue(eq0.data(), tx.data(), eq0.size());
  expect_tx_status(tx, RpcStatus::OK, 1);
  auto eq1 = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                           make_enqueue_payload(1, {0}));
  session->handle_enqueue(eq1.data(), tx.data(), eq1.size());
  expect_tx_status(tx, RpcStatus::OK, 2);

  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(1, false));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::INTERNAL_ERROR, 3);
  // Sticky until reset.
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::INTERNAL_ERROR, 3);

  decoder->throw_on_decode = false;
  auto rst = make_cqr_slot(kResetDecoderFunctionId, 4, make_reset_payload());
  session->handle_reset(rst.data(), tx.data(), rst.size());
  expect_tx_status(tx, RpcStatus::OK, 4);

  session->handle_enqueue(eq0.data(), tx.data(), eq0.size());
  session->handle_enqueue(eq1.data(), tx.data(), eq1.size());
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
}

TEST(DecodingSessionInline, RejectsMeasurementVolumeOverflow) {
  auto [session, decoder] = make_session();
  (void)decoder;
  std::vector<uint8_t> tx(64, 0);

  // {1} then {0,1}: 3 bits into a 2-bit volume — deferred INTERNAL_ERROR.
  auto eq0 = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                           make_enqueue_payload(0, {1}));
  session->handle_enqueue(eq0.data(), tx.data(), eq0.size());
  expect_tx_status(tx, RpcStatus::OK, 1);
  auto eq1 = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                           make_enqueue_payload(1, {0, 1}));
  session->handle_enqueue(eq1.data(), tx.data(), eq1.size());
  expect_tx_status(tx, RpcStatus::OK, 2);

  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(1, false));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::INTERNAL_ERROR, 3);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::INTERNAL_ERROR, 3);

  auto rst = make_cqr_slot(kResetDecoderFunctionId, 4, make_reset_payload());
  session->handle_reset(rst.data(), tx.data(), rst.size());
  expect_tx_status(tx, RpcStatus::OK, 4);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 3);
}

TEST(DecodingSessionInline, MalformedRequestsRespondBadRequest) {
  auto [session, decoder] = make_session();
  (void)decoder;
  std::vector<uint8_t> tx(64, 0);

  // Enqueue whose advertised bits are not physically present.
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 5,
                          make_enqueue_payload(0, {1}));
  reinterpret_cast<RPCHeader *>(eq.data())->arg_len = 4 * sizeof(uint64_t);
  session->handle_enqueue(eq.data(), tx.data(), eq.size());
  expect_tx_status(tx, RpcStatus::BAD_REQUEST, 5);

  // get_corrections with a short payload.
  auto gc =
      make_cqr_slot(kGetCorrectionsFunctionId, 6, std::vector<uint8_t>(4, 0));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::BAD_REQUEST, 6);

  // reset with a short payload.
  auto rst =
      make_cqr_slot(kResetDecoderFunctionId, 7, std::vector<uint8_t>(4, 0));
  session->handle_reset(rst.data(), tx.data(), rst.size());
  expect_tx_status(tx, RpcStatus::BAD_REQUEST, 7);
}

// A decoder that records the round shape each enqueue carried, so a FINAL row
// can be shown to reach the decode target rather than stopping at the table.
namespace {
class TagRecordingDecoder final : public cudaq::qec::decoder {
public:
  TagRecordingDecoder()
      : cudaq::qec::decoder(graph_stub_inputs(/*with_observables=*/true),
                            cudaq::qec::decode_result_type::observables) {}
  bool enqueue_syndrome(const uint8_t *s, std::size_t n,
                        cudaq::qec::round_type type) override {
    tags.push_back(type);
    return cudaq::qec::decoder::enqueue_syndrome(s, n);
  }
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {true, {0.0}};
  }
  std::vector<cudaq::qec::round_type> tags;
};

SyndromeMappingTable two_row_table() {
  return {{0, syndrome_mapping_row{cudaq::qec::round_type::BULK, {{}}}},
          {1, syndrome_mapping_row{cudaq::qec::round_type::FINAL, {{}}}}};
}
} // namespace

TEST(DecodingSessionInline, AnExplicitlyEmptyTableStillAcceptsMappingIdZero) {
  // An empty map is not the identity row: taken literally it would reject the
  // only id anything sends. A caller passing {} means "no rows configured",
  // which is the same thing an undeclared section means.
  // Break-it: drop the empty() check in create().
  auto session = DecodingSession::create(std::make_unique<ControlledDecoder>(),
                                         /*want_device_graph=*/false,
                                         SyndromeMappingTable{});
  std::vector<uint8_t> tx(64, 0);
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                          make_enqueue_payload(0, {1}, /*mapping_id=*/0));
  session->handle_enqueue(eq.data(), tx.data(), eq.size());
  expect_tx_status(tx, RpcStatus::OK, 1);

  // And the default-argument path, which is what every existing caller uses.
  auto [defaulted, decoder] = make_session();
  (void)decoder;
  auto eq2 = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                           make_enqueue_payload(0, {1}, /*mapping_id=*/0));
  defaulted->handle_enqueue(eq2.data(), tx.data(), eq2.size());
  expect_tx_status(tx, RpcStatus::OK, 2);
}

TEST(DecodingSessionInline, ADeclaredMappingIdIsAcceptedAndTagsTheRound) {
  auto decoder = std::make_unique<TagRecordingDecoder>();
  auto *raw = decoder.get();
  auto session =
      DecodingSession::create(std::move(decoder),
                              /*want_device_graph=*/false, two_row_table());
  std::vector<uint8_t> tx(64, 0);

  auto bulk = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                            make_enqueue_payload(0, {1}, /*mapping_id=*/0));
  session->handle_enqueue(bulk.data(), tx.data(), bulk.size());
  expect_tx_status(tx, RpcStatus::OK, 1);

  auto final_round = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                                   make_enqueue_payload(1, {1},
                                                        /*mapping_id=*/1));
  session->handle_enqueue(final_round.data(), tx.data(), final_round.size());
  expect_tx_status(tx, RpcStatus::OK, 2);

  // Break-it: drop the type from the enqueue_core/decode_target call chain.
  ASSERT_EQ(raw->tags.size(), 2u);
  EXPECT_EQ(raw->tags[0], cudaq::qec::round_type::BULK);
  EXPECT_EQ(raw->tags[1], cudaq::qec::round_type::FINAL);
}

TEST(SessionRegistryLoad, DeclaringAFinalRowKeepsIdZeroUsable) {
  // Declaring a section is additive, not a replacement.  A producer sends
  // syndrome_mapping_id = 0 unless it names a row, so a document that adds only
  // a `final` row must not turn every ordinary enqueue into BAD_REQUEST.
  // Break-it: have make_mapping_table start from an empty table when the
  // document declares rows, or invert its `"final"` ternary -- the shape
  // assertions below fail on the second mutation, the status ones on the
  // first.
  const std::string yaml = R"(decoders:
  - id: 0
    type: single_error_lut
    block_size: 1
    syndrome_size: 1
    H_sparse: [0,-1]
    O_sparse: [0,-1]
    D_sparse: [0,-1]
syndrome_mapping:
  - {id: 1, type: final}
)";
  SessionRegistry registry;
  registry.load_from_config(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml),
      "keeps_id_zero");
  auto &session = registry.get(0);

  // The row carries the shape the document asked for, not just an id that is
  // accepted -- both shapes return OK, so a status-only assertion cannot tell
  // make_mapping_table's string -> round_type conversion apart from its
  // inverse.
  ASSERT_EQ(session.mapping_table.count(1), 1u);
  EXPECT_EQ(session.mapping_table.at(0).type, cudaq::qec::round_type::BULK);
  EXPECT_EQ(session.mapping_table.at(1).type, cudaq::qec::round_type::FINAL);

  std::vector<uint8_t> tx(64, 0);

  auto bulk = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                            make_enqueue_payload(0, {1}, /*mapping_id=*/0));
  session.handle_enqueue(bulk.data(), tx.data(), bulk.size());
  expect_tx_status(tx, RpcStatus::OK, 1);

  // The declared row works too, and an undeclared one is still refused.
  auto fin = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                           make_enqueue_payload(1, {1}, /*mapping_id=*/1));
  session.handle_enqueue(fin.data(), tx.data(), fin.size());
  expect_tx_status(tx, RpcStatus::OK, 2);

  auto bad = make_cqr_slot(kEnqueueSyndromesFunctionId, 3,
                           make_enqueue_payload(2, {1}, /*mapping_id=*/9));
  session.handle_enqueue(bad.data(), tx.data(), bad.size());
  expect_tx_status(tx, RpcStatus::BAD_REQUEST, 3);
}

TEST(DecodingSessionInline, AnUndeclaredMappingIdIsStillRejected) {
  // The table replaced a hard-coded `!= 0`, so the rejection must survive with
  // its status intact rather than becoming an OK with a late failure.
  auto session =
      DecodingSession::create(std::make_unique<TagRecordingDecoder>(),
                              /*want_device_graph=*/false, two_row_table());
  std::vector<uint8_t> tx(64, 0);
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                          make_enqueue_payload(0, {1}, /*mapping_id=*/9));
  session->handle_enqueue(eq.data(), tx.data(), eq.size());
  expect_tx_status(tx, RpcStatus::BAD_REQUEST, 1);
}

TEST(DeviceSyndromeMapping, TheTableCarriesEveryDeclaredRow) {
  // The DEVICE-mode handler validates ids against its own copy of the table,
  // so every declared row has to survive the flattening into the POD.  A row
  // dropped here is an id the device refuses that the host accepts.
  //
  // Break-it: skip the `declared |=` in device_syndrome_mapping_table.
  const auto empty = device_syndrome_mapping_table({});
  EXPECT_EQ(empty.declared, 0u); // the handler reads this as "id 0 only"

  const auto table = device_syndrome_mapping_table(two_row_table());
  // Dense by id: bit i marks id i declared, and shapes[i] is its shape.
  EXPECT_EQ(table.declared, 0b11u);
  EXPECT_EQ(table.shapes[0],
            static_cast<uint8_t>(cudaq::qec::round_type::BULK));
  EXPECT_EQ(table.shapes[1],
            static_cast<uint8_t>(cudaq::qec::round_type::FINAL));
}

TEST(DeviceSyndromeMapping, EveryLegalIdFitsTheTable) {
  // The bound is on the id, not the row count, so the highest legal id must be
  // addressable rather than written past the end of shapes[].
  //
  // Break-it: make the builder reject an id of kMaxSyndromeMappingRows - 1.
  // (Shrinking the arrays is a static_assert, not a test failure.)
  const auto shape_of = [](uint64_t id) {
    // Alternating, so the check below sees each id's own value rather than one
    // repeated constant.
    return id % 2 ? cudaq::qec::round_type::FINAL
                  : cudaq::qec::round_type::BULK;
  };
  SyndromeMappingTable rows;
  for (uint64_t id = 0; id < kMaxSyndromeMappingRows; ++id)
    rows.emplace(id, syndrome_mapping_row{shape_of(id), {{}}});

  const auto table = device_syndrome_mapping_table(rows);
  // Every id in range is declared: all 16 bits set.
  EXPECT_EQ(table.declared, static_cast<uint16_t>(0xFFFFu));
  for (uint64_t id = 0; id < kMaxSyndromeMappingRows; ++id)
    EXPECT_EQ(table.shapes[id], static_cast<uint8_t>(shape_of(id)))
        << "id " << id;
}

TEST(DeviceSyndromeMapping, AnOutOfRangeIdIsRejectedRatherThanTruncated) {
  // validate_syndrome_mapping() enforces the same bound, so this is the guard
  // for a table built programmatically that never passed through it.  Without
  // it the shift would be undefined and the row could alias a real id.
  //
  // Break-it: drop the range check in device_syndrome_mapping_table.
  SyndromeMappingTable rows = default_syndrome_mapping_table();
  rows.emplace(kMaxSyndromeMappingRows,
               syndrome_mapping_row{cudaq::qec::round_type::BULK, {{}}});
  EXPECT_THROW(device_syndrome_mapping_table(rows), std::runtime_error);

  // In range is accepted, so the check is a bound and not a blanket refusal.
  rows.erase(kMaxSyndromeMappingRows);
  rows.emplace(kMaxSyndromeMappingRows - 1,
               syndrome_mapping_row{cudaq::qec::round_type::BULK, {{}}});
  EXPECT_NO_THROW(device_syndrome_mapping_table(rows));
}

TEST(DeviceSyndromeMapping,
     OnlyADocumentBeyondThePassThroughRowNeedsInstalling) {
  // The device default already services id 0 alone, so installing for a
  // document that declares nothing more would make the dispatch archive a hard
  // requirement of every device_graph session for no behavior change.
  //
  // Break-it: return true unconditionally from
  // needs_device_syndrome_mapping_install; the sc4/mixed-dispatch builds
  // without the cudevice archive then fail at session creation.
  const auto needs_install = [](const SyndromeMappingTable &rows) {
    return needs_device_syndrome_mapping_install(
        device_syndrome_mapping_table(rows));
  };
  EXPECT_FALSE(needs_install({}));
  EXPECT_FALSE(needs_install(default_syndrome_mapping_table()));
  // Redeclaring id 0 is still just id 0.
  EXPECT_FALSE(needs_install(
      {{0, syndrome_mapping_row{cudaq::qec::round_type::FINAL, {{}}}}}));

  EXPECT_TRUE(needs_install(two_row_table()));
}

TEST(DecodingSessionInline, RejectsNonIdentitySyndromeMappings) {
  auto [session, decoder] = make_session();
  (void)decoder;
  std::vector<uint8_t> tx(64, 0);

  // The wire carries syndrome_mapping_id, but only the identity mapping
  // (id 0) has ever been honorable; anything else is rejected rather than
  // silently decoded as identity, and the shot is poisoned until reset.
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                          make_enqueue_payload(0, {1}, /*mapping_id=*/5));
  session->handle_enqueue(eq.data(), tx.data(), eq.size());
  expect_tx_status(tx, RpcStatus::BAD_REQUEST, 1);

  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 2,
                          make_get_corrections_payload(1, false));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::INTERNAL_ERROR, 2);

  auto rst = make_cqr_slot(kResetDecoderFunctionId, 3, make_reset_payload());
  session->handle_reset(rst.data(), tx.data(), rst.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::NOT_READY, 2);
}

TEST(DecodingSessionInline, CorrectionsLargerThanTheCapacityFailExplicitly) {
  auto [session, decoder] = make_session();
  (void)decoder;
  std::vector<uint8_t> tx(64, 0);
  auto eq0 = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                           make_enqueue_payload(0, {1}));
  auto eq1 = make_cqr_slot(kEnqueueSyndromesFunctionId, 2,
                           make_enqueue_payload(1, {0}));
  session->handle_enqueue(eq0.data(), tx.data(), eq0.size());
  session->handle_enqueue(eq1.data(), tx.data(), eq1.size());

  // A result that does not fit the caller's capacity must fail the RPC
  // explicitly (truncation would advertise bytes that were never written) —
  // and must NOT consume the pending result.
  std::size_t out_len = 0;
  EXPECT_EQ(session->get_corrections_core(/*return_size=*/1, /*reset=*/true,
                                          /*out=*/nullptr, /*out_capacity=*/0,
                                          out_len),
            RpcStatus::INTERNAL_ERROR);
  EXPECT_EQ(out_len, 0u);

  // The result is still there for a correctly sized caller.
  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 3,
                          make_get_corrections_payload(1, false));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 3);
}

std::vector<uint8_t> g_captured_syndrome_bytes;
void record_captured_syndromes(const uint8_t *data, size_t len) {
  g_captured_syndrome_bytes.assign(data, data + len);
}

TEST(DecodingSessionInline, SaveSyndromeCaptureMatchesTheLegacyFormat) {
  auto [session, decoder] = make_session();
  (void)decoder;
  std::vector<uint8_t> tx(64, 0);

  g_captured_syndrome_bytes.clear();
  cudaq::qec::decoding::host::_set_syndrome_capture_callback(
      record_captured_syndromes);
  // Wire bits are LSB-first; the legacy saved-syndrome format is MSB-first.
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                          make_enqueue_payload(0, {1}));
  session->handle_enqueue(eq.data(), tx.data(), eq.size());
  cudaq::qec::decoding::host::_set_syndrome_capture_callback(nullptr);

  ASSERT_EQ(g_captured_syndrome_bytes.size(), 1u);
  EXPECT_EQ(g_captured_syndrome_bytes[0], 0x80u); // bit 0, MSB-first
}

// ---------------------------------------------------------------------------
// SessionRegistry: which kind of session a config entry builds
// ---------------------------------------------------------------------------

// A device_graph ring fires a captured cudaGraphExec_t, which a CPU-hosted
// graph has none of, so the combination is rejected at load — before the graph
// is built, and long before DecodingServer would fail on a missing capture.
TEST(SessionRegistryLoad, RejectsDeviceGraphDispatchForTaskGraphs) {
  const std::string yaml = R"(decoders:
  - id: 0
    dispatch: device_graph
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    task_graph:
      type: single_window
      inner_type: single_error_lut
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  SessionRegistry registry;
  EXPECT_THROW(registry.load_from_config(config, "device_graph_task_graph"),
               std::runtime_error);
}

// The `nodes`/`edges` topology form had no coverage through
// SessionRegistry/DecodingSession: only the single-node shorthand is
// exercised above and by TwoProcessHostDispatchTaskGraph. Same X/Z-split-XOR
// topology as DecodingTaskGraph.NodesFormMultiNodeGraphRunsOnFlatMatrices
// (proven there via the global decoder registry), driven here through the
// decoding-server session machinery instead.
TEST(SessionRegistryLoad, NodesFormMultiNodeGraphServesAShot) {
  const std::string yaml = R"(decoders:
  - id: 0
    task_graph:
      nodes:
        - id: x
          type: single_window
          inner: {type: single_error_lut}
        - id: z
          type: single_window
          inner: {type: single_error_lut}
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: x}
        - {from: measurements, to: z}
        - {from: x, to: combine.a}
        - {from: z, to: combine.b}
        - {from: combine, to: output}
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1, 1, -1]
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  SessionRegistry registry;
  registry.load_from_config(config, "nodes_form_task_graph");

  ASSERT_EQ(registry.dispatch_for(0), DecoderDispatch::host);
  auto *session = registry.find(0);
  ASSERT_NE(session, nullptr);
  EXPECT_EQ(session->dec, nullptr) << "a graph session owns no decoder";
  ASSERT_NE(session->graph, nullptr);

  std::vector<uint8_t> tx(64, 0xEE);
  auto eq = make_cqr_slot(kEnqueueSyndromesFunctionId, 1,
                          make_enqueue_payload(0, {0, 0}));
  session->handle_enqueue(eq.data(), tx.data(), eq.size());
  expect_tx_status(tx, RpcStatus::OK, 1);

  auto gc = make_cqr_slot(kGetCorrectionsFunctionId, 2,
                          make_get_corrections_payload(1, /*reset=*/true));
  session->handle_get_corrections(gc.data(), tx.data(), gc.size());
  expect_tx_status(tx, RpcStatus::OK, 2);
  const auto *response = reinterpret_cast<const RPCResponse *>(tx.data());
  ASSERT_EQ(response->result_len, 1u);
  EXPECT_EQ(tx[sizeof(RPCResponse)] & 1u, 0u)
      << "zero syndrome on both branches XORs to a zero correction";
}

// build_task_graph() never calls decoder::set_decoder_id() the way
// create_realtime_decoder() does for a plain decoder, so SessionRegistry must
// set it explicitly before capture: the generic get_corrections/reset_decoder
// DEVICE_CALL handlers look up g_decoder_state_table[decoder_id] keyed by the
// wire request's decoder_id (this config's id), and an inner decoder left at
// its default id (0) would register under the wrong slot for any id != 0.
TEST(SessionRegistryLoad, TaskGraphInnerDecoderGetsTheConfiguredId) {
  const std::string yaml = R"(decoders:
  - id: 3
    task_graph:
      type: single_window
      inner_type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  SessionRegistry registry;
  registry.load_from_config(config, "task_graph_decoder_id");

  auto *session = registry.find(3);
  ASSERT_NE(session, nullptr);
  ASSERT_NE(session->graph, nullptr);
  auto *dec = session->graph->inner_decoder();
  ASSERT_NE(dec, nullptr);
  EXPECT_EQ(dec->get_decoder_id(), 3u);
}

} // namespace
