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

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace cudaq::qec::decoding_server;
using namespace cudaq::qec::decoding::rpc;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

class ControlledDecoder final : public cudaq::qec::decoder {
public:
  ControlledDecoder()
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
            cudaq::qec::decode_result_type::errors) {}

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    if (throw_on_decode)
      throw std::runtime_error("controlled decoder failure");
    cudaq::qec::decoder_result result;
    result.converged = converged;
    result.result = {syndrome.at(0)};
    return result;
  }

  bool converged = false;
  bool throw_on_decode = false;
};

std::pair<std::unique_ptr<DecodingSession>, ControlledDecoder *>
make_session() {
  auto decoder = std::make_unique<ControlledDecoder>();
  auto *raw_decoder = decoder.get();
  return {DecodingSession::create(std::move(decoder)), raw_decoder};
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

} // namespace
