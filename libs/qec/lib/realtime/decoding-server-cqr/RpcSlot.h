/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// In-place views over decoder-server RPC slots, and in-place response
/// writers.  The single point of truth for interpreting the wire layouts
/// pinned in decoder_rpc_wire_format.h (per decoder_server_runtime.md).
///
/// Every helper here is DELIBERATELY device-portable: no allocation, no
/// exceptions, no STL containers -- plain pointer/length views and stores --
/// so the same source can later be compiled into the GPU dispatch kernel
/// (the device dispatcher parses the identical slot layout; sharing the text
/// keeps the two dialects from drifting).

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudaq::qec::decoding_server::slot {

using cudaq::qec::decoding::rpc::bit_packed_bytes;
using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::qec::decoding::rpc::kMaxSyndromeBits;
using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
using cudaq::qec::decoding::rpc::RpcStatus;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

/// Validated view of an enqueue_syndromes request slot.
///
/// Wire layout (identical for the CUDAQ device_call lowering and the internal
/// EnqueueRequestPayload -- 4 u64s then the packed bits):
///   [RPCHeader][u64 decoder_id][u64 counter][u64 syndrome_mapping_id]
///   [u64 num_syndromes][u8 x ceil(num_syndromes/8), LSB-first]
/// The 4th u64 doubles as the std::vector<bool> array-length prefix of the
/// device_call ABI; the byte count is derived, never carried on the wire.
struct EnqueueView {
  const RPCHeader *header = nullptr;
  uint64_t decoder_id = 0;
  uint64_t counter = 0;
  uint64_t syndrome_mapping_id = 0;
  uint64_t num_syndromes = 0;
  uint64_t byte_count = 0;
  const uint8_t *packed_bits = nullptr;
};

/// Parse an enqueue request only after proving the advertised payload is
/// physically present in the supplied slot.
inline bool parse_enqueue(const void *rx_slot, std::size_t slot_size,
                          EnqueueView &out) {
  if (!rx_slot || slot_size < sizeof(RPCHeader))
    return false;

  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      header->function_id != kEnqueueSyndromesFunctionId)
    return false;

  const std::size_t physical_payload = slot_size - sizeof(RPCHeader);
  const std::size_t arg_len = header->arg_len;
  if (arg_len > physical_payload)
    return false;

  const auto *payload =
      static_cast<const uint8_t *>(rx_slot) + sizeof(RPCHeader);
  std::size_t offset = 0;
  auto read_u64 = [&](uint64_t &value) {
    if (offset > arg_len || sizeof(uint64_t) > arg_len - offset)
      return false;
    std::memcpy(&value, payload + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    return true;
  };

  EnqueueView parsed;
  parsed.header = header;
  if (!read_u64(parsed.decoder_id) || !read_u64(parsed.counter) ||
      !read_u64(parsed.syndrome_mapping_id) || !read_u64(parsed.num_syndromes))
    return false;

  parsed.byte_count =
      bit_packed_bytes(static_cast<std::size_t>(parsed.num_syndromes));
  if (parsed.num_syndromes == 0 || parsed.num_syndromes > kMaxSyndromeBits ||
      offset > arg_len || parsed.byte_count > arg_len - offset)
    return false;

  parsed.packed_bits = payload + offset;
  out = parsed;
  return true;
}

/// Validated view of a get_corrections request slot
/// ([RPCHeader][GetCorrectionsRequestPayload]).
struct GetCorrectionsView {
  const RPCHeader *header = nullptr;
  int64_t decoder_id = 0;
  int64_t return_size = 0;
  bool reset = false;
};

inline bool parse_get_corrections(const void *rx_slot, std::size_t slot_size,
                                  GetCorrectionsView &out) {
  using cudaq::qec::decoding::rpc::GetCorrectionsRequestPayload;
  if (!rx_slot || slot_size < sizeof(RPCHeader))
    return false;
  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      header->function_id != kGetCorrectionsFunctionId)
    return false;
  if (header->arg_len < sizeof(GetCorrectionsRequestPayload) ||
      header->arg_len > slot_size - sizeof(RPCHeader))
    return false;
  GetCorrectionsRequestPayload req;
  std::memcpy(&req, static_cast<const uint8_t *>(rx_slot) + sizeof(RPCHeader),
              sizeof(req));
  out.header = header;
  out.decoder_id = req.decoder_id;
  out.return_size = req.return_size;
  out.reset = req.reset != 0;
  return true;
}

/// Validated view of a reset_decoder request slot
/// ([RPCHeader][ResetRequestPayload]).
struct ResetView {
  const RPCHeader *header = nullptr;
  int64_t decoder_id = 0;
};

inline bool parse_reset(const void *rx_slot, std::size_t slot_size,
                        ResetView &out) {
  using cudaq::qec::decoding::rpc::ResetRequestPayload;
  if (!rx_slot || slot_size < sizeof(RPCHeader))
    return false;
  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      header->function_id != kResetDecoderFunctionId)
    return false;
  if (header->arg_len < sizeof(ResetRequestPayload) ||
      header->arg_len > slot_size - sizeof(RPCHeader))
    return false;
  ResetRequestPayload req;
  std::memcpy(&req, static_cast<const uint8_t *>(rx_slot) + sizeof(RPCHeader),
              sizeof(req));
  out.header = header;
  out.decoder_id = req.decoder_id;
  return true;
}

/// Routing peek: every decoder-server request payload carries decoder_id as
/// its first u64 (per decoder_server_runtime.md), so the shim can resolve the
/// session before parsing the full request.  A negative wire value maps to a
/// huge uint64 that no registry contains, so it resolves to INVALID_DECODER
/// exactly like today's int64 -> uint64 registry lookup.
inline bool peek_decoder_id(const void *rx_slot, std::size_t slot_size,
                            uint64_t &out) {
  if (!rx_slot || slot_size < sizeof(RPCHeader))
    return false;
  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return false;
  if (header->arg_len < sizeof(uint64_t) ||
      sizeof(uint64_t) > slot_size - sizeof(RPCHeader))
    return false;
  std::memcpy(&out, static_cast<const uint8_t *>(rx_slot) + sizeof(RPCHeader),
              sizeof(uint64_t));
  return true;
}

/// Callers gate on `slot_size >= sizeof(RPCHeader)` (dispatch_rpc) and then
/// write an RPCResponse into a slot of that same size, so the response header
/// must fit wherever a request header did.  Both are 24 bytes today; assert
/// the relation rather than leave it implicit -- these structs live in the
/// cudaq realtime headers and can change independently of this repo.
static_assert(sizeof(RPCResponse) <= sizeof(RPCHeader),
              "an RPCResponse must fit in any slot large enough to have "
              "carried an RPCHeader");

/// Write a header-only RPCResponse (no result payload) into \p tx_slot.
/// The magic is release-stored LAST so the CUDAQ runtime sees a complete
/// response before observing the magic word.
inline void write_response(void *tx_slot, uint32_t request_id,
                           uint64_t ptp_timestamp, RpcStatus status) noexcept {
  auto *resp = static_cast<RPCResponse *>(tx_slot);
  resp->status = static_cast<int32_t>(status);
  resp->result_len = 0;
  resp->request_id = request_id;
  resp->ptp_timestamp = ptp_timestamp;
  __atomic_store_n(reinterpret_cast<uint32_t *>(tx_slot),
                   cudaq::realtime::RPC_MAGIC_RESPONSE, __ATOMIC_RELEASE);
}

/// Payload-bearing response written straight into a tx slot: payload() hands
/// out tx_slot + sizeof(RPCResponse) after bounds-checking result_len against
/// the slot capacity (nullptr on overflow -- truncating would advertise bytes
/// that were never written, so the caller must fail the RPC explicitly
/// instead), and commit() fills the header and release-stores the magic last.
class ResultWriter {
public:
  ResultWriter(void *tx_slot, std::size_t slot_size) noexcept
      : tx_(tx_slot), capacity_(slot_size) {}

  /// Bytes available for the result after the response header, 0 if the slot
  /// cannot even hold the header.  Order matters: the capacity check must
  /// precede any subtraction, or a short slot underflows to a huge size_t.
  std::size_t payload_capacity() const noexcept {
    if (!tx_ || capacity_ < sizeof(RPCResponse))
      return 0;
    return capacity_ - sizeof(RPCResponse);
  }

  uint8_t *payload(std::size_t result_len) noexcept {
    if (!tx_ || result_len > payload_capacity())
      return nullptr;
    return static_cast<uint8_t *>(tx_) + sizeof(RPCResponse);
  }

  void commit(RpcStatus status, uint32_t request_id, uint64_t ptp_timestamp,
              std::size_t result_len) noexcept {
    auto *resp = static_cast<RPCResponse *>(tx_);
    resp->status = static_cast<int32_t>(status);
    resp->result_len = static_cast<uint32_t>(result_len);
    resp->request_id = request_id;
    resp->ptp_timestamp = ptp_timestamp;
    __atomic_store_n(reinterpret_cast<uint32_t *>(tx_),
                     cudaq::realtime::RPC_MAGIC_RESPONSE, __ATOMIC_RELEASE);
  }

private:
  void *tx_ = nullptr;
  std::size_t capacity_ = 0;
};

} // namespace cudaq::qec::decoding_server::slot
