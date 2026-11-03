/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::decoding_server {

using cudaq::qec::round_type;

/// One row of the syndrome mapping table: the round shape the row describes,
/// and per virtual port the round-relative detector indices that port's bits
/// scatter into.
///
/// The type is what carries end-of-shot to the decode target.  A caller picks
/// a row per round via the `syndrome_mapping_id` in its enqueue payload, so
/// tagging the last round means naming a FINAL row rather than adding anything
/// to the wire.
struct syndrome_mapping_row {
  round_type type = round_type::BULK;
  /// Only the single pass-through port (one empty list) is honored today; the
  /// config layer refuses anything else.
  std::vector<std::vector<uint32_t>> vps;
};

/// table[syndrome_mapping_id] -> one row.
using SyndromeMappingTable = std::unordered_map<uint64_t, syndrome_mapping_row>;

/// The table a session gets when a configuration declares none: a single
/// pass-through BULK row under id 0, which is what every configuration written
/// before this section existed behaves as.
inline SyndromeMappingTable default_syndrome_mapping_table() {
  return {{0, syndrome_mapping_row{round_type::BULK, {{}}}}};
}

/// Flatten the table into the POD the DEVICE-mode enqueue handler reads.
///
/// Only `declared` is load bearing: the handler validates the request's id
/// against it and reads nothing else per row (see DeviceSyndromeMappingTable on
/// why a shape is inert on the device). The shapes travel anyway, so the device
/// table stays the same table the host resolves against.
///
/// @throws std::runtime_error when an id exceeds what the device table holds.
/// validate_syndrome_mapping() enforces the same bound, so reaching this means
/// a programmatically built table that never passed through it.
inline cudaq::qec::decoding::rpc::DeviceSyndromeMappingTable
device_syndrome_mapping_table(const SyndromeMappingTable &table) {
  cudaq::qec::decoding::rpc::DeviceSyndromeMappingTable out{};
  for (const auto &[id, row] : table) {
    if (id >= cudaq::qec::decoding::rpc::kMaxSyndromeMappingRows)
      throw std::runtime_error(
          "syndrome_mapping: id " + std::to_string(id) +
          " is out of range; ids must be below " +
          std::to_string(cudaq::qec::decoding::rpc::kMaxSyndromeMappingRows));
    out.shapes[id] = static_cast<std::uint8_t>(row.type);
    out.declared |= static_cast<std::uint16_t>(1u << id);
  }
  return out;
}

/// True when \p table names a row the device default cannot serve.  The
/// handler's default (nothing installed) services id 0 alone, which is exactly
/// what a document declaring only the pass-through row asks for.
inline bool needs_device_syndrome_mapping_install(
    const cudaq::qec::decoding::rpc::DeviceSyndromeMappingTable &table) {
  constexpr std::uint16_t pass_through_only = 1u; // bit 0 == id 0
  return (table.declared & ~pass_through_only) != 0;
}

} // namespace cudaq::qec::decoding_server
