/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Checks the decoding server's HOST_CALL table against the request payload
// structs in decoder_rpc_wire_format.h: callers serialize a request from an
// entry's schema, and the server parses it with the struct.

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" const cudaq_function_entry_t *
cudaqx_qec_decoding_server_host_call_table(std::uint32_t *count);

namespace rpc = cudaq::qec::decoding::rpc;

namespace {

// The table's entry for @p function_id, or null. With no decoder config set,
// no decoder starts.
const cudaq_function_entry_t *find_entry(std::uint32_t function_id) {
  std::uint32_t count = 0;
  const cudaq_function_entry_t *entries = nullptr;

  ::unsetenv("CUDAQ_QEC_DECODER_CONFIG");
  entries = cudaqx_qec_decoding_server_host_call_table(&count);
  for (std::uint32_t i = 0; i < count; ++i)
    if (entries[i].function_id == function_id)
      return &entries[i];
  return nullptr;
}

// The bytes of an entry's fixed-size arguments; a bit-packed one counts 0.
std::size_t request_bytes(const cudaq_function_entry_t &entry) {
  std::size_t bytes = 0;

  for (std::uint8_t i = 0; i < entry.schema.num_args; ++i)
    bytes += entry.schema.args[i].size_bytes;
  return bytes;
}

} // namespace

// get_logical_measurements takes decoder_id, first, return_size and reset,
// and returns bit-packed bits.
TEST(DecodingServerHostCallTable, LogicalMeasurementsTakeARange) {
  const cudaq_function_entry_t *entry =
      find_entry(rpc::kGetLogicalMeasurementsFunctionId);

  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->dispatch_mode, CUDAQ_DISPATCH_HOST_CALL);
  ASSERT_EQ(entry->schema.num_args, 4);
  EXPECT_EQ(entry->schema.args[0].type_id, CUDAQ_TYPE_INT64);
  EXPECT_EQ(entry->schema.args[1].type_id, CUDAQ_TYPE_INT64);
  EXPECT_EQ(entry->schema.args[2].type_id, CUDAQ_TYPE_INT64);
  EXPECT_EQ(entry->schema.args[3].type_id, CUDAQ_TYPE_UINT8);
  ASSERT_EQ(entry->schema.num_results, 1);
  EXPECT_EQ(entry->schema.results[0].type_id, CUDAQ_TYPE_BIT_PACKED);
}

// Each fixed-size request's arguments lay out exactly its payload struct.
TEST(DecodingServerHostCallTable, ArgumentsLayOutEachPayload) {
  const cudaq_function_entry_t *corrections =
      find_entry(rpc::kGetCorrectionsFunctionId);
  const cudaq_function_entry_t *logical =
      find_entry(rpc::kGetLogicalMeasurementsFunctionId);
  const cudaq_function_entry_t *reset =
      find_entry(rpc::kResetDecoderFunctionId);

  ASSERT_NE(corrections, nullptr);
  ASSERT_NE(logical, nullptr);
  ASSERT_NE(reset, nullptr);
  EXPECT_EQ(request_bytes(*corrections),
            sizeof(rpc::GetCorrectionsRequestPayload));
  EXPECT_EQ(request_bytes(*logical),
            sizeof(rpc::GetLogicalMeasurementsRequestPayload));
  EXPECT_EQ(request_bytes(*reset), sizeof(rpc::ResetRequestPayload));
}
