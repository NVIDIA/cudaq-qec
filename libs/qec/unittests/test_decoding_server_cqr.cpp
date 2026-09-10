/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_cc_test_helpers.h"

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/device_call_service.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" const cudaq_function_entry_t *
cudaqx_qec_decoding_server_host_call_table(std::uint32_t *count);
extern "C" void cudaqx_qec_decoding_server_shutdown();
extern "C" void *cudaqx_qec_decoding_server_graph_resources(std::uint64_t);
extern "C" cudaq::realtime::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo();

namespace {

using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

std::vector<uint8_t> make_slot(uint32_t function_id, uint32_t request_id,
                               const std::vector<uint8_t> &payload) {
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

std::vector<uint8_t> enqueue_payload(uint64_t decoder_id) {
  std::vector<uint8_t> payload(4 * sizeof(uint64_t) + 1, 0);
  const uint64_t fields[4] = {decoder_id, 0, 0, 1};
  std::memcpy(payload.data(), fields, sizeof(fields));
  payload[sizeof(fields)] = 1;
  return payload;
}

const cudaq_function_entry_t *enqueue_entry(std::uint32_t *count) {
  const auto *entries = cudaqx_qec_decoding_server_host_call_table(count);
  if (!entries || !*count)
    return nullptr;
  return entries;
}

int run_helper(const char *name) {
  if (std::strcmp(name, "graph_before_init") == 0)
    return cudaqx_qec_decoding_server_graph_resources(0) ? 2 : 0;
  if (std::strcmp(name, "bad_config") == 0) {
    std::uint32_t count = 99;
    const auto *entries = cudaqx_qec_decoding_server_host_call_table(&count);
    if (entries != nullptr || count != 0)
      return 2;
    auto info = cudaqGetDeviceCallServicePluginInfo();
    if (!info.getService)
      return 3;
    auto *svc = info.getService();
    if (!svc)
      return 4;
    auto session = svc->createDispatchSession(
        cudaq::realtime::DeviceCallDispatchMode::Host);
    return session ? 5 : 0;
  }
  if (std::strcmp(name, "lazy_init_throw") == 0) {
    ::unsetenv("CUDAQ_QEC_DECODER_CONFIG");
    std::uint32_t count = 0;
    const auto *entries = cudaqx_qec_decoding_server_host_call_table(&count);
    if (!entries || count == 0)
      return 2;
    auto rx = make_slot(kEnqueueSyndromesFunctionId, 1, enqueue_payload(0));
    std::vector<uint8_t> tx(64, 0);
    entries[0].handler.host_fn(rx.data(), tx.data(), rx.size());
    const auto *resp = reinterpret_cast<const RPCResponse *>(tx.data());
    return resp->status ==
                   static_cast<int32_t>(
                       cudaq::qec::decoding::rpc::RpcStatus::INTERNAL_ERROR)
               ? 0
               : 3;
  }
  return 1;
}

TEST(DecodingServerCqr, DispatchRpcFailureModesAndShutdown) {
  const auto path = qec_cc::write_temp(qec_cc::lut_yaml());
  ::setenv("CUDAQ_QEC_DECODER_CONFIG", path.c_str(), 1);
  std::uint32_t count = 0;
  const auto *entries = enqueue_entry(&count);
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(count, 3u);

  auto host_fn = entries[0].handler.host_fn;
  ASSERT_NE(host_fn, nullptr);

  std::vector<uint8_t> tx(64, 0);
  host_fn(nullptr, tx.data(), 64);
  host_fn(tx.data(), nullptr, 64);
  auto tiny = make_slot(kEnqueueSyndromesFunctionId, 1, enqueue_payload(0));
  host_fn(tiny.data(), tx.data(), 4);

  auto bad_magic =
      make_slot(kEnqueueSyndromesFunctionId, 2, enqueue_payload(0));
  reinterpret_cast<RPCHeader *>(bad_magic.data())->magic = 0xDEADBEEF;
  host_fn(bad_magic.data(), tx.data(), bad_magic.size());

  auto no_id = make_slot(kEnqueueSyndromesFunctionId, 3, {});
  host_fn(no_id.data(), tx.data(), no_id.size());

  auto unknown = make_slot(kEnqueueSyndromesFunctionId, 4, enqueue_payload(99));
  host_fn(unknown.data(), tx.data(), unknown.size());
  const auto *resp = reinterpret_cast<const RPCResponse *>(tx.data());
  EXPECT_EQ(resp->status,
            static_cast<int32_t>(
                cudaq::qec::decoding::rpc::RpcStatus::INVALID_DECODER));

  EXPECT_EQ(cudaqx_qec_decoding_server_graph_resources(0), nullptr);

  cudaqx_qec_decoding_server_shutdown();
  auto after = make_slot(kEnqueueSyndromesFunctionId, 5, enqueue_payload(0));
  host_fn(after.data(), tx.data(), after.size());
  resp = reinterpret_cast<const RPCResponse *>(tx.data());
  EXPECT_EQ(resp->status,
            static_cast<int32_t>(
                cudaq::qec::decoding::rpc::RpcStatus::INTERNAL_ERROR));
}

TEST(DecodingServerCqr, DeviceCallServiceNonHostIsNull) {
  auto info = cudaqGetDeviceCallServicePluginInfo();
  ASSERT_NE(info.pluginName, nullptr);
  ASSERT_NE(info.getService, nullptr);
  auto *svc = info.getService();
  ASSERT_NE(svc, nullptr);
  EXPECT_EQ(
      svc->createDispatchSession(cudaq::realtime::DeviceCallDispatchMode::Gpu),
      nullptr);
}

TEST(DecodingServerCqr, IsolatedHelpers) {
  EXPECT_EQ(qec_cc::exec_self("graph_before_init"), 0);
  EXPECT_EQ(qec_cc::exec_self("bad_config",
                              {{"CUDAQ_QEC_DECODER_CONFIG", "/no/such.yaml"}}),
            0);
  EXPECT_EQ(qec_cc::exec_self("lazy_init_throw"), 0);
}

} // namespace

int main(int argc, char **argv) {
  if (const char *helper = qec_cc::helper_name(argc, argv))
    return run_helper(helper);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
