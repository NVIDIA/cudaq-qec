/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingServer.h"
#include "ITransceiver.h"
#include "cc_test_graph_decoder.h"
#include "qec_cc_test_helpers.h"
#include "qec_test_cuda_pass.h"
#include "../lib/hardware_guards.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::qec {
CUDAQ_EXT_PT_REGISTER_TYPE(cc_test_graph_decoder)
CUDAQ_EXT_PT_REGISTER_TYPE(cc_test_null_graph_decoder)
} // namespace cudaq::qec

namespace {

using namespace cudaq::qec::decoding_server;
using cudaq::qec::decoding::config::transport_shape_override;

std::atomic<int> g_launch_mode{0}; // 0 success, 1 return false
std::atomic<int> g_factory_calls{0};

class FakeTransceiver final : public ITransceiver {
public:
  RxFrame recv() override {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stopped_; });
    return {};
  }
  void send(const PeerId &, const uint8_t *, size_t) override {}
  void shutdown() override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stopped_ = true;
    }
    cv_.notify_all();
  }
  bool launch_device_scheduler(void *graph_resources) override {
    stored_ = graph_resources;
    return g_launch_mode.load() == 0;
  }
  void *stored_ = nullptr;

private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool stopped_ = false;
};

} // namespace

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver(
    int, const cudaq::qec::decoding::config::transport_shape_override *) {
  ++g_factory_calls;
  return new FakeTransceiver();
}

namespace {

std::string graph_yaml(const char *type, int n_decoders = 1) {
  std::string yaml = "decoders:\n";
  for (int i = 0; i < n_decoders; ++i) {
    yaml += "  - id: " + std::to_string(i) + "\n    type: " + type +
            "\n    dispatch: device_graph\n"
            "    block_size: 1\n    syndrome_size: 1\n"
            "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n"
            "    D_sparse: [0, -1]\n";
  }
  yaml += "transport:\n  provider: gpu_roce\n";
  return yaml;
}

TEST(DecodingServerLifecycle, ConstructsRunsStopsAndReportsStats) {
  g_launch_mode.store(0);
  cudaq::qec::cc_test_graph_decoder::release_count.store(0);
  cudaq::qec::cc_test_graph_decoder::last_reserved_sms.store(-1);
  ::setenv("QEC_DEVICE_GRAPH_RESERVED_SMS", "1", 1);
  const auto path = qec_cc::write_temp(graph_yaml("cc_test_graph_decoder"));
  DecodingServer server(path);
  EXPECT_NE(server.graph_resources_for(0), nullptr);
  EXPECT_EQ(server.graph_resources_for(99), nullptr);
  EXPECT_EQ(cudaq::qec::cc_test_graph_decoder::last_reserved_sms.load(), 1);
  server.print_session_stats();
  std::thread runner([&server] { server.run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  server.stop();
  runner.join();
  std::filesystem::remove(path);
}

TEST(DecodingServerLifecycle, JunkReservedSmsKeepsFloor) {
  ::setenv("QEC_DEVICE_GRAPH_RESERVED_SMS", "nope", 1);
  const auto path = qec_cc::write_temp(graph_yaml("cc_test_graph_decoder"));
  DecodingServer server(path);
  EXPECT_EQ(cudaq::qec::cc_test_graph_decoder::last_reserved_sms.load(), 1);
  std::filesystem::remove(path);
}

TEST(DecodingServerLifecycle, LaunchFalseThrows) {
  g_launch_mode.store(1);
  const auto path = qec_cc::write_temp(graph_yaml("cc_test_graph_decoder"));
  EXPECT_THROW(DecodingServer server(path), std::runtime_error);
  g_launch_mode.store(0);
  std::filesystem::remove(path);
}

TEST(DecodingServerLifecycle, TwoGraphDecodersThrow) {
  g_launch_mode.store(0);
  const auto path = qec_cc::write_temp(graph_yaml("cc_test_graph_decoder", 2));
  EXPECT_THROW(DecodingServer server(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(DecodingServerLifecycle, NullCaptureThrows) {
  g_launch_mode.store(0);
  const auto path =
      qec_cc::write_temp(graph_yaml("cc_test_null_graph_decoder"));
  EXPECT_THROW(DecodingServer server(path), std::runtime_error);
  std::filesystem::remove(path);
}

class pinned_test_decoder : public cudaq::qec::decoder {
public:
  int capture_device = -99;
  pinned_test_decoder()
      : decoder(
            cudaq::qec::decoder_init(
                cudaq::qec::sparse_binary_matrix::from_nested_csr(1, 1, {{0}})),
            cudaq::qec::decode_result_type::errors) {
    cuda_device_id_ = 1;
  }
  void set_cuda_device_id(int id) { cuda_device_id_ = id; }
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {};
  }
  void *capture_decode_graph(int) override {
    int dev = -1;
    (void)cudaGetDevice(&dev);
    capture_device = dev;
    return this;
  }
};

class HardwareGuards : public ::testing::Test {
protected:
  // Restore the always-success wrap after each case so the original
  // DecodingServerLifecycle tests stay equivalent under gtest shuffle.
  void TearDown() override { qec_test_cuda().reset(); }
};

TEST_F(HardwareGuards, CachedPinAndCaptureObserveDevice) {
  qec_test_cuda().reset();
  qec_test_cuda().current = 0;
  pinned_test_decoder dec;
  cudaq::qec::detail_affinity::pin_decode_device_cached(dec);
  ASSERT_EQ(qec_test_cuda().sets.size(), 1u);
  EXPECT_EQ(qec_test_cuda().sets[0], 1);
  dec.set_cuda_device_id(2);
  cudaq::qec::detail_affinity::pin_decode_device_cached(dec);
  ASSERT_EQ(qec_test_cuda().sets.size(), 2u);
  EXPECT_EQ(qec_test_cuda().sets[1], 2);
  EXPECT_EQ(qec_test_cuda().current, 2);

  void *graph = cudaq::qec::detail_affinity::capture_graph_pinned(dec);
  EXPECT_EQ(graph, &dec);
  EXPECT_EQ(dec.capture_device, 2);
}

TEST_F(HardwareGuards, CudaDeviceGuardRestoreCases) {
  {
    qec_test_cuda().reset();
    qec_test_cuda().current = 0;
    qec_test_cuda().get_status = cudaErrorUnknown;
    qec_test_cuda().count = 8;
    cudaq::qec::detail_affinity::CudaDeviceGuard guard(1);
    EXPECT_EQ(qec_test_cuda().sets, (std::vector<int>{1}));
  }
  EXPECT_EQ(qec_test_cuda().sets, (std::vector<int>{1}));

  qec_test_cuda().reset();
  qec_test_cuda().set_status = cudaErrorUnknown;
  EXPECT_THROW(cudaq::qec::detail_affinity::CudaDeviceGuard guard(1),
               std::runtime_error);

  {
    qec_test_cuda().reset();
    qec_test_cuda().current = 3;
    cudaq::qec::detail_affinity::CudaDeviceGuard guard(3);
    EXPECT_TRUE(qec_test_cuda().sets.empty() ||
                qec_test_cuda().sets == std::vector<int>{3});
  }
  EXPECT_EQ(qec_test_cuda().sets.size(), 1u);

  {
    qec_test_cuda().reset();
    qec_test_cuda().current = 1;
    cudaq::qec::detail_affinity::CudaDeviceGuard guard(2);
    EXPECT_EQ(qec_test_cuda().sets, (std::vector<int>{2}));
  }
  EXPECT_EQ(qec_test_cuda().sets, (std::vector<int>{2, 1}));
}

} // namespace
