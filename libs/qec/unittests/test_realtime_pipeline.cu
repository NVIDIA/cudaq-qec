/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

using namespace cudaq::qec;
using namespace cudaq::qec::realtime::experimental;
namespace rt_exp = cudaq::qec::realtime::experimental;
namespace rt_sdk = cudaq::realtime;

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

static constexpr size_t kSkipTrtFloats = 1600;
static constexpr size_t kSkipTrtBytes = kSkipTrtFloats * sizeof(float);
static constexpr size_t kSlotSize = 8192;
static constexpr size_t kNumSlots = 8;
static constexpr uint32_t kTestFunctionId =
    rt_sdk::fnv1a_hash("test_predecoder");

// Helper function to check if a GPU is available
bool isGpuAvailable() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return (err == cudaSuccess && deviceCount > 0);
}

// ============================================================================
// Pre-launch DMA callback (mirrors production code)
// ============================================================================

struct PreLaunchCopyCtx {
  void *d_trt_input;
  size_t input_size;
  void **h_ring_ptrs;
};

static void pre_launch_input_copy(void *user_data, void *slot_dev,
                                  cudaStream_t stream) {
  auto *ctx = static_cast<PreLaunchCopyCtx *>(user_data);
  ctx->h_ring_ptrs[0] = slot_dev;
  cudaMemcpyAsync(ctx->d_trt_input,
                  static_cast<uint8_t *>(slot_dev) + CUDAQ_RPC_HEADER_SIZE,
                  ctx->input_size, cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// Ring buffer helpers (mapped pinned memory)
// ============================================================================

static bool allocate_mapped_buffer(size_t size, uint8_t **host_out,
                                   uint8_t **dev_out) {
  void *h = nullptr;
  if (cudaHostAlloc(&h, size, cudaHostAllocMapped) != cudaSuccess)
    return false;
  void *d = nullptr;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
    cudaFreeHost(h);
    return false;
  }
  std::memset(h, 0, size);
  *host_out = static_cast<uint8_t *>(h);
  *dev_out = static_cast<uint8_t *>(d);
  return true;
}

static void free_mapped_buffer(uint8_t *host_ptr) {
  if (host_ptr)
    cudaFreeHost(host_ptr);
}

// ============================================================================
// Write an RPC request (RPCHeader + payload) into a mapped buffer slot
// ============================================================================

static void write_rpc_slot(uint8_t *slot_host, uint32_t function_id,
                           const void *payload, size_t payload_len) {
  rt_sdk::RPCHeader hdr{};
  hdr.magic = rt_sdk::RPC_MAGIC_REQUEST;
  hdr.function_id = function_id;
  hdr.arg_len = static_cast<uint32_t>(payload_len);
  std::memcpy(slot_host, &hdr, sizeof(hdr));
  if (payload && payload_len > 0)
    std::memcpy(slot_host + sizeof(hdr), payload, payload_len);
}

// ============================================================================
// Test Fixture
// ============================================================================

class RealtimePipelineTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping realtime pipeline tests";
    }

    ASSERT_TRUE(allocate_mapped_buffer(kNumSlots * sizeof(uint64_t),
                                       &rx_flags_host_, &rx_flags_dev_));
    ASSERT_TRUE(allocate_mapped_buffer(kNumSlots * sizeof(uint64_t),
                                       &tx_flags_host_, &tx_flags_dev_));
    ASSERT_TRUE(allocate_mapped_buffer(kNumSlots * kSlotSize, &rx_data_host_,
                                       &rx_data_dev_));
    ASSERT_TRUE(allocate_mapped_buffer(kNumSlots * kSlotSize, &tx_data_host_,
                                       &tx_data_dev_));

    CUDA_CHECK(cudaHostAlloc(&mailbox_bank_host_, kMaxWorkers * sizeof(void *),
                             cudaHostAllocMapped));
    std::memset(mailbox_bank_host_, 0, kMaxWorkers * sizeof(void *));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&mailbox_bank_dev_), mailbox_bank_host_, 0));

    CUDA_CHECK(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_)
      cudaStreamDestroy(stream_);
    if (mailbox_bank_host_)
      cudaFreeHost(mailbox_bank_host_);
    free_mapped_buffer(rx_flags_host_);
    free_mapped_buffer(tx_flags_host_);
    free_mapped_buffer(rx_data_host_);
    free_mapped_buffer(tx_data_host_);
  }

  std::unique_ptr<ai_predecoder_service> create_predecoder(int mailbox_idx) {
    auto pd = ai_predecoder_service::create_passthrough(
        reinterpret_cast<void **>(mailbox_bank_dev_ + mailbox_idx));
    pd->capture_graph(stream_, false);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    return pd;
  }

  void submit_rpc_to_slot(size_t slot, uint32_t function_id,
                          const void *payload, size_t payload_len) {
    uint8_t *slot_host = rx_data_host_ + slot * kSlotSize;
    write_rpc_slot(slot_host, function_id, payload, payload_len);
    auto *flags = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
    flags[slot].store(reinterpret_cast<uint64_t>(slot_host),
                      cuda::std::memory_order_release);
  }

  bool wait_ready_flag(ai_predecoder_service *pd, int timeout_ms = 2000) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
      auto *flags = pd->get_host_ready_flags();
      int val = flags[0].load(cuda::std::memory_order_acquire);
      if (val >= 1)
        return true;
      usleep(100);
    }
    return false;
  }

  static constexpr size_t kMaxWorkers = 8;

  uint8_t *rx_flags_host_ = nullptr;
  uint8_t *rx_flags_dev_ = nullptr;
  uint8_t *tx_flags_host_ = nullptr;
  uint8_t *tx_flags_dev_ = nullptr;
  uint8_t *rx_data_host_ = nullptr;
  uint8_t *rx_data_dev_ = nullptr;
  uint8_t *tx_data_host_ = nullptr;
  uint8_t *tx_data_dev_ = nullptr;
  void **mailbox_bank_host_ = nullptr;
  void **mailbox_bank_dev_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

// ============================================================================
// ai_decoder_service Unit Tests (passthrough)
// ============================================================================

TEST_F(RealtimePipelineTest, SkipTrtSizes) {
  auto svc = ai_decoder_service::create_passthrough(mailbox_bank_dev_);
  EXPECT_EQ(svc->get_input_size(), kSkipTrtBytes);
  EXPECT_EQ(svc->get_output_size(), kSkipTrtBytes);
}

TEST_F(RealtimePipelineTest, SkipTrtBuffersAllocated) {
  auto svc = ai_decoder_service::create_passthrough(mailbox_bank_dev_);
  EXPECT_NE(svc->get_trt_input_ptr(), nullptr);
}

TEST_F(RealtimePipelineTest, SkipTrtGraphExecNull_BeforeCapture) {
  auto svc = ai_decoder_service::create_passthrough(mailbox_bank_dev_);
  EXPECT_EQ(svc->get_executable_graph(), nullptr);
}

// ============================================================================
// ai_predecoder_service Unit Tests (passthrough)
// ============================================================================

TEST_F(RealtimePipelineTest, PreDecoderConstruction) {
  auto pd = create_predecoder(0);
  EXPECT_NE(pd->get_host_ready_flags(), nullptr);
  EXPECT_NE(pd->get_host_ring_ptrs(), nullptr);
  EXPECT_EQ(pd->get_queue_depth(), 1);
  EXPECT_EQ(pd->get_input_size(), kSkipTrtBytes);
  EXPECT_EQ(pd->get_output_size(), kSkipTrtBytes);
}

TEST_F(RealtimePipelineTest, PreDecoderGraphCaptured) {
  auto pd = create_predecoder(0);
  EXPECT_NE(pd->get_executable_graph(), nullptr);
}

TEST_F(RealtimePipelineTest, PollReturnsFalseWhenIdle) {
  auto pd = create_predecoder(0);
  pre_decoder_job job{};
  EXPECT_FALSE(pd->poll_next_job(job));
}

TEST_F(RealtimePipelineTest, PollAndRelease) {
  auto pd = create_predecoder(0);

  auto *flags = pd->get_host_ready_flags();
  flags[0].store(1, cuda::std::memory_order_release);

  pre_decoder_job job{};
  EXPECT_TRUE(pd->poll_next_job(job));
  EXPECT_EQ(job.slot_idx, 0);
  EXPECT_NE(job.inference_data, nullptr);

  int val = flags[0].load(cuda::std::memory_order_acquire);
  EXPECT_EQ(val, 2);

  pd->release_job(0);
  val = flags[0].load(cuda::std::memory_order_acquire);
  EXPECT_EQ(val, 0);
}

TEST_F(RealtimePipelineTest, GraphLaunchableFromHost) {
  auto pd = create_predecoder(0);
  cudaGraphExec_t exec = pd->get_executable_graph();
  ASSERT_NE(exec, nullptr);

  CUDA_CHECK(cudaGraphLaunch(exec, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ============================================================================
// Correctness Tests (Identity Passthrough)
//
// Data flow: payload -> (pre_launch DMA to d_trt_input_) ->
//   passthrough_copy_kernel (identity) -> d_trt_output_ ->
//   cudaMemcpyAsync -> d_outputs_ (mapped pinned) ->
//   poll_next_job() -> inference_data
// ============================================================================

class CorrectnessTest : public RealtimePipelineTest {
protected:
  void SetUp() override {
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping correctness tests";
    }
    RealtimePipelineTest::SetUp();
  }

  void run_passthrough(ai_predecoder_service *pd, int mailbox_idx,
                       const float *payload, size_t num_floats, float *output) {
    size_t payload_bytes = num_floats * sizeof(float);
    ASSERT_LE(payload_bytes, kSkipTrtBytes);

    uint8_t *slot_host = rx_data_host_;
    write_rpc_slot(slot_host, kTestFunctionId, payload, payload_bytes);

    ptrdiff_t offset = slot_host - rx_data_host_;
    void *slot_dev = static_cast<void *>(rx_data_dev_ + offset);

    PreLaunchCopyCtx ctx;
    ctx.d_trt_input = pd->get_trt_input_ptr();
    ctx.input_size = pd->get_input_size();
    ctx.h_ring_ptrs = pd->get_host_ring_ptrs();

    pre_launch_input_copy(&ctx, slot_dev, stream_);
    CUDA_CHECK(cudaGraphLaunch(pd->get_executable_graph(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    ASSERT_TRUE(wait_ready_flag(pd));

    pre_decoder_job job{};
    ASSERT_TRUE(pd->poll_next_job(job));
    std::memcpy(output, job.inference_data, payload_bytes);
    pd->release_job(0);
  }
};

TEST_F(CorrectnessTest, IdentityPassthrough_Zeros) {
  auto pd = create_predecoder(0);
  float input[kSkipTrtFloats] = {};
  float output[kSkipTrtFloats];
  std::memset(output, 0xFF, sizeof(output));

  run_passthrough(pd.get(), 0, input, kSkipTrtFloats, output);
  EXPECT_EQ(std::memcmp(input, output, kSkipTrtBytes), 0)
      << "Zero payload should pass through unchanged";
}

TEST_F(CorrectnessTest, IdentityPassthrough_KnownPattern) {
  auto pd = create_predecoder(0);
  float input[kSkipTrtFloats];
  for (size_t i = 0; i < kSkipTrtFloats; ++i)
    input[i] = static_cast<float>(i + 1);
  float output[kSkipTrtFloats] = {};

  run_passthrough(pd.get(), 0, input, kSkipTrtFloats, output);
  EXPECT_EQ(std::memcmp(input, output, kSkipTrtBytes), 0)
      << "Known pattern {1..16} should pass through unchanged";
}

TEST_F(CorrectnessTest, IdentityPassthrough_RandomData) {
  auto pd = create_predecoder(0);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1e6f, 1e6f);

  float input[kSkipTrtFloats];
  for (size_t i = 0; i < kSkipTrtFloats; ++i)
    input[i] = dist(rng);
  float output[kSkipTrtFloats] = {};

  run_passthrough(pd.get(), 0, input, kSkipTrtFloats, output);
  EXPECT_EQ(std::memcmp(input, output, kSkipTrtBytes), 0)
      << "Random payload should pass through bitwise-identical";
}

TEST_F(CorrectnessTest, IdentityPassthrough_MaxValues) {
  auto pd = create_predecoder(0);
  std::vector<float> input(kSkipTrtFloats);
  const float extremes[] = {FLT_MAX,  -FLT_MAX,    FLT_MIN,     -FLT_MIN,
                            INFINITY, -INFINITY,   NAN,         0.0f,
                            -0.0f,    1.0f,        -1.0f,       1e-38f,
                            1e38f,    3.14159265f, 2.71828183f, 0.5f};
  for (size_t i = 0; i < kSkipTrtFloats; ++i)
    input[i] = extremes[i % (sizeof(extremes) / sizeof(extremes[0]))];
  std::vector<float> output(kSkipTrtFloats, 0.0f);

  run_passthrough(pd.get(), 0, input.data(), kSkipTrtFloats, output.data());
  EXPECT_EQ(std::memcmp(input.data(), output.data(), kSkipTrtBytes), 0)
      << "Extreme float values should pass through bitwise-identical";
}

TEST_F(CorrectnessTest, IdentityPassthrough_MultipleRequests) {
  auto pd = create_predecoder(0);
  constexpr int kNumRequests = 5000;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
  int failures = 0;

  for (int r = 0; r < kNumRequests; ++r) {
    float input[kSkipTrtFloats];
    for (size_t i = 0; i < kSkipTrtFloats; ++i)
      input[i] = dist(rng);
    float output[kSkipTrtFloats] = {};

    run_passthrough(pd.get(), 0, input, kSkipTrtFloats, output);
    if (std::memcmp(input, output, kSkipTrtBytes) != 0) {
      failures++;
      if (failures <= 5)
        ADD_FAILURE() << "Request " << r << ": output does not match input";
    }
  }
  EXPECT_EQ(failures, 0) << failures << " of " << kNumRequests
                         << " requests had mismatched output";
}

// ============================================================================
// Host Dispatcher Unit Tests
// ============================================================================

class HostDispatcherTest : public RealtimePipelineTest {
protected:
  void SetUp() override {
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping host dispatcher tests";
    }
    RealtimePipelineTest::SetUp();
    idle_mask_ = new atomic_uint64_sys(0);
    live_dispatched_ = new atomic_uint64_sys(0);
    inflight_slot_tags_ = new int[kMaxWorkers]();
    shutdown_flag_ = new atomic_int_sys(0);
    stats_counter_ = 0;
    function_table_ = new cudaq_function_entry_t[kMaxWorkers];
    std::memset(function_table_, 0,
                kMaxWorkers * sizeof(cudaq_function_entry_t));
  }

  void TearDown() override {
    if (shutdown_flag_ && !loop_stopped_) {
      shutdown_flag_->store(1, cuda::std::memory_order_release);
      __sync_synchronize();
      if (loop_thread_.joinable())
        loop_thread_.join();
    }
    for (auto &s : worker_streams_) {
      if (s)
        cudaStreamDestroy(s);
    }
    delete idle_mask_;
    delete live_dispatched_;
    delete[] inflight_slot_tags_;
    delete shutdown_flag_;
    delete[] function_table_;
    RealtimePipelineTest::TearDown();
  }

  void add_worker(uint32_t function_id, cudaGraphExec_t exec,
                  PreLaunchCopyCtx *plc = nullptr) {
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    worker_streams_.push_back(s);

    cudaq_host_dispatch_worker_t w{};
    w.graph_exec = exec;
    w.stream = s;
    w.function_id = function_id;
    w.pre_launch_fn = plc ? pre_launch_input_copy : nullptr;
    w.pre_launch_data = plc;
    workers_.push_back(w);

    size_t idx = ft_count_;
    function_table_[idx].handler.graph_exec = exec;
    function_table_[idx].function_id = function_id;
    function_table_[idx].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    ft_count_++;
  }

  void start_loop() {
    idle_mask_->store((1ULL << workers_.size()) - 1,
                      cuda::std::memory_order_release);

    std::memset(&config_, 0, sizeof(config_));

    config_.ringbuffer.rx_flags =
        reinterpret_cast<volatile uint64_t *>(rx_flags_dev_);
    config_.ringbuffer.tx_flags =
        reinterpret_cast<volatile uint64_t *>(tx_flags_dev_);
    config_.ringbuffer.rx_data = rx_data_dev_;
    config_.ringbuffer.tx_data = tx_data_dev_;
    config_.ringbuffer.rx_stride_sz = kSlotSize;
    config_.ringbuffer.tx_stride_sz = kSlotSize;
    config_.ringbuffer.rx_flags_host =
        reinterpret_cast<volatile uint64_t *>(rx_flags_host_);
    config_.ringbuffer.tx_flags_host =
        reinterpret_cast<volatile uint64_t *>(tx_flags_host_);
    config_.ringbuffer.rx_data_host = rx_data_host_;
    config_.ringbuffer.tx_data_host = tx_data_host_;

    config_.config.num_slots = static_cast<uint32_t>(kNumSlots);
    config_.config.slot_size = static_cast<uint32_t>(kSlotSize);
    config_.config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    config_.config.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;

    config_.function_table.entries = function_table_;
    config_.function_table.count = static_cast<uint32_t>(ft_count_);

    config_.workers = workers_.data();
    config_.num_workers = workers_.size();
    config_.h_mailbox_bank = mailbox_bank_host_;
    config_.shutdown_flag = shutdown_flag_;
    config_.stats_counter = &stats_counter_;
    config_.live_dispatched = live_dispatched_;
    config_.idle_mask = idle_mask_;
    config_.inflight_slot_tags = inflight_slot_tags_;

    loop_thread_ =
        std::thread([this]() { cudaq_host_dispatcher_loop(&config_); });
  }

  void stop_loop() {
    shutdown_flag_->store(1, cuda::std::memory_order_release);
    __sync_synchronize();
    if (loop_thread_.joinable())
      loop_thread_.join();
    loop_stopped_ = true;
  }

  void restore_worker(int id) {
    idle_mask_->fetch_or(1ULL << id, cuda::std::memory_order_release);
  }

  bool poll_tx_flag(size_t slot, int timeout_ms = 2000) {
    auto *flags = reinterpret_cast<atomic_uint64_sys *>(tx_flags_host_);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
      uint64_t val = flags[slot].load(cuda::std::memory_order_acquire);
      if (val != 0)
        return true;
      usleep(100);
    }
    return false;
  }

  void clear_tx_flag(size_t slot) {
    auto *flags = reinterpret_cast<atomic_uint64_sys *>(tx_flags_host_);
    flags[slot].store(0, cuda::std::memory_order_release);
  }

  atomic_uint64_sys *idle_mask_ = nullptr;
  atomic_uint64_sys *live_dispatched_ = nullptr;
  int *inflight_slot_tags_ = nullptr;
  atomic_int_sys *shutdown_flag_ = nullptr;
  uint64_t stats_counter_ = 0;
  bool loop_stopped_ = false;

  cudaq_function_entry_t *function_table_ = nullptr;
  size_t ft_count_ = 0;
  std::vector<cudaq_host_dispatch_worker_t> workers_;
  std::vector<cudaStream_t> worker_streams_;
  cudaq_host_dispatch_loop_ctx_t config_;
  std::thread loop_thread_;
};

TEST_F(HostDispatcherTest, ShutdownImmediate) {
  auto pd = create_predecoder(0);
  add_worker(kTestFunctionId, pd->get_executable_graph());

  shutdown_flag_->store(1, cuda::std::memory_order_release);
  start_loop();
  if (loop_thread_.joinable())
    loop_thread_.join();
  loop_stopped_ = true;

  EXPECT_EQ(stats_counter_, 0u);
}

TEST_F(HostDispatcherTest, ShutdownClean) {
  auto pd = create_predecoder(0);
  add_worker(kTestFunctionId, pd->get_executable_graph());
  start_loop();
  usleep(10000);
  stop_loop();
  EXPECT_EQ(stats_counter_, 0u);
}

TEST_F(HostDispatcherTest, StatsCounter) {
  auto pd = create_predecoder(0);
  PreLaunchCopyCtx plc;
  plc.d_trt_input = pd->get_trt_input_ptr();
  plc.input_size = pd->get_input_size();
  plc.h_ring_ptrs = pd->get_host_ring_ptrs();
  add_worker(kTestFunctionId, pd->get_executable_graph(), &plc);
  start_loop();

  constexpr int kN = 5;
  for (int i = 0; i < kN; ++i) {
    size_t slot = static_cast<size_t>(i % kNumSlots);
    if (i > 0)
      clear_tx_flag((i - 1) % kNumSlots);

    float payload[kSkipTrtFloats] = {};
    payload[0] = static_cast<float>(i);
    submit_rpc_to_slot(slot, kTestFunctionId, payload, kSkipTrtBytes);

    ASSERT_TRUE(poll_tx_flag(slot)) << "Timeout on request " << i;
    CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT_TRUE(wait_ready_flag(pd.get()));
    pre_decoder_job job{};
    if (pd->poll_next_job(job))
      pd->release_job(0);

    restore_worker(0);
  }

  stop_loop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kN));
}

TEST_F(HostDispatcherTest, InvalidMagicDropped) {
  auto pd = create_predecoder(0);
  add_worker(kTestFunctionId, pd->get_executable_graph());
  start_loop();

  uint8_t *slot_host = rx_data_host_;
  rt_sdk::RPCHeader bad_hdr;
  bad_hdr.magic = 0xDEADBEEF;
  bad_hdr.function_id = kTestFunctionId;
  bad_hdr.arg_len = 4;
  std::memcpy(slot_host, &bad_hdr, sizeof(bad_hdr));

  auto *flags = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
  flags[0].store(reinterpret_cast<uint64_t>(slot_host),
                 cuda::std::memory_order_release);

  usleep(50000);

  uint64_t rx_val = flags[0].load(cuda::std::memory_order_acquire);
  EXPECT_EQ(rx_val, 0u) << "Invalid magic should be consumed (rx_flag cleared)";

  stop_loop();
  EXPECT_EQ(stats_counter_, 0u)
      << "Invalid magic should not count as dispatched";
}

TEST_F(HostDispatcherTest, SlotWraparound) {
  auto pd = create_predecoder(0);
  PreLaunchCopyCtx plc;
  plc.d_trt_input = pd->get_trt_input_ptr();
  plc.input_size = pd->get_input_size();
  plc.h_ring_ptrs = pd->get_host_ring_ptrs();
  add_worker(kTestFunctionId, pd->get_executable_graph(), &plc);
  start_loop();

  constexpr int kTotal = static_cast<int>(kNumSlots) + 2;
  for (int i = 0; i < kTotal; ++i) {
    size_t slot = static_cast<size_t>(i % kNumSlots);

    auto *rx = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
    while (rx[slot].load(cuda::std::memory_order_acquire) != 0)
      usleep(100);
    clear_tx_flag(slot);

    float payload[kSkipTrtFloats] = {};
    payload[0] = static_cast<float>(i);
    submit_rpc_to_slot(slot, kTestFunctionId, payload, kSkipTrtBytes);

    ASSERT_TRUE(poll_tx_flag(slot))
        << "Timeout on request " << i << " (slot " << slot << ")";
    CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT_TRUE(wait_ready_flag(pd.get()));
    pre_decoder_job job{};
    if (pd->poll_next_job(job))
      pd->release_job(0);

    restore_worker(0);
  }

  stop_loop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kTotal));
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(HostDispatcherTest, SingleRequestRoundTrip) {
  auto pd = create_predecoder(0);
  PreLaunchCopyCtx plc;
  plc.d_trt_input = pd->get_trt_input_ptr();
  plc.input_size = pd->get_input_size();
  plc.h_ring_ptrs = pd->get_host_ring_ptrs();
  add_worker(kTestFunctionId, pd->get_executable_graph(), &plc);
  start_loop();

  float input[kSkipTrtFloats];
  for (size_t i = 0; i < kSkipTrtFloats; ++i)
    input[i] = static_cast<float>(i + 1);
  submit_rpc_to_slot(0, kTestFunctionId, input, kSkipTrtBytes);

  ASSERT_TRUE(poll_tx_flag(0)) << "Timeout waiting for dispatcher to process";
  CUDA_CHECK(cudaDeviceSynchronize());

  ASSERT_TRUE(wait_ready_flag(pd.get())) << "Predecoder ready flag not set";

  pre_decoder_job job{};
  ASSERT_TRUE(pd->poll_next_job(job));
  float output[kSkipTrtFloats];
  std::memcpy(output, job.inference_data, kSkipTrtBytes);
  pd->release_job(0);

  EXPECT_EQ(std::memcmp(input, output, kSkipTrtBytes), 0)
      << "Round-trip data should match (identity passthrough)";

  stop_loop();
  EXPECT_EQ(stats_counter_, 1u);
}

TEST_F(HostDispatcherTest, MultiPredecoderConcurrency) {
  constexpr int kNPd = 4;
  std::vector<std::unique_ptr<ai_predecoder_service>> pds;
  std::vector<PreLaunchCopyCtx> plcs(kNPd);
  std::vector<uint32_t> fids;

  for (int i = 0; i < kNPd; ++i) {
    pds.push_back(create_predecoder(i));
    std::string name = "predecoder_" + std::to_string(i);
    fids.push_back(rt_sdk::fnv1a_hash(name.c_str()));
    plcs[i].d_trt_input = pds[i]->get_trt_input_ptr();
    plcs[i].input_size = pds[i]->get_input_size();
    plcs[i].h_ring_ptrs = pds[i]->get_host_ring_ptrs();
    add_worker(fids[i], pds[i]->get_executable_graph(), &plcs[i]);
  }
  start_loop();

  float inputs[kNPd][kSkipTrtFloats];
  for (int i = 0; i < kNPd; ++i)
    for (size_t j = 0; j < kSkipTrtFloats; ++j)
      inputs[i][j] = static_cast<float>(i * 100 + j);

  for (int i = 0; i < kNPd; ++i)
    submit_rpc_to_slot(static_cast<size_t>(i), fids[i], inputs[i],
                       kSkipTrtBytes);

  for (int i = 0; i < kNPd; ++i)
    ASSERT_TRUE(poll_tx_flag(static_cast<size_t>(i)))
        << "Timeout on predecoder " << i;
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < kNPd; ++i) {
    ASSERT_TRUE(wait_ready_flag(pds[i].get()))
        << "Ready flag not set for predecoder " << i;
    pre_decoder_job job{};
    ASSERT_TRUE(pds[i]->poll_next_job(job));
    float output[kSkipTrtFloats];
    std::memcpy(output, job.inference_data, kSkipTrtBytes);
    pds[i]->release_job(0);

    EXPECT_EQ(std::memcmp(inputs[i], output, kSkipTrtBytes), 0)
        << "Predecoder " << i << ": output should match input";
  }

  stop_loop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kNPd));
}

TEST_F(HostDispatcherTest, SustainedThroughput_200Requests) {
  constexpr int kNPd = 2;
  constexpr int kTotalRequests = 200;

  std::vector<std::unique_ptr<ai_predecoder_service>> pds;
  std::vector<PreLaunchCopyCtx> plcs(kNPd);
  std::vector<uint32_t> fids;

  for (int i = 0; i < kNPd; ++i) {
    pds.push_back(create_predecoder(i));
    std::string name = "sustained_pd_" + std::to_string(i);
    fids.push_back(rt_sdk::fnv1a_hash(name.c_str()));
    plcs[i].d_trt_input = pds[i]->get_trt_input_ptr();
    plcs[i].input_size = pds[i]->get_input_size();
    plcs[i].h_ring_ptrs = pds[i]->get_host_ring_ptrs();
    add_worker(fids[i], pds[i]->get_executable_graph(), &plcs[i]);
  }
  start_loop();

  std::mt19937 rng(999);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  int completed = 0;

  for (int r = 0; r < kTotalRequests; ++r) {
    int pd_idx = r % kNPd;
    size_t slot = static_cast<size_t>(r % kNumSlots);

    auto *rx = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (rx[slot].load(cuda::std::memory_order_acquire) != 0) {
      if (std::chrono::steady_clock::now() > deadline)
        FAIL() << "Timeout waiting for slot " << slot << " to clear at request "
               << r;
      usleep(100);
    }
    clear_tx_flag(slot);

    float payload[kSkipTrtFloats];
    for (size_t i = 0; i < kSkipTrtFloats; ++i)
      payload[i] = dist(rng);

    submit_rpc_to_slot(slot, fids[pd_idx], payload, kSkipTrtBytes);

    ASSERT_TRUE(poll_tx_flag(slot))
        << "Timeout on request " << r << " (slot " << slot << ")";
    CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT_TRUE(wait_ready_flag(pds[pd_idx].get()))
        << "Ready flag not set for request " << r;
    pre_decoder_job job{};
    if (pds[pd_idx]->poll_next_job(job))
      pds[pd_idx]->release_job(0);

    restore_worker(pd_idx);
    completed++;
  }

  stop_loop();
  EXPECT_EQ(completed, kTotalRequests);
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kTotalRequests));
}

} // namespace
