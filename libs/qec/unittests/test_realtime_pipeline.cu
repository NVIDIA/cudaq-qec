/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/realtime/pipeline.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

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
                           const void *payload, size_t payload_len,
                           uint32_t request_id = 0) {
  rt_sdk::RPCHeader hdr{};
  hdr.magic = rt_sdk::RPC_MAGIC_REQUEST;
  hdr.function_id = function_id;
  hdr.arg_len = static_cast<uint32_t>(payload_len);
  hdr.request_id = request_id;
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

    // Device mailbox slots handed to create_passthrough(); the passthrough
    // graph takes its input through pre_launch_fn and does not read them.
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
                          const void *payload, size_t payload_len,
                          uint32_t request_id = 0) {
    uint8_t *slot_host = rx_data_host_ + slot * kSlotSize;
    write_rpc_slot(slot_host, function_id, payload, payload_len, request_id);
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
// Pipeline dispatch tests
//
// Drive realtime_pipeline through its public API with passthrough
// predecoders (no TensorRT model). The CPU stage reads the request header
// from the RX slot, frames an RPCResponse in ctx.response_buffer (the TX
// slot; on the internal in-place ring that is the RX slot itself) and
// records what it saw.
// ============================================================================

/// Per-worker context handed to the CPU stage through
/// gpu_worker_resources::user_context.
struct WorkerCtx {
  ai_predecoder_service *pd = nullptr;
  uint32_t function_id = 0;
  int worker_id = -1;
  /// Job claimed by the gated stage and held until the gate opens.
  pre_decoder_job held_job{};
  bool has_held_job = false;
};

/// What the CPU stage observed for one request.
struct RecordedJob {
  uint32_t request_id = 0;
  int worker_id = -1;
  int origin_slot = -1;
  uint32_t function_id = 0;
  std::vector<float> payload;
};

/// Request parked by the deferring stage until the test completes it.
struct DeferredJob {
  int origin_slot;
  uint32_t request_id;
  void *response_buffer;
};

static void write_rpc_response(void *dst, uint32_t request_id, int32_t status,
                               uint32_t result_len,
                               uint64_t ptp_timestamp = 0) {
  rt_sdk::RPCResponse rsp{};
  rsp.magic = rt_sdk::RPC_MAGIC_RESPONSE;
  rsp.status = status;
  rsp.result_len = result_len;
  rsp.request_id = request_id;
  rsp.ptp_timestamp = ptp_timestamp;
  std::memcpy(dst, &rsp, sizeof(rsp));
}

static std::vector<float> make_payload(float base) {
  std::vector<float> v(kSkipTrtFloats);
  for (size_t j = 0; j < kSkipTrtFloats; ++j)
    v[j] = base + static_cast<float>(j);
  return v;
}

template <typename Pred>
static bool wait_until(Pred pred, int timeout_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!pred()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return pred();
    usleep(100);
  }
  return true;
}

class PipelineDispatchTest : public RealtimePipelineTest {
protected:
  void SetUp() override {
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping pipeline dispatch tests";
    }
    RealtimePipelineTest::SetUp();
  }

  void TearDown() override {
    // stop() synchronizes the worker streams, so the pipeline goes first.
    pipeline_.reset();
    for (auto s : worker_streams_)
      if (s)
        cudaStreamDestroy(s);
    worker_streams_.clear();
    RealtimePipelineTest::TearDown();
  }

  // ---- pipeline construction ----------------------------------------------

  /// Pipeline over its own in-place ring (RX slot == TX slot).
  void make_internal_pipeline(int num_workers,
                              int num_slots = static_cast<int>(kNumSlots)) {
    pipeline_.reset();
    pipeline_stage_config cfg;
    cfg.num_workers = num_workers;
    cfg.num_slots = num_slots;
    cfg.slot_size = kSlotSize;
    pipeline_ = std::make_unique<realtime_pipeline>(cfg);
    rx_base_ = pipeline_->ringbuffer_bases().rx_data_host;
  }

  /// Pipeline over the fixture's buffers as an external ring with separate
  /// RX and TX data: the test is the producer and plays the transport.
  void make_external_pipeline(int num_workers) {
    pipeline_.reset();
    std::memset(&ext_rb_, 0, sizeof(ext_rb_));
    ext_rb_.rx_flags = reinterpret_cast<volatile uint64_t *>(rx_flags_dev_);
    ext_rb_.tx_flags = reinterpret_cast<volatile uint64_t *>(tx_flags_dev_);
    ext_rb_.rx_data = rx_data_dev_;
    ext_rb_.tx_data = tx_data_dev_;
    ext_rb_.rx_stride_sz = kSlotSize;
    ext_rb_.tx_stride_sz = kSlotSize;
    ext_rb_.rx_flags_host =
        reinterpret_cast<volatile uint64_t *>(rx_flags_host_);
    ext_rb_.tx_flags_host =
        reinterpret_cast<volatile uint64_t *>(tx_flags_host_);
    ext_rb_.rx_data_host = rx_data_host_;
    ext_rb_.tx_data_host = tx_data_host_;

    pipeline_stage_config cfg;
    cfg.num_workers = num_workers;
    cfg.num_slots = static_cast<int>(kNumSlots);
    cfg.slot_size = kSlotSize;
    cfg.external_ringbuffer = &ext_rb_;
    pipeline_ = std::make_unique<realtime_pipeline>(cfg);
    rx_base_ = rx_data_host_;
  }

  /// One passthrough predecoder per worker, each on its own stream, serving
  /// fids[w]. The pre-launch callback copies the slot payload into the
  /// predecoder input (D2D from the mapped slot).
  void install_passthrough_workers(
      realtime_pipeline &p,
      const std::vector<std::unique_ptr<ai_predecoder_service>> &pds,
      const std::vector<uint32_t> &fids) {
    const size_t n = pds.size();
    ASSERT_EQ(fids.size(), n);
    plcs_.assign(n, PreLaunchCopyCtx{});
    wctx_.assign(n, WorkerCtx{});
    std::vector<cudaStream_t> streams(n, nullptr);
    for (size_t w = 0; w < n; ++w) {
      plcs_[w].d_trt_input = pds[w]->get_trt_input_ptr();
      plcs_[w].input_size = pds[w]->get_input_size();
      plcs_[w].h_ring_ptrs = pds[w]->get_host_ring_ptrs();
      wctx_[w].pd = pds[w].get();
      wctx_[w].function_id = fids[w];
      wctx_[w].worker_id = static_cast<int>(w);
      ASSERT_EQ(cudaStreamCreate(&streams[w]), cudaSuccess);
      worker_streams_.push_back(streams[w]);
    }
    p.set_gpu_stage([this, streams](int w) {
      gpu_worker_resources r;
      r.graph_exec = wctx_[w].pd->get_executable_graph();
      r.stream = streams[w];
      r.pre_launch_fn = pre_launch_input_copy;
      r.pre_launch_data = &plcs_[w];
      r.post_launch_fn = nullptr;
      r.function_id = wctx_[w].function_id;
      r.user_context = &wctx_[w];
      return r;
    });
  }

  /// Single worker whose graph handle is null: every launch fails.
  void install_null_graph_worker(realtime_pipeline &p) {
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    worker_streams_.push_back(s);
    p.set_gpu_stage([s](int) {
      gpu_worker_resources r;
      r.graph_exec = nullptr;
      r.stream = s;
      r.function_id = kTestFunctionId;
      return r;
    });
  }

  /// Builds pipeline_ (not started) with one passthrough predecoder per
  /// entry of fids, the given CPU stage (none = GPU-only mode) and the
  /// completion recorder.
  void make_passthrough_pipeline(const std::vector<uint32_t> &fids,
                                 bool external, cpu_stage_callback stage) {
    pipeline_.reset();
    pds_.clear();
    for (size_t i = 0; i < fids.size(); ++i)
      pds_.push_back(create_predecoder(static_cast<int>(i)));
    const int n = static_cast<int>(fids.size());
    if (external)
      make_external_pipeline(n);
    else
      make_internal_pipeline(n);
    ASSERT_NO_FATAL_FAILURE(
        install_passthrough_workers(*pipeline_, pds_, fids));
    if (stage)
      pipeline_->set_cpu_stage(std::move(stage));
    install_completion_recorder(*pipeline_);
  }

  // ---- CPU stages -----------------------------------------------------------

  /// Shared tail of the passthrough and gated stages: record the request,
  /// frame RPCResponse + payload in the TX slot, release the predecoder.
  size_t finish_job(const cpu_stage_context &ctx, WorkerCtx *wc,
                    const pre_decoder_job &job) {
    // The request's identity comes from the context, captured at dispatch;
    // the RX slot is not re-read (on the internal ring ctx.response_buffer
    // is that same slot, on an external ring the page may be overwritten a
    // lap later).
    RecordedJob rec;
    rec.request_id = static_cast<uint32_t>(ctx.request_id);
    rec.worker_id = ctx.worker_id;
    rec.origin_slot = ctx.origin_slot;
    rec.function_id = wc->function_id;
    rec.payload.resize(kSkipTrtFloats);
    std::memcpy(rec.payload.data(), job.inference_data, kSkipTrtBytes);
    wc->pd->release_job(job.slot_idx);

    const size_t written = sizeof(rt_sdk::RPCResponse) + kSkipTrtBytes;
    EXPECT_GE(ctx.max_response_size, written);
    auto *out = static_cast<uint8_t *>(ctx.response_buffer);
    write_rpc_response(out, rec.request_id, 0,
                       static_cast<uint32_t>(kSkipTrtBytes), ctx.ptp_timestamp);
    std::memcpy(out + sizeof(rt_sdk::RPCResponse), rec.payload.data(),
                kSkipTrtBytes);

    std::lock_guard<std::mutex> lock(mu_);
    records_[rec.request_id] = std::move(rec);
    return written;
  }

  /// Completes a request as soon as its predecoder output is ready.
  size_t passthrough_stage(const cpu_stage_context &ctx) {
    auto *wc = static_cast<WorkerCtx *>(ctx.user_context);
    pre_decoder_job job{};
    if (!wc->pd->poll_next_job(job))
      return 0;
    return finish_job(ctx, wc, job);
  }

  /// Claims the predecoder output but holds the worker (returns 0) until
  /// gate_ opens.
  size_t gated_stage(const cpu_stage_context &ctx) {
    auto *wc = static_cast<WorkerCtx *>(ctx.user_context);
    if (!wc->has_held_job) {
      if (!wc->pd->poll_next_job(wc->held_job))
        return 0;
      wc->has_held_job = true;
    }
    if (gate_.load(std::memory_order_acquire) == 0)
      return 0;
    wc->has_held_job = false;
    return finish_job(ctx, wc, wc->held_job);
  }

  /// Parks the request for the test and releases the worker.
  size_t deferring_stage(const cpu_stage_context &ctx) {
    auto *wc = static_cast<WorkerCtx *>(ctx.user_context);
    pre_decoder_job job{};
    if (!wc->pd->poll_next_job(job))
      return 0;
    wc->pd->release_job(job.slot_idx);
    std::lock_guard<std::mutex> lock(mu_);
    deferred_.push_back({ctx.origin_slot, static_cast<uint32_t>(ctx.request_id),
                         ctx.response_buffer});
    return DEFERRED_COMPLETION;
  }

  cpu_stage_callback passthrough_cb() {
    return [this](const cpu_stage_context &c) { return passthrough_stage(c); };
  }
  cpu_stage_callback gated_cb() {
    return [this](const cpu_stage_context &c) { return gated_stage(c); };
  }
  cpu_stage_callback deferring_cb() {
    return [this](const cpu_stage_context &c) { return deferring_stage(c); };
  }

  // ---- completion recorder and waits --------------------------------------

  void install_completion_recorder(realtime_pipeline &p) {
    p.set_completion_handler([this](const completion &c) {
      std::lock_guard<std::mutex> lock(mu_);
      completions_.push_back(c);
      completion_count_.fetch_add(1, std::memory_order_release);
    });
  }

  bool wait_completions(int n, int timeout_ms = 5000) {
    return wait_until(
        [&] { return completion_count_.load(std::memory_order_acquire) >= n; },
        timeout_ms);
  }

  std::vector<completion> completions() {
    std::lock_guard<std::mutex> lock(mu_);
    return completions_;
  }

  bool find_record(uint32_t request_id, RecordedJob &out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end())
      return false;
    out = it->second;
    return true;
  }

  std::vector<DeferredJob> deferred_jobs() {
    std::lock_guard<std::mutex> lock(mu_);
    return deferred_;
  }

  bool wait_dispatched(uint64_t n, int timeout_ms = 5000) {
    return wait_until([&] { return pipeline_->stats().dispatched >= n; },
                      timeout_ms);
  }

  /// External-ring producer side: the dispatcher zeroes rx_flags[slot] once
  /// it has taken (or dropped) the request.
  bool wait_rx_cleared(size_t slot, int timeout_ms = 2000) {
    auto *flags = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
    return wait_until(
        [&] { return flags[slot].load(cuda::std::memory_order_acquire) == 0; },
        timeout_ms);
  }

  void reset_recorder() {
    std::lock_guard<std::mutex> lock(mu_);
    completions_.clear();
    records_.clear();
    deferred_.clear();
    completion_count_.store(0, std::memory_order_release);
  }

  void expect_stats(uint64_t submitted, uint64_t completed,
                    uint64_t dispatched) {
    auto st = pipeline_->stats();
    EXPECT_EQ(st.submitted, submitted);
    EXPECT_EQ(st.completed, completed);
    EXPECT_EQ(st.dispatched, dispatched);
  }

  /// One launch-failure round trip: a null graph fails every launch, the
  /// failure is reported in the completion and in the TX slot, and the worker
  /// is handed the next request afterwards.
  void run_launch_failure(bool gpu_only) {
    SCOPED_TRACE(gpu_only ? "gpu-only mode" : "cpu-stage mode");
    reset_recorder();
    make_internal_pipeline(1);
    ASSERT_NO_FATAL_FAILURE(install_null_graph_worker(*pipeline_));
    if (!gpu_only)
      pipeline_->set_cpu_stage(
          [](const cpu_stage_context &) -> size_t { return 0; });
    install_completion_recorder(*pipeline_);
    pipeline_->start();
    auto inj = pipeline_->create_injector();
    auto payload = make_payload(0.0f);

    inj.submit(kTestFunctionId, payload.data(), kSkipTrtBytes, 31);
    ASSERT_TRUE(wait_completions(1)) << "No completion for the failed launch";
    auto cs = completions();
    ASSERT_EQ(cs.size(), 1u);
    EXPECT_FALSE(cs[0].success);
    EXPECT_EQ(cs[0].cuda_error, static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(cs[0].request_id, 31u);
    EXPECT_EQ(cs[0].slot, 0);

    rt_sdk::RPCResponse rsp;
    std::memcpy(&rsp, rx_base_ + static_cast<size_t>(cs[0].slot) * kSlotSize,
                sizeof(rsp));
    EXPECT_EQ(rsp.magic, rt_sdk::RPC_MAGIC_RESPONSE);
    EXPECT_NE(rsp.status, 0);
    EXPECT_EQ(rsp.result_len, 0u);
    EXPECT_EQ(rsp.request_id, 31u);

    inj.submit(kTestFunctionId, payload.data(), kSkipTrtBytes, 32);
    ASSERT_TRUE(wait_completions(2))
        << "Worker was not released after the failed launch";
    cs = completions();
    ASSERT_EQ(cs.size(), 2u);
    EXPECT_FALSE(cs[1].success);
    EXPECT_EQ(cs[1].cuda_error, static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(cs[1].request_id, 32u);

    pipeline_->stop();
    expect_stats(2, 2, 2);
  }

  std::unique_ptr<realtime_pipeline> pipeline_;
  std::vector<std::unique_ptr<ai_predecoder_service>> pds_;
  std::vector<PreLaunchCopyCtx> plcs_;
  std::vector<WorkerCtx> wctx_;
  std::vector<cudaStream_t> worker_streams_;
  cudaq_ringbuffer_t ext_rb_{};
  const uint8_t *rx_base_ = nullptr;
  std::atomic<int> gate_{0};

  std::mutex mu_;
  std::map<uint32_t, RecordedJob> records_;
  std::vector<completion> completions_;
  std::vector<DeferredJob> deferred_;
  std::atomic<int> completion_count_{0};
};

TEST_F(PipelineDispatchTest, ShutdownImmediate) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, passthrough_cb()));
  pipeline_->start();
  pipeline_->stop();
  expect_stats(0, 0, 0);
  EXPECT_EQ(completion_count_.load(), 0);
}

TEST_F(PipelineDispatchTest, ShutdownClean) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, passthrough_cb()));
  pipeline_->start();
  usleep(10000);
  pipeline_->stop();
  expect_stats(0, 0, 0);
  EXPECT_EQ(completion_count_.load(), 0);
}

// A pipeline is single use: its stop flags are never reset, so a restart
// would spawn threads that exit at once while the injector kept accepting
// requests. start() after stop() throws, and the injector refuses.
TEST_F(PipelineDispatchTest, RestartThrows) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, passthrough_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();
  pipeline_->stop();
  EXPECT_THROW(pipeline_->start(), std::logic_error);
  auto in = make_payload(1.0f);
  EXPECT_FALSE(inj.try_submit(kTestFunctionId, in.data(), kSkipTrtBytes, 1))
      << "The injector must refuse once the pipeline has stopped";
  expect_stats(0, 0, 0);
}

TEST_F(PipelineDispatchTest, StatsCounter) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, passthrough_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  constexpr int kN = 5;
  for (int i = 0; i < kN; ++i) {
    auto payload = make_payload(static_cast<float>(i));
    inj.submit(kTestFunctionId, payload.data(), kSkipTrtBytes,
               static_cast<uint64_t>(i));
    ASSERT_TRUE(wait_completions(i + 1)) << "Timeout on request " << i;
  }

  pipeline_->stop();
  expect_stats(kN, kN, kN);
  for (const auto &c : completions()) {
    EXPECT_TRUE(c.success);
    EXPECT_EQ(c.cuda_error, 0);
  }
}

TEST_F(PipelineDispatchTest, InvalidMagicDropped) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, true, passthrough_cb()));
  pipeline_->start();

  uint8_t *slot_host = rx_data_host_;
  rt_sdk::RPCHeader bad_hdr{};
  bad_hdr.magic = 0xDEADBEEF;
  bad_hdr.function_id = kTestFunctionId;
  bad_hdr.arg_len = 4;
  std::memcpy(slot_host, &bad_hdr, sizeof(bad_hdr));

  auto *rx = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
  rx[0].store(reinterpret_cast<uint64_t>(slot_host),
              cuda::std::memory_order_release);

  EXPECT_TRUE(wait_rx_cleared(0))
      << "Invalid magic should be consumed (rx_flag cleared)";
  usleep(20000);

  auto *tx = reinterpret_cast<atomic_uint64_sys *>(tx_flags_host_);
  EXPECT_EQ(tx[0].load(cuda::std::memory_order_acquire), 0u)
      << "A dropped request must not publish a response";
  EXPECT_EQ(completion_count_.load(), 0);

  // The dispatcher keeps running: a valid request on the next slot completes
  // through the external-ring path. The response is framed in the TX slot,
  // the published flag is that slot's host address, and the pipeline leaves
  // the flag for the transport to consume.
  auto input = make_payload(1.0f);
  submit_rpc_to_slot(1, kTestFunctionId, input.data(), kSkipTrtBytes, 5);
  ASSERT_TRUE(wait_completions(1))
      << "Valid request after the drop did not complete";
  EXPECT_TRUE(wait_rx_cleared(1));
  auto cs = completions();
  ASSERT_EQ(cs.size(), 1u);
  EXPECT_EQ(cs[0].request_id, 5u);
  EXPECT_EQ(cs[0].slot, 1);
  EXPECT_TRUE(cs[0].success);

  uint8_t *tx_slot = tx_data_host_ + kSlotSize;
  EXPECT_EQ(tx[1].load(cuda::std::memory_order_acquire),
            reinterpret_cast<uint64_t>(tx_slot));
  rt_sdk::RPCResponse rsp;
  std::memcpy(&rsp, tx_slot, sizeof(rsp));
  EXPECT_EQ(rsp.magic, rt_sdk::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(rsp.status, 0);
  EXPECT_EQ(rsp.result_len, static_cast<uint32_t>(kSkipTrtBytes));
  EXPECT_EQ(rsp.request_id, 5u);
  RecordedJob rec;
  ASSERT_TRUE(find_record(5, rec));
  EXPECT_EQ(std::memcmp(input.data(), rec.payload.data(), kSkipTrtBytes), 0);
  EXPECT_EQ(tx[0].load(cuda::std::memory_order_acquire), 0u);

  pipeline_->stop();
  EXPECT_EQ(tx[1].load(cuda::std::memory_order_acquire),
            reinterpret_cast<uint64_t>(tx_slot))
      << "The pipeline must not clear tx_flags on an external ring";
  EXPECT_EQ(pipeline_->stats().dispatched, 1u)
      << "Only the valid request counts as dispatched";
  EXPECT_EQ(pipeline_->stats().completed, 1u);
}

TEST_F(PipelineDispatchTest, SlotWraparound) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, passthrough_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  constexpr int kTotal = 3 * static_cast<int>(kNumSlots) + 2;
  for (int i = 0; i < kTotal; ++i) {
    auto payload = make_payload(static_cast<float>(i));
    inj.submit(kTestFunctionId, payload.data(), kSkipTrtBytes,
               static_cast<uint64_t>(i));
    ASSERT_TRUE(wait_completions(i + 1)) << "Timeout on request " << i;
  }

  auto cs = completions();
  ASSERT_EQ(cs.size(), static_cast<size_t>(kTotal));
  for (int i = 0; i < kTotal; ++i) {
    EXPECT_EQ(cs[i].request_id, static_cast<uint64_t>(i));
    EXPECT_EQ(cs[i].slot, static_cast<int>(i % kNumSlots))
        << "Request " << i << " should cycle through the ring";
    EXPECT_TRUE(cs[i].success);
  }

  pipeline_->stop();
  expect_stats(kTotal, kTotal, kTotal);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(PipelineDispatchTest, SingleRequestRoundTrip) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, passthrough_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  auto input = make_payload(1.0f);
  inj.submit(kTestFunctionId, input.data(), kSkipTrtBytes, 7);
  ASSERT_TRUE(wait_completions(1)) << "Timeout waiting for completion";

  auto cs = completions();
  ASSERT_EQ(cs.size(), 1u);
  EXPECT_EQ(cs[0].request_id, 7u);
  EXPECT_EQ(cs[0].slot, 0);
  EXPECT_TRUE(cs[0].success);
  EXPECT_EQ(cs[0].cuda_error, 0);

  RecordedJob rec;
  ASSERT_TRUE(find_record(7, rec));
  EXPECT_EQ(rec.worker_id, 0);
  EXPECT_EQ(rec.origin_slot, 0);
  EXPECT_EQ(std::memcmp(input.data(), rec.payload.data(), kSkipTrtBytes), 0)
      << "Round-trip data should match (identity passthrough)";

  // The response was framed in the TX slot, which is the RX slot itself on
  // the internal ring.
  rt_sdk::RPCResponse rsp;
  std::memcpy(&rsp, rx_base_, sizeof(rsp));
  EXPECT_EQ(rsp.magic, rt_sdk::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(rsp.status, 0);
  EXPECT_EQ(rsp.result_len, static_cast<uint32_t>(kSkipTrtBytes));
  EXPECT_EQ(rsp.request_id, 7u);

  pipeline_->stop();
  expect_stats(1, 1, 1);
}

TEST_F(PipelineDispatchTest, MultiPredecoderConcurrency) {
  constexpr int kNPd = 4;
  constexpr int kNReq = 2 * kNPd;
  std::vector<uint32_t> fids;
  for (int i = 0; i < kNPd; ++i) {
    std::string name = "predecoder_" + std::to_string(i);
    fids.push_back(rt_sdk::fnv1a_hash(name.c_str()));
  }
  ASSERT_NO_FATAL_FAILURE(make_passthrough_pipeline(fids, false, gated_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  std::vector<std::vector<float>> inputs(kNReq);
  for (int r = 0; r < kNReq; ++r) {
    inputs[r] = make_payload(static_cast<float>(r * 100));
    inj.submit(fids[r % kNPd], inputs[r].data(), kSkipTrtBytes,
               static_cast<uint64_t>(r));
  }

  // Every worker holds one request behind the closed gate; the second
  // request for each function id waits in the ring until its worker is idle.
  ASSERT_TRUE(wait_dispatched(kNPd)) << "Workers were not all dispatched";
  usleep(20000);
  EXPECT_EQ(pipeline_->stats().dispatched, static_cast<uint64_t>(kNPd));
  EXPECT_EQ(completion_count_.load(), 0);

  gate_.store(1, std::memory_order_release);
  ASSERT_TRUE(wait_completions(kNReq)) << "Timeout waiting for completions";

  for (int r = 0; r < kNReq; ++r) {
    RecordedJob rec;
    ASSERT_TRUE(find_record(static_cast<uint32_t>(r), rec))
        << "Request " << r << " not recorded";
    EXPECT_EQ(rec.function_id, fids[r % kNPd])
        << "Request " << r << " ran on a worker with the wrong function id";
    EXPECT_EQ(rec.worker_id, r % kNPd);
    EXPECT_EQ(std::memcmp(inputs[r].data(), rec.payload.data(), kSkipTrtBytes),
              0)
        << "Request " << r << ": output should match input";
  }
  for (const auto &c : completions())
    EXPECT_TRUE(c.success);

  pipeline_->stop();
  expect_stats(kNReq, kNReq, kNReq);
}

TEST_F(PipelineDispatchTest, SustainedThroughput_200Requests) {
  constexpr int kNPd = 2;
  constexpr int kTotalRequests = 200;
  std::vector<uint32_t> fids(kNPd, kTestFunctionId);
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline(fids, false, passthrough_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  std::mt19937 rng(999);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<std::vector<float>> inputs(kTotalRequests);
  for (int r = 0; r < kTotalRequests; ++r) {
    inputs[r].resize(kSkipTrtFloats);
    for (auto &v : inputs[r])
      v = dist(rng);
    // Blocks when all kNumSlots slots are occupied (injector backpressure).
    inj.submit(kTestFunctionId, inputs[r].data(), kSkipTrtBytes,
               static_cast<uint64_t>(r));
  }
  ASSERT_TRUE(wait_completions(kTotalRequests, 30000))
      << "Timeout waiting for " << kTotalRequests << " completions";

  for (int r = 0; r < kTotalRequests; ++r) {
    RecordedJob rec;
    ASSERT_TRUE(find_record(static_cast<uint32_t>(r), rec))
        << "Request " << r << " not recorded";
    EXPECT_EQ(std::memcmp(inputs[r].data(), rec.payload.data(), kSkipTrtBytes),
              0)
        << "Request " << r << ": output should match input";
  }
  for (const auto &c : completions())
    EXPECT_TRUE(c.success);

  pipeline_->stop();
  expect_stats(kTotalRequests, kTotalRequests, kTotalRequests);
}

// External ring: a slot belongs to the pipeline from dispatch until its
// response is published. The rx_flag keeps holding the frame, the dispatcher
// does not take the slot again when its cursor comes back around, publish()
// hands the RX slot back, and a new frame in a slot whose previous response
// the transport has not taken yet waits for that. The test plays the
// transport: frames land in ring order, one per slot.
TEST_F(PipelineDispatchTest, ExternalRingHoldsSlotUntilResponsePublished) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, true, deferring_cb()));
  pipeline_->start();
  auto *rx = reinterpret_cast<atomic_uint64_sys *>(rx_flags_host_);
  auto *tx = reinterpret_cast<atomic_uint64_sys *>(tx_flags_host_);
  const auto n = static_cast<uint32_t>(kNumSlots);

  auto in = make_payload(1.0f);
  submit_rpc_to_slot(0, kTestFunctionId, in.data(), kSkipTrtBytes, 11);
  ASSERT_TRUE(wait_until([&] { return deferred_jobs().size() == 1; }, 5000));
  EXPECT_EQ(deferred_jobs()[0].request_id, 11u);

  // Held: the frame stays published to the pipeline, nothing is on TX.
  EXPECT_EQ(rx[0].load(cuda::std::memory_order_acquire),
            reinterpret_cast<uint64_t>(rx_data_host_))
      << "rx_flag must hold the frame while the request is in flight";
  EXPECT_EQ(tx[0].load(cuda::std::memory_order_acquire), 0u);

  // One lap of frames: the cursor comes back to the held slot 0 and must not
  // dispatch its stale frame again.
  for (uint32_t s = 1; s < n; ++s)
    submit_rpc_to_slot(s, kTestFunctionId, in.data(), kSkipTrtBytes, 11 + s);
  ASSERT_TRUE(wait_until([&] { return deferred_jobs().size() == n; }, 5000));
  usleep(20000);
  EXPECT_EQ(pipeline_->stats().dispatched, n)
      << "A held slot must not be dispatched again";

  // Publishing the response hands the RX slot back.
  auto first = deferred_jobs()[0];
  write_rpc_response(first.response_buffer, first.request_id, 0, 0);
  pipeline_->complete_deferred(first.origin_slot);
  ASSERT_TRUE(wait_rx_cleared(0)) << "publish() must release the RX slot";
  EXPECT_EQ(tx[0].load(cuda::std::memory_order_acquire),
            reinterpret_cast<uint64_t>(tx_data_host_));
  ASSERT_TRUE(wait_completions(1));

  // The transport (this test) has not taken the response yet: the next frame
  // in slot 0 waits until it does.
  submit_rpc_to_slot(0, kTestFunctionId, in.data(), kSkipTrtBytes, 11 + n);
  usleep(20000);
  EXPECT_EQ(pipeline_->stats().dispatched, n)
      << "A slot whose previous response is still on TX must wait";
  tx[0].store(0, cuda::std::memory_order_release);
  ASSERT_TRUE(
      wait_until([&] { return deferred_jobs().size() == n + 1; }, 5000));
  EXPECT_EQ(deferred_jobs()[n].request_id, 11u + n);
  EXPECT_EQ(deferred_jobs()[n].origin_slot, 0);

  for (size_t i = 1; i <= n; ++i) {
    auto j = deferred_jobs()[i];
    write_rpc_response(j.response_buffer, j.request_id, 0, 0);
    pipeline_->complete_deferred(j.origin_slot);
  }
  ASSERT_TRUE(wait_completions(static_cast<int>(n) + 1));
  pipeline_->stop();
}

TEST_F(PipelineDispatchTest, DeferredCompletion) {
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, deferring_cb()));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  auto in1 = make_payload(1.0f);
  auto in2 = make_payload(2.0f);
  inj.submit(kTestFunctionId, in1.data(), kSkipTrtBytes, 11);
  inj.submit(kTestFunctionId, in2.data(), kSkipTrtBytes, 12);

  // DEFERRED_COMPLETION releases the single worker, so the second request is
  // dispatched while neither has completed.
  ASSERT_TRUE(wait_until([&] { return deferred_jobs().size() == 2; }, 5000))
      << "Both requests should reach the deferring stage";
  EXPECT_EQ(pipeline_->stats().dispatched, 2u);
  EXPECT_EQ(pipeline_->stats().completed, 0u);
  EXPECT_EQ(completion_count_.load(), 0);

  auto jobs = deferred_jobs();
  for (const auto &job : jobs) {
    write_rpc_response(job.response_buffer, job.request_id, 0, 0);
    pipeline_->complete_deferred(job.origin_slot);
  }
  ASSERT_TRUE(wait_completions(2)) << "Deferred slots did not complete";

  auto cs = completions();
  ASSERT_EQ(cs.size(), 2u);
  for (const auto &c : cs) {
    EXPECT_TRUE(c.success);
    bool matched = false;
    for (const auto &job : jobs)
      if (job.request_id == c.request_id) {
        matched = true;
        EXPECT_EQ(c.slot, job.origin_slot);
      }
    EXPECT_TRUE(matched) << "Unexpected request id " << c.request_id;
  }
  EXPECT_NE(cs[0].request_id, cs[1].request_id);
  EXPECT_TRUE((cs[0].request_id == 11 && cs[1].request_id == 12) ||
              (cs[0].request_id == 12 && cs[1].request_id == 11));

  pipeline_->stop();
  expect_stats(2, 2, 2);
}

TEST_F(PipelineDispatchTest, GpuOnlyRoundTrip) {
  // No CPU stage: completion is published from the host callback the
  // dispatcher enqueues after each graph.
  ASSERT_NO_FATAL_FAILURE(
      make_passthrough_pipeline({kTestFunctionId}, false, nullptr));
  pipeline_->start();
  auto inj = pipeline_->create_injector();

  auto input = make_payload(1.0f);
  inj.submit(kTestFunctionId, input.data(), kSkipTrtBytes, 21);
  ASSERT_TRUE(wait_completions(1)) << "Timeout waiting for completion";

  auto cs = completions();
  ASSERT_EQ(cs.size(), 1u);
  EXPECT_EQ(cs[0].request_id, 21u);
  EXPECT_EQ(cs[0].slot, 0);
  EXPECT_TRUE(cs[0].success);
  EXPECT_EQ(cs[0].cuda_error, 0);

  // The graph ran: its last node raised the predecoder ready flag.
  ASSERT_TRUE(wait_ready_flag(pds_[0].get()))
      << "Predecoder ready flag not set";
  pre_decoder_job job{};
  ASSERT_TRUE(pds_[0]->poll_next_job(job));
  EXPECT_EQ(std::memcmp(input.data(), job.inference_data, kSkipTrtBytes), 0)
      << "Round-trip data should match (identity passthrough)";
  pds_[0]->release_job(job.slot_idx);

  constexpr int kBurst = 20;
  for (int i = 0; i < kBurst; ++i)
    inj.submit(kTestFunctionId, input.data(), kSkipTrtBytes,
               static_cast<uint64_t>(100 + i));
  ASSERT_TRUE(wait_completions(1 + kBurst))
      << "Timeout waiting for the back-to-back completions";
  for (const auto &c : completions())
    EXPECT_TRUE(c.success);

  pipeline_->stop();
  expect_stats(1 + kBurst, 1 + kBurst, 1 + kBurst);
}

TEST_F(PipelineDispatchTest, LaunchFailureReportsError) {
  run_launch_failure(false);
  if (HasFatalFailure())
    return;
  run_launch_failure(true);
}

} // namespace
