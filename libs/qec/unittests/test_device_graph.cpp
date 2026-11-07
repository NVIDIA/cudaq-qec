/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeviceGraphRingConsumer.h"

#include "cudaq/qec/realtime/graph_resources.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <unordered_set>

namespace {

std::mutex g_mu;
std::unordered_set<void *> g_live;
int g_host_alloc_n = 0;
int g_host_ptr_n = 0;

struct WrapCtl {
  int host_alloc_fail_on = 0;
  int host_ptr_fail_on = 0;
  cudaError_t malloc_st = cudaSuccess;
  cudaError_t memset_st = cudaSuccess;
  bool dlsym_omit_populate = false;
  bool invalid_device_call = false;
} g_ctl;

void *track(void *p) {
  if (p) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_live.insert(p);
  }
  return p;
}
void untrack(void *p) {
  if (!p)
    return;
  std::lock_guard<std::mutex> lk(g_mu);
  g_live.erase(p);
  std::free(p);
}
void *zalloc(size_t n) { return track(std::calloc(1, n ? n : 1)); }

void reset_ctl() {
  g_ctl = WrapCtl{};
  g_host_alloc_n = 0;
  g_host_ptr_n = 0;
  std::lock_guard<std::mutex> lk(g_mu);
  g_live.clear();
}

size_t live_count() {
  std::lock_guard<std::mutex> lk(g_mu);
  return g_live.size();
}

struct FakeBridge {
  uint64_t rx_flags = 1;
  uint64_t tx_flags = 1;
  uint8_t rx_data[256]{};
  uint8_t tx_data[256]{};
};

cudaq_ringbuffer_t valid_ring(FakeBridge &b) {
  cudaq_ringbuffer_t ring{};
  ring.rx_flags = &b.rx_flags;
  ring.tx_flags = &b.tx_flags;
  ring.rx_data = b.rx_data;
  ring.tx_data = b.tx_data;
  ring.rx_stride_sz = 64;
  ring.tx_stride_sz = 64;
  return ring;
}

cudaq::qec::realtime::graph_resources make_graph() {
  cudaq::qec::realtime::graph_resources gr{};
  gr.graph_exec = reinterpret_cast<cudaGraphExec_t>(&gr);
  return gr;
}

} // namespace

extern "C" {

void *__real_dlsym(void *handle, const char *symbol);

cudaError_t __wrap_cudaSetDevice(int) { return cudaSuccess; }

#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif
// CUDA 12 aliases this API to cudaGetDeviceProperties_v2, so both linker
// wraps must exist or the constructor hits real CUDA.
cudaError_t __wrap_cudaGetDeviceProperties(cudaDeviceProp *p, int) {
  if (!p)
    return cudaErrorInvalidValue;
  std::memset(p, 0, sizeof(*p));
  p->major = 9;
  p->minor = 0;
  std::strncpy(p->name, "qec-test-gpu", sizeof(p->name) - 1);
  return cudaSuccess;
}
cudaError_t __wrap_cudaGetDeviceProperties_v2(cudaDeviceProp *p, int device) {
  return __wrap_cudaGetDeviceProperties(p, device);
}
cudaError_t __wrap_cudaHostAlloc(void **p, size_t bytes, unsigned int) {
  ++g_host_alloc_n;
  if (g_ctl.host_alloc_fail_on && g_host_alloc_n == g_ctl.host_alloc_fail_on)
    return cudaErrorInvalidValue;
  *p = zalloc(bytes);
  return cudaSuccess;
}
cudaError_t __wrap_cudaHostGetDevicePointer(void **d, void *h, unsigned int) {
  ++g_host_ptr_n;
  if (g_ctl.host_ptr_fail_on && g_host_ptr_n == g_ctl.host_ptr_fail_on)
    return cudaErrorInvalidValue;
  *d = h;
  return cudaSuccess;
}
cudaError_t __wrap_cudaFreeHost(void *p) {
  untrack(p);
  return cudaSuccess;
}
cudaError_t __wrap_cudaMalloc(void **p, size_t bytes) {
  if (g_ctl.malloc_st != cudaSuccess)
    return g_ctl.malloc_st;
  *p = zalloc(bytes);
  return cudaSuccess;
}
cudaError_t __wrap_cudaMemset(void *p, int v, size_t n) {
  if (g_ctl.memset_st != cudaSuccess)
    return g_ctl.memset_st;
  if (p)
    std::memset(p, v, n);
  return cudaSuccess;
}
cudaError_t __wrap_cudaFree(void *p) {
  untrack(p);
  return cudaSuccess;
}
const char *__wrap_cudaGetErrorString(cudaError_t) { return "qec-test-cuda"; }

static void populate_ok(void *entry) {
  auto *e = static_cast<cudaq_function_entry_t *>(entry);
  *e = {};
  e->dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  e->handler.device_fn_ptr =
      reinterpret_cast<void *>(static_cast<uintptr_t>(1));
}
static void populate_bad(void *entry) {
  auto *e = static_cast<cudaq_function_entry_t *>(entry);
  *e = {};
  e->dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
}

cudaError_t test_create_dispatch(volatile std::uint64_t *,
                                 volatile std::uint64_t *, std::uint8_t *,
                                 std::uint8_t *, std::size_t, std::size_t,
                                 cudaq_function_entry_t *, std::size_t, void *,
                                 volatile int *, std::uint64_t *, std::size_t,
                                 std::uint32_t, std::uint32_t, cudaGraphExec_t,
                                 cudaStream_t,
                                 cudaq_dispatch_graph_context **out) {
  *out = static_cast<cudaq_dispatch_graph_context *>(zalloc(8));
  return cudaSuccess;
}
cudaError_t test_launch_dispatch(cudaq_dispatch_graph_context *, cudaStream_t) {
  return cudaSuccess;
}
cudaError_t test_destroy_dispatch(cudaq_dispatch_graph_context *ctx) {
  untrack(ctx);
  return cudaSuccess;
}

void *__wrap_dlsym(void *handle, const char *symbol) {
  if (!symbol)
    return __real_dlsym(handle, symbol);
  if (std::strstr(symbol, "populate_") != nullptr) {
    if (g_ctl.dlsym_omit_populate)
      return nullptr;
    return g_ctl.invalid_device_call ? reinterpret_cast<void *>(populate_bad)
                                     : reinterpret_cast<void *>(populate_ok);
  }
  if (std::strcmp(symbol, "cudaq_create_dispatch_graph_regular") == 0)
    return reinterpret_cast<void *>(test_create_dispatch);
  if (std::strcmp(symbol, "cudaq_launch_dispatch_graph") == 0)
    return reinterpret_cast<void *>(test_launch_dispatch);
  if (std::strcmp(symbol, "cudaq_destroy_dispatch_graph") == 0)
    return reinterpret_cast<void *>(test_destroy_dispatch);
  return __real_dlsym(handle, symbol);
}

} // extern "C"

namespace {

using namespace cudaq::qec::decoding_server;

// Missing and invalid DEVICE_CALL populate entries must throw and release the
// pinned function table. CUDA_QEC_ERROR currently throws before populate
// returns false, so the constructor's cudaFreeHost(ft_host_) does not run.
TEST(DeviceGraphRingConsumer, PopulateFailureReleasesPinnedMemory) {
  FakeBridge b;
  auto ring = valid_ring(b);
  auto gr = make_graph();
  auto expect_released = [&](auto set) {
    reset_ctl();
    set();
    // Construction must fail. The throw is the reported error.
    EXPECT_THROW((DeviceGraphRingConsumer(ring, 4, 64, 0, &gr)),
                 std::runtime_error);
    // The destructor does not run, so a leftover pointer is the pinned leak.
    EXPECT_EQ(live_count(), 0u);
  };
  expect_released([] { g_ctl.dlsym_omit_populate = true; });
  expect_released([] { g_ctl.invalid_device_call = true; });
}

// cudaMalloc success followed by cudaMemset failure must free d_stats_.
// The combined check frees only the host allocations, so the device buffer
// stays in the live set.
TEST(DeviceGraphRingConsumer, MemsetFailureReleasesDeviceStats) {
  FakeBridge b;
  auto ring = valid_ring(b);
  auto gr = make_graph();
  reset_ctl();
  g_ctl.memset_st = cudaErrorInvalidValue;
  // The constructor must fail once the device buffer cannot be cleared.
  EXPECT_THROW((DeviceGraphRingConsumer(ring, 4, 64, 0, &gr)),
               std::runtime_error);
  // Host allocations are freed on this path. A live pointer is d_stats_.
  EXPECT_EQ(live_count(), 0u);
}

// The C factory documents nullptr on failure. The catch calls CUDA_QEC_ERROR,
// which throws, so the nullptr return is unreachable.
TEST(DeviceGraphRingConsumer, FactoryReturnsNullptr) {
  reset_ctl();
  FakeBridge b;
  auto ring = valid_ring(b);
  cudaqx_qec_device_graph_ring_consumer_t made = nullptr;
  // A null graph must not escape the factory as an exception.
  EXPECT_NO_THROW(made = cudaqx_qec_make_device_graph_ring_consumer(
                      &ring, 4, 64, 0, nullptr));
  // The documented failure result is a null handle.
  EXPECT_EQ(made, nullptr);
}

} // namespace
