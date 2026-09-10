/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingServer.h"
#include "DeviceGraphRingConsumer.h"
#include "DeviceGraphTransceiver.h"
#include "qec_cc_test_helpers.h"

#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

struct cudaq_dispatch_graph_context {
  int token;
};

namespace {

std::mutex g_mu;
std::unordered_set<void *> g_live;
int g_bridge_create = 0, g_bridge_destroy = 0, g_bridge_connect = 0,
    g_bridge_launch = 0, g_bridge_disconnect = 0;
std::string g_last_library;
int g_host_alloc_n = 0, g_host_ptr_n = 0;

struct WrapCtl {
  cudaq_status_t create_st = CUDAQ_OK;
  bool create_null_handle = false;
  cudaq_status_t ctx_st = CUDAQ_OK;
  bool null_ring = false;
  cudaq_status_t geom_st = CUDAQ_OK;
  cudaq_status_t endpoint_st = CUDAQ_OK;
  std::string endpoint = "transport=test port=9 rest=foo";
  cudaq_status_t connect_st = CUDAQ_OK;
  cudaq_status_t launch_st = CUDAQ_OK;
  cudaError_t set_device = cudaSuccess;
  int cc_major = 9;
  int host_alloc_fail_on = 0;
  int host_ptr_fail_on = 0;
  cudaError_t malloc_st = cudaSuccess;
  cudaError_t memset_st = cudaSuccess;
  cudaError_t stream_st = cudaSuccess;
  cudaError_t create_dispatch = cudaSuccess;
  cudaError_t launch_dispatch = cudaSuccess;
  bool dlsym_omit_populate = false;
  bool invalid_device_call = false;
  bool dlsym_omit_dispatch = false;
  bool dlsym_omit_debug = false;
  cudaError_t get_device = cudaSuccess;
  cudaError_t memcpy_async = cudaSuccess;
  cudaError_t stream_flags = cudaSuccess;
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
  g_last_library.clear();
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

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver(
    int, const cudaq::qec::decoding::config::transport_shape_override *);

extern "C" {

void *__real_dlsym(void *handle, const char *symbol);

cudaError_t __wrap_cudaSetDevice(int) { return g_ctl.set_device; }
cudaError_t __wrap_cudaGetDevice(int *d) {
  if (d)
    *d = 0;
  return g_ctl.get_device;
}
cudaError_t __wrap_cudaGetDeviceProperties(cudaDeviceProp *p, int) {
  if (!p)
    return cudaErrorInvalidValue;
  std::memset(p, 0, sizeof(*p));
  p->major = g_ctl.cc_major;
  p->minor = 0;
  std::strncpy(p->name, "qec-test-gpu", sizeof(p->name) - 1);
  return cudaSuccess;
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
cudaError_t __wrap_cudaStreamCreate(cudaStream_t *s) {
  if (g_ctl.stream_st != cudaSuccess)
    return g_ctl.stream_st;
  *s = reinterpret_cast<cudaStream_t>(zalloc(8));
  return cudaSuccess;
}
cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int) {
  if (g_ctl.stream_flags != cudaSuccess)
    return g_ctl.stream_flags;
  *s = reinterpret_cast<cudaStream_t>(zalloc(8));
  return cudaSuccess;
}
cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t __wrap_cudaStreamDestroy(cudaStream_t s) {
  untrack(s);
  return cudaSuccess;
}
cudaError_t __wrap_cudaMemcpyAsync(void *dst, const void *src, size_t n,
                                   cudaMemcpyKind, cudaStream_t) {
  if (g_ctl.memcpy_async != cudaSuccess)
    return g_ctl.memcpy_async;
  if (dst && src)
    std::memcpy(dst, src, n);
  return cudaSuccess;
}
const char *__wrap_cudaGetErrorString(cudaError_t) { return "qec-test-cuda"; }

cudaq_status_t
__wrap_cudaq_bridge_create_from_library(cudaq_realtime_bridge_handle_t *out,
                                        const char *library, int, char **) {
  ++g_bridge_create;
  g_last_library = library ? library : "";
  if (g_ctl.create_st != CUDAQ_OK)
    return g_ctl.create_st;
  if (g_ctl.create_null_handle) {
    *out = nullptr;
    return CUDAQ_OK;
  }
  *out = zalloc(sizeof(FakeBridge));
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t h) {
  ++g_bridge_destroy;
  untrack(h);
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_bridge_get_transport_context(cudaq_realtime_bridge_handle_t h,
                                          cudaq_realtime_transport_context_t,
                                          void *out) {
  if (g_ctl.ctx_st != CUDAQ_OK)
    return g_ctl.ctx_st;
  auto *ring = static_cast<cudaq_ringbuffer_t *>(out);
  *ring = {};
  if (g_ctl.null_ring)
    return CUDAQ_OK;
  auto *b = static_cast<FakeBridge *>(h);
  ring->rx_flags = &b->rx_flags;
  ring->tx_flags = &b->tx_flags;
  ring->rx_data = b->rx_data;
  ring->tx_data = b->tx_data;
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_bridge_get_ring_geometry(cudaq_realtime_bridge_handle_t,
                                      uint32_t *slots, uint32_t *size) {
  if (g_ctl.geom_st != CUDAQ_OK)
    return g_ctl.geom_st;
  if (slots)
    *slots = 4;
  if (size)
    *size = 64;
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_bridge_get_endpoint_info(cudaq_realtime_bridge_handle_t, char *buf,
                                      size_t len) {
  if (g_ctl.endpoint_st != CUDAQ_OK)
    return g_ctl.endpoint_st;
  std::snprintf(buf, len, "%s", g_ctl.endpoint.c_str());
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_bridge_connect(cudaq_realtime_bridge_handle_t) {
  ++g_bridge_connect;
  return g_ctl.connect_st;
}
cudaq_status_t __wrap_cudaq_bridge_launch(cudaq_realtime_bridge_handle_t) {
  ++g_bridge_launch;
  return g_ctl.launch_st;
}
cudaq_status_t __wrap_cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t) {
  ++g_bridge_disconnect;
  return CUDAQ_OK;
}

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
  if (g_ctl.create_dispatch != cudaSuccess)
    return g_ctl.create_dispatch;
  *out = static_cast<cudaq_dispatch_graph_context *>(zalloc(8));
  return cudaSuccess;
}
cudaError_t test_launch_dispatch(cudaq_dispatch_graph_context *, cudaStream_t) {
  return g_ctl.launch_dispatch;
}
cudaError_t test_destroy_dispatch(cudaq_dispatch_graph_context *ctx) {
  untrack(ctx);
  return cudaSuccess;
}
cudaError_t test_get_debug(int *rc, unsigned long long *f,
                           unsigned long long *t) {
  if (rc)
    *rc = -1000;
  if (f)
    *f = 0;
  if (t)
    *t = 0;
  return cudaSuccess;
}

void *__wrap_dlsym(void *handle, const char *symbol) {
  if (!symbol)
    return __real_dlsym(handle, symbol);
  const bool pop = std::strstr(symbol, "populate_") != nullptr;
  if (pop) {
    if (g_ctl.dlsym_omit_populate)
      return nullptr;
    return g_ctl.invalid_device_call ? reinterpret_cast<void *>(populate_bad)
                                     : reinterpret_cast<void *>(populate_ok);
  }
  if (std::strcmp(symbol, "cudaq_create_dispatch_graph_regular") == 0)
    return g_ctl.dlsym_omit_dispatch
               ? nullptr
               : reinterpret_cast<void *>(test_create_dispatch);
  if (std::strcmp(symbol, "cudaq_launch_dispatch_graph") == 0)
    return g_ctl.dlsym_omit_dispatch
               ? nullptr
               : reinterpret_cast<void *>(test_launch_dispatch);
  if (std::strcmp(symbol, "cudaq_destroy_dispatch_graph") == 0)
    return g_ctl.dlsym_omit_dispatch
               ? nullptr
               : reinterpret_cast<void *>(test_destroy_dispatch);
  if (std::strcmp(symbol, "cudaq_dispatch_get_trigger_debug") == 0)
    return g_ctl.dlsym_omit_debug ? nullptr
                                  : reinterpret_cast<void *>(test_get_debug);
  return __real_dlsym(handle, symbol);
}

} // extern "C"

namespace {

using namespace cudaq::qec::decoding_server;

cudaq::qec::decoding::config::transport_shape_override
shape(const char *provider) {
  cudaq::qec::decoding::config::transport_shape_override t;
  t.provider = provider;
  return t;
}

TEST(DeviceGraphFactory, NullTransportThrows) {
  EXPECT_THROW((cudaqx_qec_make_device_graph_transceiver(0, nullptr)),
               std::invalid_argument);
}

TEST(DeviceGraphTransceiver, ValidationAndBridgeFailures) {
  DeviceGraphConfig empty;
  EXPECT_THROW(DeviceGraphTransceiver tx(empty), std::runtime_error);

  DeviceGraphConfig gpu_arg;
  gpu_arg.provider = "gpu_roce";
  gpu_arg.provider_args = {"--gpu=0"};
  EXPECT_THROW(DeviceGraphTransceiver tx(gpu_arg), std::runtime_error);

  auto try_cfg = [](auto set_fail) {
    reset_ctl();
    set_fail();
    g_bridge_create = g_bridge_destroy = 0;
    DeviceGraphConfig cfg;
    cfg.provider = "gpu_roce";
    EXPECT_THROW(DeviceGraphTransceiver tx(cfg), std::runtime_error);
  };
  try_cfg([] { g_ctl.create_st = CUDAQ_ERR_INTERNAL; });
  try_cfg([] { g_ctl.create_null_handle = true; });
  try_cfg([] { g_ctl.ctx_st = CUDAQ_ERR_UNSUPPORTED; });
  try_cfg([] { g_ctl.null_ring = true; });
  try_cfg([] { g_ctl.geom_st = CUDAQ_ERR_UNSUPPORTED; });
  try_cfg([] { g_ctl.endpoint_st = CUDAQ_ERR_UNSUPPORTED; });
  try_cfg([] { g_ctl.connect_st = CUDAQ_ERR_INTERNAL; });
  EXPECT_EQ(live_count(), 0u);
}

TEST(DeviceGraphTransceiver, ProviderPathAndCandidateAndSuccess) {
  reset_ctl();
  g_bridge_create = g_bridge_destroy = g_bridge_launch = 0;
  {
    DeviceGraphConfig slash;
    slash.provider = "/tmp/custom_provider.so";
    DeviceGraphTransceiver tx(slash);
    EXPECT_EQ(g_last_library, "/tmp/custom_provider.so");
  }
  reset_ctl();
  {
    DeviceGraphConfig cfg;
    cfg.provider = "gpu_roce";
    auto sh = shape("gpu_roce");
    auto *raw = cudaqx_qec_make_device_graph_transceiver(0, &sh);
    ASSERT_NE(raw, nullptr);
    std::unique_ptr<ITransceiver> t(raw);
    auto gr = make_graph();
    EXPECT_TRUE(t->launch_device_scheduler(&gr));
    std::thread waiter([&] { t->recv(); });
    t->shutdown();
    waiter.join();
    t->shutdown();
    EXPECT_THROW((t->send({}, nullptr, 0)), std::logic_error);
  }
  EXPECT_GT(g_bridge_destroy, 0);
  EXPECT_EQ(live_count(), 0u);
}

TEST(DeviceGraphTransceiver, LaunchProviderFailAndDtorWithoutShutdown) {
  reset_ctl();
  g_ctl.launch_st = CUDAQ_ERR_INTERNAL;
  DeviceGraphConfig cfg;
  cfg.provider = "gpu_roce";
  DeviceGraphTransceiver tx(cfg);
  auto gr = make_graph();
  EXPECT_THROW(tx.launch_scheduler(&gr), std::runtime_error);
}

TEST(DeviceGraphRingConsumer, NullGraphAndNullRingFields) {
  reset_ctl();
  FakeBridge b;
  auto ring = valid_ring(b);
  auto gr = make_graph();
  EXPECT_THROW((DeviceGraphRingConsumer(ring, 4, 64, 0, nullptr)),
               std::runtime_error);
  gr.graph_exec = nullptr;
  EXPECT_THROW((DeviceGraphRingConsumer(ring, 4, 64, 0, &gr)),
               std::runtime_error);
  cudaq_ringbuffer_t empty{};
  auto gr2 = make_graph();
  EXPECT_THROW((DeviceGraphRingConsumer(empty, 4, 64, 0, &gr2)),
               std::runtime_error);
}

TEST(DeviceGraphRingConsumer, CudaAndDlsymFailures) {
  FakeBridge b;
  auto ring = valid_ring(b);
  auto gr = make_graph();
  auto expect_throw = [&](auto set) {
    reset_ctl();
    set();
    EXPECT_THROW((DeviceGraphRingConsumer(ring, 4, 64, 0, &gr)),
                 std::runtime_error);
    EXPECT_EQ(live_count(), 0u);
  };
  expect_throw([] { g_ctl.set_device = cudaErrorInvalidValue; });
  expect_throw([] { g_ctl.cc_major = 8; });
  expect_throw([] { g_ctl.host_alloc_fail_on = 1; });
  expect_throw([] { g_ctl.host_ptr_fail_on = 1; });
  expect_throw([] { g_ctl.dlsym_omit_populate = true; });
  expect_throw([] { g_ctl.invalid_device_call = true; });
  expect_throw([] { g_ctl.dlsym_omit_dispatch = true; });
  expect_throw([] { g_ctl.host_alloc_fail_on = 2; });
  expect_throw([] { g_ctl.malloc_st = cudaErrorInvalidValue; });
  expect_throw([] { g_ctl.memset_st = cudaErrorInvalidValue; });
  expect_throw([] { g_ctl.stream_st = cudaErrorInvalidValue; });
  expect_throw([] { g_ctl.create_dispatch = cudaErrorInvalidValue; });
  expect_throw([] { g_ctl.launch_dispatch = cudaErrorInvalidValue; });
}

TEST(DeviceGraphRingConsumer, SuccessShutdownDispatchedAndCAbi) {
  reset_ctl();
  FakeBridge b;
  auto ring = valid_ring(b);
  auto gr = make_graph();
  {
    DeviceGraphRingConsumer c(ring, 4, 64, 0, &gr);
    c.shutdown();
    c.shutdown();
    (void)c.dispatched();
  }
  auto *ok = cudaqx_qec_make_device_graph_ring_consumer(&ring, 4, 64, 0, &gr);
  ASSERT_NE(ok, nullptr);
  cudaqx_qec_device_graph_ring_consumer_shutdown(ok);
  (void)cudaqx_qec_device_graph_ring_consumer_dispatched(ok);
  cudaqx_qec_device_graph_ring_consumer_destroy(ok);
  EXPECT_EQ(
      cudaqx_qec_make_device_graph_ring_consumer(&ring, 4, 64, 0, nullptr),
      nullptr);
  cudaqx_qec_device_graph_ring_consumer_shutdown(nullptr);
  EXPECT_EQ(cudaqx_qec_device_graph_ring_consumer_dispatched(nullptr), 0u);
  cudaqx_qec_device_graph_ring_consumer_destroy(nullptr);

  g_ctl.get_device = cudaErrorInvalidValue;
  {
    DeviceGraphRingConsumer c2(ring, 4, 64, 0, &gr);
    EXPECT_EQ(c2.dispatched(), 0u);
    g_ctl.get_device = cudaSuccess;
    g_ctl.dlsym_omit_debug = true;
    c2.shutdown();
  }
  EXPECT_EQ(live_count(), 0u);
}

} // namespace
