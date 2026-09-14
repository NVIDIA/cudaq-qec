/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_cc_test_helpers.h"

#ifdef QEC_CC_STRONG_DEVICE_GRAPH
#include "ITransceiver.h"
#include "cc_test_graph_decoder.h"
#endif

#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern int decoding_server_main(int argc, char **argv);

extern "C" {
int __real_open(const char *path, int flags, ...);
int __real_open64(const char *path, int flags, ...);
int __real_close(int fd);
}

namespace {

cudaq_function_entry_t g_entries[3]{};
int g_table_null = 0;
int g_graph_null = 0;
int g_consumer_fail = 0;
int g_dma_fail = 0;
int g_dma_readonly = 0;
int g_dma_fd = -1;
int g_dma_closes = 0;
int g_bridge_fail_on = 0; // 1-based create that fails
int g_bridge_creates = 0;
std::string g_last_lib;
std::string g_endpoint = "transport=udp port=9 rest=foo";
int g_bridge_ok = 1;
int g_factory_throw = 0;
int g_fail = 0; // 1 geometry, 2 endpoint, 3 manager, 4 connect, 5 context,
                // 6 dg-launch, 7 dispatcher-start, 8 host-launch
int g_n_bridge_destroy = 0;
int g_n_bridge_disconnect = 0;
int g_n_bridge_connect = 0;
int g_n_bridge_launch = 0;
int g_n_mgr_create = 0;
int g_n_mgr_destroy = 0;
int g_n_disp_create = 0;
int g_n_disp_destroy = 0;
int g_n_disp_stop = 0;
int g_n_cons_create = 0;
int g_n_cons_shutdown = 0;
int g_n_cons_destroy = 0;

#ifdef QEC_CC_STRONG_DEVICE_GRAPH
class FakeTx final : public cudaq::qec::decoding_server::ITransceiver {
public:
  cudaq::qec::decoding_server::RxFrame recv() override {
    while (!stop_)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return {};
  }
  void send(const cudaq::qec::decoding_server::PeerId &, const uint8_t *,
            size_t) override {}
  void shutdown() override { stop_ = true; }
  bool launch_device_scheduler(void *) override { return true; }
  std::atomic<bool> stop_{false};
};
#endif

int call_main(std::vector<std::string> args) {
  std::vector<char *> argv;
  std::string argv0 = "decoding_server";
  argv.push_back(argv0.data());
  for (auto &a : args)
    argv.push_back(a.data());
  argv.push_back(nullptr);
  return decoding_server_main(static_cast<int>(argv.size() - 1), argv.data());
}

int dma_open(const char *path, int flags, bool creat, mode_t mode) {
  if (path && std::strcmp(path, "/dev/cpu_dma_latency") == 0) {
    if (g_dma_fail) {
      errno = EACCES;
      return -1;
    }
    int fd;
    if (g_dma_readonly)
      fd = __real_open("/dev/null", O_RDONLY);
    else
      fd = creat ? __real_open("/dev/null", flags, mode)
                 : __real_open("/dev/null", flags);
    g_dma_fd = fd;
    return fd;
  }
  return creat ? __real_open(path, flags, mode) : __real_open(path, flags);
}

} // namespace

extern "C" {

int __wrap_open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
    return dma_open(path, flags, true, mode);
  }
  return dma_open(path, flags, false, 0);
}
int __wrap_open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
    return dma_open(path, flags, true, mode);
  }
  return dma_open(path, flags, false, 0);
}

int __wrap_close(int fd) {
  if (g_dma_fd >= 0 && fd == g_dma_fd)
    ++g_dma_closes;
  return __real_close(fd);
}

cudaq_status_t
__wrap_cudaq_bridge_create_from_library(cudaq_realtime_bridge_handle_t *out,
                                        const char *library, int, char **) {
  ++g_bridge_creates;
  g_last_lib = library ? library : "";
  if (g_bridge_fail_on && g_bridge_creates == g_bridge_fail_on)
    return CUDAQ_ERR_INTERNAL;
  if (!g_bridge_ok)
    return CUDAQ_ERR_INTERNAL;
  *out = reinterpret_cast<cudaq_realtime_bridge_handle_t>(
      static_cast<uintptr_t>(g_bridge_creates));
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t) {
  ++g_n_bridge_destroy;
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t) {
  ++g_n_bridge_disconnect;
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_bridge_connect(cudaq_realtime_bridge_handle_t) {
  ++g_n_bridge_connect;
  if (g_fail == 4)
    return CUDAQ_ERR_INTERNAL;
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_bridge_launch(cudaq_realtime_bridge_handle_t) {
  ++g_n_bridge_launch;
  // g_fail==6 is mixed-dispatch: let the host ring launch, then fail the
  // device-graph ring after its consumer exists.
  if (g_fail == 6 && g_n_cons_create > 0)
    return CUDAQ_ERR_INTERNAL;
  if (g_fail == 8)
    return CUDAQ_ERR_INTERNAL;
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_bridge_get_ring_geometry(cudaq_realtime_bridge_handle_t,
                                      uint32_t *slots, uint32_t *size) {
  if (g_fail == 1)
    return CUDAQ_ERR_INTERNAL;
  if (slots)
    *slots = 4;
  if (size)
    *size = 64;
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_bridge_get_endpoint_info(cudaq_realtime_bridge_handle_t, char *buf,
                                      size_t len) {
  if (g_fail == 2)
    return CUDAQ_ERR_INTERNAL;
  std::snprintf(buf, len, "%s", g_endpoint.c_str());
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_bridge_get_transport_context(cudaq_realtime_bridge_handle_t,
                                          cudaq_realtime_transport_context_t,
                                          void *out) {
  if (g_fail == 5)
    return CUDAQ_ERR_INTERNAL;
  auto *ring = static_cast<cudaq_ringbuffer_t *>(out);
  *ring = {};
  static uint64_t flags = 1;
  static uint8_t data[64];
  ring->rx_flags = &flags;
  ring->tx_flags = &flags;
  ring->rx_data = data;
  ring->tx_data = data;
  return CUDAQ_OK;
}

cudaq_status_t
__wrap_cudaq_dispatch_manager_create(cudaq_dispatch_manager_t **m) {
  ++g_n_mgr_create;
  if (g_fail == 3)
    return CUDAQ_ERR_INTERNAL;
  *m = reinterpret_cast<cudaq_dispatch_manager_t *>(static_cast<uintptr_t>(1));
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_dispatch_manager_destroy(cudaq_dispatch_manager_t *) {
  ++g_n_mgr_destroy;
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_dispatcher_create(cudaq_dispatch_manager_t *,
                                              const cudaq_dispatcher_config_t *,
                                              cudaq_dispatcher_t **d) {
  ++g_n_disp_create;
  *d = reinterpret_cast<cudaq_dispatcher_t *>(static_cast<uintptr_t>(2));
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_dispatcher_destroy(cudaq_dispatcher_t *) {
  ++g_n_disp_destroy;
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_dispatcher_set_ringbuffer(cudaq_dispatcher_t *,
                                       const cudaq_ringbuffer_t *) {
  return CUDAQ_OK;
}
cudaq_status_t
__wrap_cudaq_dispatcher_set_function_table(cudaq_dispatcher_t *,
                                           const cudaq_function_table_t *) {
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_dispatcher_set_control(cudaq_dispatcher_t *,
                                                   volatile int *, uint64_t *) {
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_dispatcher_start(cudaq_dispatcher_t *) {
  if (g_fail == 7)
    return CUDAQ_ERR_INTERNAL;
  return CUDAQ_OK;
}
cudaq_status_t __wrap_cudaq_dispatcher_stop(cudaq_dispatcher_t *) {
  ++g_n_disp_stop;
  return CUDAQ_OK;
}

const cudaq_function_entry_t *
cudaqx_qec_decoding_server_host_call_table(std::uint32_t *count) {
  if (g_table_null) {
    if (count)
      *count = 0;
    return nullptr;
  }
  if (count)
    *count = 3;
  return g_entries;
}
void *cudaqx_qec_decoding_server_graph_resources(std::uint64_t) {
  if (g_graph_null)
    return nullptr;
  static int token;
  return &token;
}
void cudaqx_qec_decoding_server_shutdown() {}
void cudaqx_qec_decoding_server_print_stats() {}
std::uint64_t cudaqx_qec_device_call_dispatch_count() { return 0; }
std::uint64_t cudaqx_qec_decoding_server_max_concurrent() { return 0; }

#ifdef QEC_CC_STRONG_DEVICE_GRAPH
cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver(
    int, const cudaq::qec::decoding::config::transport_shape_override *) {
  if (g_factory_throw)
    throw std::runtime_error("factory-boom");
  return new FakeTx();
}
void *cudaqx_qec_make_device_graph_ring_consumer(const void *, std::size_t,
                                                 std::size_t, int, void *) {
  ++g_n_cons_create;
  if (g_consumer_fail)
    return nullptr;
  static int token;
  return &token;
}
void cudaqx_qec_device_graph_ring_consumer_shutdown(void *) {
  ++g_n_cons_shutdown;
}
std::uint64_t cudaqx_qec_device_graph_ring_consumer_dispatched(void *) {
  return 0;
}
void cudaqx_qec_device_graph_ring_consumer_destroy(void *) {
  ++g_n_cons_destroy;
}
#endif

} // extern "C"

#ifdef QEC_CC_STRONG_DEVICE_GRAPH
namespace cudaq::qec {
CUDAQ_EXT_PT_REGISTER_TYPE(cc_test_graph_decoder)
CUDAQ_EXT_PT_REGISTER_TYPE(cc_test_null_graph_decoder)
} // namespace cudaq::qec
#endif

namespace {

int run_helper(const char *name) {
  g_table_null = 0;
  g_graph_null = 0;
  g_consumer_fail = 0;
  g_dma_fail = 0;
  g_dma_readonly = 0;
  g_dma_fd = -1;
  g_dma_closes = 0;
  g_bridge_fail_on = 0;
  g_bridge_creates = 0;
  g_bridge_ok = 1;
  g_factory_throw = 0;
  g_fail = 0;
  g_n_bridge_destroy = 0;
  g_n_bridge_disconnect = 0;
  g_n_bridge_connect = 0;
  g_n_bridge_launch = 0;
  g_n_mgr_create = 0;
  g_n_mgr_destroy = 0;
  g_n_disp_create = 0;
  g_n_disp_destroy = 0;
  g_n_disp_stop = 0;
  g_n_cons_create = 0;
  g_n_cons_shutdown = 0;
  g_n_cons_destroy = 0;
  g_endpoint = "transport=udp port=9 rest=foo";

  auto two_host = [] {
    return qec_cc::lut_yaml(0, "host") +
           "  - id: 1\n    type: single_error_lut\n    dispatch: host\n"
           "    block_size: 1\n    syndrome_size: 1\n"
           "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, "
           "-1]\n";
  };
  auto host_ok = [&](int rc, int creates, int destroys, int disc, int conn,
                     int launch, int mgr_c, int mgr_d, int disp_c, int disp_s,
                     int disp_d) {
    return rc == 1 && g_bridge_creates == creates &&
           g_n_bridge_destroy == destroys && g_n_bridge_disconnect == disc &&
           g_n_bridge_connect == conn && g_n_bridge_launch == launch &&
           g_n_mgr_create == mgr_c && g_n_mgr_destroy == mgr_d &&
           g_n_disp_create == disp_c && g_n_disp_stop == disp_s &&
           g_n_disp_destroy == disp_d;
  };

  if (std::strcmp(name, "help") == 0)
    return call_main({"--help"}) == 1 ? 0 : 2;
  if (std::strcmp(name, "no_config") == 0)
    return call_main({}) == 1 ? 0 : 2;
  if (std::strcmp(name, "bad_timeout") == 0)
    return call_main({"--config=/tmp/x.yaml", "--timeout=abc"}) == 1 ? 0 : 2;
  if (std::strcmp(name, "missing") == 0)
    return call_main({"--config=/no/such/qec-cc.yaml"}) == 1 ? 0 : 2;
  if (std::strcmp(name, "empty") == 0) {
    auto p = qec_cc::write_temp("decoders: []\n");
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "not_yaml") == 0) {
    auto p = qec_cc::write_temp("decoders: [");
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "dma_fail") == 0) {
    g_dma_fail = 1;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return call_main({"--config=" + p, "--timeout=-1"}) == 0 ? 0 : 2;
  }
  if (std::strcmp(name, "dma_ok") == 0) {
    g_dma_fail = 0;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return call_main({"--config=" + p, "--timeout=-1"}) == 0 ? 0 : 2;
  }
  if (std::strcmp(name, "dma_readonly") == 0) {
    g_dma_readonly = 1;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    const int rc = call_main({"--config=" + p, "--timeout=-1"});
    return (rc == 0 && g_dma_closes == 1) ? 0 : 2;
  }
  if (std::strcmp(name, "bridge_fail") == 0) {
    g_bridge_ok = 0;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "cpu_roce") == 0) {
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    int rc =
        call_main({"--config=" + p, "--transport=cpu_roce", "--timeout=-1"});
    return (rc == 0 && g_last_lib.find("cpu-roce") != std::string::npos) ? 0
                                                                         : 2;
  }
  if (std::strcmp(name, "slash_path") == 0) {
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    int rc = call_main(
        {"--config=" + p, "--transport=/tmp/libfoo.so", "--timeout=-1"});
    return (rc == 0 && g_last_lib == "/tmp/libfoo.so") ? 0 : 2;
  }
  if (std::strcmp(name, "bad_port") == 0) {
    g_endpoint = "transport=udp port=xyz rest=foo";
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return call_main({"--config=" + p, "--timeout=-1"}) == 0 ? 0 : 2;
  }
  if (std::strcmp(name, "explicit_port") == 0) {
    auto yaml = two_host();
    auto p = qec_cc::write_temp(yaml);
    g_bridge_fail_on = 2;
    return call_main({"--config=" + p, "--port=1234"}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "port_in_args") == 0) {
    auto yaml = two_host() + "transport:\n  provider: udp\n  args:\n    - "
                             "--port=1234\n";
    auto p = qec_cc::write_temp(yaml);
    g_bridge_fail_on = 2;
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "fail_geometry") == 0) {
    g_fail = 1;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return host_ok(call_main({"--config=" + p}), 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
               ? 0
               : 2;
  }
  if (std::strcmp(name, "fail_endpoint") == 0) {
    g_fail = 2;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return call_main({"--config=" + p, "--timeout=-1"}) == 0 ? 0 : 2;
  }
  if (std::strcmp(name, "fail_manager") == 0) {
    g_fail = 3;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return host_ok(call_main({"--config=" + p}), 1, 1, 1, 0, 0, 1, 0, 0, 0, 0)
               ? 0
               : 2;
  }
  if (std::strcmp(name, "fail_connect") == 0) {
    g_fail = 4;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return host_ok(call_main({"--config=" + p}), 1, 1, 1, 1, 0, 1, 1, 0, 0, 0)
               ? 0
               : 2;
  }
  if (std::strcmp(name, "fail_context") == 0) {
    g_fail = 5;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return host_ok(call_main({"--config=" + p}), 1, 1, 1, 1, 0, 1, 1, 0, 0, 0)
               ? 0
               : 2;
  }
  if (std::strcmp(name, "fail_disp_start") == 0) {
    g_fail = 7;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return host_ok(call_main({"--config=" + p}), 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)
               ? 0
               : 2;
  }
  if (std::strcmp(name, "fail_host_launch") == 0) {
    g_fail = 8;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return host_ok(call_main({"--config=" + p}), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
               ? 0
               : 2;
  }
  if (std::strcmp(name, "table_null") == 0) {
    g_table_null = 1;
    auto p = qec_cc::write_temp(qec_cc::lut_yaml());
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
#ifdef QEC_CC_STRONG_DEVICE_GRAPH
  if (std::strcmp(name, "mixed_no_graph") == 0) {
    g_graph_null = 1;
    auto p = qec_cc::write_temp(qec_cc::mixed_lut_yaml() +
                                "transport:\n  provider: udp\n  device_graph:\n"
                                "    provider: gpu_roce\n");
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "mixed_consumer_fail") == 0) {
    g_consumer_fail = 1;
    auto p = qec_cc::write_temp(qec_cc::mixed_lut_yaml() +
                                "transport:\n  provider: udp\n  device_graph:\n"
                                "    provider: gpu_roce\n");
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "all_graph") == 0) {
    std::string yaml =
        "decoders:\n  - id: 0\n    type: cc_test_graph_decoder\n"
        "    dispatch: device_graph\n    block_size: 1\n    syndrome_size: 1\n"
        "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1]\n"
        "transport:\n  provider: gpu_roce\n";
    auto p = qec_cc::write_temp(yaml);
    return call_main({"--config=" + p, "--timeout=-1"}) == 0 ? 0 : 2;
  }
  if (std::strcmp(name, "all_graph_sigterm") == 0) {
    std::string yaml =
        "decoders:\n  - id: 0\n    type: cc_test_graph_decoder\n"
        "    dispatch: device_graph\n    block_size: 1\n    syndrome_size: 1\n"
        "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1]\n"
        "transport:\n  provider: gpu_roce\n";
    auto p = qec_cc::write_temp(yaml);
    std::thread killer([] {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      ::kill(::getpid(), SIGTERM);
    });
    killer.detach();
    return call_main({"--config=" + p, "--timeout=30"}) == 0 ? 0 : 2;
  }
  if (std::strcmp(name, "all_graph_factory_throw") == 0) {
    g_factory_throw = 1;
    std::string yaml =
        "decoders:\n  - id: 0\n    type: cc_test_graph_decoder\n"
        "    dispatch: device_graph\n    block_size: 1\n    syndrome_size: 1\n"
        "    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1]\n"
        "transport:\n  provider: gpu_roce\n";
    auto p = qec_cc::write_temp(yaml);
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "fail_dg_launch") == 0) {
    g_fail = 6;
    auto p = qec_cc::write_temp(qec_cc::mixed_lut_yaml() +
                                "transport:\n  provider: udp\n  device_graph:\n"
                                "    provider: gpu_roce\n");
    const int rc = call_main({"--config=" + p});
    const bool ok = rc == 1 && g_bridge_creates == 2 &&
                    g_n_bridge_destroy == 2 && g_n_bridge_disconnect == 2 &&
                    g_n_cons_create == 1 && g_n_cons_shutdown == 1 &&
                    g_n_cons_destroy == 1 && g_n_disp_create == 1 &&
                    g_n_disp_stop == 1 && g_n_disp_destroy == 1 &&
                    g_n_mgr_create == 1 && g_n_mgr_destroy == 1;
    return ok ? 0 : 2;
  }
  if (std::strcmp(name, "mixed_ok") == 0) {
    auto p = qec_cc::write_temp(qec_cc::mixed_lut_yaml() +
                                "transport:\n  provider: udp\n  device_graph:\n"
                                "    provider: gpu_roce\n");
    return call_main({"--config=" + p, "--timeout=-1"}) == 0 ? 0 : 2;
  }
#else
  if (std::strcmp(name, "mixed_not_linked") == 0) {
    auto p = qec_cc::write_temp(qec_cc::mixed_lut_yaml() +
                                "transport:\n  provider: udp\n  device_graph:\n"
                                "    provider: gpu_roce\n");
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
  if (std::strcmp(name, "all_graph_not_linked") == 0) {
    auto p = qec_cc::write_temp(qec_cc::lut_yaml(0, "device_graph") +
                                "transport:\n  provider: gpu_roce\n");
    return call_main({"--config=" + p}) == 1 ? 0 : 2;
  }
#endif
  return 1;
}

TEST(DecodingServerCli, HelperScenarios) {
  EXPECT_EQ(qec_cc::exec_self("help"), 0);
  EXPECT_EQ(qec_cc::exec_self("no_config"), 0);
  EXPECT_EQ(qec_cc::exec_self("bad_timeout"), 0);
  EXPECT_EQ(qec_cc::exec_self("missing"), 0);
  EXPECT_EQ(qec_cc::exec_self("empty"), 0);
  EXPECT_EQ(qec_cc::exec_self("not_yaml"), 0);
  EXPECT_EQ(qec_cc::exec_self(
                "dma_fail", {{"QEC_DECODING_SERVER_CPU_DMA_LATENCY_US", "1"}}),
            0);
  EXPECT_EQ(qec_cc::exec_self(
                "dma_ok", {{"QEC_DECODING_SERVER_CPU_DMA_LATENCY_US", "1"}}),
            0);
  EXPECT_EQ(
      qec_cc::exec_self("dma_readonly",
                        {{"QEC_DECODING_SERVER_CPU_DMA_LATENCY_US", "1"}}),
      0);
  EXPECT_EQ(qec_cc::exec_self("bridge_fail"), 0);
  EXPECT_EQ(qec_cc::exec_self("cpu_roce"), 0);
  EXPECT_EQ(qec_cc::exec_self("slash_path"), 0);
  EXPECT_EQ(qec_cc::exec_self("bad_port"), 0);
  EXPECT_EQ(qec_cc::exec_self("explicit_port"), 0);
  EXPECT_EQ(qec_cc::exec_self("port_in_args"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_geometry"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_endpoint"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_manager"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_connect"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_context"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_disp_start"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_host_launch"), 0);
  EXPECT_EQ(qec_cc::exec_self("table_null"), 0);
#ifdef QEC_CC_STRONG_DEVICE_GRAPH
  EXPECT_EQ(qec_cc::exec_self("mixed_no_graph"), 0);
  EXPECT_EQ(qec_cc::exec_self("mixed_consumer_fail"), 0);
  EXPECT_EQ(qec_cc::exec_self("all_graph"), 0);
  EXPECT_EQ(qec_cc::exec_self("all_graph_sigterm"), 0);
  EXPECT_EQ(qec_cc::exec_self("all_graph_factory_throw"), 0);
  EXPECT_EQ(qec_cc::exec_self("fail_dg_launch"), 0);
  EXPECT_EQ(qec_cc::exec_self("mixed_ok"), 0);
#else
  EXPECT_EQ(qec_cc::exec_self("mixed_not_linked"), 0);
  EXPECT_EQ(qec_cc::exec_self("all_graph_not_linked"), 0);
#endif
}

} // namespace

int main(int argc, char **argv) {
  if (const char *helper = qec_cc::helper_name(argc, argv))
    return run_helper(helper);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
