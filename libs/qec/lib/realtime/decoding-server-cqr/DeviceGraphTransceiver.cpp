/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

#include "DeviceGraphTransceiver.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/graph_resources.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// CUDA-Q realtime bridge-provider interface.
#include "cudaq/realtime/gpu_roce_bridge_common.h"

namespace cudaq::qec::decoding_server {

namespace {

std::string resolve_provider_library(const std::string &provider) {
  if (provider.find('/') != std::string::npos)
    return provider;

  std::string hyphenated = provider;
  std::replace(hyphenated.begin(), hyphenated.end(), '_', '-');
  const std::string soname = "libcudaq-realtime-bridge-" + hyphenated + ".so";
#ifdef QEC_BRIDGE_PROVIDER_DIR
  const std::string literal = "libcudaq-realtime-bridge-" + provider + ".so";
  for (const auto &name : {soname, literal}) {
    const std::string candidate =
        std::string(QEC_BRIDGE_PROVIDER_DIR) + "/" + name;
    if (std::ifstream(candidate).good())
      return candidate;
  }
#endif
  return soname;
}

bool is_gpu_argument(const std::string &arg) {
  return arg == "--gpu" || arg.rfind("--gpu=", 0) == 0;
}

} // namespace

// ---------------------------------------------------------------------------
// DeviceGraphTransceiver constructor
// ---------------------------------------------------------------------------

DeviceGraphTransceiver::DeviceGraphTransceiver(const DeviceGraphConfig &config)
    : gpu_id_(config.gpu_id) {
  if (config.provider.empty())
    throw std::runtime_error(
        "DeviceGraphTransceiver: device_graph transport provider must be set "
        "in YAML");
  if (std::any_of(config.provider_args.begin(), config.provider_args.end(),
                  is_gpu_argument))
    throw std::runtime_error(
        "DeviceGraphTransceiver: transport arguments must not set --gpu; "
        "set decoder cuda_device_id in YAML instead");

  // Bring the GpuRoceTransceiver up through the bridge-provider interface:
  // create() = gpu_roce_create_transceiver + gpu_roce_start (3-kernel shape:
  // no --forward / --unified => rx_only + tx_only kernels, with dispatch
  // supplied by our device-graph scheduler in launch_scheduler()).
  // args[0] is a program-name placeholder: the provider's parse_bridge_args
  // follows the C argv convention and starts parsing at argv[1] -- without
  // the placeholder the first real option would be silently skipped (and the
  // bridge would fall back to its built-in device default).
  std::vector<std::string> args{"device-graph-transceiver"};
  args.insert(args.end(), config.provider_args.begin(),
              config.provider_args.end());
  // The decoder's YAML cuda_device_id is authoritative for graph capture,
  // provider rings, and scheduler launch, so place it last.
  args.push_back("--gpu=" + std::to_string(config.gpu_id));
  std::vector<char *> argv;
  argv.reserve(args.size());
  for (auto &a : args)
    argv.push_back(a.data());

  const std::string provider_lib = resolve_provider_library(config.provider);
  if (cudaq_bridge_create_from_library(&bridge_, provider_lib.c_str(),
                                       static_cast<int>(argv.size()),
                                       argv.data()) != CUDAQ_OK ||
      !bridge_)
    throw std::runtime_error(
        "DeviceGraphTransceiver: bridge provider create failed for '" +
        config.provider + "' (resolved as " + provider_lib + ")");

  // Adopt the DOCA ring buffer GPU VRAM pointers from the provider.
  cudaq_ringbuffer_t ring{};
  if (cudaq_bridge_get_transport_context(bridge_, RING_BUFFER, &ring) !=
      CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider has no ring-buffer context");
  }
  rx_ring_data_ = ring.rx_data;
  rx_ring_flag_ = ring.rx_flags;
  tx_ring_data_ = ring.tx_data;
  tx_ring_flag_ = ring.tx_flags;
  if (!rx_ring_data_ || !rx_ring_flag_ || !tx_ring_data_ || !tx_ring_flag_) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: null DOCA ring pointer(s) from provider");
  }

  // Ring geometry and RDMA target identity come from the provider's
  // interface-v2 queries; the scheduler and the orchestration handshake both
  // depend on them, so a provider without v2 support is an error.
  uint32_t num_slots = 0, slot_size = 0;
  if (cudaq_bridge_get_ring_geometry(bridge_, &num_slots, &slot_size) !=
      CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider does not report ring geometry "
        "(bridge interface v2 required)");
  }
  num_pages_ = num_slots;
  page_size_ = slot_size;

  char info[512] = {0};
  if (cudaq_bridge_get_endpoint_info(bridge_, info, sizeof(info)) != CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider does not report endpoint info "
        "(bridge interface v2 required)");
  }
  endpoint_info_ = info;

  // connect(): the provider finalizes whatever rendezvous its wire needs
  // (no wire traffic for gpu_roce; the playback tool alone programs the
  // FPGA control plane).
  if (cudaq_bridge_connect(bridge_) != CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider connect() failed");
  }

  CUDA_QEC_INFO("DeviceGraphTransceiver: provider started  gpu={} pages={} "
                "page_size={}  endpoint: {}  "
                "(call launch_scheduler() before run())",
                config.gpu_id, num_pages_, page_size_, endpoint_info_);
}

// ---------------------------------------------------------------------------
// launch_scheduler
// ---------------------------------------------------------------------------

void DeviceGraphTransceiver::launch_scheduler(void *raw_graph_resources) {
  // All scheduler wiring (pinned function table + populate shims + dispatch
  // graph create/launch) lives in DeviceGraphRingConsumer; this transceiver
  // contributes only its provider's ring context and geometry.
  cudaq_ringbuffer_t ring{};
  ring.rx_flags = rx_ring_flag_;
  ring.tx_flags = tx_ring_flag_;
  ring.rx_data = rx_ring_data_;
  ring.tx_data = tx_ring_data_;
  ring.rx_stride_sz = page_size_;
  ring.tx_stride_sz = page_size_;
  consumer_ = std::make_unique<DeviceGraphRingConsumer>(
      ring, num_pages_, page_size_, gpu_id_, raw_graph_resources);

  // Start the provider's I/O loop (GpuRoceTransceiver RX/TX kernels + monitor
  // thread, owned by the provider) now that the scheduler is polling the rings.
  if (cudaq_bridge_launch(bridge_) != CUDAQ_OK) {
    consumer_->shutdown();
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: provider launch() failed");
  }

  CUDA_QEC_INFO("DeviceGraphTransceiver: GPU scheduler launched ({})",
                endpoint_info_);

  // Publish the provider's endpoint description VERBATIM so the
  // orchestration layer can scrape whatever rendezvous tokens its wire
  // needs (qp=/rkey=/buffer_addr= for RDMA playback, port= for sockets).
  // This class does not know or care which tokens are present.
  std::cout << "QEC_DECODING_SERVER_ENDPOINT " << endpoint_info_ << "\n";
  std::cout.flush();
}

// ---------------------------------------------------------------------------
// ITransceiver interface stubs (GPU scheduler handles the data path)
// ---------------------------------------------------------------------------

RxFrame DeviceGraphTransceiver::recv() {
  // The GPU device-graph scheduler handles RX→dispatch→decode→TX autonomously.
  // This method only exists so DecodingServer::run()'s recv loop blocks until
  // shutdown() is called.
  while (!stopped_.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return {}; // shutdown sentinel: empty buf causes the recv loop to exit
}

void DeviceGraphTransceiver::send(const PeerId & /*peer*/,
                                  const uint8_t * /*data*/, size_t /*len*/) {
  throw std::logic_error(
      "DeviceGraphTransceiver::send() must not be called: the CUDAQ "
      "device-graph "
      "scheduler writes TX responses directly to the GpuRoceTransceiver ring "
      "buffer");
}

// ---------------------------------------------------------------------------
// shutdown / destructor
// ---------------------------------------------------------------------------

void DeviceGraphTransceiver::shutdown() {
  if (stopped_.exchange(true, std::memory_order_acq_rel))
    return; // already stopped

  // Signal the GPU scheduler's self-relaunch loop to stop.
  if (consumer_)
    consumer_->shutdown();

  // Stop the GpuRoceTransceiver RX/TX kernels and join the provider's monitor
  // thread.
  if (bridge_)
    cudaq_bridge_disconnect(bridge_);
}

DeviceGraphTransceiver::~DeviceGraphTransceiver() {
  // Ensure clean shutdown even if the caller omitted shutdown().
  if (!stopped_.exchange(true, std::memory_order_acq_rel)) {
    if (consumer_)
      consumer_->shutdown();
    if (bridge_)
      cudaq_bridge_disconnect(bridge_);
  }
  // Drain + destroy the scheduler BEFORE the provider (it polls the
  // provider's ring memory).
  consumer_.reset();
  if (bridge_)
    cudaq_bridge_destroy(bridge_);
}

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
