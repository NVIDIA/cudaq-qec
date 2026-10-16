/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingServer.h"
#include "../../hardware_guards.h"

#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

// Device-graph dispatch is an optional component
// (cudaq-qec-decoding-server-device-graph) so this core library carries no DOCA
// / GpuRoceTransceiver / CUDA-driver dependencies: those .so's require
// libcuda.so.1 at load time, which core consumers (unit tests, the CQR plugin)
// must not impose on driverless machines.  Binaries that want device_graph
// dispatch link the component WHOLE_ARCHIVE, whose DeviceGraphFactory.cpp
// provides the strong definition of this factory; anywhere else the weak
// reference is null and make_transport throws.
extern "C" __attribute__((weak)) cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver(
    int pinned_cuda_device,
    const cudaq::qec::decoding::config::transport_shape_override *transport);

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::DecoderDispatch;
using cudaq::qec::decoding::config::transport_shape_override;

/// Resolve the CUDA device a decode pipeline runs on from the decoder's
/// cuda_device_id pin; an unpinned decoder (-1) defaults to device 0. The
/// device_graph path relies on this to place its rings, dispatch scheduler,
/// and device-side graph fire on the one GPU the FPGA/NIC is affine to --
/// CUDA graphs cannot split capture and launch across devices, so the decoder
/// must be pinned to that device.
int resolve_decode_device(int decoder_pin) {
  return detail_affinity::decode_device_for(decoder_pin);
}

std::unique_ptr<ITransceiver>
DecodingServer::make_transport(DecoderDispatch dispatch, int pinned_cuda_device,
                               const transport_shape_override &transport) {
  switch (dispatch) {
  case DecoderDispatch::device_graph:
    // device_graph lives in the cudaq-qec-decoding-server-device-graph
    // component, reached through the weak factory.  The device is the
    // decoder's cuda_device_id pin, resolved inside the factory where the
    // transceiver config lives; we just thread the pin to it.
    if (cudaqx_qec_make_device_graph_transceiver)
      return std::unique_ptr<ITransceiver>(
          cudaqx_qec_make_device_graph_transceiver(pinned_cuda_device,
                                                   &transport));
    throw std::runtime_error(
        "device_graph dispatch requested but the device-graph component is "
        "not linked into this binary. Link "
        "cudaq-qec-decoding-server-device-graph (whole-archive).");

  case DecoderDispatch::host:
    // Host dispatch runs through the CQR HOST_CALL plugin, which serves each
    // request inline on the CUDAQ dispatcher thread; the standalone
    // DecodingServer has no host transport of its own.
    throw std::runtime_error(
        "host dispatch is served by the CQR HOST_CALL plugin, not the "
        "standalone DecodingServer; use dispatch: device_graph here");
  }
  throw std::runtime_error("make_transport: unknown DecoderDispatch value");
}

DecodingServer::DecodingServer(const std::string &config_yaml) {
  // Parse the YAML once: SessionRegistry validates the decoder entries and
  // required_dispatch() then drives transceiver creation (a mixed config
  // throws — heterogeneous deployments are composed by the decoding_server
  // process, which binds a consumer per decoder).
  std::ifstream f(config_yaml);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config: " + config_yaml);
  std::string yaml_str((std::istreambuf_iterator<char>(f)), {});
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_str);
  if (config.decoders.empty())
    throw std::runtime_error("No decoders in config: " + config_yaml);
  registry_.load_from_config(
      config, config_yaml,
      std::filesystem::absolute(config_yaml).parent_path());

  const auto dispatch = registry_.required_dispatch();
  // device_graph must run on the GPU the FPGA/NIC is affine to; when exactly
  // one session is booting, pass its decoder's cuda_device_id so the factory
  // can place the transport on that device.
  const auto &boot_sessions = registry_.sessions();
  const int pinned_cuda_device =
      boot_sessions.size() == 1
          ? boot_sessions.begin()->second->dec->get_cuda_device_id()
          : -1;
  auto t = make_transport(dispatch, pinned_cuda_device,
                          config.transport.resolve_device_graph());
  ITransceiver *raw = t.get();
  owned_transports_.push_back(std::move(t));

  // Wire the first (and only) session's decoder graph to the ring buffer via
  // the CUDAQ device-graph scheduler.  Multi-decoder device_graph binding is
  // deferred to a follow-up.
  if (dispatch == DecoderDispatch::device_graph) {
    const auto &sessions = registry_.sessions();
    if (sessions.size() != 1)
      throw std::runtime_error(
          "device_graph dispatch currently supports exactly one decoder "
          "session; "
          "found " +
          std::to_string(sessions.size()) +
          ". Multi-decoder device_graph dispatch is deferred.");
    auto *session = sessions.begin()->second.get();
    if (!session->graph_resources)
      throw std::runtime_error(
          "device_graph dispatch requires a decoder that supports graph "
          "dispatch "
          "(supports_graph_dispatch() must return true and "
          "capture_decode_graph() must succeed)");
    if (!raw->launch_device_scheduler(session->graph_resources.get()))
      throw std::runtime_error(
          "device_graph transceiver did not provide a device scheduler");
  }
}

void *DecodingServer::graph_resources_for(uint64_t decoder_id) const {
  const auto &sessions = registry_.sessions();
  const auto iter = sessions.find(decoder_id);
  if (iter == sessions.end() || !iter->second->graph_resources)
    return nullptr;
  return iter->second->graph_resources.get();
}

DecodingServer::~DecodingServer() { stop(); }

// ---------------------------------------------------------------------------
// run / stop
// ---------------------------------------------------------------------------

void DecodingServer::run() {
  // The GPU scheduler owns the entire data path (RX→dispatch→decode→TX);
  // there is nothing to receive on the CPU.  Park until stop().
  std::unique_lock<std::mutex> lk(stop_mutex_);
  stop_cv_.wait(lk, [this] { return shutdown_; });
}

void DecodingServer::print_session_stats() const {
  for (const auto &[id, session] : registry_.sessions()) {
    std::cout << "QEC_DECODING_SERVER_DECODER_STATS id=" << id
              << " decodes=" << session->decode_count.load()
              << " enqueues=" << session->enqueue_count.load()
              << " corrections=" << session->get_corrections_count.load()
              << " resets=" << session->reset_count.load()
              << " errors=" << session->error_count.load() << std::endl;
  }
}

void DecodingServer::stop() {
  {
    std::lock_guard<std::mutex> lk(stop_mutex_);
    shutdown_ = true;
  }
  stop_cv_.notify_all();
  for (auto &t : owned_transports_)
    t->shutdown();
}

} // namespace cudaq::qec::decoding_server
