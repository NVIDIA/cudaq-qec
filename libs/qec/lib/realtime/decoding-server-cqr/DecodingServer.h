/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "SessionRegistry.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cudaq::qec::decoding_server {

/// Resolve the CUDA device a decode pipeline runs on from the decoder's
/// cuda_device_id (-1 when unpinned). An unpinned decoder defaults to device 0.
int resolve_decode_device(int decoder_pin);

/// Lifecycle shell for the device_graph dispatch path: owns the session
/// registry and the device-graph transceiver, and wires the decoder's
/// captured CUDA graph to the on-device scheduler.
///
/// Host dispatch does not live here: it is served by the CQR HOST_CALL
/// plugin (decoding_server_cqr.cpp), which executes each request inline on
/// the CUDAQ dispatcher thread that delivered it.  After the scheduler
/// launches, the GPU handles the full RX→dispatch→decode→TX loop
/// autonomously; run() only parks the calling thread until stop().
class DecodingServer {
public:
  /// Config-driven constructor: loads the decoder sessions from
  /// \p config_yaml, creates the device-graph transceiver, and launches the
  /// on-device scheduler.  Throws for host-dispatch configs (served by the
  /// CQR plugin) and when the device-graph component is not linked.
  explicit DecodingServer(const std::string &config_yaml);

  /// Opaque graph resources of one decoder session
  /// (decoder::capture_decode_graph()), or nullptr when the decoder does not
  /// support graph dispatch / the id is unknown.  Used by the decoding_server
  /// process to wire a device-graph ring consumer to a decoder this server
  /// hosts.
  void *graph_resources_for(uint64_t decoder_id) const;

  /// This server's session registry (read-only after construction).
  const SessionRegistry &registry() const { return registry_; }

  ~DecodingServer();

  /// Block until stop() is called (the GPU scheduler owns the data path).
  void run();

  /// Thread-safe; releases run() and shuts the transports down.
  void stop();

  /// Print one QEC_DECODING_SERVER_DECODER_STATS line per session to stdout
  /// (test/diagnostic evidence; callers gate on QEC_DECODING_SERVER_STATS).
  void print_session_stats() const;

private:
  /// Create a transceiver for \p dispatch and its resolved YAML transport.
  /// Throws for host dispatch (served by the CQR plugin) and when the
  /// device-graph component is not linked.
  static std::unique_ptr<ITransceiver> make_transport(
      cudaq::qec::decoding::config::DecoderDispatch dispatch,
      int pinned_cuda_device,
      const cudaq::qec::decoding::config::transport_shape_override &transport);

  // Destruction order matters: the device-graph scheduler (inside
  // owned_transports_) holds a cudaGraphExec_t captured from a session's
  // decoder.  The scheduler must be destroyed (cudaStreamSynchronize +
  // cudaq_destroy_dispatch_graph) before registry_ releases the decoder and
  // its graph resources. C++ destroys members in reverse declaration order,
  // so registry_ must be declared BEFORE owned_transports_.
  SessionRegistry registry_;
  std::vector<std::unique_ptr<ITransceiver>> owned_transports_;

  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  bool shutdown_ = false;
};

} // namespace cudaq::qec::decoding_server
