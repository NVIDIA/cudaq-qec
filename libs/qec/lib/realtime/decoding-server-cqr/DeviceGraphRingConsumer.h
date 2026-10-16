/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// The C ABI block at the bottom is visible regardless of the component
// macro (the decoding_server tool includes this header for its weak
// references), so its includes live outside the #ifdef.
#include <cstddef>
#include <cstdint>

#ifdef CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime.h>

namespace cudaq::qec::decoding_server {

/// The device-graph RING CONSUMER: attaches the CUDAQ device-graph scheduler
/// (a self-relaunching GPU dispatch graph with the three QEC DEVICE_CALL
/// entries and one decoder's captured decode graph) to a ring buffer it does
/// NOT own.  This is the "GPU consumer" counterpart of a host dispatcher
/// object: give it any GPU-pollable ring (GpuRoceTransceiver DOCA rings, or
/// pinned+ mapped rings from a provider's --pinned-rings) plus the decoder's
/// graph resources, and RX->dispatch->decode->TX runs autonomously on the GPU.
///
/// Extracted from DeviceGraphTransceiver (which now delegates to this class)
/// so the decoding_server process can attach it directly to a ring created by
/// its own per-decoder provider loop -- host decoders and a device_graph
/// decoder then differ only in which consumer their ring gets.
class DeviceGraphRingConsumer {
public:
  /// Launches the scheduler.  \p ring must carry DEVICE-visible pointers in
  /// its rx/tx fields; \p raw_graph_resources is the opaque pointer from
  /// decoder::capture_decode_graph().  Throws on failure.
  DeviceGraphRingConsumer(const cudaq_ringbuffer_t &ring, std::size_t num_slots,
                          std::size_t slot_size, int gpu_id,
                          void *raw_graph_resources);
  ~DeviceGraphRingConsumer();

  DeviceGraphRingConsumer(const DeviceGraphRingConsumer &) = delete;
  DeviceGraphRingConsumer &operator=(const DeviceGraphRingConsumer &) = delete;

  /// Signal the scheduler's self-relaunch loop to stop.  Idempotent.
  void shutdown();

  /// Requests dispatched by the scheduler.  Call after shutdown() (the
  /// device-side counter is read back with a synchronizing copy).
  std::uint64_t dispatched() const;

private:
  int gpu_id_ = 0;
  cudaq_dispatch_graph_context *sched_ctx_ = nullptr;
  cudaStream_t sched_stream_ = nullptr;
  void *ft_host_ = nullptr;
  volatile int *shutdown_host_ = nullptr;
  volatile int *shutdown_dev_ = nullptr;
  std::uint64_t *d_stats_ = nullptr;
  cudaError_t (*fn_destroy_dispatch_graph_)(cudaq_dispatch_graph_context *) =
      nullptr;
  bool stopped_ = false;
};

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

//==============================================================================
// C ABI for consumers outside the device-graph component (e.g. the
// decoding_server tool, which references these WEAKLY so a build without the
// component still links; see the weak-factory pattern in DecodingServer.cpp).
//==============================================================================
extern "C" {
/// Opaque handle to a DeviceGraphRingConsumer.
typedef void *cudaqx_qec_device_graph_ring_consumer_t;

/// Create + launch; returns nullptr on failure (details on stderr).  `ring`
/// is a `const cudaq_ringbuffer_t *`.
cudaqx_qec_device_graph_ring_consumer_t
cudaqx_qec_make_device_graph_ring_consumer(const void *ring,
                                           std::size_t num_slots,
                                           std::size_t slot_size, int gpu_id,
                                           void *graph_resources);
void cudaqx_qec_device_graph_ring_consumer_shutdown(
    cudaqx_qec_device_graph_ring_consumer_t consumer);
std::uint64_t cudaqx_qec_device_graph_ring_consumer_dispatched(
    cudaqx_qec_device_graph_ring_consumer_t consumer);
void cudaqx_qec_device_graph_ring_consumer_destroy(
    cudaqx_qec_device_graph_ring_consumer_t consumer);
}
