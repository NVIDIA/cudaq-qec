/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Strong definition of the device-graph transceiver factory that
// DecodingServer.cpp declares
// weakly.  This translation unit lives in
// cudaq-qec-decoding-server-device-graph (NOT the core library) so that only
// binaries linking that component carry the DOCA / GpuRoceTransceiver /
// CUDA-driver dependencies.  Consumers must link the component WHOLE_ARCHIVE:
// the sole reference to this symbol is weak, which does not pull archive
// members on its own.

#include "DecodingServer.h" // resolve_decode_device (core symbol)
#include "DeviceGraphTransceiver.h"

#include <stdexcept>

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver(
    int pinned_cuda_device,
    const cudaq::qec::decoding::config::transport_shape_override *transport) {
  using namespace cudaq::qec::decoding_server;
  if (!transport)
    throw std::invalid_argument(
        "device-graph transport configuration is missing");

  // The device-graph GPU is the decoder's cuda_device_id pin; resolve it
  // here, inside the component, where DeviceGraphConfig is visible.
  DeviceGraphConfig cfg;
  cfg.provider = transport->provider;
  cfg.provider_args = transport->args;
  cfg.gpu_id = resolve_decode_device(pinned_cuda_device);
  return new DeviceGraphTransceiver(cfg);
}
