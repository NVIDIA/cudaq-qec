/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Link canary for the device-graph component -- not meant to be executed
// (running it would require a transport provider, a GPU driver, and
// RDMA-capable hardware). Building it forces the linker to resolve
// DeviceGraphTransceiver's
// full dependency chain (gpu_roce, DOCA, CUDA driver stubs), so
// GpuRoceTransceiver API drift is caught at build time even on machines where
// nothing links the component into a runnable binary (driverless CI: the
// decoding_server tool's device_graph block is additionally gated on the
// proprietary cudevice archive, which CI does not provision).

namespace cudaq::qec::decoding_server {
struct ITransceiver;
}
namespace cudaq::qec::decoding::config {
struct transport_shape_override;
}

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver(
    int pinned_cuda_device,
    const cudaq::qec::decoding::config::transport_shape_override *transport);

using DeviceGraphFactoryFn = cudaq::qec::decoding_server::ITransceiver
    *(*)(int, const cudaq::qec::decoding::config::transport_shape_override *);

static DeviceGraphFactoryFn volatile device_graph_factory =
    &cudaqx_qec_make_device_graph_transceiver;

int main() { return device_graph_factory ? 0 : 1; }
