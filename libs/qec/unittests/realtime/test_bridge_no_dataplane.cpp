/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test-only transport provider that serves NO CPU data plane.
//
// decoding_server --host-loop=unified probes each host ring's provider for
// its CPU data plane (cudaq_bridge_get_cpu_dataplane) before READY and
// refuses to start in two ways: "does not serve the unified CPU data plane"
// when the provider answers CUDAQ_ERR_UNSUPPORTED, and "returned an
// incomplete CPU data plane" when it answers CUDAQ_OK with null hooks, null
// ring host views or zero slot strides (the unified loop dereferences those
// unchecked).  This provider triggers either one on demand: by default
// get_cpu_dataplane returns CUDAQ_ERR_UNSUPPORTED; with the provider argument
// --incomplete-dataplane it returns CUDAQ_OK and a zeroed
// cudaq_cpu_dataplane_t.
//
// The in-tree udp provider can no longer reach these paths from a test: the
// server appends --unified to every host ring's provider arguments itself
// under --host-loop=unified, so a udp build that serves the plane always
// passes the probe (and one that predates the plane only skips the test).
// test_decoding_server loads this library by path through --transport=, which
// also exercises the drop-in (out-of-tree) provider path end to end.
//
// Everything else is the minimum that keeps the server's startup sequence
// going until the probe: create()/destroy() manage a small heap handle,
// connect()/launch()/disconnect() succeed trivially, get_transport_context
// serves neither shape, get_endpoint_info reports a fixed string and
// get_ring_geometry reports 8 slots x 256 bytes.  Plain C++17, no CUDA calls;
// bridge_interface.h pulls in <cuda_runtime.h>, so the CUDA include directory
// is needed at compile time only.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

struct NoDataplaneHandle {
  bool incomplete_dataplane = false;
};

cudaq_status_t nd_create(cudaq_realtime_bridge_handle_t *out, int argc,
                         char **argv) {
  if (!out)
    return CUDAQ_ERR_INVALID_ARG;
  auto *h = new (std::nothrow) NoDataplaneHandle();
  if (!h)
    return CUDAQ_ERR_INTERNAL;
  // argv[0] is the program-name placeholder; every other argument (--port=,
  // the --unified the server appends, ...) is ignored, as the in-tree
  // providers do with arguments they do not know.
  for (int i = 1; i < argc; ++i)
    if (argv[i] && std::strcmp(argv[i], "--incomplete-dataplane") == 0)
      h->incomplete_dataplane = true;
  *out = h;
  return CUDAQ_OK;
}

cudaq_status_t nd_destroy(cudaq_realtime_bridge_handle_t handle) {
  delete static_cast<NoDataplaneHandle *>(handle);
  return CUDAQ_OK;
}

cudaq_status_t nd_get_transport_context(cudaq_realtime_bridge_handle_t,
                                        cudaq_realtime_transport_context_t,
                                        void *) {
  // Neither RING_BUFFER nor UNIFIED: this provider has no transport at all.
  return CUDAQ_ERR_UNSUPPORTED;
}

cudaq_status_t nd_connect(cudaq_realtime_bridge_handle_t) { return CUDAQ_OK; }
cudaq_status_t nd_launch(cudaq_realtime_bridge_handle_t) { return CUDAQ_OK; }
cudaq_status_t nd_disconnect(cudaq_realtime_bridge_handle_t) {
  return CUDAQ_OK;
}

cudaq_status_t nd_get_cpu_dataplane(cudaq_realtime_bridge_handle_t handle,
                                    cudaq_cpu_dataplane_t *out) {
  if (!handle || !out)
    return CUDAQ_ERR_INVALID_ARG;
  if (!static_cast<NoDataplaneHandle *>(handle)->incomplete_dataplane)
    return CUDAQ_ERR_UNSUPPORTED;
  // --incomplete-dataplane: "success" with an unusable plane (null hooks,
  // null ring views, zero strides) for the server's completeness check.
  std::memset(out, 0, sizeof(*out));
  return CUDAQ_OK;
}

cudaq_status_t nd_get_endpoint_info(cudaq_realtime_bridge_handle_t, char *buf,
                                    size_t buf_len) {
  if (!buf || buf_len == 0)
    return CUDAQ_ERR_INVALID_ARG;
  const int n =
      std::snprintf(buf, buf_len, "transport=test-no-dataplane port=0");
  return (n > 0 && static_cast<size_t>(n) < buf_len) ? CUDAQ_OK
                                                     : CUDAQ_ERR_INVALID_ARG;
}

cudaq_status_t nd_get_ring_geometry(cudaq_realtime_bridge_handle_t,
                                    uint32_t *out_num_slots,
                                    uint32_t *out_slot_size) {
  if (!out_num_slots || !out_slot_size)
    return CUDAQ_ERR_INVALID_ARG;
  *out_num_slots = 8;
  *out_slot_size = 256;
  return CUDAQ_OK;
}

} // namespace

extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_bridge_interface() {
  // Fields in header order.  set_function_table is NULL: the API wrapper then
  // returns CUDAQ_OK without calling into the provider.
  static cudaq_realtime_bridge_interface_t iface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      nd_create,
      nd_destroy,
      nd_get_transport_context,
      nd_connect,
      nd_launch,
      nd_disconnect,
      nd_get_cpu_dataplane,
      nd_get_endpoint_info,
      nd_get_ring_geometry,
      nullptr, // set_function_table
  };
  return &iface;
}
