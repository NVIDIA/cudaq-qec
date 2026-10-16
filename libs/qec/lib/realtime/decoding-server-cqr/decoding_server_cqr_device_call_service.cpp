/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// In-process DeviceCallService shim for the decoding server.
///
/// CUDA-Q discovers a host-dispatch service by dlsym'ing
/// cudaqGetDeviceCallServicePluginInfo out of the loaded image and then
/// driving it through the DeviceCallService/DeviceCallServiceSession virtual
/// interface.  That handshake is CUDA-Q's, so this file is the only part of
/// the CQR plugin that names CUDA-Q types -- and the only part that needs
/// libcudaq-device-call-runtime, which defines those two base classes' key
/// functions.
///
/// It is therefore compiled only when a full CUDA-Q install is present.  The
/// function table itself, and every handler behind it, lives in
/// decoding_server_cqr.cpp and is reachable through the plain-C
/// cudaqx_qec_decoding_server_host_call_table() accessor -- which is how the
/// standalone decoding_server process gets it without a CUDA-Q install.  Both
/// paths therefore serve one table built one way.
///
/// Consumers of this shim are the in-process host_dispatch users:
/// unittests/decoders/pymatching/test_pymatching_device_call_realtime.cpp and
/// unittests/realtime/app_examples/surface_code-5-per-decoder-rings.cpp.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/device_call_service.h"

#include <cstdint>
#include <memory>

/// Defined in decoding_server_cqr.cpp.
extern "C" const cudaq_function_entry_t *
cudaqx_qec_decoding_server_host_call_table(std::uint32_t *count);

namespace {

using cudaq::realtime::DeviceCallDispatchMode;
using cudaq::realtime::DeviceCallDispatchTable;
using cudaq::realtime::DeviceCallService;
using cudaq::realtime::DeviceCallServiceSession;

constexpr std::int32_t kHostDispatchDeviceId = 0;

class QecDeviceCallSession : public DeviceCallServiceSession {
public:
  QecDeviceCallSession(const cudaq_function_entry_t *entries,
                       std::uint32_t count) {
    table_.mode = DeviceCallDispatchMode::Host;
    // The accessor owns the entries for the life of the process, so handing
    // out a pointer to them is safe for as long as the channel is up.
    table_.entries = const_cast<cudaq_function_entry_t *>(entries);
    table_.count = count;
    table_.deviceId = kHostDispatchDeviceId;
    table_.mailbox = nullptr;
  }

  const DeviceCallDispatchTable &dispatchTable() const noexcept override {
    return table_;
  }

private:
  DeviceCallDispatchTable table_;
};

class QecDeviceCallService : public DeviceCallService {
public:
  std::unique_ptr<DeviceCallServiceSession>
  createDispatchSession(DeviceCallDispatchMode mode) override {
    if (mode != DeviceCallDispatchMode::Host)
      return nullptr;
    std::uint32_t count = 0;
    const auto *entries = cudaqx_qec_decoding_server_host_call_table(&count);
    // A null table means decoder initialization failed; the accessor has
    // already reported why.  CUDA-Q core does not expect plugin session
    // creation to throw -- a propagating exception would escape the
    // channel-setup path and terminate -- so decline the session instead.
    if (!entries || count == 0)
      return nullptr;
    return std::make_unique<QecDeviceCallSession>(entries, count);
  }
};

QecDeviceCallService g_service;
DeviceCallService *get_service() { return &g_service; }

} // namespace

extern "C" __attribute__((visibility("default")))
cudaq::realtime::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"cudaq-qec-realtime-device-call", &get_service};
}

extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_realtime_device_call_service_force_link() {}
