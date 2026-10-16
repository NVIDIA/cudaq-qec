/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// For full test script: surface_code-5-per-decoder-rings-test.sh
//
// ONE RING BUFFER (AND ONE DISPATCHER) PER DECODER
// ================================================
//
// This example demonstrates the per-decoder data-plane topology: each logical
// qubit's decoder is fed by its OWN ring buffer with its OWN dispatcher, in
// one process driving one (simulated) QPU.
//
// The mechanism is `cudaq::device_call`'s device-id overload.  The QEC device
// wrappers (simulation_cqr_device.cpp) route every decoding RPC with
// `device_id == decoder_id`:
//
//   cudaq::device_call(decoder_id, ::enqueue_syndromes, decoder_id, ...);
//
// The `-frealtime-lowering` pass threads that device id into
// `__cudaq_device_call_acquire_realtime_frame(deviceId, functionId, ...)`,
// and the CUDA-Q device_call runtime keys EVERYTHING by device id: the first
// RPC for a new decoder id creates a dedicated session with its own ring
// buffer and its own dispatcher (see DeviceCallDispatch.cpp `sessions` map and
// HostDispatchChannel's per-instance RingBufferWrapper + dispatcher).  No
// shared ring, no head-of-line blocking between decoders, per-decoder
// backpressure.
//
// CPU-memory vs GPU-memory rings
// ------------------------------
// The channel selected for a device id decides where its ring lives:
//   host_dispatch    pinned+mapped host memory ring, HOST_CALL dispatcher
//                    thread on the CPU (this example; works everywhere).
//   device_dispatch  pinned+mapped ring polled by a persistent GPU dispatch
//                    kernel; the service session must supply
//                    DeviceCallDispatchMode::Gpu entries (DEVICE_CALL device
//                    function pointers) and a launch function.
// With the per-device channel override
// (CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch,1=device_dispatch), decoder 0 runs
// on a CPU-memory ring and decoder 1 on a GPU-dispatched ring simultaneously.
// This example runs both decoders on host_dispatch so it is runnable on any
// machine with a GPU; the test script exercises the override parsing.
//
// What the example does
// ---------------------
// Two pymatching decoders (id 0 and id 1) with 3-bit identity syndrome maps.
// One kernel enqueues a DIFFERENT syndrome to each decoder and reads each
// decoder's corrections back.  Expected: decoder 0 flips exactly bit 1;
// decoder 1 flips exactly bits 0 and 2.  The test script additionally asserts
// (via CUDA-Q runtime logs) that TWO device sessions were created -- i.e. two
// rings, two dispatchers.
//
// Run:
//   CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch ./surface_code-5-per-decoder-rings

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime.h"

#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();

namespace {

namespace config = cudaq::qec::decoding::config;

constexpr std::uint64_t kNumDecoders = 2;
constexpr std::uint64_t kBlockSize = 3;
constexpr std::uint64_t kSyndromeSize = 3;
constexpr double kUniformErrorRate = 0.1;
// Each decoder id gets its own ring + dispatcher; each RPC below crosses the
// ring belonging to its decoder.  3 RPCs per decoder (reset, enqueue, get).
constexpr std::uint64_t kExpectedDispatches = kNumDecoders * 3;

std::vector<std::int64_t> make_identity_sparse_matrix() {
  std::vector<std::int64_t> sparse;
  sparse.reserve(2 * kSyndromeSize);
  for (std::uint64_t column = 0; column < kSyndromeSize; ++column) {
    sparse.push_back(static_cast<std::int64_t>(column));
    sparse.push_back(-1);
  }
  return sparse;
}

config::multi_decoder_config make_config() {
  config::multi_decoder_config multi;
  const auto identity = make_identity_sparse_matrix();
  for (std::uint64_t id = 0; id < kNumDecoders; ++id) {
    config::decoder_config dc;
    dc.id = static_cast<std::int64_t>(id);
    dc.type = "pymatching";
    dc.dispatch = config::DecoderDispatch::host;
    dc.block_size = kBlockSize;
    dc.syndrome_size = kSyndromeSize;
    dc.H_sparse = identity;
    dc.O_sparse = identity;
    dc.D_sparse = identity;
    dc.error_rate_vec = std::vector<double>(kBlockSize, kUniformErrorRate);
    cudaqx::heterogeneous_map pm_args;
    pm_args.insert("merge_strategy", "smallest_weight");
    dc.decoder_custom_args = pm_args;
    multi.decoders.push_back(dc);
  }
  return multi;
}

} // namespace

// One QPU kernel drives BOTH decoders.  Each decoding RPC below travels over
// the ring belonging to its decoder id (device_id == decoder_id in the device
// wrappers).  Returns the corrections packed as
// bit i         = decoder 0 correction bit i
// bit (3 + i)   = decoder 1 correction bit i
__qpu__ std::int64_t two_decoder_kernel() {
  constexpr std::uint64_t kKernelSyndromeSize = 3;
  constexpr std::uint64_t kKernelBlockSize = 3;

  // Decoder 0: syndrome {0,1,0} -> expect correction exactly at bit 1.
  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/0);
  std::vector<bool> syndrome0(kKernelSyndromeSize);
  for (std::size_t i = 0; i < kKernelSyndromeSize; ++i)
    syndrome0[i] = false;
  syndrome0[1] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(/*decoder_id=*/0, syndrome0,
                                               /*tag=*/1);
  auto corrections0 = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/0, /*return_size=*/kKernelBlockSize, /*reset=*/true);

  // Decoder 1: syndrome {1,0,1} -> expect corrections exactly at bits 0 and 2.
  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/1);
  std::vector<bool> syndrome1(kKernelSyndromeSize);
  for (std::size_t i = 0; i < kKernelSyndromeSize; ++i)
    syndrome1[i] = false;
  syndrome1[0] = true;
  syndrome1[2] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(/*decoder_id=*/1, syndrome1,
                                               /*tag=*/2);
  auto corrections1 = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/1, /*return_size=*/kKernelBlockSize, /*reset=*/true);

  // Pack both decoders' correction bits with plain arithmetic (kernel-safe).
  std::int64_t packed = 0;
  if (corrections0[0])
    packed = packed + 1;
  if (corrections0[1])
    packed = packed + 2;
  if (corrections0[2])
    packed = packed + 4;
  if (corrections1[0])
    packed = packed + 8;
  if (corrections1[1])
    packed = packed + 16;
  if (corrections1[2])
    packed = packed + 32;
  return packed;
}

int main() {
  // Keep the service library loaded so CUDA-Q can discover its
  // cudaqGetDeviceCallServicePluginInfo symbol via dlsym(RTLD_DEFAULT).
  cudaqx_qec_realtime_device_call_service_force_link();

  auto decoder_config = make_config();
  if (config::configure_decoders(decoder_config) != 0) {
    std::fprintf(stderr, "FAILED: configure_decoders\n");
    return 1;
  }

  {
    int argc = 1;
    char program[] = "surface_code-5-per-decoder-rings";
    char *argv[] = {program, nullptr};
    cudaq::realtime::initialize(argc, argv);
  }

  const auto results = cudaq::run(1, two_decoder_kernel);

  // decoder 0 -> 0b000010 (bit 1); decoder 1 -> 0b101 << 3 (bits 3 and 5).
  constexpr std::int64_t kExpectedPacked = 0b101010;
  bool ok = results.size() == 1 && results[0] == kExpectedPacked;
  std::printf(
      "corrections packed = 0x%llx (expected 0x%llx) [%s]\n",
      static_cast<unsigned long long>(results.empty() ? -1 : results[0]),
      static_cast<unsigned long long>(kExpectedPacked), ok ? "OK" : "MISMATCH");

  // Self-verify the RPCs actually traversed the device_call rings (a correct
  // result alone does not prove this -- the direct trampolines would also
  // produce it).
  const auto dispatches = cudaqx_qec_device_call_dispatch_count();
  std::printf("device_call dispatches = %llu (expected %llu)\n",
              static_cast<unsigned long long>(dispatches),
              static_cast<unsigned long long>(kExpectedDispatches));
  if (dispatches < kExpectedDispatches)
    ok = false;

  cudaq::realtime::finalize();
  config::finalize_decoders();

  std::printf("%s\n", ok ? "PER-DECODER-RINGS PASSED" : "FAILED");
  return ok ? 0 : 1;
}
