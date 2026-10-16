/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder.h"

#include <cassert>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace cudaq::qec::detail_affinity {

/// Point the calling thread at \p target before work that allocates or
/// launches on it (no restore; set-if-different). No-op for target < 0.
/// Throws on failure: never silently decode on the wrong GPU.
inline void set_cuda_device_for_decode(int target) {
  if (target < 0)
    return;
  int current = -1;
  if (cudaGetDevice(&current) == cudaSuccess && current == target)
    return;
  cudaError_t err = cudaSetDevice(target);
  if (err != cudaSuccess)
    throw std::runtime_error("set_cuda_device_for_decode: cudaSetDevice(" +
                             std::to_string(target) +
                             ") failed: " + cudaGetErrorString(err));
}

/// Resolve a decoder's device: its cuda_device_id, or 0 when unpinned. For
/// paths that need a concrete device (graph capture/launch), unlike
/// set_cuda_device_for_decode() which no-ops on < 0.
inline int decode_device_for(int cuda_device_id) {
  return cuda_device_id >= 0 ? cuda_device_id : 0;
}

/// Pin before dispatch / decode / get_corrections / reset. No-op for an
/// unpinned decoder (cuda_device_id < 0): a CPU decoder must not be forced onto
/// a GPU. The sanctioned dispatch pin for every transport.
inline void pin_decode_device(const cudaq::qec::decoder &dec) {
  set_cuda_device_for_decode(dec.get_cuda_device_id());
}

/// Cached pin for dispatcher threads that call decoders inline on the hot
/// path: the first call per (thread, device) transition pays the CUDA call;
/// steady-state repeats cost one thread_local compare.  Debug builds
/// re-verify against cudaGetDevice to catch a foreign cudaSetDevice on the
/// same thread (the cache assumes none).  No-op for unpinned decoders
/// (cuda_device_id < 0), same as pin_decode_device().
inline void pin_decode_device_cached(const cudaq::qec::decoder &dec) {
  const int target = dec.get_cuda_device_id();
  if (target < 0)
    return;
  thread_local int pinned = -1;
  if (pinned != target) {
    set_cuda_device_for_decode(target);
    pinned = target;
  }
#ifndef NDEBUG
  int current = -1;
  if (cudaGetDevice(&current) == cudaSuccess)
    assert(current == target &&
           "pin_decode_device_cached: a foreign cudaSetDevice moved this "
           "thread off the decoder's device");
#endif
}

/// Capture a decoder's realtime graph, pinned to its device so capture lands on
/// the GPU every launch uses. Unpinned resolves to device 0 (a graph needs a
/// concrete device). The only sanctioned caller of capture_decode_graph().
inline void *capture_graph_pinned(cudaq::qec::decoder &dec,
                                  int reserved_sms = 0) {
  const int device = decode_device_for(dec.get_cuda_device_id());
  set_cuda_device_for_decode(device);
  void *raw = dec.capture_decode_graph(reserved_sms);
#ifndef NDEBUG
  int current = -1;
  (void)cudaGetDevice(&current);
  assert(current == device &&
         "capture_graph_pinned: capture did not land on the decoder's device");
#endif
  return raw;
}

/// RAII: set the calling thread's CUDA device, restore the previous device on
/// scope exit. No-op for target < 0. Lib-private and header-only so decoder
/// plugins built as separate .so files can reuse it (PR2 extends this header
/// with NUMA guards; the nv-qldpc follow-up mirrors its use).
///
/// This guard is for threads that do NOT follow the one-thread-owns-one-
/// decoder persistent pin (e.g. the fresh worker spawned by decode_async).
class CudaDeviceGuard {
public:
  explicit CudaDeviceGuard(int target) {
    if (target < 0)
      return;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || target >= count)
      throw std::runtime_error("cuda_device_id " + std::to_string(target) +
                               " is out of range: " + std::to_string(count) +
                               " CUDA device(s) visible");
    // If the current device is unreadable, skip restoration rather than
    // restore to a guessed device; the set below still applies.
    if (cudaGetDevice(&prev_) != cudaSuccess)
      prev_ = -1;
    cudaError_t err = cudaSetDevice(target);
    if (err != cudaSuccess)
      throw std::runtime_error("CudaDeviceGuard: cudaSetDevice(" +
                               std::to_string(target) +
                               ") failed: " + cudaGetErrorString(err));
    restore_ = (prev_ >= 0 && prev_ != target);
  }
  ~CudaDeviceGuard() {
    if (restore_)
      (void)cudaSetDevice(prev_);
  }
  CudaDeviceGuard(const CudaDeviceGuard &) = delete;
  CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
  int prev_ = -1;
  bool restore_ = false;
};

} // namespace cudaq::qec::detail_affinity
