/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include <atomic>
#include <cuda/atomic>

// Portable CPU Yield Macro for busy-polling (skip if already defined by
// realtime API)
#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  std::atomic_thread_fence(std::memory_order_seq_cst)
#endif
#endif

namespace cudaq::qec::realtime::experimental {

struct pre_decoder_job {
  int slot_idx;    ///< Worker/slot index (for release_job; always 0)
  int origin_slot; ///< FPGA ring slot for tx_flags routing (dynamic pool)
  void *ring_buffer_ptr;
  void *inference_data; ///< Points into the pinned output (single slot)

  // Performance Tracking
  uint64_t submit_ts_ns;
  uint64_t dispatch_ts_ns;
  uint64_t poll_ts_ns;
};

class ai_predecoder_service : public ai_decoder_service {
public:
  ai_predecoder_service(const std::string &engine_path,
                        void **device_mailbox_slot, int queue_depth = 1,
                        const std::string &engine_save_path = "",
                        network_typing_override typing_override =
                            network_typing_override::automatic);

  /// Create a passthrough (identity copy) instance for testing without TRT.
  static std::unique_ptr<ai_predecoder_service>
  create_passthrough(void **device_mailbox_slot, int queue_depth = 1,
                     size_t input_bytes = 1600 * sizeof(float),
                     size_t output_bytes = 1600 * sizeof(float));

  virtual ~ai_predecoder_service();

  void capture_graph(cudaStream_t stream, bool device_launch);
  void capture_graph(cudaStream_t stream) override {
    capture_graph(stream, true);
  }

  bool poll_next_job(pre_decoder_job &out_job);
  void release_job(int slot_idx);

  /// Stub for device-dispatcher batch path (returns nullptr; streaming uses
  /// host dispatcher)
  int *get_device_queue_idx() const { return nullptr; }
  cuda::atomic<int, cuda::thread_scope_system> *get_device_ready_flags() const {
    return d_ready_flags_;
  }
  int *get_device_inflight_flag() const { return nullptr; }

  cuda::atomic<int, cuda::thread_scope_system> *get_host_ready_flags() const {
    return h_ready_flags_;
  }
  volatile int *get_host_queue_idx() const { return nullptr; }
  int get_queue_depth() const { return queue_depth_; }

  void **get_host_ring_ptrs() const { return h_ring_ptrs_; }

private:
  /// Passthrough constructor (delegates to base passthrough constructor).
  ai_predecoder_service(void **device_mailbox_slot, int queue_depth,
                        size_t input_bytes, size_t output_bytes);

  int queue_depth_; // Always 1

  cuda::atomic<int, cuda::thread_scope_system> *h_ready_flags_ = nullptr;
  void **h_ring_ptrs_ = nullptr;
  void *h_predecoder_outputs_ = nullptr;

  cuda::atomic<int, cuda::thread_scope_system> *d_ready_flags_ = nullptr;
  void **d_ring_ptrs_ = nullptr;
  void *d_predecoder_outputs_ = nullptr;
};

} // namespace cudaq::qec::realtime::experimental
