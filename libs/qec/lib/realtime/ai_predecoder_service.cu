/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/realtime/nvtx_helpers.h"
#include <cstdlib>
#include <cuda/atomic>
#include <stdexcept>
#include <string>

#define SERVICE_CUDA_CHECK(call)                                               \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("CUDA Error in ai_predecoder_service: ") +               \
          cudaGetErrorString(err));                                            \
    }                                                                          \
  } while (0)

namespace cudaq::qec::realtime::experimental {

// System scope for NVLink/PCIe visibility to host (design: no
// __threadfence_system)
using atomic_int_sys = cuda::atomic<int, cuda::thread_scope_system>;

// =============================================================================
// Kernels (single slot 0 only; queue removed for host-side dynamic pool)
// =============================================================================

__global__ void predecoder_signal_ready_kernel(atomic_int_sys *d_ready_flags) {
  if (threadIdx.x == 0)
    d_ready_flags[0].store(1, cuda::std::memory_order_release);
}

__global__ void passthrough_copy_kernel(void *dst, const void *src,
                                        size_t num_bytes) {
  const uint4 *src4 = (const uint4 *)src;
  uint4 *dst4 = (uint4 *)dst;
  size_t n4 = num_bytes / sizeof(uint4);
  for (size_t i = threadIdx.x; i < n4; i += blockDim.x)
    dst4[i] = src4[i];

  size_t done = n4 * sizeof(uint4);
  for (size_t i = done + threadIdx.x; i < num_bytes; i += blockDim.x)
    ((char *)dst)[i] = ((const char *)src)[i];
}

// =============================================================================
// Class Implementation
// =============================================================================

static void
allocate_predecoder_buffers(atomic_int_sys *&h_ready_flags, void **&h_ring_ptrs,
                            void *&h_predecoder_outputs,
                            atomic_int_sys *&d_ready_flags, void **&d_ring_ptrs,
                            void *&d_predecoder_outputs, size_t output_size) {
  void *buf = nullptr;
  SERVICE_CUDA_CHECK(
      cudaHostAlloc(&buf, sizeof(atomic_int_sys), cudaHostAllocMapped));
  h_ready_flags = static_cast<atomic_int_sys *>(buf);
  new (h_ready_flags) atomic_int_sys(0);

  SERVICE_CUDA_CHECK(
      cudaHostAlloc(&h_ring_ptrs, sizeof(void *), cudaHostAllocMapped));
  SERVICE_CUDA_CHECK(
      cudaHostAlloc(&h_predecoder_outputs, output_size, cudaHostAllocMapped));

  SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_ready_flags,
                                              (void *)h_ready_flags, 0));
  SERVICE_CUDA_CHECK(
      cudaHostGetDevicePointer((void **)&d_ring_ptrs, (void *)h_ring_ptrs, 0));
  SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_predecoder_outputs,
                                              (void *)h_predecoder_outputs, 0));
}

ai_predecoder_service::ai_predecoder_service(
    const std::string &path, void **mailbox,
    int /* queue_depth (ignored; always 1) */,
    const std::string &engine_save_path,
    network_typing_override typing_override)
    : ai_decoder_service(path, mailbox, engine_save_path, typing_override),
      queue_depth_(1) {
  allocate_predecoder_buffers(
      h_ready_flags_, h_ring_ptrs_, h_predecoder_outputs_, d_ready_flags_,
      d_ring_ptrs_, d_predecoder_outputs_, get_output_size());
}

ai_predecoder_service::ai_predecoder_service(void **device_mailbox_slot,
                                             int queue_depth,
                                             size_t input_bytes,
                                             size_t output_bytes)
    : ai_decoder_service(device_mailbox_slot, input_bytes, output_bytes),
      queue_depth_(queue_depth) {
  allocate_predecoder_buffers(
      h_ready_flags_, h_ring_ptrs_, h_predecoder_outputs_, d_ready_flags_,
      d_ring_ptrs_, d_predecoder_outputs_, get_output_size());
}

std::unique_ptr<ai_predecoder_service>
ai_predecoder_service::create_passthrough(void **device_mailbox_slot,
                                          int queue_depth, size_t input_bytes,
                                          size_t output_bytes) {
  return std::unique_ptr<ai_predecoder_service>(new ai_predecoder_service(
      device_mailbox_slot, queue_depth, input_bytes, output_bytes));
}

ai_predecoder_service::~ai_predecoder_service() {
  if (h_ready_flags_) {
    h_ready_flags_[0].~atomic_int_sys();
    cudaFreeHost((void *)h_ready_flags_);
    h_ready_flags_ = nullptr;
    d_ready_flags_ = nullptr;
  }
  if (h_ring_ptrs_) {
    cudaFreeHost(h_ring_ptrs_);
    h_ring_ptrs_ = nullptr;
  }
  if (h_predecoder_outputs_) {
    cudaFreeHost(h_predecoder_outputs_);
    h_predecoder_outputs_ = nullptr;
  }
}

void ai_predecoder_service::capture_graph(cudaStream_t stream,
                                          bool device_launch) {
  bool has_trt = (context_ != nullptr);

  if (has_trt) {
    for (auto &b : all_bindings_) {
      context_->setTensorAddress(b.name.c_str(), b.d_buffer);
    }
    if (!context_->enqueueV3(stream))
      throw std::runtime_error(
          "TRT enqueueV3 warmup failed in ai_predecoder_service");
  }
  SERVICE_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph;
  SERVICE_CUDA_CHECK(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  if (!has_trt) {
    passthrough_copy_kernel<<<1, 256, 0, stream>>>(d_trt_output_, d_trt_input_,
                                                   get_input_size());
  } else {
    if (!context_->enqueueV3(stream))
      throw std::runtime_error(
          "TRT enqueueV3 failed during graph capture in ai_predecoder_service");
  }

  SERVICE_CUDA_CHECK(cudaMemcpyAsync(d_predecoder_outputs_, d_trt_output_,
                                     get_output_size(),
                                     cudaMemcpyDeviceToDevice, stream));

  predecoder_signal_ready_kernel<<<1, 1, 0, stream>>>(
      static_cast<atomic_int_sys *>(d_ready_flags_));

  SERVICE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  if (device_launch) {
    cudaError_t inst_err = cudaGraphInstantiateWithFlags(
        &graph_exec_, graph, cudaGraphInstantiateFlagDeviceLaunch);
    if (inst_err != cudaSuccess) {
      cudaGraphDestroy(graph);
      throw std::runtime_error(
          std::string("cudaGraphInstantiateWithFlags (DeviceLaunch) FAILED: ") +
          cudaGetErrorString(inst_err));
    }
    SERVICE_CUDA_CHECK(cudaGraphUpload(graph_exec_, stream));
  } else {
    cudaError_t inst_err = cudaGraphInstantiate(&graph_exec_, graph, 0);
    if (inst_err != cudaSuccess) {
      cudaGraphDestroy(graph);
      throw std::runtime_error(std::string("cudaGraphInstantiate FAILED: ") +
                               cudaGetErrorString(inst_err));
    }
  }

  cudaGraphDestroy(graph);
  SERVICE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool ai_predecoder_service::poll_next_job(pre_decoder_job &out_job) {
  auto *sys_flags = static_cast<atomic_int_sys *>(h_ready_flags_);
  int expected = 1;
  // Atomically claim: 1 (Ready) -> 2 (Processing) so we enqueue the job exactly
  // once. Use relaxed on failure so spinning doesn't add barriers that delay
  // seeing GPU's store(1).
  if (sys_flags[0].compare_exchange_strong(expected, 2,
                                           cuda::std::memory_order_acquire,
                                           cuda::std::memory_order_relaxed)) {
    NVTX_PUSH("PollJob");
    out_job.slot_idx = 0;
    out_job.ring_buffer_ptr = h_ring_ptrs_[0];
    out_job.inference_data = h_predecoder_outputs_;
    NVTX_POP();
    return true;
  }
  return false;
}

void ai_predecoder_service::release_job(int /* slot_idx */) {
  NVTX_PUSH("ReleaseJob");
  auto *sys_flags = static_cast<atomic_int_sys *>(h_ready_flags_);
  // PyMatching done: 2 (Processing) -> 0 (Idle)
  sys_flags[0].store(0, cuda::std::memory_order_release);
  NVTX_POP();
}

} // namespace cudaq::qec::realtime::experimental
