/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file pipeline.h
/// @brief Public API for the realtime decoding pipeline.
///
/// Provides configuration structs, callback types, a software ring buffer
/// injector, and the @p realtime_pipeline class that orchestrates GPU
/// inference and CPU post-processing for low-latency QEC decoding.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>

namespace cudaq::qec::realtime::experimental {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// @brief CPU core affinity settings for pipeline threads.
struct core_pinning {
  /// @brief Core for the host dispatcher thread. -1 disables pinning.
  int dispatcher = -1;
  /// @brief Core for the consumer (completion) thread. -1 disables pinning.
  int consumer = -1;
  /// @brief Base core for worker threads. Workers pin to base, base+1, etc.
  /// -1 disables pinning.
  int worker_base = -1;
};

/// @brief Configuration for a single pipeline stage.
struct pipeline_stage_config {
  /// @brief Number of GPU worker threads (max 64).
  int num_workers = 8;
  /// @brief Number of ring buffer slots.
  int num_slots = 32;
  /// @brief Size of each ring buffer slot in bytes.
  size_t slot_size = 16384;
  /// @brief CPU core affinity settings.
  core_pinning cores;

  /// @brief When non-null, the pipeline uses this caller-owned ring buffer
  /// (cudaq_ringbuffer_t*) instead of allocating its own. The caller is
  /// responsible for lifetime. ring_buffer_injector is unavailable in
  /// this mode (the FPGA/emulator owns the producer side).
  void *external_ringbuffer = nullptr;
};

// ---------------------------------------------------------------------------
// GPU Stage Factory
// ---------------------------------------------------------------------------

/// @brief Per-worker GPU resources returned by the gpu_stage_factory.
///
/// Each worker owns a captured CUDA graph, a dedicated stream, and optional
/// pre/post launch callbacks for DMA staging or result extraction.
struct gpu_worker_resources {
  /// @brief Instantiated CUDA graph for this worker.
  cudaGraphExec_t graph_exec = nullptr;
  /// @brief Dedicated CUDA stream for graph launches.
  cudaStream_t stream = nullptr;
  /// @brief Optional callback invoked before graph launch (e.g. DMA copy).
  void (*pre_launch_fn)(void *user_data, void *slot_dev,
                        cudaStream_t stream) = nullptr;
  /// @brief Opaque user data passed to @p pre_launch_fn.
  void *pre_launch_data = nullptr;
  /// @brief Optional callback invoked after graph launch.
  void (*post_launch_fn)(void *user_data, void *slot_dev,
                         cudaStream_t stream) = nullptr;
  /// @brief Opaque user data passed to @p post_launch_fn.
  void *post_launch_data = nullptr;
  /// @brief RPC function ID that this worker handles.
  uint32_t function_id = 0;
  /// @brief Opaque user context passed to cpu_stage_callback.
  void *user_context = nullptr;
};

/// @brief Factory called once per worker during start().
/// @param worker_id Zero-based worker index assigned by the pipeline.
/// @return GPU resources for the given worker. Any handles, callbacks, and
/// user data returned here must remain valid until the pipeline stops.
using gpu_stage_factory = std::function<gpu_worker_resources(int worker_id)>;

// ---------------------------------------------------------------------------
// CPU Stage Callback
// ---------------------------------------------------------------------------

/// @brief Context passed to the CPU stage callback for each completed GPU
/// workload.
///
/// The callback reads @p gpu_output, performs post-processing (e.g. MWPM
/// decoding), and writes the result into @p response_buffer.
struct cpu_stage_context {
  /// @brief Index of the worker thread invoking this callback.
  int worker_id;
  /// @brief Ring buffer slot that originated this request.
  int origin_slot;
  /// @brief Pointer to GPU inference output (nullptr in poll mode).
  const void *gpu_output;
  /// @brief Size of GPU output in bytes.
  size_t gpu_output_size;
  /// @brief Destination buffer for the RPC response.
  void *response_buffer;
  /// @brief Maximum number of bytes that can be written to @p response_buffer.
  size_t max_response_size;
  /// @brief Opaque user context from gpu_worker_resources::user_context.
  void *user_context;
};

/// @brief CPU stage callback type.
/// @param ctx Poll-mode view of the current worker state and response buffer.
/// @return Number of bytes written into @p ctx.response_buffer.
/// Return 0 if no GPU result is ready yet (poll again).
/// Return DEFERRED_COMPLETION to release the worker immediately while
/// deferring slot completion to a later complete_deferred() call.
using cpu_stage_callback = std::function<size_t(const cpu_stage_context &ctx)>;

/// @brief Sentinel return value from cpu_stage_callback: release the worker
/// (idle_mask) but do NOT signal slot completion (tx_flags). The caller
/// is responsible for calling realtime_pipeline::complete_deferred(slot)
/// once the deferred work (e.g. a separate decode thread) finishes.
static constexpr size_t DEFERRED_COMPLETION = SIZE_MAX;

// ---------------------------------------------------------------------------
// Completion Callback
// ---------------------------------------------------------------------------

/// @brief Metadata for a completed (or errored) pipeline request.
struct completion {
  /// @brief Original request ID from the RPC header.
  uint64_t request_id;
  /// @brief Ring buffer slot that held this request.
  int slot;
  /// @brief True if the request completed without CUDA errors.
  bool success;
  /// @brief CUDA error code (0 on success).
  int cuda_error;
};

/// @brief Callback invoked by the consumer thread for each completed request.
/// @param c Metadata for the completed or errored request.
using completion_callback = std::function<void(const completion &c)>;

// ---------------------------------------------------------------------------
// Ring Buffer Injector (software-only test/replay data source)
// ---------------------------------------------------------------------------

/// @brief Writes RPC-framed requests into the pipeline's ring buffer,
/// simulating FPGA DMA deposits.
///
/// Created via realtime_pipeline::create_injector(). The parent
/// realtime_pipeline must outlive the injector. Not available when the
/// pipeline is configured with an external ring buffer.
class ring_buffer_injector {
public:
  /// @brief Destroy the injector state.
  ~ring_buffer_injector();
  /// @brief Move-construct an injector.
  ring_buffer_injector(ring_buffer_injector &&) noexcept;
  /// @brief Move-assign an injector.
  ring_buffer_injector &operator=(ring_buffer_injector &&) noexcept;

  ring_buffer_injector(const ring_buffer_injector &) = delete;
  ring_buffer_injector &operator=(const ring_buffer_injector &) = delete;

  /// @brief Try to submit a request without blocking.
  /// @param function_id RPC function identifier.
  /// @param payload Pointer to the payload data.
  /// @param payload_size Size of the payload in bytes.
  /// @param request_id Caller-assigned request identifier.
  /// @return True if accepted, false if all slots are busy (backpressure).
  bool try_submit(uint32_t function_id, const void *payload,
                  size_t payload_size, uint64_t request_id);

  /// @brief Submit a request, spinning until a slot becomes available.
  /// @param function_id RPC function identifier.
  /// @param payload Pointer to the payload data.
  /// @param payload_size Size of the payload in bytes.
  /// @param request_id Caller-assigned request identifier.
  void submit(uint32_t function_id, const void *payload, size_t payload_size,
              uint64_t request_id);

  /// @brief Return the cumulative number of backpressure stalls.
  /// @return Number of times submit() had to spin-wait for a free slot.
  uint64_t backpressure_stalls() const;

private:
  friend class realtime_pipeline;
  struct State;
  std::unique_ptr<State> state_;
  explicit ring_buffer_injector(std::unique_ptr<State> s);
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// @brief Orchestrates GPU inference and CPU post-processing for low-latency
/// realtime QEC decoding.
///
/// The pipeline manages a ring buffer, a host dispatcher thread, per-worker
/// GPU streams with captured CUDA graphs, optional CPU worker threads for
/// post-processing (e.g. PyMatching), and a consumer thread for completion
/// signaling. It supports both an internal ring buffer (for software testing
/// via ring_buffer_injector) and an external ring buffer (for FPGA RDMA).
class realtime_pipeline {
public:
  /// @brief Construct a pipeline and allocate ring buffer resources.
  /// @param config Stage configuration (slots, slot size, workers, etc.).
  /// @note Construction allocates the backing ring buffer or binds the
  /// caller-provided external ring so @ref ringbuffer_bases can be queried
  /// before @ref start.
  explicit realtime_pipeline(const pipeline_stage_config &config);
  /// @brief Stop the pipeline if needed and release owned resources.
  ~realtime_pipeline();

  realtime_pipeline(const realtime_pipeline &) = delete;
  realtime_pipeline &operator=(const realtime_pipeline &) = delete;

  /// @brief Register the GPU stage factory. Must be called before start().
  /// @param factory Callback that returns gpu_worker_resources per worker.
  void set_gpu_stage(gpu_stage_factory factory);

  /// @brief Register the CPU worker callback. Must be called before start().
  /// @param callback Function invoked by each worker thread to poll for and
  ///   process completed GPU workloads. If not set, the pipeline operates in
  ///   GPU-only mode with completion signaled via cudaLaunchHostFunc.
  void set_cpu_stage(cpu_stage_callback callback);

  /// @brief Register the completion callback. Must be called before start().
  /// @param handler Function invoked by the consumer thread for each
  ///   completed or errored request.
  void set_completion_handler(completion_callback handler);

  /// @brief Allocate resources, build dispatcher config, and spawn all threads.
  /// @throws std::logic_error If the GPU stage factory was not registered.
  /// @throws std::logic_error If GPU-only mode is requested with an external
  /// ring buffer.
  void start();

  /// @brief Signal shutdown, join all threads, free resources.
  /// @note Safe to call multiple times. Subsequent calls are no-ops once the
  /// pipeline has fully stopped.
  void stop();

  /// @brief Create a software injector for testing without FPGA hardware.
  /// @return A ring_buffer_injector bound to this pipeline's ring buffer.
  /// @throws std::logic_error if the pipeline uses an external ring buffer.
  ring_buffer_injector create_injector();

  /// @brief Pipeline throughput and backpressure statistics.
  struct Stats {
    /// @brief Total requests submitted to the ring buffer.
    uint64_t submitted;
    /// @brief Total requests that completed (success or error).
    uint64_t completed;
    /// @brief Total packets dispatched by the host dispatcher.
    uint64_t dispatched;
    /// @brief Cumulative producer backpressure stalls.
    uint64_t backpressure_stalls;
  };

  /// @brief Thread-safe, lock-free stats snapshot.
  /// @return Current pipeline statistics.
  Stats stats() const;

  /// @brief Signal that deferred processing for a slot is complete.
  ///
  /// Call from any thread after the cpu_stage callback returned
  /// DEFERRED_COMPLETION and the deferred work has finished writing the
  /// response into the slot's ring buffer area.
  /// @param slot Ring buffer slot index to complete.
  void complete_deferred(int slot);

  /// @brief Host and device base addresses of the RX data ring.
  struct ring_buffer_bases {
    /// @brief Host-mapped base pointer for the RX data ring.
    uint8_t *rx_data_host;
    /// @brief Device-mapped base pointer for the RX data ring.
    uint8_t *rx_data_dev;
  };

  /// @brief Return the host and device base addresses of the RX data ring.
  /// @return Struct containing both base pointers.
  /// @note In external-ring mode these pointers are the caller-provided ring
  /// addresses. In internal mode they refer to the owned mapped ring buffer.
  ring_buffer_bases ringbuffer_bases() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaq::qec::realtime::experimental
