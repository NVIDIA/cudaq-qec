/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Realtime pipeline implementation.
//
// Implements the mapped ring buffer, the pipeline's host dispatcher thread
// (RX polling and per-worker CUDA graph launch), the GPU-only completion
// path, the CPU polling worker threads, and completion reporting: a consumer
// thread harvests the pipeline-owned (internal) ring, while a response on a
// caller-owned (external) ring is reported as soon as it is published.

#include "cudaq/qec/realtime/nvtx_helpers.h"
#include "cudaq/qec/realtime/pipeline.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda/std/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::qec::realtime::experimental {

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define PIPELINE_CUDA_CHECK(call)                                              \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "realtime_pipeline CUDA error: " << cudaGetErrorString(err) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Pin a thread to a specific CPU core when requested.
static void pin_thread(std::thread &t, int core) {
  if (core < 0)
    return;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// ---------------------------------------------------------------------------
// GPU-only mode: completion signaling via cudaLaunchHostFunc
// ---------------------------------------------------------------------------

// Per-worker state for GPU-only completion signaling. When no cpu_stage
// callback is installed, the dispatcher enqueues a host function on the
// worker's CUDA stream after the graph; it publishes the TX slot address of
// the originating slot and then returns the worker to the dispatcher.
// GPU-only mode runs on the internal ring only (start() rejects it with an
// external ring), so the flag store below is the whole publish step.
struct GpuOnlyWorkerCtx {
  volatile uint64_t *tx_flags_host;
  atomic_uint64_sys *idle_mask;
  int worker_id;
  int origin_slot;
  uint64_t tx_value;
};

// Host callback launched on the worker CUDA stream in GPU-only mode. It runs
// on a CUDA runtime thread after the graph and makes no CUDA calls. The idle
// bit is set last: the dispatcher rewrites origin_slot/tx_value only after
// it has acquired that bit again.
static void gpu_only_host_callback(void *user_data) {
  auto *ctx = static_cast<GpuOnlyWorkerCtx *>(user_data);
  const int slot = ctx->origin_slot;
  const uint64_t tx_value = ctx->tx_value;
  __atomic_store_n(&ctx->tx_flags_host[slot], tx_value, __ATOMIC_RELEASE);
  ctx->idle_mask->fetch_or(1ULL << ctx->worker_id,
                           cuda::std::memory_order_release);
}

// ---------------------------------------------------------------------------
// RingBufferManager
// ---------------------------------------------------------------------------

// Manage a pinned, GPU-mapped ring buffer for host-device communication.
//
// This allocates rx/tx flag arrays and a data region using cudaHostAllocMapped
// so both CPU and GPU can access them via mapped pointers. The ring is
// in-place: the TX view aliases the RX data, so a slot's response overwrites
// its request.
class RingBufferManager {
public:
  // Allocate a ring buffer with the given slot count and size.
  RingBufferManager(size_t num_slots, size_t slot_size)
      : num_slots_(num_slots), slot_size_(slot_size) {
    PIPELINE_CUDA_CHECK(cudaHostAlloc(
        &buf_rx_, num_slots * sizeof(atomic_uint64_sys), cudaHostAllocMapped));
    rx_flags_ = static_cast<atomic_uint64_sys *>(buf_rx_);
    for (size_t i = 0; i < num_slots; ++i)
      new (rx_flags_ + i) atomic_uint64_sys(0);

    PIPELINE_CUDA_CHECK(cudaHostAlloc(
        &buf_tx_, num_slots * sizeof(atomic_uint64_sys), cudaHostAllocMapped));
    tx_flags_ = static_cast<atomic_uint64_sys *>(buf_tx_);
    for (size_t i = 0; i < num_slots; ++i)
      new (tx_flags_ + i) atomic_uint64_sys(0);

    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&rx_flags_dev_), buf_rx_, 0));
    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&tx_flags_dev_), buf_tx_, 0));

    PIPELINE_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&rx_data_host_),
                                      num_slots * slot_size,
                                      cudaHostAllocMapped));
    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&rx_data_dev_), rx_data_host_, 0));

    rb_.rx_flags = reinterpret_cast<volatile uint64_t *>(rx_flags_);
    rb_.tx_flags = reinterpret_cast<volatile uint64_t *>(tx_flags_);
    rb_.rx_data = rx_data_dev_;
    rb_.tx_data = rx_data_dev_;
    rb_.rx_stride_sz = slot_size;
    rb_.tx_stride_sz = slot_size;
    rb_.rx_flags_host = reinterpret_cast<volatile uint64_t *>(rx_flags_);
    rb_.tx_flags_host = reinterpret_cast<volatile uint64_t *>(tx_flags_);
    rb_.rx_data_host = rx_data_host_;
    rb_.tx_data_host = rx_data_host_;
  }

  ~RingBufferManager() {
    for (size_t i = 0; i < num_slots_; ++i) {
      rx_flags_[i].~atomic_uint64_sys();
      tx_flags_[i].~atomic_uint64_sys();
    }
    cudaFreeHost(buf_rx_);
    cudaFreeHost(buf_tx_);
    cudaFreeHost(rx_data_host_);
  }

  // Check whether a slot's rx_flag is zero and therefore available.
  bool slot_available(uint32_t slot) const {
    auto *flags = reinterpret_cast<const volatile uint64_t *>(rx_flags_);
    return __atomic_load_n(&flags[slot], __ATOMIC_ACQUIRE) == 0;
  }

  // Write an RPC request into a slot and signal the dispatcher.
  void write_and_signal(uint32_t slot, uint32_t function_id,
                        const void *payload, uint32_t payload_len,
                        uint32_t request_id = 0, uint64_t ptp_timestamp = 0) {
    cudaq_host_ringbuffer_write_rpc_request(&rb_, slot, function_id, payload,
                                            payload_len, request_id,
                                            ptp_timestamp);
    cudaq_host_ringbuffer_signal_slot(&rb_, slot);
  }

  // Poll the TX flag for a slot to check completion status.
  cudaq_tx_status_t poll_tx(uint32_t slot, int *cuda_error) const {
    return cudaq_host_ringbuffer_poll_tx_flag(&rb_, slot, cuda_error);
  }

  // Clear a slot's tx flag after its response has been consumed.
  void clear_slot(uint32_t slot) {
    cudaq_host_ringbuffer_clear_slot(&rb_, slot);
  }

  // Return the number of slots.
  size_t num_slots() const { return num_slots_; }
  // Return the slot size in bytes.
  size_t slot_size() const { return slot_size_; }

  // Return the host-side RX flag array.
  atomic_uint64_sys *rx_flags() { return rx_flags_; }
  // Return the host-side TX flag array.
  atomic_uint64_sys *tx_flags() { return tx_flags_; }
  // Return the host-mapped RX data base pointer.
  uint8_t *rx_data_host() { return rx_data_host_; }
  // Return the device-mapped RX data base pointer.
  uint8_t *rx_data_dev() { return rx_data_dev_; }
  // Return a const reference to the underlying cudaq_ringbuffer_t.
  const cudaq_ringbuffer_t &ringbuffer() const { return rb_; }

private:
  size_t num_slots_;
  size_t slot_size_;
  void *buf_rx_ = nullptr;
  void *buf_tx_ = nullptr;
  atomic_uint64_sys *rx_flags_ = nullptr;
  atomic_uint64_sys *tx_flags_ = nullptr;
  uint64_t *rx_flags_dev_ = nullptr;
  uint64_t *tx_flags_dev_ = nullptr;
  uint8_t *rx_data_host_ = nullptr;
  uint8_t *rx_data_dev_ = nullptr;
  cudaq_ringbuffer_t rb_{};
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

// PIMPL implementation backing realtime_pipeline. Owns the ring buffer views,
// the dispatcher state, worker resources, completion accounting, and thread
// lifecycle machinery hidden from the public header.
struct realtime_pipeline::Impl {
  pipeline_stage_config config;

  gpu_stage_factory gpu_factory;
  cpu_stage_callback cpu_stage;
  completion_callback completion_handler;

  // Owned ring buffer (nullptr when using an external ring)
  std::unique_ptr<RingBufferManager> ring;

  // Active ring buffer: either a copy of ring->ringbuffer() or the
  // caller-supplied external ring buffer. The pipeline itself touches only
  // the host views (rx_flags_host, tx_flags_host, rx_data_host,
  // tx_data_host); the device view of a slot is handed to the pre/post-launch
  // hooks only.
  cudaq_ringbuffer_t active_rb_{};
  bool external_ring_ = false;

  // Dispatcher state
  atomic_int_sys shutdown_flag{0};
  // Requests handed to a worker (launch attempted); drops are not counted.
  atomic_uint64_sys live_dispatched{0};
  // Bit w set == worker w is free. Only the dispatcher clears bits. A bit is
  // set again by the worker thread (CPU-stage mode), by the GPU-only host
  // callback, or by the dispatcher itself after a failed launch in GPU-only
  // mode.
  atomic_uint64_sys idle_mask{0};
  // Slot each worker is serving: release-stored by the dispatcher before the
  // launch, acquire-loaded by the worker thread.
  std::vector<int> inflight_slot_tags;
  // Set by the dispatcher when cudaGraphLaunch failed for a worker. The
  // worker thread consumes it and releases its own idle bit, so in CPU-stage
  // mode that thread is the only one to set its bit. A worker reads its tag
  // only after observing its idle bit clear, and the dispatcher stores the tag
  // before clearing the bit, so the tag is stable until the worker releases
  // the bit itself.
  std::vector<uint8_t> worker_launch_failed;
  // cudaError_t of a failed launch, per slot: written by the dispatcher
  // before it publishes the slot, read and reset by whoever reports the
  // completion.
  std::vector<int> slot_cuda_error;
  // Device the factory's streams and graphs belong to: the device current on
  // the thread that called start(); selected in the dispatcher and worker
  // threads.
  int device_id = 0;

  // Per-worker GPU resources (from factory)
  std::vector<gpu_worker_resources> worker_resources;

  // GPU-only mode state
  bool gpu_only = false;
  std::vector<GpuOnlyWorkerCtx> gpu_only_ctxs;

  // Slot-to-request mapping. On the internal ring the injector fills
  // slot_request/slot_occupied before it signals the slot and the consumer
  // clears slot_occupied. On an external ring the dispatcher records the
  // header's request id and takes slot_occupied at dispatch, and publish()
  // clears it: the slot belongs to the pipeline in between (see
  // dispatcher_loop). slot_ptp holds the header's PTP timestamp for both
  // rings; stages get it and the request id through cpu_stage_context rather
  // than re-reading the RX slot.
  std::vector<uint64_t> slot_request;
  std::vector<uint64_t> slot_ptp;
  std::vector<uint8_t> slot_occupied;

  // Stats (atomic counters)
  std::atomic<uint64_t> total_submitted{0};
  std::atomic<uint64_t> total_completed{0};
  std::atomic<uint64_t> backpressure_stalls{0};

  // Thread coordination
  std::atomic<bool> producer_stop{false};
  std::atomic<bool> consumer_stop{false};

  // Threads
  std::thread dispatcher_thread;
  std::thread consumer_thread;
  std::vector<std::thread> worker_threads;

  std::atomic<bool> started{false};
  // Set by stop_all(): the stop flags are never reset, so a pipeline is
  // single use and start() refuses to run again.
  std::atomic<bool> stopped{false};

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  // Bind the active ring buffer (owned or external) and size the per-slot
  // and per-worker bookkeeping.
  void allocate(const pipeline_stage_config &cfg) {
    if (cfg.num_workers > 64) {
      throw std::invalid_argument("num_workers (" +
                                  std::to_string(cfg.num_workers) +
                                  ") exceeds idle_mask capacity of 64");
    }

    config = cfg;

    if (cfg.external_ringbuffer) {
      active_rb_ = *static_cast<cudaq_ringbuffer_t *>(cfg.external_ringbuffer);
      external_ring_ = true;
    } else {
      ring = std::make_unique<RingBufferManager>(
          static_cast<size_t>(cfg.num_slots), cfg.slot_size);
      active_rb_ = ring->ringbuffer();
    }

    inflight_slot_tags.resize(cfg.num_workers, 0);
    worker_launch_failed.resize(cfg.num_workers, 0);
    slot_cuda_error.resize(cfg.num_slots, 0);
    slot_request.resize(cfg.num_slots, 0);
    slot_ptp.resize(cfg.num_slots, 0);
    slot_occupied.resize(cfg.num_slots, 0);
  }

  // Build worker resources and spawn the dispatcher, worker, and consumer
  // threads.
  void start_threads() {
    if (!gpu_factory) {
      throw std::logic_error("gpu_factory must be set before calling start()");
    }

    const int nw = config.num_workers;
    gpu_only = !cpu_stage;

    // In GPU-only mode the graph and the post-launch hook own the TX slot
    // content: nothing frames an RPCResponse in it. A transport behind an
    // external ring would send that unframed slot, so the combination is
    // rejected.
    if (gpu_only && external_ring_) {
      throw std::logic_error(
          "GPU-only mode (no cpu_stage) is not supported with an external "
          "ring buffer: nothing frames an RPCResponse in the TX slot; "
          "provide a cpu_stage callback");
    }

    // Build GPU resources via user factory
    worker_resources.resize(nw);
    for (int i = 0; i < nw; ++i)
      worker_resources[i] = gpu_factory(i);

    // The factory's streams and graphs belong to the device current on this
    // thread; the dispatcher and worker threads select it before using them.
    PIPELINE_CUDA_CHECK(cudaGetDevice(&device_id));

    worker_launch_failed.assign(nw, 0);
    slot_cuda_error.assign(config.num_slots, 0);

    // In GPU-only mode, set up per-worker contexts for the cudaLaunchHostFunc
    // completion callback.
    if (gpu_only) {
      gpu_only_ctxs.resize(nw);
      for (int i = 0; i < nw; ++i) {
        auto &c = gpu_only_ctxs[i];
        c.tx_flags_host = active_rb_.tx_flags_host;
        c.idle_mask = &idle_mask;
        c.worker_id = i;
        c.origin_slot = 0;
        c.tx_value = 0;
      }
    }

    // Initialize idle_mask with all workers free
    uint64_t initial_idle = (nw >= 64) ? ~0ULL : ((1ULL << nw) - 1);
    idle_mask.store(initial_idle, cuda::std::memory_order_release);

    // --- Dispatcher thread ---
    dispatcher_thread = std::thread([this]() { dispatcher_loop(); });
    pin_thread(dispatcher_thread, config.cores.dispatcher);

    // --- Worker threads (skipped in GPU-only mode) ---
    if (!gpu_only) {
      worker_threads.resize(nw);
      for (int i = 0; i < nw; ++i) {
        worker_threads[i] = std::thread([this, i]() { worker_loop(i); });
        int core =
            (config.cores.worker_base >= 0) ? config.cores.worker_base + i : -1;
        pin_thread(worker_threads[i], core);
      }
    }

    // --- Consumer thread (internal ring only) ---
    // On an external ring the transport consumes tx_flags and the completion
    // is reported at publish time, so there is nothing to harvest.
    if (!external_ring_) {
      consumer_thread = std::thread([this]() { consumer_loop(); });
      pin_thread(consumer_thread, config.cores.consumer);
    }

    started = true;
  }

  /// @brief Stop the dispatcher, consumer, and worker threads in dependency
  /// order.
  void stop_all() {
    if (!started)
      return;

    producer_stop.store(true, std::memory_order_release);

    // Grace period for in-flight requests. Only the internal ring can drain
    // by count: on an external ring the transport owns the producer side and
    // total_submitted is never incremented.
    if (!external_ring_) {
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (total_completed.load(std::memory_order_relaxed) <
                 total_submitted.load(std::memory_order_relaxed) &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    // Dispatcher first: on exit it synchronizes every worker stream, so all
    // launched graphs have run and every GPU-only host callback has
    // published.
    shutdown_flag.store(1, cuda::std::memory_order_release);
    dispatcher_thread.join();

    consumer_stop.store(true, std::memory_order_release);
    if (consumer_thread.joinable())
      consumer_thread.join();

    // Workers check consumer_stop between cpu_stage calls; the user's stage
    // must return (0 when nothing is ready) rather than block.
    for (auto &t : worker_threads) {
      if (t.joinable())
        t.join();
    }

    started = false;
    stopped = true;
  }

  /// @brief Release the owned ring buffer after shutdown.
  void free_resources() { ring.reset(); }

  // -----------------------------------------------------------------------
  // Publishing and completion
  // -----------------------------------------------------------------------

  // Host address of the TX slot for `slot` (the RX slot itself on the
  // in-place internal ring).
  uint8_t *tx_slot_host(int slot) const {
    return active_rb_.tx_data_host +
           static_cast<size_t>(slot) * active_rb_.tx_stride_sz;
  }

  // Report the completion of `slot` to the completion handler and the
  // counters. The slot's recorded launch error, if any, is consumed here.
  void report_completion(int slot) {
    const int err = slot_cuda_error[slot];
    slot_cuda_error[slot] = 0;
    if (completion_handler) {
      completion c;
      c.request_id = slot_request[slot];
      c.slot = slot;
      c.success = (err == 0);
      c.cuda_error = err;
      completion_handler(c);
    }
    total_completed.fetch_add(1, std::memory_order_relaxed);
  }

  // Publish the response held at `tx_host` (the TX slot of `slot`): a release
  // store of the TX slot host address into tx_flags_host[slot]. On the
  // internal ring the consumer thread picks the flag up, reports the
  // completion and clears the flag. On an external ring the transport's TX
  // kernel consumes and zeroes the flag itself, so the pipeline never polls
  // or clears it; the completion is reported right here, on the publishing
  // thread (a worker thread, the complete_deferred() caller, or the
  // dispatcher's launch-failure path), and the RX slot, owned by the
  // pipeline since dispatch, is handed back to the transport.
  void publish(int slot, uint8_t *tx_host) {
    __atomic_store_n(&active_rb_.tx_flags_host[slot],
                     reinterpret_cast<uint64_t>(tx_host), __ATOMIC_RELEASE);
    if (external_ring_) {
      // Report while slot_request still describes this request. Then the
      // rx_flag, so the transport may publish the next frame into the page,
      // and only then the occupancy the dispatcher waits on, so it never
      // takes the stale flag value for a new frame.
      report_completion(slot);
      __atomic_store_n(&active_rb_.rx_flags_host[slot], 0, __ATOMIC_RELEASE);
      __atomic_store_n(&slot_occupied[slot], 0, __ATOMIC_RELEASE);
    }
  }

  // -----------------------------------------------------------------------
  // Dispatcher loop
  // -----------------------------------------------------------------------

  /// @brief Host dispatcher: poll the ring and launch worker graphs.
  /// @details Walks the ring in slot order. A non-zero rx_flag is the host
  /// address of a slot holding an RPCHeader. The request is handed to a free
  /// worker that serves its function id (pre-launch hook, cudaGraphLaunch,
  /// post-launch hook, all on the worker's stream) and the cursor advances.
  /// On the internal ring the rx_flag is cleared once the launch is
  /// enqueued; the injector's slot_occupied keeps the slot until the
  /// response is harvested. On an external ring the slot is owned by the
  /// pipeline from dispatch until publish(): the rx_flag keeps holding the
  /// frame, which stops a transport that publishes only into a released page
  /// (the GpuRoce transceiver) from presenting the next lap's frame while
  /// this request is in flight, and so keeps the TX slot and the per-slot
  /// request state to one request at a time. The page bytes are not covered
  /// by that: the transceiver re-arms the receive as soon as it publishes a
  /// frame, so they are valid only until num_slots more frames have arrived.
  /// Everything the pipeline needs is therefore taken at dispatch (header
  /// fields, and the input by pre_launch_fn), and stages read request_id
  /// and ptp_timestamp from cpu_stage_context. Memory ordering mirrors the
  /// ring producers: acquire on rx_flags, release on the rx clear and on
  /// every tx publish.
  void dispatcher_loop() {
    cudaSetDevice(device_id);

    const cudaq_ringbuffer_t &rb = active_rb_;
    const uint32_t ns = static_cast<uint32_t>(config.num_slots);
    const int nw = config.num_workers;
    uint32_t cursor = 0;

    // Drop the request at the cursor: clear its rx_flag and move on. Drops
    // are not counted. On the internal ring the injector only emits valid
    // headers for known function ids, so a drop there is a caller bug and
    // the slot stays occupied.
    auto drop = [&]() {
      __atomic_store_n(&rb.rx_flags_host[cursor], 0, __ATOMIC_RELEASE);
      cursor = (cursor + 1) % ns;
    };

    while (shutdown_flag.load(cuda::std::memory_order_acquire) == 0) {
      if (external_ring_ &&
          __atomic_load_n(&slot_occupied[cursor], __ATOMIC_ACQUIRE)) {
        // Still owned by the request dispatched from it: the rx_flag holds
        // that frame until publish() releases the slot. Checked before the
        // flag, so a stale value is never taken for a new frame. Waiting
        // keeps dispatch in ring order and makes the ring depth the
        // in-flight window, as the injector's round robin does on the
        // internal ring.
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }
      const uint64_t rx =
          __atomic_load_n(&rb.rx_flags_host[cursor], __ATOMIC_ACQUIRE);
      if (rx == 0) {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }
      if (external_ring_ &&
          __atomic_load_n(&rb.tx_flags_host[cursor], __ATOMIC_ACQUIRE) != 0) {
        // The transport has not taken the previous response out of this TX
        // slot yet; the next request must not overwrite it.
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }

      uint8_t *slot_host = reinterpret_cast<uint8_t *>(rx);
      const auto *hdr = reinterpret_cast<const RPCHeader *>(slot_host);
      if (hdr->magic != RPC_MAGIC_REQUEST) {
        drop();
        continue;
      }

      // Copy the header fields now: on the in-place internal ring the
      // response overwrites them.
      const uint32_t fid = hdr->function_id;
      const uint32_t rid = hdr->request_id;
      const uint64_t ptp = hdr->ptp_timestamp;

      bool known = false;
      for (int i = 0; i < nw && !known; ++i)
        known = (worker_resources[i].function_id == fid);
      if (!known) {
        drop();
        continue;
      }

      // Pick the lowest free worker serving this function id. None free:
      // backpressure. Stay on this slot with its rx_flag untouched and
      // re-check shutdown.
      int w = -1;
      for (uint64_t mask = idle_mask.load(cuda::std::memory_order_acquire);
           mask != 0; mask &= mask - 1) {
        const int b = __builtin_ctzll(mask);
        if (worker_resources[b].function_id == fid) {
          w = b;
          break;
        }
      }
      if (w < 0) {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }

      const uint64_t bit = 1ULL << w;
      slot_ptp[cursor] = ptp;
      if (external_ring_) {
        // Take the slot; publish() gives it back (see the class comment on
        // ownership above).
        slot_request[cursor] = rid;
        __atomic_store_n(&slot_occupied[cursor], 1, __ATOMIC_RELEASE);
      }
      // Store the tag now; the worker is taken (idle bit cleared) only after
      // the launch is enqueued and, on the internal ring, the RX flag is
      // cleared, see below.
      __atomic_store_n(&inflight_slot_tags[w], static_cast<int>(cursor),
                       __ATOMIC_RELEASE);

      uint8_t *slot_dev = rb.rx_data + (slot_host - rb.rx_data_host);
      uint8_t *tx_host = tx_slot_host(static_cast<int>(cursor));
      gpu_worker_resources &wr = worker_resources[w];

      NVTX_PUSH("Dispatch");
      if (wr.pre_launch_fn)
        wr.pre_launch_fn(wr.pre_launch_data, slot_dev, wr.stream);
      const cudaError_t err = cudaGraphLaunch(wr.graph_exec, wr.stream);
      if (err == cudaSuccess && wr.post_launch_fn)
        wr.post_launch_fn(wr.post_launch_data, slot_dev, wr.stream);
      NVTX_POP();

      live_dispatched.fetch_add(1, cuda::std::memory_order_relaxed);

      // Internal ring: release the RX flag once the launch work is enqueued,
      // and only then take the worker: nothing can complete this slot (and
      // let the injector refill it) before its rx_flag is cleared, so a
      // follow-up request in the same slot never loses its rx_flag to this
      // clear. External ring: the flag stays set until publish() releases
      // the slot.
      if (!external_ring_)
        __atomic_store_n(&rb.rx_flags_host[cursor], 0, __ATOMIC_RELEASE);
      idle_mask.fetch_and(~bit, cuda::std::memory_order_acq_rel);

      if (err != cudaSuccess) {
        // Launch failed: nothing runs on the stream for this request. Frame
        // an error response in the TX slot, record the error for the
        // completion callback, publish, and free the worker.
        (void)cudaGetLastError();
        RPCResponse resp;
        resp.magic = RPC_MAGIC_RESPONSE;
        resp.status = static_cast<int32_t>(err);
        resp.result_len = 0;
        resp.request_id = rid;
        resp.ptp_timestamp = ptp;
        std::memcpy(tx_host, &resp, sizeof(resp));
        slot_cuda_error[cursor] = static_cast<int>(err);
        publish(static_cast<int>(cursor), tx_host);
        if (gpu_only)
          idle_mask.fetch_or(bit, cuda::std::memory_order_release);
        else
          __atomic_store_n(&worker_launch_failed[w], 1, __ATOMIC_RELEASE);
      } else if (gpu_only) {
        // The callback publishes the TX slot address after the graph and
        // frees the worker. If it cannot be enqueued, wait for the graph
        // here and run it inline.
        GpuOnlyWorkerCtx &ctx = gpu_only_ctxs[w];
        ctx.origin_slot = static_cast<int>(cursor);
        ctx.tx_value = reinterpret_cast<uint64_t>(tx_host);
        if (cudaLaunchHostFunc(wr.stream, gpu_only_host_callback, &ctx) !=
            cudaSuccess) {
          (void)cudaGetLastError();
          cudaStreamSynchronize(wr.stream);
          gpu_only_host_callback(&ctx);
        }
      }

      cursor = (cursor + 1) % ns;
    }

    // Drain: once the worker streams are idle every launched graph and every
    // GPU-only host callback has run.
    for (auto &wr : worker_resources)
      cudaStreamSynchronize(wr.stream);
  }

  // -----------------------------------------------------------------------
  // Worker loop (one per worker thread)
  // -----------------------------------------------------------------------

  /// @brief Poll-mode CPU worker loop.
  /// @param worker_id Zero-based worker index.
  /// @details Each turn first consumes a launch failure flagged by the
  /// dispatcher (release the idle bit; there is nothing to process), then
  /// polls the user-supplied cpu_stage callback for the slot the worker is
  /// serving. The callback polls its own GPU-ready mechanism and, when a
  /// result is available, writes the RPC response into response_buffer (the
  /// slot's TX area) and returns the byte count; the pipeline publishes it
  /// and returns the worker to the dispatcher.
  void worker_loop(int worker_id) {
    cudaSetDevice(device_id);

    auto *wr = &worker_resources[worker_id];
    const uint64_t bit = 1ULL << worker_id;

    while (!consumer_stop.load(std::memory_order_relaxed)) {
      if (__atomic_exchange_n(&worker_launch_failed[worker_id], 0,
                              __ATOMIC_ACQ_REL)) {
        idle_mask.fetch_or(bit, cuda::std::memory_order_release);
        continue;
      }

      // Idle bit still set: no request assigned to this worker yet.
      if (idle_mask.load(cuda::std::memory_order_acquire) & bit) {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }

      const int origin =
          __atomic_load_n(&inflight_slot_tags[worker_id], __ATOMIC_ACQUIRE);
      uint8_t *tx_host = tx_slot_host(origin);

      cpu_stage_context ctx;
      ctx.worker_id = worker_id;
      ctx.origin_slot = origin;
      ctx.request_id = slot_request[origin];
      ctx.ptp_timestamp = slot_ptp[origin];
      ctx.gpu_output = nullptr;
      ctx.gpu_output_size = 0;
      ctx.response_buffer = tx_host;
      ctx.max_response_size = active_rb_.tx_stride_sz;
      ctx.user_context = wr->user_context;

      NVTX_PUSH("WorkerPoll");
      size_t written = cpu_stage(ctx);
      NVTX_POP();
      if (written == 0) {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }

      if (written == DEFERRED_COMPLETION) {
        // The slot stays in flight until complete_deferred(origin).
        idle_mask.fetch_or(bit, cuda::std::memory_order_release);
        continue;
      }

      publish(origin, tx_host);
      idle_mask.fetch_or(bit, cuda::std::memory_order_release);
    }
  }

  // -----------------------------------------------------------------------
  // Consumer loop (internal ring only)
  // -----------------------------------------------------------------------

  /// @brief Harvest completed slots of the internal ring and report them.
  /// @details The consumer owns slot completion accounting and the ordering
  /// required when clearing @c slot_occupied before resetting the shared TX
  /// flag on ARM and x86 hosts.
  void consumer_loop() {
    const uint32_t ns = static_cast<uint32_t>(config.num_slots);

    while (true) {
      if (consumer_stop.load(std::memory_order_acquire))
        break;

      bool pdone = producer_stop.load(std::memory_order_acquire);
      uint64_t nsub = total_submitted.load(std::memory_order_acquire);
      uint64_t ncomp = total_completed.load(std::memory_order_relaxed);
      if (pdone && ncomp >= nsub)
        break;

      bool found_any = false;
      for (uint32_t s = 0; s < ns; ++s) {
        if (!__atomic_load_n(&slot_occupied[s], __ATOMIC_ACQUIRE))
          continue;

        // A non-zero tx_flag is the TX slot address published for the
        // response; poll_tx_flag acquires it.
        int cuda_error = 0;
        if (cudaq_host_ringbuffer_poll_tx_flag(&active_rb_, s, &cuda_error) !=
            CUDAQ_TX_READY)
          continue;

        NVTX_PUSH("ConsumerComplete");
        report_completion(static_cast<int>(s));

        // Clear the TX flag before freeing the slot: once slot_occupied
        // reads 0 the injector may refill the slot and a worker may publish
        // the next response, which a late clear would wipe.
        cudaq_host_ringbuffer_clear_slot(&active_rb_, s);
        __atomic_store_n(&slot_occupied[s], 0, __ATOMIC_RELEASE);
        found_any = true;
        NVTX_POP();
      }

      if (!found_any)
        CUDAQ_REALTIME_CPU_RELAX();
    }
  }
};

// ---------------------------------------------------------------------------
// realtime_pipeline public API
// ---------------------------------------------------------------------------

// Construction eagerly binds the active ring buffer so callers can inspect
// ring addresses before start().
realtime_pipeline::realtime_pipeline(const pipeline_stage_config &config)
    : impl_(std::make_unique<Impl>()) {
  impl_->allocate(config);
}

// Destruction is shutdown-safe: any still-running threads are joined before
// the owned ring buffer is released.
realtime_pipeline::~realtime_pipeline() {
  if (impl_->started)
    impl_->stop_all();
  impl_->free_resources();
}

// The factory is stored and invoked later during start() so each worker can
// build stream-local graph state exactly once.
void realtime_pipeline::set_gpu_stage(gpu_stage_factory factory) {
  impl_->gpu_factory = std::move(factory);
}

// With a CPU stage installed, worker threads poll for GPU readiness, write
// the response and publish it. Without one the pipeline runs GPU-only and
// publishes from a host callback enqueued on the worker stream.
void realtime_pipeline::set_cpu_stage(cpu_stage_callback callback) {
  impl_->cpu_stage = std::move(callback);
}

// The completion handler runs on the consumer thread for the internal ring
// and on the publishing thread for an external ring.
void realtime_pipeline::set_completion_handler(completion_callback handler) {
  impl_->completion_handler = std::move(handler);
}

// Repeated calls after a successful start are ignored so callers can treat
// start() as idempotent during setup sequences.
void realtime_pipeline::start() {
  if (impl_->started)
    return;
  if (impl_->stopped)
    throw std::logic_error(
        "realtime_pipeline is single use: start() after stop() is not "
        "supported; construct a new pipeline");
  impl_->start_threads();
}

// stop() delegates to the internal shutdown path, which first allows
// in-flight requests to drain and then tears down the dispatcher, consumer
// and worker threads.
void realtime_pipeline::stop() { impl_->stop_all(); }

// The returned counters are sampled lock-free and may race with live updates,
// but each field is individually coherent.
realtime_pipeline::Stats realtime_pipeline::stats() const {
  return {impl_->total_submitted.load(std::memory_order_relaxed),
          impl_->total_completed.load(std::memory_order_relaxed),
          impl_->live_dispatched.load(cuda::std::memory_order_relaxed),
          impl_->backpressure_stalls.load(std::memory_order_relaxed)};
}

// In external-ring mode this exposes the caller-owned ring pointers; otherwise
// it returns the internally allocated mapped ring buffer bases.
realtime_pipeline::ring_buffer_bases
realtime_pipeline::ringbuffer_bases() const {
  return {impl_->active_rb_.rx_data_host, impl_->active_rb_.rx_data};
}

// Deferred completions publish the TX slot host address into tx_flags with
// release ordering so the reader of the flag also sees the response payload.
// Callable after stop(): the ring memory lives until the destructor.
void realtime_pipeline::complete_deferred(int slot) {
  impl_->publish(slot, impl_->tx_slot_host(slot));
}

// ---------------------------------------------------------------------------
// ring_buffer_injector
// ---------------------------------------------------------------------------

struct ring_buffer_injector::State {
  RingBufferManager *ring = nullptr;
  std::vector<uint64_t> *slot_request = nullptr;
  std::vector<uint8_t> *slot_occupied = nullptr;
  std::atomic<uint64_t> *total_submitted = nullptr;
  std::atomic<uint64_t> *backpressure_stalls = nullptr;
  std::atomic<bool> *producer_stop = nullptr;
  int num_slots = 0;
  std::atomic<uint32_t> next_slot{0};
};

// The injector captures pointers into the pipeline's submission and
// bookkeeping state so software tests can emulate FPGA DMA writes.
ring_buffer_injector realtime_pipeline::create_injector() {
  if (impl_->external_ring_) {
    throw std::logic_error(
        "create_injector() is not available with an external ring buffer; "
        "the FPGA/emulator owns the producer side");
  }
  auto s = std::make_unique<ring_buffer_injector::State>();
  s->ring = impl_->ring.get();
  s->slot_request = &impl_->slot_request;
  s->slot_occupied = &impl_->slot_occupied;
  s->total_submitted = &impl_->total_submitted;
  s->backpressure_stalls = &impl_->backpressure_stalls;
  s->producer_stop = &impl_->producer_stop;
  s->num_slots = impl_->config.num_slots;
  return ring_buffer_injector(std::move(s));
}

// Ownership of the shared injector state transfers with the move.
ring_buffer_injector::ring_buffer_injector(std::unique_ptr<State> s)
    : state_(std::move(s)) {}

// Destruction is trivial because the parent pipeline owns the ring buffer and
// completion bookkeeping.
ring_buffer_injector::~ring_buffer_injector() = default;
// Moving an injector transfers the submission cursor and shared state handle
// without touching the underlying ring buffer.
ring_buffer_injector::ring_buffer_injector(ring_buffer_injector &&) noexcept =
    default;
// Moving an injector transfers the submission cursor and shared state handle
// without touching the underlying ring buffer.
ring_buffer_injector &
ring_buffer_injector::operator=(ring_buffer_injector &&) noexcept = default;

// try_submit() attempts a single-slot claim using the shared round-robin
// cursor and returns immediately if the chosen slot is still occupied or
// another thread wins the cursor race. The cursor is a wrapping 32-bit
// counter reduced modulo num_slots, so a slot count that is not a power of
// two skips slots once every 2^32 submissions.
bool ring_buffer_injector::try_submit(uint32_t function_id, const void *payload,
                                      size_t payload_size,
                                      uint64_t request_id) {
  // Nothing services the ring once stop() has begun.
  if (state_->producer_stop->load(std::memory_order_acquire))
    return false;
  uint32_t cur = state_->next_slot.load(std::memory_order_relaxed);
  uint32_t slot = cur % static_cast<uint32_t>(state_->num_slots);
  if (__atomic_load_n(&(*state_->slot_occupied)[slot], __ATOMIC_ACQUIRE))
    return false;

  if (!state_->next_slot.compare_exchange_weak(
          cur, cur + 1, std::memory_order_acq_rel, std::memory_order_relaxed))
    return false;

  NVTX_PUSH("Submit");
  // Record the request before the slot is signalled: the consumer reads
  // slot_request as soon as it sees the response flag.
  (*state_->slot_request)[slot] = request_id;
  (*state_->slot_occupied)[slot] = 1;
  state_->ring->write_and_signal(slot, function_id, payload,
                                 static_cast<uint32_t>(payload_size),
                                 static_cast<uint32_t>(request_id));
  state_->total_submitted->fetch_add(1, std::memory_order_release);
  NVTX_POP();
  return true;
}

// submit() spin-waits with CUDAQ_REALTIME_CPU_RELAX until a slot becomes
// available or the producer stop flag is raised during shutdown.
void ring_buffer_injector::submit(uint32_t function_id, const void *payload,
                                  size_t payload_size, uint64_t request_id) {
  while (!try_submit(function_id, payload, payload_size, request_id)) {
    if (state_->producer_stop &&
        state_->producer_stop->load(std::memory_order_acquire))
      return;
    state_->backpressure_stalls->fetch_add(1, std::memory_order_relaxed);
    CUDAQ_REALTIME_CPU_RELAX();
  }
}

// This mirrors the pipeline-wide counter used for throughput and backpressure
// reporting.
uint64_t ring_buffer_injector::backpressure_stalls() const {
  return state_->backpressure_stalls->load(std::memory_order_relaxed);
}

} // namespace cudaq::qec::realtime::experimental
