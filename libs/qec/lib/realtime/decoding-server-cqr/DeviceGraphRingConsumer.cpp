/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

#include "DeviceGraphRingConsumer.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/graph_resources.h"

#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;

namespace {

// Allocate \p bytes of CUDA pinned+mapped host memory and return both the host
// pointer and its device-mapped counterpart.  The memory is zero-initialised.
bool alloc_pinned_mapped(size_t bytes, void **host_out, void **dev_out) {
  void *h = nullptr;
  if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  void *d = nullptr;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
    cudaFreeHost(h);
    return false;
  }
  std::memset(h, 0, bytes);
  *host_out = h;
  *dev_out = d;
  return true;
}

// Resolve a proprietary DEVICE_CALL populate shim via dlsym and stamp the
// function table entry.  The consuming process must absorb
// libcudaq-qec-realtime-cudevice-proprietary.a (WHOLE_ARCHIVE) and link with
// --export-dynamic so the symbols are visible.
using populate_fn = void (*)(void *);
bool populate_device_call(cudaq_function_entry_t &entry, const char *symbol,
                          uint32_t function_id) {
  auto fn = reinterpret_cast<populate_fn>(::dlsym(RTLD_DEFAULT, symbol));
  if (!fn) {
    CUDA_QEC_ERROR(
        "DeviceGraphRingConsumer: dlsym({}) failed -- the process must "
        "absorb libcudaq-qec-realtime-cudevice-proprietary.a as WHOLE_ARCHIVE "
        "and link with --export-dynamic",
        symbol);
    return false;
  }
  fn(&entry);
  entry.function_id = function_id;
  entry.routing_key = 0;
  if (entry.dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
      !entry.handler.device_fn_ptr) {
    CUDA_QEC_ERROR("DeviceGraphRingConsumer: {} did not produce a valid "
                   "DEVICE_CALL entry",
                   symbol);
    return false;
  }
  return true;
}

#define DGRC_CUDA_CHECK(expr)                                                  \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess)                                                   \
      throw std::runtime_error(                                                \
          std::string("DeviceGraphRingConsumer CUDA error: ") +                \
          cudaGetErrorString(_err) + " (" #expr ")");                          \
  } while (0)

} // namespace

DeviceGraphRingConsumer::DeviceGraphRingConsumer(const cudaq_ringbuffer_t &ring,
                                                 std::size_t num_slots,
                                                 std::size_t slot_size,
                                                 int gpu_id,
                                                 void *raw_graph_resources)
    : gpu_id_(gpu_id) {
  auto *graph_res =
      static_cast<cudaq::qec::realtime::graph_resources *>(raw_graph_resources);
  if (!graph_res || !graph_res->graph_exec)
    throw std::runtime_error(
        "DeviceGraphRingConsumer: null graph_exec "
        "(decoder must support_graph_dispatch() and capture_decode_graph())");
  if (!ring.rx_flags || !ring.tx_flags || !ring.rx_data || !ring.tx_data)
    throw std::runtime_error(
        "DeviceGraphRingConsumer: ring must carry device-visible pointers "
        "(GPU-pollable ring memory, e.g. DOCA rings or --pinned-rings)");

  DGRC_CUDA_CHECK(cudaSetDevice(gpu_id_));

  // Device-side graph launch (the scheduler's decode-graph trigger path) only
  // exists for __CUDA_ARCH__ >= 900; the dispatch kernel compiles it out on
  // older architectures. On such a GPU every host-side call below still
  // succeeds and the scheduler services the DEVICE_CALL RPCs, but the decode
  // graph can never fire and the decoder silently returns all-zero
  // corrections. Fail loudly instead.
  cudaDeviceProp props{};
  DGRC_CUDA_CHECK(cudaGetDeviceProperties(&props, gpu_id_));
  if (props.major < 9)
    throw std::runtime_error(
        "DeviceGraphRingConsumer: device_graph dispatch requires compute "
        "capability >= 9.0 for device-side graph launch; device " +
        std::to_string(gpu_id_) + " (" + props.name + ") is " +
        std::to_string(props.major) + "." + std::to_string(props.minor));

  void *ft_dev = nullptr;
  if (!alloc_pinned_mapped(3 * sizeof(cudaq_function_entry_t), &ft_host_,
                           &ft_dev))
    throw std::runtime_error(
        "DeviceGraphRingConsumer: function-table pinned alloc failed");

  auto *entries = static_cast<cudaq_function_entry_t *>(ft_host_);
  bool ok =
      populate_device_call(entries[0],
                           "cudaqx_qec_realtime_dispatch_populate_enqueue_"
                           "syndromes_device_entry",
                           kEnqueueSyndromesFunctionId) &&
      populate_device_call(
          entries[1],
          "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry",
          kGetCorrectionsFunctionId) &&
      populate_device_call(
          entries[2],
          "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry",
          kResetDecoderFunctionId);
  if (!ok) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphRingConsumer: populate_device_call failed "
        "(see error log above)");
  }

  // Resolve the dispatch graph API via dlsym; cudaq-realtime-dispatch is
  // linked into the process (not this library) to keep the CUDA module in
  // one copy.  Signatures must match cudaq-realtime's
  // create/launch/destroy_dispatch_graph exports exactly.
  using create_fn_t = cudaError_t (*)(
      volatile std::uint64_t *, volatile std::uint64_t *, std::uint8_t *,
      std::uint8_t *, std::size_t, std::size_t, cudaq_function_entry_t *,
      std::size_t, void *, volatile int *, std::uint64_t *, std::size_t,
      std::uint32_t, std::uint32_t, cudaGraphExec_t, cudaStream_t,
      cudaq_dispatch_graph_context **);
  using launch_fn_t =
      cudaError_t (*)(cudaq_dispatch_graph_context *, cudaStream_t);
  using destroy_fn_t = cudaError_t (*)(cudaq_dispatch_graph_context *);

  auto create_dispatch = reinterpret_cast<create_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_create_dispatch_graph_regular"));
  auto launch_dispatch = reinterpret_cast<launch_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_launch_dispatch_graph"));
  auto destroy_dispatch = reinterpret_cast<destroy_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_destroy_dispatch_graph"));
  if (!create_dispatch || !launch_dispatch || !destroy_dispatch) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphRingConsumer: cudaq dispatch API not found "
        "(cudaq_create/launch/destroy_dispatch_graph_regular); link "
        "cudaq-realtime-dispatch into the process with --export-dynamic");
  }
  fn_destroy_dispatch_graph_ = destroy_dispatch;

  void *sd_host = nullptr, *sd_dev = nullptr;
  if (!alloc_pinned_mapped(sizeof(int), &sd_host, &sd_dev)) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphRingConsumer: shutdown-flag pinned alloc failed");
  }
  shutdown_host_ = static_cast<volatile int *>(sd_host);
  shutdown_dev_ = static_cast<volatile int *>(sd_dev);

  if (cudaMalloc(&d_stats_, sizeof(std::uint64_t)) != cudaSuccess ||
      cudaMemset(d_stats_, 0, sizeof(std::uint64_t)) != cudaSuccess) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error("DeviceGraphRingConsumer: d_stats_ alloc failed");
  }

  if (cudaError_t serr = cudaStreamCreate(&sched_stream_);
      serr != cudaSuccess) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    cudaFree(d_stats_);
    d_stats_ = nullptr;
    throw std::runtime_error(
        std::string("DeviceGraphRingConsumer: stream create failed: ") +
        cudaGetErrorString(serr));
  }

  cudaError_t cerr = create_dispatch(
      ring.rx_flags, ring.tx_flags, ring.rx_data, ring.tx_data, slot_size,
      slot_size, static_cast<cudaq_function_entry_t *>(ft_dev),
      /*func_count=*/3,
      /*graph_io_ctx=*/nullptr, shutdown_dev_, d_stats_, num_slots,
      /*num_blocks=*/1, /*threads_per_block=*/64, graph_res->graph_exec,
      sched_stream_, &sched_ctx_);
  if (cerr == cudaSuccess)
    cerr = launch_dispatch(sched_ctx_, sched_stream_);
  if (cerr != cudaSuccess) {
    if (sched_ctx_) {
      fn_destroy_dispatch_graph_(sched_ctx_);
      sched_ctx_ = nullptr;
    }
    cudaStreamDestroy(sched_stream_);
    sched_stream_ = nullptr;
    cudaFree(d_stats_);
    d_stats_ = nullptr;
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        std::string("DeviceGraphRingConsumer: scheduler launch: ") +
        cudaGetErrorString(cerr));
  }

  CUDA_QEC_INFO("DeviceGraphRingConsumer: GPU scheduler launched  "
                "slots={} slot_size={} (3 DEVICE_CALL entries, "
                "graph_exec={:p})",
                num_slots, slot_size,
                static_cast<void *>(graph_res->graph_exec));
}

void DeviceGraphRingConsumer::shutdown() {
  if (stopped_)
    return;
  stopped_ = true;
  // Read the dispatch archive's trigger-path debug state BEFORE signalling
  // shutdown / draining (the drain hangs if a triggered decode graph never
  // completed, and this diagnostic is exactly for that case).
  using debug_fn_t =
      cudaError_t (*)(int *, unsigned long long *, unsigned long long *);
  if (auto get_debug = reinterpret_cast<debug_fn_t>(
          ::dlsym(RTLD_DEFAULT, "cudaq_dispatch_get_trigger_debug"))) {
    // The getter reads per-device module globals via cudaMemcpyFromSymbol in
    // the CALLER's current device context. shutdown() can run on a teardown
    // thread whose current device differs from gpu_id_ (same hazard as
    // dispatched() below); reading the wrong device's copy would report the
    // pristine "never fired" state. Pin gpu_id_ for the call and restore.
    int prev_device = 0;
    bool device_pinned = cudaGetDevice(&prev_device) == cudaSuccess &&
                         cudaSetDevice(gpu_id_) == cudaSuccess;
    int trigger_rc = -1;
    unsigned long long fires = 0, tails = 0;
    if (device_pinned && get_debug(&trigger_rc, &fires, &tails) == cudaSuccess)
      CUDA_QEC_INFO(
          "DeviceGraphRingConsumer: trigger debug rc={} ({}) fires={} "
          "tail_relaunches={}",
          trigger_rc,
          trigger_rc == -1000
              ? "never fired"
              : cudaGetErrorString(static_cast<cudaError_t>(trigger_rc)),
          fires, tails);
    if (device_pinned)
      cudaSetDevice(prev_device);
  }
  if (shutdown_host_)
    __atomic_store_n(shutdown_host_, 1, __ATOMIC_RELEASE);
}

std::uint64_t DeviceGraphRingConsumer::dispatched() const {
  // Async copy on a private non-blocking stream: a legacy default-stream
  // cudaMemcpy would synchronize with the (persistent, possibly wedged)
  // scheduler graph and deadlock the caller.
  std::uint64_t value = 0;
  if (!d_stats_)
    return value;
  // d_stats_ was allocated on gpu_id_, but this const accessor may be called
  // from a thread whose current device differs (e.g. the server teardown
  // path), which would make the stream/copy target the wrong context and
  // silently return 0. Pin gpu_id_ for the copy and restore the caller's
  // device afterward.
  int prev_device = 0;
  if (cudaGetDevice(&prev_device) != cudaSuccess)
    return value;
  if (cudaSetDevice(gpu_id_) != cudaSuccess)
    return value;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
      cudaSuccess) {
    if (cudaMemcpyAsync(&value, d_stats_, sizeof(value), cudaMemcpyDeviceToHost,
                        stream) == cudaSuccess)
      cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  cudaSetDevice(prev_device);
  return value;
}

DeviceGraphRingConsumer::~DeviceGraphRingConsumer() {
  shutdown();
  if (sched_stream_) {
    cudaStreamSynchronize(sched_stream_); // drain the self-relaunch chain
    if (sched_ctx_ && fn_destroy_dispatch_graph_)
      fn_destroy_dispatch_graph_(sched_ctx_);
    cudaStreamDestroy(sched_stream_);
  }
  if (ft_host_)
    cudaFreeHost(ft_host_);
  if (shutdown_host_)
    cudaFreeHost(const_cast<int *>(shutdown_host_));
  if (d_stats_)
    cudaFree(d_stats_);
}

} // namespace cudaq::qec::decoding_server

//==============================================================================
// C ABI (weak-linkable from the decoding_server tool)
//==============================================================================

using cudaq::qec::decoding_server::DeviceGraphRingConsumer;

extern "C" cudaqx_qec_device_graph_ring_consumer_t
cudaqx_qec_make_device_graph_ring_consumer(const void *ring,
                                           std::size_t num_slots,
                                           std::size_t slot_size, int gpu_id,
                                           void *graph_resources) {
  try {
    return new DeviceGraphRingConsumer(
        *static_cast<const cudaq_ringbuffer_t *>(ring), num_slots, slot_size,
        gpu_id, graph_resources);
  } catch (const std::exception &e) {
    CUDA_QEC_ERROR("cudaqx_qec_make_device_graph_ring_consumer: {}", e.what());
    return nullptr;
  }
}

extern "C" void cudaqx_qec_device_graph_ring_consumer_shutdown(
    cudaqx_qec_device_graph_ring_consumer_t consumer) {
  if (consumer)
    static_cast<DeviceGraphRingConsumer *>(consumer)->shutdown();
}

extern "C" std::uint64_t cudaqx_qec_device_graph_ring_consumer_dispatched(
    cudaqx_qec_device_graph_ring_consumer_t consumer) {
  return consumer
             ? static_cast<DeviceGraphRingConsumer *>(consumer)->dispatched()
             : 0;
}

extern "C" void cudaqx_qec_device_graph_ring_consumer_destroy(
    cudaqx_qec_device_graph_ring_consumer_t consumer) {
  delete static_cast<DeviceGraphRingConsumer *>(consumer);
}

#endif // CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
