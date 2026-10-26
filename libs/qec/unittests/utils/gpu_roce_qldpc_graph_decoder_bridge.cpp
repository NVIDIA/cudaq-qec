/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file gpu_roce_qldpc_graph_decoder_bridge.cpp
/// @brief QLDPC Relay-BP decoder bridge: GpuRoceTransceiver GPU-RoCE ring <->
/// the
///        self-relaunching device-graph scheduler.
///
/// This bridge wires the per-round decode-server protocol onto a real
/// (or emulated) FPGA over RoCE.  The bridge runs the device-graph scheduler
/// directly on the GpuRoceTransceiver DOCA ring:
///
///   FPGA --RDMA--> GpuRoceTransceiver RX kernel --writes rx_flags--> scheduler
///   graph scheduler graph --DEVICE_CALL append/get/reset; fires decode on a
///   full
///     window (CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH); tail-self-relaunches-->
///   scheduler writes RPCResponse + tx_flags --> GpuRoceTransceiver TX kernel
///   --RDMA--> FPGA
///
/// Flow:
///   1. Parse --config (Relay BP YAML) + generic bridge args.
///   2. Create the nv-qldpc decoder; capture_decode_graph(reserved_sms) gives a
///      device-launchable cooperative decode graph and registers the decoder's
///      GpuDecoderState (read by the append/get/reset DEVICE_CALL handlers).
///   3. Create + start the GpuRoceTransceiver (RX + TX kernels) and adopt its
///      DOCA ring (rx/tx flags + data are GPU pointers).
///   4. Build a pinned-mapped 3-entry DEVICE_CALL function table (enqueue
///      accumulate, get_corrections, reset_decoder) via the proprietary
///      populate shims (resolved by name -- the bridge exe absorbs
///      libcudaq-qec-realtime-cudevice-proprietary.a and device-links it with
///      the dispatch kernel).
///   5. Launch the device-graph scheduler on the DOCA ring with the decode
///      graph as its triggered graph, then run the GpuRoceTransceiver RX/TX
///      kernels (blocking_monitor) on a worker thread.
///   6. Run until --timeout or SIGINT, then shut down cleanly.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "cudaq/realtime/daemon/bridge/gpu_roce/gpu_roce_wrapper.h"
#include "cudaq/realtime/gpu_roce_bridge_common.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/qec/realtime/sparse_to_csr.h"

namespace {
/// Flat, -1-terminated sparse rows (the YAML/config encoding) to a shaped
/// sparse matrix, so O and D can be handed to the decoder as part of its model.
cudaq::qec::sparse_binary_matrix
sparse_matrix_from_flat_rows(const std::vector<std::int64_t> &flat,
                             std::uint32_t num_rows) {
  std::vector<std::vector<std::uint32_t>> rows;
  std::vector<std::uint32_t> row;
  std::uint32_t num_cols = 0;
  for (std::int64_t entry : flat) {
    if (entry < 0) {
      rows.push_back(std::move(row));
      row.clear();
      continue;
    }
    const auto col = static_cast<std::uint32_t>(entry);
    num_cols = std::max(num_cols, col + 1);
    row.push_back(col);
  }
  if (!row.empty())
    rows.push_back(std::move(row));
  rows.resize(num_rows);
  return cudaq::qec::sparse_binary_matrix::from_nested_csr(num_rows, num_cols,
                                                           rows);
}
} // namespace

namespace {

std::atomic<bool> g_stop{false};
void handle_sigint(int) { g_stop.store(true, std::memory_order_release); }

constexpr auto kGpuRoceMonitorStartupGrace = std::chrono::milliseconds(250);

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: Cannot open file: " << path << std::endl;
    return {};
  }
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Resolve a proprietary DEVICE_CALL populate shim by name and stamp the entry.
// dlsym(RTLD_DEFAULT) contract: the symbols are exported from this executable
// because it absorbs the cudevice proprietary archive (WHOLE_ARCHIVE) and links
// with --export-dynamic.
using populate_fn = void (*)(void *);
bool populate_device_call(cudaq_function_entry_t &entry, const char *symbol,
                          std::uint32_t function_id) {
  auto fn = reinterpret_cast<populate_fn>(::dlsym(RTLD_DEFAULT, symbol));
  if (!fn) {
    std::cerr << "ERROR: dlsym(" << symbol
              << ") failed -- the bridge must absorb "
                 "libcudaq-qec-realtime-cudevice-proprietary.a"
              << std::endl;
    return false;
  }
  fn(&entry);
  entry.function_id = function_id;
  entry.routing_key = 0;
  if (entry.dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
      !entry.handler.device_fn_ptr) {
    std::cerr << "ERROR: " << symbol
              << " did not produce a valid DEVICE_CALL entry" << std::endl;
    return false;
  }
  return true;
}

bool alloc_pinned_mapped(std::size_t bytes, void **host_out, void **dev_out) {
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

} // namespace

int main(int argc, char *argv[]) {
  namespace rpc = cudaq::qec::decoding::rpc;

  std::string config_path;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--config=") == 0)
      config_path = arg.substr(9);
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " --config=PATH [bridge options]\n\n"
                << "QLDPC Relay-BP bridge: GpuRoceTransceiver GPU-RoCE ring "
                   "<-> device-graph "
                   "scheduler.\n\n"
                << "  --config=PATH      Relay BP config YAML (required)\n"
                << "  --device=NAME      IB device (default: rocep1s0f0)\n"
                << "  --peer-ip=ADDR     FPGA/emulator IP (default: 10.0.0.2)\n"
                << "  --remote-qp=N      Remote QP number (default: 0x2)\n"
                << "  --gpu=N            GPU device ID (default: 0)\n"
                << "  --timeout=N        Timeout seconds (default: 60)\n"
                << "  --page-size=N      Ring slot size (default: 384)\n"
                << "  --num-pages=N      Ring slots (default: 64)\n"
                << "  --reserved-sms=N   SMs reserved for GpuRoceTransceiver "
                   "RX/TX (default: "
                   "2)\n";
      return 0;
    }
  }
  if (config_path.empty()) {
    std::cerr << "ERROR: --config=PATH is required" << std::endl;
    return 1;
  }

  cudaq::realtime::BridgeConfig config;
  cudaq::realtime::parse_bridge_args(argc, argv, config);

  int reserved_sms = 2;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--reserved-sms=") == 0)
      reserved_sms = std::stoi(arg.substr(15));
  }

  // Guard: clamp num_pages to the GpuRoceTransceiver receive/send work-queue
  // depth.
  //
  // The GpuRoceTransceiver (HSB 2.6.0-EA2) posts WQE_NUM=64
  // receive/send WQEs and runs one kernel thread per WQE.  When the ring is
  // deeper than that, a single thread services multiple ring slots (slot t and
  // t+64 share one WQE / CQ position), and the free-running RX/TX kernels race
  // on that shared resource -- empirically a duplicated frame W plus a dropped
  // frame W+64 (verified on the emulator: every failure was an exact (W, W+64)
  // pair on one thread, with no RDMA timeouts).  A 1:1 slot<->WQE mapping
  // (num_pages <= WQE_NUM) is the only safe configuration.  We clamp rather
  // than abort so a stale/oversized --num-pages can't silently corrupt data.
  constexpr unsigned kGpuRoceWqeNum =
      64; // == GpuRoceTransceiver WQE_NUM (gpu_roce_transceiver_common.hpp)
  if (config.num_pages > kGpuRoceWqeNum) {
    std::cerr << "WARNING: --num-pages=" << config.num_pages
              << " exceeds the GpuRoceTransceiver's WQE depth ("
              << kGpuRoceWqeNum << "); clamping to " << kGpuRoceWqeNum
              << " (a deeper ring multiplexes >1 slot per WQE and races the "
                 "RX/TX kernels -> duplicate/drop)."
              << std::endl;
    config.num_pages = kGpuRoceWqeNum;
  }

  std::cout << "=== GpuRoceTransceiver QLDPC Relay-BP Bridge (device-graph "
               "scheduler) ==="
            << std::endl;

  // -- Load decoder config + build the decoder --------------------------------
  std::string yaml_str = read_file(config_path);
  if (yaml_str.empty())
    return 1;
  auto mdc = cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
      yaml_str);
  if (mdc.decoders.empty()) {
    std::cerr << "ERROR: No decoders found in config" << std::endl;
    return 1;
  }
  auto &dec = mdc.decoders[0];

  std::vector<uint32_t> h_row_ptr, h_col_idx;
  cudaq::qec::realtime::sparse_vec_to_csr(dec.H_sparse, h_row_ptr, h_col_idx);
  std::size_t bs = dec.block_size;
  std::size_t ss = dec.syndrome_size;
  cudaqx::tensor<uint8_t> H_tensor({ss, bs});
  for (std::size_t r = 0; r < ss; ++r)
    for (uint32_t j = h_row_ptr[r]; j < h_row_ptr[r + 1]; ++j)
      H_tensor.at({r, static_cast<std::size_t>(h_col_idx[j])}) = 1;

  auto params = dec.decoder_custom_args_to_heterogeneous_map();
  // O and D belong to the model, so they are supplied at construction rather
  // than installed afterwards.
  const auto num_observable_rows = static_cast<std::uint32_t>(
      std::count(dec.O_sparse.begin(), dec.O_sparse.end(), -1));
  auto decoder = cudaq::qec::decoder::get(
      "nv-qldpc-decoder",
      cudaq::qec::decoder_init(
          cudaq::qec::sparse_binary_matrix(H_tensor),
          sparse_matrix_from_flat_rows(dec.O_sparse, num_observable_rows),
          dec.effective_error_rate_vec(),
          sparse_matrix_from_flat_rows(dec.D_sparse,
                                       static_cast<std::uint32_t>(ss))),
      params);
  if (!decoder) {
    std::cerr << "ERROR: Failed to create nv-qldpc-decoder" << std::endl;
    return 1;
  }

  std::vector<uint32_t> d_rp, d_ci;
  cudaq::qec::realtime::sparse_vec_to_csr(dec.D_sparse, d_rp, d_ci);
  std::size_t num_measurements = 0;
  for (auto c : d_ci)
    num_measurements =
        std::max(num_measurements, static_cast<std::size_t>(c + 1));
  std::vector<uint32_t> o_rp, o_ci;
  std::size_t num_observables =
      cudaq::qec::realtime::sparse_vec_to_csr(dec.O_sparse, o_rp, o_ci);

  std::cout << "  block_size=" << bs << " syndrome_size=" << ss
            << " num_measurements=" << num_measurements
            << " num_observables=" << num_observables << std::endl;

  if (!decoder->supports_graph_dispatch()) {
    std::cerr << "ERROR: nv-qldpc-decoder does not support graph dispatch"
              << std::endl;
    return 1;
  }

  BRIDGE_CUDA_CHECK(cudaSetDevice(config.gpu_id));

  // -- Capture the device-launchable cooperative decode graph -----------------
  // Reserve SMs for the GpuRoceTransceiver RX/TX kernels so the cooperative
  // decode can still co-reside.  This also registers the decoder's
  // GpuDecoderState with the proprietary device table (read by the DEVICE_CALL
  // handlers).
  void *raw_res = decoder->capture_decode_graph(reserved_sms);
  if (!raw_res) {
    std::cerr << "ERROR: capture_decode_graph() returned null" << std::endl;
    return 1;
  }
  auto *graph_res =
      static_cast<cudaq::qec::realtime::graph_resources *>(raw_res);
  if (!graph_res->graph_exec) {
    std::cerr << "ERROR: capture_decode_graph() produced no graph_exec"
              << std::endl;
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  std::cout << "  Decode graph captured (device-launchable), reserved_sms="
            << reserved_sms << std::endl;

  // -- Size the ring frame for the largest per-round RPC ----------------------
  // Per-round protocol: enqueue carries EnqueueRequestPayload + bit-packed
  // per-round syndromes; get_corrections returns bit-packed observables.  Size
  // the page for the worst case (whole-window measurements as an upper bound)
  // and let --page-size override upward.
  const std::size_t enqueue_max =
      sizeof(cudaq::realtime::RPCHeader) +
      rpc::align_to_8(sizeof(rpc::EnqueueRequestPayload) +
                      rpc::bit_packed_bytes(num_measurements));
  const std::size_t get_resp_max =
      sizeof(cudaq::realtime::RPCResponse) +
      rpc::align_to_8(rpc::bit_packed_bytes(num_observables));
  std::size_t min_frame = std::max(enqueue_max, get_resp_max);
  config.frame_size = std::max<std::size_t>(config.frame_size, min_frame);
  if (config.page_size < config.frame_size)
    config.page_size = config.frame_size;
  // 128-byte page granularity for GpuRoceTransceiver.
  config.page_size = (config.page_size + 127) & ~static_cast<std::size_t>(127);
  std::cout << "  frame_size=" << config.frame_size
            << " page_size=" << config.page_size
            << " num_pages=" << config.num_pages << std::endl;

  // -- Create + start the GpuRoceTransceiver (RX + TX kernels) --------------
  gpu_roce_transceiver_t transceiver = gpu_roce_create_transceiver(
      config.device.c_str(), 1, config.remote_qp, config.gpu_id,
      config.frame_size, config.page_size, config.num_pages,
      config.peer_ip.c_str(), /*forward=*/0, /*rx_only=*/1, /*tx_only=*/1);
  if (!transceiver) {
    std::cerr << "ERROR: Failed to create GpuRoceTransceiver" << std::endl;
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  if (!gpu_roce_start(transceiver)) {
    // Do NOT destroy a half-initialized transceiver here: the DOCA teardown
    // path double-frees GPU memory that start() never allocated and segfaults,
    // which obscures the real error.  This is a fatal exit, so let the OS
    // reclaim.  The most common cause is a missing IPv4-mapped RoCE v2 GID --
    // ensure the bridge netdev has its IPv4 address assigned and is up.
    std::cerr << "ERROR: gpu_roce_start failed (often a missing RoCE v2 GID -- "
                 "check the IB device's IPv4 address is assigned)"
              << std::endl;
    return 1;
  }
  BRIDGE_CUDA_CHECK(cudaSetDevice(config.gpu_id));

  // -- Adopt the DOCA ring (GPU pointers) -------------------------------------
  auto *rx_data =
      reinterpret_cast<uint8_t *>(gpu_roce_get_rx_ring_data_addr(transceiver));
  auto *rx_flag = gpu_roce_get_rx_ring_flag_addr(transceiver);
  auto *tx_data =
      reinterpret_cast<uint8_t *>(gpu_roce_get_tx_ring_data_addr(transceiver));
  auto *tx_flag = gpu_roce_get_tx_ring_flag_addr(transceiver);
  if (!rx_data || !rx_flag || !tx_data || !tx_flag) {
    std::cerr << "ERROR: null DOCA ring pointer(s)" << std::endl;
    gpu_roce_close(transceiver);
    gpu_roce_destroy_transceiver(transceiver);
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  const std::size_t num_slots = gpu_roce_get_num_pages(transceiver);
  const std::size_t slot_size = gpu_roce_get_page_size(transceiver);

  // -- Build the pinned-mapped 3-entry DEVICE_CALL function table -------------
  void *ft_host = nullptr;
  void *ft_dev = nullptr;
  if (!alloc_pinned_mapped(3 * sizeof(cudaq_function_entry_t), &ft_host,
                           &ft_dev)) {
    std::cerr << "ERROR: function-table alloc failed" << std::endl;
    gpu_roce_close(transceiver);
    gpu_roce_destroy_transceiver(transceiver);
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  auto *entries = static_cast<cudaq_function_entry_t *>(ft_host);
  bool ok =
      populate_device_call(
          entries[0],
          "cudaqx_qec_realtime_dispatch_populate_enqueue_syndromes_device_"
          "entry",
          rpc::kEnqueueSyndromesFunctionId) &&
      populate_device_call(
          entries[1],
          "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry",
          rpc::kGetCorrectionsFunctionId) &&
      populate_device_call(
          entries[2],
          "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry",
          rpc::kResetDecoderFunctionId);
  if (!ok) {
    cudaFreeHost(ft_host);
    gpu_roce_close(transceiver);
    gpu_roce_destroy_transceiver(transceiver);
    decoder->release_decode_graph(raw_res);
    return 1;
  }

  // -- Shutdown flag (pinned-mapped) + stats ----------------------------------
  void *sd_host = nullptr;
  void *sd_dev = nullptr;
  if (!alloc_pinned_mapped(sizeof(int), &sd_host, &sd_dev)) {
    std::cerr << "ERROR: shutdown-flag alloc failed" << std::endl;
    cudaFreeHost(ft_host);
    gpu_roce_close(transceiver);
    gpu_roce_destroy_transceiver(transceiver);
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  auto *shutdown_host = static_cast<volatile int *>(sd_host);
  auto *shutdown_dev = static_cast<volatile int *>(sd_dev);
  uint64_t *d_stats = nullptr;
  BRIDGE_CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
  BRIDGE_CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));

  // -- Launch the device-graph scheduler on the DOCA ring ---------------------
  // Strict-FIFO consumption: shared_ring_mode is OFF.  This scheduler is the
  // SOLE consumer of the DOCA RX ring (no peer dispatcher), and the
  // GpuRoceTransceiver RX kernel fills slots strictly in order (window N ->
  // slot N % num_pages).  The persistent cursor in dispatch_kernel_with_graph
  // keeps current_slot across the tail self-relaunch, so the scheduler waits at
  // the next slot in order rather than rescanning from 0 -- which is what
  // avoids the slot-reuse race (out-of-order grab + flag-clear vs. refill) that
  // shared_ring scanning introduced here.  The dispatch kernel default is
  // shared-ring OFF, so no setter call is needed.

  cudaStream_t sched_stream = nullptr;
  BRIDGE_CUDA_CHECK(cudaStreamCreate(&sched_stream));

  cudaq_dispatch_graph_context *sched_ctx = nullptr;
  cudaError_t cerr = cudaq_create_dispatch_graph_regular(
      reinterpret_cast<volatile uint64_t *>(rx_flag),
      reinterpret_cast<volatile uint64_t *>(tx_flag), rx_data, tx_data,
      slot_size, slot_size, static_cast<cudaq_function_entry_t *>(ft_dev),
      /*func_count=*/3, /*graph_io_ctx=*/nullptr, shutdown_dev, d_stats,
      num_slots, /*num_blocks=*/1, /*threads_per_block=*/64,
      graph_res->graph_exec, sched_stream, &sched_ctx);
  if (cerr != cudaSuccess) {
    std::cerr << "ERROR: cudaq_create_dispatch_graph_regular: "
              << cudaGetErrorString(cerr) << std::endl;
    cudaFree(d_stats);
    cudaFreeHost(ft_host);
    cudaFreeHost(sd_host);
    gpu_roce_close(transceiver);
    gpu_roce_destroy_transceiver(transceiver);
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  cerr = cudaq_launch_dispatch_graph(sched_ctx, sched_stream);
  if (cerr != cudaSuccess) {
    std::cerr << "ERROR: cudaq_launch_dispatch_graph: "
              << cudaGetErrorString(cerr) << std::endl;
    cudaq_destroy_dispatch_graph(sched_ctx);
    cudaFree(d_stats);
    cudaFreeHost(ft_host);
    cudaFreeHost(sd_host);
    gpu_roce_close(transceiver);
    gpu_roce_destroy_transceiver(transceiver);
    decoder->release_decode_graph(raw_res);
    return 1;
  }
  std::cout << "  Scheduler launched (3 DEVICE_CALL entries, triggered decode "
               "graph)"
            << std::endl;

  // -- Run the GpuRoceTransceiver RX/TX kernels on a worker thread
  // ----------------------
  std::signal(SIGINT, handle_sigint);
  std::thread monitor([&]() { gpu_roce_blocking_monitor(transceiver); });

  // blocking_monitor() launches the GpuRoceTransceiver RX/TX kernels inside the
  // worker thread.  Give those launches a short grace period before the
  // orchestration script sees "Bridge Ready" and starts playback.
  std::this_thread::sleep_for(kGpuRoceMonitorStartupGrace);

  // Emit the bridge's RDMA target info in the exact format the orchestration
  // script greps (extract_hex for QP/Buffer, extract_decimal for RKey) so the
  // playback tool can point the FPGA/emulator SIF at our GPU ring, and signal
  // readiness AFTER the scheduler + RX/TX kernels are live.
  std::cout << "QP Number: 0x" << std::hex
            << gpu_roce_get_qp_number(transceiver) << std::dec << "\n";
  std::cout << "RKey: " << gpu_roce_get_rkey(transceiver) << "\n";
  std::cout << "Buffer Addr: 0x" << std::hex
            << gpu_roce_get_buffer_addr(transceiver) << std::dec << "\n";
  std::cout << "Bridge Ready" << std::endl;

  std::cout << "  Running for up to " << config.timeout_sec
            << " s (Ctrl-C to stop)..." << std::endl;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(config.timeout_sec);
  while (!g_stop.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // -- Shutdown ---------------------------------------------------------------
  std::cout << "  Stopping..." << std::endl;
  __atomic_store_n(shutdown_host, 1, __ATOMIC_RELEASE);
  __sync_synchronize();
  cudaStreamSynchronize(sched_stream); // drain the self-relaunch chain
  gpu_roce_close(transceiver);         // stop RX/TX monitor
  if (monitor.joinable())
    monitor.join();

  cudaq_destroy_dispatch_graph(sched_ctx);
  cudaStreamDestroy(sched_stream);
  cudaFree(d_stats);
  cudaFreeHost(ft_host);
  cudaFreeHost(sd_host);
  gpu_roce_destroy_transceiver(transceiver);
  decoder->release_decode_graph(raw_res);

  std::cout << "=== Bridge exited cleanly ===" << std::endl;
  return 0;
}
