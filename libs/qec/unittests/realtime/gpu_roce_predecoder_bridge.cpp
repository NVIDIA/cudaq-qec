/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_roce_predecoder_bridge.cpp
/// @brief GpuRoceTransceiver bridge for the AI predecoder + PyMatching
/// pipeline.
///
/// Combines GpuRoceTransceiver RDMA transport with the realtime_pipeline
/// infrastructure to run AI pre-decoding and PyMatching on syndrome data
/// arriving from an FPGA or emulator.
///
/// GpuRoceTransceiver setup is extracted from bridge_run() (HOST_LOOP path).
/// The ring buffer pointers from the transceiver are fed into realtime_pipeline
/// via the external_ringbuffer mechanism, which avoids the double-dispatcher
/// conflict that would arise from using bridge_run() directly.

#include "predecoder_pipeline_common.h"

#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

#include "cudaq/realtime/daemon/bridge/gpu_roce/gpu_roce_wrapper.h"

#define BRIDGE_CUDA_CHECK(call)                                                \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// =============================================================================
// Signal handling
// =============================================================================

static std::atomic<bool> g_shutdown{false};
static void signal_handler(int) { g_shutdown = true; }

// =============================================================================
// Main
// =============================================================================

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // -- Parse arguments -------------------------------------------------------

  std::string config_name = "d7";
  std::string ib_device = "rocep1s0f0";
  std::string peer_ip = "10.0.0.2";
  uint32_t remote_qp = 0x2;
  int gpu_id = 0;
  int timeout_sec = 60;
  size_t page_size = 0;
  unsigned num_pages = 128;
  std::string data_dir;
  network_typing_override typing_override = network_typing_override::automatic;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--config=") == 0)
      config_name = arg.substr(9);
    else if (arg.find("--device=") == 0)
      ib_device = arg.substr(9);
    else if (arg.find("--peer-ip=") == 0)
      peer_ip = arg.substr(10);
    else if (arg.find("--remote-qp=") == 0)
      remote_qp = std::stoul(arg.substr(12), nullptr, 0);
    else if (arg.find("--gpu=") == 0)
      gpu_id = std::stoi(arg.substr(6));
    else if (arg.find("--timeout=") == 0)
      timeout_sec = std::stoi(arg.substr(10));
    else if (arg.find("--page-size=") == 0)
      page_size = std::stoull(arg.substr(12));
    else if (arg.find("--num-pages=") == 0)
      num_pages = std::stoul(arg.substr(12));
    else if (arg.find("--data-dir=") == 0)
      data_dir = arg.substr(11);
    else if (arg.find("--typing=") == 0) {
      try {
        typing_override = parse_network_typing(arg.substr(9));
      } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n\n"
          << "GpuRoceTransceiver predecoder + PyMatching bridge.\n\n"
          << "Decoder options:\n"
          << "  --config=NAME         d7|d13|d13_r104|d21|d31 (default: d7)\n"
          << "  --typing=MODE         auto|weak|strong (default: auto)\n\n"
          << "Correctness:\n"
          << "  --data-dir=PATH       Test data dir with observables.bin for\n"
          << "                        ground-truth verification\n\n"
          << "Bridge options:\n"
          << "  --device=NAME         IB device (default: rocep1s0f0)\n"
          << "  --peer-ip=ADDR        FPGA/emulator IP (default: 10.0.0.2)\n"
          << "  --remote-qp=N         Remote QP number (default: 0x2)\n"
          << "  --gpu=N               GPU device ID (default: 0)\n"
          << "  --timeout=N           Timeout in seconds (default: 60)\n"
          << "  --page-size=N         Ring buffer slot size (default: auto)\n"
          << "  --num-pages=N         Ring buffer slots (default: 128)\n\n"
          << "Config overrides (applied after --config preset):\n"
          << "  --distance=N          QEC code distance\n"
          << "  --num-rounds=N        Syndrome measurement rounds\n"
          << "  --onnx-filename=FILE  ONNX model filename\n"
          << "  --num-predecoders=N   Parallel TRT instances\n"
          << "  --num-workers=N       Pipeline GPU workers\n"
          << "  --num-decode-workers=N  PyMatching threads\n"
          << "  --label=NAME          Config label for reports\n";
      return 0;
    }
  }

  auto pcfg_opt = PipelineConfig::from_name(config_name);
  if (!pcfg_opt) {
    std::cerr << "ERROR: Unknown config: " << config_name << std::endl;
    return 1;
  }
  PipelineConfig pcfg = *pcfg_opt;
  pcfg.apply_cli_overrides(argc, argv);

  std::cout << "=== GpuRoceTransceiver Predecoder + PyMatching Bridge ==="
            << std::endl;
  std::cout << "  Config: " << pcfg.label << std::endl;

  // -- Initialize CUDA -------------------------------------------------------

  BRIDGE_CUDA_CHECK(cudaSetDevice(gpu_id));
  BRIDGE_CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  // -- Create predecoder instances -------------------------------------------

  std::string engine_file = pcfg.engine_path();
  std::string onnx_file = pcfg.onnx_path();
  std::string model_path;
  {
    std::ifstream probe(engine_file, std::ios::binary);
    model_path = probe.good() ? engine_file : onnx_file;
  }
  std::cout << "  Model: " << model_path << std::endl;

  void **h_mailbox_bank = nullptr;
  void **d_mailbox_bank = nullptr;
  BRIDGE_CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank,
                                  pcfg.num_predecoders * sizeof(void *),
                                  cudaHostAllocMapped));
  std::memset(h_mailbox_bank, 0, pcfg.num_predecoders * sizeof(void *));
  BRIDGE_CUDA_CHECK(cudaHostGetDevicePointer(
      reinterpret_cast<void **>(&d_mailbox_bank), h_mailbox_bank, 0));

  std::vector<cudaStream_t> predecoder_streams(pcfg.num_predecoders);
  std::vector<std::unique_ptr<ai_predecoder_service>> predecoders;
  bool need_save = (model_path == onnx_file);

  for (int i = 0; i < pcfg.num_predecoders; ++i) {
    BRIDGE_CUDA_CHECK(cudaStreamCreate(&predecoder_streams[i]));
    std::string save_path = (need_save && i == 0) ? engine_file : "";
    auto pd = std::make_unique<ai_predecoder_service>(
        model_path, d_mailbox_bank + i, 1, save_path, typing_override);
    cudaStream_t cap;
    BRIDGE_CUDA_CHECK(cudaStreamCreate(&cap));
    pd->capture_graph(cap, false);
    BRIDGE_CUDA_CHECK(cudaStreamDestroy(cap));
    predecoders.push_back(std::move(pd));
  }

  const size_t model_input_bytes = predecoders[0]->get_input_size();
  const size_t model_output_bytes = predecoders[0]->get_output_size();
  const size_t slot_size =
      round_up_pow2(CUDAQ_RPC_HEADER_SIZE + model_input_bytes);

  // Trust the engine: use the actual tensor volumes reported by TRT.
  const size_t num_input_detectors = predecoders[0]->get_input_num_elements();
  const size_t num_output_elements = predecoders[0]->get_output_num_elements();
  const int residual_detectors = static_cast<int>(num_output_elements) - 1;

  std::cout << "  Input detectors: " << num_input_detectors
            << ", Residual detectors: " << residual_detectors << std::endl;

  size_t frame_size = sizeof(rt_sdk::RPCHeader) + model_input_bytes;
  if (page_size == 0)
    page_size = slot_size;
  if (page_size < frame_size)
    page_size = frame_size;

  // -- Build PyMatching decoder pool -----------------------------------------

  DecoderContext decoder_ctx;
  decoder_ctx.num_residual_detectors = residual_detectors;
  decoder_ctx.num_input_detectors = static_cast<int>(num_input_detectors);
  cudaqx::heterogeneous_map pm_params;
  pm_params.insert("merge_strategy", std::string("smallest_weight"));

  {
    auto surface_code =
        cudaq::qec::get_code("surface_code", {{"distance", pcfg.distance}});
    auto H_z = surface_code->get_parity_z();
    const int z_stabilizers = static_cast<int>(H_z.shape()[0]);
    if (residual_detectors > 0 && residual_detectors % z_stabilizers == 0)
      decoder_ctx.spatial_slices = residual_detectors / z_stabilizers;
    decoder_ctx.z_stabilizers = z_stabilizers;

    for (int i = 0; i < pcfg.num_decode_workers; ++i)
      decoder_ctx.decoders.push_back(
          cudaq::qec::decoder::get("pymatching", H_z, pm_params));
  }
  std::cout << "  PyMatching: " << pcfg.num_decode_workers << " workers"
            << std::endl;

  // -- Load ground-truth observables (optional) ------------------------------

  std::vector<int32_t> ground_truth_obs;
  uint32_t obs_num_samples = 0, obs_num_observables = 0;
  if (!data_dir.empty()) {
    std::string obs_path = data_dir + "/observables.bin";
    std::ifstream obs_f(obs_path, std::ios::binary);
    if (obs_f.good()) {
      obs_f.read(reinterpret_cast<char *>(&obs_num_samples), sizeof(uint32_t));
      obs_f.read(reinterpret_cast<char *>(&obs_num_observables),
                 sizeof(uint32_t));
      ground_truth_obs.resize(static_cast<size_t>(obs_num_samples) *
                              obs_num_observables);
      obs_f.read(reinterpret_cast<char *>(ground_truth_obs.data()),
                 ground_truth_obs.size() * sizeof(int32_t));
      std::cout << "  Observables: " << obs_num_samples << " samples x "
                << obs_num_observables << " from " << obs_path << std::endl;
    } else {
      std::cerr << "  WARNING: Could not load " << obs_path << std::endl;
    }
  }

  constexpr int MAX_RESULTS = 4096;
  std::vector<int32_t> result_corrections(MAX_RESULTS, -1);
  std::vector<int32_t> result_logical_pred(MAX_RESULTS, -1);
  std::vector<int32_t> result_converged(MAX_RESULTS, -1);

  // -- Create GpuRoceTransceiver -------------------------------------------

  std::cout << "\n[1/3] Creating GpuRoceTransceiver..." << std::endl;
  std::cout << "  Device: " << ib_device << ", Peer: " << peer_ip << std::endl;
  std::cout << "  Frame: " << frame_size << " B, Page: " << page_size
            << " B, Pages: " << num_pages << std::endl;

  gpu_roce_transceiver_t transceiver = gpu_roce_create_transceiver(
      ib_device.c_str(), 1, remote_qp, gpu_id, frame_size, page_size, num_pages,
      peer_ip.c_str(),
      0, // forward
      1, // rx_only
      1  // tx_only
  );
  if (!transceiver) {
    std::cerr << "ERROR: Failed to create GpuRoceTransceiver" << std::endl;
    return 1;
  }

  gpu_roce_set_cpu_ring_buffers(transceiver, 1);

  if (!gpu_roce_start(transceiver)) {
    std::cerr << "ERROR: Failed to start GpuRoceTransceiver" << std::endl;
    gpu_roce_destroy_transceiver(transceiver);
    return 1;
  }
  BRIDGE_CUDA_CHECK(cudaSetDevice(gpu_id));

  uint32_t our_qp = gpu_roce_get_qp_number(transceiver);
  uint32_t our_rkey = gpu_roce_get_rkey(transceiver);
  uint64_t our_buffer = gpu_roce_get_buffer_addr(transceiver);

  uint8_t *rx_ring_data =
      reinterpret_cast<uint8_t *>(gpu_roce_get_rx_ring_data_addr(transceiver));
  uint64_t *rx_ring_flag = gpu_roce_get_rx_ring_flag_addr(transceiver);
  uint8_t *tx_ring_data =
      reinterpret_cast<uint8_t *>(gpu_roce_get_tx_ring_data_addr(transceiver));
  uint64_t *tx_ring_flag = gpu_roce_get_tx_ring_flag_addr(transceiver);

  if (!rx_ring_data || !rx_ring_flag || !tx_ring_data || !tx_ring_flag) {
    std::cerr << "ERROR: Failed to get ring buffer pointers" << std::endl;
    gpu_roce_destroy_transceiver(transceiver);
    return 1;
  }

  // -- Build external cudaq_ringbuffer_t from GpuRoceTransceiver pointers
  // --------------

  cudaq_ringbuffer_t ext_rb{};
  ext_rb.rx_flags = reinterpret_cast<volatile uint64_t *>(rx_ring_flag);
  ext_rb.tx_flags = reinterpret_cast<volatile uint64_t *>(tx_ring_flag);
  ext_rb.rx_data = rx_ring_data;
  ext_rb.tx_data = tx_ring_data;
  ext_rb.rx_stride_sz = page_size;
  ext_rb.tx_stride_sz = page_size;
  ext_rb.rx_flags_host = reinterpret_cast<volatile uint64_t *>(rx_ring_flag);
  ext_rb.tx_flags_host = reinterpret_cast<volatile uint64_t *>(tx_ring_flag);
  ext_rb.rx_data_host = rx_ring_data;
  ext_rb.tx_data_host = tx_ring_data;

  // -- Create realtime_pipeline with external ring ---------------------------

  std::cout << "\n[2/3] Building pipeline..." << std::endl;

  rt_pipeline::pipeline_stage_config stage_cfg;
  stage_cfg.num_workers = pcfg.num_workers;
  stage_cfg.num_slots =
      std::min(static_cast<size_t>(num_pages), static_cast<size_t>(NUM_SLOTS));
  stage_cfg.slot_size = page_size;
  stage_cfg.cores = {.dispatcher = 2, .consumer = 4, .worker_base = 10};
  stage_cfg.external_ringbuffer = &ext_rb;

  rt_pipeline::realtime_pipeline pipeline(stage_cfg);

  std::vector<PreLaunchCopyCtx> pre_launch_ctxs(pcfg.num_predecoders);
  for (int i = 0; i < pcfg.num_predecoders; ++i) {
    pre_launch_ctxs[i].d_trt_input = predecoders[i]->get_trt_input_ptr();
    pre_launch_ctxs[i].input_size = predecoders[i]->get_input_size();
    pre_launch_ctxs[i].h_ring_ptrs = predecoders[i]->get_host_ring_ptrs();
  }

  auto rb_bases = pipeline.ringbuffer_bases();
  for (int i = 0; i < pcfg.num_predecoders; ++i) {
    pre_launch_ctxs[i].rx_data_dev_base = rb_bases.rx_data_dev;
    pre_launch_ctxs[i].rx_data_host_base = rb_bases.rx_data_host;
  }

  std::vector<WorkerCtx> worker_ctxs(pcfg.num_workers);
  for (int i = 0; i < pcfg.num_workers; ++i) {
    worker_ctxs[i].predecoder = predecoders[i].get();
    worker_ctxs[i].decoder_ctx = &decoder_ctx;
  }

  const uint32_t shared_fid = rt_sdk::fnv1a_hash("predecode");
  std::vector<uint32_t> function_ids(pcfg.num_workers, shared_fid);

  std::vector<std::vector<uint8_t>> deferred_outputs(
      stage_cfg.num_slots, std::vector<uint8_t>(model_output_bytes));

  PyMatchQueue pymatch_queue;

  // -- GPU stage factory ---
  pipeline.set_gpu_stage([&](int w) -> rt_pipeline::gpu_worker_resources {
    return {.graph_exec = predecoders[w]->get_executable_graph(),
            .stream = predecoder_streams[w],
            .pre_launch_fn = pre_launch_input_copy,
            .pre_launch_data = &pre_launch_ctxs[w],
            .function_id = function_ids[w],
            .user_context = &worker_ctxs[w]};
  });

  // -- CPU stage callback ---
  pipeline.set_cpu_stage(
      [&deferred_outputs, &pymatch_queue, out_sz = model_output_bytes](
          const rt_pipeline::cpu_stage_context &ctx) -> size_t {
        auto *wctx = static_cast<WorkerCtx *>(ctx.user_context);
        auto *pd = wctx->predecoder;

        pre_decoder_job job;
        if (!pd->poll_next_job(job))
          return 0;

        NVTX_PUSH("PredecoderPoll");
        int origin_slot = ctx.origin_slot;

        std::memcpy(deferred_outputs[origin_slot].data(), job.inference_data,
                    out_sz);

        pd->release_job(job.slot_idx);

        auto *rpc_hdr =
            static_cast<const rt_sdk::RPCHeader *>(job.ring_buffer_ptr);
        uint32_t rid = rpc_hdr->request_id;

        const uint8_t *pred_out = deferred_outputs[origin_slot].data();
        int input_nz = 0;
        const uint8_t *inp = static_cast<const uint8_t *>(job.ring_buffer_ptr) +
                             CUDAQ_RPC_HEADER_SIZE;
        for (size_t k = 0; k < out_sz - 1; ++k)
          input_nz += (inp[k] != 0);
        int output_nz = 0;
        for (size_t k = 1; k < out_sz; ++k)
          output_nz += (pred_out[k] != 0);

        std::cout << "  [RDMA+TRT] Shot " << rid << ": received "
                  << (out_sz - 1) << " detectors"
                  << " (input_nonzero=" << input_nz << ")"
                  << ", predecoder logical_pred=" << (int)pred_out[0]
                  << ", residual_nonzero=" << output_nz << std::endl;

        pymatch_queue.push({origin_slot, rid, job.ring_buffer_ptr});

        NVTX_POP();
        return rt_pipeline::DEFERRED_COMPLETION;
      });

  // -- Completion handler ---
  std::atomic<uint64_t> n_completed{0};
  pipeline.set_completion_handler([&](const rt_pipeline::completion &c) {
    n_completed.fetch_add(1, std::memory_order_relaxed);
  });

  // -- PyMatching thread pool ---
  std::vector<std::thread> pymatch_threads(pcfg.num_decode_workers);
  for (int t = 0; t < pcfg.num_decode_workers; ++t) {
    pymatch_threads[t] = std::thread([&pipeline, &pymatch_queue,
                                      &deferred_outputs, &decoder_ctx,
                                      &result_corrections, &result_logical_pred,
                                      &result_converged]() {
      PyMatchJob job;
      while (pymatch_queue.pop(job)) {
        NVTX_PUSH("PyMatchDecode");
        auto decode_start = std::chrono::high_resolution_clock::now();

        const uint8_t *output_u8 = deferred_outputs[job.origin_slot].data();
        const int32_t logical_pred = output_u8[0];
        int total_corrections = 0;
        bool all_converged = true;

#if !defined(DISABLE_PYMATCHING)
        const uint8_t *residual_u8 = output_u8 + 1;
        auto *my_decoder = decoder_ctx.acquire_decoder();

        thread_local cudaqx::tensor<uint8_t> syndrome_tensor(
            {(size_t)decoder_ctx.z_stabilizers});
        uint8_t *syn_data = syndrome_tensor.data();
        for (int s = 0; s < decoder_ctx.spatial_slices; ++s) {
          const uint8_t *slice = residual_u8 + s * decoder_ctx.z_stabilizers;
          std::memcpy(syn_data, slice, decoder_ctx.z_stabilizers);
          auto result = my_decoder->decode(syndrome_tensor);
          all_converged &= result.converged;
          for (auto v : result.result)
            if (v > 0.5)
              total_corrections++;
        }
        total_corrections += logical_pred;
#endif

        auto decode_end = std::chrono::high_resolution_clock::now();
        NVTX_POP();

        DecodeResponse resp{total_corrections, all_converged ? 1 : 0};
        char *response_payload =
            (char *)job.ring_buffer_ptr + sizeof(rt_sdk::RPCResponse);
        std::memcpy(response_payload, &resp, sizeof(resp));

        auto *header = static_cast<rt_sdk::RPCResponse *>(job.ring_buffer_ptr);
        header->magic = rt_sdk::RPC_MAGIC_RESPONSE;
        header->status = 0;
        header->result_len = sizeof(resp);

        pipeline.complete_deferred(job.origin_slot);

        uint32_t rid = static_cast<uint32_t>(job.request_id);
        if (rid < result_corrections.size()) {
          result_corrections[rid] = total_corrections;
          result_logical_pred[rid] = logical_pred;
          result_converged[rid] = all_converged ? 1 : 0;
        }

        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      decode_end - decode_start)
                      .count();
        decoder_ctx.total_decode_us.fetch_add(us, std::memory_order_relaxed);
        decoder_ctx.decode_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // -- Start pipeline and GpuRoceTransceiver
  // -------------------------------------------

  std::cout << "\n[3/3] Starting..." << std::endl;
  pipeline.start();

  std::thread gpu_roce_thread(
      [transceiver]() { gpu_roce_blocking_monitor(transceiver); });
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec << std::endl;
  std::cout << "  RKey: " << our_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
            << std::endl;
  std::cout << "\nWaiting for data (Ctrl+C to stop, timeout=" << timeout_sec
            << "s)..." << std::endl;

  // -- Main loop -------------------------------------------------------------

  auto start_time = std::chrono::steady_clock::now();
  uint64_t last_count = 0;

  while (!g_shutdown) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start_time)
                       .count();
    if (elapsed > timeout_sec) {
      std::cout << "\nTimeout reached (" << timeout_sec << "s)" << std::endl;
      break;
    }

    uint64_t cur = n_completed.load(std::memory_order_relaxed);
    if (cur != last_count) {
      std::cout << "  [" << elapsed << "s] Completed " << cur << " requests"
                << std::endl;
      last_count = cur;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // -- Results ---------------------------------------------------------------

  uint64_t total = n_completed.load();
  int n_dec = decoder_ctx.decode_count.load();

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "  Total completed: " << total << std::endl;
  if (n_dec > 0) {
    double avg_us =
        static_cast<double>(decoder_ctx.total_decode_us.load()) / n_dec;
    std::cout << "  Avg PyMatching decode: " << std::fixed
              << std::setprecision(1) << avg_us << " us (" << n_dec
              << " samples)" << std::endl;
  }

  int verified = 0, mismatches = 0, pred_only_mismatches = 0;
  for (int i = 0; i < static_cast<int>(total) && i < MAX_RESULTS; ++i) {
    if (result_corrections[i] < 0)
      continue;

    int32_t total_corr = result_corrections[i];
    int32_t lpred = result_logical_pred[i];
    int32_t pipeline_parity = total_corr % 2;

    std::cout << "  Shot " << i << ": logical_pred=" << lpred
              << " total_corrections=" << total_corr
              << " converged=" << result_converged[i];

    if (!ground_truth_obs.empty() &&
        static_cast<uint32_t>(i) < obs_num_samples) {
      int32_t gt =
          ground_truth_obs[static_cast<size_t>(i) * obs_num_observables];
      bool match = (pipeline_parity == gt);
      bool pred_match = ((lpred % 2) == gt);
      std::cout << " ground_truth=" << gt
                << " pipeline_parity=" << pipeline_parity
                << (match ? " PASS" : " MISMATCH");
      if (!match)
        mismatches++;
      if (!pred_match)
        pred_only_mismatches++;
      verified++;
    }
    std::cout << std::endl;
  }

  if (verified > 0) {
    double ler = static_cast<double>(mismatches) / verified;
    double pred_ler = static_cast<double>(pred_only_mismatches) / verified;
    std::cout << "\n  [Correctness] Verified: " << verified << " shots"
              << std::endl;
    std::cout << "  [Correctness] Pipeline (pred+pymatch) mismatches: "
              << mismatches << "  LER: " << std::setprecision(4) << ler
              << std::endl;
    std::cout << "  [Correctness] Predecoder-only mismatches:         "
              << pred_only_mismatches << "  LER: " << std::setprecision(4)
              << pred_ler << std::endl;
  } else if (total > 0) {
    std::cout << "\n  (No ground-truth data; pass --data-dir= to enable "
                 "correctness checking)"
              << std::endl;
  }

  // -- Shutdown --------------------------------------------------------------

  std::cout << "\n=== Shutting down ===" << std::endl;

  pymatch_queue.shutdown();
  for (auto &t : pymatch_threads)
    if (t.joinable())
      t.join();

  pipeline.stop();

  gpu_roce_close(transceiver);
  if (gpu_roce_thread.joinable())
    gpu_roce_thread.join();
  gpu_roce_destroy_transceiver(transceiver);

  for (auto &s : predecoder_streams) {
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
  }
  cudaFreeHost(h_mailbox_bank);

  std::cout << "*** Bridge shutdown complete ***" << std::endl;
  return 0;
}
