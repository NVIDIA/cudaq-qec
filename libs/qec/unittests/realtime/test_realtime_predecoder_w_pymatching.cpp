/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Hybrid Realtime Pipeline Benchmark with AI Pre-Decoder + PyMatching
 *
 * Uses the RealtimePipeline scaffolding to hide all ring buffer, atomics,
 * and thread management. Application code only provides:
 *   1. GPU stage factory (ai_predecoder_service instances)
 *   2. CPU stage callback (PyMatching decode)
 *   3. Completion callback (timestamp recording)
 *
 * Usage: test_realtime_predecoder_w_pymatching [d7|d13|d13_r104|d21|d31]
 *[rate_us] [duration_s]
 ******************************************************************************/

#include "predecoder_pipeline_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unistd.h>

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================
// Streaming Config
// =============================================================================

struct StreamingConfig {
  int rate_us = 0;
  int duration_s = 5;
  int warmup_count = 20;
  std::string data_dir;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char *argv[]) {
  using hrclock = std::chrono::high_resolution_clock;

  // --- Parse arguments ---
  std::string config_name = "d7";
  StreamingConfig scfg;

  int num_gpus = 1;
  network_typing_override typing_override = network_typing_override::automatic;
  std::string onnx_override;

  // Scan for named flags (can appear anywhere)
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--data-dir" && i + 1 < argc) {
      scfg.data_dir = argv[i + 1];
    } else if (std::string(argv[i]) == "--num-gpus" && i + 1 < argc) {
      num_gpus = std::stoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--typing" && i + 1 < argc) {
      try {
        typing_override = parse_network_typing(argv[i + 1]);
      } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
      }
    } else if (std::string(argv[i]) == "--onnx" && i + 1 < argc) {
      onnx_override = argv[i + 1];
    }
  }
  // Multi-GPU dispatch is not yet supported: the host dispatcher thread
  // does not call cudaSetDevice before cudaGraphLaunch/cudaStreamQuery,
  // causing hangs when workers span multiple devices.
  if (num_gpus > 1) {
    std::cerr << "[WARN] --num-gpus " << num_gpus
              << " requested, but multi-GPU dispatch is not yet supported. "
                 "Clamping to 1.\n";
    num_gpus = 1;
  }
  // Positional: config_name [rate_us] [duration_s]
  if (argc > 1 && std::string(argv[1]).substr(0, 2) != "--")
    config_name = argv[1];
  if (argc > 2 && std::isdigit(argv[2][0]))
    scfg.rate_us = std::stoi(argv[2]);
  if (argc > 3 && std::isdigit(argv[3][0]))
    scfg.duration_s = std::stoi(argv[3]);

  auto config_opt = PipelineConfig::from_name(config_name);
  if (!config_opt) {
    std::cerr << "Usage: " << argv[0]
              << " [d7|d13|d13_r104|d21|d21_r42|d31] [rate_us] [duration_s]\n"
              << "  d7       - distance 7, 7 rounds (default)\n"
              << "  d13      - distance 13, 13 rounds\n"
              << "  d13_r104 - distance 13, 104 rounds\n"
              << "  d21      - distance 21, 21 rounds\n"
              << "  d21_r42  - distance 21, 42 rounds\n"
              << "  d31      - distance 31, 31 rounds\n"
              << "  rate_us    - inter-arrival time in us (0 = open-loop)\n"
              << "  duration_s - test duration in seconds (default: 5)\n"
              << "\nOverride flags (applied after preset):\n"
              << "  --distance=N          QEC code distance\n"
              << "  --num-rounds=N        Syndrome measurement rounds\n"
              << "  --onnx-filename=FILE  ONNX model filename\n"
              << "  --num-predecoders=N   Parallel TRT instances\n"
              << "  --num-workers=N       Pipeline GPU workers\n"
              << "  --num-decode-workers=N  PyMatching threads\n"
              << "  --label=NAME          Config label for reports\n";
    return 1;
  }
  PipelineConfig config = *config_opt;
  config.apply_cli_overrides(argc, argv);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (num_gpus < 1 || num_gpus > device_count) {
    std::cerr << "ERROR: --num-gpus " << num_gpus << " is out of range (1.."
              << device_count << ")\n";
    return 1;
  }

  std::cout << "--- Initializing Hybrid AI Realtime Pipeline (" << config.label
            << ") ---\n";

  // Enable mapped host allocations on all GPUs used.
  for (int g = 0; g < num_gpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  }
  CUDA_CHECK(cudaSetDevice(0));

  // --- Model path ---
  // --onnx=PATH overrides the preset's ONNX filename; the cached engine
  // path is derived from the ONNX path by replacing .onnx with .engine.
  std::string engine_file = config.engine_path();
  std::string onnx_file = config.onnx_path();
  if (!onnx_override.empty()) {
    onnx_file = onnx_override;
    engine_file = onnx_override;
    auto dot = engine_file.rfind(".onnx");
    if (dot != std::string::npos)
      engine_file = engine_file.substr(0, dot) + ".engine";
    std::cout << "[Setup] ONNX override: " << onnx_file << "\n";
  }
  std::string model_path;

  std::ifstream engine_probe(engine_file, std::ios::binary);
  if (engine_probe.good()) {
    engine_probe.close();
    model_path = engine_file;
    std::cout << "[Setup] Loading cached TRT engine: " << engine_file << "\n";
  } else {
    model_path = onnx_file;
    std::cout << "[Setup] Building TRT engines from ONNX: " << onnx_file
              << "\n";
  }

  // --- Create GPU resources (predecoders, streams, mailbox) ---
  // Mailbox is allocated on GPU 0 (the ring buffer device).
  void **h_mailbox_bank = nullptr;
  void **d_mailbox_bank = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank,
                           config.num_predecoders * sizeof(void *),
                           cudaHostAllocMapped));
  std::memset(h_mailbox_bank, 0, config.num_predecoders * sizeof(void *));
  CUDA_CHECK(cudaHostGetDevicePointer(
      reinterpret_cast<void **>(&d_mailbox_bank), h_mailbox_bank, 0));

  std::cout << "[Setup] Capturing " << config.num_predecoders
            << "x AIPreDecoder Graphs across " << num_gpus << " GPU(s)...\n";

  std::vector<cudaStream_t> predecoder_streams(config.num_predecoders);
  std::vector<std::unique_ptr<ai_predecoder_service>> predecoders;
  bool need_save = (model_path == onnx_file);
  for (int i = 0; i < config.num_predecoders; ++i) {
    int gpu = i % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu));

    CUDA_CHECK(cudaStreamCreate(&predecoder_streams[i]));

    std::string save_path = (need_save && i == 0) ? engine_file : "";
    auto pd = std::make_unique<ai_predecoder_service>(
        model_path, d_mailbox_bank + i, 1, save_path, typing_override);

    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));
    pd->capture_graph(capture_stream, false);
    CUDA_CHECK(cudaStreamDestroy(capture_stream));

    std::cout << "[Setup] Predecoder " << i << " (GPU " << gpu
              << "): input_size=" << pd->get_input_size()
              << " output_size=" << pd->get_output_size() << "\n";
    predecoders.push_back(std::move(pd));
  }
  CUDA_CHECK(cudaSetDevice(0));

  // --- Derive dimensions from TRT model bindings ---
  const size_t model_input_bytes = predecoders[0]->get_input_size();
  const size_t model_output_bytes = predecoders[0]->get_output_size();
  const size_t slot_size =
      round_up_pow2(CUDAQ_RPC_HEADER_SIZE + model_input_bytes);

  // Trust the engine: use the actual tensor volumes reported by TRT.
  // This works uniformly across FP16/FP8/INT32/uint8 IO dtypes, whereas
  // the old heuristic "output == input + 1 ? uint8 : int32" misidentifies
  // FP16 IO as int32 and halves the detector count.
  const size_t num_input_detectors = predecoders[0]->get_input_num_elements();
  const size_t num_output_elements = predecoders[0]->get_output_num_elements();
  const size_t input_elem_bytes =
      (num_input_detectors > 0)
          ? (model_input_bytes + num_input_detectors - 1) / num_input_detectors
          : 0;

  std::cout << "[Setup] Model I/O: " << num_input_detectors
            << " input elements ("
            << (model_input_bytes * 8 /
                std::max<size_t>(num_input_detectors, 1))
            << " bits each), " << num_output_elements << " output elements\n";
  std::cout << "[Setup] Input detectors: " << num_input_detectors
            << ", Output elements: " << num_output_elements
            << ", input_elem_bytes=" << input_elem_bytes << "\n";

  const int residual_detectors = static_cast<int>(num_output_elements) - 1;

  std::cout << "[Config] distance=" << config.distance
            << " rounds=" << config.num_rounds
            << " residual_detectors=" << residual_detectors
            << " model_input=" << model_input_bytes
            << " model_output=" << model_output_bytes
            << " slot_size=" << slot_size << "\n";

  // --- Load test data (optional) ---
  TestData test_data;
  StimData stim;
  if (!scfg.data_dir.empty()) {
    test_data = load_test_data(scfg.data_dir);
    if (!test_data.loaded()) {
      std::cerr << "ERROR: Failed to load test data from " << scfg.data_dir
                << "\n";
      return 1;
    }
    if (test_data.num_detectors != num_input_detectors) {
      std::cerr << "ERROR: detector count mismatch: data has "
                << test_data.num_detectors << " but model expects "
                << num_input_detectors << "\n";
      return 1;
    }
    stim = load_stim_data(scfg.data_dir);
  }

  // --- Build PyMatching decoder ---
  DecoderContext decoder_ctx;
  decoder_ctx.num_residual_detectors = residual_detectors;
  decoder_ctx.num_input_detectors = static_cast<int>(num_input_detectors);
  cudaqx::heterogeneous_map pm_params;
  pm_params.insert("merge_strategy", std::string("smallest_weight"));

  // Observable row from O matrix (for projecting edge corrections → logical)
  std::vector<uint8_t> obs_row;

  if (stim.H.loaded() && static_cast<int>(stim.H.nrows) == residual_detectors) {
    decoder_ctx.use_full_H = true;
    std::cout << "[Setup] Converting sparse H (" << stim.H.nrows << "x"
              << stim.H.ncols << ") to dense tensor...\n";
    auto H_full = stim.H.to_dense();
    std::cout << "[Setup] H tensor: [" << H_full.shape()[0] << " x "
              << H_full.shape()[1] << "]\n";

    auto O = cudaq::qec::sparse_binary_matrix::from_csr(
        0, stim.H.ncols, std::vector<std::uint32_t>{0}, {});
    if (stim.O.loaded()) {
      obs_row = stim.O.row_dense(0);
      O = cudaq::qec::sparse_binary_matrix(stim.O.to_dense());
    }
    auto rates = !stim.priors.empty() && stim.priors.size() == stim.H.ncols
                     ? stim.priors
                     : std::vector<double>{};
    auto inputs =
        cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H_full),
                                 std::move(O), std::move(rates));

    std::cout << "[Setup] Creating " << config.num_decode_workers
              << " PyMatching decoders (full H)...\n";
    for (int i = 0; i < config.num_decode_workers; ++i)
      decoder_ctx.decoders.push_back(cudaq::qec::decoder::get(
          "pymatching", inputs, cudaq::qec::decode_result_type::errors,
          pm_params));
  } else {
    // Fallback: per-slice decode with CUDA-Q surface code H_z
    std::cout << "[Setup] Creating PyMatching decoder (d=" << config.distance
              << " surface code, Z stabilizers)...\n";
    auto surface_code =
        cudaq::qec::get_code("surface_code", {{"distance", config.distance}});
    auto H_z = surface_code->get_parity_z();

    const int z_stabilizers = static_cast<int>(H_z.shape()[0]);
    if (residual_detectors > 0 && residual_detectors % z_stabilizers == 0)
      decoder_ctx.spatial_slices = residual_detectors / z_stabilizers;
    decoder_ctx.z_stabilizers = z_stabilizers;

    std::cout << "[Setup] H_z shape: [" << H_z.shape()[0] << " x "
              << H_z.shape()[1]
              << "], spatial_slices=" << decoder_ctx.spatial_slices << "\n";

    std::cout << "[Setup] Creating " << config.num_decode_workers
              << " PyMatching decoders (per-slice)...\n";
    for (int i = 0; i < config.num_decode_workers; ++i)
      decoder_ctx.decoders.push_back(
          cudaq::qec::decoder::get("pymatching", H_z, pm_params));
  }
  std::cout << "[Setup] PyMatching decoder pool ready.\n";

  // Pre-launch DMA contexts
  std::vector<PreLaunchCopyCtx> pre_launch_ctxs(config.num_predecoders);
  for (int i = 0; i < config.num_predecoders; ++i) {
    pre_launch_ctxs[i].d_trt_input = predecoders[i]->get_trt_input_ptr();
    pre_launch_ctxs[i].input_size = predecoders[i]->get_input_size();
    pre_launch_ctxs[i].h_ring_ptrs = predecoders[i]->get_host_ring_ptrs();
  }

  if (config.num_workers != config.num_predecoders) {
    std::cerr << "[WARN] num_workers (" << config.num_workers
              << ") != num_predecoders (" << config.num_predecoders
              << "); pipeline workers should match predecoders for 1:1 poll\n";
  }

  // Worker contexts (per-worker, application-specific)
  std::vector<WorkerCtx> worker_ctxs(config.num_workers);
  for (int i = 0; i < config.num_workers; ++i) {
    worker_ctxs[i].predecoder = predecoders[i].get();
    worker_ctxs[i].decoder_ctx = &decoder_ctx;
  }

  // Build function table for RPC dispatch — all workers share a single
  // function_id so the dispatcher can pick any idle worker (no HOL blocking).
  const uint32_t shared_fid = rt_sdk::fnv1a_hash("predecode");
  std::vector<uint32_t> function_ids(config.num_workers, shared_fid);

  // =========================================================================
  // Per-slot output buffers (predecoder output copied here before release)
  // =========================================================================
  // Predecoder workers copy GPU output into deferred_outputs[slot], then
  // PyMatching workers read from it.  No lock is needed because the slot's
  // tx_flags stays IN_FLIGHT until complete_deferred() is called after
  // decoding, so the consumer cannot recycle the slot in between.

  std::vector<std::vector<uint8_t>> deferred_outputs(
      NUM_SLOTS, std::vector<uint8_t>(model_output_bytes));

  PyMatchQueue pymatch_queue;

  // =========================================================================
  // Create pipeline (all atomics hidden inside)
  // =========================================================================

  rt_pipeline::pipeline_stage_config stage_cfg;
  stage_cfg.num_workers = config.num_workers;
  stage_cfg.num_slots = NUM_SLOTS;
  stage_cfg.slot_size = slot_size;
  stage_cfg.cores = {.dispatcher = 2, .consumer = 4, .worker_base = 10};

  rt_pipeline::realtime_pipeline pipeline(stage_cfg);

  // Wire ring buffer base pointers into pre-launch contexts so the H2D
  // copy callback can derive the host pointer from the device pointer the
  // dispatcher passes.
  auto rb_bases = pipeline.ringbuffer_bases();
  for (int i = 0; i < config.num_predecoders; ++i) {
    pre_launch_ctxs[i].rx_data_dev_base = rb_bases.rx_data_dev;
    pre_launch_ctxs[i].rx_data_host_base = rb_bases.rx_data_host;
  }

  // --- GPU stage factory ---
  pipeline.set_gpu_stage([&](int w) -> rt_pipeline::gpu_worker_resources {
    return {.graph_exec = predecoders[w]->get_executable_graph(),
            .stream = predecoder_streams[w],
            .pre_launch_fn = pre_launch_input_copy,
            .pre_launch_data = &pre_launch_ctxs[w],
            .function_id = function_ids[w],
            .user_context = &worker_ctxs[w]};
  });

  // --- CPU stage callback (poll GPU + copy + enqueue to PyMatch queue) ---
  // Predecoder workers only poll GPU completion, copy the output to a
  // per-slot buffer, release the predecoder, and enqueue a PyMatchJob.
  // Returns DEFERRED_COMPLETION so the pipeline releases the worker
  // (idle_mask) without signaling slot completion (tx_flags).
  pipeline.set_cpu_stage(
      [&deferred_outputs, &pymatch_queue, out_sz = model_output_bytes](
          const rt_pipeline::cpu_stage_context &ctx) -> size_t {
        auto *wctx = static_cast<WorkerCtx *>(ctx.user_context);
        auto *pd = wctx->predecoder;
        auto *dctx = wctx->decoder_ctx;

        pre_decoder_job job;
        if (!pd->poll_next_job(job))
          return 0;

        NVTX_PUSH("PredecoderPoll");

        int origin_slot = ctx.origin_slot;

        std::memcpy(deferred_outputs[origin_slot].data(), job.inference_data,
                    out_sz);

        // Syndrome density: count nonzero in input and output residuals
        const uint8_t *input_u8 =
            static_cast<const uint8_t *>(job.ring_buffer_ptr) +
            CUDAQ_RPC_HEADER_SIZE;
        int input_nz = 0;
        for (int k = 0; k < dctx->num_input_detectors; ++k)
          input_nz += (input_u8[k] != 0);
        const uint8_t *out_buf = deferred_outputs[origin_slot].data();
        int output_nz = 0;
        for (int k = 0; k < dctx->num_residual_detectors; ++k)
          output_nz += (out_buf[1 + k] != 0);
        dctx->total_input_nonzero.fetch_add(input_nz,
                                            std::memory_order_relaxed);
        dctx->total_output_nonzero.fetch_add(output_nz,
                                             std::memory_order_relaxed);

        pd->release_job(job.slot_idx);

        auto *rpc_hdr =
            static_cast<const rt_sdk::RPCHeader *>(job.ring_buffer_ptr);
        uint32_t rid = rpc_hdr->request_id;

        pymatch_queue.push({origin_slot, rid, job.ring_buffer_ptr});

        NVTX_POP(); // PredecoderPoll
        return rt_pipeline::DEFERRED_COMPLETION;
      });

  // --- Completion callback (record timestamps) ---
  const int max_requests = 500000;
  std::vector<hrclock::time_point> submit_ts(max_requests);
  std::vector<hrclock::time_point> complete_ts(max_requests);
  std::vector<uint8_t> completed(max_requests, 0);
  std::vector<int32_t> decode_corrections(max_requests, -1);
  std::vector<int32_t> decode_logical_pred(max_requests, -1);

  pipeline.set_completion_handler([&](const rt_pipeline::completion &c) {
    if (c.request_id < static_cast<uint64_t>(max_requests)) {
      complete_ts[c.request_id] = hrclock::now();
      completed[c.request_id] = c.success;
    }
  });

  // =========================================================================
  // Start pipeline and run producer
  // =========================================================================

  for (int i = 0; i < config.num_workers; ++i) {
    worker_ctxs[i].decode_corrections = decode_corrections.data();
    worker_ctxs[i].decode_logical_pred = decode_logical_pred.data();
    worker_ctxs[i].max_requests = max_requests;
    if (!obs_row.empty()) {
      worker_ctxs[i].obs_row = obs_row.data();
      worker_ctxs[i].obs_row_size = obs_row.size();
    }
  }

  // =========================================================================
  // PyMatching thread pool (decoupled from predecoder workers)
  // =========================================================================

  std::vector<std::thread> pymatch_threads(config.num_decode_workers);
  for (int t = 0; t < config.num_decode_workers; ++t) {
    pymatch_threads[t] = std::thread([&pipeline, &pymatch_queue,
                                      &deferred_outputs, &decoder_ctx,
                                      &decode_corrections, &decode_logical_pred,
                                      &obs_row, max_requests]() {
      PyMatchJob job;
      while (pymatch_queue.pop(job)) {
        NVTX_PUSH("PyMatchDecode");
        using hrclock = std::chrono::high_resolution_clock;
        auto decode_start = hrclock::now();

        const uint8_t *output_u8 = deferred_outputs[job.origin_slot].data();
        const int32_t logical_pred = output_u8[0];
        int total_corrections = 0;
        bool all_converged = true;

#if !defined(DISABLE_PYMATCHING)
        const uint8_t *residual_u8 = output_u8 + 1;
        auto *my_decoder = decoder_ctx.acquire_decoder();

        if (decoder_ctx.use_full_H) {
          thread_local cudaqx::tensor<uint8_t> syndrome_tensor(
              {(size_t)decoder_ctx.num_residual_detectors});
          std::memcpy(syndrome_tensor.data(), residual_u8,
                      decoder_ctx.num_residual_detectors);
          auto result = my_decoder->decode(syndrome_tensor);
          all_converged = result.converged;
          if (!obs_row.empty() && !result.result.empty()) {
            if (result.result[0] > 0.5)
              total_corrections++;
          } else {
            for (auto v : result.result)
              if (v > 0.5)
                total_corrections++;
          }
        } else {
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
        }
        total_corrections += logical_pred;
#endif

        auto decode_end = hrclock::now();
        NVTX_POP(); // PyMatchDecode

        // Write RPC response into ring buffer slot
        DecodeResponse resp{total_corrections, all_converged ? 1 : 0};
        char *response_payload =
            (char *)job.ring_buffer_ptr + sizeof(rt_sdk::RPCResponse);
        std::memcpy(response_payload, &resp, sizeof(resp));

        auto *header = static_cast<rt_sdk::RPCResponse *>(job.ring_buffer_ptr);
        header->magic = rt_sdk::RPC_MAGIC_RESPONSE;
        header->status = 0;
        header->result_len = sizeof(resp);

        pipeline.complete_deferred(job.origin_slot);

        auto worker_end = hrclock::now();
        auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             decode_end - decode_start)
                             .count();
        auto worker_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             worker_end - decode_start)
                             .count();
        decoder_ctx.total_decode_us.fetch_add(decode_us,
                                              std::memory_order_relaxed);
        decoder_ctx.total_worker_us.fetch_add(worker_us,
                                              std::memory_order_relaxed);
        decoder_ctx.decode_count.fetch_add(1, std::memory_order_relaxed);

        uint32_t rid = static_cast<uint32_t>(job.request_id);
        if (rid < static_cast<uint32_t>(max_requests)) {
          decode_corrections[rid] = total_corrections;
          decode_logical_pred[rid] = logical_pred;
        }
      }
    });
  }
  std::cout << "[Setup] Started " << config.num_decode_workers
            << " PyMatching decode workers.\n";

  std::cout << "[Setup] Starting pipeline...\n";
  auto injector = pipeline.create_injector();
  pipeline.start();

  auto run_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(scfg.duration_s);

  std::string rate_label =
      (scfg.rate_us > 0) ? std::to_string(scfg.rate_us) + " us" : "open-loop";

  std::cout << "\n[Stream] Starting streaming test (" << config.label << ")\n"
            << "  Rate:       " << rate_label << "\n"
            << "  Duration:   " << scfg.duration_s << " s\n"
            << "  Warmup:     " << scfg.warmup_count << " requests\n"
            << "  Predecoders:" << config.num_predecoders
            << " (dedicated streams)\n"
            << "  Decode workers:" << config.num_decode_workers << "\n"
            << "  Max reqs:   " << max_requests << "\n\n"
            << std::flush;

  // --- Producer loop (runs on main thread) ---
  std::mt19937 rng(42);
  const size_t payload_bytes =
      std::min(model_input_bytes,
               slot_size - static_cast<size_t>(CUDAQ_RPC_HEADER_SIZE));
  std::vector<uint8_t> payload_buf(CUDAQ_RPC_HEADER_SIZE + payload_bytes);
  int req_id = 0;

  auto next_submit_time = hrclock::now();

  while (std::chrono::steady_clock::now() < run_deadline &&
         req_id < max_requests) {

    if (scfg.rate_us > 0) {
      while (hrclock::now() < next_submit_time)
        CUDAQ_REALTIME_CPU_RELAX();
    }

    uint8_t *payload = payload_buf.data() + CUDAQ_RPC_HEADER_SIZE;
    if (test_data.loaded()) {
      const int32_t *src = test_data.sample(req_id);
      for (size_t d = 0; d < num_input_detectors; ++d)
        payload[d] = static_cast<uint8_t>(src[d]);
    } else {
      std::bernoulli_distribution err_dist(0.01);
      for (size_t d = 0; d < num_input_detectors; ++d)
        payload[d] = err_dist(rng) ? 1 : 0;
    }

    submit_ts[req_id] = hrclock::now();
    NVTX_PUSH("ProducerSubmit");
    injector.submit(shared_fid, payload, static_cast<uint32_t>(payload_bytes),
                    static_cast<uint64_t>(req_id));
    NVTX_POP();
    req_id++;

    if (scfg.rate_us > 0)
      next_submit_time += std::chrono::microseconds(scfg.rate_us);
  }

  // --- Shutdown ---
  pipeline.stop();

  pymatch_queue.shutdown();
  for (auto &t : pymatch_threads)
    if (t.joinable())
      t.join();

  // =========================================================================
  // Report
  // =========================================================================

  auto final_stats = pipeline.stats();
  uint64_t nsub = final_stats.submitted;
  uint64_t ncomp = final_stats.completed;

  if (ncomp < nsub)
    std::cerr << "  [WARN] " << (nsub - ncomp)
              << " requests did not complete.\n";

  int warmup = std::min(scfg.warmup_count, static_cast<int>(nsub));
  std::vector<double> latencies;
  latencies.reserve(nsub - warmup);

  for (uint64_t i = warmup; i < nsub; ++i) {
    if (!completed[i])
      continue;
    auto dt =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            complete_ts[i] - submit_ts[i]);
    latencies.push_back(dt.count());
  }

  std::sort(latencies.begin(), latencies.end());

  auto pct = [&](double p) -> double {
    if (latencies.empty())
      return 0;
    double idx = (p / 100.0) * (latencies.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = std::min(lo + 1, latencies.size() - 1);
    double frac = idx - lo;
    return latencies[lo] * (1.0 - frac) + latencies[hi] * frac;
  };

  double mean = 0;
  for (auto v : latencies)
    mean += v;
  mean = latencies.empty() ? 0 : mean / latencies.size();

  double stddev = 0;
  for (auto v : latencies)
    stddev += (v - mean) * (v - mean);
  stddev = latencies.empty() ? 0 : std::sqrt(stddev / latencies.size());

  auto wall_us =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() -
          (run_deadline - std::chrono::seconds(scfg.duration_s)))
          .count();
  double throughput = (wall_us > 0) ? (ncomp * 1e6 / wall_us) : 0;

  double actual_rate = (nsub > 1)
                           ? std::chrono::duration_cast<
                                 std::chrono::duration<double, std::micro>>(
                                 submit_ts[nsub - 1] - submit_ts[0])
                                     .count() /
                                 (nsub - 1)
                           : 0;

  std::cout << std::fixed;
  std::cout
      << "\n================================================================\n";
  std::cout << "  Streaming Benchmark: " << config.label << "\n";
  std::cout
      << "================================================================\n";
  std::cout << "  Submitted:          " << nsub << "\n";
  std::cout << "  Completed:          " << ncomp << "\n";
  std::cout << std::setprecision(1);
  std::cout << "  Wall time:          " << wall_us / 1000.0 << " ms\n";
  std::cout << "  Throughput:         " << throughput << " req/s\n";
  std::cout << "  Actual arrival rate:" << std::setw(8) << actual_rate
            << " us/req\n";
  std::cout << "  Backpressure stalls:" << std::setw(8)
            << final_stats.backpressure_stalls << "\n";
  std::cout
      << "  ---------------------------------------------------------------\n";
  std::cout << "  Latency (us)  [steady-state, " << latencies.size()
            << " requests after " << warmup << " warmup]\n";
  if (!latencies.empty()) {
    std::cout << "    min    = " << std::setw(10) << latencies.front() << "\n";
    std::cout << "    p50    = " << std::setw(10) << pct(50) << "\n";
    std::cout << "    mean   = " << std::setw(10) << mean << "\n";
    std::cout << "    p90    = " << std::setw(10) << pct(90) << "\n";
    std::cout << "    p95    = " << std::setw(10) << pct(95) << "\n";
    std::cout << "    p99    = " << std::setw(10) << pct(99) << "\n";
    std::cout << "    max    = " << std::setw(10) << latencies.back() << "\n";
    std::cout << "    stddev = " << std::setw(10) << stddev << "\n";
  }

  int n_decoded = decoder_ctx.decode_count.load();
  if (n_decoded > 0) {
    double avg_decode = (double)decoder_ctx.total_decode_us.load() / n_decoded;
    double avg_worker = (double)decoder_ctx.total_worker_us.load() / n_decoded;
    double avg_overhead = avg_worker - avg_decode;
    std::cout
        << "  "
           "---------------------------------------------------------------\n";
    std::cout << "  Worker-level averages (" << n_decoded << " completed):\n";
    std::cout << "    PyMatching decode:    " << std::setw(9) << avg_decode
              << " us\n";
    std::cout << "    Total worker:         " << std::setw(9) << avg_worker
              << " us\n";
    std::cout << "    Worker overhead:      " << std::setw(9) << avg_overhead
              << " us\n";
  }
  if (n_decoded > 0) {
    double avg_in_nz =
        (double)decoder_ctx.total_input_nonzero.load() / n_decoded;
    double avg_out_nz =
        (double)decoder_ctx.total_output_nonzero.load() / n_decoded;
    double in_density = avg_in_nz / decoder_ctx.num_input_detectors;
    double out_density = avg_out_nz / decoder_ctx.num_residual_detectors;
    double reduction = (in_density > 0) ? (1.0 - out_density / in_density) : 0;
    std::cout
        << "  "
           "---------------------------------------------------------------\n";
    std::cout << "  Syndrome density (" << n_decoded << " samples):\n";
    std::cout << "    Input:  " << std::fixed << std::setprecision(1)
              << avg_in_nz << " / " << decoder_ctx.num_input_detectors << "  ("
              << std::setprecision(4) << in_density << ")\n";
    std::cout << "    Output: " << std::fixed << std::setprecision(1)
              << avg_out_nz << " / " << decoder_ctx.num_residual_detectors
              << "  (" << std::setprecision(4) << out_density << ")\n";
    std::cout << "    Reduction: " << std::setprecision(1)
              << (reduction * 100.0) << "%\n";
  }

  std::cout
      << "  ---------------------------------------------------------------\n";
  std::cout << "  Host dispatcher processed " << final_stats.dispatched
            << " packets.\n";
  std::cout
      << "================================================================\n";

  // --- Correctness verification (when using real data) ---
  if (test_data.loaded()) {
    int verified = 0, mismatches = 0, missing = 0;
    int pred_only_mismatches = 0;
    int64_t sum_total_corr = 0, sum_logical_pred = 0;
    int nonzero_logical = 0, nonzero_pymatch = 0;
    for (int i = 0; i < nsub; ++i) {
      if (decode_corrections[i] < 0) {
        missing++;
        continue;
      }
      int32_t total_corr = decode_corrections[i];
      int32_t lpred = decode_logical_pred[i];
      int32_t pymatch_corr = total_corr - lpred;
      int32_t pipeline_parity = total_corr % 2;
      int32_t ground_truth = test_data.observable(i, 0);

      if (pipeline_parity != ground_truth)
        mismatches++;
      if ((lpred % 2) != ground_truth)
        pred_only_mismatches++;

      sum_total_corr += total_corr;
      sum_logical_pred += lpred;
      if (lpred != 0)
        nonzero_logical++;
      if (pymatch_corr != 0)
        nonzero_pymatch++;
      verified++;
    }
    double ler =
        (verified > 0) ? static_cast<double>(mismatches) / verified : 0;
    double pred_ler = (verified > 0)
                          ? static_cast<double>(pred_only_mismatches) / verified
                          : 0;
    std::cout << "\n[Correctness] Verified " << verified << "/" << nsub
              << " requests (" << missing << " missing)\n";
    std::cout << "[Correctness] Pipeline (pred+pymatch) mismatches: "
              << mismatches << "  LER: " << std::setprecision(4) << ler << "\n";
    std::cout << "[Correctness] Predecoder-only mismatches:         "
              << pred_only_mismatches << "  LER: " << std::setprecision(4)
              << pred_ler << "\n";
    std::cout << "[Correctness] Avg logical_pred: " << std::setprecision(3)
              << (verified > 0 ? (double)sum_logical_pred / verified : 0)
              << "  nonzero: " << nonzero_logical << "/" << verified << "\n";
    std::cout << "[Correctness] Avg pymatch_corr: " << std::setprecision(3)
              << (verified > 0
                      ? (double)(sum_total_corr - sum_logical_pred) / verified
                      : 0)
              << "  nonzero: " << nonzero_pymatch << "/" << verified << "\n";
    std::cout << "[Correctness] Ground truth ones: ";
    int gt_ones = 0;
    int gt_count = static_cast<int>(
        std::min(nsub, static_cast<uint64_t>(test_data.num_samples)));
    for (int i = 0; i < gt_count; ++i)
      if (test_data.observable(i, 0))
        gt_ones++;
    std::cout << gt_ones << "/" << gt_count << "\n";
  }

  // --- Cleanup ---
  std::cout << "[Teardown] Shutting down...\n";
  for (auto &s : predecoder_streams) {
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
  }
  cudaFreeHost(h_mailbox_bank);

  std::cout << "Done.\n";
  return 0;
}
