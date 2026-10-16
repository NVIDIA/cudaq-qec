/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Composite TensorRT Decoder Demo
 *
 * This is a software-only trt_decoder demo of test_predecoder_w_pymatching
 * It consumes the same Stim-exported files:
 *   detectors.bin, observables.bin, H_csr.bin, O_csr.bin, priors.bin
 *
 * Instead of manually wiring ai_predecoder_service -> PyMatching, it creates
 *the trt_decoder plugin from an ONNX model and asks it to run PyMatching as the
 * global decoder:
 *
 *   input detectors -> TRT predecoder -> [pre_L, residual syndromes]
 *     -> PyMatching(H, O, priors) -> final logical frame
 *
 * Usage:
 *   test_trt_decoder_composite [d7|d13|d13_r104|d21|d21_r42|d31]
 *       --data-dir DIR [--max-samples=N] [--onnx-path=FILE]
 *       [--engine-save-path=FILE] [--batch-size=N] [--warmup=N]
 *       [--no-cuda-graph]
 *
 *   test_trt_decoder_composite --data-dir DIR --config-yaml FILE
 *       [--decoder-id=N] [--max-samples=N] [--warmup=N]
 ******************************************************************************/

#include "predecoder_pipeline_common.h"
#include "../../lib/realtime/realtime_decoding.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace {

using hrclock = std::chrono::high_resolution_clock;

struct DemoConfig {
  std::string data_dir;
  std::string config_yaml_path;
  std::string onnx_path;
  std::string engine_save_path;
  int64_t decoder_id = -1;
  int max_samples = 0;
  int warmup_count = 20;
  size_t batch_size = 1;
  bool use_cuda_graph = true;
};

bool starts_with(const std::string &s, const std::string &prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::string value_after_equals(const std::string &arg,
                               const std::string &prefix) {
  return arg.substr(prefix.size());
}

bool parse_bool(const std::string &v) {
  if (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON" ||
      v == "yes" || v == "YES")
    return true;
  if (v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF" ||
      v == "no" || v == "NO")
    return false;
  throw std::runtime_error("Expected boolean value, got '" + v + "'");
}

bool file_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

std::string replace_extension(const std::string &path,
                              const std::string &new_ext) {
  auto slash = path.find_last_of('/');
  auto dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
    return path + new_ext;
  return path.substr(0, dot) + new_ext;
}

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [d7|d13|d13_r104|d21|d21_r42|d31] --data-dir DIR [options]\n\n"
      << "Options:\n"
      << "  --data-dir DIR             Directory with "
         "detectors/observables/H/O\n"
      << "  --max-samples=N            Limit samples decoded (0 = all)\n"
      << "  --warmup=N                 Samples excluded from latency stats\n"
      << "  --config-yaml=FILE         Build composite decoder from YAML "
         "config\n"
      << "  --decoder-id=N             Decoder ID to select from YAML config\n"
      << "  --onnx-path=FILE           Override full ONNX path\n"
      << "  --engine-save-path=FILE    Where the built TRT engine is saved\n"
      << "  --batch-size=N             TRT dynamic batch profile size (default "
         "1)\n"
      << "  --use-cuda-graph=0|1       Enable CUDA graph executor (default 1)\n"
      << "  --no-cuda-graph            Shorthand for --use-cuda-graph=0\n"
      << "\nPipelineConfig overrides are also accepted:\n"
      << "  --distance=N --num-rounds=N --onnx-filename=FILE --label=NAME\n";
}

DemoConfig parse_demo_config(int argc, char *argv[]) {
  DemoConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--data-dir" && i + 1 < argc) {
      cfg.data_dir = argv[++i];
    } else if (starts_with(arg, "--data-dir=")) {
      cfg.data_dir = value_after_equals(arg, "--data-dir=");
    } else if (starts_with(arg, "--max-samples=")) {
      cfg.max_samples = std::stoi(value_after_equals(arg, "--max-samples="));
    } else if (arg == "--max-samples" && i + 1 < argc) {
      cfg.max_samples = std::stoi(argv[++i]);
    } else if (starts_with(arg, "--warmup=")) {
      cfg.warmup_count = std::stoi(value_after_equals(arg, "--warmup="));
    } else if (arg == "--warmup" && i + 1 < argc) {
      cfg.warmup_count = std::stoi(argv[++i]);
    } else if (starts_with(arg, "--config-yaml=")) {
      cfg.config_yaml_path = value_after_equals(arg, "--config-yaml=");
    } else if (arg == "--config-yaml" && i + 1 < argc) {
      cfg.config_yaml_path = argv[++i];
    } else if (starts_with(arg, "--decoder-id=")) {
      cfg.decoder_id = std::stoll(value_after_equals(arg, "--decoder-id="));
    } else if (arg == "--decoder-id" && i + 1 < argc) {
      cfg.decoder_id = std::stoll(argv[++i]);
    } else if (starts_with(arg, "--onnx-path=")) {
      cfg.onnx_path = value_after_equals(arg, "--onnx-path=");
    } else if (arg == "--onnx-path" && i + 1 < argc) {
      cfg.onnx_path = argv[++i];
    } else if (starts_with(arg, "--engine-save-path=")) {
      cfg.engine_save_path = value_after_equals(arg, "--engine-save-path=");
    } else if (arg == "--engine-save-path" && i + 1 < argc) {
      cfg.engine_save_path = argv[++i];
    } else if (starts_with(arg, "--batch-size=")) {
      cfg.batch_size = static_cast<size_t>(
          std::stoul(value_after_equals(arg, "--batch-size=")));
    } else if (arg == "--batch-size" && i + 1 < argc) {
      cfg.batch_size = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (starts_with(arg, "--use-cuda-graph=")) {
      cfg.use_cuda_graph =
          parse_bool(value_after_equals(arg, "--use-cuda-graph="));
    } else if (arg == "--no-cuda-graph") {
      cfg.use_cuda_graph = false;
    }
  }
  return cfg;
}

std::string read_text_file(const std::string &path) {
  std::ifstream in(path);
  if (!in.good())
    throw std::runtime_error("Failed to open " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

size_t sparse_vec_nnz(const std::vector<int64_t> &sparse) {
  return static_cast<size_t>(std::count_if(sparse.begin(), sparse.end(),
                                           [](int64_t v) { return v >= 0; }));
}

size_t sparse_vec_rows(const std::vector<int64_t> &sparse) {
  return static_cast<size_t>(std::count(sparse.begin(), sparse.end(), -1));
}

std::vector<cudaq::qec::float_t> sample_to_syndrome(const TestData &data,
                                                    int sample_idx) {
  std::vector<cudaq::qec::float_t> syndrome(data.num_detectors);
  const int32_t *sample = data.sample(sample_idx);
  for (uint32_t i = 0; i < data.num_detectors; ++i)
    syndrome[i] = static_cast<cudaq::qec::float_t>(sample[i] != 0 ? 1.0 : 0.0);
  return syndrome;
}

int bit_from_float(cudaq::qec::float_t v) { return v >= 0.5 ? 1 : 0; }

double percentile(const std::vector<double> &sorted, double p) {
  if (sorted.empty())
    return 0.0;
  double idx = (p / 100.0) * (sorted.size() - 1);
  size_t lo = static_cast<size_t>(idx);
  size_t hi = std::min(lo + 1, sorted.size() - 1);
  double frac = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

struct CompositeStats {
  int decoded = 0;
  int converged = 0;
  int missing = 0;
  int first_obs_mismatches = 0;
  int any_obs_mismatches = 0;
  int ground_truth_ones = 0;
  int result_ones = 0;
  std::vector<double> latencies_us;
  double wall_us = 0.0;
};

struct DecoderSetup {
  std::unique_ptr<cudaq::qec::decoder> decoder;
  cudaqx::tensor<uint8_t> H;
  cudaqx::heterogeneous_map trt_params;
  std::string label;
  std::string init_mode;
  std::string config_yaml_path;
  std::string onnx_path;
  std::string engine_save_path;
  size_t H_rows = 0;
  size_t H_cols = 0;
  size_t H_nnz = 0;
  size_t O_rows = 0;
  size_t O_cols = 0;
  size_t O_nnz = 0;
};

CompositeStats run_composite_decoder(cudaq::qec::decoder &decoder,
                                     const TestData &test_data, int n_samples,
                                     size_t num_observables) {
  CompositeStats stats;
  stats.latencies_us.reserve(n_samples);

  const size_t check_observables =
      std::min(num_observables, static_cast<size_t>(test_data.num_observables));

  auto wall_start = std::chrono::steady_clock::now();
  for (int i = 0; i < n_samples; ++i) {
    auto syndrome = sample_to_syndrome(test_data, i);

    auto t0 = hrclock::now();
    auto result = decoder.decode(syndrome);
    auto t1 = hrclock::now();

    stats.latencies_us.push_back(
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            t1 - t0)
            .count());

    if (result.result.size() < check_observables || check_observables == 0) {
      stats.missing++;
      continue;
    }

    stats.decoded++;
    if (result.converged)
      stats.converged++;

    bool any_mismatch = false;
    for (size_t obs = 0; obs < check_observables; ++obs) {
      int pred = bit_from_float(result.result[obs]);
      int truth = test_data.observable(i, static_cast<int>(obs));
      if (obs == 0) {
        stats.ground_truth_ones += truth != 0;
        stats.result_ones += pred != 0;
        if (pred != truth)
          stats.first_obs_mismatches++;
      }
      any_mismatch |= (pred != truth);
    }
    if (any_mismatch)
      stats.any_obs_mismatches++;
  }
  auto wall_end = std::chrono::steady_clock::now();
  stats.wall_us =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
          wall_end - wall_start)
          .count();
  return stats;
}

const cudaq::qec::decoding::config::decoder_config &select_yaml_decoder(
    const cudaq::qec::decoding::config::multi_decoder_config &config,
    int64_t decoder_id) {
  if (decoder_id >= 0) {
    auto it = std::find_if(config.decoders.begin(), config.decoders.end(),
                           [&](const auto &decoder_config) {
                             return decoder_config.id == decoder_id;
                           });
    if (it == config.decoders.end())
      throw std::runtime_error("Decoder ID " + std::to_string(decoder_id) +
                               " not found in YAML config.");
    return *it;
  }

  if (config.decoders.size() != 1)
    throw std::runtime_error("YAML config contains " +
                             std::to_string(config.decoders.size()) +
                             " decoders; pass --decoder-id to select one.");
  return config.decoders.front();
}

DecoderSetup create_decoder_from_yaml(const DemoConfig &demo_cfg) {
  using cudaq::qec::decoding::config::multi_decoder_config;

  auto config = multi_decoder_config::from_yaml_str(
      read_text_file(demo_cfg.config_yaml_path));
  const auto &decoder_config = select_yaml_decoder(config, demo_cfg.decoder_id);
  if (decoder_config.type != "trt_decoder") {
    throw std::runtime_error("YAML decoder type must be trt_decoder, got '" +
                             decoder_config.type + "'.");
  }

  DecoderSetup setup;
  setup.label = "yaml decoder " + std::to_string(decoder_config.id);
  setup.init_mode = "YAML config";
  setup.config_yaml_path = demo_cfg.config_yaml_path;
  setup.H_rows = static_cast<size_t>(decoder_config.syndrome_size);
  setup.H_cols = static_cast<size_t>(decoder_config.block_size);
  setup.H_nnz = sparse_vec_nnz(decoder_config.H_sparse);
  setup.O_rows = sparse_vec_rows(decoder_config.O_sparse);
  setup.O_cols = static_cast<size_t>(decoder_config.block_size);
  setup.O_nnz = sparse_vec_nnz(decoder_config.O_sparse);

  setup.H = cudaq::qec::pcm_from_sparse_vec(decoder_config.H_sparse,
                                            decoder_config.syndrome_size,
                                            decoder_config.block_size);
  auto O = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.O_sparse, setup.O_rows, decoder_config.block_size);
  setup.trt_params =
      cudaq::qec::decoding::host::prepare_decoder_params(decoder_config);
  if (setup.trt_params.contains("onnx_load_path"))
    setup.onnx_path = setup.trt_params.get<std::string>("onnx_load_path");
  if (setup.trt_params.contains("engine_save_path"))
    setup.engine_save_path =
        setup.trt_params.get<std::string>("engine_save_path");
  if (setup.trt_params.contains("engine_load_path") &&
      setup.engine_save_path.empty())
    setup.engine_save_path =
        setup.trt_params.get<std::string>("engine_load_path");

  setup.decoder = cudaq::qec::decoder::get(
      decoder_config.type,
      cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(setup.H),
                               cudaq::qec::sparse_binary_matrix(O)),
      cudaq::qec::decode_result_type::observables, setup.trt_params);
  return setup;
}

DecoderSetup create_decoder_from_cli(const PipelineConfig &config,
                                     const DemoConfig &demo_cfg,
                                     const StimData &stim) {
  std::string onnx_path =
      demo_cfg.onnx_path.empty() ? config.onnx_path() : demo_cfg.onnx_path;
  std::string engine_save_path = demo_cfg.engine_save_path.empty()
                                     ? replace_extension(onnx_path, ".engine")
                                     : demo_cfg.engine_save_path;

  if (!file_exists(onnx_path))
    throw std::runtime_error("ONNX file not found: " + onnx_path);

  auto H = stim.H.to_dense();
  auto O = stim.O.to_dense();

  cudaqx::heterogeneous_map pm_params;
  pm_params.insert("merge_strategy", std::string("smallest_weight"));
  if (!stim.priors.empty()) {
    if (stim.priors.size() != stim.H.ncols) {
      throw std::runtime_error(
          "priors.bin has " + std::to_string(stim.priors.size()) +
          " entries, but H has " + std::to_string(stim.H.ncols) + " columns.");
    }
  }

  DecoderSetup setup;
  setup.H = H;
  setup.label = config.label;
  setup.init_mode = "manual CLI args";
  setup.onnx_path = onnx_path;
  setup.engine_save_path = engine_save_path;
  setup.H_rows = stim.H.nrows;
  setup.H_cols = stim.H.ncols;
  setup.H_nnz = stim.H.nnz;
  setup.O_rows = stim.O.nrows;
  setup.O_cols = stim.O.ncols;
  setup.O_nnz = stim.O.nnz;

  setup.trt_params.insert("onnx_load_path", onnx_path);
  setup.trt_params.insert("engine_save_path", engine_save_path);
  setup.trt_params.insert("batch_size", demo_cfg.batch_size);
  setup.trt_params.insert("use_cuda_graph", demo_cfg.use_cuda_graph);
  setup.trt_params.insert("engine_output_format",
                          "observables_and_residual_detectors");
  setup.trt_params.insert("global_decoder", std::string("pymatching"));
  setup.trt_params.insert("global_decoder_params", pm_params);

  setup.decoder = cudaq::qec::decoder::get(
      "trt_decoder",
      cudaq::qec::decoder_init(cudaq::qec::sparse_binary_matrix(H),
                               cudaq::qec::sparse_binary_matrix(O),
                               stim.priors),
      cudaq::qec::decode_result_type::observables, setup.trt_params);
  return setup;
}

} // namespace

int main(int argc, char *argv[]) {
  std::string config_name = "d7";
  if (argc > 1 && std::string(argv[1]).substr(0, 2) != "--")
    config_name = argv[1];

  if (argc > 1 &&
      (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    print_usage(argv[0]);
    return 0;
  }

  auto config_opt = PipelineConfig::from_name(config_name);
  if (!config_opt) {
    print_usage(argv[0]);
    return 1;
  }

  PipelineConfig config = *config_opt;
  config.apply_cli_overrides(argc, argv);
  DemoConfig demo_cfg = parse_demo_config(argc, argv);

  if (demo_cfg.data_dir.empty()) {
    std::cerr
        << "ERROR: --data-dir is required for composite TRT decoder demo.\n";
    print_usage(argv[0]);
    return 1;
  }
  if (demo_cfg.batch_size < 1) {
    std::cerr << "ERROR: --batch-size must be >= 1.\n";
    return 1;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "ERROR: no CUDA device available.\n";
    return 1;
  }

  TestData test_data = load_test_data(demo_cfg.data_dir);
  if (!test_data.loaded()) {
    std::cerr << "ERROR: failed to load detector/observable test data from "
              << demo_cfg.data_dir << "\n";
    return 1;
  }

  const bool use_yaml_config = !demo_cfg.config_yaml_path.empty();
  std::optional<StimData> stim;
  if (!use_yaml_config) {
    stim = load_stim_data(demo_cfg.data_dir);
    if (!stim->H.loaded()) {
      std::cerr << "ERROR: H_csr.bin is required in " << demo_cfg.data_dir
                << "\n";
      return 1;
    }
    if (!stim->O.loaded()) {
      std::cerr << "ERROR: O_csr.bin is required in " << demo_cfg.data_dir
                << "\n";
      return 1;
    }
    if (stim->O.nrows == 0) {
      std::cerr << "ERROR: O_csr.bin contains zero observables.\n";
      return 1;
    }
  }

  DecoderSetup setup;
  try {
    std::cout << "--- Initializing Composite TensorRT Decoder ("
              << (use_yaml_config ? demo_cfg.config_yaml_path : config.label)
              << ") ---\n";
    setup = use_yaml_config ? create_decoder_from_yaml(demo_cfg)
                            : create_decoder_from_cli(config, demo_cfg, *stim);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: failed to create composite trt_decoder: " << e.what()
              << "\n";
    return 1;
  }

  if (setup.O_rows == 0) {
    std::cerr << "ERROR: observable matrix contains zero observables.\n";
    return 1;
  }
  if (test_data.num_detectors != setup.H_rows) {
    std::cerr << "ERROR: detectors.bin has " << test_data.num_detectors
              << " detectors, but decoder H has " << setup.H_rows << " rows.\n";
    return 1;
  }
  if (test_data.num_observables < setup.O_rows) {
    std::cerr << "ERROR: observables.bin has " << test_data.num_observables
              << " observable column(s), but decoder O has " << setup.O_rows
              << " row(s).\n";
    return 1;
  }
  if (setup.decoder->get_result_type() !=
      cudaq::qec::decode_result_type::observables) {
    std::cerr << "ERROR: composite trt_decoder must use observable output.\n";
    return 1;
  }

  std::cout << "[Setup] Init mode:   " << setup.init_mode << "\n";
  if (!setup.config_yaml_path.empty())
    std::cout << "[Setup] YAML:        " << setup.config_yaml_path << "\n";
  if (!setup.onnx_path.empty())
    std::cout << "[Setup] ONNX:        " << setup.onnx_path << "\n";
  if (!setup.engine_save_path.empty())
    std::cout << "[Setup] Engine:      " << setup.engine_save_path << "\n";
  std::cout << "[Setup] Data dir:    " << demo_cfg.data_dir << "\n";
  std::cout << "[Setup] H:           " << setup.H_rows << " x " << setup.H_cols
            << " (" << setup.H_nnz << " nnz)\n";
  std::cout << "[Setup] O:           " << setup.O_rows << " x " << setup.O_cols
            << " (" << setup.O_nnz << " nnz)\n";
  std::cout << "[Setup] Samples:     " << test_data.num_samples
            << ", detectors/sample=" << test_data.num_detectors
            << ", observables/sample=" << test_data.num_observables << "\n";
  std::cout << "[Setup] PyMatching:  ";
  if (use_yaml_config) {
    std::cout << "from YAML global_decoder_params\n";
  } else {
    std::cout << "merge_strategy=smallest_weight"
              << (stim->priors.empty() ? ", no priors\n" : ", priors loaded\n");
  }

  const int available_samples = static_cast<int>(test_data.num_samples);
  const int n_samples = (demo_cfg.max_samples > 0)
                            ? std::min(demo_cfg.max_samples, available_samples)
                            : available_samples;
  if (n_samples <= 0) {
    std::cerr << "ERROR: no samples selected.\n";
    return 1;
  }

  std::cout << "[Run] Decoding " << n_samples
            << " sample(s) through composite TRT+PyMatching decoder...\n";

  CompositeStats stats =
      run_composite_decoder(*setup.decoder, test_data, n_samples, setup.O_rows);

  int warmup = std::min(demo_cfg.warmup_count,
                        static_cast<int>(stats.latencies_us.size()));
  std::vector<double> steady_latencies(stats.latencies_us.begin() + warmup,
                                       stats.latencies_us.end());
  std::sort(steady_latencies.begin(), steady_latencies.end());
  double mean = 0.0;
  for (double v : steady_latencies)
    mean += v;
  mean = steady_latencies.empty() ? 0.0 : mean / steady_latencies.size();
  double stddev = 0.0;
  for (double v : steady_latencies)
    stddev += (v - mean) * (v - mean);
  stddev = steady_latencies.empty()
               ? 0.0
               : std::sqrt(stddev / steady_latencies.size());
  double throughput =
      stats.wall_us > 0.0
          ? (static_cast<double>(stats.decoded) * 1e6 / stats.wall_us)
          : 0.0;
  double ler = stats.decoded > 0
                   ? static_cast<double>(stats.first_obs_mismatches) /
                         static_cast<double>(stats.decoded)
                   : 0.0;
  double any_obs_ler = stats.decoded > 0
                           ? static_cast<double>(stats.any_obs_mismatches) /
                                 static_cast<double>(stats.decoded)
                           : 0.0;

  std::cout << std::fixed;
  std::cout
      << "\n================================================================\n";
  std::cout << "  Composite TRT Decoder Benchmark: " << setup.label << "\n";
  std::cout
      << "================================================================\n";
  std::cout << "  Submitted:          " << n_samples << "\n";
  std::cout << "  Decoded:            " << stats.decoded << "\n";
  std::cout << "  Missing/malformed:  " << stats.missing << "\n";
  std::cout << "  Converged:          " << stats.converged << "\n";
  std::cout << std::setprecision(1);
  std::cout << "  Wall time:          " << stats.wall_us / 1000.0 << " ms\n";
  std::cout << "  Throughput:         " << throughput << " samples/s\n";
  std::cout
      << "  ---------------------------------------------------------------\n";
  std::cout << "  Latency (us)  [steady-state, " << steady_latencies.size()
            << " samples after " << warmup << " warmup]\n";
  if (!steady_latencies.empty()) {
    std::cout << "    min    = " << std::setw(10) << steady_latencies.front()
              << "\n";
    std::cout << "    p50    = " << std::setw(10)
              << percentile(steady_latencies, 50) << "\n";
    std::cout << "    mean   = " << std::setw(10) << mean << "\n";
    std::cout << "    p90    = " << std::setw(10)
              << percentile(steady_latencies, 90) << "\n";
    std::cout << "    p95    = " << std::setw(10)
              << percentile(steady_latencies, 95) << "\n";
    std::cout << "    p99    = " << std::setw(10)
              << percentile(steady_latencies, 99) << "\n";
    std::cout << "    max    = " << std::setw(10) << steady_latencies.back()
              << "\n";
    std::cout << "    stddev = " << std::setw(10) << stddev << "\n";
  }

  std::cout
      << "  ---------------------------------------------------------------\n";
  std::cout << std::setprecision(4);
  std::cout << "  Correctness [observable 0]:\n";
  std::cout << "    Composite mismatches: " << stats.first_obs_mismatches
            << "  LER: " << ler << "\n";
  std::cout << "    Any-observable mismatches: " << stats.any_obs_mismatches
            << "  rate: " << any_obs_ler << "\n";
  std::cout << "    Composite ones: " << stats.result_ones << "/"
            << stats.decoded << "\n";
  std::cout << "    Ground truth ones: " << stats.ground_truth_ones << "/"
            << stats.decoded << "\n";

  std::cout
      << "================================================================\n";
  std::cout << "Done.\n";
  return 0;
}
