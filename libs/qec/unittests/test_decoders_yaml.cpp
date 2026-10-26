/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SessionRegistry.h"
#include "task_graph_test_fixtures.h"
#include "../lib/decoding_task/decoding_task_data_artifacts.h"
#include "../lib/decoding_task/decoding_task_decoding_window.h"
#include "../lib/decoding_task/decoding_task_parallel_window_pattern.h"
#include "../lib/realtime/realtime_decoding.h"
#include "../lib/realtime/window_request_yaml.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace cudaq::qec {

/// Records the measurement-to-detector map exactly as a plugin sees it at
/// construction, so a test can pin what the construction inputs carry.
/// The whole model exactly as the plugin received it, so two construction
/// paths can be compared field by field.
struct captured_model {
  std::vector<std::vector<std::uint32_t>> H;
  std::vector<std::vector<std::uint32_t>> O;
  bool has_observable_model = false;
  std::vector<double> rates;
  std::vector<std::vector<std::uint32_t>> D;
  bool has_d = false;
  bool has_stim_dem = false;
  std::size_t num_detectors = 0;
  std::size_t num_error_mechanisms = 0;
  std::size_t num_observables = 0;

  bool operator==(const captured_model &) const = default;
};

struct construction_d_probe {
  static inline bool has_d = false;
  static inline std::vector<std::vector<std::uint32_t>> rows;
  static inline std::uint32_t num_cols = 0;
  /// Detector syndrome handed to decode(), i.e. D as the realtime path applies
  /// it, so a test can compare that against the construction copy above.
  static inline std::vector<float_t> last_decode_syndrome;

  static inline captured_model model;
};

class d_capture_decoder : public decoder {
public:
  d_capture_decoder(decoder_init inputs, decoder_output_request request,
                    const cudaqx::heterogeneous_map &)
      : decoder(std::move(inputs), request) {
    if (request.has_auxiliary_outputs())
      throw std::invalid_argument(
          "d_capture_decoder does not produce auxiliary outputs");
    const auto &in = get_inputs();
    const auto *D = in.measurement_to_detectors();
    construction_d_probe::has_d = D != nullptr;
    construction_d_probe::rows.clear();
    construction_d_probe::num_cols = 0;
    if (D) {
      construction_d_probe::rows = D->to_nested_csr();
      construction_d_probe::num_cols = D->num_cols();
    }

    captured_model captured;
    captured.H = in.detector_error_matrix().canonicalize().to_nested_csr();
    captured.has_observable_model = in.has_observable_model();
    if (captured.has_observable_model)
      captured.O = in.observable_flips_matrix().canonicalize().to_nested_csr();
    captured.rates = in.error_rates();
    captured.has_d = D != nullptr;
    if (D)
      captured.D = D->to_nested_csr();
    captured.has_stim_dem = in.has_stim_dem();
    captured.num_detectors = in.num_detectors();
    captured.num_error_mechanisms = in.num_error_mechanisms();
    captured.num_observables = in.num_observables();
    construction_d_probe::model = std::move(captured);
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    construction_d_probe::last_decode_syndrome = syndrome;
    return decoder_result{true,
                          std::vector<float_t>(get_num_observables(), 0.0)};
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      d_capture_decoder,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decoder_output_request> output,
          const cudaqx::heterogeneous_map &params) {
        const auto request = output.value_or(
            decoder_output_request{decode_result_type::observables});
        return std::make_unique<d_capture_decoder>(std::move(inputs), request,
                                                   params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(d_capture_decoder)

} // namespace cudaq::qec

namespace {
// A Stim DEM on disk, removed when the test finishes. Two detectors, three
// error mechanisms, one observable.
constexpr const char *kTinyDem = "error(0.1) D0 L0\n"
                                 "error(0.1) D0 D1\n"
                                 "error(0.2) D1\n";

class ScopedDemFile {
public:
  explicit ScopedDemFile(const char *contents = kTinyDem) {
    // GoogleTest binaries run concurrently under ctest, and this file is
    // discovered by more than one target, so a process-local counter alone
    // collides. Qualify by pid.
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("cudaqx_resolver_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter++) + ".dem");
    std::ofstream(path_) << contents;
  }
  ~ScopedDemFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

/// Config carrying only what both model branches need: an id, a type, and a
/// two-row measurement-to-detector map matching the tiny DEM's detectors.
cudaq::qec::decoding::config::decoder_config
make_dem_config(const std::filesystem::path &dem_path) {
  cudaq::qec::decoding::config::decoder_config config;
  config.id = 0;
  config.type = "d_capture_decoder";
  config.stim_dem_path = dem_path.string();
  config.D_sparse = {0, -1, 1, -1};
  return config;
}

class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name(name) {
    if (const char *old = std::getenv(name))
      oldValue = old;
    setenv(name, value, 1);
  }

  ~ScopedEnv() {
    if (oldValue.has_value())
      setenv(name.c_str(), oldValue->c_str(), 1);
    else
      unsetenv(name.c_str());
  }

private:
  std::string name;
  std::optional<std::string> oldValue;
};
} // namespace

namespace cudaq::qec::decoding::simulation {
void enqueue_syndromes(std::uint64_t decoder_id, uint8_t *syndromes,
                       std::uint64_t syndrome_length, std::uint64_t tag);
void get_corrections(std::uint64_t decoder_id, uint8_t *corrections,
                     std::uint64_t correction_length, bool reset);
} // namespace cudaq::qec::decoding::simulation

TEST(DecoderYAMLTest, RejectsParserErrors) {
  const std::string unknown_root_key = R"(
decoders:
  - id: 0
    type: pymatching
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
unexpected: true
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          unknown_root_key),
      std::runtime_error);

  const std::string misspelled_decoder_argument = R"(
decoders:
  - id: 0
    type: pymatching
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      merge_stratgey: smallest_weight
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          misspelled_decoder_argument),
      std::runtime_error);

  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          "decoders: ["),
      std::runtime_error);
}

TEST(DecoderYAMLTest, AcceptsBlockStyleSparseMatrices) {
  const std::string yaml = R"(
decoders:
- id: 0
  type: multi_error_lut
  dispatch: device_graph
  cuda_device_id: 0
  block_size: 2
  syndrome_size: 2
  H_sparse:
  - 0
  - -1
  - 1
  - -1
  O_sparse:
  - 1
  - -1
  D_sparse:
  - 0
  - -1
  - 3
  - -1
transport:
  provider: gpu_roce
  args:
  - --device=mlx5_4
  - --peer-ip=192.168.0.2
)";

  const auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);

  ASSERT_EQ(config.decoders.size(), 1u);
  const auto &decoder = config.decoders.front();
  EXPECT_EQ(decoder.H_sparse, (std::vector<std::int64_t>{0, -1, 1, -1}));
  EXPECT_EQ(decoder.O_sparse, (std::vector<std::int64_t>{1, -1}));
  EXPECT_EQ(decoder.D_sparse, (std::vector<std::int64_t>{0, -1, 3, -1}));
  EXPECT_EQ(config.transport.provider, "gpu_roce");
  EXPECT_EQ(
      config.transport.args,
      (std::vector<std::string>{"--device=mlx5_4", "--peer-ip=192.168.0.2"}));
}

/// Helper function to test that a decoder configuration can be serialized to
/// and from YAML.
void test_decoder_yaml_roundtrip(
    cudaq::qec::decoding::config::multi_decoder_config &multi_config) {
  // Serialize to YAML
  std::string config_str = multi_config.to_yaml_str(200);
  // Deserialize from YAML
  auto multi_config_from_yaml =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          config_str);
  // And now serialize the deserialized configuration back to YAML, just for
  // good measure.
  std::string round_trip_config_str = multi_config_from_yaml.to_yaml_str(200);
  // Validate
  bool matchStrings = round_trip_config_str == config_str;
  bool matchConfigs = multi_config_from_yaml == multi_config;
  EXPECT_TRUE(matchStrings);
  EXPECT_TRUE(matchConfigs);

  // Retain for debug:
  // if (!matchStrings || !matchConfigs) {
  //   std::cout << "Orig config string: " << config_str << std::endl;
  //   std::cout << "Round trip config string: " <<
  //   multi_config_from_yaml.to_yaml_str(200) << std::endl;
  // }
}

/// Helper function to create and finalize a decoder configuration.
void test_decoder_creation(
    cudaq::qec::decoding::config::multi_decoder_config &multi_config) {
  int status = cudaq::qec::decoding::config::configure_decoders(multi_config);
  EXPECT_EQ(status, 0);
  cudaq::qec::decoding::config::finalize_decoders();
}

/// Helper function to create a sample, skeleton test decoder configuration for
/// a single error LUT decoder.
cudaq::qec::decoding::config::decoder_config
create_test_empty_decoder_config(int id) {
  cudaq::qec::decoding::config::decoder_config config;
  config.id = id;
  config.type = "single_error_lut";
  config.block_size = 20;
  config.syndrome_size = 10;
  cudaqx::tensor<uint8_t> H({config.syndrome_size, config.block_size});
  cudaqx::tensor<uint8_t> O({2, config.block_size});
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(H);
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
      config.syndrome_size, 2, /*include_first_round=*/false);
  return config;
}

cudaq::qec::decoding::config::decoder_config
create_test_sample_realtime_decoder_config(int id) {
  auto config = create_test_empty_decoder_config(id);
  config.type = "sample_decoder";
  cudaqx::tensor<uint8_t> O({2, config.block_size});
  O.at({0, 0}) = 1;
  O.at({1, 1}) = 1;
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  return config;
}

/// Helper function to create a sample, skeleton test decoder configuration for
/// the NV-QLDPC decoder.
cudaq::qec::decoding::config::decoder_config
create_test_decoder_config_nv_qldpc(int id) {
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(id);
  config.type = "nv-qldpc-decoder";
  config.error_rate_vec = std::vector<double>(config.block_size, 0.1);

  cudaqx::heterogeneous_map nv_args;
  nv_args.insert("use_sparsity", true);
  nv_args.insert("max_iterations", 50);
  nv_args.insert("use_osd", true);
  nv_args.insert("osd_order", 60);
  nv_args.insert("osd_method", 3);
  nv_args.insert("n_threads", 128);
  nv_args.insert("bp_batch_size", 1);
  nv_args.insert("osd_batch_size", 16);
  nv_args.insert("iter_per_check", 2);
  nv_args.insert("clip_value", 10.0);
  nv_args.insert("bp_method", 3);
  nv_args.insert("scale_factor", 1.0);
  nv_args.insert("proc_float", "fp64");
  nv_args.insert("gamma0", 0.0);
  nv_args.insert("gamma_dist", std::vector<double>{0.1, 0.2});
  cudaqx::heterogeneous_map srelay_args;
  srelay_args.insert("pre_iter", std::size_t{5});
  srelay_args.insert("num_sets", std::size_t{10});
  srelay_args.insert("stopping_criterion", "NConv");
  srelay_args.insert("stop_nconv", std::size_t{10});
  nv_args.insert("srelay_config", srelay_args);
  // explicit_gammas must have num_sets rows (10 in this case)
  nv_args.insert("explicit_gammas",
                 std::vector<std::vector<double>>(
                     10, std::vector<double>(config.block_size, 0.1)));
  nv_args.insert("bp_seed", 42);
  nv_args.insert("composition", 1);
  config.decoder_custom_args = nv_args;

  return config;
}

// The trt_decoder schema is registered by the trt_decoder plugin, which is
// only built when TensorRT is available. YAML paths for trt configs require
// it; typed-struct conversions do not.
bool is_trt_decoder_schema_available() {
  return cudaq::qec::decoding::config::find_decoder_schema("trt_decoder") !=
         nullptr;
}

// The nv-qldpc-decoder schema is registered by the proprietary plugin.
bool is_nv_qldpc_schema_available() {
  return cudaq::qec::decoding::config::find_decoder_schema(
             "nv-qldpc-decoder") != nullptr;
}

bool is_nv_qldpc_decoder_available() {
  try {
    std::size_t block_size = 7;
    std::size_t syndrome_size = 3;
    cudaqx::tensor<uint8_t> H;
    // clang-format off
    std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1,
                                  0, 1, 0, 1, 1, 0, 1,
                                  0, 0, 1, 0, 1, 1, 1};
    // clang-format on
    H.copy(H_vec.data(), {syndrome_size, block_size});

    auto d = cudaq::qec::decoder::get("nv-qldpc-decoder", H);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

TEST(DecoderYAMLTest, SingleDecoder) {
  if (!is_nv_qldpc_decoder_available()) {
    GTEST_SKIP() << "nv-qldpc-decoder is not available";
  }
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_decoder_config_nv_qldpc(0);
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, MultiDecoder) {
  if (!is_nv_qldpc_decoder_available()) {
    GTEST_SKIP() << "nv-qldpc-decoder is not available";
  }
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config1 =
      create_test_decoder_config_nv_qldpc(0);
  cudaq::qec::decoding::config::decoder_config config2 =
      create_test_decoder_config_nv_qldpc(1);
  multi_config.decoders.push_back(config1);
  multi_config.decoders.push_back(config2);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, MultiLUTDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "multi_error_lut";
  cudaqx::heterogeneous_map lut_args;
  lut_args.insert("lut_error_depth", 2);
  config.decoder_custom_args = lut_args;
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, TransportSectionAndMixedDispatch) {
  // The top-level transport section (server-level deployment config,
  // shape-keyed override) and a host + device_graph decoder mix must
  // survive a YAML round trip...
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  auto host_decoder = create_test_empty_decoder_config(0);
  auto dg_decoder = create_test_empty_decoder_config(1);
  dg_decoder.dispatch =
      cudaq::qec::decoding::config::DecoderDispatch::device_graph;
  multi_config.decoders.push_back(host_decoder);
  multi_config.decoders.push_back(dg_decoder);
  multi_config.transport.provider = "udp";
  multi_config.transport.args = {"--num-slots=8"};
  multi_config.transport.device_graph.provider = "gpu_roce";
  multi_config.transport.device_graph.args = {"--pinned-rings"};
  test_decoder_yaml_roundtrip(multi_config);

  // ...and the exact YAML key spelling is part of the contract (a round
  // trip alone cannot catch a symmetric key rename).
  const std::string yaml_text = R"(%YAML 1.2
---
transport:
  provider:      udp
  args:          [--num-slots=8]
  device_graph:
    provider:    gpu_roce
    args:        [--pinned-rings]
decoders:
  - id:            0
    type:          single_error_lut
    block_size:    3
    syndrome_size: 3
    H_sparse:      [0, -1, 1, -1, 2, -1]
    O_sparse:      [0, -1, 1, -1, 2, -1]
    D_sparse:      [0, -1, 1, -1, 2, -1]
  - id:            1
    type:          single_error_lut
    dispatch:      device_graph
    block_size:    3
    syndrome_size: 3
    H_sparse:      [0, -1, 1, -1, 2, -1]
    O_sparse:      [0, -1, 1, -1, 2, -1]
    D_sparse:      [0, -1, 1, -1, 2, -1]
)";
  const auto parsed =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_text);
  EXPECT_EQ(parsed.transport.provider, "udp");
  ASSERT_EQ(parsed.transport.args.size(), 1u);
  EXPECT_EQ(parsed.transport.args[0], "--num-slots=8");
  EXPECT_EQ(parsed.transport.device_graph.provider, "gpu_roce");
  ASSERT_EQ(parsed.transport.device_graph.args.size(), 1u);
  EXPECT_EQ(parsed.transport.device_graph.args[0], "--pinned-rings");
  const auto resolved = parsed.transport.resolve_device_graph();
  EXPECT_EQ(resolved.provider, "gpu_roce");
  ASSERT_EQ(resolved.args.size(), 2u);
  EXPECT_EQ(resolved.args[0], "--num-slots=8");
  EXPECT_EQ(resolved.args[1], "--pinned-rings");
  ASSERT_EQ(parsed.decoders.size(), 2u);
  EXPECT_EQ(parsed.decoders[0].dispatch,
            cudaq::qec::decoding::config::DecoderDispatch::host);
  EXPECT_EQ(parsed.decoders[1].dispatch,
            cudaq::qec::decoding::config::DecoderDispatch::device_graph);
}

TEST(DecoderYAMLTest, SingleLUTDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "single_error_lut";
  config.decoder_custom_args = cudaqx::heterogeneous_map();
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

namespace {

std::string error_rate_location_yaml(bool top_level, bool custom_args) {
  std::string yaml = R"(
decoders:
  - id: 0
    type: single_error_lut
    block_size: 2
    syndrome_size: 2
    H_sparse: [0, -1, 1, -1]
    O_sparse: [0, -1, 1, -1]
    D_sparse: [0, -1, 1, -1]
)";
  if (top_level)
    yaml += "    error_rate_vec: [0.1, 0.2]\n";
  if (custom_args)
    yaml += "    decoder_custom_args:\n"
            "      error_rate_vec: [0.1, 0.2]\n";
  return yaml;
}

} // namespace

TEST(DecoderYAMLTest, LegacyErrorRatesKeepTheEstablishedWireShape) {
  using cudaq::qec::decoding::config::multi_decoder_config;

  auto legacy = multi_decoder_config::from_yaml_str(
      error_rate_location_yaml(/*top_level=*/false, /*custom_args=*/true));
  ASSERT_EQ(legacy.decoders.size(), 1u);
  const auto &config = legacy.decoders.front();
  EXPECT_EQ(config.error_rate_vec, (std::vector<double>{0.1, 0.2}));
  EXPECT_FALSE(config.decoder_custom_args.map().contains("error_rate_vec"));

  const auto emitted = legacy.to_yaml_str(200);
  EXPECT_NE(emitted.find("    decoder_custom_args:\n"
                         "      error_rate_vec:"),
            std::string::npos)
      << emitted;
  EXPECT_EQ(emitted.find("\n    error_rate_vec:"), std::string::npos)
      << emitted;

  auto reparsed = multi_decoder_config::from_yaml_str(emitted);
  EXPECT_EQ(reparsed, legacy);
  EXPECT_EQ(reparsed.to_yaml_str(200), emitted);

  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());
  EXPECT_EQ(inputs.error_rates(), (std::vector<double>{0.1, 0.2}));
  EXPECT_FALSE(config.decoder_custom_args_to_heterogeneous_map().contains(
      "error_rate_vec"));
}

TEST(DecoderYAMLTest, TopLevelErrorRatesNormalizeToEstablishedWireShape) {
  using cudaq::qec::decoding::config::multi_decoder_config;

  auto top_level = multi_decoder_config::from_yaml_str(
      error_rate_location_yaml(/*top_level=*/true, /*custom_args=*/false));
  auto legacy = multi_decoder_config::from_yaml_str(
      error_rate_location_yaml(/*top_level=*/false, /*custom_args=*/true));

  EXPECT_EQ(top_level, legacy);
  EXPECT_EQ(top_level.to_yaml_str(200), legacy.to_yaml_str(200));
}

TEST(DecoderYAMLTest, DuplicateErrorRateLocationsAreRejected) {
  using cudaq::qec::decoding::config::multi_decoder_config;

  try {
    (void)multi_decoder_config::from_yaml_str(
        error_rate_location_yaml(/*top_level=*/true, /*custom_args=*/true));
    ADD_FAILURE() << "expected duplicate error_rate_vec locations to fail";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("both"), std::string::npos)
        << error.what();
    EXPECT_NE(std::string(error.what()).find("decoder_custom_args"),
              std::string::npos)
        << error.what();
  }
}

TEST(DecoderYAMLTest, ProgrammaticLegacyErrorRatesDoNotReachPluginParams) {
  auto config = create_test_empty_decoder_config(0);
  cudaqx::heterogeneous_map args;
  args.insert("error_rate_vec", std::vector<double>{0.1, 0.2});
  config.block_size = 2;
  config.syndrome_size = 2;
  config.H_sparse = {0, -1, 1, -1};
  config.O_sparse = {0, -1, 1, -1};
  config.D_sparse = {0, -1, 1, -1};
  config.decoder_custom_args = args;

  EXPECT_EQ(config.effective_error_rate_vec(), (std::vector<double>{0.1, 0.2}));
  EXPECT_FALSE(config.decoder_custom_args_to_heterogeneous_map().contains(
      "error_rate_vec"));

  config.error_rate_vec = {0.1, 0.2};
  EXPECT_THROW(config.effective_error_rate_vec(), std::runtime_error);
  EXPECT_THROW(config.to_yaml_str(200), std::runtime_error);
}

TEST(DecoderYAMLTest, NestedLegacyErrorRatesAreUniqueModelInput) {
  using cudaq::qec::decoding::config::multi_decoder_config;
  const std::string nested = R"(
decoders:
  - id: 0
    type: sliding_window
    block_size: 2
    syndrome_size: 2
    H_sparse: [0, -1, 1, -1]
    O_sparse: [0, -1, 1, -1]
    D_sparse: [0, -1, 1, -1]
    decoder_custom_args:
      window_size: 1
      inner_decoder_name: single_error_lut
      inner_decoder_params:
        error_rate_vec: [0.1, 0.2]
)";

  auto parsed = multi_decoder_config::from_yaml_str(nested);
  const auto &config = parsed.decoders.front();
  EXPECT_EQ(config.error_rate_vec, (std::vector<double>{0.1, 0.2}));
  auto params = config.decoder_custom_args_to_heterogeneous_map();
  EXPECT_FALSE(params.contains("error_rate_vec"));
  EXPECT_FALSE(params.get<cudaqx::heterogeneous_map>("inner_decoder_params")
                   .contains("error_rate_vec"));

  const auto duplicate = nested.substr(0, nested.find("      window_size:")) +
                         "      error_rate_vec: [0.1, 0.2]\n" +
                         nested.substr(nested.find("      window_size:"));
  EXPECT_THROW(multi_decoder_config::from_yaml_str(duplicate),
               std::runtime_error);
}

cudaq::qec::decoding::config::decoder_config
create_test_decoder_config_trt(int id) {
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(id);
  config.type = "trt_decoder";

  cudaqx::tensor<uint8_t> O({2, config.block_size});
  O.at({0, 1}) = 1;
  O.at({1, 3}) = 1;
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  config.error_rate_vec = std::vector<double>(config.block_size, 0.1);

  cudaqx::heterogeneous_map trt_args;
  trt_args.insert("onnx_load_path", "/tmp/predecoder.onnx");
  trt_args.insert("engine_save_path", "/tmp/predecoder.engine");
  trt_args.insert("precision", "best");
  trt_args.insert("memory_workspace", std::size_t{1ULL << 20});
  trt_args.insert("batch_size", std::size_t{4});
  trt_args.insert("use_cuda_graph", false);
  trt_args.insert("engine_output_format", "observables_and_residual_detectors");
  trt_args.insert("global_decoder", "pymatching");
  cudaqx::heterogeneous_map pymatching_params;
  pymatching_params.insert("merge_strategy", "smallest_weight");
  trt_args.insert("global_decoder_params", pymatching_params);
  config.decoder_custom_args = trt_args;

  return config;
}

TEST(DecoderYAMLTest, TrtDecoderConfigRoundTrip) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(create_test_decoder_config_trt(0));

  test_decoder_yaml_roundtrip(multi_config);
  const auto &args = multi_config.decoders[0].decoder_custom_args.map();
  ASSERT_TRUE(args.contains("global_decoder_params"));
  EXPECT_EQ(args.get<cudaqx::heterogeneous_map>("global_decoder_params")
                .get<std::string>("merge_strategy"),
            "smallest_weight");
}

TEST(DecoderYAMLTest, TrtDecoderConfigToHeterogeneousMap) {
  auto config = create_test_decoder_config_trt(0);
  auto params = config.decoder_custom_args_to_heterogeneous_map();

  EXPECT_EQ(params.get<std::string>("onnx_load_path"), "/tmp/predecoder.onnx");
  EXPECT_EQ(params.get<std::string>("engine_save_path"),
            "/tmp/predecoder.engine");
  EXPECT_EQ(params.get<std::string>("precision"), "best");
  EXPECT_EQ(params.get<std::size_t>("memory_workspace"), 1ULL << 20);
  EXPECT_EQ(params.get<std::size_t>("batch_size"), 4u);
  EXPECT_FALSE(params.get<bool>("use_cuda_graph"));
  EXPECT_EQ(params.get<std::string>("engine_output_format"),
            "observables_and_residual_detectors");
  EXPECT_EQ(params.get<std::string>("global_decoder"), "pymatching");

  auto global_params =
      params.get<cudaqx::heterogeneous_map>("global_decoder_params");
  EXPECT_EQ(global_params.get<std::string>("merge_strategy"),
            "smallest_weight");
  EXPECT_FALSE(global_params.contains("error_rate_vec"));
}

TEST(DecoderYAMLTest, RealtimeParamsDoNotInjectObservableMatrix) {
  auto config = create_test_decoder_config_trt(0);
  auto params = cudaq::qec::decoding::host::prepare_decoder_params(config);

  EXPECT_FALSE(params.contains("O"));
  EXPECT_FALSE(params.contains("error_rate_vec"));

  auto global_params =
      params.get<cudaqx::heterogeneous_map>("global_decoder_params");
  EXPECT_FALSE(global_params.contains("O"));
}

TEST(DecoderYAMLTest, TrtDecoderEmptyGlobalDecoderParams) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  // An explicitly empty global params section round-trips and reaches the
  // realtime decoder params.
  auto config = create_test_decoder_config_trt(0);
  auto args = config.decoder_custom_args.map();
  args.insert("global_decoder_params", cudaqx::heterogeneous_map());
  config.decoder_custom_args = args;

  auto params = config.decoder_custom_args_to_heterogeneous_map();
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_TRUE(
      params.get<cudaqx::heterogeneous_map>("global_decoder_params").empty());

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto yaml = multi_config.to_yaml_str(200);
  EXPECT_NE(yaml.find("global_decoder_params"), std::string::npos);
  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_EQ(round_tripped.to_yaml_str(200), yaml);
  const auto &round_tripped_args =
      round_tripped.decoders[0].decoder_custom_args.map();
  ASSERT_TRUE(round_tripped_args.contains("global_decoder_params"));
  EXPECT_TRUE(
      round_tripped_args.get<cudaqx::heterogeneous_map>("global_decoder_params")
          .empty());

  params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_FALSE(params.contains("O"));

  config.O_sparse.clear();
  params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_FALSE(params.contains("O"));
}

TEST(DecoderYAMLTest, TrtDecoderDefaultGlobalDecoderParams) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  // When the YAML names a global decoder with a registered schema but gives
  // no params, an empty section is materialized on parse.
  const std::string yaml_without_params = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      engine_output_format: residual_detectors
      global_decoder: chromobius
)";
  auto parsed =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_without_params);
  const auto &args = parsed.decoders[0].decoder_custom_args.map();
  ASSERT_TRUE(args.contains("global_decoder_params"));
  EXPECT_TRUE(
      args.get<cudaqx::heterogeneous_map>("global_decoder_params").empty());

  // Emission after materialization is stable.
  const auto emitted = parsed.to_yaml_str(200);
  EXPECT_NE(emitted.find("global_decoder_params"), std::string::npos);
  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped, parsed);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);
}

TEST(DecoderYAMLTest, UnknownTrtGlobalDecoderParamsThrow) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  const std::string yaml_with_unknown_params = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      engine_output_format: residual_detectors
      global_decoder: my_plugin
      global_decoder_params: {}
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_with_unknown_params),
      std::runtime_error);

  // A global decoder without a registered schema is allowed as long as no
  // params section is given (nothing is materialized for it).
  const std::string yaml_without_params = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      engine_output_format: residual_detectors
      global_decoder: my_plugin
)";
  auto parsed =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_without_params);
  const auto &args = parsed.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(args.get<std::string>("global_decoder"), "my_plugin");
  EXPECT_FALSE(args.contains("global_decoder_params"));
}

TEST(DecoderYAMLTest, TrtDecoderParamsWithoutDecoderThrows) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  const std::string yaml_params_without_decoder = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      engine_output_format: errors
      onnx_load_path: /tmp/predecoder.onnx
      global_decoder_params:
        merge_strategy: smallest_weight
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_params_without_decoder),
      std::runtime_error);
}

TEST(DecoderYAMLTest, SlidingWindowDecoder) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);

  // Top-level decoder config
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "sliding_window";
  config.block_size = n_cols;
  config.syndrome_size = n_rows;

  // Sliding window config
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(pcm);
  config.O_sparse =
      cudaq::qec::pcm_to_sparse_vec(cudaqx::tensor<uint8_t>({2, n_cols}));
  config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
      config.syndrome_size, 2, /*include_first_round=*/false);
  config.error_rate_vec = std::vector<double>(config.block_size, 0.1);
  cudaqx::heterogeneous_map sw_args;
  sw_args.insert("window_size", std::size_t{1});
  sw_args.insert("step_size", std::size_t{1});
  sw_args.insert("num_syndromes_per_round", n_syndromes_per_round);
  sw_args.insert("straddle_start_round", false);
  sw_args.insert("straddle_end_round", true);

  // Inner decoder config
  sw_args.insert("inner_decoder_name", "multi_error_lut");
  cudaqx::heterogeneous_map inner_lut_args;
  inner_lut_args.insert("lut_error_depth", 2);
  sw_args.insert("inner_decoder_params", inner_lut_args);
  config.decoder_custom_args = sw_args;

  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, TrtDecoderConfigRoundTripWithoutInstantiation) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config multi_config;
  decoder_config config = create_test_empty_decoder_config(0);
  config.type = "trt_decoder";
  cudaqx::heterogeneous_map trt_args;
  trt_args.insert("engine_load_path", "/tmp/prebuilt.engine");
  trt_args.insert("engine_save_path", "/tmp/saved.engine");
  trt_args.insert("precision", "best");
  trt_args.insert("memory_workspace", std::size_t{1 << 20});
  trt_args.insert("engine_output_format", "errors");
  config.decoder_custom_args = trt_args;
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
}

TEST(DecoderYAMLTest, SlidingWindowInnerDecoderVariantRoundTrips) {
  using namespace cudaq::qec::decoding::config;

  auto check_roundtrip = [](const cudaqx::heterogeneous_map &sw_args) {
    multi_decoder_config multi_config;
    decoder_config config = create_test_empty_decoder_config(0);
    config.type = "sliding_window";
    config.block_size = 6;
    config.syndrome_size = 4;
    cudaqx::tensor<uint8_t> H({config.syndrome_size, config.block_size});
    cudaqx::tensor<uint8_t> O({1, config.block_size});
    config.H_sparse = cudaq::qec::pcm_to_sparse_vec(H);
    config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
    config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, /*include_first_round=*/false);
    config.error_rate_vec = std::vector<double>(config.block_size, 0.1);
    config.decoder_custom_args = sw_args;
    multi_config.decoders.push_back(config);
    test_decoder_yaml_roundtrip(multi_config);
  };

  cudaqx::heterogeneous_map single_lut_sw;
  single_lut_sw.insert("window_size", std::size_t{1});
  single_lut_sw.insert("step_size", std::size_t{1});
  single_lut_sw.insert("num_syndromes_per_round", std::size_t{2});
  single_lut_sw.insert("num_boundary_syndromes", std::size_t{1});
  single_lut_sw.insert("inner_decoder_name", "single_error_lut");
  check_roundtrip(single_lut_sw);

  if (is_nv_qldpc_schema_available()) {
    auto nv_sw = single_lut_sw;
    nv_sw.insert("inner_decoder_name", "nv-qldpc-decoder");
    cudaqx::heterogeneous_map nv_inner;
    nv_inner.insert("max_iterations", 5);
    nv_sw.insert("inner_decoder_params", nv_inner);
    check_roundtrip(nv_sw);
  }
}

TEST(DecoderConfigTest, ConfigureRejectsDuplicateAndNegativeIds) {
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config duplicate_ids;
  duplicate_ids.decoders.push_back(create_test_empty_decoder_config(0));
  duplicate_ids.decoders.push_back(create_test_empty_decoder_config(0));
  EXPECT_EQ(configure_decoders(duplicate_ids), 1);

  multi_decoder_config negative_id;
  negative_id.decoders.push_back(create_test_empty_decoder_config(-1));
  negative_id.decoders.push_back(create_test_empty_decoder_config(0));
  EXPECT_EQ(configure_decoders(negative_id), 3);
}

TEST(DecoderConfigTest, CreateRealtimeDecoderConfiguresRuntimeState) {
  auto config = create_test_sample_realtime_decoder_config(7);

  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      config, cudaq::qec::decoding::host::resolve_decoder_init(
                  config, std::filesystem::current_path()));

  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->get_decoder_id(), 7u);
  EXPECT_EQ(decoder->get_num_observables(), 2u);
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 20u);
}

// A repeated index within a D row cancels under the realtime detector XOR. The
// construction inputs must encode that same rule, or a plugin reading its
// inputs sees a different D from the one the realtime path applies.
TEST(DecoderConfigTest, DuplicateDetectorIndicesCollapseInConstructionInputs) {
  auto config = create_test_empty_decoder_config(0);
  config.type = "d_capture_decoder";
  // Ten non-empty detector rows, as the configuration layer requires. Row 0
  // names measurement 9 twice, which cancels, plus measurement 2, which
  // survives. No other row references measurement 9, so the inferred
  // measurement width stays 10 only if width is taken before cancellation.
  config.D_sparse = {9,  9, 2,  -1, 0,  -1, 1,  -1, 2,  -1, 3,
                     -1, 4, -1, 5,  -1, 6,  -1, 7,  -1, 8,  -1};

  // Round-trip so the fixture is a configuration the server would accept.
  auto parsed = cudaq::qec::decoding::config::decoder_config::from_yaml_str(
      config.to_yaml_str(200));

  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      parsed,
      cudaq::qec::decoding::host::resolve_decoder_init(
          parsed, std::filesystem::current_path()),
      {cudaq::qec::decode_result_type::observables});
  ASSERT_NE(decoder, nullptr);

  // Construction copy: the duplicate pair has cancelled, measurement 2 remains.
  ASSERT_TRUE(cudaq::qec::construction_d_probe::has_d);
  ASSERT_EQ(cudaq::qec::construction_d_probe::rows.size(),
            parsed.syndrome_size);
  EXPECT_EQ(cudaq::qec::construction_d_probe::rows[0],
            std::vector<std::uint32_t>{2});
  EXPECT_EQ(cudaq::qec::construction_d_probe::rows[1],
            std::vector<std::uint32_t>{0});
  // Width survives the cancellation of its only referencing entry.
  EXPECT_EQ(cudaq::qec::construction_d_probe::num_cols, 10u);
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 10u);

  // Realtime application: feed one shot and check the detector syndrome the
  // decoder receives matches the same canonical D.
  cudaq::qec::construction_d_probe::last_decode_syndrome.clear();
  std::vector<uint8_t> measurements(10, 0);
  measurements[2] = 1; // survives in row 0 and row 3
  measurements[9] = 1; // cancelled in row 0, referenced nowhere else
  ASSERT_TRUE(decoder->enqueue_syndrome(measurements));

  const std::vector<cudaq::qec::float_t> expected_detectors = {1, 0, 0, 1, 0,
                                                               0, 0, 0, 0, 0};
  EXPECT_EQ(cudaq::qec::construction_d_probe::last_decode_syndrome,
            expected_detectors);
}

// --- raw Stim DEM model source ---------------------------------------------

TEST(ResolveDecoderInputs, DemSourceCarriesRawProvenanceAndDerivedSizes) {
  ScopedDemFile dem;
  auto config = make_dem_config(dem.path());

  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());

  // The DEM stays authoritative, so a DEM-native decoder can read it back.
  ASSERT_TRUE(inputs.has_stim_dem());
  EXPECT_NE(inputs.stim_dem().find("error(0.1) D0 L0"), std::string::npos);
  // Sizes come from the DEM rather than the configuration.
  EXPECT_EQ(inputs.num_detectors(), 2u);
  EXPECT_EQ(inputs.num_error_mechanisms(), 3u);
  EXPECT_EQ(inputs.num_observables(), 1u);
  // D is orthogonal to the model source and survives resolution.
  ASSERT_NE(inputs.measurement_to_detectors(), nullptr);
  EXPECT_EQ(inputs.measurement_to_detectors()->num_rows(), 2u);
}

TEST(ResolveDecoderInputs, DemSourceRejectsCompetingMatrixKeys) {
  ScopedDemFile dem;
  const std::filesystem::path cwd = std::filesystem::current_path();

  auto with_H = make_dem_config(dem.path());
  with_H.H_sparse = {0, -1, 1, -1};
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(with_H, cwd),
               std::runtime_error);

  auto with_O = make_dem_config(dem.path());
  with_O.O_sparse = {0, -1};
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(with_O, cwd),
               std::runtime_error);

  auto with_rates = make_dem_config(dem.path());
  with_rates.error_rate_vec = {0.1, 0.1, 0.1};
  EXPECT_THROW(
      cudaq::qec::decoding::host::resolve_decoder_init(with_rates, cwd),
      std::runtime_error);
}

TEST(ResolveDecoderInputs, DemSourceRejectsUnreadableFile) {
  auto config = make_dem_config("/nonexistent/definitely-not-here.dem");
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   config, std::filesystem::current_path()),
               std::runtime_error);
}

TEST(ResolveDecoderInputs, DemSourceTreatsSuppliedSizesAsAssertions) {
  ScopedDemFile dem;
  const std::filesystem::path cwd = std::filesystem::current_path();

  // Matching values are accepted.
  auto matching = make_dem_config(dem.path());
  matching.syndrome_size = 2;
  matching.block_size = 3;
  EXPECT_NO_THROW(
      cudaq::qec::decoding::host::resolve_decoder_init(matching, cwd));

  auto wrong_detectors = make_dem_config(dem.path());
  wrong_detectors.syndrome_size = 99;
  EXPECT_THROW(
      cudaq::qec::decoding::host::resolve_decoder_init(wrong_detectors, cwd),
      std::runtime_error);

  auto wrong_mechanisms = make_dem_config(dem.path());
  wrong_mechanisms.block_size = 99;
  EXPECT_THROW(
      cudaq::qec::decoding::host::resolve_decoder_init(wrong_mechanisms, cwd),
      std::runtime_error);
}

TEST(ResolveDecoderInputs, DemSourceResolvesRelativePathAgainstBaseDir) {
  ScopedDemFile dem;
  auto config = make_dem_config(dem.path().filename());
  ASSERT_TRUE(std::filesystem::path(config.stim_dem_path).is_relative());

  // Against the containing directory it resolves; against an unrelated one it
  // does not, which is what makes the base directory meaningful.
  EXPECT_NO_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
      config, dem.path().parent_path()));
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   config, "/definitely/not/the/right/place"),
               std::runtime_error);
}

TEST(ResolveDecoderInputs, MatrixSourceStillRequiresItsDimensions) {
  auto config = create_test_empty_decoder_config(0);
  config.block_size = 0;
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   config, std::filesystem::current_path()),
               std::runtime_error);
}

TEST(ResolveDecoderInputs, MatrixSourceRequiresAnObservableMapping) {
  auto config = create_test_empty_decoder_config(0);
  config.O_sparse.clear();
  // The realtime path returns observable corrections, so a model with no
  // observable mapping cannot serve it. Without this it constructed happily
  // and decoded to a zero-length observable frame.
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   config, std::filesystem::current_path()),
               std::runtime_error);
}

// Acceptance: a DEM-native decoder can be configured and constructed for the
// decoding server straight from a raw Stim DEM, with no decoder-specific
// branch anywhere in the configuration or construction path. Chromobius
// requires the DEM itself -- it throws when handed only matrices -- so this
// only passes if raw provenance survives resolution.
// Acceptance: a decoding-server configuration on disk names its model with a
// path relative to itself, and the server resolves it through its own session
// path. Deliberately goes through SessionRegistry rather than calling the
// resolver directly: an earlier version of this test bypassed the registry and
// therefore missed that the registry resolved every model against the process
// working directory.
// --- configuration lifecycle -----------------------------------------------
//
// Applying a configuration must not damage a working one. Resolution happens
// before any process state is touched, and the configuration is cached and
// published only once it is actually in effect.

TEST(ConfigureDecodersLifecycle, InvalidModelLeavesPriorConfigurationInPlace) {
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config good;
  good.decoders.push_back(create_test_sample_realtime_decoder_config(0));
  ASSERT_EQ(configure_decoders(good), 0);
  const auto cached_after_good = last_configured_multi_decoder_config();
  ASSERT_NE(cached_after_good, nullptr);

  // An unresolvable model: resolution failures propagate as exceptions rather
  // than a status code, and must happen before anything is replaced.
  multi_decoder_config bad;
  auto broken = create_test_sample_realtime_decoder_config(0);
  broken.O_sparse.clear(); // no observable mapping for an observable server
  bad.decoders.push_back(broken);
  EXPECT_THROW(configure_decoders(bad), std::runtime_error);

  // The previously applied configuration is still the cached one, and is not
  // replaced by the configuration that failed to apply.
  const auto cached_after_bad = last_configured_multi_decoder_config();
  ASSERT_NE(cached_after_bad, nullptr);
  EXPECT_EQ(*cached_after_bad, *cached_after_good);

  finalize_decoders();
}

TEST(ConfigureDecodersLifecycle, EmptyLeadingDetectorRowIsRejected) {
  using namespace cudaq::qec::decoding::config;

  // A -1 in first position is an empty detector row: that detector maps to no
  // measurement and would decode as permanently zero. The row-emptiness check
  // once looked only for adjacent -1 pairs, so a leading one reached
  // construction and produced a silently wrong decoder.
  multi_decoder_config config;
  auto leading_empty = create_test_sample_realtime_decoder_config(0);
  auto &d = leading_empty.D_sparse;
  d.erase(d.begin(), std::find(d.begin(), d.end(), -1));
  ASSERT_EQ(d.front(), -1);
  ASSERT_EQ(std::count(d.begin(), d.end(), -1),
            std::count(leading_empty.D_sparse.begin(),
                       leading_empty.D_sparse.end(), -1));
  config.decoders.push_back(leading_empty);
  EXPECT_THROW(configure_decoders(config), std::runtime_error);

  finalize_decoders();
}

TEST(ConfigureDecodersLifecycle, ConstructionFailureIsNotAdvertised) {
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config good;
  good.decoders.push_back(create_test_sample_realtime_decoder_config(0));
  ASSERT_EQ(configure_decoders(good), 0);
  const auto cached_after_good = last_configured_multi_decoder_config();
  ASSERT_NE(cached_after_good, nullptr);

  // Resolves cleanly, then fails in the factory: an unregistered decoder type.
  multi_decoder_config unbuildable;
  auto unknown = create_test_sample_realtime_decoder_config(0);
  unknown.type = "no-such-decoder-is-registered";
  unbuildable.decoders.push_back(unknown);
  EXPECT_NE(configure_decoders(unbuildable), 0);

  // A configuration that never took effect must not be cached or published.
  const auto cached_after_failure = last_configured_multi_decoder_config();
  ASSERT_NE(cached_after_failure, nullptr);
  EXPECT_EQ(*cached_after_failure, *cached_after_good);

  finalize_decoders();
}

// Acceptance 2: a plugin must receive the same model at construction whether it
// is built offline or through the decoding server. This is the test that finds
// divergences between the two construction paths -- the class of defect where
// one path canonicalized a matrix and the other did not, which was invisible
// end to end because only the plugin could see both.
TEST(DecodingServerAcceptance,
     ConstructionInputsAgreeAcrossOfflineAndServerPaths) {
  using namespace cudaq::qec::decoding::config;

  auto config = create_test_empty_decoder_config(0);
  config.type = "d_capture_decoder";
  config.error_rate_vec = std::vector<double>(config.block_size, 0.01);
  // A duplicate index, so the two paths must agree on GF(2) collapse too.
  config.D_sparse = {9,  9, 2,  -1, 0,  -1, 1,  -1, 2,  -1, 3,
                     -1, 4, -1, 5,  -1, 6,  -1, 7,  -1, 8,  -1};

  // Server path: resolve the configuration, then construct through the factory.
  auto server_inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());
  auto server_decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      config, server_inputs);
  ASSERT_NE(server_decoder, nullptr);
  const auto server_model = cudaq::qec::construction_d_probe::model;

  // Offline path: build the model the way an offline caller does, directly from
  // the same matrices. Reusing the handle the server just produced would only
  // prove that an object equals itself.
  auto offline_H = cudaq::qec::pcm_from_sparse_vec(
      config.H_sparse, config.syndrome_size, config.block_size);
  const auto offline_num_obs =
      std::count(config.O_sparse.begin(), config.O_sparse.end(), -1);
  auto offline_O = cudaq::qec::pcm_from_sparse_vec(
      config.O_sparse, offline_num_obs, config.block_size);
  std::vector<std::vector<std::uint32_t>> offline_d_rows;
  {
    std::vector<std::uint32_t> row;
    for (std::int64_t entry : config.D_sparse) {
      if (entry < 0) {
        offline_d_rows.push_back(std::move(row));
        row.clear();
      } else {
        row.push_back(static_cast<std::uint32_t>(entry));
      }
    }
  }
  std::uint32_t offline_measurements = 0;
  for (const auto &r : offline_d_rows)
    for (auto c : r)
      offline_measurements = std::max(offline_measurements, c + 1);
  auto offline_D = cudaq::qec::sparse_binary_matrix::from_nested_csr(
                       static_cast<std::uint32_t>(offline_d_rows.size()),
                       offline_measurements, offline_d_rows)
                       .canonicalize();
  cudaq::qec::decoder_init offline_inputs(
      std::move(offline_H), std::move(offline_O), config.error_rate_vec,
      std::move(offline_D));

  auto offline_decoder =
      cudaq::qec::decoder::get("d_capture_decoder", offline_inputs,
                               cudaq::qec::decode_result_type::observables);
  ASSERT_NE(offline_decoder, nullptr);
  const auto offline_model = cudaq::qec::construction_d_probe::model;

  EXPECT_EQ(server_model, offline_model);
  // And the model is complete, not merely equal: all four fields present.
  EXPECT_FALSE(server_model.H.empty());
  EXPECT_TRUE(server_model.has_observable_model);
  EXPECT_EQ(server_model.rates.size(), config.block_size);
  EXPECT_TRUE(server_model.has_d);
  // The duplicate pair cancelled on both paths.
  EXPECT_EQ(server_model.D[0], std::vector<std::uint32_t>{2});
}

// Acceptance 1: an H-based plugin remains usable offline and through the server
// without any decoder-specific framework change. Uses the same registered
// plugin on both routes and asserts each produces a working decoder.
TEST(DecodingServerAcceptance, MatrixSourcePluginWorksOfflineAndOnServer) {
  using namespace cudaq::qec::decoding::config;

  auto config = create_test_sample_realtime_decoder_config(0);

  // Server route.
  multi_decoder_config multi;
  multi.decoders.push_back(config);
  ASSERT_EQ(configure_decoders(multi), 0);
  finalize_decoders();

  // Offline route, same plugin and the same resolved model.
  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());
  auto offline = cudaq::qec::decoder::get(
      config.type, inputs, cudaq::qec::decode_result_type::errors);
  ASSERT_NE(offline, nullptr);
  auto result = offline->decode(
      std::vector<cudaq::qec::float_t>(config.syndrome_size, 0.0));
  EXPECT_EQ(result.result.size(), config.block_size);
}

TEST(ConfigureDecodersLifecycle,
     AppliedConfigurationStoresAnAbsoluteModelPath) {
  using namespace cudaq::qec::decoding::config;

  const auto root = std::filesystem::temp_directory_path() /
                    ("cudaqx_abs_" + std::to_string(::getpid()));
  std::filesystem::create_directories(root / "configs");
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
  } cleanup{root};
  std::ofstream(root / "configs" / "model.dem") << kTinyDem;

  const auto previous_cwd = std::filesystem::current_path();
  std::filesystem::current_path(root);
  struct RestoreCwd {
    std::filesystem::path path;
    ~RestoreCwd() { std::filesystem::current_path(path); }
  } restore{previous_cwd};

  decoder_config dc;
  dc.id = 0;
  dc.type = "d_capture_decoder";
  dc.stim_dem_path = "model.dem";
  dc.D_sparse = {0, -1, 1, -1};
  multi_decoder_config mc;
  mc.decoders.push_back(dc);

  // A RELATIVE base directory: normalizing the join without absolutizing it
  // would store "configs/model.dem", which stops resolving once the working
  // directory moves.
  ASSERT_EQ(configure_decoders(mc, "configs"), 0);

  EXPECT_TRUE(
      std::filesystem::path(mc.decoders[0].stim_dem_path).is_absolute());
  const auto cached = last_configured_multi_decoder_config();
  ASSERT_NE(cached, nullptr);
  EXPECT_TRUE(
      std::filesystem::path(cached->decoders[0].stim_dem_path).is_absolute());
  EXPECT_TRUE(std::filesystem::exists(cached->decoders[0].stim_dem_path));

  finalize_decoders();
}

TEST(ConfigureDecodersLifecycle, FailedResolutionLeavesCallerConfigUnmodified) {
  using namespace cudaq::qec::decoding::config;

  ScopedDemFile dem;
  const auto base = dem.path().parent_path();

  decoder_config first;
  first.id = 0;
  first.type = "d_capture_decoder";
  first.stim_dem_path = dem.path().filename().string();
  first.D_sparse = {0, -1, 1, -1};

  decoder_config second = first;
  second.id = 1;
  second.stim_dem_path = "definitely-not-present.dem";

  multi_decoder_config mc;
  mc.decoders.push_back(first);
  mc.decoders.push_back(second);
  const std::string original_path = mc.decoders[0].stim_dem_path;

  // The second entry cannot resolve, so nothing is applied -- including the
  // path rewrite on the entry that did resolve. Otherwise a retry against a
  // different base directory would silently keep the first one.
  EXPECT_THROW(configure_decoders(mc, base), std::runtime_error);
  EXPECT_EQ(mc.decoders[0].stim_dem_path, original_path);
}

TEST(ResolveDecoderInputs, DetectorMapIndicesMustBeRepresentable) {
  auto config = create_test_empty_decoder_config(0);
  // Only -1 terminates a row; anything else must be a measurement index that
  // fits the sparse index type. Narrowing would alias onto a real measurement.
  config.D_sparse = {
      static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()), -1};
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   config, std::filesystem::current_path()),
               std::runtime_error);

  auto negative = create_test_empty_decoder_config(0);
  negative.D_sparse = {-2, -1};
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   negative, std::filesystem::current_path()),
               std::runtime_error);
}

TEST(DecodingServerAcceptance,
     ServerLoadsFileRelativeDemThroughSessionRegistry) {
  if (cudaq::qec::decoding::config::find_decoder_schema("chromobius") ==
      nullptr)
    GTEST_SKIP() << "chromobius plugin not built in this configuration";

  // A config directory holding both the document and its model, so the model
  // is findable only by resolving relative to the document.
  const auto dir = std::filesystem::temp_directory_path() /
                   ("cudaqx_server_acc_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
  } cleanup{dir};

  std::ofstream(dir / "model.dem") << "error(0.1) D0 L0\n"
                                      "error(0.1) D0 D1 L1\n"
                                      "error(0.1) D1 L2\n"
                                      "detector(0, 0, 0, 0) D0\n"
                                      "detector(0, 0, 0, 1) D1\n";

  cudaq::qec::decoding::config::decoder_config config;
  config.id = 0;
  config.type = "chromobius";
  config.stim_dem_path = "model.dem"; // relative to the document, not the CWD
  config.D_sparse = {0, -1, 1, -1};
  cudaq::qec::decoding::config::multi_decoder_config multi;
  multi.decoders.push_back(config);
  const auto config_path = dir / "decoders.yml";
  std::ofstream(config_path) << multi.to_yaml_str(200);

  // Run from somewhere else entirely, so a CWD-relative resolution fails.
  const auto previous_cwd = std::filesystem::current_path();
  std::filesystem::current_path(std::filesystem::temp_directory_path());
  struct RestoreCwd {
    std::filesystem::path path;
    ~RestoreCwd() { std::filesystem::current_path(path); }
  } restore{previous_cwd};

  // The model is genuinely unreachable from the working directory, so this
  // fixture fails unless the registry resolves against the document.
  EXPECT_THROW(cudaq::qec::decoding::host::resolve_decoder_init(
                   config, std::filesystem::current_path()),
               std::runtime_error);

  cudaq::qec::decoding_server::SessionRegistry registry;
  ASSERT_NO_THROW(registry.load_from_config(config_path.string()));
  EXPECT_NO_THROW((void)registry.get(0));
}

TEST(DecodingServerAcceptance, ChromobiusConstructsFromRawDemSource) {
  if (cudaq::qec::decoding::config::find_decoder_schema("chromobius") ==
      nullptr)
    GTEST_SKIP() << "chromobius plugin not built in this configuration";

  // The reference case from quantumlib/chromobius: two detectors carrying
  // colour coordinates, three error mechanisms, three observables.
  ScopedDemFile dem("error(0.1) D0 L0\n"
                    "error(0.1) D0 D1 L1\n"
                    "error(0.1) D1 L2\n"
                    "detector(0, 0, 0, 0) D0\n"
                    "detector(0, 0, 0, 1) D1\n");

  cudaq::qec::decoding::config::decoder_config config;
  config.id = 0;
  config.type = "chromobius";
  config.stim_dem_path = dem.path().string();
  config.D_sparse = {0, -1, 1, -1};

  // Round-trip so this is provably a configuration the server would accept.
  auto parsed = cudaq::qec::decoding::config::decoder_config::from_yaml_str(
      config.to_yaml_str(200));
  EXPECT_EQ(parsed.stim_dem_path, config.stim_dem_path);
  EXPECT_TRUE(parsed.H_sparse.empty());

  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      parsed, std::filesystem::current_path());
  ASSERT_TRUE(inputs.has_stim_dem());

  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      parsed, std::move(inputs));
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->get_num_observables(), 3u);
  EXPECT_EQ(decoder->get_result_type(),
            cudaq::qec::decode_result_type::observables);

  // Decoding works off the DEM-derived detector basis, and returns one entry
  // per observable the DEM declares.
  ASSERT_EQ(decoder->get_syndrome_size(), 2u);
  auto result = decoder->decode(
      std::vector<cudaq::qec::float_t>(decoder->get_syndrome_size(), 0.0));
  EXPECT_EQ(result.result.size(), 3u);
}

TEST(DecoderConfigTest, CreateRealtimeDecoderRequiresDetectorMatrix) {
  auto config = create_test_sample_realtime_decoder_config(0);
  config.D_sparse.clear();

  EXPECT_THROW(cudaq::qec::decoding::host::create_realtime_decoder(
                   config, cudaq::qec::decoding::host::resolve_decoder_init(
                               config, std::filesystem::current_path())),
               std::runtime_error);
}

TEST(DecoderConfigTest, CreateRealtimeDecoderRejectsUnrepresentableId) {
  auto config = create_test_sample_realtime_decoder_config(0);
  config.id =
      static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;

  EXPECT_THROW(cudaq::qec::decoding::host::create_realtime_decoder(
                   config, cudaq::qec::decoding::host::resolve_decoder_init(
                               config, std::filesystem::current_path())),
               std::invalid_argument);
}

TEST(DecoderConfigTest, SessionRegistryUsesConfiguredRealtimeDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config config;
  auto decoder_config = create_test_sample_realtime_decoder_config(0);
  config.decoders.push_back(std::move(decoder_config));

  cudaq::qec::decoding_server::SessionRegistry registry;
  registry.load_from_config(config, "unit test");

  const auto &decoder = registry.get(0).dec;
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->get_decoder_id(), 0u);
  EXPECT_EQ(decoder->get_num_observables(), 2u);
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 20u);
}

TEST(DecoderConfigTest, SessionRegistryRejectsMissingDetectorMatrix) {
  cudaq::qec::decoding::config::multi_decoder_config config;
  auto decoder_config = create_test_sample_realtime_decoder_config(0);
  decoder_config.D_sparse.clear();
  config.decoders.push_back(std::move(decoder_config));

  cudaq::qec::decoding_server::SessionRegistry registry;
  EXPECT_THROW(registry.load_from_config(config, "unit test"),
               std::runtime_error);
}

TEST(DecoderConfigTest, SessionRegistryRejectsNegativeDecoderId) {
  cudaq::qec::decoding::config::multi_decoder_config config;
  auto decoder_config = create_test_sample_realtime_decoder_config(-1);
  config.decoders.push_back(std::move(decoder_config));

  cudaq::qec::decoding_server::SessionRegistry registry;
  EXPECT_THROW(registry.load_from_config(config, "unit test"),
               std::runtime_error);
}

TEST(DecoderConfigTest, ConfigureFromFileWithDebugLogging) {
  using namespace cudaq::qec::decoding::config;

  ScopedEnv debugEnv("CUDAQ_QEC_DEBUG_DECODER", "1");

  multi_decoder_config multi_config;
  multi_config.decoders.push_back(create_test_empty_decoder_config(0));
  const auto path =
      std::filesystem::temp_directory_path() / "cudaq_qec_decoders.yaml";
  {
    std::ofstream out(path);
    out << multi_config.to_yaml_str(200);
  }

  EXPECT_EQ(configure_decoders_from_file(path.c_str()), 0);
  finalize_decoders();
  std::filesystem::remove(path);
}

TEST(DecoderConfigTest, ConfigureFromMissingFileReturnsError) {
  using namespace cudaq::qec::decoding::config;

  // Missing config files should return the documented nonzero status instead
  // of attempting to parse an empty or invalid YAML payload.
  const auto missing_path = std::filesystem::temp_directory_path() /
                            "cudaq_qec_missing_decoders.yaml";
  std::filesystem::remove(missing_path);
  EXPECT_EQ(configure_decoders_from_file(missing_path.c_str()), 1);
}

TEST(DecoderSchemaTest, ThirdPartySchemaRegistrationEnablesCustomArgs) {
  using namespace cudaq::qec::decoding::config;

  // A third-party decoder plugin registers a parameter schema (normally from
  // a static initializer in its own shared library); the YAML layer then
  // accepts and round-trips its decoder_custom_args with no framework
  // changes.
  register_decoder_schema({"third_party_demo_engine",
                           {
                               {"gain", param_kind::f64},
                           }});
  register_decoder_schema(
      {"third_party_demo_decoder",
       {
           {"strength", param_kind::f64},
           {"passes", param_kind::int32},
           {"mode", param_kind::string, /*required=*/true},
           {"weights", param_kind::f64_vec},
           {"engine", param_kind::string},
           {"engine_params", param_kind::discriminated, false, "", "engine",
            /*materialize_empty=*/true},
       }});

  const std::string yaml = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      strength: 1.5
      passes: 3
      mode: fast
      weights: [0.25, 0.75]
      engine: third_party_demo_engine
)";
  auto config = multi_decoder_config::from_yaml_str(yaml);
  const auto &args = config.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(args.get<double>("strength"), 1.5);
  EXPECT_EQ(args.get<int>("passes"), 3);
  EXPECT_EQ(args.get<std::string>("mode"), "fast");
  EXPECT_EQ(args.get<std::vector<double>>("weights"),
            (std::vector<double>{0.25, 0.75}));
  // The discriminated engine_params section is materialized (empty) because
  // "engine" names a registered schema and materialize_empty is set.
  ASSERT_TRUE(args.contains("engine_params"));
  EXPECT_TRUE(args.get<cudaqx::heterogeneous_map>("engine_params").empty());

  const auto emitted = config.to_yaml_str(200);
  auto round_tripped = multi_decoder_config::from_yaml_str(emitted);
  EXPECT_EQ(round_tripped, config);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);

  // Unknown keys are rejected against the schema.
  const std::string misspelled = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      strenght: 1.5
      mode: fast
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(misspelled),
               std::runtime_error);

  // Missing required keys are rejected when the section is present.
  const std::string missing_required = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      strength: 1.5
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(missing_required),
               std::runtime_error);

  // A populated discriminated section round-trips, and one that names an
  // unregistered schema is rejected.
  const std::string with_engine_params = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      mode: fast
      engine: third_party_demo_engine
      engine_params:
        gain: 2.5
)";
  auto engine_config = multi_decoder_config::from_yaml_str(with_engine_params);
  const auto &engine_args = engine_config.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(engine_args.get<cudaqx::heterogeneous_map>("engine_params")
                .get<double>("gain"),
            2.5);
  auto engine_round_tripped =
      multi_decoder_config::from_yaml_str(engine_config.to_yaml_str(200));
  EXPECT_EQ(engine_round_tripped, engine_config);

  const std::string unknown_engine = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      mode: fast
      engine: engine_without_schema
      engine_params: {}
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(unknown_engine),
               std::runtime_error);
}

TEST(DecoderSchemaTest, CustomArgsForUnregisteredTypeThrow) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: decoder_without_registered_schema
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      anything: 1
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml),
      std::runtime_error);
}

TEST(DecoderSchemaTest, ExamplePluginRegistersSchema) {
  // The in-tree example decoder plugin registers a (parameter-less) schema
  // from its own shared library; its presence here proves the end-to-end
  // plugin registration path works.
  EXPECT_NE(cudaq::qec::decoding::config::find_decoder_schema(
                "single_error_lut_example"),
            nullptr);
}

TEST(DecoderSchemaTest, ValidateCustomArgsChecksProgrammaticMaps) {
  using namespace cudaq::qec::decoding::config;

  // Maps built programmatically (or from Python dicts) never pass through the
  // YAML parser, so validate_custom_args applies the same schema checks
  // explicitly.
  register_decoder_schema({"third_party_demo_engine",
                           {
                               {"gain", param_kind::f64},
                           }});
  register_decoder_schema(
      {"third_party_demo_decoder",
       {
           {"strength", param_kind::f64},
           {"passes", param_kind::int32},
           {"mode", param_kind::string, /*required=*/true},
           {"weights", param_kind::f64_vec},
           {"engine", param_kind::string},
           {"engine_params", param_kind::discriminated, false, "", "engine",
            /*materialize_empty=*/true},
       }});

  decoder_config config;
  config.type = "third_party_demo_decoder";
  cudaqx::heterogeneous_map args;
  args.insert("strength", 1.5);
  args.insert("mode", std::string("fast"));
  config.decoder_custom_args = args;
  EXPECT_NO_THROW(config.validate_custom_args());

  // Unknown key.
  args.insert("strenght", 1.5);
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Missing required key.
  cudaqx::heterogeneous_map missing_mode;
  missing_mode.insert("strength", 1.5);
  config.decoder_custom_args = missing_mode;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Nested discriminated sections are validated with the schema named by the
  // discriminator.
  cudaqx::heterogeneous_map engine_params;
  engine_params.insert("gain", 2.5);
  cudaqx::heterogeneous_map with_engine;
  with_engine.insert("mode", std::string("fast"));
  with_engine.insert("engine", std::string("third_party_demo_engine"));
  with_engine.insert("engine_params", engine_params);
  config.decoder_custom_args = with_engine;
  EXPECT_NO_THROW(config.validate_custom_args());

  engine_params.insert("gian", 2.5);
  with_engine.insert("engine_params", engine_params);
  config.decoder_custom_args = with_engine;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Unregistered decoder types reject non-empty args (and accept empty ones).
  decoder_config unregistered;
  unregistered.type = "decoder_without_registered_schema";
  EXPECT_NO_THROW(unregistered.validate_custom_args());
  cudaqx::heterogeneous_map anything;
  anything.insert("anything", 1);
  unregistered.decoder_custom_args = anything;
  EXPECT_THROW(unregistered.validate_custom_args(), std::runtime_error);

  // multi_decoder_config validates every decoder.
  multi_decoder_config multi;
  multi.decoders.push_back(unregistered);
  EXPECT_THROW(multi.validate_custom_args(), std::runtime_error);
}

// get<std::size_t> reads a stored int, long or short, so a negative one would
// read as a huge count and pass any minimum. A uint64 parameter rejects it
// instead, whichever signed type holds it, and still takes a non-negative one.
TEST(DecoderSchemaTest, Uint64ParamsRejectNegativeIntegers) {
  using namespace cudaq::qec::decoding::config;
  const auto with_count = [](auto count) {
    cudaqx::heterogeneous_map args;
    args.insert("count", count);
    return args;
  };
  const auto rejects = [](const cudaqx::heterogeneous_map &args) {
    try {
      validate_custom_args("uint64_sign_demo", args);
      ADD_FAILURE() << "a negative count validated";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(
                    "Key 'count' in 'uint64_sign_demo' parameters does not "
                    "hold a non-negative int"),
                std::string::npos)
          << e.what();
    }
  };
  register_decoder_schema({"uint64_sign_demo",
                           {
                               {"count", param_kind::uint64},
                           }});

  rejects(with_count(-1));
  rejects(with_count(-1L));
  rejects(with_count(short{-1}));
  EXPECT_NO_THROW(validate_custom_args("uint64_sign_demo", with_count(0)));
  EXPECT_NO_THROW(validate_custom_args("uint64_sign_demo", with_count(3L)));
  EXPECT_NO_THROW(
      validate_custom_args("uint64_sign_demo", with_count(std::size_t{7})));
}

TEST(DecoderSchemaTest, ProgrammaticConfigsMaterializeSchemaDefaults) {
  using namespace cudaq::qec::decoding::config;

  // Schema-declared defaults (materialize_empty discriminated sections, e.g.
  // trt_decoder's global_decoder_params) must apply to programmatically
  // built configs at the decoder-construction seam, not only on the YAML
  // parse path.
  register_decoder_schema({"third_party_demo_engine",
                           {
                               {"gain", param_kind::f64},
                           }});
  register_decoder_schema(
      {"third_party_demo_decoder",
       {
           {"mode", param_kind::string, /*required=*/true},
           {"engine", param_kind::string},
           {"engine_params", param_kind::discriminated, false, "", "engine",
            /*materialize_empty=*/true},
       }});

  decoder_config config;
  config.type = "third_party_demo_decoder";
  cudaqx::heterogeneous_map args;
  args.insert("mode", std::string("fast"));
  args.insert("engine", std::string("third_party_demo_engine"));
  config.decoder_custom_args = args;

  auto materialized = config.decoder_custom_args_to_heterogeneous_map();
  ASSERT_TRUE(materialized.contains("engine_params"));
  EXPECT_TRUE(
      materialized.get<cudaqx::heterogeneous_map>("engine_params").empty());
  // The stored args are untouched; only the constructor-facing view defaults.
  EXPECT_FALSE(config.decoder_custom_args.map().contains("engine_params"));

  // A decoder type without a registered schema passes its args through.
  decoder_config unregistered;
  unregistered.type = "decoder_without_registered_schema";
  unregistered.decoder_custom_args = args;
  EXPECT_TRUE(custom_args_maps_equal(
      unregistered.decoder_custom_args_to_heterogeneous_map(), args));
}

TEST(DecoderSchemaTest, CustomArgsEqualityIsSignAware) {
  using namespace cudaq::qec::decoding::config;

  // size_t(2^64-1) must not compare equal to int(-1) via wraparound.
  cudaqx::heterogeneous_map a;
  a.insert("seed", std::numeric_limits<std::size_t>::max());
  cudaqx::heterogeneous_map b;
  b.insert("seed", int(-1));
  EXPECT_FALSE(custom_args_maps_equal(a, b));

  // Same-value cross-width comparisons still hold.
  cudaqx::heterogeneous_map c;
  c.insert("seed", std::size_t(7));
  cudaqx::heterogeneous_map d;
  d.insert("seed", int(7));
  EXPECT_TRUE(custom_args_maps_equal(c, d));
  cudaqx::heterogeneous_map e;
  e.insert("seed", int(-1));
  EXPECT_TRUE(custom_args_maps_equal(b, e));
}

TEST(DecoderSchemaTest, SlidingWindowValidateHookRejectsBadWindowing) {
  using namespace cudaq::qec::decoding::config;

  // The sliding_window schema registers a validate hook for the cross-field
  // constraints its per-key specs can't express; the hook runs both when YAML
  // is parsed and from validate_custom_args.
  const std::string yaml_template = R"(
decoders:
  - id: 0
    type: sliding_window
    block_size: 2
    syndrome_size: 2
    H_sparse: [0, -1, 1, -1]
    O_sparse: [0, -1, 1, -1]
    D_sparse: [0, -1, 1, -1]
    error_rate_vec: [0.01, 0.01]
    decoder_custom_args:
      window_size: WINDOW
      step_size: STEP
      inner_decoder_name: single_error_lut
)";
  auto make_yaml = [&](const std::string &window, const std::string &step) {
    std::string yaml = yaml_template;
    yaml.replace(yaml.find("WINDOW"), 6, window);
    yaml.replace(yaml.find("STEP"), 4, step);
    return yaml;
  };

  EXPECT_NO_THROW(multi_decoder_config::from_yaml_str(make_yaml("4", "2")));
  // step_size > window_size
  EXPECT_THROW(multi_decoder_config::from_yaml_str(make_yaml("2", "4")),
               std::runtime_error);
  // step_size == 0
  EXPECT_THROW(multi_decoder_config::from_yaml_str(make_yaml("2", "0")),
               std::runtime_error);

  decoder_config config;
  config.type = "sliding_window";
  config.error_rate_vec = {0.01, 0.01};
  cudaqx::heterogeneous_map args;
  args.insert("window_size", std::size_t(2));
  args.insert("step_size", std::size_t(4));
  args.insert("inner_decoder_name", std::string("single_error_lut"));
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  args.insert("step_size", std::size_t(2));
  config.decoder_custom_args = args;
  EXPECT_NO_THROW(config.validate_custom_args());

  // num_boundary_syndromes must be <= num_syndromes_per_round (the boundary
  // layers can be narrower than the interior, never wider).
  args.insert("num_syndromes_per_round", std::size_t(2));
  args.insert("num_boundary_syndromes", std::size_t(3));
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  args.insert("num_boundary_syndromes", std::size_t(2));
  config.decoder_custom_args = args;
  EXPECT_NO_THROW(config.validate_custom_args());
}

TEST(DecoderSchemaTest, CEntryPointReturnsTheSameSchema) {
  // The C symbol exists for tools that load the library without the Python
  // package; it must hand out the same document as the C++ function.
  using namespace cudaq::qec::decoding::config;
  const char *text = cudaq_qec_decoder_config_json_schema();
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(std::string(text), decoder_config_json_schema());
}

TEST(DecoderSchemaTest, JsonSchemaExportReflectsRegistry) {
  using namespace cudaq::qec::decoding::config;

  // Structural spot checks; the python test suite parses the document and
  // exercises it against real YAML configurations with the jsonschema
  // package.
  const std::string text = decoder_config_json_schema();
  EXPECT_NE(text.find("\"https://json-schema.org/draft/2020-12/schema\""),
            std::string::npos);
  EXPECT_NE(text.find("\"decoder_params\""), std::string::npos);
  EXPECT_NE(text.find("\"decoder_config\""), std::string::npos);
  EXPECT_NE(text.find("\"sparse_matrix\""), std::string::npos);

  // Every registered schema (built-in and plugin-registered alike) has a
  // $defs entry, referenced from the per-type dispatch.
  for (const auto &name : registered_decoder_schema_names()) {
    EXPECT_NE(text.find("\"" + name + "\""), std::string::npos) << name;
    EXPECT_NE(text.find("\"#/$defs/decoder_params/" + name + "\""),
              std::string::npos)
        << name;
  }

  // Required keys and unknown-key rejection are carried over.
  EXPECT_NE(text.find("\"error_rate_vec\""), std::string::npos);
  EXPECT_NE(text.find("\"additionalProperties\": false"), std::string::npos);

  // An int32 parameter carries the width of its stored type, so a document
  // that overflows it fails offline exactly as it fails when parsed.
  EXPECT_NE(text.find("\"maximum\": 2147483647"), std::string::npos);
  EXPECT_NE(text.find("\"minimum\": -2147483648"), std::string::npos);
  // Envelope: id is int64, cuda_device_id a non-negative int32.
  EXPECT_NE(text.find("\"maximum\": 9223372036854775807"), std::string::npos);
  EXPECT_NE(text.find("\"minimum\": -9223372036854775808"), std::string::npos);
}

TEST(DecoderConfigTest, NegativeCudaDeviceIdIsRejectedWhenParsed) {
  using namespace cudaq::qec::decoding::config;
  // decoder construction refuses a negative device id and the exported
  // schema bounds it at 0; the parser must agree with both.
  const std::string yaml = R"(decoders:
  - id: 0
    type: pymatching
    cuda_device_id: -1
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args:
      merge_strategy: smallest_weight
      error_rate_vec: [0.1, 0.1, 0.1]
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(yaml), std::runtime_error);
  std::string ok = yaml;
  ok.replace(ok.find("cuda_device_id: -1"), 18, "cuda_device_id: 0");
  EXPECT_NO_THROW(multi_decoder_config::from_yaml_str(ok));
}

TEST(DecoderConfigTest, SimulationHostPointerWrappersForwardToHostRuntime) {
  using namespace cudaq::qec::decoding::config;

  // The simulation namespace pointer overloads are host trampolines; configure
  // a simple decoder and verify enqueue/get_corrections reaches the host state.
  multi_decoder_config multi_config;
  auto config = create_test_empty_decoder_config(0);
  cudaqx::tensor<uint8_t> O({1, config.block_size});
  O.at({0, 0}) = 1;
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  multi_config.decoders.push_back(config);
  ASSERT_EQ(configure_decoders(multi_config), 0);

  std::vector<uint8_t> syndromes(config.syndrome_size * 2, 0);
  syndromes[0] = 1;
  cudaq::qec::decoding::simulation::enqueue_syndromes(
      /*decoder_id=*/0, syndromes.data(), syndromes.size(), /*tag=*/17);

  std::vector<uint8_t> corrections(1, 0xff);
  cudaq::qec::decoding::simulation::get_corrections(
      /*decoder_id=*/0, corrections.data(), corrections.size(), /*reset=*/true);
  EXPECT_EQ(corrections, (std::vector<uint8_t>{0}));
  finalize_decoders();
}

TEST(DecoderYAMLTest, CudaDeviceIdRoundTrip) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  auto config = create_test_empty_decoder_config(0);
  config.cuda_device_id = 2;
  multi_config.decoders.push_back(config);
  test_decoder_yaml_roundtrip(multi_config);
}

// describe_decoder_type() is the display counterpart to
// effective_decoder_type(): the two agree except on the nodes form, where the
// decoders sit on the nodes and entry-level schema lookup has nothing to
// resolve.
TEST(DecoderYAMLTest, DescribeDecoderTypeNamesEitherTaskGraphForm) {
  using cudaq::qec::decoding::config::decoder_config;

  decoder_config plain;
  plain.type = "pymatching";
  EXPECT_EQ(plain.describe_decoder_type(), "pymatching");

  decoder_config shorthand;
  shorthand.task_graph.emplace();
  shorthand.task_graph->type = "single_window";
  shorthand.task_graph->inner_type = "pymatching";
  EXPECT_EQ(shorthand.describe_decoder_type(), "pymatching");
  EXPECT_EQ(shorthand.effective_decoder_type(), "pymatching");

  decoder_config nodes;
  nodes.task_graph.emplace();
  auto &node_list = nodes.task_graph->nodes;
  node_list.emplace_back();
  node_list.back().id = "x";
  node_list.back().type = "single_window";
  node_list.back().inner_type = "pymatching";
  node_list.emplace_back();
  node_list.back().id = "z";
  node_list.back().type = "single_window";
  node_list.back().inner_type = "nv-qldpc-decoder";
  EXPECT_EQ(nodes.describe_decoder_type(), "pymatching,nv-qldpc-decoder");
  EXPECT_TRUE(nodes.effective_decoder_type().empty());

  // A kind that takes no decoder has no name to contribute, so the kinds
  // themselves are reported rather than an empty string.
  decoder_config decoderless;
  decoderless.task_graph.emplace();
  decoderless.task_graph->nodes.emplace_back();
  decoderless.task_graph->nodes.back().id = "merge";
  decoderless.task_graph->nodes.back().type = "XOR";
  EXPECT_EQ(decoderless.describe_decoder_type(), "xor");

  // An ensemble entry names its members.
  decoder_config raced;
  raced.task_graph.emplace();
  raced.task_graph->nodes.emplace_back();
  auto &ens = raced.task_graph->nodes.back();
  ens.id = "ens";
  ens.ensemble.emplace();
  ens.ensemble->members.resize(2);
  ens.ensemble->members[0].nodes = node_list;
  ens.ensemble->members[1].nodes.emplace_back();
  ens.ensemble->members[1].nodes.back().type = "single_window";
  ens.ensemble->members[1].nodes.back().inner_type = "chromobius";
  EXPECT_EQ(raced.describe_decoder_type(),
            "ensemble(pymatching,nv-qldpc-decoder|chromobius)");
}

TEST(DecoderYAMLTest, PrepareDecoderParamsSurfacesCudaDeviceId) {
  // Non-trt type: the insert must happen before prepare_decoder_params()'s
  // trt-only early return, so the knob reaches every decoder type.
  auto config = create_test_empty_decoder_config(0);
  config.cuda_device_id = 3;
  auto params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  ASSERT_TRUE(params.contains("cuda_device_id"));
  EXPECT_EQ(params.get<int>("cuda_device_id"), 3);

  // Absent -> key absent (decoder::get() treats absence as unpinned).
  auto config2 = create_test_empty_decoder_config(1);
  auto params2 = cudaq::qec::decoding::host::prepare_decoder_params(config2);
  EXPECT_FALSE(params2.contains("cuda_device_id"));

  // trt type: still surfaced on the trt branch. prepare_decoder_params only
  // manipulates the params map (no schema lookup, no filesystem), so empty
  // custom args exercise the trt path without needing the trt plugin.
  auto config3 = create_test_empty_decoder_config(2);
  config3.type = "trt_decoder";
  config3.cuda_device_id = 1;
  auto params3 = cudaq::qec::decoding::host::prepare_decoder_params(config3);
  ASSERT_TRUE(params3.contains("cuda_device_id"));
  EXPECT_EQ(params3.get<int>("cuda_device_id"), 1);
}

TEST(DecoderYAMLTest, ValidateCustomArgsChecksValueKinds) {
  if (!is_nv_qldpc_schema_available())
    GTEST_SKIP() << "nv-qldpc-decoder plugin (and its parameter schema) not "
                    "available";
  // A validated map is guaranteed to serialize: every value must be readable
  // as its schema kind's canonical storage type, not just have a known key.
  using cudaq::qec::decoding::config::decoder_config;

  decoder_config config;
  config.type = "nv-qldpc-decoder";

  cudaqx::heterogeneous_map args;
  args.insert("clip_value", std::string("oops")); // f64 param
  config.decoder_custom_args = args;
  try {
    config.validate_custom_args();
    FAIL() << "expected kind mismatch to be rejected";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("clip_value"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("float"), std::string::npos);
  }

  // A std::size_t stored under an f64 param (the generic conversion used
  // for dicts assigned before `type` is set) is equally unreadable at
  // emission and must be rejected too.
  cudaqx::heterogeneous_map generic;
  generic.insert("clip_value", std::size_t{2});
  config.decoder_custom_args = generic;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Canonically-typed values pass.
  cudaqx::heterogeneous_map good;
  good.insert("clip_value", 2.0);
  good.insert("max_iterations", 50);
  config.decoder_custom_args = good;
  EXPECT_NO_THROW(config.validate_custom_args());
}

TEST(DecoderYAMLTest, TrtFirstEmissionMaterializesGlobalDecoderParams) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  // A programmatic config with only global_decoder set serializes with the
  // defaulted empty global_decoder_params on FIRST emission (as the old
  // typed path did), so emitted YAML is stable across round trips.
  auto config = create_test_empty_decoder_config(0);
  config.type = "trt_decoder";
  cudaqx::heterogeneous_map args;
  args.insert("engine_output_format", std::string("residual_detectors"));
  args.insert("global_decoder", std::string("pymatching"));
  config.decoder_custom_args = args;

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto first = multi_config.to_yaml_str(200);
  EXPECT_NE(first.find("global_decoder_params"), std::string::npos);

  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(first);
  EXPECT_EQ(round_tripped.to_yaml_str(200), first);
}

TEST(DecoderYAMLTest, NonSchemaKeysDroppedFromDecoderParamsAndEmission) {
  // A key outside the registered schema can never round-trip through YAML,
  // so the constructor-facing map must not contain it either: local decoders
  // and remote targets see the same configuration.
  auto config = create_test_empty_decoder_config(0);
  config.type = "multi_error_lut";
  cudaqx::heterogeneous_map args;
  args.insert("lut_error_depth", 2);
  args.insert("not_a_real_param", 42);
  config.decoder_custom_args = args;

  auto params = config.decoder_custom_args_to_heterogeneous_map();
  EXPECT_TRUE(params.contains("lut_error_depth"));
  EXPECT_FALSE(params.contains("not_a_real_param"));

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto yaml = multi_config.to_yaml_str(200);
  EXPECT_NE(yaml.find("lut_error_depth"), std::string::npos);
  EXPECT_EQ(yaml.find("not_a_real_param"), std::string::npos);

  // The stored args are untouched -- only the derived views are filtered.
  EXPECT_TRUE(config.decoder_custom_args.map().contains("not_a_real_param"));
}

// ---------------------------------------------------------------------------
// dem_chunks: named-phase DEM for a repeated-round decomposition
// ---------------------------------------------------------------------------

namespace {

// A simple 1-check, 2-fault code:
//   H: [[f0, f1]] (1 syndrome check)
//   O: [[f0]]     (f0 flips the observable)
// Phases:
//   init:  num_faults=2, next_round seam only (H_sparse=[0,1,-1])
//   bulk:  num_faults=2, both seams   (H_sparse=[0,1,-1])
//   final: num_faults=2, prev_round seam only (H_sparse=[0,1,-1])
std::string dem_chunks_yaml(unsigned num_rounds = 3) {
  return R"(
decoders:
  - id: 0
    type: sample_decoder
    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      num_rounds: )" +
         std::to_string(num_rounds) + R"(
      phases:
        - name: init
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
        - name: bulk
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
        - name: final
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
)";
}

// As above, but every phase carries its own D_sparse rows and the final
// round declares three measurements while its detector reads only one, as a
// data readout whose trailing bits no detector differences.
std::string dem_chunks_yaml_with_trailing_measurements() {
  return R"(
decoders:
  - id: 0
    type: sample_decoder
    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      num_rounds: 3
      measurements_per_round: 1
      phases:
        - name: init
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
            D_sparse: [0, -1]
        - name: bulk
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
            D_sparse: [0, 1, -1]
        - name: final
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
            D_sparse: [0, 1, -1]
            num_measurements: 3
)";
}

cudaq::qec::decoding::config::decoder_config
parse_one(const std::string &yaml) {
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_EQ(config.decoders.size(), 1u);
  return config.decoders.at(0);
}

} // namespace

TEST(DecoderDemChunksYAMLTest, ParsesAllPhases) {
  const auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          dem_chunks_yaml(3));
  ASSERT_EQ(config.decoders.size(), 1u);
  const auto &chunks = config.decoders[0].dem_chunks;
  ASSERT_TRUE(chunks.has_value());
  EXPECT_EQ(chunks->num_rounds, 3u);
  EXPECT_EQ(chunks->phases.size(), 3u);
  EXPECT_EQ(chunks->connections.size(), 3u);
  EXPECT_FALSE(chunks->is_empty());
  EXPECT_TRUE(chunks->has_repeating_phase());
}

TEST(DecoderDemChunksYAMLTest, ParsedPhasesExpandAndClose) {
  const auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          dem_chunks_yaml(4));
  const auto &spec = *config.decoders[0].dem_chunks;

  const auto expanded = cudaq::qec::dem_chunks_from_spec(spec);
  ASSERT_EQ(expanded.size(), 4u);

  const auto flat = cudaq::qec::dem_close_all(expanded);
  // init's prev_round band + 3 inter-chunk boundaries = 4 detectors;
  // 4 rounds × 2 faults = 8 fault columns.
  EXPECT_EQ(flat.detector_error_matrix.shape()[0], 4u);
  EXPECT_EQ(flat.detector_error_matrix.shape()[1], 8u);
  EXPECT_EQ(flat.observables_flips_matrix.shape()[0], 1u);
}

TEST(DecoderDemChunksYAMLTest, SectionIsOptional) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: multi_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_FALSE(config.decoders[0].dem_chunks.has_value());
  EXPECT_EQ(config.to_yaml_str(200).find("dem_chunks"), std::string::npos);
}

// Chunk form: dem_chunks describes the DEM as named phases
TEST(DecoderChunkFormTest, ParsesWithNoFlatMatrices) {
  const auto config = parse_one(dem_chunks_yaml(5));
  EXPECT_TRUE(config.H_sparse.empty());
  EXPECT_TRUE(config.O_sparse.empty());
  EXPECT_TRUE(config.D_sparse.empty());
  EXPECT_EQ(config.block_size, 0u);
  EXPECT_EQ(config.syndrome_size, 0u);
  ASSERT_TRUE(config.dem_chunks.has_value());
  EXPECT_EQ(config.dem_chunks->num_rounds, 5u);
}

TEST(DecoderChunkFormTest, ExpandsToASelfConsistentFlatConfig) {
  auto config = parse_one(dem_chunks_yaml(5));
  const auto closed = cudaq::qec::decoding::config::expand_dem_chunks(config);

  ASSERT_TRUE(closed.has_value());
  EXPECT_GT(config.block_size, 0u);
  EXPECT_GT(config.syndrome_size, 0u);

  const auto count_rows = [](const std::vector<std::int64_t> &sparse) {
    return static_cast<std::uint64_t>(
        std::count(sparse.begin(), sparse.end(), -1));
  };
  EXPECT_EQ(count_rows(config.H_sparse), config.syndrome_size);
  EXPECT_EQ(count_rows(config.O_sparse), 1u);

  // 5 rounds × 2 faults = 10 fault columns
  EXPECT_EQ(config.block_size, 10u);
  EXPECT_EQ(config.error_rate_vec, closed->error_rates);
  // init's prev_round band + 4 inter-chunk boundaries = 5 detectors
  EXPECT_EQ(config.syndrome_size, 5u);

  // Expanded config is flat, so re-expanding is a no-op.
  EXPECT_FALSE(
      cudaq::qec::decoding::config::expand_dem_chunks(config).has_value());
}

TEST(DecoderChunkFormTest, SameConnectionsDifferentRoundCounts) {
  // Two configs with the same phases but different num_rounds should expand
  // to different sizes.
  auto three = parse_one(dem_chunks_yaml(3));
  auto five = parse_one(dem_chunks_yaml(5));

  cudaq::qec::decoding::config::expand_dem_chunks(three);
  cudaq::qec::decoding::config::expand_dem_chunks(five);
  EXPECT_LT(three.syndrome_size, five.syndrome_size);
  EXPECT_EQ(three.block_size, 6u); // 3 rounds × 2 faults
  EXPECT_EQ(five.block_size, 10u); // 5 rounds × 2 faults
}

TEST(DecoderChunkFormTest, DerivedFieldsAreRejected) {
  for (const std::string derived :
       {"    syndrome_size: 3\n", "    block_size: 10\n",
        "    O_sparse: [0, -1]\n", "    D_sparse: [0, -1]\n"}) {
    const std::string yaml =
        R"(
decoders:
  - id: 3
    type: sample_decoder
)" + derived +
        dem_chunks_yaml(3).substr(dem_chunks_yaml(3).find("    dem_chunks:"));
    try {
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected rejection of derived field: " << derived;
    } catch (const std::runtime_error &) {
    }
  }
}

TEST(DecoderChunkFormTest, ExpandedConfigIsItselfAValidFlatConfig) {
  auto config = parse_one(dem_chunks_yaml(3));
  cudaq::qec::decoding::config::expand_dem_chunks(config);

  cudaq::qec::decoding::config::multi_decoder_config wrapper;
  wrapper.decoders.push_back(config);
  const auto emitted = wrapper.to_yaml_str(200);
  EXPECT_NE(emitted.find("H_sparse"), std::string::npos);
  EXPECT_NE(emitted.find("dem_chunks"), std::string::npos);
}

TEST(DecoderChunkFormTest, ChunkFormRoundTripsThroughYAML) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(parse_one(dem_chunks_yaml(5)));
  const auto emitted = multi_config.to_yaml_str(200);
  for (const char *name : {"init", "bulk", "final", "prev_round", "next_round"})
    EXPECT_NE(emitted.find(std::string(" ") + name + "\n"), std::string::npos)
        << name << " in " << emitted;

  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped, multi_config);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);
}

// A spec built in C++ may name its seams and phases with ids never registered.
// They are written as seam:<hex> and must read back as the same ids.
TEST(DecoderChunkFormTest, UnregisteredChunkNamesRoundTripThroughYAML) {
  using cudaq::qec::phase_id;
  using cudaq::qec::seam_id;
  auto config = parse_one(dem_chunks_yaml(5));
  auto &spec = *config.dem_chunks;
  const seam_id out_seam{"test_unregistered_out_seam"};
  const phase_id bulk{"test_unregistered_bulk_phase"};
  ASSERT_EQ(out_seam.name().rfind("seam:", 0), 0u);
  ASSERT_EQ(bulk.name().rfind("seam:", 0), 0u);
  for (auto &phase : spec.phases)
    if (phase.id == cudaq::qec::phase_name::dem_bulk)
      phase.id = bulk;
  for (auto &connection : spec.connections) {
    if (connection.from_phase == cudaq::qec::phase_name::dem_bulk)
      connection.from_phase = bulk;
    if (connection.to_phase == cudaq::qec::phase_name::dem_bulk)
      connection.to_phase = bulk;
  }
  spec.seam.from_seam = out_seam;

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto emitted = multi_config.to_yaml_str(200);
  EXPECT_NE(emitted.find(out_seam.name()), std::string::npos) << emitted;
  EXPECT_NE(emitted.find(bulk.name()), std::string::npos) << emitted;

  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped, multi_config);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);
  EXPECT_EQ(out_seam.name().rfind("seam:", 0), 0u)
      << "reading the placeholder must not register it";
}

// Per-phase measurement maps: one measurement a round, except a final round
// reading two bits that both enter its one detector.
// phases_supply_D_sparse changes how the rows are read, so a config that
// sets it and is emitted without it comes back meaning something else. The
// config layer's parse -> emit -> parse identity is what makes the key
// necessary rather than convenient.
TEST(DecoderDemChunksYAMLTest, PhasesSupplyDSparseSurvivesAYAMLRoundTrip) {
  namespace cfg = cudaq::qec::decoding::config;
  auto yaml = dem_chunks_yaml(3);
  const auto at = yaml.find("      phases:");
  yaml.insert(at, "      measurements_per_round: 1\n"
                  "      phases_supply_D_sparse: false\n");

  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_TRUE(parsed.decoders.at(0).dem_chunks.has_value());
  const auto &spec = *parsed.decoders.at(0).dem_chunks;
  ASSERT_TRUE(spec.phases_supply_D_sparse.has_value());
  EXPECT_FALSE(*spec.phases_supply_D_sparse);
  EXPECT_FALSE(spec.has_D_sparse());

  const auto emitted = parsed.to_yaml_str(200);
  EXPECT_NE(emitted.find("phases_supply_D_sparse"), std::string::npos);
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed);
}

TEST(DecoderDemChunksYAMLTest, PerPhaseMeasurementMapsRoundTripAndExpand) {
  auto yaml = dem_chunks_yaml(3);
  const auto add_map = [&](const std::string &phase, const std::string &map) {
    const auto at = yaml.find("error_rates", yaml.find("name: " + phase));
    const auto eol = yaml.find('\n', at);
    yaml.insert(eol + 1, map);
  };
  add_map("init", "            D_sparse: [0, -1]\n");
  add_map("bulk", "            D_sparse: [0, 1, -1]\n");
  add_map("final", "            num_measurements: 2\n"
                   "            D_sparse: [0, 1, 2, -1]\n");
  const auto at = yaml.find("      phases:");
  yaml.insert(at, "      measurements_per_round: 1\n");

  auto multi =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  const auto emitted = multi.to_yaml_str(200);
  EXPECT_NE(emitted.find("measurements_per_round"), std::string::npos);
  EXPECT_NE(emitted.find("num_measurements"), std::string::npos);
  EXPECT_EQ(cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
                emitted),
            multi);

  auto config = multi.decoders.at(0);
  const auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());
  ASSERT_TRUE(
      cudaq::qec::decoding::config::expand_dem_chunks(config).has_value());
  const std::vector<std::int64_t> want{0, -1, 0, 1, -1, 1, 2, 3, -1};
  EXPECT_EQ(config.D_sparse, want);
  ASSERT_NE(inputs.measurement_to_detectors(), nullptr);
  EXPECT_EQ(inputs.measurement_to_detectors()->num_cols(), 4u);
}

// dem_chunks_yaml() with extra keys on the named phase and a round width.
std::string dem_chunks_yaml_with_readout(const std::string &phase,
                                         const std::string &extra) {
  auto yaml = dem_chunks_yaml(3);
  const auto at = yaml.find("error_rates", yaml.find("name: " + phase));
  yaml.insert(yaml.find('\n', at) + 1, extra);
  yaml.insert(yaml.find("      phases:"), "      measurements_per_round: 1\n");
  return yaml;
}

// m2o_sparse and observable_names round-trip and are in the schema.
TEST(DecoderDemChunksYAMLTest, ReadoutMapAndNamesRoundTripAndAreInTheSchema) {
  namespace cfg = cudaq::qec::decoding::config;
  auto yaml = dem_chunks_yaml_with_readout("final",
                                           "            m2o_sparse: [0, -1]\n");
  yaml.insert(yaml.find("      phases:"), "      observable_names: [Z_L]\n");
  auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  const std::vector<std::int64_t> want{0, -1};
  const std::vector<std::string> names{"Z_L"};
  const auto emitted = parsed.to_yaml_str(200);
  const auto schema = cfg::decoder_config_json_schema();
  auto plain = cfg::multi_decoder_config::from_yaml_str(dem_chunks_yaml(3));

  ASSERT_TRUE(parsed.decoders.at(0).dem_chunks.has_value());
  const auto &spec = *parsed.decoders.at(0).dem_chunks;
  EXPECT_EQ(spec.get_phase(cudaq::qec::phase_name::dem_final).m2o_sparse, want);
  EXPECT_EQ(spec.observable_names, names);
  EXPECT_NE(emitted.find("m2o_sparse"), std::string::npos);
  EXPECT_NE(emitted.find("observable_names"), std::string::npos);
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed);

  EXPECT_NE(schema.find("\"m2o_sparse\""), std::string::npos);
  EXPECT_NE(schema.find("\"observable_names\""), std::string::npos);

  // A spec without the new keys emits neither.
  EXPECT_EQ(plain.to_yaml_str(200).find("m2o_sparse"), std::string::npos);
  EXPECT_EQ(plain.to_yaml_str(200).find("observable_names"), std::string::npos);
} // end - TEST(..., ReadoutMapAndNamesRoundTripAndAreInTheSchema)

// Parsing validates m2o_sparse and observable_names, with or without
// num_rounds.
TEST(DecoderDemChunksYAMLTest, ReadoutMapAndNamesAreCheckedWhenParsed) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const std::string &text) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  const auto streaming = [](std::string yaml) {
    return yaml.substr(0, yaml.find("      num_rounds:")) +
           yaml.substr(yaml.find("      measurements_per_round:"));
  };
  const auto good = dem_chunks_yaml_with_readout(
      "final", "            m2o_sparse: [0, -1]\n");
  auto named = dem_chunks_yaml_with_readout("final", "");

  EXPECT_NO_THROW(cfg::multi_decoder_config::from_yaml_str(good));
  EXPECT_NO_THROW(cfg::multi_decoder_config::from_yaml_str(streaming(good)));
  rejects(dem_chunks_yaml_with_readout("bulk", "            m2o_sparse: "
                                               "[0, -1]\n"),
          "only the last phase");
  rejects(streaming(dem_chunks_yaml_with_readout("bulk", "            "
                                                         "m2o_sparse: "
                                                         "[0, -1]\n")),
          "only the last phase");
  rejects(dem_chunks_yaml_with_readout("final", "            m2o_sparse: "
                                                "[0, -1, 0, -1]\n"),
          "2 m2o_sparse rows for 1 observables");
  rejects(dem_chunks_yaml_with_readout("final", "            m2o_sparse: "
                                                "[1, -1]\n"),
          "m2o_sparse reads measurement 1, past the 1");
  named.insert(named.find("      phases:"),
               "      observable_names: [Z_L, X_L]\n");
  rejects(named, "2 observable_names for 1 observables");
} // end - TEST(..., ReadoutMapAndNamesAreCheckedWhenParsed)

// ---------------------------------------------------------------------------
// Window requests
// ---------------------------------------------------------------------------

// A hand-written parallel-window layer A: c_A = b = 3, c_B = 9.
std::string window_a_yaml() {
  return R"(period: 12
zones: {buffers: [[-3, 0], [3, 6]], commit: [0, 3]}
ports:
  input: {disposition: open, prior: 0.01}
  output: {disposition: open}
sends: [{id: forward}, {id: backward}]
)";
}

// The same layer as a C++ request.
cudaq::qec::window_request window_a() {
  using namespace cudaq::qec;
  window_request a;
  a.period = 12;
  a.commit = round_range{0, 3};
  a.buffers = {{-3, 0}, {3, 6}};
  a.input = {port_disposition::open, 0.01};
  a.output = {port_disposition::open, std::nullopt};
  a.sends = {{"forward", std::nullopt, {}}, {"backward", std::nullopt, {}}};
  return a;
}

TEST(WindowRequestYAML, ParsesThePlansLayerA) {
  EXPECT_EQ(cudaq::qec::window_request_from_yaml_str(window_a_yaml()),
            window_a());
}

// Every range may be written [b, e] or {rounds: [b, e]}, and both build the
// same window.
TEST(WindowRequestYAML, MapFormRangesEqualListForm) {
  using namespace cudaq::qec;
  const std::string list = R"(period: 12
offset: -9
zones: {commit: [0, 9]}
ports:
  input: {disposition: closed}
  output: {disposition: closed, imports: [9, 10]}
sends: [{id: back, mode: residual, rounds: [0, 1]}]
)";
  const std::string map = R"(period: 12
offset: -9
zones: {commit: {rounds: [0, 9]}}
ports:
  input: {disposition: closed}
  output: {disposition: closed, imports: {rounds: [9, 10]}}
sends: [{id: back, mode: residual, rounds: {rounds: [0, 1]}}]
)";
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto from_list = window_request_from_yaml_str(list);
  const auto from_map = window_request_from_yaml_str(map);
  ASSERT_EQ(from_list, from_map);
  EXPECT_EQ(from_list.output.imports, (round_range{9, 10}));
  const auto a = window_from_dem_chunks(tmpl, from_list, 2, 40);
  const auto b = window_from_dem_chunks(tmpl, from_map, 2, 40);
  EXPECT_NO_THROW(check_same_window_up_to_shift(a, b));
} // end - TEST(WindowRequestYAML, MapFormRangesEqualListForm)

// Emitting and parsing again gives the same request, for every key.
TEST(WindowRequestYAML, RoundTrips) {
  using namespace cudaq::qec;
  auto full = window_a();
  auto closed = window_a();
  full.input.prior.reset();
  full.input.boundary = input_boundary::truncate;
  full.sends.push_back({"wide", send_mode::delta, {{-3, 6}}});
  full.columns.merge_identical = true;
  full.columns.conflicting_parallel = parallel_resolution::most_likely;
  closed.offset = -9;
  closed.input = {port_disposition::closed, std::nullopt};
  closed.output = {port_disposition::closed, round_range{6, 8}};
  for (const auto &request : {window_a(), full, closed}) {
    const auto text = window_request_to_yaml_str(request);
    EXPECT_EQ(window_request_from_yaml_str(text), request) << text;
  }
}

// A send's rounds are one range, [b, e] or {rounds: [b, e]}, or a list of
// ranges in either form; several ranges are written as a list and read back.
// A list that mixes numbers and ranges, or holds a malformed range, is
// rejected.
TEST(WindowRequestYAML, SendsWithSeveralRangesRoundTrip) {
  using namespace cudaq::qec;
  const auto with_backward = [](const std::string &send) {
    auto yaml = window_a_yaml();
    yaml.replace(yaml.find("{id: backward}"), 14, send);
    return yaml;
  };
  const auto rejects = [&](const std::string &send, const std::string &text) {
    try {
      window_request_from_yaml_str(with_backward(send));
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(std::string(e.what()).find("window request: " + text),
                std::string::npos)
          << e.what();
    }
  };
  auto request = window_a();
  request.sends.push_back({"wide", send_mode::delta, {{-3, -2}, {3, 5}}});
  const auto text = window_request_to_yaml_str(request);
  EXPECT_EQ(window_request_from_yaml_str(text), request) << text;
  const auto listed = window_request_from_yaml_str(with_backward(
      "{id: backward, mode: residual, rounds: [[0, 1], {rounds: [2, 3]}]}"));
  EXPECT_EQ(listed.sends.at(1).rounds,
            (std::vector<round_range>{{0, 1}, {2, 3}}));
  rejects("{id: backward, mode: residual, rounds: [0, [1, 2]]}",
          "a send's rounds are [begin, end] or a list of ranges");
  rejects("{id: backward, mode: residual, rounds: [[0, 1, 2]]}",
          "a round range is [begin, end]");
} // end - TEST(WindowRequestYAML, SendsWithSeveralRangesRoundTrip)

// The forward-compatible shapes reserved for spatial windows are each
// rejected with their own message, as are malformed ranges and unknown keys.
TEST(WindowRequestYAML, RejectsReservedShapes) {
  using namespace cudaq::qec;
  const auto rejects = [](const std::string &from, const std::string &to,
                          const std::string &text) {
    auto yaml = window_a_yaml();
    const auto at = yaml.find(from);
    ASSERT_NE(at, std::string::npos) << from;
    yaml.replace(at, from.size(), to);
    try {
      window_request_from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  rejects("commit: [0, 3]", "commit: {rounds: [0, 3], occurrences: [[0, 1]]}",
          "'occurrences' is reserved");
  rejects("commit: [0, 3]", "commit: {rounds: [0, 3], block: 0}",
          "'block' is reserved");
  rejects("  output: {disposition: open}",
          "  output: {disposition: open}\n  west: {disposition: open}",
          "spatial ports are not supported yet");
  rejects("period: 12", "period: 12\nresponsibility: [0]",
          "'responsibility' is reserved");
  rejects("period: 12", "period: 12\nsoundness_class: strict",
          "'soundness_class' is reserved");
  rejects("commit: [0, 3]", "commit: [0, 3, 5]",
          "a round range is [begin, end]");
  rejects("period: 12", "period: 12\nperiodd: 5", "unknown key 'periodd'");
} // end - TEST(WindowRequestYAML, RejectsReservedShapes)

// period, zones and ports are required, as the schema says, rather than
// left at defaults that build no window.
TEST(WindowRequestYAML, RejectsMissingRequiredKeys) {
  using namespace cudaq::qec;
  const auto rejects = [](const std::string &line, const std::string &key) {
    auto yaml = window_a_yaml();
    const auto at = yaml.find(line);
    ASSERT_NE(at, std::string::npos) << line;
    yaml.erase(at, line.size());
    try {
      window_request_from_yaml_str(yaml);
      ADD_FAILURE() << "parsed without '" << key << "'";
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(
          std::string(e.what()).find("missing required key '" + key + "'"),
          std::string::npos)
          << e.what();
    }
  };
  rejects("period: 12\n", "period");
  rejects("zones: {buffers: [[-3, 0], [3, 6]], commit: [0, 3]}\n", "zones");
  rejects("ports:\n  input: {disposition: open, prior: 0.01}\n"
          "  output: {disposition: open}\n",
          "ports");
} // end - TEST(WindowRequestYAML, RejectsMissingRequiredKeys)

// A well-formed request that breaks a construction rule is rejected when
// parsed.
TEST(WindowRequestYAML, RejectsInvalidRequests) {
  using namespace cudaq::qec;
  const auto rejects = [](const std::string &from, const std::string &to,
                          const std::string &text) {
    auto yaml = window_a_yaml();
    const auto at = yaml.find(from);
    ASSERT_NE(at, std::string::npos) << from;
    yaml.replace(at, from.size(), to);
    try {
      window_request_from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(std::string(e.what()).find("window request: " + text),
                std::string::npos)
          << e.what();
    }
  };
  rejects("period: 12", "period: 0", "period must be at least 1");
  rejects("[3, 6]]", "[2, 6]]", "zones overlap");
  rejects("{disposition: open, prior: 0.01}", "{prior: 0.01}",
          "the input port has no disposition");
  rejects("prior: 0.01", "prior: 1.5",
          "the input prior must be strictly between 0 and 1");
} // end - TEST(WindowRequestYAML, RejectsInvalidRequests)

// An empty value where a range or a send's rounds belong is rejected, where
// LLVM's YAML reader would otherwise stop the process.
TEST(WindowRequestYAML, EmptyValuesFailCleanly) {
  using namespace cudaq::qec;
  for (const auto &[from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"commit: [0, 3]", "commit: "},
           {"{id: backward}", "{id: backward, mode: residual, rounds: }"}}) {
    auto yaml = window_a_yaml();
    yaml.replace(yaml.find(from), from.size(), to);
    EXPECT_THROW(window_request_from_yaml_str(yaml), std::invalid_argument)
        << to;
  }
}

// A hand-written parallel-window graph, with a two-input xor, over
// a streaming rep-code template. edit() changes the text before parsing.
std::string pw_graph_yaml() {
  return R"(
decoders:
  - id: 0
    dem_chunks:
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      phases:
        - {name: init, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: bulk, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: final, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
    task_graph:
      nodes:
        - id: ingest
          type: ingest
          windows:
            A:
              period: 12
              zones: {buffers: [[-3, 0], [3, 6]], commit: [0, 3]}
              ports:
                input: {disposition: open, prior: 0.01}
                output: {disposition: open}
              sends: [{id: forward}, {id: backward}]
            B:
              period: 12
              offset: -9
              zones: {commit: [0, 9]}
              ports:
                input: {disposition: closed}
                output: {disposition: closed}
        - id: A
          type: decode_window
          max_in_flight: 4
          inner: {type: single_error_lut}
        - id: B
          type: decode_window
          num_receives: 2
          max_in_flight: 4
          verify_residual: report
          inner: {type: single_error_lut}
        - id: sum
          type: xor
      edges:
        - {from: measurements, to: ingest}
        - {from: ingest.A, to: A}
        - {from: ingest.B, to: B}
        - {from: A.forward, to: B.receive_0, index_offset: -1}
        - {from: A.backward, to: B.receive_1}
        - {from: A, to: sum.a}
        - {from: B, to: sum.b}
        - {from: sum, to: output}
)";
}

// pw_graph_yaml() with `from` replaced by `to`.
std::string pw_graph_yaml_with(const std::string &from, const std::string &to) {
  auto yaml = pw_graph_yaml();
  const auto at = yaml.find(from);
  if (at == std::string::npos)
    throw std::logic_error("pw_graph_yaml has no '" + from + "'");
  return yaml.replace(at, from.size(), to);
}

// pw_graph_yaml() with an xor of num_inputs inputs: A, B and the raw
// logical value, as a corrected-logical graph wires it.
std::string three_input_sum_yaml(const std::string &num_inputs) {
  auto yaml = pw_graph_yaml_with("        - id: sum\n          type: xor\n",
                                 "        - id: sum\n          type: xor\n"
                                 "          num_inputs: " +
                                     num_inputs + "\n");
  const std::string edges = "        - {from: A, to: sum.a}\n"
                            "        - {from: B, to: sum.b}\n";
  return yaml.replace(yaml.find(edges), edges.size(),
                      "        - {from: A, to: sum.in_0}\n"
                      "        - {from: B, to: sum.in_1}\n"
                      "        - {from: ingest.logical, to: sum.in_2}\n");
}

// xor's num_inputs parses and round-trips, and below 2 or above 1024 it is
// rejected.
TEST(XorNumInputs, ParsesAndIsChecked) {
  namespace cfg = cudaq::qec::decoding::config;
  auto parsed =
      cfg::multi_decoder_config::from_yaml_str(three_input_sum_yaml("3"));
  const auto &sum = parsed.decoders.at(0).task_graph->nodes.at(3);
  const auto emitted = parsed.to_yaml_str(200);
  ASSERT_EQ(sum.id, "sum");
  EXPECT_EQ(sum.node_params.map().get<std::size_t>("num_inputs"), 3u);
  EXPECT_NE(emitted.find("num_inputs"), std::string::npos) << emitted;
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
      << emitted;
  for (const auto *n : {"0", "1"})
    try {
      cfg::multi_decoder_config::from_yaml_str(three_input_sum_yaml(n));
      ADD_FAILURE() << "num_inputs " << n << " was accepted";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(
          std::string(e.what()).find("xor: num_inputs must be at least 2"),
          std::string::npos)
          << e.what();
    }
  try {
    cfg::multi_decoder_config::from_yaml_str(three_input_sum_yaml("2000"));
    ADD_FAILURE() << "num_inputs 2000 was accepted";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(
        std::string(e.what()).find("xor: num_inputs must be at most 1024"),
        std::string::npos)
        << e.what();
  }
}

// The layers parse into the ingest node, sorted by name, as the window
// requests the C++ API writes; decode_window's parameters go to its kind;
// and the document round-trips.
TEST(TaskGraphWindowKinds, ParsesThePlansParallelWindowGraph) {
  namespace cfg = cudaq::qec::decoding::config;
  using namespace cudaq::qec::experimental;
  auto parsed = cfg::multi_decoder_config::from_yaml_str(pw_graph_yaml());
  const auto &graph = *parsed.decoders.at(0).task_graph;
  auto b = window_a();
  b.offset = -9;
  b.commit = round_range{0, 9};
  b.buffers.clear();
  b.input = {port_disposition::closed, std::nullopt};
  b.output = {port_disposition::closed, std::nullopt};
  b.sends.clear();
  ASSERT_EQ(graph.nodes.size(), 4u);
  const auto &ingest = graph.nodes[0];
  ASSERT_EQ(ingest.windows.size(), 2u);
  EXPECT_EQ(ingest.windows[0], (ingest_window{"A", window_a()}));
  EXPECT_EQ(ingest.windows[1], (ingest_window{"B", b}));
  const auto &params = graph.nodes[2].node_params.map();
  EXPECT_EQ(params.get<std::size_t>("num_receives"), 2u);
  EXPECT_EQ(graph.nodes[2].max_in_flight, 4u);
  EXPECT_EQ(params.get<std::string>("verify_residual"), "report");
  EXPECT_EQ(graph.nodes[2].inner_type, "single_error_lut");
  EXPECT_EQ(graph.edges[3].index_offset, -1);

  const auto emitted = parsed.to_yaml_str(200);
  EXPECT_NE(emitted.find("windows:"), std::string::npos);
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
      << emitted;
} // end - TEST(TaskGraphWindowKinds, ParsesThePlansParallelWindowGraph)

// A kind's parameter minimums are checked by its parameter validation and
// published in its clause of the JSON schema, so the two agree. A minimum
// for a key the kind does not declare as a uint64 parameter is rejected.
TEST(NodeKindMinimums, AreCheckedAndPublished) {
  namespace cfg = cudaq::qec::decoding::config;
  cudaq::qec::task_graph_node_kind_spec spec;
  cfg::decoder_schema params;
  cudaqx::heterogeneous_map low;
  cudaqx::heterogeneous_map enough;
  spec.name = "minimum_probe";
  spec.requires_decoder = false;
  spec.requires_dem = false;
  spec.parameter_minimums = {{"count", 3}};
  params.params.push_back({"count", cfg::param_kind::uint64});
  params.params.push_back({"label", cfg::param_kind::string});
  low.insert("count", std::size_t{2});
  enough.insert("count", std::size_t{3});
  cudaq::qec::register_task_graph_node_kind(spec, params);
  const auto *schema = cfg::find_decoder_schema(
      cudaq::qec::task_graph_node_schema_name("minimum_probe"));
  const auto json = cfg::decoder_config_json_schema();
  ASSERT_NE(schema, nullptr);
  try {
    schema->validate(low);
    ADD_FAILURE() << "a count below its minimum was accepted";
  } catch (const std::runtime_error &e) {
    EXPECT_EQ(std::string(e.what()), "minimum_probe: count must be at least 3");
  }
  EXPECT_NO_THROW(schema->validate(enough));
  EXPECT_NE(json.find("\"minimum\": 3"), std::string::npos);
  cudaq::qec::unregister_task_graph_node_kind("minimum_probe");
  for (const std::string key : {"missing", "label"}) {
    auto bad = spec;
    bad.parameter_minimums = {{key, 1}};
    try {
      cudaq::qec::register_task_graph_node_kind(bad, params);
      cudaq::qec::unregister_task_graph_node_kind("minimum_probe");
      ADD_FAILURE() << "a minimum for " << key << " was accepted";
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(std::string(e.what()).find("'" + key +
                                           "', which is not one of its uint64 "
                                           "parameters"),
                std::string::npos)
          << e.what();
    }
  }
} // end - TEST(NodeKindMinimums, AreCheckedAndPublished)

// pw_graph_yaml() with both of each round's faults flipping the observable,
// and `from` replaced by `to`. pw_graph_yaml()'s faults touch the same
// detector with different observable effects, which no decoder can tell
// apart, so validation would reject its windows; as identical twins they are
// no conflict. Its shot has no length, as a stream's does not.
std::string pw_build_yaml(const std::string &from = {},
                          const std::string &to = {}) {
  auto yaml = pw_graph_yaml();
  for (auto at = yaml.find("O_sparse: [0, -1]"); at != std::string::npos;
       at = yaml.find("O_sparse: [0, -1]", at))
    yaml.replace(at, 17, "O_sparse: [0, 1, -1]");
  if (from.empty())
    return yaml;
  const auto at = yaml.find(from);
  if (at == std::string::npos)
    throw std::logic_error("pw_build_yaml has no '" + from + "'");
  return yaml.replace(at, from.size(), to);
}

// Expect building decoder 0 of `config` to fail with `text` in the message.
void expect_build_error(
    const cudaq::qec::decoding::config::multi_decoder_config &config,
    const std::string &text) {
  try {
    cudaq::qec::decoding::host::build_task_graph(config.decoders.at(0));
    ADD_FAILURE() << "the graph built; expected: " << text;
  } catch (const std::exception &e) {
    EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
  }
}

// The kinds are registered for configuration, so a graph naming them parses
// and validates, and building it says the kinds are not available yet.
TEST(TaskGraphWindowKinds, BuildingSaysTheKindsAreNotAvailableYet) {
  namespace cfg = cudaq::qec::decoding::config;
  auto parsed = cfg::multi_decoder_config::from_yaml_str(pw_build_yaml());
  try {
    cudaq::qec::decoding::host::build_task_graph(parsed.decoders.at(0));
    ADD_FAILURE() << "the graph built";
  } catch (const std::exception &e) {
    EXPECT_NE(std::string(e.what()).find("'ingest'"), std::string::npos)
        << e.what();
    EXPECT_NE(std::string(e.what()).find("not available yet"),
              std::string::npos)
        << e.what();
  }
}

// Building a windowed graph validates its windows before anything is built,
// so each mistake is reported with its own message, ahead of the kinds
// saying they are not available yet.
TEST(TaskGraphWindowKinds, BuildingValidatesTheWindowsFirst) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &from, const std::string &to,
                          const std::string &text) {
    SCOPED_TRACE(text);
    expect_build_error(
        cfg::multi_decoder_config::from_yaml_str(pw_build_yaml(from, to)),
        text);
  };
  rejects("        - {from: measurements, to: ingest}\n", "",
          "ingest node 'ingest' must be fed by measurements alone");
  rejects("{from: ingest.A, to: A}", "{from: ingest.C, to: A}",
          "decode_window node 'A' is fed by 'ingest.C', which is not a window "
          "port of an ingest node");
  rejects("{from: ingest.A, to: A}", "{from: measurements, to: A}",
          "decode_window node 'A' is fed by 'measurements.output', which is "
          "not a window port of an ingest node");
  rejects("          type: ingest\n",
          "          type: ingest\n          read_logical: true\n",
          "ingest node 'ingest' reads the logical value, but its template's "
          "last phase has no m2o_sparse");
  rejects("to: B.receive_1}", "to: B.receive_2}",
          "decode_window node 'B' has num_receives 2, so it has no input "
          "'receive_2'");
  rejects("to: B.receive_1}", "to: B.receive_01}",
          "decode_window node 'B' has no input 'receive_01'");
  rejects("to: B.receive_1}", "to: B.receive_99999999999999999999}",
          "decode_window node 'B' has no input "
          "'receive_99999999999999999999'");
  rejects("{from: ingest.B, to: B}",
          "{from: ingest.B, to: B, index_offset: -1}",
          "edge ingest.B -> B.input (index_offset -1) has a nonzero "
          "index_offset");
  rejects("        - {from: A.backward, to: B.receive_1}\n", "",
          "validate_commitment: F3: ");
  rejects(", index_offset: -1}", "}", "reaches round");
  rejects("              period: 12\n              offset: -9",
          "              period: 13\n              offset: -9",
          "have periods 12 and 13");
  rejects("                input: {disposition: closed}\n",
          "                input: {disposition: open, prior: 0.01}\n", "F1: ");
} // end - TEST(TaskGraphWindowKinds, BuildingValidatesTheWindowsFirst)

// A configuration built in code skips the parser, so building also rejects
// a positive index_offset. Windows of two ingest nodes are two shots: a send
// between them is rejected, and so are two ingest nodes with different
// templates that reach one node.
TEST(TaskGraphWindowKinds, BuildingRejectsMixedShots) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto parsed = cfg::multi_decoder_config::from_yaml_str(pw_build_yaml());
  const auto with_second_ingest = [&] {
    auto config = parsed;
    auto &decoder = config.decoders.at(0);
    auto &graph = *decoder.task_graph;
    auto second = graph.nodes.at(0);
    second.id = "ingest2";
    second.dem_chunks = *decoder.dem_chunks;
    second.dem_chunks->phases.at(0).spec.error_rates.at(0) = 0.02;
    graph.nodes.push_back(second);
    graph.edges.push_back({"measurements", "output", "ingest2", "input"});
    for (auto &edge : graph.edges)
      if (edge.from_node == "ingest" && edge.from_output == "B")
        edge.from_node = "ingest2";
    return config;
  };
  auto positive = parsed;
  auto across = with_second_ingest();
  auto apart = with_second_ingest();
  for (auto &edge : positive.decoders.at(0).task_graph->edges)
    if (edge.from_output == "forward")
      edge.index_offset = 1;
  auto &apart_edges = apart.decoders.at(0).task_graph->edges;
  std::erase_if(apart_edges, [](const auto &edge) {
    return edge.to_node == "B" && edge.to_input != "input";
  });
  expect_build_error(positive, "edge A.forward -> B.receive_0 (index_offset "
                               "1) has a positive index_offset");
  expect_build_error(across, "edge A.forward -> B.receive_0 (index_offset -1) "
                             "joins the windows of ingest nodes 'ingest' and "
                             "'ingest2'");
  expect_build_error(apart,
                     "ingest nodes 'ingest' and 'ingest2' have different "
                     "templates but both reach 'sum'");
} // end - TEST(TaskGraphWindowKinds, BuildingRejectsMixedShots)

// A window waits on each send it receives at index_offset 0, from its own
// index, so receives at offset 0 that close a cycle are rejected, a window
// receiving its own send among them: no window on the cycle could start. A
// cycle through an earlier index is a chain through the shot instead; it
// passes on to the commitment checks, which here find B has no such send.
TEST(TaskGraphWindowKinds, BuildingRejectsWindowsWaitingOnTheirOwnIndex) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto parsed = cfg::multi_decoder_config::from_yaml_str(pw_build_yaml());
  // parsed with A taking one receive, on `edge`.
  const auto with = [&](const cudaq::qec::task_graph_edge &edge) {
    auto config = parsed;
    auto &graph = *config.decoders.at(0).task_graph;
    for (auto &node : graph.nodes)
      if (node.id == "A")
        node.node_params.map().insert("num_receives", std::size_t{1});
    graph.edges.push_back(edge);
    return config;
  };
  expect_build_error(with({"A", "forward", "A", "receive_0"}),
                     "decode_window receives at index_offset 0 form a cycle, "
                     "'A' -> 'A'");
  expect_build_error(with({"B", "forward", "A", "receive_0"}),
                     "decode_window receives at index_offset 0 form a cycle, "
                     "'A' -> 'B' -> 'A'");
  expect_build_error(with({"B", "forward", "A", "receive_0", -1}),
                     "B has no send 'forward'");
} // end - TEST(TaskGraphWindowKinds,
  // BuildingRejectsWindowsWaitingOnTheirOwnIndex)

// A window's decoders are built from its window models, so building a graph
// makes no decoder for a decode_window node from the whole shot: none is
// constructed before the layout is validated, and a layout mistake is
// reported as itself.
TEST(TaskGraphWindowKinds, BuildingMakesNoWholeShotDecoderForAWindow) {
  namespace cfg = cudaq::qec::decoding::config;
  auto yaml = pw_build_yaml();
  for (auto at = yaml.find("single_error_lut"); at != std::string::npos;
       at = yaml.find("single_error_lut", at))
    yaml.replace(at, std::string("single_error_lut").size(),
                 "d_capture_decoder");
  const auto parsed = cfg::multi_decoder_config::from_yaml_str(yaml);
  auto mistaken = parsed;
  std::erase_if(
      mistaken.decoders.at(0).task_graph->edges,
      [](const auto &edge) { return edge.from_output == "backward"; });
  cudaq::qec::construction_d_probe::model = {};
  expect_build_error(parsed, "the 'ingest' kind is not available yet");
  expect_build_error(mistaken, "validate_commitment: F3: ");
  EXPECT_EQ(cudaq::qec::construction_d_probe::model.num_detectors, 0u);
} // end - TEST(TaskGraphWindowKinds, BuildingMakesNoWholeShotDecoderForAWindow)

// Two ingest nodes with different templates may share a graph as long as
// their windows meet in no node: graph outputs are separate results, not a
// node they share. Each pipeline validates on its own, so the build reaches
// the kinds' own report.
TEST(TaskGraphWindowKinds, BuildingAcceptsTwoTemplatesOnTwoOutputs) {
  namespace cfg = cudaq::qec::decoding::config;
  auto config = cfg::multi_decoder_config::from_yaml_str(pw_build_yaml());
  auto &decoder = config.decoders.at(0);
  auto &graph = *decoder.task_graph;
  const auto nodes = graph.nodes;
  const auto edges = graph.edges;
  const auto renamed = [](const std::string &id) {
    return id == "measurements" || id == "output" ? id : id + "2";
  };
  for (auto node : nodes) {
    node.id = renamed(node.id);
    if (node.id == "ingest2") {
      node.dem_chunks = *decoder.dem_chunks;
      node.dem_chunks->phases.at(0).spec.error_rates.at(0) = 0.02;
    }
    graph.nodes.push_back(node);
  }
  for (auto edge : edges) {
    edge.from_node = renamed(edge.from_node);
    edge.to_node = renamed(edge.to_node);
    graph.edges.push_back(edge);
  }
  expect_build_error(config, "the 'ingest' kind is not available yet");
} // end - TEST(TaskGraphWindowKinds, BuildingAcceptsTwoTemplatesOnTwoOutputs)

// Node ids and parameter types are checked before the window layout, which
// reads both: a repeated id is named as such, and a parameter of the wrong
// type, or a negative count, is not read as its default or as a huge count.
TEST(TaskGraphWindowKinds, BuildingChecksIdsAndParametersFirst) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto parsed = cfg::multi_decoder_config::from_yaml_str(pw_build_yaml());
  auto repeated = parsed;
  auto mistyped = parsed;
  auto negative = parsed;
  for (auto &node : repeated.decoders.at(0).task_graph->nodes)
    if (node.id == "sum")
      node.id = "A";
  for (auto &node : mistyped.decoders.at(0).task_graph->nodes)
    if (node.id == "B") {
      node.node_params.map().insert("num_receives", 2.0);
    }
  for (auto &node : negative.decoders.at(0).task_graph->nodes)
    if (node.id == "B")
      node.node_params.map().insert("num_receives", -1);
  expect_build_error(repeated, "duplicate node id 'A'");
  expect_build_error(mistyped, "Key 'num_receives' in from_topology: node 'B' "
                               "parameters does not hold a non-negative int");
  expect_build_error(negative, "Key 'num_receives' in from_topology: node 'B' "
                               "parameters does not hold a non-negative int");
}

// Misplaced or malformed windows, and a bad verify_residual, are each
// rejected when parsed, with their own message. verify_residual takes no
// boolean, which YAML tools would read as a non-string.
TEST(TaskGraphWindowKinds, RejectsMisplacedWindowsAndBadParameters) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const std::string &text) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  rejects(
      pw_graph_yaml_with("        - id: sum\n          type: xor\n",
                         "        - id: sum\n          type: xor\n"
                         "          windows: {C: {period: 1, zones: {commit: "
                         "[0, 1]}, ports: {input: {disposition: closed}, "
                         "output: {disposition: closed}}}}\n"),
      "sets windows, which only an ingest node takes");
  rejects(pw_graph_yaml_with("          windows:\n            A:",
                             "          read_logical: false\n"
                             "          other:\n            A:"),
          "'other'");
  rejects(pw_graph_yaml_with("            B:\n", "            logical:\n"),
          "layer 'logical'");
  rejects(pw_graph_yaml_with("              offset: -9\n",
                             "              offset: -9\n"
                             "              responsibility: [0]\n"),
          "'responsibility' is reserved");
  rejects(
      pw_graph_yaml_with("verify_residual: report", "verify_residual: maybe"),
      "verify_residual");
  for (const std::string value : {"false", "true"})
    rejects(
        pw_graph_yaml_with("verify_residual: report",
                           "verify_residual: " + value),
        "decode_window: verify_residual must be none, fail or report; got '" +
            value + "'");
} // end - TEST(TaskGraphWindowKinds, RejectsMisplacedWindowsAndBadParameters)

// A layer missing a required key, or well formed but breaking a window
// construction rule, fails when the config loads rather than when the graph
// is built.
TEST(TaskGraphWindowKinds, RejectsInvalidRequestsWhenLoaded) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &from, const std::string &to,
                          const std::string &text) {
    const auto yaml = pw_graph_yaml_with(from, to);
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  const std::string a = "            A:\n              period: 12\n";
  rejects(a, "            A:\n",
          "window layer 'A': window request: missing required key 'period'");
  rejects("              zones: {commit: [0, 9]}\n", "",
          "window layer 'B': window request: missing required key 'zones'");
  rejects("              ports:\n"
          "                input: {disposition: closed}\n"
          "                output: {disposition: closed}\n",
          "", "window layer 'B': window request: missing required key 'ports'");

  rejects(a, "            A:\n              period: 0\n",
          "window layer 'A': window request: period must be at least 1");
  rejects("[[-3, 0], [3, 6]]", "[[-3, 0], [2, 6]]",
          "window layer 'A': window request: zones overlap");
  rejects("input: {disposition: closed}", "input: {}",
          "window layer 'B': window request: the input port has no "
          "disposition");
  rejects("output: {disposition: open}", "output: {}",
          "window layer 'A': window request: the output port has no "
          "disposition");
} // end - TEST(TaskGraphWindowKinds, RejectsInvalidRequestsWhenLoaded)

// Only an ingest node takes windows: a node whose type is a decoder plugin,
// and an ensemble entry, are rejected rather than having theirs dropped.
TEST(TaskGraphWindowKinds, RejectsWindowsOnPluginNodesAndEnsembles) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const std::string &text) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  rejects(pw_graph_yaml_with("type: ingest", "type: single_error_lut"),
          "sets windows, which only an ingest node takes");
  rejects(R"(decoders:
  - id: 0
    task_graph:
      nodes:
        - id: e
          windows:
            A: {period: 4, zones: {commit: [0, 2]}, ports: {input: {disposition: closed}, output: {disposition: open}}}
          ensemble:
            join_strategy: deadline
            join_params: {deadline_us: 200000}
            members:
              - nodes:
                  - {id: a, type: single_window, inner: {type: single_error_lut}}
                edges: [{from: input, to: a}, {from: a, to: output}]
      edges:
        - {from: measurements, to: e}
        - {from: e, to: output}
)",
          "is an ensemble entry, which takes no windows");
} // end - TEST(TaskGraphWindowKinds, RejectsWindowsOnPluginNodesAndEnsembles)

// An empty value where a value of several forms belongs (dem_chunks, a
// window layer, a partition) is a parse error, where LLVM's YAML reader
// would otherwise stop the process.
TEST(TaskGraphWindowKinds, EmptyValuesFailCleanly) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const std::string &text) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  rejects("decoders:\n  - id: 0\n    type: single_error_lut\n    dem_chunks:\n",
          "dem_chunks needs a value");
  rejects(
      pw_graph_yaml_with("          windows:\n            A:",
                         "          windows:\n            C:\n            A:"),
      "a window layer needs a value");
  rejects(pw_graph_yaml_with("        - id: sum\n          type: xor\n",
                             "        - id: sum\n          type: xor\n"
                             "          partition:\n"),
          "partition needs a value");
} // end - TEST(TaskGraphWindowKinds, EmptyValuesFailCleanly)

// An ingest node with no windows has nothing to publish.
TEST(TaskGraphWindowKinds, IngestNeedsAWindow) {
  namespace cfg = cudaq::qec::decoding::config;
  auto yaml = pw_graph_yaml();
  const auto begin = yaml.find("          windows:");
  const auto end = yaml.find("        - id: A");
  yaml.erase(begin, end - begin);
  try {
    cfg::multi_decoder_config::from_yaml_str(yaml);
    ADD_FAILURE() << "an ingest node without windows parsed";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("needs at least one window"),
              std::string::npos)
        << e.what();
  }
}

// ---------------------------------------------------------------------------
// data_artifacts
// ---------------------------------------------------------------------------

// A template, a Stim file pinned by its hash, and the parallel-window
// layers' requests, named once and referenced from a decoder and its
// ingest node.
std::string artifacts_yaml() {
  return R"(
data_artifacts:
  rep_memory:
    dem_chunks:
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      phases:
        - {name: init, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: bulk, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: final, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
  rep_stim:
    stim_dem_path: dems/rep.dem
    sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  pw_A:
    window:
      context: rep_memory
      period: 12
      zones: {buffers: [[-3, 0], [3, 6]], commit: [0, 3]}
      ports:
        input: {disposition: open, prior: 0.01}
        output: {disposition: open}
      sends: [{id: forward}, {id: backward}]
  pw_B:
    window:
      context: rep_memory
      period: 12
      offset: -9
      zones: {commit: [0, 9]}
      ports:
        input: {disposition: closed}
        output: {disposition: closed}
decoders:
  - id: 0
    dem_chunks: rep_memory
    task_graph:
      nodes:
        - id: ingest
          type: ingest
          windows: {A: pw_A, B: pw_B}
        - {id: A, type: decode_window, inner: {type: single_error_lut}}
        - id: B
          type: decode_window
          num_receives: 2
          inner: {type: single_error_lut}
        - {id: sum, type: xor}
      edges:
        - {from: measurements, to: ingest}
        - {from: ingest.A, to: A}
        - {from: ingest.B, to: B}
        - {from: A.forward, to: B.receive_0, index_offset: -1}
        - {from: A.backward, to: B.receive_1}
        - {from: A, to: sum.a}
        - {from: B, to: sum.b}
        - {from: sum, to: output}
)";
}

// artifacts_yaml() with `from` replaced by `to`.
std::string artifacts_yaml_with(const std::string &from,
                                const std::string &to) {
  auto yaml = artifacts_yaml();
  const auto at = yaml.find(from);
  if (at == std::string::npos)
    throw std::logic_error("artifacts_yaml has no '" + from + "'");
  return yaml.replace(at, from.size(), to);
}

// Entries parse sorted by name; references resolve to copies of the named
// entries, which keep the name, so emission writes the name back and the
// document round-trips.
TEST(DataArtifactsYAML, ReferencesResolveToTheNamedEntries) {
  namespace cfg = cudaq::qec::decoding::config;
  using namespace cudaq::qec::experimental;
  auto parsed = cfg::multi_decoder_config::from_yaml_str(artifacts_yaml());
  const auto &artifacts = parsed.data_artifacts;
  const auto &decoder = parsed.decoders.at(0);
  const auto &ingest = decoder.task_graph->nodes.at(0);
  ASSERT_EQ(artifacts.size(), 4u);
  EXPECT_EQ(artifacts[0].name, "pw_A");
  EXPECT_EQ(artifacts[1].name, "pw_B");
  EXPECT_EQ(artifacts[2].name, "rep_memory");
  EXPECT_EQ(artifacts[3].name, "rep_stim");
  ASSERT_TRUE(artifacts[0].window.has_value());
  EXPECT_EQ(*artifacts[0].window, window_a());
  EXPECT_EQ(artifacts[0].context, "rep_memory");
  EXPECT_EQ(artifacts[3].stim_dem_path, "dems/rep.dem");
  EXPECT_EQ(artifacts[3].sha256.size(), 64u);

  EXPECT_EQ(decoder.dem_chunks_artifact, "rep_memory");
  ASSERT_TRUE(decoder.dem_chunks.has_value());
  EXPECT_EQ(*decoder.dem_chunks, *artifacts[2].dem_chunks);
  ASSERT_EQ(ingest.windows.size(), 2u);
  EXPECT_EQ(ingest.windows[0].artifact, "pw_A");
  EXPECT_EQ(ingest.windows[0].request, window_a());
  EXPECT_EQ(ingest.windows[1].artifact, "pw_B");
  EXPECT_EQ(ingest.windows[1].request, *artifacts[1].window);

  // The requests are written once, in data_artifacts; the uses name them.
  const auto emitted = parsed.to_yaml_str(200);
  std::size_t periods = 0;
  for (auto at = emitted.find("period:"); at != std::string::npos;
       at = emitted.find("period:", at + 1))
    ++periods;
  EXPECT_EQ(periods, 2u) << emitted;
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
      << emitted;
} // end - TEST(DataArtifactsYAML, ReferencesResolveToTheNamedEntries)

// A decoder that names its template builds the same graph as one that writes
// it inline, and a file without data_artifacts emits none.
TEST(DataArtifactsYAML, ByNameBuildsWhatInlineBuilds) {
  namespace cfg = cudaq::qec::decoding::config;
  const std::string tmpl = R"(
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      num_rounds: 5
      phases:
        - {name: init, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: bulk, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: final, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
)";
  const std::string graph = R"(
    task_graph:
      nodes:
        - {id: w, type: single_window, inner: {type: single_error_lut}}
      edges:
        - {from: measurements, to: w}
        - {from: w, to: output}
)";
  auto inline_form = cfg::multi_decoder_config::from_yaml_str(
      "decoders:\n  - id: 0\n    dem_chunks:" + tmpl + graph);
  auto by_name = cfg::multi_decoder_config::from_yaml_str(
      "data_artifacts:\n  t:\n    dem_chunks:" + tmpl +
      "decoders:\n  - id: 0\n    dem_chunks: t\n" + graph);
  const auto a =
      cudaq::qec::decoding::host::build_task_graph(inline_form.decoders.at(0));
  const auto b =
      cudaq::qec::decoding::host::build_task_graph(by_name.decoders.at(0));
  EXPECT_EQ(by_name.decoders.at(0).dem_chunks,
            inline_form.decoders.at(0).dem_chunks);
  EXPECT_EQ(a->get_num_msyn_per_decode(), b->get_num_msyn_per_decode());
  EXPECT_EQ(a->get_num_msyn_per_round(), b->get_num_msyn_per_round());
  EXPECT_EQ(a->get_num_observables(), b->get_num_observables());
  EXPECT_EQ(inline_form.to_yaml_str().find("data_artifacts"),
            std::string::npos);
} // end - TEST(DataArtifactsYAML, ByNameBuildsWhatInlineBuilds)

// Each malformed entry or reference is rejected when parsed, with its own
// message.
TEST(DataArtifactsYAML, RejectsMalformedEntriesAndReferences) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto rejects = [](const std::string &yaml, const std::string &text) {
    try {
      cfg::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  rejects(artifacts_yaml_with("dem_chunks: rep_memory", "dem_chunks: nowhere"),
          "names data_artifacts entry 'nowhere', which does not exist");
  rejects(artifacts_yaml_with("dem_chunks: rep_memory", "dem_chunks: pw_A"),
          "'pw_A' is a window entry, not a dem_chunks entry");
  rejects(artifacts_yaml_with("{A: pw_A, B: pw_B}", "{A: rep_memory, B: pw_B}"),
          "'rep_memory' is a dem_chunks entry, not a window entry");
  rejects(artifacts_yaml_with("    stim_dem_path: dems/rep.dem\n",
                              "    stim_dem_path: dems/rep.dem\n"
                              "    edem_path: x.edem\n"),
          "entry 'rep_stim' has two kinds");
  rejects(artifacts_yaml_with("    stim_dem_path: dems/rep.dem\n", ""),
          "entry 'rep_stim' has no kind");
  rejects(artifacts_yaml_with("      context: rep_memory\n      period: 12\n"
                              "      zones: {buffers",
                              "      context: rep_stim\n      period: 12\n"
                              "      zones: {buffers"),
          "context 'rep_stim' is not a dem_chunks entry");
  rejects(artifacts_yaml_with("stim_dem_path: dems/rep.dem",
                              "edem_path: gadgets/sec.edem"),
          "edem_path is not supported until the cudaq-edem migration");
  rejects(artifacts_yaml_with("stim_dem_path: dems/rep.dem",
                              "stim_dem: \"error(0.1) D0\""),
          "sha256 pins a file");
  rejects(artifacts_yaml_with("sha256: 0123456789abcdef", "sha256: 0123"),
          "sha256 must be 64 hexadecimal digits");
  rejects(artifacts_yaml_with("      offset: -9\n",
                              "      offset: -9\n"
                              "      responsibility: [0]\n"),
          "'responsibility' is reserved");
} // end - TEST(DataArtifactsYAML, RejectsMalformedEntriesAndReferences)

// A reference is written back by name only while the copy still equals the
// named entry; a copy edited after parsing is written inline, so emitting
// and parsing again keeps the edit, as the YAML published to remote targets
// must. A decoder written on its own has no entries to name, so it writes
// its copies inline.
TEST(DataArtifactsYAML, EditsAfterParsingAreWrittenInline) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto parsed =
      cfg::multi_decoder_config::from_yaml_str(artifacts_yaml());
  const auto unchanged = cfg::multi_decoder_config::from_yaml_str(
      cfg::multi_decoder_config(parsed).to_yaml_str(200));
  auto edited = parsed;
  auto &layers = edited.decoders.at(0).task_graph->nodes.at(0).windows;
  edited.decoders.at(0).dem_chunks->num_rounds = 7;
  layers.at(0).request.period = 24;
  const auto again =
      cfg::multi_decoder_config::from_yaml_str(edited.to_yaml_str(200));
  const auto &again_layers =
      again.decoders.at(0).task_graph->nodes.at(0).windows;
  EXPECT_EQ(unchanged, parsed);
  EXPECT_EQ(again.decoders.at(0).dem_chunks, edited.decoders.at(0).dem_chunks);
  EXPECT_TRUE(again.decoders.at(0).dem_chunks_artifact.empty());
  EXPECT_EQ(again_layers.at(0).request, layers.at(0).request);
  EXPECT_TRUE(again_layers.at(0).artifact.empty());
  EXPECT_EQ(again_layers.at(1).artifact, "pw_B");
  const auto alone = cfg::decoder_config::from_yaml_str(
      cfg::decoder_config(parsed.decoders.at(0)).to_yaml_str(200));
  EXPECT_EQ(alone.dem_chunks, parsed.decoders.at(0).dem_chunks);
  EXPECT_EQ(
      alone.task_graph->nodes.at(0).windows.at(0).request,
      parsed.decoders.at(0).task_graph->nodes.at(0).windows.at(0).request);
} // end - TEST(DataArtifactsYAML, EditsAfterParsingAreWrittenInline)

// A decoder parsed on its own has no data_artifacts to name.
TEST(DataArtifactsYAML, AStandaloneDecoderCannotNameAnEntry) {
  namespace cfg = cudaq::qec::decoding::config;
  try {
    cfg::decoder_config::from_yaml_str("id: 0\ntype: single_error_lut\n"
                                       "dem_chunks: rep_memory\n");
    ADD_FAILURE() << "a standalone decoder resolved a name";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("outside a multi_decoder_config"),
              std::string::npos)
        << e.what();
  }
}

// A name is written back only while the document holds an entry of its kind
// to resolve it. A copy built in code that keeps a name the document lacks
// -- written on its own, or beside a model of another form -- writes no
// reference, so the YAML parses again. With the entry present, a name with
// no copy is still written, and resolves to the entry.
TEST(DataArtifactsYAML, NamesWithoutTheirEntryAreNotWritten) {
  namespace cfg = cudaq::qec::decoding::config;
  auto flat = cfg::decoder_config::from_yaml_str(R"(
id: 0
type: multi_error_lut
block_size: 2
syndrome_size: 1
H_sparse: [0, 1, -1]
O_sparse: [0, -1]
D_sparse: [0, -1]
)");
  auto named = flat;
  cfg::multi_decoder_config document;
  auto parsed = cfg::multi_decoder_config::from_yaml_str(artifacts_yaml());
  auto without_copy = parsed;
  named.stim_dem_artifact = "by_file";
  named.dem_chunks_artifact = "rep_memory";
  document.decoders = {named};
  const auto alone = named.to_yaml_str();
  const auto in_document = document.to_yaml_str();
  for (const auto &yaml : {alone, in_document}) {
    EXPECT_EQ(yaml.find("by_file"), std::string::npos) << yaml;
    EXPECT_EQ(yaml.find("rep_memory"), std::string::npos) << yaml;
  }
  EXPECT_EQ(cfg::decoder_config::from_yaml_str(alone), flat);
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(in_document).decoders,
            std::vector<cfg::decoder_config>{flat});

  without_copy.decoders.at(0).dem_chunks.reset();
  const auto emitted = without_copy.to_yaml_str(200);
  EXPECT_TRUE(
      std::regex_search(emitted, std::regex("dem_chunks:\\s+rep_memory\\n")))
      << emitted;
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
      << emitted;
} // end - TEST(DataArtifactsYAML, NamesWithoutTheirEntryAreNotWritten)

// ---------------------------------------------------------------------------
// Data artifact store, filled from a configuration
// ---------------------------------------------------------------------------

// The Stim DEM artifacts_yaml() names as dems/rep.dem, and the placeholder
// sha256 it pins the file to.
constexpr const char *rep_dem_text = "error(0.01) D0\n";
constexpr const char *placeholder_pin =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// A temporary directory holding dems/rep.dem, removed on destruction.
class ScopedArtifactDir {
public:
  ScopedArtifactDir() {
    // Qualified by pid, as ScopedDemFile is, since ctest runs binaries
    // concurrently.
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("cudaqx_artifacts_" + std::to_string(::getpid()) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(dir_ / "dems");
    std::ofstream(dir_ / "dems" / "rep.dem") << rep_dem_text;
  }
  ~ScopedArtifactDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  const std::filesystem::path &path() const { return dir_; }

private:
  std::filesystem::path dir_;
};

// artifacts_yaml() with dems/rep.dem pinned to its real sha256.
std::string pinned_artifacts_yaml() {
  return artifacts_yaml_with(placeholder_pin,
                             cudaq::qec::sha256_hex(rep_dem_text));
}

// A store's records of one kind.
std::vector<cudaq::qec::data_artifact_record>
records_of(const cudaq::qec::data_artifact_store &store,
           const std::string &kind) {
  std::vector<cudaq::qec::data_artifact_record> out;
  for (const auto &record : store.records())
    if (record.kind == kind)
      out.push_back(record);
  return out;
}

// The distinct variant keys of request on tmpl, over the shot lengths that
// hold every key.
std::set<cudaq::qec::variant_key>
variant_keys(const cudaq::qec::experiment_template &tmpl,
             const cudaq::qec::experimental::window_request &request) {
  std::set<cudaq::qec::variant_key> keys;
  for (auto R = tmpl.min_rounds();
       R < cudaq::qec::variant_probe_rounds(tmpl, request); ++R) {
    const auto instances = cudaq::qec::window_instances(request, R);
    for (auto k = instances.begin; k < instances.end; ++k)
      keys.insert(cudaq::qec::window_variant(tmpl, request, k, R));
  }
  return keys;
}

// Filling holds the template once, though the decoder and both window
// entries name it; the Stim DEM, read from its file and checked against its
// pin; and every variant of both window entries, which the ingest node's
// layers name again.
TEST(DataArtifactStore, FillsFromTheConfig) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  const auto config =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  const auto store =
      cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
  const auto &artifacts = config.data_artifacts;
  const auto tmpl = store.find_template(*config.decoders.at(0).dem_chunks);
  std::size_t variants = 0;
  ASSERT_EQ(artifacts.size(), 4u);
  EXPECT_EQ(tmpl, store.find_template(*artifacts[2].dem_chunks));
  EXPECT_EQ(records_of(store, "dem_chunks").size(), 1u);
  EXPECT_EQ(records_of(store, "stim_dem"),
            (std::vector<cudaq::qec::data_artifact_record>{
                {"stim_dem", std::string(rep_dem_text).size(),
                 cudaq::qec::sha256_hex(rep_dem_text)}}));
  for (const auto &entry : {artifacts[0], artifacts[1]}) {
    const auto keys = variant_keys(*tmpl, *entry.window);
    variants += keys.size();
    for (const auto &key : keys)
      EXPECT_NE(store.find_window(*tmpl, *entry.window, key), nullptr);
  }
  EXPECT_EQ(records_of(store, "window").size(), variants);
} // end - TEST(DataArtifactStore, FillsFromTheConfig)

// Two decoders naming one template, and a third writing the same content
// inline, each building windows on it, share one entry and one handle.
TEST(DataArtifactStore, DecodersNamingOneTemplateShareIt) {
  namespace cfg = cudaq::qec::decoding::config;
  const std::string tmpl = R"(
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      phases:
        - {name: init, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: bulk, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: final, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
)";
  const std::string window = R"(
  w:
    window:
      context: t
      period: 12
      offset: -9
      zones: {commit: [0, 9]}
      ports:
        input: {disposition: closed}
        output: {disposition: closed}
)";
  const auto decoder = [](int id, const std::string &dem_chunks) {
    return "  - id: " + std::to_string(id) + "\n    dem_chunks:" + dem_chunks +
           R"(    task_graph:
      nodes:
        - {id: ingest, type: ingest, windows: {A: w}}
        - {id: A, type: decode_window, inner: {type: single_error_lut}}
      edges:
        - {from: measurements, to: ingest}
        - {from: ingest.A, to: A}
        - {from: A, to: output}
)";
  };
  const auto config = cfg::multi_decoder_config::from_yaml_str(
      "data_artifacts:\n  t:\n    dem_chunks:" + tmpl + window + "decoders:\n" +
      decoder(0, " t\n") + decoder(1, " t\n") + decoder(2, tmpl));
  const auto store = cudaq::qec::decoding::host::fill_data_artifact_store(
      config, std::filesystem::current_path());
  const auto shared = store.find_template(*config.decoders.at(0).dem_chunks);
  EXPECT_EQ(records_of(store, "dem_chunks").size(), 1u);
  EXPECT_EQ(store.find_template(*config.decoders.at(1).dem_chunks), shared);
  EXPECT_EQ(store.find_template(*config.decoders.at(2).dem_chunks), shared);
} // end - TEST(DataArtifactStore, DecodersNamingOneTemplateShareIt)

// Only the templates windows are built on are filled: a window entry's
// context, and the template of a node with window layers. experiment_template
// is stricter than a plain decoder's dem_chunks, so a model it rejects -- here
// a suffix phase before FINAL wider than bulk -- leaves filling alone where no
// window is built on it: on a decoder without a task graph, on a node without
// layers, and in an entry no window names.
TEST(DataArtifactStore, FillsOnlyTheTemplatesWindowsAreBuiltOn) {
  namespace cfg = cudaq::qec::decoding::config;
  using node_spec = cfg::decoder_config::task_graph_spec::node_spec;
  const ScopedArtifactDir dir;
  auto config =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  auto wide = cudaq::qec::multi_terminal_memory_spec(5, 2);
  auto plain = config.decoders.at(0);
  cfg::data_artifact entry;
  node_spec node;
  for (auto &phase : wide.phases)
    if (phase.id == cudaq::qec::phase_id::from_name("terminal0"))
      phase.spec.num_measurements = 5;
  EXPECT_THROW(cudaq::qec::experiment_template{wide}, std::invalid_argument);

  // A plain decoder expands the model as it is.
  plain.task_graph.reset();
  plain.dem_chunks = wide;
  plain.dem_chunks_artifact.clear();
  auto expanded = plain;
  EXPECT_NO_THROW(cfg::expand_dem_chunks(expanded));
  plain.id = 1;
  config.decoders.push_back(plain);
  entry.name = "wide";
  entry.dem_chunks = wide;
  config.data_artifacts.push_back(entry);
  plain.id = 2;
  plain.dem_chunks_artifact = "wide";
  config.decoders.push_back(plain);
  node.id = "w";
  node.dem_chunks = wide;
  plain.id = 3;
  plain.dem_chunks.reset();
  plain.dem_chunks_artifact.clear();
  plain.task_graph.emplace();
  plain.task_graph->nodes.push_back(node);
  config.decoders.push_back(plain);

  const auto store =
      cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
  EXPECT_EQ(records_of(store, "dem_chunks").size(), 1u);
  EXPECT_NO_THROW(store.find_template(*config.decoders.at(0).dem_chunks));
  EXPECT_THROW(store.find_template(wide), std::logic_error);
} // end - TEST(DataArtifactStore, FillsOnlyTheTemplatesWindowsAreBuiltOn)

// A node's template is the one translate_task_graph_node() builds its DEM
// from: its own dem_chunks, else its decoder's while the decoder has no flat
// matrices. expand_dem_chunks() leaves H_sparse beside dem_chunks, and the
// node then runs on the flat matrices, so its layers have no template to be
// built on and are rejected. With dem_chunks of its own, the node builds its
// windows on that.
TEST(DataArtifactStore, ANodeOnFlatMatricesHasNoTemplate) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  const auto parsed =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  auto config = parsed;
  auto &decoder = config.decoders.at(0);
  auto &ingest = decoder.task_graph->nodes.at(0);
  decoder.dem_chunks->num_rounds = 12;
  cfg::expand_dem_chunks(decoder);
  ASSERT_FALSE(decoder.H_sparse.empty());
  ASSERT_TRUE(decoder.dem_chunks.has_value());
  try {
    cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
    ADD_FAILURE() << "a layer was built on a template the node does not use";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "decoder 0, node 'ingest' has windows but no dem_chunks "
                  "template: its decoder's dem_chunks is expanded into "
                  "H_sparse"),
              std::string::npos)
        << e.what();
  }

  ingest.dem_chunks = parsed.data_artifacts.at(2).dem_chunks;
  const auto store =
      cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
  EXPECT_NO_THROW(store.find_template(*ingest.dem_chunks));
  EXPECT_EQ(records_of(store, "dem_chunks").size(), 1u);
} // end - TEST(DataArtifactStore, ANodeOnFlatMatricesHasNoTemplate)

// By name and inline, a configuration fills a store with the same template
// and window entries, in the same order; the by-name file adds its Stim DEM.
TEST(DataArtifactStore, ByNameFillsWhatInlineFills) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  const auto by_name = cudaq::qec::decoding::host::fill_data_artifact_store(
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml()),
      dir.path());
  const auto inline_form = cudaq::qec::decoding::host::fill_data_artifact_store(
      cfg::multi_decoder_config::from_yaml_str(pw_graph_yaml()), dir.path());
  for (const auto *kind : {"dem_chunks", "window"})
    EXPECT_EQ(records_of(by_name, kind), records_of(inline_form, kind)) << kind;
  EXPECT_TRUE(records_of(inline_form, "stim_dem").empty());
}

// A file that does not match its pin is rejected, naming the entry, the file
// and both hashes, and so is a missing file. Without a pin, any bytes do.
TEST(DataArtifactStore, RejectsAFileThatDoesNotMatchItsSha256) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  const auto fill = [&](const std::string &yaml) {
    cudaq::qec::decoding::host::fill_data_artifact_store(
        cfg::multi_decoder_config::from_yaml_str(yaml), dir.path());
  };
  const auto rejects = [&](const std::string &yaml, const std::string &text) {
    try {
      fill(yaml);
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  };
  rejects(artifacts_yaml(),
          "data_artifacts.rep_stim: dems/rep.dem has sha256 " +
              cudaq::qec::sha256_hex(rep_dem_text) + ", but the entry pins " +
              placeholder_pin);
  EXPECT_NO_THROW(fill(artifacts_yaml_with(
      "    sha256: " + std::string(placeholder_pin) + "\n", "")));
  std::filesystem::remove(dir.path() / "dems" / "rep.dem");
  rejects(pinned_artifacts_yaml(), "data_artifacts.rep_stim");
} // end - TEST(DataArtifactStore, RejectsAFileThatDoesNotMatchItsSha256)

// A configuration built in code skips the parser's checks, so filling checks
// a window entry's context itself: an empty one, one naming no entry, and
// one naming an entry that is not a template are each rejected, naming the
// window entry.
TEST(DataArtifactStore, RejectsAWindowWhoseContextIsNotATemplate) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  const auto parsed =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  const std::vector<std::pair<std::string, std::string>> cases{
      {"", "a window entry needs a context naming a dem_chunks entry"},
      {"nowhere", "context 'nowhere' is not a dem_chunks entry"},
      {"rep_stim", "context 'rep_stim' is not a dem_chunks entry"}};
  ASSERT_EQ(parsed.data_artifacts.at(0).name, "pw_A");
  for (const auto &[context, text] : cases) {
    auto config = parsed;
    config.data_artifacts[0].context = context;
    try {
      cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find("data_artifacts.pw_A: " + text),
                std::string::npos)
          << e.what();
    }
  }
} // end - TEST(DataArtifactStore, RejectsAWindowWhoseContextIsNotATemplate)

// An ingest layer that names a window entry is built on the node's template,
// so that entry's context must be the same template; one on another
// template is rejected, naming the layer, the entry and its context, and so
// is a layer naming an entry that is not a window. With the same template,
// the layer reuses the entry's variants.
TEST(DataArtifactStore, RejectsALayerWhoseEntryIsOnAnotherTemplate) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  const auto parsed =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  auto config = parsed;
  config.decoders.at(0).dem_chunks->phases.at(0).spec.error_rates.at(0) = 0.02;
  try {
    cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
    ADD_FAILURE() << "a layer was built on another template";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "decoder 0, node 'ingest', layer 'A': window entry 'pw_A' "
                  "is built on 'rep_memory', a different template from the "
                  "node's"),
              std::string::npos)
        << e.what();
  }
  config = parsed;
  config.decoders.at(0).task_graph->nodes.at(0).windows.at(0).artifact =
      "rep_stim";
  try {
    cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
    ADD_FAILURE() << "a layer named an entry that is not a window";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("layer 'A': 'rep_stim' is not a "
                                         "window entry"),
              std::string::npos)
        << e.what();
  }
  const auto store =
      cudaq::qec::decoding::host::fill_data_artifact_store(parsed, dir.path());
  EXPECT_EQ(records_of(store, "dem_chunks").size(), 1u);
} // end - TEST(DataArtifactStore, RejectsALayerWhoseEntryIsOnAnotherTemplate)

// A layer builds its own request. Once that differs from the entry it names,
// to_yaml_str() writes it inline, since the name no longer describes it; so
// the entry's template is no longer checked against the node's.
TEST(DataArtifactStore, ALayerBuildsItsOwnRequest) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  auto config =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  auto &decoder = config.decoders.at(0);
  auto &layers = decoder.task_graph->nodes.at(0).windows;
  decoder.dem_chunks->phases.at(0).spec.error_rates.at(0) = 0.02;
  for (auto &layer : layers)
    ++layer.request.offset;
  const auto store =
      cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
  const auto tmpl = store.find_template(*decoder.dem_chunks);
  for (const auto &layer : layers)
    for (const auto &key : variant_keys(*tmpl, layer.request))
      EXPECT_NO_THROW(store.find_window(*tmpl, layer.request, key));
} // end - TEST(DataArtifactStore, ALayerBuildsItsOwnRequest)

// A configuration built in code skips the parser, so filling rejects a pin
// on an inline Stim DEM as parsing does.
TEST(DataArtifactStore, RejectsAPinOnAnInlineStimDem) {
  namespace cfg = cudaq::qec::decoding::config;
  const ScopedArtifactDir dir;
  auto config =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  auto &entry = config.data_artifacts.at(3);
  ASSERT_EQ(entry.name, "rep_stim");
  entry.stim_dem = rep_dem_text;
  entry.stim_dem_path.clear();
  try {
    cudaq::qec::decoding::host::fill_data_artifact_store(config, dir.path());
    ADD_FAILURE() << "a pin on an inline Stim DEM was accepted";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "data_artifacts.rep_stim: sha256 pins a file, so it needs a "
                  "file-backed entry such as stim_dem_path"),
              std::string::npos)
        << e.what();
  }
} // end - TEST(DataArtifactStore, RejectsAPinOnAnInlineStimDem)

// A caller reads each pinned file once, and both the pinned copy of the
// configuration and the store take its text and hash from that read, so with
// the file gone afterwards both still work. A read missing a pinned entry, or
// made against another pin, is a logic error, not a way past the pin: a
// configuration whose entry now pins other bytes cannot reuse it.
TEST(DataArtifactStore, ReusesThePinnedReads) {
  namespace cfg = cudaq::qec::decoding::config;
  namespace host = cudaq::qec::decoding::host;
  const ScopedArtifactDir dir;
  const host::pinned_stim_dems none;
  const auto config =
      cfg::multi_decoder_config::from_yaml_str(pinned_artifacts_yaml());
  auto repinned = config;
  const auto pinned = host::read_pinned_stim_dems(config, dir.path());
  const auto hash = cudaq::qec::sha256_hex(rep_dem_text);
  EXPECT_EQ(pinned,
            (host::pinned_stim_dems{{"rep_stim", {rep_dem_text, hash}}}));
  std::filesystem::remove(dir.path() / "dems" / "rep.dem");
  EXPECT_EQ(host::with_pinned_stim_dems(config, pinned), config);
  const auto store = host::fill_data_artifact_store(config, dir.path(), pinned);
  EXPECT_EQ(records_of(store, "stim_dem"),
            (std::vector<cudaq::qec::data_artifact_record>{
                {"stim_dem", std::string(rep_dem_text).size(), hash}}));
  EXPECT_THROW(host::with_pinned_stim_dems(config, none), std::logic_error);
  EXPECT_THROW(host::fill_data_artifact_store(config, dir.path(), none),
               std::logic_error);

  const auto rejects_stale_read = [&](auto &&use) {
    try {
      use();
      ADD_FAILURE() << "a read made against another pin was reused";
    } catch (const std::logic_error &e) {
      EXPECT_NE(
          std::string(e.what()).find("data_artifacts.rep_stim pins sha256 " +
                                     std::string(placeholder_pin) +
                                     ", but its text was read against " + hash),
          std::string::npos)
          << e.what();
    }
  };
  ASSERT_EQ(repinned.data_artifacts.at(3).name, "rep_stim");
  repinned.data_artifacts.at(3).sha256 = placeholder_pin;
  rejects_stale_read([&] { host::with_pinned_stim_dems(repinned, pinned); });
  rejects_stale_read(
      [&] { host::fill_data_artifact_store(repinned, dir.path(), pinned); });
} // end - TEST(DataArtifactStore, ReusesThePinnedReads)

// ---------------------------------------------------------------------------
// parallel_window pattern and pool options
// ---------------------------------------------------------------------------

// The reference's pattern form: one node, wired measurements -> pw -> output,
// with the pool's threads pinned.
std::string pattern_yaml() {
  return R"(
decoders:
  - id: 0
    dem_chunks:
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      phases:
        - {name: init, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: bulk, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
        - {name: final, spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.01, 0.01]}}
    task_graph:
      async_streaming: true
      async_coordinator_cpu: 2
      async_num_workers: 4
      async_worker_cpus: [4, 5, 6, 7]
      nodes:
        - id: pw
          type: parallel_window
          commit_rounds: 3
          buffer_rounds: 3
          gap_rounds: 9
          output: corrected_logical
          max_in_flight: 4
          inner: {type: single_error_lut}
      edges:
        - {from: measurements, to: pw}
        - {from: pw, to: output}
)";
}

// pattern_yaml() with `from` replaced by `to`.
std::string pattern_yaml_with(const std::string &from, const std::string &to) {
  auto yaml = pattern_yaml();
  const auto at = yaml.find(from);
  if (at == std::string::npos)
    throw std::logic_error("pattern_yaml has no '" + from + "'");
  return yaml.replace(at, from.size(), to);
}

// Expect parsing yaml to fail with a message containing text.
void expect_parse_error(const std::string &yaml, const std::string &text) {
  namespace cfg = cudaq::qec::decoding::config;
  try {
    cfg::multi_decoder_config::from_yaml_str(yaml);
    ADD_FAILURE() << "expected: " << text;
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
  }
}

// The pattern's parameters and the pool options parse, and the document
// round-trips; unset options are not written.
TEST(ParallelWindowPattern, ParsesThePatternForm) {
  namespace cfg = cudaq::qec::decoding::config;
  auto parsed = cfg::multi_decoder_config::from_yaml_str(pattern_yaml());
  const auto &graph = *parsed.decoders.at(0).task_graph;
  const auto &params = graph.nodes.at(0).node_params.map();
  EXPECT_EQ(graph.nodes.at(0).type, "parallel_window");
  EXPECT_EQ(params.get<std::size_t>("commit_rounds"), 3u);
  EXPECT_EQ(params.get<std::size_t>("buffer_rounds"), 3u);
  EXPECT_EQ(params.get<std::size_t>("gap_rounds"), 9u);
  EXPECT_EQ(params.get<std::string>("output"), "corrected_logical");
  EXPECT_EQ(graph.nodes.at(0).max_in_flight, 4u);
  EXPECT_EQ(graph.options.async_num_workers, std::optional<std::size_t>{4});
  EXPECT_EQ(graph.options.async_worker_cpus, (std::vector<int>{4, 5, 6, 7}));
  const auto emitted = parsed.to_yaml_str(200);
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
      << emitted;

  auto plain = cfg::multi_decoder_config::from_yaml_str(pattern_yaml_with(
      "      async_num_workers: 4\n      async_worker_cpus: [4, 5, 6, 7]\n",
      ""));
  EXPECT_FALSE(plain.decoders.at(0).task_graph->options.async_num_workers);
  EXPECT_EQ(plain.to_yaml_str(200).find("async_num_workers"),
            std::string::npos);
  EXPECT_EQ(plain.to_yaml_str(200).find("async_worker_cpus"),
            std::string::npos);
} // end - TEST(ParallelWindowPattern, ParsesThePatternForm)

// Building lowers the pattern into pw.ingest, pw.A, pw.B and pw.sum and
// validates them as a hand-written graph, so the pattern is checked end to
// end: corrected_logical needs the template's m2o_sparse, and two faults on
// one detector with different observable effects are ambiguous. With
// identical twins, or the pattern's conflicting_parallel resolving them in
// its layers, the build reaches the kinds' own report.
TEST(ParallelWindowPattern, BuildingLowersThePatternAndValidatesIt) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto unpooled = [](std::string yaml) {
    const std::string pool =
        "      async_num_workers: 4\n      async_worker_cpus: [4, 5, 6, 7]\n";
    return yaml.replace(yaml.find(pool), pool.size(), "");
  };
  const auto corrections = unpooled(
      pattern_yaml_with("output: corrected_logical", "output: corrections"));
  auto twins = corrections;
  for (auto at = twins.find("O_sparse: [0, -1]"); at != std::string::npos;
       at = twins.find("O_sparse: [0, -1]", at))
    twins.replace(at, 17, "O_sparse: [0, 1, -1]");
  expect_build_error(
      cfg::multi_decoder_config::from_yaml_str(unpooled(pattern_yaml())),
      "from_topology: parallel_window 'pw' has output: corrected_logical, "
      "which reads the logical value through the last phase's m2o_sparse, "
      "and its template has none; use output: corrections");
  auto resolved = corrections;
  resolved.replace(resolved.find("          max_in_flight: 4\n"), 0,
                   "          conflicting_parallel: most_likely\n");
  expect_build_error(cfg::multi_decoder_config::from_yaml_str(corrections),
                     "validate_commitment: pw.A[0] (position 0, terminal 3) "
                     "of the 3-round shot: commit column 0 (round 0) is "
                     "parallel to column 1 (round 0), which has a different "
                     "observable effect");
  expect_build_error(cfg::multi_decoder_config::from_yaml_str(twins),
                     "the 'ingest' kind is not available yet");
  expect_build_error(cfg::multi_decoder_config::from_yaml_str(resolved),
                     "the 'ingest' kind is not available yet");
} // end - TEST(ParallelWindowPattern, BuildingLowersThePatternAndValidatesIt)

// Filling the store from the pattern holds every variant of its two layers,
// the very windows of the parallel-window graph written by hand with the
// same sizes and prior. A pattern whose parameters or template give no
// windows is named.
TEST(ParallelWindowPattern, FillsTheStoreWithItsWindows) {
  namespace cfg = cudaq::qec::decoding::config;
  using cudaq::qec::decoding::host::fill_data_artifact_store;
  // The prior the hand-written graph states.
  const auto pattern =
      cfg::multi_decoder_config::from_yaml_str(pattern_yaml_with(
          "          max_in_flight: 4\n", "          boundary_prior: 0.01\n"
                                          "          max_in_flight: 4\n"));
  const auto by_hand =
      cfg::multi_decoder_config::from_yaml_str(pw_graph_yaml());
  const auto store = fill_data_artifact_store(pattern, ".");
  const auto tmpl = store.find_template(*pattern.decoders.at(0).dem_chunks);
  const auto requests = cudaq::qec::parallel_window_requests(
      pattern.decoders.at(0).task_graph->nodes.at(0).node_params.map(), *tmpl);
  std::size_t variants = 0;
  for (const auto &request : {requests.first, requests.second}) {
    const auto keys = variant_keys(*tmpl, request);
    variants += keys.size();
    for (const auto &key : keys)
      EXPECT_NE(store.find_window(*tmpl, request, key), nullptr);
  }
  EXPECT_EQ(records_of(store, "window").size(), variants);
  EXPECT_EQ(records_of(store, "window"),
            records_of(fill_data_artifact_store(by_hand, "."), "window"));

  auto uncrossed = cfg::multi_decoder_config::from_yaml_str(pattern_yaml());
  auto zero = uncrossed;
  uncrossed.decoders.at(0).dem_chunks = cudaq::qec::single_round_spec();
  zero.decoders.at(0).task_graph->nodes.at(0).node_params.map().insert(
      "commit_rounds", std::size_t{0});
  for (const auto &[config, text] :
       {std::pair{uncrossed, "decoder 0, node 'pw': parallel_window: no fault "
                             "crosses into the next round"},
        std::pair{zero, "decoder 0, node 'pw': parallel_window: "
                        "commit_rounds must be at least 1"}}) {
    try {
      fill_data_artifact_store(config, ".");
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
  }
} // end - TEST(ParallelWindowPattern, FillsTheStoreWithItsWindows)

// Configuring builds no decoder for a pattern, whose layers build their own
// from their windows: a pattern beside flat matrices is reported as having
// no template, and one inside an ensemble as expanding only as a node of the
// graph, with no inner decoder constructed for either.
TEST(ParallelWindowPattern, ConfiguringBuildsNoDecoderForAPattern) {
  namespace cfg = cudaq::qec::decoding::config;
  const std::string flat = R"(
decoders:
  - id: 0
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, 1, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    task_graph:
      nodes:
)";
  const std::string pattern =
      "{id: pw, type: parallel_window, commit_rounds: 3, buffer_rounds: 3, "
      "inner: {type: d_capture_decoder}}";
  const std::string top = flat + "        - " + pattern + R"(
      edges:
        - {from: measurements, to: pw}
        - {from: pw, to: output}
)";
  const std::string nested = flat + R"(        - id: e
          ensemble:
            join_strategy: deadline
            join_params: {deadline_us: 200000}
            members:
              - nodes:
                  - )" + pattern +
                             R"(
                edges: [{from: input, to: pw}, {from: pw, to: output}]
      edges:
        - {from: measurements, to: e}
        - {from: e, to: output}
)";
  for (const auto &[yaml, text] :
       {std::pair{top, std::string("from_topology: parallel_window 'pw' has "
                                   "no dem_chunks template")},
        std::pair{nested,
                  std::string("from_topology: ensemble 'e' member 0 has "
                              "parallel_window 'pw', which expands only as a "
                              "node of the graph")}}) {
    cudaq::qec::construction_d_probe::model = {};
    expect_build_error(cfg::multi_decoder_config::from_yaml_str(yaml), text);
    EXPECT_EQ(cudaq::qec::construction_d_probe::model.num_detectors, 0u);
  }
} // end - TEST(ParallelWindowPattern, ConfiguringBuildsNoDecoderForAPattern)

// Each bad pattern parameter is rejected when parsed, with its own message.
TEST(ParallelWindowPattern, RejectsBadParameters) {
  expect_parse_error(pattern_yaml_with("          commit_rounds: 3\n", ""),
                     "commit_rounds");
  expect_parse_error(
      pattern_yaml_with("          commit_rounds: 3\n",
                        "          commit_rounds: 2147483649\n"),
      "parallel_window: commit_rounds must be at most 2147483648");
  expect_parse_error(pattern_yaml_with("buffer_rounds: 3", "buffer_rounds: 0"),
                     "parallel_window: buffer_rounds must be at least 1");
  expect_parse_error(pattern_yaml_with("gap_rounds: 9", "gap_rounds: 0"),
                     "parallel_window: gap_rounds must be at least 1");
  expect_parse_error(
      pattern_yaml_with("output: corrected_logical", "output: everything"),
      "parallel_window: output must be corrected_logical or corrections");
  expect_parse_error(pattern_yaml_with("gap_rounds: 9",
                                       "gap_rounds: 9\n"
                                       "          boundary_prior: 1.5"),
                     "parallel_window: boundary_prior must be strictly "
                     "between 0 and 1");
  expect_parse_error(pattern_yaml_with("gap_rounds: 9",
                                       "gap_rounds: 9\n"
                                       "          boundary_prior: 0.01\n"
                                       "          input_boundary: truncate"),
                     "parallel_window: boundary_prior is for input_boundary "
                     "columns");
  expect_parse_error(
      pattern_yaml_with("gap_rounds: 9", "gap_rounds: 9\n"
                                         "          input_boundary: sideways"),
      "parallel_window: input_boundary must be columns or truncate");
  expect_parse_error(pattern_yaml_with("gap_rounds: 9",
                                       "gap_rounds: 9\n"
                                       "          conflicting_parallel: any"),
                     "parallel_window: conflicting_parallel must be reject "
                     "or most_likely");
  expect_parse_error(pattern_yaml_with("max_in_flight: 4", "max_in_flight: 0"),
                     "task_graph node 'pw' has max_in_flight 0");
  expect_parse_error(pattern_yaml_with("max_in_flight: 4",
                                       "max_in_flight: 4\n"
                                       "          verify_residual: true"),
                     "parallel_window: verify_residual must be none, fail or "
                     "report; got 'true'");
} // end - TEST(ParallelWindowPattern, RejectsBadParameters)

// A count built in code as a negative int never reaches the pattern's
// minimum, which would read it as a huge count and pass it.
TEST(ParallelWindowPattern, RejectsANegativeCountBuiltInCode) {
  namespace cfg = cudaq::qec::decoding::config;
  const auto parsed = cfg::multi_decoder_config::from_yaml_str(pattern_yaml());
  auto params = parsed.decoders.at(0).task_graph->nodes.at(0).node_params.map();
  params.insert("commit_rounds", -1);
  try {
    cfg::validate_custom_args(
        cudaq::qec::task_graph_node_schema_name("parallel_window"), params);
    ADD_FAILURE() << "a negative commit_rounds validated";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(
                  "Key 'commit_rounds' in 'task_graph_node.parallel_window' "
                  "parameters does not hold a non-negative int"),
              std::string::npos)
        << e.what();
  }
}

// Bad pool options are rejected when parsed, and the graph rejects set ones
// until the pool applies them.
TEST(TaskGraphPoolOptions, AreChecked) {
  expect_parse_error(
      pattern_yaml_with("async_num_workers: 4", "async_num_workers: 0"),
      "task_graph.async_num_workers must be at least 1");
  expect_parse_error(pattern_yaml_with("[4, 5, 6, 7]", "[4, -5]"),
                     "task_graph.async_worker_cpus has negative CPU -5");
  // Two workers on one CPU only warn.
  EXPECT_NO_THROW(
      (void)cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          pattern_yaml_with("[4, 5, 6, 7]", "[4, 5, 4]")));
  expect_parse_error(pattern_yaml_with("[4, 5, 6, 7]", "[2, 5]"),
                     "task_graph.async_worker_cpus includes CPU 2, which "
                     "async_coordinator_cpu uses");
} // end - TEST(TaskGraphPoolOptions, AreChecked)

// The chunk form stays the authoritative source, carrying the closed model
// expand_dem_chunks() would write for any decoder that reads matrices.
TEST(ResolveDecoderInputs, ChunkFormCarriesItsClosedModelForAMatrixDecoder) {
  const auto config = parse_one(dem_chunks_yaml(5));
  auto flat_config = config;
  ASSERT_TRUE(
      cudaq::qec::decoding::config::expand_dem_chunks(flat_config).has_value());

  const auto cwd = std::filesystem::current_path();
  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(config, cwd);
  const auto flat =
      cudaq::qec::decoding::host::resolve_decoder_init(flat_config, cwd);
  ASSERT_TRUE(inputs.has_dem_chunks());
  ASSERT_TRUE(inputs.has_matrices());
  EXPECT_EQ(inputs.num_detectors(), flat_config.syndrome_size);
  EXPECT_EQ(inputs.num_error_mechanisms(), flat_config.block_size);
  ASSERT_NE(inputs.measurement_to_detectors(), nullptr);
  ASSERT_NE(flat.measurement_to_detectors(), nullptr);
  EXPECT_EQ(inputs.measurement_to_detectors()->to_nested_csr(),
            flat.measurement_to_detectors()->to_nested_csr());
  EXPECT_EQ(inputs.detector_error_matrix().to_nested_csc(),
            flat.detector_error_matrix().to_nested_csc());
  EXPECT_EQ(inputs.observable_flips_matrix().to_nested_csr(),
            flat.observable_flips_matrix().to_nested_csr());
  EXPECT_EQ(inputs.error_rates(), flat.error_rates());

  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      config, std::move(inputs));
  EXPECT_EQ(decoder->get_syndrome_size(), flat_config.syndrome_size);
  EXPECT_EQ(decoder->get_block_size(), flat_config.block_size);
}

// The decoder is fed a measurement stream as wide as the rounds declare. The
// final round's two trailing bits are read by no detector, so a width taken
// from the largest referenced index would leave the decoder expecting 3
// measurements per shot where the stream carries 5, mis-framing every shot.
TEST(ResolveDecoderInputs, ChunkFormDWidthIsTheDeclaredMeasurementCount) {
  const auto config = parse_one(dem_chunks_yaml_with_trailing_measurements());
  ASSERT_TRUE(config.dem_chunks.has_value());
  const auto declared =
      cudaq::qec::dem_chunks_measurement_count(*config.dem_chunks);
  ASSERT_TRUE(declared.has_value());
  EXPECT_EQ(*declared, 5u);

  const auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());
  ASSERT_NE(inputs.measurement_to_detectors(), nullptr);
  EXPECT_EQ(inputs.measurement_to_detectors()->num_cols(), 5u);
  EXPECT_EQ(inputs.measurement_to_detectors()->num_rows(),
            inputs.num_detectors());
}

// A decoder that declared it reads chunks gets them without their closed
// model, and the same D. The declaration is made against the decoder name,
// with no schema involved: a chunk-native decoder need not take custom args.
TEST(ResolveDecoderInputs, ChunkFormLeavesAChunkDecoderWithoutMatrices) {
  cudaq::qec::register_decoder_native_model_source(
      "test_chunk_native_decoder",
      cudaq::qec::decoder_model_source::dem_chunks);

  auto config = parse_one(dem_chunks_yaml(5));
  const auto cwd = std::filesystem::current_path();
  const auto closed =
      cudaq::qec::decoding::host::resolve_decoder_init(config, cwd);
  config.type = "test_chunk_native_decoder";
  const auto inputs =
      cudaq::qec::decoding::host::resolve_decoder_init(config, cwd);
  ASSERT_TRUE(inputs.has_dem_chunks());
  EXPECT_FALSE(inputs.has_matrices());
  EXPECT_TRUE(inputs.has_observable_model());
  EXPECT_EQ(inputs.num_detectors(), closed.num_detectors());
  EXPECT_EQ(inputs.num_error_mechanisms(), closed.num_error_mechanisms());
  ASSERT_NE(inputs.measurement_to_detectors(), nullptr);
  EXPECT_EQ(inputs.measurement_to_detectors()->to_nested_csr(),
            closed.measurement_to_detectors()->to_nested_csr());
}

// A config carrying both forms parses as flat, which is what
// expand_dem_chunks() leaves behind. It cannot be an error for that reason,
// but a hand-written document that reaches it has silently lost its whole
// chunk spec, so the parse has to say so.
TEST(DecoderChunkFormTest, BothFormsWarnAndFlatFormWins) {
  const std::string chunks = dem_chunks_yaml(3);
  const auto insert_at = chunks.find("    dem_chunks:");
  ASSERT_NE(insert_at, std::string::npos);
  const std::string yaml = chunks.substr(0, insert_at) +
                           "    block_size: 2\n"
                           "    syndrome_size: 2\n"
                           "    H_sparse: [0, -1, 1, -1]\n"
                           "    O_sparse: [0, -1]\n"
                           "    D_sparse: [0, -1, 1, -1]\n" +
                           chunks.substr(insert_at);

  const auto previous = cudaq::qec::detail::get_log_level();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::warn);
  testing::internal::CaptureStderr();
  const auto config = parse_one(yaml);
  cudaq::qec::detail::flush_logs();
  const std::string logged = testing::internal::GetCapturedStderr();
  cudaq::qec::detail::set_log_level(previous);

  EXPECT_NE(logged.find("both H_sparse and dem_chunks"), std::string::npos)
      << logged;
  // The flat matrix wins, and dem_chunks survives so the document still
  // round-trips.
  EXPECT_EQ(config.syndrome_size, 2u);
  EXPECT_EQ(config.H_sparse, (std::vector<std::int64_t>{0, -1, 1, -1}));
  EXPECT_TRUE(config.dem_chunks.has_value());
}

// The warning must stay quiet for a config that uses the chunk form as
// intended, otherwise it is noise on the common path.
TEST(DecoderChunkFormTest, ChunkFormAloneDoesNotWarn) {
  const auto previous = cudaq::qec::detail::get_log_level();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::warn);
  testing::internal::CaptureStderr();
  const auto config = parse_one(dem_chunks_yaml(3));
  cudaq::qec::detail::flush_logs();
  const std::string logged = testing::internal::GetCapturedStderr();
  cudaq::qec::detail::set_log_level(previous);

  EXPECT_EQ(logged.find("both H_sparse and dem_chunks"), std::string::npos)
      << logged;
  EXPECT_TRUE(config.H_sparse.empty());
}

TEST(DecoderChunkFormTest, HSparseStillRequiredWithoutChunks) {
  const std::string yaml = R"(
decoders:
  - id: 7
    type: multi_error_lut
    block_size: 3
    syndrome_size: 3
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
)";
  try {
    cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
    FAIL() << "expected a missing-H_sparse failure";
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("H_sparse"), std::string::npos) << message;
  }
}

TEST(DecoderChunkFormTest, IncompleteFlatConfigIsRejectedAtParse) {
  const std::vector<std::pair<std::string, std::string>> omissions{
      {"block_size", "    syndrome_size: 3\n    H_sparse: [0, -1, 1, -1, 2, "
                     "-1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1, 1, -1, "
                     "2, -1]\n"},
      {"syndrome_size", "    block_size: 3\n    H_sparse: [0, -1, 1, -1, 2, "
                        "-1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1, 1, "
                        "-1, 2, -1]\n"},
      {"O_sparse", "    block_size: 3\n    syndrome_size: 3\n    H_sparse: [0, "
                   "-1, 1, -1, 2, -1]\n    D_sparse: [0, -1, 1, -1, 2, -1]\n"},
      {"D_sparse", "    block_size: 3\n    syndrome_size: 3\n    H_sparse: [0, "
                   "-1, 1, -1, 2, -1]\n    O_sparse: [0, -1]\n"},
  };

  for (const auto &[omitted, body] : omissions) {
    const std::string yaml =
        "decoders:\n  - id: 0\n    type: multi_error_lut\n" + body;
    try {
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected rejection when " << omitted << " is omitted";
    } catch (const std::runtime_error &error) {
      const std::string message = error.what();
      EXPECT_NE(message.find(omitted), std::string::npos) << message;
    }
  }
}

TEST(DecoderChunkFormTest, ConfigWithNeitherFormIsRejectedAtParse) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: multi_error_lut
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml),
      std::runtime_error);
}

TEST(DecoderChunkFormTest, NumRoundsBelowTwoIsRejected) {
  // num_rounds must be at least 2 (init + final).
  for (const unsigned rounds : {0u, 1u}) {
    const auto yaml = dem_chunks_yaml(rounds);
    try {
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected rejection of num_rounds: " << rounds;
    } catch (const std::runtime_error &error) {
      EXPECT_NE(std::string(error.what()).find("num_rounds"), std::string::npos)
          << error.what();
    }
  }
  // Two rounds (init + final, no bulk) is accepted.
  const auto config = parse_one(dem_chunks_yaml(2));
  EXPECT_EQ(config.dem_chunks->num_rounds, 2u);
}

TEST(DecoderChunkFormTest, JsonSchemaNumRoundsMinimumMatchesTheParser) {
  const auto schema =
      cudaq::qec::decoding::config::decoder_config_json_schema();
  const auto at = schema.find("\"num_rounds\"");
  ASSERT_NE(at, std::string::npos);
  EXPECT_NE(schema.find("\"minimum\": 2", at), std::string::npos)
      << schema.substr(at, 200);
}

TEST(DecoderChunkFormTest, StreamingConfigParsesWithoutNumRounds) {
  const auto yaml = dem_chunks_yaml(3);
  const auto without = yaml.substr(0, yaml.find("      num_rounds:")) +
                       yaml.substr(yaml.find("      phases:"));
  const auto config = parse_one(without);
  ASSERT_TRUE(config.dem_chunks.has_value());
  EXPECT_FALSE(config.dem_chunks->num_rounds.has_value());

  // The schema has to agree, or a validating tool would reject a streaming
  // configuration the parser accepts.
  const auto schema =
      cudaq::qec::decoding::config::decoder_config_json_schema();
  const auto at = schema.find("\"dem_chunks\"");
  ASSERT_NE(at, std::string::npos);
  const auto required = schema.find("\"required\"", at);
  ASSERT_NE(required, std::string::npos);
  EXPECT_EQ(schema.find("num_rounds", required),
            schema.find("num_rounds", schema.find(']', required)))
      << schema.substr(required, 200);

  // Expansion is where the missing round count is caught.
  auto expandable = config;
  EXPECT_THROW(cudaq::qec::decoding::config::expand_dem_chunks(expandable),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// stim_dem_path: the DEM text itself as the model a decoder is built from
// ---------------------------------------------------------------------------

namespace {

// Four detectors, six error mechanisms and one observable. The fourth detector
// coordinate is the colour/basis annotation chromobius decodes from, matching
// the model used by the chromobius unit tests; other decoders ignore it.
constexpr const char *kStimDem = R"DEM(
error(0.1) D0 D1
error(0.1) D0 D1 D2
error(0.1) D0 L0
error(0.1) D1 D2 D3
error(0.1) D2 D3
error(0.1) D3
detector(0, 0, 0, 1) D0
detector(1, 0, 0, 2) D1
detector(2, 0, 0, 0) D2
detector(3, 0, 0, 1) D3
)DEM";

// One D row per detector in kStimDem, mapping detector i to measurement bit i.
const std::vector<std::int64_t> kStimDemDSparse{0, -1, 1, -1, 2, -1, 3, -1};

/// Writes the model to a file, since a config names a DEM by path.
class ScopedServerDemFile {
public:
  ScopedServerDemFile() {
    static std::atomic<int> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            ("cudaqx-qec-test-" + std::to_string(getpid()) + "-" +
             std::to_string(counter++) + ".dem");
    std::ofstream out(path_);
    out << kStimDem;
    out.close();
    // Everything below reads this back through the config, so a failed write
    // has to fail here rather than as a confusing DEM parse error later.
    EXPECT_TRUE(out) << "could not write " << path_;
  }
  ~ScopedServerDemFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  std::string path() const { return path_.string(); }

private:
  std::filesystem::path path_;
};

std::string dem_form_yaml(const std::string &dem_path,
                          const std::string &extra_keys = {}) {
  return R"(
decoders:
  - id: 0
    type: multi_error_lut
    stim_dem_path: )" +
         dem_path + R"(
    D_sparse: [0, -1, 1, -1, 2, -1, 3, -1]
)" + extra_keys;
}

} // namespace

TEST(DecoderDemFormTest, ParsesADemFormDocument) {
  ScopedServerDemFile dem;
  const auto config = parse_one(dem_form_yaml(dem.path()));
  EXPECT_EQ(config.stim_dem_path, dem.path());
  EXPECT_TRUE(config.H_sparse.empty());
  EXPECT_TRUE(config.O_sparse.empty());
  EXPECT_EQ(config.block_size, 0u);
  EXPECT_EQ(config.syndrome_size, 0u);
  EXPECT_EQ(config.D_sparse, kStimDemDSparse);
}

TEST(DecoderDemFormTest, ArbitraryLengthModeRoundTripsAndExecutes) {
  auto yaml = dem_chunks_yaml(5);
  const std::string type = "    type: sample_decoder\n";
  yaml.replace(yaml.find(type), type.size(), R"(    task_graph:
      nodes:
        - id: memory
          type: sliding_window
          window_size: 5
          step_size: 3
          guard_rounds: 1
          arbitrary_length: true
          inner: {type: single_error_lut}
)");
  const std::string old_h = "H_sparse: [0, 1, -1]";
  for (auto at = yaml.find(old_h); at != std::string::npos;
       at = yaml.find(old_h))
    yaml.replace(at, old_h.size(), "H_sparse: [0, -1]");
  namespace cfg = cudaq::qec::decoding::config;
  auto config = cfg::multi_decoder_config::from_yaml_str(yaml);
  auto roundtrip =
      cfg::multi_decoder_config::from_yaml_str(config.to_yaml_str(200));
  auto graph =
      cudaq::qec::decoding::host::build_task_graph(roundtrip.decoders.front());
  const std::array<uint8_t, 6> shot{0, 0, 0, 0, 0, 1};
  graph->enqueue_syndrome(shot.data(), shot.size(),
                          cudaq::qec::round_type::FINAL);
  EXPECT_EQ(graph->get_obs_corrections()[0], 1);
  EXPECT_EQ(graph->streaming_result().outputs.front().coverage_end_round,
            shot.size());
}

TEST(DecoderDemFormTest, EmittedDocumentReParsesAsDemForm) {
  ScopedServerDemFile dem;
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(parse_one(dem_form_yaml(dem.path())));

  // The fields DEM form derives have to stay absent, or the emitted document
  // would come back as an incomplete flat configuration.
  const auto emitted = multi_config.to_yaml_str(200);
  EXPECT_NE(emitted.find("stim_dem_path"), std::string::npos) << emitted;
  EXPECT_EQ(emitted.find("H_sparse"), std::string::npos) << emitted;
  EXPECT_EQ(emitted.find("O_sparse"), std::string::npos) << emitted;
  EXPECT_EQ(emitted.find("block_size"), std::string::npos) << emitted;
  EXPECT_EQ(emitted.find("syndrome_size"), std::string::npos) << emitted;

  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped, multi_config);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);
}

TEST(DecoderDemFormTest, RejectsStimDemPathTogetherWithDemChunks) {
  ScopedServerDemFile dem;
  // Each names the whole DEM on its own, and there is no precedence between
  // them the way a flat H_sparse wins over dem_chunks.
  const auto yaml =
      dem_chunks_yaml(3) + R"(    stim_dem_path: )" + dem.path() + "\n";
  try {
    cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
    ADD_FAILURE() << "expected dem_chunks + stim_dem_path to be rejected";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("stim_dem_path"),
              std::string::npos)
        << error.what();
  }
}

TEST(DecoderDemFormTest, RejectsStimDemPathTogetherWithFlatMatrices) {
  ScopedServerDemFile dem;
  for (const char *derived :
       {"    H_sparse: [0, 1, 2, -1]\n", "    O_sparse: [2, -1]\n",
        "    error_rate_vec: [0.1]\n"}) {
    const auto yaml = dem_form_yaml(dem.path(), derived);
    try {
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected rejection of " << derived;
    } catch (const std::runtime_error &error) {
      EXPECT_NE(std::string(error.what()).find("stim_dem_path"),
                std::string::npos)
          << error.what();
    }
  }
}

TEST(DecoderDemFormTest, RequiresDSparseAlongsideStimDemPath) {
  ScopedServerDemFile dem;
  // The DEM describes detectors but not which measurement bits form them, so
  // this is the one model field DEM form cannot derive.
  const auto yaml = R"(
decoders:
  - id: 0
    type: multi_error_lut
    stim_dem_path: )" +
                    dem.path() + "\n";
  try {
    cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
    ADD_FAILURE() << "expected a missing D_sparse to be rejected";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("D_sparse"), std::string::npos)
        << error.what();
  }
}

// ---------------------------------------------------------------------------
// Node-level stim_dem / stim_dem_path: one model, two spellings
// ---------------------------------------------------------------------------

namespace {

/// Indent every line of `text`, for inlining a model as a YAML block scalar
/// among a node's own keys.
std::string indent_lines(const std::string &text, std::size_t spaces) {
  const std::string pad(spaces, ' ');
  std::string out;
  for (std::size_t at = 0; at < text.size();) {
    const auto eol = text.find('\n', at);
    const auto end = eol == std::string::npos ? text.size() : eol + 1;
    // Padding a blank line would leave trailing whitespace, which clang-format
    // strips and YAML does not need.
    if (end - at > 1)
      out += pad;
    out.append(text, at, end - at);
    at = end;
  }
  return out;
}

/// A one-node task_graph whose inner decoder is built from a Stim model.
/// `model_keys` carries that model in whichever spelling a test exercises, and
/// `num_rounds` sets the node's detector count -- one detector per round, so
/// it has to equal the model's for the graph to build.
std::string node_model_yaml(const std::string &model_keys,
                            unsigned num_rounds = 4,
                            const std::string &kind = "sliding_window") {
  return R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: w
          type: )" +
         kind + R"(
          inner:
            type: sample_decoder
)" + model_keys +
         R"(    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      num_rounds: )" +
         std::to_string(num_rounds) + R"(
      phases:
        - name: init
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
        - name: bulk
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
        - name: final
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
)";
}

} // namespace

TEST(DecoderNodeModelTest, NodeLevelStimDemPathParsesAndRoundTrips) {
  ScopedDemFile dem;
  const auto config = parse_one(node_model_yaml(
      "          stim_dem_path: " + dem.path().string() + "\n"));
  ASSERT_TRUE(config.task_graph.has_value());
  ASSERT_EQ(config.task_graph->nodes.size(), 1u);
  EXPECT_EQ(config.task_graph->nodes.front().stim_dem_path,
            dem.path().string());
  // The path is a path: nothing resolves it at parse time, so the sibling text
  // field stays empty until the graph builder runs.
  EXPECT_TRUE(config.task_graph->nodes.front().stim_dem.empty());

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto emitted = multi_config.to_yaml_str(200);
  EXPECT_NE(emitted.find("stim_dem_path"), std::string::npos) << emitted;
  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);

  // Field-wise rather than whole-config, because a nodes-plus-dem_chunks
  // document does not satisfy struct equality across a round trip on this
  // branch -- the difference is inside dem_chunks and is there with or without
  // this key, so it is not step 2's to assert or to fix.  The task_graph half,
  // which is the half a node-level key lives in, does compare equal.
  const auto &before = multi_config.decoders.front();
  const auto &after = round_tripped.decoders.front();
  EXPECT_EQ(before.task_graph, after.task_graph);
  EXPECT_TRUE(after.dem_chunks.has_value());
}

TEST(DecoderNodeModelTest, AsyncStreamingOptionsParseAndRoundTrip) {
  ScopedDemFile dem;
  auto yaml =
      node_model_yaml("          stim_dem_path: " + dem.path().string() + "\n");
  const std::string marker = "    task_graph:\n";
  const auto options_at = yaml.find(marker);
  ASSERT_NE(options_at, std::string::npos);
  yaml.insert(options_at + marker.size(), "      async_streaming: true\n"
                                          "      async_queue_capacity: 3\n"
                                          "      async_coordinator_cpu: 7\n"
                                          "      async_spin_pause_iters: 99\n"
                                          "      async_num_workers: 5\n"
                                          "      async_worker_cpus: [2, 3]\n");

  const auto config = parse_one(yaml);
  ASSERT_TRUE(config.task_graph.has_value());
  EXPECT_EQ(config.task_graph->options.async_num_workers, 5u);
  EXPECT_EQ(config.task_graph->options.async_worker_cpus,
            (std::vector<int>{2, 3}));
  EXPECT_TRUE(config.task_graph->options.async_streaming);
  EXPECT_EQ(config.task_graph->options.async_queue_capacity, 3u);
  EXPECT_EQ(config.task_graph->options.async_coordinator_cpu, 7);
  EXPECT_EQ(config.task_graph->options.async_spin_pause_iters, 99u);

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto emitted = multi_config.to_yaml_str(200);
  EXPECT_NE(emitted.find("async_streaming: true"), std::string::npos);
  EXPECT_NE(emitted.find("async_queue_capacity: 3"), std::string::npos);
  EXPECT_NE(emitted.find("async_coordinator_cpu: 7"), std::string::npos);
  EXPECT_NE(emitted.find("async_spin_pause_iters: 99"), std::string::npos);
  // Short keys are padded to align values.
  EXPECT_TRUE(std::regex_search(emitted, std::regex("async_num_workers: +5")));
  EXPECT_TRUE(std::regex_search(
      emitted, std::regex("async_worker_cpus: +\\[ 2, 3 \\]")));
  const auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped.decoders.front().task_graph->options,
            config.task_graph->options);
}

TEST(DecoderNodeModelTest, RejectsCoordinatorCpuBelowSentinel) {
  ScopedDemFile dem;
  auto yaml =
      node_model_yaml("          stim_dem_path: " + dem.path().string() + "\n");
  const std::string marker = "    task_graph:\n";
  const auto options_at = yaml.find(marker);
  ASSERT_NE(options_at, std::string::npos);
  yaml.insert(options_at + marker.size(), "      async_streaming: true\n"
                                          "      async_queue_capacity: 3\n"
                                          "      async_coordinator_cpu: -2\n");

  EXPECT_THROW(
      (void)cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml),
      std::runtime_error);
}

TEST(DecoderNodeModelTest, RejectsInvalidAsyncWorkerOptions) {
  ScopedDemFile dem;
  for (const char *options :
       {"      async_worker_cpus: [0, -1]\n", "      async_num_workers: 0\n"}) {
    auto yaml = node_model_yaml(
        "          stim_dem_path: " + dem.path().string() + "\n");
    const std::string marker = "    task_graph:\n";
    const auto options_at = yaml.find(marker);
    ASSERT_NE(options_at, std::string::npos);
    yaml.insert(options_at + marker.size(), options);
    EXPECT_THROW(
        (void)cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
            yaml),
        std::runtime_error)
        << options;
  }
}

TEST(ConfigureDecodersLifecycle, WarnsWhenAsyncGraphsContendForCpus) {
  using namespace cudaq::qec::decoding::config;
  if (std::thread::hardware_concurrency() < 2)
    GTEST_SKIP() << "needs two CPUs to pin";
  // Decoder 0 pins its coordinator to CPU 0 and its workers to every other
  // CPU; decoder 1 pins its workers there too. Synchronous decoder 2 pins
  // CPU 1 as well, which it ignores. The pool runs one worker per pinned
  // CPU, so each async graph has as many threads as the machine and
  // together they oversubscribe it.
  const auto cpus = std::thread::hardware_concurrency();
  const auto workers = cpus - 1;
  std::vector<int> worker_cpus;
  for (unsigned cpu = 1; cpu < cpus; ++cpu)
    worker_cpus.push_back(static_cast<int>(cpu));
  const auto async_decoder = [&](int id) {
    auto config = create_test_sample_realtime_decoder_config(id);
    config.task_graph.emplace();
    config.task_graph->type = "single_window";
    config.task_graph->inner_type = config.type;
    config.type.clear();
    config.task_graph->options.async_streaming = true;
    config.task_graph->options.async_queue_capacity = 2;
    config.task_graph->options.async_worker_cpus = worker_cpus;
    config.task_graph->options.async_num_workers = cpus;
    return config;
  };
  multi_decoder_config multi;
  multi.decoders.push_back(async_decoder(0));
  multi.decoders.back().task_graph->options.async_coordinator_cpu = 0;
  multi.decoders.push_back(async_decoder(1));
  multi.decoders.push_back(async_decoder(2));
  multi.decoders.back().task_graph->options.async_streaming = false;

  const auto previous = cudaq::qec::detail::get_log_level();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::warn);
  testing::internal::CaptureStderr();
  const int status = configure_decoders(multi);
  cudaq::qec::detail::flush_logs();
  const std::string logged = testing::internal::GetCapturedStderr();
  cudaq::qec::detail::set_log_level(previous);
  finalize_decoders();

  EXPECT_EQ(status, 0) << logged;
  EXPECT_NE(logged.find("decoders 0 and 1 both pin CPU 1"), std::string::npos)
      << logged;
  EXPECT_EQ(logged.find("decoders 0 and 2"), std::string::npos) << logged;
  EXPECT_EQ(logged.find("decoders 1 and 2"), std::string::npos) << logged;
  const auto threads = std::to_string(2 * (workers + 1));
  EXPECT_NE(logged.find("need " + threads + " threads"), std::string::npos)
      << logged;
  EXPECT_NE(logged.find("decoder 1: " + std::to_string(workers + 1)),
            std::string::npos)
      << logged;
}

TEST(DecoderNodeModelTest, NodeMaxInFlightParsesRoundTripsAndRejectsZero) {
  ScopedDemFile dem;
  auto yaml =
      node_model_yaml("          stim_dem_path: " + dem.path().string() + "\n");
  const std::string node_marker = "        - id: w\n";
  const auto node_at = yaml.find(node_marker);
  ASSERT_NE(node_at, std::string::npos);
  yaml.insert(node_at + node_marker.size(), "          max_in_flight: 3\n");
  const auto config = parse_one(yaml);
  ASSERT_TRUE(config.task_graph.has_value());
  ASSERT_EQ(config.task_graph->nodes.size(), 1u);
  EXPECT_EQ(config.task_graph->nodes.front().max_in_flight, 3u);

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto emitted = multi_config.to_yaml_str(200);
  EXPECT_TRUE(std::regex_search(emitted, std::regex("max_in_flight: +3")))
      << emitted;
  const auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(
      round_tripped.decoders.front().task_graph->nodes.front().max_in_flight,
      3u);

  auto zero =
      node_model_yaml("          stim_dem_path: " + dem.path().string() + "\n");
  zero.insert(zero.find(node_marker) + node_marker.size(),
              "          max_in_flight: 0\n");
  EXPECT_THROW(
      (void)cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          zero),
      std::runtime_error);
}

TEST(DecoderNodeModelTest, RejectsNegativeAsyncNumWorkers) {
  ScopedDemFile dem;
  auto yaml =
      node_model_yaml("          stim_dem_path: " + dem.path().string() + "\n");
  const std::string marker = "    task_graph:\n";
  const auto options_at = yaml.find(marker);
  ASSERT_NE(options_at, std::string::npos);
  yaml.insert(options_at + marker.size(), "      async_num_workers: -1\n");

  EXPECT_THROW(
      (void)cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml),
      std::runtime_error);
}

TEST(DecoderNodeModelTest, RejectsNegativeAsyncSpinPauseIters) {
  ScopedDemFile dem;
  auto yaml =
      node_model_yaml("          stim_dem_path: " + dem.path().string() + "\n");
  const std::string marker = "    task_graph:\n";
  const auto options_at = yaml.find(marker);
  ASSERT_NE(options_at, std::string::npos);
  yaml.insert(options_at + marker.size(), "      async_streaming: true\n"
                                          "      async_queue_capacity: 3\n"
                                          "      async_spin_pause_iters: -1\n");

  EXPECT_THROW(
      (void)cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml),
      std::runtime_error);
}

TEST(DecoderNodeModelTest, RejectsBothModelSpellingsOnOneNode) {
  ScopedDemFile dem;
  // The path is read into stim_dem before the builder runs, so accepting both
  // would mean one silently losing to the other.
  const auto yaml =
      node_model_yaml("          stim_dem_path: " + dem.path().string() +
                      "\n          stim_dem: |\n" + indent_lines(kStimDem, 12));
  try {
    cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
    ADD_FAILURE() << "expected stim_dem + stim_dem_path to be rejected";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("stim_dem_path"),
              std::string::npos)
        << error.what();
    EXPECT_NE(std::string(error.what()).find("exactly one form"),
              std::string::npos)
        << error.what();
  }
}

TEST(DecoderNodeModelTest, RejectsEitherModelSpellingOnADecoderlessKind) {
  ScopedDemFile dem;
  // Either spelling exists only to build an inner decoder, so a kind that
  // wraps none has no use for it.  XOR takes no decoder and no DEM.
  for (const auto &keys :
       {"          stim_dem_path: " + dem.path().string() + "\n",
        "          stim_dem: |\n" + indent_lines(kStimDem, 12)}) {
    const auto yaml = R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: merge
          type: XOR
)" + keys + R"(    H_sparse: [0, 1, -1]
    block_size: 2
    syndrome_size: 1
    O_sparse: [0, -1]
    D_sparse: [0, -1]
)";
    try {
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
      ADD_FAILURE() << "expected the model to be rejected on kind XOR: "
                    << keys;
    } catch (const std::runtime_error &error) {
      EXPECT_NE(std::string(error.what()).find("wraps no decoder"),
                std::string::npos)
          << error.what();
    }
  }
}

TEST(DecoderNodeModelTest, TheBuilderTreatsTheTwoSpellingsIdentically) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedDemFile dem(kStimDem);

  // Control: the identical node carrying no model builds from the chunk DEM.
  // It is what makes the two expectations below mean something -- a
  // `stim_dem_path` that silently failed to resolve would take this branch and
  // succeed, so "supplying a model changes the outcome" is the observation
  // that catches it.
  auto control = cfg::multi_decoder_config::from_yaml_str(node_model_yaml(""));
  EXPECT_EQ(cfg::configure_decoders(control), 0);
  cfg::finalize_decoders();

  // sample_decoder supports the observable output requested by this node. Both
  // spellings therefore have to build successfully from the same model; a
  // path that silently failed to resolve would instead fall back to the chunk
  // DEM and leave this test unable to distinguish the two inputs.
  for (const auto &keys :
       {"          stim_dem_path: " + dem.path().string() + "\n",
        "          stim_dem: |\n" + indent_lines(kStimDem, 12)}) {
    auto with_model =
        cfg::multi_decoder_config::from_yaml_str(node_model_yaml(keys));
    EXPECT_EQ(cfg::configure_decoders(with_model), 0) << keys;
    cfg::finalize_decoders();
  }
}

// ---------------------------------------------------------------------------
// Stim DEMs named in data_artifacts
// ---------------------------------------------------------------------------

namespace {

/// data_artifacts holding kStimDem by file, as by_file, pinned to `pin` when
/// one is given, and inline, as inline_text; then `rest`.
std::string stim_artifacts_yaml(const std::string &dem_path,
                                const std::string &rest,
                                const std::string &pin = {}) {
  return "data_artifacts:\n"
         "  by_file:\n"
         "    stim_dem_path: " +
         dem_path + "\n" + (pin.empty() ? "" : "    sha256: " + pin + "\n") +
         "  inline_text:\n"
         "    stim_dem: |\n" +
         indent_lines(kStimDem, 6) + rest;
}

/// A DEM-form decoder whose stim_dem holds `model`, an entry's name or text,
/// then `extra` keys.
std::string stim_dem_decoder(const std::string &model, int id = 0,
                             const std::string &extra = {}) {
  return "  - id: " + std::to_string(id) +
         "\n"
         "    type: multi_error_lut\n"
         "    stim_dem: " +
         model +
         "\n"
         "    D_sparse: [0, -1, 1, -1, 2, -1, 3, -1]\n" +
         extra;
}

/// Whether `yaml` writes `key: value`, as the emitter aligns and quotes it.
bool writes(const std::string &yaml, const std::string &key,
            const std::string &value) {
  return std::regex_search(yaml, std::regex(key + ":\\s+'?" + value + "'?\\n"));
}

/// Text without its leading and trailing whitespace, which a YAML block
/// scalar need not keep as the file has it.
std::string trimmed(const std::string &text) {
  const auto begin = text.find_first_not_of(" \n");
  const auto end = text.find_last_not_of(" \n");
  return begin == std::string::npos ? std::string{}
                                    : text.substr(begin, end - begin + 1);
}

} // namespace

// A decoder's stim_dem with no whitespace names an entry: a file-backed one
// gives it the entry's path, an inline one the entry's text. The name is
// kept, so the document writes it back and round-trips, and each decoder
// resolves to the model its file form resolves to.
TEST(StimDemReferences, DecodersNameEntries) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  const auto cwd = std::filesystem::current_path();
  auto parsed = cfg::multi_decoder_config::from_yaml_str(stim_artifacts_yaml(
      dem.path(), "decoders:\n" + stim_dem_decoder("by_file") +
                      stim_dem_decoder("inline_text", 1)));
  const auto &by_file = parsed.decoders.at(0);
  const auto &by_text = parsed.decoders.at(1);
  const auto emitted = parsed.to_yaml_str(200);
  const auto reference = cudaq::qec::decoding::host::resolve_decoder_init(
      parse_one(dem_form_yaml(dem.path())), cwd);
  EXPECT_EQ(by_file.stim_dem_artifact, "by_file");
  EXPECT_EQ(by_file.stim_dem_path, dem.path());
  EXPECT_TRUE(by_file.stim_dem.empty());
  EXPECT_EQ(by_text.stim_dem_artifact, "inline_text");
  EXPECT_EQ(by_text.stim_dem, parsed.data_artifacts.at(1).stim_dem);
  EXPECT_TRUE(by_text.stim_dem_path.empty());
  EXPECT_TRUE(writes(emitted, "stim_dem", "by_file")) << emitted;
  EXPECT_TRUE(writes(emitted, "stim_dem", "inline_text")) << emitted;
  EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
      << emitted;
  for (const auto *decoder : {&by_file, &by_text}) {
    const auto inputs =
        cudaq::qec::decoding::host::resolve_decoder_init(*decoder, cwd);
    ASSERT_TRUE(inputs.has_stim_dem());
    EXPECT_EQ(trimmed(inputs.stim_dem()), trimmed(reference.stim_dem()));
    EXPECT_EQ(inputs.num_error_mechanisms(), reference.num_error_mechanisms());
  }
} // end - TEST(StimDemReferences, DecodersNameEntries)

// A decoder's stim_dem holds the model text as a node's does, and like a
// node's it excludes stim_dem_path.
TEST(StimDemReferences, DecodersTakeStimDemText) {
  ScopedServerDemFile dem;
  cudaq::qec::decoding::config::multi_decoder_config written;
  const auto text = "|\n" + indent_lines(kStimDem, 6);
  const auto config = parse_one("decoders:\n" + stim_dem_decoder(text));
  const auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      config, std::filesystem::current_path());
  EXPECT_TRUE(config.stim_dem_artifact.empty());
  EXPECT_EQ(trimmed(config.stim_dem), trimmed(kStimDem));
  ASSERT_TRUE(inputs.has_stim_dem());
  EXPECT_EQ(inputs.num_detectors(), 4u);
  written.decoders.push_back(config);
  EXPECT_EQ(parse_one(written.to_yaml_str(200)), config);
  expect_parse_error(
      "decoders:\n" +
          stim_dem_decoder(text, 0, "    stim_dem_path: " + dem.path() + "\n"),
      "sets both stim_dem and stim_dem_path");
}

// A node's stim_dem names an entry the same way.
TEST(StimDemReferences, NodesNameEntries) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  for (const std::string name : {"by_file", "inline_text"}) {
    auto parsed = cfg::multi_decoder_config::from_yaml_str(stim_artifacts_yaml(
        dem.path(), node_model_yaml("          stim_dem: " + name + "\n")));
    const auto &node = parsed.decoders.at(0).task_graph->nodes.at(0);
    const auto &entry = parsed.data_artifacts.at(name == "by_file" ? 0 : 1);
    const auto emitted = parsed.to_yaml_str(200);
    EXPECT_EQ(node.stim_dem_artifact, name);
    EXPECT_EQ(node.stim_dem, entry.stim_dem);
    EXPECT_EQ(node.stim_dem_path, entry.stim_dem_path);
    EXPECT_TRUE(writes(emitted, "stim_dem", name)) << emitted;
    EXPECT_EQ(cfg::multi_decoder_config::from_yaml_str(emitted), parsed)
        << emitted;
  }
}

// A stim_dem with no whitespace names an entry, so one that names none, or
// names an entry of another kind, is rejected, as is a name outside a
// document with data_artifacts, or one whose file meets a stim_dem_path of
// its own.
TEST(StimDemReferences, RejectsBadNames) {
  ScopedServerDemFile dem;
  const auto with = [&](const std::string &decoders) {
    return stim_artifacts_yaml(dem.path(), "decoders:\n" + decoders);
  };
  const auto window_entry = stim_artifacts_yaml(
      dem.path(),
      "  rep_memory:\n"
      "    dem_chunks:\n"
      "      seam: {from: next_round, to: prev_round}\n"
      "      connections: [{from: bulk, to: bulk}]\n"
      "      phases:\n"
      "        - {name: bulk, spec: {num_faults: 2, H_sparse: [0, 1, -1], "
      "O_sparse: [0, -1], error_rates: [0.01, 0.01]}}\n"
      "decoders:\n" +
          stim_dem_decoder("rep_memory"));
  expect_parse_error(with(stim_dem_decoder("nowhere")),
                     "decoder 0 stim_dem names data_artifacts entry "
                     "'nowhere', which does not exist");
  expect_parse_error(window_entry, "'rep_memory' is a dem_chunks entry, not "
                                   "a Stim DEM entry");
  expect_parse_error(
      with(stim_dem_decoder("by_file", 0,
                            "    stim_dem_path: " + dem.path() + "\n")),
      "sets both stim_dem and stim_dem_path");
  try {
    cudaq::qec::decoding::config::decoder_config::from_yaml_str(
        "id: 0\ntype: multi_error_lut\nstim_dem: by_file\n"
        "D_sparse: [0, -1]\n");
    ADD_FAILURE() << "a standalone decoder resolved a name";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("outside a multi_decoder_config"),
              std::string::npos)
        << e.what();
  }
} // end - TEST(StimDemReferences, RejectsBadNames)

namespace {

/// The entries of stim_artifacts_yaml(), out of name order, and a sliding
/// window node; then a DEM-form decoder naming each entry, all as built in
/// code. The node and the decoders hold `node_name`, `file_name` and
/// `text_name` as their stim_dem.
cudaq::qec::decoding::config::multi_decoder_config
code_built_names(const std::string &dem_path, const std::string &node_name,
                 const std::string &file_name, const std::string &text_name) {
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          stim_artifacts_yaml(dem_path, node_model_yaml("")));
  auto decoder = parse_one("decoders:\n" +
                           stim_dem_decoder("|\n" + indent_lines(kStimDem, 6)));
  std::reverse(config.data_artifacts.begin(), config.data_artifacts.end());
  config.decoders.at(0).task_graph->nodes.at(0).stim_dem = node_name;
  decoder.id = 1;
  decoder.stim_dem = file_name;
  config.decoders.push_back(decoder);
  decoder.id = 2;
  decoder.stim_dem = text_name;
  config.decoders.push_back(decoder);
  return config;
}

} // namespace

// A configuration built in code names entries as a parsed one does: each
// decoder and node built from it gets the entry's model, whatever the order
// of data_artifacts. with_pinned_stim_dems() leaves the caller's
// configuration as it was; configure_decoders() resolves its names once the
// configuration is in effect, as it makes its paths absolute, so it is
// written back by name.
TEST(StimDemReferences, CodeBuiltConfigsNameEntries) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  auto config =
      code_built_names(dem.path(), "inline_text", "by_file", "inline_text");
  const auto before = config;
  const auto named = cudaq::qec::decoding::host::with_pinned_stim_dems(
      config, std::filesystem::current_path());
  const auto &node = named.decoders.at(0).task_graph->nodes.at(0);
  const auto &by_file = named.decoders.at(1);
  const auto &by_text = named.decoders.at(2);
  const auto &text = config.data_artifacts.at(0).stim_dem;
  EXPECT_EQ(config, before);
  EXPECT_EQ(node.stim_dem_artifact, "inline_text");
  EXPECT_EQ(node.stim_dem, text);
  EXPECT_EQ(by_file.stim_dem_artifact, "by_file");
  EXPECT_EQ(by_file.stim_dem_path, dem.path());
  EXPECT_TRUE(by_file.stim_dem.empty());
  EXPECT_EQ(by_text.stim_dem_artifact, "inline_text");
  EXPECT_EQ(by_text.stim_dem, text);
  ASSERT_EQ(cfg::configure_decoders(config), 0);
  cfg::finalize_decoders();
  const auto &entry = config.data_artifacts.at(1);
  const auto emitted = config.to_yaml_str(200);
  ASSERT_EQ(entry.name, "by_file");
  EXPECT_EQ(config.decoders.at(0).task_graph->nodes.at(0).stim_dem_artifact,
            "inline_text");
  EXPECT_EQ(config.decoders.at(1).stim_dem_artifact, "by_file");
  EXPECT_EQ(config.decoders.at(1).stim_dem_path, entry.stim_dem_path);
  EXPECT_EQ(config.decoders.at(2).stim_dem_artifact, "inline_text");
  EXPECT_TRUE(writes(emitted, "stim_dem", "by_file")) << emitted;
  EXPECT_TRUE(writes(emitted, "stim_dem", "inline_text")) << emitted;
} // end - TEST(StimDemReferences, CodeBuiltConfigsNameEntries)

// A configuration built in code is refused, before anything is built, for
// the names parsing refuses, its own errors naming the decoder or node.
TEST(StimDemReferences, CodeBuiltConfigsRejectBadNames) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  const auto refused = [&](cfg::multi_decoder_config config,
                           const std::string &text) {
    try {
      cudaq::qec::decoding::host::with_pinned_stim_dems(
          config, std::filesystem::current_path());
      ADD_FAILURE() << "expected: " << text;
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
    }
    const auto before = config;
    EXPECT_THROW(cfg::configure_decoders(config), std::runtime_error) << text;
    EXPECT_EQ(config, before) << text;
  };
  auto spaced = code_built_names(dem.path(), "", "by_file", "inline_text");
  auto twice = spaced;
  auto unnamed = spaced;
  auto none = spaced;
  spaced.data_artifacts.at(0).name = "inline text";
  twice.data_artifacts.at(0).name = "by_file";
  unnamed.data_artifacts.at(0).name.clear();
  none.data_artifacts.clear();
  refused(code_built_names(dem.path(), "nowhere", "by_file", "inline_text"),
          "decoder 0, task_graph node 'w' stim_dem names data_artifacts "
          "entry 'nowhere', which does not exist");
  refused(code_built_names(dem.path(), "", "by_file", "nowhere"),
          "decoder 2 stim_dem names data_artifacts entry 'nowhere', which "
          "does not exist");
  refused(none, "decoder 1 stim_dem names data_artifacts entry 'by_file', "
                "which does not exist");
  refused(spaced, "data_artifacts entry 'inline text' needs a name "
                  "without spaces");
  refused(unnamed, "data_artifacts entry '' needs a name without spaces");
  refused(twice, "data_artifacts has two entries named 'by_file'");
} // end - TEST(StimDemReferences, CodeBuiltConfigsRejectBadNames)

// A name holds no spaces, tabs or line breaks, so that a stim_dem naming it
// is told apart from model text: an entry whose name holds one is refused
// where it is declared, rather than a decoder naming it being read as a
// model.
TEST(StimDemReferences, NamesHaveNoWhitespace) {
  expect_parse_error(R"(
data_artifacts:
  "my dem":
    stim_dem: "error(0.1) D0 L0\nerror(0.1) D0 D1\n"
decoders:
  - id: 0
    type: multi_error_lut
    stim_dem: "my dem"
    D_sparse: [0, -1, 1, -1]
)",
                     "data_artifacts entry 'my dem' needs a name without "
                     "spaces");
}

// A name's error names its node, and keys arrive in no fixed order, so a
// node without an id is reported as such rather than as node ''.
TEST(StimDemReferences, NodesWithoutAnIdAreReportedAsSuch) {
  for (const std::string key : {"stim_dem", "dem_chunks"})
    expect_parse_error(R"(
decoders:
  - id: 0
    task_graph:
      nodes:
        - )" + key + R"(: nowhere
          type: sliding_window
          inner: {type: sample_decoder}
)",
                       "every task_graph node needs an id");
}

// with_pinned_stim_dems() gives each decoder and node that names a pinned
// file entry the entry's text, read once and checked against the pin, in
// place of its path, so the model built is the one checked. A name of an
// unpinned entry, and a copy edited after parsing, keep their paths.
TEST(StimDemReferences, PinnedModelsAreReadOnce) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  const auto pin = cudaq::qec::sha256_hex(kStimDem);
  const auto yaml = [&](const std::string &with) {
    return stim_artifacts_yaml(dem.path(),
                               "decoders:\n" + stim_dem_decoder("by_file") +
                                   stim_dem_decoder("by_file", 1),
                               with);
  };
  auto config = cfg::multi_decoder_config::from_yaml_str(yaml(pin));
  auto unpinned = cfg::multi_decoder_config::from_yaml_str(yaml(""));
  auto bad = cfg::multi_decoder_config::from_yaml_str(yaml(placeholder_pin));
  config.decoders.at(1).stim_dem_path = dem.path() + ".copy";
  const auto pinned = cudaq::qec::decoding::host::with_pinned_stim_dems(
      config, std::filesystem::current_path());
  const auto kept = cudaq::qec::decoding::host::with_pinned_stim_dems(
      unpinned, std::filesystem::current_path());
  EXPECT_EQ(pinned.decoders.at(0).stim_dem, kStimDem);
  EXPECT_TRUE(pinned.decoders.at(0).stim_dem_path.empty());
  EXPECT_EQ(pinned.decoders.at(0).stim_dem_artifact, "by_file");
  EXPECT_EQ(pinned.decoders.at(1), config.decoders.at(1));
  EXPECT_EQ(pinned.data_artifacts, config.data_artifacts);
  EXPECT_EQ(kept, unpinned);
  try {
    cudaq::qec::decoding::host::with_pinned_stim_dems(
        bad, std::filesystem::current_path());
    ADD_FAILURE() << "a file that does not match its pin was read";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("data_artifacts.by_file: " +
                                         dem.path() + " has sha256 " + pin),
              std::string::npos)
        << e.what();
  }
} // end - TEST(StimDemReferences, PinnedModelsAreReadOnce)

// A node that names a pinned entry is built from the checked text too.
TEST(StimDemReferences, PinnedNodeModelsAreReadOnce) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  const auto config =
      cfg::multi_decoder_config::from_yaml_str(stim_artifacts_yaml(
          dem.path(), node_model_yaml("          stim_dem: by_file\n"),
          cudaq::qec::sha256_hex(kStimDem)));
  const auto pinned = cudaq::qec::decoding::host::with_pinned_stim_dems(
      config, std::filesystem::current_path());
  const auto &node = pinned.decoders.at(0).task_graph->nodes.at(0);
  EXPECT_EQ(node.stim_dem, kStimDem);
  EXPECT_TRUE(node.stim_dem_path.empty());
  EXPECT_EQ(node.stim_dem_artifact, "by_file");
}

// A declared dimension that disagrees with the model names the model as the
// decoder gives it: the entry it names, though a pin has read the entry's
// file into text; the file of its own stim_dem_path; or stim_dem, for text
// it holds itself.
TEST(StimDemReferences, DimensionErrorsNameTheModel) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  const std::string declared = "    syndrome_size: 9\n";
  const auto config =
      cfg::multi_decoder_config::from_yaml_str(stim_artifacts_yaml(
          dem.path(),
          "decoders:\n" + stim_dem_decoder("by_file", 0, declared) +
              stim_dem_decoder("inline_text", 1, declared) +
              stim_dem_decoder("\"error(0.1) D0 D1 D2 D3\"", 2, declared) +
              "  - id: 3\n"
              "    type: multi_error_lut\n"
              "    stim_dem_path: " +
              dem.path() + "\n    D_sparse: [0, -1, 1, -1, 2, -1, 3, -1]\n" +
              declared,
          cudaq::qec::sha256_hex(kStimDem)));
  const auto pinned = cudaq::qec::decoding::host::with_pinned_stim_dems(
      config, std::filesystem::current_path());
  const std::vector<std::string> sources{"data_artifacts.by_file",
                                         "data_artifacts.inline_text",
                                         "stim_dem", dem.path()};
  ASSERT_TRUE(pinned.decoders.at(0).stim_dem_path.empty());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    SCOPED_TRACE(sources[i]);
    try {
      cudaq::qec::decoding::host::resolve_decoder_init(
          pinned.decoders.at(i), std::filesystem::current_path());
      ADD_FAILURE() << "a declared syndrome_size of 9 was accepted";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(
                    "syndrome_size (9) does not match the detector count of " +
                    sources[i] + " (4)"),
                std::string::npos)
          << e.what();
    }
  }
} // end - TEST(StimDemReferences, DimensionErrorsNameTheModel)

// The decoding server builds its decoders without configure_decoders(), and
// checks pins the same way.
TEST(StimDemReferences, TheServerChecksPins) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedDemFile dem;
  const auto base = dem.path().parent_path();
  const auto yaml = [&](const std::string &pin) {
    return "data_artifacts:\n"
           "  tiny:\n"
           "    stim_dem_path: " +
           dem.path().filename().string() + "\n    sha256: " + pin +
           "\n"
           "decoders:\n"
           "  - id: 0\n"
           "    type: d_capture_decoder\n"
           "    stim_dem: tiny\n"
           "    D_sparse: [0, -1, 1, -1]\n";
  };
  cudaq::qec::decoding_server::SessionRegistry rejected;
  cudaq::qec::decoding_server::SessionRegistry accepted;
  EXPECT_THROW(
      rejected.load_from_config(
          cfg::multi_decoder_config::from_yaml_str(yaml(placeholder_pin)),
          "test", base),
      std::runtime_error);
  EXPECT_NO_THROW(
      accepted.load_from_config(cfg::multi_decoder_config::from_yaml_str(
                                    yaml(cudaq::qec::sha256_hex(kTinyDem))),
                                "test", base));
}

// As for dem_chunks, an edit made after parsing is written inline, since the
// name no longer describes the copy.
TEST(StimDemReferences, EditsAfterParsingAreWrittenInline) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedServerDemFile dem;
  auto parsed = cfg::multi_decoder_config::from_yaml_str(stim_artifacts_yaml(
      dem.path(), "decoders:\n" + stim_dem_decoder("by_file")));
  parsed.decoders.at(0).stim_dem_path = "other.dem";
  const auto emitted = parsed.to_yaml_str(200);
  EXPECT_FALSE(writes(emitted, "stim_dem", "by_file")) << emitted;
  EXPECT_TRUE(writes(emitted, "stim_dem_path", "other.dem")) << emitted;
}

// Configuring a decoder that names a pinned entry reads the entry's file, so
// the pin is checked first. Once configured, entries' paths are absolute like
// the decoders', and the names are still written back.
TEST(StimDemReferences, ConfigureChecksPinsAndKeepsNames) {
  namespace cfg = cudaq::qec::decoding::config;
  ScopedDemFile dem;
  const auto base = dem.path().parent_path();
  const auto file = dem.path().filename().string();
  const auto yaml = [&](const std::string &pin) {
    return "data_artifacts:\n"
           "  tiny:\n"
           "    stim_dem_path: " +
           file + "\n    sha256: " + pin +
           "\n"
           "decoders:\n"
           "  - id: 0\n"
           "    type: d_capture_decoder\n"
           "    stim_dem: tiny\n"
           "    D_sparse: [0, -1, 1, -1]\n";
  };
  auto bad = cfg::multi_decoder_config::from_yaml_str(yaml(placeholder_pin));
  auto good = cfg::multi_decoder_config::from_yaml_str(
      yaml(cudaq::qec::sha256_hex(kTinyDem)));
  try {
    cfg::configure_decoders(bad, base);
    ADD_FAILURE() << "a file that does not match its pin was configured";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("data_artifacts.tiny: " + file +
                                         " has sha256 " +
                                         cudaq::qec::sha256_hex(kTinyDem)),
              std::string::npos)
        << e.what();
  }
  ASSERT_EQ(cfg::configure_decoders(good, base), 0);
  EXPECT_TRUE(std::filesystem::path(good.data_artifacts.at(0).stim_dem_path)
                  .is_absolute());
  EXPECT_EQ(good.decoders.at(0).stim_dem_path,
            good.data_artifacts.at(0).stim_dem_path);
  EXPECT_TRUE(writes(good.to_yaml_str(200), "stim_dem", "tiny"));
  cfg::finalize_decoders();
} // end - TEST(StimDemReferences, ConfigureChecksPinsAndKeepsNames)

TEST(DecoderDemFormTest, JsonSchemaAcceptsDemFormAlongsideTheOtherTwo) {
  const auto schema =
      cudaq::qec::decoding::config::decoder_config_json_schema();
  EXPECT_NE(schema.find("\"stim_dem_path\""), std::string::npos);

  // The three forms are alternatives, so none of them may appear in the
  // envelope's own required list -- that would invalidate the other two.
  // Keys are emitted alphabetically at a two-space indent, so allOf and anyOf
  // come first and carry required arrays of their own; the envelope's is the
  // one indented six spaces, directly under the $defs entry.
  const auto at = schema.find("\n    \"decoder_config\": {");
  ASSERT_NE(at, std::string::npos) << schema;
  const auto required = schema.find("\n      \"required\": [", at);
  ASSERT_NE(required, std::string::npos) << schema;
  const auto required_end = schema.find(']', required);
  ASSERT_NE(required_end, std::string::npos);
  const auto envelope = schema.substr(required, required_end - required);

  // Without this the checks below pass on any string, including the wrong
  // required array or none at all.  `id` is the only universally required
  // key, so it is the only anchor available.
  EXPECT_NE(envelope.find("\"id\""), std::string::npos) << envelope;

  // `type` is absent for the same reason the DEM forms are: a `task_graph:`
  // block supplies the decoder in its place, so requiring `type` would
  // invalidate that form exactly as requiring one DEM form invalidates the
  // other two.  Pinned from the other side by
  // test_json_schema_accepts_a_task_graph_without_a_top_level_type.
  for (const char *key :
       {"\"type\"", "stim_dem_path", "dem_chunks", "H_sparse", "D_sparse"})
    EXPECT_EQ(envelope.find(key), std::string::npos)
        << key << " in " << envelope;
}

TEST(DecoderDemFormTest, DerivesObservablesForADecoderThatInstallsNone) {
  ScopedServerDemFile dem;

  // The LUT decoders return error frames and never install an observable
  // mapping of their own, and DEM form names no O_sparse to install for them.
  // Without deriving one from the model, this decoder would come back with
  // nothing to project onto and every realtime correction would be empty.
  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "multi_error_lut";
  dc.stim_dem_path = dem.path();
  dc.D_sparse = kStimDemDSparse;

  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      dc, std::filesystem::current_path());
  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      dc, std::move(inputs));
  EXPECT_EQ(decoder->get_syndrome_size(), 4u);
  EXPECT_EQ(decoder->get_num_observables(), 1u);
}

TEST(DecoderDemFormTest, ReportsAStimDemPathThatIsNotAFile) {
  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "multi_error_lut";
  dc.stim_dem_path = std::filesystem::temp_directory_path().string();
  dc.D_sparse = kStimDemDSparse;

  // A directory opens as a stream and only fails on the first read, so the
  // error has to name the key and the path rather than the stream buffer.
  try {
    (void)cudaq::qec::decoding::host::resolve_decoder_init(
        dc, std::filesystem::current_path());
    ADD_FAILURE() << "expected a directory to be rejected";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("stim_dem_path"),
              std::string::npos)
        << error.what();
    EXPECT_NE(std::string(error.what()).find(dc.stim_dem_path),
              std::string::npos)
        << error.what();
  }
}

// The parser's exclusivity rules reached the other way: decoder_config has
// writable fields, so a caller can build one that never was a document.
TEST(DecoderDemFormTest, RejectsDemChunksWithStimDemPathAtConstruction) {
  ScopedServerDemFile dem;

  // Four rounds close to four detectors, as many as kStimDem declares, so the
  // D row check downstream cannot catch this. What reaches the decoder is the
  // chunk-derived O_sparse and error_rate_vec over an unrelated model.
  auto dc = parse_one(dem_chunks_yaml(4));
  dc.stim_dem_path = dem.path();

  try {
    (void)cudaq::qec::decoding::host::resolve_decoder_init(
        dc, std::filesystem::current_path());
    ADD_FAILURE() << "expected dem_chunks + stim_dem_path to be rejected";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("stim_dem_path"),
              std::string::npos)
        << error.what();
    EXPECT_NE(std::string(error.what()).find("dem_chunks"), std::string::npos)
        << error.what();
  }
}

// H_sparse and O_sparse are competing model sources beside a DEM. In
// particular O_sparse cannot overwrite the mapping the authoritative source
// supplies.
TEST(DecoderDemFormTest, RejectsOSparseWithStimDemPathAtConstruction) {
  ScopedServerDemFile dem;

  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "multi_error_lut";
  dc.stim_dem_path = dem.path();
  dc.D_sparse = kStimDemDSparse;
  dc.O_sparse = {2, -1};

  try {
    (void)cudaq::qec::decoding::host::resolve_decoder_init(
        dc, std::filesystem::current_path());
    ADD_FAILURE() << "expected O_sparse + stim_dem_path to be rejected";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("O_sparse"), std::string::npos)
        << error.what();
    EXPECT_NE(std::string(error.what()).find("stim_dem_path"),
              std::string::npos)
        << error.what();
  }
}

#ifdef CUDAQX_QEC_HAS_CHROMOBIUS
TEST(ChromobiusOnDecodingServer, ConstructsFromARawDemSource) {
  ScopedServerDemFile dem;

  // No H_sparse and no O_sparse: the DEM is the model, and chromobius installs
  // the observable mapping its results are expressed in.
  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "chromobius";
  dc.stim_dem_path = dem.path();
  dc.D_sparse = kStimDemDSparse;

  cudaq::qec::decoding::config::multi_decoder_config mc;
  mc.decoders.push_back(dc);
  ASSERT_EQ(cudaq::qec::decoding::config::configure_decoders(mc), 0);
  cudaq::qec::decoding::config::finalize_decoders();
}

TEST(ChromobiusOnDecodingServer, KeepsItsOwnObservableMapping) {
  ScopedServerDemFile dem;

  // Chromobius predicts observables directly and installs an identity mapping
  // over them, which is expressed in a different space than the DEM-derived
  // matrix other decoders get. Deriving one here would have to leave it alone.
  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "chromobius";
  dc.stim_dem_path = dem.path();
  dc.D_sparse = kStimDemDSparse;

  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      dc, std::filesystem::current_path());
  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
      dc, std::move(inputs));
  EXPECT_EQ(decoder->get_syndrome_size(), 4u);
  EXPECT_EQ(decoder->get_num_observables(), 1u);
  EXPECT_EQ(decoder->get_block_size(), 6u);
}

TEST(ChromobiusOnDecodingServer, MatrixOnlyConfigStillFails) {
  // The converse: without a DEM source chromobius cannot be built, which is
  // what naming a DEM lifts. If this ever passes, the test above proves
  // nothing.
  auto dc = create_test_sample_realtime_decoder_config(0);
  dc.type = "chromobius";

  cudaq::qec::decoding::config::multi_decoder_config mc;
  mc.decoders.push_back(dc);
  EXPECT_NE(cudaq::qec::decoding::config::configure_decoders(mc), 0);
  cudaq::qec::decoding::config::finalize_decoders();

  // configure_decoders() only reports a status code, so check that the refusal
  // came from chromobius rejecting a matrix and not from something incidental.
  try {
    auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
        dc, std::filesystem::current_path());
    (void)cudaq::qec::decoding::host::create_realtime_decoder(
        dc, std::move(inputs));
    ADD_FAILURE() << "expected chromobius to reject a matrix-only config";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("detector error model"),
              std::string::npos)
        << error.what();
  }
}

TEST(ChromobiusOnDecodingServer, DemReachesANestedGlobalDecoder) {
  ScopedServerDemFile dem;

  // The nesting half: a DEM-sourced entry hands the same immutable model to
  // the nested decoder after removing only the parent's measurement map.
  cudaq::qec::decoding::config::decoder_config dc;
  dc.id = 0;
  dc.type = "trt_decoder";
  dc.stim_dem_path = dem.path();
  dc.D_sparse = kStimDemDSparse;
  dc.decoder_custom_args.map().insert("global_decoder",
                                      std::string{"chromobius"});
  dc.decoder_custom_args.map().insert("global_decoder_params",
                                      cudaqx::heterogeneous_map{});

  auto inputs = cudaq::qec::decoding::host::resolve_decoder_init(
      dc, std::filesystem::current_path());
  ASSERT_TRUE(inputs.has_stim_dem());
  auto global_decoder =
      cudaq::qec::decoder::get("chromobius", inputs.decoder_init_without_d(),
                               cudaq::qec::decode_result_type::observables, {});
  EXPECT_EQ(global_decoder->get_syndrome_size(), 4);
}
#endif // CUDAQX_QEC_HAS_CHROMOBIUS

// ---------------------------------------------------------------------------
// Syndrome mapping table -- config surface only.  Nothing consumes a row yet;
// these pin what a document may say.
// ---------------------------------------------------------------------------

namespace {
std::string syndrome_mapping_yaml(const std::string &rows) {
  return R"(decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
)" + rows;
}
} // namespace

TEST(SyndromeMappingYAMLTest, ParsesBulkAndFinalRows) {
  const auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          syndrome_mapping_yaml(R"(syndrome_mapping:
  - {id: 0, type: bulk, vps: [[]]}
  - {id: 7, type: final, vps: [[]]}
)"));
  ASSERT_EQ(config.syndrome_mapping.size(), 2u);
  EXPECT_EQ(config.syndrome_mapping[0].id, 0u);
  EXPECT_EQ(config.syndrome_mapping[0].type, "bulk");
  EXPECT_EQ(config.syndrome_mapping[1].id, 7u);
  EXPECT_EQ(config.syndrome_mapping[1].type, "final");
}

TEST(SyndromeMappingYAMLTest, SectionIsOptionalAndTypeDefaultsToBulk) {
  // The optional half is what makes step 3 behavior-neutral: every
  // configuration written before this section existed still parses.
  const auto absent =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          syndrome_mapping_yaml(""));
  EXPECT_TRUE(absent.syndrome_mapping.empty());

  // A row is just an id and a shape: `type` defaults to bulk and `vps` to the
  // single pass-through port, which is the only shape the runtime honors.
  //
  // Declaring rows is additive at the registry: id 0 stays a pass-through BULK
  // row unless the document names it, so adding only a `final` row does not
  // strip the id every producer sends.  Pinned by
  // SessionRegistryLoad.DeclaringAFinalRowKeepsIdZeroUsable -- this test is
  // about parsing alone, so it does not imply id 3 is usable on its own.
  const auto defaulted =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          syndrome_mapping_yaml(
              "syndrome_mapping:\n  - {id: 1, type: final}\n"));
  ASSERT_EQ(defaulted.syndrome_mapping.size(), 1u);
  EXPECT_EQ(defaulted.syndrome_mapping[0].id, 1u);
  EXPECT_EQ(defaulted.syndrome_mapping[0].vps,
            (std::vector<std::vector<uint64_t>>{{}}));
}

TEST(SyndromeMappingYAMLTest, RejectsMalformedRows) {
  const auto rejected = [](const char *rows) {
    try {
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          syndrome_mapping_yaml(rows));
      return std::string("accepted");
    } catch (const std::runtime_error &e) {
      return std::string(e.what());
    }
  };

  EXPECT_NE(
      rejected("syndrome_mapping:\n  - {id: 0, type: middle, vps: [[]]}\n")
          .find("bulk"),
      std::string::npos);
  // Note `vps: []` is NOT tested here: mapOptional cannot distinguish an
  // explicitly empty sequence from an absent key, so it normalizes to the
  // pass-through default.  Harmless -- both mean "no port configuration" and
  // the server behaves identically -- but do not add an assertion for it, it
  // cannot fail.
  // Duplicate keys would otherwise resolve arbitrarily when the table is built.
  EXPECT_NE(rejected("syndrome_mapping:\n  - {id: 2, vps: [[]]}\n  - {id: 2, "
                     "type: final, vps: [[]]}\n")
                .find("Duplicate"),
            std::string::npos);
  // Per-port scatter is parsed but not honored at runtime, so a populated
  // index list is refused rather than silently ignored.
  EXPECT_NE(rejected("syndrome_mapping:\n  - {id: 0, vps: [[0,1]]}\n")
                .find("does not scatter"),
            std::string::npos);
  EXPECT_NE(rejected("syndrome_mapping:\n  - {id: 0, vps: [[],[]]}\n")
                .find("single pass-through"),
            std::string::npos);
}

TEST(SyndromeMappingYAMLTest, JsonSchemaAcceptsTheSectionAndPolicesRows) {
  const auto schema =
      cudaq::qec::decoding::config::decoder_config_json_schema();
  // Without a root property the schema's additionalProperties:false would
  // reject a document the parser accepts.
  EXPECT_NE(schema.find("\"syndrome_mapping\""), std::string::npos);
  EXPECT_NE(schema.find("\"syndrome_mapping_config\""), std::string::npos);
  // Not vacuous: the row schema constrains `type` rather than allowing any
  // string.
  EXPECT_NE(schema.find("\"bulk\""), std::string::npos);
  EXPECT_NE(schema.find("\"final\""), std::string::npos);
}
