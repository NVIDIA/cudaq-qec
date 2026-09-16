/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SessionRegistry.h"
#include "../lib/realtime/realtime_decoding.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>
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
  d_capture_decoder(decoder_init inputs, decode_result_type requested_output,
                    const cudaqx::heterogeneous_map &)
      : decoder(std::move(inputs), requested_output) {
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
          decoder_init inputs, std::optional<decode_result_type> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<d_capture_decoder>(
            std::move(inputs), output.value_or(decode_result_type::observables),
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

  // Constructor-facing parameters also leave the section absent because an
  // unregistered global decoder has no schema from which to materialize it.
  auto params =
      cudaq::qec::decoding::host::prepare_decoder_params(parsed.decoders[0]);
  EXPECT_FALSE(params.contains("global_decoder_params"));
  EXPECT_FALSE(params.contains("O"));
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
      parsed, cudaq::qec::decoding::host::resolve_decoder_init(
                  parsed, std::filesystem::current_path()));
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

// Every param_kind accepts its canonical type and RelatedTypesMap entries, and
// a wrong type names that kind in the error. Equality covers the integral
// family, float cross-family, and unequal map shapes.
TEST(DecoderSchemaTest, KindMatrixAndEqualityFamily) {
  using namespace cudaq::qec::decoding::config;
  register_decoder_schema(
      {"cc_test_all_kinds",
       {{"flag", param_kind::boolean},
        {"count", param_kind::int32},
        {"n", param_kind::uint64},
        {"gain", param_kind::f64},
        {"name", param_kind::string},
        {"weights", param_kind::f64_vec},
        {"matrix", param_kind::f64_matrix},
        {"nested", param_kind::subschema, false, "cc_test_nested_kind"},
        {"engine_params", param_kind::discriminated, false, "", "engine"}}});
  register_decoder_schema({"cc_test_nested_kind", {{"gain", param_kind::f64}}});
  register_decoder_schema({"cc_test_engine", {{"gain", param_kind::f64}}});

  auto expect_ok = [](const char *key, auto value) {
    cudaqx::heterogeneous_map args;
    args.insert(key, value);
    EXPECT_NO_THROW(validate_custom_args("cc_test_all_kinds", args)) << key;
  };
  expect_ok("flag", true);
  expect_ok("flag", 1);
  expect_ok("count", 3);
  expect_ok("count", std::size_t{3});
  expect_ok("n", std::size_t{4});
  expect_ok("n", 4);
  expect_ok("gain", 1.25);
  expect_ok("gain", 1.25f);
  expect_ok("name", std::string("x"));
  expect_ok("weights", std::vector<double>{1.0, 2.0});
  expect_ok("matrix", std::vector<std::vector<double>>{{1.0}, {2.0}});
  {
    cudaqx::heterogeneous_map nested;
    nested.insert("gain", 0.5);
    expect_ok("nested", nested);
  }

  auto expect_kind = [](const char *key, auto value, const char *desc) {
    cudaqx::heterogeneous_map args;
    args.insert(key, value);
    try {
      validate_custom_args("cc_test_all_kinds", args);
      FAIL() << key << " should reject a wrong type";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find(desc), std::string::npos)
          << e.what();
    }
  };
  expect_kind("flag", std::string("no"), "boolean");
  expect_kind("count", std::string("no"), "32-bit int");
  expect_kind("n", std::string("no"), "non-negative int");
  expect_kind("gain", std::string("no"), "float");
  expect_kind("name", 1, "string");
  expect_kind("weights", 1, "list-of-float");
  expect_kind("matrix", 1, "list-of-list-of-float");
  expect_kind("nested", 1, "mapping");
  expect_kind("engine_params", 1, "mapping");

  auto eq_one = [](auto a, auto b) {
    cudaqx::heterogeneous_map left;
    left.insert("v", a);
    cudaqx::heterogeneous_map right;
    right.insert("v", b);
    return custom_args_maps_equal(left, right);
  };
  EXPECT_TRUE(eq_one(int{5}, long{5}));
  EXPECT_TRUE(eq_one(long{5}, static_cast<long long>(5)));
  EXPECT_TRUE(eq_one(static_cast<short>(5), int{5}));
  EXPECT_TRUE(eq_one(5u, static_cast<unsigned long>(5)));
  EXPECT_TRUE(eq_one(static_cast<unsigned long>(5),
                     static_cast<unsigned long long>(5)));
  EXPECT_TRUE(eq_one(static_cast<unsigned short>(5), 5u));
  EXPECT_TRUE(eq_one(5, 5.0));
  EXPECT_TRUE(eq_one(5.0f, 5));
  EXPECT_FALSE(eq_one(5, 6.0));
  EXPECT_FALSE(eq_one(5.0, 6));
  EXPECT_FALSE(eq_one(true, 1));
  EXPECT_FALSE(eq_one(1, true));
  EXPECT_FALSE(eq_one(std::string("a"), std::vector<double>{1.0}));
  EXPECT_FALSE(eq_one(std::vector<double>{1.0},
                      std::vector<std::vector<double>>{{1.0}}));
  EXPECT_FALSE(eq_one(std::vector<int>{1}, 1));

  cudaqx::heterogeneous_map bigger;
  bigger.insert("a", 1);
  bigger.insert("b", 2);
  cudaqx::heterogeneous_map smaller;
  smaller.insert("a", 1);
  EXPECT_FALSE(custom_args_maps_equal(bigger, smaller));
  cudaqx::heterogeneous_map other;
  other.insert("a", 1);
  other.insert("c", 2);
  EXPECT_FALSE(custom_args_maps_equal(bigger, other));
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

TEST(DecoderYAMLTest, PrepareDecoderParamsSurfacesCudaDeviceId) {
  // Placement is common factory policy, so every decoder type receives it.
  auto config = create_test_empty_decoder_config(0);
  config.cuda_device_id = 3;
  auto params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  ASSERT_TRUE(params.contains("cuda_device_id"));
  EXPECT_EQ(params.get<int>("cuda_device_id"), 3);

  // Absent -> key absent (decoder::get() treats absence as unpinned).
  auto config2 = create_test_empty_decoder_config(1);
  auto params2 = cudaq::qec::decoding::host::prepare_decoder_params(config2);
  EXPECT_FALSE(params2.contains("cuda_device_id"));

  // The TRT type follows the same common path without constructing a decoder.
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
  // required array or none at all.
  EXPECT_NE(envelope.find("\"id\""), std::string::npos) << envelope;
  EXPECT_NE(envelope.find("\"type\""), std::string::npos) << envelope;

  for (const char *key :
       {"stim_dem_path", "dem_chunks", "H_sparse", "D_sparse"})
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

TEST(DecoderSchemaTest, F64MatrixAndSubschemaRoundTripAndJsonSchema) {
  using namespace cudaq::qec::decoding::config;

  register_decoder_schema(
      {"cc_test_nested_engine", {{"gain", param_kind::f64}}});
  register_decoder_schema(
      {"cc_test_matrix_decoder",
       {{"weights", param_kind::f64_matrix},
        {"nested", param_kind::subschema, false, "cc_test_nested_engine"},
        {"gain", param_kind::f64}}});

  const std::string yaml = R"(
decoders:
  - id: 0
    type: cc_test_matrix_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      gain: 1.25
      weights:
        - [1.0, 0.0]
        - [0.0, 1.0]
      nested:
        gain: 2.5
)";
  auto config = multi_decoder_config::from_yaml_str(yaml);
  const auto &args = config.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(args.get<double>("gain"), 1.25);
  EXPECT_EQ(args.get<std::vector<std::vector<double>>>("weights"),
            (std::vector<std::vector<double>>{{1.0, 0.0}, {0.0, 1.0}}));
  EXPECT_EQ(args.get<cudaqx::heterogeneous_map>("nested").get<double>("gain"),
            2.5);

  const auto emitted = config.to_yaml_str(200);
  EXPECT_NE(emitted.find("weights"), std::string::npos);
  EXPECT_NE(emitted.find("nested"), std::string::npos);
  auto round_tripped = multi_decoder_config::from_yaml_str(emitted);
  EXPECT_EQ(round_tripped, config);

  const auto schema = decoder_config_json_schema();
  EXPECT_NE(schema.find("cc_test_matrix_decoder"), std::string::npos);
  EXPECT_NE(schema.find("cc_test_nested_engine"), std::string::npos);
}

TEST(DecoderSchemaTest, JsonPointerEscapesTildeAndSlashInSchemaName) {
  using namespace cudaq::qec::decoding::config;
  register_decoder_schema({"cc~test/name", {{"x", param_kind::f64}}});
  const auto schema = decoder_config_json_schema();
  EXPECT_NE(schema.find("cc~0test~1name"), std::string::npos) << schema;
}

TEST(DecoderSchemaTest, UnknownKeyWarnsOnYamlEmit) {
  using namespace cudaq::qec::decoding::config;
  register_decoder_schema(
      {"cc_test_unknown_key_decoder", {{"gain", param_kind::f64}}});

  decoder_config dc;
  dc.id = 0;
  dc.type = "cc_test_unknown_key_decoder";
  dc.block_size = 1;
  dc.syndrome_size = 1;
  dc.H_sparse = {0, -1};
  dc.O_sparse = {0, -1};
  dc.D_sparse = {0, -1};
  dc.decoder_custom_args.map().insert("gain", 0.5);
  dc.decoder_custom_args.map().insert("not_in_schema", 1);
  multi_decoder_config multi;
  multi.decoders.push_back(dc);

  const auto previous = cudaq::qec::detail::get_log_level();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::warn);
  testing::internal::CaptureStderr();
  const auto emitted = multi.to_yaml_str(200);
  cudaq::qec::detail::flush_logs();
  const std::string logged = testing::internal::GetCapturedStderr();
  cudaq::qec::detail::set_log_level(previous);

  EXPECT_NE(logged.find("not_in_schema"), std::string::npos) << logged;
  EXPECT_EQ(emitted.find("not_in_schema"), std::string::npos);
}

TEST(DecoderDemChunksYAMLTest, SeamSpecsMappingRoundTripsWithoutExpand) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: sample_decoder
    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {from: init, to: final}
      num_rounds: 2
      phases:
        - name: init
          spec:
            num_faults: 1
            seam_specs:
              - name: prev_round
                spec: {H_sparse: [0, -1], O_sparse: []}
              - name: next_round
                spec: {H_sparse: [0, -1], O_sparse: []}
            O_sparse: [0, -1]
            error_rates: [0.01]
        - name: final
          spec:
            num_faults: 1
            seam_specs:
              - name: prev_round
                spec: {H_sparse: [0, -1], O_sparse: []}
              - name: next_round
                spec: {H_sparse: [0, -1], O_sparse: []}
            O_sparse: [0, -1]
            error_rates: [0.01]
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(config.decoders.size(), 1u);
  ASSERT_TRUE(config.decoders[0].dem_chunks.has_value());
  ASSERT_FALSE(config.decoders[0].dem_chunks->phases.empty());
  EXPECT_FALSE(
      config.decoders[0].dem_chunks->phases[0].spec.seam_specs.empty());

  const auto emitted = config.to_yaml_str(200);
  EXPECT_NE(emitted.find("seam_specs"), std::string::npos);
}

TEST(DecoderDemChunksYAMLTest, ValidateRejectsErrorRateLengthMismatch) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: sample_decoder
    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: final}
      num_rounds: 2
      phases:
        - name: init
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01]
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
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml),
      std::runtime_error);
}

TEST(DecoderDemChunksYAMLTest, ExpandCloseAllRejectsANonContractedSeam) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: sample_decoder
    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {from: init, to: final}
      num_rounds: 2
      phases:
        - name: init
          spec:
            num_faults: 1
            seam_specs:
              - name: prev_round
                spec: {H_sparse: [0, -1], O_sparse: []}
              - name: next_round
                spec: {H_sparse: [0, -1], O_sparse: []}
              - name: side
                spec: {H_sparse: [0, -1], O_sparse: []}
            O_sparse: [0, -1]
            error_rates: [0.01]
        - name: final
          spec:
            num_faults: 1
            seam_specs:
              - name: prev_round
                spec: {H_sparse: [0, -1], O_sparse: []}
              - name: next_round
                spec: {H_sparse: [0, -1], O_sparse: []}
              - name: side
                spec: {H_sparse: [0, -1], O_sparse: []}
            O_sparse: [0, -1]
            error_rates: [0.01]
)";
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(config.decoders.size(), 1u);
  EXPECT_THROW(
      cudaq::qec::decoding::config::expand_dem_chunks(config.decoders[0]),
      std::runtime_error);
}

TEST(DecoderConfigTest, ConfigureFromStrLogsInfoWhenDebugDumpEnabled) {
  using namespace cudaq::qec::decoding::config;
  ScopedEnv debugEnv("CUDAQ_QEC_DEBUG_DECODER", "1");
  const auto previous = cudaq::qec::detail::get_log_level();
  cudaq::qec::detail::set_log_level(cudaq::qec::detail::log_level::info);

  multi_decoder_config multi_config;
  multi_config.decoders.push_back(create_test_empty_decoder_config(0));
  const auto yaml = multi_config.to_yaml_str(200);
  EXPECT_EQ(configure_decoders_from_str(yaml.c_str()), 0);
  finalize_decoders();

  cudaq::qec::detail::set_log_level(previous);
}
