/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "server_config.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::qec::memory_experiment {
namespace {

namespace config = cudaq::qec::decoding::config;

struct temporary_server_directory {
  std::filesystem::path path;

  temporary_server_directory() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "cudaq-memory-server-XXXXXX")
            .string();
    if (!::mkdtemp(pattern.data()))
      throw std::runtime_error("cannot create temporary server directory");
    path = pattern;
  }

  ~temporary_server_directory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

bool has_pattern_schema() {
  return find_task_graph_node_kind("parallel_window") &&
         config::find_decoder_schema(
             task_graph_node_schema_name("parallel_window"));
}

server_config_options reference_options() {
  return {.decoder = "single_error_lut",
          .commit_rounds = 3,
          .buffer_rounds = 3,
          .gap_rounds = 9};
}

// Only the parameter schema is replaced, for a real registered decoder. The
// test never constructs it and never registers a stand-in pattern or executor.
struct scoped_parameter_schema {
  config::decoder_schema original;

  scoped_parameter_schema()
      : original(*config::find_decoder_schema("single_error_lut")) {
    using config::param_kind;
    config::register_decoder_schema({"single_error_lut",
                                     {{"gain", param_kind::f64},
                                      {"weights", param_kind::f64_vec},
                                      {"matrix", param_kind::f64_matrix},
                                      {"label", param_kind::string}}});
  }

  ~scoped_parameter_schema() {
    config::register_decoder_schema(std::move(original));
  }
};

TEST(MemoryExperimentServerConfig, MissingPublicCapabilityFailsExplicitly) {
  if (has_pattern_schema())
    GTEST_SKIP() << "This build supplies the actual public pattern schema";
  try {
    validate_server_config_options(reference_options());
    FAIL() << "server configuration accepted without the public capability";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find("public parallel_window"),
              std::string::npos);
  }
}

TEST(MemoryExperimentServerConfig, PublicSchemaChecksGeometryAndDecoderArgs) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  auto options = reference_options();
  EXPECT_NO_THROW(validate_server_config_options(options));
  options.gap_rounds.reset();
  EXPECT_NO_THROW(validate_server_config_options(options));
  options.commit_rounds = 0;
  EXPECT_THROW(validate_server_config_options(options), std::exception);
  options = reference_options();
  options.buffer_rounds = 0;
  EXPECT_THROW(validate_server_config_options(options), std::exception);
  options = reference_options();
  options.gap_rounds = 0;
  EXPECT_THROW(validate_server_config_options(options), std::exception);

  options = reference_options();
  options.decoder = "not_a_registered_decoder";
  EXPECT_THROW(validate_server_config_options(options), std::invalid_argument);
  options = reference_options();
  options.decoder = "multi_error_lut";
  for (const auto &arguments : std::vector<std::vector<std::string>>{
           {"lut_error_depth=2", "lut_error_depth=3"},
           {"lut_error_depth=2junk"},
           {"lut_error_depth=2147483648"},
           {"unknown=2"},
           {"lut_error_depth.child=2"},
           {".lut_error_depth=2"},
           {"lut_error_depth"}}) {
    options.decoder_args = arguments;
    EXPECT_THROW(validate_server_config_options(options), std::exception);
  }
  options.decoder_args = {"lut_error_depth=2"};
  EXPECT_NO_THROW(validate_server_config_options(options));
}

TEST(MemoryExperimentServerConfig,
     FullYamlRoundtripPreservesTypedPrecisionAndLiteralStrings) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  ASSERT_TRUE(cudaq::qec::decoder::is_registered("single_error_lut"));
  ASSERT_NE(config::find_decoder_schema("single_error_lut"), nullptr);
  const scoped_parameter_schema schema;
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  auto options = reference_options();
  const std::string literal = "001,alpha;beta='quoted':a=b/with spaces\nnext";
  options.decoder_args = {"gain=0.12345678901234567",
                          "weights=0.9876543210123456,1.2345678901234567e-20",
                          "matrix=0.12345678901234567,0.9876543210123456;"
                          "1.2345678901234567e-20,0.12345678901234567",
                          "label=" + literal};
  const auto yaml = emit_server_config(model, options, 1);
  const auto value = config::multi_decoder_config::from_yaml_str(yaml);
  const auto &arguments = value.decoders.front()
                              .task_graph->nodes.front()
                              .decoder_custom_args.map();
  EXPECT_EQ(arguments.get<double>("gain"), 0.12345678901234567);
  EXPECT_EQ(arguments.get<std::vector<double>>("weights"),
            (std::vector<double>{0.9876543210123456, 1.2345678901234567e-20}));
  EXPECT_EQ(arguments.get<std::vector<std::vector<double>>>("matrix"),
            (std::vector<std::vector<double>>{
                {0.12345678901234567, 0.9876543210123456},
                {1.2345678901234567e-20, 0.12345678901234567}}));
  EXPECT_EQ(arguments.get<std::string>("label"), literal);
  options.decoder_args = {"label=", "weights=", "matrix="};
  EXPECT_NO_THROW(emit_server_config(model, options, 1));
  for (const auto &argument : {"gain=nan", "gain=inf", "gain=1junk",
                               "weights=1,,2", "matrix=1,x;2,3"}) {
    options.decoder_args = {argument};
    EXPECT_THROW(validate_server_config_options(options),
                 std::invalid_argument);
  }
}

TEST(MemoryExperimentServerConfig,
     RealNestedDecoderSchemaDoesNotDependOnArgumentOrder) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  auto options = reference_options();
  options.decoder = "sliding_window";
  options.decoder_args = {"inner_decoder_params.lut_error_depth=2",
                          "straddle_start_round=true", "window_size=6",
                          "step_size=3", "inner_decoder_name=multi_error_lut"};
  EXPECT_NO_THROW(validate_server_config_options(options));
  options.decoder_args.push_back("step_size=4");
  EXPECT_THROW(validate_server_config_options(options), std::invalid_argument);
}

TEST(MemoryExperimentServerConfig,
     RealPublicConfigurationRoundtripsSharedModelWithoutExecution) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  const auto options = reference_options();
  const auto yaml = emit_server_config(model, options, 4);
  EXPECT_NO_THROW(validate_server_config(yaml, model.chunks, 4));
  EXPECT_THROW(validate_server_config(yaml, model.chunks, 1),
               std::runtime_error);
  EXPECT_NE(yaml.find("memory_model"), std::string::npos);
  EXPECT_NE(yaml.find("corrections"), std::string::npos);
  EXPECT_EQ(yaml.find("async_num_workers"), std::string::npos);
  EXPECT_EQ(yaml.find("async_worker_cpus"), std::string::npos);
  const auto value = config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(value.syndrome_mapping.size(), 2);
  EXPECT_EQ(value.syndrome_mapping[0].id, 0);
  EXPECT_EQ(value.syndrome_mapping[0].type, "bulk");
  EXPECT_EQ(value.syndrome_mapping[1].id, 1);
  EXPECT_EQ(value.syndrome_mapping[1].type, "final");
  ASSERT_EQ(value.decoders.size(), 4);
  for (std::size_t patch = 0; patch < value.decoders.size(); ++patch) {
    const auto &decoder = value.decoders[patch];
    EXPECT_EQ(decoder.id, patch);
    ASSERT_TRUE(decoder.task_graph);
    ASSERT_EQ(decoder.task_graph->nodes.size(), 1);
    EXPECT_EQ(
        decoder.task_graph->nodes.front().node_params.map().get<std::string>(
            "output"),
        "corrections");
  }
}

TEST(MemoryExperimentServerConfig, BillionRoundConfigurationRemainsCompact) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  preparation_request request{{"surface_code", 3, 26, 1, memory_basis::z}};
  const auto small = prepare_model(request);
  const auto options = reference_options();
  const auto small_yaml = emit_server_config(small, options, 1);
  request.experiment.rounds = 1000000000;
  const auto large = prepare_model(request);
  const auto large_yaml = emit_server_config(large, options, 1);
  EXPECT_EQ(large.chunks.phases.size(), small.chunks.phases.size());
  EXPECT_LT(large_yaml.size(), small_yaml.size() + 1024);
  const auto parsed = config::multi_decoder_config::from_yaml_str(large_yaml);
  ASSERT_EQ(parsed.decoders.size(), 1);
  ASSERT_TRUE(parsed.decoders.front().dem_chunks);
  EXPECT_EQ(parsed.decoders.front().dem_chunks->num_rounds, 1000000001);
  EXPECT_EQ(parsed.decoders.front().dem_chunks->phases, small.chunks.phases);
}

TEST(MemoryExperimentServerConfig, PublishesOnlyServerYamlWithoutOverwriting) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  const auto options = reference_options();
  temporary_server_directory temporary;
  const auto output = temporary.path / "custom server.yaml";
  ASSERT_NO_THROW(write_server_config(request, model, options, output));
  std::vector<std::string> files;
  for (const auto &entry : std::filesystem::directory_iterator(temporary.path))
    files.push_back(entry.path().filename().string());
  EXPECT_EQ(files, (std::vector<std::string>{"custom server.yaml"}));
  std::ifstream input(output);
  const std::string yaml((std::istreambuf_iterator<char>(input)), {});
  EXPECT_NO_THROW(validate_server_config(yaml, model.chunks, 1));
  EXPECT_THROW(write_server_config(request, model, options, output),
               std::exception);
  std::ifstream unchanged(output);
  EXPECT_EQ(std::string((std::istreambuf_iterator<char>(unchanged)), {}), yaml);
  const auto missing = temporary.path / "missing parent" / "server.yaml";
  EXPECT_THROW(write_server_config(request, model, options, missing),
               std::exception);
  EXPECT_FALSE(std::filesystem::exists(missing.parent_path()));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(temporary.path),
                          std::filesystem::directory_iterator()),
            1);
}

TEST(MemoryExperimentServerConfig,
     ExistingEntriesAndUnrelatedFilesArePreserved) {
  if (!has_pattern_schema())
    GTEST_SKIP() << "Requires the real public parallel-window schema";
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  const auto options = reference_options();
  for (const std::string kind : {"file", "directory", "file symlink",
                                 "directory symlink", "dangling symlink"}) {
    SCOPED_TRACE(kind);
    temporary_server_directory temporary;
    const auto output = temporary.path / "server.yaml";
    const auto target = temporary.path / "target";
    const auto sentinel = temporary.path / "unrelated.txt";
    std::ofstream(sentinel) << "unrelated user content";
    if (kind == "file")
      std::ofstream(output) << "existing output";
    else if (kind == "directory")
      std::filesystem::create_directory(output);
    else {
      if (kind == "file symlink")
        std::ofstream(target) << "existing output";
      else if (kind == "directory symlink")
        std::filesystem::create_directory(target);
      std::filesystem::create_symlink(target, output);
    }
    const auto output_type = std::filesystem::symlink_status(output).type();
    const auto count =
        std::distance(std::filesystem::directory_iterator(temporary.path),
                      std::filesystem::directory_iterator());
    EXPECT_THROW(write_server_config(request, model, options, output),
                 std::exception);
    EXPECT_EQ(std::filesystem::symlink_status(output).type(), output_type);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(temporary.path),
                            std::filesystem::directory_iterator()),
              count);
    std::ifstream unchanged(sentinel);
    EXPECT_EQ(std::string((std::istreambuf_iterator<char>(unchanged)), {}),
              "unrelated user content");
    if (kind == "file" || kind == "file symlink") {
      std::ifstream existing(output);
      EXPECT_EQ(std::string((std::istreambuf_iterator<char>(existing)), {}),
                "existing output");
    }
    if (kind == "dangling symlink")
      EXPECT_FALSE(std::filesystem::exists(target));
  }
}

TEST(MemoryExperimentServerConfig, InvalidPreparedReadoutCreatesNoOutput) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  auto model = prepare_model(request);
  model.final_m2o = {{0}};
  temporary_server_directory temporary;
  const auto output = temporary.path / "not published.yaml";
  EXPECT_THROW(write_server_config(request, model, reference_options(), output),
               std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(output));
  EXPECT_TRUE(std::filesystem::is_empty(temporary.path));
}

} // namespace
} // namespace cudaq::qec::memory_experiment
