/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include <limits>
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

server_config_options reference_options() {
  return {.decoder = "single_error_lut",
          .commit_rounds = 3,
          .buffer_rounds = 3,
          .gap_rounds = 9};
}

void expect_worker_policy(const task_graph_options &options,
                          std::uint64_t workers) {
  auto expected = task_graph_options{};
  expected.async_streaming = true;
  expected.async_num_workers = static_cast<std::size_t>(workers);
  EXPECT_EQ(options, expected);
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

// Extend only the actual pattern's parameter schema, never its kind or builder.
// Restoring the complete original also preserves its cross-field validation.
struct scoped_pattern_schema_extension {
  config::decoder_schema original;

  scoped_pattern_schema_extension()
      : original(*config::find_decoder_schema(
            task_graph_node_schema_name("parallel_window"))) {
    auto extended = original;
    extended.params.push_back({"profile_hint", config::param_kind::f64});
    config::register_decoder_schema(std::move(extended));
  }

  ~scoped_pattern_schema_extension() {
    config::register_decoder_schema(std::move(original));
  }
};

TEST(MemoryExperimentServerConfig, ConfiguredWorkerTotalIsExplicitAndChecked) {
  auto options = reference_options();
  EXPECT_EQ(configured_worker_total(options, 3), std::nullopt);
  options.workers_per_graph = 4;
  EXPECT_EQ(configured_worker_total(options, 3), 12);
  EXPECT_THROW(configured_worker_total(options, 0), std::invalid_argument);
  options.workers_per_graph = 0;
  EXPECT_THROW(configured_worker_total(options, 3), std::invalid_argument);
  options.workers_per_graph = std::numeric_limits<std::uint64_t>::max();
  if (std::numeric_limits<std::size_t>::max() >=
      std::numeric_limits<std::uint64_t>::max())
    EXPECT_EQ(configured_worker_total(options, 1),
              std::numeric_limits<std::uint64_t>::max());
  EXPECT_THROW(configured_worker_total(options, 2), std::overflow_error);
}

TEST(MemoryExperimentServerConfig, PublicSchemaChecksGeometryAndDecoderArgs) {
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
     PatternParametersRoundtripWithTypedOverridesForEveryPatch) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  auto options = reference_options();
  options.workers_per_graph = 2;
  options.pattern_args = {"commit_rounds=5",
                          "buffer_rounds=2",
                          "gap_rounds=11",
                          "output=corrected_logical",
                          "boundary_prior=1.2345678901234567e-6",
                          "input_boundary=columns",
                          "merge_identical=false",
                          "conflicting_parallel=most_likely",
                          "verify_residual=report"};
  const auto yaml = emit_server_config(model, options, 2);
  const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(parsed.decoders.size(), 2);
  for (const auto &decoder : parsed.decoders) {
    ASSERT_TRUE(decoder.task_graph);
    expect_worker_policy(decoder.task_graph->options, 2);
    ASSERT_EQ(decoder.task_graph->nodes.size(), 1);
    const auto &parameters =
        decoder.task_graph->nodes.front().node_params.map();
    EXPECT_EQ(parameters.get<std::size_t>("commit_rounds"), 5);
    EXPECT_EQ(parameters.get<std::size_t>("buffer_rounds"), 2);
    EXPECT_EQ(parameters.get<std::size_t>("gap_rounds"), 11);
    EXPECT_EQ(parameters.get<std::string>("output"), "corrected_logical");
    EXPECT_EQ(parameters.get<double>("boundary_prior"), 1.2345678901234567e-6);
    EXPECT_EQ(parameters.get<std::string>("input_boundary"), "columns");
    EXPECT_FALSE(parameters.get<bool>("merge_identical"));
    EXPECT_EQ(parameters.get<std::string>("conflicting_parallel"),
              "most_likely");
    EXPECT_EQ(parameters.get<std::string>("verify_residual"), "report");
  }
}

TEST(MemoryExperimentServerConfig, PatternSchemaRejectsInvalidArguments) {
  for (const auto &arguments : std::vector<std::vector<std::string>>{
           {"gap_rounds=2", "gap_rounds=3"},
           {"not_registered=1"},
           {"lut_error_depth=2"},
           {".output=corrections"},
           {"gap_rounds..child=2"},
           {"gap_rounds"},
           {"gap_rounds=-1"},
           {"gap_rounds=18446744073709551616"},
           {"gap_rounds=2junk"},
           {"commit_rounds=0"},
           {"buffer_rounds=0"},
           {"gap_rounds=0"},
           {"boundary_prior=nan"},
           {"boundary_prior=inf"},
           {"boundary_prior=0"},
           {"boundary_prior=1"},
           {"boundary_prior=-0.1"},
           {"input_boundary=unknown"},
           {"output=unknown"},
           {"merge_identical=1"},
           {"merge_identical=TRUE"},
           {"conflicting_parallel=unknown"},
           {"verify_residual=unknown"},
           {"verify_residual=false"},
           {"input_boundary=truncate", "boundary_prior=0.1"}}) {
    SCOPED_TRACE(::testing::PrintToString(arguments));
    auto options = reference_options();
    options.pattern_args = arguments;
    EXPECT_THROW(validate_server_config_options(options), std::exception);
  }
  auto options = reference_options();
  options.pattern_args = {"gap_rounds=2junk"};
  try {
    validate_server_config_options(options);
    FAIL() << "Invalid pattern number was accepted";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find("pattern argument"),
              std::string::npos);
    EXPECT_EQ(std::string(error.what()).find("decoder argument"),
              std::string::npos);
  }
}

TEST(MemoryExperimentServerConfig, PatternSchemaAcceptsItsOtherValidChoices) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  for (const std::string residual : {"none", "fail"}) {
    auto options = reference_options();
    options.pattern_args = {"input_boundary=truncate", "merge_identical=true",
                            "conflicting_parallel=reject",
                            "verify_residual=" + residual};
    const auto yaml = emit_server_config(model, options, 1);
    const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
    const auto &parameters =
        parsed.decoders.front().task_graph->nodes.front().node_params.map();
    EXPECT_EQ(parameters.get<std::string>("input_boundary"), "truncate");
    EXPECT_TRUE(parameters.get<bool>("merge_identical"));
    EXPECT_EQ(parameters.get<std::string>("conflicting_parallel"), "reject");
    EXPECT_EQ(parameters.get<std::string>("verify_residual"), residual);
    EXPECT_EQ(parameters.get<std::string>("output"), "corrections");
  }
}

TEST(MemoryExperimentServerConfig, PatternSchemaExtensionsNeedNoToolWhitelist) {
  ASSERT_NE(config::find_decoder_schema(
                task_graph_node_schema_name("parallel_window")),
            nullptr);
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  auto options = reference_options();
  options.pattern_args = {"profile_hint=0.12345678901234567"};
  {
    const scoped_pattern_schema_extension schema;
    const auto yaml = emit_server_config(model, options, 2);
    const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
    ASSERT_EQ(parsed.decoders.size(), 2);
    for (const auto &decoder : parsed.decoders)
      EXPECT_EQ(decoder.task_graph->nodes.front().node_params.map().get<double>(
                    "profile_hint"),
                0.12345678901234567);
  }
  EXPECT_THROW(validate_server_config_options(options), std::invalid_argument);
}

TEST(MemoryExperimentServerConfig,
     FullYamlRoundtripPreservesTypedPrecisionAndLiteralStrings) {
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
     PyMatchingProfileDefaultAndExplicitPoliciesRoundtripForEveryPatch) {
  if (!cudaq::qec::decoder::is_registered("pymatching"))
    GTEST_SKIP() << "Requires the installed PyMatching plugin";
  ASSERT_NE(config::find_decoder_schema("pymatching"), nullptr);
  auto options = reference_options();
  options.decoder = "pymatching";
  ASSERT_NO_THROW(validate_server_config_options(options));
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  for (const std::string policy :
       {"", "smallest_weight", "independent", "disallow"}) {
    SCOPED_TRACE(policy.empty() ? "profile default" : policy);
    options.decoder_args.clear();
    if (!policy.empty())
      options.decoder_args.push_back("merge_strategy=" + policy);
    const auto yaml = emit_server_config(model, options, 2);
    const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
    ASSERT_EQ(parsed.decoders.size(), 2);
    for (const auto &decoder : parsed.decoders) {
      ASSERT_TRUE(decoder.task_graph);
      ASSERT_EQ(decoder.task_graph->nodes.size(), 1);
      const auto &node = decoder.task_graph->nodes.front();
      EXPECT_EQ(node.inner_type, "pymatching");
      const auto &arguments = node.decoder_custom_args.map();
      EXPECT_EQ(arguments.get<std::string>("merge_strategy"),
                policy.empty() ? "smallest_weight" : policy);
      EXPECT_NO_THROW(config::validate_custom_args("pymatching", arguments));
    }
  }
}

TEST(MemoryExperimentServerConfig,
     RealPublicConfigurationRoundtripsSharedModelWithoutExecution) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  const auto options = reference_options();
  const auto yaml = emit_server_config(model, options, 4);
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
    EXPECT_EQ(decoder.task_graph->options, task_graph_options{});
    ASSERT_EQ(decoder.task_graph->nodes.size(), 1);
    const auto &node = decoder.task_graph->nodes.front();
    EXPECT_EQ(node.type, "parallel_window");
    const std::vector<task_graph_edge> edges{
        {"measurements", "output", node.id, "input"},
        {node.id, "output", "output", "input"}};
    EXPECT_EQ(decoder.task_graph->edges, edges);
    EXPECT_EQ(node.node_params.map().get<std::string>("output"), "corrections");
  }
}

TEST(MemoryExperimentServerConfig,
     WorkersPerGraphRoundtripWithoutChangingDecoderOrAffinity) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  auto options = reference_options();
  const auto baseline = config::multi_decoder_config::from_yaml_str(
      emit_server_config(model, options, 3));
  for (const std::uint64_t workers : {1, 4}) {
    SCOPED_TRACE(workers);
    options.workers_per_graph = workers;
    const auto yaml = emit_server_config(model, options, 3);
    const auto value = config::multi_decoder_config::from_yaml_str(yaml);
    ASSERT_EQ(value.decoders.size(), 3);
    for (const auto &decoder : value.decoders) {
      ASSERT_TRUE(decoder.task_graph);
      expect_worker_policy(decoder.task_graph->options, workers);
      EXPECT_EQ(decoder.task_graph->nodes,
                baseline.decoders.front().task_graph->nodes);
      EXPECT_EQ(decoder.task_graph->edges,
                baseline.decoders.front().task_graph->edges);
    }
  }
}

TEST(MemoryExperimentServerConfig, BillionRoundConfigurationRemainsCompact) {
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
  const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
  ASSERT_EQ(parsed.decoders.size(), 1);
  ASSERT_TRUE(parsed.decoders.front().dem_chunks);
  EXPECT_EQ(*parsed.decoders.front().dem_chunks, model.chunks);
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
