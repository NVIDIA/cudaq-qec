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
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace cudaq::qec::memory_experiment {
namespace {

namespace config = cudaq::qec::decoding::config;
constexpr auto pattern_name = "parallel_window";
constexpr auto model_name = "memory_model";
constexpr auto node_name = "parallel_memory";
constexpr auto corrections_output = "corrections";
constexpr std::uint64_t bulk_mapping_id = 0;
constexpr std::uint64_t final_mapping_id = 1;
constexpr auto unavailable_message =
    "server YAML generation requires the public parallel_window configuration "
    "and named-model/readout support; this QEC build does not provide it";

using argument_leaves = std::map<std::string, std::string>;

std::vector<config::syndrome_mapping_config> syndrome_mapping() {
  return {{bulk_mapping_id, "bulk", {{}}}, {final_mapping_id, "final", {{}}}};
}

template <typename Config, typename Model>
constexpr bool has_public_server_model = requires(Config &value, Model &model) {
  value.data_artifacts;
  value.decoders.front().dem_chunks_artifact;
  model.phases.front().spec.m2o_sparse;
};

const config::decoder_schema &require_public_server_config() {
  // These are public, installed registry APIs. Do not register a replacement
  // schema: old builds must reject this mode instead of emitting guessed YAML.
  if (!find_task_graph_node_kind(pattern_name))
    throw std::invalid_argument(unavailable_message);
  const auto *schema =
      config::find_decoder_schema(task_graph_node_schema_name(pattern_name));
  if (!schema ||
      !has_public_server_model<config::multi_decoder_config, dem_chunks_spec>)
    throw std::invalid_argument(unavailable_message);
  return *schema;
}

template <typename Number>
Number number(std::string_view text, const std::string &path) {
  Number value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc() ||
      parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("decoder argument '" + path +
                                "' has an invalid or out-of-range number");
  return value;
}

std::vector<double> double_vector(std::string_view text,
                                  const std::string &path) {
  std::vector<double> result;
  if (text.empty())
    return result;
  for (;;) {
    const auto separator = text.find(',');
    const auto value = number<double>(text.substr(0, separator), path);
    if (!std::isfinite(value))
      throw std::invalid_argument("decoder argument '" + path +
                                  "' must contain finite numbers");
    result.push_back(value);
    if (separator == std::string_view::npos)
      return result;
    text.remove_prefix(separator + 1);
  }
}

std::vector<std::vector<double>> double_matrix(std::string_view text,
                                               const std::string &path) {
  std::vector<std::vector<double>> result;
  if (text.empty())
    return result;
  for (;;) {
    const auto separator = text.find(';');
    result.push_back(double_vector(text.substr(0, separator), path));
    if (separator == std::string_view::npos)
      return result;
    text.remove_prefix(separator + 1);
  }
}

argument_leaves split_arguments(const std::vector<std::string> &arguments) {
  argument_leaves result;
  for (const auto &argument : arguments) {
    const auto equals = argument.find('=');
    if (equals == std::string::npos || equals == 0)
      throw std::invalid_argument("--decoder-arg expects PATH=VALUE");
    const auto path = argument.substr(0, equals);
    if (path.front() == '.' || path.back() == '.' ||
        path.find("..") != std::string::npos)
      throw std::invalid_argument("invalid decoder argument path: " + path);
    if (!result.emplace(path, argument.substr(equals + 1)).second)
      throw std::invalid_argument("duplicate decoder argument: " + path);
  }
  return result;
}

cudaqx::heterogeneous_map parse_arguments(const config::decoder_schema &schema,
                                          const argument_leaves &leaves,
                                          const std::string &prefix = {}) {
  cudaqx::heterogeneous_map result;
  argument_leaves remaining = leaves;
  // Resolve all direct leaves before nested sections, so a discriminator may
  // follow its section on the command line or in the schema declaration.
  for (const auto &parameter : schema.params) {
    const auto found = leaves.find(parameter.key);
    if (found == leaves.end())
      continue;
    const auto &text = found->second;
    const auto path = prefix + parameter.key;
    using config::param_kind;
    switch (parameter.kind) {
    case param_kind::boolean:
      if (text != "true" && text != "false")
        throw std::invalid_argument("decoder argument '" + path +
                                    "' must be true or false");
      result.insert(parameter.key, text == "true");
      break;
    case param_kind::int32:
      result.insert(parameter.key, number<int>(text, path));
      break;
    case param_kind::uint64:
      result.insert(parameter.key, number<std::size_t>(text, path));
      break;
    case param_kind::f64: {
      const auto value = number<double>(text, path);
      if (!std::isfinite(value))
        throw std::invalid_argument("decoder argument '" + path +
                                    "' must be finite");
      result.insert(parameter.key, value);
      break;
    }
    case param_kind::string:
      result.insert(parameter.key, text);
      break;
    case param_kind::f64_vec:
      result.insert(parameter.key, double_vector(text, path));
      break;
    case param_kind::f64_matrix:
      result.insert(parameter.key, double_matrix(text, path));
      break;
    case param_kind::subschema:
    case param_kind::discriminated:
      throw std::invalid_argument("decoder argument '" + path +
                                  "' is a section; name its leaf parameters");
    }
    remaining.erase(parameter.key);
  }
  for (const auto &parameter : schema.params) {
    if (parameter.kind != config::param_kind::subschema &&
        parameter.kind != config::param_kind::discriminated)
      continue;
    const auto child_prefix = parameter.key + '.';
    argument_leaves children;
    for (auto it = remaining.lower_bound(child_prefix);
         it != remaining.end() && it->first.starts_with(child_prefix);) {
      children.emplace(it->first.substr(child_prefix.size()), it->second);
      it = remaining.erase(it);
    }
    if (children.empty())
      continue;
    auto schema_name = parameter.subschema;
    if (parameter.kind == config::param_kind::discriminated) {
      if (!result.contains(parameter.discriminator))
        throw std::invalid_argument("decoder argument section '" + prefix +
                                    parameter.key + "' requires '" + prefix +
                                    parameter.discriminator + "'");
      schema_name = parameter.discriminator_prefix +
                    result.get<std::string>(parameter.discriminator);
    }
    const auto *nested = config::find_decoder_schema(schema_name);
    if (!nested)
      throw std::invalid_argument("no registered decoder parameter schema: " +
                                  schema_name);
    result.insert(parameter.key,
                  parse_arguments(*nested, children, prefix + child_prefix));
  }
  if (!remaining.empty())
    throw std::invalid_argument("unknown decoder argument: " + prefix +
                                remaining.begin()->first);
  config::materialize_default_args(schema, result);
  config::validate_custom_args(schema, result, "decoder arguments");
  return result;
}

cudaqx::heterogeneous_map
decoder_arguments(const server_config_options &options) {
  const auto leaves = split_arguments(options.decoder_args);
  const auto *schema = config::find_decoder_schema(options.decoder);
  if (!schema) {
    if (leaves.empty())
      return {};
    throw std::invalid_argument("decoder '" + options.decoder +
                                "' has no registered parameter schema");
  }
  return parse_arguments(*schema, leaves);
}

cudaqx::heterogeneous_map
pattern_arguments(const server_config_options &options) {
  if (options.commit_rounds > std::numeric_limits<std::size_t>::max() ||
      options.buffer_rounds > std::numeric_limits<std::size_t>::max() ||
      (options.gap_rounds &&
       *options.gap_rounds > std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("parallel-window geometry exceeds size_t");
  cudaqx::heterogeneous_map result;
  result.insert("commit_rounds",
                static_cast<std::size_t>(options.commit_rounds));
  result.insert("buffer_rounds",
                static_cast<std::size_t>(options.buffer_rounds));
  if (options.gap_rounds)
    result.insert("gap_rounds", static_cast<std::size_t>(*options.gap_rounds));
  result.insert("output", std::string(corrections_output));
  return result;
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(std::string("invalid server configuration: ") +
                             message);
}

void validate_patch_count(std::uint64_t patches) {
  if (patches == 0 ||
      patches > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()) ||
      patches > std::vector<config::decoder_config>{}.max_size())
    throw std::invalid_argument(
        "server configuration patch count is out of range");
}

template <typename Config>
Config make_configuration(const dem_chunks_spec &model,
                          const server_config_options &options,
                          std::uint64_t logical_qubits) {
  if constexpr (has_public_server_model<Config, dem_chunks_spec>) {
    Config result;
    result.syndrome_mapping = syndrome_mapping();
    auto &artifact = result.data_artifacts.emplace_back();
    artifact.name = model_name;
    artifact.dem_chunks = model;
    const auto inner = decoder_arguments(options);
    const auto parameters = pattern_arguments(options);
    result.decoders.reserve(static_cast<std::size_t>(logical_qubits));
    for (std::uint64_t patch = 0; patch < logical_qubits; ++patch) {
      auto &decoder = result.decoders.emplace_back();
      decoder.id = static_cast<std::int64_t>(patch);
      // The public configuration owns resolved models as well as their names;
      // the serializer and parser require this bounded copy per decoder.
      decoder.dem_chunks = model;
      decoder.dem_chunks_artifact = model_name;
      auto &graph = decoder.task_graph.emplace();
      auto &node = graph.nodes.emplace_back();
      node.id = node_name;
      node.type = pattern_name;
      node.inner_type = options.decoder;
      node.decoder_custom_args = inner;
      node.node_params = parameters;
      graph.edges = {{"measurements", "output", node_name, "input"},
                     {node_name, "output", "output", "input"}};
    }
    return result;
  } else {
    throw std::invalid_argument(unavailable_message);
  }
}

template <typename Config>
void validate_configuration(const Config &value,
                            const dem_chunks_spec &expected_model,
                            std::uint64_t logical_qubits) {
  if constexpr (has_public_server_model<Config, dem_chunks_spec>) {
    require(value.transport == config::transport_config{} &&
                value.syndrome_mapping == syndrome_mapping(),
            "unexpected deployment or syndrome-mapping overrides");
    require(value.data_artifacts.size() == 1, "expected one shared model");
    const auto &artifact = value.data_artifacts.front();
    require(artifact.name == model_name && artifact.dem_chunks &&
                *artifact.dem_chunks == expected_model,
            "shared model disagrees with the prepared model");
    require(value.decoders.size() == logical_qubits,
            "decoder count disagrees with independent patches");
    for (std::size_t patch = 0; patch < value.decoders.size(); ++patch) {
      const auto &decoder = value.decoders[patch];
      require(decoder.id == static_cast<std::int64_t>(patch),
              "decoder identity disagrees with patch index");
      require(decoder.type.empty() && decoder.H_sparse.empty() &&
                  decoder.D_sparse.empty() && decoder.O_sparse.empty() &&
                  decoder.block_size == 0 && decoder.syndrome_size == 0 &&
                  decoder.error_rate_vec.empty() &&
                  decoder.decoder_custom_args.empty() &&
                  decoder.dispatch == config::DecoderDispatch::host &&
                  !decoder.cuda_device_id && decoder.stim_dem_path.empty() &&
                  decoder.dem_chunks &&
                  decoder.dem_chunks_artifact == model_name &&
                  *decoder.dem_chunks == expected_model,
              "decoder must reference the shared prepared model");
      require(decoder.task_graph.has_value(), "missing task graph");
      const auto &graph = *decoder.task_graph;
      require(graph.type.empty() && graph.inner_type.empty() &&
                  graph.nodes.size() == 1 &&
                  graph.options == task_graph_options{},
              "expected one pattern with no worker-policy overrides");
      const auto &node = graph.nodes.front();
      require(node.id == node_name && node.type == pattern_name &&
                  cudaq::qec::decoder::is_registered(node.inner_type) &&
                  !node.dem_chunks && node.dem_chunks_artifact.empty() &&
                  node.H_sparse.empty() && node.D_sparse.empty() &&
                  node.O_sparse.empty() && node.stim_dem.empty() &&
                  node.block_size == 0 && node.syndrome_size == 0 &&
                  node.stim_dem_path.empty() && !node.ensemble &&
                  node.partition_label.empty() &&
                  node.partition_columns.empty() && node.windows.empty(),
              "unexpected parallel-window node or model override");
      require(node.node_params.map().template get<std::string>("output") ==
                  corrections_output,
              "parallel-window output must be corrections");
      for (const auto &parameter : node.node_params.map())
        require(parameter.first == "commit_rounds" ||
                    parameter.first == "buffer_rounds" ||
                    parameter.first == "gap_rounds" ||
                    parameter.first == "output",
                "unexpected parallel-window policy override");
      const std::vector<task_graph_edge> edges{
          {"measurements", "output", node_name, "input"},
          {node_name, "output", "output", "input"}};
      require(graph.edges == edges, "unexpected parallel-window wiring");
      const auto &first = value.decoders.front().task_graph->nodes.front();
      require(node.inner_type == first.inner_type &&
                  node.decoder_custom_args == first.decoder_custom_args &&
                  node.node_params == first.node_params,
              "independent patches must share the decoder recipe and geometry");
    }
  } else {
    throw std::invalid_argument(unavailable_message);
  }
}

} // namespace

void validate_server_config_options(const server_config_options &options) {
  const auto &schema = require_public_server_config();
  if (options.decoder.empty())
    throw std::invalid_argument("--decoder must not be empty");
  if (!cudaq::qec::decoder::is_registered(options.decoder))
    throw std::invalid_argument(
        "decoder is not registered in this QEC build: " + options.decoder);
  config::validate_custom_args(schema, pattern_arguments(options),
                               "parallel-window configuration");
  (void)decoder_arguments(options);
}

std::string emit_server_config(const prepared_model &model,
                               const server_config_options &options,
                               std::uint64_t logical_qubits) {
  validate_server_config_options(options);
  validate_patch_count(logical_qubits);
  model.chunks.validate();
  validate_final_readout(model.chunks, model.final_m2o, true);
  auto value = make_configuration<config::multi_decoder_config>(
      model.chunks, options, logical_qubits);
  const auto yaml = value.to_yaml_str();
  const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
  require(
      parsed == value,
      "public YAML roundtrip changed model, geometry or decoder parameters");
  validate_configuration(parsed, model.chunks, logical_qubits);
  return yaml;
}

void validate_server_config(std::string_view yaml,
                            const dem_chunks_spec &expected_model,
                            std::uint64_t logical_qubits) {
  (void)require_public_server_config();
  validate_patch_count(logical_qubits);
  const auto parsed = config::multi_decoder_config::from_yaml_str(yaml);
  validate_configuration(parsed, expected_model, logical_qubits);
}

void write_server_config(const preparation_request &request,
                         const prepared_model &model,
                         const server_config_options &options,
                         const std::filesystem::path &config_file) {
  validate_prepared_model(request, model);
  const auto yaml =
      emit_server_config(model, options, request.experiment.logical_qubits);
  // Keep the incomplete write invisible to a server reading the default path.
  // mkstemp reserves a sibling uniquely and never interprets its directory as
  // a template; only the trailing XXXXXX characters are replaced.
  auto temporary = config_file.string() + ".XXXXXX";
  const auto descriptor = ::mkstemp(temporary.data());
  if (descriptor < 0)
    throw std::system_error(errno, std::generic_category(),
                            "cannot stage server configuration: " +
                                config_file.string());
  const llvm::FileRemover cleanup(temporary);
  try {
    llvm::raw_fd_ostream stream(descriptor, /*shouldClose=*/true);
    stream << yaml;
    stream.close();
    if (const auto error = stream.error()) {
      stream.clear_error();
      throw std::system_error(error);
    }
    // Unlike rename(), a hard-link publication is atomic and refuses an
    // existing destination, including symlinks and concurrent writers.
    std::filesystem::create_hard_link(temporary, config_file);
  } catch (const std::exception &error) {
    throw std::runtime_error("cannot write server configuration: " +
                             config_file.string() + ": " + error.what());
  }
}

} // namespace cudaq::qec::memory_experiment
