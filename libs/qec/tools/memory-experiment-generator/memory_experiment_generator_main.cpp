/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "server_config.h"
#include "../common/decoding_server_config.h"
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::uint64_t integer(std::string_view text, std::string_view flag) {
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc() ||
      parsed.ptr != text.data() + text.size())
    throw std::invalid_argument(std::string(flag) +
                                " must be an unsigned whole number");
  return value;
}

double probability(std::string_view text) {
  double value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc() ||
      parsed.ptr != text.data() + text.size() || !std::isfinite(value) ||
      value <= 0 || value >= 1)
    throw std::invalid_argument("--error-rate must be finite and in (0,1)");
  return value;
}

void usage() {
  std::cout
      << "Usage: memory-experiment-generator --distance=N --rounds=T "
         "--basis=X|Z\n"
         "       --decoder=NAME [--logical-qubits=N] [--error-rate=P]\n"
         "       [--commit-rounds=N] [--buffer-rounds=N]\n"
         "       [--gap-rounds=N] [--decoder-arg=PATH=VALUE ...]\n"
         "\nGenerates server.yaml for a Surface Code memory experiment.\n"
         "--rounds counts extraction cycles, excluding terminal readout.\n"
         "Multiple logical qubits represent independent patches.\n"
         "Defaults: code=surface_code, logical-qubits=1, CNOT "
         "depolarization=0.001.\n"
         "A/B commit and buffer widths default to distance for Surface Code;\n"
         "the gap uses the framework rule unless explicitly overridden.\n"
         "Writes server.yaml in the working directory, shared with the "
         "server.\n"
         "CUDAQ_QEC_DECODER_CONFIG selects a deployment configuration path.\n"
         "The parent must exist; an existing configuration is never replaced.\n"
         "The linked QEC build must provide the public parallel_window\n"
         "configuration API. Configuration validation does not establish\n"
         "that the runtime can execute the pattern.\n"
         "Server output is corrections, not corrected raw logical values.\n"
         "Decoder arguments use registered schema types, dotted nested paths,\n"
         "comma-separated vectors and semicolon-separated matrix rows.\n"
         "No JSON or YAML parameter input is accepted.\n";
}

} // namespace

int main(int argc, char **argv) {
  namespace generator = cudaq::qec::memory_experiment;
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      usage();
      return 0;
    }
    std::map<std::string, std::string> values;
    std::vector<std::string> decoder_args;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      const auto equals = argument.find('=');
      if (!argument.starts_with("--") || equals == std::string::npos ||
          equals + 1 == argument.size())
        throw std::invalid_argument("expected --name=value, got: " + argument);
      const auto name = argument.substr(2, equals - 2);
      if (name == "decoder-arg") {
        decoder_args.push_back(argument.substr(equals + 1));
        continue;
      }
      if (name != "distance" && name != "rounds" && name != "basis" &&
          name != "logical-qubits" && name != "error-rate" && name != "code" &&
          name != "decoder" && name != "commit-rounds" &&
          name != "buffer-rounds" && name != "gap-rounds")
        throw std::invalid_argument("unknown option: --" + name);
      if (!values.emplace(name, argument.substr(equals + 1)).second)
        throw std::invalid_argument("duplicate option: --" + name);
    }
    for (const auto *required : {"distance", "rounds", "basis", "decoder"})
      if (!values.contains(required))
        throw std::invalid_argument(std::string("missing --") + required);
    generator::preparation_request request;
    auto &spec = request.experiment;
    spec.distance = integer(values.at("distance"), "--distance");
    spec.rounds = integer(values.at("rounds"), "--rounds");
    if (values.contains("code"))
      spec.code_family = values.at("code");
    if (values.contains("logical-qubits"))
      spec.logical_qubits =
          integer(values.at("logical-qubits"), "--logical-qubits");
    const auto &basis = values.at("basis");
    if (basis != "X" && basis != "Z")
      throw std::invalid_argument("--basis must be X or Z");
    spec.basis =
        basis == "X" ? generator::memory_basis::x : generator::memory_basis::z;
    if (values.contains("error-rate"))
      request.cnot_error_rate = probability(values.at("error-rate"));
    generator::server_config_options options;
    options.decoder = values.at("decoder");
    // Window geometry belongs to the A/B schedule, not the selected decoder.
    // This Surface Code profile supplies the framework's required widths.
    options.commit_rounds =
        values.contains("commit-rounds")
            ? integer(values.at("commit-rounds"), "--commit-rounds")
            : spec.distance;
    options.buffer_rounds =
        values.contains("buffer-rounds")
            ? integer(values.at("buffer-rounds"), "--buffer-rounds")
            : spec.distance;
    if (values.contains("gap-rounds"))
      options.gap_rounds = integer(values.at("gap-rounds"), "--gap-rounds");
    options.decoder_args = std::move(decoder_args);
    generator::validate_preparation_request(request);
    generator::validate_server_config_options(options);
    const auto output = cudaq::qec::tools::default_server_config_path();
    if (std::filesystem::exists(std::filesystem::symlink_status(output)))
      throw std::invalid_argument("configuration already exists: " +
                                  output.string());
    const auto parent = output.parent_path();
    if (!std::filesystem::is_directory(parent))
      throw std::invalid_argument("configuration parent must exist: " +
                                  parent.string());
    const auto model = generator::prepare_model(request);
    generator::write_server_config(request, model, options, output);
    std::cout << "Generated server configuration: " << output << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "memory-experiment-generator: " << error.what() << '\n';
    return 1;
  }
}
