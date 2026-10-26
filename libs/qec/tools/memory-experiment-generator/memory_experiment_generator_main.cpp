/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server_config.h"
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
         "       --decoder=NAME --output=PATH [--logical-qubits=N] "
         "[--error-rate=P]\n"
         "       [--code=surface_code]\n"
         "       [--commit-rounds=N] [--buffer-rounds=N]\n"
         "       [--gap-rounds=N] [--pattern-arg=PATH=VALUE ...]\n"
         "       [--decoder-arg=PATH=VALUE ...]\n"
         "       [--workers-per-graph=N]\n"
         "\nGenerates standalone YAML for a Surface Code memory experiment.\n"
         "--distance must be odd and at least 3.\n"
         "--rounds must be at least 4 extraction cycles, excluding readout.\n"
         "Logical-qubit, geometry and worker counts must be positive.\n"
         "--error-rate must be finite and strictly between 0 and 1.\n"
         "Multiple logical qubits represent independent patches.\n"
         "Defaults: code=surface_code, logical-qubits=1, CNOT "
         "depolarization=0.001.\n"
         "A/B commit and buffer widths default to distance for Surface Code;\n"
         "the gap uses the framework rule unless explicitly overridden.\n"
         "Workers are configured per graph, not shared across patches.\n"
         "Each patch has its own graph; coordinator threads are additional.\n"
         "Omitting --workers-per-graph preserves framework execution "
         "defaults.\n"
         "--output=PATH is required; there is no default output file.\n"
         "The parent must exist; an existing configuration is never replaced.\n"
         "Pass the generated path explicitly: decoding_server --config=PATH.\n"
         "Server execution requires parallel-window runtime support.\n"
         "Pattern output defaults to corrections (logical correction bits).\n"
         "PyMatching uses merge_strategy=smallest_weight unless overridden.\n"
         "Pattern and decoder arguments use registered types and dotted "
         "paths.\n"
         "Vectors use commas; matrix rows use semicolons.\n";
}

} // namespace

int main(int argc, char **argv) {
  namespace generator = cudaq::qec::memory_experiment;
  try {
    for (int i = 1; i < argc; ++i)
      if (const std::string_view argument(argv[i]);
          argument == "--help" || argument == "-h") {
        usage();
        return 0;
      }
    std::map<std::string, std::string> values;
    std::vector<std::string> decoder_args;
    std::vector<std::string> pattern_args;
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
      if (name == "pattern-arg") {
        pattern_args.push_back(argument.substr(equals + 1));
        continue;
      }
      if (name != "distance" && name != "rounds" && name != "basis" &&
          name != "logical-qubits" && name != "error-rate" && name != "code" &&
          name != "decoder" && name != "commit-rounds" &&
          name != "buffer-rounds" && name != "gap-rounds" &&
          name != "workers-per-graph" && name != "output")
        throw std::invalid_argument("unknown option: --" + name);
      if (!values.emplace(name, argument.substr(equals + 1)).second)
        throw std::invalid_argument("duplicate option: --" + name);
    }
    for (const auto *required :
         {"distance", "rounds", "basis", "decoder", "output"})
      if (!values.contains(required))
        throw std::invalid_argument(std::string("missing --") + required);
    // Geometry aliases must not compete with explicit dedicated flags. Other
    // pattern leaves are parsed and validated by the registered schema.
    for (const auto &argument : pattern_args) {
      const auto key = argument.substr(0, argument.find('='));
      const auto *flag = key == "commit_rounds"   ? "commit-rounds"
                         : key == "buffer_rounds" ? "buffer-rounds"
                         : key == "gap_rounds"    ? "gap-rounds"
                                                  : nullptr;
      if (flag && values.contains(flag))
        throw std::invalid_argument(std::string("cannot combine --") + flag +
                                    " with --pattern-arg=" + key);
    }
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
    if (values.contains("workers-per-graph"))
      options.workers_per_graph =
          integer(values.at("workers-per-graph"), "--workers-per-graph");
    options.decoder_args = std::move(decoder_args);
    options.pattern_args = std::move(pattern_args);
    generator::validate_preparation_request(request);
    const auto total_workers =
        generator::configured_worker_total(options, spec.logical_qubits);
    generator::validate_server_config_options(options);
    const auto output = std::filesystem::absolute(values.at("output"));
    if (std::filesystem::exists(std::filesystem::symlink_status(output)))
      throw std::invalid_argument("configuration already exists: " +
                                  output.string());
    const auto parent = output.parent_path();
    if (!std::filesystem::is_directory(parent))
      throw std::invalid_argument("configuration parent must exist: " +
                                  parent.string());
    const auto model = generator::prepare_model(request);
    generator::write_server_config(request, model, options, output);
    std::cout << "Generated server configuration: " << output.string() << '\n';
    if (total_workers)
      std::cout << "Workers per graph: " << *options.workers_per_graph
                << "; graphs: " << spec.logical_qubits
                << "; total configured workers: " << *total_workers
                << " (excluding coordinators).\n";
    else
      std::cout
          << "Workers per graph: framework-selected (total not specified).\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "memory-experiment-generator: " << error.what() << '\n';
    return 1;
  }
}
