/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "bundle.h"
#include "raw_replay.h"
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string_view>

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
    throw std::invalid_argument(
        "--error-rate must be finite and in (0,1); noiseless phases require "
        "the pending public zero-fault-phase support");
  return value;
}

void usage() {
  std::cout
      << "Usage: memory-experiment-generator --distance=N --rounds=T "
         "--basis=X|Z\n"
         "       --output=NEW_DIRECTORY [--logical-qubits=N] "
         "[--batch-rounds=N]\n"
         "       [--error-rate=P]\n"
         "       [--replay-shots=N --seed=N]\n"
         "       memory-experiment-generator --validate-bundle=DIRECTORY\n\n"
         "Generates Surface Code memory-model YAML.\n"
         "--rounds counts extraction cycles, excluding terminal readout.\n"
         "Multiple logical qubits represent independent patches.\n"
         "With --replay-shots=N, also simulates N experiment shots per "
         "logical\n"
         "qubit and saves their measurement records for offline decoder "
         "testing.\n"
         "No measurements are simulated by default.\n"
         "Defaults: logical-qubits=1, batch-rounds=1, CNOT "
         "depolarization=0.001.\n"
         "The output directory must not exist; its parent must exist.\n"
         "A/B decoding-server YAML generation is not yet supported.\n";
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
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      const auto equals = argument.find('=');
      if (!argument.starts_with("--") || equals == std::string::npos ||
          equals + 1 == argument.size())
        throw std::invalid_argument("expected --name=value, got: " + argument);
      const auto name = argument.substr(2, equals - 2);
      if (name != "distance" && name != "rounds" && name != "basis" &&
          name != "output" && name != "logical-qubits" &&
          name != "batch-rounds" && name != "error-rate" &&
          name != "validate-bundle" && name != "code" &&
          name != "replay-shots" && name != "seed")
        throw std::invalid_argument("unknown option: --" + name);
      if (!values.emplace(name, argument.substr(equals + 1)).second)
        throw std::invalid_argument("duplicate option: --" + name);
    }
    if (values.contains("validate-bundle")) {
      if (values.size() != 1)
        throw std::invalid_argument("--validate-bundle cannot be combined with "
                                    "experiment-generation options");
      generator::validate_model_bundle(values.at("validate-bundle"));
      std::cout
          << "Model bundle validated; graph construction and replay remain "
             "pending.\n";
      return 0;
    }
    for (const auto *required : {"distance", "rounds", "basis", "output"})
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
    generator::bundle_options options;
    if (values.contains("batch-rounds"))
      options.batch_rounds =
          integer(values.at("batch-rounds"), "--batch-rounds");
    if (values.contains("replay-shots"))
      options.replay_shots =
          integer(values.at("replay-shots"), "--replay-shots");
    if (values.contains("seed"))
      options.seed = integer(values.at("seed"), "--seed");
    generator::validate_preparation_request(request);
    if (options.batch_rounds == 0)
      throw std::invalid_argument("--batch-rounds must be positive");
    generator::validate_raw_replay_options(spec.logical_qubits,
                                           options.replay_shots, options.seed);
    const std::filesystem::path output(values.at("output"));
    if (std::filesystem::exists(std::filesystem::symlink_status(output)))
      throw std::invalid_argument("output already exists: " + output.string());
    const auto parent = output.has_parent_path() ? output.parent_path()
                                                 : std::filesystem::path(".");
    if (!std::filesystem::is_directory(parent))
      throw std::invalid_argument("output parent must exist: " +
                                  parent.string());
    const auto model = generator::prepare_model(request);
    generator::write_model_bundle(request, model, options, output);
    std::cout << "Prepared model bundle: " << output
              << "\nA/B graph generation remains pending; no server YAML was "
                 "emitted.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "memory-experiment-generator: " << error.what() << '\n';
    return 1;
  }
}
