/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "bundle.h"
#include "manifest.h"
#include "model_provider.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/noise_model.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudaq::qec::memory_experiment {
namespace {

struct temporary_bundle_directory {
  std::filesystem::path path;

  temporary_bundle_directory() {
    llvm::SmallString<128> temporary;
    if (const auto error = llvm::sys::fs::createUniqueDirectory(
            "cudaq-memory-bundle", temporary))
      throw std::runtime_error(error.message());
    path = temporary.str().str();
  }

  ~temporary_bundle_directory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

std::string read_test_text(const std::filesystem::path &path) {
  std::ifstream input(path);
  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

void write_test_text(const std::filesystem::path &path,
                     const std::string &text) {
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.open(path);
  output << text;
}

manifest::document read_test_manifest(const std::filesystem::path &path) {
  return manifest::from_yaml(read_test_text(path));
}

void write_test_manifest(const std::filesystem::path &path,
                         const manifest::document &value) {
  write_test_text(path, manifest::to_yaml(value));
}

enum class yaml_field_mutation { unknown, missing, duplicate };

std::string mutate_yaml_field(const std::string &yaml,
                              const std::string &section,
                              const std::string &key,
                              yaml_field_mutation mutation) {
  const auto section_begin = section.empty() ? 0 : yaml.find(section + ':');
  if (section_begin == std::string::npos)
    throw std::runtime_error("missing test YAML section");
  const auto field = yaml.find(key + ':', section_begin);
  if (field == std::string::npos)
    throw std::runtime_error("missing test YAML field");
  const auto previous = yaml.rfind('\n', field);
  const auto begin = previous == std::string::npos ? 0 : previous + 1;
  const auto newline = yaml.find('\n', field);
  if (newline == std::string::npos)
    throw std::runtime_error("unterminated test YAML field");
  const auto past_last = newline + 1;
  auto changed = yaml;
  switch (mutation) {
  case yaml_field_mutation::unknown:
    changed.insert(begin, yaml.substr(begin, field - begin) +
                              "unsupported_field: true\n");
    break;
  case yaml_field_mutation::missing:
    changed.erase(begin, past_last - begin);
    break;
  case yaml_field_mutation::duplicate:
    changed.insert(begin, yaml.substr(begin, past_last - begin));
    break;
  }
  return changed;
}

std::string test_digest(const std::string &text) {
  const auto hash = llvm::SHA256::hash(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(text.data()), text.size()));
  return llvm::toHex(llvm::ArrayRef<std::uint8_t>(hash), true);
}

using fault_signature =
    std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;

std::map<fault_signature, double>
canonical_faults(const detector_error_model &dem) {
  std::map<fault_signature, double> result;
  for (std::size_t fault = 0; fault < dem.num_error_mechanisms(); ++fault) {
    fault_signature signature;
    for (std::size_t detector = 0; detector < dem.num_detectors(); ++detector)
      if (dem.detector_error_matrix.at({detector, fault}))
        signature.first.push_back(detector);
    for (std::size_t observable = 0; observable < dem.num_observables();
         ++observable)
      if (dem.observables_flips_matrix.at({observable, fault}))
        signature.second.push_back(observable);
    auto &probability = result[signature];
    const auto next = dem.error_rates[fault];
    probability = probability * (1 - next) + next * (1 - probability);
  }
  return result;
}

void compare_with_full_circuit(std::uint64_t distance, memory_basis basis,
                               std::uint64_t extraction_rounds = 26) {
  const preparation_request request{
      {"surface_code", distance, extraction_rounds, 1, basis}};
  const auto prepared = prepare_model(request);
  ASSERT_EQ(prepared.chunks.num_rounds, extraction_rounds + 1);
  ASSERT_EQ(prepared.characterization_rounds, representative_extraction_rounds);
  ASSERT_EQ(prepared.validation_rounds, validation_extraction_rounds);

  auto surface = get_code(
      "surface_code", cudaqx::heterogeneous_map{
                          {"distance", static_cast<std::size_t>(distance)}});
  const auto prep =
      basis == memory_basis::z ? operation::prep0 : operation::prepp;
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x",
                              two_qubit_depolarization(request.cnot_error_rate),
                              /*num_controls=*/1);
  // This independently characterizes the complete requested experiment. It
  // never calls the splitter, and must agree with the bounded representative
  // model expanded to the same horizon, including both endpoint mappings.
  const auto full =
      decoder_context_from_memory_circuit(*surface, prep, extraction_rounds,
                                          noise, request.decompose_errors)
          .full_component();
  const auto closed =
      dem_close(dem_stitch_all(dem_chunks_from_spec(prepared.chunks)));
  ASSERT_EQ(closed.num_detectors(), full.dem.num_detectors());
  ASSERT_EQ(closed.num_observables(), full.dem.num_observables());
  const auto expected = canonical_faults(full.dem);
  const auto actual = canonical_faults(closed);
  ASSERT_EQ(actual.size(), expected.size());
  for (const auto &[signature, probability] : expected) {
    const auto found = actual.find(signature);
    ASSERT_NE(found, actual.end());
    EXPECT_NEAR(found->second, probability, 1e-12);
  }
  std::vector<std::vector<std::uint32_t>> expected_d;
  for (const auto &row : full.m2d.rows)
    expected_d.emplace_back(row.begin(), row.end());
  EXPECT_EQ(dem_chunks_to_d_sparse(prepared.chunks), expected_d);
  EXPECT_EQ(dem_chunks_measurement_count(prepared.chunks),
            full.m2d.num_measurements);
  EXPECT_EQ(prepared.regular_measurements,
            surface->get_num_ancilla_x_qubits() +
                surface->get_num_ancilla_z_qubits());
  EXPECT_EQ(prepared.final_measurements, surface->get_num_data_qubits());
  EXPECT_EQ(prepared.physical_qubits,
            prepared.regular_measurements + prepared.final_measurements);

  const auto logical = basis == memory_basis::z ? surface->get_observables_z()
                                                : surface->get_observables_x();
  std::vector<std::vector<std::uint64_t>> expected_m2o(logical.shape()[0]);
  for (std::size_t observable = 0; observable < logical.shape()[0];
       ++observable)
    for (std::size_t data = 0; data < logical.shape()[1]; ++data)
      if (logical.at({observable, data}))
        expected_m2o[observable].push_back(data);
  EXPECT_EQ(prepared.final_m2o, expected_m2o);
  ASSERT_EQ(full.m2o.rows.size(), expected_m2o.size());
  const auto final_begin = extraction_rounds * prepared.regular_measurements;
  for (std::size_t observable = 0; observable < expected_m2o.size();
       ++observable) {
    auto expected_absolute = expected_m2o[observable];
    for (auto &measurement : expected_absolute)
      measurement += final_begin;
    EXPECT_EQ(full.m2o.rows[observable], expected_absolute);
  }
}

TEST(MemoryExperimentProvider, BoundedModelMatchesCompleteZMemoryCircuit) {
  compare_with_full_circuit(3, memory_basis::z);
}

TEST(MemoryExperimentProvider, BoundedModelMatchesCompleteXMemoryCircuit) {
  compare_with_full_circuit(3, memory_basis::x);
}

TEST(MemoryExperimentProvider, DistanceFiveMatchesCompleteMemoryCircuit) {
  compare_with_full_circuit(5, memory_basis::z);
  compare_with_full_circuit(5, memory_basis::x);
}

TEST(MemoryExperimentProvider,
     MinimumAndAwkwardDurationsMatchCompleteCircuits) {
  for (const auto basis : {memory_basis::x, memory_basis::z})
    for (const auto rounds : {4, 5, 25, 27}) {
      SCOPED_TRACE(rounds);
      SCOPED_TRACE(basis == memory_basis::x ? "X" : "Z");
      compare_with_full_circuit(3, basis, rounds);
    }
}

TEST(MemoryExperimentProvider, RequestedDurationAndPatchCountDoNotGrowModel) {
  preparation_request request{{"surface_code", 3, 26, 1, memory_basis::z}};
  const auto small = prepare_model(request);
  for (const auto rounds : {std::uint64_t{10000}, std::uint64_t{1000000000}}) {
    request.experiment.rounds = rounds;
    request.experiment.logical_qubits = 4;
    const auto large = prepare_model(request);
    EXPECT_EQ(large.chunks.num_rounds, rounds + 1);
    EXPECT_EQ(large.chunks.phases, small.chunks.phases);
    EXPECT_EQ(large.chunks.connections, small.chunks.connections);
    EXPECT_EQ(large.chunks.measurements_per_round,
              small.chunks.measurements_per_round);
    EXPECT_EQ(large.final_m2o, small.final_m2o);
    EXPECT_EQ(large.characterization_rounds, small.characterization_rounds);
    EXPECT_EQ(large.validation_rounds, small.validation_rounds);
    EXPECT_EQ(large.physical_qubits, small.physical_qubits);
  }
}

TEST(MemoryExperimentProvider, InvalidNoiseAndGeometryFailBeforeCircuitWork) {
  preparation_request request{{"surface_code", 3, 26, 1, memory_basis::z}};
  for (const auto probability :
       {0.0, -0.01, 1.0, 1.01, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    request.cnot_error_rate = probability;
    EXPECT_THROW(prepare_model(request), std::invalid_argument);
  }
  request.cnot_error_rate = 0.001;
  request.experiment.distance = std::numeric_limits<std::uint32_t>::max();
  EXPECT_THROW(prepare_model(request), std::overflow_error);
  request.experiment.distance = 3;
  request.experiment.rounds = std::numeric_limits<std::uint64_t>::max() - 1;
  EXPECT_THROW(prepare_model(request), std::overflow_error);
}

TEST(MemoryExperimentProvider, TooShortDurationIsRejectedWithoutRounding) {
  const preparation_request request{{"surface_code", 3, 1, 1, memory_basis::z}};
  EXPECT_THROW(prepare_model(request), std::invalid_argument);
}

TEST(MemoryExperimentProvider, PreparedGeometryIsCheckedAgainstCode) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const preparation_request request{{"surface_code", 3, 26, 1, basis}};
    const auto model = prepare_model(request);
    for (const auto field : {0, 1, 2, 3}) {
      auto invalid = model;
      switch (field) {
      case 0:
        ++invalid.regular_measurements;
        break;
      case 1:
        ++invalid.final_measurements;
        break;
      case 2:
        ++invalid.physical_qubits;
        break;
      case 3:
        invalid.final_m2o.emplace_back();
        break;
      }
      EXPECT_THROW(validate_prepared_model(request, invalid),
                   std::invalid_argument);
    }
  }
}

TEST(MemoryExperimentBundle, VersionOneAcceptsSaneCharacterizationProvenance) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto original = prepare_model(request);
  temporary_bundle_directory temporary;
  // Provenance records how a producer established repetition. A schema-v1
  // reader must not require today's 12/13 choice, adjacency or even/odd pair.
  for (const auto &[characterization, validation] :
       std::vector<std::pair<std::uint64_t, std::uint64_t>>{{16, 17},
                                                            {14, 19}}) {
    auto model = original;
    model.characterization_rounds = characterization;
    model.validation_rounds = validation;
    EXPECT_NO_THROW(validate_prepared_model(request, model));
    const auto output = temporary.path / std::to_string(characterization);
    EXPECT_NO_THROW(write_model_bundle(request, model, {}, output));
    const auto metadata = read_test_manifest(output / "manifest.yaml");
    EXPECT_EQ(metadata.model.characterization_rounds, characterization);
    EXPECT_EQ(metadata.model.validation_rounds, validation);
    EXPECT_EQ(metadata.schema_version, 1);
    EXPECT_NO_THROW(validate_model_bundle(output));
  }
}

TEST(MemoryExperimentBundle, RejectsDegenerateCharacterizationProvenance) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto original = prepare_model(request);
  temporary_bundle_directory temporary;
  const auto output = temporary.path / "bundle";
  write_model_bundle(request, original, {}, output);
  const auto path = output / "manifest.yaml";
  const auto original_metadata = read_test_manifest(path);
  const auto too_short = original.chunks.minimum_rounds() - 1;
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  for (const auto &[characterization, validation] :
       std::vector<std::pair<std::uint64_t, std::uint64_t>>{
           {0, 13},
           {12, 0},
           {12, 12},
           {too_short, 13},
           {12, too_short},
           {maximum, 13},
           {12, maximum},
           {maximum - 1, 13}}) {
    auto model = original;
    model.characterization_rounds = characterization;
    model.validation_rounds = validation;
    EXPECT_THROW(validate_prepared_model(request, model), std::exception);
    auto metadata = original_metadata;
    metadata.model.characterization_rounds = characterization;
    metadata.model.validation_rounds = validation;
    write_test_manifest(path, metadata);
    EXPECT_THROW(validate_model_bundle(output), std::exception);
  }
  for (const auto &key : {"characterization_rounds", "validation_rounds"}) {
    auto yaml = manifest::to_yaml(original_metadata);
    const auto begin = yaml.find(std::string(key) + ':');
    ASSERT_NE(begin, std::string::npos);
    const auto value = yaml.find(':', begin) + 1;
    const auto end = yaml.find('\n', value);
    ASSERT_NE(end, std::string::npos);
    yaml.replace(value, end - value, " -1");
    write_test_text(path, yaml);
    EXPECT_THROW(validate_model_bundle(output), std::runtime_error);
  }
  write_test_manifest(path, original_metadata);
  EXPECT_NO_THROW(validate_model_bundle(output));
}

TEST(MemoryExperimentBundle, RejectsExtraModelYamlDocumentsEvenWithValidHash) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  temporary_bundle_directory temporary;
  const auto output = temporary.path / "bundle";
  write_model_bundle(request, model, {}, output);
  const auto model_path = output / "model.yaml";
  const auto manifest_path = output / "manifest.yaml";
  const auto original_yaml = read_test_text(model_path);
  const auto original_metadata = read_test_manifest(manifest_path);
  for (const auto &yaml :
       {std::string("---\n") + original_yaml, original_yaml + "---\n",
        original_yaml + "---\n[]\n", original_yaml + original_yaml}) {
    write_test_text(model_path, yaml);
    auto metadata = original_metadata;
    metadata.model.sha256 = test_digest(yaml);
    write_test_manifest(manifest_path, metadata);
    EXPECT_THROW(validate_model_bundle(output), std::runtime_error);
  }
  write_test_text(model_path, original_yaml);
  write_test_manifest(manifest_path, original_metadata);
  EXPECT_NO_THROW(validate_model_bundle(output));
}

TEST(MemoryExperimentManifest, ExactProbabilityIntegersAndBitStringsRoundTrip) {
  // Exercise metadata scalar representation independently of circuit analysis
  // or model validation, including bit strings wider than the present fixture.
  manifest::document metadata;
  metadata.schema_version = 1;
  metadata.kind = "cudaq-qec-memory-model-bundle";
  metadata.status = "model_prepared_graph_pending";
  metadata.experiment = {"surface_code", 3, 26, "Z", 1};
  metadata.layout = {"cudaq_memory_circuit/v1", 8, 9, 27, 217, 17};
  metadata.delivery = {3, 8, 24, 2, 25};
  metadata.model = {"model.yaml", std::string(64, '0'), 12, 13};
  metadata.normalization = {true, "independent canonical mechanisms"};
  metadata.noise = {"cnot_depolarization", 0.001};
  metadata.final_m2o = {{0, 1, 2}};
  metadata.provenance = {"test-version", "test-revision"};
  metadata.validation = {true, false, false, false};
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  metadata.replay = manifest::replay_metadata{
      "playback-static-source/v1",
      1,
      maximum,
      {{0, 0, maximum, "raw-0-0.txt", std::string(64, '0'), "0010"}},
      false};
  for (const auto probability :
       {std::nextafter(0.001, 1.0), std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(), std::nextafter(1.0, 0.0),
        0.12345678901234567}) {
    metadata.noise.probability = probability;
    const auto yaml = manifest::to_yaml(metadata);
    const auto parsed = manifest::from_yaml(yaml);
    EXPECT_EQ(parsed.noise.probability, probability);
    ASSERT_TRUE(parsed.replay.has_value());
    EXPECT_EQ(parsed.replay->base_seed, maximum);
    ASSERT_EQ(parsed.replay->files.size(), 1);
    EXPECT_EQ(parsed.replay->files[0].seed, maximum);
    EXPECT_EQ(parsed.replay->files[0].raw_logical, "0010");
    EXPECT_NE(yaml.find("'0010'"), std::string::npos);
    EXPECT_FALSE(parsed.validation.graph_generated);
    EXPECT_FALSE(parsed.validation.graph_constructed);
    EXPECT_FALSE(parsed.validation.replayed);
  }
  metadata.replay.reset();
  const auto without_replay = manifest::to_yaml(metadata);
  EXPECT_EQ(without_replay.find("\nreplay:"), std::string::npos);
  EXPECT_FALSE(manifest::from_yaml(without_replay).replay.has_value());
}

TEST(MemoryExperimentManifest,
     RejectsMalformedUnknownDuplicateAndExtraDocuments) {
  const auto valid = manifest::to_yaml(manifest::document{});
  EXPECT_NO_THROW(manifest::from_yaml(valid));
  for (const auto &invalid :
       {std::string{}, std::string("---\n"), std::string("[]\n"),
        std::string("schema_version: [\n"), std::string("---\n") + valid,
        valid + "---\n", valid + valid})
    EXPECT_THROW(manifest::from_yaml(invalid), std::runtime_error);
  EXPECT_THROW(
      manifest::from_yaml(mutate_yaml_field(valid, "", "schema_version",
                                            yaml_field_mutation::duplicate)),
      std::runtime_error);
  EXPECT_THROW(
      manifest::from_yaml(mutate_yaml_field(valid, "noise", "probability",
                                            yaml_field_mutation::unknown)),
      std::runtime_error);
  EXPECT_THROW(manifest::from_yaml(mutate_yaml_field(
                   valid, "model", "path", yaml_field_mutation::missing)),
               std::runtime_error);
}

TEST(MemoryExperimentBundle,
     WrongInRangeReadoutSupportIsRejectedWithoutReplay) {
  llvm::SmallString<128> temporary;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory("cudaq-memory-bundle", temporary));
  struct cleanup_directory {
    std::filesystem::path path;
    ~cleanup_directory() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{temporary.str().str()};
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const preparation_request request{{"surface_code", 3, 26, 1, basis}};
    const auto model = prepare_model(request);
    auto wrong = model;
    wrong.final_m2o = {{0}};
    ASSERT_NE(wrong.final_m2o, model.final_m2o);
    const auto name = basis == memory_basis::x ? "x" : "z";
    const auto rejected = cleanup.path / (std::string(name) + "-rejected");
    EXPECT_THROW(write_model_bundle(request, wrong, {}, rejected),
                 std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(rejected));

    const auto output = cleanup.path / name;
    write_model_bundle(request, model, {}, output);
    const auto manifest_path = output / "manifest.yaml";
    auto metadata = read_test_manifest(manifest_path);
    metadata.final_m2o = {{0}};
    write_test_manifest(manifest_path, metadata);
    EXPECT_THROW(validate_model_bundle(output), std::runtime_error);
  }
}

TEST(MemoryExperimentBundle, PreparationIdentityIsCheckedBeforePublication) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  ASSERT_TRUE(model.preparation.has_value());
  temporary_bundle_directory temporary;
  std::vector<preparation_request> mismatched(5, request);
  mismatched[0].cnot_error_rate = 0.002;
  mismatched[1].decompose_errors = false;
  mismatched[2].experiment.basis = memory_basis::x;
  mismatched[3].experiment.distance = 5;
  mismatched[4].experiment.rounds = 27;
  for (std::size_t i = 0; i < mismatched.size(); ++i) {
    const auto output = temporary.path / std::to_string(i);
    EXPECT_THROW(validate_prepared_model(mismatched[i], model),
                 std::invalid_argument);
    EXPECT_THROW(write_model_bundle(mismatched[i], model, {}, output),
                 std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(output));
  }
  auto unbound = model;
  unbound.preparation.reset();
  EXPECT_THROW(
      write_model_bundle(request, unbound, {}, temporary.path / "unbound"),
      std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(temporary.path / "unbound"));
  auto replicated = request;
  replicated.experiment.logical_qubits = 4;
  EXPECT_NO_THROW(validate_prepared_model(replicated, model));
  EXPECT_NO_THROW(
      write_model_bundle(replicated, model, {}, temporary.path / "replicated"));
}

TEST(MemoryExperimentBundle,
     VersionOneRejectsUnknownMissingAndDuplicateManifestFields) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  temporary_bundle_directory temporary;
  const auto output = temporary.path / "bundle";
  write_model_bundle(request, model, {.batch_rounds = 3, .replay_shots = 1},
                     output);
  const auto path = output / "manifest.yaml";
  const auto original = read_test_text(path);
  const std::vector<std::pair<std::string, std::string>> fields{
      {"", "schema_version"},
      {"experiment", "code_family"},
      {"layout", "convention"},
      {"delivery", "batch_rounds"},
      {"model", "path"},
      {"normalization", "decompose_errors"},
      {"noise", "kind"},
      {"provenance", "qec_version"},
      {"validation", "phase_model_roundtrip"},
      {"replay", "format"},
      {"files", "shot"}};
  for (const auto &[section, key] : fields) {
    for (const auto mutation :
         {yaml_field_mutation::unknown, yaml_field_mutation::missing,
          yaml_field_mutation::duplicate}) {
      write_test_text(path,
                      mutate_yaml_field(original, section, key, mutation));
      EXPECT_THROW(validate_model_bundle(output), std::runtime_error)
          << "invalid field in " << section;
    }
  }
  write_test_text(path, original);
  EXPECT_NO_THROW(validate_model_bundle(output));
}

TEST(MemoryExperimentBundle,
     ReplayValidationStreamsAcrossRecordAndBufferEdges) {
  const preparation_request request{
      {"surface_code", 3, 1024, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  temporary_bundle_directory temporary;
  for (const auto batch :
       {std::uint64_t{3}, std::uint64_t{512}, std::uint64_t{1025}}) {
    const auto output = temporary.path / std::to_string(batch);
    write_model_bundle(request, model,
                       {.batch_rounds = batch, .replay_shots = 1}, output);
    const auto manifest_path = output / "manifest.yaml";
    const auto original = read_test_manifest(manifest_path);
    const auto raw_path = output / "raw-0-0.txt";
    std::ifstream input(raw_path, std::ios::binary);
    const std::string raw((std::istreambuf_iterator<char>(input)), {});
    ASSERT_GT(raw.size(), 8192);
    ASSERT_EQ(raw.back(), '\n');
    input.close();
    const auto replace_raw = [&](const std::string &replacement) {
      {
        std::ofstream rewritten(raw_path, std::ios::binary);
        rewritten << replacement;
        ASSERT_TRUE(rewritten.good());
      }
      auto metadata = original;
      ASSERT_TRUE(metadata.replay.has_value());
      ASSERT_EQ(metadata.replay->files.size(), 1);
      metadata.replay->files[0].sha256 = test_digest(replacement);
      write_test_manifest(manifest_path, metadata);
    };
    // An absent final newline was accepted by the original v1 reader.
    replace_raw(raw.substr(0, raw.size() - 1));
    EXPECT_NO_THROW(validate_model_bundle(output));
    auto malformed = raw;
    malformed[4096] = 'x';
    replace_raw(malformed);
    EXPECT_THROW(validate_model_bundle(output), std::runtime_error);
    replace_raw(raw + "\n");
    EXPECT_THROW(validate_model_bundle(output), std::runtime_error);
    replace_raw(raw.substr(0, raw.size() - 2) + "\n");
    EXPECT_THROW(validate_model_bundle(output), std::runtime_error);
    replace_raw(raw);
    EXPECT_NO_THROW(validate_model_bundle(output));
  }
}

} // namespace
} // namespace cudaq::qec::memory_experiment
