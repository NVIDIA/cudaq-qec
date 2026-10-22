/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "bundle.h"
#include "manifest.h"
#include "raw_replay.h"
#include "cudaq/qec/decoder_init.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/version.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/YAMLParser.h>
#include <llvm/Support/YAMLTraits.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace manifest_data = cudaq::qec::memory_experiment::manifest;

namespace {

// Match the QEC phase-prior serializer: LLVM's default double emitter rounds
// to six significant digits, which is not enough for model provenance.
struct manifest_probability {
  double value = 0;
};

struct logical_bit_string {
  std::string value;
};

void require_single_yaml_mapping(llvm::StringRef source,
                                 const char *description) {
  llvm::SourceMgr source_manager;
  llvm::yaml::Stream documents(source, source_manager, /*ShowColors=*/false);
  std::size_t count = 0;
  for (auto &entry : documents) {
    if (++count != 1)
      throw std::runtime_error(std::string(description) +
                               " requires exactly one document");
    auto *root = entry.getRoot();
    if (!root || !llvm::isa<llvm::yaml::MappingNode>(root))
      throw std::runtime_error(std::string(description) +
                               " requires a mapping document");
    root->skip();
  }
  if (documents.failed() || count != 1)
    throw std::runtime_error(std::string("invalid or empty ") + description);
}

} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(manifest_data::replay_file_metadata)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<std::uint64_t>)

namespace llvm::yaml {

template <>
struct ScalarTraits<manifest_probability> {
  static void output(const manifest_probability &value, void *,
                     raw_ostream &stream) {
    std::array<char, 64> buffer;
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value.value);
    assert(result.ec == std::errc() && "double serialization buffer too small");
    stream << StringRef(buffer.data(), result.ptr - buffer.data());
  }
  static StringRef input(StringRef text, void *context,
                         manifest_probability &value) {
    return ScalarTraits<double>::input(text, context, value.value);
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

template <>
struct ScalarTraits<logical_bit_string> {
  static void output(const logical_bit_string &value, void *,
                     raw_ostream &out) {
    out << value.value;
  }
  static StringRef input(StringRef text, void *, logical_bit_string &value) {
    value.value = text.str();
    return {};
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }
};

template <>
struct MappingTraits<manifest_data::experiment_metadata> {
  static void mapping(IO &io, manifest_data::experiment_metadata &value) {
    io.mapRequired("code_family", value.code_family);
    io.mapRequired("distance", value.distance);
    io.mapRequired("rounds", value.rounds);
    io.mapRequired("basis", value.basis);
    io.mapRequired("logical_qubits", value.logical_qubits);
  }
};

template <>
struct MappingTraits<manifest_data::layout_metadata> {
  static void mapping(IO &io, manifest_data::layout_metadata &value) {
    io.mapRequired("convention", value.convention);
    io.mapRequired("regular_measurements", value.regular_measurements);
    io.mapRequired("final_measurements", value.final_measurements);
    io.mapRequired("input_rounds", value.input_rounds);
    io.mapRequired("raw_measurements", value.raw_measurements);
    io.mapRequired("physical_qubits_per_patch",
                   value.physical_qubits_per_patch);
  }
};

template <>
struct MappingTraits<manifest_data::delivery_metadata> {
  static void mapping(IO &io, manifest_data::delivery_metadata &value) {
    io.mapRequired("batch_rounds", value.batch_rounds);
    io.mapRequired("regular_batches", value.regular_batches);
    io.mapRequired("regular_batch_measurements",
                   value.regular_batch_measurements);
    io.mapRequired("terminal_extraction_rounds",
                   value.terminal_extraction_rounds);
    io.mapRequired("terminal_batch_measurements",
                   value.terminal_batch_measurements);
  }
};

template <>
struct MappingTraits<manifest_data::model_metadata> {
  static void mapping(IO &io, manifest_data::model_metadata &value) {
    io.mapRequired("path", value.path);
    io.mapRequired("sha256", value.sha256);
    io.mapRequired("characterization_rounds", value.characterization_rounds);
    io.mapRequired("validation_rounds", value.validation_rounds);
  }
};

template <>
struct MappingTraits<manifest_data::normalization_metadata> {
  static void mapping(IO &io, manifest_data::normalization_metadata &value) {
    io.mapRequired("decompose_errors", value.decompose_errors);
    io.mapRequired("correlations", value.correlations);
  }
};

template <>
struct MappingTraits<manifest_data::noise_metadata> {
  static void mapping(IO &io, manifest_data::noise_metadata &value) {
    io.mapRequired("kind", value.kind);
    manifest_probability probability{value.probability};
    io.mapRequired("probability", probability);
    if (!io.outputting())
      value.probability = probability.value;
  }
};

template <>
struct MappingTraits<manifest_data::replay_file_metadata> {
  static void mapping(IO &io, manifest_data::replay_file_metadata &value) {
    io.mapRequired("patch", value.patch);
    io.mapRequired("shot", value.shot);
    io.mapRequired("seed", value.seed);
    io.mapRequired("path", value.path);
    io.mapRequired("sha256", value.sha256);
    logical_bit_string logical{value.raw_logical};
    io.mapRequired("raw_logical", logical);
    if (!io.outputting())
      value.raw_logical = std::move(logical.value);
  }
};

template <>
struct MappingTraits<manifest_data::replay_metadata> {
  static void mapping(IO &io, manifest_data::replay_metadata &value) {
    io.mapRequired("format", value.format);
    io.mapRequired("shots_per_patch", value.shots_per_patch);
    io.mapRequired("base_seed", value.base_seed);
    io.mapRequired("files", value.files);
    io.mapRequired("decoded", value.decoded);
  }
};

template <>
struct MappingTraits<manifest_data::provenance_metadata> {
  static void mapping(IO &io, manifest_data::provenance_metadata &value) {
    io.mapRequired("qec_version", value.qec_version);
    io.mapRequired("qec_repository_version", value.qec_repository_version);
  }
};

template <>
struct MappingTraits<manifest_data::validation_metadata> {
  static void mapping(IO &io, manifest_data::validation_metadata &value) {
    io.mapRequired("phase_model_roundtrip", value.phase_model_roundtrip);
    io.mapRequired("graph_generated", value.graph_generated);
    io.mapRequired("graph_constructed", value.graph_constructed);
    io.mapRequired("replayed", value.replayed);
  }
};

template <>
struct MappingTraits<manifest_data::document> {
  static void mapping(IO &io, manifest_data::document &value) {
    io.mapRequired("schema_version", value.schema_version);
    io.mapRequired("kind", value.kind);
    io.mapRequired("status", value.status);
    io.mapRequired("experiment", value.experiment);
    io.mapRequired("layout", value.layout);
    io.mapRequired("delivery", value.delivery);
    io.mapRequired("model", value.model);
    io.mapRequired("normalization", value.normalization);
    io.mapRequired("noise", value.noise);
    io.mapRequired("final_m2o", value.final_m2o);
    io.mapOptional("replay", value.replay);
    io.mapRequired("provenance", value.provenance);
    io.mapRequired("validation", value.validation);
  }
};

} // namespace llvm::yaml

namespace cudaq::qec::memory_experiment::manifest {

std::string to_yaml(document metadata) {
  // Take ownership for LLVM's mutable mapping interface. Bundle publication
  // moves its metadata here instead of copying every replay-file entry.
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::yaml::Output output(stream);
  output << metadata;
  return text;
}

document from_yaml(std::string_view yaml) {
  const llvm::StringRef source(yaml.data(), yaml.size());
  require_single_yaml_mapping(source, "manifest YAML");
  document metadata;
  llvm::yaml::Input input(source);
  input >> metadata;
  if (const auto error = input.error())
    throw std::runtime_error("invalid manifest YAML: " + error.message());
  return metadata;
}

} // namespace cudaq::qec::memory_experiment::manifest

namespace cudaq::qec::memory_experiment {
namespace {

namespace fs = std::filesystem;
namespace config = cudaq::qec::decoding::config;
constexpr std::uint64_t bundle_version = 1;
constexpr auto bundle_kind = "cudaq-qec-memory-model-bundle";
constexpr auto bundle_status = "model_prepared_graph_pending";
constexpr auto model_filename = "model.yaml";
constexpr auto manifest_filename = "manifest.yaml";
constexpr std::size_t replay_validation_buffer_bytes = 4096;

std::string replay_filename(std::uint64_t patch, std::uint64_t shot) {
  return "raw-" + std::to_string(patch) + "-" + std::to_string(shot) + ".txt";
}

std::string sha256(llvm::StringRef text) {
  const auto digest = llvm::SHA256::hash(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(text.data()), text.size()));
  return llvm::toHex(llvm::ArrayRef<std::uint8_t>(digest), true);
}

std::ifstream open_input(const fs::path &path) {
  if (fs::is_symlink(path) || !fs::is_regular_file(path))
    throw std::runtime_error(
        "bundle file must be a regular, non-symlink file: " + path.string());
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open bundle file: " + path.string());
  return input;
}

std::string read_file(const fs::path &path) {
  auto input = open_input(path);
  std::string text((std::istreambuf_iterator<char>(input)), {});
  if (input.bad())
    throw std::runtime_error("cannot read bundle file: " + path.string());
  return text;
}

void write_file(const fs::path &path, const std::string &text) {
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.open(path, std::ios::binary);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  output.close();
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(std::string("invalid model bundle: ") + message);
}

struct replay_file_result {
  std::string digest;
  std::string raw_logical;
};

replay_file_result validate_replay_file(
    const fs::path &path, const delivery_layout &batches,
    std::uint64_t final_measurements,
    const std::vector<std::vector<std::uint64_t>> &readout_support) {
  auto input = open_input(path);
  llvm::SHA256 digest;
  std::array<char, replay_validation_buffer_bytes> buffer;
  std::uint64_t record = 0;
  std::uint64_t position = 0;
  std::vector<std::size_t> support_cursors(readout_support.size());
  std::string logical(readout_support.size(), '0');
  const auto readout_begin =
      batches.terminal_batch_measurements - final_measurements;
  while (input) {
    input.read(buffer.data(), buffer.size());
    const auto count = static_cast<std::size_t>(input.gcount());
    digest.update(llvm::StringRef(buffer.data(), count));
    for (std::size_t i = 0; i < count; ++i) {
      require(record <= batches.regular_batches,
              "too many replay input records");
      const auto width = record < batches.regular_batches
                             ? batches.regular_batch_measurements
                             : batches.terminal_batch_measurements;
      const auto bit = buffer[i];
      if (bit == '\n') {
        require(position == width, "invalid replay record width");
        ++record;
        position = 0;
        continue;
      }
      require(position < width && (bit == '0' || bit == '1'),
              "invalid replay record width or bits");
      if (record == batches.regular_batches && position >= readout_begin) {
        const auto data = position - readout_begin;
        for (std::size_t observable = 0; observable < readout_support.size();
             ++observable) {
          const auto &row = readout_support[observable];
          auto &cursor = support_cursors[observable];
          if (cursor < row.size() && row[cursor] == data) {
            if (bit == '1')
              logical[observable] = logical[observable] == '0' ? '1' : '0';
            ++cursor;
          }
        }
      }
      ++position;
    }
  }
  if (input.bad() || !input.eof())
    throw std::runtime_error("cannot read replay file: " + path.string());
  // The existing v1 reader accepts an unterminated final line. Preserve that
  // contract while the generator continues writing canonical LF-terminated
  // records. No other missing delimiter or extra record is accepted.
  if (record == batches.regular_batches &&
      position == batches.terminal_batch_measurements) {
    ++record;
    position = 0;
  }
  require(record == batches.regular_batches + 1 && position == 0,
          "missing replay terminal record");
  const auto hash = digest.final();
  return {llvm::toHex(llvm::ArrayRef<std::uint8_t>(hash), true),
          std::move(logical)};
}

void validate_phase_model(const dem_chunks_spec &spec) {
  spec.validate();
  require(spec.num_rounds.has_value() && spec.has_D_sparse(),
          "closed horizon and per-phase detector mapping are required");
  // Validate connections, seams and D at bounded horizons, not by expanding a
  // potentially billion-round experiment. The second bulk visit tests its
  // self-connection as well as both endpoint connections.
  (void)spec.repeating_rounds();
  auto bounded = spec;
  bounded.num_rounds = std::max<std::uint64_t>(2, spec.minimum_rounds()) + 1;
  (void)decoder_init::from_dem_chunks(bounded);
}

manifest::document make_manifest(const preparation_request &request,
                                 const prepared_model &model,
                                 const bundle_options &options,
                                 const std::string &model_yaml) {
  const auto &spec = request.experiment;
  const memory_record_layout layout(spec.rounds, model.regular_measurements,
                                    model.final_measurements);
  const auto delivery = layout.batches(options.batch_rounds);
  manifest::document result;
  result.schema_version = bundle_version;
  result.kind = bundle_kind;
  result.status = bundle_status;
  result.experiment = {spec.code_family, spec.distance, spec.rounds,
                       spec.basis == memory_basis::x ? "X" : "Z",
                       spec.logical_qubits};
  result.layout = {"cudaq_memory_circuit/v1", model.regular_measurements,
                   model.final_measurements,  layout.input_rounds(),
                   layout.raw_measurements(), model.physical_qubits};
  result.delivery = {options.batch_rounds, delivery.regular_batches,
                     delivery.regular_batch_measurements,
                     delivery.terminal_extraction_rounds,
                     delivery.terminal_batch_measurements};
  result.model = {model_filename, sha256(model_yaml),
                  model.characterization_rounds, model.validation_rounds};
  result.normalization = {request.decompose_errors,
                          "independent canonical mechanisms"};
  result.noise = {"cnot_depolarization", request.cnot_error_rate};
  result.final_m2o = model.final_m2o;
  result.provenance = {getVersion(), getFullRepositoryVersion()};
  result.validation = {true, false, false, false};
  return result;
}

} // namespace

void validate_model_bundle(const fs::path &directory) {
  const auto metadata =
      manifest::from_yaml(read_file(directory / manifest_filename));
  require(metadata.schema_version == bundle_version,
          "unsupported schema_version");
  require(metadata.kind == bundle_kind && metadata.status == bundle_status,
          "not a prepared memory-model bundle");
  const auto &experiment = metadata.experiment;
  const auto &basis = experiment.basis;
  require(basis == "X" || basis == "Z", "basis must be X or Z");
  const experiment_spec spec{experiment.code_family, experiment.distance,
                             experiment.rounds, experiment.logical_qubits,
                             basis == "X" ? memory_basis::x : memory_basis::z};
  validate_experiment_spec(spec);
  const auto surface = create_experiment_code(spec);
  const auto expected_data = surface->get_num_data_qubits();
  const auto expected_regular =
      surface->get_num_ancilla_x_qubits() + surface->get_num_ancilla_z_qubits();
  require(metadata.model.path == model_filename,
          "model path must be model.yaml");
  const auto yaml = read_file(directory / model_filename);
  require(metadata.model.sha256 == sha256(yaml), "model SHA-256 mismatch");
  require_single_yaml_mapping(yaml, "model YAML");
  const auto model = config::dem_chunks_from_yaml_str(yaml);
  validate_phase_model(model);
  validate_characterization_provenance(model, expected_regular, expected_data,
                                       metadata.model.characterization_rounds,
                                       metadata.model.validation_rounds);
  require(model.num_rounds == spec.rounds + 1, "model horizon mismatch");
  const auto &widths = metadata.layout;
  require(widths.convention == "cudaq_memory_circuit/v1",
          "unsupported record convention");
  const auto regular = widths.regular_measurements;
  const auto final = widths.final_measurements;
  const memory_record_layout layout(spec.rounds, regular, final);
  require(
      final == expected_data && regular == expected_regular &&
          widths.physical_qubits_per_patch ==
              surface->get_num_ancilla_qubits() + expected_data,
      "record widths or physical qubits disagree with Surface Code geometry");
  require(widths.input_rounds == layout.input_rounds() &&
              widths.raw_measurements == layout.raw_measurements(),
          "raw record counts disagree");
  require(model.measurements_per_round == regular,
          "regular measurement width disagrees with model");
  auto compact = model;
  compact.num_rounds = std::max<std::uint64_t>(2, model.minimum_rounds()) + 1;
  const auto sequence = compact.phase_sequence();
  for (std::size_t i = 0; i < sequence.size(); ++i)
    require(model.measurements_of(model.get_phase(sequence[i])) ==
                (i + 1 == sequence.size() ? final : regular),
            "phase measurement width disagrees with layout");
  const auto &delivery = metadata.delivery;
  const auto batches = layout.batches(delivery.batch_rounds);
  require(delivery.regular_batches == batches.regular_batches &&
              delivery.regular_batch_measurements ==
                  batches.regular_batch_measurements &&
              delivery.terminal_extraction_rounds ==
                  batches.terminal_extraction_rounds &&
              delivery.terminal_batch_measurements ==
                  batches.terminal_batch_measurements,
          "delivery counts disagree with record layout");
  const auto &readout = metadata.final_m2o;
  const auto observables = spec.basis == memory_basis::x
                               ? surface->get_observables_x()
                               : surface->get_observables_z();
  const auto expected_readout = final_readout_support(observables);
  require(
      expected_readout.size() == 1,
      "one logical observable per independent patch is currently supported");
  require(readout.size() == expected_readout.size(),
          "final raw-logical map count disagrees with code and basis");
  for (const auto &phase : model.phases)
    require(std::count(phase.spec.O_sparse.begin(), phase.spec.O_sparse.end(),
                       std::int64_t{-1}) == expected_readout.size(),
            "phase fault-observable count disagrees with raw-logical count");
  for (const auto &row : readout) {
    require(!row.empty(), "empty raw-logical map");
    std::optional<std::uint64_t> previous;
    for (const auto index : row) {
      require(index < final && (!previous || *previous < index),
              "invalid raw-logical measurement index");
      previous = index;
    }
  }
  require(readout == expected_readout,
          "raw-logical map disagrees with code and basis");
  const auto &noise = metadata.noise;
  require(noise.kind == "cnot_depolarization" &&
              std::isfinite(noise.probability) && noise.probability > 0 &&
              noise.probability < 1,
          "unsupported or invalid noise prescription");
  const auto &normalization = metadata.normalization;
  require(normalization.correlations == "independent canonical mechanisms",
          "unsupported model normalization");
  const auto &provenance = metadata.provenance;
  require(!provenance.qec_version.empty() &&
              !provenance.qec_repository_version.empty(),
          "missing framework provenance");
  if (metadata.replay) {
    const auto &replay = *metadata.replay;
    require(replay.format == "playback-static-source/v1",
            "unsupported replay input format");
    const auto shots = replay.shots_per_patch;
    const auto seed = replay.base_seed;
    require(seed > 0 && shots > 0 &&
                spec.logical_qubits <=
                    std::numeric_limits<std::uint64_t>::max() / shots,
            "invalid replay shot count");
    const auto count = shots * spec.logical_qubits;
    require(seed <= std::numeric_limits<std::uint64_t>::max() - (count - 1),
            "replay seed range overflows");
    const auto &files = replay.files;
    require(files.size() == count, "replay file count disagrees");
    for (std::size_t i = 0; i < files.size(); ++i) {
      const auto &file = files[i];
      const auto name = replay_filename(i / shots, i % shots);
      require(file.patch == i / shots && file.shot == i % shots &&
                  file.seed == seed + i && file.path == name,
              "replay identity or path disagrees");
      const auto sampled =
          validate_replay_file(directory / name, batches, final, readout);
      require(file.sha256 == sampled.digest, "replay SHA-256 mismatch");
      require(file.raw_logical == sampled.raw_logical,
              "raw-logical value disagrees with terminal measurements");
    }
    require(!replay.decoded,
            "raw replay inputs must not claim decoder execution");
  }
  const auto &validation = metadata.validation;
  require(validation.phase_model_roundtrip && !validation.graph_generated &&
              !validation.graph_constructed && !validation.replayed,
          "model-only bundle must not claim graph execution");
}

void write_model_bundle(const preparation_request &request,
                        const prepared_model &model,
                        const bundle_options &options, const fs::path &output) {
  validate_prepared_model(request, model);
  validate_raw_replay_options(request.experiment.logical_qubits,
                              options.replay_shots, options.seed);
  validate_phase_model(model.chunks);
  const auto yaml = config::dem_chunks_to_yaml_str(model.chunks);
  require(config::dem_chunks_from_yaml_str(yaml) == model.chunks,
          "phase-model YAML roundtrip changed the model");
  auto metadata = make_manifest(request, model, options, yaml);
  // Exclusive directory creation is the publication reservation. Never erase
  // or replace a pre-existing output, including an empty directory or symlink.
  if (!fs::create_directory(output))
    throw std::runtime_error("output already exists: " + output.string());
  try {
    write_file(output / model_filename, yaml);
    if (options.replay_shots != 0) {
      std::vector<manifest::replay_file_metadata> files;
      const memory_record_layout layout(request.experiment.rounds,
                                        model.regular_measurements,
                                        model.final_measurements);
      const auto batches = layout.batches(options.batch_rounds);
      sample_raw_replay(
          request, model, options.replay_shots, options.seed,
          [&](const raw_replay_shot &shot) {
            const auto name = replay_filename(shot.patch, shot.shot);
            std::ofstream raw;
            raw.exceptions(std::ios::badbit | std::ios::failbit);
            raw.open(output / name, std::ios::binary);
            llvm::SHA256 digest;
            std::size_t offset = 0;
            for (std::uint64_t batch = 0; batch <= batches.regular_batches;
                 ++batch) {
              const auto width = batch < batches.regular_batches
                                     ? batches.regular_batch_measurements
                                     : batches.terminal_batch_measurements;
              const llvm::StringRef bits(shot.measurements.data() + offset,
                                         width);
              raw.write(bits.data(), bits.size());
              raw.put('\n');
              digest.update(bits);
              digest.update("\n");
              offset += width;
            }
            raw.close();
            const auto hash = digest.final();
            files.push_back(
                {shot.patch, shot.shot, shot.seed, name,
                 llvm::toHex(llvm::ArrayRef<std::uint8_t>(hash), true),
                 shot.raw_logical});
          });
      metadata.replay = manifest::replay_metadata{
          "playback-static-source/v1", options.replay_shots, options.seed,
          std::move(files), false};
    }
    // Manifest is the completion marker and is written after all model data.
    write_file(output / manifest_filename,
               manifest::to_yaml(std::move(metadata)));
    validate_model_bundle(output);
  } catch (...) {
    // Only this call's freshly created directory is eligible for cleanup.
    std::error_code ignored;
    fs::remove_all(output, ignored);
    throw;
  }
}

} // namespace cudaq::qec::memory_experiment
