/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "realtime_decoding.h"
#include "cudaq/qec/decoder_config_payload.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

bool decoder_custom_args_t::operator==(
    const decoder_custom_args_t &other) const {
  return custom_args_maps_equal(map_, other.map_);
}

namespace {

constexpr const char *kErrorRateVec = "error_rate_vec";

struct legacy_error_rates {
  std::vector<double> values;
  std::string path;
};

const param_spec *find_param(const decoder_schema &schema,
                             const std::string &key) {
  for (const auto &candidate : schema.params)
    if (candidate.key == key)
      return &candidate;
  return nullptr;
}

const decoder_schema *nested_schema_for(const param_spec &spec,
                                        const cudaqx::heterogeneous_map &map) {
  if (spec.kind == param_kind::subschema)
    return find_decoder_schema(spec.subschema);
  if (spec.kind != param_kind::discriminated ||
      !map.contains(spec.discriminator))
    return nullptr;
  try {
    return find_decoder_schema(map.get<std::string>(spec.discriminator));
  } catch (const std::runtime_error &) {
    // Canonical schema validation below reports the ill-typed discriminator.
    return nullptr;
  }
}

void record_legacy_error_rates(const cudaqx::heterogeneous_map &map,
                               const std::string &path,
                               std::optional<legacy_error_rates> &found) {
  std::vector<double> values;
  try {
    values = map.get<std::vector<double>>(kErrorRateVec);
  } catch (const std::runtime_error &) {
    throw std::runtime_error("'" + path +
                             "' must be a sequence of floating-point values.");
  }
  if (found)
    throw std::runtime_error(
        "error_rate_vec is supplied more than once in decoder_custom_args: '" +
        found->path + "' and '" + path +
        "'. Supply model error rates in exactly one location.");
  found = legacy_error_rates{std::move(values), path};
}

cudaqx::heterogeneous_map strip_legacy_error_rates(
    const decoder_schema *schema, const cudaqx::heterogeneous_map &map,
    const std::string &path, std::optional<legacy_error_rates> &found) {
  cudaqx::heterogeneous_map filtered;
  for (const auto &kv : map) {
    if (kv.first == kErrorRateVec) {
      record_legacy_error_rates(map, path + "." + kErrorRateVec, found);
      continue;
    }

    const param_spec *spec = schema ? find_param(*schema, kv.first) : nullptr;
    const decoder_schema *nested_schema =
        spec ? nested_schema_for(*spec, map) : nullptr;
    if (nested_schema) {
      if (const auto *nested =
              std::any_cast<cudaqx::heterogeneous_map>(&kv.second)) {
        filtered.insert(kv.first,
                        strip_legacy_error_rates(nested_schema, *nested,
                                                 path + "." + kv.first, found));
        continue;
      }
    }
    filtered.insert(kv.first, kv.second);
  }
  return filtered;
}

struct normalized_decoder_args {
  cudaqx::heterogeneous_map plugin_args;
  std::optional<legacy_error_rates> legacy_rates;
};

normalized_decoder_args normalize_decoder_args(const decoder_config &config) {
  normalized_decoder_args normalized;
  normalized.plugin_args = strip_legacy_error_rates(
      find_decoder_schema(config.type), config.decoder_custom_args.map(),
      "decoder_custom_args", normalized.legacy_rates);
  if (!config.error_rate_vec.empty() && normalized.legacy_rates)
    throw std::runtime_error(
        "error_rate_vec is supplied both at the decoder configuration level "
        "and at '" +
        normalized.legacy_rates->path +
        "'. Supply model error rates in exactly one location.");
  return normalized;
}

} // namespace

void decoder_config::validate_custom_args() const {
  auto normalized = normalize_decoder_args(*this);
  config::validate_custom_args(type, normalized.plugin_args);
}

std::vector<double> decoder_config::effective_error_rate_vec() const {
  auto normalized = normalize_decoder_args(*this);
  if (normalized.legacy_rates)
    return normalized.legacy_rates->values;
  return error_rate_vec;
}

/// Flatten the nested sparse form the decoder API uses into the -1-terminated
/// row form the configuration uses.
static std::vector<std::int64_t>
flatten_sparse_rows(const std::vector<std::vector<std::uint32_t>> &rows) {
  std::vector<std::int64_t> flat;
  std::size_t total = rows.size();
  for (const auto &row : rows)
    total += row.size();
  flat.reserve(total);
  for (const auto &row : rows) {
    for (const auto column : row)
      flat.push_back(static_cast<std::int64_t>(column));
    flat.push_back(-1);
  }
  return flat;
}

std::optional<cudaq::qec::detector_error_model>
expand_dem_chunks(decoder_config &config) {
  if (!config.dem_chunks.has_value() || !config.H_sparse.empty())
    return std::nullopt;

  const auto &spec = *config.dem_chunks;
  const cudaq::qec::seam_id from_seam = spec.seam.from_seam;
  const cudaq::qec::seam_id to_seam = spec.seam.to_seam;

  std::vector<cudaq::qec::extended_dem> chunks;
  try {
    chunks = cudaq::qec::dem_chunks_from_spec(spec);
  } catch (const std::invalid_argument &error) {
    throw std::runtime_error("Cannot expand dem_chunks for decoder " +
                             std::to_string(config.id) + ": " + error.what());
  }

  cudaq::qec::detector_error_model closed;
  std::vector<std::vector<std::uint32_t>> d_sparse;
  try {
    closed = cudaq::qec::dem_close_all(chunks, from_seam, to_seam);
    d_sparse = cudaq::qec::dem_chunks_to_d_sparse(chunks, from_seam, to_seam);
  } catch (const std::exception &error) {
    throw std::runtime_error("Cannot close dem_chunks for decoder " +
                             std::to_string(config.id) + ": " + error.what());
  }

  config.block_size = closed.num_error_mechanisms();
  config.syndrome_size = closed.num_detectors();
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(closed.detector_error_matrix);
  config.O_sparse =
      cudaq::qec::pcm_to_sparse_vec(closed.observables_flips_matrix);
  config.D_sparse = flatten_sparse_rows(d_sparse);
  config.error_rate_vec = closed.error_rates;
  return closed;
}

cudaqx::heterogeneous_map
decoder_config::decoder_custom_args_to_heterogeneous_map() const {
  auto args = normalize_decoder_args(*this).plugin_args;
  if (const auto *schema = find_decoder_schema(type)) {
    // Non-schema keys are warned-and-dropped and schema-declared defaults are
    // materialized. The YAML serializer starts from this plugin-safe map and
    // adds the framework-owned error-rate compatibility alias back only to
    // the wire representation.
    drop_non_schema_keys(*schema, args);
    materialize_default_args(*schema, args);
  }
  return args;
}

void multi_decoder_config::validate_custom_args() const {
  for (const auto &decoder : decoders)
    decoder.validate_custom_args();
}

// Post-parse pass over a schema-parsed custom-args map: materialize defaulted
// discriminated sections, then run the canonical registry validation
// (required keys, per-schema hooks; its unknown-key check is a no-op here
// because the parser already rejected unknown keys). Only invoked when the
// section was present in the input document, mirroring the previous behavior
// where an absent decoder_custom_args section skipped its mapping (and
// therefore its required-key checks) entirely.
static void finalize_parsed_args(const decoder_schema &schema,
                                 cudaqx::heterogeneous_map &map,
                                 const std::string &context) {
  materialize_default_args(schema, map);
  validate_custom_args(schema, map, context);
}

} // namespace cudaq::qec::decoding::config

LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::decoding::config::decoder_config)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::phase_connection)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::seam_spec_entry)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::phase_spec_entry)

namespace llvm::yaml {

// Binds a heterogeneous_map to the decoder_schema that describes it so the
// generic mapping traits below can convert between the two. The schema drives
// everything: which keys are legal, the canonical storage type of each value,
// and how nested sections resolve their schemas.
struct schema_binding {
  cudaqx::heterogeneous_map *map = nullptr;
  const cudaq::qec::decoding::config::decoder_schema *schema = nullptr;
};

namespace {

template <typename T>
void input_schema_scalar(IO &io, const std::string &key,
                         cudaqx::heterogeneous_map &map) {
  T value{};
  io.mapRequired(key.c_str(), value);
  map.insert(key, value);
}

template <typename T>
void output_schema_scalar(IO &io, const std::string &key,
                          const cudaqx::heterogeneous_map &map) {
  T value = map.get<T>(key);
  io.mapRequired(key.c_str(), value);
}

} // namespace

template <>
struct CustomMappingTraits<schema_binding> {
  using param_kind = cudaq::qec::decoding::config::param_kind;
  using param_spec = cudaq::qec::decoding::config::param_spec;

  static void inputOne(IO &io, StringRef key, schema_binding &binding) {
    const std::string key_str = key.str();
    // Compatibility spelling for the framework-owned model rates. It is
    // accepted in every decoder parameter section, then promoted out before
    // schema validation or plugin construction.
    if (key_str == "error_rate_vec") {
      input_schema_scalar<std::vector<double>>(io, key_str, *binding.map);
      return;
    }
    const param_spec *spec = nullptr;
    for (const auto &candidate : binding.schema->params) {
      if (candidate.key == key_str) {
        spec = &candidate;
        break;
      }
    }
    if (!spec)
      throw std::runtime_error("Unknown key '" + key_str + "' in '" +
                               binding.schema->name + "' parameters.");

    switch (spec->kind) {
    case param_kind::boolean:
      input_schema_scalar<bool>(io, key_str, *binding.map);
      break;
    case param_kind::int32:
      input_schema_scalar<int>(io, key_str, *binding.map);
      break;
    case param_kind::uint64:
      input_schema_scalar<std::size_t>(io, key_str, *binding.map);
      break;
    case param_kind::f64:
      input_schema_scalar<double>(io, key_str, *binding.map);
      break;
    case param_kind::string:
      input_schema_scalar<std::string>(io, key_str, *binding.map);
      break;
    case param_kind::f64_vec:
      input_schema_scalar<std::vector<double>>(io, key_str, *binding.map);
      break;
    case param_kind::f64_matrix:
      input_schema_scalar<std::vector<std::vector<double>>>(io, key_str,
                                                            *binding.map);
      break;
    case param_kind::subschema: {
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(spec->subschema);
      if (!nested_schema)
        throw std::runtime_error("No schema registered under '" +
                                 spec->subschema + "' (needed to parse '" +
                                 key_str + "').");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    case param_kind::discriminated: {
      // The nested schema is named by a sibling key. Read it through the IO
      // (document order does not matter; mapping keys are random access).
      std::string discriminator_value;
      io.mapOptional(spec->discriminator.c_str(), discriminator_value);
      if (discriminator_value.empty())
        throw std::runtime_error("'" + key_str + "' is present but '" +
                                 spec->discriminator + "' is not set.");
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(
              discriminator_value);
      if (!nested_schema)
        throw std::runtime_error(
            "'" + key_str + "' does not support " + spec->discriminator + " '" +
            discriminator_value +
            "': no parameter schema is registered under that name.");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    }
  }

  static void output(IO &io, schema_binding &binding) {
    // Only schema keys and the framework compatibility alias are emitted;
    // surface anything else (a typo in a programmatically built map) instead
    // of dropping it silently.
    for (const auto &kv : *binding.map) {
      bool known = false;
      for (const auto &spec : binding.schema->params) {
        if (spec.key == kv.first) {
          known = true;
          break;
        }
      }
      if (!known && kv.first != "error_rate_vec")
        CUDA_QEC_WARN("Key '{}' is not in the '{}' parameter schema; it is "
                      "omitted from the emitted YAML.",
                      kv.first, binding.schema->name);
    }
    if (binding.map->contains("error_rate_vec"))
      output_schema_scalar<std::vector<double>>(io, "error_rate_vec",
                                                *binding.map);
    // Emit in schema declaration order so output is deterministic.
    for (const auto &spec : binding.schema->params) {
      if (spec.key == "error_rate_vec")
        continue;
      if (!binding.map->contains(spec.key))
        continue;
      switch (spec.kind) {
      case param_kind::boolean:
        output_schema_scalar<bool>(io, spec.key, *binding.map);
        break;
      case param_kind::int32:
        output_schema_scalar<int>(io, spec.key, *binding.map);
        break;
      case param_kind::uint64:
        output_schema_scalar<std::size_t>(io, spec.key, *binding.map);
        break;
      case param_kind::f64:
        output_schema_scalar<double>(io, spec.key, *binding.map);
        break;
      case param_kind::string:
        output_schema_scalar<std::string>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_vec:
        output_schema_scalar<std::vector<double>>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_matrix:
        output_schema_scalar<std::vector<std::vector<double>>>(io, spec.key,
                                                               *binding.map);
        break;
      case param_kind::subschema: {
        const auto *nested_schema =
            cudaq::qec::decoding::config::find_decoder_schema(spec.subschema);
        if (!nested_schema)
          throw std::runtime_error("No schema registered under '" +
                                   spec.subschema + "' (needed to emit '" +
                                   spec.key + "').");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      case param_kind::discriminated: {
        std::string discriminator_value;
        if (binding.map->contains(spec.discriminator))
          discriminator_value =
              binding.map->get<std::string>(spec.discriminator);
        const auto *nested_schema =
            discriminator_value.empty()
                ? nullptr
                : cudaq::qec::decoding::config::find_decoder_schema(
                      discriminator_value);
        if (!nested_schema)
          throw std::runtime_error("'" + spec.key +
                                   "' is present but no parameter schema is "
                                   "registered for " +
                                   spec.discriminator + " '" +
                                   discriminator_value + "'.");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      }
    }
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::decoding::config::DecoderDispatch> {
  static void
  enumeration(IO &io, cudaq::qec::decoding::config::DecoderDispatch &value) {
    io.enumCase(value, "host",
                cudaq::qec::decoding::config::DecoderDispatch::host);
    io.enumCase(value, "device_graph",
                cudaq::qec::decoding::config::DecoderDispatch::device_graph);
  }
};

// Per-seam sparse spec: H_sparse and O_sparse for one named seam boundary.
template <>
struct MappingTraits<cudaq::qec::dem_seam_spec> {
  static void mapping(IO &io, cudaq::qec::dem_seam_spec &spec) {
    io.mapOptional("H_sparse", spec.H_sparse);
    io.mapOptional("O_sparse", spec.O_sparse, std::vector<std::int64_t>{});
  }
};

// seam_spec_entry holds a seam_id (hash) and its spec. On input the key
// is the name string; on output we cannot recover the string from the hash,
// so the seam_specs sequence form is input-only (the flat form is used for
// output after expand_dem_chunks() runs).
template <>
struct MappingTraits<cudaq::qec::seam_spec_entry> {
  static void mapping(IO &io, cudaq::qec::seam_spec_entry &entry) {
    std::string name;
    io.mapRequired("name", name);
    io.mapRequired("spec", entry.spec);
    if (!io.outputting()) {
      entry.id = cudaq::qec::seam_id{name.c_str()};
      cudaq::qec::seam_id::register_name(entry.id, name);
    }
  }
};

// One DEM phase/chunk as flat index lists.
// Shorthand: H_sparse at chunk level (memory experiments).
// Full: seam_specs sequence with per-seam H_sparse/O_sparse.
template <>
struct MappingTraits<cudaq::qec::dem_chunk_spec> {
  static void mapping(IO &io, cudaq::qec::dem_chunk_spec &spec) {
    io.mapRequired("num_faults", spec.num_faults);
    io.mapOptional("H_sparse", spec.H_sparse);
    io.mapOptional("seam_specs", spec.seam_specs);
    io.mapOptional("O_sparse", spec.O_sparse, std::vector<std::int64_t>{});
    io.mapRequired("error_rates", spec.error_rates);
  }
};

// seam_connection: {from: name, to: name}
template <>
struct MappingTraits<cudaq::qec::seam_connection> {
  static void mapping(IO &io, cudaq::qec::seam_connection &conn) {
    std::string from_str, to_str;
    io.mapRequired("from", from_str);
    io.mapRequired("to", to_str);
    if (!io.outputting()) {
      conn.from_seam = cudaq::qec::seam_id{from_str.c_str()};
      conn.to_seam = cudaq::qec::seam_id{to_str.c_str()};
      cudaq::qec::seam_id::register_name(conn.from_seam, from_str);
      cudaq::qec::seam_id::register_name(conn.to_seam, to_str);
    }
  }
};

// phase_connection: {from: name, to: name}
template <>
struct MappingTraits<cudaq::qec::phase_connection> {
  static void mapping(IO &io, cudaq::qec::phase_connection &conn) {
    std::string from_str, to_str;
    io.mapRequired("from", from_str);
    io.mapRequired("to", to_str);
    if (!io.outputting()) {
      conn.from_phase = cudaq::qec::phase_id{from_str.c_str()};
      conn.to_phase = cudaq::qec::phase_id{to_str.c_str()};
      cudaq::qec::seam_id::register_name(conn.from_phase, from_str);
      cudaq::qec::seam_id::register_name(conn.to_phase, to_str);
    }
  }
};

// phase_spec_entry: {name: string, spec: dem_chunk_spec}
template <>
struct MappingTraits<cudaq::qec::phase_spec_entry> {
  static void mapping(IO &io, cudaq::qec::phase_spec_entry &entry) {
    std::string name;
    io.mapRequired("name", name);
    io.mapRequired("spec", entry.spec);
    if (!io.outputting()) {
      entry.id = cudaq::qec::phase_id{name.c_str()};
      cudaq::qec::seam_id::register_name(entry.id, name);
    }
  }
};

// Multi-phase DEM decomposition.
// YAML format:
//   seam: {from: next_round, to: prev_round}
//   connections:
//     - {from: init, to: bulk}
//     - {from: bulk, to: bulk}
//     - {from: bulk, to: final}
//   num_rounds: 5
//   phases:
//     - {name: init, spec: {num_faults: 3, H_sparse: [...], ...}}
//     - {name: bulk, spec: {...}}
//     - {name: final, spec: {...}}
template <>
struct MappingTraits<cudaq::qec::dem_chunks_spec> {
  static void mapping(IO &io, cudaq::qec::dem_chunks_spec &spec) {
    io.mapRequired("seam", spec.seam);
    io.mapRequired("connections", spec.connections);
    io.mapOptional("num_rounds", spec.num_rounds);
    io.mapRequired("phases", spec.phases);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::decoder_config> {
  static void mapping(IO &io,
                      cudaq::qec::decoding::config::decoder_config &config) {
    // io.keys() reports every key the traits below ask for as well as the ones
    // the document actually carries, so it can only tell absent from present
    // before any mapping call has registered a name. The flat-form check at
    // the bottom needs that distinction, because a field that is missing and
    // one that is present but empty parse to the same value.
    std::vector<std::string> document_keys;
    if (!io.outputting())
      for (const auto key : io.keys())
        document_keys.emplace_back(key.str());
    const auto in_document = [&document_keys](const char *name) {
      return std::find(document_keys.begin(), document_keys.end(), name) !=
             document_keys.end();
    };

    io.mapRequired("id", config.id);
    io.mapRequired("type", config.type);
    const auto *schema =
        cudaq::qec::decoding::config::find_decoder_schema(config.type);
    io.mapOptional("dispatch", config.dispatch,
                   cudaq::qec::decoding::config::DecoderDispatch::host);
    io.mapOptional("cuda_device_id", config.cuda_device_id);
    // The DEM arrives in one of three forms (see decoder_config). Everything
    // below the mapping calls enforces that exactly one of them is described,
    // because the other two forms derive the flat fields and a config that
    // spelled out more than one could disagree with itself.
    // A flat configuration names all five on the way out as well as the way
    // in. Emitting only the ones that differ from their defaults would drop a
    // legitimately empty O_sparse (a DEM with no observables) and leave behind
    // a document that no longer parses. A chunk-form or DEM-sourced
    // configuration takes the optional path so its derived fields stay absent,
    // which is what makes the emitted document re-parse in the form it was
    // written in.
    const bool emitting_flat_form =
        io.outputting() &&
        !((config.dem_chunks.has_value() || !config.stim_dem_path.empty()) &&
          config.H_sparse.empty());
    if (emitting_flat_form) {
      io.mapRequired("block_size", config.block_size);
      io.mapRequired("syndrome_size", config.syndrome_size);
      io.mapRequired("H_sparse", config.H_sparse);
      io.mapRequired("O_sparse", config.O_sparse);
      io.mapRequired("D_sparse", config.D_sparse);
    } else {
      io.mapOptional("block_size", config.block_size, std::uint64_t{0});
      io.mapOptional("syndrome_size", config.syndrome_size, std::uint64_t{0});
      io.mapOptional("H_sparse", config.H_sparse, std::vector<std::int64_t>{});
      io.mapOptional("O_sparse", config.O_sparse, std::vector<std::int64_t>{});
      io.mapOptional("D_sparse", config.D_sparse, std::vector<std::int64_t>{});
    }
    // Continue accepting the PR765 top-level spelling, but registered decoder
    // configurations are emitted with rates under decoder_custom_args so the
    // established researcher-facing wire YAML does not change.
    if (!io.outputting() || !schema)
      io.mapOptional("error_rate_vec", config.error_rate_vec);
    io.mapOptional("dem_chunks", config.dem_chunks);
    io.mapOptional("stim_dem_path", config.stim_dem_path, std::string{});

    // Convert decoder_custom_args through the schema registered for this
    // decoder type. error_rate_vec is a reserved compatibility alias: accept
    // it at any schema-recognized nesting level, promote it to model state,
    // and never leave it in the map a plugin receives.
    if (io.outputting()) {
      auto args_map = config.decoder_custom_args_to_heterogeneous_map();
      const auto rates = config.effective_error_rate_vec();
      if (!rates.empty())
        args_map.insert("error_rate_vec", rates);
      if (!args_map.empty()) {
        if (!schema) {
          // Match the historical emission behavior for unknown types.
          CUDA_QEC_WARN(
              "decoder_custom_args set for decoder type '{}' but no parameter "
              "schema is registered under that name; the args are omitted "
              "from the emitted YAML.",
              config.type);
        } else {
          cudaqx::heterogeneous_map emitted_args = std::move(args_map);
          schema_binding binding{&emitted_args, schema};
          io.mapRequired("decoder_custom_args", binding);
        }
      }
    } else if (schema) {
      bool args_present = false;
      for (const auto key : io.keys()) {
        if (key == "decoder_custom_args") {
          args_present = true;
          break;
        }
      }
      cudaqx::heterogeneous_map args_map;
      schema_binding binding{&args_map, schema};
      io.mapOptional("decoder_custom_args", binding);
      if (args_present) {
        cudaq::qec::decoding::config::decoder_config args_config;
        args_config.type = config.type;
        args_config.decoder_custom_args = args_map;
        auto normalized =
            cudaq::qec::decoding::config::normalize_decoder_args(args_config);
        if (in_document("error_rate_vec") && normalized.legacy_rates)
          throw std::runtime_error(
              "error_rate_vec is supplied both at the decoder configuration "
              "level and at '" +
              normalized.legacy_rates->path +
              "'. Supply model error rates in exactly one location.");
        if (normalized.legacy_rates)
          config.error_rate_vec = std::move(normalized.legacy_rates->values);
        args_map = std::move(normalized.plugin_args);
        cudaq::qec::decoding::config::finalize_parsed_args(
            *schema, args_map, "decoder_custom_args (" + config.type + ")");
      }
      config.decoder_custom_args = args_map;
    }

    // LLVM's YAML parser records a diagnostic and keeps going, so a malformed
    // document arrives here half-populated. Validate state only if error-free.
    if (io.error())
      return;

    // Chunk form when dem_chunks is set and H_sparse is empty. A nonempty
    // H_sparse makes the config flat — the matrix wins — which is also what
    // expand_dem_chunks() leaves behind after it runs.
    const bool chunk_form =
        config.dem_chunks.has_value() && config.H_sparse.empty();

    // DEM form when a Stim model file is named. The decoder is built from that
    // text and derives its own dimensions and observable mapping, so the
    // checks below that compare against syndrome_size do not apply to it; the
    // D row count is checked against the constructed decoder instead.
    const bool from_dem = !config.stim_dem_path.empty();

    // Both forms at once is legal but lopsided: H_sparse wins and the chunk
    // spec never runs. expand_dem_chunks() produces exactly this state, so it
    // cannot be an error, but a hand-written document that reaches it has lost
    // its dem_chunks without any other signal.
    if (!io.outputting() && config.dem_chunks.has_value() &&
        !config.H_sparse.empty())
      CUDA_QEC_WARN(
          "decoder {} sets both H_sparse and dem_chunks. The flat H_sparse "
          "matrix is what decoders are built from, and dem_chunks is ignored, "
          "including its num_rounds. This is expected for a config already "
          "expanded by expand_dem_chunks(); otherwise drop H_sparse to decode "
          "from dem_chunks.",
          config.id);

    // dem_chunks and stim_dem_path each describe the whole DEM on their own,
    // and unlike the flat form there is no precedence between them to fall
    // back on, so a document naming both has to be rejected outright.
    if (config.dem_chunks.has_value() && from_dem)
      throw std::runtime_error(
          "dem_chunks and stim_dem_path must not both be set for decoder " +
          std::to_string(config.id) +
          "; each describes the whole DEM on its own.");

    // Whichever form is in play derives these from the model it names;
    // accepting them here would let a config disagree with itself.
    const auto reject_derived = [&config](const char *key, bool present,
                                          const char *source) {
      if (present)
        throw std::runtime_error(
            std::string(key) + " must not be set for decoder " +
            std::to_string(config.id) + " because it is derived from " +
            source +
            ". Remove it, or describe the whole experiment with "
            "H_sparse instead of " +
            source + ".");
    };

    if (chunk_form) {
      if (config.dem_chunks->num_rounds.has_value() &&
          *config.dem_chunks->num_rounds < 2)
        throw std::runtime_error(
            "dem_chunks.num_rounds must be at least 2 for decoder " +
            std::to_string(config.id) + "; got " +
            std::to_string(*config.dem_chunks->num_rounds) + ".");
      reject_derived("block_size", config.block_size != 0, "dem_chunks");
      reject_derived("syndrome_size", config.syndrome_size != 0, "dem_chunks");
      reject_derived("O_sparse", !config.O_sparse.empty(), "dem_chunks");
      reject_derived("D_sparse", !config.D_sparse.empty(), "dem_chunks");
      reject_derived("error_rate_vec", !config.error_rate_vec.empty(),
                     "dem_chunks");
    } else if (from_dem) {
      reject_derived("H_sparse", !config.H_sparse.empty(), "stim_dem_path");
      reject_derived("O_sparse", !config.O_sparse.empty(), "stim_dem_path");
      reject_derived("error_rate_vec", !config.error_rate_vec.empty(),
                     "stim_dem_path");
      // D_sparse is the one model field the DEM text does not carry: it maps
      // detectors onto the syndrome bits this decoder is sent. Its row count is
      // checked against the constructed decoder's detector count, which is only
      // known once the model has been read.
      if (!io.outputting() && config.D_sparse.empty())
        throw std::runtime_error(
            "D_sparse is required for decoder " + std::to_string(config.id) +
            " alongside stim_dem_path, which describes the DEM but not the "
            "detectors this decoder is sent.");
    } else {
      // A flat document spells out its whole DEM, so every one of these fields
      // has to be there. They are mapOptional only because the other two forms
      // derive them; without this check a document that omits one parses into
      // a zero-sized DEM and fails much later, at decoder construction.
      if (!io.outputting()) {
        std::string missing;
        for (const char *name : {"block_size", "syndrome_size", "H_sparse",
                                 "O_sparse", "D_sparse"}) {
          if (!in_document(name)) {
            if (!missing.empty())
              missing += ", ";
            missing += name;
          }
        } // end - for(name)
        if (!missing.empty())
          throw std::runtime_error(
              "decoder " + std::to_string(config.id) +
              " is missing required field(s): " + missing +
              ". A flat DEM names block_size, syndrome_size, H_sparse, "
              "O_sparse and D_sparse together; use dem_chunks "
              "to describe the same DEM one round at a time, or stim_dem_path "
              "to name the model text itself, instead.");
      } // end - if(!io.outputting())

      if (config.H_sparse.empty() && config.syndrome_size > 0)
        throw std::runtime_error(
            "H_sparse is required for decoder " + std::to_string(config.id) +
            " unless dem_chunks is present, which describes the same DEM one "
            "round at a time instead.");
      if (config.block_size == 0 && !config.H_sparse.empty())
        throw std::runtime_error("block_size is required for decoder " +
                                 std::to_string(config.id));

      // Validate that the number of rows in the H_sparse vector is equal to
      // syndrome_size.
      auto num_H_rows =
          std::count(config.H_sparse.begin(), config.H_sparse.end(), -1);
      if (num_H_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in H_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_H_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
    }

    // Validate that no values in the H_sparse vector are out of range.
    for (auto value : config.H_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in H_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that no values in the O_sparse vector are out of range.
    for (auto value : config.O_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in O_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that if the D_sparse is provided, it is a valid D matrix. That
    // means that the number of rows in the D_sparse matrix should be equal to
    // the number of rows in the H_sparse matrix, and no row should be empty.
    if (!config.D_sparse.empty()) {
      auto num_D_rows =
          std::count(config.D_sparse.begin(), config.D_sparse.end(), -1);
      if (!from_dem && num_D_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in D_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_D_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
      // No row should be empty, including the first row.
      if (config.D_sparse.front() == -1)
        throw std::runtime_error("D_sparse row is empty for decoder " +
                                 std::to_string(config.id));
      for (std::size_t i = 0; i < config.D_sparse.size() - 1; ++i) {
        if (config.D_sparse.at(i) == -1 && config.D_sparse.at(i + 1) == -1) {
          throw std::runtime_error("D_sparse row is empty for decoder " +
                                   std::to_string(config.id));
        }
      }
    }
  }
};

// transport section mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::transport_shape_override> {
  static void
  mapping(IO &io,
          cudaq::qec::decoding::config::transport_shape_override &override_) {
    io.mapOptional("provider", override_.provider, std::string());
    io.mapOptional("args", override_.args);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::transport_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::transport_config &transport) {
    io.mapOptional("provider", transport.provider, std::string());
    io.mapOptional("args", transport.args);
    io.mapOptional("device_graph", transport.device_graph,
                   cudaq::qec::decoding::config::transport_shape_override());
  }
};

// multi_decoder_config mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::multi_decoder_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::multi_decoder_config &config) {
    io.mapRequired("decoders", config.decoders);
    io.mapOptional("transport", config.transport,
                   cudaq::qec::decoding::config::transport_config());
  }
};

} // namespace llvm::yaml

namespace {

// Run dem_chunks_spec's cross-phase checks after a successful parse, and
// present failures as std::runtime_error so every configuration error out of
// from_yaml_str() has the one type callers already catch.
void validate_parsed_dem_chunks(
    const cudaq::qec::decoding::config::decoder_config &config) {
  if (!config.dem_chunks)
    return;
  try {
    config.dem_chunks->validate();
  } catch (const std::invalid_argument &e) {
    throw std::runtime_error("Invalid dem_chunks for decoder " +
                             std::to_string(config.id) + ": " + e.what());
  }
}

} // namespace

// Static method to convert a YAML string to a multi_decoder_config.
cudaq::qec::decoding::config::multi_decoder_config
cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
    const std::string_view yaml_str) {
  multi_decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  for (const auto &decoder : config.decoders)
    validate_parsed_dem_chunks(decoder);
  return config;
}

std::string cudaq::qec::decoding::config::multi_decoder_config::to_yaml_str(
    int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

cudaq::qec::decoding::config::decoder_config
cudaq::qec::decoding::config::decoder_config::from_yaml_str(
    const std::string &yaml_str) {
  decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  validate_parsed_dem_chunks(config);
  return config;
}

std::string
cudaq::qec::decoding::config::decoder_config::to_yaml_str(int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

namespace cudaq::qec::decoding::config {

// ---------------------------------------------------------------------------
// JSON Schema export
//
// Translates the registered decoder parameter schemas plus the fixed
// decoder_config envelope (the fields MappingTraits<decoder_config> maps
// above) into a JSON Schema draft 2020-12 document, so standard tooling can
// validate user-provided configuration YAML offline. The document is a
// snapshot of what this installation can parse: it enumerates the schemas
// registered at call time, exactly as the runtime parser resolves them.
// ---------------------------------------------------------------------------

namespace {

// JSON-pointer token escaping for schema names used inside $ref paths.
std::string json_pointer_escape(const std::string &name) {
  std::string out;
  for (char c : name) {
    if (c == '~')
      out += "~0";
    else if (c == '/')
      out += "~1";
    else
      out += c;
  }
  return out;
}

std::string params_ref(const std::string &name) {
  return "#/$defs/decoder_params/" + json_pointer_escape(name);
}

llvm::json::Object json_schema_for_param(const param_spec &spec) {
  using k = param_kind;
  switch (spec.kind) {
  case k::boolean:
    return llvm::json::Object{{"type", "boolean"}};
  case k::int32:
    return llvm::json::Object{{"type", "integer"}};
  case k::uint64:
    return llvm::json::Object{{"type", "integer"}, {"minimum", 0}};
  case k::f64:
    return llvm::json::Object{{"type", "number"}};
  case k::string:
    return llvm::json::Object{{"type", "string"}};
  case k::f64_vec:
    return llvm::json::Object{
        {"type", "array"}, {"items", llvm::json::Object{{"type", "number"}}}};
  case k::f64_matrix:
    return llvm::json::Object{
        {"type", "array"},
        {"items", llvm::json::Object{
                      {"type", "array"},
                      {"items", llvm::json::Object{{"type", "number"}}}}}};
  case k::subschema:
    return llvm::json::Object{{"$ref", params_ref(spec.subschema)}};
  case k::discriminated:
    // The concrete shape is selected by the discriminator value; the
    // dispatch clauses emitted below refine this.
    return llvm::json::Object{{"type", "object"}};
  }
  return llvm::json::Object{};
}

llvm::json::Array registered_name_array(const std::vector<std::string> &names) {
  llvm::json::Array arr;
  for (const auto &name : names)
    arr.push_back(name);
  return arr;
}

llvm::json::Object
decoder_params_json_schema(const decoder_schema &schema,
                           const std::vector<std::string> &all_names) {
  // Framework-owned compatibility alias. Runtime normalization removes it
  // before the map reaches the decoder plugin.
  llvm::json::Object properties{
      {"error_rate_vec",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "number"}}}}}};
  llvm::json::Array required;
  llvm::json::Array all_of;
  for (const auto &spec : schema.params) {
    properties[spec.key] = json_schema_for_param(spec);
    if (spec.required)
      required.push_back(spec.key);
    if (spec.kind == param_kind::discriminated) {
      // When the section is present, its discriminator must be present and
      // name a registered schema (mirrors the parser's checks).
      all_of.push_back(llvm::json::Object{
          {"if", llvm::json::Object{{"required", llvm::json::Array{spec.key}}}},
          {"then",
           llvm::json::Object{
               {"required", llvm::json::Array{spec.discriminator}},
               {"properties",
                llvm::json::Object{
                    {spec.discriminator,
                     llvm::json::Object{
                         {"enum", registered_name_array(all_names)}}}}}}}});
      // Each candidate discriminator value selects that schema for the
      // section.
      for (const auto &name : all_names)
        all_of.push_back(llvm::json::Object{
            {"if",
             llvm::json::Object{
                 {"properties",
                  llvm::json::Object{{spec.discriminator,
                                      llvm::json::Object{{"const", name}}}}},
                 {"required",
                  llvm::json::Array{spec.discriminator, spec.key}}}},
            {"then", llvm::json::Object{
                         {"properties",
                          llvm::json::Object{
                              {spec.key, llvm::json::Object{
                                             {"$ref", params_ref(name)}}}}}}}});
    }
  }
  llvm::json::Object out{{"type", "object"},
                         {"additionalProperties", false},
                         {"properties", std::move(properties)}};
  if (!required.empty())
    out["required"] = std::move(required);
  if (!all_of.empty())
    out["allOf"] = std::move(all_of);
  return out;
}

} // namespace

std::string decoder_config_json_schema() {
  const auto names = registered_decoder_schema_names();

  llvm::json::Object decoder_params;
  for (const auto &name : names)
    decoder_params[name] =
        decoder_params_json_schema(*find_decoder_schema(name), names);

  // The fixed decoder_config envelope; keep in sync with
  // MappingTraits<decoder_config> above.
  llvm::json::Object config_properties{
      {"id", llvm::json::Object{{"type", "integer"}}},
      {"type", llvm::json::Object{{"type", "string"}}},
      {"dispatch",
       llvm::json::Object{{"enum", llvm::json::Array{"host", "device_graph"}}}},
      {"cuda_device_id",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"block_size", llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"syndrome_size",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"stim_dem_path", llvm::json::Object{{"type", "string"}}},
      {"H_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"O_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"D_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"error_rate_vec",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "number"}}}}},
      {"dem_chunks", llvm::json::Object{{"$ref", "#/$defs/dem_chunks"}}},
      {"decoder_custom_args", llvm::json::Object{{"type", "object"}}},
  };

  // Per-type dispatch of decoder_custom_args, generated from the registry:
  // a registered type's args follow its schema; a type with no registered
  // schema accepts no args (the parser rejects the section outright).
  llvm::json::Array dispatch;
  // Direct duplicate spelling is representable in JSON Schema. Duplicates in
  // schema-recognized nested parameter sections are rejected by the runtime
  // normalization walk.
  llvm::json::Array duplicate_rate_locations;
  duplicate_rate_locations.push_back(llvm::json::Object{
      {"required",
       llvm::json::Array{"error_rate_vec", "decoder_custom_args"}}});
  llvm::json::Object custom_args_with_rates{
      {"required", llvm::json::Array{"error_rate_vec"}}};
  llvm::json::Object custom_args_constraint{
      {"properties", llvm::json::Object{{"decoder_custom_args",
                                         std::move(custom_args_with_rates)}}}};
  duplicate_rate_locations.push_back(std::move(custom_args_constraint));
  dispatch.push_back(llvm::json::Object{
      {"not",
       llvm::json::Object{{"allOf", std::move(duplicate_rate_locations)}}}});
  for (const auto &name : names)
    dispatch.push_back(llvm::json::Object{
        {"if", llvm::json::Object{{"properties",
                                   llvm::json::Object{
                                       {"type",
                                        llvm::json::Object{{"const", name}}}}},
                                  {"required", llvm::json::Array{"type"}}}},
        {"then", llvm::json::Object{
                     {"properties", llvm::json::Object{
                                        {"decoder_custom_args",
                                         llvm::json::Object{
                                             {"$ref", params_ref(name)}}}}}}}});
  dispatch.push_back(llvm::json::Object{
      {"if",
       llvm::json::Object{
           {"properties",
            llvm::json::Object{
                {"type",
                 llvm::json::Object{
                     {"not", llvm::json::Object{{"enum", registered_name_array(
                                                             names)}}}}}}},
           {"required", llvm::json::Array{"type"}}}},
      {"then",
       llvm::json::Object{
           {"properties",
            llvm::json::Object{{"decoder_custom_args",
                                llvm::json::Object{{"maxProperties", 0}}}}}}}});

  // The cross-phase rules dem_chunks_spec::validate() enforces (empty init
  // incoming seam, empty final outgoing seam, equal seam widths, one error rate
  // per fault) are not expressible here, so a document that passes this schema
  // may still be rejected when parsed.
  llvm::json::Object dem_chunk_properties{
      {"num_faults", llvm::json::Object{{"type", "integer"}, {"minimum", 1}}},
      {"H_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"O_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"error_rates",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "number"},
                                                       {"minimum", 0},
                                                       {"maximum", 1}}}}},
  };

  llvm::json::Object defs{
      {"sparse_matrix",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "integer"},
                                                       {"minimum", -1}}}}},
      // Server-level transport section (see transport_config /
      // transport_shape_override); the wire is deployment config and lives
      // outside the decoders list.
      {"transport_shape_override",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"provider", llvm::json::Object{{"type", "string"}}},
                {"args",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}}}}}},
           {"additionalProperties", false}}},
      {"transport_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"provider", llvm::json::Object{{"type", "string"}}},
                {"args",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}}}},
                {"device_graph",
                 llvm::json::Object{
                     {"$ref", "#/$defs/transport_shape_override"}}}}},
           {"additionalProperties", false}}},
      {"dem_chunk",
       llvm::json::Object{
           {"type", "object"},
           {"properties", std::move(dem_chunk_properties)},
           {"required", llvm::json::Array{"num_faults", "error_rates"}},
           {"additionalProperties", false}}},
      {"dem_chunks",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"seam", llvm::json::Object{{"type", "object"}}},
                {"connections", llvm::json::Object{{"type", "array"}}},
                {"num_rounds",
                 llvm::json::Object{{"type", "integer"}, {"minimum", 2}}},
                {"phases", llvm::json::Object{{"type", "array"}}}}},
           // num_rounds is absent from a streaming configuration, whose round
           // count is only known once the experiment runs, so the parser
           // accepts it as optional and expand_dem_chunks() is what demands it.
           {"required", llvm::json::Array{"seam", "connections", "phases"}},
           {"additionalProperties", false}}},
      {"decoder_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties", std::move(config_properties)},
           {"required", llvm::json::Array{"id", "type"}},
           // The DEM is described flat, as repeated phases, or as a nonempty
           // Stim model path. Expanded chunk configurations retain dem_chunks
           // beside their flat fields, so that combination belongs to the
           // flat branch. The other derived-field exclusions mirror the YAML
           // parser exactly.
           {"oneOf",
            llvm::json::Array{
                llvm::json::Object{
                    {"required", llvm::json::Array{"H_sparse", "block_size",
                                                   "syndrome_size", "O_sparse",
                                                   "D_sparse"}},
                    {"properties",
                     llvm::json::Object{
                         {"stim_dem_path",
                          llvm::json::Object{{"maxLength", 0}}}}}},
                llvm::json::Object{
                    {"required", llvm::json::Array{"dem_chunks"}},
                    {"properties", llvm::json::Object{{"stim_dem_path",
                                                       llvm::json::Object{
                                                           {"maxLength", 0}}}}},
                    {"not",
                     llvm::json::Object{
                         {"anyOf",
                          llvm::json::Array{
                              llvm::json::Object{
                                  {"required",
                                   llvm::json::Array{"block_size"}}},
                              llvm::json::Object{
                                  {"required",
                                   llvm::json::Array{"syndrome_size"}}},
                              llvm::json::Object{
                                  {"required", llvm::json::Array{"H_sparse"}}},
                              llvm::json::Object{
                                  {"required", llvm::json::Array{"O_sparse"}}},
                              llvm::json::Object{
                                  {"required", llvm::json::Array{"D_sparse"}}},
                              llvm::json::Object{
                                  {"required",
                                   llvm::json::Array{"error_rate_vec"}}}}}}}},
                llvm::json::Object{
                    {"required",
                     llvm::json::Array{"stim_dem_path", "D_sparse"}},
                    {"properties", llvm::json::Object{{"stim_dem_path",
                                                       llvm::json::Object{
                                                           {"minLength", 1}}}}},
                    {"not",
                     llvm::json::Object{
                         {"anyOf",
                          llvm::json::Array{
                              llvm::json::Object{
                                  {"required", llvm::json::Array{"H_sparse"}}},
                              llvm::json::Object{
                                  {"required", llvm::json::Array{"O_sparse"}}},
                              llvm::json::Object{
                                  {"required",
                                   llvm::json::Array{"error_rate_vec"}}},
                              llvm::json::Object{
                                  {"required",
                                   llvm::json::Array{"dem_chunks"}}}}}}}}}},
           {"additionalProperties", false},
           {"allOf", std::move(dispatch)}}},
      {"decoder_params", std::move(decoder_params)},
  };

  llvm::json::Object root{
      {"$schema", "https://json-schema.org/draft/2020-12/schema"},
      {"title", "CUDA-Q QEC realtime decoding configuration"},
      {"description",
       "Validates multi_decoder_config YAML documents. Generated from the "
       "decoder parameter schemas registered in this installation, so it "
       "reflects the decoder plugins loaded at generation time. Per-schema "
       "validate hooks (arbitrary cross-field checks) are not representable "
       "in JSON Schema; a document that passes may still be rejected when "
       "parsed."},
      {"type", "object"},
      {"properties",
       llvm::json::Object{
           {"decoders",
            llvm::json::Object{
                {"type", "array"},
                {"items",
                 llvm::json::Object{{"$ref", "#/$defs/decoder_config"}}}}},
           {"transport",
            llvm::json::Object{{"$ref", "#/$defs/transport_config"}}}}},
      {"required", llvm::json::Array{"decoders"}},
      {"additionalProperties", false},
      {"$defs", std::move(defs)},
  };

  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::OStream json_out(os, /*IndentSize=*/2);
  json_out.value(llvm::json::Value(std::move(root)));
  return out;
}

// Stash a copy for consumers that build their own decoder instances from the
// process-wide configuration -- the decoding-server DeviceCallService plugin
// reads it when CUDAQ_QEC_DECODER_CONFIG is not set (in-process path).
// shared_ptr + mutex: the plugin reads this from the realtime dispatcher
// thread while the application thread may call configure_decoders() again;
// shared ownership keeps the reader's config alive across a concurrent
// replacement.
static std::mutex g_last_multi_decoder_config_mutex;
static std::shared_ptr<const multi_decoder_config> g_last_multi_decoder_config;

int configure_decoders(multi_decoder_config &config,
                       const std::filesystem::path &base_dir) {
  CUDA_QEC_INFO("Initializing realtime decoding library with config object");
  const int status =
      cudaq::qec::decoding::host::configure_decoders(config, base_dir);
  if (status != 0)
    return status;

  // Stash and publish only once the configuration is actually in effect, so a
  // failed application cannot leave a configuration cached here or advertised
  // to remote targets that nothing is honoring.
  {
    std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
    g_last_multi_decoder_config =
        std::make_shared<const multi_decoder_config>(config);
  }
  // The cudaq integration (ExtraPayloadProvider) is installed by cudaq-qec at
  // load time; this call is a no-op when cudaq-qec is not loaded, keeping this
  // library free of any direct cudaq-common dependency.
  cudaq::qec::publish_decoder_config_payload(config.to_yaml_str());
  return status;
}

int configure_decoders(multi_decoder_config &config) {
  // No originating file: relative model paths resolve against the working
  // directory as it stands when resolution starts.
  return configure_decoders(config, std::filesystem::current_path());
}

std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config() {
  std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
  return g_last_multi_decoder_config;
}

void log_config(const char *config_str, bool from_file) {
  const bool dump_config = []() {
    if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
      if (ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y')
        return true;
    return false;
  }();

  if (dump_config) {
    if (cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info)) {
      CUDA_QEC_INFO(
          "Initializing realtime decoding library with config string: {}",
          config_str);
    } else {
      printf("Initializing realtime decoding library with config string: %s\n",
             config_str);
    }
  }
}

int configure_decoders_from_file(const char *config_file) {
  std::string config_file_str(config_file);
  CUDA_QEC_INFO("Initializing realtime decoding library with config file: {}",
                config_file_str);

  // Verify that the file exists.
  if (!std::filesystem::exists(config_file_str)) {
    CUDA_QEC_WARN("Config file does not exist: {}", config_file_str);
    return 1;
  }

  // Read the config file into a string.
  std::string config_str;
  std::ifstream config_file_stream(config_file_str);
  config_str = std::string(std::istreambuf_iterator<char>(config_file_stream),
                           std::istreambuf_iterator<char>());
  log_config(config_str.c_str(), /*from_file=*/true);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  // Relative model paths resolve against the configuration file's directory,
  // absolute so the resolved paths stay valid if the working directory moves.
  return configure_decoders(
      config, std::filesystem::absolute(config_file_str).parent_path());
}

int configure_decoders_from_str(const char *config_str) {
  CUDA_QEC_INFO(
      "Initializing realtime decoding library with raw config string");
  log_config(config_str, /*from_file=*/false);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

void finalize_decoders() { cudaq::qec::decoding::host::finalize_decoders(); }

} // namespace cudaq::qec::decoding::config
