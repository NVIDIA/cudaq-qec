/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder_config_schema.h"
#include "cuda-qx/core/tuple_utils.h"
#include "cuda-qx/core/type_traits.h"
#include "cudaq/qec/logger.h"
#include <any>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace cudaq::qec::decoding::config {

namespace {

// Node-based map: pointers returned by find_decoder_schema stay valid across
// later registrations. Schemas are never erased or unregistered -- including
// when the plugin that registered one is dlclose'd (cleanup_decoder_plugins
// runs from a library destructor at exit). The registry is intentionally
// leaked, like get_plugin_handles() and INSTANTIATE_REGISTRY: destroying it
// at static-destruction time could run std::function (validate hook)
// destructors whose code lives in an already-unloaded plugin.
std::map<std::string, decoder_schema> &schema_registry() {
  static auto *registry = new std::map<std::string, decoder_schema>();
  return *registry;
}

std::mutex &schema_registry_mutex() {
  static auto *m = new std::mutex();
  return *m;
}

} // namespace

void register_decoder_schema(decoder_schema schema) {
  std::lock_guard<std::mutex> lock(schema_registry_mutex());
  auto name = schema.name;
  schema_registry().insert_or_assign(std::move(name), std::move(schema));
}

const decoder_schema *find_decoder_schema(const std::string &name) {
  std::lock_guard<std::mutex> lock(schema_registry_mutex());
  auto &registry = schema_registry();
  auto iter = registry.find(name);
  if (iter == registry.end())
    return nullptr;
  return &iter->second;
}

std::vector<std::string> registered_decoder_schema_names() {
  std::lock_guard<std::mutex> lock(schema_registry_mutex());
  std::vector<std::string> names;
  for (const auto &[name, schema] : schema_registry())
    names.push_back(name);
  return names;
}

// ---------------------------------------------------------------------------
// Schema validation of programmatically built custom-args maps
// ---------------------------------------------------------------------------

namespace {

// True when `value` would be readable as T by heterogeneous_map::get<T> --
// i.e. it holds T or one of T's RelatedTypesMap entries (the same tuple
// get<T> iterates). Pointer-based probing so validation never copies a value.
template <typename T>
bool value_readable_as(const std::any &value) {
  if (std::any_cast<T>(&value))
    return true;
  bool readable = false;
  cudaqx::tuple_for_each(
      typename cudaqx::RelatedTypesMap<T>::types(), [&](auto &&el) {
        readable = readable ||
                   std::any_cast<
                       std::remove_cv_t<std::remove_reference_t<decltype(el)>>>(
                       &value);
      });
  return readable;
}

// True when `value` would be readable as `kind`'s canonical storage type.
bool value_matches_kind(param_kind kind, const std::any &value) {
  switch (kind) {
  case param_kind::boolean:
    return value_readable_as<bool>(value);
  case param_kind::int32:
    return value_readable_as<int>(value);
  case param_kind::uint64:
    return value_readable_as<std::size_t>(value);
  case param_kind::f64:
    return value_readable_as<double>(value);
  case param_kind::string:
    return value_readable_as<std::string>(value);
  case param_kind::f64_vec:
    return value_readable_as<std::vector<double>>(value);
  case param_kind::f64_matrix:
    return value_readable_as<std::vector<std::vector<double>>>(value);
  case param_kind::subschema:
  case param_kind::discriminated:
    return value_readable_as<cudaqx::heterogeneous_map>(value);
  }
  return false;
}

const char *kind_description(param_kind kind) {
  switch (kind) {
  case param_kind::boolean:
    return "boolean";
  case param_kind::int32:
    return "32-bit int";
  case param_kind::uint64:
    return "non-negative int";
  case param_kind::f64:
    return "float";
  case param_kind::string:
    return "string";
  case param_kind::f64_vec:
    return "list-of-float";
  case param_kind::f64_matrix:
    return "list-of-list-of-float";
  case param_kind::subschema:
  case param_kind::discriminated:
    return "mapping";
  }
  return "unknown";
}

} // namespace

void validate_custom_args(const decoder_schema &schema,
                          const cudaqx::heterogeneous_map &args,
                          const std::string &context) {
  for (const auto &kv : args) {
    const std::string &key = kv.first;
    const param_spec *spec = nullptr;
    for (const auto &candidate : schema.params) {
      if (candidate.key == key) {
        spec = &candidate;
        break;
      }
    }
    if (!spec)
      throw std::runtime_error("Unknown key '" + key + "' in " + context + ".");

    // Every value must be readable as its kind's canonical storage type;
    // this is what YAML emission and decoder construction will do with it,
    // so a map that validates is guaranteed to serialize (round-trip
    // invariant). Values that arrived through the YAML parser or the typed
    // Python setter always pass; this catches maps built by other means
    // (e.g. a Python dict assigned before `type`, stored generically).
    if (!value_matches_kind(spec->kind, kv.second))
      throw std::runtime_error(
          "Key '" + key + "' in " + context + " does not hold a " +
          kind_description(spec->kind) + " value as its schema kind requires.");

    if (spec->kind == param_kind::subschema) {
      const auto *nested_schema = find_decoder_schema(spec->subschema);
      if (!nested_schema)
        throw std::runtime_error("No schema registered under '" +
                                 spec->subschema + "' (needed to validate '" +
                                 key + "').");
      validate_custom_args(
          *nested_schema, *std::any_cast<cudaqx::heterogeneous_map>(&kv.second),
          context + "." + key);
    } else if (spec->kind == param_kind::discriminated) {
      std::string discriminator_value;
      if (args.contains(spec->discriminator))
        discriminator_value = args.get<std::string>(spec->discriminator);
      if (discriminator_value.empty())
        throw std::runtime_error("'" + key + "' is present but '" +
                                 spec->discriminator + "' is not set in " +
                                 context + ".");
      const auto *nested_schema = find_decoder_schema(discriminator_value);
      if (!nested_schema)
        throw std::runtime_error(
            "'" + key + "' does not support " + spec->discriminator + " '" +
            discriminator_value +
            "': no parameter schema is registered under that name.");
      validate_custom_args(
          *nested_schema, *std::any_cast<cudaqx::heterogeneous_map>(&kv.second),
          context + "." + key);
    }
  }
  for (const auto &spec : schema.params)
    if (spec.required && !args.contains(spec.key))
      throw std::runtime_error("Missing required key '" + spec.key + "' in " +
                               context + ".");
  if (schema.validate)
    schema.validate(args);
}

void validate_custom_args(const std::string &schema_name,
                          const cudaqx::heterogeneous_map &args) {
  const auto *schema = find_decoder_schema(schema_name);
  if (!schema) {
    if (args.empty())
      return;
    throw std::runtime_error(
        "Decoder type '" + schema_name +
        "' has no registered parameter schema; its decoder_custom_args "
        "cannot be validated (or serialized to YAML). Register a schema from "
        "the decoder's plugin library with register_decoder_schema().");
  }
  validate_custom_args(*schema, args, "'" + schema_name + "' parameters");
}

void materialize_default_args(const decoder_schema &schema,
                              cudaqx::heterogeneous_map &args) {
  for (const auto &spec : schema.params) {
    const decoder_schema *nested_schema = nullptr;
    if (spec.kind == param_kind::discriminated) {
      std::string discriminator_value;
      if (args.contains(spec.discriminator))
        discriminator_value = args.get<std::string>(spec.discriminator);
      nested_schema = discriminator_value.empty()
                          ? nullptr
                          : find_decoder_schema(discriminator_value);
      if (spec.materialize_empty && nested_schema && !args.contains(spec.key))
        args.insert(spec.key, cudaqx::heterogeneous_map());
    } else if (spec.kind == param_kind::subschema) {
      nested_schema = find_decoder_schema(spec.subschema);
    }
    if (nested_schema && args.contains(spec.key)) {
      auto nested = args.get<cudaqx::heterogeneous_map>(spec.key);
      materialize_default_args(*nested_schema, nested);
      args.insert(spec.key, nested);
    }
  }
}

void drop_non_schema_keys(const decoder_schema &schema,
                          cudaqx::heterogeneous_map &args) {
  // Single pass: recurse into nested sections in place (no copies), and note
  // whether any key at this level is unknown. Only when one is does the map
  // get rebuilt -- the common all-keys-known case leaves `args` untouched.
  bool has_unknown = false;
  for (auto &kv : args) {
    const param_spec *spec = nullptr;
    for (const auto &candidate : schema.params) {
      if (candidate.key == kv.first) {
        spec = &candidate;
        break;
      }
    }
    if (!spec) {
      CUDA_QEC_WARN("Key '{}' is not in the '{}' parameter schema; it is "
                    "excluded from the decoder parameters and from emitted "
                    "YAML.",
                    kv.first, schema.name);
      has_unknown = true;
      continue;
    }
    const decoder_schema *nested_schema = nullptr;
    if (spec->kind == param_kind::subschema) {
      nested_schema = find_decoder_schema(spec->subschema);
    } else if (spec->kind == param_kind::discriminated &&
               args.contains(spec->discriminator)) {
      nested_schema =
          find_decoder_schema(args.get<std::string>(spec->discriminator));
    }
    if (nested_schema)
      if (auto *nested = std::any_cast<cudaqx::heterogeneous_map>(&kv.second))
        drop_non_schema_keys(*nested_schema, *nested);
  }
  if (!has_unknown)
    return;
  cudaqx::heterogeneous_map kept;
  for (const auto &kv : args) {
    for (const auto &spec : schema.params) {
      if (spec.key == kv.first) {
        kept.insert(kv.first, kv.second);
        break;
      }
    }
  }
  args = std::move(kept);
}

// ---------------------------------------------------------------------------
// Deep equality over canonical custom-args value kinds
// ---------------------------------------------------------------------------

namespace {

// Sign-aware integer representation so unsigned values above 2^63 never
// alias negative signed values (a plain long long cast would wrap
// size_t(2^64-1) to -1 and report it equal to int(-1)).
struct integer_value {
  bool negative = false;
  unsigned long long magnitude = 0;
  bool operator==(const integer_value &other) const {
    return negative == other.negative && magnitude == other.magnitude;
  }
  double as_double() const {
    double d = static_cast<double>(magnitude);
    return negative ? -d : d;
  }
};

std::optional<integer_value> as_integer(const std::any &v) {
  auto from_signed = [](long long s) {
    integer_value iv;
    iv.negative = s < 0;
    iv.magnitude = iv.negative ? ~static_cast<unsigned long long>(s) + 1ULL
                               : static_cast<unsigned long long>(s);
    return iv;
  };
  auto from_unsigned = [](unsigned long long u) {
    return integer_value{false, u};
  };
  if (auto *p = std::any_cast<int>(&v))
    return from_signed(*p);
  if (auto *p = std::any_cast<long>(&v))
    return from_signed(*p);
  if (auto *p = std::any_cast<long long>(&v))
    return from_signed(*p);
  if (auto *p = std::any_cast<short>(&v))
    return from_signed(*p);
  if (auto *p = std::any_cast<unsigned int>(&v))
    return from_unsigned(*p);
  if (auto *p = std::any_cast<unsigned long>(&v))
    return from_unsigned(*p);
  if (auto *p = std::any_cast<unsigned long long>(&v))
    return from_unsigned(*p);
  if (auto *p = std::any_cast<unsigned short>(&v))
    return from_unsigned(*p);
  return std::nullopt;
}

std::optional<double> as_floating(const std::any &v) {
  if (auto *p = std::any_cast<double>(&v))
    return *p;
  if (auto *p = std::any_cast<float>(&v))
    return static_cast<double>(*p);
  return std::nullopt;
}

bool any_values_equal(const std::any &a, const std::any &b) {
  // bool is kept distinct from the integer family so `true` never silently
  // matches `1` from a differently-built configuration.
  if (auto *pa = std::any_cast<bool>(&a)) {
    auto *pb = std::any_cast<bool>(&b);
    return pb && *pa == *pb;
  }
  if (std::any_cast<bool>(&b))
    return false;

  if (auto ia = as_integer(a)) {
    if (auto ib = as_integer(b))
      return *ia == *ib;
    if (auto fb = as_floating(b))
      return ia->as_double() == *fb;
    return false;
  }
  if (auto fa = as_floating(a)) {
    if (auto fb = as_floating(b))
      return *fa == *fb;
    if (auto ib = as_integer(b))
      return *fa == ib->as_double();
    return false;
  }
  if (auto *pa = std::any_cast<std::string>(&a)) {
    auto *pb = std::any_cast<std::string>(&b);
    return pb && *pa == *pb;
  }
  if (auto *pa = std::any_cast<std::vector<double>>(&a)) {
    auto *pb = std::any_cast<std::vector<double>>(&b);
    return pb && *pa == *pb;
  }
  if (auto *pa = std::any_cast<std::vector<std::vector<double>>>(&a)) {
    auto *pb = std::any_cast<std::vector<std::vector<double>>>(&b);
    return pb && *pa == *pb;
  }
  if (auto *pa = std::any_cast<cudaqx::heterogeneous_map>(&a)) {
    auto *pb = std::any_cast<cudaqx::heterogeneous_map>(&b);
    return pb && custom_args_maps_equal(*pa, *pb);
  }
  // Unknown value type: conservatively unequal.
  return false;
}

} // namespace

bool custom_args_maps_equal(const cudaqx::heterogeneous_map &a,
                            const cudaqx::heterogeneous_map &b) {
  if (a.size() != b.size())
    return false;

  for (const auto &[key_a, val_a] : a) {
    bool found = false;
    for (const auto &[key_b, val_b] : b) {
      if (key_a != key_b)
        continue;
      found = true;
      if (!any_values_equal(val_a, val_b))
        return false;
      break;
    }
    if (!found)
      return false;
  }
  return true;
}

// Decoder schemas are registered by the shared library that ships the
// decoder: lut.cpp and sliding_window.cpp register theirs in this library,
// and the decoder plugins (pymatching, chromobius, trt_decoder,
// nv-qldpc-decoder, ...) register theirs from their own .so. Third-party
// decoders register their schema from their own plugin library instead of
// editing this file.

} // namespace cudaq::qec::decoding::config
