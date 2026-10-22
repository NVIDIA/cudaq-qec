/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the YAML forms of a window request and their llvm::yaml
// traits, shared by the config parser, which reads an ingest node's
// `windows`, and the standalone parser in window_request_yaml.h, so both
// apply the same rules. Reserved keys are rejected by name, so files written
// for spatial windows fail with a message rather than an unknown-key error.
// Included only where llvm::yaml is linked.

#pragma once

#include "llvm/Support/YAMLTraits.h"
#include "cudaq/qec/window_request.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::window_yaml {

using namespace cudaq::qec::experimental;

/// Throws std::invalid_argument for a malformed request.
[[noreturn]] void reject(const std::string &what);

/// The map form of a round range, {rounds: [b, e]}; the list form [b, e]
/// fills the same rounds.
struct range_map_form {
  std::vector<std::int64_t> rounds;
};

/// A round range in either form. A scalar is kept only to be rejected.
struct range_form {
  range_map_form map;
  std::string scalar;
};

struct zones_form {
  std::optional<range_form> commit;
  std::vector<range_form> buffers;
};

struct input_form {
  std::optional<port_disposition> disposition;
  std::optional<input_boundary> boundary;
  std::optional<double> prior;
};

struct output_form {
  std::optional<port_disposition> disposition;
  std::optional<range_form> imports;
};

struct ports_form {
  input_form input;
  output_form output;
};

struct send_form {
  std::string id;
  std::optional<send_mode> mode;
  std::optional<range_form> rounds;
};

struct columns_form {
  bool merge_identical = false;
  parallel_resolution conflicting_parallel = parallel_resolution::reject;
  bool operator==(const columns_form &) const = default;
};

/// One window request, as YAML writes it. The required keys are optional
/// here so that a missing one is told apart from its default.
struct request_form {
  std::optional<std::int64_t> period;
  std::int64_t offset = 0;
  std::optional<zones_form> zones;
  std::optional<ports_form> ports;
  std::vector<send_form> sends;
  columns_form columns;
};

/// The request a form describes.
/// @throws std::invalid_argument for a missing period, zones or ports, or a
/// malformed range.
window_request request_from_form(const request_form &form);

/// The form that writes @p request, every range in list form.
/// @throws std::invalid_argument for a send with more than one round range.
request_form form_from_request(const window_request &request);

} // namespace cudaq::qec::window_yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::window_yaml::range_form)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::window_yaml::send_form)

namespace llvm::yaml {

template <>
struct ScalarEnumerationTraits<cudaq::qec::experimental::port_disposition> {
  static void enumeration(IO &io,
                          cudaq::qec::experimental::port_disposition &value) {
    using cudaq::qec::experimental::port_disposition;
    io.enumCase(value, "open", port_disposition::open);
    io.enumCase(value, "closed", port_disposition::closed);
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::experimental::input_boundary> {
  static void enumeration(IO &io,
                          cudaq::qec::experimental::input_boundary &value) {
    using cudaq::qec::experimental::input_boundary;
    io.enumCase(value, "columns", input_boundary::columns);
    io.enumCase(value, "truncate", input_boundary::truncate);
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::experimental::send_mode> {
  static void enumeration(IO &io, cudaq::qec::experimental::send_mode &value) {
    using cudaq::qec::experimental::send_mode;
    io.enumCase(value, "delta", send_mode::delta);
    io.enumCase(value, "residual", send_mode::residual);
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::experimental::parallel_resolution> {
  static void
  enumeration(IO &io, cudaq::qec::experimental::parallel_resolution &value) {
    using cudaq::qec::experimental::parallel_resolution;
    io.enumCase(value, "reject", parallel_resolution::reject);
    io.enumCase(value, "most_likely", parallel_resolution::most_likely);
  }
};

/// {rounds: [b, e]}, with occurrences and block reserved.
template <>
struct CustomMappingTraits<cudaq::qec::window_yaml::range_map_form> {
  static void inputOne(IO &io, StringRef key,
                       cudaq::qec::window_yaml::range_map_form &form) {
    if (key == "rounds") {
      io.mapRequired("rounds", form.rounds);
      return;
    }
    if (key == "occurrences" || key == "block")
      cudaq::qec::window_yaml::reject(
          "'" + key.str() +
          "' is reserved for spatial windows, which are not supported yet");
    cudaq::qec::window_yaml::reject("unknown key '" + key.str() +
                                    "' in a round range");
  }
  static void output(IO &io, cudaq::qec::window_yaml::range_map_form &form) {
    io.mapRequired("rounds", form.rounds);
  }
};

/// [b, e] or {rounds: [b, e]}; written as [b, e].
template <>
struct PolymorphicTraits<cudaq::qec::window_yaml::range_form> {
  static NodeKind getKind(const cudaq::qec::window_yaml::range_form &) {
    return NodeKind::Sequence;
  }
  static std::string &getAsScalar(cudaq::qec::window_yaml::range_form &form) {
    return form.scalar;
  }
  static cudaq::qec::window_yaml::range_map_form &
  getAsMap(cudaq::qec::window_yaml::range_form &form) {
    return form.map;
  }
  static std::vector<std::int64_t> &
  getAsSequence(cudaq::qec::window_yaml::range_form &form) {
    return form.map.rounds;
  }
};

template <>
struct MappingTraits<cudaq::qec::window_yaml::zones_form> {
  static void mapping(IO &io, cudaq::qec::window_yaml::zones_form &form) {
    io.mapOptional("commit", form.commit);
    io.mapOptional("buffers", form.buffers);
  }
};

template <>
struct MappingTraits<cudaq::qec::window_yaml::input_form> {
  static void mapping(IO &io, cudaq::qec::window_yaml::input_form &form) {
    io.mapOptional("disposition", form.disposition);
    io.mapOptional("boundary", form.boundary);
    io.mapOptional("prior", form.prior);
  }
};

template <>
struct MappingTraits<cudaq::qec::window_yaml::output_form> {
  static void mapping(IO &io, cudaq::qec::window_yaml::output_form &form) {
    io.mapOptional("disposition", form.disposition);
    io.mapOptional("imports", form.imports);
  }
};

/// Only the temporal ports; spatial ones are reserved.
template <>
struct CustomMappingTraits<cudaq::qec::window_yaml::ports_form> {
  static void inputOne(IO &io, StringRef key,
                       cudaq::qec::window_yaml::ports_form &form) {
    if (key == "input") {
      io.mapRequired("input", form.input);
      return;
    }
    if (key == "output") {
      io.mapRequired("output", form.output);
      return;
    }
    cudaq::qec::window_yaml::reject(
        "port '" + key.str() +
        "': spatial ports are not supported yet; use input and output");
  }
  static void output(IO &io, cudaq::qec::window_yaml::ports_form &form) {
    io.mapRequired("input", form.input);
    io.mapRequired("output", form.output);
  }
};

template <>
struct MappingTraits<cudaq::qec::window_yaml::send_form> {
  static void mapping(IO &io, cudaq::qec::window_yaml::send_form &form) {
    io.mapRequired("id", form.id);
    io.mapOptional("mode", form.mode);
    io.mapOptional("rounds", form.rounds);
  }
};

template <>
struct MappingTraits<cudaq::qec::window_yaml::columns_form> {
  static void mapping(IO &io, cudaq::qec::window_yaml::columns_form &form) {
    using cudaq::qec::experimental::parallel_resolution;
    io.mapOptional("merge_identical", form.merge_identical, false);
    io.mapOptional("conflicting_parallel", form.conflicting_parallel,
                   parallel_resolution::reject);
  }
};

/// The window keys, with responsibility and soundness_class reserved.
template <>
struct CustomMappingTraits<cudaq::qec::window_yaml::request_form> {
  static void inputOne(IO &io, StringRef key,
                       cudaq::qec::window_yaml::request_form &form) {
    if (key == "period")
      io.mapRequired("period", form.period.emplace());
    else if (key == "offset")
      io.mapRequired("offset", form.offset);
    else if (key == "zones")
      io.mapRequired("zones", form.zones.emplace());
    else if (key == "ports")
      io.mapRequired("ports", form.ports.emplace());
    else if (key == "sends")
      io.mapRequired("sends", form.sends);
    else if (key == "columns")
      io.mapRequired("columns", form.columns);
    else if (key == "responsibility" || key == "soundness_class")
      cudaq::qec::window_yaml::reject(
          "'" + key.str() +
          "' is reserved for spatial windows, which are not supported yet");
    else
      cudaq::qec::window_yaml::reject("unknown key '" + key.str() + "'");
  }
  static void output(IO &io, cudaq::qec::window_yaml::request_form &form) {
    io.mapRequired("period", form.period.value());
    io.mapOptional("offset", form.offset, std::int64_t{0});
    io.mapRequired("zones", form.zones.value());
    io.mapRequired("ports", form.ports.value());
    io.mapOptional("sends", form.sends);
    io.mapOptional("columns", form.columns,
                   cudaq::qec::window_yaml::columns_form{});
  }
};

} // namespace llvm::yaml
