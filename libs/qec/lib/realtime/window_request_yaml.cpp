/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// YAML for window requests: forms that mirror the YAML shapes, their
// llvm::yaml traits, and conversion to and from window_request. Reserved
// keys are rejected by name, so files written for spatial windows fail with
// a message rather than an unknown-key error.

#include "window_request_yaml.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cudaq::qec;

// Throws std::invalid_argument naming the parse.
[[noreturn]] void reject(const std::string &what) {
  throw std::invalid_argument("window_request_from_yaml_str: " + what);
}

// The map form of a round range, {rounds: [b, e]}; the list form [b, e]
// fills the same rounds.
struct range_map_form {
  std::vector<std::int64_t> rounds;
};

// A round range in either form. A scalar is kept only to be rejected.
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

struct request_form {
  std::int64_t period = 0;
  std::int64_t offset = 0;
  zones_form zones;
  ports_form ports;
  std::vector<send_form> sends;
  columns_form columns;
};

} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(range_form)
LLVM_YAML_IS_SEQUENCE_VECTOR(send_form)

namespace llvm::yaml {

template <>
struct ScalarEnumerationTraits<port_disposition> {
  static void enumeration(IO &io, port_disposition &value) {
    io.enumCase(value, "open", port_disposition::open);
    io.enumCase(value, "closed", port_disposition::closed);
  }
};

template <>
struct ScalarEnumerationTraits<input_boundary> {
  static void enumeration(IO &io, input_boundary &value) {
    io.enumCase(value, "columns", input_boundary::columns);
    io.enumCase(value, "truncate", input_boundary::truncate);
  }
};

template <>
struct ScalarEnumerationTraits<send_mode> {
  static void enumeration(IO &io, send_mode &value) {
    io.enumCase(value, "delta", send_mode::delta);
    io.enumCase(value, "residual", send_mode::residual);
  }
};

template <>
struct ScalarEnumerationTraits<parallel_resolution> {
  static void enumeration(IO &io, parallel_resolution &value) {
    io.enumCase(value, "reject", parallel_resolution::reject);
    io.enumCase(value, "most_likely", parallel_resolution::most_likely);
  }
};

// {rounds: [b, e]}, with occurrences and block reserved.
template <>
struct CustomMappingTraits<range_map_form> {
  static void inputOne(IO &io, StringRef key, range_map_form &form) {
    if (key == "rounds") {
      io.mapRequired("rounds", form.rounds);
      return;
    }
    if (key == "occurrences" || key == "block")
      reject("'" + key.str() +
             "' is reserved for spatial windows, which are not supported yet");
    reject("unknown key '" + key.str() + "' in a round range");
  }
  static void output(IO &io, range_map_form &form) {
    io.mapRequired("rounds", form.rounds);
  }
};

// [b, e] or {rounds: [b, e]}; written as [b, e].
template <>
struct PolymorphicTraits<range_form> {
  static NodeKind getKind(const range_form &) { return NodeKind::Sequence; }
  static std::string &getAsScalar(range_form &form) { return form.scalar; }
  static range_map_form &getAsMap(range_form &form) { return form.map; }
  static std::vector<std::int64_t> &getAsSequence(range_form &form) {
    return form.map.rounds;
  }
};

template <>
struct MappingTraits<zones_form> {
  static void mapping(IO &io, zones_form &form) {
    io.mapOptional("commit", form.commit);
    io.mapOptional("buffers", form.buffers);
  }
};

template <>
struct MappingTraits<input_form> {
  static void mapping(IO &io, input_form &form) {
    io.mapOptional("disposition", form.disposition);
    io.mapOptional("boundary", form.boundary);
    io.mapOptional("prior", form.prior);
  }
};

template <>
struct MappingTraits<output_form> {
  static void mapping(IO &io, output_form &form) {
    io.mapOptional("disposition", form.disposition);
    io.mapOptional("imports", form.imports);
  }
};

// Only the temporal ports; spatial ones are reserved.
template <>
struct CustomMappingTraits<ports_form> {
  static void inputOne(IO &io, StringRef key, ports_form &form) {
    if (key == "input") {
      io.mapRequired("input", form.input);
      return;
    }
    if (key == "output") {
      io.mapRequired("output", form.output);
      return;
    }
    reject("port '" + key.str() +
           "': spatial ports are not supported yet; use input and output");
  }
  static void output(IO &io, ports_form &form) {
    io.mapRequired("input", form.input);
    io.mapRequired("output", form.output);
  }
};

template <>
struct MappingTraits<send_form> {
  static void mapping(IO &io, send_form &form) {
    io.mapRequired("id", form.id);
    io.mapOptional("mode", form.mode);
    io.mapOptional("rounds", form.rounds);
  }
};

template <>
struct MappingTraits<columns_form> {
  static void mapping(IO &io, columns_form &form) {
    io.mapOptional("merge_identical", form.merge_identical, false);
    io.mapOptional("conflicting_parallel", form.conflicting_parallel,
                   parallel_resolution::reject);
  }
};

// The window keys, with responsibility and soundness_class reserved.
template <>
struct CustomMappingTraits<request_form> {
  static void inputOne(IO &io, StringRef key, request_form &form) {
    if (key == "period")
      io.mapRequired("period", form.period);
    else if (key == "offset")
      io.mapRequired("offset", form.offset);
    else if (key == "zones")
      io.mapRequired("zones", form.zones);
    else if (key == "ports")
      io.mapRequired("ports", form.ports);
    else if (key == "sends")
      io.mapRequired("sends", form.sends);
    else if (key == "columns")
      io.mapRequired("columns", form.columns);
    else if (key == "responsibility" || key == "soundness_class")
      reject("'" + key.str() +
             "' is reserved for spatial windows, which are not supported yet");
    else
      reject("unknown key '" + key.str() + "'");
  }
  static void output(IO &io, request_form &form) {
    io.mapRequired("period", form.period);
    io.mapOptional("offset", form.offset, std::int64_t{0});
    io.mapRequired("zones", form.zones);
    io.mapRequired("ports", form.ports);
    io.mapOptional("sends", form.sends);
    io.mapOptional("columns", form.columns, columns_form{});
  }
};

} // namespace llvm::yaml

namespace {

// A range form as [begin, end).
round_range to_range(const range_form &form) {
  const auto &rounds = form.map.rounds;
  if (!form.scalar.empty() || rounds.size() != 2)
    reject("a round range is [begin, end]");
  return {rounds[0], rounds[1]};
}

// [begin, end) as a range form.
range_form to_form(round_range range) {
  range_form form;
  form.map.rounds = {range.begin, range.end};
  return form;
}

} // namespace

namespace cudaq::qec {

// Parses into the forms, then converts. Shape errors inside the traits throw
// directly; the rest surface as the input's error.
window_request window_request_from_yaml_str(const std::string &text) {
  request_form form;
  window_request request;
  llvm::yaml::Input in(text);
  in >> form;
  if (in.error())
    reject("invalid YAML (" + in.error().message() + ")");
  request.period = form.period;
  request.offset = form.offset;
  if (form.zones.commit)
    request.commit = to_range(*form.zones.commit);
  for (const auto &buffer : form.zones.buffers)
    request.buffers.push_back(to_range(buffer));
  request.input = {form.ports.input.disposition, form.ports.input.prior,
                   form.ports.input.boundary};
  request.output.disposition = form.ports.output.disposition;
  if (form.ports.output.imports)
    request.output.imports = to_range(*form.ports.output.imports);
  for (const auto &send : form.sends) {
    auto &out = request.sends.emplace_back();
    out.id = send.id;
    out.mode = send.mode;
    if (send.rounds)
      out.rounds.push_back(to_range(*send.rounds));
  }
  request.columns = {form.columns.merge_identical,
                     form.columns.conflicting_parallel};
  return request;
} // end - window_request_from_yaml_str()

// The inverse of window_request_from_yaml_str(): every range in list form.
std::string window_request_to_yaml_str(const window_request &request) {
  request_form form;
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::yaml::Output out(os);
  form.period = request.period;
  form.offset = request.offset;
  if (request.commit)
    form.zones.commit = to_form(*request.commit);
  for (const auto buffer : request.buffers)
    form.zones.buffers.push_back(to_form(buffer));
  form.ports.input = {request.input.disposition, request.input.boundary,
                      request.input.prior};
  form.ports.output.disposition = request.output.disposition;
  if (request.output.imports)
    form.ports.output.imports = to_form(*request.output.imports);
  for (const auto &send : request.sends) {
    auto &f = form.sends.emplace_back();
    if (send.rounds.size() > 1)
      throw std::invalid_argument("window_request_to_yaml_str: send '" +
                                  send.id + "' has more than one round range");
    f.id = send.id;
    f.mode = send.mode;
    if (!send.rounds.empty())
      f.rounds = to_form(send.rounds.front());
  }
  form.columns = {request.columns.merge_identical,
                  request.columns.conflicting_parallel};
  out << form;
  os.flush();
  return text;
} // end - window_request_to_yaml_str()

} // namespace cudaq::qec
