/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// YAML for window requests: conversion between the YAML forms and
// window_request, and the standalone parser and emitter. The forms and their
// traits live in window_request_yaml_traits.h, which the config parser
// shares.

#include "window_request_yaml.h"
#include "llvm/Support/raw_ostream.h"
#include "window_request_yaml_traits.h"
#include <stdexcept>
#include <string>

namespace cudaq::qec::window_yaml {

// Every request error names what it is about; callers add where.
void reject(const std::string &what) {
  throw std::invalid_argument("window request: " + what);
}

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

// A send's rounds: one range, or a list whose every element is a range.
std::vector<round_range> to_ranges(const send_rounds_form &form) {
  std::vector<round_range> out;
  std::vector<std::int64_t> numbers;
  const auto *shape = "a send's rounds are [begin, end] or a list of ranges";
  if (!form.scalar.empty())
    reject(shape);
  if (!form.map.rounds.empty())
    return {to_range({form.map, ""})};
  for (const auto &item : form.items)
    if (item.number.set)
      numbers.push_back(item.number.value);
  if (!numbers.empty() && numbers.size() == form.items.size())
    return {to_range({{numbers}, ""})};
  if (!numbers.empty() || form.items.empty())
    reject(shape);
  for (const auto &item : form.items)
    out.push_back(to_range(
        {item.list.empty() ? item.map : range_map_form{item.list}, ""}));
  return out;
}

// One range as [begin, end], several as a list of them.
send_rounds_form to_send_form(const std::vector<round_range> &rounds) {
  send_rounds_form form;
  if (rounds.size() == 1) {
    form.items.push_back({{rounds[0].begin, true}, {}, {}});
    form.items.push_back({{rounds[0].end, true}, {}, {}});
    return form;
  }
  for (const auto r : rounds)
    form.items.push_back({{}, {r.begin, r.end}, {}});
  return form;
}

} // namespace

// Field for field; a send's rounds become its list of ranges.
window_request request_from_form(const request_form &form) {
  if (!form.period)
    reject("missing required key 'period'");
  if (!form.zones)
    reject("missing required key 'zones'");
  if (!form.ports)
    reject("missing required key 'ports'");
  const auto &zones = *form.zones;
  const auto &ports = *form.ports;
  window_request request;
  request.period = *form.period;
  request.offset = form.offset;
  if (zones.commit)
    request.commit = to_range(*zones.commit);
  for (const auto &buffer : zones.buffers)
    request.buffers.push_back(to_range(buffer));
  request.input = {ports.input.disposition, ports.input.prior,
                   ports.input.boundary};
  request.output.disposition = ports.output.disposition;
  if (ports.output.imports)
    request.output.imports = to_range(*ports.output.imports);
  for (const auto &send : form.sends) {
    auto &out = request.sends.emplace_back();
    out.id = send.id;
    out.mode = send.mode;
    if (send.rounds)
      out.rounds = to_ranges(*send.rounds);
  }
  request.columns = {form.columns.merge_identical,
                     form.columns.conflicting_parallel};
  return request;
} // end - request_from_form()

// The inverse of request_from_form(), every range in list form.
request_form form_from_request(const window_request &request) {
  request_form form;
  form.period = request.period;
  form.offset = request.offset;
  auto &zones = form.zones.emplace();
  auto &ports = form.ports.emplace();
  if (request.commit)
    zones.commit = to_form(*request.commit);
  for (const auto buffer : request.buffers)
    zones.buffers.push_back(to_form(buffer));
  ports.input = {request.input.disposition, request.input.boundary,
                 request.input.prior};
  ports.output.disposition = request.output.disposition;
  if (request.output.imports)
    ports.output.imports = to_form(*request.output.imports);
  for (const auto &send : request.sends) {
    auto &out = form.sends.emplace_back();
    out.id = send.id;
    out.mode = send.mode;
    if (!send.rounds.empty())
      out.rounds = to_send_form(send.rounds);
  }
  form.columns = {request.columns.merge_identical,
                  request.columns.conflicting_parallel};
  return form;
} // end - form_from_request()

} // namespace cudaq::qec::window_yaml

namespace cudaq::qec {

// Parses into the form, converts, and checks the construction rules. Shape
// errors inside the traits throw directly; the rest surface as the input's
// error.
experimental::window_request
window_request_from_yaml_str(const std::string &text) {
  window_yaml::request_form form;
  llvm::yaml::Input in(text);
  in >> form;
  if (in.error())
    window_yaml::reject("invalid YAML (" + in.error().message() + ")");
  auto request = window_yaml::request_from_form(form);
  experimental::check_window_request(request);
  return request;
}

// Emits the request's form.
std::string
window_request_to_yaml_str(const experimental::window_request &request) {
  auto form = window_yaml::form_from_request(request);
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::yaml::Output out(os);
  out << form;
  os.flush();
  return text;
}

} // namespace cudaq::qec
