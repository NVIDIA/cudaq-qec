/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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

} // namespace

// Field for field; a send's one range becomes its list of ranges.
window_request request_from_form(const request_form &form) {
  window_request request;
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
} // end - request_from_form()

// The inverse of request_from_form(), every range in list form.
request_form form_from_request(const window_request &request) {
  request_form form;
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
    auto &out = form.sends.emplace_back();
    if (send.rounds.size() > 1)
      throw std::invalid_argument("window request: send '" + send.id +
                                  "' has more than one round range, which "
                                  "YAML cannot write");
    out.id = send.id;
    out.mode = send.mode;
    if (!send.rounds.empty())
      out.rounds = to_form(send.rounds.front());
  }
  form.columns = {request.columns.merge_identical,
                  request.columns.conflicting_parallel};
  return form;
} // end - form_from_request()

} // namespace cudaq::qec::window_yaml

namespace cudaq::qec {

// Parses into the form, then converts. Shape errors inside the traits throw
// directly; the rest surface as the input's error.
experimental::window_request
window_request_from_yaml_str(const std::string &text) {
  window_yaml::request_form form;
  llvm::yaml::Input in(text);
  in >> form;
  if (in.error())
    window_yaml::reject("invalid YAML (" + in.error().message() + ")");
  return window_yaml::request_from_form(form);
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
