/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The decode_window node's core: a layer's variants with their decoders, and
// the fire that runs the canonical workload on any instance of one. A
// received send lands on the rows of the shot it names, which a window's
// own rows and then its imported ones cover from its first row, so the
// same fire serves every instance of a variant. Construction sizes every
// buffer; a fire allocates nothing beyond its decoder's result.

#include "decoding_task_decode_window_core.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cudaq::qec {

namespace {

// A request send the variant lacks.
constexpr std::size_t no_send = std::numeric_limits<std::size_t>::max();

} // namespace

// Each variant's sends are matched to the request's by id; a send whose
// rounds lie outside a clipped window may be missing.
decode_window_core::decode_window_core(
    const data_artifact_store &store,
    std::shared_ptr<const experiment_template> context, window_request request,
    const std::string &decoder_name, const cudaqx::heterogeneous_map &params,
    std::size_t slots)
    : context_(std::move(context)), request_(std::move(request)) {
  if (slots == 0)
    fail("no slot to fire into");
  for (auto &[key, window] : store.find_windows(*context_, request_)) {
    auto &v = variants_.emplace_back();
    v.key = key;
    v.window = window;
    v.dec = make_window_decoder(*window, decoder_name, params);
    for (const auto &wanted : request_.sends) {
      const auto it = std::find_if(
          window->sends.begin(), window->sends.end(),
          [&](const window_send &send) { return send.id == wanted.id; });
      v.send_of.push_back(
          it == window->sends.end()
              ? no_send
              : static_cast<std::size_t>(it - window->sends.begin()));
    }
  } // end - for(key)
  workspaces_.resize(slots);
  detectors_.resize(slots);
  fires_.resize(slots);
  for (std::size_t s = 0; s < slots; ++s) {
    for (const auto &v : variants_) {
      workspaces_[s].push_back(make_window_workspace(*v.window));
      detectors_[s].emplace_back(v.window->model.h.num_rows(), 0);
    }
    fires_[s].sends.resize(request_.sends.size());
  }
} // end - decode_window_core()

// A send's row i lies at its window's first row plus the send's row, so it
// is row (that) - (this window's first row) here: a delta must land on an
// own row and a residual on an imported one. The rows are checked in one
// pass and applied in a second.
const window_fire &
decode_window_core::fire(std::size_t slot, const window_view &view,
                         std::span<const window_send_value *const> receives) {
  std::size_t v = 0;
  std::size_t row = 0;
  if (slot >= fires_.size())
    fail("slot " + std::to_string(slot) + " does not exist; there are " +
         std::to_string(fires_.size()));
  while (v < variants_.size() && !(variants_[v].key == view.variant))
    ++v;
  if (v == variants_.size())
    fail("window " + std::to_string(view.k) +
         "'s variant is none of this layer's");
  if (!view.source)
    fail("window " + std::to_string(view.k) + "'s view has no ring");
  const auto &w = *variants_[v].window;
  auto &d = detectors_[slot][v];
  auto &ws = workspaces_[slot][v];
  auto &out = fires_[slot];
  const auto own = w.num_own_rows;
  const auto all = d.size();
  // The receiving row of row i of send r, or all when it lies before this
  // window's rows.
  const auto target = [&](const window_send_value &r, std::size_t i) {
    const auto shot_row = r.first_row + r.send->rows[i];
    return shot_row < view.first_row ? all : shot_row - view.first_row;
  };

  for (auto r = view.rounds.begin; r < view.rounds.end; ++r) {
    const auto rows = view.source->detectors(r);
    if (row + rows.size() > own)
      break;
    std::copy(rows.begin(), rows.end(), d.begin() + row);
    row += rows.size();
  }
  if (row != own || view.rounds.empty())
    fail("window " + std::to_string(view.k) + "'s view holds other rows " +
         "than the " + std::to_string(own) + " of its variant");
  std::fill(d.begin() + own, d.end(), 0);

  for (const auto *r : receives) {
    if (!r || !r->send)
      continue;
    if (r->values.size() != r->send->rows.size())
      fail("a send '" + r->send->id + "' to window " + std::to_string(view.k) +
           " has " + std::to_string(r->values.size()) + " values for " +
           std::to_string(r->send->rows.size()) + " rows");
    for (std::size_t i = 0; i < r->values.size(); ++i) {
      const auto t = target(*r, i);
      const bool delta = r->send->mode == send_mode::delta;
      if (delta ? t >= own : t < own || t >= all)
        fail("send '" + r->send->id + "' reaches shot row " +
             std::to_string(r->first_row + r->send->rows[i]) +
             ", which window " + std::to_string(view.k) +
             " does not take as a " + (delta ? "delta" : "residual"));
    }
  } // end - for(r)
  for (const auto *r : receives) {
    if (!r || !r->send)
      continue;
    for (std::size_t i = 0; i < r->values.size(); ++i)
      if (r->send->mode == send_mode::delta)
        d[target(*r, i)] ^= r->values[i] & 1;
      else
        d[target(*r, i)] = r->values[i] & 1;
  }

  solve_window(w, variants_[v].dec.get(), d, ws);
  const auto anchor = request_.offset + view.k * request_.period;
  out.k = view.k;
  out.fired = true;
  out.commit = {std::max(view.rounds.begin, anchor + request_.commit->begin),
                std::min(view.rounds.end, anchor + request_.commit->end)};
  out.detectors = d;
  out.solution = &ws.solution;
  for (std::size_t s = 0; s < out.sends.size(); ++s) {
    const auto i = variants_[v].send_of[s];
    out.sends[s] = i == no_send
                       ? window_send_value{}
                       : window_send_value{ws.solution.outputs.sends[i],
                                           &w.sends[i], view.first_row};
  }
  return out;
} // end - fire()

// The slots are few, so a scan finds the one holding window k.
const window_fire *decode_window_core::find(std::int64_t k) const {
  for (const auto &f : fires_)
    if (f.fired && f.k == k)
      return &f;
  return nullptr;
}

// In the request's order, which every fire's sends follow.
std::size_t decode_window_core::send_index(const std::string &id) const {
  for (std::size_t s = 0; s < request_.sends.size(); ++s)
    if (request_.sends[s].id == id)
      return s;
  fail("the layer has no send '" + id + "'");
}

// Errors name the node kind.
void decode_window_core::fail(const std::string &what) const {
  throw std::invalid_argument("decode_window: " + what);
}

} // namespace cudaq::qec
