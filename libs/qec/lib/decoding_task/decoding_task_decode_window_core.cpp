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
#include <stdexcept>

namespace cudaq::qec {

// A window has every send of its request, in the request's order, with no
// rows for one whose rounds it clips away. That is checked here, so a fire
// publishes the window's send s as the request's send s. The bulk, which
// any window of a shot may be, has a decoder per slot, so fires into
// different slots never share one.
decode_window_core::decode_window_core(
    const data_artifact_store &store,
    std::shared_ptr<const experiment_template> context, window_request request,
    const std::string &decoder_name, const cudaqx::heterogeneous_map &params,
    std::size_t slots)
    : context_(std::move(context)), request_(std::move(request)) {
  std::size_t imported = 0;
  if (slots == 0)
    fail("no slot to fire into");
  span_ = zone_span(request_);
  for (auto &[key, window] : store.find_windows(*context_, request_)) {
    auto &v = variants_.emplace_back();
    v.key = key;
    v.window = window;
    for (std::size_t s = 0; s < (key.bulk() ? slots : 1); ++s)
      v.decoders.push_back(make_window_decoder(*window, decoder_name, params));
    if (window->sends.size() != request_.sends.size() ||
        !std::equal(window->sends.begin(), window->sends.end(),
                    request_.sends.begin(),
                    [](const window_send &send, const auto &wanted) {
                      return send.id == wanted.id;
                    }))
      throw std::logic_error("decode_window: a window's sends are not its "
                             "request's, in order");
    imported =
        std::max(imported, window->model.h.num_rows() - window->num_own_rows);
  } // end - for(key)
  covered_.assign(slots, std::vector<std::uint8_t>(imported, 0));
  decoding_ = std::make_unique<std::atomic<bool>[]>(variants_.size());
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
// own row and a residual on an imported one, and every imported row must be
// reached, since a window imports rows only when the instance that values
// them exists. The view must be this layer's window k: its rounds are the
// request's zones at k's anchor, clipped to the shot, whose end a terminal
// variant gives. The view and every received row are checked before the
// slot is touched, so a rejected fire changes nothing; the slot is marked
// unfired while it is rewritten, so a decoder that fails leaves it empty
// rather than torn.
const window_fire &
decode_window_core::fire(std::size_t slot, const window_view &view,
                         std::span<const window_send_value *const> receives) {
  std::size_t v = 0;
  std::size_t row = 0;
  std::size_t rows = 0;
  // Frees a prefix or terminal variant's decoder however the fire ends.
  struct free_on_exit {
    std::atomic<bool> *decoding = nullptr;
    ~free_on_exit() {
      if (decoding)
        *decoding = false;
    }
  } claim;
  check_slot(slot);
  while (v < variants_.size() && !(variants_[v].key == view.variant))
    ++v;
  if (v == variants_.size())
    fail("window " + std::to_string(view.k) +
         "'s variant is none of this layer's");
  if (!view.source)
    fail("window " + std::to_string(view.k) + "'s view has no ring");
  const auto anchor = request_.offset + view.k * request_.period;
  const round_range rounds{
      std::max<std::int64_t>(0, anchor + span_.begin),
      view.variant.terminal
          ? std::min(anchor + span_.end, anchor + *view.variant.terminal)
          : anchor + span_.end};
  if (view.rounds != rounds)
    fail("window " + std::to_string(view.k) + "'s view spans rounds [" +
         std::to_string(view.rounds.begin) + ", " +
         std::to_string(view.rounds.end) + "), not this layer's window " +
         std::to_string(view.k) + "'s [" + std::to_string(rounds.begin) + ", " +
         std::to_string(rounds.end) + ")");
  const auto &w = *variants_[v].window;
  auto &d = detectors_[slot][v];
  auto &ws = workspaces_[slot][v];
  auto &out = fires_[slot];
  auto &covered = covered_[slot];
  const auto &decoders = variants_[v].decoders;
  const auto own = w.num_own_rows;
  const auto all = d.size();
  // The receiving row of row i of send r, or all when it lies before this
  // window's rows.
  const auto target = [&](const window_send_value &r, std::size_t i) {
    const auto shot_row = r.first_row + r.send->rows[i];
    return shot_row < view.first_row ? all : shot_row - view.first_row;
  };

  // The view's rows, counted until they are too many.
  for (auto r = view.rounds.begin; r < view.rounds.end && rows <= own; ++r)
    rows += view.source->view_detectors(view, r).size();
  if (rows != own || view.rounds.empty())
    fail("window " + std::to_string(view.k) + "'s view holds other rows " +
         "than the " + std::to_string(own) + " of its variant");
  std::fill(covered.begin(), covered.begin() + (all - own), 0);
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
      if (!delta)
        covered[t - own] = 1;
    }
  } // end - for(r)
  for (std::size_t i = 0; i < all - own; ++i)
    if (!covered[i])
      fail("window " + std::to_string(view.k) + " imports " +
           std::to_string(all - own) + " rows, but no residual reaches " +
           "its imported row " + std::to_string(i));
  // A prefix or terminal variant's one decoder is claimed before the slot
  // is touched, so a second fire of it leaves its slot as it was.
  if (!view.variant.bulk()) {
    if (decoding_[v].exchange(true))
      throw std::logic_error("decode_window: window " + std::to_string(view.k) +
                             "'s variant is decoding in another slot; a "
                             "window fires into one slot at a time");
    claim.decoding = &decoding_[v];
  }

  out.fired = false;
  for (auto r = view.rounds.begin; r < view.rounds.end; ++r) {
    const auto round = view.source->view_detectors(view, r);
    std::copy(round.begin(), round.end(), d.begin() + row);
    row += round.size();
  }
  std::fill(d.begin() + own, d.end(), 0);
  for (const auto *r : receives) {
    if (!r || !r->send)
      continue;
    for (std::size_t i = 0; i < r->values.size(); ++i)
      if (r->send->mode == send_mode::delta)
        d[target(*r, i)] ^= r->values[i] & 1;
      else
        d[target(*r, i)] = r->values[i] & 1;
  }

  solve_window(w, decoders[decoders.size() > 1 ? slot : 0].get(), d, ws);
  out.k = view.k;
  out.fired = true;
  out.commit = {std::max(view.rounds.begin, anchor + request_.commit->begin),
                std::min(view.rounds.end, anchor + request_.commit->end)};
  out.detectors = d;
  out.solution = &ws.solution;
  for (std::size_t s = 0; s < out.sends.size(); ++s)
    out.sends[s] = {ws.solution.outputs.sends[s], &w.sends[s], view.first_row};
  return out;
} // end - fire()

// A new shot: no slot holds a fire.
void decode_window_core::reset() {
  for (auto &f : fires_)
    f.fired = false;
}

// Reads nothing of another slot.
const window_fire &decode_window_core::held(std::size_t slot) const {
  check_slot(slot);
  return fires_[slot];
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

// The slot count is fixed at construction.
void decode_window_core::check_slot(std::size_t slot) const {
  if (slot >= fires_.size())
    fail("slot " + std::to_string(slot) + " does not exist; there are " +
         std::to_string(fires_.size()));
}

} // namespace cudaq::qec
