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
#include "cudaq/qec/decoding_task_graph.h"
#include <algorithm>
#include <stdexcept>

namespace cudaq::qec {

// A fire's sends travel on edges by value.
CUDAQ_REGISTER_EDGE_VALUE_TYPE(window_send_payload, "window_send")

// A window has every send of its request, in the request's order, with no
// rows for one whose rounds it clips away. That is checked here, so a fire
// publishes the window's send s as the request's send s. The bulk, which
// any window of a shot may be, has a decoder per slot, so fires into
// different slots never share one.
decode_window_core::decode_window_core(
    const data_artifact_store &store,
    std::shared_ptr<const experiment_template> context, window_request request,
    const std::string &decoder_name, const cudaqx::heterogeneous_map &params,
    std::size_t slots, verify_residual_mode verify)
    : context_(std::move(context)), request_(std::move(request)),
      verify_(verify) {
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
  // Almost every window of a shot is the bulk, so a fire's variant lookup
  // tries it first.
  std::stable_partition(variants_.begin(), variants_.end(),
                        [](const variant &v) { return v.key.bulk(); });
  covered_.assign(slots, std::vector<std::uint8_t>(imported, 0));
  decoding_ = std::make_unique<claim_flag[]>(variants_.size());
  workspaces_.resize(slots);
  detectors_.resize(slots);
  fires_.resize(slots);
  for (std::size_t s = 0; s < slots; ++s) {
    for (const auto &v : variants_) {
      workspaces_[s].push_back(make_window_workspace(*v.window));
      detectors_[s].emplace_back(v.window->model.h.num_rows(), 0);
    }
    fires_[s].fire.sends.resize(request_.sends.size());
  }
} // end - decode_window_core()

// A send's row i lies at its window's first row plus the send's row, so it
// is row (that) - (this window's first row) here: a delta must land on an
// own row and a residual on an imported one, and every imported row must be
// reached, since a window imports rows only when the instance that values
// them exists. The window must be this layer's window k: its rounds are the
// request's zones at k's anchor, clipped to the shot, whose end a terminal
// variant gives. The window and every received row are checked before the
// slot is touched, so a rejected fire changes nothing; the slot is marked
// unfired while it is rewritten, so a decoder that fails, or a solution that
// verify_residual fail rejects, leaves it empty rather than torn. Whether
// H e = d is always known: solve_window() finds it as it commits.
const window_fire &
decode_window_core::fire(std::size_t slot, const window_rows &window,
                         std::span<const window_send_value *const> receives) {
  std::size_t v = 0;
  std::size_t missed = 0;      // The first row H e does not explain
  std::size_t num_covered = 0; // Imported rows some residual reaches
  // Frees a prefix or terminal variant's decoder however the fire ends.
  struct free_on_exit {
    std::atomic<bool> *decoding = nullptr;
    ~free_on_exit() {
      if (decoding)
        *decoding = false;
    }
  } claim;
  check_slot(slot);
  while (v < variants_.size() && !(variants_[v].key == window.variant))
    ++v;
  if (v == variants_.size())
    fail("window " + std::to_string(window.k) +
         "'s variant is none of this layer's");
  const auto anchor = request_.offset + window.k * request_.period;
  const round_range rounds{
      std::max<std::int64_t>(0, anchor + span_.begin),
      window.variant.terminal
          ? std::min(anchor + span_.end, anchor + *window.variant.terminal)
          : anchor + span_.end};
  if (window.variant.position && *window.variant.position != window.k)
    fail("window " + std::to_string(window.k) + " carries window " +
         std::to_string(*window.variant.position) + "'s variant");
  if (window.first_row != first_row(window))
    fail("window " + std::to_string(window.k) + "'s first row is " +
         std::to_string(window.first_row) + ", not " +
         std::to_string(first_row(window)) + ", where its round " +
         std::to_string(window.rounds.begin) + " starts");
  if (window.rounds != rounds)
    fail("window " + std::to_string(window.k) + " spans rounds [" +
         std::to_string(window.rounds.begin) + ", " +
         std::to_string(window.rounds.end) + "), not this layer's window " +
         std::to_string(window.k) + "'s [" + std::to_string(rounds.begin) +
         ", " + std::to_string(rounds.end) + ")");
  const auto &w = *variants_[v].window;
  auto &d = detectors_[slot][v];
  auto &ws = workspaces_[slot][v];
  auto &out = fires_[slot].fire;
  auto &covered = covered_[slot];
  const auto &decoders = variants_[v].decoders;
  const auto own = w.num_own_rows;
  const auto all = d.size();
  // The receiving row of row i of send r, or all when it lies before this
  // window's rows.
  const auto target = [&](const window_send_value &r, std::size_t i) {
    const auto shot_row = r.first_row + r.send->rows[i];
    return shot_row < window.first_row ? all : shot_row - window.first_row;
  };

  if (window.detectors.size() != own || window.rounds.empty())
    fail("window " + std::to_string(window.k) + " holds " +
         std::to_string(window.detectors.size()) + " rows, not the " +
         std::to_string(own) + " of its variant");
  std::fill(covered.begin(), covered.begin() + (all - own), 0);
  for (const auto *r : receives) {
    if (!r || !r->send)
      continue;
    if (r->values.size() != r->send->rows.size())
      fail("a send '" + r->send->id + "' to window " +
           std::to_string(window.k) + " has " +
           std::to_string(r->values.size()) + " values for " +
           std::to_string(r->send->rows.size()) + " rows");
    for (std::size_t i = 0; i < r->values.size(); ++i) {
      const auto t = target(*r, i);
      const bool delta = r->send->mode == send_mode::delta;
      if (delta ? t >= own : t < own || t >= all)
        fail("send '" + r->send->id + "' reaches shot row " +
             std::to_string(r->first_row + r->send->rows[i]) +
             ", which window " + std::to_string(window.k) +
             " does not take as a " + (delta ? "delta" : "residual"));
      if (!delta && !covered[t - own]) {
        covered[t - own] = 1;
        ++num_covered;
      }
    }
  } // end - for(r)
  // Only a rejected fire looks for the row that was missed.
  if (num_covered != all - own) {
    std::size_t i = 0;
    while (covered[i])
      ++i;
    fail("window " + std::to_string(window.k) + " imports " +
         std::to_string(all - own) + " rows, but no residual reaches " +
         "its imported row " + std::to_string(i));
  }
  // A prefix or terminal variant's one decoder is claimed before the slot
  // is touched, so a second fire of it leaves its slot as it was.
  if (!window.variant.bulk()) {
    if (decoding_[v].decoding.exchange(true))
      throw std::logic_error("decode_window: window " +
                             std::to_string(window.k) +
                             "'s variant is decoding in another slot; a "
                             "window fires into one slot at a time");
    claim.decoding = &decoding_[v].decoding;
  }

  out.fired = false;
  out.failure.clear();
  std::copy(window.detectors.begin(), window.detectors.end(), d.begin());
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
  if (verify_ == verify_residual_mode::fail && !ws.solution.satisfied) {
    while (ws.explained[missed] == (d[missed] & 1))
      ++missed;
    // Reported, not thrown: the caller decides how the failure surfaces.
    out.failure =
        "decode_window: window " + std::to_string(window.k) +
        "'s solution does not satisfy its detectors (verify_residual: "
        "fail): H e first differs from d at its " +
        (missed < own ? "row " + std::to_string(missed) + ", shot row " +
                            std::to_string(window.first_row + missed)
                      : "imported row " + std::to_string(missed - own));
    return out;
  }
  out.k = window.k;
  out.fired = true;
  out.checked = verify_ != verify_residual_mode::none;
  out.commit = {std::max(window.rounds.begin, anchor + request_.commit->begin),
                std::min(window.rounds.end, anchor + request_.commit->end)};
  out.detectors = d;
  out.solution = &ws.solution;
  for (std::size_t s = 0; s < out.sends.size(); ++s)
    out.sends[s] = {ws.solution.outputs.sends[s], &w.sends[s],
                    window.first_row};
  return out;
} // end - fire()

// A new shot: no slot holds a fire.
void decode_window_core::reset() {
  for (auto &f : fires_)
    f.fire.fired = false;
}

// Touches nothing of another slot.
void decode_window_core::reset(std::size_t slot) {
  check_slot(slot);
  fires_[slot].fire.fired = false;
}

// Reads nothing of another slot.
const window_fire &decode_window_core::held(std::size_t slot) const {
  check_slot(slot);
  return fires_[slot].fire;
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

// A terminal window's key gives the shot's length. Any other window's
// rounds come before the shot's suffix, whose rows do not depend on the
// length, so the shortest length that keeps them there serves.
std::size_t decode_window_core::first_row(const window_rows &window) const {
  const auto anchor = request_.offset + window.k * request_.period;
  const auto begin =
      static_cast<std::size_t>(std::max<std::int64_t>(0, window.rounds.begin));
  std::size_t R = context_->min_rounds();
  if (window.variant.terminal)
    R = static_cast<std::size_t>(anchor + *window.variant.terminal);
  else if (context_->repeats())
    R = std::max(R, static_cast<std::size_t>(
                        std::max<std::int64_t>(0, window.rounds.end)) +
                        context_->suffix_rounds());
  return context_->row_offset(begin, R);
}

// Variants clip a send's rows differently near the shot's ends.
std::size_t decode_window_core::max_send_rows(std::size_t s) const {
  std::size_t most = 0;
  for (const auto &v : variants_)
    most = std::max(most, v.window->sends.at(s).rows.size());
  return most;
}

// The slot count is fixed at construction.
void decode_window_core::check_slot(std::size_t slot) const {
  if (slot >= fires_.size())
    fail("slot " + std::to_string(slot) + " does not exist; there are " +
         std::to_string(fires_.size()));
}

} // namespace cudaq::qec
