/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The ingest node's core: cutting records into rounds, forming each round's
// detectors once into a ring of fixed size, and handing every layer's window
// k over together, as views of the ring. Construction measures and checks
// the template's rounds and sizes every buffer, so enqueue(), poll() and
// release(), which run per record, check only their input and allocate
// nothing.

#include "decoding_task_ingest_core.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq::qec {
namespace {

// Measurements in round r of an R-round shot.
std::size_t width_of(const experiment_template &t, std::size_t r,
                     std::size_t R) {
  return t.measurement_offset(r + 1, R) - t.measurement_offset(r, R);
}

// Detectors in round r of an R-round shot.
std::size_t rows_of(const experiment_template &t, std::size_t r,
                    std::size_t R) {
  return t.row_offset(r + 1, R) - t.row_offset(r, R);
}

// The least power of two not below n.
std::size_t power_of_two_at_least(std::size_t n) {
  std::size_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

// "1 round", "3 rounds".
std::string rounds_text(std::size_t n) {
  return std::to_string(n) + (n == 1 ? " round" : " rounds");
}

} // namespace

// Rounds are measured on a shot with bulk rounds between the prefix and the
// suffix. Each round's D_sparse is then checked against the widths of its
// round and the one before, on that shot and the shortest ones, which hold
// every pair of neighbouring phases, so forming detectors needs no check.
// The ring holds one index's windows with their reach, a period for each
// view a pool can have out and one more, a record, and the suffix.
ingest_core::ingest_core(std::shared_ptr<const experiment_template> context,
                         std::vector<ingest_layer> layers, bool read_logical)
    : context_(std::move(context)), layers_(std::move(layers)),
      read_logical_(read_logical) {
  const auto reject = [](const std::string &what) {
    throw std::invalid_argument("ingest: " + what);
  };
  if (!context_)
    reject("no template");
  if (layers_.empty())
    reject("no layer");
  const auto &t = *context_;
  const bool repeats = t.repeats();
  const auto probe = repeats ? t.min_rounds() + 2 : t.min_rounds();
  std::int64_t lowest = std::numeric_limits<std::int64_t>::max();
  std::int64_t highest = std::numeric_limits<std::int64_t>::min();
  std::size_t max_views = 0;
  std::size_t readout_rows = 0;

  for (std::size_t l = 0; l < layers_.size(); ++l) {
    const auto &request = layers_[l].request;
    experimental::check_window_request(request);
    if (request.period != layers_.front().request.period)
      reject("the layers' periods differ: " +
             std::to_string(layers_.front().request.period) + " and " +
             std::to_string(request.period));
    if (layers_[l].max_views == 0)
      reject("layer " + std::to_string(l) + "'s view pool is empty");
    first_k_.push_back(window_instances(request, 0).begin);
    spans_.push_back(zone_span(request));
    lowest = std::min(lowest, request.offset + spans_.back().begin);
    highest = std::max(highest, request.offset + model_reach(t, request).end);
    max_views = std::max(max_views, layers_[l].max_views);
  } // end - for(l)
  period_ = layers_.front().request.period;

  // The rounds, by kind. A spec that does not repeat is all prefix.
  has_d_ = t.spec().has_D_sparse();
  prefix_rounds_ = t.prefix_rounds();
  suffix_rounds_ = repeats ? t.suffix_rounds() : 0;
  const auto d_of = [&](std::size_t r) {
    return &t.spec().get_phase(t.phase(r, probe)).D_sparse;
  };
  for (std::size_t r = 0; r < prefix_rounds_; ++r) {
    prefix_widths_.push_back(width_of(t, r, probe));
    prefix_rows_.push_back(rows_of(t, r, probe));
    prefix_d_.push_back(d_of(r));
  }
  if (repeats) {
    bulk_width_ = width_of(t, prefix_rounds_, probe);
    bulk_rows_ = rows_of(t, prefix_rounds_, probe);
    bulk_d_ = d_of(prefix_rounds_);
    for (auto r = probe - suffix_rounds_; r < probe; ++r) {
      suffix_rows_.push_back(rows_of(t, r, probe));
      suffix_d_.push_back(d_of(r));
    }
  }
  last_width_ = width_of(t, probe - 1, probe);
  min_width_ = std::min(last_width_, repeats ? bulk_width_ : last_width_);
  max_width_ = std::max(last_width_, bulk_width_);
  max_rows_ = std::max(bulk_rows_, rows_of(t, probe - 1, probe));
  for (std::size_t r = 0; r < prefix_rounds_; ++r) {
    min_width_ = std::min(min_width_, prefix_widths_[r]);
    max_width_ = std::max(max_width_, prefix_widths_[r]);
    max_rows_ = std::max(max_rows_, prefix_rows_[r]);
  }
  for (const auto rows : suffix_rows_)
    max_rows_ = std::max(max_rows_, rows);
  if (min_width_ == 0)
    reject("a round holds no measurements, so a record has no bound in "
           "rounds");
  record_limit_ = static_cast<std::size_t>(period_) * max_width_;
  // Rounds cut before FINAL: bulk and every prefix round, but the last
  // round of a shot that does not repeat, which is its readout round.
  min_forward_width_ = repeats ? bulk_width_ : last_width_;
  for (std::size_t r = 0; r + (repeats ? 0 : 1) < prefix_rounds_; ++r)
    min_forward_width_ = std::min(min_forward_width_, prefix_widths_[r]);
  for (std::size_t r = 0; !repeats && r < prefix_rounds_; ++r)
    fixed_measurements_ += prefix_widths_[r];

  // Every round's D_sparse, as forming detectors will read it.
  for (auto R = t.min_rounds(); R <= probe; ++R)
    for (std::size_t r = 0; r < R; ++r) {
      const auto width = width_of(t, r, R);
      const auto before = r == 0 ? 0 : width_of(t, r - 1, R);
      const auto rows = rows_of(t, r, R);
      const auto *d = &t.spec().get_phase(t.phase(r, R)).D_sparse;
      std::size_t counted = 0;
      const auto where =
          "round " + std::to_string(r) + " of a " + rounds_text(R) + " shot";
      if (width != (r + 1 == R ? last_width_ : forward_width(r)))
        reject(where + " is " + std::to_string(width) +
               " measurements wide, unlike its phase elsewhere");
      if (!has_d_ && (rows != width || (r > 0 && before != width)))
        reject(where + " has " + std::to_string(rows) + " detectors from " +
               std::to_string(width) +
               " measurements, but without D_sparse "
               "every round has one measurement per detector");
      if (!has_d_)
        continue;
      for (const auto index : *d) {
        if (index == -1)
          ++counted;
        else if (static_cast<std::size_t>(index) >= before + width)
          reject(where + ": its phase's D_sparse reads measurement " +
                 std::to_string(index) + ", past the " +
                 std::to_string(before + width) +
                 " of its round and the one before");
      }
      if (counted != rows)
        reject(where + ": its phase's D_sparse has " + std::to_string(counted) +
               " rows for " + std::to_string(rows) + " detectors");
    } // end - for(r)

  if (read_logical_) {
    if (t.readout().empty())
      reject("read_logical needs the last phase's m2o_sparse");
    for (const auto index : t.readout()) {
      if (index == -1)
        ++readout_rows;
      else if (static_cast<std::size_t>(index) >= last_width_)
        reject("the last phase's m2o_sparse reads measurement " +
               std::to_string(index) + ", past its round's " +
               std::to_string(last_width_));
    }
    logical_.reserve(readout_rows);
  }

  // Buffers. Raw rounds keep those not yet formed, with the one before them
  // and the one being cut: the suffix, and the rounds the readout round may
  // have begun in.
  staged_slots_ = suffix_rounds_ + 4 + last_width_ / min_forward_width_;
  staged_.assign(staged_slots_ * max_width_, 0);
  staged_width_.assign(staged_slots_, 0);
  const auto reach = static_cast<std::size_t>(highest - lowest) +
                     std::max<std::size_t>(suffix_rounds_, 1);
  const auto ring = power_of_two_at_least(
      reach + (max_views + 1) * static_cast<std::size_t>(period_) +
      (record_limit_ + max_width_) / min_forward_width_ + 1 + staged_slots_);
  ring_.assign(ring * max_rows_, 0);
  ring_rows_.assign(ring, 0);
  ring_round_.assign(ring, -1);
  ring_first_row_.assign(ring, 0);
  ring_mask_ = ring - 1;
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    pools_.emplace_back(layers_[l].max_views);
    for (std::size_t s = 0; s < layers_[l].max_views; ++s) {
      pools_.back()[s].layer = l;
      pools_.back()[s].slot = s;
    }
    out_.emplace_back(layers_[l].max_views, false);
  }
  publication_.views.assign(layers_.size(), nullptr);
  final_instances_.resize(layers_.size());
  reset();
} // end - ingest_core()

// Before FINAL the readout round is never cut, so every round after the
// prefix is bulk-wide: the template keeps every suffix round but the last
// as wide as bulk.
std::size_t ingest_core::forward_width(std::size_t r) const {
  return r < prefix_rounds_ ? prefix_widths_[r] : bulk_width_;
}

// Unpublished windows start no earlier than the next index's, and a window
// starts no earlier than the least first round of its layers' spans.
std::int64_t ingest_core::first_held_round() const {
  auto held = static_cast<std::int64_t>(formed_);
  if (!done())
    for (std::size_t l = 0; l < layers_.size(); ++l) {
      const auto &request = layers_[l].request;
      held = std::min(held, std::max<std::int64_t>(
                                0, request.offset + next_index_ * period_ +
                                       spans_[l].begin));
    }
  for (std::size_t l = 0; l < pools_.size(); ++l)
    for (std::size_t s = 0; s < pools_[l].size(); ++s)
      if (out_[l][s])
        held = std::min(held, pools_[l][s].rounds.begin);
  return held;
} // end - first_held_round()

// Rounds [first held, end) must all fit.
bool ingest_core::has_room(std::size_t end) const {
  return static_cast<std::int64_t>(end) - first_held_round() <=
         static_cast<std::int64_t>(ring_mask_ + 1);
}

// A shot that does not repeat ends at a known round, which is its readout
// round, so input before FINAL may fill it but never completes it.
void ingest_core::cut(const std::uint8_t *data, std::size_t len) {
  while (len > 0) {
    const auto r = rounds_cut_;
    const bool readout =
        !context_->repeats() && r + 1 == prefix_rounds_ && !shot_rounds_;
    const auto width = readout ? last_width_ : forward_width(r);
    const auto take = std::min(len, width - partial_);
    std::copy_n(data, take,
                staged_.begin() + (r % staged_slots_) * max_width_ + partial_);
    partial_ += take;
    data += take;
    len -= take;
    if (partial_ < width)
      continue;
    if (readout) {
      if (len > 0)
        fail("a record runs past the shot's last round, " +
             rounds_text(prefix_rounds_) + " long");
      return;
    }
    complete_round(width);
  } // end - while(len)
  form_known_rounds();
} // end - cut()

// A round is complete; those whose phase is now known are formed.
void ingest_core::complete_round(std::size_t width) {
  staged_width_[rounds_cut_ % staged_slots_] = width;
  ++rounds_cut_;
  cut_bytes_ += width;
  partial_ = 0;
  form_known_rounds();
}

// The readout round must end at or past the last measurement received, so
// it starts no earlier than the first round whose start lies within its
// width of that measurement, and the shot has at least one round more than
// those before it. Each step back passes a round of at least the narrowest
// width, so the walk is a few rounds long.
std::size_t ingest_core::least_shot_rounds() const {
  const auto received = cut_bytes_ + partial_;
  const auto earliest = received > last_width_ ? received - last_width_ : 0;
  auto first = rounds_cut_ + 1; // starts past what has arrived
  auto start = cut_bytes_;      // where the round before `first` starts
  if (shot_rounds_)
    return *shot_rounds_;
  if (!context_->repeats())
    return context_->min_rounds();
  while (first > 0 && start >= earliest) {
    --first;
    if (first > 0)
      start -= forward_width(first - 1);
  }
  return std::max(first + 1, context_->min_rounds());
} // end - least_shot_rounds()

// A round after the prefix is bulk once the shot is known to hold the
// suffix after it.
void ingest_core::form_known_rounds() {
  const auto least = least_shot_rounds();
  while (formed_ < rounds_cut_ && (shot_rounds_ || formed_ < prefix_rounds_ ||
                                   formed_ + suffix_rounds_ < least)) {
    form_detectors(formed_);
    ++formed_;
  }
}

// Each detector is the XOR of the measurements its D_sparse row reads,
// indexed over the round before and then this round, as window maps read
// them; without D_sparse, of this round's measurement and the one before.
void ingest_core::form_detectors(std::size_t r) {
  const auto *current = staged_.data() + (r % staged_slots_) * max_width_;
  const auto *before =
      r == 0 ? nullptr
             : staged_.data() + ((r - 1) % staged_slots_) * max_width_;
  const auto before_width =
      r == 0 ? std::size_t{0} : staged_width_[(r - 1) % staged_slots_];
  const auto slot = r & ring_mask_;
  auto *out = ring_.data() + slot * max_rows_;
  const std::vector<std::int64_t> *d = bulk_d_;
  std::size_t rows = bulk_rows_;
  std::size_t row = 0;
  std::uint8_t value = 0;
  if (r < prefix_rounds_) {
    d = prefix_d_[r];
    rows = prefix_rows_[r];
  } else if (shot_rounds_ && r + suffix_rounds_ >= *shot_rounds_) {
    const auto j = r + suffix_rounds_ - *shot_rounds_;
    d = suffix_d_[j];
    rows = suffix_rows_[j];
  }
  if (has_d_) {
    for (const auto index : *d) {
      const auto i = static_cast<std::size_t>(index);
      if (index == -1) {
        out[row++] = value;
        value = 0;
      } else {
        value ^= (i < before_width ? before[i] : current[i - before_width]) & 1;
      }
    }
  } else {
    for (std::size_t j = 0; j < rows; ++j)
      out[j] = (current[j] ^ (before ? before[j] : 0)) & 1;
  }
  ring_rows_[slot] = rows;
  ring_round_[slot] = static_cast<std::int64_t>(r);
  ring_first_row_[slot] = rows_formed_;
  rows_formed_ += rows;
} // end - form_detectors()

// The shot's last measurements are its readout round, and the rounds before
// it are whole. It may begin in the FINAL record or in rounds already cut,
// none of them formed, since a round is formed only once the readout round
// cannot have begun in it; a shot whose last phase is bulk forms its readout
// round as bulk, which it is. Counting the rounds first fixes the shot's
// length, so every round's phase is known before any is formed, and a
// record that ends no shot changes nothing.
ingest_status ingest_core::enqueue(const std::uint8_t *data, std::size_t len,
                                   bool final) {
  check_open();
  if (len > record_limit_)
    fail("a record of " + std::to_string(len) +
         " measurements is longer than one period of rounds, " +
         std::to_string(record_limit_));
  if (len > 0 && !data)
    fail("a record of " + std::to_string(len) + " measurements has no data");
  if (!final) {
    // Checked before any round is cut, so a rejected record forms none.
    if (!context_->repeats() &&
        cut_bytes_ + partial_ + len > fixed_measurements_)
      fail("a record runs past the shot's last round, " +
           rounds_text(prefix_rounds_) + " long");
    if (!has_room(rounds_cut_ + (partial_ + len) / min_forward_width_))
      return ingest_status::deferred;
    cut(data, len);
    return ingest_status::accepted;
  }

  const auto total = cut_bytes_ + partial_ + len;
  std::size_t r = rounds_cut_;
  std::size_t at = cut_bytes_; // where round r starts
  std::size_t head = 0;
  if (total < last_width_)
    fail("the shot holds " + std::to_string(total) +
         " measurements, fewer than its readout round's " +
         std::to_string(last_width_));
  const auto readout_start = total - last_width_;
  // Forward from the round being cut, or back over rounds already cut. A
  // shot that does not repeat has a last round, which the readout round
  // must be: rounds before it are all prefix rounds, of known width.
  while (at < readout_start) {
    const auto width = forward_width(r);
    if (!context_->repeats() && r + 1 >= prefix_rounds_)
      fail("the FINAL record runs past the shot's last round, " +
           rounds_text(prefix_rounds_) + " long");
    if (readout_start - at < width)
      fail("the FINAL record ends mid-round: round " + std::to_string(r) +
           " has " + std::to_string(readout_start - at) + " of its " +
           std::to_string(width) + " measurements before the readout round");
    at += width;
    ++r;
  }
  while (at > readout_start && r > 0)
    at -= forward_width(--r);
  if (at != readout_start)
    fail("the FINAL record ends mid-round: round " + std::to_string(r) +
         " has " + std::to_string(readout_start - at) + " of its " +
         std::to_string(forward_width(r)) +
         " measurements before the readout round");
  const auto R = r + 1;
  if (context_->repeats() ? R < context_->min_rounds()
                          : R != context_->min_rounds())
    fail("the FINAL record ends the shot at " + rounds_text(R) +
         ", but the template's shots have " +
         (context_->repeats() ? "at least " : "exactly ") +
         std::to_string(context_->min_rounds()));
  if (r < formed_ && suffix_rounds_ > 0)
    fail("the FINAL record ends the shot at " + rounds_text(R) + ", but " +
         rounds_text(formed_) + " are already formed as bulk");
  if (!has_room(R))
    return ingest_status::deferred;

  shot_rounds_ = R;
  auto *readout = staged_.data() + (r % staged_slots_) * max_width_;
  if (r >= rounds_cut_) {
    // The readout round starts at the round being cut or in the record.
    if (r > rounds_cut_) {
      head = readout_start - cut_bytes_ - partial_;
      cut(data, head);
    }
    std::copy_n(data + head, len - head, readout + partial_);
    complete_round(last_width_);
  } else if (r >= formed_) {
    // It began in rounds already cut: their measurements, the partial
    // round's and the record's join round r's in its slot.
    std::size_t filled = forward_width(r);
    for (auto q = r + 1; q < rounds_cut_; ++q) {
      std::copy_n(staged_.data() + (q % staged_slots_) * max_width_,
                  forward_width(q), readout + filled);
      filled += forward_width(q);
    }
    std::copy_n(staged_.data() + (rounds_cut_ % staged_slots_) * max_width_,
                partial_, readout + filled);
    std::copy_n(data, len, readout + filled + partial_);
    rounds_cut_ = r;
    cut_bytes_ = readout_start;
    partial_ = 0;
    complete_round(last_width_);
  }
  if (read_logical_) {
    const auto *readout =
        staged_.data() + ((R - 1) % staged_slots_) * max_width_;
    std::uint8_t value = 0;
    logical_.clear();
    for (const auto index : context_->readout()) {
      if (index == -1) {
        logical_.push_back(value);
        value = 0;
      } else {
        value ^= readout[static_cast<std::size_t>(index)] & 1;
      }
    }
  }
  index_end_ = next_index_;
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    final_instances_[l] = window_instances(layers_[l].request, R);
    index_end_ = std::max(index_end_, final_instances_[l].end);
  }
  return ingest_status::accepted;
} // end - enqueue()

// Before FINAL a layer has a window at every index from its first; after
// it, at the indices whose commit zone the shot holds, found at FINAL.
bool ingest_core::window_exists(std::size_t l, std::int64_t k) const {
  if (!shot_rounds_)
    return k >= first_k_[l];
  return k >= final_instances_[l].begin && k < final_instances_[l].end;
}

// A window that is not terminal in the shortest shot the measurements allow
// is terminal in no longer one; its rounds end before its reach, so once
// they are formed it is complete.
bool ingest_core::window_complete(std::size_t l, std::int64_t k,
                                  std::size_t least) const {
  const auto &request = layers_[l].request;
  if (shot_rounds_)
    return true;
  return !window_variant(*context_, request, k, least).terminal &&
         request.offset + k * period_ + spans_[l].end <=
             static_cast<std::int64_t>(formed_);
}

// Index by index, in order: an index waits for its last window and for a
// view in every pool it needs. After FINAL an index with no window is
// passed over, and a failed shot hands nothing over.
const ingest_publication *ingest_core::poll() {
  const auto least = least_shot_rounds();
  if (failed_)
    return nullptr;
  for (;;) {
    const auto k = next_index_;
    bool any = false;
    if (shot_rounds_ && k >= index_end_)
      return nullptr;
    for (std::size_t l = 0; l < layers_.size(); ++l) {
      if (!window_exists(l, k))
        continue;
      if (!window_complete(l, k, least))
        return nullptr;
      if (std::find(out_[l].begin(), out_[l].end(), false) == out_[l].end())
        return nullptr;
      any = true;
    }
    ++next_index_;
    if (!any)
      continue;
    publication_.k = k;
    for (std::size_t l = 0; l < layers_.size(); ++l) {
      const auto &request = layers_[l].request;
      const auto anchor = request.offset + k * period_;
      const auto end = shot_rounds_ ? static_cast<std::int64_t>(*shot_rounds_)
                                    : std::numeric_limits<std::int64_t>::max();
      std::size_t s = 0;
      publication_.views[l] = nullptr;
      if (!window_exists(l, k))
        continue;
      while (out_[l][s])
        ++s;
      auto &view = pools_[l][s];
      out_[l][s] = true;
      view.k = k;
      view.rounds = {std::max<std::int64_t>(0, anchor + spans_[l].begin),
                     std::min(end, anchor + spans_[l].end)};
      view.variant = window_variant(*context_, request, k, least);
      view.first_row =
          ring_first_row_[static_cast<std::size_t>(view.rounds.begin) &
                          ring_mask_];
      view.source = this;
      publication_.views[l] = &view;
    } // end - for(l)
    return &publication_;
  } // end - for(;;)
} // end - poll()

// The view must be one of the pools', and out.
void ingest_core::release(const window_view &view) {
  if (view.layer >= pools_.size() || view.slot >= pools_[view.layer].size() ||
      &pools_[view.layer][view.slot] != &view || !out_[view.layer][view.slot])
    throw std::logic_error("ingest: release of a view that is not out");
  out_[view.layer][view.slot] = false;
}

// The slot of a round formed this shot holds it until a later round takes
// the slot.
std::span<const std::uint8_t> ingest_core::detectors(std::int64_t r) const {
  const auto slot = static_cast<std::size_t>(r) & ring_mask_;
  if (r < 0 || r >= static_cast<std::int64_t>(formed_) ||
      ring_round_[slot] != r)
    throw std::out_of_range("ingest: round " + std::to_string(r) +
                            "'s detectors are not in the ring");
  return {ring_.data() + slot * max_rows_, ring_rows_[slot]};
}

// A round a view out holds keeps its place in the ring, which enqueue()
// writes only for rounds no view holds; formed_, which it does write, is
// not read.
std::span<const std::uint8_t>
ingest_core::view_detectors(const window_view &view, std::int64_t r) const {
  const auto slot = static_cast<std::size_t>(r) & ring_mask_;
  if (r < view.rounds.begin || r >= view.rounds.end || ring_round_[slot] != r)
    throw std::out_of_range("ingest: window " + std::to_string(view.k) +
                            "'s view does not hold round " + std::to_string(r) +
                            " in the ring");
  return {ring_.data() + slot * max_rows_, ring_rows_[slot]};
}

// Round by round, as view_detectors() reads them.
void ingest_core::copy_window(const window_view &view, window_rows &out) const {
  out.k = view.k;
  out.rounds = view.rounds;
  out.variant = view.variant;
  out.first_row = view.first_row;
  out.detectors.clear();
  for (auto r = view.rounds.begin; r < view.rounds.end; ++r) {
    const auto round = view_detectors(view, r);
    out.detectors.insert(out.detectors.end(), round.begin(), round.end());
  }
}

// Every index is handed over once FINAL fixes the last.
bool ingest_core::done() const {
  return shot_rounds_ && next_index_ >= index_end_;
}

// The buffers keep their contents: a round's are written before they are
// read.
void ingest_core::reset() {
  for (const auto &pool : out_)
    if (std::find(pool.begin(), pool.end(), true) != pool.end())
      throw std::logic_error("ingest: reset() with views out");
  rounds_cut_ = 0;
  cut_bytes_ = 0;
  partial_ = 0;
  formed_ = 0;
  rows_formed_ = 0;
  shot_rounds_.reset();
  next_index_ = first_index();
  index_end_ = 0;
  logical_.clear();
  failed_ = false;
} // end - reset()

// A malformed record leaves the shot unusable until reset().
void ingest_core::fail(const std::string &what) {
  failed_ = true;
  throw std::invalid_argument("ingest: " + what);
}

// Input waits for reset() after a failure or after FINAL.
void ingest_core::check_open() const {
  if (failed_)
    throw std::logic_error("ingest: reset() is needed after a failure");
  if (shot_rounds_)
    throw std::logic_error(
        "ingest: input after FINAL; reset() starts the next shot");
}

} // namespace cudaq::qec
