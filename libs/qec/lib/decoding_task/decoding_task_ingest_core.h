/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the ingest node's core, apart from the executor. It cuts
// a stream of raw measurement records into rounds, forms each round's
// detectors once into a ring of fixed size, and hands every layer's window k
// over together, as views of that ring, once all of them are complete. At
// FINAL it settles the shot's length and, on request, reads the raw logical
// value. A caller polls for what is ready; publishing it to other nodes is
// the executor's part. Never included from a public header.

#pragma once

#include "decoding_task_decoding_window.h"
#include "decoding_task_experiment_template.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cudaq::qec {

class ingest_core;

/// One layer of an ingest node: its windows' request, and how many of its
/// views may be out at once.
struct ingest_layer {
  window_request request;
  std::size_t max_views = 1; ///< The size of the layer's view pool
};

/// A layer's window k as ingest hands it over: where its own rows lie and
/// which of the layer's models it is. Its rounds' detectors stay in the ring
/// until it is released.
struct window_view {
  std::size_t layer = 0; ///< Index into the ingest node's layers
  std::int64_t k = 0;
  round_range rounds;        ///< The window's rounds, clipped to the shot
  variant_key variant;       ///< Its model among the layer's variants
  std::size_t first_row = 0; ///< Its first round's first row in the shot
  const ingest_core *source = nullptr; ///< The core whose ring holds its rows
  std::size_t slot = 0;                ///< Its place in the layer's view pool
};

/// Every layer's window k, handed over together.
struct ingest_publication {
  std::int64_t k = 0;
  std::vector<const window_view *> views; ///< One per layer; null for none
};

/// Whether a record was taken.
enum class ingest_status {
  accepted, ///< The record is ingested
  deferred, ///< Nothing was taken: the ring has no room for it yet
};

/// Rounds, detectors and window views for one ingest node. The rounds of a
/// shot are prefix, then bulk, then suffix rounds, the last of them the
/// readout round. Before FINAL, rounds are cut at their widths before the
/// readout round, and the readout round must end at or past the last
/// measurement received, which bounds the shot's length from below. A round
/// is known to be bulk once that bound puts it before the suffix, and only
/// then are its detectors formed, so measurements of the readout round that
/// arrive early are never formed as a bulk round. A FINAL record fixes the
/// shot's length and so every remaining round's phase. A round's
/// detectors are its phase's D_sparse rows over the previous round's
/// measurements and its own, or without D_sparse the memory convention.
///
/// Index k is ready once every layer's window k is complete: before FINAL,
/// once its rounds' detectors are formed and its variant can no longer be
/// terminal; after FINAL, always, with the windows clipped to the shot. A
/// view copies nothing: it names rounds whose detectors stay in the ring
/// while an unpublished window or a view out holds them.
///
/// One thread enqueues, polls, releases and resets. Another thread may read
/// a view's rows through view_detectors() while that one runs, between the
/// view's hand-over and its release, when both pass through something that
/// orders the two threads, such as a queue or a lock. The ring does not
/// reuse a round that a view out holds.
class ingest_core {
public:
  /// @param context The experiment template.
  /// @param layers The layers, which share one period.
  /// @param read_logical Read the raw logical value at FINAL, through the
  /// last phase's m2o_sparse.
  /// @throws std::invalid_argument naming the operation if there is no
  /// layer, a request is invalid, the periods differ, a view pool is empty,
  /// a round holds no measurements, a round's D_sparse does not fit its
  /// detectors or its rounds' measurements, or read_logical is set and the
  /// last phase's m2o_sparse is missing or reads past its round.
  ingest_core(std::shared_ptr<const experiment_template> context,
              std::vector<ingest_layer> layers, bool read_logical);

  /// Take one record of raw measurements, one byte each. A record may cut a
  /// round anywhere, and the readout round may arrive in records before the
  /// FINAL one, which ends the shot and may be empty: the readout round is
  /// the shot's last measurements. Allocates nothing.
  /// @return deferred, taking nothing, when the ring has no room for the
  /// record's rounds until views are polled or released.
  /// @throws std::invalid_argument if the record is longer than
  /// record_limit(), or a FINAL record does not end a shot the template
  /// admits; std::logic_error after FINAL or a failure, until reset().
  ingest_status enqueue(const std::uint8_t *data, std::size_t len, bool final);

  /// The next index whose windows are all complete, with a view of each, or
  /// null when none is ready, a pool it needs is empty, or the shot failed.
  /// Valid until the next call. Allocates nothing.
  const ingest_publication *poll();

  /// Return @p view to its pool, once its consumer is done with its rows.
  /// @throws std::logic_error if the view is not out.
  void release(const window_view &view);

  /// The detectors of round @p r, while a view out or an unpublished window
  /// holds it. Not to be called while enqueue() runs; a consumer on another
  /// thread uses view_detectors().
  /// @throws std::out_of_range if round r's detectors are not in the ring.
  std::span<const std::uint8_t> detectors(std::int64_t r) const;

  /// The detectors of round @p r of @p view, a view out. Reads only that
  /// round's place in the ring, so a consumer may call it while enqueue()
  /// and poll() run on another thread.
  /// @throws std::out_of_range if the view does not hold round r in the
  /// ring.
  std::span<const std::uint8_t> view_detectors(const window_view &view,
                                               std::int64_t r) const;

  /// Copy @p view's window into @p out: its index, rounds, variant, first
  /// row and own rows' detectors, so the view may be released at once.
  /// Allocates nothing once @p out has held a window as large.
  /// @throws std::out_of_range as view_detectors() does.
  void copy_window(const window_view &view, window_rows &out) const;

  /// The rounds in the shot, once a FINAL record is taken.
  std::optional<std::size_t> shot_rounds() const { return shot_rounds_; }
  /// The raw logical value, m2o times the readout round's measurements, once
  /// a FINAL record is taken with read_logical set; empty otherwise.
  const std::vector<std::uint8_t> &logical() const { return logical_; }
  /// Whether a FINAL record is taken and every index handed over.
  bool done() const;

  /// The most measurements one record may hold: a period of the widest
  /// rounds.
  std::size_t record_limit() const { return record_limit_; }
  /// The measurements in a typical round: the repeating round's, or, for a
  /// template that does not repeat, the widest round's.
  std::size_t round_width() const {
    return bulk_width_ != 0 ? bulk_width_ : max_width_;
  }
  /// The layers' shared period, in rounds: the most one record holds.
  std::int64_t period() const { return period_; }
  /// The rounds the ring holds, fixed at construction.
  std::size_t ring_rounds() const { return ring_mask_ + 1; }
  /// The first index any layer has a window at, fixed by the layout.
  /// The most indices one record can ready: a record holds at most a
  /// period of rounds, which completes at most two indices, while a FINAL
  /// record readies every index left in the ring.
  std::size_t max_ready_per_record() const {
    return ring_rounds() / static_cast<std::size_t>(period_) + 2;
  }
  std::int64_t first_index() const {
    return *std::min_element(first_k_.begin(), first_k_.end());
  }

  /// Start a new shot.
  /// @throws std::logic_error if a view is out.
  void reset();

private:
  /// Measurements in round r, cut before the shot's length is known.
  std::size_t forward_width(std::size_t r) const;
  /// The fewest rounds the shot can have: its length once FINAL is taken,
  /// else a length whose readout round ends at or past the last measurement
  /// received.
  std::size_t least_shot_rounds() const;
  /// Form the detectors of every round cut whose phase is known.
  void form_known_rounds();
  /// The first round still held: by the next unpublished index, or a view
  /// out.
  std::int64_t first_held_round() const;
  /// Whether the ring can take rounds up to @p end without dropping a round
  /// still held.
  bool has_room(std::size_t end) const;
  /// Append @p len measurements to the round being cut, completing rounds
  /// and forming the detectors of those whose phase is known.
  void cut(const std::uint8_t *data, std::size_t len);
  /// Complete the round being cut, of @p width measurements.
  void complete_round(std::size_t width);
  /// Form round r's detectors into the ring.
  void form_detectors(std::size_t r);
  /// Whether layer @p l's window k is complete, in a shot of at least
  /// @p least rounds.
  bool window_complete(std::size_t l, std::int64_t k, std::size_t least) const;
  /// Whether layer @p l has a window k.
  bool window_exists(std::size_t l, std::int64_t k) const;
  /// Throws std::invalid_argument for @p what, leaving the core failed.
  [[noreturn]] void fail(const std::string &what);
  /// Throws std::logic_error unless input is allowed.
  void check_open() const;

  std::shared_ptr<const experiment_template> context_;
  std::vector<ingest_layer> layers_;
  bool read_logical_ = false;

  // The template's rounds, by kind. A spec that does not repeat has only
  // prefix rounds.
  bool has_d_ = false; ///< Rounds read D_sparse, not the memory convention
  std::size_t prefix_rounds_ = 0;
  std::size_t suffix_rounds_ = 0;
  std::int64_t period_ = 0;
  std::vector<std::size_t> prefix_widths_;
  std::vector<std::size_t> prefix_rows_;
  std::vector<const std::vector<std::int64_t> *> prefix_d_;
  std::size_t bulk_width_ = 0;
  std::size_t bulk_rows_ = 0;
  const std::vector<std::int64_t> *bulk_d_ = nullptr;
  std::vector<std::size_t> suffix_rows_;
  std::vector<const std::vector<std::int64_t> *> suffix_d_;
  std::size_t last_width_ = 0;
  std::size_t min_width_ = 0;
  std::size_t min_forward_width_ = 0;  ///< The narrowest round cut before FINAL
  std::size_t fixed_measurements_ = 0; ///< A fixed shot's, when not repeating
  std::size_t max_width_ = 0;
  std::size_t max_rows_ = 0;
  std::size_t record_limit_ = 0;

  // Per layer: the first index with a window, and the rounds a window spans
  // relative to its anchor.
  std::vector<std::int64_t> first_k_;
  std::vector<index_range> final_instances_; ///< Once FINAL fixes them
  std::vector<round_range> spans_;

  // Raw rounds: the round being cut and those before it whose detectors are
  // not yet formed, with the one before them, whose measurements are v_in.
  std::vector<std::uint8_t> staged_;
  std::vector<std::size_t> staged_width_;
  std::size_t staged_slots_ = 0;

  // The ring of formed rounds, by round modulo its power-of-two size.
  std::vector<std::uint8_t> ring_;
  std::vector<std::size_t> ring_rows_;
  std::vector<std::int64_t> ring_round_;
  std::vector<std::size_t> ring_first_row_; ///< Each round's first shot row
  std::size_t ring_mask_ = 0;

  // View pools, one per layer, and the publication poll() fills.
  std::vector<std::vector<window_view>> pools_;
  std::vector<std::vector<bool>> out_;
  ingest_publication publication_;

  // The shot.
  std::size_t rounds_cut_ = 0;  ///< Complete rounds
  std::size_t cut_bytes_ = 0;   ///< Their measurements
  std::size_t partial_ = 0;     ///< Measurements of the round being cut
  std::size_t formed_ = 0;      ///< Rounds whose detectors are formed
  std::size_t rows_formed_ = 0; ///< Their detectors
  std::optional<std::size_t> shot_rounds_;
  std::int64_t next_index_ = 0;
  std::int64_t index_end_ = 0; ///< Past the last index, once FINAL is taken
  std::vector<std::uint8_t> logical_;
  bool failed_ = false;
};

} // namespace cudaq::qec
