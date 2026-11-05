/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the logical_readout node. It reads a shot's raw logical
// value: each reading round's measurements through its phase's m2o_sparse,
// XORed per observable. Prefix reading rounds are read as they arrive, suffix
// ones at FINAL. Never included from a public header.

#pragma once

#include "decoding_task_experiment_template.h"
#include "decoding_task_graph_node.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// Reads the raw logical value. Streaming, it takes records whole: a record
/// completing a prefix reading round publishes the value so far, and flush()
/// after FINAL adds the suffix's and publishes once. Not thread-safe.
class logical_readout_node final : public task_graph_node,
                                   public streaming_task_graph_node {
public:
  /// @param context The experiment template.
  /// @throws std::invalid_argument if no phase of the template carries
  /// m2o_sparse.
  explicit logical_readout_node(
      std::shared_ptr<const experiment_template> context);

  /// The logical value of the shot on input 0.
  /// @throws std::invalid_argument if the shot is shorter than its reading
  /// rounds span.
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override;
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override;
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override;
  std::optional<std::size_t> output_value_size(std::size_t slot) const override;
  streaming_task_graph_node *streaming() override { return this; }
  /// The value carries across deliveries; copies would split it.
  bool supports_fire_pool() const override { return false; }
  bool reads_logical() const override { return true; }
  /// Each observable's last reading round if it is in the prefix; else none
  /// (fixed only at FINAL).
  std::vector<std::optional<std::uint64_t>>
  logical_completion_rounds() const override {
    return completion_;
  }

  /// Read prefix reading rounds and keep the last suffix-width measurements;
  /// a FINAL chunk ends the shot. True when a prefix reading round completes.
  /// Allocates nothing.
  /// @throws std::logic_error after FINAL until the next shot;
  /// std::invalid_argument if the shot is shorter than its reading rounds.
  bool enqueue(const uint8_t *msyn, std::size_t len, round_type type) override;
  /// After FINAL, add the suffix's readouts and publish, once.
  bool flush() override;
  /// The value read so far while a publication is current; else null.
  const edge_value *streaming_output(std::size_t slot) const override;
  /// [0, end of the last published prefix reading round), or the whole shot
  /// after flush(); none before any publication.
  std::optional<committed_coverage> streaming_coverage() const override;
  /// The value read so far; zero before any reading round.
  const uint8_t *corrections() const override { return corrections_.data(); }
  /// Starts the next shot once this one has ended; a no-op before.
  void clear_corrections() override;
  void reset_streaming_state() override;
  /// A typical round's measurements; the node keeps no window.
  std::size_t num_msyn_per_decode() const override { return round_width_; }
  /// A typical round's measurements, as ingest reports them.
  std::size_t num_msyn_per_round() const override { return round_width_; }
  /// Publishes as reading rounds complete, not on a schedule.
  std::optional<window_spec> streaming_schedule() const override {
    return std::nullopt;
  }
  /// The template's observables.
  std::size_t num_observables() const override { return corrections_.size(); }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }

private:
  /// A round that reads observables: its position and, per measurement, the
  /// observables it flips (CSR).
  struct reading_round {
    std::uint64_t round = 0; ///< Its round; prefix rounds only
    std::size_t begin = 0;   ///< From the shot's start, or the suffix's
    std::size_t width = 0;
    std::vector<std::size_t> start;         ///< width + 1 offsets
    std::vector<std::uint32_t> observables; ///< Read by each measurement
  };

  /// Append round @p r's reading of @p m2o to @p out if it reads anything,
  /// marking the observables it reads in @p reads.
  static void add_reading_round(const std::vector<std::int64_t> &m2o,
                                std::uint64_t r, std::size_t begin,
                                std::size_t width,
                                std::vector<reading_round> &out,
                                std::vector<std::uint8_t> &reads);
  /// XOR into @p value the observables @p round's measurement @p local flips.
  static void add_reading(const reading_round &round, std::size_t local,
                          std::vector<std::uint8_t> &value);

  std::shared_ptr<const experiment_template> context_;
  std::size_t round_width_ = 0; ///< A typical round's
  std::size_t head_ = 0;        ///< Measurements of the prefix rounds
  std::size_t tail_ = 0;        ///< Measurements of the suffix rounds
  std::vector<reading_round> head_rounds_; ///< In round order
  std::vector<reading_round> tail_rounds_; ///< From the suffix's start
  std::vector<std::optional<std::uint64_t>> completion_; ///< By observable

  // Ring of the last tail_ measurements; oldest at next_ once seen_ >= tail_.
  std::vector<std::uint8_t> kept_;
  std::size_t next_ = 0;
  std::size_t seen_ = 0;
  std::size_t next_round_ = 0; ///< The next prefix reading round to complete
  std::uint64_t read_end_ = 0; ///< Coverage end of the value read so far
  std::vector<std::uint8_t> read_; ///< The prefix's readouts so far
  bool ended_ = false;             ///< A FINAL chunk ended the shot
  bool published_ = false;         ///< flush() published the value this shot
  bool current_ = false;           ///< A publication is current
  edge_value logical_;             ///< logical_outcome
  std::vector<std::uint8_t> corrections_;
};

} // namespace cudaq::qec
