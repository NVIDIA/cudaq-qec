/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the ingest node, the only raw-fed node of a windowed
// graph. It cuts records into rounds through ingest_core and publishes each
// index's windows, one output port per layer, copied as window_rows so that
// no view of its ring is ever out. A logical_readout node, not ingest, reads
// the shot's raw logical value. Never included from a public header.

#pragma once

#include "decoding_task_graph_node.h"
#include "decoding_task_ingest_core.h"
#include "cudaq/qec/window_request.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// Publications are numbered from 0, the first index any layer has. One
/// record may ready several indices, and FINAL readies all that remain, so
/// enqueue() makes the first current and publish_next() hands over each of
/// the rest; enqueue() and flush() reject a call while any still wait.
/// Not thread-safe: one thread calls every member.
class ingest_node final : public task_graph_node,
                          public streaming_task_graph_node {
public:
  /// @param context The experiment template.
  /// @param windows The layers, in port order: each one's port name and
  /// request.
  /// @throws std::invalid_argument as ingest_core's constructor does.
  ingest_node(std::shared_ptr<const experiment_template> context,
              const std::vector<experimental::ingest_window> &windows);

  /// Make the next waiting publication current; false when none waits,
  /// leaving the last one current until enqueue() or flush().
  bool publish_next() override;
  /// The current publication's index on a window port.
  std::optional<std::int64_t> published_index(std::size_t slot) const override;

  /// @throws std::logic_error: the node streams; it has no one-shot path.
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override;
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return outputs_; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override;
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override;
  streaming_task_graph_node *streaming() override { return this; }
  /// A shot's partial rounds and publications carry across deliveries, so
  /// copies would split one stream.
  bool supports_fire_pool() const override { return false; }
  bool has_one_shot_path() const override { return false; }

  /// Take one chunk of a record; true when it readied a publication, which
  /// is then current.
  /// @throws std::logic_error while publications wait to be handed over,
  /// or as ingest_core::enqueue() does; std::invalid_argument as it does.
  bool enqueue(const uint8_t *msyn, std::size_t len, round_type type) override;
  /// Note the shot's end; FINAL has readied every window, so it publishes
  /// nothing.
  /// @throws std::logic_error while publications wait to be handed over.
  bool flush() override;
  /// The current publication's value on @p slot: a layer's window, or null
  /// where the layer has none at its index.
  const edge_value *streaming_output(std::size_t slot) const override;
  /// Every publication reports its coverage exactly: a window's carries no
  /// delta and none.
  bool exact_interval_coverage() const override { return true; }
  /// Windows carry no correction.
  const uint8_t *corrections() const override { return nullptr; }
  /// After the shot is flushed, start the next shot, as a caller does
  /// between shots; before, there is nothing to clear.
  void clear_corrections() override;
  void reset_streaming_state() override;
  std::size_t num_msyn_per_decode() const override {
    return core_.record_limit();
  }
  /// A typical round's measurements, not the record limit's period of them.
  std::size_t num_msyn_per_round() const override {
    return core_.round_width();
  }
  /// A record's period of rounds, committing nothing: the base derives it
  /// by dividing the record limit by a round, which rounds of unequal width
  /// need not divide.
  std::optional<window_spec> streaming_schedule() const override {
    const auto period = static_cast<std::uint32_t>(core_.period());
    return window_spec{period, period, 0, 0};
  }
  /// None: windows carry no correction.
  std::size_t num_observables() const override { return 0; }
  std::optional<std::size_t> raw_chunk_limit() const override {
    return core_.record_limit();
  }
  std::size_t max_publications_per_delivery() const override {
    return core_.max_ready_per_record();
  }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }

private:
  /// One ready index: its number, and each layer's window, where it has one.
  struct publication {
    std::int64_t index = 0;
    std::vector<edge_value> windows; ///< window_rows, one per layer
    std::vector<std::uint8_t> present;
  };

  /// Copy every ready index's windows out of the ring, releasing the views.
  void collect();
  /// Throws std::logic_error, naming @p call, while publications wait.
  void check_handed_over(const char *call) const;
  /// Forget the publications handed over.
  void clear_publications();

  // No publication is current.
  static constexpr std::size_t none = std::numeric_limits<std::size_t>::max();

  ingest_core core_;
  std::size_t layers_ = 0;
  std::vector<std::string> outputs_;
  // Ready publications, reused shot to shot: filled_ of them hold this
  // record's, and current_ is the one handed over.
  std::vector<publication> ready_;
  std::size_t filled_ = 0;
  std::size_t current_ = none;
  bool flushed_ = false; ///< flush() followed FINAL
};

} // namespace cudaq::qec
