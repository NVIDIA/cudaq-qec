/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the decode_window node, which decodes one layer of a
// windowed graph through decode_window_core. Its instance k fires once its
// window and the sends its receives name are joined, and publishes the
// frame update its commit zone makes, with that zone's rounds, and each of
// its sends. Never included from a public header.

#pragma once

#include "decoding_task_decode_window_core.h"
#include "decoding_task_graph_node.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// Joins its inputs by index: input 0 is the window, required, and input
/// 1 + i is receive_i's send, absent where its producer instance does not
/// exist. Its only state across fires is the shot's index offset, which
/// every instance of a shot shares, so an executor may run copies of it at
/// once, each with its own core and decoders. Not thread-safe: one thread
/// at a time calls every member of a copy.
class decode_window_node final : public task_graph_node,
                                 public streaming_task_graph_node,
                                 public indexed_join_node {
public:
  /// @param store The store holding every variant of @p request.
  /// @param context The experiment template.
  /// @param request The layer's request, from the ingest port it reads.
  /// @param decoder_name The inner decoder, and @p params its parameters.
  /// @param num_receives Receive ports: receive_0 ... receive_{n-1}.
  /// @param verify What a fire does with a solution that does not satisfy
  /// its window's detectors. With report, only the core's fire record says
  /// whether it held; the node publishes nothing of it yet.
  /// @throws as decode_window_core's constructor does.
  decode_window_node(const data_artifact_store &store,
                     std::shared_ptr<const experiment_template> context,
                     window_request request, const std::string &decoder_name,
                     const cudaqx::heterogeneous_map &params,
                     std::size_t num_receives, verify_residual_mode verify);

  /// Whether an instance exists only with input @p slot: its window.
  bool requires_input(std::size_t slot) const override { return slot == 0; }
  /// Fire instance @p k: its window on inputs[0] and each receive's send,
  /// or null. Publishes its frame update, commit zone and sends.
  /// @throws std::invalid_argument if there are not one input per port,
  /// the window is absent, an input holds another type, or k is not the
  /// window's index less the offset of this shot's earlier fires, or as
  /// decode_window_core::fire() does, as do its other exceptions. A fire
  /// that throws publishes nothing.
  bool fire(std::int64_t k,
            const std::vector<const edge_value *> &inputs) override;
  /// The index of the instance last fired, on every port, while its
  /// publication stands: the frame update and each send are instance k's.
  std::optional<std::int64_t> published_index(std::size_t slot) const override;
  /// Every publication reports the exact rounds its frame update commits.
  bool exact_interval_coverage() const override { return true; }

  /// @throws std::logic_error: the node joins by index; it has no one-shot
  /// path.
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override;
  std::vector<std::string> inputs() const override { return inputs_; }
  std::vector<std::string> outputs() const override { return outputs_; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override;
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override;
  streaming_task_graph_node *streaming() override { return this; }
  indexed_join_node *indexed_join() override { return this; }

  /// @throws std::logic_error: the node takes windows, not raw
  /// measurements.
  bool enqueue(const uint8_t *msyn, std::size_t len, round_type type) override;
  /// The last fire's frame update on `output`, or a send on its port.
  const edge_value *streaming_output(std::size_t slot) const override;
  /// The last fire's commit zone: the rounds its frame update covers.
  std::optional<committed_coverage> streaming_coverage() const override;
  /// Every fire's frame update since the corrections were last cleared,
  /// XORed together.
  const uint8_t *corrections() const override { return corrections_.data(); }
  void clear_corrections() override;
  void reset_streaming_state() override;
  /// It reads no raw measurements.
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return corrections_.size(); }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot < inputs_.size();
  }
  /// Every output follows the window input's schedule. Publications follow
  /// window indices, not rounds; outputs fold by exact coverage, which needs
  /// no shared schedule, so this only orders the graph's compile.
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override;

private:
  decode_window_core core_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  // A fire's receives, as the core takes them, and pointers to the present
  // ones; sized at construction.
  std::vector<window_send_value> receives_;
  std::vector<const window_send_value *> received_;
  // The last fire's publication.
  bool published_ = false;
  std::int64_t index_ = 0;
  edge_value frame_;              ///< logical_outcome
  std::vector<edge_value> sends_; ///< window_send_payload, one per send
  committed_coverage commit_{};
  std::vector<std::uint8_t> corrections_;
  /// A window's index less its instance's, the same for a shot's fires.
  std::optional<std::int64_t> offset_;
};

} // namespace cudaq::qec
