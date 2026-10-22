/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: window_node — round-accumulating decode node, and the
// `sliding_window` and `single_window` node kinds it implements (registered
// from the accompanying .cpp).  A node type is one header and one source: the
// node class, its parameter schema, and its registration live together.
// Never included from a public header.

#pragma once
#include "decoding_task_graph_node.h"
#include "decoding_task_node_kind_registry.h"
#include "cudaq/qec/cancellation.h"
#include "cudaq/qec/candidate_sink.h"
#include "cudaq/qec/decoder.h"
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <vector>

namespace cudaq::qec {

/// Accumulates syndrome rounds and fires D→decode→O on a configurable trigger.
///
/// Implements the core DAG path, streaming accumulation, and decoder-backed
/// async, affinity, and graph-dispatch operations. The D matrix is shared with
/// impl::d so there is exactly one copy regardless of which path is used.
class window_node : public task_graph_node,
                    public streaming_task_graph_node,
                    public decoder_backed_task_graph_node {
public:
  /// @param d    Measurement-to-detector CSR matrix (shared with impl::d).
  /// @param o    Observable-flip CSR matrix.
  /// @param dec  Inner decoder.
  /// @param spr  Syndrome bytes per round.
  /// @param ws   Window trigger/stride configuration (in rounds).
  /// @param num_measurements  Bytes in one whole window when its rounds differ
  ///        in width; zero for `ws.size * spr`. Such a window cannot slide.
  /// @param fault_offsets  Cumulative per-round fault-column offsets from
  ///        dem_chunks_to_fault_offsets(), `ws.size + 1` entries.  Required
  ///        only when a guard is non-zero, since guards are expressed in
  ///        rounds and committed in columns.  May be empty otherwise.
  window_node(std::shared_ptr<const sparse_binary_matrix> d,
              std::shared_ptr<const sparse_binary_matrix> o,
              std::unique_ptr<decoder> dec, uint32_t spr, window_spec ws,
              std::vector<std::size_t> fault_offsets = {},
              std::size_t num_measurements = 0, bool push_candidates = false,
              bool cancellable = false, std::string id = {});

  streaming_task_graph_node *streaming() override { return this; }
  decoder_backed_task_graph_node *decoder_backed() override { return this; }
  const decoder_backed_task_graph_node *decoder_backed() const override {
    return this;
  }

  // --- Streaming capability ---

  bool enqueue(const uint8_t *msyn, std::size_t len, round_type type) override;
  const uint8_t *corrections() const override;
  void clear_corrections() override;
  void reset_streaming_state() override;

  /// Commits the last fire's end_guard rounds, replaying that fire's own
  /// decode masked to the complement of what it committed.  Buffered rounds
  /// that never filled a window are dropped: D and O are sized for
  /// `window_size`, and padding a short window makes D's difference rows
  /// manufacture detector events that corrupt the rounds that did arrive.
  bool flush() override;
  std::size_t num_msyn_per_decode() const override;
  std::size_t num_msyn_per_round() const override;
  std::size_t num_observables() const override;
  std::optional<window_spec> streaming_schedule() const override { return ws_; }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override;

  /// Room left in the buffer: one stride after a fire, not a whole window.
  std::optional<std::size_t> raw_chunk_limit() const override;
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }

  // --- Core path-independent edge typing ---

  /// slot 0 → logical_outcome; all other slots → none.
  std::optional<edge_value_type> output_type(std::size_t slot) const override;
  /// Slot 0 accepts raw measurements or detection events.
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override;
  std::optional<std::size_t> output_value_size(std::size_t slot) const override;
  void
  validate_input_shape(std::size_t slot, edge_value_type type,
                       std::optional<std::size_t> producer_size) const override;

  // --- Streaming edge routing ---

  /// Streaming edges may feed detection events into slot 0.
  bool supports_streaming_input(std::size_t slot) const override;

  /// Returns pointer to pre-allocated logical_outcome on slot 0 after each
  /// fire, or nullptr before the first fire or after reset_streaming_state()
  /// (including through reset_decoder()).
  const edge_value *streaming_output(std::size_t slot) const override;

  /// Decodes one whole window of detection events, skipping the D-apply the
  /// producer already did.  Throws std::invalid_argument for wrong slot or
  /// wrong edge_value type, std::out_of_range for the wrong detector count.
  bool enqueue_edge_value(const edge_value &v, std::size_t to_slot) override;

  /// Rejects an edge into a sliding window: an edge value is one whole producer
  /// window, and a slid consumer cannot rebuild its leading detector frame.
  void validate_edge_input(std::size_t to_slot) const override;

  // --- Core DAG execution ---

  /// inputs[0] = measurement_results (D applied here) or detection_events
  /// (already in detector space); outputs[0] = logical_outcome.
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override;

  std::vector<std::string> inputs() const override;
  std::vector<std::string> outputs() const override;

  // --- Decoder-backed capability ---

  /// Resets streaming state, then the inner decoder.
  void reset_decoder() override;

  /// Applies D synchronously, then dec().decode_async(); projection deferred.
  std::future<logical_outcome>
  run_async(const measurement_results &m) const override;

  /// dec_'s own device pin, or nullopt when dec_ is unpinned
  /// (get_cuda_device_id() < 0) — same convention decoder itself uses.
  std::optional<int> cuda_device_id() const override;

  /// False for sliding or guarded windows, whose node-owned streaming state is
  /// not representable in the captured decode. Otherwise delegates to
  /// dec_->supports_graph_dispatch().
  bool supports_graph_dispatch() const override;

  /// Sets dec_'s D matrix, then delegates to dec_->capture_decode_graph().
  /// No device buffers of window_node's own -- see the class comment.
  void *capture_graph(int reserved_sms = 0) override;

  /// dec_->release_decode_graph(graph_resources).
  void release_graph(void *graph_resources) override;

  /// dec_.get().
  decoder *inner_decoder() const override;

  // --- Accessors used by decoding_task_graph ---

  const sparse_binary_matrix &d() const;
  const sparse_binary_matrix &o() const;
  std::shared_ptr<const sparse_binary_matrix> d_ptr() const;
  std::shared_ptr<const sparse_binary_matrix> o_ptr() const;

private:
  void require_width(std::size_t result_size) const;

  /// Throw if the stream is resuming after flush() closed it.
  void reject_after_flush(const char *call) const;

  /// decode → mask → commit, for a full window of detection events however it
  /// was produced.  Always fires: reaching here is the trigger.  Returns true.
  bool fire(const std::vector<float_t> &dets);

  /// The inner decode, through `tok` when cancellable; `std::nullopt` when a
  /// stop was honored before a result.
  std::optional<decoder_result> decode(const std::vector<float_t> &dets,
                                       const cancellation_token &tok) const;

  /// The outcome as this node's candidate, costed when the decoder reports a
  /// `cost` among its optional results.
  void push_candidate(const candidate_sink &sink, const logical_outcome &lo,
                      const decoder_result &dr) const;

  /// Input slots of the optional ports, present per flag after `input`.
  std::optional<std::size_t> sink_slot() const {
    return pushes_ ? std::optional<std::size_t>(1) : std::nullopt;
  }
  std::optional<std::size_t> token_slot() const {
    return cancellable_ ? std::optional<std::size_t>(pushes_ ? 2 : 1)
                        : std::nullopt;
  }

  /// True when dec_ hands back an observable frame rather than fault columns,
  /// so O must not be applied a second time.
  bool projects_to_observables() const;

  /// Zero the guard rounds' fault columns so they contribute nothing to the
  /// commit.  Equivalent to dropping those columns from O, because
  /// project_into() indexes the result by O's global column ids.
  ///
  /// @param first_fire  Keep the leading guard on the first fire of a shot: no
  ///        earlier window covered those rounds, so nothing else commits them.
  void mask_guard_columns(std::vector<float_t> &result, bool first_fire) const;

  std::shared_ptr<const sparse_binary_matrix> d_;
  std::shared_ptr<const sparse_binary_matrix> o_;
  std::unique_ptr<decoder> dec_;
  uint32_t spr_; ///< syndrome bytes per round
  window_spec ws_;
  /// Committed column range [lo_col_, hi_col_).  Without guards this is the
  /// whole result: lo_col_ is 0 and hi_col_ is SIZE_MAX.
  std::size_t lo_col_ = 0;
  std::size_t hi_col_ = std::numeric_limits<std::size_t>::max();
  std::size_t fires_since_reset_ = 0;
  // Raw-syndrome accumulation (enqueue path).
  std::size_t bytes_in_ = 0;
  std::vector<uint8_t> buf_;
  mutable std::vector<float_t> dets_; ///< D-apply scratch (stream + one-shot)
  std::vector<uint8_t> corr_;
  /// The last fire's raw decoder result, snapshot for flush() only when a
  /// guard withholds a tail.  Copied rather than re-decoded: commit_into()
  /// masks in place, and decode() may be stateful.
  std::vector<float_t> last_result_;
  std::vector<float_t> flush_scratch_; ///< masked copy of last_result_
  bool last_result_valid_ = false;
  bool flushed_ = false; ///< one commit per tail; see flush()
  /// Landing buffer for a routed detection_events value: one whole window,
  /// d_->num_rows() floats, decoded on arrival (enqueue_edge_value path).
  std::vector<float_t> det_buf_;
  // Pre-allocated streaming output; reused on every fire (no per-fire alloc).
  mutable edge_value last_lo_val_; ///< logical_outcome with pre-sized bits
  mutable bool last_lo_valid_ = false;
  /// Optional ports: `push_candidates` adds `sink`, into which every outcome
  /// goes as `id_`'s candidate; `cancellable` adds `token`, which the decode
  /// honors. Streaming delivers the handles before the window.
  bool pushes_ = false;
  bool cancellable_ = false;
  std::string id_;
  candidate_sink sink_;
  cancellation_token token_;
};

/// Build a window_node from a node-kind build context, shared by the
/// `sliding_window` and `single_window` kinds: they differ only in whether the
/// window advances by fewer rounds than it holds.  Exported so an out-of-tree
/// kind can wrap a window without reimplementing the parameter handling.
///
/// @param sliding  When true, `step_size` (default 1) sets the stride and
///        guards are available.  When false the window covers its whole DEM in
///        one decode, so the stride is the size and neither `step_size` nor a
///        guard is a legal parameter (the schema does not declare them).
__attribute__((visibility("default"))) std::unique_ptr<task_graph_node>
make_window_node(node_build_context &ctx, bool sliding);

/// Parameters shared by every window kind, in emission order. A kind appends
/// its own (a sliding window adds `step_size` and the guards).
__attribute__((visibility("default"))) std::vector<decoding::config::param_spec>
window_kind_common_params();

} // namespace cudaq::qec
