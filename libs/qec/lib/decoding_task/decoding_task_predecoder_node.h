/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: predecoder_node — a cheap decode that commits what it
// explains and passes on what it does not, and the `predecoder` node kind it
// implements (registered from the accompanying .cpp).
// Never included from a public header.

#pragma once
#include "decoding_task_graph_node.h"
#include "decoding_task_node_kind_registry.h"
#include "cudaq/qec/decoder.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cudaq::qec {

/// Decodes one window and emits two values per fire: the observable correction
/// it committed (slot 0) and the detection events its prediction does not
/// explain (slot 1), so a slower decoder downstream sees only what is left.
///
/// Either derivation leaves the residual in this node's own detector space,
/// and the consumer decodes it as it stands (it skips its own D-apply), so the
/// consumer has to share this node's DEM and window size. Neither node slides.
///
/// Completeness, which is what makes the split safe under `parity_check`: with
/// the residual computed as `dets XOR H*e`, `H*(e XOR g) == dets` for any `g`
/// explaining it, so this node can never hand the downstream decoder an
/// unsolvable problem however badly it decodes. Exactness is a separate matter
/// -- the two corrections XOR to the monolithic answer only when `e` lies on an
/// optimal explanation, which a cheap decoder does not guarantee. Under
/// `model` neither property is structural: both halves come from the model, so
/// the guarantee is only as good as its training. See the node kind's
/// registration comment.
class predecoder_node : public task_graph_node,
                        public streaming_task_graph_node,
                        public decoder_backed_task_graph_node {
public:
  /// Where the residual comes from. Declared, never inferred: an inner decoder
  /// built without O reports `decode_to_errs` under either contract, so its
  /// result type cannot say which it holds, and telling them apart by result
  /// width would make a mis-declared model a reinterpretation instead of an
  /// error.
  enum class residual_source {
    /// The inner returns an error pattern; this node computes both halves,
    /// `pre_L = O*e` and `residual = dets XOR H*e`. Needs H.
    parity_check,
    /// The inner returns `pre_L` as its observable result and residual
    /// detectors in `opt_results["residual_detectors"]`. Ignores H.
    model,
  };

  /// @param src  Which derivation the inner decoder's result feeds.
  /// @param d    Measurement-to-detector CSR matrix for the window.
  /// @param o    Observable-flip CSR matrix. Its row count is the width of
  ///        `pre_L`.
  /// @param h    The inner decoder's parity-check matrix, CSR: rows are the
  ///        detectors `d` produces, columns the fault mechanisms `o` indexes.
  ///        Required under `parity_check`, where the residual maps a predicted
  ///        error back to the detectors it explains and D and O cannot express
  ///        that; unused under `model`, which reads the residual off the inner.
  /// @param dec  Inner decoder. It returns an error pattern under
  ///        `parity_check`, or observables plus residual metadata under
  ///        `model`.
  /// @param spr  Syndrome bytes per round.
  /// @param rounds  Rounds this window covers; one fire consumes all of them.
  /// @param num_measurements  Bytes in the whole window when its rounds differ
  ///        in width; zero for `rounds * spr`.
  predecoder_node(residual_source src,
                  std::shared_ptr<const sparse_binary_matrix> d,
                  std::shared_ptr<const sparse_binary_matrix> o,
                  std::shared_ptr<const sparse_binary_matrix> h,
                  std::unique_ptr<decoder> dec, uint32_t spr, uint32_t rounds,
                  std::size_t num_measurements = 0);
  ~predecoder_node() override;

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

  /// Never has a tail: no guards, and a fire empties the buffer.  Overridden
  /// only to warn about a partial window rather than inherit a silent false.
  bool flush() override;
  std::size_t num_msyn_per_decode() const override { return buf_.size(); }

  /// Room left in the buffer; this window does not slide, so a fire empties it.
  std::optional<std::size_t> raw_chunk_limit() const override {
    return buf_.size() - bytes_in_;
  }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  std::size_t num_msyn_per_round() const override { return spr_; }
  std::size_t num_observables() const override { return corr_.size(); }

  // --- Core path-independent edge typing ---

  /// slot 0 → logical_outcome, slot 1 → detection_events, else none; the slot
  /// numbering is the order outputs() declares.
  std::optional<edge_value_type> output_type(std::size_t slot) const override;

  /// The one-shot input accepts raw measurement results on slot 0.
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override;
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    if (slot == 0)
      return corr_.size();
    if (slot == 1)
      return d_->num_rows();
    return std::nullopt;
  }

  // --- Streaming output routing ---

  /// Both slots after a fire, nullptr before the first and after
  /// reset_streaming_state() (including through reset_decoder()). Valid until
  /// the next enqueue() or reset.
  const edge_value *streaming_output(std::size_t slot) const override;

  // supports_streaming_input() stays false: this node is fed raw measurement
  // bytes, so an edge into it is a build-time error during streaming.

  // --- Core DAG execution ---

  /// inputs[0] = measurement_results;
  /// outputs[0] = logical_outcome, outputs[1] = detection_events.
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override;

  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override {
    return {"pre_L", "residual"};
  }

  // --- Decoder-backed capability ---

  /// Resets streaming state, then the inner decoder.
  void reset_decoder() override;

  /// dec_'s own device pin, or nullopt when it is unpinned — the same
  /// convention decoder itself uses.
  std::optional<int> cuda_device_id() const override;
  /// dec_.get(). run_async and graph dispatch use the unsupported
  /// decoder_backed_task_graph_node defaults.
  decoder *inner_decoder() const override;

  // --- Accessors (tests and diagnostics) ---

  const sparse_binary_matrix &d() const { return *d_; }
  const sparse_binary_matrix &o() const { return *o_; }
  residual_source source() const { return src_; }

private:
  /// decode → commit → residual, shared by both paths. Fills `lo` and
  /// `residual`, both pre-sized by the caller; only the streaming caller folds
  /// `lo` into corr_.
  void decode_into(const std::vector<float_t> &dets, logical_outcome &lo,
                   detection_events &residual) const;

  residual_source src_;
  std::shared_ptr<const sparse_binary_matrix> d_, o_, h_;
  std::unique_ptr<decoder> dec_;
  uint32_t spr_;
  std::size_t bytes_in_ = 0;
  std::vector<uint8_t> buf_;          ///< raw accumulation, rounds * spr
  mutable std::vector<float_t> dets_; ///< D-apply scratch (stream + one-shot)
  std::vector<uint8_t> corr_;         ///< committed correction, O rows
  // Pre-allocated streaming outputs, refilled in place on every fire: nothing
  // on the hot path allocates.
  edge_value lo_val_, residual_val_;
  bool outputs_valid_ = false;
};

} // namespace cudaq::qec
