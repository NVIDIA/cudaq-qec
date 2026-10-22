/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
#pragma once

#include "decoding_task_window_node.h"
#include "cudaq/qec/extended_dem.h"
#include <map>
#include <tuple>

namespace cudaq::qec {

/// Fixed-width memory execution with an arbitrary-length terminal tail. The
/// inherited finite-shot path uses the configured closed model; streaming uses
/// distinct boundary models. Storage depends on the window, never the duration
/// of the shot.
/// Eager preparation builds floor(p / stride) + p + 4 + W + lookahead - m
/// models plus the base decoder (p = prefix length, m = minimum shot length).
/// Short-shot models can require O(W^2) storage. Decoder allocations and
/// convergence remain backend-dependent; no model is built while streaming.
/// Multiple equal-width terminal phases are retained until FINAL identifies
/// them. They require p_tail extra input rounds of lookahead, W + p_tail + 1
/// ring slots, and a leading guard of at least p_tail. Initial FINAL models
/// also cover W + p_tail rounds when FINAL preempts the first delayed regular
/// fire. Single-terminal-phase execution is unchanged.
class memory_window_node final : public window_node {
public:
  memory_window_node(node_build_context &ctx, window_spec schedule);
  std::size_t prepared_decoder_count() const { return models_.size() + 1; }
  std::size_t retained_round_capacity() const { return ring_.size(); }
  bool enqueue(const uint8_t *, std::size_t, round_type) override;
  bool flush() override;
  // Graph failure handling materializes diagnostic values while marking the
  // result failed; this accessor must not mask the original decoding error.
  const uint8_t *corrections() const override { return corrections_.data(); }
  void clear_corrections() override;
  void reset_streaming_state() override;
  void reset_decoder() override;
  std::size_t num_msyn_per_decode() const override {
    return static_cast<std::size_t>(schedule_.size) * bulk_width_;
  }
  std::size_t num_msyn_per_round() const override { return bulk_width_; }
  std::optional<std::size_t> raw_chunk_limit() const override;
  std::optional<window_spec> streaming_schedule() const override {
    return delivery_schedule_;
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t slot,
      const std::vector<input_schedule> &inputs) const override {
    return slot == 0 && inputs.empty() ? std::optional{delivery_schedule_}
                                       : std::nullopt;
  }
  std::optional<committed_coverage> streaming_coverage() const override {
    return coverage_;
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return slot == 0 && coverage_ && !failed_ ? &output_ : nullptr;
  }
  bool supports_streaming_input(std::size_t) const override { return false; }
  void validate_edge_input(std::size_t) const override;
  bool supports_graph_dispatch() const override { return false; }

private:
  struct model {
    // Nonterminal H is stored in CSC for sparse carry updates; the decoder
    // owns its independent model. Terminal models never update carry.
    sparse_binary_matrix d, h, o;
    std::unique_ptr<decoder> dec;
    std::vector<std::size_t> measurement_offsets, detector_offsets,
        fault_offsets;
    std::vector<uint8_t> measurements;
    std::vector<float_t> residual;
    std::vector<std::size_t> detector_rounds;
  };
  struct round_buffer {
    std::vector<uint8_t> measurements, carry;
    std::size_t width = 0;
  };
  using model_key = std::tuple<std::size_t, bool, std::size_t>;
  model compile_model(std::size_t start_phase, bool terminal,
                      std::size_t rounds, const node_build_context &ctx);
  bool decode_window(bool terminal);
  std::size_t measurement_width(const dem_chunk_spec &phase) const;
  std::size_t next_width() const;
  round_buffer &round(std::uint64_t index) {
    return ring_[index % ring_.size()];
  }
  const round_buffer &round(std::uint64_t index) const {
    return ring_[index % ring_.size()];
  }

  std::shared_ptr<const dem_chunks_spec> spec_;
  std::vector<phase_id> prefix_phases_, final_phases_;
  phase_id bulk_phase_;
  window_spec schedule_, delivery_schedule_;
  std::vector<std::size_t> prefix_widths_;
  std::size_t bulk_width_ = 0, final_width_ = 0;
  std::size_t tail_lookahead_ = 0;
  std::size_t partial_bytes_ = 0;
  std::uint64_t rounds_ = 0, committed_ = 0, next_fire_ = 0;
  bool terminal_complete_ = false, flushed_ = false;
  bool failed_ = false;
  std::vector<round_buffer> ring_;
  std::map<model_key, model> models_;
  std::vector<uint8_t> corrections_;
  edge_value output_;
  std::optional<committed_coverage> coverage_;
};
} // namespace cudaq::qec
