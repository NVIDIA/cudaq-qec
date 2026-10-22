/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Internal header: streaming_decoder_node — transparent access to an inner
// decoder's native enqueue/get/clear/reset streaming API. Never included from
// a public header.

#pragma once

#include "decoding_task_graph_node.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <memory>
#include <vector>

namespace cudaq::qec {

/// A decoder-backed node that forwards raw streaming chunks without buffering
/// them in the graph. This preserves a decoder's native incremental execution.
class streaming_decoder_node : public task_graph_node,
                               public streaming_task_graph_node,
                               public decoder_backed_task_graph_node {
public:
  streaming_decoder_node(std::shared_ptr<const sparse_binary_matrix> d,
                         std::shared_ptr<const sparse_binary_matrix> o,
                         std::unique_ptr<decoder> dec, uint32_t spr,
                         bool equal_rounds = true);

  streaming_task_graph_node *streaming() override { return this; }
  decoder_backed_task_graph_node *decoder_backed() override { return this; }
  const decoder_backed_task_graph_node *decoder_backed() const override {
    return this;
  }

  bool enqueue(const uint8_t *msyn, std::size_t len, round_type type) override;
  bool enqueue_untagged(const uint8_t *msyn, std::size_t len) override;
  /// Untagged rounds make a fixed-length shot, so a chunk stops at its end.
  std::optional<std::size_t> untagged_chunk_limit() const override {
    return num_msyn_per_decode() - untagged_fill_;
  }
  bool flush() override;
  const uint8_t *corrections() const override;
  void clear_corrections() override;
  void reset_streaming_state() override;
  std::size_t num_msyn_per_decode() const override;
  std::size_t num_msyn_per_round() const override;
  std::size_t num_observables() const override;
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  const edge_value *streaming_output(std::size_t slot) const override;
  std::optional<committed_coverage> streaming_coverage() const override;

  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override;
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override;
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override;
  std::optional<std::size_t> output_value_size(std::size_t slot) const override;

  void reset_decoder() override;
  std::future<logical_outcome>
  run_async(const measurement_results &m) const override;
  std::optional<int> cuda_device_id() const override;
  decoder *inner_decoder() const override { return dec_.get(); }

private:
  void capture_inner_corrections(bool final);
  void publish_inner_frame(const uint8_t *current, bool final);
  logical_outcome decode_one_shot(const measurement_results &m) const;
  logical_outcome project_result(decoder_result result) const;

  std::shared_ptr<const sparse_binary_matrix> d_;
  std::shared_ptr<const sparse_binary_matrix> o_;
  std::unique_ptr<decoder> dec_;
  streaming_correction_provider *correction_provider_ = nullptr;
  uint32_t spr_;
  mutable std::vector<float_t> dets_;
  mutable edge_value last_output_;
  std::vector<uint8_t> correction_frame_;
  bool pending_inner_input_ = false;
  bool output_valid_ = false;
  bool published_final_ = false;
  // Measurements of the untagged shot taken so far.
  std::size_t untagged_fill_ = 0;
  // Rounds of the stream's completed shots, where the next one begins.
  std::uint64_t completed_rounds_ = 0;
  // The last FINAL shot's rounds, when the decoder reports them.
  std::optional<committed_coverage> final_coverage_;
};

} // namespace cudaq::qec
