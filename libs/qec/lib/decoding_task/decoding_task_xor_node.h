/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: xor_node — GF(2) combiner for logical outcomes, and the
// `xor` node kind it implements (registered from the accompanying .cpp).
// Never included from a public header.

#pragma once
#include "decoding_task_graph_node.h"
#include "cudaq/qec/decoding_task_types.h"

namespace cudaq::qec {

/// GF(2) XOR of two logical_outcomes. DAG-only: streaming() and
/// decoder_backed() stay null, so it has no streaming state, decoder reset,
/// asynchronous decode, affinity, or graph dispatch.
class xor_node : public task_graph_node {
public:
  ~xor_node() override;

  // inputs[0]="a", inputs[1]="b"  →  outputs[0]="output"
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = xor_logical(inputs[0].get<logical_outcome>(),
                             inputs[1].get<logical_outcome>());
  }

  std::vector<std::string> inputs() const override { return {"a", "b"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return logical_outcome_type;
    return std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot < 2 && type == logical_outcome_type;
  }
  bool supports_streaming_xor_fold() const override { return true; }
};

} // namespace cudaq::qec
