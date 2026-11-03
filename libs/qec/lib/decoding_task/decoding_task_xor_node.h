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
#include <cstddef>
#include <string>
#include <vector>

namespace cudaq::qec {

/// GF(2) XOR of logical_outcomes: two, on inputs `a` and `b`, or any number
/// from 2 up, on `in_0` ... `in_{N-1}`. DAG-only: streaming() and
/// decoder_backed() stay null, so it has no streaming state, decoder reset,
/// asynchronous decode, affinity, or graph dispatch.
class xor_node : public task_graph_node {
public:
  /// The most inputs an xor takes. Far wider than any graph folds; it stops
  /// a mistyped count from exhausting memory naming ports before any edge is
  /// checked.
  static constexpr std::size_t max_inputs = 1024;

  /// The two-input xor, on `a` and `b`, as combine() wires it.
  xor_node();
  /// An xor of @p num_inputs inputs, on `in_0` ... `in_{num_inputs-1}`.
  /// @throws std::invalid_argument if @p num_inputs is below 2 or above
  /// max_inputs.
  explicit xor_node(std::size_t num_inputs);
  ~xor_node() override;

  // inputs[i]  →  outputs[0]="output", the XOR of every input
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    // One copy of the first input, folded in place: the result is the only
    // allocation however many inputs there are.
    auto folded = inputs[0].get<logical_outcome>();
    for (std::size_t i = 1; i < inputs.size(); ++i)
      xor_logical_into(folded, inputs[i].get<logical_outcome>());
    outputs[0] = std::move(folded);
  }

  std::vector<std::string> inputs() const override { return inputs_; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    if (slot == 0)
      return logical_outcome_type;
    return std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot < inputs_.size() && type == logical_outcome_type;
  }
  bool supports_streaming_xor_fold() const override { return true; }

private:
  std::vector<std::string> inputs_;
};

} // namespace cudaq::qec
