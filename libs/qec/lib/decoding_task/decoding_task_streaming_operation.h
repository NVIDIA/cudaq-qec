// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "decoding_task_graph_node.h"
#include <algorithm>
#include <stdexcept>

namespace cudaq::qec {

/// A synchronous graph operation with one input value per slot per firing.
/// The same execute_indexed implementation serves complete-shot and streaming
/// execution. Equal schedules and slot rendezvous keep separate shots apart.
class streaming_operation : public task_graph_node,
                            public streaming_task_graph_node {
  std::vector<edge_value> pending_, values_;
  std::vector<bool> arrived_;
  std::vector<uint8_t> corrections_;

protected:
  mutable std::size_t observable_width_;

public:
  streaming_operation(std::size_t inputs, std::size_t outputs,
                      std::size_t observables)
      : pending_(inputs), values_(outputs), arrived_(inputs),
        corrections_(observables), observable_width_(observables) {}
  streaming_task_graph_node *streaming() override { return this; }
  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    throw std::logic_error("operation requires routed graph inputs");
  }
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return observable_width_; }
  const uint8_t *corrections() const override { return corrections_.data(); }
  void clear_corrections() override {
    std::fill(corrections_.begin(), corrections_.end(), 0);
  }
  void reset_streaming_state() override {
    std::fill(arrived_.begin(), arrived_.end(), false);
    clear_corrections();
  }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot < pending_.size();
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t, const std::vector<input_schedule> &inputs) const override {
    if (inputs.empty() || inputs.size() != pending_.size())
      return std::nullopt;
    const auto schedule = inputs.front().schedule;
    for (const auto &input : inputs)
      if (!(input.schedule == schedule))
        throw std::invalid_argument(
            "operation inputs have different window_size, "
            "step_size, start_guard or end_guard schedules");
    return schedule;
  }
  bool enqueue_edge_value(const edge_value &value, std::size_t slot) override {
    if (slot >= pending_.size() || arrived_[slot])
      throw std::logic_error("operation input arrived twice before its peers");
    pending_[slot] = value;
    arrived_[slot] = true;
    if (!std::all_of(arrived_.begin(), arrived_.end(),
                     [](bool b) { return b; }))
      return false;
    execute_indexed(pending_, values_);
    // Only the first logical output is this capability's correction stream.
    // Additional detector outputs carry residuals, not correction parity.
    if (const auto *logical = values_.front().get_if<logical_outcome>()) {
      if (logical->bits.size() != observable_width_)
        throw std::runtime_error(
            "operation produced an invalid logical result");
      corrections_.resize(observable_width_, 0);
      for (std::size_t i = 0; i < observable_width_; ++i)
        corrections_[i] ^= logical->bits[i];
    }
    std::fill(arrived_.begin(), arrived_.end(), false);
    return true;
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return slot < values_.size() ? &values_[slot] : nullptr;
  }
  bool flush() override {
    if (std::any_of(arrived_.begin(), arrived_.end(), [](bool b) { return b; }))
      throw std::runtime_error("FINAL leaves unmatched operation inputs");
    return false;
  }
};
} // namespace cudaq::qec
