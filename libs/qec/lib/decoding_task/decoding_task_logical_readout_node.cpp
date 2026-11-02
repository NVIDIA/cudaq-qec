/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `logical_readout` node and its kind. Prefix reading rounds sit at fixed
// places, so they are read as they arrive; suffix rounds depend on the shot's
// length, so they are read from its last measurements at FINAL. A graph XORs
// the value with the corrections into its `logical` output.

#include "decoding_task_logical_readout_node.h"
#include "decoding_task_node_kind_registry.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// Places rounds as ingest does: prefix rounds at fixed offsets from the
// shot's start, suffix rounds at fixed offsets from its end. A template that
// does not repeat is all prefix.
logical_readout_node::logical_readout_node(
    std::shared_ptr<const experiment_template> context)
    : context_(std::move(context)) {
  std::vector<std::uint8_t> reads; // observables one round reads
  if (!context_)
    throw std::invalid_argument("logical_readout: no template");
  const auto &t = *context_;
  const auto observables = t.num_observables();
  const auto probe = t.repeats() ? t.min_rounds() + 2 : t.min_rounds();
  const auto prefix = t.prefix_rounds();
  const auto suffix = t.repeats() ? t.suffix_rounds() : std::size_t{0};
  const auto first_tail = probe - suffix;
  const auto width_of = [&](std::size_t r) {
    return t.measurement_offset(r + 1, probe) - t.measurement_offset(r, probe);
  };
  if (!t.has_readout())
    throw std::invalid_argument(
        "logical_readout: no phase of the template carries m2o_sparse");
  if (t.repeats())
    round_width_ = width_of(prefix);
  else
    for (std::size_t r = 0; r < probe; ++r)
      round_width_ = std::max(round_width_, width_of(r));

  completion_.assign(observables, std::nullopt);
  for (std::size_t r = 0; r < prefix; ++r) {
    reads.assign(observables, 0);
    add_reading_round(t.readout_of(r, probe), r, t.measurement_offset(r, probe),
                      width_of(r), head_rounds_, reads);
    for (std::size_t i = 0; i < observables; ++i)
      if (reads[i])
        completion_[i] = r;
  }
  head_ = t.measurement_offset(prefix, probe);
  tail_ = t.measurement_offset(probe, probe) -
          t.measurement_offset(first_tail, probe);
  for (auto r = first_tail; r < probe; ++r) {
    reads.assign(observables, 0);
    add_reading_round(t.readout_of(r, probe), 0,
                      t.measurement_offset(r, probe) -
                          t.measurement_offset(first_tail, probe),
                      width_of(r), tail_rounds_, reads);
    for (std::size_t i = 0; i < observables; ++i)
      if (reads[i])
        completion_[i].reset();
  }
  kept_.assign(tail_, 0);
  read_.assign(observables, 0);
  corrections_.assign(observables, 0);
  logical_ = logical_outcome{corrections_, true};
} // end - logical_readout_node()

// Each m2o_sparse row (ended by -1) is an observable; each value a
// measurement of the round. dem_chunks_spec::validate() keeps every value
// inside its round and unrepeated within a row; the index is checked again
// here, at construction, since it addresses by_measurement.
void logical_readout_node::add_reading_round(
    const std::vector<std::int64_t> &m2o, std::uint64_t r, std::size_t begin,
    std::size_t width, std::vector<reading_round> &out,
    std::vector<std::uint8_t> &reads) {
  std::vector<std::vector<std::uint32_t>> by_measurement(width);
  std::uint32_t observable = 0;
  reading_round round;
  bool any = false;
  for (const auto value : m2o) {
    if (value == -1) {
      ++observable;
      continue;
    }
    if (value < 0 || static_cast<std::uint64_t>(value) >= width)
      throw std::invalid_argument(
          "logical_readout: round " + std::to_string(r) +
          "'s m2o_sparse reads measurement " + std::to_string(value) +
          " of the " + std::to_string(width) + " its round has");
    by_measurement[static_cast<std::size_t>(value)].push_back(observable);
    reads[observable] = 1;
    any = true;
  }
  if (!any)
    return;
  round.round = r;
  round.begin = begin;
  round.width = width;
  round.start.push_back(0);
  for (const auto &observables : by_measurement) {
    round.observables.insert(round.observables.end(), observables.begin(),
                             observables.end());
    round.start.push_back(round.observables.size());
  }
  out.push_back(std::move(round));
} // end - add_reading_round()

void logical_readout_node::add_reading(const reading_round &round,
                                       std::size_t local,
                                       std::vector<std::uint8_t> &value) {
  for (auto k = round.start[local]; k < round.start[local + 1]; ++k)
    value[round.observables[k]] ^= 1;
}

// The prefix's reading rounds from the shot's start, the suffix's from its
// end.
void logical_readout_node::execute_indexed(
    const std::vector<edge_value> &inputs,
    std::vector<edge_value> &outputs) const {
  const auto &bits = inputs.at(0).get<measurement_results>().bits;
  logical_outcome out{std::vector<uint8_t>(corrections_.size(), 0), true};
  if (bits.size() < head_ + tail_)
    throw std::invalid_argument(
        "logical_readout: the shot holds " + std::to_string(bits.size()) +
        " measurements, fewer than the " + std::to_string(head_ + tail_) +
        " of its prefix and suffix rounds");
  for (const auto &round : head_rounds_)
    for (std::size_t p = 0; p < round.width; ++p)
      if (bits[round.begin + p] & 1)
        add_reading(round, p, out.bits);
  for (const auto &round : tail_rounds_)
    for (std::size_t p = 0; p < round.width; ++p)
      if (bits[bits.size() - tail_ + round.begin + p] & 1)
        add_reading(round, p, out.bits);
  outputs.at(0) = std::move(out);
} // end - execute_indexed()

// The value, on `output` alone.
std::optional<edge_value_type>
logical_readout_node::output_type(std::size_t slot) const {
  if (slot == 0)
    return logical_outcome_type;
  return std::nullopt;
}

// Raw measurements, on `input` alone.
bool logical_readout_node::accepts_input_type(std::size_t slot,
                                              edge_value_type type) const {
  return slot == 0 && type == measurement_results_type;
}

std::optional<std::size_t>
logical_readout_node::output_value_size(std::size_t slot) const {
  if (slot == 0)
    return corrections_.size();
  return std::nullopt;
}

// Reads prefix reading rounds as the chunk carries them (in order, so only
// one can be partly read) and keeps only the chunk's last tail_ measurements.
bool logical_readout_node::enqueue(const uint8_t *msyn, std::size_t len,
                                   round_type type) {
  const auto end = seen_ + len;
  const auto skip = len > tail_ ? len - tail_ : 0;
  bool read = false; // the chunk completed a prefix reading round
  if (ended_)
    throw std::logic_error("logical_readout: a chunk after FINAL; clear the "
                           "corrections or reset to start the next shot");
  current_ = false;
  while (next_round_ < head_rounds_.size()) {
    const auto &round = head_rounds_[next_round_];
    const auto from = std::max(round.begin, seen_);
    const auto to = std::min(round.begin + round.width, end);
    for (auto p = from; p < to; ++p)
      if (msyn[p - seen_] & 1)
        add_reading(round, p - round.begin, read_);
    if (end < round.begin + round.width)
      break;
    read_end_ = round.round + 1;
    ++next_round_;
    read = true;
  } // end - while(next_round_)
  for (auto i = skip; tail_ != 0 && i < len; ++i) {
    kept_[next_] = msyn[i];
    next_ = next_ + 1 == tail_ ? 0 : next_ + 1;
  }
  seen_ = end;
  // Copies between buffers sized at construction, so nothing allocates.
  if (read) {
    std::copy(read_.begin(), read_.end(), corrections_.begin());
    std::copy(corrections_.begin(), corrections_.end(),
              logical_.get<logical_outcome>().bits.begin());
    current_ = true;
  }
  if (type != round_type::FINAL)
    return read;
  if (seen_ < head_ + tail_)
    throw std::invalid_argument(
        "logical_readout: the shot holds " + std::to_string(seen_) +
        " measurements, fewer than the " + std::to_string(head_ + tail_) +
        " of its prefix and suffix rounds");
  ended_ = true;
  return read;
} // end - enqueue()

// After FINAL the ring holds the suffix, oldest at next_; publish once.
bool logical_readout_node::flush() {
  current_ = false;
  if (!ended_ || published_)
    return false;
  std::copy(read_.begin(), read_.end(), corrections_.begin());
  for (const auto &round : tail_rounds_)
    for (std::size_t p = 0; p < round.width; ++p) {
      auto at = next_ + round.begin + p;
      if (at >= tail_)
        at -= tail_;
      if (kept_[at] & 1)
        add_reading(round, p, corrections_);
    }
  std::copy(corrections_.begin(), corrections_.end(),
            logical_.get<logical_outcome>().bits.begin());
  read_end_ = whole_shot;
  current_ = true;
  published_ = true;
  return true;
} // end - flush()

const edge_value *
logical_readout_node::streaming_output(std::size_t slot) const {
  return slot == 0 && current_ ? &logical_ : nullptr;
}

// Reported between publications too, for snapshots taken then.
std::optional<streaming_task_graph_node::committed_coverage>
logical_readout_node::streaming_coverage() const {
  if (read_end_ == 0)
    return std::nullopt;
  return committed_coverage{0, read_end_};
}

// As for every streaming node, clearing an ended shot starts the next one.
void logical_readout_node::clear_corrections() {
  if (ended_)
    reset_streaming_state();
}

void logical_readout_node::reset_streaming_state() {
  next_ = 0;
  seen_ = 0;
  next_round_ = 0;
  read_end_ = 0;
  ended_ = false;
  published_ = false;
  current_ = false;
  std::fill(read_.begin(), read_.end(), 0);
  std::fill(corrections_.begin(), corrections_.end(), 0);
}

namespace {

// Built from the DEM's template; takes no parameters and wraps no decoder.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "logical_readout";
  spec.requires_decoder = false;
  spec.requires_dem = true;
  spec.takes_template = true;
  register_node_kind(
      std::move(spec), {},
      [](node_build_context &ctx) -> std::unique_ptr<task_graph_node> {
        if (!ctx.context)
          throw std::invalid_argument(
              "task graph node '" + ctx.id +
              "': a logical_readout node reads its template from dem_chunks, "
              "which it has none of");
        return std::make_unique<logical_readout_node>(ctx.context);
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
