/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The node kinds an ensemble pattern lowers to (decoding_task_graph_topology):
// `ensemble_fanout`, `candidate_push` and `ensemble_join`.
// A fan-out arms one run: it passes its data input, candidate sink, and
// cancellation token to the members, and hands the join node the run's
// `ensemble_selection`, which contains references to the shared state. The
// selection strategy decides as candidates are pushed, on the pushing member's
// thread, and a commit hard-stops the other members; the join, which the
// lowering orders after every member through their implicit `done` outputs,
// only reads that decision, or asks the strategy to finish.

#include "decoding_task_graph_node.h"
#include "decoding_task_node_kind_registry.h"
#include "cudaq/qec/cancellation.h"
#include "cudaq/qec/candidate_sink.h"
#include "cudaq/qec/selection_strategy.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace cudaq::qec {

namespace {

/// An ensemble decision, shared by the fan-out that arms it,
/// the sink observer that writes it and the join that reads it. Copies share
/// one state.
class ensemble_selection {
public:
  ensemble_selection() : s_(std::make_shared<state>()) {}
  explicit ensemble_selection(std::unique_ptr<selection_strategy> strategy)
      : ensemble_selection() {
    s_->strategy = std::move(strategy);
  }

  /// Start a run under `source`, pushing into `sink`: forget the previous
  /// decision and have the strategy do the same. The previous run's members
  /// have all finished.
  void begin(cancellation_source source, const candidate_sink &sink) const {
    s_->source = std::move(source);
    s_->committed.store(false, std::memory_order_release);
    s_->strategy->begin(s_->source, sink);
  }

  /// The sink's observer: offer a pushed candidate to the strategy, whose
  /// one value per run commits the run and hard-stops the members. Called on
  /// member threads.
  void offer(const candidate &c) const {
    if (s_->committed.load(std::memory_order_acquire))
      return;
    auto result = s_->strategy->offer(c);
    if (!result)
      return;
    s_->result = std::move(*result);
    s_->committed.store(true, std::memory_order_release);
    s_->source.request_hard_stop();
  }

  /// The committed outcome or, when nothing committed, the strategy's last
  /// word. Called once every member has finished.
  std::optional<logical_outcome> conclude() const {
    if (s_->committed.load(std::memory_order_acquire))
      return s_->result;
    return s_->strategy->finish();
  }

private:
  struct state {
    std::unique_ptr<selection_strategy> strategy;
    cancellation_source source;
    std::atomic<bool> committed{false};
    logical_outcome result;
  };
  std::shared_ptr<state> s_;
};

} // namespace

CUDAQ_REGISTER_EDGE_VALUE_TYPE(candidate_sink, "candidate_sink")
CUDAQ_REGISTER_EDGE_VALUE_TYPE(cancellation_token, "cancellation_token")
CUDAQ_REGISTER_EDGE_VALUE_TYPE(ensemble_selection, "ensemble_selection")

namespace {

constexpr auto sink_type = edge_value_type::of<candidate_sink>();
constexpr auto token_type = edge_value_type::of<cancellation_token>();
constexpr auto selection_type = edge_value_type::of<ensemble_selection>();

/// inputs[0]="input" (whatever feeds it: a shot, or a streaming producer's
/// windows of measurements or detection events; its type is fixed when that
/// edge is bound) → outputs "output" (the same value), "sink" and "token" for
/// the members, "selection" for the join. Knows nothing about rounds or
/// windows: when streaming it forwards each value its producer delivers as one
/// run and adopts the producer's schedule, so it cannot be fed raw bytes.
class ensemble_fanout_node : public task_graph_node,
                             public streaming_task_graph_node {
public:
  ensemble_fanout_node(const std::string &strategy,
                       const cudaqx::heterogeneous_map &params)
      : selection_(make_selection_strategy(strategy, params)),
        observer_([selection = selection_](const candidate &c) {
          selection.offer(c);
        }) {
    out_.emplace_back(edge_value{});
    out_.emplace_back(sink_);
    out_.emplace_back(cancellation_token{});
    out_.emplace_back(selection_);
  }

  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    arm();
    outputs[0] = inputs[0];
    for (std::size_t slot = 1; slot < out_.size(); ++slot)
      outputs[slot] = out_[slot];
  }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override {
    return {"output", "sink", "token", "selection"};
  }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    switch (slot) {
    case 0:
      return data_type_; // known once the input edge is bound
    case 1:
      return sink_type;
    case 2:
      return token_type;
    case 3:
      return selection_type;
    default:
      return std::nullopt;
    }
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 &&
           (type == measurement_results_type || type == detection_events_type);
  }
  /// The lowering wires the input edge before any edge leaving `output`, so
  /// the pass-through type is known by the time a member's edge is checked.
  void validate_input_shape(std::size_t slot, edge_value_type type,
                            std::optional<std::size_t>) const override {
    if (slot == 0)
      data_type_ = type;
  }
  streaming_task_graph_node *streaming() override { return this; }

  // Edge-fed streaming: each delivered value is one run.
  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    return false;
  }
  std::size_t num_msyn_per_decode() const override { return 0; }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  bool enqueue_edge_value(const edge_value &v, std::size_t) override {
    arm();
    out_[0] = v;
    return fired_ = true;
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t, const std::vector<input_schedule> &inputs) const override {
    return inputs.size() == 1 ? std::optional(inputs[0].schedule)
                              : std::nullopt;
  }
  const uint8_t *corrections() const override { return nullptr; }
  void clear_corrections() override {}
  void reset_streaming_state() override { fired_ = false; }
  std::size_t num_observables() const override { return 0; }
  const edge_value *streaming_output(std::size_t slot) const override {
    return fired_ && slot < out_.size() ? &out_[slot] : nullptr;
  }

private:
  /// Start a run: a fresh cancellation source, the strategy and selection
  /// reset, the sink emptied. The previous run's members have all finished.
  void arm() const {
    cancellation_source source;
    sink_.reset(observer_);
    selection_.begin(source, sink_);
    out_[2] = source.get_token();
  }

  mutable std::optional<edge_value_type> data_type_;
  candidate_sink sink_;
  ensemble_selection selection_;
  candidate_sink::observer observer_;
  /// Outputs parallel to outputs(); refilled in place per run.
  mutable std::vector<edge_value> out_;
  bool fired_ = false;
};

/// inputs "input" (a logical outcome) and "sink" → pushes the outcome as this
/// node's candidate. The lowering appends one to a member whose edges reach
/// `output`.
class candidate_push_node : public task_graph_node,
                            public streaming_task_graph_node {
public:
  explicit candidate_push_node(std::string id) : id_(std::move(id)) {}
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &) const override {
    inputs[1].get<candidate_sink>().push(inputs[0].get<logical_outcome>(), id_);
  }
  std::vector<std::string> inputs() const override { return {"input", "sink"}; }
  std::vector<std::string> outputs() const override { return {}; }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return type == (slot == 0 ? logical_outcome_type : sink_type);
  }
  streaming_task_graph_node *streaming() override { return this; }

  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    return false;
  }
  const uint8_t *corrections() const override { return nullptr; }
  void clear_corrections() override {}
  void reset_streaming_state() override {}
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return 0; }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot < 2;
  }
  bool enqueue_edge_value(const edge_value &v, std::size_t slot) override {
    if (slot == 1)
      sink_ = v.get<candidate_sink>();
    else
      sink_.push(v.get<logical_outcome>(), id_);
    return false;
  }

private:
  std::string id_;
  candidate_sink sink_;
};

/// inputs "selection" and one "done_<i>" per candidate-producing member node →
/// outputs[0]="output". Runs once every producer has completed and emits the
/// run's decision. Streams when given its width.
class ensemble_join_node : public task_graph_node,
                           public streaming_task_graph_node {
public:
  /// @param done_inputs  Number of `done_<i>` inputs (one per producer).
  /// @param num_obs      Output width; 0 disables streaming, which must size
  ///        its output up front.
  ensemble_join_node(std::size_t done_inputs, std::size_t num_obs)
      : inputs_{"selection"}, corr_(num_obs, 0), last_(logical_outcome{}) {
    for (std::size_t i = 0; i < done_inputs; ++i)
      inputs_.push_back("done_" + std::to_string(i));
  }

  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = decide(inputs[0].get<ensemble_selection>());
  }
  std::vector<std::string> inputs() const override { return inputs_; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    return slot == 0 ? std::optional(logical_outcome_type) : std::nullopt;
  }
  std::optional<std::size_t>
  output_value_size(std::size_t slot) const override {
    if (slot != 0 || corr_.empty())
      return std::nullopt;
    return corr_.size();
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return type == (slot == 0 ? selection_type : completion_type);
  }
  streaming_task_graph_node *streaming() override {
    return corr_.empty() ? nullptr : this;
  }

  // Streaming: one decision per fan-out window, XORed into the corrections,
  // taken once every input of the window has arrived, whatever their order.
  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    return false;
  }
  const uint8_t *corrections() const override { return corr_.data(); }
  void clear_corrections() override {
    std::fill(corr_.begin(), corr_.end(), uint8_t{0});
  }
  void reset_streaming_state() override {
    clear_corrections();
    fired_ = false;
    received_ = 0;
  }
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return corr_.size(); }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot < inputs_.size();
  }
  std::optional<window_spec> derive_streaming_schedule(
      std::size_t output_slot,
      const std::vector<input_schedule> &inputs) const override {
    if (output_slot != 0 || inputs.empty())
      return std::nullopt;
    for (const auto &in : inputs)
      if (!(in.schedule == inputs[0].schedule))
        return std::nullopt;
    return inputs[0].schedule;
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return slot == 0 && fired_ ? &last_ : nullptr;
  }
  bool enqueue_edge_value(const edge_value &v, std::size_t slot) override {
    if (slot == 0)
      selection_ = v.get<ensemble_selection>();
    if (++received_ < inputs_.size())
      return false;
    received_ = 0;
    auto lo = decide(selection_);
    for (std::size_t i = 0; i < corr_.size(); ++i)
      corr_[i] ^= lo.bits[i];
    last_ = std::move(lo);
    return fired_ = true;
  }

private:
  logical_outcome decide(const ensemble_selection &selection) const {
    auto lo = selection.conclude();
    if (!lo)
      throw std::runtime_error("ensemble_join: no member produced a candidate");
    if (!corr_.empty() && lo->bits.size() != corr_.size())
      throw std::out_of_range(
          "ensemble_join: the winning candidate has the wrong number of "
          "observables");
    return std::move(*lo);
  }

  std::vector<std::string> inputs_;
  std::vector<uint8_t> corr_;
  edge_value last_;
  bool fired_ = false;
  // Streaming delivery state for the window being assembled.
  ensemble_selection selection_;
  std::size_t received_ = 0;
};

const bool registered = [] {
  using decoding::config::param_kind;

  task_graph_node_kind_spec fanout;
  fanout.name = "ensemble_fanout";
  fanout.requires_decoder = fanout.requires_dem = false;
  fanout.lowered_only = true;
  decoding::config::decoder_schema fanout_params;
  fanout_params.params = {{"strategy", param_kind::string, /*required=*/true},
                          {"strategy_params", param_kind::discriminated, false,
                           "", "strategy", /*materialize_empty=*/true,
                           "selection_strategy."}};
  fanout_params.validate = [](const cudaqx::heterogeneous_map &p) {
    const auto names = registered_selection_strategies();
    const auto s = p.get<std::string>("strategy", "");
    if (std::find(names.begin(), names.end(), s) == names.end())
      throw std::runtime_error("ensemble: strategy '" + s +
                               "' is not a registered selection strategy");
  };
  register_node_kind(
      std::move(fanout), std::move(fanout_params), [](node_build_context &ctx) {
        const auto &p = *ctx.node_params;
        return std::unique_ptr<task_graph_node>(new ensemble_fanout_node(
            p.get<std::string>("strategy"),
            p.get<cudaqx::heterogeneous_map>("strategy_params",
                                             cudaqx::heterogeneous_map{})));
      });

  task_graph_node_kind_spec push;
  push.name = "candidate_push";
  push.requires_decoder = push.requires_dem = false;
  push.lowered_only = true;
  register_node_kind(std::move(push), {}, [](node_build_context &ctx) {
    return std::unique_ptr<task_graph_node>(new candidate_push_node(ctx.id));
  });

  task_graph_node_kind_spec join;
  join.name = "ensemble_join";
  join.requires_decoder = join.requires_dem = false;
  join.lowered_only = true;
  decoding::config::decoder_schema join_params;
  // Both set by the lowering once the members are built: the number of
  // producers whose `done` the join waits for, and its streaming width.
  join_params.params = {{"done_inputs", param_kind::uint64},
                        {"num_observables", param_kind::uint64}};
  register_node_kind(
      std::move(join), std::move(join_params), [](node_build_context &ctx) {
        const auto &p = *ctx.node_params;
        return std::unique_ptr<task_graph_node>(
            new ensemble_join_node(p.get<std::size_t>("done_inputs", 0),
                                   p.get<std::size_t>("num_observables", 0)));
      });
  return true;
}();

} // namespace

} // namespace cudaq::qec
