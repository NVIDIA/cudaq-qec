/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../lib/decoding_task/decoding_task_node_kind_registry.h"
#include "cudaq/qec/candidate_sink.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/selection_strategy.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <map>
#include <thread>

using namespace cudaq::qec;
using namespace std::chrono_literals;

namespace {

const auto sink_type = edge_value_type::of<candidate_sink>();
const auto token_type = edge_value_type::of<cancellation_token>();

std::vector<uint8_t> marker_bits(std::size_t marker) {
  std::vector<uint8_t> bits(3, 0);
  bits[marker] = 1;
  return bits;
}

logical_outcome marker(std::size_t m, bool converged = true) {
  return {marker_bits(m), converged};
}

/// Shared handle the test keeps on a member after it moved into its graph.
struct probe {
  std::atomic<int> latency_ms{0}, calls{0};
  std::atomic<bool> converged{true}, token_aware{true}, throws{false},
      saw_stop{false};
  std::optional<double> cost;
  /// Cost of the best candidate the sink held when this member pushed.
  std::atomic<double> best_seen{-1};
};

std::shared_ptr<probe> make_probe(int latency_ms, bool converged = true,
                                  bool token_aware = true,
                                  std::optional<double> cost = std::nullopt) {
  auto p = std::make_shared<probe>();
  p->latency_ms = latency_ms;
  p->converged = converged;
  p->token_aware = token_aware;
  p->cost = cost;
  return p;
}

/// Probe and marker for each debug member id, read when its graph is built.
std::map<std::string, std::pair<std::shared_ptr<probe>, std::size_t>> specs;

/// Sleeps `latency_ms` in 1 ms slices while polling the run's token; returns
/// false when a stop was seen.
bool wait_or_stop(probe &p, const cancellation_token &tok) {
  const auto until = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(p.latency_ms);
  while (std::chrono::steady_clock::now() < until) {
    if (p.token_aware && tok.stop_requested()) {
      p.saw_stop = true;
      return false;
    }
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

/// Debug member that pushes its own candidate: waits, then pushes fault
/// `marker` into the sink. It takes the sink and the token; its `output` is
/// declared (so a streaming join can size its corrections) but never produced.
class test_member : public task_graph_node, public streaming_task_graph_node {
public:
  explicit test_member(const std::string &id)
      : id_(id), p_(specs.at(id).first), marker_(specs.at(id).second) {}

  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &) const override {
    run(inputs[1].get<candidate_sink>(), inputs[2].get<cancellation_token>());
  }
  std::vector<std::string> inputs() const override {
    return {"input", "sink", "token"};
  }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    return slot == 0 ? std::optional(logical_outcome_type) : std::nullopt;
  }
  std::optional<std::size_t> output_value_size(std::size_t) const override {
    return 3;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    switch (slot) {
    case 0:
      return type == measurement_results_type || type == detection_events_type;
    case 1:
      return type == sink_type;
    default:
      return type == token_type;
    }
  }
  streaming_task_graph_node *streaming() override { return this; }

  bool enqueue(const uint8_t *, std::size_t, round_type) override {
    return false;
  }
  const uint8_t *corrections() const override { return nullptr; }
  void clear_corrections() override {}
  void reset_streaming_state() override {}
  std::size_t num_msyn_per_decode() const override { return 0; }
  std::size_t num_observables() const override { return 3; }
  bool supports_streaming_input(std::size_t slot) const override {
    return slot < 3;
  }
  // The lowering delivers the handles before the window that fires the node.
  bool enqueue_edge_value(const edge_value &v, std::size_t slot) override {
    if (slot == 1)
      sink_ = v.get<candidate_sink>();
    else if (slot == 2)
      tok_ = v.get<cancellation_token>();
    else
      run(sink_, tok_);
    return false;
  }

private:
  void run(const candidate_sink &sink, const cancellation_token &tok) const {
    ++p_->calls;
    if (p_->throws)
      throw std::runtime_error("test_member: configured to throw");
    if (!wait_or_stop(*p_, tok))
      return;
    if (const auto best = sink.best_so_far())
      p_->best_seen = *best->cost;
    sink.push(marker(marker_, p_->converged), id_, p_->cost);
  }

  std::string id_;
  std::shared_ptr<probe> p_;
  std::size_t marker_;
  candidate_sink sink_;
  cancellation_token tok_;
};

/// Debug member that is cancellable but produces no candidates itself: it
/// outputs its marker as an ordinary logical outcome, which the lowering's
/// push tail turns into the member's candidate.
class test_cancel_only : public task_graph_node {
public:
  explicit test_cancel_only(const std::string &id)
      : p_(specs.at(id).first), marker_(specs.at(id).second) {}
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    ++p_->calls;
    wait_or_stop(*p_, inputs[1].get<cancellation_token>());
    outputs[0] = marker(marker_);
  }
  std::vector<std::string> inputs() const override {
    return {"input", "token"};
  }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    return slot == 0 ? std::optional(logical_outcome_type) : std::nullopt;
  }
  std::optional<std::size_t> output_value_size(std::size_t) const override {
    return 3;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 ? type == measurement_results_type : type == token_type;
  }

private:
  std::shared_ptr<probe> p_;
  std::size_t marker_;
};

/// Debug detector former: passes the shot's bits through as detection events,
/// so an ensemble downstream is fed detectors rather than measurements.
class test_detectors : public task_graph_node {
public:
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    const auto &bits = inputs[0].get<measurement_results>().bits;
    outputs[0] = detection_events{{bits.begin(), bits.end()}};
  }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    return slot == 0 ? std::optional(detection_events_type) : std::nullopt;
  }
  std::optional<std::size_t> output_value_size(std::size_t) const override {
    return 2;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == measurement_results_type;
  }
};

/// Debug raw-fed streaming node: one window of two measurement bytes per fire,
/// passed through unchanged, so the ensemble downstream is fed windows without
/// knowing anything about rounds itself. One-shot it passes the shot through.
class test_windower : public task_graph_node, public streaming_task_graph_node {
public:
  void execute_indexed(const std::vector<edge_value> &inputs,
                       std::vector<edge_value> &outputs) const override {
    outputs[0] = inputs[0];
  }
  std::vector<std::string> inputs() const override { return {"input"}; }
  std::vector<std::string> outputs() const override { return {"output"}; }
  std::optional<edge_value_type> output_type(std::size_t slot) const override {
    return slot == 0 ? std::optional(measurement_results_type) : std::nullopt;
  }
  bool accepts_input_type(std::size_t slot,
                          edge_value_type type) const override {
    return slot == 0 && type == measurement_results_type;
  }
  streaming_task_graph_node *streaming() override { return this; }

  bool enqueue(const uint8_t *msyn, std::size_t len, round_type) override {
    std::copy_n(msyn, len, buf_.begin() + bytes_in_);
    bytes_in_ += len;
    if (bytes_in_ < buf_.size())
      return false;
    bytes_in_ = 0;
    out_.get<measurement_results>().bits = buf_;
    return fired_ = true;
  }
  const uint8_t *corrections() const override { return nullptr; }
  void clear_corrections() override {}
  void reset_streaming_state() override { bytes_in_ = 0, fired_ = false; }
  std::size_t num_msyn_per_decode() const override { return buf_.size(); }
  std::size_t num_observables() const override { return 0; }
  std::optional<std::size_t> raw_chunk_limit() const override {
    return buf_.size() - bytes_in_;
  }
  bool supports_raw_streaming_input(std::size_t slot) const override {
    return slot == 0;
  }
  const edge_value *streaming_output(std::size_t slot) const override {
    return fired_ && slot == 0 ? &out_ : nullptr;
  }

private:
  std::size_t bytes_in_ = 0;
  std::vector<uint8_t> buf_ = std::vector<uint8_t>(2, 0);
  edge_value out_{measurement_results{}};
  bool fired_ = false;
};

template <typename T>
void register_test_kind(const char *name) {
  task_graph_node_kind_spec spec;
  spec.name = name;
  spec.requires_decoder = spec.requires_dem = false;
  register_node_kind(std::move(spec), {}, [](node_build_context &ctx) {
    if constexpr (std::is_constructible_v<T, const std::string &>)
      return std::unique_ptr<task_graph_node>(new T(ctx.id));
    else
      return std::unique_ptr<task_graph_node>(new T());
  });
}

const bool registered = [] {
  register_test_kind<test_member>("test_member");
  register_test_kind<test_cancel_only>("test_cancel_only");
  register_test_kind<test_detectors>("test_detectors");
  register_test_kind<test_windower>("test_windower");
  return true;
}();

task_graph_node_config node(std::string id, std::string kind) {
  task_graph_node_config c;
  c.id = std::move(id);
  c.kind = std::move(kind);
  return c;
}

using members = std::vector<std::pair<std::shared_ptr<probe>, std::size_t>>;

cudaqx::heterogeneous_map deadline(std::size_t deadline_us,
                                   std::size_t soft_before_us = 0) {
  cudaqx::heterogeneous_map p;
  p.insert("deadline_us", deadline_us);
  if (soft_before_us)
    p.insert("soft_stop_before_us", soft_before_us);
  return p;
}

/// An ensemble pattern `id` of single-node members (kind overridable per
/// member, `test_member` by default) with the given strategy; member ids
/// become `<id>.m<i>.n`.
task_graph_node_config ensemble(const std::string &id, const members &ms,
                                const std::string &strategy = "first",
                                cudaqx::heterogeneous_map join_params = {},
                                const std::vector<std::string> &kinds = {}) {
  task_graph_ensemble_spec spec;
  for (std::size_t i = 0; i < ms.size(); ++i) {
    specs[id + ".m" + std::to_string(i) + ".n"] = ms[i];
    const std::string kind = i < kinds.size() ? kinds[i] : "test_member";
    task_graph_ensemble_spec::member m;
    m.nodes.push_back(node("n", kind));
    m.edges.push_back({"input", "output", "n", "input"});
    if (kind != "test_member") // pushes nothing itself: its output is pushed
      m.edges.push_back({"n", "output", "output", "input"});
    spec.members.push_back(std::move(m));
  }
  spec.join_strategy = strategy;
  spec.join_params = std::move(join_params);
  auto c = node(id, "");
  c.ensemble = std::move(spec);
  return c;
}

/// A LUT window over two detectors and three faults, each flipping one
/// observable: detector 0 alone is explained by fault 0 alone.
task_graph_node_config lut_window(std::string id) {
  auto window = node(std::move(id), "single_window");
  window.d_sparse = {{0}, {1}};
  window.o_sparse = {{0}, {1}, {2}};
  window.decoder_instance = get_decoder(
      "single_error_lut",
      sparse_binary_matrix::from_nested_csr(2, 3, {{0, 1}, {1, 2}}));
  return window;
}

/// A member made of one LUT window whose outcome is its candidate.
task_graph_ensemble_spec::member lut_member() {
  task_graph_ensemble_spec::member m;
  m.nodes.push_back(lut_window("lut"));
  m.edges = {{"input", "output", "lut", "input"},
             {"lut", "output", "output", "input"}};
  return m;
}

/// A member of one LUT window that pushes its own candidates and takes the
/// token, so its edges need not reach `output`.
task_graph_ensemble_spec::member lut_member_pushing() {
  task_graph_ensemble_spec::member m;
  auto window = lut_window("lut");
  window.node_params.insert("push_candidates", true);
  window.node_params.insert("cancellable", true);
  m.nodes.push_back(std::move(window));
  m.edges = {{"input", "output", "lut", "input"}};
  return m;
}

/// Debug decoder behind a window: waits like a probe while honoring the
/// token, then answers fault `marker` with the probe's cost.
class probe_decoder : public decoder {
public:
  probe_decoder(std::shared_ptr<probe> p, std::size_t marker)
      : decoder(decoder_init{
            sparse_binary_matrix::from_nested_csr(2, 3, {{0, 1}, {1, 2}})}),
        p_(std::move(p)), marker_(marker) {}
  decoder_result decode(const std::vector<cudaq::qec::float_t> &) override {
    decoder_result r;
    r.converged = true;
    r.result.assign(3, 0.0f);
    r.result[marker_] = 1.0f;
    if (p_->cost)
      r.opt_results.emplace().insert("cost", *p_->cost);
    return r;
  }
  std::optional<decoder_result>
  decode(const std::vector<cudaq::qec::float_t> &dets,
         cancellation_token tok) override {
    if (!wait_or_stop(*p_, tok))
      return std::nullopt;
    return decode(dets);
  }

private:
  std::shared_ptr<probe> p_;
  std::size_t marker_;
};

/// A self-pushing, cancellable window around a probe_decoder.
task_graph_ensemble_spec::member probe_member(std::shared_ptr<probe> p,
                                              std::size_t marker) {
  auto m = lut_member_pushing();
  m.nodes[0].decoder_instance =
      std::make_unique<probe_decoder>(std::move(p), marker);
  return m;
}

/// An ensemble `ens` of the given members.
task_graph_node_config
ensemble_of(std::vector<task_graph_ensemble_spec::member> ms,
            const std::string &strategy = "first") {
  auto c = node("ens", "");
  c.ensemble.emplace();
  c.ensemble->members = std::move(ms);
  c.ensemble->join_strategy = strategy;
  return c;
}

std::unique_ptr<decoding_task_graph> build(task_graph_node_config ens) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(std::move(ens));
  return decoding_task_graph::from_topology(std::move(nodes));
}

/// The ensemble `ens` fed by a two-byte windower, so it streams one run per
/// window.
std::unique_ptr<decoding_task_graph>
make_streaming(task_graph_node_config ens, task_graph_options options = {}) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("win", "test_windower"));
  nodes.push_back(std::move(ens));
  return decoding_task_graph::from_topology(
      std::move(nodes),
      {{"measurements", "output", "win", "input"},
       {"win", "output", "ens", "input"},
       {"ens", "output", "output", "input"}},
      std::move(options));
}

std::unique_ptr<decoding_task_graph>
make_ensemble(const members &ms, const std::string &strategy = "first",
              cudaqx::heterogeneous_map join_params = {}, bool stream = false,
              task_graph_options options = {}) {
  auto ens = ensemble("ens", ms, strategy, std::move(join_params));
  if (stream)
    return make_streaming(std::move(ens), std::move(options));
  return build(std::move(ens));
}

const measurement_results shot{{1, 0}};

/// Members race only under the event runtime; the synchronous `run()` walks
/// the members in order and then the join, as documented.
logical_outcome race(decoding_task_graph &g) { return g.run_async(shot).get(); }

template <typename F>
std::chrono::milliseconds time_it(F &&f) {
  const auto start = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
}

std::vector<uint8_t> corrections(const decoding_task_graph &g) {
  const auto *corr = g.get_obs_corrections();
  return {corr, corr + g.get_num_observables()};
}

} // namespace

// ---------------------------------------------------------------------------
// candidate_sink
// ---------------------------------------------------------------------------

TEST(CandidateSink, PushDrainIsFifoAndBestTracksTheCheapest) {
  candidate_sink sink;
  EXPECT_FALSE(sink.best_so_far());
  sink.push(marker(0), "a", 2.0);
  sink.push(marker(1), "b");
  EXPECT_EQ(sink.best_so_far()->producer, "a");
  sink.push(marker(2), "c", 1.0);
  sink.push(marker(2), "d", 1.0); // a tie keeps the earlier one
  sink.push(marker(2), "e", 3.0);
  EXPECT_EQ(sink.best_so_far()->producer, "c");
  const auto all = sink.drain();
  ASSERT_EQ(all.size(), 5u);
  EXPECT_EQ(all[0].producer, "a");
  EXPECT_EQ(all[4].producer, "e");
  EXPECT_EQ(sink.best_so_far()->producer, "c"); // draining leaves it alone
  EXPECT_TRUE(sink.drain().empty());
  sink.reset();
  EXPECT_FALSE(sink.best_so_far());
  sink.push(marker(1), "f"); // an uncosted-only round has no best
  EXPECT_FALSE(sink.best_so_far());
}

TEST(CandidateSink, ConcurrentPushesAllArriveAndReachTheObserver) {
  candidate_sink sink;
  std::atomic<int> observed{0};
  sink.reset([&](const candidate &c) {
    observed += static_cast<int>(c.result.bits.size());
  });
  constexpr int threads = 4, per_thread = 2000;
  std::vector<std::thread> pushers;
  for (int t = 0; t < threads; ++t)
    pushers.emplace_back([&, t] {
      for (int i = 0; i < per_thread; ++i)
        sink.push(marker(t % 3), "t" + std::to_string(t), double(i));
    });
  for (auto &t : pushers)
    t.join();
  EXPECT_EQ(sink.drain().size(), std::size_t(threads * per_thread));
  EXPECT_EQ(observed, 3 * threads * per_thread);
  EXPECT_DOUBLE_EQ(*sink.best_so_far()->cost, 0.0);
}

// ---------------------------------------------------------------------------
// Selection strategies
// ---------------------------------------------------------------------------

TEST(SelectionStrategy, FirstAcceptsOnlyConvergedCandidates) {
  auto s = make_selection_strategy("first", {});
  s->begin(cancellation_source{}, candidate_sink{});
  EXPECT_FALSE(s->offer({marker(0, false), "a", std::nullopt}));
  const auto lo = s->offer({marker(1), "b", 7.0});
  ASSERT_TRUE(lo);
  EXPECT_EQ(lo->bits, marker_bits(1));
  EXPECT_FALSE(s->offer({marker(2), "c"})); // committed once, then deaf
  EXPECT_FALSE(s->finish());
  EXPECT_THROW(make_selection_strategy("best", {}), std::invalid_argument);
  const auto names = registered_selection_strategies();
  EXPECT_NE(std::find(names.begin(), names.end(), "deadline"), names.end());
}

/// A sink whose observer is the strategy, as the fan-out wires it.
candidate_sink sink_for(selection_strategy &s) {
  candidate_sink sink;
  sink.reset([&s](const candidate &c) { EXPECT_FALSE(s.offer(c)); });
  return sink;
}

TEST(SelectionStrategy, RegistrationIsReleasedWithItsHandle) {
  /// Commits to whatever is offered first.
  struct any_strategy : selection_strategy {
    std::optional<logical_outcome> offer(const candidate &c) override {
      return c.result;
    }
  };
  {
    const auto held = register_selection_strategy(
        "test_any", {}, [](const cudaqx::heterogeneous_map &) {
          return std::make_unique<any_strategy>();
        });
    const auto names = registered_selection_strategies();
    EXPECT_NE(std::find(names.begin(), names.end(), "test_any"), names.end());
    EXPECT_EQ(
        make_selection_strategy("test_any", {})->offer({marker(2), "a"})->bits,
        marker_bits(2));
    EXPECT_EQ(
        make_ensemble({{make_probe(0, false), 1}}, "test_any")->run(shot).bits,
        marker_bits(1));
  }
  EXPECT_THROW(make_selection_strategy("test_any", {}), std::invalid_argument);
}

TEST(SelectionStrategy, DeadlineStopsTheRunAndPicksTheCheapestOrTheFirst) {
  auto s = make_selection_strategy("deadline", deadline(30000));
  cancellation_source cancel;
  auto sink = sink_for(*s);
  s->begin(cancel, sink);
  sink.push(marker(0), "a", 5.0);
  sink.push(marker(1), "b", 2.0);
  sink.push(marker(2), "c");
  EXPECT_EQ(cancel.level(), cancellation_level::none);
  std::this_thread::sleep_for(45ms);
  // The timer fired the hard stop and fixed the choice; a late, cheaper
  // candidate still enters the sink but not the decision.
  EXPECT_EQ(cancel.level(), cancellation_level::hard);
  sink.push(marker(2), "late", 1.0);
  EXPECT_EQ(sink.best_so_far()->producer, "late");
  EXPECT_EQ(s->finish()->bits, marker_bits(1));
  EXPECT_EQ(sink.drain().size(), 4u);

  auto uncosted = make_selection_strategy("deadline", deadline(1000000));
  auto quiet = sink_for(*uncosted);
  uncosted->begin(cancellation_source{}, quiet);
  EXPECT_FALSE(uncosted->finish());
  quiet.push(marker(2, false), "c");
  quiet.push(marker(0), "a");
  EXPECT_EQ(uncosted->finish()->bits, marker_bits(2));
}

TEST(SelectionStrategy, DeadlineRequestsASoftStopAheadAndDisarmsOnFinish) {
  auto s = make_selection_strategy("deadline", deadline(80000, 60000));
  cancellation_source cancel;
  s->begin(cancel, candidate_sink{});
  std::this_thread::sleep_for(40ms);
  EXPECT_TRUE(cancel.get_token().stop_requested());
  EXPECT_FALSE(cancel.get_token().stop_requested(cancellation_level::hard));
  std::this_thread::sleep_for(60ms);
  EXPECT_TRUE(cancel.get_token().stop_requested(cancellation_level::hard));
  // Finishing before the deadline disarms it: no stop is ever requested.
  auto early = make_selection_strategy("deadline", deadline(20000));
  cancellation_source quiet;
  early->begin(quiet, candidate_sink{});
  early->offer({marker(0), "a", 1.0});
  EXPECT_EQ(early->finish()->bits, marker_bits(0));
  std::this_thread::sleep_for(35ms);
  EXPECT_EQ(quiet.level(), cancellation_level::none);
}

// ---------------------------------------------------------------------------
// The ensemble pattern, one-shot
// ---------------------------------------------------------------------------

TEST(EnsembleGraph, BuildValidatesThePattern) {
  EXPECT_THROW(make_ensemble({}), std::invalid_argument);
  EXPECT_THROW(make_ensemble({{make_probe(0), 0}}, "best"), std::runtime_error);
  EXPECT_THROW(make_ensemble({{make_probe(0), 0}}, "deadline"),
               std::runtime_error);
  EXPECT_THROW(
      make_ensemble({{make_probe(0), 0}}, "deadline", deadline(100, 200)),
      std::runtime_error);
  // The lowering names the ensemble's ports.
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(ensemble("ens", {{make_probe(0), 0}}));
  EXPECT_THROW(decoding_task_graph::from_topology(
                   std::move(nodes), {{"measurements", "output", "ens", "dets"},
                                      {"ens", "output", "output", "input"}}),
               std::invalid_argument);
  // Every member lists its edges.
  auto ens = ensemble("ens", {{make_probe(0), 0}});
  ens.ensemble->members[0].edges.clear();
  EXPECT_THROW(build(std::move(ens)), std::invalid_argument);
  // A member that neither takes the sink nor reaches `output` is rejected.
  ens = ensemble("ens", {{make_probe(0), 0}}, "first", {}, {"test_detectors"});
  ens.ensemble->members[0].edges.pop_back();
  EXPECT_THROW(build(std::move(ens)), std::invalid_argument);
}

TEST(EnsembleGraph, SingleMemberReturnsItsResult) {
  const auto lo = make_ensemble({{make_probe(0), 1}})->run(shot);
  EXPECT_EQ(lo.bits, marker_bits(1));
  EXPECT_TRUE(lo.converged);
}

TEST(EnsembleGraph, SerialRunWalksTheMembersInOrderThenTheJoin) {
  // run() does not race: the first member in order answers first whatever its
  // latency, and the commit hard-stops the later ones before they start.
  auto a = make_probe(50), b = make_probe(1);
  auto g = make_ensemble({{a, 0}, {b, 1}});
  EXPECT_EQ(g->run(shot).bits, marker_bits(0));
  EXPECT_TRUE(b->saw_stop);
  EXPECT_EQ(g->run_batch({shot, shot}).at(1).bits, marker_bits(0));
  EXPECT_EQ(g->run_all(shot).at(0).bits, marker_bits(0));
}

TEST(EnsembleGraph, PlainDecoderMemberGetsAPushTail) {
  // A LUT window member whose outcome reaches `output` races a slow member.
  auto slow = make_probe(5000);
  auto ens = ensemble("ens", {{slow, 1}});
  ens.ensemble->members.push_back(lut_member());
  auto g = build(std::move(ens));
  // Detector 0 alone fires, which only fault 0 explains.
  EXPECT_EQ(race(*g).bits, marker_bits(0));
  EXPECT_TRUE(slow->saw_stop);
}

TEST(EnsembleGraph, MemberMayBeASubgraphWithExplicitEdges) {
  auto slow = make_probe(5000);
  auto ens = ensemble("ens", {{slow, 1}});
  task_graph_ensemble_spec::member m;
  m.nodes.push_back(lut_window("lut"));
  m.nodes.push_back(node("x", "xor"));
  m.edges = {{"input", "output", "lut", "input"},
             {"lut", "output", "x", "a"},
             {"lut", "output", "x", "b"},
             {"x", "output", "output", "input"}};
  ens.ensemble->members.push_back(std::move(m));
  auto g = build(std::move(ens));
  // The LUT's answer XORed with itself is the zero correction.
  EXPECT_EQ(race(*g).bits, std::vector<uint8_t>(3, 0));
  EXPECT_TRUE(slow->saw_stop);
}

TEST(EnsembleGraph, CancellableOnlyMemberGetsTheTokenAndAPushTail) {
  auto fast = make_probe(1), cancellable = make_probe(5000);
  auto g = build(ensemble("ens", {{fast, 0}, {cancellable, 1}}, "first", {},
                          {"test_member", "test_cancel_only"}));
  logical_outcome lo;
  EXPECT_LT(time_it([&] { lo = race(*g); }), 2s);
  EXPECT_EQ(lo.bits, marker_bits(0));
  EXPECT_TRUE(cancellable->saw_stop);
  EXPECT_EQ(cancellable->calls, 1);
  // Its push tail commits when it answers first.
  cancellable->latency_ms = 0;
  fast->latency_ms = 50;
  EXPECT_EQ(race(*g).bits, marker_bits(1));
}

TEST(EnsembleGraph, FastestWinsAndSlowerIsHardStopped) {
  auto fast = make_probe(5), slow = make_probe(5000);
  auto g = make_ensemble({{slow, 2}, {fast, 0}});
  logical_outcome lo;
  EXPECT_LT(time_it([&] { lo = race(*g); }), 2s);
  EXPECT_EQ(lo.bits, marker_bits(0));
  EXPECT_TRUE(slow->saw_stop);
}

TEST(EnsembleGraph, FirstIgnoresUnconvergedAndThrowsWhenNoneConverges) {
  auto fast = make_probe(1, /*converged=*/false), slow = make_probe(50);
  auto g = make_ensemble({{fast, 0}, {slow, 1}});
  auto lo = g->run(shot);
  EXPECT_EQ(lo.bits, marker_bits(1));
  EXPECT_TRUE(lo.converged);
  // Every member finished without a converged answer: a miss.
  slow->converged = false;
  EXPECT_THROW(g->run(shot), std::runtime_error);
  EXPECT_THROW(race(*g), std::runtime_error);
  EXPECT_EQ(slow->calls, 3);
}

TEST(EnsembleGraph, DeadlineCommitsTheCheapestCandidate) {
  auto a = make_probe(1, true, true, 5.0), b = make_probe(5, true, true, 2.0),
       c = make_probe(5000);
  auto g = make_ensemble({{a, 0}, {b, 1}, {c, 2}}, "deadline", deadline(50000));
  logical_outcome lo;
  const auto took = time_it([&] { lo = race(*g); });
  EXPECT_GE(took, 50ms);
  EXPECT_LT(took, 2s);
  EXPECT_EQ(lo.bits, marker_bits(1));
  EXPECT_TRUE(c->saw_stop);
  // b saw a's candidate as the best so far when it pushed its own.
  EXPECT_DOUBLE_EQ(b->best_seen, 5.0);
}

TEST(EnsembleGraph, DeadlineSoftStopLetsMembersEndEarly) {
  auto a = make_probe(1, true, true, 1.0), c = make_probe(5000);
  auto g =
      make_ensemble({{a, 0}, {c, 2}}, "deadline", deadline(400000, 350000));
  logical_outcome lo;
  // c honours the soft stop at ~50 ms; every member has then finished, so the
  // join runs well before the 400 ms deadline.
  EXPECT_LT(time_it([&] { lo = race(*g); }), 300ms);
  EXPECT_EQ(lo.bits, marker_bits(0));
  EXPECT_TRUE(c->saw_stop);
}

TEST(EnsembleGraph, DeadlineMissThrowsAndStopsTheMembers) {
  auto a = make_probe(5000), b = make_probe(5000);
  auto g = make_ensemble({{a, 0}, {b, 1}}, "deadline", deadline(20000));
  EXPECT_LT(time_it([&] { EXPECT_THROW(race(*g), std::runtime_error); }), 2s);
  EXPECT_TRUE(a->saw_stop);
  EXPECT_TRUE(b->saw_stop);
  // The serial walk is bounded by the same deadline.
  EXPECT_LT(time_it([&] { EXPECT_THROW(g->run(shot), std::runtime_error); }),
            2s);
}

TEST(EnsembleGraph, NonTokenAwareMemberDelaysOnlyTheRunsEnd) {
  auto fast = make_probe(1), deaf = make_probe(200, true, false);
  auto g = make_ensemble({{fast, 0}, {deaf, 1}});
  logical_outcome lo;
  EXPECT_GE(time_it([&] { lo = race(*g); }), 200ms);
  EXPECT_EQ(lo.bits, marker_bits(0));
  EXPECT_FALSE(deaf->saw_stop);
}

TEST(EnsembleGraph, FailedMemberFailsTheRunAndTheGraphStaysUsable) {
  auto bad = make_probe(0), slow = make_probe(100);
  bad->throws = true;
  auto g = make_ensemble({{bad, 0}, {slow, 1}});
  EXPECT_ANY_THROW(g->run(shot));
  EXPECT_ANY_THROW(race(*g));
  bad->throws = false;
  EXPECT_EQ(g->run(shot).bits, marker_bits(0));
  EXPECT_EQ(race(*g).bits, marker_bits(0));
}

TEST(EnsembleGraph, RepeatedRacesReuseTheEnsemble) {
  auto a = make_probe(1), b = make_probe(100);
  auto g = make_ensemble({{a, 0}, {b, 1}});
  for (int i = 0; i < 4; ++i) {
    const bool a_wins = i % 2 == 0;
    a->latency_ms = a_wins ? 1 : 100;
    b->latency_ms = a_wins ? 100 : 1;
    EXPECT_EQ(race(*g).bits, marker_bits(a_wins ? 0 : 1)) << i;
  }
}

TEST(EnsembleGraph, TwoEnsemblesComposeThroughXor) {
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(ensemble("a", {{make_probe(1), 0}, {make_probe(2000), 1}}));
  nodes.push_back(ensemble("b", {{make_probe(2000), 0}, {make_probe(1), 2}}));
  nodes.push_back(node("x", "xor"));
  auto g = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "a", "input"},
                         {"measurements", "output", "b", "input"},
                         {"a", "output", "x", "a"},
                         {"b", "output", "x", "b"},
                         {"x", "output", "output", "input"}});
  logical_outcome lo;
  EXPECT_LT(time_it([&] { lo = race(*g); }), 1500ms);
  EXPECT_EQ(lo.bits, (std::vector<uint8_t>{1, 0, 1}));
}

TEST(EnsembleGraph, MoreEnsemblesThanThreadsCannotDeadlock) {
  // Nothing blocks a worker, so any pool size makes progress: chain more
  // ensembles than this machine has threads through XOR nodes.
  const std::size_t count = 2 * std::thread::hardware_concurrency() + 1;
  std::vector<task_graph_node_config> nodes;
  std::vector<task_graph_edge> edges;
  for (std::size_t i = 0; i < count; ++i) {
    const auto id = "e" + std::to_string(i);
    nodes.push_back(ensemble(id, {{make_probe(1), 0}, {make_probe(1000), 1}}));
    edges.push_back({"measurements", "output", id, "input"});
  }
  std::string acc = "e0";
  for (std::size_t i = 1; i < count; ++i) {
    const auto x = "x" + std::to_string(i);
    nodes.push_back(node(x, "xor"));
    edges.push_back({acc, "output", x, "a"});
    edges.push_back({"e" + std::to_string(i), "output", x, "b"});
    acc = x;
  }
  edges.push_back({acc, "output", "output", "input"});
  auto g =
      decoding_task_graph::from_topology(std::move(nodes), std::move(edges));
  logical_outcome lo;
  EXPECT_LT(time_it([&] { lo = race(*g); }), 5s);
  EXPECT_EQ(lo.bits, marker_bits(0)); // an odd number of marker-0 answers
}

TEST(EnsembleGraph, DetectionEventsFormedUpstreamFeedEveryMember) {
  // Detectors are formed once; the fan-out passes them through typed as such,
  // and a window member decodes them directly.
  auto slow = make_probe(5000);
  auto ens = ensemble("ens", {{slow, 1}});
  ens.ensemble->members.push_back(lut_member());
  std::vector<task_graph_node_config> nodes;
  nodes.push_back(node("form", "test_detectors"));
  nodes.push_back(std::move(ens));
  auto g = decoding_task_graph::from_topology(
      std::move(nodes), {{"measurements", "output", "form", "input"},
                         {"form", "output", "ens", "input"},
                         {"ens", "output", "output", "input"}});
  EXPECT_EQ(race(*g).bits, marker_bits(0));
  EXPECT_TRUE(slow->saw_stop);
}

// ---------------------------------------------------------------------------
// Streaming: one ensemble run per window the producer delivers
// ---------------------------------------------------------------------------

TEST(EnsembleGraph, WindowMemberPushesItsOwnCandidate) {
  std::vector<task_graph_ensemble_spec::member> ms;
  ms.push_back(lut_member_pushing());
  ms.push_back(lut_member());
  auto g = build(ensemble_of(std::move(ms)));
  EXPECT_EQ(g->run(shot).bits, (std::vector<uint8_t>{1, 0, 0}));
  EXPECT_EQ(race(*g).bits, (std::vector<uint8_t>{1, 0, 0}));
}

TEST(EnsembleGraph, StoppedWindowCommitsNothing) {
  auto slow = make_probe(2000);
  std::vector<task_graph_ensemble_spec::member> ms;
  ms.push_back(probe_member(slow, 1));
  ms.push_back(lut_member());
  auto g = build(ensemble_of(std::move(ms)));
  EXPECT_EQ(race(*g).bits, marker_bits(0));
  EXPECT_TRUE(slow->saw_stop);
}

TEST(EnsembleGraph, WindowsPushTheirDecoderCost) {
  auto dear = make_probe(1, true, true, 5.0),
       cheap = make_probe(1, true, true, 1.0);
  auto ens = ensemble_of({}, "deadline");
  ens.ensemble->members.push_back(probe_member(dear, 0));
  ens.ensemble->members.push_back(probe_member(cheap, 1));
  ens.ensemble->join_params = deadline(50000);
  EXPECT_EQ(race(*build(std::move(ens))).bits, marker_bits(1));
}

TEST(EnsembleGraph, BuildRejectsAMemberThatPushesAndReachesOutput) {
  std::vector<task_graph_ensemble_spec::member> ms;
  ms.push_back(lut_member_pushing());
  ms.back().edges.push_back({"lut", "output", "output", "input"});
  EXPECT_THROW(build(ensemble_of(std::move(ms))), std::invalid_argument);
}

TEST(EnsembleGraph, BuildRejectsMembersOfDifferentWidths) {
  std::vector<task_graph_ensemble_spec::member> ms;
  ms.push_back(lut_member());
  ms.push_back(lut_member());
  ms.back().nodes[0].o_sparse = {{0}, {1}}; // two observables, not three
  EXPECT_THROW(build(ensemble_of(std::move(ms))), std::invalid_argument);
}

TEST(EnsembleStreaming, NeedsAStreamingProducer) {
  // Fed raw measurements, the fan-out has no windows to forward: the ensemble
  // knows nothing about rounds, so it is one-shot only.
  auto raw = make_ensemble({{make_probe(0), 0}});
  EXPECT_FALSE(raw->supports_streaming());
  EXPECT_FALSE(raw->streaming_incompatibility_reason().empty());
}

TEST(EnsembleStreaming, OutputWidthIsInferredFromAMemberThatReachesOutput) {
  // Any member answer or member output of known width sizes the join's
  // corrections; here both members offer one.
  auto ens = ensemble("ens", {{make_probe(0), 0}}, "deadline", deadline(1000));
  ens.ensemble->members.push_back(lut_member());
  auto g = make_streaming(std::move(ens));
  EXPECT_EQ(g->get_num_observables(), 3u);
  EXPECT_EQ(g->run(shot).bits.size(), 3u);
}

TEST(EnsembleStreaming, EachWindowCommitsTheWinner) {
  auto a = make_probe(1), b = make_probe(2000);
  auto g = make_ensemble({{a, 0}, {b, 1}}, "first", {}, true);
  ASSERT_TRUE(g->supports_streaming()) << g->streaming_incompatibility_reason();
  EXPECT_EQ(g->get_num_msyn_per_decode(), 2u);

  const uint8_t window[2] = {1, 0};
  EXPECT_FALSE(g->enqueue_syndrome(window, 1));
  EXPECT_TRUE(g->enqueue_syndrome(window + 1, 1));
  EXPECT_EQ(corrections(*g), marker_bits(0));
  EXPECT_TRUE(b->saw_stop);

  // The other member wins the next window; corrections accumulate by XOR.
  a->latency_ms = 2000;
  b->latency_ms = 1;
  EXPECT_TRUE(g->enqueue_syndrome(window, 2));
  EXPECT_EQ(corrections(*g), (std::vector<uint8_t>{1, 1, 0}));

  // Two windows in one call fire twice and cancel.
  g->reset_decoder();
  b->latency_ms = 2000;
  a->latency_ms = 1;
  const uint8_t two[4] = {1, 0, 0, 1};
  EXPECT_TRUE(g->enqueue_syndrome(two, 4));
  EXPECT_EQ(corrections(*g), std::vector<uint8_t>(3, 0));
  EXPECT_EQ(a->calls, 4);
}

TEST(EnsembleStreaming, DeadlineDecidesEachWindowAfresh) {
  auto a = make_probe(1, true, true, 2.0), b = make_probe(200, true, true, 1.0);
  auto g = make_ensemble({{a, 0}, {b, 1}}, "deadline", deadline(30000), true);
  const uint8_t window[2] = {1, 0};
  EXPECT_TRUE(g->enqueue_syndrome(window, 2));
  EXPECT_EQ(corrections(*g), marker_bits(0)); // b was stopped at the deadline
  EXPECT_TRUE(b->saw_stop);
  a->latency_ms = 200;
  b->latency_ms = 1;
  EXPECT_TRUE(g->enqueue_syndrome(window, 2)); // a fresh decision, XORed in
  EXPECT_EQ(corrections(*g), (std::vector<uint8_t>{1, 1, 0}));
}

TEST(EnsembleStreaming, TheJoinRunsOncePerWindowAfterItsMembers) {
  // Both members answer within a few ms; the deadline is far away. The join
  // follows the members' completion, so the window ends as soon as they do.
  auto a = make_probe(1, true, true, 2.0), b = make_probe(3, true, true, 1.0);
  auto g = make_ensemble({{a, 0}, {b, 1}}, "deadline", deadline(5000000), true);
  ASSERT_TRUE(g->supports_streaming()) << g->streaming_incompatibility_reason();
  const uint8_t window[2] = {1, 0};
  EXPECT_LT(time_it([&] { EXPECT_TRUE(g->enqueue_syndrome(window, 2)); }), 2s);
  EXPECT_EQ(corrections(*g), marker_bits(1));
}

TEST(EnsembleStreaming, AsyncStreamingCommitsEveryWindow) {
  task_graph_options options;
  options.async_streaming = true;
  auto g = make_ensemble({{make_probe(1), 1}, {make_probe(2000), 0}}, "first",
                         {}, true, std::move(options));
  ASSERT_TRUE(g->supports_streaming()) << g->streaming_incompatibility_reason();
  const uint8_t window[2] = {0, 1};
  for (int i = 0; i < 3; ++i)
    g->enqueue_syndrome(window, 2);
  EXPECT_EQ(corrections(*g), marker_bits(1));
}
