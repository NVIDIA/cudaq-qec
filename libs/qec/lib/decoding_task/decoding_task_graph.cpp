/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decoding_task_graph_impl.h"
#include "decoding_task_graph_utils.h"
#include "../hardware_guards.h"
#include "cudaq/qec/decoding_task_types.h"
#include "cudaq/qec/logger.h"

#include <algorithm>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cudaq::qec {

std::future<logical_outcome>
decoder_backed_task_graph_node::run_async(const measurement_results &) const {
  throw std::logic_error(
      "task_graph_node: run_async not implemented for this node type");
}

// ---------------------------------------------------------------------------
// Primitive task functions (public API in decoding_task_types.h)
// ---------------------------------------------------------------------------

detection_events
apply_d_sparse(const measurement_results &m,
               const std::vector<std::vector<uint32_t>> &d_sparse) {
  const nested_rows rows{d_sparse};
  if (const auto bad = first_column_at_or_above(rows, m.bits.size()))
    throw std::out_of_range(
        "apply_d_sparse: column " + std::to_string(*bad) +
        " >= num_measurements=" + std::to_string(m.bits.size()));
  detection_events result;
  result.bits.resize(d_sparse.size(), 0.0f);
  detect_into(rows, m.bits.data(), result.bits.data());
  return result;
}

logical_outcome
apply_o_sparse(const error_pattern &e,
               const std::vector<std::vector<uint32_t>> &o_sparse) {
  const nested_rows rows{o_sparse};
  if (const auto bad = first_column_at_or_above(rows, e.result.size()))
    throw std::out_of_range(
        "apply_o_sparse: column " + std::to_string(*bad) +
        " >= num_mechanisms=" + std::to_string(e.result.size()));
  logical_outcome result;
  result.converged = e.converged;
  result.bits.resize(o_sparse.size(), 0);
  project_into(rows, e.result.data(), result.bits.data());
  return result;
}

logical_outcome xor_logical(const logical_outcome &a,
                            const logical_outcome &b) {
  if (a.bits.size() != b.bits.size())
    throw std::invalid_argument("xor_logical: size mismatch (" +
                                std::to_string(a.bits.size()) + " vs " +
                                std::to_string(b.bits.size()) + ")");
  logical_outcome result;
  result.converged = a.converged && b.converged;
  result.bits.resize(a.bits.size());
  for (std::size_t i = 0; i < result.bits.size(); ++i)
    result.bits[i] = a.bits[i] ^ b.bits[i];
  return result;
}

namespace {

void validate_core_edge(
    const std::vector<std::unique_ptr<task_graph_node>> &nodes,
    const task_graph_edge &edge, int from_node, int from_slot, int to_node,
    int to_slot, std::optional<edge_value_type> output_type) {
  if (to_node < 0)
    return;
  if (!output_type ||
      !nodes[to_node]->accepts_input_type(to_slot, *output_type))
    throw std::invalid_argument(
        "compile_wires: edge type mismatch — node '" + edge.to_node +
        "' input '" + edge.to_input + "' does not accept what node '" +
        edge.from_node + "' output '" + edge.from_output + "' produces");
  const auto producer_size =
      from_node < 0 || *output_type == completion_type
          ? std::nullopt
          : nodes[from_node]->output_value_size(from_slot);
  nodes[to_node]->validate_input_shape(to_slot, *output_type, producer_size);
}

std::string
validate_streaming_edge_input(const streaming_task_graph_node &consumer,
                              std::size_t input_slot) {
  try {
    consumer.validate_edge_input(input_slot);
    return {};
  } catch (const std::exception &e) {
    return e.what();
  }
}

std::string derive_streaming_output_schedules(
    const streaming_task_graph_node &node,
    const std::vector<streaming_task_graph_node::input_schedule> &inputs,
    std::vector<std::optional<window_spec>> &outputs) {
  try {
    for (std::size_t slot = 0; slot < outputs.size(); ++slot)
      outputs[slot] = node.derive_streaming_schedule(slot, inputs);
    return {};
  } catch (const std::exception &e) {
    return e.what();
  }
}

// Compute the DAG width: the largest set of pairwise-independent nodes, which
// bounds every dynamic ready/running frontier. By Dilworth's theorem this is
// N minus a maximum matching in the reachability bipartite graph.
std::size_t maximum_parallel_width(
    const std::vector<std::vector<std::size_t>> &dependents) {
  const std::size_t node_count = dependents.size();
  std::vector<std::vector<uint8_t>> reachable(
      node_count, std::vector<uint8_t>(node_count, 0));
  std::vector<int> matched_left(node_count, -1);
  std::size_t matching_size = 0;

  // Nodes are already topologically ordered, so descendants of every direct
  // dependent are complete when its producer is visited in reverse.
  for (std::size_t node = node_count; node-- > 0;) {
    for (const std::size_t dependent : dependents[node]) {
      reachable[node][dependent] = 1;
      for (std::size_t descendant = 0; descendant < node_count; ++descendant)
        reachable[node][descendant] |= reachable[dependent][descendant];
    }
  }

  for (std::size_t node = 0; node < node_count; ++node) {
    std::vector<uint8_t> visited(node_count, 0);
    const auto augment = [&](auto &&self, std::size_t left) -> bool {
      for (std::size_t right = 0; right < node_count; ++right) {
        if (reachable[left][right] == 0 || visited[right] != 0)
          continue;
        visited[right] = 1;
        if (matched_left[right] < 0 ||
            self(self, static_cast<std::size_t>(matched_left[right]))) {
          matched_left[right] = static_cast<int>(left);
          return true;
        }
      }
      return false;
    };
    if (augment(augment, node))
      ++matching_size;
  }

  return node_count - matching_size;
}

} // namespace

// ---------------------------------------------------------------------------
// compile_wires
// ---------------------------------------------------------------------------

void decoding_task_graph::impl::record_streaming_conflict(std::string msg) {
  if (streaming_conflict.empty())
    streaming_conflict = std::move(msg);
}

void decoding_task_graph::impl::require_streaming_support(
    const char *who) const {
  if (!streaming_conflict.empty())
    throw std::logic_error(std::string("decoding_task_graph::") + who + ": " +
                           streaming_conflict +
                           "; this graph decodes only in batch, via "
                           "run() or run_all()");
}

void decoding_task_graph::impl::compile_wires() {
  const auto N = nodes.size();
  std::size_t one_shot_parallel_width = 1;
  std::size_t streaming_wire_count = 0;
  wires_to.assign(N + 1, {});
  node_input_counts.resize(N);
  node_outputs.resize(N);
  std::vector<std::vector<std::string>> node_input_names(N);
  std::vector<std::vector<std::string>> node_output_names(N);
  for (std::size_t i = 0; i < N; ++i) {
    node_input_names[i] = nodes[i]->inputs();
    node_output_names[i] = nodes[i]->outputs();
  }
  std::vector<compiled_wire> wires; // every wire, in declared edge order
  std::unordered_map<std::string, int> id_map;
  id_map.reserve(N);
  for (std::size_t i = 0; i < N; ++i)
    id_map[node_ids[i]] = static_cast<int>(i);

  // A node an edge names as `<node>.done` gets that implicit last output,
  // carrying a `completion` once the node has finished: it orders a consumer
  // after the node without the node declaring anything.
  std::vector<uint8_t> has_done(N, 0);
  for (const auto &e : edges) {
    const auto it = id_map.find(e.from_node);
    if (e.from_output != "done" || it == id_map.end() || has_done[it->second])
      continue;
    auto &outs = node_output_names[it->second];
    if (std::find(outs.begin(), outs.end(), "done") != outs.end())
      throw std::invalid_argument("compile_wires: node '" + e.from_node +
                                  "' declares an output named 'done', which "
                                  "is the implicit completion output");
    outs.push_back("done");
    has_done[it->second] = 1;
  }
  const auto is_done_slot = [&](int node, int slot) {
    return node >= 0 && has_done[node] &&
           static_cast<std::size_t>(slot) + 1 == node_output_names[node].size();
  };
  const auto output_type_at = [&](int node,
                                  int slot) -> std::optional<edge_value_type> {
    if (node < 0)
      return measurement_results_type;
    if (is_done_slot(node, slot))
      return completion_type;
    return nodes[node]->output_type(static_cast<std::size_t>(slot));
  };

  for (const auto &e : edges) {
    int fn = -1, fs = 0;
    if (e.from_node == "measurements" && e.from_output != "output")
      throw std::invalid_argument(
          "compile_wires: synthetic source 'measurements' has only output "
          "'output'");
    if (e.from_node != "measurements") {
      auto it = id_map.find(e.from_node);
      if (it == id_map.end())
        throw std::invalid_argument("compile_wires: unknown node '" +
                                    e.from_node + "'");
      fn = it->second;
      const auto &outs = node_output_names[fn];
      auto sit = std::find(outs.begin(), outs.end(), e.from_output);
      if (sit == outs.end())
        throw std::invalid_argument("compile_wires: node '" + e.from_node +
                                    "' has no output '" + e.from_output + "'");
      fs = static_cast<int>(sit - outs.begin());
    }

    int tn = -1, ts = 0;
    if (e.to_node == "output") {
      // The sink has no ports, so to_input names the output instead. The
      // default ("input") means the author did not name it; fall back to the
      // producing endpoint so every output has a name.
      output_names.push_back(e.to_input != "input" ? e.to_input
                             : e.from_output != "output"
                                 ? e.from_node + "." + e.from_output
                                 : e.from_node);
    } else {
      auto it = id_map.find(e.to_node);
      if (it == id_map.end())
        throw std::invalid_argument("compile_wires: unknown node '" +
                                    e.to_node + "'");
      tn = it->second;
      const auto &ins = node_input_names[tn];
      auto sit = std::find(ins.begin(), ins.end(), e.to_input);
      if (sit == ins.end())
        throw std::invalid_argument("compile_wires: node '" + e.to_node +
                                    "' has no input '" + e.to_input + "'");
      ts = static_cast<int>(sit - ins.begin());
    }

    if (fn >= 0 && tn >= 0 && fn >= tn)
      throw std::invalid_argument(
          "compile_wires: edge from node " + std::to_string(fn) + " ('" +
          node_ids[fn] + "') to node " + std::to_string(tn) + " ('" +
          node_ids[tn] + "') violates topological order");

    validate_core_edge(nodes, e, fn, fs, tn, ts, output_type_at(fn, fs));

    const int group = (tn < 0) ? static_cast<int>(N) : tn;
    wires_to[group].push_back({fn, fs, tn, ts});
    wires.push_back(wires_to[group].back());
  }

  if (wires_to.back().empty())
    throw std::invalid_argument(
        "compile_wires: graph must have at least one output connection");

  for (std::size_t i = 0; i < N; ++i) {
    const auto &inputs = node_input_names[i];
    std::vector<std::size_t> producer_counts(inputs.size(), 0);
    for (const auto &wire : wires_to[i])
      ++producer_counts[wire.to_slot];
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
      if (producer_counts[slot] == 0)
        throw std::invalid_argument("compile_wires: input '" + inputs[slot] +
                                    "' of node '" + node_ids[i] +
                                    "' has no producer");
      if (producer_counts[slot] > 1)
        throw std::invalid_argument("compile_wires: input '" + inputs[slot] +
                                    "' of node '" + node_ids[i] +
                                    "' has multiple producers");
    }
    node_input_counts[i] = inputs.size();
    node_outputs[i].resize(node_output_names[i].size());
    if (has_done[i])
      node_outputs[i].back() = edge_value{completion{}};
  }
  meas_val = edge_value{measurement_results{}};

  // Compile the dependency graph once. Raw measurement edges are immediately
  // available and therefore do not contribute to a node's predecessor count.
  {
    predecessor_counts.assign(N, 0);
    dependents.assign(N, {});
    source_nodes.clear();
    parallel_node_inputs.resize(N);
    for (std::size_t node = 0; node < N; ++node) {
      parallel_node_inputs[node].resize(node_input_counts[node]);
      for (const auto &wire : wires_to[node]) {
        if (wire.from_node < 0)
          continue;
        ++predecessor_counts[node];
        dependents[static_cast<std::size_t>(wire.from_node)].push_back(node);
      }
      if (predecessor_counts[node] == 0)
        source_nodes.push_back(node);
    }

    one_shot_parallel_width = maximum_parallel_width(dependents);
  }

  // Capability presence, rather than a sentinel byte count, declares which
  // nodes participate in streaming.
  streaming_by_node.assign(N, -1);
  for (std::size_t i = 0; i < N; ++i) {
    if (auto *streaming = nodes[i]->streaming()) {
      streaming_nodes.push_back({static_cast<int>(i), streaming});
      streaming_by_node[i] = static_cast<int>(streaming_nodes.size() - 1);
    }
  }
  if (streaming_nodes.empty())
    record_streaming_conflict("graph has no streaming-capable nodes");

  // Only logical outcomes contribute to the corrections buffer. With
  // path-independent edge typing, `none` means that the slot carries no graph
  // value; one-shot XOR nodes explicitly declare a logical_outcome output.
  const auto carries_corrections = [&](const compiled_wire &w) {
    return output_type_at(w.from_node, w.from_slot) == logical_outcome_type;
  };

  // A sink has no input declaration of its own, so validate its producer here.
  for (const auto &sw : wires_to.back())
    if (sw.from_node < 0)
      throw std::invalid_argument(
          "compile_wires: graph output must come from a node's logical-result "
          "port, not the raw 'measurements' source");
    else if (!carries_corrections(sw))
      throw std::invalid_argument(
          "compile_wires: node '" + node_ids[sw.from_node] + "' port '" +
          node_output_names[sw.from_node][sw.from_slot] +
          "' is not a logical outcome and cannot be a graph output");

  // Path parity drives XOR routing; reachability keeps shape information when
  // an even number of paths cancels to zero.
  struct path_state {
    bool parity = false;
    bool reachable = false;
  };
  std::vector<std::vector<path_state>> reach;
  std::vector<std::vector<std::vector<path_state>>> output_reach;
  std::vector<bool> output_has_parity_contributor;

  if (!streaming_nodes.empty()) {
    // A streaming node XOR-accumulates only into the outputs it reaches, so
    // distinct outputs never fold together while commits to the same one
    // still XOR.  That is also what makes a batch-only fold work without
    // firing: combine()'s xor_node never runs while streaming, but both
    // windows reach the one output through it, so their XOR into its segment
    // is the value it would have produced.
    //
    // reach[j][k] = node j feeds sink k; seeded at the sink edges and
    // propagated feeder-ward in reverse topological order.
    const auto &sink_wires = wires_to.back();
    const auto num_sinks = sink_wires.size();
    reach.assign(N, std::vector<path_state>(num_sinks));
    output_reach.resize(N);
    for (std::size_t i = 0; i < N; ++i)
      output_reach[i].assign(node_output_names[i].size(),
                             std::vector<path_state>(num_sinks));
    for (std::size_t k = 0; k < num_sinks; ++k) {
      const auto &wire = sink_wires[k];
      auto &state = output_reach[wire.from_node][wire.from_slot][k];
      state.parity = true;
      state.reachable = true;
    }

    std::vector<bool> is_streaming(N, false);
    for (const auto &entry : streaming_nodes)
      is_streaming[entry.node_index] = true;
    std::string xor_fold_conflict;

    // Only a logical output carries reach.  A syndrome edge -- a predecoder's
    // residual -- gives its consumer something to decode, not a correction to
    // fold into whatever that consumer feeds. Path parity preserves XOR
    // topology, including a source used twice cancelling to zero.
    for (std::size_t j = N; j-- > 0;) {
      bool has_correction_input = false;
      bool reaches_output = false;
      for (std::size_t slot = 0; slot < output_reach[j].size(); ++slot)
        for (std::size_t k = 0; k < num_sinks; ++k) {
          const auto &output_state = output_reach[j][slot][k];
          if (output_state.parity)
            reach[j][k].parity = !reach[j][k].parity;
          reach[j][k].reachable =
              reach[j][k].reachable || output_state.reachable;
        }
      for (const auto &w : wires_to[j])
        if (w.from_node >= 0 && carries_corrections(w))
          has_correction_input = true;
      for (std::size_t k = 0; k < num_sinks; ++k)
        reaches_output = reaches_output || reach[j][k].reachable;
      if (has_correction_input && reaches_output && !is_streaming[j] &&
          !nodes[j]->supports_streaming_xor_fold() && xor_fold_conflict.empty())
        xor_fold_conflict =
            "one-shot-only node '" + node_ids[j] +
            "' lies on a streaming correction path but does not declare "
            "XOR-fold semantics";

      // A streaming node executes and owns the correction it emits, so it is
      // the terminal contributor for this path. Propagating through it would
      // also include its upstream producer and XOR the same correction twice.
      if (is_streaming[j])
        continue;

      for (const auto &w : wires_to[j])
        if (w.from_node >= 0 && carries_corrections(w))
          for (std::size_t k = 0; k < num_sinks; ++k) {
            auto &upstream = output_reach[w.from_node][w.from_slot][k];
            upstream.reachable = upstream.reachable || reach[j][k].reachable;
            if (reach[j][k].parity)
              upstream.parity = !upstream.parity;
          }
    }

    // A structurally connected node whose paths cancel does not constrain a
    // real value contributor. If every path cancels, retain the reachable
    // nodes so the constant-zero output still acquires a publication event.
    output_has_parity_contributor.assign(num_sinks, false);
    for (const auto &entry : streaming_nodes)
      for (std::size_t k = 0; k < num_sinks; ++k)
        output_has_parity_contributor[k] = output_has_parity_contributor[k] ||
                                           reach[entry.node_index][k].parity;

    std::vector<std::size_t> output_offset(num_sinks, 0),
        output_width(num_sinks, 0);
    bool obs_conflict = false;
    const auto obs_conflicted = [&](std::string msg) {
      record_streaming_conflict(std::move(msg));
      obs_conflict = true;
    };

    n_obs = 0;
    for (std::size_t k = 0; k < num_sinks; ++k) {
      // Nodes feeding one output must agree on its width; nodes feeding
      // different outputs need not agree on anything.
      std::size_t width = 0;
      bool fed = false;
      for (const auto &entry : streaming_nodes) {
        const int idx = entry.node_index;
        if (!reach[idx][k].reachable)
          continue;
        const auto w = entry.capability->num_observables();
        if (!fed) {
          width = w;
          fed = true;
        } else if (w != width)
          obs_conflicted(
              "streaming nodes disagree on num_observables for output '" +
              output_names[k] + "'");
      }
      // Reached only by batch-only nodes: no streaming source to XOR from,
      // so it would serve stale zeros forever.
      if (!fed)
        obs_conflicted("output '" + output_names[k] +
                       "' is fed by no streaming node");
      output_offset[k] = n_obs;
      output_width[k] = width;
      n_obs += width;
    }
    if (obs_conflict)
      n_obs = 0;
    if (!xor_fold_conflict.empty())
      record_streaming_conflict(std::move(xor_fold_conflict));

    // The layout a client needs to read the buffer at all.  Multi-output only:
    // one output is always offset 0 for the whole width, which says nothing.
    if (num_sinks > 1 && !obs_conflict)
      for (std::size_t k = 0; k < num_sinks; ++k)
        CUDA_QEC_INFO("decoding_task_graph: output '{}' occupies corrections "
                      "bytes [{}, {}) of {}",
                      output_names[k], output_offset[k],
                      output_offset[k] + output_width[k], n_obs);

    streaming_node_output_offsets.assign(streaming_nodes.size(), {});
    output_streaming_nodes.assign(num_sinks, {});
    if (!obs_conflict)
      for (std::size_t si = 0; si < streaming_nodes.size(); ++si) {
        for (std::size_t k = 0; k < num_sinks; ++k) {
          if (reach[streaming_nodes[si].node_index][k].parity) {
            streaming_node_output_offsets[si].push_back(output_offset[k]);
          }
          const auto &path = reach[streaming_nodes[si].node_index][k];
          if (output_has_parity_contributor[k] ? path.parity : path.reachable) {
            output_streaming_nodes[k].push_back(si);
          }
        }
      }
    streaming_result_state.outputs.resize(num_sinks);
    streaming_result_state.values.assign(n_obs, 0);
    for (std::size_t k = 0; k < num_sinks; ++k) {
      auto &output = streaming_result_state.outputs[k];
      output.name = output_names[k];
      output.value_offset = output_offset[k];
      output.value_width = output_width[k];
    }
  }

  // Compile streaming routes and schedules.
  {
    streaming_wires_from.assign(N, {});
    completion_wires_from.assign(N, {});
    streaming_wires_to_count.assign(N, 0);
    streaming_deliveries_seen.assign(N, 0);
    // Walked in declared edge order: a producer publishes to its consumers in
    // that order, which is what the event tree's sibling blocking assumes.
    for (const auto &w : wires) {
      if (w.from_node < 0 || w.to_node < 0 ||
          streaming_by_node[w.from_node] < 0 ||
          streaming_by_node[w.to_node] < 0)
        continue; // not an edge between two streaming nodes
      (is_done_slot(w.from_node, w.from_slot)
           ? completion_wires_from
           : streaming_wires_from)[w.from_node]
          .push_back(w);
      ++streaming_wires_to_count[w.to_node];
      const auto &consumer = streaming_nodes[streaming_by_node[w.to_node]];
      if (!consumer.capability->supports_streaming_input(w.to_slot))
        record_streaming_conflict("node '" + node_ids[w.to_node] +
                                  "' has no streaming input on slot '" +
                                  node_input_names[w.to_node][w.to_slot] + "'");
    }

    // Classify every input edge to a streaming node. enqueue() has one implicit
    // raw-input slot, while enqueue_edge_value() names its slot explicitly, so
    // mixing the two delivery modes cannot be represented by the runtime.
    const auto M = streaming_nodes.size();
    std::vector<std::size_t> raw_input_count(M, 0);
    std::vector<bool> edge_fed(M, false), batch_fed(M, false);
    for (std::size_t si = 0; si < M; ++si) {
      const auto &entry = streaming_nodes[si];
      const int idx = entry.node_index;
      for (const auto &wire : wires_to[idx]) {
        if (wire.from_node < 0) {
          if (!entry.capability->supports_raw_streaming_input(wire.to_slot))
            record_streaming_conflict(
                "streaming node '" + node_ids[idx] +
                "' cannot consume raw measurements on input '" +
                node_input_names[idx][wire.to_slot] + "'");
          ++raw_input_count[si];
        } else if (streaming_by_node[wire.from_node] >= 0)
          edge_fed[si] = true;
        else
          batch_fed[si] = true;
      }
    }
    streaming_node_has_raw_input.assign(M, false);
    bool any_raw_input = false;
    for (std::size_t si = 0; si < M; ++si) {
      const auto &entry = streaming_nodes[si];
      const int idx = entry.node_index;
      if (batch_fed[si])
        record_streaming_conflict(
            "streaming node '" + node_ids[idx] +
            "' has an input from a non-streaming node, which cannot be routed "
            "by streaming execution");
      if (raw_input_count[si] > 0 && edge_fed[si])
        record_streaming_conflict(
            "streaming node '" + node_ids[idx] +
            "' mixes raw measurement and streaming-edge inputs");
      if (raw_input_count[si] > 1)
        record_streaming_conflict(
            "streaming node '" + node_ids[idx] +
            "' has multiple raw measurement inputs, but enqueue() has no "
            "input-slot argument");
      if (edge_fed[si])
        continue;
      if (raw_input_count[si] == 0)
        record_streaming_conflict("streaming node '" + node_ids[idx] +
                                  "' has no streaming input");
      if (entry.capability->num_msyn_per_decode() == 0)
        record_streaming_conflict("raw-fed streaming node '" + node_ids[idx] +
                                  "' accepts no measurement bytes");
      streaming_node_has_raw_input[si] = true;

      // Window sizes may differ: enqueue_syndrome() meters to each node's own
      // room, so two windows read the same bytes on their own strides, and the
      // capacity reported is the largest.  Round size still has to agree --
      // one byte stream cannot have two, and the accessor reports just one.
      const auto node_round = entry.capability->num_msyn_per_round();
      n_msyn = std::max(n_msyn, entry.capability->num_msyn_per_decode());
      if (!any_raw_input) {
        n_round = node_round;
        any_raw_input = true;
        continue;
      }
      if (node_round != n_round)
        record_streaming_conflict(
            "streaming nodes disagree on num_msyn_per_round");
    }

    // Propagate effective schedules in node topological order. Each consumer
    // validates every individual edge after its producer's output schedule is
    // known, then derives schedules for its own output slots from all inputs.
    std::vector<std::vector<std::optional<window_spec>>> effective_schedule(N);
    for (const auto &entry : streaming_nodes)
      effective_schedule[entry.node_index].resize(
          node_output_names[entry.node_index].size());
    for (std::size_t si = 0; si < M && streaming_conflict.empty(); ++si) {
      const auto &entry = streaming_nodes[si];
      const int idx = entry.node_index;
      std::vector<streaming_task_graph_node::input_schedule> input_schedules;
      for (const auto &sw : wires_to[idx]) {
        if (sw.from_node < 0 || streaming_by_node[sw.from_node] < 0)
          continue;
        const auto &producer_schedule =
            effective_schedule[sw.from_node][sw.from_slot];
        if (!producer_schedule) {
          record_streaming_conflict(
              "streaming edge from '" + node_ids[sw.from_node] + "." +
              node_output_names[sw.from_node][sw.from_slot] + "' to '" +
              node_ids[idx] + "." + node_input_names[idx][sw.to_slot] +
              "' has no declared producer schedule");
          break;
        }
        const auto validation_error =
            validate_streaming_edge_input(*entry.capability, sw.to_slot);
        if (!validation_error.empty()) {
          record_streaming_conflict("streaming edge into '" + node_ids[idx] +
                                    "." + node_input_names[idx][sw.to_slot] +
                                    "' is incompatible: " + validation_error);
          break;
        }
        input_schedules.push_back(
            {static_cast<std::size_t>(sw.to_slot), *producer_schedule});
      }
      if (!streaming_conflict.empty())
        break;
      const auto derivation_error = derive_streaming_output_schedules(
          *entry.capability, input_schedules, effective_schedule[idx]);
      if (!derivation_error.empty())
        record_streaming_conflict(
            "streaming node '" + node_ids[idx] +
            "' could not derive its output schedule: " + derivation_error);
      // `done` fires once per window the node receives
      if (!completion_wires_from[idx].empty())
        for (const auto &in : input_schedules)
          if (!(in.schedule == input_schedules[0].schedule))
            record_streaming_conflict(
                "node '" + node_ids[idx] +
                "' feeds its implicit 'done' output but its inputs deliver on "
                "different streaming schedules, so it has no single "
                "completion per window");
      if (has_done[idx])
        effective_schedule[idx].back() =
            input_schedules.empty() ? entry.capability->streaming_schedule()
                                    : input_schedules[0].schedule;
    }

    // Sharing an output requires the same firing and commit schedule. Equal
    // window sizes alone are insufficient: unequal strides diverge after the
    // first fire, while unequal guards commit different regions and flush
    // different tails at the shot boundary. Odd-parity value contributors are
    // the alignment set. Structural reachability is used only when every path
    // cancels, because that constant-zero output still needs a publication
    // schedule to distinguish it from an unavailable value.
    const auto num_sinks = wires_to.back().size();
    output_streaming_schedules.assign(num_sinks, {});
    for (std::size_t k = 0; k < num_sinks && streaming_conflict.empty(); ++k) {
      std::optional<window_spec> shared;
      std::string first_endpoint;
      for (std::size_t si = 0; si < M; ++si) {
        const auto &entry = streaming_nodes[si];
        const int idx = entry.node_index;
        for (std::size_t slot = 0; slot < output_reach[idx].size(); ++slot) {
          const auto &path = output_reach[idx][slot][k];
          if (output_has_parity_contributor[k] ? !path.parity : !path.reachable)
            continue;
          const auto &schedule = effective_schedule[idx][slot];
          const auto endpoint =
              node_ids[idx] + "." + node_output_names[idx][slot];
          if (!schedule) {
            record_streaming_conflict(
                "streaming output '" + endpoint + "' feeds graph output '" +
                output_names[k] +
                "' but does not declare its effective firing and commit "
                "schedule");
            break;
          }
          if (first_endpoint.empty()) {
            shared = schedule;
            first_endpoint = endpoint;
            continue;
          }
          if (!(*schedule == *shared))
            record_streaming_conflict(
                "streaming outputs '" + first_endpoint + "' and '" + endpoint +
                "' feed graph output '" + output_names[k] +
                "' but disagree on their effective streaming schedule "
                "(window_size, step_size, start_guard, or end_guard); nodes "
                "folded into one output must fire and commit the same regions "
                "together, so align their schedules or wire them to separate "
                "outputs");
        }
      }
      if (shared)
        output_streaming_schedules[k] = *shared;
    }
    if (!any_raw_input)
      record_streaming_conflict(
          "no streaming node consumes measurement results, so "
          "enqueue_syndrome would never reach the graph");

    // Compile the streaming-only dependency graph. Reachability orders
    // overlapping roots and reconvergent fan-out while predecessor counts let
    // independent roots flush without a global barrier.
    // Completion wires order a consumer after a producer but carry no value,
    // so they count as dependencies without making the two branches overlap:
    // reachability for blocking follows value wires only.
    std::vector<std::vector<std::size_t>> value_dependents(M);
    streaming_dependents.assign(M, {});
    streaming_predecessor_counts.assign(M, 0);
    for (std::size_t si = 0; si < M; ++si) {
      const auto node_idx = streaming_nodes[si].node_index;
      for (const auto &wire : wires_to[node_idx]) {
        if (wire.from_node < 0 || streaming_by_node[wire.from_node] < 0)
          continue;
        const auto producer_si =
            static_cast<std::size_t>(streaming_by_node[wire.from_node]);
        streaming_dependents[producer_si].push_back(si);
        if (!is_done_slot(wire.from_node, wire.from_slot))
          value_dependents[producer_si].push_back(si);
        ++streaming_predecessor_counts[si];
        ++streaming_wire_count;
      }
    }
    streaming_reachability.assign(M, std::vector<uint8_t>(M, uint8_t{0}));
    for (std::size_t si = M; si-- > 0;) {
      streaming_reachability[si][si] = 1;
      for (const auto dependent : value_dependents[si])
        for (std::size_t descendant = 0; descendant < M; ++descendant)
          streaming_reachability[si][descendant] |=
              streaming_reachability[dependent][descendant];
    }
    streaming_overlap.assign(M, std::vector<uint8_t>(M, uint8_t{0}));
    for (std::size_t left = 0; left < M; ++left)
      for (std::size_t right = left; right < M; ++right)
        for (std::size_t descendant = 0; descendant < M; ++descendant)
          if (streaming_reachability[left][descendant] &&
              streaming_reachability[right][descendant]) {
            streaming_overlap[left][right] = 1;
            streaming_overlap[right][left] = 1;
            break;
          }
    if (M > 0)
      streaming_parallel_width = maximum_parallel_width(streaming_dependents);

    streaming_node_generations.assign(M, 0);
    streaming_output_fire_ordinals.assign(num_sinks, 0);
    reset_streaming_progress();
  }

  {
    const auto max_parallel_width =
        std::max(one_shot_parallel_width, streaming_parallel_width);
    const auto hardware_threads =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const auto pool_threads = std::min(max_parallel_width, hardware_threads);
    // Width-one graphs still use the same event loop, with the caller as the
    // sole worker, so execution semantics do not depend on pool width.
    if (N > 1 || !streaming_nodes.empty())
      parallel_pool = std::make_unique<detail::thread_pool>(pool_threads);
    if (parallel_pool) {
      // Causal ancestors remain resident until their descendants drain.
      // This linear multiple covers all simultaneously live frontier,
      // publication, continuation, and flush descriptors without hot-path
      // allocation.
      const auto event_capacity =
          std::max<std::size_t>(64, 16 * (N + streaming_wire_count + 1));
      prepare_event_runtime(event_capacity);
    }
  }

  node_ids.clear();
  node_ids.shrink_to_fit();
  edges.clear();
  edges.shrink_to_fit();
}

// ---------------------------------------------------------------------------
// decoding_task_graph — impl-taking constructor / destructor / move
// ---------------------------------------------------------------------------

decoding_task_graph::decoding_task_graph(std::unique_ptr<impl> i)
    : impl_(std::move(i)) {}

decoding_task_graph::~decoding_task_graph() = default;
decoding_task_graph::decoding_task_graph(decoding_task_graph &&) noexcept =
    default;
decoding_task_graph &
decoding_task_graph::operator=(decoding_task_graph &&) noexcept = default;

// ---------------------------------------------------------------------------
// Threading-contract tripwire
// ---------------------------------------------------------------------------

namespace {

/// Detects an overlapping call instead of serialising it. Decoder-backed inner
/// decoders, streaming accumulators, and impl's scratch buffers are
/// unsynchronised, so overlap is corruption -- two overlapping run_async calls
/// abort the process on a corrupted heap -- and the caller has to be told
/// rather than quietly slowed down.
///
/// Overlap-based, never thread-identity based: handing a graph from one thread
/// to another is legal as long as the first is done, which is what the
/// realtime session does when it joins its host loop.
///
/// Always on rather than #ifndef NDEBUG, because CI builds Release with tests
/// enabled: an assert here would enforce nothing in the configuration that
/// actually ships, and could not be tested.
class in_flight_guard {
public:
  in_flight_guard(std::atomic<uint32_t> &counter, const char *caller)
      : counter_(counter) {
    if (counter_.fetch_add(1, std::memory_order_acq_rel) != 0) {
      // Undo our own increment before unwinding. A constructor that throws
      // runs no destructor, so this is the only decrement on this path.
      counter_.fetch_sub(1, std::memory_order_acq_rel);
      throw std::logic_error(
          std::string("decoding_task_graph::") + caller +
          ": another call is already in flight on this graph instance. The "
          "decoder-backed inner decoders, streaming accumulators, and the "
          "graph's scratch buffers are stateful and unsynchronised, so "
          "overlapping calls corrupt each other. Collect "
          "the previous run_async() future before starting the next, keep "
          "streaming (enqueue_syndrome) and one-shot/batch (run/run_batch) use "
          "separated in time, finish or cancel an asynchronous streaming "
          "session before using synchronous APIs, or give each thread its own "
          "graph instance.");
    }
  }
  ~in_flight_guard() { release(); }

  /// Idempotent and callable from any thread: an async run releases from
  /// whichever thread collects the future, and the destructor covers the
  /// future a caller discarded instead.
  void release() noexcept {
    if (armed_.exchange(false, std::memory_order_acq_rel))
      counter_.fetch_sub(1, std::memory_order_acq_rel);
  }

  in_flight_guard(const in_flight_guard &) = delete;
  in_flight_guard &operator=(const in_flight_guard &) = delete;

private:
  std::atomic<uint32_t> &counter_;
  std::atomic<bool> armed_{true};
};

/// Frees the graph when the multi-node run_async body ends, rather than when
/// the caller gets around to collecting the future.
struct scoped_release {
  in_flight_guard &g;
  ~scoped_release() { g.release(); }
};

/// Single-node run_async: decode_async() launches with std::launch::async, so
/// the inner decoder is already running by the time run_async() returns and
/// the token has to outlive the call frame.
///
/// Member order is load-bearing. On a future the caller discards, ~inner joins
/// the decode and must run before the token is released; members are destroyed
/// in reverse declaration order, which gives exactly that. A lambda cannot
/// substitute here -- the order of closure members is unspecified.
struct pending_async_run {
  std::shared_ptr<in_flight_guard> token;
  std::future<logical_outcome> inner;
  logical_outcome operator()() {
    auto lo = inner.get();
    token->release();
    return lo;
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Streaming interface
// ---------------------------------------------------------------------------

// Count publications per streaming node; output barriers consume these counts.
void decoding_task_graph::impl::record_streaming_publication(int node_idx) {
  const int streaming_idx = streaming_by_node[node_idx];
  if (streaming_idx >= 0)
    ++streaming_node_generations[static_cast<std::size_t>(streaming_idx)];
}

// Publish each output through the slowest contributor, then derive the exact
// round range represented by that generation from its compiled schedule.
void decoding_task_graph::impl::refresh_streaming_progress(bool final) {
  auto &result = streaming_result_state;
  const auto previous_graph_generation = result.generation;
  const auto previous_graph_status = result.status;
  std::uint64_t graph_generation =
      result.outputs.empty() ? 0 : std::numeric_limits<std::uint64_t>::max();
  bool all_outputs_available = !result.outputs.empty();
  bool changed = false;

  for (std::size_t k = 0; k < result.outputs.size(); ++k) {
    auto &output = result.outputs[k];
    const auto previous_generation = output.generation;
    const auto previous_begin = output.coverage_begin_round;
    const auto previous_end = output.coverage_end_round;
    const auto previous_status = output.status;
    const auto &contributors = output_streaming_nodes[k];
    std::uint64_t generation =
        contributors.empty() ? 0 : std::numeric_limits<std::uint64_t>::max();
    for (const auto si : contributors)
      generation = std::min(generation, streaming_node_generations[si]);

    if (generation > output.generation) {
      const auto &schedule = output_streaming_schedules[k];
      std::optional<streaming_task_graph_node::committed_coverage> exact;
      std::size_t exact_count = 0;
      for (const auto si : contributors) {
        const auto coverage =
            streaming_nodes[si].capability->streaming_coverage();
        if (!coverage)
          continue;
        ++exact_count;
        if (exact && (exact->begin_round != coverage->begin_round ||
                      exact->end_round != coverage->end_round))
          throw std::logic_error(
              "task graph: contributors disagree on committed coverage");
        exact = coverage;
      }
      if (exact_count && exact_count != contributors.size())
        throw std::logic_error(
            "task graph: contributors mix exact and scheduled coverage");
      if (exact) {
        output.coverage_begin_round = exact->begin_round;
        output.coverage_end_round = exact->end_round;
        // A native shot reports exact coverage once FINAL ends its stream, so
        // scheduled fires after it number a new one from round 0.
        streaming_output_fire_ordinals[k] = 0;
      } else if (final && schedule.end_guard > 0) {
        const std::uint64_t fire_ordinal =
            streaming_output_fire_ordinals[k] == 0
                ? 0
                : streaming_output_fire_ordinals[k] - 1;
        output.coverage_begin_round =
            fire_ordinal * schedule.stride + schedule.size - schedule.end_guard;
        output.coverage_end_round =
            fire_ordinal * schedule.stride + schedule.size;
      } else {
        streaming_output_fire_ordinals[k] += generation - output.generation;
        const std::uint64_t ordinal = streaming_output_fire_ordinals[k] - 1;
        output.coverage_begin_round =
            ordinal == 0 ? 0 : ordinal * schedule.stride + schedule.start_guard;
        output.coverage_end_round =
            ordinal * schedule.stride + schedule.size - schedule.end_guard;
      }
      output.generation = generation;
      output.status = task_graph_streaming_status::available;
    } else if (final && output.generation > 0 &&
               output_streaming_schedules[k].end_guard > 0 &&
               std::none_of(contributors.begin(), contributors.end(),
                            [&](auto si) {
                              return streaming_nodes[si]
                                  .capability->streaming_coverage()
                                  .has_value();
                            })) {
      // A guarded tail is operational progress even when it projects to no
      // fault columns and flush() therefore has no correction bytes to XOR.
      const auto &schedule = output_streaming_schedules[k];
      const std::uint64_t fire_ordinal =
          streaming_output_fire_ordinals[k] == 0
              ? 0
              : streaming_output_fire_ordinals[k] - 1;
      output.coverage_begin_round =
          fire_ordinal * schedule.stride + schedule.size - schedule.end_guard;
      output.coverage_end_round =
          fire_ordinal * schedule.stride + schedule.size;
      ++output.generation;
    }
    if (final && output.generation > 0)
      output.status = task_graph_streaming_status::complete;
    all_outputs_available =
        all_outputs_available &&
        (output.status == task_graph_streaming_status::available ||
         output.status == task_graph_streaming_status::complete) &&
        output.generation > 0;
    graph_generation = std::min(graph_generation, output.generation);
    changed = changed || output.generation != previous_generation ||
              output.coverage_begin_round != previous_begin ||
              output.coverage_end_round != previous_end ||
              output.status != previous_status;
  }

  result.generation = graph_generation;
  result.status = task_graph_streaming_status::pending;
  if (all_outputs_available)
    result.status = final ? task_graph_streaming_status::complete
                          : task_graph_streaming_status::available;
  changed = changed || result.generation != previous_graph_generation ||
            result.status != previous_graph_status;
  if (changed)
    ++result.result_revision;
}

// Consume published values without disturbing window overlap or generations.
void decoding_task_graph::impl::consume_streaming_result() {
  auto &result = streaming_result_state;
  result.operation_completed = false;
  std::fill(result.values.begin(), result.values.end(), uint8_t{0});
  if (result.status == task_graph_streaming_status::failed) {
    ++result.result_revision;
    return;
  }
  result.status = task_graph_streaming_status::pending;
  for (std::size_t k = 0; k < result.outputs.size(); ++k) {
    auto &output = result.outputs[k];
    output.status = task_graph_streaming_status::pending;
    // A zero-value guarded tail is an output publication without a node fire.
    // Keep contributor barriers caught up before streaming resumes.
    for (const auto si : output_streaming_nodes[k])
      streaming_node_generations[si] =
          std::max(streaming_node_generations[si], output.generation);
  }
  ++result.result_revision;
}

// Begin a clean result epoch without reallocating any hot-path storage.
void decoding_task_graph::impl::reset_streaming_progress() {
  auto &result = streaming_result_state;
  ++result.shot_epoch;
  result.result_revision = 0;
  result.completed_operations = 0;
  result.generation = 0;
  result.status = task_graph_streaming_status::pending;
  result.operation_completed = false;
  result.error.clear();
  std::fill(streaming_node_generations.begin(),
            streaming_node_generations.end(), 0);
  std::fill(streaming_output_fire_ordinals.begin(),
            streaming_output_fire_ordinals.end(), 0);
  std::fill(result.values.begin(), result.values.end(), uint8_t{0});
  for (auto &output : result.outputs) {
    output.generation = 0;
    output.coverage_begin_round = 0;
    output.coverage_end_round = 0;
    output.status = task_graph_streaming_status::pending;
  }
}

// Latch an operational failure on the graph and every named output.
void decoding_task_graph::impl::fail_streaming_progress(
    const std::string &message) {
  auto &result = streaming_result_state;
  const bool changed = result.status != task_graph_streaming_status::failed ||
                       result.error != message;
  result.status = task_graph_streaming_status::failed;
  result.operation_completed = false;
  result.error = message;
  for (auto &output : result.outputs)
    output.status = task_graph_streaming_status::failed;
  if (changed)
    ++result.result_revision;
}

// Materialize the current parity-folded values beside their progress metadata.
void decoding_task_graph::impl::compose_streaming_values() const {
  auto &values = streaming_result_state.values;
  std::fill(values.begin(), values.end(), uint8_t{0});
  for (std::size_t si = 0; si < streaming_nodes.size(); ++si) {
    const auto &node = *streaming_nodes[si].capability;
    const uint8_t *corrections = node.corrections();
    const auto width = node.num_observables();
    for (const auto offset : streaming_node_output_offsets[si])
      for (std::size_t i = 0; i < width; ++i)
        values[offset + i] ^= corrections[i];
  }
}

namespace {
/// True when a round arriving now must be dropped because a FINAL round
/// already closed the shot.  Both enqueue overloads consult it: resuming
/// through the untagged one corrupts exactly as much as through the tagged.
bool dropped_after_final(bool finalized) {
  if (!finalized)
    return false;
  CUDA_QEC_WARN("decoding_task_graph: dropping a round after the shot was "
                "closed by a FINAL round; read the corrections and call "
                "clear_corrections() (or reset_decoder()) to start the next "
                "shot");
  return true;
}
} // namespace

// Run one queue record through the graph and publish the same progress model
// used by synchronous streaming. The caller owns graph execution exclusivity.
bool decoding_task_graph::impl::process_streaming_record(
    const uint8_t *syndrome, std::size_t len, std::optional<round_type> type) {
  auto &result = streaming_result_state;
  result.operation_completed = false;
  if (result.status == task_graph_streaming_status::failed)
    throw std::logic_error(
        "decoding_task_graph: streaming failed; reset the graph before "
        "enqueueing more measurements");
  if (dropped_after_final(finalized))
    return false;
  const auto generation = result.generation;
  try {
    const int cuda_device_id =
        parallel_pool && streaming_parallel_width > 1
            ? (async_streaming.active.load(std::memory_order_acquire)
                   ? async_streaming.cuda_device_id
                   : resolve_cuda_device_id())
            : -1;
    // Materialized, not short-circuited: the FINAL flush must run even when
    // the body also fires.
    const bool fired = enqueue_sweep(syndrome, len, type, cuda_device_id);
    refresh_streaming_progress(false);
    bool flushed = false;
    if (type == round_type::FINAL) {
      flushed = flush_sweep(cuda_device_id);
      finalized = true;
      refresh_streaming_progress(true);
    }
    result.operation_completed =
        result.generation > generation ||
        (type == round_type::FINAL && result.generation > 0 &&
         result.status == task_graph_streaming_status::complete);
    compose_streaming_values();
    return fired || flushed;
  } catch (const std::exception &e) {
    fail_streaming_progress(e.what());
    compose_streaming_values();
    throw;
  } catch (...) {
    fail_streaming_progress("unknown streaming failure");
    compose_streaming_values();
    throw;
  }
}

// Apply end-of-stream semantics when finish_async_streaming() closes a session
// that did not carry an explicit FINAL record.
bool decoding_task_graph::impl::finish_streaming_record() {
  auto &result = streaming_result_state;
  result.operation_completed = false;
  const auto generation = result.generation;
  try {
    const int cuda_device_id =
        parallel_pool && streaming_parallel_width > 1
            ? (async_streaming.active.load(std::memory_order_acquire)
                   ? async_streaming.cuda_device_id
                   : resolve_cuda_device_id())
            : -1;
    const bool flushed = flush_sweep(cuda_device_id);
    finalized = true;
    refresh_streaming_progress(true);
    result.operation_completed =
        result.generation > generation ||
        (result.generation > 0 &&
         result.status == task_graph_streaming_status::complete);
    compose_streaming_values();
    return flushed;
  } catch (const std::exception &e) {
    fail_streaming_progress(e.what());
    compose_streaming_values();
    throw;
  } catch (...) {
    fail_streaming_progress("unknown streaming failure");
    compose_streaming_values();
    throw;
  }
}

// The record stays untagged down to the nodes, so a decoder that tells an
// untagged round from BULK keeps its legacy fixed-length shot contract.
bool decoding_task_graph::enqueue_syndrome(const uint8_t *syndrome,
                                           std::size_t len) {
  if (impl_->async_streaming.enabled)
    return impl_->enqueue_syndrome_async(syndrome, len, std::nullopt,
                                         cuda_device_id().value_or(-1));
  const in_flight_guard guard(impl_->in_flight, "enqueue_syndrome");
  const bool fired =
      impl_->process_streaming_record(syndrome, len, std::nullopt);
  if (impl_->streaming_result_state.operation_completed)
    ++impl_->streaming_result_state.completed_operations;
  return fired;
}

bool decoding_task_graph::enqueue_syndrome(const uint8_t *syndrome,
                                           std::size_t len, round_type type) {
  if (type == round_type::FINAL && len == 0)
    throw std::invalid_argument("decoding_task_graph: a FINAL round must "
                                "carry the shot's final measurements");
  if (impl_->async_streaming.enabled)
    return impl_->enqueue_syndrome_async(syndrome, len, type,
                                         cuda_device_id().value_or(-1));
  const in_flight_guard guard(impl_->in_flight, "enqueue_syndrome");
  const bool fired = impl_->process_streaming_record(syndrome, len, type);
  if (impl_->streaming_result_state.operation_completed)
    ++impl_->streaming_result_state.completed_operations;
  return fired;
}

const uint8_t *decoding_task_graph::get_obs_corrections() const {
  if (impl_->async_streaming.enabled) {
    impl_->synchronize_async_streaming();
    return impl_->async_streaming_result().values.data();
  }
  const in_flight_guard guard(impl_->in_flight, "get_obs_corrections");
  auto &g = *impl_;
  g.require_streaming_support("get_obs_corrections");
  // One streaming node owning the whole buffer returns its corrections in
  // place, no copy.
  if (g.streaming_nodes.size() == 1 &&
      g.streaming_nodes[0].capability->num_observables() == g.n_obs &&
      g.streaming_node_output_offsets[0].size() == 1 &&
      g.streaming_node_output_offsets[0][0] == 0)
    return g.streaming_nodes[0].capability->corrections();

  // Multi-node/output graphs share the same graph-owned materialization used
  // by streaming_result(), avoiding a second correction buffer and XOR path.
  g.compose_streaming_values();
  return g.streaming_result_state.values.data();
}

// Refresh preallocated values; enqueue owns all metadata changes.
const task_graph_streaming_result &
decoding_task_graph::streaming_result() const {
  if (impl_->async_streaming.enabled) {
    impl_->synchronize_async_streaming();
    return impl_->async_streaming_result();
  }
  const in_flight_guard guard(impl_->in_flight, "streaming_result");
  impl_->require_streaming_support("streaming_result");
  impl_->compose_streaming_values();
  return impl_->streaming_result_state;
}

void decoding_task_graph::clear_corrections() {
  if (impl_->async_streaming.enabled &&
      impl_->async_streaming.active.load(std::memory_order_acquire))
    impl_->drain_async_streaming();
  const in_flight_guard guard(impl_->in_flight, "clear_corrections");
  impl_->invalidate_async_stream_snapshot();
  // This consumes the published accumulator mid-stream. Keep each node's
  // overlap buffer so the next stride still decodes the correct frame.
  impl_->finalized = false;
  for (const auto &entry : impl_->streaming_nodes)
    entry.capability->clear_corrections();
  impl_->consume_streaming_result();
  if (impl_->async_streaming.enabled) {
    std::lock_guard<std::mutex> lock(impl_->async_streaming.snapshot_mutex);
    impl_->async_streaming.read_snapshot = impl_->streaming_result_state;
  }
}

void decoding_task_graph::reset_decoder() {
  if (impl_->async_streaming.enabled &&
      impl_->async_streaming.active.load(std::memory_order_acquire))
    impl_->cancel_async_streaming();
  const in_flight_guard guard(impl_->in_flight, "reset_decoder");
  impl_->invalidate_async_stream_snapshot();
  impl_->finalized = false;
  impl_->async_streaming.needs_reset = false;
  for (auto &node : impl_->nodes)
    if (auto *decoder_backed = node->decoder_backed())
      decoder_backed->reset_decoder();
    else if (auto *streaming = node->streaming())
      streaming->reset_streaming_state();
  std::fill(impl_->streaming_deliveries_seen.begin(),
            impl_->streaming_deliveries_seen.end(), std::size_t{0});
  impl_->reset_streaming_progress();
  if (impl_->async_streaming.enabled) {
    std::lock_guard<std::mutex> lock(impl_->async_streaming.snapshot_mutex);
    impl_->async_streaming.read_snapshot = impl_->streaming_result_state;
  }
}

std::size_t decoding_task_graph::get_num_msyn_per_decode() const {
  return impl_->n_msyn;
}

std::size_t decoding_task_graph::get_num_msyn_per_round() const {
  return impl_->n_round;
}

std::size_t decoding_task_graph::get_num_observables() const {
  return impl_->n_obs;
}

bool decoding_task_graph::supports_streaming() const noexcept {
  return impl_->streaming_conflict.empty();
}

const std::string &
decoding_task_graph::streaming_incompatibility_reason() const noexcept {
  return impl_->streaming_conflict;
}

const sparse_binary_matrix &decoding_task_graph::d_matrix() const {
  return *impl_->d;
}

const sparse_binary_matrix &decoding_task_graph::o_matrix() const {
  if (!impl_->o)
    throw std::logic_error("o_matrix: not a single-window-node graph");
  return *impl_->o;
}

std::optional<int> decoding_task_graph::cuda_device_id() const {
  const int resolved = impl_->resolve_cuda_device_id();
  return resolved < 0 ? std::nullopt : std::optional<int>{resolved};
}

bool decoding_task_graph::supports_graph_dispatch() const {
  if (impl_->nodes.size() != 1)
    return false;
  const auto *decoder_backed = impl_->nodes[0]->decoder_backed();
  return decoder_backed && decoder_backed->supports_graph_dispatch();
}

void *decoding_task_graph::capture_graph(int reserved_sms) {
  if (impl_->nodes.size() != 1)
    throw std::logic_error(
        "decoding_task_graph::capture_graph: multi-node graphs (combine()) "
        "do not support DEVICE-mode capture; check supports_graph_dispatch() "
        "first");
  auto *decoder_backed = impl_->nodes[0]->decoder_backed();
  if (!decoder_backed || !decoder_backed->supports_graph_dispatch())
    throw std::logic_error(
        "decoding_task_graph::capture_graph: this graph's node does not "
        "support graph dispatch; check supports_graph_dispatch() first");
  return decoder_backed->capture_graph(reserved_sms);
}

void decoding_task_graph::release_graph(void *graph_resources) {
  if (impl_->nodes.size() != 1)
    throw std::logic_error(
        "decoding_task_graph::release_graph: multi-node graphs never "
        "captured a graph to release");
  if (auto *decoder_backed = impl_->nodes[0]->decoder_backed())
    decoder_backed->release_graph(graph_resources);
}

decoder *decoding_task_graph::inner_decoder() const {
  if (impl_->nodes.size() != 1)
    return nullptr;
  const auto *decoder_backed = impl_->nodes[0]->decoder_backed();
  return decoder_backed ? decoder_backed->inner_decoder() : nullptr;
}

// ---------------------------------------------------------------------------
// One-shot and batch execution interface
// ---------------------------------------------------------------------------

logical_outcome decoding_task_graph::run(const measurement_results &m) const {
  const in_flight_guard guard(impl_->in_flight, "run");
  return impl_->run_dag(m, "run");
}

// Synchronous one-shot APIs intentionally use the serial topological walk.
// It avoids ready-queue and worker-dispatch overhead when the caller is already
// waiting for the result. Multi-node run_async() uses the event runtime because
// its contract permits independent ready nodes to overlap.
void decoding_task_graph::impl::walk_dag(const measurement_results &m) {
  auto &g = *this;

  // Route the raw shot to every "measurements" source edge; each node applies
  // its own D.  Refilled in place, so a shot costs one vector copy and no
  // lookup.
  g.meas_val = m;

  auto get_val = [&](const compiled_wire &w) -> const edge_value & {
    return (w.from_node < 0) ? g.meas_val
                             : g.node_outputs[w.from_node][w.from_slot];
  };

  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    auto &inputs = g.parallel_node_inputs[i];
    for (const auto &w : g.wires_to[i])
      inputs[w.to_slot] = get_val(w);
    g.nodes[i]->execute_indexed(inputs, g.node_outputs[i]);
  }
}

void decoding_task_graph::impl::require_single_output(const char *who) const {
  const auto &sink_wires = wires_to.back();
  if (sink_wires.empty())
    throw std::logic_error("decoding_task_graph: no edge reaches output");
  if (sink_wires.size() > 1)
    throw std::logic_error(
        "decoding_task_graph: " + std::to_string(sink_wires.size()) +
        " edges reach output; " + who +
        "() returns a single outcome. Call run_all() instead, once per shot");
}

logical_outcome decoding_task_graph::impl::run_dag(const measurement_results &m,
                                                   const char *who) {
  auto &g = *this;
  // Checked before the walk: refuse rather than do the work and drop the rest.
  g.require_single_output(who);
  const auto &sink_wires = g.wires_to.back();
  g.walk_dag(m);
  const auto &sw = sink_wires[0];
  return g.node_outputs[sw.from_node][sw.from_slot].get<logical_outcome>();
}

std::vector<logical_outcome>
decoding_task_graph::run_all(const measurement_results &m) const {
  const in_flight_guard guard(impl_->in_flight, "run_all");
  auto &g = *impl_;
  const auto &sink_wires = g.wires_to.back();
  if (sink_wires.empty())
    throw std::logic_error("decoding_task_graph: no edge reaches output");
  g.walk_dag(m);
  std::vector<logical_outcome> outcomes;
  outcomes.reserve(sink_wires.size());
  for (const auto &sw : sink_wires)
    outcomes.push_back(
        g.node_outputs[sw.from_node][sw.from_slot].get<logical_outcome>());
  return outcomes;
}

const std::vector<std::string> &decoding_task_graph::output_names() const {
  return impl_->output_names;
}

std::future<logical_outcome>
decoding_task_graph::run_async(const measurement_results &m) const {
  // The single-node fast path bypasses run_dag(), so validate before selecting
  // either path or arming the in-flight guard.
  impl_->require_single_output("run_async");

  // Held for the lifetime of the work, not of this call: the inner decode is
  // already on a worker thread by the time we return.
  auto token = std::make_shared<in_flight_guard>(impl_->in_flight, "run_async");

  if (impl_->nodes.size() == 1) {
    const auto *decoder_backed = impl_->nodes[0]->decoder_backed();
    if (!decoder_backed)
      throw std::logic_error(
          "task_graph_node: run_async not implemented for this node type");
    return std::async(std::launch::deferred,
                      pending_async_run{token, decoder_backed->run_async(m)});
  }

  // The coordinator starts unpinned. Resolve the graph device here so a node
  // disagreement throws synchronously, then let each persistent pool worker
  // pin itself before executing a ready node.
  const int cuda_id = cuda_device_id().value_or(-1);
  return std::async(std::launch::async, [this, m, cuda_id, token] {
    const scoped_release rel{*token};
    cudaq::qec::detail_affinity::CudaDeviceGuard dev(cuda_id);
    return impl_->run_dag_parallel(m, "run_async", cuda_id);
  });
}

std::vector<logical_outcome> decoding_task_graph::run_batch(
    const std::vector<measurement_results> &shots) const {
  // One token for the whole batch, over the unguarded walk: a per-shot guard
  // would leave a detection gap between shots.
  const in_flight_guard guard(impl_->in_flight, "run_batch");
  std::vector<logical_outcome> results;
  results.reserve(shots.size());
  for (const auto &m : shots)
    results.push_back(impl_->run_dag(m, "run_batch"));
  return results;
}

} // namespace cudaq::qec
