/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `parallel_window` pattern: written as one node, wired measurements ->
// <id> -> output, it stands for the parallel-window graph of an ingest node,
// two decode_window layers A and B, and their xor. Its kind registers the
// parameters, so configurations that use it parse and validate, and is
// registered as a pattern: the graph builder lowers it into those nodes
// before anything is built, and a configuration declares its layers'
// windows.

#include "decoding_task_parallel_window_pattern.h"
#include "decoding_task_node_kind_registry.h"
#include "decoding_task_window_kind_params.h"
#include "cudaq/qec/decoder_config_schema.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::qec {
namespace {

/// Throws std::runtime_error unless string parameter @p key, when set, is
/// one of @p first or @p second.
void check_choice(const cudaqx::heterogeneous_map &p, const std::string &key,
                  const std::string &first, const std::string &second) {
  if (!p.contains(key))
    return;
  const auto value = p.get<std::string>(key);
  if (value != first && value != second)
    throw std::runtime_error("parallel_window: " + key + " must be " + first +
                             " or " + second + "; got '" + value + "'");
}

/// A's commit and buffer zones are required; B's gap defaults to
/// commit_rounds + 2 buffer_rounds. The rest pass through to the layers.
decoding::config::decoder_schema parallel_window_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"commit_rounds", param_kind::uint64, true});
  schema.params.push_back({"buffer_rounds", param_kind::uint64, true});
  schema.params.push_back({"gap_rounds", param_kind::uint64});
  schema.params.push_back({"boundary_prior", param_kind::f64});
  schema.params.push_back({"output", param_kind::string});
  schema.params.push_back({"input_boundary", param_kind::string});
  schema.params.push_back({"merge_identical", param_kind::boolean});
  schema.params.push_back({"conflicting_parallel", param_kind::string});
  schema.params.push_back({"verify_residual", param_kind::string});
  schema.validate = [](const cudaqx::heterogeneous_map &p) {
    check_choice(p, "output", "corrected_logical", "corrections");
    check_choice(p, "input_boundary", "columns", "truncate");
    check_choice(p, "conflicting_parallel", "reject", "most_likely");
    if (p.contains("boundary_prior")) {
      const auto prior = p.get<double>("boundary_prior");
      if (!(prior > 0.0 && prior < 1.0))
        throw std::runtime_error("parallel_window: boundary_prior must be "
                                 "strictly between 0 and 1");
      if (p.contains("input_boundary") &&
          p.get<std::string>("input_boundary") == "truncate")
        throw std::runtime_error("parallel_window: boundary_prior is for "
                                 "input_boundary columns; truncate keeps each "
                                 "crossing fault's own prior");
    }
    verify_residual_of(p, "parallel_window");
  };
  return schema;
}

// It reads the template and names the layers' inner decoder. Its lowering
// replaces it before the build, and its layers are windows A and B.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  node_pattern pattern;
  spec.name = "parallel_window";
  spec.requires_decoder = true;
  spec.requires_dem = true;
  spec.parameter_minimums = {
      {"commit_rounds", 1}, {"buffer_rounds", 1}, {"gap_rounds", 1}};
  // A round count beyond the window round limit lies past any shot, and
  // within it the layers' rounds and period stay within int64.
  spec.parameter_maximums = {
      {"commit_rounds", experimental::window_round_limit},
      {"buffer_rounds", experimental::window_round_limit},
      {"gap_rounds", experimental::window_round_limit}};
  pattern.lower = lower_parallel_window;
  pattern.windows = [](const cudaqx::heterogeneous_map &params,
                       const experiment_template &context) {
    auto [a, b] = parallel_window_requests(params, context);
    return node_pattern::window_layers{{"A", std::move(a)},
                                       {"B", std::move(b)}};
  };
  register_pattern_kind(std::move(spec), parallel_window_params(),
                        std::move(pattern));
  return true;
}();

// The largest chance, over the detectors faults cross into, that an odd
// number of those faults fire: each detector combines the priors of the
// faults that cross into it as merged columns do, p = p1 + p2 - 2 p1 p2, so
// a detector crossed by one fault keeps that fault's prior. The shot holds
// every pair of neighbouring phases, and a round's outgoing seam band holds
// the next round's rows its faults touch.
double crossing_prior(const experiment_template &t) {
  const auto R = t.repeats() ? t.min_rounds() + 2 : t.min_rounds();
  double largest = 0.0;
  for (std::size_t r = 0; r + 1 < R; ++r) {
    const auto &chunk = t.chunk(r, R);
    const auto columns = chunk.H.to_nested_csc();
    for (const auto &seam : chunk.seams) {
      std::vector<double> odd(seam.num_rows(), 0.0);
      if (seam.id != t.spec().seam.from_seam)
        continue;
      for (std::size_t c = 0; c < columns.size(); ++c)
        for (const auto row : columns[c])
          if (row >= seam.row_begin && row < seam.row_end) {
            const auto p = chunk.error_rates[c];
            auto &q = odd[row - seam.row_begin];
            q = q + p - 2 * q * p;
          }
      for (const auto q : odd)
        largest = std::max(largest, q);
    } // end - for(seam)
  } // end - for(r)
  return largest;
} // end - crossing_prior()

} // namespace

// The parameters are checked against the pattern's schema first, so the
// round counts are at least 1 and boundary_prior is not set with truncate,
// whoever built them.
std::pair<experimental::window_request, experimental::window_request>
parallel_window_requests(const cudaqx::heterogeneous_map &checked,
                         const experiment_template &context) {
  using namespace experimental;
  const auto *schema = decoding::config::find_decoder_schema(
      task_graph_node_schema_name("parallel_window"));
  auto params = checked;
  if (!schema)
    throw std::logic_error("parallel_window: the pattern is registered "
                           "without a parameter schema");
  decoding::config::materialize_default_args(*schema, params);
  decoding::config::validate_custom_args(*schema, params,
                                         "parallel_window parameters");
  const auto c_a =
      static_cast<std::int64_t>(params.get<std::size_t>("commit_rounds"));
  const auto b =
      static_cast<std::int64_t>(params.get<std::size_t>("buffer_rounds"));
  const auto c_b = static_cast<std::int64_t>(params.get<std::size_t>(
      "gap_rounds", static_cast<std::size_t>(c_a + 2 * b)));
  const bool truncate =
      params.get<std::string>("input_boundary", "columns") == "truncate";
  window_request a;
  window_request bb;
  window_request::column_options columns;
  columns.merge_identical = params.get<bool>("merge_identical", false);
  columns.conflicting_parallel =
      params.get<std::string>("conflicting_parallel", "reject") == "most_likely"
          ? parallel_resolution::most_likely
          : parallel_resolution::reject;

  a.period = c_a + c_b;
  a.commit = round_range{0, c_a};
  a.buffers = {{-b, 0}, {c_a, c_a + b}};
  a.input.disposition = port_disposition::open;
  if (truncate) {
    a.input.boundary = input_boundary::truncate;
  } else if (params.contains("boundary_prior")) {
    a.input.prior = params.get<double>("boundary_prior");
  } else {
    a.input.prior = crossing_prior(context);
    if (!(*a.input.prior > 0.0 && *a.input.prior < 1.0))
      throw std::invalid_argument(
          "parallel_window: no fault crosses into the next round with a "
          "prior strictly between 0 and 1, so A's open input has no default "
          "boundary_prior; set one");
  }
  a.output.disposition = port_disposition::open;
  a.sends = {{"forward", std::nullopt, {}}, {"backward", std::nullopt, {}}};
  a.columns = columns;

  bb.period = c_a + c_b;
  bb.offset = -c_b;
  bb.commit = round_range{0, c_b};
  bb.input.disposition = port_disposition::closed;
  bb.output.disposition = port_disposition::closed;
  bb.columns = columns;
  return {a, bb};
} // end - parallel_window_requests()

// The parameters are checked first, as the builder checks every node's, so
// a configuration built in code is held to the parsed one's rules. The
// layers share the pattern's inner decoder recipe; each builds its own.
void lower_parallel_window(task_graph_node_config cfg,
                           std::vector<task_graph_node_config> &out,
                           std::vector<task_graph_edge> &edges) {
  const std::string id = cfg.id;
  const std::string what = "from_topology: parallel_window '" + id + "'";
  const std::string ingest = id + ".ingest", a = id + ".A", b = id + ".B",
                    sum = id + ".sum";
  const auto *schema = decoding::config::find_decoder_schema(
      task_graph_node_schema_name("parallel_window"));
  std::size_t inputs_seen = 0;
  bool corrected = false;
  task_graph_node_config ingest_cfg;
  task_graph_node_config a_cfg;
  task_graph_node_config b_cfg;
  task_graph_node_config sum_cfg;
  if (id.empty())
    throw std::invalid_argument("from_topology: every node needs an id");
  if (!schema)
    throw std::logic_error(what + ": the pattern is registered without a "
                                  "parameter schema");
  decoding::config::materialize_default_args(*schema, cfg.node_params);
  decoding::config::validate_custom_args(*schema, cfg.node_params,
                                         what + " parameters");
  if (!cfg.dem)
    throw std::invalid_argument(what + " has no dem_chunks template");
  // Each layer builds its decoders from its windows' models.
  if (!cfg.stim_dem.empty() || cfg.decoder_instance)
    throw std::invalid_argument(
        what + " takes no " +
        (cfg.decoder_instance ? "built decoder" : "stim_dem") +
        ": each layer builds its decoders from its windows");
  if (!cfg.d_sparse.empty() || !cfg.o_sparse.empty() || !cfg.h_sparse.empty())
    throw std::invalid_argument(what + " takes no flat matrices; its "
                                       "dem_chunks template gives its windows");
  // Its parameters give its layers' windows; any set here would be dropped.
  if (!cfg.windows.empty())
    throw std::invalid_argument(what + " takes no windows: its layers are A "
                                       "and B, from its parameters");
  const experiment_template context(*cfg.dem);
  const auto [request_a, request_b] =
      parallel_window_requests(cfg.node_params, context);
  corrected = cfg.node_params.get<std::string>("output", "corrected_logical") ==
              "corrected_logical";

  // The outer edges: exactly one into `input`, any number out of `output`.
  for (auto &e : edges) {
    if (e.to_node == id) {
      if (e.to_input != "input")
        throw std::invalid_argument(what + " has no input '" + e.to_input +
                                    "'");
      e.to_node = ingest;
      ++inputs_seen;
    }
    if (e.from_node == id) {
      if (e.from_output != "output")
        throw std::invalid_argument(what + " has no output '" + e.from_output +
                                    "'");
      e.from_node = sum;
    }
  }
  if (inputs_seen != 1)
    throw std::invalid_argument(what + " needs exactly one edge into 'input'");
  // Reported here, naming the pattern, rather than by its lowered ingest.
  if (corrected && context.readout().empty())
    throw std::invalid_argument(
        what + " has output: corrected_logical, which reads the logical "
               "value through the last phase's m2o_sparse, and its template "
               "has none; use output: corrections, or give the last phase "
               "m2o_sparse");

  ingest_cfg.id = ingest;
  ingest_cfg.kind = "ingest";
  ingest_cfg.dem = std::move(cfg.dem);
  ingest_cfg.windows = {{"A", request_a, ""}, {"B", request_b, ""}};
  if (corrected)
    ingest_cfg.node_params.insert("read_logical", true);
  for (auto *layer : {&a_cfg, &b_cfg}) {
    layer->kind = "decode_window";
    layer->decoder_type = cfg.decoder_type;
    layer->params = cfg.params;
    // The layers are the nodes that fire as pools; ingest and the sum keep
    // one copy.
    layer->max_in_flight = cfg.max_in_flight;
    if (cfg.node_params.contains("verify_residual"))
      layer->node_params.insert(
          "verify_residual",
          cfg.node_params.get<std::string>("verify_residual"));
  }
  a_cfg.id = a;
  b_cfg.id = b;
  b_cfg.node_params.insert("num_receives", std::size_t{2});
  sum_cfg.id = sum;
  sum_cfg.kind = "xor";
  sum_cfg.node_params.insert("num_inputs", std::size_t{corrected ? 3 : 2});
  out.push_back(std::move(ingest_cfg));
  out.push_back(std::move(a_cfg));
  out.push_back(std::move(b_cfg));
  out.push_back(std::move(sum_cfg));

  edges.push_back({ingest, "A", a, "input"});
  edges.push_back({ingest, "B", b, "input"});
  edges.push_back({a, "forward", b, "receive_0", -1});
  edges.push_back({a, "backward", b, "receive_1"});
  edges.push_back({a, "output", sum, "in_0"});
  edges.push_back({b, "output", sum, "in_1"});
  if (corrected)
    edges.push_back({ingest, "logical", sum, "in_2"});
} // end - lower_parallel_window()

} // namespace cudaq::qec
