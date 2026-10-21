/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// from_topology(): the one place a task graph is built from configuration.
// from_spec() and the flat constructor express themselves as one-node
// topologies, so node construction, DEM expansion, and wire compilation each
// exist once.

#include "decoding_task_graph_impl.h"
#include "decoding_task_graph_utils.h"
#include "decoding_task_node_kind_registry.h"
#include "measurement_frame_lengths.h"
#include "cudaq/qec/cancellation.h"
#include "cudaq/qec/candidate_sink.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/extended_dem.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace cudaq::qec {

namespace {

/// Every type a node's ports name must be registered: a wire into an
/// unregistered one would fail as a bare type mismatch, and an edge_value
/// built for it would throw on the first fire.  An input is checked by asking
/// whether it accepts anything registered at all, since it is a predicate.
void require_registered_port_types(const task_graph_node &node,
                                   const std::string &id) {
  const auto registered = registered_edge_value_types();
  const auto outs = node.outputs();
  for (std::size_t slot = 0; slot < outs.size(); ++slot) {
    const auto type = node.output_type(slot);
    if (type && std::find(registered.begin(), registered.end(), *type) ==
                    registered.end())
      throw std::invalid_argument(
          "assemble: node '" + id +
          "' declares an unregistered edge-value type on output '" +
          outs[slot] + "'");
  }
  const auto ins = node.inputs();
  for (std::size_t slot = 0; slot < ins.size(); ++slot)
    if (std::none_of(registered.begin(), registered.end(),
                     [&](const edge_value_type &type) {
                       return node.accepts_input_type(slot, type);
                     }))
      throw std::invalid_argument("assemble: node '" + id + "' input '" +
                                  ins[slot] +
                                  "' accepts no registered edge-value type");
}

/// Resolve one node's DEM and inner decoder into a build context.
node_build_context prepare_node(task_graph_node_config &cfg,
                                const task_graph_node_kind_spec &kind) {
  const std::string where = "from_topology: node '" + cfg.id + "'";
  node_build_context ctx;
  const decoder_output_request decoder_output =
      kind.inner_decoder_output(cfg.node_params);
  const bool decoder_takes_observables =
      kind.inner_decoder_takes_observables(cfg.node_params);
  // Native observable streaming needs D and the observable shape, not a
  // closed H/rates model. Other kinds still need their finite window's H.
  const bool native_streaming_chunks =
      kind.name == "streaming" && decoder_takes_observables &&
      cfg.dem.has_value() && !cfg.decoder_instance && cfg.stim_dem.empty() &&
      !cfg.decoder_type.empty() &&
      !decoder_needs_model_matrices(cfg.decoder_type,
                                    decoder_model_source::dem_chunks);
  ctx.id = cfg.id;
  ctx.node_params = &cfg.node_params;

  const bool has_flat = !cfg.d_sparse.empty();
  // A kind applying O itself, including a guarded window, reads an error
  // pattern. Its stim_dem is therefore the node's whole model rather than the
  // inner decoder's alone, so O and H both come from the text and share the
  // fault columns O*e is taken in.
  const bool dem_from_text =
      !cfg.stim_dem.empty() && !decoder_takes_observables;
  if (dem_from_text && cfg.dem.has_value())
    throw std::invalid_argument(
        where + " of kind '" + kind.name +
        "' takes its whole model from stim_dem, so it cannot also set dem; "
        "give it the text and a d_sparse to read measurements with");
  if (kind.requires_dem && !cfg.dem.has_value() && !has_flat)
    throw std::invalid_argument(where + " of kind '" + kind.name +
                                "' needs a DEM: set dem or d_sparse/o_sparse");
  if (!kind.requires_dem) {
    if (cfg.dem.has_value() || has_flat)
      throw std::invalid_argument(where + " of kind '" + kind.name +
                                  "' takes no DEM");
    return ctx;
  }

  sparse_binary_matrix d_mat, o_mat;
  std::vector<std::size_t> flat_frames;
  // The closed DEM is kept rather than a prebuilt decoder handle so its model
  // data can be placed into one immutable decoder_init after the node's
  // independent input and output policies have both been resolved.
  std::shared_ptr<const detector_error_model> closed_dem;
  std::optional<dem_chunks_spec> prepared_chunks;
  if (cfg.dem.has_value()) {
    // window_size sizes the DEM as well as the buffer: it is the round count
    // the phases expand to, and the stream may then run for any number of
    // rounds beyond it.
    dem_chunks_spec working = *cfg.dem;
    if (cfg.node_params.contains("window_size"))
      working.num_rounds = cfg.node_params.get<std::size_t>("window_size");
    else if (!working.num_rounds.has_value() && working.has_repeating_phase())
      working.num_rounds = working.minimum_rounds();

    std::vector<extended_dem> chunks;
    if (!native_streaming_chunks)
      chunks = dem_chunks_from_spec(working);
    if (kind.name == "sliding_window" &&
        cfg.node_params.get<bool>("arbitrary_length", false) &&
        cfg.stim_dem.empty() && !cfg.decoder_instance) {
      ctx.memory_spec = std::make_shared<const dem_chunks_spec>(working);
      ctx.decoder_type = cfg.decoder_type;
      ctx.decoder_params = cfg.params;
    }
    const auto rounds = native_streaming_chunks
                            ? working.phase_sequence().size()
                            : chunks.size();
    if (rounds > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(where + ": too many DEM rounds");
    ctx.dem_rounds = static_cast<uint32_t>(rounds);
    if (!native_streaming_chunks)
      ctx.fault_offsets = dem_chunks_to_fault_offsets(chunks);
    // The spec form honors per-phase measurement maps; the chunk list alone
    // can only say the memory convention.
    const auto d_rows = dem_chunks_to_d_sparse(working);
    d_mat = make_csr(d_rows, "D");
    // A round's trailing measurements may be read by no detector, so the
    // declared count, not the widest referenced index, is the stream width.
    if (const auto declared = dem_chunks_measurement_count(working);
        declared && *declared > d_mat.num_cols()) {
      if (*declared >
          std::numeric_limits<sparse_binary_matrix::index_type>::max())
        throw std::invalid_argument(where + ": declared measurement count " +
                                    std::to_string(*declared) +
                                    " exceeds the index range");
      d_mat =
          sparse_binary_matrix::from_nested_csr(
              static_cast<sparse_binary_matrix::index_type>(d_rows.size()),
              static_cast<sparse_binary_matrix::index_type>(*declared), d_rows)
              .canonicalize();
    }
    if (native_streaming_chunks) {
      // Preserve o_matrix()'s real projection without expanding every fault
      // model or closing H. The compact map owns the fault-column convention.
      const auto observable_map = dem_chunks_to_o_chunked(working);
      const auto columns = observable_map.num_error_columns();
      if (columns >
          std::numeric_limits<sparse_binary_matrix::index_type>::max())
        throw std::invalid_argument(where +
                                    ": too many observable fault columns");
      std::vector<std::vector<uint32_t>> rows(
          observable_map.columns.front().num_rows());
      observable_map.for_each_flip(
          columns, [](std::size_t) { return true; },
          [&](std::size_t column, uint32_t observable) {
            rows[observable].push_back(static_cast<uint32_t>(column));
          });
      o_mat = make_csr(rows, "O");
    } else {
      o_mat = make_csr(dem_chunks_to_o_sparse(chunks), "O");
      closed_dem = std::make_shared<const detector_error_model>(
          dem_close_all(chunks, working.seam.from_seam, working.seam.to_seam));
    }
    // Rounds may differ in width (a first round with one stabilizer type's
    // detectors, a data readout last). Then there is no equal split, and spr
    // is the typical round's, with the whole stream's width kept alongside.
    if (d_mat.num_cols() % ctx.dem_rounds == 0) {
      ctx.spr = static_cast<uint32_t>(d_mat.num_cols()) / ctx.dem_rounds;
    } else {
      if (!working.measurements_per_round)
        throw std::invalid_argument(
            where + ": its rounds differ in width, so the dem needs "
                    "measurements_per_round to name the typical round's");
      ctx.spr = static_cast<uint32_t>(*working.measurements_per_round);
      ctx.num_measurements = d_mat.num_cols();
    }
    prepared_chunks.emplace(std::move(working));
  } else {
    if (cfg.node_params.contains("measurement_frame_lengths_file")) {
      flat_frames = detail::read_measurement_frame_lengths(
          cfg.node_params.get<std::string>("measurement_frame_lengths_file"));
      std::size_t width = 0;
      for (const auto frame : flat_frames)
        width += frame;
      using index_type = sparse_binary_matrix::index_type;
      if (width > std::numeric_limits<index_type>::max() ||
          required_width(cfg.d_sparse, "D") > width)
        throw std::invalid_argument(
            where + ": measurement frames cannot describe D columns");
      d_mat = sparse_binary_matrix::from_nested_csr(
                  static_cast<index_type>(cfg.d_sparse.size()),
                  static_cast<index_type>(width), cfg.d_sparse)
                  .canonicalize();
    } else {
      d_mat = make_csr(cfg.d_sparse, "D");
    }
    if (dem_from_text) {
      // The text carries no measurement map, so D stays the node's.
      auto parsed = dem_from_stim_text(cfg.stim_dem);
      o_mat = sparse_binary_matrix(parsed.observables_flips_matrix,
                                   sparse_binary_matrix_layout::csr)
                  .canonicalize();
      closed_dem =
          std::make_shared<const detector_error_model>(std::move(parsed));
    } else {
      o_mat = make_csr(cfg.o_sparse, "O");
    }
    // Flat matrices say nothing about round structure, so one measurement is
    // one "round" unless the node states otherwise. Either way the window
    // covers the whole matrix; see make_window_node().
    ctx.spr = static_cast<uint32_t>(
        cfg.node_params.get<std::size_t>("num_syndromes_per_round", 1));
    if (ctx.spr == 0)
      throw std::invalid_argument(where + ": num_syndromes_per_round is zero");
    if (cfg.node_params.contains("measurement_frame_lengths_file")) {
      if (flat_frames.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(where + ": too many measurement frames");
      ctx.dem_rounds = static_cast<uint32_t>(flat_frames.size());
    } else {
      if (d_mat.num_cols() % ctx.spr != 0)
        throw std::invalid_argument(
            where + ": num_syndromes_per_round " + std::to_string(ctx.spr) +
            " does not divide the " + std::to_string(d_mat.num_cols()) +
            " measurement columns of D");
      ctx.dem_rounds = static_cast<uint32_t>(d_mat.num_cols()) / ctx.spr;
    }
  }

  ctx.d = std::make_shared<sparse_binary_matrix>(std::move(d_mat));
  ctx.o = std::make_shared<sparse_binary_matrix>(std::move(o_mat));
  ctx.closed_dem = closed_dem;
  // Resolve H once, from whichever form described this node, so a kind that
  // needs it does not have to know which that was. CSR: its rows are the
  // detectors `d` produces.
  if (closed_dem)
    ctx.h = std::make_shared<const sparse_binary_matrix>(
        sparse_binary_matrix(closed_dem->detector_error_matrix,
                             sparse_binary_matrix_layout::csr)
            .canonicalize());
  else if (!native_streaming_chunks && !cfg.h_sparse.empty())
    ctx.h = std::make_shared<const sparse_binary_matrix>(
        make_csr(cfg.h_sparse, "H"));
  if (ctx.h && ctx.h->num_rows() != ctx.d->num_rows())
    throw std::invalid_argument(
        where + ": H has " + std::to_string(ctx.h->num_rows()) +
        " detector rows but D produces " + std::to_string(ctx.d->num_rows()));

  if (!kind.requires_decoder) {
    if (cfg.decoder_instance || !cfg.decoder_type.empty())
      throw std::invalid_argument(where + " of kind '" + kind.name +
                                  "' takes no decoder");
    return ctx;
  }
  if (cfg.decoder_instance) {
    if (!cfg.stim_dem.empty())
      throw std::invalid_argument(
          where + " supplies both stim_dem and a built decoder; the text is "
                  "only used to build one");
    ctx.dec = std::move(cfg.decoder_instance);
  } else if (!cfg.stim_dem.empty() && !dem_from_text) {
    if (cfg.decoder_type.empty())
      throw std::invalid_argument(where + " of kind '" + kind.name +
                                  "' needs an inner decoder name");
    // The text is the decoder's whole model, including the observable map and
    // the error rates make_pcm_decoder derives from it, so the chunk DEM's
    // defaults are deliberately not merged in: they index a different fault
    // column space and would be silently wrong.
    ctx.dec = get_decoder(cfg.decoder_type,
                          decoder_init::from_stim_dem(cfg.stim_dem, *ctx.d),
                          decoder_output, cfg.params);
    // Only a decoder that projects to observables itself can be driven this
    // way: any other returns an error pattern indexed in the text's mechanism
    // space, which this node's O and H do not share.
    if (ctx.dec->get_result_type() != decode_result_type::observables)
      throw std::invalid_argument(
          where +
          ": stim_dem needs an inner decoder that projects to "
          "observables itself; '" +
          cfg.decoder_type +
          "' returns an error pattern, which is indexed in the text's fault "
          "columns rather than this node's");
    if (ctx.dec->get_num_observables() != ctx.o->num_rows())
      throw std::invalid_argument(
          where + ": stim_dem declares " +
          std::to_string(ctx.dec->get_num_observables()) +
          " observables but this node's O has " +
          std::to_string(ctx.o->num_rows()) + " rows");
    if (ctx.dec->get_syndrome_size() != ctx.d->num_rows())
      throw std::invalid_argument(where + ": stim_dem declares " +
                                  std::to_string(ctx.dec->get_syndrome_size()) +
                                  " detectors but this node's D produces " +
                                  std::to_string(ctx.d->num_rows()));
  } else {
    if (cfg.decoder_type.empty())
      throw std::invalid_argument(where + " of kind '" + kind.name +
                                  "' needs an inner decoder name");
    if (!closed_dem && !native_streaming_chunks)
      throw std::invalid_argument(
          where +
          " has flat matrices, which cannot build a decoder (H and the error "
          "rates are not recoverable from D and O); supply a built decoder");
    if (!decoder_takes_observables && cfg.params.contains("O"))
      throw std::invalid_argument(
          where + " of kind '" + kind.name +
          "' handles O at the graph level, so its inner decoder must not be "
          "given one");
    auto inputs = [&] {
      // Native consumers need the authoritative recipe to preserve streaming
      // phase/FINAL semantics, not just its finite-window equivalent matrices.
      // Match the standalone decoder configuration's model-source policy.
      if (prepared_chunks &&
          !decoder_needs_model_matrices(cfg.decoder_type,
                                        decoder_model_source::dem_chunks)) {
        auto native =
            decoder_init::from_dem_chunks(std::move(*prepared_chunks), *ctx.d);
        return decoder_takes_observables ? std::move(native)
                                         : native.without_observables();
      }
      return decoder_takes_observables
                 ? decoder_init(*closed_dem, *ctx.d)
                 : decoder_init(*ctx.h, std::nullopt, closed_dem->error_rates,
                                *ctx.d, closed_dem->error_ids);
    }();
    ctx.dec = get_decoder(cfg.decoder_type, std::move(inputs), decoder_output,
                          cfg.params);
  }
  return ctx;
}

} // namespace

// ---------------------------------------------------------------------------
// Pattern lowering
// ---------------------------------------------------------------------------
//
// A pattern is a higher-level description written where a node would be and
// rewritten into plain nodes and edges before the ordinary build.

namespace {

/// A spec for one lowered ensemble. Consumed by `wire_patterns` below.
struct ensemble_wiring {
  struct member {
    std::vector<std::string> nodes; ///< prefixed ids, the push tail excluded
    std::string push;               ///< the push tail's id, or empty
  };
  std::string id, fan, join;
  std::vector<member> members;
  /// "<node id>.<output>" of every member answer (an edge reaching `output`)
  /// and of every member node's `output`: once built, they size the join.
  std::vector<std::pair<std::string, std::string>> answers, node_outputs;
};

/// Lower one ensemble pattern `cfg` (id `E`) into `E.fanout`, the prefixed
/// member nodes (`E.m<i>.<id>`), a `candidate_push` tail per member whose
/// edges reach `output`, and `E.join`; the lowered nodes are appended to `out`
/// in topological order and the edges rewritten/added in `edges`.
ensemble_wiring lower_ensemble(task_graph_node_config cfg,
                               std::vector<task_graph_node_config> &out,
                               std::vector<task_graph_edge> &edges) {
  ensemble_wiring wiring;
  const std::string id = wiring.id = cfg.id;
  const std::string what = "from_topology: ensemble '" + id + "'";
  auto spec = std::move(*cfg.ensemble);
  if (id.empty())
    throw std::invalid_argument("from_topology: every node needs an id");
  if (spec.members.empty())
    throw std::invalid_argument(what + " has no members");
  if (spec.join_strategy.empty())
    throw std::invalid_argument(what + " names no join_strategy");
  const std::string fan = wiring.fan = id + ".fanout",
                    join = wiring.join = id + ".join";

  // The outer edges: exactly one into `input`, any number out of `output`.
  // The fan-out takes its data type from whatever that one edge carries.
  std::size_t inputs_seen = 0;
  for (auto &e : edges) {
    if (e.to_node == id) {
      if (e.to_input != "input")
        throw std::invalid_argument(what + " has no input '" + e.to_input +
                                    "'");
      e.to_node = fan;
      ++inputs_seen;
    }
    if (e.from_node == id) {
      if (e.from_output != "output")
        throw std::invalid_argument(what + " has no output '" + e.from_output +
                                    "'");
      e.from_node = join;
    }
  }
  if (inputs_seen != 1)
    throw std::invalid_argument(what + " needs exactly one edge into 'input'");

  task_graph_node_config fan_cfg;
  fan_cfg.id = fan;
  fan_cfg.kind = "ensemble_fanout";
  fan_cfg.node_params.insert("strategy", spec.join_strategy);
  fan_cfg.node_params.insert("strategy_params", spec.join_params);
  out.push_back(std::move(fan_cfg));

  for (std::size_t i = 0; i < spec.members.size(); ++i) {
    auto &m = spec.members[i];
    ensemble_wiring::member mw;
    const std::string prefix = id + ".m" + std::to_string(i) + ".";
    const std::string member = what + " member " + std::to_string(i);
    if (m.nodes.empty())
      throw std::invalid_argument(member + " has no nodes");
    for (auto &n : m.nodes) {
      if (n.ensemble)
        throw std::invalid_argument(member + " nests an ensemble, which is "
                                             "not supported yet");
      if (n.id.empty())
        throw std::invalid_argument(member + " has a node without an id");
      n.id = prefix + n.id;
      mw.nodes.push_back(n.id);
      wiring.node_outputs.emplace_back(n.id, "output");
    }

    auto member_edges = std::move(m.edges);
    if (member_edges.empty())
      throw std::invalid_argument(member + " has no edges; name its data flow "
                                           "from 'input' and, unless its nodes "
                                           "push candidates, to 'output'");

    const std::string push = prefix + "push";
    bool reaches_output = false;
    for (auto &e : member_edges) {
      if (e.from_node == "input") {
        if (e.from_output != "output")
          throw std::invalid_argument(member + ": the ensemble input has only "
                                               "output 'output'");
        e.from_node = fan;
      } else {
        e.from_node = prefix + e.from_node;
      }
      if (e.to_node == "output") {
        if (reaches_output)
          throw std::invalid_argument(member + " reaches 'output' twice");
        reaches_output = true;
        wiring.answers.emplace_back(e.from_node, e.from_output);
        e.to_node = push;
        e.to_input = "input";
      } else {
        e.to_node = prefix + e.to_node;
      }
    }

    for (auto &n : m.nodes)
      out.push_back(std::move(n));
    if (reaches_output) {
      task_graph_node_config push_cfg;
      push_cfg.id = mw.push = push;
      push_cfg.kind = "candidate_push";
      out.push_back(std::move(push_cfg));
    }
    edges.insert(edges.end(), member_edges.begin(), member_edges.end());
    wiring.members.push_back(std::move(mw));
  }

  task_graph_node_config join_cfg;
  join_cfg.id = join;
  join_cfg.kind = "ensemble_join";
  out.push_back(std::move(join_cfg));
  edges.push_back({fan, "selection", join, "selection"});
  return wiring;
}

/// Rewrite every pattern entry of `nodes` into plain nodes, in place, so the
/// result is in topological order whenever the input was.
std::vector<ensemble_wiring>
lower_patterns(std::vector<task_graph_node_config> &nodes,
               std::vector<task_graph_edge> &edges) {
  std::vector<ensemble_wiring> wirings;
  if (std::none_of(nodes.begin(), nodes.end(),
                   [](const auto &cfg) { return cfg.ensemble.has_value(); }))
    return wirings;
  // A lone pattern gets the single-node wiring before it stops being one
  // node.
  if (nodes.size() == 1 && edges.empty())
    edges = {{"measurements", "output", nodes[0].id, "input"},
             {nodes[0].id, "output", "output", "input"}};
  std::vector<task_graph_node_config> lowered;
  lowered.reserve(nodes.size());
  for (auto &cfg : nodes) {
    if (cfg.ensemble)
      wirings.push_back(lower_ensemble(std::move(cfg), lowered, edges));
    else
      lowered.push_back(std::move(cfg));
  }
  nodes = std::move(lowered);
  return wirings;
}

/// The static width of a node's `logical_outcome` output, when it has one.
std::optional<std::size_t>
output_width(const std::pair<std::string, std::string> &ref,
             const std::vector<std::string> &ids,
             const std::vector<std::unique_ptr<task_graph_node>> &built) {
  const auto it = std::find(ids.begin(), ids.end(), ref.first);
  if (it == ids.end())
    return std::nullopt;
  const auto &node = *built[static_cast<std::size_t>(it - ids.begin())];
  const auto outs = node.outputs();
  const auto slot = std::find(outs.begin(), outs.end(), ref.second);
  if (slot == outs.end())
    return std::nullopt;
  const auto index = static_cast<std::size_t>(slot - outs.begin());
  if (node.output_type(index) != logical_outcome_type)
    return std::nullopt;
  return node.output_value_size(index);
}

/// The join's width: every member answer of known width, which must agree,
/// else the first member node `output` of known width. Unset when nothing is
/// known, which a streaming join reports in its own terms.
std::optional<std::size_t>
join_width(const ensemble_wiring &w, const std::vector<std::string> &ids,
           const std::vector<std::unique_ptr<task_graph_node>> &built) {
  std::optional<std::size_t> width;
  std::string first;
  for (const auto &ref : w.answers) {
    const auto this_width = output_width(ref, ids, built);
    if (!this_width)
      continue;
    const auto name = ref.first + "." + ref.second;
    if (width && *width != *this_width)
      throw std::invalid_argument(
          "from_topology: ensemble '" + w.id +
          "' members answer with different numbers of observables: '" + first +
          "' has " + std::to_string(*width) + ", '" + name + "' has " +
          std::to_string(*this_width));
    if (!width) {
      width = this_width;
      first = name;
    }
  }
  if (width)
    return width;
  for (const auto &ref : w.node_outputs)
    if (const auto node_width = output_width(ref, ids, built))
      return node_width;
  return std::nullopt;
}

/// The second step: with the nodes built, wire every member input that
/// accepts a candidate sink to the fan-out's sink and every one that accepts
/// a cancellation token to its token, whatever the ports are called; then
/// order the join after every such producer through the producers' implicit
/// `done` outputs, rebuilding the join with that many `done_<i>` inputs and
/// its width. Handle edges go in front of the data edges, so a streaming
/// member holds the run's handles before the window that fires it.
void wire_patterns(const std::vector<ensemble_wiring> &wirings,
                   const std::vector<std::string> &ids,
                   std::vector<std::unique_ptr<task_graph_node>> &built,
                   std::vector<task_graph_edge> &edges) {
  constexpr auto sink_type = edge_value_type::of<candidate_sink>();
  constexpr auto token_type = edge_value_type::of<cancellation_token>();
  const auto index_of = [&](const std::string &id) {
    return static_cast<std::size_t>(std::find(ids.begin(), ids.end(), id) -
                                    ids.begin());
  };
  std::vector<task_graph_edge> handles;
  for (const auto &w : wirings) {
    std::vector<std::string> producers;
    for (std::size_t i = 0; i < w.members.size(); ++i) {
      const auto &m = w.members[i];
      const auto before = producers.size();
      auto wire = [&](const std::string &id) {
        const auto &node = *built[index_of(id)];
        const auto ins = node.inputs();
        for (std::size_t slot = 0; slot < ins.size(); ++slot) {
          if (node.accepts_input_type(slot, sink_type)) {
            handles.push_back({w.fan, "sink", id, ins[slot]});
            producers.push_back(id);
          } else if (node.accepts_input_type(slot, token_type)) {
            handles.push_back({w.fan, "token", id, ins[slot]});
          }
        }
      };
      for (const auto &id : m.nodes)
        wire(id);
      if (producers.size() > before && !m.push.empty())
        throw std::invalid_argument(
            "from_topology: ensemble '" + w.id + "' member " +
            std::to_string(i) + ": node '" + producers.back() +
            "' pushes its own candidates, so the member must not also reach "
            "'output'");
      if (!m.push.empty())
        wire(m.push);
      if (producers.size() == before)
        throw std::invalid_argument(
            "from_topology: ensemble '" + w.id + "' member " +
            std::to_string(i) +
            " produces no candidates: no input accepts a candidate sink, and "
            "no edge reaches 'output'");
    }
    // The join follows every producer; built again now that their number and
    // the members' widths are known.
    task_graph_node_config join_cfg;
    join_cfg.id = w.join;
    if (const auto width = join_width(w, ids, built))
      join_cfg.node_params.insert("num_observables", *width);
    join_cfg.node_params.insert("done_inputs", producers.size());
    for (std::size_t k = 0; k < producers.size(); ++k)
      edges.push_back(
          {producers[k], "done", w.join, "done_" + std::to_string(k)});
    node_build_context ctx;
    ctx.id = w.join;
    ctx.node_params = &join_cfg.node_params;
    built[index_of(w.join)] = (*find_node_factory("ensemble_join"))(ctx);
  }
  edges.insert(edges.begin(), handles.begin(), handles.end());
}

} // namespace

std::unique_ptr<decoding_task_graph>
decoding_task_graph::assemble(std::unique_ptr<impl> i,
                              task_graph_options options) {
  for (std::size_t n = 0; n < i->nodes.size(); ++n)
    require_registered_port_types(*i->nodes[n], i->node_ids[n]);
  i->compile_wires();
  if (options.async_streaming) {
    i->prepare_async_streaming(options.async_queue_capacity, i->n_round,
                               options.async_coordinator_cpu,
                               options.async_spin_pause_iters);
    i->async_streaming.enabled = true;
  }
  return std::unique_ptr<decoding_task_graph>(
      new decoding_task_graph(std::move(i)));
}

std::unique_ptr<decoding_task_graph>
decoding_task_graph::from_topology(std::vector<task_graph_node_config> nodes,
                                   std::vector<task_graph_edge> edges,
                                   task_graph_options options) {
  // Keep every plugin-owned callback and node vtable below mapped through
  // factory dispatch and wire compilation. Unregister waits for this guard.
  auto kind_lock = lock_node_kind_registry();
  if (nodes.empty())
    throw std::invalid_argument("from_topology: at least one node is required");
  const auto patterns = lower_patterns(nodes, edges);

  // Inferred below, once the node exists: the sink edge has to name the node's
  // first output, and only a built node knows what that is called.  A kind
  // whose slot 0 is not the conventional "output" -- a `predecoder`'s `pre_L`
  // -- would otherwise have no single-node form at all.
  const bool infer_edges = edges.empty();
  if (infer_edges && nodes.size() != 1)
    throw std::invalid_argument(
        "from_topology: edges are required for a graph with " +
        std::to_string(nodes.size()) +
        " nodes; they are inferred only for a single node");

  auto i = std::make_unique<impl>();
  std::unordered_set<std::string> seen_ids;
  for (auto &cfg : nodes) {
    if (cfg.id.empty())
      throw std::invalid_argument("from_topology: every node needs an id");
    if (cfg.id == "measurements" || cfg.id == "dets" || cfg.id == "output")
      throw std::invalid_argument("from_topology: node id '" + cfg.id +
                                  "' is reserved for a synthetic endpoint");
    if (!seen_ids.insert(cfg.id).second)
      throw std::invalid_argument("from_topology: duplicate node id '" +
                                  cfg.id + "'");
    const auto *kind = find_task_graph_node_kind(cfg.kind);
    if (!kind)
      throw std::invalid_argument("from_topology: node '" + cfg.id +
                                  "' has unregistered kind '" + cfg.kind + "'");
    const auto *factory = find_node_factory(cfg.kind);
    if (!factory)
      throw std::invalid_argument("from_topology: node kind '" + kind->name +
                                  "' is registered but has no builder");

    const auto *schema = decoding::config::find_decoder_schema(
        task_graph_node_schema_name(kind->name));
    if (!schema)
      throw std::logic_error("from_topology: node kind '" + kind->name +
                             "' is registered without a parameter schema");
    decoding::config::materialize_default_args(*schema, cfg.node_params);
    decoding::config::validate_custom_args(*schema, cfg.node_params,
                                           "from_topology: node '" + cfg.id +
                                               "' parameters");

    auto ctx = prepare_node(cfg, *kind);
    // The first node carrying a DEM owns the graph's D; a single-node graph
    // also exposes its O.  o_matrix() is not meaningful once outcomes are
    // combined, so a multi-node graph leaves it unset, as combine() does.
    if (ctx.d && !i->d) {
      i->d = ctx.d;
      if (nodes.size() == 1)
        i->o = ctx.o;
    }
    auto node = (*factory)(ctx);
    if (!node)
      throw std::logic_error("from_topology: node kind '" + kind->name +
                             "' built nothing for node '" + cfg.id + "'");
    i->node_ids.push_back(cfg.id);
    i->nodes.push_back(std::move(node));
  }

  wire_patterns(patterns, i->node_ids, i->nodes, edges);

  if (infer_edges) {
    const auto outs = i->nodes[0]->outputs();
    if (outs.empty())
      throw std::invalid_argument("from_topology: node '" + i->node_ids[0] +
                                  "' declares no outputs, so a single-node "
                                  "graph has nothing to publish");
    edges = {{"measurements", "output", i->node_ids[0], "input"},
             {i->node_ids[0], outs.front(), "output", "input"}};
  }

  // `dets` was the original public spelling for the raw measurement source.
  // Accept it at the API boundary while storing only the unambiguous name.
  for (auto &edge : edges)
    if (edge.from_node == "dets")
      edge.from_node = "measurements";

  i->edges = std::move(edges);
  return assemble(std::move(i), options);
}

} // namespace cudaq::qec
