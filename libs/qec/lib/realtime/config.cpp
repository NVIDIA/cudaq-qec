/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "realtime_decoding.h"
#include "stim_dem_forms.h"
#include "window_request_yaml_traits.h"
#include "cudaq/qec/decoder_config_payload.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoder_rpc_limits.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/selection_strategy.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cudaq::qec::decoding::config {

bool decoder_custom_args_t::operator==(
    const decoder_custom_args_t &other) const {
  return custom_args_maps_equal(map_, other.map_);
}

const std::string &decoder_config::effective_decoder_type() const {
  return task_graph.has_value() ? task_graph->inner_type : type;
}

/// The names of a node list joined by ',', an ensemble entry as
/// `ensemble(<members joined by |>)`; nodes without a name are skipped.
static std::string describe_task_graph_nodes(
    const std::vector<decoder_config::task_graph_spec::node_spec> &nodes,
    const std::function<std::string(
        const decoder_config::task_graph_spec::node_spec &)> &name_of) {
  std::string out;
  for (const auto &node : nodes) {
    std::string name;
    if (node.ensemble.has_value()) {
      for (const auto &member : node.ensemble->members) {
        const auto inner = describe_task_graph_nodes(member.nodes, name_of);
        if (!inner.empty())
          name += (name.empty() ? "ensemble(" : "|") + inner;
      }
      if (!name.empty())
        name += ')';
    } else {
      name = name_of(node);
    }
    if (name.empty())
      continue;
    if (!out.empty())
      out += ',';
    out += name;
  }
  return out;
}

std::string decoder_config::describe_decoder_type() const {
  if (!task_graph.has_value())
    return type;
  if (!task_graph->inner_type.empty())
    return task_graph->inner_type;
  std::string decoders =
      describe_task_graph_nodes(task_graph->nodes, [](const auto &n) {
        return n.resolved_decoder_type();
      });
  if (!decoders.empty())
    return decoders;
  return describe_task_graph_nodes(
      task_graph->nodes, [](const auto &n) { return n.resolved_kind(); });
}

std::string decoder_config::task_graph_spec::node_spec::resolved_kind() const {
  return cudaq::qec::resolved_task_graph_node_kind(type);
}

std::string
decoder_config::task_graph_spec::node_spec::resolved_decoder_type() const {
  if (!inner_type.empty())
    return inner_type;
  return cudaq::qec::find_task_graph_node_kind(type) ? std::string{} : type;
}

namespace {

constexpr const char *kErrorRateVec = "error_rate_vec";

struct legacy_error_rates {
  std::vector<double> values;
  std::string path;
};

const param_spec *find_param(const decoder_schema &schema,
                             const std::string &key) {
  for (const auto &candidate : schema.params)
    if (candidate.key == key)
      return &candidate;
  return nullptr;
}

const decoder_schema *nested_schema_for(const param_spec &spec,
                                        const cudaqx::heterogeneous_map &map) {
  if (spec.kind == param_kind::subschema)
    return find_decoder_schema(spec.subschema);
  if (spec.kind != param_kind::discriminated ||
      !map.contains(spec.discriminator))
    return nullptr;
  try {
    return find_decoder_schema(spec.discriminator_prefix +
                               map.get<std::string>(spec.discriminator));
  } catch (const std::runtime_error &) {
    // Canonical schema validation below reports the ill-typed discriminator.
    return nullptr;
  }
}

void record_legacy_error_rates(const cudaqx::heterogeneous_map &map,
                               const std::string &path,
                               std::optional<legacy_error_rates> &found) {
  std::vector<double> values;
  try {
    values = map.get<std::vector<double>>(kErrorRateVec);
  } catch (const std::runtime_error &) {
    throw std::runtime_error("'" + path +
                             "' must be a sequence of floating-point values.");
  }
  if (found)
    throw std::runtime_error(
        "error_rate_vec is supplied more than once in decoder_custom_args: '" +
        found->path + "' and '" + path +
        "'. Supply model error rates in exactly one location.");
  found = legacy_error_rates{std::move(values), path};
}

cudaqx::heterogeneous_map strip_legacy_error_rates(
    const decoder_schema *schema, const cudaqx::heterogeneous_map &map,
    const std::string &path, std::optional<legacy_error_rates> &found) {
  cudaqx::heterogeneous_map filtered;
  for (const auto &kv : map) {
    if (kv.first == kErrorRateVec) {
      record_legacy_error_rates(map, path + "." + kErrorRateVec, found);
      continue;
    }

    const param_spec *spec = schema ? find_param(*schema, kv.first) : nullptr;
    const decoder_schema *nested_schema =
        spec ? nested_schema_for(*spec, map) : nullptr;
    if (nested_schema) {
      if (const auto *nested =
              std::any_cast<cudaqx::heterogeneous_map>(&kv.second)) {
        filtered.insert(kv.first,
                        strip_legacy_error_rates(nested_schema, *nested,
                                                 path + "." + kv.first, found));
        continue;
      }
    }
    filtered.insert(kv.first, kv.second);
  }
  return filtered;
}

struct normalized_decoder_args {
  cudaqx::heterogeneous_map plugin_args;
  std::optional<legacy_error_rates> legacy_rates;
};

normalized_decoder_args normalize_decoder_args(const decoder_config &config) {
  normalized_decoder_args normalized;
  normalized.plugin_args = strip_legacy_error_rates(
      find_decoder_schema(config.effective_decoder_type()),
      config.decoder_custom_args.map(), "decoder_custom_args",
      normalized.legacy_rates);
  if (!config.error_rate_vec.empty() && normalized.legacy_rates)
    throw std::runtime_error(
        "error_rate_vec is supplied both at the decoder configuration level "
        "and at '" +
        normalized.legacy_rates->path +
        "'. Supply model error rates in exactly one location.");
  return normalized;
}

} // namespace

void decoder_config::validate_custom_args() const {
  auto normalized = normalize_decoder_args(*this);
  config::validate_custom_args(effective_decoder_type(),
                               normalized.plugin_args);
}

std::vector<double> decoder_config::effective_error_rate_vec() const {
  auto normalized = normalize_decoder_args(*this);
  if (normalized.legacy_rates)
    return normalized.legacy_rates->values;
  return error_rate_vec;
}

/// Flatten the nested sparse form the decoder API uses into the -1-terminated
/// row form the configuration uses.
static std::vector<std::int64_t>
flatten_sparse_rows(const std::vector<std::vector<std::uint32_t>> &rows) {
  std::vector<std::int64_t> flat;
  std::size_t total = rows.size();
  for (const auto &row : rows)
    total += row.size();
  flat.reserve(total);
  for (const auto &row : rows) {
    for (const auto column : row)
      flat.push_back(static_cast<std::int64_t>(column));
    flat.push_back(-1);
  }
  return flat;
}

std::optional<cudaq::qec::detector_error_model>
expand_dem_chunks(decoder_config &config) {
  if (!config.dem_chunks.has_value() || !config.H_sparse.empty())
    return std::nullopt;

  const auto &spec = *config.dem_chunks;
  const cudaq::qec::seam_id from_seam = spec.seam.from_seam;
  const cudaq::qec::seam_id to_seam = spec.seam.to_seam;

  std::vector<cudaq::qec::extended_dem> chunks;
  try {
    chunks = cudaq::qec::dem_chunks_from_spec(spec);
  } catch (const std::invalid_argument &error) {
    throw std::runtime_error("Cannot expand dem_chunks for decoder " +
                             std::to_string(config.id) + ": " + error.what());
  }

  cudaq::qec::detector_error_model closed;
  std::vector<std::vector<std::uint32_t>> d_sparse;
  try {
    closed = cudaq::qec::dem_close_all(chunks, from_seam, to_seam);
    d_sparse = cudaq::qec::dem_chunks_to_d_sparse(spec);
  } catch (const std::exception &error) {
    throw std::runtime_error("Cannot close dem_chunks for decoder " +
                             std::to_string(config.id) + ": " + error.what());
  }

  config.block_size = closed.num_error_mechanisms();
  config.syndrome_size = closed.num_detectors();
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(closed.detector_error_matrix);
  config.O_sparse =
      cudaq::qec::pcm_to_sparse_vec(closed.observables_flips_matrix);
  config.D_sparse = flatten_sparse_rows(d_sparse);
  config.error_rate_vec = closed.error_rates;
  return closed;
}

cudaqx::heterogeneous_map
decoder_config::decoder_custom_args_to_heterogeneous_map() const {
  auto args = normalize_decoder_args(*this).plugin_args;
  if (const auto *schema = find_decoder_schema(effective_decoder_type())) {
    // Non-schema keys are warned-and-dropped and schema-declared defaults are
    // materialized. The YAML serializer starts from this plugin-safe map and
    // adds the framework-owned error-rate compatibility alias back only to
    // the wire representation.
    drop_non_schema_keys(*schema, args);
    materialize_default_args(*schema, args);
  }
  return args;
}

void multi_decoder_config::validate_custom_args() const {
  for (const auto &decoder : decoders)
    decoder.validate_custom_args();
}

// Post-parse pass over a schema-parsed custom-args map: materialize defaulted
// discriminated sections, then run the canonical registry validation
// (required keys, per-schema hooks; its unknown-key check is a no-op here
// because the parser already rejected unknown keys). Only invoked when the
// section was present in the input document, mirroring the previous behavior
// where an absent decoder_custom_args section skipped its mapping (and
// therefore its required-key checks) entirely.
static void finalize_parsed_args(const decoder_schema &schema,
                                 cudaqx::heterogeneous_map &map,
                                 const std::string &context) {
  materialize_default_args(schema, map);
  validate_custom_args(schema, map, context);
}

/// Checks the graph builder cannot make, because it never sees the YAML: which
/// form was named, and which keys may accompany it. Endpoint, port, and
/// topological-order errors are left to the builder, which already reports
/// them by node and port name.
using task_graph_node_spec = decoder_config::task_graph_spec::node_spec;

/// Whether a node needs no entry-level DEM: its kind takes none, it carries
/// its own, or every member node does.
static bool task_graph_node_describes_itself(const task_graph_node_spec &node) {
  if (node.ensemble.has_value()) {
    for (const auto &member : node.ensemble->members)
      for (const auto &member_node : member.nodes)
        if (!task_graph_node_describes_itself(member_node))
          return false;
    return true;
  }
  const auto *kind =
      cudaq::qec::find_task_graph_node_kind(node.resolved_kind());
  if (kind && !kind->requires_dem)
    return true;
  return node.dem_chunks.has_value() || !node.D_sparse.empty();
}

static void
reject_lowered_only_kind(const cudaq::qec::task_graph_node_kind_spec *kind,
                         const std::string &type, const std::string &where) {
  if (kind && kind->lowered_only)
    throw std::runtime_error(where + " has type '" + type +
                             "', which only the lowering of a pattern "
                             "produces; write an 'ensemble' entry instead");
}

/// The checks for an `ensemble` pattern entry. The YAML traits have resolved
/// join_strategy and its parameters and the builder's lowering checks the
/// wiring, which leaves the node form; the member nodes join `typed`.
static void validate_ensemble_node(
    const task_graph_node_spec &node, const std::string &where,
    std::vector<std::pair<const task_graph_node_spec *, std::string>> &typed) {
  if (!node.type.empty())
    throw std::runtime_error(where + " is an ensemble and takes no type");
  const auto &members = node.ensemble->members;
  for (std::size_t i = 0; i < members.size(); ++i)
    for (const auto &n : members[i].nodes) {
      const std::string node_where = where + " ensemble member " +
                                     std::to_string(i) + " node '" + n.id + "'";
      if (n.ensemble.has_value())
        throw std::runtime_error(node_where + " nests an ensemble, which is "
                                              "not supported yet");
      typed.emplace_back(&n, node_where);
    }
}

static void validate_task_graph(const decoder_config &config) {
  const auto &tg = *config.task_graph;
  const std::string who = "decoder " + std::to_string(config.id);
  const auto node_context = [&who](const std::string &id) {
    return who + ": task_graph node '" + id + "'";
  };

  if (tg.options.async_streaming && tg.options.async_queue_capacity == 0)
    throw std::runtime_error(
        who +
        ": task_graph.async_queue_capacity must be nonzero when asynchronous "
        "streaming is enabled");
  if (tg.options.async_coordinator_cpu < -1)
    throw std::runtime_error(
        who + ": task_graph.async_coordinator_cpu must be -1 or nonnegative");
  if (tg.options.async_num_workers == std::size_t{0})
    throw std::runtime_error(
        who + ": task_graph.async_num_workers must be at least 1");
  for (std::size_t i = 0; i < tg.options.async_worker_cpus.size(); ++i) {
    const int cpu = tg.options.async_worker_cpus[i];
    const auto &cpus = tg.options.async_worker_cpus;
    if (cpu < 0)
      throw std::runtime_error(who +
                               ": task_graph.async_worker_cpus has negative "
                               "CPU " +
                               std::to_string(cpu));
    // Two workers on one core can be deliberate, so it is a warning.
    if (std::find(cpus.begin(), cpus.begin() + i, cpu) != cpus.begin() + i)
      CUDA_QEC_WARN("{}: task_graph.async_worker_cpus lists CPU {} twice", who,
                    cpu);
    if (cpu == tg.options.async_coordinator_cpu)
      throw std::runtime_error(
          who + ": task_graph.async_worker_cpus includes CPU " +
          std::to_string(cpu) + ", which async_coordinator_cpu uses");
  }

  if (tg.nodes.empty()) {
    if (!tg.edges.empty())
      throw std::runtime_error(who +
                               ": task_graph.edges requires task_graph.nodes");
    if (tg.type.empty() || tg.inner_type.empty())
      throw std::runtime_error(
          who + ": task_graph needs either nodes, or type and inner_type "
                "together");
    return;
  }
  if (!tg.type.empty() || !tg.inner_type.empty())
    throw std::runtime_error(
        who + ": task_graph.nodes is mutually exclusive with task_graph.type "
              "and task_graph.inner_type; a nodes-form graph names its kind "
              "and inner decoder per node");

  // Typed nodes, top-level and ensemble members, with their error contexts.
  std::vector<std::pair<const task_graph_node_spec *, std::string>> typed;
  std::vector<std::string> ids;
  for (const auto &node : tg.nodes) {
    if (node.id.empty())
      throw std::runtime_error(who + ": every task_graph node needs an id");
    if (node.max_in_flight == 0)
      throw std::runtime_error(who + ": task_graph node '" + node.id +
                               "' has max_in_flight 0");
    // compile_wires() resolves edges through an id map, which would silently
    // collapse a repeated id onto one node.
    if (std::find(ids.begin(), ids.end(), node.id) != ids.end())
      throw std::runtime_error(who + ": duplicate task_graph node id '" +
                               node.id + "'");
    ids.push_back(node.id);
    if (node.ensemble.has_value() && !node.windows.empty())
      throw std::runtime_error(node_context(node.id) +
                               " is an ensemble entry, which takes no "
                               "windows; only an ingest node takes windows");
    if (node.ensemble.has_value())
      validate_ensemble_node(node, node_context(node.id), typed);
    else
      typed.emplace_back(&node, node_context(node.id));
  }
  for (const auto &[typed_node, where] : typed) {
    const auto &node = *typed_node;
    if (node.type.empty())
      throw std::runtime_error(where + " needs a type");

    const auto *kind = cudaq::qec::find_task_graph_node_kind(node.type);
    reject_lowered_only_kind(kind, node.type, where);
    if (!kind && !node.windows.empty())
      throw std::runtime_error(
          where + " has type '" + node.type +
          "', which is not a registered node kind, so it is taken as a "
          "decoder plugin name and sets windows, which only an ingest node "
          "takes");
    if (kind) {
      if (kind->requires_decoder && node.inner_type.empty())
        throw std::runtime_error(where + " of kind '" + kind->name +
                                 "' needs an inner decoder: set inner.type");
      if (!kind->requires_decoder &&
          (!node.inner_type.empty() || !node.decoder_custom_args.empty()))
        throw std::runtime_error(where + " of kind '" + kind->name +
                                 "' wraps no decoder, so it "
                                 "takes neither inner nor decoder_custom_args");
      if (!kind->requires_dem && node.dem_chunks.has_value())
        throw std::runtime_error(where + " of kind '" + kind->name +
                                 "' takes no dem_chunks");
      if (!kind->takes_windows && !node.windows.empty())
        throw std::runtime_error(where + " of kind '" + kind->name +
                                 "' sets windows, which only an ingest node "
                                 "takes");
      if (kind->takes_windows && node.windows.empty())
        throw std::runtime_error(where + " of kind '" + kind->name +
                                 "' needs at least one window");
      if (!kind->requires_dem &&
          (!node.H_sparse.empty() || !node.D_sparse.empty() ||
           !node.O_sparse.empty()))
        throw std::runtime_error(where + " of kind '" + kind->name +
                                 "' reads no DEM, so it takes no "
                                 "H_sparse, D_sparse or O_sparse");
      // Either spelling of the model is only ever used to build an inner
      // decoder, so a kind that wraps none has nothing to do with it.
      if (!kind->requires_decoder &&
          (!node.stim_dem.empty() || !node.stim_dem_path.empty()))
        throw std::runtime_error(
            where + " of kind '" + kind->name +
            "' wraps no decoder, so it takes neither stim_dem nor "
            "stim_dem_path");
    } else if (!node.inner_type.empty()) {
      // `type` fell through to a decoder plugin name, so there is no node kind
      // left for an inner decoder to be inner to.
      std::string kinds;
      for (const auto &name : cudaq::qec::registered_task_graph_node_kinds())
        kinds += (kinds.empty() ? "" : ", ") + name;
      throw std::runtime_error(
          where + " has type '" + node.type +
          "', which is not a registered node kind (" + kinds +
          "), so it is taken as a decoder plugin name and cannot also name an "
          "inner decoder");
    }

    for (const auto &window : node.windows) {
      try {
        cudaq::qec::experimental::check_window_request(window.request);
      } catch (const std::invalid_argument &e) {
        throw std::runtime_error(where + " window layer '" + window.layer +
                                 "': " + e.what());
      }
    }

    // Two descriptions of one node's matrices, with nothing choosing between.
    const bool node_flat = !node.H_sparse.empty() || !node.D_sparse.empty() ||
                           !node.O_sparse.empty();
    if (node_flat && node.dem_chunks.has_value())
      throw std::runtime_error(
          where +
          " sets both dem_chunks and flat matrices; a node's matrices come "
          "from exactly one of them");
    // A kind applying O itself takes it from its stim_dem, in the fault columns
    // its decoder is built in, so it states D alone.
    const auto *resolved =
        cudaq::qec::find_task_graph_node_kind(node.resolved_kind());
    const bool o_from_text =
        resolved &&
        !resolved->inner_decoder_takes_observables(node.node_params.map()) &&
        (!node.stim_dem.empty() || !node.stim_dem_path.empty());
    if (o_from_text && !node.O_sparse.empty())
      throw std::runtime_error(where + " of kind '" + resolved->name +
                               "' takes O from its stim_dem, so it states no "
                               "O_sparse of its own");
    if (node_flat && node.D_sparse.empty())
      throw std::runtime_error(where +
                               " has flat matrices but no D_sparse to read "
                               "measurements with");
    if (node_flat && node.O_sparse.empty() && !o_from_text)
      throw std::runtime_error(where +
                               " has flat matrices but not both D_sparse and "
                               "O_sparse, which come as a pair");
    // A row encoding names the columns it touches, not how many exist.
    if (!node.H_sparse.empty() &&
        (node.block_size == 0 || node.syndrome_size == 0))
      throw std::runtime_error(
          where +
          " has H_sparse but no block_size and syndrome_size to shape it");
    if (!node_flat && (node.block_size != 0 || node.syndrome_size != 0))
      throw std::runtime_error(where +
                               " sets block_size or syndrome_size but has no "
                               "flat matrices for them to size");

    // The path is read into stim_dem before the builder runs, so both set
    // would mean one silently losing to the other.
    if (!node.stim_dem.empty() && !node.stim_dem_path.empty())
      throw both_stim_forms(where);

    // Cross-field rules for the kind's own parameters live in the kind's
    // schema, so this runs the same hook for every kind, present and future.
    if (const auto *schema =
            cudaq::qec::task_graph_node_params_schema(node.type))
      validate_custom_args(*schema, node.node_params.map(),
                           where + " parameters");

    if (!node.node_params.map().contains("window_size"))
      continue;
    const auto window_size =
        node.node_params.map().get<std::size_t>("window_size");
    const auto &dem =
        node.dem_chunks.has_value() ? node.dem_chunks : config.dem_chunks;
    if (!dem.has_value())
      continue;
    // window_size is the round count the phases expand to, so it has to name a
    // chain the connections can actually describe.
    if (dem->num_rounds.has_value() && window_size > *dem->num_rounds)
      throw std::runtime_error(
          where + " has window_size " + std::to_string(window_size) +
          ", more than the " + std::to_string(*dem->num_rounds) +
          " rounds its DEM describes");
    const auto minimum = dem->minimum_rounds();
    if (window_size < minimum)
      throw std::runtime_error(node_context(node.id) + " has window_size " +
                               std::to_string(window_size) +
                               ", fewer than the " + std::to_string(minimum) +
                               " rounds its DEM's phase chain needs");
  }

  if (tg.nodes.size() > 1 && tg.edges.empty())
    throw std::runtime_error(
        who + ": task_graph.edges is required for a graph with " +
        std::to_string(tg.nodes.size()) +
        " nodes; edges are inferred only for a single node");

  bool reaches_output = false;
  for (const auto &edge : tg.edges)
    reaches_output = reaches_output || edge.to_node == "output";
  if (!tg.edges.empty() && !reaches_output)
    throw std::runtime_error(who +
                             ": no task_graph edge reaches 'output', so the "
                             "graph produces nothing");

  // Window sizes agree per output, not per graph, so compile_wires() checks
  // them against the built nodes: this layer cannot see what an edge reaches.
}

} // namespace cudaq::qec::decoding::config

namespace {

// LLVM's default double emitter rounds to six significant digits. Model priors
// and schema-typed decoder parameters must both survive export unchanged.
struct exact_yaml_double {
  double value = 0;
};

} // namespace

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(exact_yaml_double)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<exact_yaml_double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::decoding::config::decoder_config)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::phase_connection)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::seam_spec_entry)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::phase_spec_entry)

namespace {

using data_artifact_t = cudaq::qec::decoding::config::data_artifact;

/// What a parse of one configuration document shares: its data_artifacts,
/// read before the decoders, so a reference resolves while the decoder or
/// node naming it is parsed, and validation reads the resolved block.
struct config_parse_context {
  const std::vector<data_artifact_t> *artifacts = nullptr;
};

/// The entry named @p name in a list sorted by name, or nullptr. Names are
/// looked up only while a configuration is parsed.
const data_artifact_t *named_artifact(const std::vector<data_artifact_t> &list,
                                      const std::string &name) {
  const auto it =
      std::lower_bound(list.begin(), list.end(), name,
                       [](const data_artifact_t &e, const std::string &n) {
                         return e.name < n;
                       });
  return it != list.end() && it->name == name ? &*it : nullptr;
}

/// The entry a reference may be written back as: the document being written
/// holds one named @p name. Without one, as when a decoder is written on its
/// own, the copy is written inline.
const data_artifact_t *written_entry(llvm::yaml::IO &io,
                                     const std::string &name) {
  const auto *context =
      static_cast<const config_parse_context *>(io.getContext());
  if (name.empty() || !context || !context->artifacts)
    return nullptr;
  return named_artifact(*context->artifacts, name);
}

/// The kinds an entry holds, in key order, for checks and messages.
std::vector<std::string> artifact_kinds(const data_artifact_t &entry) {
  std::vector<std::string> kinds;
  if (entry.dem_chunks)
    kinds.push_back("dem_chunks");
  if (!entry.stim_dem.empty())
    kinds.push_back("stim_dem");
  if (!entry.stim_dem_path.empty())
    kinds.push_back("stim_dem_path");
  if (entry.window)
    kinds.push_back("window");
  if (!entry.edem_path.empty())
    kinds.push_back("edem_path");
  return kinds;
}

/// The entry @p name names, which must hold one of @p kinds; @p what names
/// them, and @p where the reference, in messages.
const data_artifact_t &find_artifact(llvm::yaml::IO &io,
                                     const std::string &name,
                                     const std::vector<std::string> &kinds,
                                     const std::string &what,
                                     const std::string &where) {
  const auto *context =
      static_cast<const config_parse_context *>(io.getContext());
  const auto prefix = where + " names data_artifacts entry '" + name + "'";
  if (!context || !context->artifacts)
    throw std::runtime_error(
        prefix + ", but it was parsed outside a "
                 "multi_decoder_config, which holds data_artifacts");
  const auto *entry = named_artifact(*context->artifacts, name);
  if (!entry)
    throw std::runtime_error(prefix + ", which does not exist");
  const auto held = artifact_kinds(*entry);
  if (held.size() != 1 ||
      std::find(kinds.begin(), kinds.end(), held.front()) == kinds.end())
    throw std::runtime_error(prefix + ", but '" + name + "' is a " +
                             (held.empty() ? "kindless" : held.front()) +
                             " entry, not a " + what + " entry");
  return *entry;
}

/// Checks each entry on its own and against the others: exactly one kind;
/// edem_path reserved; sha256 only on a file, as 64 hex digits; a window's
/// request valid and its context a dem_chunks entry; a template valid.
void validate_data_artifacts(const std::vector<data_artifact_t> &list) {
  for (const auto &entry : list) {
    const auto where = "data_artifacts entry '" + entry.name + "'";
    const auto kinds = artifact_kinds(entry);
    if (kinds.empty())
      throw std::runtime_error(where +
                               " has no kind: give one of dem_chunks, "
                               "stim_dem, stim_dem_path, window or edem_path");
    if (kinds.size() == 2)
      throw std::runtime_error(where + " has two kinds, '" + kinds[0] +
                               "' and '" + kinds[1] + "'; give one");
    if (kinds.size() > 2)
      throw std::runtime_error(where + " has " + std::to_string(kinds.size()) +
                               " kinds; give one");
    if (!entry.edem_path.empty())
      throw std::runtime_error(where + ": edem_path is not supported until "
                                       "the cudaq-edem migration");
    if (!entry.sha256.empty()) {
      if (entry.stim_dem_path.empty())
        throw std::runtime_error(where +
                                 ": sha256 pins a file, so it needs a "
                                 "file-backed entry such as stim_dem_path");
      const bool hex =
          std::all_of(entry.sha256.begin(), entry.sha256.end(), [](char c) {
            return std::isxdigit(static_cast<unsigned char>(c));
          });
      if (entry.sha256.size() != 64 || !hex)
        throw std::runtime_error(where +
                                 ": sha256 must be 64 hexadecimal digits");
    }
    if (!entry.context.empty() && !entry.window)
      throw std::runtime_error(where +
                               ": context belongs to a window entry only");
    if (entry.window) {
      const auto *context = named_artifact(list, entry.context);
      if (entry.context.empty())
        throw std::runtime_error(where + ": a window entry needs a context "
                                         "naming a dem_chunks entry");
      if (!context || !context->dem_chunks)
        throw std::runtime_error(where + ": context '" + entry.context +
                                 "' is not a dem_chunks entry");
      try {
        cudaq::qec::experimental::check_window_request(*entry.window);
      } catch (const std::invalid_argument &e) {
        throw std::runtime_error(where + ": " + e.what());
      }
    }
    if (entry.dem_chunks) {
      try {
        entry.dem_chunks->validate();
      } catch (const std::invalid_argument &e) {
        throw std::runtime_error("Invalid dem_chunks for " + where + ": " +
                                 e.what());
      }
    }
  } // end - for(entry)
} // end - validate_data_artifacts()

} // namespace
LLVM_YAML_IS_SEQUENCE_VECTOR(
    cudaq::qec::decoding::config::decoder_config::task_graph_spec::node_spec)
LLVM_YAML_IS_SEQUENCE_VECTOR(
    cudaq::qec::decoding::config::decoder_config::task_graph_spec::node_spec::
        ensemble_spec::member)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::task_graph_edge)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<uint64_t>)
LLVM_YAML_IS_SEQUENCE_VECTOR(
    cudaq::qec::decoding::config::syndrome_mapping_config)

namespace llvm::yaml {

template <>
struct ScalarTraits<exact_yaml_double> {
  static void output(const exact_yaml_double &probability, void *context,
                     raw_ostream &stream) {
    // Preserve LLVM's established spelling for nonfinite parameter values.
    // Phase-prior validation rejects them independently.
    if (!std::isfinite(probability.value)) {
      ScalarTraits<double>::output(probability.value, context, stream);
      return;
    }
    std::array<char, 64> buffer;
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), probability.value);
    assert(result.ec == std::errc() &&
           "the buffer must fit the shortest double representation");
    const StringRef text(buffer.data(), result.ptr - buffer.data());
    const auto exponent = text.find_first_of("eE");
    // YAML 1.1 readers such as PyYAML infer an exponent-only spelling (1e-06)
    // as a string. A decimal point preserves numeric interpretation without
    // changing the shortest representation's value or significant digits.
    if (exponent != StringRef::npos && !text.contains('.'))
      stream << text.take_front(exponent) << ".0" << text.drop_front(exponent);
    else
      stream << text;
  }
  static StringRef input(StringRef text, void *context,
                         exact_yaml_double &value) {
    return ScalarTraits<double>::input(text, context, value.value);
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

// Binds a heterogeneous_map to the decoder_schema that describes it so the
// generic mapping traits below can convert between the two. The schema drives
// everything: which keys are legal, the canonical storage type of each value,
// and how nested sections resolve their schemas.
struct schema_binding {
  cudaqx::heterogeneous_map *map = nullptr;
  const cudaq::qec::decoding::config::decoder_schema *schema = nullptr;
};

namespace {

template <typename T>
void input_schema_scalar(IO &io, const std::string &key,
                         cudaqx::heterogeneous_map &map) {
  T value{};
  io.mapRequired(key.c_str(), value);
  map.insert(key, value);
}

template <typename T>
void output_schema_scalar(IO &io, const std::string &key,
                          const cudaqx::heterogeneous_map &map) {
  auto value = map.get<T>(key);
  if constexpr (std::is_same_v<T, double>) {
    exact_yaml_double exact{value};
    io.mapRequired(key.c_str(), exact);
  } else if constexpr (std::is_same_v<T, std::vector<double>>) {
    std::vector<exact_yaml_double> exact;
    exact.reserve(value.size());
    for (const auto element : value)
      exact.push_back({element});
    io.mapRequired(key.c_str(), exact);
  } else if constexpr (std::is_same_v<T, std::vector<std::vector<double>>>) {
    std::vector<std::vector<exact_yaml_double>> exact;
    exact.reserve(value.size());
    for (const auto &row : value) {
      auto &converted = exact.emplace_back();
      converted.reserve(row.size());
      for (const auto element : row)
        converted.push_back({element});
    }
    io.mapRequired(key.c_str(), exact);
  } else {
    io.mapRequired(key.c_str(), value);
  }
}

} // namespace

template <>
struct CustomMappingTraits<schema_binding> {
  using param_kind = cudaq::qec::decoding::config::param_kind;
  using param_spec = cudaq::qec::decoding::config::param_spec;

  static void inputOne(IO &io, StringRef key, schema_binding &binding) {
    const std::string key_str = key.str();
    // Compatibility spelling for the framework-owned model rates. It is
    // accepted in every decoder parameter section, then promoted out before
    // schema validation or plugin construction.
    if (key_str == "error_rate_vec") {
      input_schema_scalar<std::vector<double>>(io, key_str, *binding.map);
      return;
    }
    const param_spec *spec = nullptr;
    for (const auto &candidate : binding.schema->params) {
      if (candidate.key == key_str) {
        spec = &candidate;
        break;
      }
    }
    if (!spec)
      throw std::runtime_error("Unknown key '" + key_str + "' in '" +
                               binding.schema->name + "' parameters.");

    switch (spec->kind) {
    case param_kind::boolean:
      input_schema_scalar<bool>(io, key_str, *binding.map);
      break;
    case param_kind::int32:
      input_schema_scalar<int>(io, key_str, *binding.map);
      break;
    case param_kind::uint64:
      input_schema_scalar<std::size_t>(io, key_str, *binding.map);
      break;
    case param_kind::f64:
      input_schema_scalar<double>(io, key_str, *binding.map);
      break;
    case param_kind::string:
      input_schema_scalar<std::string>(io, key_str, *binding.map);
      break;
    case param_kind::f64_vec:
      input_schema_scalar<std::vector<double>>(io, key_str, *binding.map);
      break;
    case param_kind::f64_matrix:
      input_schema_scalar<std::vector<std::vector<double>>>(io, key_str,
                                                            *binding.map);
      break;
    case param_kind::subschema: {
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(spec->subschema);
      if (!nested_schema)
        throw std::runtime_error("No schema registered under '" +
                                 spec->subschema + "' (needed to parse '" +
                                 key_str + "').");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    case param_kind::discriminated: {
      // The nested schema is named by a sibling key. Read it through the IO
      // (document order does not matter; mapping keys are random access).
      std::string discriminator_value;
      io.mapOptional(spec->discriminator.c_str(), discriminator_value);
      if (discriminator_value.empty())
        throw std::runtime_error("'" + key_str + "' is present but '" +
                                 spec->discriminator + "' is not set.");
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(
              spec->discriminator_prefix + discriminator_value);
      if (!nested_schema)
        throw std::runtime_error(
            "'" + key_str + "' does not support " + spec->discriminator + " '" +
            discriminator_value +
            "': no parameter schema is registered under that name.");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    }
  }

  static void output(IO &io, schema_binding &binding) {
    // Only schema keys and the framework compatibility alias are emitted;
    // surface anything else (a typo in a programmatically built map) instead
    // of dropping it silently.
    for (const auto &kv : *binding.map) {
      bool known = false;
      for (const auto &spec : binding.schema->params) {
        if (spec.key == kv.first) {
          known = true;
          break;
        }
      }
      if (!known && kv.first != "error_rate_vec")
        CUDA_QEC_WARN("Key '{}' is not in the '{}' parameter schema; it is "
                      "omitted from the emitted YAML.",
                      kv.first, binding.schema->name);
    }
    if (binding.map->contains("error_rate_vec"))
      output_schema_scalar<std::vector<double>>(io, "error_rate_vec",
                                                *binding.map);
    // Emit in schema declaration order so output is deterministic.
    for (const auto &spec : binding.schema->params) {
      if (spec.key == "error_rate_vec")
        continue;
      if (!binding.map->contains(spec.key))
        continue;
      switch (spec.kind) {
      case param_kind::boolean:
        output_schema_scalar<bool>(io, spec.key, *binding.map);
        break;
      case param_kind::int32:
        output_schema_scalar<int>(io, spec.key, *binding.map);
        break;
      case param_kind::uint64:
        output_schema_scalar<std::size_t>(io, spec.key, *binding.map);
        break;
      case param_kind::f64:
        output_schema_scalar<double>(io, spec.key, *binding.map);
        break;
      case param_kind::string:
        output_schema_scalar<std::string>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_vec:
        output_schema_scalar<std::vector<double>>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_matrix:
        output_schema_scalar<std::vector<std::vector<double>>>(io, spec.key,
                                                               *binding.map);
        break;
      case param_kind::subschema: {
        const auto *nested_schema =
            cudaq::qec::decoding::config::find_decoder_schema(spec.subschema);
        if (!nested_schema)
          throw std::runtime_error("No schema registered under '" +
                                   spec.subschema + "' (needed to emit '" +
                                   spec.key + "').");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      case param_kind::discriminated: {
        std::string discriminator_value;
        if (binding.map->contains(spec.discriminator))
          discriminator_value =
              binding.map->get<std::string>(spec.discriminator);
        const auto *nested_schema =
            discriminator_value.empty()
                ? nullptr
                : cudaq::qec::decoding::config::find_decoder_schema(
                      spec.discriminator_prefix + discriminator_value);
        if (!nested_schema)
          throw std::runtime_error("'" + spec.key +
                                   "' is present but no parameter schema is "
                                   "registered for " +
                                   spec.discriminator + " '" +
                                   discriminator_value + "'.");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      }
    }
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::decoding::config::DecoderDispatch> {
  static void
  enumeration(IO &io, cudaq::qec::decoding::config::DecoderDispatch &value) {
    io.enumCase(value, "host",
                cudaq::qec::decoding::config::DecoderDispatch::host);
    io.enumCase(value, "device_graph",
                cudaq::qec::decoding::config::DecoderDispatch::device_graph);
  }
};

// Per-seam sparse spec: H_sparse and O_sparse for one named seam boundary.
template <>
struct MappingTraits<cudaq::qec::dem_seam_spec> {
  static void mapping(IO &io, cudaq::qec::dem_seam_spec &spec) {
    io.mapOptional("H_sparse", spec.H_sparse);
    io.mapOptional("O_sparse", spec.O_sparse, std::vector<std::int64_t>{});
  }
};

// A seam or phase id is written as its name. Reading it back registers the
// name, so the id keeps it for output.
static void map_id(IO &io, const char *key, cudaq::qec::seam_id &id) {
  std::string name = io.outputting() ? id.name() : std::string{};
  io.mapRequired(key, name);
  if (!io.outputting())
    id = cudaq::qec::seam_id::from_name(name);
}

// seam_spec_entry: {name: string, spec: dem_seam_spec}
template <>
struct MappingTraits<cudaq::qec::seam_spec_entry> {
  static void mapping(IO &io, cudaq::qec::seam_spec_entry &entry) {
    map_id(io, "name", entry.id);
    io.mapRequired("spec", entry.spec);
  }
};

// One DEM phase/chunk as flat index lists.
// Shorthand: H_sparse at chunk level (memory experiments).
// Full: seam_specs sequence with per-seam H_sparse/O_sparse.
template <>
struct MappingTraits<cudaq::qec::dem_chunk_spec> {
  static void mapping(IO &io, cudaq::qec::dem_chunk_spec &spec) {
    io.mapRequired("num_faults", spec.num_faults);
    io.mapOptional("H_sparse", spec.H_sparse);
    io.mapOptional("seam_specs", spec.seam_specs);
    io.mapOptional("O_sparse", spec.O_sparse, std::vector<std::int64_t>{});
    std::vector<exact_yaml_double> rates;
    if (io.outputting()) {
      rates.reserve(spec.error_rates.size());
      for (const auto value : spec.error_rates)
        rates.push_back({value});
    }
    io.mapRequired("error_rates", rates);
    if (!io.outputting()) {
      spec.error_rates.clear();
      spec.error_rates.reserve(rates.size());
      for (const auto value : rates)
        spec.error_rates.push_back(value.value);
    }
    io.mapOptional("D_sparse", spec.D_sparse, std::vector<std::int64_t>{});
    io.mapOptional("num_measurements", spec.num_measurements);
    io.mapOptional("m2o_sparse", spec.m2o_sparse, std::vector<std::int64_t>{});
  }
};

// seam_connection: {from: name, to: name}
template <>
struct MappingTraits<cudaq::qec::seam_connection> {
  static void mapping(IO &io, cudaq::qec::seam_connection &conn) {
    map_id(io, "from", conn.from_seam);
    map_id(io, "to", conn.to_seam);
  }
};

// phase_connection: {from: name, to: name}
template <>
struct MappingTraits<cudaq::qec::phase_connection> {
  static void mapping(IO &io, cudaq::qec::phase_connection &conn) {
    map_id(io, "from", conn.from_phase);
    map_id(io, "to", conn.to_phase);
  }
};

// phase_spec_entry: {name: string, spec: dem_chunk_spec}
template <>
struct MappingTraits<cudaq::qec::phase_spec_entry> {
  static void mapping(IO &io, cudaq::qec::phase_spec_entry &entry) {
    map_id(io, "name", entry.id);
    io.mapRequired("spec", entry.spec);
  }
};

// Multi-phase DEM decomposition.
// YAML format:
//   seam: {from: next_round, to: prev_round}
//   connections:
//     - {from: init, to: bulk}
//     - {from: bulk, to: bulk}
//     - {from: bulk, to: final}
//   num_rounds: 5
//   phases:
//     - {name: init, spec: {num_faults: 3, H_sparse: [...], ...}}
//     - {name: bulk, spec: {...}}
//     - {name: final, spec: {...}}
template <>
struct MappingTraits<cudaq::qec::dem_chunks_spec> {
  static void mapping(IO &io, cudaq::qec::dem_chunks_spec &spec) {
    io.mapRequired("seam", spec.seam);
    io.mapRequired("connections", spec.connections);
    io.mapOptional("num_rounds", spec.num_rounds);
    io.mapOptional("measurements_per_round", spec.measurements_per_round);
    io.mapOptional("phases_supply_D_sparse", spec.phases_supply_D_sparse);
    io.mapOptional("observable_names", spec.observable_names,
                   std::vector<std::string>{});
    io.mapRequired("phases", spec.phases);
  }
};

// ---------------------------------------------------------------------------
// Task-graph topology: nodes, edges, ports
// ---------------------------------------------------------------------------

using task_graph_spec_t =
    cudaq::qec::decoding::config::decoder_config::task_graph_spec;

// An edge endpoint is a scalar: `node` or `node.port`.  The dot is what the
// document adds over task_graph_edge's four fields, so it lives here rather
// than in a config-only endpoint type.
static std::string join_endpoint(const std::string &node,
                                 const std::string &port,
                                 const char *default_port) {
  if (node.empty() || port == default_port)
    return node;
  return node + "." + port;
}

/// Fills `node` and `port` from a scalar, defaulting an unnamed port.  Returns
/// the error to report, empty when the scalar is well formed.
static std::string split_endpoint(const std::string &scalar, std::string &node,
                                  std::string &port, const char *default_port) {
  const auto dot = scalar.find('.');
  node = scalar.substr(0, dot);
  port = dot == std::string::npos ? default_port : scalar.substr(dot + 1);
  if (node.empty())
    return "edge endpoint is missing a node id";
  if (port.empty())
    return "edge endpoint names a node but no port after the '.'";
  return {};
}

template <>
struct MappingTraits<cudaq::qec::task_graph_edge> {
  static void mapping(IO &io, cudaq::qec::task_graph_edge &edge) {
    // Normalize the original source spelling on both input and output, so old
    // documents remain readable while every emitted document is canonical.
    if (edge.from_node == "dets")
      edge.from_node = "measurements";
    // A port equal to the default is left off.
    std::string from =
        join_endpoint(edge.from_node, edge.from_output, "output");
    std::string to = join_endpoint(edge.to_node, edge.to_input, "input");
    io.mapRequired("from", from);
    io.mapRequired("to", to);
    io.mapOptional("index_offset", edge.index_offset, std::int64_t{0});
    if (io.outputting())
      return;
    // A window waits only on the same or an earlier index, and no shot
    // holds one further back than the request limit.
    if (edge.index_offset > 0 ||
        edge.index_offset < -cudaq::qec::experimental::window_round_limit)
      throw std::runtime_error(
          "edge " + from + " -> " + to + " has index_offset " +
          std::to_string(edge.index_offset) +
          "; it must be zero or negative, and no less than -" +
          std::to_string(cudaq::qec::experimental::window_round_limit));
    if (auto err =
            split_endpoint(from, edge.from_node, edge.from_output, "output");
        !err.empty())
      io.setError(err);
    if (edge.from_node == "dets")
      edge.from_node = "measurements";
    if (auto err = split_endpoint(to, edge.to_node, edge.to_input, "input");
        !err.empty())
      io.setError(err);
  }
};

namespace {

/// The `columns` mapping of a node's `partition`, as a flat [lo, hi, ...] list
/// so a node can own more than one range.
struct partition_columns_form {
  std::vector<uint64_t> flat;
};

/// `partition:` is either a label scalar or a `{columns: [...]}` mapping.
struct partition_form {
  std::string label;
  partition_columns_form columns;
};

/// The `inner:` block: a decoder `type` with that decoder's arguments inline
/// beside it, rather than nested under a second key.
struct inner_form {
  std::string *type = nullptr;
  cudaqx::heterogeneous_map *args = nullptr;
};

} // namespace

template <>
struct MappingTraits<partition_columns_form> {
  static void mapping(IO &io, partition_columns_form &form) {
    io.mapRequired("columns", form.flat);
  }
};

template <>
struct PolymorphicTraits<partition_form> {
  static NodeKind getKind(const partition_form &form) {
    return form.label.empty() ? NodeKind::Map : NodeKind::Scalar;
  }
  static std::string &getAsScalar(partition_form &form) { return form.label; }
  static partition_columns_form &getAsMap(partition_form &form) {
    return form.columns;
  }
  static std::vector<uint64_t> &getAsSequence(partition_form &form) {
    return form.columns.flat;
  }
};

/// An empty partition is rejected before the polymorphic read, which LLVM
/// cannot do for an empty value.
inline void yamlize(IO &io, partition_form &form, bool required,
                    EmptyContext &context) {
  if (cudaq::qec::window_yaml::reading_empty(io))
    throw std::runtime_error(
        "partition needs a value: a label, or {columns: [...]}");
  yamlize<partition_form>(io, form, required, context);
}

template <>
struct CustomMappingTraits<inner_form> {
  static void inputOne(IO &io, StringRef key, inner_form &inner) {
    if (key == "type") {
      io.mapRequired("type", *inner.type);
      return;
    }
    // The decoder's own arguments sit beside its type, so the schema comes
    // from a sibling key (mapping keys are random access, so document order
    // does not matter).
    std::string type_value;
    io.mapOptional("type", type_value);
    if (type_value.empty())
      throw std::runtime_error("task_graph node 'inner' sets '" + key.str() +
                               "' but not 'type'");
    const auto *schema =
        cudaq::qec::decoding::config::find_decoder_schema(type_value);
    if (!schema)
      throw std::runtime_error(
          "task_graph node 'inner' does not support type '" + type_value +
          "': no parameter schema is registered under that name.");
    schema_binding binding{inner.args, schema};
    CustomMappingTraits<schema_binding>::inputOne(io, key, binding);
  }

  static void output(IO &io, inner_form &inner) {
    io.mapRequired("type", *inner.type);
    if (inner.args->empty())
      return;
    const auto *schema =
        cudaq::qec::decoding::config::find_decoder_schema(*inner.type);
    if (!schema) {
      CUDA_QEC_WARN("task_graph node inner decoder '{}' has arguments but no "
                    "parameter schema is registered under that name; they are "
                    "omitted from the emitted YAML.",
                    *inner.type);
      return;
    }
    schema_binding binding{inner.args, schema};
    CustomMappingTraits<schema_binding>::output(io, binding);
  }
};

// A node's fixed keys plus its kind's parameters, inline in the same mapping.
// CustomMappingTraits (rather than MappingTraits) because which keys are legal
// depends on the resolved node kind, so no key list can be written here.
using ensemble_spec_t = task_graph_spec_t::node_spec::ensemble_spec;

// The ensemble block and the node list refer to each other, so these are
// declared before the node traits and defined after them.
template <>
struct MappingTraits<ensemble_spec_t::member> {
  static void mapping(IO &io, ensemble_spec_t::member &m);
};
template <>
struct MappingTraits<ensemble_spec_t> {
  static void mapping(IO &io, ensemble_spec_t &e);
};

/// A `dem_chunks` value: the block, or the name of a data_artifacts entry.
/// A list is kept only to be rejected.
struct dem_chunks_form {
  std::string name;
  cudaq::qec::dem_chunks_spec spec;
  std::vector<std::int64_t> list;
};

template <>
struct PolymorphicTraits<dem_chunks_form> {
  static NodeKind getKind(const dem_chunks_form &form) {
    return form.name.empty() ? NodeKind::Map : NodeKind::Scalar;
  }
  static std::string &getAsScalar(dem_chunks_form &form) { return form.name; }
  static cudaq::qec::dem_chunks_spec &getAsMap(dem_chunks_form &form) {
    return form.spec;
  }
  static std::vector<std::int64_t> &getAsSequence(dem_chunks_form &form) {
    return form.list;
  }
};

/// An empty dem_chunks is rejected before the polymorphic read.
inline void yamlize(IO &io, dem_chunks_form &form, bool required,
                    EmptyContext &context) {
  if (cudaq::qec::window_yaml::reading_empty(io))
    throw std::runtime_error("dem_chunks needs a value: a mapping, or the "
                             "name of a data_artifacts entry");
  yamlize<dem_chunks_form>(io, form, required, context);
}

/// Maps a `dem_chunks` key holding the block or naming an entry. A name
/// resolves to a copy of the entry's template and is kept, so emission writes
/// the name back.
static void map_dem_chunks(IO &io,
                           std::optional<cudaq::qec::dem_chunks_spec> &spec,
                           std::string &artifact, const std::string &where) {
  std::optional<dem_chunks_form> form;
  if (io.outputting()) {
    // An edit made after parsing is kept: the name is written back only while
    // the copy still equals the entry. A name with no copy is written only
    // while the document holds its entry, which reading resolves it to.
    const auto *entry = written_entry(io, artifact);
    const bool named = entry && entry->dem_chunks;
    const bool unedited = named && spec && *entry->dem_chunks == *spec;
    if (unedited || (named && !spec))
      io.mapRequired("dem_chunks", artifact);
    else
      io.mapOptional("dem_chunks", spec);
    return;
  }
  io.mapOptional("dem_chunks", form);
  if (!form)
    return;
  if (!form->list.empty())
    throw std::runtime_error(where + " dem_chunks is a mapping, or the name "
                                     "of a data_artifacts entry");
  if (form->name.empty()) {
    spec = std::move(form->spec);
    return;
  }
  artifact = form->name;
  spec = *find_artifact(io, artifact, {"dem_chunks"}, "dem_chunks", where)
              .dem_chunks;
}

using cudaq::qec::decoding::config::both_stim_forms;

/// Resolves a stim_dem that names a data_artifacts entry: any value with no
/// whitespace, which Stim model text always has. The name is kept in @p
/// artifact, and @p text or
/// @p path becomes a copy of the entry's text or path.
static void resolve_stim_dem(IO &io, std::string &text, std::string &path,
                             std::string &artifact, const std::string &where) {
  if (text.empty() || text.find_first_of(" \t\r\n") != std::string::npos)
    return;
  artifact = std::move(text);
  text.clear();
  const auto &entry = find_artifact(io, artifact, {"stim_dem", "stim_dem_path"},
                                    "Stim DEM", where + " stim_dem");
  if (!entry.stim_dem_path.empty() && !path.empty())
    throw both_stim_forms(where);
  text = entry.stim_dem;
  if (!entry.stim_dem_path.empty())
    path = entry.stim_dem_path;
}

/// Writes a model that names an entry as the name while its copy still
/// equals the entry, as for dem_chunks, and otherwise inline. A name with no
/// copy is written only while the document holds its entry.
static void write_stim_dem(IO &io, std::string &text, std::string &path,
                           std::string &artifact) {
  const auto *entry = written_entry(io, artifact);
  const bool named =
      entry && (!entry->stim_dem.empty() || !entry->stim_dem_path.empty());
  const bool unedited = named && (!text.empty() || !path.empty()) &&
                        entry->stim_dem == text && entry->stim_dem_path == path;
  if (unedited || (named && text.empty() && path.empty())) {
    io.mapRequired("stim_dem", artifact);
    return;
  }
  if (!text.empty())
    io.mapRequired("stim_dem", text);
  if (!path.empty())
    io.mapRequired("stim_dem_path", path);
}

/// Maps a decoder entry's model keys: stim_dem_path, and stim_dem, which
/// holds the model text or names a data_artifacts entry.
static void map_stim_dem(IO &io, std::string &text, std::string &path,
                         std::string &artifact, const std::string &where) {
  if (io.outputting()) {
    write_stim_dem(io, text, path, artifact);
    return;
  }
  io.mapOptional("stim_dem_path", path, std::string{});
  io.mapOptional("stim_dem", text, std::string{});
  resolve_stim_dem(io, text, path, artifact, where);
}

/// A window layer's value: a request, or the name of a data_artifacts window
/// entry. A list is kept only to be rejected.
struct layer_form {
  std::string name;
  cudaq::qec::window_yaml::request_form request;
  std::vector<std::int64_t> list;
};

template <>
struct PolymorphicTraits<layer_form> {
  static NodeKind getKind(const layer_form &form) {
    return form.name.empty() ? NodeKind::Map : NodeKind::Scalar;
  }
  static std::string &getAsScalar(layer_form &form) { return form.name; }
  static cudaq::qec::window_yaml::request_form &getAsMap(layer_form &form) {
    return form.request;
  }
  static std::vector<std::int64_t> &getAsSequence(layer_form &form) {
    return form.list;
  }
};

/// An empty window layer is rejected before the polymorphic read.
inline void yamlize(IO &io, layer_form &form, bool required,
                    EmptyContext &context) {
  if (cudaq::qec::window_yaml::reading_empty(io))
    throw std::runtime_error("a window layer needs a value: a request, or the "
                             "name of a data_artifacts window entry");
  yamlize<layer_form>(io, form, required, context);
}

/// A `window` entry of data_artifacts: a request and the dem_chunks entry
/// it is built on.
struct window_entry_form {
  std::string context;
  cudaq::qec::window_yaml::request_form request;
};

template <>
struct CustomMappingTraits<window_entry_form> {
  static void inputOne(IO &io, StringRef key, window_entry_form &form) {
    if (key == "context") {
      io.mapRequired("context", form.context);
      return;
    }
    CustomMappingTraits<cudaq::qec::window_yaml::request_form>::inputOne(
        io, key, form.request);
  }
  static void output(IO &io, window_entry_form &form) {
    io.mapRequired("context", form.context);
    CustomMappingTraits<cudaq::qec::window_yaml::request_form>::output(
        io, form.request);
  }
};

/// One data_artifacts entry; its name is the key that holds it.
template <>
struct CustomMappingTraits<data_artifact_t> {
  static void inputOne(IO &io, StringRef key, data_artifact_t &entry) {
    const auto where = "data_artifacts entry '" + entry.name + "'";
    if (key == "dem_chunks") {
      io.mapRequired("dem_chunks", entry.dem_chunks.emplace());
    } else if (key == "stim_dem") {
      io.mapRequired("stim_dem", entry.stim_dem);
    } else if (key == "stim_dem_path") {
      io.mapRequired("stim_dem_path", entry.stim_dem_path);
    } else if (key == "edem_path") {
      io.mapRequired("edem_path", entry.edem_path);
    } else if (key == "sha256") {
      io.mapRequired("sha256", entry.sha256);
    } else if (key == "window") {
      // Request errors throw std::invalid_argument; config errors are
      // std::runtime_error, which is what from_yaml_str() callers catch.
      window_entry_form form;
      try {
        io.mapRequired("window", form);
        entry.window = cudaq::qec::window_yaml::request_from_form(form.request);
      } catch (const std::invalid_argument &e) {
        throw std::runtime_error(where + ": " + e.what());
      }
      entry.context = form.context;
    } else {
      throw std::runtime_error(where + " has unknown key '" + key.str() + "'");
    }
  }
  static void output(IO &io, data_artifact_t &entry) {
    if (entry.dem_chunks)
      io.mapRequired("dem_chunks", *entry.dem_chunks);
    if (!entry.stim_dem.empty())
      io.mapRequired("stim_dem", entry.stim_dem);
    if (!entry.stim_dem_path.empty())
      io.mapRequired("stim_dem_path", entry.stim_dem_path);
    if (entry.window) {
      window_entry_form form{
          entry.context,
          cudaq::qec::window_yaml::form_from_request(*entry.window)};
      io.mapRequired("window", form);
    }
    if (!entry.edem_path.empty())
      io.mapRequired("edem_path", entry.edem_path);
    if (!entry.sha256.empty())
      io.mapRequired("sha256", entry.sha256);
  }
};

/// The data_artifacts section: names to entries, kept sorted by name, since
/// llvm::yaml does not keep a mapping's key order.
struct artifacts_form {
  std::vector<data_artifact_t> *list = nullptr;
};

template <>
struct CustomMappingTraits<artifacts_form> {
  static void inputOne(IO &io, StringRef key, artifacts_form &form) {
    data_artifact_t entry;
    entry.name = key.str();
    io.mapRequired(entry.name.c_str(), entry);
    const auto at =
        std::lower_bound(form.list->begin(), form.list->end(), entry.name,
                         [](const data_artifact_t &e, const std::string &name) {
                           return e.name < name;
                         });
    form.list->insert(at, std::move(entry));
  }
  static void output(IO &io, artifacts_form &form) {
    for (auto &entry : *form.list)
      io.mapRequired(entry.name.c_str(), entry);
  }
};

/// An ingest node's `windows`: layer names to window requests, kept sorted by
/// name, since llvm::yaml does not keep a mapping's key order. Each layer is
/// also an output port, so it must be a port name and cannot be one ingest
/// already uses.
struct windows_form {
  std::vector<cudaq::qec::experimental::ingest_window> *windows = nullptr;
  const std::string *node = nullptr;
};

template <>
struct CustomMappingTraits<windows_form> {
  static void inputOne(IO &io, StringRef key, windows_form &form) {
    const std::string layer = key.str();
    const std::string where =
        "task_graph node '" + *form.node + "' window layer '" + layer + "'";
    layer_form value;
    const bool identifier =
        !layer.empty() && !std::isdigit(static_cast<unsigned char>(layer[0])) &&
        std::all_of(layer.begin(), layer.end(), [](char c) {
          return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
        });
    if (!identifier)
      throw std::runtime_error(where + " is not a port name");
    if (layer == "logical" || layer == "input")
      throw std::runtime_error(where +
                               " is reserved: ingest uses that port itself");
    // Request errors throw std::invalid_argument; config errors are
    // std::runtime_error, which is what from_yaml_str() callers catch.
    cudaq::qec::experimental::ingest_window window{layer, {}, {}};
    try {
      io.mapRequired(layer.c_str(), value);
      if (!value.list.empty())
        throw std::runtime_error(where + " is a window request, or the name "
                                         "of a data_artifacts window entry");
      if (value.name.empty()) {
        window.request =
            cudaq::qec::window_yaml::request_from_form(value.request);
      } else {
        window.artifact = value.name;
        window.request =
            *find_artifact(io, value.name, {"window"}, "window", where).window;
      }
    } catch (const std::invalid_argument &e) {
      throw std::runtime_error(where + ": " + e.what());
    }
    const auto at = std::lower_bound(
        form.windows->begin(), form.windows->end(), layer,
        [](const auto &w, const std::string &name) { return w.layer < name; });
    form.windows->insert(at, std::move(window));
  }
  static void output(IO &io, windows_form &form) {
    for (auto &window : *form.windows) {
      // As for dem_chunks, the name only while the copy equals the entry.
      const auto *entry = written_entry(io, window.artifact);
      if (entry && entry->window && *entry->window == window.request) {
        io.mapRequired(window.layer.c_str(), window.artifact);
        continue;
      }
      auto request = cudaq::qec::window_yaml::form_from_request(window.request);
      io.mapRequired(window.layer.c_str(), request);
    }
  }
};

template <>
struct CustomMappingTraits<task_graph_spec_t::node_spec> {
  using node_spec = task_graph_spec_t::node_spec;

  // Keys arrive in no fixed order, and the errors of a reference or a window
  // name their node, so the id is read first; a node without one is reported
  // as such.
  static std::string node_where(IO &io, node_spec &node) {
    if (node.id.empty())
      io.mapOptional("id", node.id);
    if (node.id.empty())
      throw std::runtime_error("every task_graph node needs an id");
    return "task_graph node '" + node.id + "'";
  }

  static void inputOne(IO &io, StringRef key, node_spec &node) {
    if (key == "id") {
      io.mapRequired("id", node.id);
      return;
    }
    if (key == "type") {
      io.mapRequired("type", node.type);
      return;
    }
    if (key == "dem_chunks") {
      map_dem_chunks(io, node.dem_chunks, node.dem_chunks_artifact,
                     node_where(io, node));
      return;
    }
    if (key == "windows") {
      windows_form form{&node.windows, &node.id};
      node_where(io, node);
      io.mapRequired("windows", form);
      return;
    }
    if (key == "H_sparse") {
      io.mapOptional("H_sparse", node.H_sparse);
      return;
    }
    if (key == "D_sparse") {
      io.mapOptional("D_sparse", node.D_sparse);
      return;
    }
    if (key == "O_sparse") {
      io.mapOptional("O_sparse", node.O_sparse);
      return;
    }
    if (key == "block_size") {
      io.mapOptional("block_size", node.block_size);
      return;
    }
    if (key == "syndrome_size") {
      io.mapOptional("syndrome_size", node.syndrome_size);
      return;
    }
    // A name read first may already have set the path, which a later
    // stim_dem_path would silently replace.
    if (key == "stim_dem") {
      io.mapOptional("stim_dem", node.stim_dem);
      resolve_stim_dem(io, node.stim_dem, node.stim_dem_path,
                       node.stim_dem_artifact, node_where(io, node));
      return;
    }
    if (key == "stim_dem_path") {
      if (!node.stim_dem_path.empty())
        throw both_stim_forms(node_where(io, node));
      io.mapOptional("stim_dem_path", node.stim_dem_path);
      return;
    }
    if (key == "max_in_flight") {
      io.mapOptional("max_in_flight", node.max_in_flight);
      return;
    }
    if (key == "inner") {
      auto &args = node.decoder_custom_args.map();
      inner_form inner{&node.inner_type, &args};
      io.mapRequired("inner", inner);
      const auto *schema =
          cudaq::qec::decoding::config::find_decoder_schema(node.inner_type);
      if (schema)
        cudaq::qec::decoding::config::finalize_parsed_args(
            *schema, args,
            "task_graph node inner decoder (" + node.inner_type + ")");
      return;
    }
    if (key == "partition") {
      partition_form form;
      io.mapRequired("partition", form);
      store_partition(form, node);
      return;
    }
    if (key == "decoder_custom_args") {
      // A node whose `type` is a decoder plugin can pass that decoder's
      // arguments directly, exactly as an entry-level config does.
      std::string type_value;
      io.mapOptional("type", type_value);
      const auto *schema =
          cudaq::qec::decoding::config::find_decoder_schema(type_value);
      if (!schema)
        throw std::runtime_error(
            node_where(io, node) +
            " sets decoder_custom_args but no parameter schema is registered "
            "for type '" +
            type_value +
            "'; a node whose type is a node kind passes its decoder arguments "
            "through the 'inner' block instead");
      auto &args = node.decoder_custom_args.map();
      schema_binding binding{&args, schema};
      io.mapRequired("decoder_custom_args", binding);
      cudaq::qec::decoding::config::finalize_parsed_args(
          *schema, args,
          "task_graph node decoder_custom_args (" + type_value + ")");
      return;
    }

    if (key == "ensemble") {
      // A pattern in place of a node; the builder lowers it. Everything the
      // block holds is parsed by its own traits.
      node.ensemble.emplace();
      io.mapRequired("ensemble", *node.ensemble);
      return;
    }

    // The straddle flags move a window's decode-input column range; guards
    // move which of its rounds commit. Only the latter is expressible here, so
    // say which key replaces which rather than reporting an unknown key.
    if (key.starts_with("straddle_"))
      throw std::runtime_error(
          "task_graph node sets '" + key.str() +
          "', which a task-graph window does not support; use start_guard and "
          "end_guard to exclude a window's boundary rounds from the commit");

    // Anything else belongs to the node kind, which the `type` key names.
    std::string type_value;
    io.mapOptional("type", type_value);
    const auto *schema = cudaq::qec::task_graph_node_params_schema(type_value);
    if (!schema)
      throw std::runtime_error("task_graph node sets '" + key.str() +
                               "' but no parameter schema is registered for "
                               "node type '" +
                               type_value + "'");
    schema_binding binding{&node.node_params.map(), schema};
    CustomMappingTraits<schema_binding>::inputOne(io, key, binding);
  }

  static void output(IO &io, node_spec &node) {
    io.mapRequired("id", node.id);
    if (node.ensemble.has_value()) {
      io.mapRequired("ensemble", *node.ensemble);
      return;
    }
    // An alias is an accepted spelling on input only, so emission writes the
    // canonical name and a document round-trips to one form.
    const auto *kind = cudaq::qec::find_task_graph_node_kind(node.type);
    std::string type = kind ? kind->name : node.type;
    io.mapRequired("type", type);
    map_dem_chunks(io, node.dem_chunks, node.dem_chunks_artifact,
                   "task_graph node '" + node.id + "'");
    if (!node.windows.empty()) {
      windows_form form{&node.windows, &node.id};
      io.mapRequired("windows", form);
    }
    if (!node.H_sparse.empty())
      io.mapRequired("H_sparse", node.H_sparse);
    if (!node.D_sparse.empty())
      io.mapRequired("D_sparse", node.D_sparse);
    if (!node.O_sparse.empty())
      io.mapRequired("O_sparse", node.O_sparse);
    if (node.block_size != 0)
      io.mapRequired("block_size", node.block_size);
    if (node.syndrome_size != 0)
      io.mapRequired("syndrome_size", node.syndrome_size);
    write_stim_dem(io, node.stim_dem, node.stim_dem_path,
                   node.stim_dem_artifact);
    if (node.max_in_flight != 1)
      io.mapRequired("max_in_flight", node.max_in_flight);
    if (!node.node_params.empty()) {
      const auto *schema = cudaq::qec::task_graph_node_params_schema(node.type);
      if (!schema)
        throw std::runtime_error(
            "task_graph node '" + node.id +
            "' has node parameters but no schema is registered for type '" +
            node.type + "'");
      schema_binding binding{&node.node_params.map(), schema};
      CustomMappingTraits<schema_binding>::output(io, binding);
    }
    if (!node.inner_type.empty()) {
      inner_form inner{&node.inner_type, &node.decoder_custom_args.map()};
      io.mapRequired("inner", inner);
    } else if (!node.decoder_custom_args.empty()) {
      const auto *schema =
          cudaq::qec::decoding::config::find_decoder_schema(node.type);
      if (!schema)
        throw std::runtime_error(
            "task_graph node '" + node.id +
            "' has decoder_custom_args but no parameter schema is registered "
            "for type '" +
            node.type + "'");
      auto &args = node.decoder_custom_args.map();
      schema_binding binding{&args, schema};
      io.mapRequired("decoder_custom_args", binding);
    }
    if (!node.partition_label.empty() || !node.partition_columns.empty()) {
      partition_form form = load_partition(node);
      io.mapRequired("partition", form);
    }
  }

private:
  static void store_partition(const partition_form &form, node_spec &node) {
    node.partition_label = form.label;
    node.partition_columns.clear();
    const auto &flat = form.columns.flat;
    if (!form.label.empty())
      return;
    if (flat.empty() || flat.size() % 2 != 0)
      throw std::runtime_error(
          "task_graph node '" + node.id +
          "': partition columns are [lo, hi] pairs, so the list must have an "
          "even, non-zero length; got " +
          std::to_string(flat.size()) + " entries");
    for (std::size_t i = 0; i < flat.size(); i += 2) {
      if (flat[i] > flat[i + 1])
        throw std::runtime_error(
            "task_graph node '" + node.id + "': partition column range [" +
            std::to_string(flat[i]) + ", " + std::to_string(flat[i + 1]) +
            "] is not ordered");
      if (i > 0 && flat[i] <= flat[i - 1])
        throw std::runtime_error(
            "task_graph node '" + node.id +
            "': partition column ranges must be ordered and disjoint");
      node.partition_columns.push_back({flat[i], flat[i + 1]});
    }
  }

  static partition_form load_partition(const node_spec &node) {
    partition_form form;
    form.label = node.partition_label;
    for (const auto &range : node.partition_columns) {
      form.columns.flat.push_back(range.lo);
      form.columns.flat.push_back(range.hi);
    }
    return form;
  }
};

void MappingTraits<ensemble_spec_t::member>::mapping(
    IO &io, ensemble_spec_t::member &m) {
  io.mapRequired("nodes", m.nodes);
  io.mapOptional("edges", m.edges);
}

void MappingTraits<ensemble_spec_t>::mapping(IO &io, ensemble_spec_t &e) {
  io.mapRequired("members", e.members);
  io.mapRequired("join_strategy", e.join_strategy);
  // join_params belongs to the strategy, whose schema is registered under its
  // name (keys are random access, so the strategy is known either way).
  const auto *schema = cudaq::qec::decoding::config::find_decoder_schema(
      cudaq::qec::selection_strategy_schema_name(e.join_strategy));
  if (io.outputting()) {
    if (e.join_params.empty())
      return;
    if (!schema) {
      CUDA_QEC_WARN("task_graph ensemble join_strategy '{}' has parameters "
                    "but no schema is registered under that name; they are "
                    "omitted from the emitted YAML.",
                    e.join_strategy);
      return;
    }
    schema_binding binding{&e.join_params.map(), schema};
    io.mapRequired("join_params", binding);
    return;
  }
  if (!schema)
    throw std::runtime_error("task_graph ensemble: join_strategy '" +
                             e.join_strategy +
                             "' is not a registered selection strategy");
  schema_binding binding{&e.join_params.map(), schema};
  io.mapOptional("join_params", binding);
  cudaq::qec::decoding::config::finalize_parsed_args(
      *schema, e.join_params.map(),
      "task_graph ensemble join_params (" + e.join_strategy + ")");
}

template <>
struct MappingTraits<task_graph_spec_t> {
  static void mapping(IO &io, task_graph_spec_t &tg) {
    // Both forms are optional here; decoder_config's mapping rejects a spec
    // that names neither or both.
    io.mapOptional("type", tg.type, std::string{});
    io.mapOptional("inner_type", tg.inner_type, std::string{});
    io.mapOptional("async_streaming", tg.options.async_streaming, false);
    io.mapOptional("async_queue_capacity", tg.options.async_queue_capacity,
                   std::size_t{8});
    io.mapOptional("async_coordinator_cpu", tg.options.async_coordinator_cpu,
                   -1);
    io.mapOptional("async_spin_pause_iters", tg.options.async_spin_pause_iters,
                   std::size_t{0});
    io.mapOptional("async_num_workers", tg.options.async_num_workers);
    io.mapOptional("async_worker_cpus", tg.options.async_worker_cpus,
                   std::vector<int>{});
    io.mapOptional("nodes", tg.nodes);
    io.mapOptional("edges", tg.edges);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::decoder_config> {
  static void mapping(IO &io,
                      cudaq::qec::decoding::config::decoder_config &config) {
    // io.keys() reports every key the traits below ask for as well as the ones
    // the document actually carries, so it can only tell absent from present
    // before any mapping call has registered a name. The flat-form check at
    // the bottom needs that distinction, because a field that is missing and
    // one that is present but empty parse to the same value.
    std::vector<std::string> document_keys;
    if (!io.outputting())
      for (const auto key : io.keys())
        document_keys.emplace_back(key.str());
    const auto in_document = [&document_keys](const char *name) {
      return std::find(document_keys.begin(), document_keys.end(), name) !=
             document_keys.end();
    };

    io.mapRequired("id", config.id);
    // `type` is required for plain decoder configs but absent for task_graph
    // configs; make it optional here and enforce the constraint below.
    io.mapOptional("type", config.type, std::string{});
    io.mapOptional("task_graph", config.task_graph);
    const std::string &schema_type = config.effective_decoder_type();
    const auto *schema =
        cudaq::qec::decoding::config::find_decoder_schema(schema_type);
    io.mapOptional("dispatch", config.dispatch,
                   cudaq::qec::decoding::config::DecoderDispatch::host);
    io.mapOptional("cuda_device_id", config.cuda_device_id);
    // The DEM arrives in one of three forms (see decoder_config). Everything
    // below the mapping calls enforces that exactly one of them is described,
    // because the other two forms derive the flat fields and a config that
    // spelled out more than one could disagree with itself.
    // A flat configuration names all five on the way out as well as the way
    // in. Emitting only the ones that differ from their defaults would drop a
    // legitimately empty O_sparse (a DEM with no observables) and leave behind
    // a document that no longer parses. A chunk-form or DEM-sourced
    // configuration takes the optional path so its derived fields stay absent,
    // which is what makes the emitted document re-parse in the form it was
    // written in.
    const bool emitting_flat_form =
        io.outputting() &&
        !((config.dem_chunks.has_value() || !config.stim_dem_path.empty() ||
           !config.stim_dem.empty()) &&
          config.H_sparse.empty());
    if (emitting_flat_form) {
      io.mapRequired("block_size", config.block_size);
      io.mapRequired("syndrome_size", config.syndrome_size);
      io.mapRequired("H_sparse", config.H_sparse);
      io.mapRequired("O_sparse", config.O_sparse);
      io.mapRequired("D_sparse", config.D_sparse);
    } else {
      io.mapOptional("block_size", config.block_size, std::uint64_t{0});
      io.mapOptional("syndrome_size", config.syndrome_size, std::uint64_t{0});
      io.mapOptional("H_sparse", config.H_sparse, std::vector<std::int64_t>{});
      io.mapOptional("O_sparse", config.O_sparse, std::vector<std::int64_t>{});
      io.mapOptional("D_sparse", config.D_sparse, std::vector<std::int64_t>{});
    }
    // Continue accepting the PR765 top-level spelling, but registered decoder
    // configurations are emitted with rates under decoder_custom_args so the
    // established researcher-facing wire YAML does not change.
    if (!io.outputting())
      io.mapOptional("error_rate_vec", config.error_rate_vec);
    else if (!schema) {
      std::vector<exact_yaml_double> rates;
      rates.reserve(config.error_rate_vec.size());
      for (const auto rate : config.error_rate_vec)
        rates.push_back({rate});
      io.mapOptional("error_rate_vec", rates);
    }
    map_dem_chunks(io, config.dem_chunks, config.dem_chunks_artifact,
                   "decoder " + std::to_string(config.id));
    map_stim_dem(io, config.stim_dem, config.stim_dem_path,
                 config.stim_dem_artifact,
                 "decoder " + std::to_string(config.id));

    // Convert decoder_custom_args through the schema registered for this
    // decoder type. error_rate_vec is a reserved compatibility alias: accept
    // it at any schema-recognized nesting level, promote it to model state,
    // and never leave it in the map a plugin receives.
    if (io.outputting()) {
      auto args_map = config.decoder_custom_args_to_heterogeneous_map();
      const auto rates = config.effective_error_rate_vec();
      if (!rates.empty())
        args_map.insert("error_rate_vec", rates);
      if (!args_map.empty()) {
        if (!schema) {
          // Match the historical emission behavior for unknown types.
          CUDA_QEC_WARN(
              "decoder_custom_args set for decoder type '{}' but no parameter "
              "schema is registered under that name; the args are omitted "
              "from the emitted YAML.",
              schema_type);
        } else {
          cudaqx::heterogeneous_map emitted_args = std::move(args_map);
          schema_binding binding{&emitted_args, schema};
          io.mapRequired("decoder_custom_args", binding);
        }
      }
    } else if (schema) {
      bool args_present = false;
      for (const auto key : io.keys()) {
        if (key == "decoder_custom_args") {
          args_present = true;
          break;
        }
      }
      cudaqx::heterogeneous_map args_map;
      schema_binding binding{&args_map, schema};
      io.mapOptional("decoder_custom_args", binding);
      if (args_present) {
        cudaq::qec::decoding::config::decoder_config args_config;
        args_config.type = schema_type;
        args_config.decoder_custom_args = args_map;
        auto normalized =
            cudaq::qec::decoding::config::normalize_decoder_args(args_config);
        if (in_document("error_rate_vec") && normalized.legacy_rates)
          throw std::runtime_error(
              "error_rate_vec is supplied both at the decoder configuration "
              "level and at '" +
              normalized.legacy_rates->path +
              "'. Supply model error rates in exactly one location.");
        if (normalized.legacy_rates)
          config.error_rate_vec = std::move(normalized.legacy_rates->values);
        args_map = std::move(normalized.plugin_args);
        cudaq::qec::decoding::config::finalize_parsed_args(
            *schema, args_map, "decoder_custom_args (" + schema_type + ")");
      }
      config.decoder_custom_args = args_map;
    }

    // LLVM's YAML parser records a diagnostic and keeps going, so a malformed
    // document arrives here half-populated. Validate state only if error-free.
    if (io.error())
      return;

    // Decoder construction refuses a negative cuda_device_id (decoder.cpp,
    // read_cuda_device_id) and the exported schema bounds it at 0; refuse it
    // here as well, so a document is rejected where its other mistakes are.
    if (!io.outputting() && config.cuda_device_id.has_value() &&
        *config.cuda_device_id < 0)
      throw std::runtime_error("decoder " + std::to_string(config.id) +
                               ": cuda_device_id must be >= 0 (got " +
                               std::to_string(*config.cuda_device_id) + ")");

    // Task-graph form: task_graph: is set, type: must be absent.
    if (!io.outputting() && config.task_graph.has_value()) {
      if (!config.type.empty())
        throw std::runtime_error(
            "decoder " + std::to_string(config.id) +
            ": task_graph and type are mutually exclusive");
      cudaq::qec::decoding::config::validate_task_graph(config);
      // An entry-level model is required only while some node would look for
      // one. The single-node form always does.
      const auto describes_itself =
          cudaq::qec::decoding::config::task_graph_node_describes_itself;
      bool any_falls_back = config.task_graph->nodes.empty();
      for (const auto &node : config.task_graph->nodes)
        any_falls_back = any_falls_back || !describes_itself(node);
      if (any_falls_back && !config.dem_chunks.has_value() &&
          config.H_sparse.empty())
        throw std::runtime_error(
            "decoder " + std::to_string(config.id) +
            ": task_graph requires either dem_chunks or H_sparse to supply "
            "the DEM");
      // Chunk form carries its own DEM; no flat-matrix fields to validate.
      // Flat task_graph form (H_sparse set) falls through to flat-DEM
      // validation below so block_size, syndrome_size, O_sparse, and D_sparse
      // are required and checked exactly as for a standard flat decoder.
      if (config.H_sparse.empty())
        return;
    }

    // Plain decoder forms require type: (task_graph flat form is exempt).
    if (!io.outputting() && config.type.empty() &&
        !config.task_graph.has_value())
      throw std::runtime_error("decoder " + std::to_string(config.id) +
                               ": type is required when task_graph is absent");

    // Chunk form when dem_chunks is set and H_sparse is empty. A nonempty
    // H_sparse makes the config flat — the matrix wins — which is also what
    // expand_dem_chunks() leaves behind after it runs.
    const bool chunk_form =
        config.dem_chunks.has_value() && config.H_sparse.empty();

    // DEM form when a Stim model is given, inline or by file. The decoder is
    // built from that text and derives its own dimensions and observable
    // mapping, so the checks below that compare against syndrome_size do not
    // apply to it; the D row count is checked against the constructed decoder
    // instead.
    const bool from_dem =
        !config.stim_dem_path.empty() || !config.stim_dem.empty();
    const char *dem_key =
        cudaq::qec::decoding::config::stim_dem_key(config.stim_dem_path);
    if (!config.stim_dem_path.empty() && !config.stim_dem.empty())
      throw both_stim_forms("decoder " + std::to_string(config.id));

    // Both forms at once is legal but lopsided: H_sparse wins and the chunk
    // spec never runs. expand_dem_chunks() produces exactly this state, so it
    // cannot be an error, but a hand-written document that reaches it has lost
    // its dem_chunks without any other signal.
    if (!io.outputting() && config.dem_chunks.has_value() &&
        !config.H_sparse.empty())
      CUDA_QEC_WARN(
          "decoder {} sets both H_sparse and dem_chunks. The flat H_sparse "
          "matrix is what decoders are built from, and dem_chunks is ignored, "
          "including its num_rounds. This is expected for a config already "
          "expanded by expand_dem_chunks(); otherwise drop H_sparse to decode "
          "from dem_chunks.",
          config.id);

    // dem_chunks and stim_dem_path each describe the whole DEM on their own,
    // and unlike the flat form there is no precedence between them to fall
    // back on, so a document naming both has to be rejected outright.
    if (config.dem_chunks.has_value() && from_dem)
      throw std::runtime_error(std::string("dem_chunks and ") + dem_key +
                               " must not both be set for decoder " +
                               std::to_string(config.id) +
                               "; each describes the whole DEM on its own.");

    // Whichever form is in play derives these from the model it names;
    // accepting them here would let a config disagree with itself.
    const auto reject_derived = [&config](const char *key, bool present,
                                          const char *source) {
      if (present)
        throw std::runtime_error(
            std::string(key) + " must not be set for decoder " +
            std::to_string(config.id) + " because it is derived from " +
            source +
            ". Remove it, or describe the whole experiment with "
            "H_sparse instead of " +
            source + ".");
    };

    if (chunk_form) {
      if (config.dem_chunks->num_rounds.has_value() &&
          *config.dem_chunks->num_rounds < 2)
        throw std::runtime_error(
            "dem_chunks.num_rounds must be at least 2 for decoder " +
            std::to_string(config.id) + "; got " +
            std::to_string(*config.dem_chunks->num_rounds) + ".");
      reject_derived("block_size", config.block_size != 0, "dem_chunks");
      reject_derived("syndrome_size", config.syndrome_size != 0, "dem_chunks");
      reject_derived("O_sparse", !config.O_sparse.empty(), "dem_chunks");
      reject_derived("D_sparse", !config.D_sparse.empty(), "dem_chunks");
      reject_derived("error_rate_vec", !config.error_rate_vec.empty(),
                     "dem_chunks");
    } else if (from_dem) {
      reject_derived("H_sparse", !config.H_sparse.empty(), dem_key);
      reject_derived("O_sparse", !config.O_sparse.empty(), dem_key);
      reject_derived("error_rate_vec", !config.error_rate_vec.empty(), dem_key);
      // D_sparse is the one model field the DEM text does not carry: it maps
      // detectors onto the syndrome bits this decoder is sent. Its row count is
      // checked against the constructed decoder's detector count, which is only
      // known once the model has been read.
      if (!io.outputting() && config.D_sparse.empty())
        throw std::runtime_error(
            "D_sparse is required for decoder " + std::to_string(config.id) +
            " alongside " + dem_key +
            ", which describes the DEM but not the detectors this decoder is "
            "sent.");
    } else {
      // A flat document spells out its whole DEM, so every one of these fields
      // has to be there. They are mapOptional only because the other two forms
      // derive them; without this check a document that omits one parses into
      // a zero-sized DEM and fails much later, at decoder construction.
      if (!io.outputting()) {
        std::string missing;
        for (const char *name : {"block_size", "syndrome_size", "H_sparse",
                                 "O_sparse", "D_sparse"}) {
          if (!in_document(name)) {
            if (!missing.empty())
              missing += ", ";
            missing += name;
          }
        } // end - for(name)
        if (!missing.empty())
          throw std::runtime_error(
              "decoder " + std::to_string(config.id) +
              " is missing required field(s): " + missing +
              ". A flat DEM names block_size, syndrome_size, H_sparse, "
              "O_sparse and D_sparse together; use dem_chunks "
              "to describe the same DEM one round at a time, or stim_dem or "
              "stim_dem_path to give the model text itself, inline or by "
              "file, instead.");
      } // end - if(!io.outputting())

      if (config.H_sparse.empty() && config.syndrome_size > 0)
        throw std::runtime_error(
            "H_sparse is required for decoder " + std::to_string(config.id) +
            " unless dem_chunks is present, which describes the same DEM one "
            "round at a time instead.");
      if (config.block_size == 0 && !config.H_sparse.empty())
        throw std::runtime_error("block_size is required for decoder " +
                                 std::to_string(config.id));

      // Validate that the number of rows in the H_sparse vector is equal to
      // syndrome_size.
      auto num_H_rows =
          std::count(config.H_sparse.begin(), config.H_sparse.end(), -1);
      if (num_H_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in H_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_H_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
    }

    // Validate that no values in the H_sparse vector are out of range.
    for (auto value : config.H_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in H_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that no values in the O_sparse vector are out of range.
    for (auto value : config.O_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in O_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that if the D_sparse is provided, it is a valid D matrix. That
    // means that the number of rows in the D_sparse matrix should be equal to
    // the number of rows in the H_sparse matrix, and no row should be empty.
    if (!config.D_sparse.empty()) {
      auto num_D_rows =
          std::count(config.D_sparse.begin(), config.D_sparse.end(), -1);
      if (!from_dem && num_D_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in D_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_D_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
      // No row should be empty, including the first row.
      if (config.D_sparse.front() == -1)
        throw std::runtime_error("D_sparse row is empty for decoder " +
                                 std::to_string(config.id));
      for (std::size_t i = 0; i < config.D_sparse.size() - 1; ++i) {
        if (config.D_sparse.at(i) == -1 && config.D_sparse.at(i + 1) == -1) {
          throw std::runtime_error("D_sparse row is empty for decoder " +
                                   std::to_string(config.id));
        }
      }
    }
  }
};

// transport section mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::transport_shape_override> {
  static void
  mapping(IO &io,
          cudaq::qec::decoding::config::transport_shape_override &override_) {
    io.mapOptional("provider", override_.provider, std::string());
    io.mapOptional("args", override_.args);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::transport_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::transport_config &transport) {
    io.mapOptional("provider", transport.provider, std::string());
    io.mapOptional("args", transport.args);
    io.mapOptional("device_graph", transport.device_graph,
                   cudaq::qec::decoding::config::transport_shape_override());
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::syndrome_mapping_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::syndrome_mapping_config &row) {
    io.mapRequired("id", row.id);
    io.mapOptional("type", row.type, std::string("bulk"));
    // Defaults to the single pass-through port, the only shape the runtime
    // honors -- so a row is just an id and a shape until scatter lands.
    io.mapOptional("vps", row.vps, std::vector<std::vector<uint64_t>>{{}});
  }
};

// multi_decoder_config mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::multi_decoder_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::multi_decoder_config &config) {
    // Read before the decoders, whose references resolve against it; when
    // writing, references are compared against it.
    artifacts_form artifacts{&config.data_artifacts};
    if (io.outputting()) {
      if (!config.data_artifacts.empty())
        io.mapRequired("data_artifacts", artifacts);
      if (auto *context = static_cast<config_parse_context *>(io.getContext()))
        context->artifacts = &config.data_artifacts;
    } else {
      io.mapOptional("data_artifacts", artifacts);
      validate_data_artifacts(config.data_artifacts);
      if (auto *context = static_cast<config_parse_context *>(io.getContext()))
        context->artifacts = &config.data_artifacts;
    }
    io.mapRequired("decoders", config.decoders);
    io.mapOptional("transport", config.transport,
                   cudaq::qec::decoding::config::transport_config());
    io.mapOptional("syndrome_mapping", config.syndrome_mapping);
  }
};

} // namespace llvm::yaml

// A row must name a shape the server understands and exactly one pass-through
// virtual port.  Per-port scatter is parsed but not honored at runtime yet, so
// a populated index list is refused rather than silently ignored: a config
// that reads as if it routes bits per port while doing nothing of the kind is
// worse than one that fails to load.  Refusing every non-empty list also makes
// the "which port is pass-through" question moot -- all of them are.
// Duplicate row keys would otherwise resolve arbitrarily when the table is
// built, and a bad shape would silently read as BULK.
void cudaq::qec::decoding::config::multi_decoder_config::
    validate_syndrome_mapping() const {
  std::set<uint64_t> seen;
  for (const auto &row : syndrome_mapping) {
    row.validate();
    if (!seen.insert(row.id).second)
      throw std::runtime_error("Duplicate syndrome_mapping id " +
                               std::to_string(row.id) + ".");
    // Bounded here, not at the DEVICE-mode install, so an out-of-range
    // document is refused under HOST too.
    if (row.id >= cudaq::qec::decoding::rpc::kMaxSyndromeMappingRows)
      throw std::runtime_error(
          "syndrome_mapping id " + std::to_string(row.id) +
          " is out of range; ids must be below " +
          std::to_string(cudaq::qec::decoding::rpc::kMaxSyndromeMappingRows) +
          ".");
  }
} // end - multi_decoder_config::validate_syndrome_mapping()

void cudaq::qec::decoding::config::syndrome_mapping_config::validate() const {
  if (type != "bulk" && type != "final")
    throw std::runtime_error("syndrome_mapping row " + std::to_string(id) +
                             " has type \"" + type +
                             "\", but the only round shapes are \"bulk\" and "
                             "\"final\".");

  // Reachable from the object API, not from YAML: mapOptional normalizes an
  // explicitly empty sequence to the pass-through default.
  if (vps.empty())
    throw std::runtime_error(
        "syndrome_mapping row " + std::to_string(id) +
        " lists no virtual ports, so no enqueue could ever complete a round "
        "under it. Give it one entry per virtual port; an empty index list is "
        "the pass-through identity.");

  if (vps.size() != 1)
    throw std::runtime_error(
        "syndrome_mapping row " + std::to_string(id) + " lists " +
        std::to_string(vps.size()) +
        " virtual ports. Only a single pass-through port is supported today.");

  if (!vps.front().empty())
    throw std::runtime_error(
        "syndrome_mapping row " + std::to_string(id) +
        " gives per-port detector indices, but the server does not scatter by "
        "them yet and would ignore them. Use an empty list for pass-through.");
} // end - syndrome_mapping_config::validate()

namespace {

// Run dem_chunks_spec's cross-phase checks after a successful parse, and
// present failures as std::runtime_error so every configuration error out of
// from_yaml_str() has the one type callers already catch.
void validate_parsed_dem_chunks(
    const cudaq::qec::decoding::config::decoder_config &config) {
  if (!config.dem_chunks)
    return;
  try {
    config.dem_chunks->validate();
  } catch (const std::invalid_argument &e) {
    throw std::runtime_error("Invalid dem_chunks for decoder " +
                             std::to_string(config.id) + ": " + e.what());
  }
}

} // namespace

// Static method to convert a YAML string to a multi_decoder_config.
cudaq::qec::decoding::config::multi_decoder_config
cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
    const std::string_view yaml_str) {
  multi_decoder_config config;
  config_parse_context context;
  llvm::yaml::Input yaml_in(yaml_str, &context);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  for (const auto &decoder : config.decoders)
    validate_parsed_dem_chunks(decoder);
  config.validate_syndrome_mapping();
  return config;
}

std::string cudaq::qec::decoding::config::multi_decoder_config::to_yaml_str(
    int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  config_parse_context context;
  llvm::yaml::Output yaml_out(yaml_stream, &context, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

cudaq::qec::decoding::config::decoder_config
cudaq::qec::decoding::config::decoder_config::from_yaml_str(
    const std::string &yaml_str) {
  decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  validate_parsed_dem_chunks(config);
  return config;
}

std::string
cudaq::qec::decoding::config::decoder_config::to_yaml_str(int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

namespace cudaq::qec::decoding::config {

// ---------------------------------------------------------------------------
// JSON Schema export
//
// Translates the registered decoder parameter schemas plus the fixed
// decoder_config envelope (the fields MappingTraits<decoder_config> maps
// above) into a JSON Schema draft 2020-12 document, so standard tooling can
// validate user-provided configuration YAML offline. The document is a
// snapshot of what this installation can parse: it enumerates the schemas
// registered at call time, exactly as the runtime parser resolves them.
// ---------------------------------------------------------------------------

namespace {

// JSON-pointer token escaping for schema names used inside $ref paths.
std::string json_pointer_escape(const std::string &name) {
  std::string out;
  for (char c : name) {
    if (c == '~')
      out += "~0";
    else if (c == '/')
      out += "~1";
    else
      out += c;
  }
  return out;
}

std::string params_ref(const std::string &name) {
  return "#/$defs/decoder_params/" + json_pointer_escape(name);
}

llvm::json::Object json_schema_for_param(const param_spec &spec) {
  using k = param_kind;
  switch (spec.kind) {
  case k::boolean:
    return llvm::json::Object{{"type", "boolean"}};
  case k::int32:
    // The stored type is a 32-bit int; a document that overflows it is
    // rejected when parsed, so the schema says so too and offline
    // validation agrees with the parser.
    return llvm::json::Object{
        {"type", "integer"},
        {"minimum", static_cast<int64_t>(std::numeric_limits<int32_t>::min())},
        {"maximum", static_cast<int64_t>(std::numeric_limits<int32_t>::max())}};
  case k::uint64:
    // Stored as std::size_t. Only the lower bound is representable: JSON
    // Schema numbers are IEEE doubles to most validators, and llvm::json
    // holds int64, so 2^64 - 1 cannot be written exactly.
    return llvm::json::Object{{"type", "integer"}, {"minimum", 0}};
  case k::f64:
    return llvm::json::Object{{"type", "number"}};
  case k::string:
    return llvm::json::Object{{"type", "string"}};
  case k::f64_vec:
    return llvm::json::Object{
        {"type", "array"}, {"items", llvm::json::Object{{"type", "number"}}}};
  case k::f64_matrix:
    return llvm::json::Object{
        {"type", "array"},
        {"items", llvm::json::Object{
                      {"type", "array"},
                      {"items", llvm::json::Object{{"type", "number"}}}}}};
  case k::subschema:
    return llvm::json::Object{{"$ref", params_ref(spec.subschema)}};
  case k::discriminated:
    // The concrete shape is selected by the discriminator value; the
    // dispatch clauses emitted below refine this.
    return llvm::json::Object{{"type", "object"}};
  }
  return llvm::json::Object{};
}

llvm::json::Array registered_name_array(const std::vector<std::string> &names) {
  llvm::json::Array arr;
  for (const auto &name : names)
    arr.push_back(name);
  return arr;
}

llvm::json::Object
decoder_params_json_schema(const decoder_schema &schema,
                           const std::vector<std::string> &all_names) {
  // Framework-owned compatibility alias. Runtime normalization removes it
  // before the map reaches the decoder plugin.
  llvm::json::Object properties{
      {"error_rate_vec",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "number"}}}}}};
  llvm::json::Array required;
  llvm::json::Array all_of;
  for (const auto &spec : schema.params) {
    properties[spec.key] = json_schema_for_param(spec);
    if (spec.required)
      required.push_back(spec.key);
    if (spec.kind == param_kind::discriminated) {
      // When the section is present, its discriminator must be present and
      // name a registered schema (mirrors the parser's checks).
      all_of.push_back(llvm::json::Object{
          {"if", llvm::json::Object{{"required", llvm::json::Array{spec.key}}}},
          {"then",
           llvm::json::Object{
               {"required", llvm::json::Array{spec.discriminator}},
               {"properties",
                llvm::json::Object{
                    {spec.discriminator,
                     llvm::json::Object{
                         {"enum", registered_name_array(all_names)}}}}}}}});
      // Each candidate discriminator value selects that schema for the
      // section.
      for (const auto &name : all_names)
        all_of.push_back(llvm::json::Object{
            {"if",
             llvm::json::Object{
                 {"properties",
                  llvm::json::Object{{spec.discriminator,
                                      llvm::json::Object{{"const", name}}}}},
                 {"required",
                  llvm::json::Array{spec.discriminator, spec.key}}}},
            {"then", llvm::json::Object{
                         {"properties",
                          llvm::json::Object{
                              {spec.key, llvm::json::Object{
                                             {"$ref", params_ref(name)}}}}}}}});
    }
  }
  llvm::json::Object out{{"type", "object"},
                         {"additionalProperties", false},
                         {"properties", std::move(properties)}};
  if (!required.empty())
    out["required"] = std::move(required);
  if (!all_of.empty())
    out["allOf"] = std::move(all_of);
  return out;
}

/// The `task_graph` sub-schema, generated from the node-kind registry: a kind's
/// parameters appear as node keys, and each kind's clause forbids the keys that
/// belong only to other kinds. A newly registered kind shows up here without an
/// edit to this function.
/// The `ensemble` pattern entry: members of nodes (the node schema itself)
/// and edges, a registered join_strategy and, per strategy, its join_params.
llvm::json::Object ensemble_json_schema() {
  llvm::json::Array names, per_strategy;
  for (const auto &name : cudaq::qec::registered_selection_strategies()) {
    names.push_back(name);
    const auto *schema =
        find_decoder_schema(cudaq::qec::selection_strategy_schema_name(name));
    if (!schema)
      continue;
    llvm::json::Object properties;
    llvm::json::Array required;
    for (const auto &param : schema->params) {
      properties[param.key] = json_schema_for_param(param);
      if (param.required)
        required.push_back(param.key);
    }
    llvm::json::Object params{{"type", "object"},
                              {"properties", std::move(properties)},
                              {"additionalProperties", false}};
    if (!required.empty())
      params["required"] = std::move(required);
    per_strategy.push_back(llvm::json::Object{
        {"if",
         llvm::json::Object{
             {"properties",
              llvm::json::Object{
                  {"join_strategy", llvm::json::Object{{"const", name}}}}},
             {"required", llvm::json::Array{"join_strategy"}}}},
        {"then",
         llvm::json::Object{
             {"properties",
              llvm::json::Object{{"join_params", std::move(params)}}}}}});
  }
  llvm::json::Object member{
      {"type", "object"},
      {"properties",
       llvm::json::Object{
           {"nodes",
            llvm::json::Object{
                {"type", "array"},
                {"minItems", 1},
                {"items",
                 llvm::json::Object{{"$ref", "#/$defs/task_graph_node"}}}}},
           {"edges",
            llvm::json::Object{
                {"type", "array"},
                {"items",
                 llvm::json::Object{{"$ref", "#/$defs/task_graph_edge"}}}}}}},
      {"required", llvm::json::Array{"nodes", "edges"}},
      {"additionalProperties", false}};
  llvm::json::Object ensemble{
      {"type", "object"},
      {"properties",
       llvm::json::Object{
           {"members", llvm::json::Object{{"type", "array"},
                                          {"minItems", 1},
                                          {"items", std::move(member)}}},
           {"join_strategy",
            llvm::json::Object{{"type", "string"}, {"enum", std::move(names)}}},
           {"join_params", llvm::json::Object{{"type", "object"}}}}},
      {"required", llvm::json::Array{"members", "join_strategy"}},
      {"additionalProperties", false}};
  if (!per_strategy.empty())
    ensemble["allOf"] = std::move(per_strategy);
  return ensemble;
}

llvm::json::Object task_graph_json_schema() {
  auto kinds = cudaq::qec::registered_task_graph_node_kinds();
  // A kind only a pattern's lowering produces is not a `type` a user writes.
  kinds.erase(std::remove_if(kinds.begin(), kinds.end(),
                             [](const std::string &kind) {
                               const auto *spec =
                                   cudaq::qec::find_task_graph_node_kind(kind);
                               return spec && spec->lowered_only;
                             }),
              kinds.end());

  // Union of every kind's parameters: node parameters sit inline among the
  // node's own keys, so `additionalProperties: false` has to know all of them.
  llvm::json::Object node_properties{
      {"id", llvm::json::Object{{"type", "string"}}},
      {"type", llvm::json::Object{{"type", "string"}}},
      {"inner", llvm::json::Object{{"type", "object"},
                                   {"required", llvm::json::Array{"type"}}}},
      {"decoder_custom_args", llvm::json::Object{{"type", "object"}}},
      {"max_in_flight",
       llvm::json::Object{{"type", "integer"}, {"minimum", 1}, {"default", 1}}},
      {"dem_chunks",
       llvm::json::Object{{"$ref", "#/$defs/dem_chunks_or_name"}}},
      // An ingest node's layers; each name is also an output port.
      {"windows",
       llvm::json::Object{
           {"type", "object"},
           {"propertyNames",
            llvm::json::Object{
                {"pattern", "^[A-Za-z_][A-Za-z0-9_]*$"},
                {"not",
                 llvm::json::Object{
                     {"enum", llvm::json::Array{"logical", "input"}}}}}},
           {"additionalProperties",
            llvm::json::Object{
                {"oneOf",
                 llvm::json::Array{
                     llvm::json::Object{{"$ref", "#/$defs/window_request"}},
                     llvm::json::Object{{"type", "string"}}}}}}}},
      {"H_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"O_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"D_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"block_size", llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"syndrome_size",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"stim_dem", llvm::json::Object{{"type", "string"}}},
      {"stim_dem_path", llvm::json::Object{{"type", "string"}}},
      {"partition",
       llvm::json::Object{{"$ref", "#/$defs/task_graph_partition"}}},
      {"ensemble", ensemble_json_schema()},
  };
  llvm::json::Array per_kind;
  for (const auto &kind : kinds) {
    const auto *schema =
        find_decoder_schema(cudaq::qec::task_graph_node_schema_name(kind));
    if (!schema)
      continue;
    for (const auto &param : schema->params)
      node_properties[param.key] = json_schema_for_param(param);
  }
  for (const auto &kind : kinds) {
    const auto *schema =
        find_decoder_schema(cudaq::qec::task_graph_node_schema_name(kind));
    if (!schema)
      continue;
    llvm::json::Array required;
    for (const auto &param : schema->params)
      if (param.required)
        required.push_back(param.key);
    // Keys another kind declares and this one does not are as wrong here as an
    // unknown key would be, which is what the parser tells a user.
    std::vector<std::string> foreign;
    for (const auto &other : kinds) {
      if (other == kind)
        continue;
      const auto *other_schema =
          find_decoder_schema(cudaq::qec::task_graph_node_schema_name(other));
      if (!other_schema)
        continue;
      for (const auto &param : other_schema->params) {
        const bool mine = std::any_of(
            schema->params.begin(), schema->params.end(),
            [&](const param_spec &p) { return p.key == param.key; });
        if (!mine && std::find(foreign.begin(), foreign.end(), param.key) ==
                         foreign.end())
          foreign.push_back(param.key);
      }
    }
    llvm::json::Array forbidden;
    for (const auto &key : foreign)
      forbidden.push_back(llvm::json::Object{
          {"not", llvm::json::Object{{"required", llvm::json::Array{key}}}}});
    const auto *spec = cudaq::qec::find_task_graph_node_kind(kind);
    llvm::json::Array spellings{kind};
    if (spec)
      for (const auto &alias : spec->aliases)
        spellings.push_back(alias);
    llvm::json::Object then_clause;
    if (!required.empty())
      then_clause["required"] = std::move(required);
    if (!forbidden.empty())
      then_clause["allOf"] = std::move(forbidden);
    if (then_clause.empty())
      continue;
    per_kind.push_back(llvm::json::Object{
        {"if", llvm::json::Object{{"properties",
                                   llvm::json::Object{
                                       {"type",
                                        llvm::json::Object{
                                            {"enum", std::move(spellings)}}}}},
                                  {"required", llvm::json::Array{"type"}}}},
        {"then", std::move(then_clause)}});
  }

  // A node names a `type`; a pattern entry names `ensemble` instead.
  llvm::json::Object node{
      {"type", "object"},
      {"properties", std::move(node_properties)},
      {"required", llvm::json::Array{"id"}},
      {"oneOf",
       llvm::json::Array{
           llvm::json::Object{{"required", llvm::json::Array{"type"}}},
           llvm::json::Object{{"required", llvm::json::Array{"ensemble"}}}}},
      {"additionalProperties", false}};
  if (!per_kind.empty())
    node["allOf"] = std::move(per_kind);

  // A window request, as an ingest node's `windows` holds it. Each helper
  // returns a fresh object, since json::Value takes objects by move. Round
  // values lie within window_round_limit of zero, as the request check has.
  constexpr auto limit = cudaq::qec::experimental::window_round_limit;
  const auto bounded = [](std::int64_t least, std::int64_t most) {
    return llvm::json::Object{
        {"type", "integer"}, {"minimum", least}, {"maximum", most}};
  };
  const auto rounds_pair = [&] {
    return llvm::json::Object{{"type", "array"},
                              {"items", bounded(-limit, limit)},
                              {"minItems", 2},
                              {"maxItems", 2}};
  };
  const auto range_ref = [] {
    return llvm::json::Object{{"$ref", "#/$defs/round_range"}};
  };
  const auto closed_object = [](llvm::json::Object properties) {
    return llvm::json::Object{{"type", "object"},
                              {"properties", std::move(properties)},
                              {"additionalProperties", false}};
  };
  const auto one_of = [](std::initializer_list<const char *> values) {
    llvm::json::Array list;
    for (const auto *value : values)
      list.push_back(value);
    return llvm::json::Object{{"enum", std::move(list)}};
  };
  llvm::json::Object range_map =
      closed_object(llvm::json::Object{{"rounds", rounds_pair()}});
  range_map["required"] = llvm::json::Array{"rounds"};
  // A send's rounds are one range, or a list of them.
  llvm::json::Object send = closed_object(llvm::json::Object{
      {"id", llvm::json::Object{{"type", "string"}}},
      {"mode", one_of({"delta", "residual"})},
      {"rounds",
       llvm::json::Object{
           {"oneOf", llvm::json::Array{
                         range_ref(), llvm::json::Object{{"type", "array"},
                                                         {"items", range_ref()},
                                                         {"minItems", 1}}}}}}});
  send["required"] = llvm::json::Array{"id"};
  llvm::json::Object input = closed_object(llvm::json::Object{
      {"disposition", one_of({"open", "closed"})},
      {"boundary", one_of({"columns", "truncate"})},
      {"prior", llvm::json::Object{{"type", "number"},
                                   {"exclusiveMinimum", 0},
                                   {"exclusiveMaximum", 1}}}});
  llvm::json::Object output = closed_object(llvm::json::Object{
      {"disposition", one_of({"open", "closed"})}, {"imports", range_ref()}});
  llvm::json::Object zones = closed_object(llvm::json::Object{
      {"commit", range_ref()},
      {"buffers",
       llvm::json::Object{{"type", "array"}, {"items", range_ref()}}}});
  llvm::json::Object columns = closed_object(llvm::json::Object{
      {"merge_identical", llvm::json::Object{{"type", "boolean"}}},
      {"conflicting_parallel", one_of({"reject", "most_likely"})}});
  llvm::json::Object window_request = closed_object(llvm::json::Object{
      {"period", bounded(1, limit)},
      {"offset", bounded(-limit, limit)},
      {"zones", std::move(zones)},
      {"ports",
       closed_object(llvm::json::Object{{"input", std::move(input)},
                                        {"output", std::move(output)}})},
      {"sends",
       llvm::json::Object{{"type", "array"}, {"items", std::move(send)}}},
      {"columns", std::move(columns)}});
  window_request["required"] = llvm::json::Array{"period", "zones", "ports"};
  // A data_artifacts window entry: the request and the dem_chunks entry it
  // is built on.
  llvm::json::Object artifact_window = window_request;
  (*artifact_window.getObject("properties"))["context"] =
      llvm::json::Object{{"type", "string"}};
  artifact_window["required"] =
      llvm::json::Array{"context", "period", "zones", "ports"};
  llvm::json::Array one_kind;
  for (const auto *kind :
       {"dem_chunks", "stim_dem", "stim_dem_path", "window", "edem_path"})
    one_kind.push_back(
        llvm::json::Object{{"required", llvm::json::Array{kind}}});
  llvm::json::Object data_artifact = closed_object(llvm::json::Object{
      {"dem_chunks", llvm::json::Object{{"$ref", "#/$defs/dem_chunks"}}},
      {"stim_dem", llvm::json::Object{{"type", "string"}}},
      {"stim_dem_path", llvm::json::Object{{"type", "string"}}},
      {"window", llvm::json::Object{{"$ref", "#/$defs/artifact_window"}}},
      {"edem_path", llvm::json::Object{{"type", "string"}}},
      {"sha256", llvm::json::Object{{"type", "string"},
                                    {"pattern", "^[0-9a-fA-F]{64}$"}}}});
  data_artifact["oneOf"] = std::move(one_kind);

  return llvm::json::Object{
      {"round_range",
       llvm::json::Object{
           {"oneOf", llvm::json::Array{rounds_pair(), std::move(range_map)}}}},
      {"window_request", std::move(window_request)},
      {"artifact_window", std::move(artifact_window)},
      {"data_artifact", std::move(data_artifact)},
      {"dem_chunks_or_name",
       llvm::json::Object{
           {"oneOf",
            llvm::json::Array{
                llvm::json::Object{{"$ref", "#/$defs/dem_chunks"}},
                llvm::json::Object{{"type", "string"}}}}}},
      {"task_graph_partition",
       llvm::json::Object{
           {"oneOf",
            llvm::json::Array{
                llvm::json::Object{{"type", "string"}},
                llvm::json::Object{
                    {"type", "object"},
                    {"properties",
                     llvm::json::Object{
                         {"columns",
                          llvm::json::Object{
                              {"type", "array"},
                              {"items", llvm::json::Object{{"type", "integer"},
                                                           {"minimum", 0}}}}}}},
                    {"required", llvm::json::Array{"columns"}},
                    {"additionalProperties", false}}}}}},
      {"task_graph_endpoint",
       // `node` or `node.port`; `measurements` and `output` are implicit.
       llvm::json::Object{{"type", "string"},
                          {"pattern", "^[A-Za-z_][A-Za-z0-9_]*(\\.[A-Za-z_]["
                                      "A-Za-z0-9_]*)?$"}}},
      {"task_graph_edge",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"from",
                 llvm::json::Object{{"$ref", "#/$defs/task_graph_endpoint"}}},
                {"to",
                 llvm::json::Object{{"$ref", "#/$defs/task_graph_endpoint"}}},
                {"index_offset",
                 llvm::json::Object{
                     {"type", "integer"},
                     {"minimum", -cudaq::qec::experimental::window_round_limit},
                     {"maximum", 0}}}}},
           {"required", llvm::json::Array{"from", "to"}},
           {"additionalProperties", false}}},
      {"task_graph_node", std::move(node)},
      {"task_graph",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"type", llvm::json::Object{{"type", "string"}}},
                {"inner_type", llvm::json::Object{{"type", "string"}}},
                {"async_streaming",
                 llvm::json::Object{{"type", "boolean"}, {"default", false}}},
                {"async_queue_capacity", llvm::json::Object{{"type", "integer"},
                                                            {"minimum", 1},
                                                            {"default", 8}}},
                {"async_coordinator_cpu",
                 llvm::json::Object{
                     {"type", "integer"}, {"minimum", -1}, {"default", -1}}},
                {"async_spin_pause_iters",
                 llvm::json::Object{
                     {"type", "integer"}, {"minimum", 0}, {"default", 0}}},
                {"async_num_workers",
                 llvm::json::Object{{"type", "integer"}, {"minimum", 1}}},
                {"async_worker_cpus",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "integer"},
                                                  {"minimum", 0}}}}},
                {"nodes",
                 llvm::json::Object{
                     {"type", "array"},
                     {"minItems", 1},
                     {"items",
                      llvm::json::Object{
                          {"$ref", "#/$defs/task_graph_node"}}}}},
                {"edges",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items",
                      llvm::json::Object{
                          {"$ref", "#/$defs/task_graph_edge"}}}}}}},
           {"additionalProperties", false},
           // Either the single-node form or the topology form, never both.
           {"oneOf",
            llvm::json::Array{
                llvm::json::Object{
                    {"required", llvm::json::Array{"type", "inner_type"}},
                    {"not", llvm::json::Object{{"required",
                                                llvm::json::Array{"nodes"}}}}},
                llvm::json::Object{
                    {"required", llvm::json::Array{"nodes"}},
                    {"allOf",
                     llvm::json::Array{
                         llvm::json::Object{
                             {"not",
                              llvm::json::Object{
                                  {"required", llvm::json::Array{"type"}}}}},
                         llvm::json::Object{
                             {"not",
                              llvm::json::Object{
                                  {"required",
                                   llvm::json::Array{"inner_type"}}}}}}}}}}}},
  };
}

// Properties that leave both Stim model keys empty, for the DEM forms that
// take neither.
llvm::json::Object no_stim_dem() {
  return llvm::json::Object{
      {"stim_dem", llvm::json::Object{{"maxLength", 0}}},
      {"stim_dem_path", llvm::json::Object{{"maxLength", 0}}}};
}

// A schema matching an object that has any of @p keys.
llvm::json::Object any_required(std::initializer_list<const char *> keys) {
  llvm::json::Array any;
  for (const auto *key : keys)
    any.push_back(llvm::json::Object{{"required", llvm::json::Array{key}}});
  return llvm::json::Object{{"anyOf", std::move(any)}};
}

// The DEM form with its model in @p key, nonempty, beside D_sparse, and
// @p other, the other spelling, empty.
llvm::json::Object stim_dem_form(const char *key, const char *other) {
  return llvm::json::Object{
      {"required", llvm::json::Array{key, "D_sparse"}},
      {"properties",
       llvm::json::Object{{key, llvm::json::Object{{"minLength", 1}}},
                          {other, llvm::json::Object{{"maxLength", 0}}}}},
      {"not",
       any_required({"H_sparse", "O_sparse", "error_rate_vec", "dem_chunks"})}};
}

} // namespace

std::string decoder_config_json_schema() {
  const auto names = registered_decoder_schema_names();

  llvm::json::Object decoder_params;
  for (const auto &name : names)
    decoder_params[name] =
        decoder_params_json_schema(*find_decoder_schema(name), names);

  // The fixed decoder_config envelope; keep in sync with
  // MappingTraits<decoder_config> above.
  // Envelope integers carry the width of their C++ field, like the int32
  // parameters above, so overflow fails offline as it fails when parsed.
  llvm::json::Object config_properties{
      {"id",
       llvm::json::Object{{"type", "integer"},
                          {"minimum", std::numeric_limits<int64_t>::min()},
                          {"maximum", std::numeric_limits<int64_t>::max()}}},
      {"type", llvm::json::Object{{"type", "string"}}},
      {"dispatch",
       llvm::json::Object{{"enum", llvm::json::Array{"host", "device_graph"}}}},
      {"cuda_device_id",
       llvm::json::Object{
           {"type", "integer"},
           {"minimum", 0},
           {"maximum",
            static_cast<int64_t>(std::numeric_limits<int32_t>::max())}}},
      {"block_size", llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"syndrome_size",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"stim_dem_path", llvm::json::Object{{"type", "string"}}},
      // Model text, or, with no whitespace, a data_artifacts entry's name.
      {"stim_dem", llvm::json::Object{{"type", "string"}}},
      {"H_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"O_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"D_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"error_rate_vec",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "number"}}}}},
      {"dem_chunks",
       llvm::json::Object{{"$ref", "#/$defs/dem_chunks_or_name"}}},
      {"decoder_custom_args", llvm::json::Object{{"type", "object"}}},
      {"task_graph", llvm::json::Object{{"$ref", "#/$defs/task_graph"}}},
  };

  // Per-type dispatch of decoder_custom_args, generated from the registry:
  // a registered type's args follow its schema; a type with no registered
  // schema accepts no args (the parser rejects the section outright).
  llvm::json::Array dispatch;
  // Direct duplicate spelling is representable in JSON Schema. Duplicates in
  // schema-recognized nested parameter sections are rejected by the runtime
  // normalization walk.
  llvm::json::Array duplicate_rate_locations;
  duplicate_rate_locations.push_back(llvm::json::Object{
      {"required",
       llvm::json::Array{"error_rate_vec", "decoder_custom_args"}}});
  llvm::json::Object custom_args_with_rates{
      {"required", llvm::json::Array{"error_rate_vec"}}};
  llvm::json::Object custom_args_constraint{
      {"properties", llvm::json::Object{{"decoder_custom_args",
                                         std::move(custom_args_with_rates)}}}};
  duplicate_rate_locations.push_back(std::move(custom_args_constraint));
  dispatch.push_back(llvm::json::Object{
      {"not",
       llvm::json::Object{{"allOf", std::move(duplicate_rate_locations)}}}});
  for (const auto &name : names)
    dispatch.push_back(llvm::json::Object{
        {"if", llvm::json::Object{{"properties",
                                   llvm::json::Object{
                                       {"type",
                                        llvm::json::Object{{"const", name}}}}},
                                  {"required", llvm::json::Array{"type"}}}},
        {"then", llvm::json::Object{
                     {"properties", llvm::json::Object{
                                        {"decoder_custom_args",
                                         llvm::json::Object{
                                             {"$ref", params_ref(name)}}}}}}}});
  dispatch.push_back(llvm::json::Object{
      {"if",
       llvm::json::Object{
           {"properties",
            llvm::json::Object{
                {"type",
                 llvm::json::Object{
                     {"not", llvm::json::Object{{"enum", registered_name_array(
                                                             names)}}}}}}},
           {"required", llvm::json::Array{"type"}}}},
      {"then",
       llvm::json::Object{
           {"properties",
            llvm::json::Object{{"decoder_custom_args",
                                llvm::json::Object{{"maxProperties", 0}}}}}}}});

  // `type` names the decoder for a plain configuration; a task-graph
  // configuration names its kinds and inner decoders per node instead, and the
  // two are mutually exclusive.
  dispatch.push_back(llvm::json::Object{
      {"if", llvm::json::Object{{"required", llvm::json::Array{"task_graph"}}}},
      {"then",
       llvm::json::Object{
           {"not",
            llvm::json::Object{{"required", llvm::json::Array{"type"}}}}}},
      {"else", llvm::json::Object{{"required", llvm::json::Array{"type"}}}}});

  // A {from, to} pair of names: the seam connection, and each phase
  // connection.
  const auto from_to = [] {
    return llvm::json::Object{
        {"type", "object"},
        {"properties",
         llvm::json::Object{{"from", llvm::json::Object{{"type", "string"}}},
                            {"to", llvm::json::Object{{"type", "string"}}}}},
        {"required", llvm::json::Array{"from", "to"}},
        {"additionalProperties", false}};
  };
  // A {name, spec} entry, as a phase or per-seam spec is written.
  const auto named_spec = [](const std::string &spec_def) {
    return llvm::json::Object{
        {"type", "object"},
        {"properties",
         llvm::json::Object{
             {"name", llvm::json::Object{{"type", "string"}}},
             {"spec", llvm::json::Object{{"$ref", "#/$defs/" + spec_def}}}}},
        {"required", llvm::json::Array{"name", "spec"}},
        {"additionalProperties", false}};
  };

  // The cross-phase rules dem_chunks_spec::validate() enforces (empty init
  // incoming seam, empty final outgoing seam, equal seam widths, one error rate
  // per fault) are not expressible here, so a document that passes this schema
  // may still be rejected when parsed.
  llvm::json::Object dem_chunk_properties{
      {"num_faults", llvm::json::Object{{"type", "integer"}, {"minimum", 1}}},
      {"H_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"seam_specs",
       llvm::json::Object{{"type", "array"},
                          {"items", named_spec("dem_seam_spec")}}},
      {"O_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"error_rates",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "number"},
                                                       {"minimum", 0},
                                                       {"maximum", 1}}}}},
      {"D_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"num_measurements",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"m2o_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
  };

  llvm::json::Object defs{
      {"sparse_matrix",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "integer"},
                                                       {"minimum", -1}}}}},
      // Server-level transport section (see transport_config /
      // transport_shape_override); the wire is deployment config and lives
      // outside the decoders list.
      {"transport_shape_override",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"provider", llvm::json::Object{{"type", "string"}}},
                {"args",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}}}}}},
           {"additionalProperties", false}}},
      {"syndrome_mapping_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"id", llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
                {"type",
                 llvm::json::Object{
                     {"enum", llvm::json::Array{"bulk", "final"}}}},
                {"vps",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items",
                      llvm::json::Object{
                          {"type", "array"},
                          {"items", llvm::json::Object{{"type", "integer"},
                                                       {"minimum", 0}}}}}}}}},
           {"required", llvm::json::Array{"id"}},
           {"additionalProperties", false}}},
      {"transport_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"provider", llvm::json::Object{{"type", "string"}}},
                {"args",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}}}},
                {"device_graph",
                 llvm::json::Object{
                     {"$ref", "#/$defs/transport_shape_override"}}}}},
           {"additionalProperties", false}}},
      {"dem_seam_spec",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"H_sparse",
                 llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
                {"O_sparse",
                 llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}}}},
           {"additionalProperties", false}}},
      {"dem_chunk",
       llvm::json::Object{
           {"type", "object"},
           {"properties", std::move(dem_chunk_properties)},
           {"required", llvm::json::Array{"num_faults", "error_rates"}},
           // H_sparse and seam_specs are the shorthand and per-seam forms
           // of one phase's H; a phase uses one or the other.
           {"not",
            llvm::json::Object{
                {"required", llvm::json::Array{"H_sparse", "seam_specs"}}}},
           {"additionalProperties", false}}},
      {"dem_chunks",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"seam", from_to()},
                {"connections",
                 llvm::json::Object{{"type", "array"}, {"items", from_to()}}},
                {"num_rounds",
                 llvm::json::Object{{"type", "integer"}, {"minimum", 2}}},
                {"measurements_per_round",
                 llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
                {"phases_supply_D_sparse",
                 llvm::json::Object{{"type", "boolean"}}},
                {"observable_names",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}},
                     {"uniqueItems", true}}},
                {"phases",
                 llvm::json::Object{{"type", "array"},
                                    {"items", named_spec("dem_chunk")}}}}},
           // num_rounds is absent from a streaming configuration, whose round
           // count is only known once the experiment runs, so the parser
           // accepts it as optional and expand_dem_chunks() is what demands it.
           {"required", llvm::json::Array{"seam", "connections", "phases"}},
           {"additionalProperties", false}}},
      {"decoder_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties", std::move(config_properties)},
           {"required", llvm::json::Array{"id"}},
           // The DEM is described flat, as repeated phases, or as a nonempty
           // Stim model, by path or inline (or as a name with no whitespace,
           // which the schema cannot tell from text). Expanded chunk
           // configurations retain dem_chunks beside their flat fields, so
           // that combination belongs to the flat branch. A model given both
           // ways matches no branch. The other derived-field exclusions
           // mirror the YAML parser exactly.
           {"oneOf",
            llvm::json::Array{
                llvm::json::Object{
                    {"required", llvm::json::Array{"H_sparse", "block_size",
                                                   "syndrome_size", "O_sparse",
                                                   "D_sparse"}},
                    {"properties", no_stim_dem()}},
                llvm::json::Object{
                    {"required", llvm::json::Array{"dem_chunks"}},
                    {"properties", no_stim_dem()},
                    {"not",
                     any_required({"block_size", "syndrome_size", "H_sparse",
                                   "O_sparse", "D_sparse", "error_rate_vec"})}},
                stim_dem_form("stim_dem_path", "stim_dem"),
                stim_dem_form("stim_dem", "stim_dem_path")}},
           {"additionalProperties", false},
           {"allOf", std::move(dispatch)}}},
      {"decoder_params", std::move(decoder_params)},
  };

  for (auto &&entry : task_graph_json_schema())
    defs[entry.first] = std::move(entry.second);

  llvm::json::Object root{
      {"$schema", "https://json-schema.org/draft/2020-12/schema"},
      {"title", "CUDA-Q QEC realtime decoding configuration"},
      {"description",
       "Validates multi_decoder_config YAML documents. Generated from the "
       "decoder parameter schemas registered in this installation, so it "
       "reflects the decoder plugins loaded at generation time. Per-schema "
       "validate hooks (arbitrary cross-field checks) are not representable "
       "in JSON Schema; a document that passes may still be rejected when "
       "parsed."},
      {"type", "object"},
      {"properties",
       llvm::json::Object{
           {"data_artifacts",
            llvm::json::Object{
                {"type", "object"},
                {"additionalProperties",
                 llvm::json::Object{{"$ref", "#/$defs/data_artifact"}}}}},
           {"decoders",
            llvm::json::Object{
                {"type", "array"},
                {"items",
                 llvm::json::Object{{"$ref", "#/$defs/decoder_config"}}}}},
           {"transport",
            llvm::json::Object{{"$ref", "#/$defs/transport_config"}}},
           {"syndrome_mapping",
            llvm::json::Object{
                {"type", "array"},
                {"items",
                 llvm::json::Object{
                     {"$ref", "#/$defs/syndrome_mapping_config"}}}}}}},
      {"required", llvm::json::Array{"decoders"}},
      {"additionalProperties", false},
      {"$defs", std::move(defs)},
  };

  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::OStream json_out(os, /*IndentSize=*/2);
  json_out.value(llvm::json::Value(std::move(root)));
  return out;
}

// Stash a copy for consumers that build their own decoder instances from the
// process-wide configuration -- the decoding-server DeviceCallService plugin
// reads it when CUDAQ_QEC_DECODER_CONFIG is not set (in-process path).
// shared_ptr + mutex: the plugin reads this from the realtime dispatcher
// thread while the application thread may call configure_decoders() again;
// shared ownership keeps the reader's config alive across a concurrent
// replacement.
static std::mutex g_last_multi_decoder_config_mutex;
static std::shared_ptr<const multi_decoder_config> g_last_multi_decoder_config;

int configure_decoders(multi_decoder_config &config,
                       const std::filesystem::path &base_dir) {
  CUDA_QEC_INFO("Initializing realtime decoding library with config object");
  const int status =
      cudaq::qec::decoding::host::configure_decoders(config, base_dir);
  if (status != 0)
    return status;

  // Stash and publish only once the configuration is actually in effect, so a
  // failed application cannot leave a configuration cached here or advertised
  // to remote targets that nothing is honoring.
  {
    std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
    g_last_multi_decoder_config =
        std::make_shared<const multi_decoder_config>(config);
  }
  // The cudaq integration (ExtraPayloadProvider) is installed by cudaq-qec at
  // load time; this call is a no-op when cudaq-qec is not loaded, keeping this
  // library free of any direct cudaq-common dependency.
  cudaq::qec::publish_decoder_config_payload(config.to_yaml_str());
  return status;
}

int configure_decoders(multi_decoder_config &config) {
  // No originating file: relative model paths resolve against the working
  // directory as it stands when resolution starts.
  return configure_decoders(config, std::filesystem::current_path());
}

std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config() {
  std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
  return g_last_multi_decoder_config;
}

void log_config(const char *config_str, bool from_file) {
  const bool dump_config = []() {
    if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
      if (ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y')
        return true;
    return false;
  }();

  if (dump_config) {
    if (cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info)) {
      CUDA_QEC_INFO(
          "Initializing realtime decoding library with config string: {}",
          config_str);
    } else {
      printf("Initializing realtime decoding library with config string: %s\n",
             config_str);
    }
  }
}

int configure_decoders_from_file(const char *config_file) {
  std::string config_file_str(config_file);
  CUDA_QEC_INFO("Initializing realtime decoding library with config file: {}",
                config_file_str);

  // Verify that the file exists.
  if (!std::filesystem::exists(config_file_str)) {
    CUDA_QEC_WARN("Config file does not exist: {}", config_file_str);
    return 1;
  }

  // Read the config file into a string.
  std::string config_str;
  std::ifstream config_file_stream(config_file_str);
  config_str = std::string(std::istreambuf_iterator<char>(config_file_stream),
                           std::istreambuf_iterator<char>());
  log_config(config_str.c_str(), /*from_file=*/true);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  // Relative model paths resolve against the configuration file's directory,
  // absolute so the resolved paths stay valid if the working directory moves.
  return configure_decoders(
      config, std::filesystem::absolute(config_file_str).parent_path());
}

int configure_decoders_from_str(const char *config_str) {
  CUDA_QEC_INFO(
      "Initializing realtime decoding library with raw config string");
  log_config(config_str, /*from_file=*/false);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

void finalize_decoders() { cudaq::qec::decoding::host::finalize_decoders(); }

} // namespace cudaq::qec::decoding::config

extern "C" const char *cudaq_qec_decoder_config_json_schema() {
  // Regenerated per call; see the declaration for the ownership rule.
  static std::string buffer;
  buffer = cudaq::qec::decoding::config::decoder_config_json_schema();
  return buffer.c_str();
}
