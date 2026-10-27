/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the node-kind factory side of the registry whose public
// metadata lives in cudaq/qec/decoding_task_graph.h.  The factory is private
// because it returns a task_graph_node.  A pattern kind registers a lowering
// in its place, which rewrites the node into others before the build.  Never
// included from a public header.

#pragma once
#include "cuda-qx/core/heterogeneous_map.h"
#include "decoding_task_graph_node.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include "cudaq/qec/window_request.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

class data_artifact_store;
class experiment_template;

/// Everything a node factory is handed.  from_topology() resolves the DEM and
/// builds the inner decoder before dispatching, so a factory only has to
/// interpret its own parameters and hand back a node.
struct node_build_context {
  std::string id;
  /// Measurement-to-detector and observable-flip matrices; null for a kind
  /// declaring `requires_dem = false`.
  std::shared_ptr<const sparse_binary_matrix> d, o;
  /// Inner decoder; null for a kind declaring `requires_decoder = false`.
  std::unique_ptr<decoder> dec;
  uint32_t spr = 0; ///< syndrome bytes in one round
  /// Measurement bytes in the whole stream, when rounds differ in width and
  /// `spr` is therefore only the typical round's. Zero means `dem_rounds *
  /// spr`.
  std::size_t num_measurements = 0;
  uint32_t dem_rounds = 0; ///< rounds the node's DEM spans
  /// Cumulative per-round fault-column offsets (`dem_rounds + 1` entries), or
  /// empty for flat matrices with unknown layout or native observable
  /// streaming, which does not expand the chain's fault frames.
  std::vector<std::size_t> fault_offsets;
  /// Detectors to the fault columns that trigger them, over the same detector
  /// rows `d` produces and the same columns `o` indexes -- what a kind needs
  /// to map a predicted error back into detector space. Resolved from whichever
  /// form described the node, so a kind need not know which that was; null
  /// when the flat form omitted it or native observable streaming needs no
  /// closed matrix. Kinds needing only D and O ignore it.
  std::shared_ptr<const sparse_binary_matrix> h;
  /// The node's closed DEM when its matrices came from a `dem`; null for the
  /// flat form or native observable streaming. Carries the error rates and
  /// observable model beside `h`, for a kind that needs more of the DEM than
  /// its matrices.
  std::shared_ptr<const detector_error_model> closed_dem;
  /// Rebuildable fixed-width memory recipe for arbitrary-length FINAL tails.
  std::shared_ptr<const dem_chunks_spec> memory_spec;
  std::string decoder_type;
  cudaqx::heterogeneous_map decoder_params;
  const cudaqx::heterogeneous_map *node_params = nullptr;
  /// In a windowed topology: every variant of every layer, built for the
  /// graph, whose handles outlive it; null elsewhere.
  const data_artifact_store *store = nullptr;
  /// An ingest node's template, or that of the ingest node a decode_window
  /// node reads.
  std::shared_ptr<const experiment_template> context;
  /// An ingest node's layers, in port order.
  std::vector<experimental::ingest_window> windows;
  /// A decode_window node's layer: the request of the ingest port it reads.
  std::optional<experimental::window_request> layer;
  /// A decode_window node's receives, by index: the first instance from
  /// which each is guaranteed, if any (first_guaranteed_receive()).
  std::vector<std::optional<std::int64_t>> guaranteed_receives;
};

using node_factory =
    std::function<std::unique_ptr<task_graph_node>(node_build_context &)>;

/// How a pattern kind expands. A node of a pattern kind is never built:
/// from_topology() rewrites it into plain nodes and edges before any node is
/// checked, as it does an ensemble entry, so the expanded graph is validated
/// as a hand-written one is.
struct node_pattern {
  using lowering = std::function<void(task_graph_node_config cfg,
                                      std::vector<task_graph_node_config> &out,
                                      std::vector<task_graph_edge> &edges)>;
  using window_layers =
      std::vector<std::pair<std::string, experimental::window_request>>;
  using window_declaration = std::function<window_layers(
      const cudaqx::heterogeneous_map &params, const experiment_template &)>;

  /// Append the nodes replacing pattern node `cfg` to `out`, move the edges
  /// naming cfg.id onto them, and append their own edges. Throws
  /// std::invalid_argument naming the pattern when `cfg` is not one it can
  /// expand. Required.
  lowering lower;
  /// The window requests the lowered nodes read, by layer name, from the
  /// pattern's parameters and template. A configuration registers each as
  /// window `<id>.<layer>` before the build. Empty when the lowered nodes
  /// read no windows.
  /// Must declare every request `lower` gives its nodes; an undeclared one
  /// fails only at build, when its windows are not in the store. Derive both
  /// hooks from one function, as parallel_window does.
  window_declaration windows;
};

using node_kind_registry_lock = std::unique_lock<std::recursive_mutex>;

/// Hold the node-kind registry stable across a multi-step lookup or factory
/// call. Registry entry points may be called while this recursive lock is held.
node_kind_registry_lock lock_node_kind_registry();

/// Register a kind's public metadata and parameter schema together with the
/// factory that builds it.  One registrar per kind covers both registries.
__attribute__((visibility("default"))) void
register_node_kind(task_graph_node_kind_spec spec,
                   decoding::config::decoder_schema params,
                   node_factory factory);

/// Unregister `name` only if `registration_id` still owns it.
__attribute__((visibility("default"))) void
unregister_task_graph_node_kind_if_current(const std::string &name,
                                           uint64_t registration_id);

/// Unregisters its kind when destroyed. Hold one at namespace scope in the
/// plugin that registered the kind.
class [[nodiscard]] node_kind_registration {
public:
  node_kind_registration() = default;
  node_kind_registration(std::string name, uint64_t registration_id)
      : name_(std::move(name)), registration_id_(registration_id) {}
  node_kind_registration(node_kind_registration &&other) noexcept
      : name_(std::exchange(other.name_, {})),
        registration_id_(std::exchange(other.registration_id_, 0)) {}
  node_kind_registration &operator=(node_kind_registration &&other) noexcept {
    if (this != &other) {
      reset();
      name_ = std::exchange(other.name_, {});
      registration_id_ = std::exchange(other.registration_id_, 0);
    }
    return *this;
  }
  node_kind_registration(const node_kind_registration &) = delete;
  node_kind_registration &operator=(const node_kind_registration &) = delete;
  ~node_kind_registration() { reset(); }

  void reset() {
    if (!name_.empty()) {
      const uint64_t registration_id = std::exchange(registration_id_, 0);
      unregister_task_graph_node_kind_if_current(std::exchange(name_, {}),
                                                 registration_id);
    }
  }

private:
  std::string name_;
  uint64_t registration_id_ = 0;
};

/// Register a kind and return a token that unregisters it on destruction.
__attribute__((visibility("default"))) node_kind_registration
register_scoped_node_kind(task_graph_node_kind_spec spec,
                          decoding::config::decoder_schema params,
                          node_factory factory);

/// Factory registered for a canonical kind name or alias, or nullptr. A kind
/// registered through the public metadata-only entry point has no factory, so
/// it parses and validates but cannot be built here.
__attribute__((visibility("default"))) const node_factory *
find_node_factory(const std::string &kind);

/// Register a pattern kind's public metadata and parameter schema together
/// with its lowering. A kind is a pattern or has a factory, never both:
/// registering it either way replaces the other.
/// @throws std::invalid_argument if `pattern.lower` is empty, or as
/// register_task_graph_node_kind() does.
__attribute__((visibility("default"))) void
register_pattern_kind(task_graph_node_kind_spec spec,
                      decoding::config::decoder_schema params,
                      node_pattern pattern);

/// Register a pattern kind and return a token that unregisters it on
/// destruction, before the plugin holding its hooks is unloaded. Hold the
/// token at namespace scope in that plugin, as for register_scoped_node_kind().
/// @throws as register_pattern_kind() does.
__attribute__((visibility("default"))) node_kind_registration
register_scoped_pattern_kind(task_graph_node_kind_spec spec,
                             decoding::config::decoder_schema params,
                             node_pattern pattern);

/// Lowering registered for a canonical kind name or alias, or nullptr when
/// the kind is not a pattern. The pointer stays valid while the caller holds
/// lock_node_kind_registry(), or until the kind is replaced or unregistered.
__attribute__((visibility("default"))) const node_pattern *
find_node_pattern(const std::string &kind);

} // namespace cudaq::qec
