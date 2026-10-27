/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: the node-kind factory side of the registry whose public
// metadata lives in cudaq/qec/decoding_task_graph.h.  The factory is private
// because it returns a task_graph_node.  Never included from a public header.

#pragma once
#include "cuda-qx/core/heterogeneous_map.h"
#include "decoding_task_graph_node.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

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
};

using node_factory =
    std::function<std::unique_ptr<task_graph_node>(node_build_context &)>;

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

} // namespace cudaq::qec
