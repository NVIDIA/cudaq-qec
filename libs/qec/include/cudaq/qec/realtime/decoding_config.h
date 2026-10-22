/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/decoding_task_graph_config.h"
#include "cudaq/qec/extended_dem.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// Dispatch shape for a decoder session -- HOW its RPCs are executed, not
/// which wire the bytes arrive on (the wire is a server-level transport
/// provider, selected independently).
/// host:         requests are dispatched on the CPU (HOST_CALL); works with
///               any transport provider (dev, CI, no GPU required).
/// device_graph: requests are dispatched on the GPU by the self-relaunching
///               device-graph scheduler (DeviceGraphTransceiver); requires a
///               decoder with a captured decode graph and a provider whose
///               rings are GPU-visible (e.g. GpuRoceTransceiver/DOCA).
/// YAML key: `dispatch: host|device_graph`.
enum class DecoderDispatch { host, device_graph };

/// @brief Decoder-specific constructor arguments, stored as a
/// `cudaqx::heterogeneous_map` -- the form every decoder's constructor
/// consumes. YAML conversion and key validation are driven by the parameter
/// schema the decoder registered (see cudaq/qec/decoder_config_schema.h), so
/// out-of-tree decoders participate without any framework changes. The
/// framework-owned `error_rate_vec` model input is also accepted here as a
/// legacy YAML alias, but is removed before plugin construction.
class decoder_custom_args_t {
public:
  decoder_custom_args_t() = default;
  decoder_custom_args_t(const cudaqx::heterogeneous_map &m) : map_(m) {}

  decoder_custom_args_t &operator=(const cudaqx::heterogeneous_map &m) {
    map_ = m;
    return *this;
  }

  cudaqx::heterogeneous_map &map() { return map_; }
  const cudaqx::heterogeneous_map &map() const { return map_; }
  bool empty() const { return map_.empty(); }

  /// Deep equality over the canonical custom-args value kinds.
  __attribute__((visibility("default"))) bool
  operator==(const decoder_custom_args_t &other) const;

private:
  cudaqx::heterogeneous_map map_;
};

/// @brief Configuration structure for decoder options.
struct decoder_config {
  int64_t id = 0;
  std::string type;
  /// Dispatch shape for this decoder's RPCs.  Defaults to host.  Set to
  /// device_graph for decoders where syndrome bits are DMA'd directly to GPU
  /// VRAM and decoded by a captured CUDA graph (e.g. nv_qldpc_decoder with
  /// RelayBP).
  DecoderDispatch dispatch = DecoderDispatch::host;
  /// CUDA device this decoder is pinned to at construction (see the
  /// "cuda_device_id" decoder parameter). Placement knob common to any
  /// GPU-accelerated decoder, hence at this level rather than inside the
  /// per-decoder custom args. Unset = unpinned.
  std::optional<int> cuda_device_id;
  /// The fields below describe the DEM three alternative ways, and exactly
  /// one of them applies:
  ///
  ///   - Flat form: H_sparse plus block_size, syndrome_size, O_sparse,
  ///     D_sparse and optional error_rate_vec, all sized for the whole
  ///     experiment.
  ///   - Chunk form: dem_chunks (which carries phases, connections, seam, and
  ///     num_rounds internally). The flat fields are derived by expanding the
  ///     phases, and must be omitted.
  ///   - DEM form: stim_dem_path, the Stim model text itself. The decoder
  ///     derives its dimensions, observable mapping and error rates from it.
  ///     H_sparse, O_sparse and error_rate_vec must be omitted; D_sparse is
  ///     still required, while block_size and syndrome_size are optional
  ///     checked assertions.
  ///
  /// Throughout a configuration, a `*_path` key names a file and a key without
  /// the suffix carries its contents inline: `stim_dem_path` is read from disk,
  /// while a node's `stim_dem` is the model text itself.
  ///
  /// Chunk form stays authoritative: decoder construction hands the chunks to
  /// the decoder, alone for one that declared it reads chunks natively
  /// (decoder_init::from_dem_chunks(); see
  /// register_decoder_native_model_source()) and with their closed model for
  /// any other (decoder_init::from_dem_chunks_closed()). expand_dem_chunks()
  /// is the explicit derivation of the equivalent flat form. DEM form stays
  /// authoritative: the model text is what the decoder is built from.
  uint64_t block_size = 0;
  uint64_t syndrome_size = 0;
  /// Path to an authoritative Stim detector error model. Relative paths are
  /// resolved against the configuration file's directory, or the process
  /// working directory for a programmatic or raw-string configuration.
  std::string stim_dem_path;
  std::vector<std::int64_t> H_sparse;
  std::vector<std::int64_t> O_sparse;
  /// Maps raw measurements to detectors. Orthogonal to the model source and
  /// required by both.
  std::vector<std::int64_t> D_sparse;
  /// Error probability per H column. Input YAML also accepts the established
  /// `decoder_custom_args.error_rate_vec` spelling, and emitted YAML uses that
  /// spelling to preserve the researcher-facing wire format.
  std::vector<double> error_rate_vec;
  /// Optional per-phase DEM for a streaming, repeated-round decomposition.
  /// H_sparse above describes the whole experiment as one flat matrix; these
  /// phases describe one round each so the round count can be chosen at run
  /// time. num_rounds lives inside dem_chunks_spec. See
  /// cudaq::qec::dem_chunks_from_spec() for expansion to a chunk sequence.
  ///
  /// A configuration that also has a nonempty H_sparse is flat, and that
  /// matrix is the one decoders are built from. Form selection keys off
  /// H_sparse.empty(). Nonempty H_sparse is exactly the state
  /// expand_dem_chunks() leaves behind, allowing round-trip through YAML.
  std::optional<cudaq::qec::dem_chunks_spec> dem_chunks;
  decoder_custom_args_t decoder_custom_args;

  /// Decoder task graph topology.  When set, `type` must be absent and either
  /// `dem_chunks` or the flat matrices supply the DEM.
  ///
  /// Two forms, mutually exclusive:
  ///
  ///   - Single node, named by `type` and `inner_type`.  `decoder_custom_args`
  ///     are forwarded to the inner decoder.  `type` is a node kind, in any
  ///     spelling the registry accepts: "sliding_window" for a window that
  ///     advances one round per decode, or "single_window" for one window over
  ///     a fixed round count (which needs that count: `H_sparse` or
  ///     `dem_chunks.num_rounds`).
  ///
  ///   - `nodes` and `edges`, describing an arbitrary topology.  A node names
  ///     a registered node kind or a decoder plugin (see
  ///     %registered_task_graph_node_kinds()), carries that kind's parameters
  ///     inline, and may override the DEM.  `edges` may be omitted for a
  ///     single node, which is wired `measurements -> node -> output`.
  struct task_graph_spec {
    /// A range of DEM fault columns a node owns, `[lo, hi]` inclusive.
    struct column_range {
      uint64_t lo = 0, hi = 0;
      bool operator==(const column_range &) const = default;
    };

    struct node_spec {
      std::string id;
      /// A registered node kind (canonical or aliased spelling), or a decoder
      /// plugin name, in which case the kind defaults to "sliding_window".
      std::string type;
      /// Inner decoder plugin name, when `type` named a node kind.
      std::string inner_type;
      /// The inner decoder's constructor arguments, from either the `inner:`
      /// block or a node-level `decoder_custom_args:`.
      decoder_custom_args_t decoder_custom_args;
      /// The node kind's own parameters: window_size, step_size, guards, ...
      /// Parsed against the schema registered for the resolved kind.
      decoder_custom_args_t node_params;
      /// DEM for this node alone; falls back to the entry-level DEM.
      std::optional<cudaq::qec::dem_chunks_spec> dem_chunks;
      /// This node's own matrices, in the entry-level flat form and the same
      /// -1-separated row encoding. Use them for a detector map that follows no
      /// round structure; a node takes these or `dem_chunks`, never both.
      ///
      /// `D_sparse` and `O_sparse` are required together. `block_size` and
      /// `syndrome_size` give `H_sparse` its shape, which sparse rows do not
      /// carry. `H_sparse` also builds this node's inner decoder, along with
      /// its `decoder_custom_args`, so a node with its own `D` and `O` needs
      /// either an `H_sparse` or a `stim_dem`.
      std::vector<std::int64_t> H_sparse;
      std::vector<std::int64_t> D_sparse;
      std::vector<std::int64_t> O_sparse;
      std::uint64_t block_size = 0;
      std::uint64_t syndrome_size = 0;
      /// Stim DEM text to build the inner decoder from, in place of this
      /// node's H -- the YAML spelling of the string alternative of
      /// `decoder_init`, for a decoder whose model needs annotations the
      /// matrices cannot carry. The text is inline, as everywhere else in this
      /// library: callers pass DEM contents, never a path. The node's own D/O
      /// still come from its DEM, so the text describes the decoder and the
      /// chunks describe the window.
      std::string stim_dem;
      /// The same model as `stim_dem`, named by file rather than inlined, for
      /// callers who keep DEMs on disk. Read and folded into `stim_dem` before
      /// the graph builder runs, so nothing downstream distinguishes the two
      /// spellings; setting both on one node is an error rather than a
      /// precedence.
      std::string stim_dem_path;
      /// Which DEM columns this node owns, as a label ("local", "boundary") or
      /// explicit ranges.  Parsed, validated, and emitted; not yet honored at
      /// construction, which rejects a partitioned node.
      std::string partition_label;
      std::vector<column_range> partition_columns;

      /// An ensemble pattern written in place of a node: member subgraphs
      /// raced against each other, one answer kept (the YAML spelling of
      /// `cudaq::qec::task_graph_ensemble_spec`). `type` stays empty and the
      /// node fields above are ignored; the entry is lowered by the builder.
      struct ensemble_spec {
        struct member {
          std::vector<node_spec> nodes;
          std::vector<cudaq::qec::task_graph_edge> edges;
          bool operator==(const member &) const = default;
        };
        std::vector<member> members;
        std::string join_strategy;
        decoder_custom_args_t join_params;
        bool operator==(const ensemble_spec &) const = default;
      };
      std::optional<ensemble_spec> ensemble;
      bool operator==(const node_spec &) const = default;

      /// The node kind this node builds: `type` resolved to its canonical name
      /// when it names a registered kind, otherwise the default kind, because
      /// `type` then named a decoder plugin instead.
      __attribute__((visibility("default"))) std::string resolved_kind() const;

      /// The inner decoder plugin: `inner_type`, or `type` when that named a
      /// plugin rather than a node kind. Empty for a kind wrapping no decoder.
      __attribute__((visibility("default"))) std::string
      resolved_decoder_type() const;
    };

    std::string type;       ///< single-node form: node kind
    std::string inner_type; ///< single-node form: inner decoder registry name
    /// Construction-time streaming execution policy. Async mode is opt-in.
    cudaq::qec::task_graph_options options;
    std::vector<node_spec> nodes;
    /// The graph's wires, in the same form from_topology() takes, so the
    /// configuration is the topology rather than a parallel description of it.
    /// YAML writes an endpoint as `node` or `node.port` and omits the port when
    /// it is the default one task_graph_edge fills in.
    std::vector<cudaq::qec::task_graph_edge> edges;
    bool operator==(const task_graph_spec &) const = default;
  };
  std::optional<task_graph_spec> task_graph;

  bool operator==(const decoder_config &) const = default;

  /// The decoder type used for schema lookup and custom-args handling.
  /// Returns task_graph->inner_type when task_graph is set (type must be
  /// absent in that case), otherwise returns type.
  __attribute__((visibility("default"))) const std::string &
  effective_decoder_type() const;

  /// Decoder names shown to a user, which is not always the one schema
  /// lookup wants: a `nodes`-form graph keeps its decoders on the nodes, so
  /// effective_decoder_type() is empty there while this joins what the nodes
  /// name. Falls back to the node kinds for a graph whose nodes take no
  /// decoder, so a log line always names something.
  __attribute__((visibility("default"))) std::string
  describe_decoder_type() const;

  /// Return the parameter map a decoder's constructor should receive: the
  /// stored custom args with the legacy error-rate alias removed and
  /// schema-declared defaults materialized (see materialize_default_args in
  /// cudaq/qec/decoder_config_schema.h) when a schema is registered for
  /// the effective decoder type, so programmatically built configs get the
  /// same normalization the YAML parse path applies.
  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  decoder_custom_args_to_heterogeneous_map() const;

  /// Return the model error rates from either the canonical C++ field or the
  /// legacy `decoder_custom_args.error_rate_vec` YAML spelling. The legacy
  /// key is configuration compatibility data and is never included in the
  /// parameter map handed to a decoder plugin.
  ///
  /// @throws std::runtime_error if rates are supplied in both locations, or
  ///         more than once in nested decoder custom arguments.
  __attribute__((visibility("default"))) std::vector<double>
  effective_error_rate_vec() const;

  /// Validate `decoder_custom_args` against the parameter schema registered
  /// for the effective decoder type: unknown keys, missing required keys, and
  /// the schema's own validate hook (if any). Throws std::runtime_error on
  /// the first violation. YAML parsing applies the same checks automatically;
  /// call this to vet a configuration built programmatically before using it.
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static decoder_config
  from_yaml_str(const std::string &yaml_str);
};

/// Transport override applied to the rings of one dispatch shape (see the
/// `device_graph` member of `transport_config`).
struct transport_shape_override {
  /// Provider name (e.g. udp, cpu_roce, gpu_roce) or /path/to/lib.so.
  /// A bare name resolves to the CUDA-Q realtime provider library whose
  /// soname is "libcudaq-realtime-bridge-" + name + ".so", with '_' in the
  /// name mapping to '-' to match the shipped hyphenated sonames (so
  /// gpu_roce loads libcudaq-realtime-bridge-gpu-roce.so).
  /// Empty = inherit the section default.
  std::string provider;
  /// Extra provider arguments appended for this shape's rings.
  std::vector<std::string> args;

  bool operator==(const transport_shape_override &) const = default;
};

/// Server-level transport section: the WIRE is deployment configuration and
/// lives OUTSIDE the decoders list.  Transports differ between rings only by
/// dispatch shape (a device_graph ring must be GPU-pollable), so the only
/// override is shape-keyed -- decoder entries carry no transport
/// information.
///
///   transport:
///     provider: udp
///     args: [--slot-size=256]
///     device_graph:
///       provider: udp          # "gpu_roce" on an HSB rig
///       args: [--pinned-rings]
///
/// Resolution per ring: shape override (device_graph rings) > this section's
/// provider/args. The standalone server permits a CLI fallback only for
/// non-GPU-RoCE, host-only configurations whose YAML names no provider. A
/// configuration containing device_graph dispatch or using GPU RoCE must
/// define its provider and arguments in YAML (the deployment file is the
/// source of truth for GPU-visible rings).
struct transport_config {
  std::string provider;
  std::vector<std::string> args;
  transport_shape_override device_graph;

  /// Resolve the provider and ordered arguments for a device-graph ring.
  /// The shape-specific provider overrides the section provider, while its
  /// arguments are appended to the section arguments.
  transport_shape_override resolve_device_graph() const {
    transport_shape_override resolved;
    resolved.provider =
        device_graph.provider.empty() ? provider : device_graph.provider;
    resolved.args = args;
    resolved.args.insert(resolved.args.end(), device_graph.args.begin(),
                         device_graph.args.end());
    return resolved;
  }

  bool operator==(const transport_config &) const = default;
};

/// One row of the server's syndrome mapping table.
///
/// A caller names a row per round through the `syndrome_mapping_id` field of
/// an enqueue request.  The row says what shape that round has, and per
/// virtual port which round-relative detector indices the port's bits scatter
/// into.
///
/// The shape is what carries end-of-shot: a decoder that decides its length at
/// run time cannot see the end in the syndrome stream alone, because the last
/// round's fault set differs from a bulk round's.  Naming a `final` row
/// conveys it, with nothing added to the wire.
///
/// ```yaml
/// syndrome_mapping:
///   - id: 0
///     type: bulk
///     vps: [[]]      # one virtual port, pass-through
///   - id: 1
///     type: final
///     vps: [[]]
/// ```
struct syndrome_mapping_config {
  /// Row key; the value a caller places in an enqueue request.
  uint64_t id = 0;
  /// Round shape: "bulk" (mid-circuit) or "final" (last round of a shot).
  std::string type = "bulk";
  /// Per virtual port, the round-relative detector indices that port's bits
  /// scatter into.  An empty list is the pass-through identity: the fragment
  /// becomes the round at whatever length it arrived, which is what lets a
  /// caller batch a whole round into one message.
  ///
  /// Defaults to one pass-through port; only that is accepted today, so a row
  /// need not name it.  See validate().
  std::vector<std::vector<uint64_t>> vps{{}};

  bool operator==(const syndrome_mapping_config &) const = default;

  /// @brief Throw std::runtime_error if the row is not well formed.
  __attribute__((visibility("default"))) void validate() const;
};

class multi_decoder_config {
public:
  std::vector<decoder_config> decoders;
  /// Optional server-level transport section. The standalone server's CLI
  /// defaults may fill an empty section only for non-GPU-RoCE, host-only
  /// configurations.
  transport_config transport;
  /// Optional syndrome mapping table.  When empty the server installs a single
  /// pass-through `bulk` row under id 0, which is what every configuration
  /// written before this section existed gets.
  std::vector<syndrome_mapping_config> syndrome_mapping;

  bool operator==(const multi_decoder_config &) const = default;

  /// @brief Throw std::runtime_error if the syndrome mapping table is
  /// malformed: an unknown round shape, a duplicate row key, or a row that
  /// validate() refuses.  Called by %from_yaml_str() and again where the table
  /// is built, so a programmatically assembled config cannot skip it.
  __attribute__((visibility("default"))) void validate_syndrome_mapping() const;

  /// Validate every decoder's custom args (see
  /// decoder_config::validate_custom_args).
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static multi_decoder_config
  from_yaml_str(const std::string_view yaml_str);
};

/// @brief Rewrite a chunk-form configuration into the equivalent flat form,
/// filling block_size, syndrome_size, H_sparse, O_sparse, D_sparse and
/// error_rate_vec from `dem_chunks` expanded `num_rounds` times. Decoder
/// construction does not need this: it keeps the chunk form (see
/// decoder_init::from_dem_chunks_closed()).
///
/// Does nothing to a configuration that is already flat (one whose `H_sparse`
/// is nonempty, or which carries no `dem_chunks` at all), so it is safe to
/// call unconditionally. An empty H_sparse with dem_chunks present is still
/// treated as chunk form.
///
/// @return The closed DEM the flat fields were derived from, so a caller that
///         also wants its per-fault priors does not have to expand a second
///         time. Empty when the configuration was already flat.
/// @throws std::runtime_error if `num_rounds` is missing, or if the phases
///         cannot be expanded to that many rounds.
__attribute__((visibility("default")))
std::optional<cudaq::qec::detector_error_model>
expand_dem_chunks(decoder_config &config);

/// @brief Generate a JSON Schema (draft 2020-12) document describing valid
/// `multi_decoder_config` YAML files, so third-party tools (check-jsonschema,
/// python jsonschema, yaml-language-server, ...) can validate user-provided
/// configurations offline. The per-decoder `decoder_custom_args` sections are
/// generated from the parameter schemas registered at call time, so decoder
/// plugins (in-tree and out-of-tree alike) appear automatically once their
/// library is loaded. Schema `validate` hooks are arbitrary C++ and are not
/// representable; a document that passes the JSON Schema may still be
/// rejected by those hooks when parsed.
__attribute__((visibility("default"))) std::string decoder_config_json_schema();

/// @brief Configure the decoders (`multi_decoder_config` variant). This
/// function configures both local decoders, and if running on remote target
/// hardware, will submit the configuration to the remote target for further
/// processing.
/// @param config The configuration to use.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders(multi_decoder_config &config);

/// @brief Configure the decoders, resolving relative model paths (such as
/// `stim_dem_path`) against @p base_dir. The overload above uses the process
/// working directory as it stands when resolution starts.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders(multi_decoder_config &config,
                   const std::filesystem::path &base_dir);

/// @brief Configure the decoders from a file. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_file The file to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_file(const char *config_file);

/// @brief Configure the decoders from a string. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_str The string to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_str(const char *config_str);

/// @brief Finalize the decoders. This function finalizes local decoders.
__attribute__((visibility("default"))) void finalize_decoders();

/// @brief Return the most recently passed multi_decoder_config, or an empty
/// pointer if configure_decoders() has not been called in this process.
/// Used by the decoding-server DeviceCallService plugin to build
/// DecodingSessions on the in-process host_dispatch path without requiring
/// CUDAQ_QEC_DECODER_CONFIG.  Returns shared ownership: a concurrent
/// configure_decoders() replaces the stored config but cannot free it out
/// from under the caller.
__attribute__((visibility("default")))
std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config();

} // namespace cudaq::qec::decoding::config
