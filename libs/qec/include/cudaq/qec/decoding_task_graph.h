/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// decoding_task_graph: a node/edge DAG representing a decode pipeline.
//
// from_topology() is the general builder: nodes of any registered kind, wired
// by explicit edges. The narrower factories are shorthands for one of its
// shapes — from_spec() and the public constructor for a single window node
// owning D, O, and a decoder; combine() for two window nodes feeding an xor
// node. The public interface is stable; internal graph structure is hidden via
// pimpl. The node-kind registry below is the vocabulary a topology is written
// in: the names a node's `kind` may take, and what each one needs.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/decoding_task_graph_config.h"
#include "cudaq/qec/decoding_task_types.h"
#include "cudaq/qec/extended_dem.h"
#include <any>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudaq::qec {

/// Task-graph node kinds.
///
/// A node kind is what a node's `kind` (or a `task_graph.nodes[].type` value in
/// YAML) names. Kinds are an open registry rather than an enumeration: a new
/// kind registers itself from the translation unit that implements its node,
/// and becomes parseable, validatable, emittable, and visible in the exported
/// JSON schema without any edit to the config layer, the bindings, or this
/// header. Adding a node type is therefore one header and one source, holding
/// the node class, its parameter schema, and its registration.
///
/// Only the metadata a configuration needs is public. The factory that builds
/// the node itself lives behind the library boundary, because the node
/// interface is internal.
///
/// Node parameters (`window_size`, `step_size`, guards, ...) are described by
/// an ordinary `decoding::config::decoder_schema`, registered under the name
/// `task_graph_node.<kind>` so it cannot collide with a decoder plugin of the
/// same name (`sliding_window` is both a plugin and a node kind, with
/// different keys). That reuses the whole existing schema machinery:
/// per-kind parsing, emission, required-key and unknown-key checks, defaults,
/// and JSON-schema generation.
struct task_graph_node_kind_spec {
  /// Canonical name, snake_case. This is what emission writes.
  std::string name;
  /// Accepted alternative spellings, e.g. "SlidingWindow" for
  /// "sliding_window". Parsing accepts these; emission never produces them.
  std::vector<std::string> aliases;
  /// Whether the kind wraps a decoder plugin, and so requires a decoder name
  /// (an `inner:` block or a plugin-valued `type`). An `xor` node does not.
  bool requires_decoder = true;
  /// Whether the kind needs a DEM (`dem_chunks` or flat matrices) to build its
  /// D and O matrices.
  bool requires_dem = true;
  /// Optional policy describing the inner decoder's outputs. When absent, the
  /// output preserves the original observable-model policy: a decoder given O
  /// returns observables, while one without O returns errors. Auxiliary
  /// outputs are requested independently.
  std::function<decoder_output_request(const cudaqx::heterogeneous_map &)>
      inner_decoder_output_policy;

  decoder_output_request
  inner_decoder_output(const cudaqx::heterogeneous_map &params) const {
    if (inner_decoder_output_policy)
      return inner_decoder_output_policy(params);
    return {inner_decoder_takes_observables(params)
                ? decode_result_type::observables
                : decode_result_type::errors};
  }
  /// Whether the kind exists only as a lowering target and is not part of the
  /// graph description a user writes. The config layer rejects such a `type`.
  bool lowered_only = false;
  /// Optional policy deciding whether the inner decoder needs the DEM's
  /// observable model. An absent policy means yes. When the policy returns
  /// false, the topology withholds O and leaves projection to the node.
  std::function<bool(const cudaqx::heterogeneous_map &)>
      inner_decoder_observable_policy;

  bool inner_decoder_takes_observables(
      const cudaqx::heterogeneous_map &params) const {
    return !inner_decoder_observable_policy ||
           inner_decoder_observable_policy(params);
  }
  // Port names are deliberately absent: a node declares them itself, and
  // compile_wires() checks every edge against the node it actually built, so a
  // copy here could only drift out of sync with the real thing.
};

/// The schema-registry name holding `kind`'s node parameters.
inline std::string task_graph_node_schema_name(const std::string &kind) {
  return "task_graph_node." + kind;
}

/// Register (or replace) a node kind, and its parameter schema under
/// `task_graph_node_schema_name(spec.name)`. A kind registered here has no
/// factory, so it parses and validates but is built only where its factory is
/// registered: that is the point for a configuration authored in one process
/// and executed in another.
///
/// The registry container is intentionally leaked to avoid static-destruction
/// ordering hazards. A plugin that registers executable metadata must arrange
/// to unregister its kind before that plugin is unloaded.
__attribute__((visibility("default"))) void
register_task_graph_node_kind(task_graph_node_kind_spec spec,
                              decoding::config::decoder_schema params);

/// Unregister a canonical kind or alias while its defining image is mapped.
/// Plugin cleanup must call this before unloading executable metadata.
__attribute__((visibility("default"))) void
unregister_task_graph_node_kind(const std::string &name);

/// Look up a kind by canonical name or alias. Returns nullptr when no kind
/// matches. The returned pointer stays valid until that kind is replaced or
/// unregistered.
__attribute__((visibility("default"))) const task_graph_node_kind_spec *
find_task_graph_node_kind(const std::string &name);

/// Canonical names of all registered kinds, for diagnostics and schema export.
__attribute__((visibility("default"))) std::vector<std::string>
registered_task_graph_node_kinds();

/// Ids of all registered edge-value types, for diagnostics.
__attribute__((visibility("default"))) std::vector<edge_value_type>
registered_edge_value_types();

/// The kind a node gets when its `type` names a decoder plugin instead of a
/// kind, because such a node is a window over that decoder. Also the default
/// for task_graph_node_config::kind, so the two cannot disagree.
inline constexpr const char *default_task_graph_node_kind = "sliding_window";

/// The kind a `type` value builds: its canonical name when `type` names a kind
/// or one of its aliases, else the default kind, because `type` then named a
/// decoder plugin.
inline std::string resolved_task_graph_node_kind(const std::string &type) {
  if (const auto *kind = find_task_graph_node_kind(type))
    return kind->name;
  return default_task_graph_node_kind;
}

/// Schema describing the node parameters legal beside a given `type`, or null
/// when the kind it resolves to registered none.
inline const decoding::config::decoder_schema *
task_graph_node_params_schema(const std::string &type) {
  return decoding::config::find_decoder_schema(
      task_graph_node_schema_name(resolved_task_graph_node_kind(type)));
}

struct task_graph_node_config;

/// An ensemble pattern: member subgraphs run concurrently, one answer
/// kept. Written in place of a node (`task_graph_node_config::ensemble`) and
/// lowered by from_topology() into an `ensemble_fanout`, `ensemble_join`,
/// and the members. Every member node input that accepts a `%candidate_sink` is
/// wired to the ensemble's candidate sink and every one that accepts a
/// `cancellation_token` to its cancellation token. A member whose edges reach
/// `output` (a `logical_outcome`) has that value pushed into the sink.
///
/// The pattern node has one input, `input`, and one output, `output`. Whatever
/// feeds `input` (a shot, or a streaming producer's windows of measurements or
/// detection events) is passed to every member.
struct task_graph_ensemble_spec {
  struct member {
    std::vector<task_graph_node_config> nodes;
    std::vector<task_graph_edge> edges;
  };
  // Members own node configs, which may own a decoder: move-only.
  task_graph_ensemble_spec() = default;
  task_graph_ensemble_spec(task_graph_ensemble_spec &&) = default;
  task_graph_ensemble_spec &operator=(task_graph_ensemble_spec &&) = default;
  std::vector<member> members;
  /// A registered selection strategy (see %selection_strategy.h) and its
  /// parameters, validated against that strategy's schema.
  std::string join_strategy;
  cudaqx::heterogeneous_map join_params;
};

/// Build inputs for one node of a task graph.
///
/// `kind` is a registered node-kind name rather than an enumerator, and
/// kind-specific settings ride in `node_params`, so a new kind never edits this
/// struct. registered_task_graph_node_kinds() lists what is available.
///
/// Move-only: it may carry an already-built decoder.
struct task_graph_node_config {
  std::string id;
  /// Registered node kind, e.g. "streaming", "sliding_window",
  /// "single_window", "xor".
  std::string kind = default_task_graph_node_kind;
  /// The kind's own parameters: window_size, step_size, guards, ...
  cudaqx::heterogeneous_map node_params;

  /// DEM for this node. When absent, `d_sparse`/`o_sparse` below supply the
  /// matrices directly.
  std::optional<dem_chunks_spec> dem;
  /// Pre-expanded matrices, used when `dem` is absent. Rows are CSR:
  /// `d_sparse` maps measurements to detectors, `o_sparse` fault columns to
  /// observables, and `h_sparse` detectors to the fault columns that trigger
  /// them. `h_sparse` is optional and only a kind that maps a predicted error
  /// back into detector space needs it; the others are the flat form's usual
  /// pair. Flat matrices still describe one fixed-T window with no per-round
  /// column layout, so this does not make the flat form a substitute for
  /// `dem` in a windowed topology.
  std::vector<std::vector<uint32_t>> d_sparse, o_sparse, h_sparse;

  /// Inner decoder plugin name and its constructor parameters. Ignored by a
  /// kind that wraps no decoder.
  std::string decoder_type;
  cudaqx::heterogeneous_map params;
  /// Stim DEM text to build the inner decoder from, instead of this node's H.
  /// For a decoder whose model needs what the text carries and the matrices do
  /// not: `detector` coordinate annotations have no home in
  /// `detector_error_model`, so a decoder that reads them can only be built
  /// from the text. The node's own D/O still come from `dem`/`d_sparse` — the
  /// text describes the decoder, the chunks describe the window. Requires an
  /// inner decoder that projects to observables itself, since the text's fault
  /// columns are not this node's.
  std::string stim_dem;
  /// Already-built inner decoder, taking precedence over `decoder_type`. The
  /// flat form needs this: a decoder is built from H and the error rates,
  /// neither of which is recoverable from D and O.
  std::unique_ptr<decoder> decoder_instance;

  /// When set, this entry is an ensemble pattern rather than a node: `kind` and
  /// the fields above are ignored, and from_topology() lowers it in place.
  std::optional<task_graph_ensemble_spec> ensemble;
};

/// Operational state of a streaming graph or one of its named outputs.
enum class task_graph_streaming_status { pending, available, complete, failed };

/// Progress of one named graph output. `generation` is a monotonic publication
/// watermark and advances only when every contributor to this output has
/// published the corresponding update. Consuming corrections preserves this
/// watermark but changes `status` to `pending`; generation alone never implies
/// that a value is currently readable.
struct task_graph_output_progress {
  std::string name;
  std::size_t value_offset = 0;
  std::size_t value_width = 0;
  std::uint64_t generation = 0;
  std::uint64_t coverage_begin_round = 0;
  std::uint64_t coverage_end_round = 0;
  task_graph_streaming_status status = task_graph_streaming_status::pending;
};

/// Graph-owned streaming result. `values` contains all named outputs at their
/// declared offsets. Synchronous graphs return a live graph-owned view that
/// must not be read concurrently with graph activity. Async-streaming graphs
/// return a stable reader snapshot after synchronizing the accepted prefix.
/// Copy either form to retain a historical result across later graph calls.
/// `operation_completed` is an edge-triggered indication that the latest
/// synchronized public enqueue advanced the aggregate barrier; level-triggered
/// readability is determined by `status`: only `available` or `complete` is
/// readable (and either necessarily has a nonzero generation). Consuming
/// corrections changes status to `pending` but deliberately preserves
/// generation as a monotonic publication watermark for detecting the next
/// update. Generation alone must therefore never be used as a readiness test.
/// `result_revision` advances whenever any output or graph status changes.
/// `completed_operations` counts public enqueue operations that advanced the
/// aggregate result, even when several complete before the next read.
/// Only reset_decoder() advances shot_epoch.
struct task_graph_streaming_result {
  std::uint64_t shot_epoch = 0;
  std::uint64_t result_revision = 0;
  /// Aggregate-completing public enqueue operations in this shot epoch.
  std::uint64_t completed_operations = 0;
  std::uint64_t generation = 0;
  task_graph_streaming_status status = task_graph_streaming_status::pending;
  bool operation_completed = false;
  std::string error;
  std::vector<task_graph_output_progress> outputs;
  std::vector<uint8_t> values;
};

/// @brief Decode pipeline as a composable, testable object.
///
/// Owns one or more decoder plugins and the D/O matrices that connect
/// raw measurement bits to detector events and observables.
///
/// **Streaming interface**:
///   - `enqueue_syndrome` supplies raw measurement chunks to streaming-capable
///     source nodes. Window nodes accumulate them; native-streaming nodes
///     forward each public call to their inner decoder. Published node outputs
///     then flow through the streaming DAG.
///
/// **One-shot execution**:
///   - `run()` walks the internal node/edge DAG for one full shot.
///
/// **Asynchronous execution**:
///   - `run_async()` returns one future for the shot. Independent ready nodes
///     in a multi-node DAG may execute concurrently.
///
/// **Batch execution**:
///   - `run_batch()` walks the graph once for each supplied shot.
///
/// The decoders are stateful: do not call into the same graph from two
/// threads, collect one `run_async` before starting the next, and do not
/// overlap streaming with one-shot calls. Overlap is detected rather than
/// serialised -- an overlapping call throws `std::logic_error`. An
/// async-streaming graph instead has one internal coordinator as its sole
/// node-state mutator; its public streaming calls provide copied input and
/// coherent snapshots.
class decoding_task_graph {
public:
  /// Construct from an already-built decoder and D/O in nested-row form.
  ///
  /// Uses a fixed-T window: the full D matrix is applied once per decode,
  /// and the accumulation buffer is exactly `d_sparse` column-width bytes.
  ///
  /// @param dec      Inner decoder plugin.
  /// @param d_sparse Measurement-to-detector map (CSR row form).
  /// @param o_sparse Observable-flip map (CSR row form).
  /// @param options  Construction-time execution policy.
  decoding_task_graph(std::unique_ptr<decoder> dec,
                      std::vector<std::vector<uint32_t>> d_sparse,
                      std::vector<std::vector<uint32_t>> o_sparse,
                      task_graph_options options = {});

  ~decoding_task_graph();
  decoding_task_graph(decoding_task_graph &&) noexcept;
  decoding_task_graph &operator=(decoding_task_graph &&) noexcept;

  // Non-copyable: owns the inner decoder.
  decoding_task_graph(const decoding_task_graph &) = delete;
  decoding_task_graph &operator=(const decoding_task_graph &) = delete;

  // ---------------------------------------------------------------------------
  // Factory
  // ---------------------------------------------------------------------------

  /// Build from a `dem_chunks_spec` and a named decoder plugin.
  ///
  /// When `spec.num_rounds` is set, builds a fixed-T window.
  /// When absent (streaming), builds a sliding window (stride = 1) sized to
  /// the minimum rounds required by the spec's phase structure.
  ///
  /// @param spec         Phase/connection spec describing the DEM structure.
  /// @param decoder_type Registry name of the inner decoder.
  /// @param params       Additional constructor parameters for the decoder.
  /// @param options      Construction-time execution policy.
  static std::unique_ptr<decoding_task_graph>
  from_spec(const dem_chunks_spec &spec, const std::string &decoder_type,
            const cudaqx::heterogeneous_map &params = {},
            task_graph_options options = {});

  /// Build an arbitrary topology: nodes of any registered kind, wired by
  /// explicit edges.
  ///
  /// Each node's D and O come from its own `dem` (expanded to the round count
  /// its `window_size` asks for) or from its flat `d_sparse`/`o_sparse`.
  ///
  /// `edges` may be omitted for a single-node graph, which is wired
  /// `measurements -> node -> output`. Nodes must be given in topological
  /// order; an edge that runs backwards is rejected. Each node's registered
  /// parameter schema is applied here, including default materialization, so
  /// YAML and programmatic construction have the same parameter contract.
  /// `options` selects the graph's streaming execution policy.
  ///
  /// @throws std::invalid_argument for an unregistered kind, a node missing
  ///         what its kind requires, a dangling edge endpoint, an unknown port
  ///         name, a reserved node id, an unconnected or multiply connected
  ///         input, incompatible port value types, an unsupported graph
  ///         output, or incompatible statically known edge-value shapes.
  /// @throws std::runtime_error for an unknown, incorrectly typed, missing, or
  ///         cross-field-invalid node parameter.
  static std::unique_ptr<decoding_task_graph>
  from_topology(std::vector<task_graph_node_config> nodes,
                std::vector<task_graph_edge> edges = {},
                task_graph_options options = {});

  /// Combine two single-node graphs into a three-node DAG that XORs their
  /// logical outcomes.  Both input graphs are consumed.
  ///
  /// **Streaming**: both branches must have the same effective firing and
  /// commit schedule. An incompatibility is recorded on the combined graph:
  /// its complete one-shot and batch DAG remains usable, while
  /// ``enqueue_syndrome()`` throws ``std::logic_error``.
  ///
  /// @throws std::invalid_argument if either graph is null, is not a
  ///         single-node graph, or produces a different observable width.
  static std::unique_ptr<decoding_task_graph>
  combine(std::unique_ptr<decoding_task_graph> left,
          std::unique_ptr<decoding_task_graph> right,
          task_graph_options options = {});

  // ---------------------------------------------------------------------------
  // Streaming interface
  // ---------------------------------------------------------------------------

  /// Accumulate one round of measurement syndromes.
  ///
  /// Returns true when any window fired (D → decode → O). For independently
  /// progressing outputs this does not mean that an aggregate result is ready;
  /// inspect streaming_result() for output and operation status. For fixed-T,
  /// a window fires once when its full buffer is filled. For sliding-window,
  /// it fires every `stride` rounds once the buffer is full.
  ///
  /// Any chunking works: the graph meters a chunk to each node's remaining
  /// room, so one call may drive several fires.  `get_num_msyn_per_round()` is
  /// the chunk to prefer, not a limit.  Unlike `decoder::enqueue_syndrome`,
  /// which still ignores a chunk that does not fit, nothing is dropped here.
  ///
  /// With construction-time asynchronous streaming enabled, a BULK call
  /// returns false after copying its input into the bounded queue; call
  /// streaming_result() to wait for accepted input and inspect completion. If
  /// the queue is full, this call waits only until one slot is available. A
  /// FINAL call waits for all accepted input and the final flush before
  /// returning.
  ///
  /// @throws std::logic_error if the graph's streaming routes or effective
  ///         output schedules are incompatible. Batch execution remains valid.
  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t len);

  /// Same, carrying the round's shape.  A `FINAL` round commits whatever a
  /// guarded window is still withholding, so the tail of a shot is not lost.
  /// Returns true if a window fired or a tail was committed.
  ///
  /// `FINAL` ends the shot: further rounds are warned about and dropped until
  /// clear_corrections() or reset_decoder(), because the rounds it committed
  /// are the ones the next window's core would commit and corrections
  /// accumulate by XOR.
  ///
  /// @throws std::invalid_argument if a `FINAL` round is empty: it carries
  ///         the shot's final measurements.
  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t len,
                        round_type type);

  /// Host-accessible corrections buffer (num_observables bytes).
  ///
  /// During synchronous streaming, this is a current-value read between calls
  /// to `enqueue_syndrome()`: an output appears as soon as its node fires, and
  /// one whose node has not fired yet reads as zero -- indistinguishable from a
  /// genuine zero. For an async-streaming graph it waits for records accepted
  /// before the call, then returns a coherent coordinator snapshot without
  /// closing the session or flushing it. It never aliases state the coordinator
  /// mutates.
  /// Its pointer remains valid only until the next result accessor or mutating
  /// graph call; copy the bytes if they must outlive either.
  ///
  /// @throws std::logic_error If streaming is unsupported or, for a
  /// synchronous graph, another graph operation is in flight.
  const uint8_t *get_obs_corrections() const;

  /// Return values and per-output publication state from the latest streaming
  /// operation. A zero-valued output is distinguishable from an unavailable
  /// one through its status and generation. Coverage is the half-open round
  /// range incorporated by the latest generation. The returned reference is a
  /// live view for synchronous graphs. For async-streaming graphs, each call
  /// waits for records accepted before the call, then refreshes a stable
  /// reader-owned snapshot; copy it before another result accessor call if a
  /// historical snapshot is required.
  ///
  /// @throws std::logic_error If streaming is unsupported or, for a
  /// synchronous graph, another graph operation is in flight.
  const task_graph_streaming_result &streaming_result() const;

  /// Consume the current corrections without starting a new result epoch.
  ///
  /// This zeroes values and marks aggregate and output statuses `pending`, but
  /// preserves sliding-window overlap, coverage, and generation watermarks so
  /// streaming can continue and the next publication can be detected. Use
  /// reset_decoder() instead to start a clean shot epoch. For an asynchronous
  /// graph, this first drains all accepted records without flushing guarded
  /// tails.
  ///
  /// @throws std::logic_error If another graph operation is in flight.
  /// @throws std::runtime_error If asynchronous processing failed.
  void clear_corrections();

  /// Reset all per-shot state (accumulation buffer, corrections) on every node.
  /// For an asynchronous graph, this first cancels queued records and waits for
  /// any currently executing decoder call to return.
  ///
  /// @throws std::logic_error If another graph operation is in flight.
  void reset_decoder();

  /// Total syndrome bytes per full window (= buffer capacity).  This is what
  /// the first decode of a shot consumes; a slid window then fires every
  /// `stride` rounds.  Nodes reading the stream may hold windows of different
  /// sizes, in which case this is the largest, i.e. the bytes needed before
  /// every one of them has fired.
  std::size_t get_num_msyn_per_decode() const;

  /// Common syndrome-byte unit used by the graph's streaming schedules. This
  /// is the preferred chunk size, not an enqueue limit; for variable-width
  /// native rounds it may be a common divisor of caller-visible round sizes.
  /// A window that does not slide also accepts
  /// `get_num_msyn_per_decode()` per call.
  std::size_t get_num_msyn_per_round() const;

  /// Number of observables (= corrections buffer length).
  std::size_t get_num_observables() const;

  /// Whether the compiled graph supports incremental streaming execution.
  /// A false result does not affect run(), run_all(), or other batch APIs.
  bool supports_streaming() const noexcept;

  /// Empty when supports_streaming() is true; otherwise the first deterministic
  /// build-time reason streaming execution is unavailable.
  const std::string &streaming_incompatibility_reason() const noexcept;

  /// Which bits are which: one name per output, in the order
  /// get_obs_corrections() concatenates them and run_all() returns them.  Each
  /// is the sink edge's `to_input`, or its producer's name when left at the
  /// default.  Fixed at build.
  const std::vector<std::string> &output_names() const;

  // ---------------------------------------------------------------------------
  // One-shot and batch execution interface
  // ---------------------------------------------------------------------------

  /// Walk the node/edge DAG for one shot: detect(m) then execute each node.
  ///
  /// Does not touch the streaming accumulation buffer.
  ///
  /// @throws std::logic_error if more than one edge reaches `output`: a
  /// multi-output program reduced to its first output is a different program,
  /// so it is refused rather than answered.  Use run_all().
  logical_outcome run(const measurement_results &m) const;

  /// run() for any number of outputs: one outcome per edge reaching `output`,
  /// in output_names() order.
  std::vector<logical_outcome> run_all(const measurement_results &m) const;

  /// @brief Run one full shot asynchronously.
  ///
  /// The shot is copied before this function returns, so `m` need not outlive
  /// the call. Single-node graphs apply D synchronously and use the decoder's
  /// own async path. Multi-node graphs execute dependency-ready nodes
  /// concurrently on a bounded persistent worker pool. Each node owns its
  /// decoder instance, so independent instances may decode concurrently. The
  /// graph's single-flight contract prevents concurrent entry into the same
  /// node instance.
  ///
  /// @param m Raw measurement bits for one complete shot.
  /// @return A future containing the graph's single logical outcome.
  /// @throws std::logic_error If another graph operation is already in flight
  /// or the graph does not have exactly one output.
  /// @throws std::runtime_error If graph nodes request different CUDA devices.
  /// Node execution failures are stored in the returned future and rethrown by
  /// `std::future::get()`.
  std::future<logical_outcome> run_async(const measurement_results &m) const;

  /// Multi-record version: run() for each element in order.
  std::vector<logical_outcome>
  run_batch(const std::vector<measurement_results> &shots) const;

  // ---------------------------------------------------------------------------
  // Accessors (primarily for testing and diagnostics)
  // ---------------------------------------------------------------------------

  /// Measurement-to-detector matrix (CSR). Rows = detectors, cols =
  /// measurements.
  const sparse_binary_matrix &d_matrix() const;
  /// Observable-flip matrix (CSR). Rows = observables, cols = fault columns.
  /// Throws for combined graphs (no single O matrix).
  const sparse_binary_matrix &o_matrix() const;

  /// The CUDA device every GPU-affine node agrees on, or nullopt if none has
  /// GPU affinity. A caller must pin to this device before a node's decoder
  /// runs CUDA kernels of its own, same as detail_affinity::pin_decode_device
  /// already requires for a plain decoder.
  ///
  /// @throws std::runtime_error if GPU-affine nodes disagree on device --
  /// one thread can't pin to two devices for one shot. Pin every GPU-affine
  /// node in a graph to the same cuda_device_id.
  std::optional<int> cuda_device_id() const;

  // ---------------------------------------------------------------------------
  // DEVICE-mode graph dispatch
  // ---------------------------------------------------------------------------
  //
  // Initial cut is single-node only: a combine()-produced graph (more than
  // one streaming node) always reports false / throws below rather than
  // attempting a multi-decoder capture, which is deferred.

  /// True if this graph's one streaming node supports graph dispatch. Always
  /// false for a multi-node (combine()) graph.
  bool supports_graph_dispatch() const;

  /// Capture the graph's single streaming node's full decode pipeline
  /// (D-apply -> decode -> O-apply) into a CUDA graph. Same void* convention
  /// as decoder::capture_decode_graph() (avoids <cuda_runtime.h> here);
  /// caller casts to cudaq::qec::realtime::graph_resources* and owns it.
  ///
  /// @throws std::logic_error if called on a multi-node graph, or a
  /// single-node graph whose node does not supports_graph_dispatch() --
  /// callers are expected to check supports_graph_dispatch() first, so this
  /// is a misuse guard, not the primary gate.
  void *capture_graph(int reserved_sms = 0);

  /// Release graph resources previously returned by capture_graph().
  void release_graph(void *graph_resources);

  /// This graph's inner decoder, when it has exactly one streaming node that
  /// wraps one; nullptr for a multi-node graph or a node kind that wraps no
  /// decoder. Exposed for the realtime session to assign the decoder_id
  /// DEVICE-mode capture needs (see capture_graph()).
  decoder *inner_decoder() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;

  /// Wraps a compiled impl. Only assemble() calls this, so no graph can reach a
  /// caller without having gone through compile_wires().
  explicit decoding_task_graph(std::unique_ptr<impl> i);

  /// Finish a hand-filled impl: compile its string edges to wires and wrap it.
  /// The one path every builder ends on.
  static std::unique_ptr<decoding_task_graph>
  assemble(std::unique_ptr<impl> i, task_graph_options options = {});
};

} // namespace cudaq::qec

namespace cudaq::qec::detail {

/// One registration of a payload type under a name, held for the object's
/// lifetime.  CUDAQ_REGISTER_EDGE_VALUE_TYPE creates one at namespace scope,
/// so the type is registered when the module loads and released when it
/// unloads.  A type may be registered under several names, and a name by
/// several modules; the name stays until every registration is released.
class __attribute__((visibility("default"))) edge_value_registration {
public:
  /// Registers T under `name`.  Throws std::invalid_argument if the name is
  /// empty or already names another type.
  template <typename T>
  edge_value_registration(std::string name, std::in_place_type_t<T>)
      : edge_value_registration(
            std::move(name), edge_value_type::of<T>(),
            [](std::any &dst, const std::any &src) {
              std::any_cast<T &>(dst) = std::any_cast<const T &>(src);
            },
            [](std::any &dst, std::any &src) {
              std::any_cast<T &>(dst) = std::move(std::any_cast<T &>(src));
            }) {
    static_assert(std::is_copy_assignable_v<T> && std::is_move_assignable_v<T>,
                  "an edge-value type must be copy- and move-assignable");
  }

  /// Releases the registration.  Must not run while an edge_value of the type,
  /// or a graph carrying one, is alive: the registry holds function pointers
  /// into the registering module.
  ~edge_value_registration();

  edge_value_registration(const edge_value_registration &) = delete;
  edge_value_registration &operator=(const edge_value_registration &) = delete;

private:
  edge_value_registration(std::string name, edge_value_type type,
                          void (*copy_assign)(std::any &, const std::any &),
                          void (*move_assign)(std::any &, std::any &));
  std::string name_;
};

} // namespace cudaq::qec::detail

#define CUDAQ_EDGE_VALUE_CONCAT_(a, b) a##b
#define CUDAQ_EDGE_VALUE_CONCAT(a, b) CUDAQ_EDGE_VALUE_CONCAT_(a, b)

/// Register TYPE under NAME at namespace scope in the .cpp that owns TYPE, for
/// as long as that module is loaded.
#define CUDAQ_REGISTER_EDGE_VALUE_TYPE(TYPE, NAME)                             \
  static const ::cudaq::qec::detail::edge_value_registration                   \
      CUDAQ_EDGE_VALUE_CONCAT(cudaq_edge_value_registration_,                  \
                              __LINE__){NAME, std::in_place_type<TYPE>};
