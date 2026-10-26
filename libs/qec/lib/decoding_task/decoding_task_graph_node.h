/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: types shared between decoding_task_graph and future
// node implementations.  Never included from a public header.

#pragma once
#include "decoding_task_edge_value.h"
#include "cudaq/qec/decoding_task_types.h"
#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

class decoder;

struct window_spec {
  uint32_t size;        ///< total rounds in the accumulation buffer
  uint32_t stride;      ///< rounds to advance per decode
  uint32_t start_guard; ///< leading context rounds (excluded from commit)
  uint32_t end_guard;   ///< trailing context rounds (excluded from commit)

  uint32_t core() const { return size - start_guard - end_guard; }

  bool operator==(const window_spec &other) const {
    return size == other.size && stride == other.stride &&
           start_guard == other.start_guard && end_guard == other.end_guard;
  }
};

/// Construction-time edges are the public task_graph_edge (string names);
/// compile_wires() turns them into compiled_wires, which is what runs.
struct compiled_wire {
  int from_node; ///< index into nodes[]; -1 = "measurements" source
  int from_slot; ///< index into that node's outputs vector
  int to_node;   ///< index into nodes[]; -1 = "output" sink
  int to_slot;   ///< index into that node's inputs vector
};

/// Optional per-round streaming capability. DAG-only nodes do not implement it.
class streaming_task_graph_node {
public:
  /// Effective schedule delivered to one input slot by a streaming edge.
  struct input_schedule {
    std::size_t slot;
    window_spec schedule;
  };

  virtual ~streaming_task_graph_node() = default;
  struct committed_coverage {
    std::uint64_t begin_round;
    std::uint64_t end_round;
  };
  /// Actual ownership when FINAL ends between regular scheduled firings.
  virtual std::optional<committed_coverage> streaming_coverage() const {
    return std::nullopt;
  }
  /// Accumulate one raw chunk.  `type` is FINAL only on the chunk that ends a
  /// FINAL round -- the last piece of it this node receives -- and BULK
  /// otherwise.  flush() follows a FINAL round either way, so a kind that
  /// commits its tail there may ignore `type`.
  virtual bool enqueue(const uint8_t *msyn, std::size_t len,
                       round_type type) = 0;
  /// Accumulate one raw chunk of a record the caller did not tag.  Only a kind
  /// whose decoder tells untagged rounds from BULK ones needs to override it.
  virtual bool enqueue_untagged(const uint8_t *msyn, std::size_t len) {
    return enqueue(msyn, len, round_type::BULK);
  }
  virtual const uint8_t *corrections() const = 0;
  virtual void clear_corrections() = 0;
  virtual void reset_streaming_state() = 0;
  virtual std::size_t num_msyn_per_decode() const = 0;
  virtual std::size_t num_observables() const = 0;

  /// Syndrome bytes in one measurement round
  virtual std::size_t num_msyn_per_round() const {
    return num_msyn_per_decode();
  }

  /// Firing and commit schedule for a raw-input streaming node. The default
  /// describes a non-overlapping window; sliding or guarded kinds override it.
  virtual std::optional<window_spec> streaming_schedule() const {
    const std::size_t per_round = num_msyn_per_round();
    const std::size_t per_decode = num_msyn_per_decode();
    if (per_round == 0 || per_decode == 0 || per_decode % per_round != 0)
      return std::nullopt;
    const auto rounds = static_cast<uint32_t>(per_decode / per_round);
    return window_spec{rounds, rounds, 0, 0};
  }

  /// Derive the effective schedule emitted on `output_slot`. For a raw-fed
  /// node, `inputs` is empty and the default returns streaming_schedule().
  /// Edge-fed nodes must override this: unknown timing is deliberately not
  /// assumed to preserve the producer's schedule.
  virtual std::optional<window_spec>
  derive_streaming_schedule(std::size_t /*output_slot*/,
                            const std::vector<input_schedule> &inputs) const {
    if (!inputs.empty())
      return std::nullopt;
    return streaming_schedule();
  }

  /// Room left in this node's accumulator: the most one enqueue() call can
  /// take.  nullopt for a kind that keeps none, which is handed whole chunks.
  /// enqueue_syndrome() meters to it, so a caller may cut the stream anywhere.
  virtual std::optional<std::size_t> raw_chunk_limit() const {
    return std::nullopt;
  }
  /// The same for an untagged chunk, which a kind whose decoder ends a
  /// fixed-length shot on untagged rounds meters to that shot's end.
  virtual std::optional<std::size_t> untagged_chunk_limit() const {
    return raw_chunk_limit();
  }

  /// Whether `enqueue()` represents raw measurements delivered to `slot`.
  /// Unlike enqueue_edge_value(), enqueue() has no slot argument, so a node
  /// must explicitly identify the one input for which it is valid.
  virtual bool supports_raw_streaming_input(std::size_t /*slot*/) const {
    return false;
  }

  /// Whether input slot `slot` can consume an edge during streaming.
  virtual bool supports_streaming_input(std::size_t /*slot*/) const {
    return false;
  }

  /// Called after this node fires during streaming execution.
  /// Returns a pointer to the pre-allocated output value on `slot`, or nullptr
  /// if not supported.  The pointer is valid until the next call that mutates
  /// node state (enqueue, flush, reset_streaming_state), and holds whatever
  /// that call most recently produced -- after a flush that is the committed
  /// tail, not the last fire.
  virtual const edge_value *streaming_output(std::size_t slot) const {
    return nullptr;
  }

  /// Accumulate a pre-routed value into this node's streaming buffer.
  /// Called by enqueue_syndrome when an upstream streaming node fires and
  /// its output slot is wired to this node.  Returns true when this node
  /// fires as a result.  Default: not supported (returns false).
  virtual bool enqueue_edge_value(const edge_value &v, std::size_t to_slot) {
    return false;
  }

  /// Commit whatever this node is still withholding, because no more syndrome
  /// is coming.  Returns true if that produced a new streaming output.  Must
  /// be idempotent: corrections accumulate by XOR, so a second commit of the
  /// same data cancels it out.  Default: nothing withheld.
  virtual bool flush() { return false; }

  /// Called once per incoming streaming edge by compile_wires(). This validates
  /// constraints local to one input; derive_streaming_schedule() validates
  /// relationships among multiple input schedules and declares their resulting
  /// output timing. Throw to reject. Default: accept.
  virtual void validate_edge_input(std::size_t /*to_slot*/) const {}
};

/// Optional capability for a node that owns a decoder. These operations all
/// describe that decoder or optimized execution paths it may provide.
class decoder_backed_task_graph_node {
public:
  virtual ~decoder_backed_task_graph_node() = default;
  /// Reset the owned decoder and all node state coupled to it. A node that also
  /// streams must include reset_streaming_state() in this operation.
  virtual void reset_decoder() = 0;
  virtual std::future<logical_outcome>
  run_async(const measurement_results &m) const;
  virtual std::optional<int> cuda_device_id() const = 0;
  virtual bool supports_graph_dispatch() const { return false; }
  virtual void *capture_graph(int reserved_sms = 0) {
    (void)reserved_sms;
    return nullptr;
  }
  virtual void release_graph(void *graph_resources) { (void)graph_resources; }
  virtual decoder *inner_decoder() const = 0;
};

/// Core DAG node contract. Optional execution modes are exposed as capabilities
/// so a node implements only operations that have meaning for it.
class task_graph_node {
public:
  virtual ~task_graph_node() = default;

  /// inputs[i]  = value at the i-th input slot (order declared by inputs()).
  /// outputs[i] = value at the i-th output slot (order declared by outputs()).
  /// Slots are assigned by compile_wires(); no string ops at runtime.
  ///
  /// A multi-node run_async() may call this concurrently on independent node
  /// objects. The graph's single-flight contract ensures the same node object
  /// is never entered concurrently by two graph operations.
  virtual void execute_indexed(const std::vector<edge_value> &inputs,
                               std::vector<edge_value> &outputs) const = 0;

  /// Input name declarations — used once at construction by compile_wires().
  virtual std::vector<std::string> inputs() const = 0;
  /// Output name declarations — used once at construction by compile_wires().
  virtual std::vector<std::string> outputs() const = 0;

  /// Path-independent value type id produced on output slot `slot`; nullopt
  /// for no output edge on that slot.
  virtual std::optional<edge_value_type>
  output_type(std::size_t /*slot*/) const {
    return std::nullopt;
  }

  /// Whether input `slot` accepts values of the registered type `type`. A
  /// predicate is used because a decoder input may accept either raw
  /// measurements or already-computed detection events.
  virtual bool accepts_input_type(std::size_t /*slot*/,
                                  edge_value_type /*type*/) const {
    return false;
  }

  /// Number of scalar values emitted on an output slot when statically known.
  /// Used to validate DAG shape independently of execution mode.
  virtual std::optional<std::size_t>
  output_value_size(std::size_t /*slot*/) const {
    return std::nullopt;
  }

  /// Validate an incoming value's static shape. Throw for an incompatibility
  /// that makes both batch and streaming execution invalid.
  virtual void
  validate_input_shape(std::size_t /*slot*/, edge_value_type /*type*/,
                       std::optional<std::size_t> /*producer_size*/) const {}

  /// Capability queries. A null pointer means this node does not support that
  /// execution mode; no sentinel dimensions or no-op implementations needed.
  virtual streaming_task_graph_node *streaming() { return nullptr; }
  virtual decoder_backed_task_graph_node *decoder_backed() { return nullptr; }
  virtual const decoder_backed_task_graph_node *decoder_backed() const {
    return nullptr;
  }

  /// True when this one-shot-only node combines logical inputs by GF(2) XOR.
  /// compile_wires() may then reduce its streaming correction paths to parity
  /// without executing the node on every fire.
  virtual bool supports_streaming_xor_fold() const { return false; }
};

} // namespace cudaq::qec
