/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../hardware_guards.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoding_task_graph.h"

#include <cstdint>
#include <fmt/core.h>
#include <optional>
#include <stdexcept>

namespace cudaq::qec::detail_dispatch {

/// The decode endpoint behind a decoder id: either a registered task graph or
/// a plain decoder.  A task graph shadows the decoder slot at the same id, and
/// the two expose the same streaming interface, so callers work through this
/// instead of writing the pair of branches once per operation.
///
/// Resolution stays with the caller: the inline dispatch path indexes its own
/// tables and throws, while the realtime session reads the published atomics
/// and maps a miss to an RPC status.
class decode_target {
public:
  struct result_update {
    bool operation_completed = false;
    std::uint64_t completed_results = 0;
  };

  decode_target() = default;
  explicit decode_target(cudaq::qec::decoder *dec) : decoder_(dec) {}
  explicit decode_target(cudaq::qec::decoding_task_graph *graph)
      : graph_(graph) {}

  explicit operator bool() const { return decoder_ || graph_; }

  /// Reject a syndrome buffer this target cannot consume.
  void validate_syndromes(const uint8_t *syndromes,
                          std::uint64_t syndrome_length) const {
    if (syndrome_length == 0)
      throw std::invalid_argument("syndrome_length must be greater than 0");
    if (!syndromes)
      throw std::invalid_argument("syndromes buffer is null");
    if (num_msyn_per_decode() == 0)
      throw std::invalid_argument(
          fmt::format("{} has no measurement syndromes configured", kind()));
    const auto limit = chunk_limit();
    if (limit && syndrome_length > *limit)
      throw std::invalid_argument(fmt::format(
          "syndrome_length ({}) exceeds configured measurement count ({})",
          syndrome_length, *limit));
  }

  /// Reject a corrections buffer that does not match this target's shape;
  /// copying into a shorter one would read past the target's own buffer.
  void validate_corrections(const uint8_t *corrections,
                            std::uint64_t correction_length) const {
    if (correction_length == 0)
      throw std::invalid_argument("correction_length must be greater than 0");
    if (!corrections)
      throw std::invalid_argument("corrections buffer is null");
    const auto observables = num_observables();
    if (correction_length != observables)
      throw std::invalid_argument(
          fmt::format("correction_length ({}) does not match number of "
                      "observables ({})",
                      correction_length, observables));
  }

  /// Point the calling thread at the device this target decodes on, before
  /// work that allocates or launches on it.  Decoders are built sequentially
  /// on one thread, leaving the LAST decoder's device current, so a dispatch
  /// path must pin before touching plugin state.  Set-and-leave (no restore):
  /// one dispatcher thread serves every decoder, so it simply converges to the
  /// device of whichever it is serving, and pinning an already-current device
  /// is a cheap no-op.  Throws on failure (fail fast) rather than decoding on
  /// whichever device happened to be current.
  ///
  /// A task graph's dispatch is CPU-hosted, but an inner decoder can still run
  /// CUDA kernels on whatever device is current, so it needs the same pin.
  void pin_device() const {
    if (decoder_)
      cudaq::qec::detail_affinity::pin_decode_device(*decoder_);
    else if (const auto device = graph_device())
      cudaq::qec::detail_affinity::set_cuda_device_for_decode(*device);
  }

  /// Same, but cached per calling thread -- see
  /// detail_affinity::set_cuda_device_for_decode_cached(). For dispatcher
  /// threads that serve one target per call and would otherwise re-pin an
  /// already-current device on every call.
  void pin_device_cached() const {
    if (decoder_)
      cudaq::qec::detail_affinity::pin_decode_device_cached(*decoder_);
    else if (const auto device = graph_device())
      cudaq::qec::detail_affinity::set_cuda_device_for_decode_cached(*device);
  }

  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t len) const {
    return graph_ ? graph_->enqueue_syndrome(syndrome, len)
                  : decoder_->enqueue_syndrome(syndrome, len);
  }

  /// Same, carrying the round's shape.  A graph commits its guarded tail on a
  /// FINAL round; a decoder that tracks its own length overrides the tagged
  /// overload.  Either way the caller reads the one boolean it always read.
  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t len,
                        cudaq::qec::round_type type) const {
    return graph_ ? graph_->enqueue_syndrome(syndrome, len, type)
                  : decoder_->enqueue_syndrome(syndrome, len, type);
  }

  /// Interpret one enqueue result and freeze a newly completed graph aggregate
  /// into caller-owned storage, so independently progressing outputs cannot
  /// change what was published. Completion is edge-triggered for both kinds:
  /// a plain decoder's enqueue return value, or a graph aggregate completed
  /// since graph_result was frozen.
  result_update
  capture_result_update(bool decoder_enqueue_result,
                        task_graph_streaming_result &graph_result) const {
    if (!graph_)
      return {decoder_enqueue_result, decoder_enqueue_result ? 1u : 0u};

    const auto &result = graph_->streaming_result();
    if (result.status == task_graph_streaming_status::failed)
      throw std::runtime_error("task graph streaming failed: " + result.error);
    const bool current_available =
        result.generation > 0 &&
        (result.status == task_graph_streaming_status::available ||
         result.status == task_graph_streaming_status::complete);
    const bool operation_completed =
        current_available &&
        result.completed_operations > graph_result.completed_operations;
    const std::uint64_t completed_results =
        operation_completed
            ? result.completed_operations - graph_result.completed_operations
            : 0;
    if (operation_completed) {
      if (result.values.size() != graph_result.values.size())
        throw std::logic_error(
            "task graph result snapshot does not match its output width");
      graph_result = result;
    }
    return {operation_completed, completed_results};
  }

  const uint8_t *obs_corrections() const {
    if (!graph_)
      return decoder_->get_obs_corrections();
    // The host realtime API performs one corrections read, not a poll. Wait
    // through records accepted before that read, then return its stable bytes.
    (void)graph_->streaming_result();
    return graph_->get_obs_corrections();
  }

  /// Return a session-frozen graph aggregate or a decoder's live completed
  /// result. The caller owns and preserves graph_snapshot while readable.
  const uint8_t *published_corrections(const uint8_t *graph_snapshot) const {
    return graph_ ? graph_snapshot : decoder_->get_obs_corrections();
  }

  void clear_corrections() const {
    if (graph_)
      graph_->clear_corrections();
    else
      decoder_->clear_corrections();
  }

  void reset() const {
    if (graph_)
      graph_->reset_decoder();
    else
      decoder_->reset_decoder();
  }

  std::size_t num_msyn_per_decode() const {
    return graph_ ? graph_->get_num_msyn_per_decode()
                  : decoder_->get_num_msyn_per_decode();
  }

  /// Measurement bytes this target always accepts in one enqueue call: one
  /// round for a sliding window, which frees only that much per fire, and a
  /// whole buffer for a decoder, which empties on every fire.
  std::size_t num_msyn_per_round() const {
    return graph_ ? graph_->get_num_msyn_per_round() : num_msyn_per_decode();
  }

  /// Largest chunk one enqueue call may carry, or nullopt when the target
  /// meters internally and so takes any length.  A graph meters; a decoder
  /// still ignores a chunk that does not fit, leaving its buffer the bound.
  std::optional<std::size_t> chunk_limit() const {
    if (!decoder_)
      return std::nullopt;
    return decoder_->get_num_msyn_per_decode();
  }

  std::size_t num_observables() const {
    return graph_ ? graph_->get_num_observables()
                  : decoder_->get_num_observables();
  }

  /// Names this target in error messages.
  const char *kind() const { return graph_ ? "Task graph" : "Decoder"; }
  bool is_graph() const { return graph_ != nullptr; }

private:
  /// graph_'s device, or nullopt if there's no graph or it isn't GPU-affine.
  std::optional<int> graph_device() const {
    return graph_ ? graph_->cuda_device_id() : std::nullopt;
  }

  cudaq::qec::decoder *decoder_ = nullptr;
  cudaq::qec::decoding_task_graph *graph_ = nullptr;
};

} // namespace cudaq::qec::detail_dispatch
