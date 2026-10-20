/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Nominal I/O types for the decoder task graph.
//
// These are thin value types wrapping the arrays that flow between graph nodes,
// plus the metadata (tag, converged, opt_results) that the plain array-based
// realtime API cannot express.
//
// Storage note: all bit/result fields are currently std::vector (host memory).
// This is sufficient while graph nodes (apply_d_sparse, apply_o_sparse) run on
// CPU, since GPU decoders manage their own H→D copies inside decode().  When
// GPU-accelerated node execution is added, these fields should become
// cudaqx::tensor<T> so that nodes can write directly into pinned/device memory
// and avoid an extra copy on the hot path.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/decoder.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <typeinfo>
#include <vector>

namespace cudaq::qec {

// ---------------------------------------------------------------------------
// I/O types
// ---------------------------------------------------------------------------

/// Raw per-shot measurement record from the QPU.
struct measurement_results {
  std::vector<uint8_t> bits; ///< length = num_measurements

  /// Round tag identifying the shot, to be forwarded from the tag the decoder
  /// server receives with enqueue_syndromes. Not yet populated by the graph.
  uint64_t tag = 0;
};

/// Marker carried by every node's implicit `done` output, which fires when the
/// node has completed execution. It lets an edge
/// express "after this node" without an explicit data dependence.
struct completion {};

/// Aggregated detection events: s = D * m (mod 2).
/// Stored as float_t so it can be passed directly to decoder::decode().
struct detection_events {
  std::vector<float_t> bits; ///< length = num_detectors
};

/// Predicted error vector from a decoder.
struct error_pattern {
  bool converged = false;
  std::vector<float_t> result; ///< length = num_error_mechanisms

  /// Plugin-specific diagnostics, to be forwarded from decoder_result.
  std::optional<cudaqx::heterogeneous_map> opt_results;
};

/// Final observable correction. The role (L_raw, L_flip, or L_obs) is carried
/// by the graph topology, not the type.
struct logical_outcome {
  std::vector<uint8_t> bits; ///< length = num_observables
  bool converged = false;    ///< forwarded from error_pattern
};

// ---------------------------------------------------------------------------
// Edge-value type identity
// ---------------------------------------------------------------------------

/// Identity of an edge-value type: the C++ payload type.  Callers obtain one
/// from a type via of<T>() or from a registered name; several names may refer
/// to one type, and ports declared through any of them wire together.
class edge_value_type {
public:
  /// The identity of payload type `T`.
  template <typename T>
  static constexpr edge_value_type of() {
    return edge_value_type(&typeid(T));
  }

  /// The type registered under `name`.  Throws std::invalid_argument when no
  /// such name is registered.
  __attribute__((visibility("default"))) explicit edge_value_type(
      std::string_view name);

  bool operator==(const edge_value_type &other) const noexcept {
    return *info_ == *other.info_;
  }
  bool operator!=(const edge_value_type &other) const noexcept {
    return !(*this == other);
  }

  const std::type_info &info() const noexcept { return *info_; }

private:
  constexpr explicit edge_value_type(const std::type_info *info)
      : info_(info) {}
  const std::type_info *info_;
};

/// Ids of the built-in payloads above.
inline constexpr edge_value_type measurement_results_type =
    edge_value_type::of<measurement_results>();
inline constexpr edge_value_type detection_events_type =
    edge_value_type::of<detection_events>();
inline constexpr edge_value_type logical_outcome_type =
    edge_value_type::of<logical_outcome>();
inline constexpr edge_value_type completion_type =
    edge_value_type::of<completion>();

// ---------------------------------------------------------------------------
// Primitive task functions
// ---------------------------------------------------------------------------

/// Dets: aggregate measurements into detection events via D_sparse (XOR).
/// d_sparse[i] lists the measurement indices that XOR to form detector i.
detection_events
apply_d_sparse(const measurement_results &m,
               const std::vector<std::vector<uint32_t>> &d_sparse);

/// Obs_flips: project an error pattern onto observable flips via O_sparse.
/// o_sparse[i] lists the fault-column indices whose flip toggles observable i.
/// Assumes the error pattern is in decode_to_errs form (result per fault
/// column).
logical_outcome
apply_o_sparse(const error_pattern &e,
               const std::vector<std::vector<uint32_t>> &o_sparse);

/// GF(2) XOR of two logical outcomes (L_raw ⊕ L_flip → L_obs).
logical_outcome xor_logical(const logical_outcome &a, const logical_outcome &b);

} // namespace cudaq::qec

///@cond INTERNAL
template <>
struct std::hash<cudaq::qec::edge_value_type> {
  std::size_t operator()(cudaq::qec::edge_value_type t) const noexcept {
    return t.info().hash_code();
  }
};
///@endcond
