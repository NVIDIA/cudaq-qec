/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuda-qx/core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// A detector error model (DEM) for a quantum error correction circuit. A
/// DEM can be created from a QEC circuit and a noise model. It contains
/// information about which errors flip which detectors. This is used by the
/// decoder to help make predictions about observables flips.
///
/// Shared size parameters among the matrix types.
/// - `detector_error_matrix`: num_detectors x num_error_mechanisms [d, e]
/// - `error_rates`: num_error_mechanisms
/// - `observables_flips_matrix`: num_observables x num_error_mechanisms [k, e]
///
/// @note The C++ API for this class may change in the future. The Python API is
/// more likely to be backwards compatible.
struct detector_error_model {
  /// The detector error matrix is a specific kind of circuit-level parity-check
  /// matrix where each row represents a detector, and each column represents
  /// an error mechanism. The entries of this matrix are H[i,j] = 1 if detector
  /// i is triggered by error mechanism j, and 0 otherwise.
  cudaqx::tensor<uint8_t> detector_error_matrix;

  /// The list of weights has length equal to the number of columns of
  /// `detector_error_matrix`, which assigns a likelihood to each error
  /// mechanism.
  std::vector<double> error_rates;

  /// The observables flips matrix is a specific kind of circuit-level parity-
  /// check matrix where each row represents a Pauli observable, and each
  /// column represents an error mechanism. The entries of this matrix are
  /// O[i,j] = 1 if Pauli observable i is flipped by error mechanism j, and 0
  /// otherwise.
  cudaqx::tensor<uint8_t> observables_flips_matrix;

  /// Error mechanism ID. From a probability perspective, each error mechanism
  /// ID is independent of all other error mechanism ID. For all errors with
  /// the *same* ID, only one of them can happen. That is - the errors
  /// containing the same ID are correlated with each other.
  std::optional<std::vector<std::size_t>> error_ids;

  /// Return the number of rows in the detector_error_matrix.
  std::size_t num_detectors() const;

  /// Return the number of columns in the detector_error_matrix, error_rates,
  /// and observables_flips_matrix.
  std::size_t num_error_mechanisms() const;

  /// Return the number of rows in the observables_flips_matrix.
  std::size_t num_observables() const;

  /// @brief Put the detector_error_matrix into canonical form for round-based
  /// decoding: topologically order the columns and merge columns that share the
  /// same detector AND observable signature, composing their rates so the
  /// canonicalized model matches the input.
  /// @param num_syndromes_per_round Number of syndromes per round, used to
  /// order columns by the rounds their detectors span; 0 disables the per-round
  /// key.
  /// @param remove_zero_syndrome_errors If false (default), zero-syndrome
  /// columns that still flip an observable (undetectable logical errors) are
  /// retained so the observable-flip probability is preserved; if true, all
  /// columns with no detector signature are dropped (appropriate when the DEM
  /// is consumed only for round-based decoding).
  /// @note Canonicalization does not preserve cross-column exclusivity
  /// structure: each output column is given a fresh unique error id and is
  /// treated as independent, so any `error_ids` correlation in the input model
  /// is discarded.
  void canonicalize_for_rounds(uint32_t num_syndromes_per_round,
                               bool remove_zero_syndrome_errors = false);

  /// @brief Boundary-aware variant of canonicalize_for_rounds for
  /// memory-experiment DEMs whose first and last detector layers (the
  /// boundaries) are narrower than the interior layers
  /// @param num_syndromes_per_round Interior-layer width (syndromes per
  /// interior round).
  /// @param num_boundary_syndromes Width of the leading and trailing boundary
  /// layers: the first and last @p num_boundary_syndromes rows form the
  /// boundary rounds and each intermediate block of @p num_syndromes_per_round
  /// rows is an interior round.
  /// @param remove_zero_syndrome_errors Same meaning as in the scalar overload.
  /// @throws std::invalid_argument if @p num_syndromes_per_round is 0 or
  /// @p num_boundary_syndromes > @p num_syndromes_per_round.
  void canonicalize_for_rounds_with_boundary(uint32_t num_syndromes_per_round,
                                             uint32_t num_boundary_syndromes,
                                             bool remove_zero_syndrome_errors);

private:
  /// Shared implementation of the canonicalize_for_rounds overloads. Given the
  /// sparse detector matrix and a precomputed topological @p column_order,
  /// merges columns sharing a full (detector, observable) signature and
  /// reorders/reduces the matrices accordingly.
  void canonicalize_for_rounds_impl(
      const std::vector<std::vector<std::uint32_t>> &row_indices,
      const std::vector<std::uint32_t> &column_order,
      bool remove_zero_syndrome_errors);
};

/// Parse the Stim DEM string @p dem_text into detector/observable flip
/// matrices and error rates. DEM-native decoders should consume raw DEM text
/// instead. By default (@p use_decomp_suggestions = false) the '^' separators
/// are ignored and each error instruction produces a single column. If
/// @p use_decomp_suggestions is true, error mechanisms that carry an explicit
/// graphlike decomposition (components separated by '^') are expanded into one
/// column per component, each inheriting the probability of the parent
/// instruction. @p error_ids is always left as nullopt. Note that this is a
/// lossy approximation of the original DEM.
detector_error_model dem_from_stim_text(const std::string &dem_text,
                                        bool use_decomp_suggestions = false);

} // namespace cudaq::qec
