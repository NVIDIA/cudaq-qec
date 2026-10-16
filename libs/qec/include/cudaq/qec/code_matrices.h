/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file defines the css_code_matrices and css_noise_params types used
// to describe a Calderbank-Shor-Steane (CSS) quantum error-correcting code
// in terms of its sparse binary generator matrices and a depolarizing noise
// model. These types serve as the primary input interface for
// dem_from_css_matrices() and are intended to be derived from a code object
// via css_matrices_from_code(), or constructed directly from raw matrix data.

#pragma once

#include "cudaq/qec/sparse_binary_matrix.h"
#include <vector>

namespace cudaq::qec {

/// @brief CSS code generator matrices: parity-check and logical-operator
/// matrices for a Calderbank-Shor-Steane (CSS) quantum error-correcting code.
///
/// All matrices are sparse binary (GF(2)). Each matrix column corresponds to
/// one data qubit, so every non-empty matrix must share the same num_cols()
/// value (the number of data qubits, n). A matrix with zero rows is a valid
/// empty block; its num_cols() may be 0 (default-constructed) or n.
///
/// Row semantics:
///   hz  — Z-type stabilizer generators [n_z_checks x n_qubits].
///          Row i is the support of the i-th Z-stabilizer generator.
///          A Z-stabilizer anticommutes with X errors, so hz[:,q] gives
///          the set of Z-checks triggered by an X error on qubit q.
///   hx  — X-type stabilizer generators [n_x_checks x n_qubits].
///          Detects Z errors symmetrically; hx[:,q] gives the X-checks
///          triggered by a Z error on qubit q.
///   lz  — Z-type logical operators [k x n_qubits].
///          lz[r, q] = 1 iff the r-th Z logical observable is flipped
///          by an X error on qubit q.
///   lx  — X-type logical operators [k x n_qubits].
///          lx[r, q] = 1 iff the r-th X logical observable is flipped
///          by a Z error on qubit q.
struct css_code_matrices {
  sparse_binary_matrix hz; ///< Z stabilizers [n_z_checks x n_qubits]
  sparse_binary_matrix hx; ///< X stabilizers [n_x_checks x n_qubits]
  sparse_binary_matrix lz; ///< Z logical operators [k x n_qubits]
  sparse_binary_matrix lx; ///< X logical operators [k x n_qubits]
};

/// @brief Noise parameters for the phenomenological noise model.
///
/// Extends the code-capacity (data-qubit-only) model with syndrome
/// measurement errors, which are the most important circuit-level effect
/// for CSS codes: each stabilizer measurement can produce the wrong outcome
/// with probability pm, creating a time-local detector event that spans
/// two consecutive rounds without flipping any logical observable.
///
/// Scalar rates (px, py, pz, pm) apply uniformly. Per-element override
/// vectors take priority when non-empty; their length must equal n_qubits
/// (for data-qubit rates) or n_checks = hz.num_rows() + hx.num_rows()
/// (for pm_per_check, Z-checks first then X-checks). Elements with
/// effective rate zero produce no DEM column.
struct css_noise_params {
  double px = 0.0; ///< Uniform X data-qubit rate; overridden by px_per_qubit
  double py = 0.0; ///< Uniform Y data-qubit rate; overridden by py_per_qubit
  double pz = 0.0; ///< Uniform Z data-qubit rate; overridden by pz_per_qubit
  double pm = 0.0; ///< Uniform syndrome measurement error rate per check per
                   ///< round; overridden by pm_per_check if set

  /// Per-qubit X rates. If non-empty, length must equal n_qubits.
  std::vector<double> px_per_qubit;
  /// Per-qubit Y rates. If non-empty, length must equal n_qubits.
  std::vector<double> py_per_qubit;
  /// Per-qubit Z rates. If non-empty, length must equal n_qubits.
  std::vector<double> pz_per_qubit;
  /// Per-check measurement error rates. If non-empty, length must equal
  /// n_checks (= hz.num_rows() + hx.num_rows()); Z-checks first.
  std::vector<double> pm_per_check;
};

} // namespace cudaq::qec
