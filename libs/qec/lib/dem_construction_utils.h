/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Internal helpers shared by dem_construction.cpp and extended_dem.cpp when
// building DEMs from css_code_matrices / css_noise_params. Not part of the
// public API; lives under lib/ so it is not installed with the public headers.

#pragma once

#include "cudaq/qec/code_matrices.h"
#include <cstddef>
#include <string>
#include <vector>

namespace cudaq::qec::detail {

using col_list = std::vector<std::vector<sparse_binary_matrix::index_type>>;

/// Determine n_qubits from the first matrix that has a nonzero column count.
/// Returns 0 when all matrices are default-constructed.
std::size_t resolve_num_qubits(const css_code_matrices &code);

/// Throw if m has nonzero rows but a column count that disagrees with n.
void check_num_cols(const sparse_binary_matrix &m, std::size_t n,
                    const char *label);

/// Throw if rates is non-empty with a length that does not match n.
/// size_label names the dimension n counts, so a per-check vector does not
/// report a per-qubit mismatch.
void check_rate_vector_size(const std::vector<double> &rates, std::size_t n,
                            const char *label, const char *size_label);

/// Throw if per_qubit is non-empty with a length that does not match n_qubits.
void check_per_qubit_size(const std::vector<double> &per_qubit, std::size_t n,
                          const char *label);

/// Throw unless p is a probability. Written as a negated range test so that
/// NaN, which compares false against everything, is rejected too.
void check_probability(double p, const std::string &label);

/// Throw unless every scalar and per-element rate is a probability. Zero rates
/// are legal and simply produce no DEM column; the point of this check is that
/// a negative or NaN rate is silently inactive under that same rule, so a
/// mistyped configuration would otherwise build a smaller DEM instead of
/// failing.
void validate_noise_rates(const css_noise_params &noise);

/// True when any noise rate is nonzero, i.e. the model asks for at least one
/// fault mechanism. Call only after validate_noise_rates(), so that a nonzero
/// rate here really is a positive probability.
bool has_any_noise(const css_noise_params &noise);

/// Nested CSC for m padded to outer size n; new entries are empty vectors.
col_list padded_nested_csc(const sparse_binary_matrix &m, std::size_t n);

/// Effective rate for qubit q: per_qubit[q] when set, else uniform.
double qubit_rate(double uniform, const std::vector<double> &per_qubit,
                  std::size_t q);

/// Qubit indices (ascending) whose effective rate is nonzero.
std::vector<std::size_t> active_qubits(double uniform,
                                       const std::vector<double> &per_qubit,
                                       std::size_t n);

/// Effective measurement error rate for check k (k indexes all checks,
/// Z-type first then X-type): pm_per_check[k] when set, else uniform pm.
double check_rate(double uniform_pm, const std::vector<double> &pm_per_check,
                  std::size_t k);

/// Check indices (ascending, 0..n_checks-1) with a nonzero effective rate.
std::vector<std::size_t> active_checks(double uniform_pm,
                                       const std::vector<double> &pm_per_check,
                                       std::size_t n_checks);

} // namespace cudaq::qec::detail
