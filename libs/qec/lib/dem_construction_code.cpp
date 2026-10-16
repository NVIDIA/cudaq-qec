/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Bridge between code objects and dem_from_css_matrices(). Extracts the
// four CSS generator matrices from a code object by calling get_parity_z(),
// get_parity_x(), get_observables_z(), and get_observables_x(), then
// delegates to the matrix-based construction path. The dense tensors
// returned by those methods are implicitly converted to sparse_binary_matrix
// via the non-explicit constructor (CSC layout, non-zeros treated as 1).

#include "cudaq/qec/dem_construction_code.h"

namespace cudaq::qec {

// Convert a dense tensor to sparse_binary_matrix, returning an empty
// (default-constructed) matrix when the tensor is not rank-2. Code methods
// such as get_parity_x() return a rank-0 tensor when the code has no
// stabilizers of that type (e.g. a Z-basis repetition code has no X checks).
static sparse_binary_matrix tensor_to_sparse(const cudaqx::tensor<uint8_t> &t) {
  if (t.rank() != 2)
    return sparse_binary_matrix{};
  return sparse_binary_matrix(t);
}

css_code_matrices css_matrices_from_code(const code &qec_code) {
  css_code_matrices m;
  m.hz = tensor_to_sparse(qec_code.get_parity_z());
  m.hx = tensor_to_sparse(qec_code.get_parity_x());
  m.lz = tensor_to_sparse(qec_code.get_observables_z());
  m.lx = tensor_to_sparse(qec_code.get_observables_x());
  return m;
}

detector_error_model dem_from_css_matrices(const code &qec_code,
                                           const css_noise_params &noise,
                                           std::size_t num_rounds) {
  return dem_from_css_matrices(css_matrices_from_code(qec_code), noise,
                               num_rounds);
}

} // namespace cudaq::qec
