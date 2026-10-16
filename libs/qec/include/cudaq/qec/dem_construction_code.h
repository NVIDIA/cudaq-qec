/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file declares convenience functions that bridge the code base class
// to css_code_matrices and dem_from_css_matrices(). They live in cudaq-qec
// (not cudaq-qec-decoders) because they depend on the code base class which
// requires the CUDA-Q framework. Callers working purely from raw matrix data
// should use dem_construction.h directly.

#pragma once

#include "cudaq/qec/code.h"
#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_construction.h"

namespace cudaq::qec {

/// @brief Extract CSS generator matrices from a code object.
///
/// Converts the four parity/observable tensors returned by the code object
/// into a css_code_matrices suitable for direct use with
/// dem_from_css_matrices(). The conversion uses the implicit
/// sparse_binary_matrix(tensor<uint8_t>) constructor (CSC layout).
///
/// @param qec_code A constructed CSS code object (repetition, surface, etc.).
/// @return css_code_matrices populated from the code's parity and logical
///         operator matrices.
css_code_matrices css_matrices_from_code(const code &qec_code);

/// @brief Build a T-round code-capacity DEM directly from a code object.
///
/// Convenience wrapper equivalent to:
///   dem_from_css_matrices(css_matrices_from_code(qec_code), noise, num_rounds)
///
/// @param qec_code   A constructed CSS code object.
/// @param noise      Per-qubit depolarizing noise parameters.
/// @param num_rounds Number of syndrome measurement rounds T (default 1).
/// @return           A T-round code-capacity DEM. See
///                   dem_from_css_matrices(css_code_matrices, css_noise_params,
///                   std::size_t) for full semantics.
detector_error_model dem_from_css_matrices(const code &qec_code,
                                           const css_noise_params &noise,
                                           std::size_t num_rounds = 1);

} // namespace cudaq::qec
