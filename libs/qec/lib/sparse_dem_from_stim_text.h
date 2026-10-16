/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/sparse_binary_matrix.h"
#include <string>
#include <vector>

// Library-private: shared by detector_error_model.cpp and decoder_init.cpp.
// Not installed and not exported, so it adds no plugin-visible API surface.
// Use dem_from_stim_text() for the public materialized model.
//
// One shared declaration rather than a hand-written one per translation unit:
// an ordinary function's return type is not part of its mangled name, so a
// return type that drifted between the definition and a local declaration
// would link cleanly and corrupt the returned object.

namespace cudaq::qec::details {

/// The sparse projection of a Stim DEM, in the layouts `decoder_init` stores.
/// Named fields rather than a tuple: H and O share a type, so positional
/// results would let them be swapped while still type-checking.
struct sparse_dem {
  /// H, detectors x error mechanisms, CSC (one compressed group per error).
  sparse_binary_matrix detector_error_matrix;
  /// O, observables x error mechanisms, CSR (one compressed group per
  /// observable).
  sparse_binary_matrix observables_flips_matrix;
  std::vector<double> error_rates;
};

/// Parse a Stim DEM straight into that sparse projection.
///
/// The parser already collects per-error hit lists, which are exactly H's
/// compressed columns, so this skips the dense intermediate entirely. That
/// matters at realistic sizes: a distance-13 model's dense H alone is ~98 MiB
/// while its sparse form is under 1 MiB, and a DEM-native decoder such as
/// Chromobius never reads the matrices at all.
///
/// Hidden explicitly: this library does not set CXX_VISIBILITY_PRESET, so a
/// non-inline symbol would otherwise reach the dynamic symbol table.
__attribute__((visibility("hidden"))) sparse_dem
sparse_dem_from_stim_text(const std::string &dem_text);

} // namespace cudaq::qec::details
