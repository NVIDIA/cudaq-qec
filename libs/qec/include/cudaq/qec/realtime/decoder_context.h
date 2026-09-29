/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace cudaq::qec::realtime {

/// @brief Base context structure for QEC decoder parameters.
///
/// This struct contains pointers to D and O matrices in CSR format,
/// along with dimension information needed for preprocessing and
/// postprocessing. Decoder implementations can extend this with additional
/// fields.
struct decoder_context_base {
  // D matrix (measurement -> detector) in CSR format
  const uint32_t *D_row_ptr =
      nullptr; ///< CSR row pointers (size: num_detectors + 1)
  const uint32_t *D_col_idx = nullptr; ///< CSR column indices

  // O matrix (decision -> observable) in CSR format
  const uint32_t *O_row_ptr =
      nullptr; ///< CSR row pointers (size: num_observables + 1)
  const uint32_t *O_col_idx = nullptr; ///< CSR column indices

  // Dimensions
  std::size_t num_measurements = 0; ///< Number of raw measurements per shot
  std::size_t num_detectors = 0;    ///< Number of detectors (syndrome bits)
  std::size_t num_observables = 0;  ///< Number of logical observables
  std::size_t num_edges = 0;        ///< Number of edges (decoder output size)

  // Soft-to-hard threshold
  double threshold = 0.5; ///< Threshold for converting soft to hard decisions
};

/// @brief Context for mock decoder with lookup table.
///
/// Extends decoder_context_base with pointers to pre-recorded expected answers.
struct mock_decoder_context : public decoder_context_base {
  // Lookup table: maps input measurements to expected corrections
  // Format: Each entry is (measurements..., corrections...)
  const uint8_t *lookup_measurements =
      nullptr; ///< Input measurements for lookup
  const uint8_t *lookup_corrections =
      nullptr;                        ///< Expected corrections for lookup
  std::size_t num_lookup_entries = 0; ///< Number of entries in lookup table
};

} // namespace cudaq::qec::realtime
