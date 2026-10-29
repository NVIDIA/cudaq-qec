/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <vector>

// Library-private sparse closure shared by decoder preparation and both
// window implementations. Observable rows stay uncompressed so column-list
// consumers need no intermediate CSR matrix.
namespace cudaq::qec::details {

struct sparse_chunk_model {
  sparse_binary_matrix detector_error_matrix; ///< Canonical CSC.
  std::vector<std::vector<sparse_binary_matrix::index_type>> observable_rows;
  std::vector<double> error_rates;
};

/// Close a validated chunk chain without dense matrices or column reordering.
/// Graph-only projections omit priors when a native decoder takes the recipe.
inline sparse_chunk_model
sparse_dem_from_chunks(const std::vector<extended_dem> &chunks,
                       seam_id from_seam, seam_id to_seam,
                       bool with_priors = true) {
  sparse_chunk_model model;
  model.detector_error_matrix = dem_chunks_to_pcm(chunks, from_seam, to_seam);
  model.observable_rows = dem_chunks_to_o_sparse(chunks);
  if (with_priors) {
    model.error_rates.reserve(model.detector_error_matrix.num_cols());
    for (const auto &chunk : chunks)
      model.error_rates.insert(model.error_rates.end(),
                               chunk.error_rates.begin(),
                               chunk.error_rates.end());
  }
  return model;
}

} // namespace cudaq::qec::details
