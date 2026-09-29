/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Memory-circuit-specific DEM function: dem_chunks_to_d_sparse.
//
// Encodes the XOR-of-consecutive-rounds detector convention used by CSS
// memory experiments. Not valid for circuits whose detectors are not pairs
// of adjacent syndrome rounds.

#include "cudaq/qec/dem_chunks_memory.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// D_sparse[det_id] = measurement bit positions that XOR-combine to form that
// detector under the memory-experiment convention:
//   det k       (r=0): {k}               — vs. zero initial state
//   det r*d+k (r>0): {(r-1)*d+k, r*d+k} — consecutive syndrome XOR
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const std::vector<extended_dem> &dem_chunks,
                       seam_id from_seam, seam_id to_seam) {
  if (dem_chunks.empty())
    throw std::invalid_argument(
        "dem_chunks_to_d_sparse: dem_chunks must be non-empty");

  uint32_t d = 0;
  if (dem_chunks[0].has_seam(to_seam))
    d = dem_chunks[0].get_seam(to_seam).num_rows();
  if (d == 0 && dem_chunks[0].has_seam(from_seam))
    d = dem_chunks[0].get_seam(from_seam).num_rows();
  if (d == 0)
    throw std::invalid_argument(
        "dem_chunks_to_d_sparse: chunk 0 has no seam rows on either side, "
        "so its rounds cannot be counted");

  // dem_chunks_to_rounds validates seam widths and contractibility.
  const std::size_t T = dem_chunks_to_rounds(dem_chunks, from_seam, to_seam);
  const std::size_t n_det = T * static_cast<std::size_t>(d);

  if (T > 0) {
    const auto max_bit = static_cast<std::size_t>(T - 1) * d + (d - 1);
    if (max_bit > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error(
          "dem_chunks_to_d_sparse: measurement bit index exceeds uint32_t max");
  }

  std::vector<std::vector<uint32_t>> d_sparse(n_det);
  for (std::size_t r = 0; r < T; ++r) {
    for (uint32_t k = 0; k < d; ++k) {
      const std::size_t det = r * d + k;
      if (r == 0) {
        d_sparse[det] = {k};
      } else {
        d_sparse[det] = {static_cast<uint32_t>((r - 1) * d + k),
                         static_cast<uint32_t>(r * d + k)};
      }
    }
  } // end - for(r)
  return d_sparse;
} // end - dem_chunks_to_d_sparse()

} // namespace cudaq::qec
