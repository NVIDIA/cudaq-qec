/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace cudaq::qec::realtime {

/// @brief Convert a sparse vector (with -1 row separators) to CSR format.
///
/// The input format uses -1 as a row separator:
///   [col0, col1, -1, col2, col3, col4, -1, col5, -1]
/// represents a matrix with 3 rows where:
///   - row 0 has non-zeros at columns 0, 1
///   - row 1 has non-zeros at columns 2, 3, 4
///   - row 2 has non-zeros at column 5
///
/// @param sparse_vec Input sparse vector with -1 separators
/// @param row_ptr Output CSR row pointers (size: num_rows + 1)
/// @param col_idx Output CSR column indices
/// @return Number of rows
inline std::size_t
sparse_vec_to_csr(const std::vector<std::int64_t> &sparse_vec,
                  std::vector<uint32_t> &row_ptr,
                  std::vector<uint32_t> &col_idx) {

  row_ptr.clear();
  col_idx.clear();

  row_ptr.push_back(0);

  for (const auto &val : sparse_vec) {
    if (val == -1) {
      // End of row
      row_ptr.push_back(static_cast<uint32_t>(col_idx.size()));
    } else {
      // Column index
      col_idx.push_back(static_cast<uint32_t>(val));
    }
  }

  return row_ptr.size() - 1; // Number of rows
}

} // namespace cudaq::qec::realtime
