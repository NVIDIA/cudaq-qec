/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuda-qx/core/tensor.h"
#include <cstdint>
#include <vector>

namespace cudaq::qec {

/// @brief Storage layout for the sparse PCM: Compressed Sparse Column (CSC)
/// or Compressed Sparse Row (CSR). All non-zero entries are assumed to be 1;
/// values are not stored.
enum class sparse_binary_matrix_layout { csc, csr };

/// @brief Sparse parity-check matrix in either CSC or CSR form.
///
/// Input index lists are stored as given: not required to be sorted or
/// GF(2)-unique. Consumers that require cuSPARSE-style compressed groups can
/// call `validate_sorted_unique_indices`; consumers that need GF(2)-collapsed
/// per-group indices can call `canonicalize` on entry.
///
/// `index_type` is `uint32_t`, so each dimension and `nnz` must fit in
/// `~4×10^9`.
class sparse_binary_matrix {
public:
  using index_type = std::uint32_t;

  /// @brief Construct a sparse PCM in CSC form.
  /// @param num_rows Number of rows.
  /// @param num_cols Number of columns.
  /// @param col_ptrs Column pointer array (length num_cols + 1); column j has
  /// indices in \p row_indices[col_ptrs[j] .. col_ptrs[j+1]-1].
  /// @param row_indices Row indices of non-zeros (length nnz).
  static sparse_binary_matrix from_csc(index_type num_rows, index_type num_cols,
                                       std::vector<index_type> col_ptrs,
                                       std::vector<index_type> row_indices);

  /// @brief Construct a sparse PCM in CSR form.
  /// @param num_rows Number of rows.
  /// @param num_cols Number of columns.
  /// @param row_ptrs Row pointer array (length num_rows + 1); row i has
  /// indices in \p col_indices[row_ptrs[i] .. row_ptrs[i+1]-1].
  /// @param col_indices Column indices of non-zeros (length nnz).
  static sparse_binary_matrix from_csr(index_type num_rows, index_type num_cols,
                                       std::vector<index_type> row_ptrs,
                                       std::vector<index_type> col_indices);

  /// @brief Construct from nested CSC: \p nested[j] is the list of row indices
  /// for column j; \p nested.size() must equal \p num_cols.
  static sparse_binary_matrix
  from_nested_csc(index_type num_rows, index_type num_cols,
                  const std::vector<std::vector<index_type>> &nested);

  /// @brief Construct from nested CSR: \p nested[i] is the list of column
  /// indices for row i; \p nested.size() must equal \p num_rows.
  static sparse_binary_matrix
  from_nested_csr(index_type num_rows, index_type num_cols,
                  const std::vector<std::vector<index_type>> &nested);

  /// @brief Construct from a rank-2 dense PCM (any non-zero treated as 1).
  /// Intentionally not `explicit` so call sites that take
  /// `sparse_binary_matrix` accept a dense `cudaqx::tensor` unchanged.
  sparse_binary_matrix(
      const cudaqx::tensor<std::uint8_t> &dense,
      sparse_binary_matrix_layout layout = sparse_binary_matrix_layout::csc);

  sparse_binary_matrix() = default;
  sparse_binary_matrix(const sparse_binary_matrix &) = default;
  sparse_binary_matrix(sparse_binary_matrix &&) noexcept = default;
  sparse_binary_matrix &operator=(const sparse_binary_matrix &) = default;
  sparse_binary_matrix &operator=(sparse_binary_matrix &&) noexcept = default;

  sparse_binary_matrix_layout layout() const { return layout_; }
  index_type num_rows() const { return num_rows_; }
  index_type num_cols() const { return num_cols_; }
  index_type num_nnz() const {
    return static_cast<index_type>(indices_.size());
  }

  /// @brief For CSC: ptr has length num_cols+1; for CSR: ptr has length
  /// num_rows+1.
  const std::vector<index_type> &ptr() const { return ptr_; }
  /// @brief For CSC: row indices; for CSR: column indices.
  const std::vector<index_type> &indices() const { return indices_; }

  /// @brief Throw if each compressed column/row does not have strictly
  /// increasing indices. This rejects duplicate entries in the stored layout.
  void validate_sorted_unique_indices(
      const char *context = "sparse_binary_matrix") const;

  /// @brief Return a GF(2)-canonical copy: each compressed column/row has its
  /// indices sorted ascending, and duplicate indices are XOR-merged. An index
  /// that appears k times is kept iff k is odd. The output has the same layout
  /// as the input and is idempotent under further `canonicalize` calls.
  ///
  /// Use this when a PCM source, such as a DEM decomposition, may emit
  /// duplicate indices within a compressed group and the caller wants GF(2)
  /// duplicate-collapse semantics before passing the matrix to consumers that
  /// require at most one entry per row/column.
  sparse_binary_matrix canonicalize() const;

  /// @brief Return a copy of this matrix in CSC layout. No-op if already CSC.
  sparse_binary_matrix to_csc() const;

  /// @brief Return a copy of this matrix in CSR layout. No-op if already CSR.
  sparse_binary_matrix to_csr() const;

  /// @brief Convert to a dense PCM tensor (rows x columns). Non-zero entries
  /// are set to 1.
  cudaqx::tensor<std::uint8_t> to_dense() const;

  /// @brief Nested CSC: outer vector has size num_cols; inner vector for
  /// column j lists row indices of non-zeros in that column.
  std::vector<std::vector<index_type>> to_nested_csc() const;

  /// @brief Nested CSR: outer vector has size num_rows; inner vector for row i
  /// lists column indices of non-zeros in that row.
  std::vector<std::vector<index_type>> to_nested_csr() const;

private:
  sparse_binary_matrix(sparse_binary_matrix_layout layout, index_type num_rows,
                       index_type num_cols, std::vector<index_type> ptr,
                       std::vector<index_type> indices);

  sparse_binary_matrix_layout layout_ = sparse_binary_matrix_layout::csc;
  index_type num_rows_ = 0;
  index_type num_cols_ = 0;
  std::vector<index_type> ptr_;
  std::vector<index_type> indices_;
};

} // namespace cudaq::qec
