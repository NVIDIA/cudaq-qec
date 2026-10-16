/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/sparse_binary_matrix.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

namespace {

/// CSC/CSR pointers and per-column/per-row runs use `index_type` (uint32_t);
/// the total nnz must fit in that type (std::vector::size may exceed it).
inline void throw_if_nnz_exceeds_index_range(std::size_t nnz,
                                             const char *context) {
  if (nnz > static_cast<std::size_t>(
                std::numeric_limits<sparse_binary_matrix::index_type>::max()))
    throw std::invalid_argument(std::string(context) +
                                ": nnz exceeds index_type (uint32_t) range");
}

} // namespace

sparse_binary_matrix::sparse_binary_matrix(sparse_binary_matrix_layout layout,
                                           index_type num_rows,
                                           index_type num_cols,
                                           std::vector<index_type> ptr,
                                           std::vector<index_type> indices)
    : layout_(layout), num_rows_(num_rows), num_cols_(num_cols),
      ptr_(std::move(ptr)), indices_(std::move(indices)) {}

void sparse_binary_matrix::validate_sorted_unique_indices(
    const char *context) const {
  const std::string where =
      context == nullptr ? "sparse_binary_matrix" : context;
  const bool is_csc = layout_ == sparse_binary_matrix_layout::csc;
  const std::size_t num_groups = is_csc ? static_cast<std::size_t>(num_cols_)
                                        : static_cast<std::size_t>(num_rows_);
  const auto index_bound = is_csc ? num_rows_ : num_cols_;

  if (num_groups == 0 && ptr_.empty() && indices_.empty())
    return;

  if (ptr_.size() != num_groups + 1) {
    throw std::invalid_argument(where +
                                ": sparse pointer array has invalid size");
  }
  if (!ptr_.empty() && ptr_.front() != 0U) {
    throw std::invalid_argument(where +
                                ": sparse pointer array must start at zero");
  }
  if (!ptr_.empty() &&
      static_cast<std::size_t>(ptr_.back()) != indices_.size()) {
    throw std::invalid_argument(where +
                                ": sparse pointer array must end at nnz");
  }

  const char *group_name = is_csc ? "column" : "row";
  for (std::size_t group = 0; group < num_groups; ++group) {
    const std::size_t begin = ptr_[group];
    const std::size_t end = ptr_[group + 1];
    if (begin > end || end > indices_.size()) {
      throw std::invalid_argument(where +
                                  ": sparse pointer array is not monotonic");
    }
    for (std::size_t p = begin; p < end; ++p) {
      const index_type idx = indices_[p];
      if (idx >= index_bound) {
        throw std::invalid_argument(where + ": sparse index out of range");
      }
      if (p > begin && indices_[p - 1] >= idx) {
        throw std::invalid_argument(
            where +
            ": sparse indices must be strictly increasing within each " +
            group_name);
      }
    }
  }
}

sparse_binary_matrix sparse_binary_matrix::canonicalize() const {
  // Sort each per-group index list, then GF(2)-collapse equal-value runs
  // (keep one iff the run length is odd).
  const bool is_csc = layout_ == sparse_binary_matrix_layout::csc;
  // num_groups from dims (not ptr_.size()) so default-constructed matrices
  // still produce a num_groups+1 ptr vector.
  const std::size_t num_groups = is_csc ? static_cast<std::size_t>(num_cols_)
                                        : static_cast<std::size_t>(num_rows_);

  std::vector<index_type> new_ptr(num_groups + 1, 0);
  std::vector<index_type> new_idx;
  new_idx.reserve(indices_.size());

  std::vector<index_type> scratch;
  const bool ptr_is_valid = ptr_.size() == num_groups + 1;
  if (ptr_is_valid) {
    std::size_t max_group_size = 0;
    for (std::size_t g = 0; g < num_groups; ++g)
      max_group_size =
          std::max<std::size_t>(max_group_size, ptr_[g + 1] - ptr_[g]);
    scratch.reserve(max_group_size);
  }

  for (std::size_t g = 0; g < num_groups && ptr_is_valid; ++g) {
    const auto begin = ptr_[g];
    const auto end = ptr_[g + 1];
    scratch.assign(indices_.begin() + begin, indices_.begin() + end);
    std::sort(scratch.begin(), scratch.end());
    auto read = scratch.begin();
    while (read != scratch.end()) {
      const index_type val = *read;
      auto run_end = read;
      while (run_end != scratch.end() && *run_end == val)
        ++run_end;
      if (((run_end - read) & 1) != 0)
        new_idx.push_back(val);
      read = run_end;
    }
    new_ptr[g + 1] = static_cast<index_type>(new_idx.size());
  }
  new_idx.shrink_to_fit();

  return is_csc ? sparse_binary_matrix::from_csc(num_rows_, num_cols_,
                                                 std::move(new_ptr),
                                                 std::move(new_idx))
                : sparse_binary_matrix::from_csr(num_rows_, num_cols_,
                                                 std::move(new_ptr),
                                                 std::move(new_idx));
}

sparse_binary_matrix
sparse_binary_matrix::from_csc(index_type num_rows, index_type num_cols,
                               std::vector<index_type> col_ptrs,
                               std::vector<index_type> row_indices) {
  const auto expected_ptr = static_cast<std::size_t>(num_cols) + 1;
  if (col_ptrs.size() != expected_ptr) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csc: col_ptrs must have size num_cols + 1");
  }
  if (col_ptrs.front() != 0U) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csc: col_ptrs[0] must be zero");
  }
  // size_t guards j from wrapping when num_cols == UINT32_MAX.
  for (std::size_t j = 1; j <= num_cols; ++j) {
    if (col_ptrs[j - 1] > col_ptrs[j]) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csc: col_ptrs must be non-decreasing");
    }
  }
  throw_if_nnz_exceeds_index_range(row_indices.size(),
                                   "sparse_binary_matrix::from_csc");
  const auto nnz = static_cast<index_type>(row_indices.size());
  if (col_ptrs.back() != nnz) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csc: last col_ptr must equal nnz "
        "(row_indices.size())");
  }
  for (index_type idx : row_indices) {
    if (idx >= num_rows) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csc: row index out of range");
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows,
                              num_cols, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix
sparse_binary_matrix::from_csr(index_type num_rows, index_type num_cols,
                               std::vector<index_type> row_ptrs,
                               std::vector<index_type> col_indices) {
  const auto expected_ptr = static_cast<std::size_t>(num_rows) + 1;
  if (row_ptrs.size() != expected_ptr) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csr: row_ptrs must have size num_rows + 1");
  }
  if (row_ptrs.front() != 0U) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csr: row_ptrs[0] must be zero");
  }
  for (std::size_t i = 1; i <= num_rows; ++i) { // see from_csc re: size_t.
    if (row_ptrs[i - 1] > row_ptrs[i]) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csr: row_ptrs must be non-decreasing");
    }
  }
  throw_if_nnz_exceeds_index_range(col_indices.size(),
                                   "sparse_binary_matrix::from_csr");
  const auto nnz = static_cast<index_type>(col_indices.size());
  if (row_ptrs.back() != nnz) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csr: last row_ptr must equal nnz "
        "(col_indices.size())");
  }
  for (index_type idx : col_indices) {
    if (idx >= num_cols) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csr: column index out of range");
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows,
                              num_cols, std::move(row_ptrs),
                              std::move(col_indices));
}

sparse_binary_matrix sparse_binary_matrix::from_nested_csc(
    index_type num_rows, index_type num_cols,
    const std::vector<std::vector<index_type>> &nested) {
  if (nested.size() != static_cast<std::size_t>(num_cols)) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_nested_csc: nested.size() must equal "
        "num_cols");
  }
  std::vector<index_type> col_ptrs(num_cols + 1);
  col_ptrs[0] = 0;
  std::size_t nnz_accum = 0;
  for (const auto &col : nested)
    nnz_accum += col.size();
  throw_if_nnz_exceeds_index_range(nnz_accum,
                                   "sparse_binary_matrix::from_nested_csc");
  std::vector<index_type> row_indices;
  row_indices.reserve(nnz_accum);
  for (index_type j = 0; j < num_cols; ++j) {
    for (index_type r : nested[j]) {
      if (r >= num_rows) {
        throw std::invalid_argument(
            "sparse_binary_matrix::from_nested_csc: row index out of range");
      }
      row_indices.push_back(r);
    }
    col_ptrs[j + 1] = static_cast<index_type>(row_indices.size());
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows,
                              num_cols, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix sparse_binary_matrix::from_nested_csr(
    index_type num_rows, index_type num_cols,
    const std::vector<std::vector<index_type>> &nested) {
  if (nested.size() != static_cast<std::size_t>(num_rows)) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_nested_csr: nested.size() must equal "
        "num_rows");
  }
  std::vector<index_type> row_ptrs(num_rows + 1);
  row_ptrs[0] = 0;
  std::size_t nnz_accum = 0;
  for (const auto &rw : nested)
    nnz_accum += rw.size();
  throw_if_nnz_exceeds_index_range(nnz_accum,
                                   "sparse_binary_matrix::from_nested_csr");
  std::vector<index_type> col_indices;
  col_indices.reserve(nnz_accum);
  for (index_type i = 0; i < num_rows; ++i) {
    for (index_type c : nested[i]) {
      if (c >= num_cols) {
        throw std::invalid_argument(
            "sparse_binary_matrix::from_nested_csr: column index out of range");
      }
      col_indices.push_back(c);
    }
    row_ptrs[i + 1] = static_cast<index_type>(col_indices.size());
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows,
                              num_cols, std::move(row_ptrs),
                              std::move(col_indices));
}

sparse_binary_matrix::sparse_binary_matrix(
    const cudaqx::tensor<std::uint8_t> &dense,
    sparse_binary_matrix_layout layout) {
  if (dense.rank() != 2) {
    throw std::invalid_argument(
        "sparse_binary_matrix: dense PCM tensor must have rank 2");
  }
  const std::size_t nrows = dense.shape()[0];
  const std::size_t ncols = dense.shape()[1];
  if (nrows >
          static_cast<std::size_t>(std::numeric_limits<index_type>::max()) ||
      ncols >
          static_cast<std::size_t>(std::numeric_limits<index_type>::max())) {
    throw std::invalid_argument(
        "sparse_binary_matrix: dense PCM dimensions exceed index_type range");
  }
  num_rows_ = static_cast<index_type>(nrows);
  num_cols_ = static_cast<index_type>(ncols);
  layout_ = layout;

  // Degenerate shape: skip the loops below — `&dense.at({r, 0})` is OOB
  // when num_cols_ == 0.
  if (num_rows_ == 0 || num_cols_ == 0) {
    const std::size_t num_groups = (layout_ == sparse_binary_matrix_layout::csc)
                                       ? static_cast<std::size_t>(num_cols_)
                                       : static_cast<std::size_t>(num_rows_);
    ptr_.assign(num_groups + 1, 0);
    return;
  }

  // size_t loops + `static_cast<size_t>(dim) + 1` ptr sizing avoid UINT32_MAX
  // wraparound; see from_csc.
  if (layout_ == sparse_binary_matrix_layout::csc) {
    std::vector<index_type> col_nnz(num_cols_, 0);
    for (std::size_t r = 0; r < num_rows_; ++r) {
      const auto *row_ptr = &dense.at({r, std::size_t{0}});
      for (std::size_t c = 0; c < num_cols_; ++c)
        if (row_ptr[c])
          ++col_nnz[c];
    }
    std::size_t total_nnz = 0;
    for (std::size_t c = 0; c < num_cols_; ++c)
      total_nnz += static_cast<std::size_t>(col_nnz[c]);
    throw_if_nnz_exceeds_index_range(total_nnz,
                                     "sparse_binary_matrix(dense,to CSC)");
    ptr_.resize(static_cast<std::size_t>(num_cols_) + 1);
    ptr_[0] = 0;
    for (std::size_t c = 0; c < num_cols_; ++c)
      ptr_[c + 1] = ptr_[c] + col_nnz[c];
    indices_.resize(ptr_.back());
    std::fill(col_nnz.begin(), col_nnz.end(), 0);
    for (std::size_t r = 0; r < num_rows_; ++r) {
      const auto *row_ptr = &dense.at({r, std::size_t{0}});
      for (std::size_t c = 0; c < num_cols_; ++c) {
        if (row_ptr[c]) {
          index_type slot = ptr_[c] + col_nnz[c];
          indices_[slot] = static_cast<index_type>(r);
          ++col_nnz[c];
        }
      }
    }
    return;
  }

  std::vector<index_type> row_nnz(num_rows_, 0);
  for (std::size_t r = 0; r < num_rows_; ++r) {
    const auto *row_ptr = &dense.at({r, std::size_t{0}});
    for (std::size_t c = 0; c < num_cols_; ++c)
      if (row_ptr[c])
        ++row_nnz[r];
  }
  std::size_t total_nnz = 0;
  for (std::size_t r = 0; r < num_rows_; ++r)
    total_nnz += static_cast<std::size_t>(row_nnz[r]);
  throw_if_nnz_exceeds_index_range(total_nnz,
                                   "sparse_binary_matrix(dense,to CSR)");
  ptr_.resize(static_cast<std::size_t>(num_rows_) + 1);
  ptr_[0] = 0;
  for (std::size_t r = 0; r < num_rows_; ++r)
    ptr_[r + 1] = ptr_[r] + row_nnz[r];
  indices_.resize(ptr_.back());
  std::fill(row_nnz.begin(), row_nnz.end(), 0);
  for (std::size_t r = 0; r < num_rows_; ++r) {
    const auto *row_ptr = &dense.at({r, std::size_t{0}});
    for (std::size_t c = 0; c < num_cols_; ++c) {
      if (row_ptr[c]) {
        index_type slot = ptr_[r] + row_nnz[r];
        indices_[slot] = static_cast<index_type>(c);
        ++row_nnz[r];
      }
    }
  }
}

sparse_binary_matrix sparse_binary_matrix::to_csc() const {
  if (layout_ == sparse_binary_matrix_layout::csc) {
    return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows_,
                                num_cols_, ptr_, indices_);
  }
  // CSR -> CSC. size_t outer + `static_cast<size_t>(dim) + 1` ptr sizing avoid
  // UINT32_MAX wraparound; inner p-loop stays index_type (nnz-bounded).
  std::vector<index_type> col_nnz(num_cols_, 0);
  for (std::size_t i = 0; i < num_rows_; ++i) {
    for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
      index_type j = indices_[p];
      ++col_nnz[j];
    }
  }
  std::vector<index_type> col_ptrs(static_cast<std::size_t>(num_cols_) + 1);
  col_ptrs[0] = 0;
  for (std::size_t j = 0; j < num_cols_; ++j) {
    col_ptrs[j + 1] = col_ptrs[j] + col_nnz[j];
  }
  std::fill(col_nnz.begin(), col_nnz.end(), 0);
  std::vector<index_type> row_indices(indices_.size());
  for (std::size_t i = 0; i < num_rows_; ++i) {
    for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
      index_type j = indices_[p];
      index_type q = col_ptrs[j] + col_nnz[j];
      row_indices[q] = static_cast<index_type>(i);
      ++col_nnz[j];
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows_,
                              num_cols_, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix sparse_binary_matrix::to_csr() const {
  if (layout_ == sparse_binary_matrix_layout::csr) {
    return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows_,
                                num_cols_, ptr_, indices_);
  }
  // CSC -> CSR; see to_csc re: size_t.
  std::vector<index_type> row_nnz(num_rows_, 0);
  for (std::size_t j = 0; j < num_cols_; ++j) {
    for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
      index_type i = indices_[p];
      ++row_nnz[i];
    }
  }
  std::vector<index_type> row_ptrs(static_cast<std::size_t>(num_rows_) + 1);
  row_ptrs[0] = 0;
  for (std::size_t i = 0; i < num_rows_; ++i) {
    row_ptrs[i + 1] = row_ptrs[i] + row_nnz[i];
  }
  std::fill(row_nnz.begin(), row_nnz.end(), 0);
  std::vector<index_type> col_indices(indices_.size());
  for (std::size_t j = 0; j < num_cols_; ++j) {
    for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
      index_type i = indices_[p];
      index_type q = row_ptrs[i] + row_nnz[i];
      col_indices[q] = static_cast<index_type>(j);
      ++row_nnz[i];
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows_,
                              num_cols_, std::move(row_ptrs),
                              std::move(col_indices));
}

cudaqx::tensor<std::uint8_t> sparse_binary_matrix::to_dense() const {
  cudaqx::tensor<std::uint8_t> dense(
      std::vector<std::size_t>{num_rows_, num_cols_});
  // `&dense.at({r, 0})` is OOB when num_cols_ == 0.
  if (num_cols_ == 0)
    return dense;
  for (std::size_t r = 0; r < num_rows_; ++r) {
    std::memset(&dense.at({r, 0}), 0, num_cols_ * sizeof(std::uint8_t));
  }
  if (layout_ == sparse_binary_matrix_layout::csc) {
    for (std::size_t j = 0; j < num_cols_; ++j) {
      for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
        index_type i = indices_[p];
        dense.at({i, j}) = 1;
      }
    }
  } else {
    for (std::size_t i = 0; i < num_rows_; ++i) {
      for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
        index_type j = indices_[p];
        dense.at({i, j}) = 1;
      }
    }
  }
  return dense;
}

std::vector<std::vector<sparse_binary_matrix::index_type>>
sparse_binary_matrix::to_nested_csc() const {
  std::vector<std::vector<index_type>> out(num_cols_);
  if (layout_ == sparse_binary_matrix_layout::csc) {
    for (std::size_t j = 0; j < num_cols_; ++j) {
      out[j].assign(indices_.begin() + ptr_[j], indices_.begin() + ptr_[j + 1]);
    }
  } else {
    for (std::size_t i = 0; i < num_rows_; ++i) {
      for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
        index_type j = indices_[p];
        out[j].push_back(static_cast<index_type>(i));
      }
    }
  }
  return out;
}

std::vector<std::vector<sparse_binary_matrix::index_type>>
sparse_binary_matrix::to_nested_csr() const {
  std::vector<std::vector<index_type>> out(num_rows_);
  if (layout_ == sparse_binary_matrix_layout::csr) {
    for (std::size_t i = 0; i < num_rows_; ++i) {
      out[i].assign(indices_.begin() + ptr_[i], indices_.begin() + ptr_[i + 1]);
    }
  } else {
    for (std::size_t j = 0; j < num_cols_; ++j) {
      for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
        index_type i = indices_[p];
        out[i].push_back(static_cast<index_type>(j));
      }
    }
  }
  return out;
}

} // namespace cudaq::qec
