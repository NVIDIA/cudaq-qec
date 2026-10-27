/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: row-view adapters and primitive decode kernels shared by
// node implementations.  Never included from a public header.

#pragma once
#include "cudaq/qec/decoder.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {

// ---------------------------------------------------------------------------
// Row-view adapters — one set of kernels serves both nested and CSR storage.
// ---------------------------------------------------------------------------

struct nested_rows {
  const std::vector<std::vector<uint32_t>> &rows;
  std::size_t size() const { return rows.size(); }
  template <typename Fn>
  void for_each_column(std::size_t i, Fn fn) const {
    for (uint32_t col : rows[i])
      fn(col);
  }
};

struct csr_rows {
  const sparse_binary_matrix &m;
  std::size_t size() const { return m.num_rows(); }
  template <typename Fn>
  void for_each_column(std::size_t i, Fn fn) const {
    const auto &ptr = m.ptr();
    const auto &idx = m.indices();
    for (auto k = ptr[i]; k < ptr[i + 1]; ++k)
      fn(idx[k]);
  }
};

template <typename Rows>
std::optional<uint32_t> first_column_at_or_above(const Rows &rows,
                                                 std::size_t limit) {
  std::optional<uint32_t> found;
  for (std::size_t i = 0, n = rows.size(); i < n; ++i)
    rows.for_each_column(i, [&](uint32_t col) {
      if (col >= limit && !found)
        found = col;
    });
  return found;
}

template <typename Rows>
void detect_into(const Rows &rows, const uint8_t *m, float_t *out) {
  for (std::size_t i = 0, n = rows.size(); i < n; ++i) {
    uint8_t parity = 0;
    rows.for_each_column(i, [&](uint32_t col) { parity ^= m[col]; });
    out[i] = static_cast<float_t>(parity);
  }
}

/// Scatter H * hard(e) over committed columns [begin, end). H is canonical
/// CSC, prepared before streaming. Calling toggle once per incident detector
/// preserves GF(2) cancellation without scanning inactive fault edges.
template <typename Toggle>
void scatter_fault_syndrome(const sparse_binary_matrix &h, const float_t *e,
                            std::size_t begin, std::size_t end, Toggle toggle) {
  const auto &ptr = h.ptr();
  const auto &indices = h.indices();
  for (auto col = begin; col < end; ++col)
    if (convert_soft_to_hard(e[col]))
      for (auto k = ptr[col]; k < ptr[col + 1]; ++k)
        toggle(indices[k]);
}

template <typename Rows>
void project_into(const Rows &rows, const float_t *e, uint8_t *out) {
  for (std::size_t i = 0, n = rows.size(); i < n; ++i) {
    uint8_t parity = 0;
    rows.for_each_column(i, [&](uint32_t col) {
      parity ^= static_cast<uint8_t>(convert_soft_to_hard(e[col]));
    });
    out[i] ^= parity;
  }
}

/// Detection events a prediction does not explain: out[i] = dets[i] XOR the
/// parity of row i of `rows` over hard(e).
///
/// The transpose partner of project_into: the same row/column walk, but the
/// result lands in detector space rather than observable space, and it
/// overwrites rather than XOR-accumulates because `dets` is its starting
/// value.  With `rows` the decoder's H and `e` its predicted error pattern,
/// this is the syndrome left for someone else to decode.  Output is hard 0/1
/// carried in float_t, which is what detection_events holds.
template <typename Rows>
void residual_into(const Rows &rows, const float_t *dets, const float_t *e,
                   float_t *out) {
  for (std::size_t i = 0, n = rows.size(); i < n; ++i) {
    uint8_t parity = static_cast<uint8_t>(convert_soft_to_hard(dets[i]));
    rows.for_each_column(i, [&](uint32_t col) {
      parity ^= static_cast<uint8_t>(convert_soft_to_hard(e[col]));
    });
    out[i] = static_cast<float_t>(parity);
  }
}

/// Normalize a matrix to CSR for the row-walking kernels above.
///
/// CSC holds the same matrix stored the other way round, so `csr_rows` over it
/// would read columns as rows -- wrong, and silently so. The tensor
/// constructor defaults to CSC, so a node normalizes what it is handed rather
/// than trusting its caller to have picked the layout its kernels assume. A
/// no-op when the matrix is already CSR (and for a null handle).
inline std::shared_ptr<const sparse_binary_matrix>
as_csr(std::shared_ptr<const sparse_binary_matrix> m) {
  if (!m || m->layout() == sparse_binary_matrix_layout::csr)
    return m;
  return std::make_shared<const sparse_binary_matrix>(m->to_csr());
}

inline uint32_t required_width(const std::vector<std::vector<uint32_t>> &rows,
                               const char *name) {
  uint32_t width = 0;
  for (const auto &row : rows)
    for (uint32_t col : row) {
      if (col == std::numeric_limits<uint32_t>::max())
        throw std::out_of_range(std::string("decoding_task_graph: ") + name +
                                " column index 2^32-1 is not representable");
      width = std::max(width, col + 1);
    }
  return width;
}

inline sparse_binary_matrix
make_csr(const std::vector<std::vector<uint32_t>> &rows, const char *name) {
  using idx = sparse_binary_matrix::index_type;
  return sparse_binary_matrix::from_nested_csr(static_cast<idx>(rows.size()),
                                               required_width(rows, name), rows)
      .canonicalize();
}

} // namespace cudaq::qec
