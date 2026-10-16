/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Conversions between sparse_binary_matrix and the Python matrix types
// (dense NumPy arrays and scipy.sparse matrices). sparse_binary_matrix is not
// exposed to Python as a class, so every binding that accepts or returns one
// goes through these helpers.
//
// These live here rather than in type_casters.h so that the general-purpose
// casters in that header do not pick up a dependency on the QEC headers.

#pragma once

#include "type_casters.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace cudaq::qec {

/// Range-checked narrow from std::size_t to sparse_binary_matrix::index_type;
/// unchecked static_cast would silently truncate 64-bit Python ints.
inline sparse_binary_matrix::index_type
checked_narrow_to_index_type(std::size_t value, const char *field_name) {
  if (value > std::numeric_limits<sparse_binary_matrix::index_type>::max())
    throw std::runtime_error(
        std::string(field_name) +
        " exceeds sparse_binary_matrix index_type (uint32_t) range; got " +
        std::to_string(value));
  return static_cast<sparse_binary_matrix::index_type>(value);
}

/// Build sparse_binary_matrix directly from a scipy sparse matrix.
/// Any scipy sparse format is accepted; it is normalized to CSR internally.
inline sparse_binary_matrix sparse_binary_matrix_from_scipy(nb::object mat) {
  // Normalize to CSR so that indptr == row_offsets, indices == col_indices.
  nb::object csr = mat.attr("tocsr")();
  // Avoid mutating the caller's matrix and clean it up for our internal use.
  csr = csr.attr("copy")();
  csr.attr("sum_duplicates")();
  csr.attr("eliminate_zeros")();
  csr.attr("sort_indices")();
  nb::tuple shape_t = nb::cast<nb::tuple>(csr.attr("shape"));
  auto num_rows = checked_narrow_to_index_type(
      nb::cast<std::size_t>(shape_t[0]), "num_rows");
  auto num_cols = checked_narrow_to_index_type(
      nb::cast<std::size_t>(shape_t[1]), "num_cols");

  // Copy a numpy integer array to vector<uint32_t> using pure C++ dtype
  // dispatch — handles int32, int64, uint32, uint64 (all common scipy dtypes).
  // Values are range-checked: scipy stores indices signed, so a corrupt or
  // hand-built matrix could otherwise wrap into a huge unsigned index.
  auto copy_to_uint32 = [](nb::handle arr_h, const char *field_name) {
    auto arr = nb::cast<nb::ndarray<>>(arr_h);
    std::vector<sparse_binary_matrix::index_type> out(arr.size());
    auto dtype = arr.dtype();
    auto narrow_signed = [&](auto *p) {
      for (size_t i = 0; i < arr.size(); ++i) {
        if (p[i] < 0)
          throw std::runtime_error(std::string(field_name) +
                                   " contains a negative index");
        out[i] = checked_narrow_to_index_type(static_cast<std::size_t>(p[i]),
                                              field_name);
      }
    };
    if (dtype == nb::dtype<int32_t>()) {
      narrow_signed(static_cast<const int32_t *>(arr.data()));
    } else if (dtype == nb::dtype<int64_t>()) {
      narrow_signed(static_cast<const int64_t *>(arr.data()));
    } else if (dtype == nb::dtype<uint32_t>()) {
      std::memcpy(out.data(), arr.data(),
                  arr.size() * sizeof(sparse_binary_matrix::index_type));
    } else if (dtype == nb::dtype<uint64_t>()) {
      auto *p = static_cast<const uint64_t *>(arr.data());
      for (size_t i = 0; i < arr.size(); ++i)
        out[i] = checked_narrow_to_index_type(static_cast<std::size_t>(p[i]),
                                              field_name);
    } else {
      throw std::runtime_error(
          "scipy sparse matrix indptr/indices has unsupported dtype; "
          "expected int32, int64, uint32, or uint64.");
    }
    return out;
  };

  auto ptr = copy_to_uint32(csr.attr("indptr"), "indptr");
  auto idx = copy_to_uint32(csr.attr("indices"), "indices");

  return sparse_binary_matrix::from_csr(num_rows, num_cols, std::move(ptr),
                                        std::move(idx));
}

/// Convert a dense 2-D NumPy uint8 array to sparse_binary_matrix without
/// any intermediate dense tensor allocation.  Strides are read directly so
/// both C-contiguous (row-major) and Fortran-contiguous (column-major) arrays
/// are handled efficiently: the inner loop always traverses contiguous memory.
inline sparse_binary_matrix
make_sparse_from_dense(const nb::ndarray<nb::numpy, uint8_t> &arr) {
  if (arr.ndim() != 2)
    throw std::invalid_argument("H must be a 2-D uint8 array");
  const std::size_t num_rows = arr.shape(0);
  const std::size_t num_cols = arr.shape(1);
  const auto rows = checked_narrow_to_index_type(num_rows, "num_rows");
  const auto cols = checked_narrow_to_index_type(num_cols, "num_cols");
  const std::ptrdiff_t rs = arr.stride(0); // bytes per row step
  const std::ptrdiff_t cs = arr.stride(1); // bytes per col step
  const uint8_t *base = static_cast<const uint8_t *>(arr.data());

  using index_t = sparse_binary_matrix::index_type;
  std::vector<index_t> ptr, idx;

  // C-order: inner loop over columns is sequential → build CSR.
  // F-order: inner loop over rows is sequential → build CSC.
  // Comparing magnitudes keeps this correct for negative (reversed) strides.
  if (std::abs(cs) <= std::abs(rs)) {
    ptr.reserve(num_rows + 1);
    ptr.push_back(0);
    for (std::size_t i = 0; i < num_rows; ++i) {
      for (std::size_t j = 0; j < num_cols; ++j) {
        if (base[static_cast<std::ptrdiff_t>(i) * rs +
                 static_cast<std::ptrdiff_t>(j) * cs])
          idx.push_back(static_cast<index_t>(j));
      }
      ptr.push_back(static_cast<index_t>(idx.size()));
    }
    return sparse_binary_matrix::from_csr(rows, cols, std::move(ptr),
                                          std::move(idx));
  }
  ptr.reserve(num_cols + 1);
  ptr.push_back(0);
  for (std::size_t j = 0; j < num_cols; ++j) {
    for (std::size_t i = 0; i < num_rows; ++i) {
      if (base[static_cast<std::ptrdiff_t>(i) * rs +
               static_cast<std::ptrdiff_t>(j) * cs])
        idx.push_back(static_cast<index_t>(i));
    }
    ptr.push_back(static_cast<index_t>(idx.size()));
  }
  return sparse_binary_matrix::from_csc(rows, cols, std::move(ptr),
                                        std::move(idx));
}

/// Accept either a scipy sparse matrix (any format) or a dense array of any
/// numeric dtype. This is the single entry point for bindings that take a
/// matrix from Python.
inline sparse_binary_matrix sparse_binary_matrix_from_python(nb::object mat) {
  // Any scipy sparse format exposes tocsr(); detect via that rather than
  // indptr/indices, which COO and some other formats lack.
  if (nb::hasattr(mat, "tocsr"))
    return sparse_binary_matrix_from_scipy(mat);
  // copy=False makes astype a no-op when the input is already uint8;
  // make_sparse_from_dense reads strides directly, so a non-contiguous uint8
  // input is also handled without a copy.
  return make_sparse_from_dense(nb::cast<nb::ndarray<nb::numpy, uint8_t>>(
      mat.attr("astype")("uint8", nb::arg("copy") = false)));
}

template <typename T>
inline nb::ndarray<nb::numpy, T> vector_to_numpy_1d(std::vector<T> values) {
  const size_t logical_size = values.size();
  auto *owned = new std::vector<T>(std::move(values));
  if (owned->empty())
    owned->resize(1);
  nb::capsule owner(
      owned, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
  size_t shape[1] = {logical_size};
  return nb::ndarray<nb::numpy, T>(owned->data(), 1, shape, owner);
}

inline std::vector<std::int64_t>
to_int64_vector(const std::vector<sparse_binary_matrix::index_type> &values) {
  std::vector<std::int64_t> out;
  out.reserve(values.size());
  for (auto value : values)
    out.push_back(static_cast<std::int64_t>(value));
  return out;
}

/// Dense NumPy copy of a sparse matrix, for the read side of matrix
/// properties. sparse_binary_matrix has no nanobind type caster, so a binding
/// that returns one directly compiles but fails at call time with "unable to
/// convert return value".
inline nb::ndarray<nb::numpy, std::uint8_t>
sparse_binary_matrix_to_numpy(const sparse_binary_matrix &matrix) {
  return cudaq::python::copyCUDAQXTensorToPyArray(matrix.to_dense());
}

inline nb::object
sparse_binary_matrix_to_scipy_csc(const sparse_binary_matrix &matrix) {
  auto csc = matrix.to_csc();
  std::vector<std::uint8_t> data(csc.num_nnz(), 1);
  auto scipy_sparse = nb::module_::import_("scipy.sparse");
  return scipy_sparse.attr("csc_matrix")(
      nb::make_tuple(vector_to_numpy_1d(std::move(data)),
                     vector_to_numpy_1d(to_int64_vector(csc.indices())),
                     vector_to_numpy_1d(to_int64_vector(csc.ptr()))),
      nb::arg("shape") = nb::make_tuple(csc.num_rows(), csc.num_cols()));
}

} // namespace cudaq::qec
