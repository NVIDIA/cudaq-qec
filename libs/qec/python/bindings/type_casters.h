/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstring>
#include <stdexcept>

#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/kwargs_utils.h"
#include "cuda-qx/core/tensor.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// Casters for CUDA-Q runtime result types (spin_op, sample_result,
// observe_result) live in cudaq_type_casters.h; they need CUDA-Q headers,
// which decoder-only translation units must not depend on.

namespace nanobind {
namespace detail {

template <>
struct type_caster<cudaqx::heterogeneous_map> {
  NB_TYPE_CASTER(cudaqx::heterogeneous_map, const_name("dict"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      if (nb::isinstance<nb::dict>(src)) {
        value = cudaqx::hetMapFromKwargs(nb::cast<nb::kwargs>(src));
        return true;
      }
      return false;
    } catch (...) {
      return false;
    }
  }

  // Take by const-reference: the map can hold large payloads (e.g. the
  // bp_llr_history LLR matrix), and a by-value parameter deep-copied all of it
  // on every conversion.
  static handle from_cpp(const cudaqx::heterogeneous_map &v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::dict result;
      for (const auto &[key, val] : v) {
        if (auto *bool_val = std::any_cast<bool>(&val)) {
          result[key.c_str()] = *bool_val;
        } else if (auto *int_val = std::any_cast<std::size_t>(&val)) {
          result[key.c_str()] = *int_val;
        } else if (auto *int_val = std::any_cast<int>(&val)) {
          result[key.c_str()] = *int_val;
        } else if (auto *int_val = std::any_cast<uint8_t>(&val)) {
          result[key.c_str()] = *int_val;
        } else if (auto *double_val = std::any_cast<double>(&val)) {
          result[key.c_str()] = *double_val;
        } else if (auto *float_val = std::any_cast<float>(&val)) {
          result[key.c_str()] = *float_val;
        } else if (auto *str_val = std::any_cast<std::string>(&val)) {
          result[key.c_str()] = *str_val;
        } else if (auto *vec_vec_val =
                       std::any_cast<std::vector<std::vector<double>>>(&val)) {
          // Convert vector<vector<double>> to Python list of lists
          nb::list outer_list;
          for (const auto &inner_vec : *vec_vec_val) {
            nb::list inner_list;
            for (double dv : inner_vec) {
              inner_list.append(dv);
            }
            outer_list.append(inner_list);
          }
          result[key.c_str()] = outer_list;
        } else if (auto *vec_val = std::any_cast<std::vector<double>>(&val)) {
          size_t n = vec_val->size();
          double *data_copy = new double[n];
          std::copy(vec_val->begin(), vec_val->end(), data_copy);
          size_t shape[] = {n};
          result[key.c_str()] = nb::ndarray<nb::numpy, double>(
              data_copy, 1, shape, nb::capsule(data_copy, [](void *p) noexcept {
                delete[] static_cast<double *>(p);
              }));
        } else if (auto *vec_int_val = std::any_cast<std::vector<int>>(&val)) {
          size_t n = vec_int_val->size();
          int *data_copy = new int[n];
          std::copy(vec_int_val->begin(), vec_int_val->end(), data_copy);
          size_t shape[] = {n};
          result[key.c_str()] = nb::ndarray<nb::numpy, int>(
              data_copy, 1, shape, nb::capsule(data_copy, [](void *p) noexcept {
                delete[] static_cast<int *>(p);
              }));
        } else if (auto *hetMap =
                       std::any_cast<cudaqx::heterogeneous_map>(&val)) {
          // Recursively convert nested heterogeneous_map
          result[key.c_str()] = nb::cast(*hetMap);
        } else {
          PyErr_SetString(PyExc_RuntimeError,
                          ("Failed to cast from heterogeneous_map to "
                           "Python dict. Unsupported data type in the '" +
                           key + "' field.")
                              .c_str());
          return handle();
        }
      }
      return result.release();
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return handle();
    } catch (...) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Unknown error in heterogeneous_map conversion");
      return handle();
    }
  }
};

} // namespace detail
} // namespace nanobind

namespace cudaq {
namespace python {

template <typename T>
auto copyCUDAQXTensorToPyArray(const cudaqx::tensor<T> &tensor) {
  auto shape = tensor.shape();
  if (shape.size() != 2)
    throw std::runtime_error(
        "Expected a rank-2 cudaqx::tensor for NumPy conversion.");

  auto rows = shape[0];
  auto cols = shape[1];
  size_t total_size = rows * cols;

  // Allocate new memory and copy the data
  T *data_copy = new T[total_size];
  if (total_size > 0)
    std::memcpy(data_copy, tensor.data(), total_size * sizeof(T));

  size_t arr_shape[] = {rows, cols};
  return nb::ndarray<nb::numpy, T>(data_copy, 2, arr_shape,
                                   nb::capsule(data_copy, [](void *p) noexcept {
                                     delete[] static_cast<T *>(p);
                                   }));
}

template <typename T>
auto copy1DCUDAQXTensorToPyArray(const cudaqx::tensor<T> &tensor) {
  auto shape = tensor.shape();
  if (shape.size() != 1)
    throw std::runtime_error(
        "Expected a rank-1 cudaqx::tensor for NumPy conversion.");

  auto rows = shape[0];
  size_t total_size = rows;

  // Allocate new memory and copy the data
  T *data_copy = new T[total_size];
  if (total_size > 0)
    std::memcpy(data_copy, tensor.data(), total_size * sizeof(T));

  size_t arr_shape[] = {rows};
  return nb::ndarray<nb::numpy, T>(data_copy, 1, arr_shape,
                                   nb::capsule(data_copy, [](void *p) noexcept {
                                     delete[] static_cast<T *>(p);
                                   }));
}

} // namespace python
} // namespace cudaq
