/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaqx {

/// @brief Return the value of given type corresponding to the provided
/// key string from the provided options `kwargs` `dict`. Return the `orVal`
/// if the key is not in the `dict`.
template <typename T>
T getValueOr(nb::kwargs &options, const std::string &key, const T &orVal) {
  if (options.contains(key))
    for (auto item : options)
      if (nb::cast<std::string>(item.first) == key)
        return nb::cast<T>(item.second);

  return orVal;
}

inline heterogeneous_map hetMapFromKwargs(const nb::kwargs &kwargs) {
  cudaqx::heterogeneous_map result;

  for (const auto &item : kwargs) {
    std::string key = nb::cast<std::string>(item.first);
    nb::handle value = item.second;

    if (nb::isinstance<nb::bool_>(value)) {
      result.insert(key, nb::cast<bool>(value));
    } else if (nb::isinstance<nb::int_>(value)) {
      result.insert(key, nb::cast<std::size_t>(value));
    } else if (nb::isinstance<nb::float_>(value)) {
      result.insert(key, nb::cast<double>(value));
    } else if (nb::isinstance<nb::str>(value)) {
      result.insert(key, nb::cast<std::string>(value));
    } else if (nb::isinstance<nb::dict>(value)) {
      // Recursively convert nested dictionary
      result.insert(key, hetMapFromKwargs(nb::cast<nb::kwargs>(value)));
    } else if (nb::isinstance<nb::list>(value)) {
      // Handle Python lists
      nb::list py_list = nb::cast<nb::list>(value);
      if (py_list.size() > 0) {
        // Check if it's a nested list (list of lists)
        if (nb::isinstance<nb::list>(py_list[0])) {
          std::vector<std::vector<double>> vec_vec;
          for (const auto &item : py_list) {
            nb::list inner_list = nb::cast<nb::list>(item);
            std::vector<double> inner_vec;
            for (const auto &v : inner_list) {
              inner_vec.push_back(nb::cast<double>(v));
            }
            vec_vec.push_back(std::move(inner_vec));
          }
          result.insert(key, std::move(vec_vec));
        } else {
          // Single-level list - try to convert to vector<double>
          std::vector<double> vec;
          for (const auto &item : py_list) {
            vec.push_back(nb::cast<double>(item));
          }
          result.insert(key, std::move(vec));
        }
      }
    } else if (nb::isinstance<nb::ndarray<>>(value)) {
      nb::ndarray<> np_array = nb::cast<nb::ndarray<>>(value);
      if (np_array.ndim() >= 2) {
        // nanobind strides are element counts; stride(0)==1 means column-major
        if (np_array.stride(0) == 1) {
          throw std::runtime_error(
              "Array in kwargs must be in row-major order, but "
              "column-major order was detected.");
        }
        std::vector<std::size_t> shape;
        shape.reserve(np_array.ndim());
        for (std::size_t d = 0; d < np_array.ndim(); d++)
          shape.push_back(np_array.shape(d));

        auto dtype = np_array.dtype();
        auto insert_tensor = [&](auto type_tag) {
          using T = decltype(type_tag);
          cudaqx::tensor<T> ten(shape);
          ten.borrow(static_cast<T *>(np_array.data()), shape);
          result.insert(key, std::move(ten));
        };
        if (dtype == nb::dtype<double>()) {
          insert_tensor(double{});
        } else if (dtype == nb::dtype<float>()) {
          insert_tensor(float{});
        } else if (dtype == nb::dtype<int>()) {
          insert_tensor(int{});
        } else if (dtype == nb::dtype<uint8_t>()) {
          insert_tensor(uint8_t{});
        } else {
          throw std::runtime_error("Unsupported array data type in kwargs.");
        }
      } else {
        // 1D array: keep as flattened vector for backward compatibility
        // (e.g. error_rate_vec used by decoders).
        auto dtype = np_array.dtype();
        auto insert_vector = [&](auto type_tag) {
          using T = decltype(type_tag);
          T *ptr = static_cast<T *>(np_array.data());
          std::vector<T> vec(ptr, ptr + np_array.size());
          result.insert(key, std::move(vec));
        };
        if (dtype == nb::dtype<double>()) {
          insert_vector(double{});
        } else if (dtype == nb::dtype<float>()) {
          insert_vector(float{});
        } else if (dtype == nb::dtype<int>()) {
          insert_vector(int{});
        } else if (dtype == nb::dtype<uint8_t>()) {
          insert_vector(uint8_t{});
        } else {
          throw std::runtime_error("Unsupported array data type in kwargs.");
        }
      }
    } else {
      throw std::runtime_error(
          "Invalid python type for mapping kwargs to a heterogeneous_map.");
    }
  }

  return result;
}

template <typename T>
tensor<T> toTensor(const nb::ndarray<nb::numpy, T> &H,
                   bool perform_pcm_checks = false) {
  std::size_t expected_stride = 1;
  for (int i = static_cast<int>(H.ndim()) - 1; i >= 0; --i) {
    if (H.shape(i) > 1 &&
        static_cast<std::size_t>(H.stride(i)) != expected_stride) {
      throw std::runtime_error("toTensor: data must be in row-major order, but "
                               "column-major order was detected.");
    }
    expected_stride *= H.shape(i);
  }

  if (perform_pcm_checks) {
    if (H.itemsize() != sizeof(uint8_t)) {
      throw std::runtime_error(
          "Parity check matrix must be an array of uint8_t.");
    }

    if (H.ndim() != 2) {
      throw std::runtime_error("Parity check matrix must be 2-dimensional.");
    }
  }

  // Create a vector of the array dimensions
  std::vector<std::size_t> shape;
  shape.reserve(H.ndim());
  for (std::size_t d = 0; d < H.ndim(); d++) {
    shape.push_back(H.shape(d));
  }

  // Create a tensor and borrow the NumPy array data
  cudaqx::tensor<T> tensor_H(shape);
  tensor_H.borrow(static_cast<T *>(H.data()), std::move(shape));
  return tensor_H;
}

/// @brief Convert a nb::ndarray<nb::numpy, T> to a tensor<T>. This is the same
/// as toTensor, but with additional checks.
template <typename T>
tensor<T> pcmToTensor(const nb::ndarray<nb::numpy, T> &H) {
  return toTensor(H, /*perform_pcm_checks=*/true);
}

} // namespace cudaqx
