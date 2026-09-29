/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "extension_point.h"

#include <complex>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
namespace cudaqx::details {

/// @brief Implementation class for tensor operations following the PIMPL idiom
template <typename Scalar = std::complex<double>>
class tensor_impl
    : public cudaqx::extension_point<tensor_impl<Scalar>, const Scalar *,
                                     const std::vector<std::size_t>> {
public:
  /// @brief Type alias for the scalar type used in the tensor
  using scalar_type = Scalar;
  using BaseExtensionPoint =
      cudaqx::extension_point<tensor_impl<Scalar>, const Scalar *,
                              const std::vector<std::size_t>>;

  /// @brief Create a tensor implementation with the given name and shape
  /// @param name The name of the tensor implementation
  /// @param shape The shape of the tensor
  /// @return A unique pointer to the created tensor implementation
  /// @throws std::runtime_error if the requested tensor implementation is
  /// invalid
  static std::unique_ptr<tensor_impl<Scalar>>
  get(const std::string &name, const std::vector<std::size_t> &shape) {
    auto [mutex, registry] = BaseExtensionPoint::get_registry();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error("invalid tensor_impl requested: " + name);

    if (shape.empty())
      return iter->second(nullptr, {});

    std::size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<size_t>());
    scalar_type *data = new scalar_type[size]();
    return iter->second(data, shape);
  }

  /// @brief Create a tensor implementation with the given set of bit strings
  /// @param name The name of the tensor implementation
  /// @param data Pointer to the tensor data as a vector of bitstrings
  /// @return A unique pointer to the created tensor implementation
  /// @throws std::runtime_error if the requested tensor implementation is
  /// invalid
  static std::unique_ptr<tensor_impl<Scalar>>
  get(const std::string &name, const std::vector<std::string> &data) {
    auto [mutex, registry] = BaseExtensionPoint::get_registry();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error("invalid tensor_impl requested: " + name);
    auto num_rows = data.size();
    if (num_rows == 0)
      return iter->second(nullptr, {});
    auto num_cols = data[0].size();
    for (size_t r = 1; r < num_rows; r++)
      if (data[r].size() != num_cols)
        throw std::runtime_error("inconsistent tensor_impl dimensions found in "
                                 "std::vector<std::string> inputs");
    std::size_t size = num_rows * num_cols;
    scalar_type *scalar_data = new scalar_type[size]();
    auto result = iter->second(scalar_data, {num_rows, num_cols});
    for (size_t r = 0; r < num_rows; r++) {
      Scalar *__restrict__ resultRow = &result->at({r, 0});
      for (size_t c = 0; c < num_cols; c++) {
        resultRow[c] = data[r][c] - '0';
      }
    }
    return std::move(result);
  }

  /// @brief Create a tensor implementation with the given name, data, and shape
  /// @param name The name of the tensor implementation
  /// @param data Pointer to the tensor data
  /// @param shape The shape of the tensor
  /// @return A unique pointer to the created tensor implementation
  /// @throws std::runtime_error if the requested tensor implementation is
  /// invalid
  static std::unique_ptr<tensor_impl<Scalar>>
  get(const std::string &name, const scalar_type *data,
      const std::vector<std::size_t> &shape) {
    auto [mutex, registry] = BaseExtensionPoint::get_registry();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error("invalid tensor_impl requested: " + name);
    return iter->second(data, shape);
  }

  /// @brief Get the rank of the tensor
  /// @return The rank of the tensor
  virtual std::size_t rank() const = 0;

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  virtual std::size_t size() const = 0;

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  virtual std::vector<std::size_t> shape() const = 0;

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  virtual scalar_type &at(const std::vector<size_t> &indices) = 0;

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  virtual const scalar_type &at(const std::vector<size_t> &indices) const = 0;

  /// @brief Copy data into the tensor
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  virtual void copy(const scalar_type *data,
                    const std::vector<std::size_t> &shape) = 0;

  /// @brief Take ownership of the given data
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  virtual void take(const scalar_type *data,
                    const std::vector<std::size_t> &shape) = 0;

  /// @brief Borrow the given data without taking ownership
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  virtual void borrow(const scalar_type *data,
                      const std::vector<std::size_t> &shape) = 0;

  virtual scalar_type sum_all() const = 0;

  virtual bool any() const = 0;

  virtual void elementwise_add(const tensor_impl<Scalar> *other,
                               tensor_impl<Scalar> *result) const = 0;

  virtual void elementwise_multiply(const tensor_impl<Scalar> *other,
                                    tensor_impl<Scalar> *result) const = 0;

  virtual void elementwise_modulo(const tensor_impl<Scalar> *other,
                                  tensor_impl<Scalar> *result) const = 0;

  virtual void scalar_modulo(Scalar value,
                             tensor_impl<Scalar> *result) const = 0;

  virtual void matrix_dot(const tensor_impl<Scalar> *other,
                          tensor_impl<Scalar> *result) const = 0;

  virtual void matrix_vector_product(const tensor_impl<Scalar> *vec,
                                     tensor_impl<Scalar> *result) const = 0;

  virtual void matrix_transpose(tensor_impl<Scalar> *result) const = 0;

  /// @brief Get a pointer to the raw data of the tensor.
  /// This method provides direct access to the underlying data storage of the
  /// tensor. It returns a pointer to the first element of the data array.
  ///
  /// @return scalar_type* A pointer to the mutable data of the tensor.
  /// @note Care should be taken when directly manipulating the raw data to
  /// avoid invalidating the tensor's internal state or violating its
  /// invariants.
  virtual scalar_type *data() = 0;

  /// @brief Get a const pointer to the raw data of the tensor.
  /// This method provides read-only access to the underlying data storage of
  /// the tensor. It returns a const pointer to the first element of the data
  /// array.
  ///
  /// @return const scalar_type * A const pointer to the immutable data of the
  /// tensor.
  /// @note This const version ensures that the tensor's data cannot be modified
  ///       through the returned pointer, preserving const correctness.
  virtual const scalar_type *data() const = 0;

  virtual void dump() const = 0;

  /// @brief Dump tensor as bits, where non-zero elements are shown as '1' and
  /// zero-elements are shown as '.'.
  virtual void dump_bits() const = 0;

  virtual ~tensor_impl() = default;
};

} // namespace cudaqx::details
