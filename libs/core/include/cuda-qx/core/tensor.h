/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "extension_point.h"
#include "tensor_impl.h"
#include "type_traits.h"

namespace cudaqx {

/// @brief A tensor class implementing the PIMPL idiom. The flattened data is
/// stored where the strides grow from right to left (similar to a
/// multi-dimensional C array).
template <typename Scalar = std::complex<double>>
class tensor {
private:
  std::shared_ptr<details::tensor_impl<Scalar>> pimpl;

  tensor<Scalar> mv_product(const tensor<Scalar> &vec) const {
    if (rank() != 2 || vec.rank() != 1) {
      throw std::runtime_error(
          "Matrix-vector product requires rank-2 matrix and rank-1 vector");
    }
    if (shape()[1] != vec.shape()[0]) {
      throw std::runtime_error("Invalid dimensions for matrix-vector product");
    }

    tensor<Scalar> result({shape()[0]});
    pimpl->matrix_vector_product(vec.pimpl.get(), result.pimpl.get());
    return result;
  }

public:
  /// @brief Type alias for the scalar type used in the tensor
  using scalar_type = typename details::tensor_impl<Scalar>::scalar_type;
  static constexpr auto ScalarAsString = type_to_string<Scalar>();

  /// @brief Construct an empty tensor
  tensor()
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(std::string("xtensor") +
                                                  std::string(ScalarAsString),
                                              std::vector<std::size_t>())
                .release())) {}

  /// @brief Construct a tensor with the given shape
  /// @param shape The shape of the tensor
  tensor(const std::vector<std::size_t> &shape)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(
                std::string("xtensor") + std::string(ScalarAsString), shape)
                .release())) {}

  /// @brief Construct a tensor with the given data and shape
  /// @param data Pointer to the tensor data. This takes ownership of the data.
  /// @param shape The shape of the tensor
  tensor(const scalar_type *data, const std::vector<std::size_t> &shape)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(std::string("xtensor") +
                                                  std::string(ScalarAsString),
                                              data, shape)
                .release())) {}

  /// @brief Construct a tensor with the given bitstrings
  /// @param data Bitstrings from which to construct tensor
  template <typename T, typename = std::enable_if_t<
                            std::is_convertible<T, std::string>::value>>
  tensor(const std::vector<T> &data)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(
                std::string("xtensor") + std::string(ScalarAsString), data)
                .release())) {}

  /// @brief Get the rank of the tensor
  /// @return The rank of the tensor
  std::size_t rank() const { return pimpl->rank(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const { return pimpl->size(); }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  std::vector<std::size_t> shape() const { return pimpl->shape(); }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  scalar_type &at(const std::vector<size_t> &indices) {
    if (indices.size() != rank())
      throw std::runtime_error("Invalid indices provided to tensor::at(), size "
                               "must be equal to rank.");
    return pimpl->at(indices);
  }

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  const scalar_type &at(const std::vector<size_t> &indices) const {
    return const_cast<const details::tensor_impl<Scalar> *>(pimpl.get())
        ->at(indices);
  }

  /// @brief Copy data into the tensor
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void copy(const scalar_type *data,
            const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error(
          "This tensor does not have a shape yet, must provide one to copy()");

    pimpl->copy(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Take ownership of the given data
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void take(const scalar_type *data,
            const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error(
          "This tensor does not have a shape yet, must provide one to take()");

    pimpl->take(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Borrow the given data without taking ownership
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void borrow(const scalar_type *data,
              const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error("This tensor does not have a shape yet, must "
                               "provide one to borrow()");

    pimpl->borrow(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  // Scalar-resulting operations
  Scalar sum_all() const { return pimpl->sum_all(); }

  // Boolean-resulting operations
  bool any() const { return pimpl->any(); }

  // Elementwise operations
  tensor<Scalar> operator+(const tensor<Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Tensor shapes must match for addition");
    }
    tensor<Scalar> result(shape());
    pimpl->elementwise_add(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  tensor<Scalar> operator*(const tensor<Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Tensor shapes must match for multiplication");
    }
    tensor<Scalar> result(shape());
    pimpl->elementwise_multiply(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  tensor<Scalar> operator%(const tensor<Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Tensor shapes must match for modulo");
    }
    tensor<Scalar> result(shape());
    pimpl->elementwise_modulo(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  // Tensor-Scalar operations
  tensor<Scalar> operator%(Scalar value) const {
    tensor<Scalar> result(shape());
    pimpl->scalar_modulo(value, result.pimpl.get());
    return result;
  }

  // Matrix operations (rank-2 specific)
  tensor<Scalar> dot(const tensor<Scalar> &other) const {

    if (rank() == 2 && other.rank() == 1)
      return mv_product(other);

    if (rank() != 2 || other.rank() != 2) {
      throw std::runtime_error("Dot product requires rank-2 tensors");
    }
    if (shape()[1] != other.shape()[0]) {
      throw std::runtime_error(
          "Invalid matrix dimensions for dot product: matrix1 dimensions: " +
          std::to_string(shape()[0]) + "x" + std::to_string(shape()[1]) +
          ", matrix2 dimensions: " + std::to_string(other.shape()[0]) + "x" +
          std::to_string(other.shape()[1]));
    }

    std::vector<std::size_t> result_shape = {shape()[0], other.shape()[1]};
    tensor<Scalar> result(result_shape);
    pimpl->matrix_dot(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  tensor<Scalar> transpose() const {
    if (rank() != 2) {
      throw std::runtime_error("Transpose requires rank-2 tensors");
    }

    std::vector<std::size_t> result_shape = {shape()[1], shape()[0]};
    tensor<Scalar> result(result_shape);
    pimpl->matrix_transpose(result.pimpl.get());
    return result;
  }

  /// @brief Get a pointer to the raw data of the tensor.
  ///
  /// This method provides direct access to the underlying data storage of the
  /// tensor. It returns a pointer to the first element of the data array.
  ///
  /// @return scalar_type* A pointer to the mutable data of the tensor.
  ///
  /// @note Care should be taken when directly manipulating the raw data to
  /// avoid
  ///       invalidating the tensor's internal state or violating its
  ///       invariants.
  scalar_type *data() { return pimpl->data(); }

  /// @brief Get a const pointer to the raw data of the tensor.
  ///
  /// This method provides read-only access to the underlying data storage of
  /// the tensor. It returns a const pointer to the first element of the data
  /// array.
  ///
  /// @return const scalar_type * A const pointer to the immutable data of the
  /// tensor.
  ///
  /// @note This const version ensures that the tensor's data cannot be modified
  ///       through the returned pointer, preserving const correctness.
  const scalar_type *data() const { return pimpl->data(); }

  void dump() const { pimpl->dump(); }

  /// @brief Dump tensor as bits, where non-zero elements are shown as '1' and
  /// zero-elements are shown as '.'.
  void dump_bits() const { pimpl->dump_bits(); }
};

} // namespace cudaqx
