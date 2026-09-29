/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda-qx/core/tensor_impl.h"
#include "cuda-qx/core/type_traits.h"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

#include <fmt/ranges.h>

namespace cudaqx {

// Use is_complex<T> to evaluate whether or not T is std::complex.
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

/// @brief An implementation of tensor_impl using xtensor library
template <typename Scalar>
class xtensor : public cudaqx::details::tensor_impl<Scalar> {
private:
  Scalar *m_data = nullptr;         ///< Pointer to the tensor data
  std::vector<std::size_t> m_shape; ///< Shape of the tensor
  bool ownsData = true; ///< Flag indicating if this object owns the data

  /// @brief Check if the given indices are valid for this tensor
  /// @param idxs Vector of indices to check
  /// @return true if indices are valid, false otherwise
  bool validIndices(const std::vector<std::size_t> &idxs) const {
    if (idxs.size() != m_shape.size())
      return false;
    for (std::size_t dim = 0; auto idx : idxs)
      if (idx < 0 || idx >= m_shape[dim++])
        return false;
    return true;
  }

public:
  /// @brief Constructor for xtensor
  /// @param d Pointer to the tensor data
  /// @param s Shape of the tensor
  xtensor(const Scalar *d, const std::vector<std::size_t> &s)
      : m_data(const_cast<Scalar *>(d)), m_shape(s) {}

  /// @brief Get the rank of the tensor
  /// @return The rank (number of dimensions) of the tensor
  std::size_t rank() const override { return m_shape.size(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const override {
    if (rank() == 0)
      return 0;
    return std::accumulate(m_shape.begin(), m_shape.end(), 1,
                           std::multiplies<size_t>());
  }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  std::vector<std::size_t> shape() const override { return m_shape; }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  Scalar &at(const std::vector<size_t> &indices) override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));

    return xt::adapt(m_data, size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  const Scalar &at(const std::vector<size_t> &indices) const override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid constant tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));
    return xt::adapt(m_data, size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Copy data into the tensor
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void copy(const Scalar *d, const std::vector<std::size_t> &shape) override {
    auto size = std::accumulate(shape.begin(), shape.end(), 1,
                                std::multiplies<size_t>());
    if (m_data)
      delete[] m_data;

    m_data = new Scalar[size];
    std::copy(d, d + size, m_data);
    m_shape = shape;
    ownsData = true;
  }

  /// @brief Take ownership of the given data
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void take(const Scalar *d, const std::vector<std::size_t> &shape) override {
    m_data = const_cast<Scalar *>(d);
    m_shape = shape;
    ownsData = true;
  }

  /// @brief Borrow the given data without taking ownership
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void borrow(const Scalar *d, const std::vector<std::size_t> &shape) override {
    m_data = const_cast<Scalar *>(d);
    m_shape = shape;
    ownsData = false;
  }

  /// @brief Sum all elements of the tensor
  /// @return A scalar sum of all elements of the tensor
  Scalar sum_all() const override {
    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    return xt::sum(x)[0];
  }

  /// @brief Check if any values are non-zero
  /// @return Returns true if any value is truthy, false otherwise
  bool any() const override {
    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    bool result;
    // For non-complex types, use regular bool casting
    if constexpr (!is_complex<Scalar>::value) {
      result = xt::any(x);
    }
    // For complex types, implement custom ny
    else {
      throw std::runtime_error("any() not supported on non-integral types.");
    }

    return result;
  }

  void elementwise_add(const details::tensor_impl<Scalar> *other,
                       details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const xtensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());
    auto z = x + y;
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void
  elementwise_multiply(const details::tensor_impl<Scalar> *other,
                       details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const xtensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());
    auto z = x * y;
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void elementwise_modulo(const details::tensor_impl<Scalar> *other,
                          details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const xtensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    // For non-complex types, use regular modulo
    if constexpr (std::is_integral_v<Scalar>) {
      auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
      auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                         other_xt->shape());
      auto z = x % y;
      std::copy(z.begin(), z.end(), result_xt->data());
    }
    // For complex types, implement custom modulo
    else {
      throw std::runtime_error("modulo not supported on non-integral types.");
    }
  }

  void scalar_modulo(Scalar value,
                     details::tensor_impl<Scalar> *result) const override {
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    // For non-complex types, use regular modulo
    if constexpr (std::is_integral_v<Scalar>) {
      auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
      auto z = x % value;
      std::copy(z.begin(), z.end(), result_xt->data());
    }
    // For complex types, implement custom modulo
    else {
      throw std::runtime_error("modulo not supported on non-integral types.");
    }
  }

  void matrix_dot(const details::tensor_impl<Scalar> *other,
                  details::tensor_impl<Scalar> *result) const override {
    auto *other_xt = dynamic_cast<const xtensor<Scalar> *>(other);
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    if (!other_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto y = xt::adapt(other_xt->data(), other_xt->size(), xt::no_ownership(),
                       other_xt->shape());
    auto z = xt::linalg::dot(x, y);
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void
  matrix_vector_product(const details::tensor_impl<Scalar> *vec,
                        details::tensor_impl<Scalar> *result) const override {
    auto *vec_xt = dynamic_cast<const xtensor<Scalar> *>(vec);
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    if (!vec_xt || !result_xt) {
      throw std::runtime_error("Invalid tensor implementation type");
    }

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto v = xt::adapt(vec_xt->data(), vec_xt->size(), xt::no_ownership(),
                       vec_xt->shape());
    auto z = xt::linalg::dot(x, v);
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  void matrix_transpose(details::tensor_impl<Scalar> *result) const override {
    auto *result_xt = dynamic_cast<xtensor<Scalar> *>(result);

    auto x = xt::adapt(m_data, size(), xt::no_ownership(), m_shape);
    auto z = xt::transpose(x, {1, 0});
    std::copy(z.begin(), z.end(), result_xt->data());
  }

  Scalar *data() override { return m_data; }
  const Scalar *data() const override { return m_data; }
  void dump() const override {
    std::cout << xt::adapt(m_data, size(), xt::no_ownership(), m_shape) << '\n';
  }

  void dump_bits() const override {
    if constexpr (is_complex<Scalar>::value) {
      throw std::runtime_error("dump_bits() unsupported for complex types");
    } else if (rank() == 1) {
      // Dump bits on 1 row
      size_t num_elements = m_shape[0];
      for (size_t ix = 0; ix < num_elements; ix++)
        std::cout << (m_data[ix] > 0 ? '1' : '.');
      std::cout << '\n';
    } else if (rank() == 2) {
      // Dump bits as a matrix
      size_t nr = m_shape[0];
      size_t nc = m_shape[1];
      size_t ix = 0;
      for (size_t r = 0; r < nr; r++) {
        for (size_t c = 0; c < nc; c++, ix++)
          std::cout << (m_data[ix] > 0 ? '1' : '.');
        std::cout << '\n';
      }
    } else {
      throw std::runtime_error("dump_bits() unsupported for rank > 2");
    }
  }

  static constexpr auto ScalarAsString = cudaqx::type_to_string<Scalar>();

  /// @brief Custom creator function for xtensor
  /// @param d Pointer to the tensor data
  /// @param s Shape of the tensor
  /// @return A unique pointer to the created xtensor object
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      xtensor<Scalar>, std::string("xtensor") + std::string(ScalarAsString),
      static std::unique_ptr<cudaqx::details::tensor_impl<Scalar>> create(
          const Scalar *d, const std::vector<std::size_t> s) {
        return std::make_unique<xtensor<Scalar>>(d, s);
      })

  /// @brief Destructor for xtensor
  ~xtensor() {
    if (ownsData)
      delete[] m_data;
  }
};

/// @brief Register the xtensor types

#define INSTANTIATE_REGISTRY_TENSOR_IMPL(TYPE)                                 \
  INSTANTIATE_REGISTRY(cudaqx::details::tensor_impl<TYPE>, const TYPE *,       \
                       const std::vector<std::size_t>)

INSTANTIATE_REGISTRY_TENSOR_IMPL(std::complex<double>)
INSTANTIATE_REGISTRY_TENSOR_IMPL(std::complex<float>)
INSTANTIATE_REGISTRY_TENSOR_IMPL(int)
INSTANTIATE_REGISTRY_TENSOR_IMPL(uint8_t)
INSTANTIATE_REGISTRY_TENSOR_IMPL(double)
INSTANTIATE_REGISTRY_TENSOR_IMPL(float)
INSTANTIATE_REGISTRY_TENSOR_IMPL(std::size_t)

template <>
const bool xtensor<std::complex<double>>::registered_ =
    xtensor<std::complex<double>>::register_type();
template <>
const bool xtensor<std::complex<float>>::registered_ =
    xtensor<std::complex<float>>::register_type();
template <>
const bool xtensor<int>::registered_ = xtensor<int>::register_type();
template <>
const bool xtensor<uint8_t>::registered_ = xtensor<uint8_t>::register_type();
template <>
const bool xtensor<double>::registered_ = xtensor<double>::register_type();
template <>
const bool xtensor<float>::registered_ = xtensor<float>::register_type();
template <>
const bool xtensor<std::size_t>::registered_ =
    xtensor<std::size_t>::register_type();

} // namespace cudaqx
