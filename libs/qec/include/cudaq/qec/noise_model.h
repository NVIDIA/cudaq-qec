/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common/NoiseModel.h"

#include <optional>

namespace cudaq::qec {

/// @brief Namespace containing utility functions for quantum error correction
namespace details {

/// @brief Typedef for a matrix wrapper using std::vector<cudaq::complex>
using matrix_wrapper = std::vector<cudaq::complex>;

/// @brief Compute the Kronecker product of two matrices
///
/// @param A First matrix
/// @param rowsA Number of rows in matrix A
/// @param colsA Number of columns in matrix A
/// @param B Second matrix
/// @param rowsB Number of rows in matrix B
/// @param colsB Number of columns in matrix B
/// @return matrix_wrapper Result of the Kronecker product
inline matrix_wrapper kron(const matrix_wrapper &A, int rowsA, int colsA,
                           const matrix_wrapper &B, int rowsB, int colsB) {
  matrix_wrapper C((rowsA * rowsB) * (colsA * colsB));
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsA; ++j) {
      for (int k = 0; k < rowsB; ++k) {
        for (int l = 0; l < colsB; ++l) {
          C[(i * rowsB + k) * (colsA * colsB) + (j * colsB + l)] =
              A[i * colsA + j] * B[k * colsB + l];
        }
      }
    }
  }
  return C;
}

} // namespace details

/// @brief Two-qubit bit flip channel implementation
class two_qubit_bitflip : public cudaq::kraus_channel {
public:
  /// @brief Construct a two qubit kraus channel that applies a bit flip on
  /// either qubit independently.
  ///
  /// @param probability The probability of a bit flip occurring
  ///
  two_qubit_bitflip(const cudaq::real probability) : kraus_channel() {
    std::vector<cudaq::complex> K0{std::sqrt(1 - probability), 0, 0,
                                   std::sqrt(1 - probability)},
        K1{0, std::sqrt(probability), std::sqrt(probability), 0};
    auto E0 = details::kron(K0, 2, 2, K0, 2, 2);
    auto E1 = details::kron(K0, 2, 2, K1, 2, 2);
    auto E2 = details::kron(K1, 2, 2, K0, 2, 2);
    auto E3 = details::kron(K1, 2, 2, K1, 2, 2);

    // Set the ops vector to contain only the Kronecker product
    ops = {std::move(E0), std::move(E1), std::move(E2), std::move(E3)};
    this->parameters.push_back(probability);
    noise_type = cudaq::noise_model_type::bit_flip_channel;
    validateCompleteness();
  }
};

class two_qubit_depolarization : public cudaq::kraus_channel {
public:
  /// @brief Construct a two qubit kraus channel that applies a depolarization
  /// channel on either qubit independently.
  ///
  /// @param probability The probability of a bit flip occurring
  ///
  two_qubit_depolarization(const cudaq::real probability) : kraus_channel() {
    auto three = static_cast<cudaq::real>(3.);
    auto negOne = static_cast<cudaq::real>(-1.);
    std::vector<std::vector<cudaq::complex>> singleQubitKraus = {
        {std::sqrt(1 - probability), 0, 0, std::sqrt(1 - probability)},
        {0, std::sqrt(probability / three), std::sqrt(probability / three), 0},
        {0, cudaq::complex{0, negOne * std::sqrt(probability / three)},
         cudaq::complex{0, std::sqrt(probability / three)}, 0},
        {std::sqrt(probability / three), 0, 0,
         negOne * std::sqrt(probability / three)}};

    // Generate 2-qubit Kraus operators
    for (const auto &k1 : singleQubitKraus) {
      for (const auto &k2 : singleQubitKraus) {
        ops.push_back(details::kron(k1, 2, 2, k2, 2, 2));
      }
    }
    this->parameters.push_back(probability);
    noise_type = cudaq::noise_model_type::depolarization_channel;
    validateCompleteness();
  }
};
} // namespace cudaq::qec
