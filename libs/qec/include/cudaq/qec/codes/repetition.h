/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudaq/qec/code.h"
#include "cudaq/qec/patch.h"

using namespace cudaqx;

namespace cudaq::qec::repetition {

/// \pure_device_kernel
///
/// @brief Apply Logical X gate to a repetition code patch
/// @param p The patch to apply the X gate to
__qpu__ void x(patch p);

/// \pure_device_kernel
///
/// @brief Prepares the given repetition code in the |0⟩ state
/// @param[in,out] p The quantum patch to initialize
__qpu__ void prep0(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a repetition code patch in the |1⟩ state
/// @param p The patch to prepare
__qpu__ void prep1(patch p);

/// @brief Measures the X and Z stabilizers for the repetition code
/// @param[in] p The quantum patch to measure stabilizers on
/// @param[in] x_stabilizers Vector of qubit indices for X stabilizer
/// measurements
/// @param[in] z_stabilizers Vector of qubit indices for Z stabilizer
/// measurements
/// @return Vector containing the measurement results
__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch p, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers);

/// @brief Implementation of the repetition quantum error correction code
class repetition : public cudaq::qec::code {
protected:
  /// @brief The code distance parameter
  std::size_t distance;

  /// @brief Gets the number of data qubits in the code
  /// @return Number of data qubits
  std::size_t get_num_data_qubits() const override;

  /// @brief Gets the total number of ancilla qubits
  /// @return Total number of ancilla qubits
  std::size_t get_num_ancilla_qubits() const override;

  /// @brief Gets the number of X-basis ancilla qubits
  /// @return Number of X ancilla qubits
  std::size_t get_num_ancilla_x_qubits() const override;

  /// @brief Gets the number of Z-basis ancilla qubits
  /// @return Number of Z ancilla qubits
  std::size_t get_num_ancilla_z_qubits() const override;

  /// @brief Get number of X stabilizer that can be measured
  /// @return Number of X-type stabilizers
  std::size_t get_num_x_stabilizers() const override;

  /// @brief Get number of Z stabilizer that can be measured
  /// @return Number of Z-type stabilizers
  std::size_t get_num_z_stabilizers() const override;

public:
  /// @brief Constructs a repetition code instance
  repetition(const heterogeneous_map &);

  /// @brief Factory function to create repetition code instances
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      repetition, static std::unique_ptr<cudaq::qec::code> create(
                      const cudaqx::heterogeneous_map &options) {
        return std::make_unique<repetition>(options);
      })
};
} // namespace cudaq::qec::repetition
