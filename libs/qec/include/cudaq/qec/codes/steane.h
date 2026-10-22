/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cudaq/qec/code.h"
#include "cudaq/qec/patch.h"

using namespace cudaqx;

namespace cudaq::qec::steane {

/// \pure_device_kernel
///
/// @brief Apply X gate to a Steane code patch
/// @param p The patch to apply the X gate to
__qpu__ void x(patch p);

/// \pure_device_kernel
///
/// @brief Apply Y gate to a Steane code patch
/// @param p The patch to apply the Y gate to
__qpu__ void y(patch p);

/// \pure_device_kernel
///
/// @brief Apply Z gate to a Steane code patch
/// @param p The patch to apply the Z gate to
__qpu__ void z(patch p);

/// \pure_device_kernel
///
/// @brief Apply Hadamard gate to a Steane code patch
/// @param p The patch to apply the Hadamard gate to
__qpu__ void h(patch p);

/// \pure_device_kernel
///
/// @brief Apply S gate to a Steane code patch
/// @param p The patch to apply the S gate to
__qpu__ void s(patch p);

/// \pure_device_kernel
///
/// @brief Apply controlled-X gate between two Steane code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cx(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Apply controlled-Y gate between two Steane code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cy(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Apply controlled-Z gate between two Steane code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cz(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Prepare a Steane code patch in the |0⟩ state
/// @param p The patch to prepare
__qpu__ void prep0(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a Steane code patch in the |1⟩ state
/// @param p The patch to prepare
__qpu__ void prep1(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a Steane code patch in the |+⟩ state
/// @param p The patch to prepare
__qpu__ void prepp(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a Steane code patch in the |-⟩ state
/// @param p The patch to prepare
__qpu__ void prepm(patch p);

/// \pure_device_kernel
///
/// @brief Perform stabilizer measurements on a Steane code patch
/// @param p The patch to measure
/// @param x_stabilizers Indices of X stabilizers to measure
/// @param z_stabilizers Indices of Z stabilizers to measure
/// @return Vector of measurement results
__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch p, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers);

/// @brief Steane code implementation
class steane : public cudaq::qec::code {
protected:
  /// @brief Get the number of data qubits in the Steane code
  /// @return Number of data qubits (7 for Steane code)
  std::size_t get_num_data_qubits() const override { return 7; }

  /// @brief Get the number of total ancilla qubits in the Steane code
  /// @return Number of data qubits (6 for Steane code)
  std::size_t get_num_ancilla_qubits() const override { return 6; }

  /// @brief Get the number of X ancilla qubits in the Steane code
  /// @return Number of data qubits (3 for Steane code)
  std::size_t get_num_ancilla_x_qubits() const override { return 3; }

  /// @brief Get the number of Z ancilla qubits in the Steane code
  /// @return Number of data qubits (3 for Steane code)
  std::size_t get_num_ancilla_z_qubits() const override { return 3; }

  /// @brief Get number of X stabilizer that can be measured
  /// @return Number of X-type stabilizers
  std::size_t get_num_x_stabilizers() const override { return 3; }

  /// @brief Get number of Z stabilizer that can be measured
  /// @return Number of Z-type stabilizers
  std::size_t get_num_z_stabilizers() const override { return 3; }

public:
  /// @brief Constructor for the Steane code
  steane(const heterogeneous_map &);

  /// @brief Extension creator function for the Steane code
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      steane, static std::unique_ptr<cudaq::qec::code> create(
                  const cudaqx::heterogeneous_map &options) {
        return std::make_unique<steane>(options);
      })
};

} // namespace cudaq::qec::steane
