/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <memory>

#include "cudaq/qis/qkernel.h"
#include "cudaq/qis/qvector.h"

#include "cudaq/qec/noise_model.h"
#include "cudaq/qec/patch.h"
#include "cudaq/qec/stabilizer_utils.h"

#include "cuda-qx/core/extension_point.h"
#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"

using namespace cudaqx;

namespace cudaq::qec {

/// @brief Enum describing all supported logical operations.
enum class operation {
  x,                ///< Logical X gate
  y,                ///< Logical Y gate
  z,                ///< Logical Z gate
  h,                ///< Logical Hadamard gate
  s,                ///< Logical S gate
  cx,               ///< Logical controlled-X gate
  cy,               ///< Logical controlled-Y gate
  cz,               ///< Logical controlled-Z gate
  stabilizer_round, ///< Stabilizer measurement round
  prep0,            ///< Prepare logical |0⟩ state
  prep1,            ///< Prepare logical |1⟩ state
  prepp,            ///< Prepare logical |+⟩ state
  prepm             ///< Prepare logical |-⟩ state
};

/// @brief Base class for quantum error correcting codes in CUDA-Q.
/// @details
/// This class provides the core interface and functionality for implementing
/// quantum error correcting codes in CUDA-Q. It defines the basic operations
/// that any QEC code must support and provides infrastructure for syndrome
/// measurement and error correction experiments.
///
/// To implement a new quantum error correcting code:
/// 1. Create a new class that inherits from code
/// 2. Implement the protected virtual methods:
///    - get_num_data_qubits()
///    - get_num_ancilla_qubits()
///    - get_num_ancilla_x_qubits()
///    - get_num_ancilla_z_qubits()
/// 3. Define quantum kernels for each required logical operation (these are
/// the fault tolerant logical operation implementations)
/// 4. Register the operations in your constructor using the
/// operation_encodings map on the base class
/// 5. Register your new code type using CUDAQ_EXT_PT_REGISTER_TYPE
///
/// Example implementation:
/// @code{.cpp}
/// __qpu__ void x_kernel(patch p);
/// __qpu__ void z_kernel(patch p);
/// class my_code : public qec::code {
/// protected:
///   std::size_t get_num_data_qubits() const override { return 7; }
///   std::size_t get_num_ancilla_qubits() const override { return 6; }
///   std::size_t get_num_ancilla_x_qubits() const override { return 3; }
///   std::size_t get_num_ancilla_z_qubits() const override { return 3; }
///
/// public:
///   my_code(const heterogeneous_map& options) : code() {
///     // Can use user-specified options, e.g. auto d =
///     options.get<int>("distance");
///     operation_encodings.insert(std::make_pair(operation::x, x_kernel));
///     operation_encodings.insert(std::make_pair(operation::z, z_kernel));
///     // Register other required operations...
///
///     // Define the default stabilizers!
///     m_stabilizers = qec::stabilizers({"XXXX", "ZZZZ"});
///   }
///
///   CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
///     my_code,
///     static std::unique_ptr<qec::code> create(const heterogeneous_map
/// &options) { return std::make_unique<my_code>(options);
///     }
///   )
/// };
///
/// CUDAQ_EXT_PT_REGISTER_TYPE(my_code)
/// @endcode
/// @brief Supported quantum operations for error correcting codes
class code : public cudaqx::extension_point<code, const heterogeneous_map &> {
public:
  /// @brief Type alias for single qubit quantum kernels
  using one_qubit_encoding = cudaq::qkernel<void(patch)>;

  /// @brief Type alias for two qubit quantum kernels
  using two_qubit_encoding = cudaq::qkernel<void(patch, patch)>;

  /// @brief Type alias for stabilizer measurement kernels. The two vector
  /// arguments are the flattened X and Z stabilizer schedule matrices (see
  /// get_stabilizer_schedule_x/z()): entry 0 = no support, entry k >= 1 =
  /// interaction at timestep k.
  using stabilizer_round = cudaq::qkernel<std::vector<cudaq::measure_result>(
      patch, const std::vector<std::size_t> &,
      const std::vector<std::size_t> &)>;

  /// @brief Type alias for quantum operation encodings
  using encoding =
      std::variant<one_qubit_encoding, two_qubit_encoding, stabilizer_round>;

protected:
  /// @brief Map storing the quantum kernels for each supported operation
  std::unordered_map<operation, encoding> operation_encodings;

  /// @brief Stabilizer generators for the code
  std::vector<cudaq::spin_op_term> m_stabilizers;

  /// @brief Pauli Logical operators
  std::vector<cudaq::spin_op_term> m_pauli_observables;

  std::vector<cudaq::spin_op_term>
  fromPauliWords(const std::vector<std::string> &words) {
    std::vector<cudaq::spin_op_term> ops;
    for (auto &os : words)
      ops.emplace_back(cudaq::spin_op::from_word(os));
    sortStabilizerOps(ops);
    return ops;
  }

public:
  /// @brief Get the number of physical data qubits needed for the code
  /// @return Number of data qubits
  virtual std::size_t get_num_data_qubits() const = 0;

  /// @brief Get the total number of ancilla qubits needed
  /// @return Total number of ancilla qubits
  virtual std::size_t get_num_ancilla_qubits() const = 0;

  /// @brief Get number of ancilla qubits needed for X stabilizer measurements
  /// @return Number of X-type ancilla qubits
  virtual std::size_t get_num_ancilla_x_qubits() const = 0;

  /// @brief Get number of ancilla qubits needed for Z stabilizer measurements
  /// @return Number of Z-type ancilla qubits
  virtual std::size_t get_num_ancilla_z_qubits() const = 0;

  /// @brief Get number of X stabilizer that can be measured
  /// @return Number of X-type stabilizers
  virtual std::size_t get_num_x_stabilizers() const = 0;

  /// @brief Get number of Z stabilizer that can be measured
  /// @return Number of Z-type stabilizers
  virtual std::size_t get_num_z_stabilizers() const = 0;

  code() = default;
  virtual ~code() {}

  /// @brief Factory method to create a code instance with specified
  /// stabilizers
  /// @param name Name of the code to create
  /// @param stabilizers Stabilizer generators for the code
  /// @param options Optional code-specific configuration options
  /// @return Unique pointer to created code instance
  static std::unique_ptr<code>
  get(const std::string &name,
      const std::vector<cudaq::spin_op_term> &stabilizers,
      const heterogeneous_map options = {});

  /// @brief Factory method to create a code instance
  /// @param name Name of the code to create
  /// @param options Optional code-specific configuration options
  /// @return Unique pointer to created code instance
  static std::unique_ptr<code> get(const std::string &name,
                                   const heterogeneous_map options = {});

  /// @brief Get the full parity check matrix H = (Hx | Hz)
  /// @return Tensor representing the parity check matrix
  cudaqx::tensor<uint8_t> get_parity() const;

  /// @brief Get the X component of the parity check matrix
  /// @return Tensor representing Hx
  cudaqx::tensor<uint8_t> get_parity_x() const;

  /// @brief Get the Z component of the parity check matrix
  /// @return Tensor representing Hz
  cudaqx::tensor<uint8_t> get_parity_z() const;

  /// @brief Get the X-stabilizer schedule matrix passed to the code's
  /// stabilizer_round kernel. Same shape and support pattern as
  /// get_parity_x(); entry 0 means no support, entry k >= 1 means the
  /// ancilla-data interaction executes at timestep k, so a code can encode an
  /// optimized (e.g. hook-error-aware) gate order.
  /// @return Tensor of scheduled interactions; the default implementation
  /// returns get_parity_x() unchanged (every interaction at timestep 1, i.e.
  /// ascending qubit-index order).
  virtual cudaqx::tensor<uint8_t> get_stabilizer_schedule_x() const {
    return get_parity_x();
  }

  /// @brief Get the Z-stabilizer schedule matrix passed to the code's
  /// stabilizer_round kernel. See get_stabilizer_schedule_x().
  /// @return Tensor of scheduled interactions; the default implementation
  /// returns get_parity_z() unchanged.
  virtual cudaqx::tensor<uint8_t> get_stabilizer_schedule_z() const {
    return get_parity_z();
  }

  /// @brief Get Lx stacked on Lz
  /// @return Tensor representing pauli observables
  cudaqx::tensor<uint8_t> get_pauli_observables_matrix() const;

  /// @brief Get the Lx observables
  /// @return Tensor representing Lx
  cudaqx::tensor<uint8_t> get_observables_x() const;

  /// @brief Get the Lz observables
  /// @return Tensor representing Lz
  cudaqx::tensor<uint8_t> get_observables_z() const;

  /// @brief Get the stabilizer generators
  /// @return Reference to stabilizers
  const std::vector<cudaq::spin_op_term> &get_stabilizers() const {
    return m_stabilizers;
  }

  /// @brief Return true if this code contains the given operation encoding.
  bool contains_operation(operation op) const {
    return operation_encodings.find(op) != operation_encodings.end();
  }

  // Return the CUDA-Q kernel for the given operation encoding.
  // User must provide the qkernel type (stabilizer_round, one_qubit_encoding,
  // or two_qubit_encoding) as the template type.
  template <typename T>
  auto &&get_operation(operation op) const {
    auto iter = operation_encodings.find(op);
    if (iter == operation_encodings.end())
      throw std::runtime_error(
          "code::get_operation error - could not find operation encoding " +
          std::to_string(static_cast<int>(op)));

    return std::get<T>(iter->second);
  }
};

/// Factory function to create a code instance with specified stabilizers
/// @param name Name of the code
/// @param options Optional code-specific configuration parameters
/// @return Unique pointer to the created code instance
std::unique_ptr<code> get_code(const std::string &name,
                               const heterogeneous_map options = {});

/// Factory function to create a code instance with specified stabilizers
/// @param name Name of the code
/// @param stab stabilizers
/// @param options Optional code-specific configuration parameters
/// @return Unique pointer to the created code instance
std::unique_ptr<code> get_code(const std::string &name,
                               const std::vector<cudaq::spin_op_term> &stab,
                               const heterogeneous_map options = {});

/// Get a list of available quantum error correcting codes
/// @return Vector of strings containing names of available codes
std::vector<std::string> get_available_codes();

} // namespace cudaq::qec
