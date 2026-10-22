/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudaq/qec/code.h"

#include "device/memory_circuit.h"

INSTANTIATE_REGISTRY(cudaq::qec::code, const cudaqx::heterogeneous_map &)

namespace cudaq::qec {

std::unique_ptr<code>
code::get(const std::string &name,
          const std::vector<cudaq::spin_op_term> &_stabilizers,
          const heterogeneous_map options) {
  auto [mutex, registry] = get_registry();
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error("invalid qec_code requested: " + name);
  auto ret = iter->second(options);
  ret->m_stabilizers = _stabilizers;
  return ret;
}

std::unique_ptr<code> code::get(const std::string &name,
                                const heterogeneous_map options) {
  auto [mutex, registry] = get_registry();
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error("invalid qec_code requested: " + name);
  auto ret = iter->second(options);
  return ret;
}

cudaqx::tensor<uint8_t> code::get_parity() const {
  return to_parity_matrix(m_stabilizers);
}
cudaqx::tensor<uint8_t> code::get_parity_x() const {
  return to_parity_matrix(m_stabilizers, stabilizer_type::X);
}

cudaqx::tensor<uint8_t> code::get_parity_z() const {
  return to_parity_matrix(m_stabilizers, stabilizer_type::Z);
}

cudaqx::tensor<uint8_t> code::get_pauli_observables_matrix() const {
  return to_parity_matrix(m_pauli_observables);
}

cudaqx::tensor<uint8_t> code::get_observables_x() const {
  return to_parity_matrix(m_pauli_observables, stabilizer_type::X);
}

cudaqx::tensor<uint8_t> code::get_observables_z() const {
  return to_parity_matrix(m_pauli_observables, stabilizer_type::Z);
}

std::unique_ptr<code> get_code(const std::string &name,
                               const std::vector<cudaq::spin_op_term> &stab,
                               const heterogeneous_map options) {
  return code::get(name, stab, options);
}

std::unique_ptr<code> get_code(const std::string &name,
                               const heterogeneous_map options) {
  return code::get(name, options);
}

std::vector<std::string> get_available_codes() {
  return code::get_registered();
}

} // namespace cudaq::qec
