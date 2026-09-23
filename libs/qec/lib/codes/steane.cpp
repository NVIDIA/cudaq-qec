/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudaq/qec/codes/steane.h"

using cudaq::qec::operation;

namespace cudaq::qec::steane {

steane::steane(const heterogeneous_map &options) : code() {
  operation_encodings.insert(std::make_pair(operation::x, x));
  operation_encodings.insert(std::make_pair(operation::y, y));
  operation_encodings.insert(std::make_pair(operation::z, z));
  operation_encodings.insert(std::make_pair(operation::h, h));
  operation_encodings.insert(std::make_pair(operation::s, s));
  operation_encodings.insert(std::make_pair(operation::cx, cx));
  operation_encodings.insert(std::make_pair(operation::cy, cy));
  operation_encodings.insert(std::make_pair(operation::cz, cz));
  operation_encodings.insert(
      std::make_pair(operation::stabilizer_round, stabilizer));
  operation_encodings.insert(std::make_pair(operation::prep0, prep0));
  operation_encodings.insert(std::make_pair(operation::prep1, prep1));
  operation_encodings.insert(std::make_pair(operation::prepp, prepp));
  operation_encodings.insert(std::make_pair(operation::prepm, prepm));

  m_stabilizers = fromPauliWords(
      {"XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI", "IIZZIZZ"});
  m_pauli_observables = fromPauliWords({"IIIIXXX", "IIIIZZZ"});

  // Sort now to avoid repeated sorts later.
  sortStabilizerOps(m_stabilizers);
  sortStabilizerOps(m_pauli_observables);
}

/// @brief Register the Steane code type
CUDAQ_EXT_PT_REGISTER_TYPE(steane)

} // namespace cudaq::qec::steane
