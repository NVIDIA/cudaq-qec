/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudaq/qec/codes/repetition.h"

namespace cudaq::qec::repetition {

__qpu__ void x(patch logicalQubit) { x(logicalQubit.data); }

__qpu__ void prep0(patch logicalQubit) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++)
    reset(logicalQubit.data[i]);
}

__qpu__ void prep1(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
}

__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch logicalQubit, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers) {

  // cnot between every data qubit
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    cudaq::x<cudaq::ctrl>(logicalQubit.data[i], logicalQubit.ancz[i]);

  for (std::size_t i = 1; i < logicalQubit.data.size(); i++)
    cudaq::x<cudaq::ctrl>(logicalQubit.data[i], logicalQubit.ancz[i - 1]);

  auto results = mz(logicalQubit.ancz);

  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);

  return results;
}

} // namespace cudaq::qec::repetition
