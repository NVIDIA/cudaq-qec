/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq.h"
#include "cudaq/qec/patch.h"

// using qec::patch;

namespace cudaq::qec::steane {

__qpu__ void x(patch logicalQubit) {
  x(logicalQubit.data[4], logicalQubit.data[5], logicalQubit.data[6]);
}
__qpu__ void y(patch logicalQubit) { y(logicalQubit.data); }
__qpu__ void z(patch logicalQubit) {
  z(logicalQubit.data[4], logicalQubit.data[5], logicalQubit.data[6]);
}
__qpu__ void h(patch logicalQubit) { h(logicalQubit.data); }
__qpu__ void s(patch logicalQubit) { s(logicalQubit.data); }

__qpu__ void cx(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < 7; i++) {
    x<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

__qpu__ void cy(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < 7; i++) {
    y<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

__qpu__ void cz(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < 7; i++) {
    z<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

__qpu__ void prep0(patch logicalQubit) {
  h(logicalQubit.data[0], logicalQubit.data[4], logicalQubit.data[6]);
  x<cudaq::ctrl>(logicalQubit.data[0], logicalQubit.data[1]);
  x<cudaq::ctrl>(logicalQubit.data[4], logicalQubit.data[5]);
  x<cudaq::ctrl>(logicalQubit.data[6], logicalQubit.data[3]);
  x<cudaq::ctrl>(logicalQubit.data[6], logicalQubit.data[5]);
  x<cudaq::ctrl>(logicalQubit.data[4], logicalQubit.data[2]);
  x<cudaq::ctrl>(logicalQubit.data[0], logicalQubit.data[3]);
  x<cudaq::ctrl>(logicalQubit.data[4], logicalQubit.data[1]);
  x<cudaq::ctrl>(logicalQubit.data[3], logicalQubit.data[2]);
}

__qpu__ void prep1(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
}

__qpu__ void prepp(patch logicalQubit) {
  prep0(logicalQubit);
  h(logicalQubit.data);
}

__qpu__ void prepm(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
  h(logicalQubit.data);
}

__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch logicalQubit, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers) {
  h(logicalQubit.ancx);
  for (std::size_t xi = 0; xi < logicalQubit.ancx.size(); ++xi)
    for (std::size_t di = 0; di < logicalQubit.data.size(); ++di)
      if (x_stabilizers[xi * logicalQubit.data.size() + di] == 1)
        cudaq::x<cudaq::ctrl>(logicalQubit.ancx[xi], logicalQubit.data[di]);
  h(logicalQubit.ancx);

  // Now apply z_stabilizer circuit
  for (size_t zi = 0; zi < logicalQubit.ancz.size(); ++zi)
    for (size_t di = 0; di < logicalQubit.data.size(); ++di)
      if (z_stabilizers[zi * logicalQubit.data.size() + di] == 1)
        cudaq::x<cudaq::ctrl>(logicalQubit.data[di], logicalQubit.ancz[zi]);

  // S = (S_X, S_Z), (x flip syndromes, z flip syndrones).
  // x flips are triggered by z-stabilizers (ancz)
  // z flips are triggered by x-stabilizers (ancx)
  auto results = mz(logicalQubit.ancz, logicalQubit.ancx);

  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);

  return results;
}

} // namespace cudaq::qec::steane
