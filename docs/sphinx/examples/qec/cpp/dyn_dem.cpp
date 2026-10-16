/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// [Begin Documentation]
// Dynamic DEM construction from CSS matrices and composable DEM chunks.
//
// Compile and run with:
// nvq++ -lcudaq-qec -lcudaq-qec-decoders dyn_dem.cpp
// ./a.out

#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/dem_construction.h"
#include "cudaq/qec/extended_dem.h"

#include <iostream>
#include <vector>

int main() {
  // d=3 repetition code (Z-basis): H_Z = [[1,1,0],[0,1,1]], L_Z = [[1,0,0]].
  cudaq::qec::css_code_matrices code;
  code.hz = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      2, 3, {{0}, {0, 1}, {1}});
  code.lz =
      cudaq::qec::sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});

  cudaq::qec::css_noise_params noise;
  noise.px = 0.01;
  noise.pm = 0.005;

  constexpr std::size_t num_rounds = 5;

  // Monolithic T-round DEM (code-capacity + measurement errors).
  auto flat = cudaq::qec::dem_from_css_matrices(code, noise, num_rounds);
  std::cout << "flat DEM: " << flat.num_detectors() << " detectors, "
            << flat.num_error_mechanisms() << " faults, "
            << flat.num_observables() << " observables\n";

  // Same experiment as T one-round chunks closed in one O(T) pass.
  auto chunk = cudaq::qec::extended_dem_from_css_matrices(code, noise);
  std::vector<cudaq::qec::extended_dem> chunks(num_rounds, chunk);
  auto closed = cudaq::qec::dem_close_all(chunks);

  if (closed.num_detectors() != flat.num_detectors() ||
      closed.num_error_mechanisms() != flat.num_error_mechanisms()) {
    std::cerr << "dem_close_all does not match dem_from_css_matrices\n";
    return 1;
  }
  std::cout << "dem_close_all matches dem_from_css_matrices\n";

  auto detector_round = cudaq::qec::dem_chunks_to_detector_round(chunks);
  auto d_sparse = cudaq::qec::dem_chunks_to_d_sparse(chunks);
  std::cout << "detector_round length " << detector_round.size()
            << ", D_sparse rows " << d_sparse.size() << "\n";
  return 0;
}
// [End Documentation]
