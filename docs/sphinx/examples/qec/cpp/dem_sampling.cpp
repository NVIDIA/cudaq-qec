/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                 *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/
// [Begin Documentation]
// DEM Sampling — sample errors and syndromes from a detector error model.
//
// Compile and run with:
// nvq++ -lcudaq-qec dem_sampling.cpp
// ./a.out

#include <cstdint>
#include <iostream>
#include <vector>

#include "cudaq/qec/dem_sampling.h"

int main() {
  // [3,1] repetition code check matrix:
  //   H = | 1 1 0 |
  //       | 0 1 1 |
  std::vector<uint8_t> H_data = {1, 1, 0, 0, 1, 1};
  size_t num_checks = 2;
  size_t num_mechanisms = 3;

  cudaqx::tensor<uint8_t> H({num_checks, num_mechanisms});
  H.copy(H_data.data(), H.shape());

  std::vector<double> error_probs = {0.05, 0.10, 0.05};
  size_t num_shots = 10;
  unsigned seed = 42;

  // CPU sampling
  auto [syndromes, errors] =
      cudaq::qec::dem_sampler::cpu::sample_dem(H, num_shots, error_probs, seed);

  std::cout << "Syndromes [" << syndromes.shape()[0] << " x "
            << syndromes.shape()[1] << "]:\n";
  for (size_t shot = 0; shot < num_shots; shot++) {
    for (size_t c = 0; c < num_checks; c++)
      std::cout << static_cast<int>(syndromes.at({shot, c})) << " ";
    std::cout << "\n";
  }

  std::cout << "\nErrors [" << errors.shape()[0] << " x " << errors.shape()[1]
            << "]:\n";
  for (size_t shot = 0; shot < num_shots; shot++) {
    for (size_t e = 0; e < num_mechanisms; e++)
      std::cout << static_cast<int>(errors.at({shot, e})) << " ";
    std::cout << "\n";
  }

  // Verify: syndromes == (errors * H^T) mod 2
  bool ok = true;
  for (size_t shot = 0; shot < num_shots && ok; shot++) {
    for (size_t c = 0; c < num_checks && ok; c++) {
      uint8_t expected = 0;
      for (size_t e = 0; e < num_mechanisms; e++)
        expected ^= errors.at({shot, e}) & H.at({c, e});
      if (syndromes.at({shot, c}) != expected)
        ok = false;
    }
  }
  std::cout << "\nVerification: " << (ok ? "PASSED" : "FAILED") << "\n";

  return ok ? 0 : 1;
}
