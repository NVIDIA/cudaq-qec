/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// [Begin Documentation]
// Compile and run with:
// nvq++ -lcudaq-qec -lcudaq-qec-decoders stim_dem_decoder.cpp
// ./a.out

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
  const std::string dem_text = R"(error(0.1) D0 L0
error(0.1) D1 L0
error(0.05) D0 D1
error(0.02) D0 ^ D1
)";

  // Decoder construction uses default parsing (use_decomp_suggestions=false):
  // '^' hints in the DEM text are ignored.
  auto decoder = cudaq::qec::get_decoder("single_error_lut", dem_text);
  auto dem = cudaq::qec::dem_from_stim_text(dem_text);

  std::cout << "detectors: " << dem.num_detectors() << "\n";
  std::cout << "error mechanisms (used by decoder): "
            << dem.num_error_mechanisms() << "\n";
  std::cout << "observables: " << dem.num_observables() << "\n";

  // Inspection only: show how many columns the matrix would have if
  // '^' hints were honored. This does not affect the decoder above.
  auto dem_decomposed =
      cudaq::qec::dem_from_stim_text(dem_text, /*use_decomp_suggestions=*/true);

  std::cout << "error mechanisms (if ^ hints honored): "
            << dem_decomposed.num_error_mechanisms() << "\n";

  const std::vector<std::vector<cudaq::qec::float_t>> syndromes = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

  for (const auto &syndrome : syndromes) {
    auto result = decoder->decode(syndrome);

    std::cout << "syndrome [" << syndrome[0] << ", " << syndrome[1] << "]"
              << " -> error [";
    for (std::size_t i = 0; i < result.result.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << result.result[i];
    }
    std::cout << "] -> observable flip [";
    for (std::size_t obs = 0; obs < dem.num_observables(); ++obs) {
      if (obs > 0)
        std::cout << ", ";
      std::uint8_t flip = 0;
      for (std::size_t error = 0; error < result.result.size(); ++error)
        flip ^= static_cast<std::uint8_t>(
            dem.observables_flips_matrix.at({obs, error}) &&
            result.result[error] != 0.0);
      std::cout << static_cast<int>(flip);
    }
    std::cout << "]\n";
  }
}
