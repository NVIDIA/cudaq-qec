/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// [Begin Documentation]
// Compile and run with:
// nvq++ --target=stim -lcudaq-qec -lcudaq-qec-decoders circuit_level_noise.cpp
// ./a.out

#include "cudaq.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/noise_model.h"

int main() {
  // Choose a QEC code
  auto steane = cudaq::qec::get_code("steane");

  // Access the parity check matrix
  auto H = steane->get_parity();
  std::cout << "H:\n";
  H.dump();

  // Data qubits the logical Z observable is supported on
  auto Lz = steane->get_observables_z();
  std::cout << "Lz:\n";
  Lz.dump();

  // How many shots to run the experiment
  int nShots = 3;
  // For each shot, how many rounds of stabilizer measurements
  int nRounds = 4;

  // can set seed for reproducibility
  // cudaq::set_random_seed(1337);
  cudaq::noise_model noise;

  // Add a depolarization noise channel after each cx gate
  noise.add_all_qubit_channel("x", cudaq::depolarization2(/*probability*/ 0.01),
                              /*numControls*/ 1);

  // Perform a noisy z-basis memory circuit experiment
  auto [syndromes, data] = cudaq::qec::sample_memory_circuit(
      *steane, cudaq::qec::operation::prep0, nShots, nRounds, noise);

  // Each row holds one shot's detectors: the syndrome differences between
  // consecutive rounds, plus the boundary detectors.
  std::cout << "syndromes:\n";
  syndromes.dump();

  // With noise, Lz will sometimes be flipped
  std::cout << "data:\n";
  data.dump();

  // Use z-measurements on data qubits to determine the logical mz
  // In an x-basis experiment, use Lx.
  auto logical_mz = Lz.dot(data.transpose()) % 2;
  std::cout << "logical_mz each shot:\n";
  logical_mz.dump();

  // The detector error model (DEM) of the same circuit and noise relates each
  // error mechanism to the detectors it flips and to the logical observables
  // it flips. Its detectors match the columns of the sampled syndromes.
  auto dem = cudaq::qec::dem_from_memory_circuit(
      *steane, cudaq::qec::operation::prep0, nRounds, noise);

  // Select a decoder that decodes a whole shot against the DEM
  auto decoder =
      cudaq::qec::get_decoder("single_error_lut", dem.detector_error_matrix);

  // Start a loop to count the number of logical errors
  size_t numLerrors = 0;
  size_t numDetectors = syndromes.shape()[1];
  for (size_t shot = 0; shot < nShots; ++shot) {
    std::cout << "shot: " << shot << "\n";

    // Access this shot's row of the syndrome tensor
    cudaqx::tensor<uint8_t> syndrome({numDetectors});
    syndrome.borrow(syndromes.data() + numDetectors * shot);
    std::cout << "syndrome:\n";
    syndrome.dump();

    // Decode the syndrome into the most likely error mechanisms
    auto result = decoder->decode(syndrome);
    cudaqx::tensor<uint8_t> result_tensor;
    cudaq::qec::convert_vec_soft_to_tensor_hard(result.result, result_tensor);
    std::cout << "decode result:\n";
    result_tensor.dump();

    // Predict whether those errors flipped the logical observable
    auto predicted_flip = dem.observables_flips_matrix.dot(result_tensor) % 2;
    std::cout << "predicted observable flip:\n";
    predicted_flip.dump();

    // prep0 means we expected to measure out 0.
    uint8_t expected_mz = 0;
    // Apply the predicted flip to our logical measurement
    uint8_t corrected_mz =
        (logical_mz.at({0, shot}) + predicted_flip.at({0})) % 2;

    // Check if Logical_mz + predicted flip = 0?
    std::cout << "Corrected readout: " << +corrected_mz << "\n";
    std::cout << "Expected readout: " << +expected_mz << "\n";
    if (corrected_mz != expected_mz)
      numLerrors++;
    std::cout << "\n";
  }

  std::cout << "numLogicalErrors: " << numLerrors << "\n";
}
