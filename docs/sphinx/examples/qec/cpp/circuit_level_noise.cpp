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

  // Access the logical observables
  auto observables = steane->get_pauli_observables_matrix();
  auto Lz = steane->get_observables_z();

  // Data qubits the logical Z observable is supported on
  std::cout << "Lz:\n";
  Lz.dump();

  // Observables are stacked as Z over X for mat-vec multiplication
  std::cout << "Obs:\n";
  observables.dump();

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

  // With noise, many syndromes will flip each QEC cycle, these are the
  // syndrome differences from the previous cycle.
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

  // Select a decoder
  auto decoder = cudaq::qec::get_decoder("single_error_lut", H);

  // Initialize a pauli_frame to track the logical errors
  cudaqx::tensor<uint8_t> pauli_frame({observables.shape()[0]});

  // Start a loop to count the number of logical errors
  size_t numLerrors = 0;
  for (size_t shot = 0; shot < nShots; ++shot) {
    std::cout << "shot: " << shot << "\n";

    for (size_t round = 0; round < nRounds; ++round) {
      std::cout << "round: " << round << "\n";

      // Access one row of the syndrome tensor
      size_t count = shot * nRounds + round;
      size_t stride = syndromes.shape()[1];
      cudaqx::tensor<uint8_t> syndrome({stride});
      syndrome.borrow(syndromes.data() + stride * count);
      std::cout << "syndrome:\n";
      syndrome.dump();

      // Decode the syndrome
      auto result = decoder->decode(syndrome);
      cudaqx::tensor<uint8_t> result_tensor;
      cudaq::qec::convert_vec_soft_to_tensor_hard(result.result, result_tensor);
      std::cout << "decode result:\n";
      result_tensor.dump();

      // See if the decoded result anti-commutes with observables
      auto decoded_observables = observables.dot(result_tensor);
      std::cout << "decoded observable:\n";
      decoded_observables.dump();

      // update from previous stabilizer round
      pauli_frame = (pauli_frame + decoded_observables) % 2;
      std::cout << "pauli frame:\n";
      pauli_frame.dump();
    }

    // prep0 means we expected to measure out 0.
    uint8_t expected_mz = 0;
    // Apply the pauli frame correction to our logical measurement
    uint8_t corrected_mz = (logical_mz.at({0, shot}) + pauli_frame.at({0})) % 2;

    // Check if Logical_mz + pauli_frame_X = 0?
    std::cout << "Corrected readout: " << +corrected_mz << "\n";
    std::cout << "Expected readout: " << +expected_mz << "\n";
    if (corrected_mz != expected_mz)
      numLerrors++;
    std::cout << "\n";
  }

  std::cout << "numLogicalErrors: " << numLerrors << "\n";
}
