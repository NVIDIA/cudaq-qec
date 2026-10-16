/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <array>
#include <cmath>
#include <deque>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "cudaq.h"
#include "cudaq/algorithms/dem.h"

#include "device/memory_circuit.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/patch.h"
#include "cudaq/qec/pcm_utils.h"

TEST(QECCodeTester, checkRepetitionNoiseStim) {

  auto repetition = cudaq::qec::get_code(
      "repetition", cudaqx::heterogeneous_map{{"distance", 9}});
  {
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1),
                                /*num_controls=*/1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*repetition, 2, 2, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();
    EXPECT_EQ(syndromes.shape()[0], 2u);
    EXPECT_EQ(syndromes.shape()[1], 24u); // numDetectors

    // Should have some 1s since it's noisy
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum > 0);
  }
  {
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.1),
                                /*num_controls=*/1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*repetition, 2, 2, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Should have some 1s since it's noisy
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum > 0);
  }
}

namespace {

/// @brief Splits a `sample_memory_circuit` syndrome tensor's columns by
/// stabilizer type. Returns (ancz-derived sum, ancx-derived sum).
std::pair<std::size_t, std::size_t>
sumDetectorsByType(const cudaqx::tensor<uint8_t> &syndromes,
                   std::size_t numRounds, std::size_t numZStabs,
                   std::size_t numXStabs, bool is_z_prep) {
  const std::size_t numFixed = is_z_prep ? numZStabs : numXStabs;
  const std::size_t numSynPerRound = numZStabs + numXStabs;
  const std::size_t numInteriorRounds = numRounds - 1;
  const std::size_t numCols = syndromes.shape()[1];

  std::size_t z_type_sum = 0;
  std::size_t x_type_sum = 0;
  for (std::size_t i = 0; i < syndromes.shape()[0]; i++) {
    std::size_t &fixed_sum = is_z_prep ? z_type_sum : x_type_sum;
    // Boundary detectors
    for (std::size_t b = 0; b < numFixed; b++) {
      fixed_sum += syndromes.at({i, b});
      fixed_sum += syndromes.at({i, numCols - numFixed + b});
    }
    // Interior round detectors
    for (std::size_t round = 0; round < numInteriorRounds; round++) {
      std::size_t base = numFixed + round * numSynPerRound;
      for (std::size_t z = 0; z < numZStabs; z++)
        z_type_sum += syndromes.at({i, base + z});
      for (std::size_t x = 0; x < numXStabs; x++)
        x_type_sum += syndromes.at({i, base + numZStabs + x});
    }
  }
  return {z_type_sum, x_type_sum};
}

} // namespace

TEST(QECCodeTester, checkSteaneNoiseStim) {

  auto steane = cudaq::qec::get_code("steane");
  int nShots = 10;
  int nRounds = 3;
  size_t numAncz = steane->get_num_ancilla_z_qubits();
  size_t numAncx = steane->get_num_ancilla_x_qubits();
  {
    // two qubit bitflip
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1), 1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*steane, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();
    EXPECT_EQ(syndromes.shape()[0], nShots); // numShots
    EXPECT_EQ(syndromes.shape()[1], 18);     // numDetectors

    // ancx-derived (X-type) detectors must be exactly zero, while the
    // ancz-derived (Z-type) detectors should fire.
    auto [z_type_sum, x_type_sum] = sumDetectorsByType(
        syndromes, nRounds, numAncz, numAncx, /*is_z_prep=*/true);
    EXPECT_TRUE(z_type_sum > 0);
    EXPECT_TRUE(x_type_sum == 0);
  }
  {
    // two qubit depol
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.1),
                                1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*steane, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Depolarizing noise can introduce both X- and Z-type errors, so both
    // detector types should fire.
    auto [z_type_sum, x_type_sum] = sumDetectorsByType(
        syndromes, nRounds, numAncz, numAncx, /*is_z_prep=*/true);
    EXPECT_TRUE(z_type_sum > 0);
    EXPECT_TRUE(x_type_sum > 0);
  }
  {
    // one qubit bitflip
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::bit_flip_channel(0.1));

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();
    // This noise only touches ancx (every round) and data (only at the
    // prep/readout boundary), so Z-type (ancz) detectors can never fire.
    auto [z_type_sum, x_type_sum] = sumDetectorsByType(
        syndromes, nRounds, numAncz, numAncx, /*is_z_prep=*/false);
    EXPECT_TRUE(z_type_sum == 0);
    EXPECT_TRUE(x_type_sum > 0);
  }
  {
    // one qubit phase
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::phase_flip_channel(0.1));

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Same "h"-gate-only argument as above: Z-type detectors can never
    // fire for this circuit/noise combination.
    auto [z_type_sum, x_type_sum] = sumDetectorsByType(
        syndromes, nRounds, numAncz, numAncx, /*is_z_prep=*/false);
    EXPECT_TRUE(z_type_sum == 0);
    EXPECT_TRUE(x_type_sum > 0);
  }
  {
    // one qubit depol
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.1));

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Same "h"-gate-only argument as above: Z-type detectors can never
    // fire for this circuit/noise combination.
    auto [z_type_sum, x_type_sum] = sumDetectorsByType(
        syndromes, nRounds, numAncz, numAncx, /*is_z_prep=*/false);
    EXPECT_TRUE(z_type_sum == 0);
    EXPECT_TRUE(x_type_sum > 0);
  }
}

TEST(QECCodeTester, checkSampleMemoryCircuitStim) {
  {
    // Steane tests
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 10;
    int nRounds = 4;
    {
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep0, nShots, nRounds);
      syndromes.dump();

      // No noise here, should be all zeros
      int sum = 0;
      for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
        for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
          sum += syndromes.at({i, j});
      EXPECT_TRUE(sum == 0);

      // Prep0, should measure out logical 0 each shot
      printf("data:\n");
      d.dump();
      printf("Lz:\n");
      Lz.dump();
      cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
      printf("logical_mz:\n");
      logical_mz.dump();
      EXPECT_FALSE(logical_mz.any());
    }
    {
      // Prep1, should measure out logical 1 each shot
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep1, nShots, nRounds);
      printf("data:\n");
      d.dump();
      printf("Lz:\n");
      Lz.dump();
      cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
      printf("logical_mz:\n");
      logical_mz.dump();
      EXPECT_EQ(nShots, logical_mz.sum_all());
    }
    {
      // Prepp, should measure out logical + each shot
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prepp, nShots, nRounds);
      printf("data:\n");
      d.dump();
      printf("Lx:\n");
      Lx.dump();
      cudaqx::tensor<uint8_t> logical_mx = Lx.dot(d.transpose()) % 2;
      printf("logical_mx:\n");
      logical_mx.dump();
      EXPECT_FALSE(logical_mx.any());
    }
    {
      // Prepm, should measure out logical - each shot
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prepm, nShots, nRounds);
      printf("data:\n");
      d.dump();
      printf("Lx:\n");
      Lx.dump();
      cudaqx::tensor<uint8_t> logical_mx = Lx.dot(d.transpose()) % 2;
      printf("logical_mx:\n");
      logical_mx.dump();
      EXPECT_EQ(nShots, logical_mx.sum_all());
    }
    {
      cudaq::set_random_seed(13);
      cudaq::noise_model noise;
      noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1), 1);

      nShots = 10;
      nRounds = 4;

      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep0, nShots, nRounds, noise);
      printf("syndromes:\n");
      syndromes.dump();

      // Noise here, expect a nonzero
      int sum = 0;
      for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
        for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
          sum += syndromes.at({i, j});
      EXPECT_TRUE(sum > 0);

      // With noise, Lz will sometimes be flipped
      printf("data:\n");
      d.dump();
      printf("Lz:\n");
      Lz.dump();
      cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
      printf("logical_mz:\n");
      logical_mz.dump();
      EXPECT_TRUE(logical_mz.any());
    }
  }
}

TEST(QECCodeTester, checkTwoQubitBitflipStim) {
  // This circuit should read out |00> with and without bitflip noise
  struct null1 {
    void operator()() __qpu__ {
      cudaq::qvector q(2);
      h(q);
      x<cudaq::ctrl>(q[0], q[1]);
      h(q);
    }
  };

  // This circuit should read out |00> without bitflip noise, and random values
  // with
  struct null2 {
    void operator()() __qpu__ {
      cudaq::qvector q(2);
      x<cudaq::ctrl>(q[0], q[1]);
    }
  };
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1), 1);
  cudaq::set_noise(noise);

  auto counts1 = cudaq::sample(100, null1{});
  EXPECT_FLOAT_EQ(1.0, counts1.probability("00"));

  auto counts2 = cudaq::sample(100, null2{});
  EXPECT_TRUE(counts2.probability("00") < 0.9);
  cudaq::unset_noise();
}

TEST(QECCodeTester, checkBitflip) {
  GTEST_SKIP() << "quantum optimizations cancel the h;h pair and eliminate the "
                  "qubit; re-enable once NVIDIA/cuda-quantum#5057 lands";

  // This circuit should read out |0> when noiseless
  struct null1 {
    void operator()() __qpu__ {
      cudaq::qubit q;
      h(q);
      h(q);
    }
  };

  auto counts1 = cudaq::sample(100, null1{});
  EXPECT_FLOAT_EQ(1.0, counts1.probability("0"));

  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::bit_flip_channel(0.5));
  cudaq::set_noise(noise);
  auto counts2 = cudaq::sample(100, null1{});
  cudaq::unset_noise();
  EXPECT_TRUE(counts2.probability("0") < 0.9);
}

TEST(QECCodeTester, checkNoisySampleMemoryCircuitAndDecodeStim) {
  {
    // Steane tests
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> H = steane->get_parity();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 1;
    int nRounds = 10;
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.01),
                                1);

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prep0, nShots, nRounds, noise);
    printf("syndromes:\n");
    syndromes.dump();

    // Noise here, expect a nonzero
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});
    EXPECT_TRUE(sum > 0);

    // With noise, Lz will sometimes be flipped
    printf("data:\n");
    d.dump();
    printf("Lz:\n");
    Lz.dump();
    cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
    printf("logical_mz:\n");
    logical_mz.dump();

    // s = (sx | sz)
    // sx = Hz . ex
    // sz = Hx . ez

    printf("Obs:\n");
    observables.dump();
    auto decoder = cudaq::qec::get_decoder("single_error_lut", H);
    printf("Hz:\n");
    H.dump();
    printf("end\n");
    // syndromes shape: (nShots=1, nDetectors).
    // Format: [numFixed round-0 detectors] [numCols * (nRounds-1) detectors]
    // [numFixed final-data] For prep0 (Z-prep): numFixed = numAncz = 3 numCols
    // = number of per-round syndromes
    size_t numAncz_s = steane->get_num_ancilla_z_qubits();
    size_t numAncx_s = steane->get_num_ancilla_x_qubits();
    size_t numCols = numAncz_s + numAncx_s;
    size_t numFixed = numAncz_s; // prep0 → Z-ancilla are fixed
    size_t fixedOffset = 0;      // Z-ancilla sit at cols 0..numFixed-1
    const uint8_t *shotRow = syndromes.data(); // nShots=1, shot=0
    size_t numLerrors = 0;
    cudaqx::tensor<uint8_t> pauli_frame({observables.shape()[0]});
    for (size_t i = 0; i < nRounds - 1; ++i) {
      cudaqx::tensor<uint8_t> syndrome({numCols});
      if (i == 0) {
        for (size_t c = 0; c < numFixed; ++c)
          syndrome.at({c}) = shotRow[c];
      } else {
        syndrome.borrow(shotRow + numFixed + (i - 1) * numCols);
      }
      printf("syndrome:\n");
      syndrome.dump();
      auto decoded = decoder->decode(syndrome);
      cudaqx::tensor<uint8_t> result_tensor;
      cudaq::qec::convert_vec_soft_to_tensor_hard(decoded.result,
                                                  result_tensor);
      printf("decode result:\n");
      result_tensor.dump();
      cudaqx::tensor<uint8_t> decoded_observables =
          observables.dot(result_tensor);
      printf("decoded observable:\n");
      decoded_observables.dump();
      pauli_frame = (pauli_frame + decoded_observables) % 2;
      printf("pauli frame:\n");
      pauli_frame.dump();
    }
    // prep0 means this is a z-basis experiment
    // Check if Lz + pauli_frame[0] = 0?
    printf("Lz: %d, xFlips: %d\n", Lz.at({0, 0}), pauli_frame.at({0}));
    if (Lz.at({0, 0}) != pauli_frame.at({0}))
      numLerrors++;
#ifdef __x86_64__
    // No logicals errors for this seed
    // TODO - find a comparable seed for ARM or modify test.
    EXPECT_EQ(0, numLerrors);
#endif
  }
  {
    // Test x-basis and x-flips
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> H = steane->get_parity();
    cudaqx::tensor<uint8_t> Hx = steane->get_parity_x();
    cudaqx::tensor<uint8_t> Hz = steane->get_parity_z();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 10;
    int nRounds = 4;
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.05), 1);

    // Bitflip is X-type error, detected by Z stabilizers (Hz)
    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndromes:\n");
    syndromes.dump();
    EXPECT_EQ(syndromes.shape()[0], nShots);
    EXPECT_EQ(syndromes.shape()[1], 24u); // numDetectors
    // With noise, Lx will sometimes be flipped
    printf("data:\n");
    d.dump();
    printf("Lx:\n");
    Lx.dump();
    cudaqx::tensor<uint8_t> logical_mx = Lx.dot(d.transpose()) % 2;
    // Can make a column vector
    printf("logical_mx:\n");
    logical_mx.dump();
    // bit flip errors trigger Z-type stabilizers (ZZIII)
    // these will be extracted into the ancx syndrome registers
    // (s_x | s_z ) = ( X flip syndromes, Z Flip syndromes)

    cudaqx::tensor<uint8_t> final_sx = Hz.dot(d.transpose()) % 2;
    // If x basis experiment, this would be final sx
    printf("final sx:\n");
    final_sx.dump();

    printf("Obs:\n");
    observables.dump();
    auto decoder = cudaq::qec::get_decoder("single_error_lut", H);
    printf("end\n");
    // syndromes shape: (nShots=1, nDetectors).
    // Format: [numFixed round-0 detectors] [numCols * (nRounds-1) detectors]
    // [numFixed final-data] For prepp (X-prep): numFixed = numAncx = 3 numCols
    // = number of per-round syndromes
    size_t numAncz_s2 = steane->get_num_ancilla_z_qubits();
    size_t numAncx_s2 = steane->get_num_ancilla_x_qubits();
    size_t numCols2 = numAncz_s2 + numAncx_s2;
    size_t numFixed2 = numAncx_s2; // prepp → X-ancilla are fixed
    size_t k2 = syndromes.shape()[1];
    size_t numLerrors = 0;
    for (size_t shot = 0; shot < nShots; ++shot) {
      cudaqx::tensor<uint8_t> pauli_frame({observables.shape()[0]});
      const uint8_t *shotRow2 = syndromes.data() + shot * k2;
      for (size_t i = 0; i < nRounds; ++i) {
        printf("shot: %zu, round: %zu\n", shot, i);
        cudaqx::tensor<uint8_t> syndrome({numCols2});
        if (i == 0) {
          for (size_t c = 0; c < numFixed2; ++c)
            syndrome.at({c}) = shotRow2[c];
        } else {
          syndrome.borrow(shotRow2 + numFixed2 + (i - 1) * numCols2);
        }
        printf("syndrome:\n");
        syndrome.dump();
        auto decoded = decoder->decode(syndrome);
        cudaqx::tensor<uint8_t> result_tensor;
        cudaq::qec::convert_vec_soft_to_tensor_hard(decoded.result,
                                                    result_tensor);

        printf("decode result:\n");
        result_tensor.dump();
        cudaqx::tensor<uint8_t> decoded_observables =
            observables.dot(result_tensor);
        printf("decoded observable:\n");
        decoded_observables.dump();
        pauli_frame = (pauli_frame + decoded_observables) % 2;
        printf("pauli frame:\n");
        pauli_frame.dump();
      }
      // prepp means this is a x-basis experiment
      // does LMx + pauli_frame[1] = |+>? (+ is read out as 0 after rotation)

      printf("Obs_x: %d, pfZ: %d\n", logical_mx.at({0, shot}),
             pauli_frame.at({1}));
      uint8_t corrected_obs =
          (logical_mx.at({0, shot}) + pauli_frame.at({1})) % 2;
      std::cout << "corrected_obs: " << +corrected_obs << "\n";
      if (corrected_obs != 0)
        numLerrors++;
    }
    printf("numLerrors: %zu\n", numLerrors);
    EXPECT_TRUE(numLerrors > 0);
  }
}

// Utility function to check if a matrix matches a string of bits.
void check_matrix_bits(const std::string &name,
                       const cudaqx::tensor<uint8_t> &matrix,
                       const std::vector<std::string> &expected_str) {
  EXPECT_EQ(expected_str.size(), matrix.shape()[0])
      << name << " shape mismatch";
  const auto *data_ptr = matrix.data();
  for (std::size_t i = 0; i < expected_str.size(); i++) {
    for (std::size_t j = 0; j < expected_str[i].size(); j++) {
      if (expected_str[i][j] == '.') {
        EXPECT_EQ(*data_ptr, 0)
            << name << " bit mismatch at " << i << ", " << j;
      } else {
        EXPECT_EQ(*data_ptr, 1)
            << name << " bit mismatch at " << i << ", " << j;
      }
      data_ptr++;
    }
  }
}

// End-to-end realtime decode driven by the decoder_context API,
// built from a full-basis CSS memory circuit. Steane's boundary is
// inhomogeneous: only the fixed (Z) stabilizers carry boundary detectors while
// interior rounds carry both types. The measurement
// window is streamed as the realtime protocol does (numCols ancilla per round,
// then the final data readout); the sliding accumulation must fire a decode at
// exactly the right point -- only once the full window has arrived.
TEST(QECCodeTester, checkRealtimeDecodeFromMemoryCircuit) {
  auto steane = cudaq::qec::get_code("steane");
  const std::size_t nRounds = 3;
  const std::size_t numCols =
      steane->get_num_ancilla_z_qubits() + steane->get_num_ancilla_x_qubits();
  const std::size_t numData = steane->get_num_data_qubits();
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.05),
                              1);

  auto ctx = cudaq::qec::decoder_context_from_memory_circuit(
      *steane, cudaq::qec::operation::prep0, nRounds, noise);
  auto inputs = ctx.full_component();
  const auto &dem = inputs.dem;
  const auto D = cudaq::qec::m2d_to_sparse(inputs.m2d);
  const auto m2d_rows = D.to_nested_csr();

  ASSERT_FALSE(m2d_rows.empty());
  EXPECT_EQ(m2d_rows.size(), dem.num_detectors());
  ASSERT_EQ(ctx.num_measurements(), nRounds * numCols + numData);

  // Inhomogeneous boundary: single-measurement boundary detectors coexist with
  // multi-measurement interior/final ones, so m2d rows are not all one shape.
  std::size_t minRow = m2d_rows[0].size(), maxRow = m2d_rows[0].size();
  for (const auto &row : m2d_rows) {
    minRow = std::min(minRow, row.size());
    maxRow = std::max(maxRow, row.size());
  }
  EXPECT_LT(minRow, maxRow);

  // Circuit analysis produces the model; decoder_init is what a decoder is
  // built from. Bridging the two here is one expression, and it carries O and
  // D, so construction configures the realtime path completely. There is no
  // second step, and nothing to re-supply.
  auto decoder = cudaq::qec::get_decoder("single_error_lut",
                                         cudaq::qec::decoder_init(dem, D));
  ASSERT_EQ(decoder->get_num_msyn_per_decode(), D.num_cols());

  // Stream numCols ancilla per round, then the final data readout. The window
  // must not decode until that last chunk completes it.
  for (std::size_t r = 0; r < nRounds; ++r)
    EXPECT_FALSE(
        decoder->enqueue_syndrome(std::vector<std::uint8_t>(numCols, 0)))
        << "decoded early after round " << r;
  EXPECT_TRUE(decoder->enqueue_syndrome(std::vector<std::uint8_t>(numData, 0)));

  // No errors were injected, so every observable correction is trivially zero.
  for (std::size_t k = 0; k < decoder->get_num_observables(); ++k)
    EXPECT_EQ(decoder->get_obs_corrections()[k], 0);
}

namespace {
// Records the detector vector the realtime path hands down, so a test can check
// the measurement-to-detector reduction directly rather than inferring it from
// a decode result.
class detector_recording_decoder : public cudaq::qec::decoder {
public:
  std::vector<std::vector<uint8_t>> seen;

  using cudaq::qec::decoder::decoder;

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    // enqueue_syndrome() converts the detector bits to soft values on the way
    // in, so undo that with the same threshold to recover what it computed.
    std::vector<uint8_t> hard;
    cudaq::qec::convert_vec_soft_to_hard(syndrome, hard);
    seen.push_back(std::move(hard));
    return trivial_result();
  }

private:
  cudaq::qec::decoder_result trivial_result() {
    cudaq::qec::decoder_result result;
    result.converged = true;
    result.result.assign(block_size, 0.0);
    return result;
  }
};
} // namespace

// Stream a measurement window a round at a time and require the detector vector
// enqueue_syndrome() hands to the decoder to match a direct reduction of m2d
// against the same measurements, over many random records. Steane's boundary
// layer is narrower than its interior rounds, so the window has non-uniform
// round widths.
//
// Scope: decode() only runs once the window is complete, so this pins the
// measurement-to-detector reduction and the point at which a decode fires. It
// does not pin the order in which individual detectors are folded in, which is
// not observable through decode() -- reducing the whole buffer on the final
// round yields the same vector.
TEST(QECCodeTester, checkRealtimeStreamedDetectorReduction) {
  auto steane = cudaq::qec::get_code("steane");
  const std::size_t nRounds = 3;
  const std::size_t numCols =
      steane->get_num_ancilla_z_qubits() + steane->get_num_ancilla_x_qubits();
  const std::size_t numData = steane->get_num_data_qubits();
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.05),
                              1);

  auto ctx = cudaq::qec::decoder_context_from_memory_circuit(
      *steane, cudaq::qec::operation::prep0, nRounds, noise);
  auto [dem, m2d, m2o] = ctx.full_component();
  const std::size_t numMeas = ctx.num_measurements();
  ASSERT_EQ(numMeas, nRounds * numCols + numData);

  detector_recording_decoder dec(
      cudaq::qec::decoder_init(dem, cudaq::qec::m2d_to_sparse(m2d)));
  ASSERT_EQ(dec.get_num_msyn_per_decode(), numMeas);

  std::mt19937 rng(17);
  std::size_t total_fired = 0;
  for (std::size_t shot = 0; shot < 100; shot++) {
    std::vector<uint8_t> meas(numMeas);
    for (auto &m : meas)
      m = rng() & 1;

    std::vector<uint8_t> expected(m2d.rows.size(), 0);
    for (std::size_t d = 0; d < m2d.rows.size(); d++) {
      uint8_t v = 0;
      for (const auto c : m2d.rows[d])
        v ^= meas[c];
      expected[d] = v;
      total_fired += v;
    }

    dec.reset_decoder();
    dec.seen.clear();
    std::size_t off = 0;
    for (std::size_t r = 0; r < nRounds; r++) {
      EXPECT_FALSE(dec.enqueue_syndrome(std::vector<uint8_t>(
          meas.begin() + off, meas.begin() + off + numCols)))
          << "decoded early after round " << r << " of shot " << shot;
      off += numCols;
    }
    ASSERT_TRUE(dec.enqueue_syndrome(
        std::vector<uint8_t>(meas.begin() + off, meas.end())));
    ASSERT_EQ(dec.seen.size(), 1u);
    EXPECT_EQ(dec.seen.front(), expected) << "shot " << shot;
  }
  EXPECT_GT(total_fired, 0u); // guard against an all-zero pass
}

namespace {
bool tensors_equal(const cudaqx::tensor<uint8_t> &a,
                   const cudaqx::tensor<uint8_t> &b) {
  if (a.shape() != b.shape())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a.data()[i] != b.data()[i])
      return false;
  return true;
}
} // namespace

// decoder_context_from_memory_circuit returns a decoder_context;
// canonicalization is deferred until a component method is called. Verify that
// full_component() matches dem_from_memory_circuit and that x_component() /
// z_component() reproduce x_/z_dem_from_memory_circuit and partition the
// detectors.
TEST(QECCodeTester, checkDecoderContextAndComponents) {
  auto steane = cudaq::qec::get_code("steane");
  const auto prep = cudaq::qec::operation::prep0;
  const std::size_t nRounds = 3;
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.05),
                              1);
  auto ctx = cudaq::qec::decoder_context_from_memory_circuit(*steane, prep,
                                                             nRounds, noise);
  auto dem = cudaq::qec::dem_from_memory_circuit(*steane, prep, nRounds, noise);
  auto z = cudaq::qec::z_dem_from_memory_circuit(*steane, prep, nRounds, noise);
  auto x = cudaq::qec::x_dem_from_memory_circuit(*steane, prep, nRounds, noise);

  // full_component() matches the plain entry point.
  auto fc_inputs = ctx.full_component();
  const auto &fc_dem = fc_inputs.dem;
  EXPECT_TRUE(
      tensors_equal(fc_dem.detector_error_matrix, dem.detector_error_matrix));

  // x_component() / z_component() reproduce the per-type DEMs and partition
  // the detectors without re-running dem_from_kernel.
  auto zc_inputs = ctx.z_component();
  auto xc_inputs = ctx.x_component();
  const auto &zc_dem = zc_inputs.dem;
  const auto &xc_dem = xc_inputs.dem;
  EXPECT_TRUE(
      tensors_equal(zc_dem.detector_error_matrix, z.detector_error_matrix));
  EXPECT_TRUE(
      tensors_equal(xc_dem.detector_error_matrix, x.detector_error_matrix));
  EXPECT_EQ(zc_dem.num_detectors() + xc_dem.num_detectors(),
            fc_dem.num_detectors());
  EXPECT_EQ(zc_inputs.m2d.rows.size(), zc_dem.num_detectors());
}

TEST(QECCodeTester, checkDemFromMemoryCircuit) {
  auto steane = cudaq::qec::get_code("steane");
  int num_rounds = 4;
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("mz", cudaq::bit_flip_channel(0.01));
  auto dem = cudaq::qec::dem_from_memory_circuit(
      *steane, cudaq::qec::operation::prep0, num_rounds, noise);

  // Uncomment if desired:
  // printf("dem:\n");
  // dem.detector_error_matrix.dump_bits();

  // clang-format off
  std::vector<std::string> expected_dem_str = {
      "1..............................",
      ".1.............................",
      "..1............................",
      "1.....1........................",
      ".1.....1.......................",
      "..1.....1......................",
      "...1.....1.....................",
      "....1.....1....................",
      ".....1.....1...................",
      "......1.....1..................",
      ".......1.....1.................",
      "........1.....1................",
      ".........1.....1...............",
      "..........1.....1..............",
      "...........1.....1.............",
      "............1........1.........",
      ".............1........1........",
      "..............1........1.......",
      "...............1..1............",
      "................1..1...........",
      ".................1..1..........",
      ".....................1..1111...",
      "......................1..1.111.",
      ".......................1..11.11"};
  // clang-format on
  check_matrix_bits("detector_error_matrix", dem.detector_error_matrix,
                    expected_dem_str);

  // Print the error probabilities (uncomment if desired)
  // printf("error probabilities: { ");
  // for (std::size_t i = 0; i < dem.error_rates.size(); i++) {
  //   printf("%f ", dem.error_rates[i]);
  // }
  // printf("}\n");

  // Check error rates
  ASSERT_EQ(dem.error_rates.size(), 31);
  std::vector<double> expected_error_rates(31, 0.01);
  for (std::size_t i = 0; i < dem.error_rates.size(); i++) {
    EXPECT_FLOAT_EQ(dem.error_rates[i], expected_error_rates[i])
        << "i: " << i << ", error_rates[i]: " << dem.error_rates[i]
        << ", expected_error_rates[i]: " << expected_error_rates[i];
  }

  // Uncomment if desired:
  // printf("observables_flips_matrix:\n");
  // dem.observables_flips_matrix.dump_bits();

  // Check observables flips matrix
  std::vector<std::string> expected_observables_flips_matrix_str = {
      "............................111"};
  check_matrix_bits("observables_flips_matrix", dem.observables_flips_matrix,
                    expected_observables_flips_matrix_str);
}

// Shortest undetectable logical error of a graphlike DEM. Detectors are graph
// nodes (plus one virtual boundary node), each error mechanism with <= 2
// flipped detectors is an edge labeled with its observable flip, and the
// effective distance is the shortest boundary-to-boundary walk with an odd
// number of observable flips (BFS over (node, parity) states). Error
// mechanisms flipping more than 2 detectors are skipped; that can only
// overestimate the distance, so the equality assertion below stays sound.
static std::size_t
shortest_graphlike_logical_error(const cudaq::qec::detector_error_model &dem) {
  const std::size_t numDet = dem.num_detectors();
  const std::size_t numErr = dem.num_error_mechanisms();
  const std::size_t boundary = numDet;
  struct Edge {
    std::size_t to;
    bool obs;
  };
  std::vector<std::vector<Edge>> adj(numDet + 1);
  for (std::size_t e = 0; e < numErr; ++e) {
    std::vector<std::size_t> dets;
    for (std::size_t d = 0; d < numDet && dets.size() <= 2; ++d)
      if (dem.detector_error_matrix.at({d, e}))
        dets.push_back(d);
    if (dets.size() > 2)
      continue;
    const bool obs = dem.observables_flips_matrix.at({0, e}) != 0;
    const std::size_t u = dets.empty() ? boundary : dets[0];
    const std::size_t v = dets.size() == 2 ? dets[1] : boundary;
    adj[u].push_back({v, obs});
    if (u != v)
      adj[v].push_back({u, obs});
  }
  std::vector<std::array<long, 2>> dist(numDet + 1, {-1, -1});
  std::deque<std::pair<std::size_t, int>> queue;
  dist[boundary][0] = 0;
  queue.push_back({boundary, 0});
  while (!queue.empty()) {
    auto [u, p] = queue.front();
    queue.pop_front();
    for (const auto &edge : adj[u]) {
      const int np = p ^ (edge.obs ? 1 : 0);
      if (dist[edge.to][np] < 0) {
        dist[edge.to][np] = dist[u][p] + 1;
        queue.push_back({edge.to, np});
      }
    }
  }
  return dist[boundary][1] < 0 ? std::numeric_limits<std::size_t>::max()
                               : static_cast<std::size_t>(dist[boundary][1]);
}

// Under circuit-level (two-qubit gate) noise the surface code only retains
// its full code distance if the per-plaquette CNOT order steers hook errors
// perpendicular to the logical operators. A same-order schedule for X and Z
// plaquettes halves the effective distance of one of the two memory bases.
TEST(QECCodeTester, checkSurfaceCodeEffectiveDistance) {
  for (std::size_t distance : {3, 5}) {
    // Each orientation assigns the two schedule shapes to opposite plaquette
    // types, so every orientation must retain the full distance in both
    // memory bases.
    for (const std::string orientation : {"XV", "XH", "ZV", "ZH"}) {
      auto code = cudaq::qec::get_code(
          "surface_code",
          cudaqx::heterogeneous_map{{"distance", distance},
                                    {"orientation", orientation}});
      for (auto prep :
           {cudaq::qec::operation::prep0, cudaq::qec::operation::prepp}) {
        cudaq::noise_model noise;
        noise.add_all_qubit_channel("x",
                                    cudaq::qec::two_qubit_depolarization(0.001),
                                    /*num_controls=*/1);
        auto dem = cudaq::qec::dem_from_memory_circuit(
            *code, prep, /*numRounds=*/distance, noise,
            /*decompose_errors=*/true);
        ASSERT_EQ(dem.num_observables(), 1);
        EXPECT_EQ(distance, shortest_graphlike_logical_error(dem))
            << "distance " << distance << " orientation " << orientation
            << " statePrep "
            << (prep == cudaq::qec::operation::prep0 ? "prep0" : "prepp");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Test-only custom code: the Shor [[9,1,3]] code, a CSS code with 2 X-type
// and 6 Z-type stabilizers. The unequal stabilizer counts produce a
// non-uniform detector layout, exercising the boundary-aware round machinery.
// ---------------------------------------------------------------------------
namespace shor9_test {

__qpu__ void prep0(cudaq::qec::patch logicalQubit) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++)
    reset(logicalQubit.data[i]);
}

// X-basis prep (|+...+>): a +1 eigenstate of the X-stabilizers, so X is the
// fixed (boundary) basis, exercising the X-type boundary.
__qpu__ void prepp(cudaq::qec::patch logicalQubit) {
  prep0(logicalQubit);
  h(logicalQubit.data);
}

__qpu__ std::vector<cudaq::measure_result>
stabilizer(cudaq::qec::patch logicalQubit,
           const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers) {
  h(logicalQubit.ancx);
  for (std::size_t xi = 0; xi < logicalQubit.ancx.size(); ++xi)
    for (std::size_t di = 0; di < logicalQubit.data.size(); ++di)
      if (x_stabilizers[xi * logicalQubit.data.size() + di] == 1)
        cudaq::x<cudaq::ctrl>(logicalQubit.ancx[xi], logicalQubit.data[di]);
  h(logicalQubit.ancx);

  for (std::size_t zi = 0; zi < logicalQubit.ancz.size(); ++zi)
    for (std::size_t di = 0; di < logicalQubit.data.size(); ++di)
      if (z_stabilizers[zi * logicalQubit.data.size() + di] == 1)
        cudaq::x<cudaq::ctrl>(logicalQubit.data[di], logicalQubit.ancz[zi]);

  auto results = mz(logicalQubit.ancz, logicalQubit.ancx);

  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);
  return results;
}

class shor9 : public cudaq::qec::code {
protected:
  std::size_t get_num_data_qubits() const override { return 9; }
  std::size_t get_num_ancilla_qubits() const override { return 8; }
  std::size_t get_num_ancilla_x_qubits() const override { return 2; }
  std::size_t get_num_ancilla_z_qubits() const override { return 6; }
  std::size_t get_num_x_stabilizers() const override { return 2; }
  std::size_t get_num_z_stabilizers() const override { return 6; }

public:
  shor9(const cudaqx::heterogeneous_map &) : code() {
    operation_encodings.insert(
        std::make_pair(cudaq::qec::operation::stabilizer_round, stabilizer));
    operation_encodings.insert(
        std::make_pair(cudaq::qec::operation::prep0, prep0));
    operation_encodings.insert(
        std::make_pair(cudaq::qec::operation::prepp, prepp));
    m_stabilizers =
        fromPauliWords({"XXXXXXIII", "IIIXXXXXX", "ZZIIIIIII", "IZZIIIIII",
                        "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ"});
    m_pauli_observables = fromPauliWords({"XXXIIIIII", "ZIIZIIZII"});
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      shor9, static std::unique_ptr<cudaq::qec::code> create(
                 const cudaqx::heterogeneous_map &options) {
        return std::make_unique<shor9>(options);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(shor9)

} // namespace shor9_test

namespace {

// Build the Shor [[9,1,3]] full both-basis DEM for a `num_rounds` memory
// experiment. `prep` selects the basis (prep0 => Z boundary, prepp => X).
cudaq::qec::detector_error_model
shor9_dem(std::size_t num_rounds,
          cudaq::qec::operation prep = cudaq::qec::operation::prep0) {
  auto shor = cudaq::qec::get_code("shor9");
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("mz", cudaq::bit_flip_channel(0.01));
  return cudaq::qec::dem_from_memory_circuit(*shor, prep, num_rounds, noise);
}

// Build sliding_window parameters for a boundary-layout DEM.
cudaqx::heterogeneous_map shor9_sliding_params(std::size_t window_size,
                                               std::size_t interior,
                                               std::size_t numBoundary) {
  cudaqx::heterogeneous_map inner_params;
  inner_params.insert("dummy_param", 1);
  cudaqx::heterogeneous_map params;
  params.insert("window_size", window_size);
  params.insert("step_size", static_cast<std::size_t>(1));
  params.insert("num_syndromes_per_round", interior);
  params.insert("num_boundary_syndromes", numBoundary);
  params.insert("straddle_start_round", false);
  params.insert("straddle_end_round", true);
  params.insert("inner_decoder_name", std::string("single_error_lut"));
  params.insert("inner_decoder_params", inner_params);
  return params;
}

// For every DEM column, use it as a syndrome, decode it with `decode_fn`, and
// require the resulting observable-flip prediction to match the full decoder's.
template <class DecodeFn>
void expectObservablesMatchFullDecoder(
    const cudaq::qec::detector_error_model &dem, cudaq::qec::decoder &full,
    DecodeFn decode_fn) {
  const auto &H = dem.detector_error_matrix;
  const auto &O = dem.observables_flips_matrix;
  const std::size_t rows = H.shape()[0], cols = H.shape()[1];
  const std::size_t numObs = O.shape()[0];
  std::size_t mismatches = 0;
  for (std::size_t col = 0; col < cols; ++col) {
    std::vector<cudaq::qec::float_t> syndrome(rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r)
      syndrome[r] = H.at({r, col});

    auto r_full = full.decode(syndrome).result;
    auto r_sw = decode_fn(syndrome);
    ASSERT_EQ(r_full.size(), r_sw.size());

    // Compare observable flips O @ hard(result) (mod 2).
    for (std::size_t o = 0; o < numObs; ++o) {
      std::uint8_t of_full = 0, of_sw = 0;
      for (std::size_t c = 0; c < r_full.size(); ++c)
        if (O.at({o, c})) {
          of_full ^= (r_full[c] > 0.5) ? 1 : 0;
          of_sw ^= (r_sw[c] > 0.5) ? 1 : 0;
        }
      if (of_full != of_sw)
        mismatches++;
    }
  }
  EXPECT_EQ(mismatches, 0u)
      << "sliding-window observable predictions disagree with the full decoder";
}

} // namespace

TEST(QECCodeTester, checkDemFromMemoryCircuitShor9) {
  auto shor = cudaq::qec::get_code("shor9");
  ASSERT_TRUE(shor != nullptr);

  const std::size_t numXStabs = shor->get_num_x_stabilizers();
  const std::size_t numZStabs = shor->get_num_z_stabilizers();
  EXPECT_EQ(numXStabs, 2u);
  EXPECT_EQ(numZStabs, 6u);

  const std::size_t num_rounds = 4;
  const std::uint32_t interiorWidth = numXStabs + numZStabs;

  // prep0 => Z-type boundary (numZStabs wide); prepp => X-type (numXStabs).
  for (auto cfg : {std::make_pair(cudaq::qec::operation::prep0, numZStabs),
                   std::make_pair(cudaq::qec::operation::prepp, numXStabs)}) {
    const auto prep = cfg.first;
    const std::size_t numFixed = cfg.second;

    auto dem = shor9_dem(num_rounds, prep);

    // Non-uniform layout: a numFixed boundary, (num_rounds - 1) interior rounds
    // of interiorWidth detectors, and a final numFixed boundary.
    const std::size_t expected_rows =
        2 * numFixed + (num_rounds - 1) * interiorWidth;
    EXPECT_EQ(dem.detector_error_matrix.shape()[0], expected_rows);

    // Column counts must be consistent across the three data structures.
    const std::size_t num_cols = dem.detector_error_matrix.shape()[1];
    EXPECT_GT(num_cols, 0u);
    EXPECT_EQ(dem.error_rates.size(), num_cols);
    EXPECT_EQ(dem.observables_flips_matrix.shape()[1], num_cols);
    EXPECT_EQ(dem.num_observables(), 1u);

    // Every retained column carries a nonzero rate and detector signature
    // (remove_zero_syndrome_errors was requested).
    for (std::size_t c = 0; c < num_cols; c++) {
      EXPECT_GT(dem.error_rates[c], 0.0);
      std::size_t weight = 0;
      for (std::size_t r = 0; r < expected_rows; r++)
        weight += dem.detector_error_matrix.at({r, c});
      EXPECT_GT(weight, 0u) << "column " << c << " has an empty syndrome";
    }

    // Columns must be ordered by their true rounds (first numFixed rows =
    // round 0, then interiorWidth-wide interior rounds).
    auto true_round = [&](std::uint32_t r) -> std::uint32_t {
      if (r < numFixed)
        return 0;
      return 1 + (r - static_cast<std::uint32_t>(numFixed)) / interiorWidth;
    };

    std::pair<std::uint32_t, std::uint32_t> prev = {0, 0};
    for (std::size_t c = 0; c < num_cols; c++) {
      std::uint32_t first_row = expected_rows, last_row = 0;
      for (std::size_t r = 0; r < expected_rows; r++)
        if (dem.detector_error_matrix.at({r, c})) {
          first_row = std::min<std::uint32_t>(first_row, r);
          last_row = std::max<std::uint32_t>(last_row, r);
        }
      std::pair<std::uint32_t, std::uint32_t> key = {true_round(first_row),
                                                     true_round(last_row)};
      if (c > 0)
        EXPECT_LE(prev, key) << "column " << c << " is out of true-round order";
      prev = key;
    }

    // Calling the boundary-aware canonicalize again is stable.
    dem.canonicalize_for_rounds_with_boundary(
        interiorWidth, static_cast<uint32_t>(numFixed),
        /*remove_zero_syndrome_errors=*/true);
    EXPECT_EQ(dem.detector_error_matrix.shape()[0], expected_rows);
    EXPECT_EQ(dem.detector_error_matrix.shape()[1], num_cols);
    EXPECT_EQ(dem.error_rates.size(), num_cols);
  }
}

// Sliding-window decoding of the Shor [[9,1,3]] full both-basis (boundary)
// DEM. The sliding window internally zero-pads the boundary layers up to the
// interior width via num_boundary_syndromes, so its whole-block decode must
// agree with a full decoder.
TEST(QECCodeTester, checkSlidingWindowShor9Boundary) {
  auto shor = cudaq::qec::get_code("shor9");
  const std::size_t numXStabs = shor->get_num_x_stabilizers();
  const std::size_t numZStabs = shor->get_num_z_stabilizers();
  const std::size_t interior = numXStabs + numZStabs;
  const std::size_t num_rounds = 4;

  // prep0 => Z-type boundary; prepp => X-type boundary.
  for (auto cfg : {std::make_pair(cudaq::qec::operation::prep0, numZStabs),
                   std::make_pair(cudaq::qec::operation::prepp, numXStabs)}) {
    const auto prep = cfg.first;
    const std::size_t numBoundary = cfg.second;

    auto dem = shor9_dem(num_rounds, prep);
    const std::size_t rows = dem.detector_error_matrix.shape()[0];
    // The [B | S...S | B] layout has one more detector layer than there are
    // rounds: two narrow boundary layers plus (num_rounds - 1) interior layers.
    const std::size_t num_layers =
        (rows + 2 * (interior - numBoundary)) / interior;
    ASSERT_EQ(num_layers, num_rounds + 1);

    auto full =
        cudaq::qec::decoder::get("single_error_lut", dem.detector_error_matrix);
    // A single window spanning all layers -- should match the full decoder.
    auto sw = cudaq::qec::decoder::get(
        "sliding_window", cudaq::qec::decoder_init{dem},
        shor9_sliding_params(num_layers, interior, numBoundary));

    expectObservablesMatchFullDecoder(
        dem, *full, [&](const std::vector<cudaq::qec::float_t> &syndrome) {
          return sw->decode(syndrome).result;
        });
  }
}

// Streaming sliding-window decoding of the Shor boundary layout:
// detector layers are fed one at a time. The two boundary layers arrive with
// numZStabs (6) values and the interior layers with numXStabs+numZStabs (8)
// values; the decoder consumes each layer at its native width.
// The streamed result must match a full decoder.
TEST(QECCodeTester, checkSlidingWindowShor9Streaming) {
  auto shor = cudaq::qec::get_code("shor9");
  const std::size_t numXStabs = shor->get_num_x_stabilizers();
  const std::size_t numZStabs = shor->get_num_z_stabilizers();
  const std::size_t interior = numXStabs + numZStabs;
  const std::size_t num_rounds = 4;

  // prep0 => Z-type boundary; prepp => X-type boundary.
  for (auto cfg : {std::make_pair(cudaq::qec::operation::prep0, numZStabs),
                   std::make_pair(cudaq::qec::operation::prepp, numXStabs)}) {
    const auto prep = cfg.first;
    const std::size_t numBoundary = cfg.second;

    auto dem = shor9_dem(num_rounds, prep);
    const std::size_t rows = dem.detector_error_matrix.shape()[0];
    const std::size_t num_layers =
        (rows + 2 * (interior - numBoundary)) / interior;

    // Detector-layer sizes: [numBoundary | interior*(num_layers-2) |
    // numBoundary].
    std::vector<std::size_t> layer_sizes(num_layers, interior);
    layer_sizes.front() = numBoundary;
    layer_sizes.back() = numBoundary;

    auto full =
        cudaq::qec::decoder::get("single_error_lut", dem.detector_error_matrix);
    // A genuinely sliding configuration: window of 2 rounds, stepping by 1.
    auto sw = cudaq::qec::decoder::get(
        "sliding_window", cudaq::qec::decoder_init{dem},
        shor9_sliding_params(/*window_size=*/2, interior, numBoundary));

    expectObservablesMatchFullDecoder(
        dem, *full, [&](const std::vector<cudaq::qec::float_t> &syndrome) {
          // Feed the syndrome one detector layer at a time (variable widths).
          cudaq::qec::decoder_result streamed;
          std::size_t off = 0;
          for (auto ls : layer_sizes) {
            std::vector<cudaq::qec::float_t> layer(syndrome.begin() + off,
                                                   syndrome.begin() + off + ls);
            off += ls;
            auto r = sw->decode(layer);
            if (!r.result.empty())
              streamed = std::move(r); // final layer yields the result
          }
          EXPECT_FALSE(streamed.result.empty());
          return streamed.result;
        });
  }
}

// End-to-end realtime sliding-window decoding of a boundary-layout memory
// circuit. The decoder streams one detector layer at its real width ([B | S...S
// | B]) and must match a whole-block decode of the same detectors.
TEST(QECCodeTester, checkSlidingWindowRealtimeBoundaryStreaming) {
  cudaq::set_random_seed(13);
  auto code = cudaq::qec::get_code("surface_code",
                                   cudaqx::heterogeneous_map{{"distance", 3}});
  const std::size_t numRounds = 4;
  const auto prep = cudaq::qec::operation::prep0; // Z-basis => Z boundary
  const bool is_z_prep = true;

  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.02),
                              /*num_controls=*/1);

  auto &prepOp =
      code->get_operation<cudaq::qec::code::one_qubit_encoding>(prep);
  auto &stabRound = code->get_operation<cudaq::qec::code::stabilizer_round>(
      cudaq::qec::operation::stabilizer_round);
  auto parity_x = code->get_parity_x();
  auto parity_z = code->get_parity_z();
  std::vector<std::size_t> xVec(parity_x.data(),
                                parity_x.data() + parity_x.size());
  std::vector<std::size_t> zVec(parity_z.data(),
                                parity_z.data() + parity_z.size());
  const std::size_t numData = code->get_num_data_qubits();
  const std::size_t numAncx = code->get_num_ancilla_x_qubits();
  const std::size_t numAncz = code->get_num_ancilla_z_qubits();
  const std::size_t numXStabs = code->get_num_x_stabilizers();
  const std::size_t numZStabs = code->get_num_z_stabilizers();
  auto logical_obs = code->get_observables_z();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                    logical_obs.data() + logical_obs.size());

  cudaq::M2DSparseMatrix m2d;
  cudaq::M2OSparseMatrix m2o;
  cudaq::dem_from_kernel(cudaq::qec::memory_circuit, &noise, /*options=*/{},
                         m2d, m2o, stabRound, prepOp, numData, numAncx, numAncz,
                         numRounds, xVec, zVec, obs_flat, num_obs, !is_z_prep);

  // Boundary DEM: its detector rows share m2d's [B | S...S | B] order.
  auto dem = cudaq::qec::dem_from_memory_circuit(*code, prep, numRounds, noise);
  const std::size_t S = numXStabs + numZStabs;
  const std::size_t B = numZStabs;
  const std::size_t numCols =
      numAncx + numAncz; // ancilla measurements per round

  auto to_sparse = [](std::size_t nRows, auto get_row) {
    std::vector<std::vector<uint32_t>> out(nRows);
    for (std::size_t r = 0; r < nRows; r++)
      out[r] = get_row(r);
    return out;
  };
  auto D_sparse = to_sparse(m2d.rows.size(), [&](std::size_t r) {
    return std::vector<uint32_t>(m2d.rows[r].begin(), m2d.rows[r].end());
  });
  const auto &O = dem.observables_flips_matrix;
  auto O_sparse = to_sparse(O.shape()[0], [&](std::size_t r) {
    std::vector<uint32_t> row;
    for (std::size_t c = 0; c < O.shape()[1]; c++)
      if (O.at({r, c}))
        row.push_back(static_cast<uint32_t>(c));
    return row;
  });

  auto make_sw = [&]() {
    cudaqx::heterogeneous_map params;
    params.insert("window_size", std::size_t{2});
    params.insert("step_size", std::size_t{1});
    params.insert("num_syndromes_per_round", S);
    params.insert("num_boundary_syndromes", B);
    params.insert("straddle_start_round", false);
    params.insert("straddle_end_round", true);
    params.insert("inner_decoder_name", std::string("single_error_lut"));
    params.insert("inner_decoder_params", cudaqx::heterogeneous_map{});
    // O comes from the DEM and D is handed in alongside it, so the model is
    // complete before the decoder exists.
    std::uint32_t num_measurements = 0;
    for (const auto &row : D_sparse)
      for (auto col : row)
        num_measurements = std::max(num_measurements, col + 1);
    return cudaq::qec::decoder::get(
        "sliding_window",
        cudaq::qec::decoder_init{
            dem, cudaq::qec::sparse_binary_matrix::from_nested_csr(
                     static_cast<std::uint32_t>(D_sparse.size()),
                     num_measurements, D_sparse)},
        params);
  };
  auto sw = make_sw();     // realtime streaming
  auto sw_ref = make_sw(); // whole-block reference

  // Raw measurements: numRounds*numCols ancilla, then numData data, per shot.
  const std::size_t nShots = 200;
  cudaq::sample_options opts{
      .shots = nShots, .noise = noise, .explicit_measurements = true};
  auto result = cudaq::sample(opts, cudaq::qec::memory_circuit, stabRound,
                              prepOp, numData, numAncx, numAncz, numRounds,
                              xVec, zVec, obs_flat, num_obs, !is_z_prep);
  cudaqx::tensor<uint8_t> mzTable(result.sequential_data());
  const std::size_t numMeas = mzTable.shape()[1];

  std::size_t mismatches = 0, nonzero_shots = 0;
  for (std::size_t shot = 0; shot < nShots; shot++) {
    const uint8_t *meas = &mzTable.at({shot, 0});

    // Reference: full detector vector via m2d
    std::vector<cudaq::qec::float_t> det(m2d.rows.size());
    for (std::size_t d = 0; d < m2d.rows.size(); d++) {
      uint8_t v = 0;
      for (auto m : m2d.rows[d])
        v ^= meas[m];
      det[d] = v;
    }
    auto ref = sw_ref->decode(det).result;
    std::vector<uint8_t> ref_obs(num_obs, 0);
    for (std::size_t k = 0; k < num_obs; k++)
      for (auto c : O_sparse[k])
        ref_obs[k] ^= (ref[c] > 0.5);

    // Realtime: stream measurement groups (numCols ancilla per round, then
    // data).
    sw->reset_decoder();
    sw->clear_corrections();
    std::size_t off = 0;
    for (std::size_t r = 0; r < numRounds; r++) {
      sw->enqueue_syndrome(
          std::vector<uint8_t>(meas + off, meas + off + numCols));
      off += numCols;
    }
    bool decoded =
        sw->enqueue_syndrome(std::vector<uint8_t>(meas + off, meas + numMeas));
    EXPECT_TRUE(decoded); // final measurement group must complete a decode

    const uint8_t *rt = sw->get_obs_corrections();
    for (std::size_t k = 0; k < num_obs; k++) {
      if (rt[k] != ref_obs[k])
        mismatches++;
      if (ref_obs[k])
        nonzero_shots++;
    }
  }
  EXPECT_EQ(mismatches, 0u);
  EXPECT_GT(nonzero_shots, 0u); // guard against an all-zero pass
}
