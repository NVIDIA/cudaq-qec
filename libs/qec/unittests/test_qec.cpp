/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <unistd.h>

#include "cuda-qx/core/library_utils.h"
#include "cudaq/qec/codes/surface_code.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/version.h"

namespace {

TEST(DecoderInputsUtilities, DSparseDoesNotRequireDeclaredMeasurementWidth) {
  cudaq::M2DSparseMatrix m2d;
  m2d.rows = {{2}, {}, {0, 4}};

  EXPECT_EQ(cudaq::qec::d_sparse(m2d),
            (std::vector<std::int64_t>{2, -1, -1, 0, 4, -1}));
}

using cudaq::qec::surface_code::sc_orientation;
using cudaq::qec::surface_code::stabilizer_grid;
using cudaq::qec::surface_code::surface_role;

int overlap_parity(const std::vector<std::size_t> &a,
                   const std::vector<std::size_t> &b) {
  auto a_it = a.begin();
  auto b_it = b.begin();
  int parity = 0;
  while (a_it != a.end() && b_it != b.end()) {
    if (*a_it < *b_it) {
      ++a_it;
    } else if (*b_it < *a_it) {
      ++b_it;
    } else {
      parity ^= 1;
      ++a_it;
      ++b_it;
    }
  }
  return parity;
}

std::vector<std::size_t> top_row_support(std::uint32_t distance) {
  std::vector<std::size_t> support;
  for (std::size_t i = 0; i < distance; ++i)
    support.push_back(i);
  return support;
}

std::vector<std::size_t> left_column_support(std::uint32_t distance) {
  std::vector<std::size_t> support;
  for (std::size_t i = 0; i < distance * distance; i += distance)
    support.push_back(i);
  return support;
}

// Recover the integer support (indices of non-identity Paulis) of a single
// spin_op_term pauli word, e.g. "XIXII" -> {0, 2}.
std::vector<std::size_t> support_of(const cudaq::spin_op_term &term) {
  std::vector<std::size_t> support;
  const std::string word = term.get_pauli_word();
  for (std::size_t i = 0; i < word.size(); ++i)
    if (word[i] != 'I')
      support.push_back(i);
  return support;
}

bool uses_horizontal_x_logical(sc_orientation orientation) {
  return orientation == sc_orientation::XV || orientation == sc_orientation::ZH;
}

void expect_surface_code_algebra(sc_orientation orientation) {
  constexpr std::uint32_t distance = 5;
  stabilizer_grid grid(distance, orientation);
  const auto expected_stabilizer_count = (distance * distance - 1) / 2;
  EXPECT_EQ(expected_stabilizer_count, grid.x_stabilizers.size());
  EXPECT_EQ(expected_stabilizer_count, grid.z_stabilizers.size());

  const auto top_row = top_row_support(distance);
  const auto left_column = left_column_support(distance);
  const auto expected_logical_x =
      uses_horizontal_x_logical(orientation) ? top_row : left_column;
  const auto expected_logical_z =
      uses_horizontal_x_logical(orientation) ? left_column : top_row;
  const auto observables = grid.get_spin_op_observables();
  ASSERT_EQ(2u, observables.size());
  // get_spin_op_observables() emits the X observable first, then Z.
  const auto logical_x = support_of(observables[0]);
  const auto logical_z = support_of(observables[1]);
  EXPECT_EQ(expected_logical_x, logical_x);
  EXPECT_EQ(expected_logical_z, logical_z);
  EXPECT_EQ(1, overlap_parity(logical_x, logical_z));

  for (const auto &x_stabilizer : grid.x_stabilizers) {
    EXPECT_EQ(0, overlap_parity(x_stabilizer, logical_z));
    for (const auto &z_stabilizer : grid.z_stabilizers)
      EXPECT_EQ(0, overlap_parity(x_stabilizer, z_stabilizer));
  }

  for (const auto &z_stabilizer : grid.z_stabilizers)
    EXPECT_EQ(0, overlap_parity(z_stabilizer, logical_x));
}

} // namespace

TEST(StabilizerTester, checkConstructFromSpinOps) {
  {
    // Constructor will always auto sort
    std::vector<cudaq::spin_op_term> stab{cudaq::spin_op::from_word("ZZZZIII"),
                                          cudaq::spin_op::from_word("XXXXIII"),
                                          cudaq::spin_op::from_word("IXXIXXI"),
                                          cudaq::spin_op::from_word("IIXXIXX"),
                                          cudaq::spin_op::from_word("IZZIZZI"),
                                          cudaq::spin_op::from_word("IIZZIZZ")};
    EXPECT_EQ(stab.size(), 6);
    auto parity = cudaq::qec::to_parity_matrix(stab);
    parity.dump();
    EXPECT_EQ(parity.rank(), 2);
    std::vector<std::size_t> expected_shape{6, 14};
    EXPECT_EQ(parity.shape(), expected_shape);

    {
      std::vector<int> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */

      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    {
      auto parity_x =
          cudaq::qec::to_parity_matrix(stab, cudaq::qec::stabilizer_type::X);
      printf("Hx:\n");
      parity_x.dump();
      EXPECT_EQ(parity_x.rank(), 2);
      std::vector<std::size_t> expected_shape{3, 7};
      EXPECT_EQ(parity_x.shape(), expected_shape);
      std::vector<int> data = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                               1, 1, 0, 0, 0, 1, 1, 0, 1, 1};
      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 7; j++)
          EXPECT_EQ(t.at({i, j}), parity_x.at({i, j}));
    }
  }
  {

    // Note testing here also that constructor sorts them
    std::vector<std::string> stab{"ZZZZIII", "XXXXIII", "IXXIXXI",
                                  "IIXXIXX", "IZZIZZI", "IIZZIZZ"};
    EXPECT_EQ(stab.size(), 6);
    auto parity = cudaq::qec::to_parity_matrix(stab);
    parity.dump();
    EXPECT_EQ(parity.rank(), 2);
    std::vector<std::size_t> expected_shape{6, 14};
    EXPECT_EQ(parity.shape(), expected_shape);
    {
      std::vector<int> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */

      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    {
      auto parity_z =
          cudaq::qec::to_parity_matrix(stab, cudaq::qec::stabilizer_type::Z);
      parity_z.dump();
      EXPECT_EQ(parity_z.rank(), 2);
      std::vector<std::size_t> expected_shape{3, 7};
      EXPECT_EQ(parity_z.shape(), expected_shape);
      std::vector<int> data = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                               1, 1, 0, 0, 0, 1, 1, 0, 1, 1};
      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 7; j++)
          EXPECT_EQ(t.at({i, j}), parity_z.at({i, j}));
    }
  }
}

TEST(StabilizerTester, checkToParityMatrixEdgeCases) {
  // Test case 1: Empty stabilizers vector (triggers line 56)
  {
    std::vector<cudaq::spin_op_term> empty_stab;
    auto parity_empty = cudaq::qec::to_parity_matrix(empty_stab);

    // Should return empty tensor
    EXPECT_EQ(parity_empty.size(), 0);
    // Empty tensor has rank 0, not 2
    EXPECT_EQ(parity_empty.rank(), 0);
  }

  // Test case 2: No Z stabilizers for Z type (triggers line 105)
  {
    // Create stabilizers with only X operations (no Z)
    std::vector<cudaq::spin_op_term> x_only_stab{
        cudaq::spin_op::from_word("XXXIII"),
        cudaq::spin_op::from_word("IXXXII"),
        cudaq::spin_op::from_word("IIXXXI")};

    auto parity_z_only = cudaq::qec::to_parity_matrix(
        x_only_stab, cudaq::qec::stabilizer_type::Z);

    // Should return empty tensor because no Z stabilizers found
    EXPECT_EQ(parity_z_only.size(), 0);
    EXPECT_EQ(parity_z_only.rank(), 0);
  }
}

TEST(QECCodeTester, checkSampleMemoryCircuit) {
  {
    // Steane tests
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> parity = steane->get_parity();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 10;
    int nRounds = 4;
    {
      // Prep0 experiment. Prep all data qubits in Z basis.
      // Measure all data qubits in the Z basis.
      // To correct it, find out how many times it flipped.
      // X errors flip the Z observable.
      // So when we get the predicted error data string E = E_X | E_Z
      // from the decoder, we apply E_X to our L_mz to correct it.
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep0, nShots, nRounds);
      syndromes.dump();
      EXPECT_EQ(syndromes.shape()[0], nShots);
      EXPECT_EQ(syndromes.shape()[1], 24u);

      // No noise here, should be all zeros
      int sum = 0;
      for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
        for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
          sum += syndromes.at({i, j});
      EXPECT_TRUE(sum == 0);

      // Prep0, should measure out logical |0> each shot
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
      // Prep1, should measure out logical |1> each shot
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
      // Prepp, should measure out logical |+> each shot
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
      // Prepm, should measure out logical |-> each shot
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
  }
}

TEST(QECCodeTester, checkSteane) {
  {
    // with default stabilizers
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> parity = steane->get_parity();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();
    {
      std::vector<uint8_t> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */

      std::vector<std::size_t> expected_shape{6, 14};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    EXPECT_EQ(2, observables.rank());
    EXPECT_EQ(2, observables.shape()[0]);
    EXPECT_EQ(14, observables.shape()[1]);
    EXPECT_EQ(2, Lx.rank());
    EXPECT_EQ(1, Lx.shape()[0]);
    EXPECT_EQ(7, Lx.shape()[1]);
    EXPECT_EQ(2, Lz.rank());
    EXPECT_EQ(1, Lz.shape()[0]);
    EXPECT_EQ(7, Lz.shape()[1]);
    {
      std::vector<std::vector<uint8_t>> true_observables = {
          {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}};
      std::vector<std::vector<uint8_t>> true_Lx = {{0, 0, 0, 0, 1, 1, 1}};
      std::vector<std::vector<uint8_t>> true_Lz = {{0, 0, 0, 0, 1, 1, 1}};
      for (std::size_t i = 0; i < observables.shape()[0]; ++i)
        for (std::size_t j = 0; j < observables.shape()[1]; ++j)
          EXPECT_EQ(true_observables[i][j], observables.at({i, j}));

      for (std::size_t i = 0; i < Lx.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lx.shape()[1]; ++j)
          EXPECT_EQ(true_Lx[i][j], Lx.at({i, j}));

      for (std::size_t i = 0; i < Lz.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lz.shape()[1]; ++j)
          EXPECT_EQ(true_Lz[i][j], Lz.at({i, j}));
    }

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(*steane, 10, 4);
    syndromes.dump();

    // No noise here, should be all zeros
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum == 0);
  }
  {
    // From Stabilizers
    std::vector<std::string> words{"ZZZZIII", "XXXXIII", "IXXIXXI",
                                   "IIXXIXX", "IZZIZZI", "IIZZIZZ"};
    std::vector<cudaq::spin_op_term> ops;
    for (auto &os : words)
      ops.emplace_back(cudaq::spin_op::from_word(os));
    cudaq::qec::sortStabilizerOps(ops);
    auto steane = cudaq::qec::get_code("steane", ops);
    auto parity = steane->get_parity();
    {
      std::vector<uint8_t> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */
      std::vector<std::size_t> expected_shape{6, 14};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(*steane, 10, 4);
    syndromes.dump();
    // No noise here, should be all zeros
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum == 0);
  }
}

TEST(QECCodeTester, checkCodeCapacity) {
  {
    auto steane = cudaq::qec::get_code("steane");
    auto Hz = steane->get_parity_z();
    int nShots = 10;
    double error_prob = 0;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(Hz, nShots, error_prob);
    EXPECT_EQ(2, Hz.rank());
    EXPECT_EQ(3, Hz.shape()[0]);
    EXPECT_EQ(7, Hz.shape()[1]);
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(Hz.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(Hz.shape()[1], data.shape()[1]);

    // Error prob = 0 should be all zeros
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[1]; ++j) {
        EXPECT_EQ(0, data.at({i, j}));
      }
    }

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[0]; ++j) {
        EXPECT_EQ(0, syndromes.at({i, j}));
      }
    }
  }
  {
    auto steane = cudaq::qec::get_code("steane");
    auto Hz = steane->get_parity_z();
    int nShots = 10;
    double error_prob = 0.15;
    unsigned seed = 1337;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(Hz, nShots, error_prob, seed);
    EXPECT_EQ(2, Hz.rank());
    EXPECT_EQ(3, Hz.shape()[0]);
    EXPECT_EQ(7, Hz.shape()[1]);
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(Hz.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(Hz.shape()[1], data.shape()[1]);
    // seed = 1337, error_prob = 0.15, nShots = 10
    // produces this data set:
    // This seed happens to only have weight 0 or 1 errors,
    // which are easy to check by hand.
    std::vector<std::vector<uint8_t>> seeded_data = {
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0}};
    EXPECT_EQ(nShots, seeded_data.size());
    EXPECT_EQ(Hz.shape()[1], seeded_data[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[1]; ++j) {
        EXPECT_EQ(seeded_data[i][j], data.at({i, j}));
      }
    }

    // Hand-checked syndromes
    std::vector<std::vector<uint8_t>> checked_syndromes = {
        {1, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 0, 0},
        {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 1}, {0, 0, 0}};
    EXPECT_EQ(nShots, checked_syndromes.size());
    EXPECT_EQ(Hz.shape()[0], checked_syndromes[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[0]; ++j) {
        EXPECT_EQ(checked_syndromes[i][j], syndromes.at({i, j}));
      }
    }
  }
  {
    auto steane = cudaq::qec::get_code("steane");
    auto Hz = steane->get_parity_z();
    int nShots = 10;
    double error_prob = 0.25;
    unsigned seed = 1337;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(Hz, nShots, error_prob, seed);
    EXPECT_EQ(2, Hz.rank());
    EXPECT_EQ(3, Hz.shape()[0]);
    EXPECT_EQ(7, Hz.shape()[1]);
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(Hz.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(Hz.shape()[1], data.shape()[1]);
    // seed = 1337, error_prob = 0.25, nShots = 10
    // produces this data set:
    // This seed has some higher weight errors which
    // where checked by hand
    std::vector<std::vector<uint8_t>> seeded_data = {
        {0, 1, 0, 1, 1, 0, 0}, {0, 0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0}};
    EXPECT_EQ(nShots, seeded_data.size());
    EXPECT_EQ(Hz.shape()[1], seeded_data[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[1]; ++j) {
        EXPECT_EQ(seeded_data[i][j], data.at({i, j}));
      }
    }

    // Hand-checked syndromes
    std::vector<std::vector<uint8_t>> checked_syndromes = {
        {0, 0, 1}, {1, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0, 0},
        {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 1}, {0, 0, 0}};

    EXPECT_EQ(nShots, checked_syndromes.size());
    EXPECT_EQ(Hz.shape()[0], checked_syndromes[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[0]; ++j) {
        EXPECT_EQ(checked_syndromes[i][j], syndromes.at({i, j}));
      }
    }
  }
}

TEST(QECCodeTester, checkCodeCapacityWithCodeObject) {
  // Test sample_code_capacity(const code &code, std::size_t nShots, double
  // error_probability)
  {
    auto steane = cudaq::qec::get_code("steane");
    int nShots = 5;
    double error_prob = 0.0;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(*steane, nShots, error_prob);

    // get_parity() returns the full parity check matrix (both X and Z
    // stabilizers)
    auto H = steane->get_parity();
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(H.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(H.shape()[1], data.shape()[1]);

    // Error prob = 0 should be all zeros
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < H.shape()[1]; ++j) {
        EXPECT_EQ(0, data.at({i, j}));
      }
    }

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < H.shape()[0]; ++j) {
        EXPECT_EQ(0, syndromes.at({i, j}));
      }
    }
  }

  // Test sample_code_capacity(const code &code, std::size_t nShots, double
  // error_probability, unsigned seed)
  {
    auto steane = cudaq::qec::get_code("steane");
    int nShots = 8;
    double error_prob = 0.15;
    unsigned seed = 1337;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(*steane, nShots, error_prob, seed);

    auto H = steane->get_parity();
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(H.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(H.shape()[1], data.shape()[1]);

    // Verify that results are deterministic with the same seed
    auto [syndromes2, data2] =
        cudaq::qec::sample_code_capacity(*steane, nShots, error_prob, seed);

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < H.shape()[1]; ++j) {
        EXPECT_EQ(data.at({i, j}), data2.at({i, j}));
      }
    }

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < H.shape()[0]; ++j) {
        EXPECT_EQ(syndromes.at({i, j}), syndromes2.at({i, j}));
      }
    }
  }

  // Test with different code type
  {
    auto repetition = cudaq::qec::get_code(
        "repetition", cudaqx::heterogeneous_map{{"distance", 3}});
    int nShots = 8;
    double error_prob = 0.2;
    unsigned seed = 42;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(*repetition, nShots, error_prob, seed);

    auto H = repetition->get_parity();
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(H.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(H.shape()[1], data.shape()[1]);

    // Verify that results are consistent - if we call with same parameters
    // again, we should get the same results
    auto [syndromes2, data2] =
        cudaq::qec::sample_code_capacity(*repetition, nShots, error_prob, seed);

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < H.shape()[1]; ++j) {
        EXPECT_EQ(data.at({i, j}), data2.at({i, j}));
      }
    }

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < H.shape()[0]; ++j) {
        EXPECT_EQ(syndromes.at({i, j}), syndromes2.at({i, j}));
      }
    }
  }

  // Test without seed (random behavior)
  {
    auto steane = cudaq::qec::get_code("steane");
    int nShots = 3;
    double error_prob = 0.1;

    auto [syndromes1, data1] =
        cudaq::qec::sample_code_capacity(*steane, nShots, error_prob);
    auto [syndromes2, data2] =
        cudaq::qec::sample_code_capacity(*steane, nShots, error_prob);

    auto H = steane->get_parity();

    // Verify shape is correct
    EXPECT_EQ(nShots, syndromes1.shape()[0]);
    EXPECT_EQ(H.shape()[0], syndromes1.shape()[1]);
    EXPECT_EQ(nShots, data1.shape()[0]);
    EXPECT_EQ(H.shape()[1], data1.shape()[1]);

    // Same for second call
    EXPECT_EQ(nShots, syndromes2.shape()[0]);
    EXPECT_EQ(H.shape()[0], syndromes2.shape()[1]);
    EXPECT_EQ(nShots, data2.shape()[0]);
    EXPECT_EQ(H.shape()[1], data2.shape()[1]);

    // Since we're using random seeds, the results might be different
    // We just verify the function completes successfully and returns valid
    // shapes
  }
}

TEST(QECCodeTester, checkRepetition) {
  {
    // must provide distance
    EXPECT_THROW(cudaq::qec::get_code("repetition"), std::runtime_error);
  }
  auto repetition = cudaq::qec::get_code(
      "repetition", cudaqx::heterogeneous_map{{"distance", 9}});

  EXPECT_EQ(repetition->get_num_x_stabilizers(), 0u);
  EXPECT_EQ(repetition->get_num_z_stabilizers(), 8u);

  {
    auto stabilizers = repetition->get_stabilizers();

    std::vector<std::string> actual_stabs;
    for (auto &s : stabilizers)
      actual_stabs.push_back(s.get_pauli_word());

    std::vector<std::string> expected_strings = {
        "ZZIIIIIII", "IZZIIIIII", "IIZZIIIII", "IIIZZIIII",
        "IIIIZZIII", "IIIIIZZII", "IIIIIIZZI", "IIIIIIIZZ"};

    EXPECT_EQ(actual_stabs, expected_strings);
    auto parity = repetition->get_parity();
    auto Hx = repetition->get_parity_x();
    auto Hz = repetition->get_parity_z();
    EXPECT_EQ(0, Hx.rank());
    EXPECT_EQ(2, Hz.rank());
    parity.dump();
    std::vector<uint8_t> data = {
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 3 */
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 4 */
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 5 */
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 6 */
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}; /* row 7 */
    std::vector<std::size_t> expected_shape{8, 18};
    cudaqx::tensor<uint8_t> t(expected_shape);
    t.borrow(data.data());
    for (std::size_t i = 0; i < t.shape()[0]; i++)
      for (std::size_t j = 0; j < t.shape()[1]; j++)
        EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
  }
  {
    cudaqx::tensor<uint8_t> observables =
        repetition->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = repetition->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = repetition->get_observables_z();

    EXPECT_EQ(2, observables.rank());
    EXPECT_EQ(1, observables.shape()[0]);
    EXPECT_EQ(18, observables.shape()[1]);
    EXPECT_EQ(0, Lx.rank());
    EXPECT_EQ(2, Lz.rank());
    EXPECT_EQ(1, Lz.shape()[0]);
    EXPECT_EQ(9, Lz.shape()[1]);
    {
      std::vector<std::vector<uint8_t>> true_observables = {
          {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
      std::vector<std::vector<uint8_t>> true_Lz = {{1, 0, 0, 0, 0, 0, 0, 0, 0}};
      for (std::size_t i = 0; i < observables.shape()[0]; ++i)
        for (std::size_t j = 0; j < observables.shape()[1]; ++j)
          EXPECT_EQ(true_observables[i][j], observables.at({i, j}));

      for (std::size_t i = 0; i < Lz.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lz.shape()[1]; ++j)
          EXPECT_EQ(true_Lz[i][j], Lz.at({i, j}));
    }
  }
  {
    auto parity = repetition->get_parity();
    auto parity_z = repetition->get_parity_z();
    int nShots = 10;
    int nRounds = 4;
    auto [syndromes, data_mz] =
        cudaq::qec::sample_memory_circuit(*repetition, nShots, nRounds);
    syndromes.dump();
    data_mz.dump();
    EXPECT_EQ(syndromes.shape()[0], static_cast<std::size_t>(nShots));
    EXPECT_EQ(syndromes.shape()[1], 40u);
    EXPECT_EQ(data_mz.shape()[0], static_cast<std::size_t>(nShots));
    EXPECT_EQ(data_mz.shape()[1], parity_z.shape()[1]);

    // No noise here, all detectors should be zero
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum == 0);
  }
}

TEST(QECNoiseModelTester, ConstructsTwoQubitBitflipChannel) {
  // Direct construction covers the inline Kraus construction path.
  EXPECT_NO_THROW({ cudaq::qec::two_qubit_bitflip channel(0.05); });
}

TEST(QECCodeTester, checkSurfaceCodeOrientationFromString) {
  using cudaq::qec::surface_code::sc_orientation_from_str;

  EXPECT_EQ(sc_orientation::XV, sc_orientation_from_str("XV"));
  EXPECT_EQ(sc_orientation::XV, sc_orientation_from_str("o1"));
  EXPECT_EQ(sc_orientation::XH, sc_orientation_from_str(" xh "));
  EXPECT_EQ(sc_orientation::XH, sc_orientation_from_str("O2"));
  EXPECT_EQ(sc_orientation::ZV, sc_orientation_from_str("ZV"));
  EXPECT_EQ(sc_orientation::ZV, sc_orientation_from_str(" o3 "));
  EXPECT_EQ(sc_orientation::ZH, sc_orientation_from_str("zh"));
  EXPECT_EQ(sc_orientation::ZH, sc_orientation_from_str("O4"));
  EXPECT_THROW(sc_orientation_from_str("north"), std::runtime_error);

  auto surf_code = cudaq::qec::get_code(
      "surface_code",
      cudaqx::heterogeneous_map{{"distance", 3}, {"orientation", "O2"}});
  auto *surface =
      dynamic_cast<cudaq::qec::surface_code::surface_code *>(surf_code.get());
  ASSERT_NE(nullptr, surface);
  EXPECT_EQ(sc_orientation::XH, surface->grid.get_orientation());
}

TEST(QECCodeTester, checkSurfaceCodeDefaultZHLayout) {
  // Default construction uses the ZH orientation.
  stabilizer_grid grid(3);
  const auto role_at = [&grid](std::size_t row, std::size_t col) {
    return grid.roles[row * grid.grid_length + col];
  };

  EXPECT_EQ(surface_role::empty, role_at(0, 0));
  EXPECT_EQ(surface_role::empty, role_at(0, 1));
  EXPECT_EQ(surface_role::amz, role_at(0, 2));
  EXPECT_EQ(surface_role::empty, role_at(0, 3));

  EXPECT_EQ(surface_role::amx, role_at(1, 0));
  EXPECT_EQ(surface_role::amz, role_at(1, 1));
  EXPECT_EQ(surface_role::amx, role_at(1, 2));
  EXPECT_EQ(surface_role::empty, role_at(1, 3));

  EXPECT_EQ(surface_role::empty, role_at(2, 0));
  EXPECT_EQ(surface_role::amx, role_at(2, 1));
  EXPECT_EQ(surface_role::amz, role_at(2, 2));
  EXPECT_EQ(surface_role::amx, role_at(2, 3));

  EXPECT_EQ(surface_role::empty, role_at(3, 0));
  EXPECT_EQ(surface_role::amz, role_at(3, 1));
  EXPECT_EQ(surface_role::empty, role_at(3, 2));
  EXPECT_EQ(surface_role::empty, role_at(3, 3));
}

TEST(QECCodeTester, checkSurfaceCodeOrientationAlgebra) {
  for (auto orientation : {sc_orientation::XV, sc_orientation::XH,
                           sc_orientation::ZV, sc_orientation::ZH}) {
    SCOPED_TRACE(static_cast<int>(orientation));
    expect_surface_code_algebra(orientation);
  }
}

// Pin the EXACT role grid of every orientation at d=3 so permuting the
// orientation labels (or regressing the boundary logic) fails. Grids derived by
// tracing generate_grid_roles()/role_for_parity() and cross-checked against
// Ising's code_rotation stabilizers. Boundary summary: XV/ZH put amx on
// left/right and amz on top/bottom; XH/ZV put amx on top/bottom and amz on
// left/right (XV vs ZH and XH vs ZV differ in the bulk checkerboard).
TEST(QECCodeTester, checkSurfaceCodeAllOrientationGeometries) {
  // ---- XV ----
  {
    SCOPED_TRACE("XV");
    stabilizer_grid grid(3, sc_orientation::XV);
    const auto role_at = [&grid](std::size_t row, std::size_t col) {
      return grid.roles[row * grid.grid_length + col];
    };
    EXPECT_EQ(surface_role::empty, role_at(0, 0));
    EXPECT_EQ(surface_role::amz, role_at(0, 1));
    EXPECT_EQ(surface_role::empty, role_at(0, 2));
    EXPECT_EQ(surface_role::empty, role_at(0, 3));

    EXPECT_EQ(surface_role::empty, role_at(1, 0));
    EXPECT_EQ(surface_role::amx, role_at(1, 1));
    EXPECT_EQ(surface_role::amz, role_at(1, 2));
    EXPECT_EQ(surface_role::amx, role_at(1, 3));

    EXPECT_EQ(surface_role::amx, role_at(2, 0));
    EXPECT_EQ(surface_role::amz, role_at(2, 1));
    EXPECT_EQ(surface_role::amx, role_at(2, 2));
    EXPECT_EQ(surface_role::empty, role_at(2, 3));

    EXPECT_EQ(surface_role::empty, role_at(3, 0));
    EXPECT_EQ(surface_role::empty, role_at(3, 1));
    EXPECT_EQ(surface_role::amz, role_at(3, 2));
    EXPECT_EQ(surface_role::empty, role_at(3, 3));
  }

  // ---- XH ----
  {
    SCOPED_TRACE("XH");
    stabilizer_grid grid(3, sc_orientation::XH);
    const auto role_at = [&grid](std::size_t row, std::size_t col) {
      return grid.roles[row * grid.grid_length + col];
    };
    EXPECT_EQ(surface_role::empty, role_at(0, 0));
    EXPECT_EQ(surface_role::empty, role_at(0, 1));
    EXPECT_EQ(surface_role::amx, role_at(0, 2));
    EXPECT_EQ(surface_role::empty, role_at(0, 3));

    EXPECT_EQ(surface_role::amz, role_at(1, 0));
    EXPECT_EQ(surface_role::amx, role_at(1, 1));
    EXPECT_EQ(surface_role::amz, role_at(1, 2));
    EXPECT_EQ(surface_role::empty, role_at(1, 3));

    EXPECT_EQ(surface_role::empty, role_at(2, 0));
    EXPECT_EQ(surface_role::amz, role_at(2, 1));
    EXPECT_EQ(surface_role::amx, role_at(2, 2));
    EXPECT_EQ(surface_role::amz, role_at(2, 3));

    EXPECT_EQ(surface_role::empty, role_at(3, 0));
    EXPECT_EQ(surface_role::amx, role_at(3, 1));
    EXPECT_EQ(surface_role::empty, role_at(3, 2));
    EXPECT_EQ(surface_role::empty, role_at(3, 3));
  }

  // ---- ZV ----
  {
    SCOPED_TRACE("ZV");
    stabilizer_grid grid(3, sc_orientation::ZV);
    const auto role_at = [&grid](std::size_t row, std::size_t col) {
      return grid.roles[row * grid.grid_length + col];
    };
    EXPECT_EQ(surface_role::empty, role_at(0, 0));
    EXPECT_EQ(surface_role::amx, role_at(0, 1));
    EXPECT_EQ(surface_role::empty, role_at(0, 2));
    EXPECT_EQ(surface_role::empty, role_at(0, 3));

    EXPECT_EQ(surface_role::empty, role_at(1, 0));
    EXPECT_EQ(surface_role::amz, role_at(1, 1));
    EXPECT_EQ(surface_role::amx, role_at(1, 2));
    EXPECT_EQ(surface_role::amz, role_at(1, 3));

    EXPECT_EQ(surface_role::amz, role_at(2, 0));
    EXPECT_EQ(surface_role::amx, role_at(2, 1));
    EXPECT_EQ(surface_role::amz, role_at(2, 2));
    EXPECT_EQ(surface_role::empty, role_at(2, 3));

    EXPECT_EQ(surface_role::empty, role_at(3, 0));
    EXPECT_EQ(surface_role::empty, role_at(3, 1));
    EXPECT_EQ(surface_role::amx, role_at(3, 2));
    EXPECT_EQ(surface_role::empty, role_at(3, 3));
  }

  // ---- ZH ----
  {
    SCOPED_TRACE("ZH");
    stabilizer_grid grid(3, sc_orientation::ZH);
    const auto role_at = [&grid](std::size_t row, std::size_t col) {
      return grid.roles[row * grid.grid_length + col];
    };
    EXPECT_EQ(surface_role::empty, role_at(0, 0));
    EXPECT_EQ(surface_role::empty, role_at(0, 1));
    EXPECT_EQ(surface_role::amz, role_at(0, 2));
    EXPECT_EQ(surface_role::empty, role_at(0, 3));

    EXPECT_EQ(surface_role::amx, role_at(1, 0));
    EXPECT_EQ(surface_role::amz, role_at(1, 1));
    EXPECT_EQ(surface_role::amx, role_at(1, 2));
    EXPECT_EQ(surface_role::empty, role_at(1, 3));

    EXPECT_EQ(surface_role::empty, role_at(2, 0));
    EXPECT_EQ(surface_role::amx, role_at(2, 1));
    EXPECT_EQ(surface_role::amz, role_at(2, 2));
    EXPECT_EQ(surface_role::amx, role_at(2, 3));

    EXPECT_EQ(surface_role::empty, role_at(3, 0));
    EXPECT_EQ(surface_role::amz, role_at(3, 1));
    EXPECT_EQ(surface_role::empty, role_at(3, 2));
    EXPECT_EQ(surface_role::empty, role_at(3, 3));
  }
}

TEST(QECCodeTester, checkSurfaceCode) {
  {
    // must provide distance
    EXPECT_THROW(cudaq::qec::get_code("surface_code"), std::runtime_error);
  }
  {
    // with default stabilizers
    auto surf_code = cudaq::qec::get_code(
        "surface_code", cudaqx::heterogeneous_map{{"distance", 3}});
    cudaqx::tensor<uint8_t> parity = surf_code->get_parity();
    cudaqx::tensor<uint8_t> parity_x = surf_code->get_parity_x();
    cudaqx::tensor<uint8_t> parity_z = surf_code->get_parity_z();
    cudaqx::tensor<uint8_t> observables =
        surf_code->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = surf_code->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = surf_code->get_observables_z();

    {
      // This is just a regression check, this has not been hand tested
      std::vector<uint8_t> data = {
          1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,  /* row 5 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 6 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1}; /* row 7 */

      std::vector<std::size_t> expected_shape{8, 18};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 8; i++)
        for (std::size_t j = 0; j < 18; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    {
      // This is just a regression check, this has not been hand tested
      std::vector<uint8_t> data = {1, 0, 0, 1, 0, 0, 0, 0, 0,  /* row 0 */
                                   0, 1, 1, 0, 1, 1, 0, 0, 0,  /* row 1 */
                                   0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 2 */
                                   0, 0, 0, 0, 0, 1, 0, 0, 1}; /* row 3 */

      std::vector<std::size_t> expected_shape{4, 9};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 9; j++)
          EXPECT_EQ(t.at({i, j}), parity_x.at({i, j}));
    }
    {
      // This is just a regression check, this has not been hand tested
      std::vector<uint8_t> data = {1, 1, 0, 1, 1, 0, 0, 0, 0,  /* row 0 */
                                   0, 1, 1, 0, 0, 0, 0, 0, 0,  /* row 1 */
                                   0, 0, 0, 0, 1, 1, 0, 1, 1,  /* row 2 */
                                   0, 0, 0, 0, 0, 0, 1, 1, 0}; /* row 3 */

      std::vector<std::size_t> expected_shape{4, 9};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 9; j++)
          EXPECT_EQ(t.at({i, j}), parity_z.at({i, j}));
    }
    EXPECT_EQ(2, observables.rank());
    EXPECT_EQ(2, observables.shape()[0]);
    EXPECT_EQ(18, observables.shape()[1]);
    EXPECT_EQ(2, Lx.rank());
    EXPECT_EQ(1, Lx.shape()[0]);
    EXPECT_EQ(9, Lx.shape()[1]);
    EXPECT_EQ(2, Lz.rank());
    EXPECT_EQ(1, Lz.shape()[0]);
    EXPECT_EQ(9, Lz.shape()[1]);
    {
      std::vector<std::vector<uint8_t>> true_observables = {
          {1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // Z first
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0}}; // Then X
      std::vector<std::vector<uint8_t>> true_Lx = {{1, 1, 1, 0, 0, 0, 0, 0, 0}};
      std::vector<std::vector<uint8_t>> true_Lz = {{1, 0, 0, 1, 0, 0, 1, 0, 0}};
      for (std::size_t i = 0; i < observables.shape()[0]; ++i)
        for (std::size_t j = 0; j < observables.shape()[1]; ++j)
          EXPECT_EQ(true_observables[i][j], observables.at({i, j}));

      for (std::size_t i = 0; i < Lx.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lx.shape()[1]; ++j)
          EXPECT_EQ(true_Lx[i][j], Lx.at({i, j}));

      for (std::size_t i = 0; i < Lz.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lz.shape()[1]; ++j)
          EXPECT_EQ(true_Lz[i][j], Lz.at({i, j}));
    }

    // Test surface code qubit count methods for distance 3
    EXPECT_EQ(9, surf_code->get_num_data_qubits())
        << "Surface code distance 3 should have 9 data qubits (3*3)";
    EXPECT_EQ(8, surf_code->get_num_ancilla_qubits())
        << "Surface code distance 3 should have 8 ancilla qubits (3*3-1)";
    EXPECT_EQ(4, surf_code->get_num_ancilla_x_qubits())
        << "Surface code distance 3 should have 4 X ancilla qubits";
    EXPECT_EQ(4, surf_code->get_num_ancilla_z_qubits())
        << "Surface code distance 3 should have 4 Z ancilla qubits";

    // Verify relationships between qubit counts
    EXPECT_EQ(surf_code->get_num_ancilla_x_qubits() +
                  surf_code->get_num_ancilla_z_qubits(),
              surf_code->get_num_ancilla_qubits())
        << "X and Z ancilla qubits should sum to total ancilla qubits";
    EXPECT_EQ(surf_code->get_num_data_qubits() +
                  surf_code->get_num_ancilla_qubits(),
              17)
        << "Total qubits should be 17 for distance 3 surface code";
  }
}

// expect |0>, |+> to measure out 0 in respective bases
// expect |1>, |-> to measure out 1 in respective bases
bool noiseless_logical_SPAM_test(const cudaq::qec::code &code,
                                 cudaq::qec::operation statePrep,
                                 uint8_t expected_result) {
  cudaqx::tensor<uint8_t> Lx = code.get_observables_x();
  cudaqx::tensor<uint8_t> Lz = code.get_observables_z();

  // We measure Z observable in prep0, prep1 experiments
  cudaqx::tensor<uint8_t> measured_observable(Lz.shape());
  measured_observable.borrow(Lz.data());
  // We measure X observable in prepp, prepm experiments
  if (statePrep == cudaq::qec::operation::prepp ||
      statePrep == cudaq::qec::operation::prepm) {
    measured_observable = cudaqx::tensor<uint8_t>(Lx.shape());
    measured_observable.borrow(Lx.data());
  }

  int nShots = 10;
  // Number of rounds does not matter for noiseless, all should be zero.
  int nRounds = 4;
  auto [syndromes, d] =
      cudaq::qec::sample_memory_circuit(code, statePrep, nShots, nRounds);
  syndromes.dump();

  printf("data:\n");
  d.dump();
  printf("Obs:\n");
  measured_observable.dump();
  cudaqx::tensor<uint8_t> logical_measurement =
      measured_observable.dot(d.transpose()) % 2;
  printf("logical_measurement:\n");
  logical_measurement.dump();

  // With no noise, each shot should measure out the expected value
  for (size_t shot = 0; shot < nShots; ++shot) {
    // All codes have only 1 logical qubit for now
    for (size_t lQ = 0; lQ < 1; ++lQ) {
      if (logical_measurement.at({lQ, shot}) != expected_result) {
        printf("shot: %zu, lQ: %zu\n", shot, lQ);
        std::cout << +logical_measurement.at({lQ, shot}) << "\n";
        return false;
      }
    }
  }
  return true;
}

TEST(QECCodeTester, checkSteaneSPAM) {
  auto steane = cudaq::qec::get_code("steane");
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep0, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep1, 1));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepp, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepm, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep0, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep1, 0));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepp, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepm, 0));
}

TEST(QECCodeTester, checkSteaneQubitCounts) {
  // Test Steane code qubit count methods
  auto steane = cudaq::qec::get_code("steane");

  // Test data qubits count
  EXPECT_EQ(7, steane->get_num_data_qubits())
      << "Steane code should have 7 data qubits";

  // Test total ancilla qubits count
  EXPECT_EQ(6, steane->get_num_ancilla_qubits())
      << "Steane code should have 6 total ancilla qubits";

  // Test X ancilla qubits count
  EXPECT_EQ(3, steane->get_num_ancilla_x_qubits())
      << "Steane code should have 3 X ancilla qubits";

  // Test Z ancilla qubits count
  EXPECT_EQ(3, steane->get_num_ancilla_z_qubits())
      << "Steane code should have 3 Z ancilla qubits";

  // Verify that X + Z ancilla qubits equals total ancilla qubits
  EXPECT_EQ(steane->get_num_ancilla_qubits(),
            steane->get_num_ancilla_x_qubits() +
                steane->get_num_ancilla_z_qubits())
      << "Total ancilla qubits should equal sum of X and Z ancilla qubits";

  // Test total qubit count (data + ancilla)
  std::size_t total_qubits =
      steane->get_num_data_qubits() + steane->get_num_ancilla_qubits();
  EXPECT_EQ(13, total_qubits)
      << "Steane code should use 13 total qubits (7 data + 6 ancilla)";
}

TEST(QECCodeTester, checkRepetitionSPAM) {
  // only Z basis for repetition
  auto repetition = cudaq::qec::get_code(
      "repetition", cudaqx::heterogeneous_map{{"distance", 9}});
  EXPECT_TRUE(noiseless_logical_SPAM_test(*repetition,
                                          cudaq::qec::operation::prep0, 0));
  EXPECT_TRUE(noiseless_logical_SPAM_test(*repetition,
                                          cudaq::qec::operation::prep1, 1));
  EXPECT_FALSE(noiseless_logical_SPAM_test(*repetition,
                                           cudaq::qec::operation::prep0, 1));
  EXPECT_FALSE(noiseless_logical_SPAM_test(*repetition,
                                           cudaq::qec::operation::prep1, 0));
}

TEST(QECCodeTester, checkSurfaceCodeSPAM) {
  // Must compile with stim for larger distances
  auto surf_code = cudaq::qec::get_code(
      "surface_code", cudaqx::heterogeneous_map{{"distance", 3}});
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep0, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep1, 1));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepp, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepm, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep0, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep1, 0));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepp, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepm, 0));
}

TEST(QECCodeTester, checkStabilizerGrid) {
  {
    int distance = 3;

    cudaq::qec::surface_code::stabilizer_grid grid(distance);

    grid.print_stabilizer_grid();
    grid.print_stabilizer_coords();
    grid.print_stabilizer_indices();
    grid.print_stabilizer_maps();
    grid.print_data_grid();
    grid.print_stabilizers();

    EXPECT_EQ(3, grid.distance);
    EXPECT_EQ(4, grid.grid_length);
    EXPECT_EQ(16, grid.roles.size());
    EXPECT_EQ(4, grid.x_stab_coords.size());
    EXPECT_EQ(4, grid.z_stab_coords.size());
    EXPECT_EQ(4, grid.x_stab_indices.size());
    EXPECT_EQ(4, grid.z_stab_indices.size());
    EXPECT_EQ(9, grid.data_coords.size());
    EXPECT_EQ(9, grid.data_indices.size());
    EXPECT_EQ(4, grid.x_stabilizers.size());
    EXPECT_EQ(4, grid.z_stabilizers.size());
  }
  {
    int distance = 5;

    cudaq::qec::surface_code::stabilizer_grid grid(distance);

    grid.print_stabilizer_grid();
    grid.print_stabilizer_coords();
    grid.print_stabilizer_indices();
    grid.print_stabilizer_maps();
    grid.print_data_grid();
    grid.print_stabilizers();

    EXPECT_EQ(5, grid.distance);
    EXPECT_EQ(6, grid.grid_length);
    EXPECT_EQ(36, grid.roles.size());
    EXPECT_EQ(12, grid.x_stab_coords.size());
    EXPECT_EQ(12, grid.z_stab_coords.size());
    EXPECT_EQ(12, grid.x_stab_indices.size());
    EXPECT_EQ(12, grid.z_stab_indices.size());
    EXPECT_EQ(25, grid.data_coords.size());
    EXPECT_EQ(25, grid.data_indices.size());
    EXPECT_EQ(12, grid.x_stabilizers.size());
    EXPECT_EQ(12, grid.z_stabilizers.size());
  }
  {
    int distance = 17;

    cudaq::qec::surface_code::stabilizer_grid grid(distance);

    EXPECT_EQ(17, grid.distance);
    EXPECT_EQ(18, grid.grid_length);
    EXPECT_EQ(324, grid.roles.size());
    EXPECT_EQ(144, grid.x_stab_coords.size());
    EXPECT_EQ(144, grid.z_stab_coords.size());
    EXPECT_EQ(144, grid.x_stab_indices.size());
    EXPECT_EQ(144, grid.z_stab_indices.size());
    EXPECT_EQ(289, grid.data_coords.size());
    EXPECT_EQ(289, grid.data_indices.size());
    EXPECT_EQ(144, grid.x_stabilizers.size());
    EXPECT_EQ(144, grid.z_stabilizers.size());
  }
}

TEST(SurfaceCodeTester, checkCnotSchedule) {
  using cudaq::qec::surface_code::stabilizer_grid;

  // Verify a schedule matrix against its parity matrix and the hook-error
  // requirement: the pair of data qubits touched by the two *last* CNOTs of
  // every weight-4 plaquette must be perpendicular to the same-type logical.
  auto check_schedule = [](const cudaqx::tensor<uint8_t> &sched,
                           const cudaqx::tensor<uint8_t> &parity,
                           uint32_t distance, bool expect_vertical_hook_pair,
                           const std::string &label) {
    ASSERT_EQ(sched.shape(), parity.shape()) << label;
    const auto num_stabs = sched.shape()[0];
    const auto num_data = sched.shape()[1];
    for (std::size_t s = 0; s < num_stabs; ++s) {
      std::vector<std::size_t> support; // data idx per ascending timestep
      std::vector<bool> steps_seen(5, false);
      for (std::size_t d = 0; d < num_data; ++d) {
        auto step = sched.at({s, d});
        // Support pattern must match the parity-matrix row exactly (this
        // also pins the row ordering to the sorted parity rows).
        EXPECT_EQ(step != 0, parity.at({s, d}) != 0)
            << label << " row " << s << " col " << d;
        if (step == 0)
          continue;
        ASSERT_LE(step, 4) << label << " row " << s;
        EXPECT_FALSE(steps_seen[step])
            << label << " row " << s << " duplicate step";
        steps_seen[step] = true;
      }
      for (uint8_t step = 1; step <= 4; ++step)
        for (std::size_t d = 0; d < num_data; ++d)
          if (sched.at({s, d}) == step)
            support.push_back(d);
      ASSERT_TRUE(support.size() == 2 || support.size() == 4)
          << label << " row " << s;
      if (support.size() == 4) {
        // Hook errors on the ancilla propagate to the data qubits of the two
        // remaining CNOTs; that pair must run against the logical direction.
        auto a = support[2], b = support[3];
        int row_a = a / distance, col_a = a % distance;
        int row_b = b / distance, col_b = b % distance;
        if (expect_vertical_hook_pair) {
          EXPECT_EQ(col_a, col_b) << label << " row " << s;
          EXPECT_EQ(1, std::abs(row_a - row_b)) << label << " row " << s;
        } else {
          EXPECT_EQ(row_a, row_b) << label << " row " << s;
          EXPECT_EQ(1, std::abs(col_a - col_b)) << label << " row " << s;
        }
      }
    }
  };

  for (uint32_t distance : {3u, 5u, 7u}) {
    for (auto orientation : {sc_orientation::XV, sc_orientation::XH,
                             sc_orientation::ZV, sc_orientation::ZH}) {
      stabilizer_grid grid(distance, orientation);
      auto stabs = grid.get_spin_op_stabilizers();
      auto parity_x =
          cudaq::qec::to_parity_matrix(stabs, cudaq::qec::stabilizer_type::X);
      auto parity_z =
          cudaq::qec::to_parity_matrix(stabs, cudaq::qec::stabilizer_type::Z);
      auto sched_x = grid.get_cnot_schedule_x();
      auto sched_z = grid.get_cnot_schedule_z();

      const std::string label =
          "d=" + std::to_string(distance) +
          " orientation=" + std::to_string(static_cast<int>(orientation));

      // XV/ZH place the X logical along the top data row (horizontal), so X
      // hook pairs must be vertical and Z hook pairs horizontal; XH/ZV swap
      // the logicals and hence the requirement.
      const bool x_logical_horizontal = orientation == sc_orientation::XV ||
                                        orientation == sc_orientation::ZH;
      check_schedule(sched_x, parity_x, distance,
                     /*expect_vertical_hook_pair=*/x_logical_horizontal,
                     label + " X");
      check_schedule(sched_z, parity_z, distance,
                     /*expect_vertical_hook_pair=*/!x_logical_horizontal,
                     label + " Z");

      // The X and Z schedules must also be conflict-free if interleaved: no
      // data qubit is touched by two CNOTs in the same timestep.
      const auto num_data = distance * distance;
      for (uint8_t step = 1; step <= 4; ++step) {
        std::vector<int> touches(num_data, 0);
        for (std::size_t s = 0; s < sched_x.shape()[0]; ++s)
          for (std::size_t d = 0; d < num_data; ++d)
            if (sched_x.at({s, d}) == step)
              touches[d]++;
        for (std::size_t s = 0; s < sched_z.shape()[0]; ++s)
          for (std::size_t d = 0; d < num_data; ++d)
            if (sched_z.at({s, d}) == step)
              touches[d]++;
        for (std::size_t d = 0; d < num_data; ++d)
          EXPECT_LE(touches[d], 1)
              << label << " step " << int(step) << " data " << d;
      }

      // The surface_code plugin must expose the grid schedule.
      const std::string orientation_str =
          orientation == sc_orientation::XV   ? "XV"
          : orientation == sc_orientation::XH ? "XH"
          : orientation == sc_orientation::ZV ? "ZV"
                                              : "ZH";
      auto code = cudaq::qec::get_code(
          "surface_code", cudaqx::heterogeneous_map{
                              {"distance", static_cast<std::size_t>(distance)},
                              {"orientation", orientation_str}});
      auto code_sched_x = code->get_stabilizer_schedule_x();
      ASSERT_EQ(code_sched_x.shape(), sched_x.shape()) << label;
      for (std::size_t s = 0; s < sched_x.shape()[0]; ++s)
        for (std::size_t d = 0; d < num_data; ++d)
          EXPECT_EQ(code_sched_x.at({s, d}), sched_x.at({s, d})) << label;
      auto code_sched_z = code->get_stabilizer_schedule_z();
      ASSERT_EQ(code_sched_z.shape(), sched_z.shape()) << label;
      for (std::size_t s = 0; s < sched_z.shape()[0]; ++s)
        for (std::size_t d = 0; d < num_data; ++d)
          EXPECT_EQ(code_sched_z.at({s, d}), sched_z.at({s, d})) << label;
    }
  }
}

TEST(SurfaceCodeTester, checkVec2dOperators) {
  // Test vec2d operators that are used in stabilizer_grid maps
  using cudaq::qec::surface_code::vec2d;

  // Test constructor
  vec2d v1(2, 3);
  vec2d v2(5, 7);
  vec2d v3(2, 3); // same as v1
  vec2d v4(1, 3); // for ordering test

  // Test operator+
  vec2d sum = v1 + v2;
  EXPECT_EQ(sum.row, 7);  // 2 + 5
  EXPECT_EQ(sum.col, 10); // 3 + 7

  // Test operator-
  vec2d diff = v2 - v1;
  EXPECT_EQ(diff.row, 3); // 5 - 2
  EXPECT_EQ(diff.col, 4); // 7 - 3

  // Test operator== (equality)
  EXPECT_TRUE(v1 == v3);  // same coordinates
  EXPECT_FALSE(v1 == v2); // different coordinates
  EXPECT_FALSE(v1 == v4); // different row

  // Test operator< (ordering used in std::map)
  EXPECT_TRUE(v4 < v1);  // (1,3) < (2,3) - different row
  EXPECT_TRUE(v1 < v2);  // (2,3) < (5,7) - different row
  EXPECT_FALSE(v1 < v3); // (2,3) not < (2,3) - same coordinates
  EXPECT_FALSE(v2 < v1); // (5,7) not < (2,3)

  // Test ordering by column when row is same
  vec2d v5(2, 1);        // same row as v1, but smaller column
  vec2d v6(2, 5);        // same row as v1, but larger column
  EXPECT_TRUE(v5 < v1);  // (2,1) < (2,3) - same row, smaller col
  EXPECT_TRUE(v1 < v6);  // (2,3) < (2,5) - same row, larger col
  EXPECT_FALSE(v1 < v5); // (2,3) not < (2,1)

  // Verify operator< works correctly with std::map (this is where it's actually
  // used)
  std::map<vec2d, int> coord_map;
  coord_map[v1] = 1;
  coord_map[v2] = 2;
  coord_map[v4] = 4;
  coord_map[v5] = 5;
  coord_map[v6] = 6;

  // Verify all values are stored correctly
  EXPECT_EQ(coord_map[v1], 1);
  EXPECT_EQ(coord_map[v2], 2);
  EXPECT_EQ(coord_map[v4], 4);
  EXPECT_EQ(coord_map[v5], 5);
  EXPECT_EQ(coord_map[v6], 6);

  // Verify that v3 (same as v1) accesses the same value
  EXPECT_EQ(coord_map[v3], 1);

  // Verify map ordering by iterating (should be sorted by operator<)
  std::vector<vec2d> expected_order = {v4, v5, v1, v6,
                                       v2}; // (1,3), (2,1), (2,3), (2,5), (5,7)
  std::vector<vec2d> actual_order;
  for (const auto &pair : coord_map) {
    actual_order.push_back(pair.first);
  }

  EXPECT_EQ(actual_order.size(), expected_order.size());
  for (size_t i = 0; i < expected_order.size(); ++i) {
    EXPECT_TRUE(actual_order[i] == expected_order[i]);
  }
}

TEST(PCMUtilsTester, checkReorderPCMColumns) {
  std::vector<uint8_t> data = {
      0, 1, 0, 0, 1, 0, 0, 0, 1, /* row 0 */
      1, 0, 0, 1, 1, 0, 0, 0, 0, /* row 1 */
      0, 0, 1, 0, 1, 0, 1, 0, 0, /* row 2 */
      0, 0, 0, 1, 1, 0, 0, 1, 0, /* row 3 */
      0, 0, 0, 0, 1, 1, 1, 1, 1, /* row 4 */
  };
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{5, 9});
  pcm.borrow(data.data());
  auto column_order = cudaq::qec::get_sorted_pcm_column_indices(pcm);
  const std::vector<std::uint32_t> expected_order = {1, 8, 4, 0, 3, 2, 6, 7, 5};
  EXPECT_EQ(column_order, expected_order);
  auto pcm_reordered = cudaq::qec::reorder_pcm_columns(pcm, column_order);

  const std::vector<std::vector<uint8_t>> expected_data = {
      {1, 1, 1, 0, 0, 0, 0, 0, 0}, /* row 0 */
      {0, 0, 1, 1, 1, 0, 0, 0, 0}, /* row 1 */
      {0, 0, 1, 0, 0, 1, 1, 0, 0}, /* row 2 */
      {0, 0, 1, 0, 1, 0, 0, 1, 0}, /* row 3 */
      {0, 1, 1, 0, 0, 0, 1, 1, 1}  /* row 4 */
  };

  // Compare expected data with reordered data
  for (std::size_t i = 0; i < pcm.shape()[0]; ++i)
    for (std::size_t j = 0; j < pcm_reordered.shape()[1]; ++j)
      EXPECT_EQ(expected_data[i][j], pcm_reordered.at({i, j}));

  // Now try the whole flow with sort_pcm_columns
  auto pcm_reordered2 = cudaq::qec::sort_pcm_columns(pcm);
  for (std::size_t i = 0; i < pcm_reordered2.shape()[0]; ++i)
    for (std::size_t j = 0; j < pcm_reordered2.shape()[1]; ++j)
      EXPECT_EQ(expected_data[i][j], pcm_reordered2.at({i, j}));
}

TEST(PCMUtilsTester, checkSimplifyPCM1) {
  // No simplification occurs here, but reordering occurs.
  std::vector<uint8_t> data = {
      0, 1, /* row 0 */
      1, 0  /* row 1 */
  };
  std::vector<double> weights = {0.5, 0.5};
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{2, 2});
  pcm.borrow(data.data());
  auto column_order = cudaq::qec::get_sorted_pcm_column_indices(pcm);
  const std::vector<std::uint32_t> expected_order = {1, 0};
  EXPECT_EQ(column_order, expected_order);
  auto pcm_reordered = cudaq::qec::reorder_pcm_columns(pcm, column_order);
  auto [H_new, weights_new] = cudaq::qec::simplify_pcm(pcm_reordered, weights);
  std::vector<double> expected_weights = {0.5, 0.5};
  EXPECT_EQ(weights_new, expected_weights);
}

TEST(PCMUtilsTester, checkSimplifyPCM2) {
  // Simplification (combining columns) occurs here.
  std::vector<uint8_t> data = {
      0, 1, 0, /* row 0 */
      1, 0, 1  /* row 1 */
  };
  std::vector<double> weights = {0.1, 0.2, 0.3};
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{2, 3});
  pcm.borrow(data.data());

  auto column_order = cudaq::qec::get_sorted_pcm_column_indices(pcm);
  const std::vector<std::uint32_t> expected_order = {1, 0, 2};
  EXPECT_EQ(column_order, expected_order);

  auto [H_new, weights_new] = cudaq::qec::simplify_pcm(pcm, weights);
  std::vector<double> expected_weights = {0.2, 0.1 + 0.3 - 2 * 0.1 * 0.3};
  for (std::size_t i = 0; i < weights_new.size(); ++i)
    EXPECT_NEAR(weights_new[i], expected_weights[i], 1e-6);
}

TEST(PCMUtilsTester, checkSimplifyPCMEdgeCases) {
  // Test case 1: Zero weights (weights[column_index] == 0)
  std::vector<uint8_t> data = {
      1, 0, 1, /* row 0 */
      0, 0, 0, /* row 1 */
      1, 0, 1, /* row 2 */
      0, 0, 0, /* row 3 */
      1, 0, 1  /* row 4 */
  };
  std::vector<double> weights = {0.1, 0.0, 0.2}; // Middle weight is zero
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{5, 3});
  pcm.borrow(data.data());

  auto [H_new, weights_new] = cudaq::qec::simplify_pcm(pcm, weights, 5);

  // Should skip zero-weight column and combine duplicates
  EXPECT_EQ(H_new.shape()[1], 1); // Only one column should remain
  EXPECT_EQ(weights_new.size(), 1);
  // Combined weight: 0.1 + 0.2 - 2*0.1*0.2 = 0.26
  double expected_weight = 0.1 + 0.2 - 2 * 0.1 * 0.2;
  EXPECT_NEAR(weights_new[0], expected_weight, 1e-6);

  // Test case 2: Empty columns (curr_row_indices.size() == 0)
  std::vector<uint8_t> data_with_empty = {
      1, 0, 1, 0, /* row 0 */
      0, 0, 0, 0, /* row 1 */
      1, 0, 1, 0, /* row 2 */
      0, 0, 0, 0, /* row 3 */
      1, 0, 1, 0  /* row 4 */
  };
  std::vector<double> weights_with_empty = {0.1, 0.2, 0.3, 0.4};
  cudaqx::tensor<uint8_t> pcm_with_empty(std::vector<std::size_t>{5, 4});
  pcm_with_empty.borrow(data_with_empty.data());

  auto [H_new2, weights_new2] =
      cudaq::qec::simplify_pcm(pcm_with_empty, weights_with_empty, 5);

  // Columns 1 and 3 are empty, columns 0 and 2 are identical
  EXPECT_EQ(H_new2.shape()[1], 1); // Only one column should remain
  EXPECT_EQ(weights_new2.size(), 1);
  // Combined weight of columns 0 and 2: 0.1 + 0.3 - 2*0.1*0.3 = 0.34
  double expected_weight2 = 0.1 + 0.3 - 2 * 0.1 * 0.3;
  EXPECT_NEAR(weights_new2[0], expected_weight2, 1e-6);

  // Test case 3: All columns empty or zero weight
  std::vector<uint8_t> empty_data(15, 0); // 5x3 all zeros
  std::vector<double> zero_weights = {0.0, 0.0, 0.0};
  cudaqx::tensor<uint8_t> empty_pcm(std::vector<std::size_t>{5, 3});
  empty_pcm.borrow(empty_data.data());

  auto [H_new3, weights_new3] =
      cudaq::qec::simplify_pcm(empty_pcm, zero_weights, 5);

  // Should result in empty PCM
  EXPECT_EQ(H_new3.shape()[1], 0);
  EXPECT_EQ(weights_new3.size(), 0);
}

TEST(PCMUtilsTester, checkGetPCMForRoundsEdgeCases) {
  // Create a PCM with empty columns in specific rounds
  // 20 rows (4 rounds * 5 syndromes), 12 columns (4 rounds * 3 errors)
  std::vector<uint8_t> data(20 * 12, 0); // Initialize all zeros
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{20, 12});
  pcm.borrow(data.data());

  // Round 0 (rows 0-4): normal columns
  pcm.at({0, 0}) = 1; // column 0 has errors in round 0
  pcm.at({2, 0}) = 1;
  pcm.at({1, 1}) = 1; // column 1 has errors in round 0
  pcm.at({3, 1}) = 1;
  // column 2 is empty (all zeros)

  // Round 1 (rows 5-9): mix of normal and empty columns
  pcm.at({5, 3}) = 1; // column 3 has errors in round 1
  pcm.at({7, 3}) = 1;
  // columns 4, 5 are empty

  // Round 2 (rows 10-14): all empty columns
  // columns 6, 7, 8 are all empty

  // Round 3 (rows 15-19): normal columns
  pcm.at({15, 9}) = 1; // column 9 has errors in round 3
  pcm.at({17, 9}) = 1;
  pcm.at({16, 10}) = 1; // column 10 has errors in round 3
  pcm.at({18, 10}) = 1;
  // column 11 is empty

  auto pcm_sorted = cudaq::qec::sort_pcm_columns(pcm, 5);

  // Test case 1: Get round with some empty columns (round 1)
  auto [pcm_round1, first_col1, last_col1] =
      cudaq::qec::get_pcm_for_rounds(pcm_sorted, 5, 1, 1);

  // Should have 5 rows (1 round)
  EXPECT_EQ(pcm_round1.shape()[0], 5);
  // Number of columns depends on which non-empty columns are included
  EXPECT_GE(pcm_round1.shape()[1], 0);

  // Test case 2: Get round with all empty columns (round 2)
  auto [pcm_round2, first_col2, last_col2] =
      cudaq::qec::get_pcm_for_rounds(pcm_sorted, 5, 2, 2);

  // Should have 5 rows but might have 0 columns if all are empty
  EXPECT_EQ(pcm_round2.shape()[0], 5);
  // This tests the rows_for_this_column.size() == 0 condition
  EXPECT_GE(pcm_round2.shape()[1], 0);

  // Test case 3: Get multiple rounds including empty ones
  auto [pcm_multi, first_col_multi, last_col_multi] =
      cudaq::qec::get_pcm_for_rounds(pcm_sorted, 5, 1, 2);

  // Should have 10 rows (2 rounds)
  EXPECT_EQ(pcm_multi.shape()[0], 10);
  EXPECT_GE(pcm_multi.shape()[1], 0);

  printf("Round 1 PCM shape: (%zu, %zu)\n", pcm_round1.shape()[0],
         pcm_round1.shape()[1]);
  printf("Round 2 PCM shape: (%zu, %zu)\n", pcm_round2.shape()[0],
         pcm_round2.shape()[1]);
  printf("Multi-round PCM shape: (%zu, %zu)\n", pcm_multi.shape()[0],
         pcm_multi.shape()[1]);
}

bool are_pcms_equal(const cudaqx::tensor<uint8_t> &a,
                    const cudaqx::tensor<uint8_t> &b) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::runtime_error("PCM must be a 2D tensor");
  }
  if (a.shape() != b.shape())
    return false;
  for (std::size_t r = 0; r < a.shape()[0]; ++r)
    for (std::size_t c = 0; c < a.shape()[1]; ++c)
      if (a.at({r, c}) != b.at({r, c}))
        return false;
  return true;
}

TEST(PCMUtilsTester, GetPcmForRoundsSparseMatchesDenseTensor) {
  std::mt19937_64 rng(42);
  auto pcm_raw = cudaq::qec::generate_random_pcm(
      /*n_rounds=*/4, /*n_errs=*/5, /*n_synd=*/3, /*weight=*/2, std::move(rng));
  auto pcm =
      cudaq::qec::sort_pcm_columns(pcm_raw, /*num_syndromes_per_round=*/3);
  cudaq::qec::sparse_binary_matrix pcm_sparse(pcm);

  constexpr std::uint32_t kSp = 3;
  for (bool straddle_start : {false, true}) {
    for (bool straddle_end : {false, true}) {
      auto [d_dense, fc_d, lc_d] = cudaq::qec::get_pcm_for_rounds(
          pcm, kSp,
          /*start_round=*/1, /*end_round=*/2, straddle_start, straddle_end);
      auto [d_sparse, fc_s, lc_s] = cudaq::qec::get_pcm_for_rounds(
          pcm_sparse, kSp, 1, 2, straddle_start, straddle_end);
      EXPECT_TRUE(are_pcms_equal(d_dense, d_sparse));
      EXPECT_EQ(fc_d, fc_s);
      EXPECT_EQ(lc_d, lc_s);
    }
  }
}

void check_pcm_equality(const cudaqx::tensor<uint8_t> &a,
                        const cudaqx::tensor<uint8_t> &b,
                        bool use_assert = true) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::runtime_error("PCM must be a 2D tensor");
  }
  ASSERT_EQ(a.shape(), b.shape());
  auto num_rows = a.shape()[0];
  auto num_cols = a.shape()[1];
  for (std::size_t r = 0; r < num_rows; ++r) {
    for (std::size_t c = 0; c < num_cols; ++c) {
      if (a.at({r, c}) != b.at({r, c})) {
        if (use_assert)
          ASSERT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
        else
          EXPECT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
      }
    }
  }
}

TEST(PCMUtilsTester, checkSparsePCM) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));
  printf("--------------------------------\n");
  printf("Original PCM:\n");
  pcm.dump_bits();
  auto pcm2 = cudaq::qec::sort_pcm_columns(pcm);
  printf("--------------------------------\n");
  printf("Sorted PCM (without specifying num syndromes per round):\n");
  pcm2.dump_bits();
  pcm2 = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);
  printf("--------------------------------\n");
  printf("Sorted PCM (with num syndromes per round specified):\n");
  pcm2.dump_bits();
  printf("--------------------------------\n");

  // Make sure that the first round where an error occurs is non-decreasing
  // and the last round where an error occurs is non-increasing.
  std::size_t prev_col_first_round = 0;
  std::size_t prev_col_last_round = 0;
  for (std::size_t c = 0; c < n_cols; ++c) {
    // Find the first and last row where an error occurs in this column.
    auto first_row = std::numeric_limits<std::size_t>::max();
    auto last_row = std::numeric_limits<std::size_t>::min();
    for (std::size_t r = 0; r < n_rows; ++r) {
      if (pcm2.at({r, c}) == 1) {
        first_row = std::min(first_row, r);
        last_row = std::max(last_row, r);
      }
    }
    // Convert rows to rounds.
    auto first_round = first_row / n_syndromes_per_round;
    auto last_round = last_row / n_syndromes_per_round;

    // Make sure we are not going backwards.
    ASSERT_GE(first_round, prev_col_first_round);
    ASSERT_GE(last_round, prev_col_last_round);

    // Save for the next iteration.
    prev_col_first_round = first_round;
    prev_col_last_round = last_round;
  }

  // Make sure that the sort is stable, regardless of the input order of the
  // columns.
  int num_tests = 10;
  std::mt19937_64 rng(/*seed=*/13);
  for (int iter = 0; iter < num_tests; ++iter) {
    std::vector<std::uint32_t> shuffle_vector(n_cols);
    std::iota(shuffle_vector.begin(), shuffle_vector.end(), 0);
    std::shuffle(shuffle_vector.begin(), shuffle_vector.end(), rng);
    auto pcm_shuffled = cudaq::qec::reorder_pcm_columns(pcm, shuffle_vector);

    for (std::size_t c = 0; c < n_cols; ++c)
      for (std::size_t r = 0; r < n_rows; ++r)
        pcm_shuffled.at({r, c}) = pcm.at({r, shuffle_vector[c]});
    auto pcm_sorted =
        cudaq::qec::sort_pcm_columns(pcm_shuffled, n_syndromes_per_round);
    check_pcm_equality(pcm2, pcm_sorted);
  }
}

TEST(PCMUtilsTester, checkGetPCMForRounds) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;

  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));

  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);
  auto [pcm_for_rounds, first_column, last_column] =
      cudaq::qec::get_pcm_for_rounds(pcm, n_syndromes_per_round, 0,
                                     n_rounds - 1);
  check_pcm_equality(pcm_for_rounds, pcm);

  // Try all possible combinations of start and end rounds.
  for (int start_round = 0; start_round < n_rounds; ++start_round) {
    for (int end_round = start_round; end_round < n_rounds; ++end_round) {
      auto [pcm_test, first_column_test, last_column_test] =
          cudaq::qec::get_pcm_for_rounds(pcm, n_syndromes_per_round,
                                         start_round, end_round);
      // I don't have a good test criteria for this yet. It mainly just runs to
      // see if it runs without errors.
      printf("pcm_test for start_round = %u, end_round = %u:\n", start_round,
             end_round);
      pcm_test.dump_bits();
    }
  }
}

TEST(PCMUtilsTester, checkGetSortedColumnIndicesWithBoundary) {
  // Boundary layout: interior width S = 8, boundary width B = 6. The
  // boundary-aware overload must map detector rows to rounds as
  //   round(r) = (r < B) ? 0 : 1 + (r - B) / S,
  // whereas the plain overload uses r / S. The two disagree on the transition
  // rows [B, S), which is what this test pins down.
  const std::uint32_t S = 8, B = 6;

  // Three columns whose correct (boundary) round order differs from the order
  // the plain r/S mapping produces:
  //   c0 = {5, 12} : boundary span (0, 1),  r/S span (0, 1)
  //   c1 = {6}     : boundary span (1, 1),  r/S span (0, 0)
  //   c2 = {0}     : boundary span (0, 0),  r/S span (0, 0)
  std::vector<std::vector<std::uint32_t>> cols = {{5, 12}, {6}, {0}};

  // Boundary order sorts by (first_round, last_round): c2(0,0) < c0(0,1) <
  // c1(1,1), i.e. indices {2, 0, 1}.
  auto order = cudaq::qec::get_sorted_pcm_column_indices(cols, S, B);
  EXPECT_EQ(order, (std::vector<std::uint32_t>{2, 0, 1}));

  // The plain r/S mapping would put c1 in round 0, yielding a different order
  // ({2, 1, 0}); confirm the boundary argument actually changes the result.
  auto uniform_order = cudaq::qec::get_sorted_pcm_column_indices(cols, S);
  EXPECT_EQ(uniform_order, (std::vector<std::uint32_t>{2, 1, 0}));
  EXPECT_NE(order, uniform_order);

  // Invalid boundary arguments must throw rather than divide by zero or
  // silently misassign rounds.
  EXPECT_THROW(cudaq::qec::get_sorted_pcm_column_indices(cols, 0, B),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::get_sorted_pcm_column_indices(cols, S, S + 1),
               std::invalid_argument);
}

TEST(PCMUtilsTester, checkPCMIsSortedWithBoundary) {
  // Same boundary layout as above (S = 8, B = 6). Rows < B are round 0; the
  // boundary and uniform mappings disagree on rows [B, S).
  const std::uint32_t S = 8, B = 6;

  // Columns in boundary-sorted order: spans (0,0) < (0,1) < (1,1).
  std::vector<std::vector<std::uint32_t>> sorted = {{0}, {5, 12}, {6}};
  EXPECT_TRUE(cudaq::qec::pcm_is_sorted(sorted, S, B));

  // Swapping the last two breaks the boundary order: {6} is round 1 and must
  // follow {5, 12} (which ends in round 1), so this is not boundary-sorted.
  std::vector<std::vector<std::uint32_t>> unsorted = {{0}, {6}, {5, 12}};
  EXPECT_FALSE(cudaq::qec::pcm_is_sorted(unsorted, S, B));

  // The uniform overload disagrees: both {6} and {5, 12}
  // start in round 0, so `unsorted` *is* uniform-sorted.
  EXPECT_TRUE(cudaq::qec::pcm_is_sorted(unsorted, S));
}

TEST(PCMUtilsTester, checkGetPCMForRoundsWithBoundary) {
  // Boundary layout [B | S | S | B] with B = 2, S = 4 -> 12 rows, 4 rounds:
  //   round 0: rows [0,2), round 1: [2,6), round 2: [6,10), round 3: [10,12).
  const std::uint32_t S = 4, B = 2;
  const std::size_t n_rows = 12;

  // Identity PCM: column j triggers exactly detector row j.
  cudaqx::tensor<uint8_t> pcm({n_rows, n_rows});
  for (std::size_t j = 0; j < n_rows; ++j)
    pcm.at({j, j}) = 1;

  // The leading boundary round is B rows wide, not S: the boundary overload
  // keeps rows [0,B) while the uniform interpretation keeps rows [0,S).
  auto [sub_b, fc_b, lc_b] = cudaq::qec::get_pcm_for_rounds(
      pcm, S, /*start_round=*/0, /*end_round=*/0, /*straddle_start=*/false,
      /*straddle_end=*/false, /*num_boundary_syndromes=*/B);
  EXPECT_EQ(sub_b.shape()[0], B);
  auto [sub_u, fc_u, lc_u] = cudaq::qec::get_pcm_for_rounds(
      pcm, S, /*start_round=*/0, /*end_round=*/0);
  EXPECT_EQ(sub_u.shape()[0], S);

  // Spanning every round returns all rows (confirms the round count is right).
  auto [sub_all, fc_all, lc_all] = cudaq::qec::get_pcm_for_rounds(
      pcm, S, /*start_round=*/0, /*end_round=*/3, false, false, B);
  EXPECT_EQ(sub_all.shape()[0], n_rows);

  // Invalid boundary arguments / inconsistent row counts must throw.
  EXPECT_THROW(
      cudaq::qec::get_pcm_for_rounds(pcm, S, 0, 0, false, false, S + 1),
      std::invalid_argument);
  // 13 rows cannot be split as 2*B + K*S for B = 2, S = 4.
  cudaqx::tensor<uint8_t> bad_pcm({13, 13});
  EXPECT_THROW(
      cudaq::qec::get_pcm_for_rounds(bad_pcm, S, 0, 0, false, false, B),
      std::invalid_argument);
}

TEST(PCMUtilsTester, checkShufflePCMColumns) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t weight = 3;
  std::mt19937_64 rng1(13);
  std::mt19937_64 rng2(13);
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::move(rng1));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);
  auto pcm_permuted = cudaq::qec::shuffle_pcm_columns(pcm, std::move(rng2));
  // Verify that the new PCM is different from the original.
  EXPECT_FALSE(are_pcms_equal(pcm, pcm_permuted));
  // Verify that the resorted permutedPCM is the same as the original.
  auto pcm_permuted_and_sorted =
      cudaq::qec::sort_pcm_columns(pcm_permuted, n_syndromes_per_round);
  check_pcm_equality(pcm_permuted_and_sorted, pcm);
}

TEST(QECCodeTester, checkVersion) {
  std::string version = cudaq::qec::getVersion();
  EXPECT_FALSE(version.empty());
  EXPECT_TRUE(version.find("CUDAQX_QEC_VERSION") == std::string::npos);

  std::string fullVersion = cudaq::qec::getFullRepositoryVersion();
  EXPECT_TRUE(fullVersion.find("NVIDIA/cudaqx") != std::string::npos);
  EXPECT_TRUE(fullVersion.find("CUDAQX_QEC_COMMIT_SHA") == std::string::npos);
}

TEST(PCMUtilsTester, checkPCMToSparseString) {
  // Generate a random PCM.
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t weight = 3;
  std::mt19937_64 rng(13);
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::move(rng));
  // Get the string version of the PCM
  auto sparse_str = cudaq::qec::pcm_to_sparse_string(pcm);
  // Get the PCM from the string
  auto pcm_from_str = cudaq::qec::pcm_from_sparse_string(
      sparse_str, n_rounds * n_syndromes_per_round,
      n_rounds * n_errs_per_round);
  // Verify that the original and the from-string PCM are the same
  check_pcm_equality(pcm, pcm_from_str, true);
}

TEST(PCMUtilsTester, checkPCMToSparseVec) {
  // Generate a random PCM.
  std::size_t n_rounds = 7;
  std::size_t n_errs_per_round = 10;
  std::size_t n_syndromes_per_round = 100;
  std::size_t weight = 3;
  std::mt19937_64 rng(13);
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::move(rng));
  // Get the string version of the PCM
  auto sparse_vec = cudaq::qec::pcm_to_sparse_vec(pcm);
  // Get the PCM from the string
  auto pcm_from_vec = cudaq::qec::pcm_from_sparse_vec(
      sparse_vec, n_rounds * n_syndromes_per_round,
      n_rounds * n_errs_per_round);
  // Verify that the original and the from-string PCM are the same
  check_pcm_equality(pcm, pcm_from_vec, true);
}

TEST(PCMUtilsTester, DetectorMatrixAndExtendUtilityPaths) {
  auto detector_without_first =
      cudaq::qec::generate_timelike_sparse_detector_matrix(
          /*num_syndromes_per_round=*/2, /*num_rounds=*/3,
          /*include_first_round=*/false);
  EXPECT_EQ(detector_without_first,
            (std::vector<int64_t>{0, 2, -1, 1, 3, -1, 2, 4, -1, 3, 5, -1}));

  auto detector_with_custom_first =
      cudaq::qec::generate_timelike_sparse_detector_matrix(
          /*num_syndromes_per_round=*/2, /*num_rounds=*/3,
          std::vector<int64_t>{0, -1, 1, -1});
  EXPECT_EQ(detector_with_custom_first.front(), 0);
  EXPECT_EQ(detector_with_custom_first.back(), -1);

  // Two identical sorted rounds give pcm_extend_to_n_rounds a repeat pattern to
  // detect and extend.
  cudaqx::tensor<uint8_t> pcm({4, 4});
  std::vector<uint8_t> data = {
      1, 0, 0, 0, //
      0, 1, 0, 0, //
      0, 0, 1, 0, //
      0, 0, 0, 1, //
  };
  pcm.copy(data.data(), {4, 4});
  ASSERT_TRUE(cudaq::qec::pcm_is_sorted(pcm, /*num_syndromes_per_round=*/2));

  auto [extended, columns] =
      cudaq::qec::pcm_extend_to_n_rounds(pcm, 2, /*n_rounds=*/4);
  EXPECT_EQ(extended.shape()[0], 8u);
  EXPECT_GE(extended.shape()[1], pcm.shape()[1]);
  EXPECT_FALSE(columns.empty());
}

TEST(PluginLoaderTester, LoadPluginsCoversSuccessAndFailure) {
  namespace fs = std::filesystem;
  const fs::path tmp_dir =
      fs::temp_directory_path() /
      ("cudaq_qec_plugin_loader_" + std::to_string(getpid()));
  fs::create_directories(tmp_dir);

  const fs::path bad_plugin = tmp_dir / "bad_plugin.so";
  {
    std::ofstream out(bad_plugin);
    out << "not an ELF shared object";
  }

  const fs::path libm = "/lib/x86_64-linux-gnu/libm.so.6";
  if (!fs::exists(libm)) {
    fs::remove_all(tmp_dir);
    GTEST_SKIP() << "libm.so.6 not found for plugin loader dlopen smoke test";
  }
  const fs::path good_plugin = tmp_dir / "good_plugin.so";
  fs::create_symlink(libm, good_plugin);

  cudaq::qec::load_plugins(tmp_dir.string(), cudaq::qec::PluginType::DECODER);
  cudaq::qec::cleanup_plugins(cudaq::qec::PluginType::DECODER);
  fs::remove_all(tmp_dir);
}

TEST(LibraryUtilsTester, FindsQecLibraryPath) {
  auto path = cudaqx::__internal__::getCUDAQXLibraryPath(
      cudaqx::__internal__::CUDAQXLibraryType::QEC);
  EXPECT_NE(path.find("libcudaq-qec"), std::string::npos);
}

// Test detector_error_model methods
TEST(DetectorErrorModelTest, NumDetectors) {
  cudaq::qec::detector_error_model dem;

  // Test case 1: Empty matrix (no shape)
  EXPECT_EQ(dem.num_detectors(), 0);

  // Test case 2: 1D matrix (shape.size() != 2)
  std::vector<std::size_t> shape_1d = {5};
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(shape_1d);
  EXPECT_EQ(dem.num_detectors(), 0);

  // Test case 3: Valid 2D matrix
  std::vector<std::size_t> shape_2d = {3, 4}; // 3 detectors, 4 error mechanisms
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(shape_2d);
  EXPECT_EQ(dem.num_detectors(), 3);

  // Test case 4: Different dimensions
  std::vector<std::size_t> shape_large = {10, 8};
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(shape_large);
  EXPECT_EQ(dem.num_detectors(), 10);
}

TEST(DetectorErrorModelTest, NumErrorMechanisms) {
  cudaq::qec::detector_error_model dem;

  // Test case 1: Empty matrix (no shape)
  EXPECT_EQ(dem.num_error_mechanisms(), 0);

  // Test case 2: 1D matrix (shape.size() != 2)
  std::vector<std::size_t> shape_1d = {5};
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(shape_1d);
  EXPECT_EQ(dem.num_error_mechanisms(), 0);

  // Test case 3: Valid 2D matrix
  std::vector<std::size_t> shape_2d = {3, 4}; // 3 detectors, 4 error mechanisms
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(shape_2d);
  EXPECT_EQ(dem.num_error_mechanisms(), 4);

  // Test case 4: Different dimensions
  std::vector<std::size_t> shape_large = {10, 8};
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(shape_large);
  EXPECT_EQ(dem.num_error_mechanisms(), 8);
}

TEST(DetectorErrorModelTest, NumObservables) {
  cudaq::qec::detector_error_model dem;

  // Test case 1: Empty matrix (no shape) - covers line 32 return 0
  EXPECT_EQ(dem.num_observables(), 0);

  // Test case 2: 1D matrix (shape.size() != 2) - covers line 32 return 0
  std::vector<std::size_t> shape_1d = {5};
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(shape_1d);
  EXPECT_EQ(dem.num_observables(), 0);

  // Test case 3: 3D matrix (shape.size() != 2) - covers line 32 return 0
  std::vector<std::size_t> shape_3d = {2, 3, 4};
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(shape_3d);
  EXPECT_EQ(dem.num_observables(), 0);

  // Test case 4: Valid 2D matrix - covers line 30 return shape[0]
  std::vector<std::size_t> shape_2d = {2,
                                       5}; // 2 observables, 5 error mechanisms
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(shape_2d);
  EXPECT_EQ(dem.num_observables(), 2);

  // Test case 5: Different dimensions
  std::vector<std::size_t> shape_large = {7, 12};
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(shape_large);
  EXPECT_EQ(dem.num_observables(), 7);

  // Test case 6: Single observable
  std::vector<std::size_t> shape_single = {1, 3};
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(shape_single);
  EXPECT_EQ(dem.num_observables(), 1);
}

TEST(DetectorErrorModelTest, NumObservablesInCanonicalize) {
  // This test verifies the calling stack: canonicalize_for_rounds ->
  // num_observables()
  cudaq::qec::detector_error_model dem;

  // Set up a simple detector_error_matrix
  std::vector<std::size_t> detector_shape = {
      2, 3}; // 2 detectors, 3 error mechanisms
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(detector_shape);
  // Initialize with some data
  dem.detector_error_matrix.at({0, 0}) = 1;
  dem.detector_error_matrix.at({0, 1}) = 0;
  dem.detector_error_matrix.at({0, 2}) = 1;
  dem.detector_error_matrix.at({1, 0}) = 0;
  dem.detector_error_matrix.at({1, 1}) = 1;
  dem.detector_error_matrix.at({1, 2}) = 0;

  // Set up observables_flips_matrix
  std::vector<std::size_t> obs_shape = {1,
                                        3}; // 1 observable, 3 error mechanisms
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(obs_shape);
  dem.observables_flips_matrix.at({0, 0}) = 0;
  dem.observables_flips_matrix.at({0, 1}) = 1;
  dem.observables_flips_matrix.at({0, 2}) = 0;

  // Set up error_rates
  dem.error_rates = {0.1, 0.2, 0.15};

  // Before canonicalize: verify num_observables works
  EXPECT_EQ(dem.num_observables(), 1);

  // This will internally call num_observables() at line 73 in
  // canonicalize_for_rounds
  dem.canonicalize_for_rounds(2);

  // After canonicalize: verify num_observables still works
  EXPECT_EQ(dem.num_observables(), 1);
}

TEST(DetectorErrorModelTest, FailureOnEmptyErrorRatesCanonicalize) {
  cudaq::qec::detector_error_model dem;

  // Set up a simple detector_error_matrix
  std::vector<std::size_t> detector_shape = {
      2, 3}; // 2 detectors, 3 error mechanisms
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(detector_shape);
  // Initialize with some data
  dem.detector_error_matrix.at({0, 0}) = 1;
  dem.detector_error_matrix.at({0, 1}) = 0;
  dem.detector_error_matrix.at({0, 2}) = 1;
  dem.detector_error_matrix.at({1, 0}) = 0;
  dem.detector_error_matrix.at({1, 1}) = 1;
  dem.detector_error_matrix.at({1, 2}) = 0;

  // Set up observables_flips_matrix
  std::vector<std::size_t> obs_shape = {1,
                                        3}; // 1 observable, 3 error mechanisms
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(obs_shape);
  dem.observables_flips_matrix.at({0, 0}) = 0;
  dem.observables_flips_matrix.at({0, 1}) = 1;
  dem.observables_flips_matrix.at({0, 2}) = 0;

  // Set up error_rates
  dem.error_rates = {};

  // Before canonicalize: verify num_observables works
  EXPECT_EQ(dem.num_observables(), 1);

  EXPECT_THROW(dem.canonicalize_for_rounds(2), std::runtime_error);
}

TEST(DetectorErrorModelTest, CanonicalizeBoundaryRejectsWideBoundary) {
  // The boundary-aware overload must reject a boundary wider than the interior
  // (num_boundary_syndromes > num_syndromes_per_round)
  cudaq::qec::detector_error_model dem;
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>({4, 2});
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>({1, 2});
  dem.error_rates = {0.1, 0.2};

  EXPECT_THROW(dem.canonicalize_for_rounds_with_boundary(
                   /*num_syndromes_per_round=*/2,
                   /*num_boundary_syndromes=*/3,
                   /*remove_zero_syndrome_errors=*/true),
               std::invalid_argument);
}

TEST(DetectorErrorModelTest, CanonicalizeWithoutErrorIds) {
  // This test covers the std::numeric_limits<std::size_t>::max() branch
  // when has_error_ids is false and duplicate columns need to be merged
  cudaq::qec::detector_error_model dem;

  // Set up detector_error_matrix with duplicate columns (columns 0 and 2 are
  // identical)
  std::vector<std::size_t> detector_shape = {
      3, 4}; // 3 detectors, 4 error mechanisms
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(detector_shape);

  // Column 0: detectors 0,2 triggered
  dem.detector_error_matrix.at({0, 0}) = 1;
  dem.detector_error_matrix.at({1, 0}) = 0;
  dem.detector_error_matrix.at({2, 0}) = 1;

  // Column 1: detector 1 triggered
  dem.detector_error_matrix.at({0, 1}) = 0;
  dem.detector_error_matrix.at({1, 1}) = 1;
  dem.detector_error_matrix.at({2, 1}) = 0;

  // Column 2: detectors 0,2 triggered (same as column 0)
  dem.detector_error_matrix.at({0, 2}) = 1;
  dem.detector_error_matrix.at({1, 2}) = 0;
  dem.detector_error_matrix.at({2, 2}) = 1;

  // Column 3: detector 1 triggered
  dem.detector_error_matrix.at({0, 3}) = 0;
  dem.detector_error_matrix.at({1, 3}) = 1;
  dem.detector_error_matrix.at({2, 3}) = 0;

  // Set up observables_flips_matrix
  std::vector<std::size_t> obs_shape = {1,
                                        4}; // 1 observable, 4 error mechanisms
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(obs_shape);
  dem.observables_flips_matrix.at({0, 0}) = 0;
  dem.observables_flips_matrix.at({0, 1}) = 1;
  dem.observables_flips_matrix.at({0, 2}) = 0; // Same as column 0
  dem.observables_flips_matrix.at({0, 3}) = 1;

  // Set up error_rates
  dem.error_rates = {0.1, 0.2, 0.15, 0.25};

  // Important: DO NOT set error_ids, so has_error_ids will be false
  // This ensures the std::numeric_limits<std::size_t>::max() branch is taken

  EXPECT_EQ(dem.num_detectors(), 3);
  EXPECT_EQ(dem.num_error_mechanisms(), 4);
  EXPECT_EQ(dem.num_observables(), 1);

  // This will trigger the canonicalize logic where:
  // 1. has_error_ids = false (because error_ids.has_value() is false)
  // 2. Columns 0 and 2 have identical detector patterns, triggering merge
  // 3. prev_error_id = std::numeric_limits<std::size_t>::max() (line 98-100)
  dem.canonicalize_for_rounds(3);

  // After canonicalization, duplicate columns should be merged
  // The matrix should have fewer columns
  EXPECT_LT(dem.num_error_mechanisms(), 4);
  EXPECT_EQ(dem.num_detectors(), 3);   // detectors count shouldn't change
  EXPECT_EQ(dem.num_observables(), 1); // observables count shouldn't change
}

TEST(DetectorErrorModelTest, CanonicalizeWithMismatchedErrorIds) {
  // This test covers the case where error_ids exists but size doesn't match
  // error_rates so has_error_ids becomes false, triggering the
  // std::numeric_limits branch
  cudaq::qec::detector_error_model dem;

  // Set up detector_error_matrix with duplicate columns
  std::vector<std::size_t> detector_shape = {2, 3};
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(detector_shape);

  // Column 0 and 2 are identical
  dem.detector_error_matrix.at({0, 0}) = 1;
  dem.detector_error_matrix.at({1, 0}) = 0;
  dem.detector_error_matrix.at({0, 1}) = 0;
  dem.detector_error_matrix.at({1, 1}) = 1;
  dem.detector_error_matrix.at({0, 2}) = 1; // Same as column 0
  dem.detector_error_matrix.at({1, 2}) = 0; // Same as column 0

  // Set up observables_flips_matrix
  std::vector<std::size_t> obs_shape = {1, 3};
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(obs_shape);
  dem.observables_flips_matrix.at({0, 0}) = 0;
  dem.observables_flips_matrix.at({0, 1}) = 1;
  dem.observables_flips_matrix.at({0, 2}) = 0; // Same as column 0

  // Set up error_rates (3 elements)
  dem.error_rates = {0.1, 0.2, 0.15};

  // Set up error_ids with MISMATCHED size (2 elements instead of 3)
  // This will make has_error_ids = false because error_ids->size() !=
  // error_rates.size()
  dem.error_ids = std::vector<std::size_t>{100, 200}; // Only 2 elements, not 3

  EXPECT_EQ(dem.error_rates.size(), 3);
  EXPECT_EQ(dem.error_ids->size(), 2); // Mismatched size

  // This will trigger has_error_ids = false due to size mismatch
  // and then use std::numeric_limits<std::size_t>::max() when merging columns 0
  // and 2
  dem.canonicalize_for_rounds(2);

  // Verify the function completed successfully
  EXPECT_LT(dem.num_error_mechanisms(), 3); // Should have merged some columns
}

// Sum the probability mass of columns that flip observable 0.
static double
observable0_flip_mass(const cudaq::qec::detector_error_model &dem) {
  double mass = 0.0;
  for (std::size_t c = 0; c < dem.num_error_mechanisms(); c++)
    if (dem.observables_flips_matrix.at({0, c}))
      mass += dem.error_rates[c];
  return mass;
}

// Columns with the same detector signature but DIFFERENT observable flips must
// not be merged, otherwise observable-flip probability mass is relabeled.
TEST(DetectorErrorModelTest, CanonicalizeObservableAwareMerge) {
  cudaq::qec::detector_error_model dem;
  // Two columns, identical single-detector syndrome.
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>({1, 2});
  dem.detector_error_matrix.at({0, 0}) = 1;
  dem.detector_error_matrix.at({0, 1}) = 1;
  // Column 0 flips the observable, column 1 does not.
  dem.observables_flips_matrix = cudaqx::tensor<uint8_t>({1, 2});
  dem.observables_flips_matrix.at({0, 0}) = 1;
  dem.observables_flips_matrix.at({0, 1}) = 0;
  dem.error_rates = {0.2, 0.3};

  dem.canonicalize_for_rounds(1);

  // The two columns differ in observable, so they remain distinct.
  EXPECT_EQ(dem.num_error_mechanisms(), 2);
  // The observable-flip mass is preserved at 0.2 (neither merged to 0.38
  // nor dropped to 0.0).
  EXPECT_NEAR(observable0_flip_mass(dem), 0.2, 1e-12);
}

// A zero-syndrome column that flips an observable is an undetectable logical
// error. It is kept by default and dropped only when
// remove_zero_syndrome_errors is requested.
TEST(DetectorErrorModelTest, CanonicalizeZeroSyndromeObservable) {
  auto make = []() {
    cudaq::qec::detector_error_model dem;
    dem.detector_error_matrix = cudaqx::tensor<uint8_t>({1, 2});
    dem.detector_error_matrix.at({0, 0}) = 1; // normal column
    dem.detector_error_matrix.at({0, 1}) = 0; // no detector signature
    dem.observables_flips_matrix = cudaqx::tensor<uint8_t>({1, 2});
    dem.observables_flips_matrix.at({0, 0}) = 0;
    dem.observables_flips_matrix.at({0, 1}) = 1; // flips the observable
    dem.error_rates = {0.1, 0.01};
    return dem;
  };

  // Default: the zero-syndrome observable-flipping column is retained.
  auto dem_keep = make();
  dem_keep.canonicalize_for_rounds(1);
  EXPECT_EQ(dem_keep.num_error_mechanisms(), 2);
  EXPECT_NEAR(observable0_flip_mass(dem_keep), 0.01, 1e-12);

  // With removal requested: it is dropped (useless for round-based decoding).
  auto dem_drop = make();
  dem_drop.canonicalize_for_rounds(1, /*remove_zero_syndrome_errors=*/true);
  EXPECT_EQ(dem_drop.num_error_mechanisms(), 1);
  EXPECT_NEAR(observable0_flip_mass(dem_drop), 0.0, 1e-12);
}

// Rate composition must be independent of input column order. Two exclusive
// alternatives of one mechanism (same id) plus an independent mechanism
// (different id), all on the same syndrome, must compose to
// P = (0.1 + 0.1)(1 - 0.2) + (1 - 0.1 - 0.1)(0.2) = 0.32 regardless of order.
TEST(DetectorErrorModelTest, CanonicalizeOrderIndependentComposition) {
  auto run = [](const std::vector<double> &rates,
                const std::vector<std::size_t> &ids) {
    cudaq::qec::detector_error_model dem;
    const std::size_t n = rates.size();
    dem.detector_error_matrix = cudaqx::tensor<uint8_t>({1, n});
    for (std::size_t c = 0; c < n; c++)
      dem.detector_error_matrix.at({0, c}) =
          1; // shared single-detector syndrome
    dem.observables_flips_matrix = cudaqx::tensor<uint8_t>({1, n});
    dem.error_rates = rates;
    dem.error_ids = ids;
    dem.canonicalize_for_rounds(1);
    EXPECT_EQ(dem.num_error_mechanisms(), 1u);
    return dem.error_rates[0];
  };

  // A(0.1,id0), B(0.2,id1), C(0.1,id0) in two different column orders.
  double abc = run({0.1, 0.2, 0.1}, {0, 1, 0});
  double acb = run({0.1, 0.1, 0.2}, {0, 0, 1});
  EXPECT_NEAR(abc, 0.32, 1e-12);
  EXPECT_NEAR(acb, 0.32, 1e-12);
}

TEST(PluginLoaderTester, checkCleanupPluginsEdgeCases) {
  // Test edge cases for cleanup_plugins function to cover the else branch
  // The plugin loader is loaded in the constructor of load_decoder_plugins()
  // with type PluginType::DECODER, so cleanup with type PluginType::CODE will
  // not do anything.
  cudaq::qec::cleanup_plugins(cudaq::qec::PluginType::CODE);
}
