/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Unit tests for dem_from_css_matrices(). Tests cover:
//   - num_rounds=1 (default): repetition code X errors against manually
//     computed expected values and against dem_from_stim_text().
//   - CSS code with both X and Z errors (two detector bands, two obs bands).
//   - Y errors triggering both detector bands.
//   - Per-qubit noise rates including zero-rate qubit suppression.
//   - Edge cases: all-zero rates, default-constructed matrices, mismatches.
//   - Multi-round (num_rounds>1): detector-difference structure, bulk vs
//     last-round boundary, observable propagation across rounds, Stim
//     cross-check for T=2.

#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_construction.h"
#include "cudaq/qec/detector_error_model.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace cudaq::qec {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Return true iff the two dense uint8 tensors have identical shape and
// values. Assumes rank-2 tensors.
bool tensors_equal(const cudaqx::tensor<uint8_t> &a,
                   const cudaqx::tensor<uint8_t> &b) {
  if (a.rank() != 2 || b.rank() != 2)
    return false;
  if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1])
    return false;
  for (std::size_t r = 0; r < a.shape()[0]; ++r) {
    for (std::size_t c = 0; c < a.shape()[1]; ++c) {
      if (a.at({r, c}) != b.at({r, c}))
        return false;
    }
  }
  return true;
}

// Build css_code_matrices for the d=3 repetition code (Z-basis only).
//   H_Z = [[1,1,0],[0,1,1]]   (Z0Z1, Z1Z2 stabilizers)
//   H_X = empty
//   L_Z = [[1,0,0]]           (Z0 logical)
//   L_X = empty
css_code_matrices rep3_z_basis_matrices() {
  css_code_matrices m;
  m.hz = sparse_binary_matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});
  m.lz = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  return m;
}

// ---------------------------------------------------------------------------
// Repetition code d=3, X errors only
// ---------------------------------------------------------------------------

// Expected detector_error_matrix for d=3, px=0.01 (no Z/Y errors):
//
//   col   0    1    2     (X fault on qubit 0, 1, 2)
//   D0  [ 1    1    0 ]   (Z0Z1 stabilizer)
//   D1  [ 0    1    1 ]   (Z1Z2 stabilizer)
TEST(DemConstruction, RepetitionCode3_XOnly_DetectorMatrix) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  ASSERT_EQ(dem.num_detectors(), 2u);
  ASSERT_EQ(dem.num_error_mechanisms(), 3u);

  // Column 0: X on qubit 0 → D0
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 0u);

  // Column 1: X on qubit 1 → D0, D1
  EXPECT_EQ(dem.detector_error_matrix.at({0, 1}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 1}), 1u);

  // Column 2: X on qubit 2 → D1
  EXPECT_EQ(dem.detector_error_matrix.at({0, 2}), 0u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 2}), 1u);
}

// Expected observables_flips_matrix for d=3, px=0.01:
//
//   col   0    1    2
//   L0  [ 1    0    0 ]   (Z0 logical flipped by X on qubit 0 only)
TEST(DemConstruction, RepetitionCode3_XOnly_ObservableMatrix) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  ASSERT_EQ(dem.num_observables(), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 1}), 0u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 2}), 0u);
}

TEST(DemConstruction, RepetitionCode3_XOnly_ErrorRates) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  ASSERT_EQ(dem.error_rates.size(), 3u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.01);
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.01);
  EXPECT_DOUBLE_EQ(dem.error_rates[2], 0.01);
}

// Cross-check: the DEM produced by dem_from_css_matrices() must match
// the one produced by dem_from_stim_text() for the same scenario.
// Stim DEM (X errors, d=3 repetition code, p=0.01):
//   X on q0: flips D0 and L0
//   X on q1: flips D0 and D1
//   X on q2: flips D1
TEST(DemConstruction, RepetitionCode3_XOnly_MatchesStimText) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto from_matrices = dem_from_css_matrices(code, noise);

  const std::string stim_dem = "error(0.01) D0 L0\n"
                               "error(0.01) D0 D1\n"
                               "error(0.01) D1\n";
  auto from_stim = dem_from_stim_text(stim_dem);

  EXPECT_TRUE(tensors_equal(from_matrices.detector_error_matrix,
                            from_stim.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(from_matrices.observables_flips_matrix,
                            from_stim.observables_flips_matrix));
  ASSERT_EQ(from_matrices.error_rates.size(), from_stim.error_rates.size());
  for (std::size_t i = 0; i < from_matrices.error_rates.size(); ++i)
    EXPECT_DOUBLE_EQ(from_matrices.error_rates[i], from_stim.error_rates[i]);
}

// ---------------------------------------------------------------------------
// X and Z errors (repetition code with explicit Z-error columns)
// ---------------------------------------------------------------------------

// Add a Z observable (X-type logical) to the repetition code so that Z
// errors also have an observable effect. Z errors have no syndrome (hx is
// empty) but qubit-0 Z errors flip the X logical L_X[0,0]=1.
//
// With px = pz = 0.01:
//   Columns 0..2: X faults on q0,q1,q2 (same as X-only test)
//   Columns 3..5: Z faults on q0,q1,q2
//     col 3: obs row 1 (X logical) flipped (Z on q0, L_X[0,0]=1)
//     col 4: no effect (Z on q1, L_X[0,1]=0)
//     col 5: no effect (Z on q2, L_X[0,2]=0)
TEST(DemConstruction, XAndZErrors_Dimensions) {
  css_code_matrices code = rep3_z_basis_matrices();
  code.lx = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  css_noise_params noise;
  noise.px = 0.01;
  noise.pz = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  // 2 Z detectors + 0 X detectors = 2 total detectors.
  // 1 Z observable + 1 X observable = 2 total observables.
  // 3 X columns + 3 Z columns = 6 total columns.
  EXPECT_EQ(dem.num_detectors(), 2u);
  EXPECT_EQ(dem.num_observables(), 2u);
  EXPECT_EQ(dem.num_error_mechanisms(), 6u);
}

TEST(DemConstruction, XAndZErrors_ZFaultColumns) {
  css_code_matrices code = rep3_z_basis_matrices();
  code.lx = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  css_noise_params noise;
  noise.px = 0.01;
  noise.pz = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  // Z fault columns start at index 3.
  // col 3 (Z on q0): no detectors, obs row 1 (X logical) flipped.
  EXPECT_EQ(dem.detector_error_matrix.at({0, 3}), 0u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 3}), 0u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 3}), 0u); // Z-obs unaffected
  EXPECT_EQ(dem.observables_flips_matrix.at({1, 3}), 1u); // X-obs flipped

  // col 4 (Z on q1): no effect.
  EXPECT_EQ(dem.observables_flips_matrix.at({1, 4}), 0u);

  // col 5 (Z on q2): no effect.
  EXPECT_EQ(dem.observables_flips_matrix.at({1, 5}), 0u);
}

// ---------------------------------------------------------------------------
// Y errors
// ---------------------------------------------------------------------------

// Minimal CSS pair: 2 qubits, 1 Z-stabilizer (Z0Z1), 1 X-stabilizer (X0X1).
//   hz = [[1,1]]  →  Z0Z1 detects X errors
//   hx = [[1,1]]  →  X0X1 detects Z errors
//   lz = [[1,0]]  →  Z logical = Z0
//   lx = [[1,0]]  →  X logical = X0
//
// Y fault on qubit 0: X component flips D0 (Z-check) and L0 (Z-obs);
//                     Z component flips D1 (X-check) and L1 (X-obs).
TEST(DemConstruction, YErrors_BothBandsTriggered) {
  css_code_matrices code;
  code.hz = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {0}});
  code.hx = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {0}});
  code.lz = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {}});
  code.lx = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {}});
  css_noise_params noise;
  noise.py = 0.005;

  auto dem = dem_from_css_matrices(code, noise);

  // 1 Z-det + 1 X-det = 2 detectors, 2 observables, 2 Y-fault columns.
  EXPECT_EQ(dem.num_detectors(), 2u);
  EXPECT_EQ(dem.num_observables(), 2u);
  EXPECT_EQ(dem.num_error_mechanisms(), 2u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.005);
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.005);

  // Y on qubit 0 (col 0): det row 0 (Z-check) and det row 1 (X-check).
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);    // Z-det from X comp
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 1u);    // X-det from Z comp
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 1u); // Z-obs
  EXPECT_EQ(dem.observables_flips_matrix.at({1, 0}), 1u); // X-obs

  // Y on qubit 1 (col 1): both detectors triggered, no observable flip.
  EXPECT_EQ(dem.detector_error_matrix.at({0, 1}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 1}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 1}), 0u);
  EXPECT_EQ(dem.observables_flips_matrix.at({1, 1}), 0u);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(DemConstruction, AllRatesZero_ReturnsEmptyDem) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise; // px = py = pz = 0

  auto dem = dem_from_css_matrices(code, noise);

  EXPECT_EQ(dem.num_error_mechanisms(), 0u);
  EXPECT_TRUE(dem.error_rates.empty());
}

TEST(DemConstruction, AllMatricesDefault_ReturnsEmptyDem) {
  css_code_matrices code; // all default-constructed (0x0)
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  EXPECT_EQ(dem.num_error_mechanisms(), 0u);
}

// A default-constructed hx (0 columns) must not cause bounds errors even
// when pz > 0, because the Z-fault columns will have no detector entries.
TEST(DemConstruction, DefaultHxLx_ZFaultColumnsAreEmpty) {
  css_code_matrices code = rep3_z_basis_matrices();
  // hx and lx are default-constructed (0x0); Z errors are undetectable.
  css_noise_params noise;
  noise.pz = 0.01;

  auto dem = dem_from_css_matrices(code, noise);

  // 2 Z-detectors, 1 Z-observable, 3 Z-fault columns.
  EXPECT_EQ(dem.num_detectors(), 2u);
  EXPECT_EQ(dem.num_observables(), 1u);
  EXPECT_EQ(dem.num_error_mechanisms(), 3u);

  // All Z-fault columns should be zero in detector matrix (no X stabilizers).
  for (std::size_t col = 0; col < 3u; ++col) {
    EXPECT_EQ(dem.detector_error_matrix.at({0, col}), 0u) << "col=" << col;
    EXPECT_EQ(dem.detector_error_matrix.at({1, col}), 0u) << "col=" << col;
  }
}

TEST(DemConstruction, DimensionMismatch_Throws) {
  css_code_matrices code;
  // hz has 3 columns, hx has 4 columns → should throw.
  code.hz = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {0}, {0}});
  code.hx = sparse_binary_matrix::from_nested_csc(1, 4, {{0}, {0}, {0}, {0}});
  css_noise_params noise;
  noise.px = 0.01;
  noise.pz = 0.01;

  EXPECT_THROW(dem_from_css_matrices(code, noise), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Per-qubit noise rates
// ---------------------------------------------------------------------------

// d=3 repetition code, non-uniform X error rates: only qubit 0 and qubit 2
// have nonzero rates. Qubit 1 (px=0) should emit no column.
TEST(DemConstruction, PerQubitRates_SkipsZeroRateQubits) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px_per_qubit = {0.01, 0.0, 0.02};

  auto dem = dem_from_css_matrices(code, noise);

  // Only 2 columns: X on q0 and X on q2.
  ASSERT_EQ(dem.num_error_mechanisms(), 2u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.01);
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.02);

  // col 0: X on q0 → D0, L0
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 0u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 1u);

  // col 1: X on q2 → D1 only (no observable flip)
  EXPECT_EQ(dem.detector_error_matrix.at({0, 1}), 0u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 1}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 1}), 0u);
}

// Verify that per-qubit rates override the uniform scalar.
TEST(DemConstruction, PerQubitRates_OverrideUniform) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;                        // uniform; ignored when per-qubit set
  noise.px_per_qubit = {0.05, 0.10, 0.0}; // q2 gets rate 0 → no column

  auto dem = dem_from_css_matrices(code, noise);

  ASSERT_EQ(dem.num_error_mechanisms(), 2u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.05); // q0
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.10); // q1
}

// A per-qubit vector with wrong length must throw.
TEST(DemConstruction, PerQubitRates_WrongLength_Throws) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px_per_qubit = {0.01, 0.02}; // only 2 entries for 3-qubit code

  EXPECT_THROW(dem_from_css_matrices(code, noise), std::invalid_argument);
}

// All per-qubit rates zero → empty DEM even with nonzero uniform scalar.
TEST(DemConstruction, PerQubitRates_AllZero_EmptyDem) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;                      // non-zero scalar
  noise.px_per_qubit = {0.0, 0.0, 0.0}; // all qubits suppressed

  auto dem = dem_from_css_matrices(code, noise);

  EXPECT_EQ(dem.num_error_mechanisms(), 0u);
}

// ---------------------------------------------------------------------------
// Multi-round DEM construction
// ---------------------------------------------------------------------------

// d=3 repetition code, T=2 rounds, X errors only.
//
// Detector layout (d=2 checks per round):
//   D0 = round 0, Z0Z1    D1 = round 0, Z1Z2
//   D2 = round 1, Z0Z1    D3 = round 1, Z1Z2
//
// Column layout (round 0 first):
//   col 0 – round 0, q0: {D0,D2}, L0
//   col 1 – round 0, q1: {D0,D1,D2,D3}
//   col 2 – round 0, q2: {D1,D3}
//   col 3 – round 1, q0: {D2},    L0   (last round → single band)
//   col 4 – round 1, q1: {D2,D3}
//   col 5 – round 1, q2: {D3}
TEST(DemConstruction, MultiRound_T2_Dimensions) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise, 2);

  EXPECT_EQ(dem.num_detectors(), 4u); // 2 rounds × 2 checks
  EXPECT_EQ(dem.num_observables(), 1u);
  EXPECT_EQ(dem.num_error_mechanisms(), 6u); // 2 rounds × 3 qubits
}

// A fault in a bulk round (not the last) must span two consecutive
// detector bands. Round 0, qubit 0: D0 and D2 both set; D1 and D3 clear.
TEST(DemConstruction, MultiRound_BulkFault_SpansTwoRounds) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise, 2);

  // col 0 = round 0, q0
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);    // D0: round 0, Z0Z1
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 0u);    // D1: round 0, Z1Z2
  EXPECT_EQ(dem.detector_error_matrix.at({2, 0}), 1u);    // D2: round 1, Z0Z1
  EXPECT_EQ(dem.detector_error_matrix.at({3, 0}), 0u);    // D3: round 1, Z1Z2
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 1u); // L0 flipped
}

// A fault in the last round must span only that round's detector band.
// Round 1 (T-1), qubit 0: D2 set; D0 clear (no back-propagation).
TEST(DemConstruction, MultiRound_LastRoundFault_SpansOneRound) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise, 2);

  // col 3 = round 1, q0
  EXPECT_EQ(dem.detector_error_matrix.at({0, 3}), 0u);    // D0: unaffected
  EXPECT_EQ(dem.detector_error_matrix.at({1, 3}), 0u);    // D1: unaffected
  EXPECT_EQ(dem.detector_error_matrix.at({2, 3}), 1u);    // D2: round 1, Z0Z1
  EXPECT_EQ(dem.detector_error_matrix.at({3, 3}), 0u);    // D3: unaffected
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 3}), 1u); // L0 flipped
}

// Faults in every round must flip the observable (the logical measurement
// is taken once at the end; any uncorrected X on the logical qubit matters).
TEST(DemConstruction, MultiRound_ObservableFlipped_AllRounds) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise, 3);

  // col 0 = round 0, q0 → L0 flipped
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 1u);
  // col 3 = round 1, q0 → L0 flipped
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 3}), 1u);
  // col 6 = round 2, q0 → L0 flipped
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 6}), 1u);

  // q1 never flips L0 regardless of round
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 1}), 0u); // round 0, q1
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 4}), 0u); // round 1, q1
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 7}), 0u); // round 2, q1
}

// Cross-check against dem_from_stim_text() for d=3, T=2, px=0.01.
// Detector numbering matches our layout: D0/D1 = round 0, D2/D3 = round 1.
//
//   error(0.01) D0 D2 L0    ← round 0, q0: bulk span + observable
//   error(0.01) D0 D1 D2 D3 ← round 0, q1: bulk span, both checks
//   error(0.01) D1 D3       ← round 0, q2: bulk span
//   error(0.01) D2 L0       ← round 1, q0: single band (last round)
//   error(0.01) D2 D3       ← round 1, q1: single band
//   error(0.01) D3          ← round 1, q2: single band
TEST(DemConstruction, MultiRound_T2_MatchesStimText) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto from_matrices = dem_from_css_matrices(code, noise, 2);

  const std::string stim_dem = "error(0.01) D0 D2 L0\n"
                               "error(0.01) D0 D1 D2 D3\n"
                               "error(0.01) D1 D3\n"
                               "error(0.01) D2 L0\n"
                               "error(0.01) D2 D3\n"
                               "error(0.01) D3\n";
  auto from_stim = dem_from_stim_text(stim_dem);

  EXPECT_TRUE(tensors_equal(from_matrices.detector_error_matrix,
                            from_stim.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(from_matrices.observables_flips_matrix,
                            from_stim.observables_flips_matrix));
  ASSERT_EQ(from_matrices.error_rates.size(), from_stim.error_rates.size());
  for (std::size_t i = 0; i < from_matrices.error_rates.size(); ++i)
    EXPECT_DOUBLE_EQ(from_matrices.error_rates[i], from_stim.error_rates[i]);
}

// T=3 rounds produces the right dimensions and a correctly placed
// interior fault. Round 1 (bulk), q0: spans rounds 1 and 2 only.
//
// Detector rows: D0..D1 = round 0, D2..D3 = round 1, D4..D5 = round 2.
// Column 3 = round 1, q0.
TEST(DemConstruction, MultiRound_T3_InteriorFaultSpan) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  auto dem = dem_from_css_matrices(code, noise, 3);

  EXPECT_EQ(dem.num_detectors(), 6u);        // 3 rounds × 2 checks
  EXPECT_EQ(dem.num_error_mechanisms(), 9u); // 3 rounds × 3 qubits

  // col 3 = round 1, q0: D2 (round 1 Z0Z1) and D4 (round 2 Z0Z1).
  EXPECT_EQ(dem.detector_error_matrix.at({0, 3}), 0u);    // D0: unaffected
  EXPECT_EQ(dem.detector_error_matrix.at({2, 3}), 1u);    // D2: round 1
  EXPECT_EQ(dem.detector_error_matrix.at({4, 3}), 1u);    // D4: round 2 span
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 3}), 1u); // L0 flipped
}

TEST(DemConstruction, MultiRound_ZeroRounds_Throws) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;

  EXPECT_THROW(dem_from_css_matrices(code, noise, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Measurement errors (phenomenological noise model)
// ---------------------------------------------------------------------------

// T=1, pm only: each check has its own column that fires ONE detector
// (last round → no next round to span).
// rep3 has 2 Z-checks → 2 meas-error columns beyond the 3 data columns.
//   col 3: meas error on Z0Z1 → D0 only
//   col 4: meas error on Z1Z2 → D1 only
TEST(DemConstruction, MeasurementErrors_T1_SingleDetector) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.pm = 0.005;

  auto dem = dem_from_css_matrices(code, noise); // T=1 default

  // 2 detectors (single round), 2 fault columns (pm>0, px=0)
  ASSERT_EQ(dem.num_detectors(), 2u);
  ASSERT_EQ(dem.num_error_mechanisms(), 2u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.005);
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.005);

  // col 0 = meas error on Z0Z1: D0 fires, D1 does not
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 0u);
  // No observable flip for measurement errors
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 0u);

  // col 1 = meas error on Z1Z2: D1 fires, D0 does not
  EXPECT_EQ(dem.detector_error_matrix.at({0, 1}), 0u);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 1}), 1u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 1}), 0u);
}

// T=2, pm only: bulk-round meas errors span two rounds; last-round errors
// span one. Column layout per round: Z-check 0, Z-check 1.
//   col 0: meas Z0Z1 round 0 → D0 and D2
//   col 1: meas Z1Z2 round 0 → D1 and D3
//   col 2: meas Z0Z1 round 1 → D2 only (last round)
//   col 3: meas Z1Z2 round 1 → D3 only (last round)
TEST(DemConstruction, MeasurementErrors_T2_BulkAndLastRound) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.pm = 0.005;

  auto dem = dem_from_css_matrices(code, noise, 2);

  ASSERT_EQ(dem.num_detectors(), 4u);        // 2 rounds × 2 checks
  ASSERT_EQ(dem.num_error_mechanisms(), 4u); // 2 rounds × 2 checks

  // Round 0, check 0: spans D0 and D2
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1u);    // D0
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 0u);    // D1 unaffected
  EXPECT_EQ(dem.detector_error_matrix.at({2, 0}), 1u);    // D2 (next round)
  EXPECT_EQ(dem.detector_error_matrix.at({3, 0}), 0u);    // D3 unaffected
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 0u); // no logical flip

  // Round 1 (last), check 0: spans D2 only
  EXPECT_EQ(dem.detector_error_matrix.at({0, 2}), 0u); // D0 unaffected
  EXPECT_EQ(dem.detector_error_matrix.at({2, 2}), 1u); // D2 only
  EXPECT_EQ(dem.detector_error_matrix.at({3, 2}), 0u); // D3 unaffected
}

// Combined px + pm: data errors and measurement errors in the same DEM.
// T=2, rep3: 3 data cols + 2 meas cols per round = 10 total columns.
// Cross-check against manually derived Stim DEM.
//
// Column order per round: [X_q0, X_q1, X_q2, meas_check0, meas_check1]
//
// Stim DEM (T=2, px=0.01, pm=0.005):
//   error(0.01)  D0 D2 L0   ← X q0 round 0
//   error(0.01)  D0 D1 D2 D3 ← X q1 round 0
//   error(0.01)  D1 D3       ← X q2 round 0
//   error(0.005) D0 D2       ← meas Z0Z1 round 0
//   error(0.005) D1 D3       ← meas Z1Z2 round 0
//   error(0.01)  D2 L0       ← X q0 round 1
//   error(0.01)  D2 D3       ← X q1 round 1
//   error(0.01)  D3           ← X q2 round 1
//   error(0.005) D2           ← meas Z0Z1 round 1
//   error(0.005) D3           ← meas Z1Z2 round 1
TEST(DemConstruction, MeasurementErrors_PxPm_T2_MatchesStimText) {
  css_code_matrices code = rep3_z_basis_matrices();
  css_noise_params noise;
  noise.px = 0.01;
  noise.pm = 0.005;

  auto from_matrices = dem_from_css_matrices(code, noise, 2);

  const std::string stim_dem = "error(0.01)  D0 D2 L0\n"
                               "error(0.01)  D0 D1 D2 D3\n"
                               "error(0.01)  D1 D3\n"
                               "error(0.005) D0 D2\n"
                               "error(0.005) D1 D3\n"
                               "error(0.01)  D2 L0\n"
                               "error(0.01)  D2 D3\n"
                               "error(0.01)  D3\n"
                               "error(0.005) D2\n"
                               "error(0.005) D3\n";
  auto from_stim = dem_from_stim_text(stim_dem);

  EXPECT_EQ(from_matrices.num_detectors(), from_stim.num_detectors());
  EXPECT_EQ(from_matrices.num_error_mechanisms(),
            from_stim.num_error_mechanisms());
  EXPECT_TRUE(tensors_equal(from_matrices.detector_error_matrix,
                            from_stim.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(from_matrices.observables_flips_matrix,
                            from_stim.observables_flips_matrix));
  ASSERT_EQ(from_matrices.error_rates.size(), from_stim.error_rates.size());
  for (std::size_t i = 0; i < from_matrices.error_rates.size(); ++i)
    EXPECT_DOUBLE_EQ(from_matrices.error_rates[i], from_stim.error_rates[i])
        << "col=" << i;
}

// hz with nonzero rows but zero columns (while n_qubits > 0 from hx) must
// throw rather than silently producing a DEM with empty Z-type detectors.
TEST(DemConstruction, MalformedHz_RowsWithZeroCols_Throws) {
  css_code_matrices code;
  // hz: 2 rows, 0 columns — nonzero rows, zero columns → inconsistent
  code.hz = sparse_binary_matrix::from_nested_csc(2, 0, {});
  // hx defines n=3; hz should be validated against that
  code.hx = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {0}, {0}});
  css_noise_params noise;
  noise.pz = 0.01;

  EXPECT_THROW(dem_from_css_matrices(code, noise), std::invalid_argument);
}

} // namespace
} // namespace cudaq::qec
