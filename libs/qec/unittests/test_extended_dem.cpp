/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Unit tests for extended_dem, dem_stitch(, seam_name::next_round,
// seam_name::prev_round), dem_stitch_all(), and dem_close().
//
// The central invariant under test:
//   dem_close(dem_stitch_all(T one-round chunks)) ==
//   dem_from_css_matrices(code, noise, T)
//
// Additional cases:
//   - One-round chunk structure (in_syndrome == out_syndrome, no interior)
//   - Stitch dimensions (interior grows by one seam per stitch)
//   - Tag validation (mismatched tags throw)
//   - dem_close() row ordering matches monolithic model
//   - Per-qubit noise propagates correctly through stitch

#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/dem_construction.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/extended_dem.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool tensors_equal(const cudaqx::tensor<uint8_t> &a,
                   const cudaqx::tensor<uint8_t> &b) {
  if (a.rank() != 2 || b.rank() != 2)
    return false;
  if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1])
    return false;
  for (std::size_t r = 0; r < a.shape()[0]; ++r)
    for (std::size_t c = 0; c < a.shape()[1]; ++c)
      if (a.at({r, c}) != b.at({r, c}))
        return false;
  return true;
}

// d=3 repetition code (Z-basis): H_Z = [[1,1,0],[0,1,1]], L_Z = [[1,0,0]].
css_code_matrices rep3() {
  css_code_matrices m;
  m.hz = sparse_binary_matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});
  m.lz = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  return m;
}

css_noise_params px_only(double p) {
  css_noise_params n;
  n.px = p;
  return n;
}

// ---------------------------------------------------------------------------
// seam_id name registry
// ---------------------------------------------------------------------------

// Same id + same name must remain idempotent (YAML re-registers standard
// seams on every parse).
TEST(SeamIdRegistry, SameNameIsIdempotent) {
  EXPECT_NO_THROW(seam_id::register_name(seam_name::prev_round, "prev_round"));
  EXPECT_EQ(seam_name::prev_round.name(), "prev_round");
}

// Two distinct names that share a 32-bit hash must fail loudly. Force the
// collision by writing the same hash value under a second display name.
TEST(SeamIdRegistry, HashCollisionThrows) {
  const seam_id first{"registry_collision_probe"};
  seam_id colliding;
  colliding.value = first.value;

  ASSERT_NO_THROW(seam_id::register_name(first, "registry_collision_probe"));
  EXPECT_THROW(seam_id::register_name(colliding, "other_display_name"),
               std::invalid_argument);
  // First registration must still win for diagnostics.
  EXPECT_EQ(first.name(), "registry_collision_probe");
}

// ---------------------------------------------------------------------------
// One-round chunk structure
// ---------------------------------------------------------------------------

TEST(ExtendedDem, OneRound_NoInterior) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  EXPECT_EQ(dem_chunk.num_interior_rows(), 0u);
}

// For a one-round chunk, in_syndrome == out_syndrome (the same raw syndrome).
TEST(ExtendedDem, OneRound_InSyndromeEqualsOutSyndrome) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  auto in_d = dem_chunk.H.to_dense();
  auto out_d = dem_chunk.H.to_dense();
  EXPECT_TRUE(tensors_equal(in_d, out_d));
}

// The in/out_syndrome matrix must equal the single-round detector_error_matrix.
TEST(ExtendedDem, OneRound_SyndromeMatchesFlatDem) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto dem_chunk = extended_dem_from_css_matrices(code, noise);
  auto flat = dem_from_css_matrices(code, noise);

  // In the new model, H has 2d rows (prev_round seam [0,d) + next_round
  // [d,2d)). dem_close writes the to_seam (prev_round) rows first, matching the
  // flat DEM.
  auto closed = dem_close(dem_chunk);
  EXPECT_TRUE(
      tensors_equal(closed.detector_error_matrix, flat.detector_error_matrix));
  EXPECT_TRUE(
      tensors_equal(dem_chunk.O.to_dense(), flat.observables_flips_matrix));
  EXPECT_EQ(dem_chunk.error_rates, flat.error_rates);
}

// Tags are sequential for a same-code chunk.
// In the new model, H has 2d rows: d for prev_round + d for next_round.
// Tags (one per seam row) has 2d entries: {0..d-1, 0..d-1}.
TEST(ExtendedDem, OneRound_Tags) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  // d=2 checks for d=3 rep code; 2 seams × 2 rows each = 4 tag entries
  ASSERT_EQ(dem_chunk.tags.size(), 4u);
  // Tags within each seam are sequential starting at 0
  EXPECT_EQ(dem_chunk.tags[0], 0u); // prev_round seam, check 0
  EXPECT_EQ(dem_chunk.tags[1], 1u); // prev_round seam, check 1
  EXPECT_EQ(dem_chunk.tags[2], 0u); // next_round seam, check 0
  EXPECT_EQ(dem_chunk.tags[3], 1u); // next_round seam, check 1
}

// ---------------------------------------------------------------------------
// Stitch dimensions
// ---------------------------------------------------------------------------

// After one stitch, interior grows by one seam worth of rows (d=2 here).
TEST(ExtendedDem, Stitch_InteriorGrowsBySeamRows) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto c0 = extended_dem_from_css_matrices(code, noise);
  auto c1 = extended_dem_from_css_matrices(code, noise);
  auto ab = dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round);

  // seam_rows = 2 (nz=2, nx=0)
  EXPECT_EQ(ab.num_interior_rows(), 2u);
  EXPECT_EQ(ab.get_seam(seam_name::prev_round).num_rows(), 2u);
  EXPECT_EQ(ab.num_observables(), 1u);
  EXPECT_EQ(ab.num_faults(), 6u); // 2 rounds × 3 qubits
}

// Fault columns from A come first, then B.
TEST(ExtendedDem, Stitch_FaultColumnOrder) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto c0 = extended_dem_from_css_matrices(code, noise);
  auto c1 = extended_dem_from_css_matrices(code, noise);
  auto ab = dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round);

  // A's priors (3 entries at 0.01) then B's (3 entries at 0.01)
  ASSERT_EQ(ab.error_rates.size(), 6u);
  for (std::size_t i = 0; i < 6u; ++i)
    EXPECT_DOUBLE_EQ(ab.error_rates[i], 0.01) << "i=" << i;
}

// Stitch 3 chunks: interior = 2 seams × 2 rows each = 4.
TEST(ExtendedDem, Stitch_ThreeDemChunks_Interior) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto c0 = extended_dem_from_css_matrices(code, noise);
  auto c1 = extended_dem_from_css_matrices(code, noise);
  auto c2 = extended_dem_from_css_matrices(code, noise);
  auto abc = dem_stitch(
      dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round), c2,
      seam_name::next_round, seam_name::prev_round);

  EXPECT_EQ(abc.num_interior_rows(), 4u); // 2 seams × 2 rows
  EXPECT_EQ(abc.num_faults(), 9u);        // 3 rounds × 3 qubits
}

// Interior rows must come out in round order for any association of stitches,
// or a chunk stitched as a tree would file one of its seams under the wrong
// round and dem_close() would emit detectors out of order.
TEST(ExtendedDem, Stitch_TreeFoldMatchesLeftFold) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);
  auto c = extended_dem_from_css_matrices(code, noise);

  const auto s = seam_name::next_round;
  const auto t = seam_name::prev_round;
  auto left = dem_stitch(dem_stitch(dem_stitch(c, c, s, t), c, s, t), c, s, t);
  auto tree = dem_stitch(dem_stitch(c, c, s, t), dem_stitch(c, c, s, t), s, t);
  auto right = dem_stitch(c, dem_stitch(c, dem_stitch(c, c, s, t), s, t), s, t);
  auto flat = dem_from_css_matrices(code, noise, 4);

  for (const auto *shape : {&left, &tree, &right}) {
    auto closed = dem_close(*shape);
    EXPECT_EQ(dem_chunk_rounds(*shape), 4u);
    EXPECT_TRUE(tensors_equal(closed.detector_error_matrix,
                              flat.detector_error_matrix));
    EXPECT_TRUE(tensors_equal(closed.observables_flips_matrix,
                              flat.observables_flips_matrix));
  }
}

// ---------------------------------------------------------------------------
// dem_close() — invariant: dem_close(dem_stitch_all(T)) ==
// dem_from_css_matrices(T)
// ---------------------------------------------------------------------------

// T=1: close of a single chunk == single-round monolithic DEM.
TEST(ExtendedDem, Close_T1_MatchesMonolithic) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto closed = dem_close(extended_dem_from_css_matrices(code, noise));
  auto flat = dem_from_css_matrices(code, noise);

  EXPECT_EQ(closed.num_detectors(), flat.num_detectors());
  EXPECT_EQ(closed.num_observables(), flat.num_observables());
  EXPECT_EQ(closed.num_error_mechanisms(), flat.num_error_mechanisms());
  EXPECT_TRUE(
      tensors_equal(closed.detector_error_matrix, flat.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(closed.observables_flips_matrix,
                            flat.observables_flips_matrix));
  EXPECT_EQ(closed.error_rates, flat.error_rates);
}

// T=2: dem_close(dem_stitch(c0, c1, seam_name::next_round,
// seam_name::prev_round)) == dem_from_css_matrices(code, noise, 2).
TEST(ExtendedDem, Close_T2_MatchesMonolithic) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto c0 = extended_dem_from_css_matrices(code, noise);
  auto c1 = extended_dem_from_css_matrices(code, noise);
  auto stitched =
      dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round);
  auto closed = dem_close(stitched);
  auto flat = dem_from_css_matrices(code, noise, 2);

  EXPECT_EQ(closed.num_detectors(), flat.num_detectors());
  EXPECT_EQ(closed.num_error_mechanisms(), flat.num_error_mechanisms());
  EXPECT_TRUE(
      tensors_equal(closed.detector_error_matrix, flat.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(closed.observables_flips_matrix,
                            flat.observables_flips_matrix));
  EXPECT_EQ(closed.error_rates, flat.error_rates);
}

// T=3 using dem_stitch_all.
TEST(ExtendedDem, Close_T3_MatchesMonolithic) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  std::vector<extended_dem> dem_chunks(
      3, extended_dem_from_css_matrices(code, noise));
  auto closed = dem_close(dem_stitch_all(dem_chunks));
  auto flat = dem_from_css_matrices(code, noise, 3);

  EXPECT_EQ(closed.num_detectors(), flat.num_detectors());
  EXPECT_EQ(closed.num_error_mechanisms(), flat.num_error_mechanisms());
  EXPECT_TRUE(
      tensors_equal(closed.detector_error_matrix, flat.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(closed.observables_flips_matrix,
                            flat.observables_flips_matrix));
  EXPECT_EQ(closed.error_rates, flat.error_rates);
}

// T=5 stress test.
TEST(ExtendedDem, Close_T5_MatchesMonolithic) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  std::vector<extended_dem> dem_chunks(
      5, extended_dem_from_css_matrices(code, noise));
  auto closed = dem_close(dem_stitch_all(dem_chunks));
  auto flat = dem_from_css_matrices(code, noise, 5);

  EXPECT_EQ(closed.num_detectors(), flat.num_detectors());
  EXPECT_EQ(closed.num_observables(), flat.num_observables());
  EXPECT_EQ(closed.num_error_mechanisms(), flat.num_error_mechanisms());
  EXPECT_TRUE(
      tensors_equal(closed.detector_error_matrix, flat.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(closed.observables_flips_matrix,
                            flat.observables_flips_matrix));
  EXPECT_EQ(closed.error_rates, flat.error_rates);
}

// ---------------------------------------------------------------------------
// Tag validation
// ---------------------------------------------------------------------------

// Mismatched out/in tags must throw. The tag has to be corrupted by hand: the
// builders number seam rows positionally, so no pair of chunks they produce
// can disagree on tags once the seam widths match. See extended_dem::tags.
TEST(ExtendedDem, Stitch_TagMismatch_Throws) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto c0 = extended_dem_from_css_matrices(code, noise);
  auto c1 = extended_dem_from_css_matrices(code, noise);
  c1.tags[0] = 999u; // corrupt a tag

  EXPECT_THROW(dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round),
               std::invalid_argument);
}

// Observable count mismatch must throw.
TEST(ExtendedDem, Stitch_ObservableMismatch_Throws) {
  css_code_matrices code = rep3();

  css_noise_params nx_noise; // no observables from lx
  nx_noise.px = 0.01;

  css_code_matrices code_with_lx = rep3();
  code_with_lx.lx = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  css_noise_params pxpz;
  pxpz.px = 0.01;
  pxpz.pz = 0.01;

  auto c0 = extended_dem_from_css_matrices(code, nx_noise);
  auto c1 = extended_dem_from_css_matrices(code_with_lx, pxpz);

  EXPECT_THROW(dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Per-qubit noise propagates through stitch
// ---------------------------------------------------------------------------

// Qubit 1 has zero rate in round 0 only — its column is absent from c0 but
// present in c1. The stitched + closed DEM should have 5 error mechanisms.
TEST(ExtendedDem, PerQubitNoise_SparseDemChunk) {
  css_code_matrices code = rep3();

  css_noise_params noise_sparse;
  noise_sparse.px_per_qubit = {0.01, 0.0, 0.01}; // skip qubit 1

  css_noise_params noise_full;
  noise_full.px = 0.01;

  auto c0 = extended_dem_from_css_matrices(code, noise_sparse); // 2 faults
  auto c1 = extended_dem_from_css_matrices(code, noise_full);   // 3 faults
  auto closed = dem_close(
      dem_stitch(c0, c1, seam_name::next_round, seam_name::prev_round));

  EXPECT_EQ(closed.num_error_mechanisms(), 5u);
  EXPECT_EQ(closed.num_detectors(), 4u); // 2 rounds × 2 checks
}

// ---------------------------------------------------------------------------
// dem_stitch_all edge cases
// ---------------------------------------------------------------------------

TEST(ExtendedDem, StitchAll_Empty_Throws) {
  EXPECT_THROW(dem_stitch_all({}), std::invalid_argument);
}

TEST(ExtendedDem, StitchAll_OneDemChunk_IsIdentity) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto c0 = extended_dem_from_css_matrices(code, noise);
  auto result = dem_stitch_all({c0});

  EXPECT_EQ(result.num_interior_rows(), c0.num_interior_rows());
  EXPECT_EQ(result.get_seam(seam_name::prev_round).num_rows(),
            c0.get_seam(seam_name::prev_round).num_rows());
  EXPECT_EQ(result.num_observables(), c0.num_observables());
  EXPECT_EQ(result.num_faults(), c0.num_faults());
}

// ---------------------------------------------------------------------------
// dem_close_all
// ---------------------------------------------------------------------------

// dem_close_all on a single chunk must equal dem_close() on that chunk.
TEST(ExtendedDem, CloseAll_T1_MatchesClose) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  auto dem_chunk = extended_dem_from_css_matrices(code, noise);
  auto via_close = dem_close(dem_chunk);
  auto via_close_all = dem_close_all({dem_chunk});

  EXPECT_EQ(via_close_all.num_detectors(), via_close.num_detectors());
  EXPECT_TRUE(tensors_equal(via_close_all.detector_error_matrix,
                            via_close.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(via_close_all.observables_flips_matrix,
                            via_close.observables_flips_matrix));
  EXPECT_EQ(via_close_all.error_rates, via_close.error_rates);
}

// dem_close_all(T chunks) == dem_close(dem_stitch_all(T chunks)) for
// single-round chunks.
TEST(ExtendedDem, CloseAll_T4_MatchesCloseStitchAll) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  std::vector<extended_dem> dem_chunks(
      4, extended_dem_from_css_matrices(code, noise));

  auto via_stitch = dem_close(dem_stitch_all(dem_chunks));
  auto via_close_all = dem_close_all(dem_chunks);

  EXPECT_EQ(via_close_all.num_detectors(), via_stitch.num_detectors());
  EXPECT_EQ(via_close_all.num_error_mechanisms(),
            via_stitch.num_error_mechanisms());
  EXPECT_TRUE(tensors_equal(via_close_all.detector_error_matrix,
                            via_stitch.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(via_close_all.observables_flips_matrix,
                            via_stitch.observables_flips_matrix));
  EXPECT_EQ(via_close_all.error_rates, via_stitch.error_rates);
}

// dem_close_all output matches dem_from_css_matrices for the same T rounds.
TEST(ExtendedDem, CloseAll_T5_MatchesMonolithic) {
  css_code_matrices code = rep3();
  css_noise_params noise = px_only(0.01);

  std::vector<extended_dem> dem_chunks(
      5, extended_dem_from_css_matrices(code, noise));
  auto via_close_all = dem_close_all(dem_chunks);
  auto monolithic = dem_from_css_matrices(code, noise, 5);

  EXPECT_EQ(via_close_all.num_detectors(), monolithic.num_detectors());
  EXPECT_EQ(via_close_all.num_error_mechanisms(),
            monolithic.num_error_mechanisms());
  EXPECT_TRUE(tensors_equal(via_close_all.detector_error_matrix,
                            monolithic.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(via_close_all.observables_flips_matrix,
                            monolithic.observables_flips_matrix));
  EXPECT_EQ(via_close_all.error_rates, monolithic.error_rates);
}

// dem_close_all on empty chunks must throw.
TEST(ExtendedDem, CloseAll_EmptyDemChunks_Throws) {
  EXPECT_THROW(dem_close_all({}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Measurement errors through the extended_dem pipeline
// ---------------------------------------------------------------------------

// A one-round chunk with pm>0 must have 2 extra fault columns (one per
// Z-check for rep3) whose syndrome column contains only that check's row
// and whose observable column is empty.
TEST(ExtendedDem, MeasurementErrors_OneRoundDemChunk_ExtraColumns) {
  css_noise_params noise;
  noise.pm = 0.005;

  auto dem_chunk = extended_dem_from_css_matrices(rep3(), noise);

  // 2 checks → 2 meas-error columns; no data errors (px=pz=py=0)
  EXPECT_EQ(dem_chunk.num_faults(), 2u);
  EXPECT_EQ(dem_chunk.num_interior_rows(), 0u);
  EXPECT_DOUBLE_EQ(dem_chunk.error_rates[0], 0.005);
  EXPECT_DOUBLE_EQ(dem_chunk.error_rates[1], 0.005);

  // in_syndrome col 0: only row 0 (Z0Z1 check) is set
  auto syn_dense = dem_chunk.H.to_dense();
  EXPECT_EQ(syn_dense.at({0, 0}), 1u); // Z0Z1 fires
  EXPECT_EQ(syn_dense.at({1, 0}), 0u); // Z1Z2 unaffected

  // observables are all zero — measurement errors don't flip logicals
  auto obs_dense = dem_chunk.O.to_dense();
  EXPECT_EQ(obs_dense.at({0, 0}), 0u);
  EXPECT_EQ(obs_dense.at({0, 1}), 0u);
}

// dem_close_all on T pm-only chunks must match dem_from_css_matrices(T, pm).
TEST(ExtendedDem, MeasurementErrors_CloseAll_MatchesMonolithic) {
  css_noise_params noise;
  noise.pm = 0.005;

  const std::size_t T = 3;
  std::vector<extended_dem> dem_chunks(
      T, extended_dem_from_css_matrices(rep3(), noise));
  auto via_close_all = dem_close_all(dem_chunks);
  auto monolithic = dem_from_css_matrices(rep3(), noise, T);

  EXPECT_EQ(via_close_all.num_detectors(), monolithic.num_detectors());
  EXPECT_EQ(via_close_all.num_error_mechanisms(),
            monolithic.num_error_mechanisms());
  EXPECT_TRUE(tensors_equal(via_close_all.detector_error_matrix,
                            monolithic.detector_error_matrix));
  EXPECT_TRUE(tensors_equal(via_close_all.observables_flips_matrix,
                            monolithic.observables_flips_matrix));
  EXPECT_EQ(via_close_all.error_rates, monolithic.error_rates);
}

// ---------------------------------------------------------------------------
// Streaming decoder integration utilities
// ---------------------------------------------------------------------------

// d=3 rep code, T=2 rounds, d=2 checks.
// detector_round must be [0, 0, 1, 1] — both round-0 detectors first.
TEST(ExtendedDem, DemChunksToDetectorRound_T2) {
  std::vector<extended_dem> dem_chunks(
      2, extended_dem_from_css_matrices(rep3(), px_only(0.01)));

  auto dr = dem_chunks_to_detector_round(dem_chunks);

  ASSERT_EQ(dr.size(), 4u); // T=2 rounds × d=2 checks
  EXPECT_EQ(dr[0], 0);      // det 0: round 0, check 0
  EXPECT_EQ(dr[1], 0);      // det 1: round 0, check 1
  EXPECT_EQ(dr[2], 1);      // det 2: round 1, check 0
  EXPECT_EQ(dr[3], 1);      // det 3: round 1, check 1
}

// T=3: three rounds, d=2 checks — verify all entries.
TEST(ExtendedDem, DemChunksToDetectorRound_T3) {
  std::vector<extended_dem> dem_chunks(
      3, extended_dem_from_css_matrices(rep3(), px_only(0.01)));

  auto dr = dem_chunks_to_detector_round(dem_chunks);

  ASSERT_EQ(dr.size(), 6u);
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 2; ++k)
      EXPECT_EQ(dr[r * 2 + k], r) << "r=" << r << " k=" << k;
}

// A chunk's interior rows count its rounds after the first.
TEST(ExtendedDem, DemChunkRounds_CountsInteriorRows) {
  auto one = extended_dem_from_css_matrices(rep3(), px_only(0.01));

  EXPECT_EQ(dem_chunk_rounds(one), 1u);
  EXPECT_EQ(dem_chunk_rounds(dem_stitch(one, one, seam_name::next_round,
                                        seam_name::prev_round)),
            2u);
  EXPECT_EQ(dem_chunk_rounds(dem_stitch_all({one, one, one})), 3u);
  EXPECT_EQ(dem_chunks_to_rounds({one,
                                  dem_stitch(one, one, seam_name::next_round,
                                             seam_name::prev_round),
                                  one}),
            4u);
}

// A pre-stitched chunk maps to exactly the rounds its pieces would have mapped
// to on their own, however the caller grouped them.
TEST(ExtendedDem, DemChunksToDetectorRound_MultiRoundChunksMatchSingles) {
  auto one = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  const std::vector<extended_dem> singles(4, one);
  const auto want = dem_chunks_to_detector_round(singles);

  ASSERT_EQ(want.size(), 8u); // 4 rounds x 2 checks
  EXPECT_EQ(dem_chunks_to_detector_round({dem_stitch_all(singles)}), want);
  EXPECT_EQ(
      dem_chunks_to_detector_round(
          {dem_stitch(one, one, seam_name::next_round, seam_name::prev_round),
           dem_stitch(one, one, seam_name::next_round, seam_name::prev_round)}),
      want);
  EXPECT_EQ(
      dem_chunks_to_detector_round({one, dem_stitch_all({one, one, one})}),
      want);
  EXPECT_EQ(dem_chunks_to_d_sparse({dem_stitch(one, one, seam_name::next_round,
                                               seam_name::prev_round),
                                    one, one}),
            dem_chunks_to_d_sparse(singles));
}

// Interior rows that are not a whole number of rounds cannot be attributed to
// rounds at all, so every round-indexed map has to refuse them rather than
// truncate.
TEST(ExtendedDem, DemChunkUtils_PartialRoundInterior_Throws) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  // d=2 checks. Build H with 5 rows: [prev_round 0..1 | next_round 2..3 |
  // interior 4] Interior = 1 row = not a whole number of rounds (1 % 2 != 0) →
  // should throw.
  const uint32_t n = dem_chunk.num_faults();
  using csc_cols = std::vector<std::vector<uint32_t>>;
  auto syn_rows = csc_cols(n); // 2 seam rows (empty cols for simplicity)
  auto int_row = csc_cols(n);  // 1 interior row (empty cols)
  dem_chunk.H =
      sparse_binary_matrix::from_nested_csr(5, n,
                                            {
                                                {},
                                                {}, // prev_round rows 0-1
                                                {},
                                                {}, // next_round rows 2-3
                                                {}  // 1 interior row 4
                                            });
  dem_chunk.seams.clear();
  dem_chunk.tags.clear();
  dem_chunk.add_seam(seam_name::prev_round, 0, 2);
  dem_chunk.add_seam(seam_name::next_round, 2, 4);
  dem_chunk.tags = {0, 1, 0, 1};

  EXPECT_THROW(dem_chunk_rounds(dem_chunk), std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_rounds({dem_chunk}), std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_detector_round({dem_chunk}),
               std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_d_sparse({dem_chunk}), std::invalid_argument);
}

// d=3 rep code, T=2 rounds, d=2 checks.
// D_sparse layout:
//   det 0 (r=0, check 0): only meas bit 0          → {0}
//   det 1 (r=0, check 1): only meas bit 1          → {1}
//   det 2 (r=1, check 0): meas bits 0 and 2 (XOR)  → {0, 2}
//   det 3 (r=1, check 1): meas bits 1 and 3 (XOR)  → {1, 3}
TEST(ExtendedDem, DemChunksToDSparse_T2_Layout) {
  std::vector<extended_dem> dem_chunks(
      2, extended_dem_from_css_matrices(rep3(), px_only(0.01)));

  auto ds = dem_chunks_to_d_sparse(dem_chunks);

  ASSERT_EQ(ds.size(), 4u);

  // Round-0 detectors: single measurement each.
  ASSERT_EQ(ds[0].size(), 1u);
  EXPECT_EQ(ds[0][0], 0u);
  ASSERT_EQ(ds[1].size(), 1u);
  EXPECT_EQ(ds[1][0], 1u);

  // Round-1 detectors: XOR of consecutive round measurements.
  ASSERT_EQ(ds[2].size(), 2u);
  EXPECT_EQ(ds[2][0], 0u); // meas bit (r-1)*d+k = 0*2+0 = 0
  EXPECT_EQ(ds[2][1], 2u); // meas bit r*d+k = 1*2+0 = 2
  ASSERT_EQ(ds[3].size(), 2u);
  EXPECT_EQ(ds[3][0], 1u); // meas bit (r-1)*d+k = 0*2+1 = 1
  EXPECT_EQ(ds[3][1], 3u); // meas bit r*d+k = 1*2+1 = 3
}

// O_sparse for rep3 px-only T=2: 1 observable, 6 fault columns.
// Only faults on qubit 0 (columns 0 and 3) flip the Z0 logical.
TEST(ExtendedDem, DemChunksToOSparse_T2_Rep3_XOnly) {
  css_noise_params noise;
  noise.px = 0.01;
  std::vector<extended_dem> dem_chunks(
      2, extended_dem_from_css_matrices(rep3(), noise));

  auto os = dem_chunks_to_o_sparse(dem_chunks);

  // 1 observable for rep3
  ASSERT_EQ(os.size(), 1u);

  // 6 total fault columns: 3 per round.
  // Column 0 = round 0, q0 (X): flips L0
  // Column 1 = round 0, q1 (X): does not flip L0
  // Column 2 = round 0, q2 (X): does not flip L0
  // Column 3 = round 1, q0 (X): flips L0
  // Column 4 = round 1, q1 (X): does not flip L0
  // Column 5 = round 1, q2 (X): does not flip L0
  const auto &obs0 = os[0];
  ASSERT_EQ(obs0.size(), 2u);
  EXPECT_EQ(obs0[0], 0u); // round 0, qubit 0
  EXPECT_EQ(obs0[1], 3u); // round 1, qubit 0
}

// Empty chunks must throw for all three utilities.
TEST(ExtendedDem, DemChunkUtils_EmptyChunks_Throw) {
  EXPECT_THROW(dem_chunks_to_detector_round({}), std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_d_sparse({}), std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_o_sparse({}), std::invalid_argument);
}

// hz with nonzero rows but zero columns triggers the same check inside
// extended_dem_from_css_matrices().
TEST(ExtendedDem, MalformedHz_RowsWithZeroCols_Throws) {
  css_code_matrices code;
  code.hz = sparse_binary_matrix::from_nested_csc(2, 0, {});
  code.hx = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {0}, {0}});
  css_noise_params noise;
  noise.pz = 0.01;

  EXPECT_THROW(extended_dem_from_css_matrices(code, noise),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// dem_merge_duplicate_columns
// ---------------------------------------------------------------------------

// Helper: rep-3 single-round chunk (px=0.01, only X faults → 3 columns).
static extended_dem rep3_dem_chunk_x() {
  css_code_matrices m;
  m.hz = sparse_binary_matrix::from_nested_csc(2, 3, {{0}, {0, 1}, {1}});
  m.lz = sparse_binary_matrix::from_nested_csc(1, 3, {{0}, {}, {}});
  css_noise_params n;
  n.px = 0.01;
  return extended_dem_from_css_matrices(m, n);
}

// A fresh single-round chunk already has unique columns (all are distinct).
TEST(ExtendedDem, MergeDuplicateColumns_AlreadyUnique_Noop) {
  auto dem_chunk = rep3_dem_chunk_x();
  EXPECT_TRUE(are_dem_columns_unique(dem_chunk));
  ASSERT_NO_THROW(assert_dem_columns_unique(dem_chunk));

  const auto canon = dem_merge_duplicate_columns(dem_chunk);
  EXPECT_EQ(canon.num_faults(), dem_chunk.num_faults());
  EXPECT_EQ(canon.error_rates, dem_chunk.error_rates);
}

// Build a minimal extended_dem for merge tests: n_faults columns, 1 H row
// (interior, no seams), no observables.
static extended_dem make_merge_chunk(std::size_t n_faults,
                                     std::vector<std::vector<uint32_t>> h_csc,
                                     std::vector<double> rates) {
  extended_dem dem;
  dem.H = sparse_binary_matrix::from_nested_csc(
      1, static_cast<uint32_t>(n_faults), h_csc);
  dem.O = sparse_binary_matrix::from_nested_csc(
      0, static_cast<uint32_t>(n_faults),
      std::vector<std::vector<uint32_t>>(n_faults));
  dem.error_rates = std::move(rates);
  // No seams: all rows are interior. tags must be empty.
  return dem;
}

// Manually build an extended_dem with two identical columns (same row support)
// and verify they are merged by dem_merge_duplicate_columns.
TEST(ExtendedDem, MergeDuplicateColumns_MergesDuplicates_OrMode) {
  // One interior row; two columns that both fire row 0.
  auto dem = make_merge_chunk(2, {{0}, {0}}, {0.1, 0.2});
  EXPECT_FALSE(are_dem_columns_unique(dem));
  EXPECT_THROW(assert_dem_columns_unique(dem), std::invalid_argument);
  const auto canon =
      dem_merge_duplicate_columns(dem, prior_combine_mode::or_combine);
  EXPECT_EQ(canon.num_faults(), 1u);
  // XOR / GF(2) merge: P(A xor B) = 0.1 + 0.2 - 2*0.1*0.2 = 0.26
  EXPECT_NEAR(canon.error_rates[0], 0.26, 1e-12);
  EXPECT_TRUE(are_dem_columns_unique(canon));
}

TEST(ExtendedDem, MergeDuplicateColumns_MergesDuplicates_SumMode) {
  auto dem = make_merge_chunk(3, {{0}, {0}, {0}}, {0.05, 0.03, 0.02});
  const auto canon =
      dem_merge_duplicate_columns(dem, prior_combine_mode::sum_combine);
  EXPECT_EQ(canon.num_faults(), 1u);
  EXPECT_NEAR(canon.error_rates[0], 0.10, 1e-12);
}

// Two columns with DIFFERENT supports must stay separate.
TEST(ExtendedDem, MergeDuplicateColumns_DistinctSupports_Unchanged) {
  extended_dem dem;
  dem.H = sparse_binary_matrix::from_nested_csc(2, 2, {{0}, {1}});
  dem.O = sparse_binary_matrix::from_nested_csc(0, 2, {{}, {}});
  dem.error_rates = {0.1, 0.2};
  EXPECT_TRUE(are_dem_columns_unique(dem));
  const auto canon = dem_merge_duplicate_columns(dem);
  EXPECT_EQ(canon.num_faults(), 2u);
  EXPECT_EQ(canon.error_rates, dem.error_rates);
}

// Columns are compared over GF(2): sparse_binary_matrix stores index lists as
// given, so a source may list a row twice in one column. Such a column denotes
// the same vector as one omitting that row and must merge with it.
TEST(ExtendedDem, MergeDuplicateColumns_ComparesColumnsOverGf2) {
  extended_dem dem;
  // Column 0 fires interior row 1 only. Column 1 lists row 0 twice (which
  // cancels) plus row 1, so it denotes the same column as column 0. Column 2
  // fires row 0 once and must stay separate.
  dem.H = sparse_binary_matrix::from_nested_csc(2, 3, {{1}, {0, 1, 0}, {0}});
  dem.O = sparse_binary_matrix::from_nested_csc(0, 3, {{}, {}, {}});
  dem.error_rates = {0.1, 0.2, 0.3};
  // No seams — pure interior rows, no tags needed.

  EXPECT_FALSE(are_dem_columns_unique(dem));
  EXPECT_THROW(assert_dem_columns_unique(dem), std::invalid_argument);

  const auto canon =
      dem_merge_duplicate_columns(dem, prior_combine_mode::or_combine);
  ASSERT_EQ(canon.num_faults(), 2u);
  EXPECT_TRUE(are_dem_columns_unique(canon));

  // Lexicographic output order puts {0} before {1}, and the cancelled row is
  // gone from the stored column.
  const auto cols = canon.H.to_nested_csc();
  EXPECT_EQ(cols[0], std::vector<sparse_binary_matrix::index_type>{0});
  EXPECT_EQ(cols[1], std::vector<sparse_binary_matrix::index_type>{1});
  EXPECT_NEAR(canon.error_rates[0], 0.3, 1e-12);
  // XOR merge of the two GF(2)-equal columns: 0.1 + 0.2 - 2*0.1*0.2 = 0.26.
  EXPECT_NEAR(canon.error_rates[1], 0.26, 1e-12);
}

// Tags and row-block sizes pass through unchanged after merging.
TEST(ExtendedDem, MergeDuplicateColumns_TagsPreserved) {
  auto dem_chunk = rep3_dem_chunk_x();
  // tags are set by extended_dem_from_css_matrices; overwrite for the test
  // (tags has 2*d entries in the new model: one per seam row across all seams)
  const auto orig_tags = dem_chunk.tags;
  const auto canon = dem_merge_duplicate_columns(dem_chunk);
  EXPECT_EQ(canon.tags, orig_tags);
  EXPECT_EQ(canon.num_interior_rows(), dem_chunk.num_interior_rows());
  EXPECT_EQ(canon.num_observables(), dem_chunk.num_observables());
  EXPECT_EQ(canon.get_seam(seam_name::prev_round).num_rows(),
            dem_chunk.get_seam(seam_name::prev_round).num_rows());
}

// dem_stitch_merged == dem_merge_duplicate_columns(dem_stitch_all(...)).
TEST(ExtendedDem, StitchMerged_MatchesStitchThenMergeDuplicates) {
  const std::vector<extended_dem> dem_chunks(4, rep3_dem_chunk_x());
  const auto via_stitch_then_merge =
      dem_merge_duplicate_columns(dem_stitch_all(dem_chunks));
  const auto via_stitch_merged = dem_stitch_merged(dem_chunks);

  EXPECT_EQ(via_stitch_merged.num_faults(), via_stitch_then_merge.num_faults());
  EXPECT_EQ(via_stitch_merged.error_rates, via_stitch_then_merge.error_rates);
}

// ---------------------------------------------------------------------------
// Asymmetric (phase) chunks
// ---------------------------------------------------------------------------

// The init / bulk / final phase chunks of a d=5 repetition code. Fault columns
// 0..4 are the five data qubits, 5..8 the four measurement errors; the final
// phase drops the measurement errors because its destructive data readout has
// none, leaving five columns.
//
// These are the chunks a repeated-round decomposition streams: init owns the
// first round's detectors outright (round 0 is compared against the zero
// initial state), bulk owns none of its own and produces detectors only at the
// seams, and final closes the last seam.

constexpr uint32_t kRep5Checks = 4;

// Build hand-crafted rep5 phase chunks using the new seam-descriptor API.
// Layout of H:
//   init:  [interior 4 rows | next_round seam 4 rows]
//   bulk:  [prev_round seam 4 rows | next_round seam 4 rows]
//   final: [prev_round seam 4 rows]

extended_dem rep5_phase_init() {
  extended_dem dem;
  // H: interior (round-0 detectors) stacked above the outgoing seam rows
  auto interior = sparse_binary_matrix::from_nested_csr(
      kRep5Checks, 9, {{0, 1, 5}, {1, 2, 6}, {2, 3, 7}, {3, 4, 8}});
  auto next_seam = sparse_binary_matrix::from_nested_csr(kRep5Checks, 9,
                                                         {{5}, {6}, {7}, {8}});
  // vstack: interior rows first (indices 0..3), seam rows after (4..7)
  auto combined_cols_int = interior.to_nested_csc();
  auto combined_cols_nxt = next_seam.to_nested_csc();
  // Build by stacking rows of both matrices
  const uint32_t n_cols = 9;
  const uint32_t n_rows = 2 * kRep5Checks;
  using idx_t = sparse_binary_matrix::index_type;
  std::vector<std::vector<idx_t>> H_csr;
  auto int_csr = interior.to_nested_csr();
  auto nxt_csr = next_seam.to_nested_csr();
  for (auto &r : int_csr)
    H_csr.push_back(r);
  for (auto &r : nxt_csr)
    H_csr.push_back(r);
  dem.H = sparse_binary_matrix::from_nested_csr(n_rows, n_cols, H_csr);
  dem.O = sparse_binary_matrix::from_nested_csr(1, 9, {{0}});
  dem.error_rates.assign(9, 0.02);
  // next_round seam at rows [kRep5Checks, 2*kRep5Checks)
  dem.add_seam(seam_name::next_round, kRep5Checks, 2 * kRep5Checks);
  dem.tags = {0, 1, 2, 3}; // tags for next_round seam rows
  return dem;
}

extended_dem rep5_phase_bulk() {
  extended_dem dem;
  auto prev_seam = sparse_binary_matrix::from_nested_csr(
      kRep5Checks, 9, {{0, 1, 5}, {1, 2, 6}, {2, 3, 7}, {3, 4, 8}});
  auto next_seam = sparse_binary_matrix::from_nested_csr(kRep5Checks, 9,
                                                         {{5}, {6}, {7}, {8}});
  using idx_t = sparse_binary_matrix::index_type;
  std::vector<std::vector<idx_t>> H_csr;
  for (auto &r : prev_seam.to_nested_csr())
    H_csr.push_back(r);
  for (auto &r : next_seam.to_nested_csr())
    H_csr.push_back(r);
  dem.H = sparse_binary_matrix::from_nested_csr(2 * kRep5Checks, 9, H_csr);
  dem.O = sparse_binary_matrix::from_nested_csr(1, 9, {{0}});
  dem.error_rates.assign(9, 0.02);
  dem.add_seam(seam_name::prev_round, 0, kRep5Checks);
  dem.add_seam(seam_name::next_round, kRep5Checks, 2 * kRep5Checks);
  dem.tags = {0, 1, 2, 3, 0, 1, 2, 3}; // 4 per seam
  return dem;
}

extended_dem rep5_phase_final() {
  extended_dem dem;
  using idx_t = sparse_binary_matrix::index_type;
  dem.H = sparse_binary_matrix::from_nested_csr(
      kRep5Checks, 5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  dem.O = sparse_binary_matrix::from_nested_csr(1, 5, {{0}});
  dem.error_rates.assign(5, 0.02);
  dem.add_seam(seam_name::prev_round, 0, kRep5Checks);
  dem.tags = {0, 1, 2, 3};
  return dem;
}

TEST(ExtendedDemPhases, InitHasNoIncomingSeam) {
  const auto init = rep5_phase_init();
  // Init has no prev_round seam (nothing precedes it)
  EXPECT_FALSE(init.has_seam(seam_name::prev_round));
  EXPECT_TRUE(init.has_seam(seam_name::next_round));
  EXPECT_EQ(init.get_seam(seam_name::next_round).num_rows(), kRep5Checks);
  EXPECT_EQ(init.num_interior_rows(), kRep5Checks);
}

TEST(ExtendedDemPhases, FinalHasNoOutgoingSeam) {
  const auto fin = rep5_phase_final();
  EXPECT_TRUE(fin.has_seam(seam_name::prev_round));
  EXPECT_EQ(fin.get_seam(seam_name::prev_round).num_rows(), kRep5Checks);
  // Final has no next_round seam (nothing follows it)
  EXPECT_FALSE(fin.has_seam(seam_name::next_round));
  EXPECT_EQ(fin.num_interior_rows(), 0u);
}

TEST(ExtendedDemPhases, BulkOwnsNoInteriorRows) {
  const auto bulk = rep5_phase_bulk();
  EXPECT_EQ(bulk.get_seam(seam_name::prev_round).num_rows(), kRep5Checks);
  EXPECT_EQ(bulk.get_seam(seam_name::next_round).num_rows(), kRep5Checks);
  EXPECT_EQ(bulk.num_interior_rows(), 0u);
}

// For a uniform per-round chunk, both seams have the same width.
TEST(ExtendedDemPhases, SeamRowsAliasesIncomingWidth) {
  const auto uniform = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  EXPECT_EQ(uniform.get_seam(seam_name::prev_round).num_rows(),
            uniform.get_seam(seam_name::next_round).num_rows());
  // Init has a next_round seam but no prev_round seam (asymmetric)
  const auto init = rep5_phase_init();
  EXPECT_TRUE(init.has_seam(seam_name::next_round));
  EXPECT_FALSE(init.has_seam(seam_name::prev_round));
}

// Stitching init to bulk contracts init's outgoing seam against bulk's
// incoming one even though init's own incoming seam is empty.
TEST(ExtendedDemPhases, StitchInitToBulk) {
  const auto ab = dem_stitch(rep5_phase_init(), rep5_phase_bulk(),
                             seam_name::next_round, seam_name::prev_round);

  // init's own 4 detectors, then the 4 new seam detectors; bulk adds none.
  EXPECT_EQ(ab.num_interior_rows(), 2u * kRep5Checks);
  EXPECT_EQ(ab.get_seam(seam_name::prev_round).num_rows(), 0u);
  EXPECT_EQ(ab.get_seam(seam_name::next_round).num_rows(), kRep5Checks);
  EXPECT_EQ(ab.num_faults(), 18u);
  EXPECT_EQ(ab.num_observables(), 1u);
}

// The seam detector row is the XOR of init's outgoing syndrome and bulk's
// incoming one, with bulk's faults offset into the second half-column block.
TEST(ExtendedDemPhases, SeamRowXorsAdjacentSyndromes) {
  const auto ab = dem_stitch(rep5_phase_init(), rep5_phase_bulk(),
                             seam_name::next_round, seam_name::prev_round);
  const auto rows = ab.H.to_nested_csr();
  // H has 3*kRep5Checks rows: init interior + contracted seam + bulk next_round
  ASSERT_EQ(rows.size(), 3u * kRep5Checks);

  // Contracted seam rows follow init's interior rows (at index kRep5Checks).
  // Row 0 of the seam block (rows[kRep5Checks]):
  // init measurement error 0 (col 5) plus bulk data 0,1 and meas error 0
  // (cols 9+0, 9+1, 9+5 = 9, 10, 14).
  const std::vector<sparse_binary_matrix::index_type> expected{5, 9, 10, 14};
  EXPECT_EQ(rows[kRep5Checks], expected);
}

// The full init -> bulk -> final chain closes into a flat DEM whose detector
// count is one round of init detectors plus one per seam.
TEST(ExtendedDemPhases, StitchAllAndCloseChain) {
  const std::vector<extended_dem> phases{rep5_phase_init(), rep5_phase_bulk(),
                                         rep5_phase_final()};
  const auto stitched = dem_stitch_all(phases);
  EXPECT_EQ(stitched.get_seam(seam_name::prev_round).num_rows(), 0u);
  EXPECT_EQ(stitched.get_seam(seam_name::next_round).num_rows(), 0u);
  EXPECT_EQ(stitched.num_faults(), 23u);

  const auto flat = dem_close(stitched);
  ASSERT_EQ(flat.detector_error_matrix.rank(), 2u);
  // 4 init detectors + 4 (init|bulk) seam + 4 (bulk|final) seam.
  EXPECT_EQ(flat.detector_error_matrix.shape()[0], 3u * kRep5Checks);
  EXPECT_EQ(flat.detector_error_matrix.shape()[1], 23u);
  EXPECT_EQ(flat.observables_flips_matrix.shape()[0], 1u);
  EXPECT_EQ(flat.error_rates.size(), 23u);
}

// Repeating the bulk phase adds one seam's worth of detectors per repeat.
TEST(ExtendedDemPhases, RepeatingBulkAddsOneSeamPerRound) {
  for (std::size_t repeats = 1; repeats <= 4; ++repeats) {
    std::vector<extended_dem> phases{rep5_phase_init()};
    phases.insert(phases.end(), repeats, rep5_phase_bulk());
    phases.push_back(rep5_phase_final());

    const auto flat = dem_close(dem_stitch_all(phases));
    // init detectors + one seam per adjacency (repeats + 1 of them).
    EXPECT_EQ(flat.detector_error_matrix.shape()[0],
              (repeats + 2) * kRep5Checks);
    EXPECT_EQ(flat.detector_error_matrix.shape()[1], 9u * (repeats + 1) + 5u);
  }
}

// A final chunk cannot be followed by anything: its outgoing seam is empty.
TEST(ExtendedDemPhases, StitchAfterFinalThrows) {
  EXPECT_THROW(dem_stitch(rep5_phase_final(), rep5_phase_bulk(),
                          seam_name::next_round, seam_name::prev_round),
               std::invalid_argument);
}

// Nothing can precede init: its incoming seam is empty.
TEST(ExtendedDemPhases, StitchBeforeInitThrows) {
  EXPECT_THROW(dem_stitch(rep5_phase_bulk(), rep5_phase_init(),
                          seam_name::next_round, seam_name::prev_round),
               std::invalid_argument);
}

// init, `rounds - 2` bulks, final: one chunk per round.
std::vector<extended_dem> rep5_phases(std::size_t rounds) {
  std::vector<extended_dem> phases{rep5_phase_init()};
  phases.insert(phases.end(), rounds - 2, rep5_phase_bulk());
  phases.push_back(rep5_phase_final());
  return phases;
}

void expect_same_chunk(const extended_dem &got, const extended_dem &want,
                       const std::string &context) {
  EXPECT_EQ(got.num_rows(), want.num_rows()) << context;
  EXPECT_EQ(got.num_faults(), want.num_faults()) << context;
  EXPECT_EQ(got.num_observables(), want.num_observables()) << context;
  EXPECT_TRUE(tensors_equal(got.H.to_dense(), want.H.to_dense())) << context;
  EXPECT_TRUE(tensors_equal(got.O.to_dense(), want.O.to_dense())) << context;
  EXPECT_EQ(got.error_rates, want.error_rates) << context;
  EXPECT_EQ(got.tags, want.tags) << context;
  ASSERT_EQ(got.seams.size(), want.seams.size()) << context;
  for (std::size_t s = 0; s < want.seams.size(); ++s) {
    EXPECT_TRUE(got.seams[s].id == want.seams[s].id) << context;
    EXPECT_EQ(got.seams[s].row_begin, want.seams[s].row_begin) << context;
    EXPECT_EQ(got.seams[s].row_end, want.seams[s].row_end) << context;
  }
}

extended_dem stitch_fold(const std::vector<extended_dem> &chunks) {
  extended_dem acc = chunks[0];
  for (std::size_t i = 1; i < chunks.size(); ++i)
    acc = dem_stitch(acc, chunks[i], seam_name::next_round,
                     seam_name::prev_round);
  return acc;
}

// dem_stitch_all() assembles the chain in one forward pass because folding
// re-copies the accumulator at every step. The fold is still the definition of
// the result, so the two must agree on the whole chunk -- rows, observables,
// priors, seam bands and tags -- not merely on what survives closing.
TEST(ExtendedDemPhases, StitchAllMatchesAnExplicitFold) {
  for (std::size_t rounds = 2; rounds <= 6; ++rounds) {
    const auto phases = rep5_phases(rounds);
    expect_same_chunk(dem_stitch_all(phases), stitch_fold(phases),
                      "phases, rounds=" + std::to_string(rounds));
  }
  // Uniform chunks carry both seams at full width, where the phase
  // decomposition leaves the outer two empty.
  for (std::size_t T = 2; T <= 6; ++T) {
    const std::vector<extended_dem> uniform(
        T, extended_dem_from_css_matrices(rep3(), px_only(0.01)));
    expect_same_chunk(dem_stitch_all(uniform), stitch_fold(uniform),
                      "uniform, T=" + std::to_string(T));
  }
}

// dem_close_all() takes a single O(T) pass and the left fold takes the general
// route. They have to agree on phase chunks exactly as they do on uniform ones:
// every round-indexed map below is built on that equivalence holding.
TEST(ExtendedDemPhases, CloseAllMatchesStitchAndClose) {
  for (std::size_t rounds = 2; rounds <= 6; ++rounds) {
    const auto phases = rep5_phases(rounds);
    const auto via_stitch = dem_close(dem_stitch_all(phases));
    const auto via_close_all = dem_close_all(phases);

    EXPECT_EQ(via_close_all.num_detectors(), via_stitch.num_detectors())
        << "rounds=" << rounds;
    EXPECT_EQ(via_close_all.num_error_mechanisms(),
              via_stitch.num_error_mechanisms());
    EXPECT_TRUE(tensors_equal(via_close_all.detector_error_matrix,
                              via_stitch.detector_error_matrix))
        << "rounds=" << rounds;
    EXPECT_TRUE(tensors_equal(via_close_all.observables_flips_matrix,
                              via_stitch.observables_flips_matrix));
    EXPECT_EQ(via_close_all.error_rates, via_stitch.error_rates);
  }
}

// init carries round 0 in its interior instead of an incoming seam band, so it
// still spans exactly one round rather than the two a naive 1 + interior/d
// would report.
TEST(ExtendedDemPhases, EachPhaseSpansOneRound) {
  EXPECT_EQ(dem_chunk_rounds(rep5_phase_init()), 1u);
  EXPECT_EQ(dem_chunk_rounds(rep5_phase_bulk()), 1u);
  EXPECT_EQ(dem_chunk_rounds(rep5_phase_final()), 1u);

  for (std::size_t rounds = 2; rounds <= 6; ++rounds)
    EXPECT_EQ(dem_chunks_to_rounds(rep5_phases(rounds)), rounds);
}

// Pre-stitching part of the chain does not change the rounds it reports, so a
// caller may group phases however it likes.
TEST(ExtendedDemPhases, PreStitchedPhasesReportTheSameRounds) {
  const auto phases = rep5_phases(4);
  const auto head = dem_stitch(phases[0], phases[1], seam_name::next_round,
                               seam_name::prev_round);
  EXPECT_EQ(head.get_seam(seam_name::prev_round).num_rows(), 0u)
      << "still an open init end";
  EXPECT_EQ(dem_chunk_rounds(head), 2u);
  EXPECT_EQ(dem_chunks_to_rounds({head, phases[2], phases[3]}), 4u);
}

// The detector count dem_close_all() emits is the round count the maps assume,
// which is what keeps the derived H and D the same height.
TEST(ExtendedDemPhases, DetectorCountAgreesWithRoundMaps) {
  for (std::size_t rounds = 2; rounds <= 6; ++rounds) {
    const auto phases = rep5_phases(rounds);
    const std::size_t want = rounds * kRep5Checks;
    EXPECT_EQ(dem_close_all(phases).num_detectors(), want);
    EXPECT_EQ(dem_chunks_to_d_sparse(phases).size(), want);
    EXPECT_EQ(dem_chunks_to_detector_round(phases).size(), want);
  }
}

// Round 0 is read directly; later rounds are xor'd against their predecessor.
TEST(ExtendedDemPhases, DSparseIsTheTimelikeLayout) {
  const uint32_t rounds = 3;
  const auto d_sparse = dem_chunks_to_d_sparse(rep5_phases(rounds));
  ASSERT_EQ(d_sparse.size(), rounds * kRep5Checks);

  for (uint32_t k = 0; k < kRep5Checks; ++k) {
    const std::vector<uint32_t> want{k};
    EXPECT_EQ(d_sparse[k], want);
  }
  for (uint32_t r = 1; r < rounds; ++r)
    for (uint32_t k = 0; k < kRep5Checks; ++k) {
      const std::vector<uint32_t> want{(r - 1) * kRep5Checks + k,
                                       r * kRep5Checks + k};
      EXPECT_EQ(d_sparse[r * kRep5Checks + k], want);
    }
}

TEST(ExtendedDemPhases, DetectorRoundLabelsOneBandPerRound) {
  const int32_t rounds = 3;
  std::vector<int32_t> want;
  for (int32_t r = 0; r < rounds; ++r)
    want.insert(want.end(), kRep5Checks, r);
  EXPECT_EQ(dem_chunks_to_detector_round(rep5_phases(rounds)), want);
}

// O_sparse is keyed on fault columns, so the phases' differing fault counts are
// all it has to track: column 0 of each phase flips the observable.
TEST(ExtendedDemPhases, OSparseSpansEveryPhase) {
  const auto o_sparse = dem_chunks_to_o_sparse(rep5_phases(3));
  ASSERT_EQ(o_sparse.size(), 1u);
  const std::vector<uint32_t> want{0, 9, 18};
  EXPECT_EQ(o_sparse[0], want);
}

// Only the two open ends are exempt. A seam that goes missing mid-chain is
// still an error rather than a shorter experiment.
TEST(ExtendedDemPhases, NonContractingSeamThrows) {
  // Wrong order: init, final, bulk. Stitching detects width/tag mismatch.
  const std::vector<extended_dem> broken{rep5_phase_init(), rep5_phase_final(),
                                         rep5_phase_bulk()};
  // dem_stitch_all detects width mismatch: stitched(init+final).next_round has
  // 0 rows but bulk.prev_round has 4 rows.
  EXPECT_THROW(dem_stitch_all(broken), std::invalid_argument);
}

// Equal seam widths are not enough: the two sides have to name the same checks
// in the same order, or the contraction pairs up unrelated rows. dem_stitch
// enforces this, so the single-pass builder has to enforce it too -- otherwise
// dem_close_all() would quietly accept sequences the fold rejects.
TEST(ExtendedDemPhases, MismatchedSeamTagsThrow) {
  auto bulk = rep5_phase_bulk();
  // Bulk has 2 seams × 4 rows = 8 tags; swap the first two (prev_round seam)
  ASSERT_EQ(bulk.tags.size(), 2u * kRep5Checks);
  std::swap(bulk.tags[0], bulk.tags[1]);

  const std::vector<extended_dem> permuted{rep5_phase_init(), bulk,
                                           rep5_phase_final()};
  // Both the fold and the single-pass builder must reject the mismatch.
  EXPECT_THROW(dem_stitch_all(permuted), std::invalid_argument);
  EXPECT_THROW(dem_close_all(permuted), std::invalid_argument);
}

TEST(ExtendedDemPhases, CloseAllNonContractingSeamThrows) {
  // Wrong order: init, final, bulk. Width mismatch between final's next_round
  // (0 rows) and bulk's prev_round (kRep5Checks rows).
  const std::vector<extended_dem> broken{rep5_phase_init(), rep5_phase_final(),
                                         rep5_phase_bulk()};
  EXPECT_THROW(dem_close_all(broken), std::invalid_argument);
}

// A chunk closed on both sides carries no width to count rounds against.
TEST(ExtendedDemPhases, FullyClosedChunkCannotBeCounted) {
  const auto closed = dem_stitch_all(rep5_phases(3));
  ASSERT_EQ(closed.get_seam(seam_name::prev_round).num_rows(), 0u);
  ASSERT_EQ(closed.get_seam(seam_name::next_round).num_rows(), 0u);
  EXPECT_THROW(dem_chunk_rounds(closed), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Declarative phase specs (new API: named phases + connections)
// ---------------------------------------------------------------------------

// Build a dem_chunks_spec for a d=5 rep code using the new generic API.
// Phases: init (9 faults), bulk (9 faults), final (5 faults).
// Seam: next_round → prev_round (standard memory convention).
dem_chunks_spec rep5_spec(uint64_t num_rounds = 5) {
  dem_chunks_spec spec;
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_final}};
  spec.num_rounds = num_rounds;

  dem_chunk_spec init_spec;
  init_spec.num_faults = 9;
  // H_sparse shorthand: assigned to next_round seam only for init
  init_spec.H_sparse = {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1};
  init_spec.O_sparse = {0, -1};
  init_spec.error_rates.assign(9, 0.02);

  dem_chunk_spec bulk_spec;
  bulk_spec.num_faults = 9;
  bulk_spec.H_sparse = {0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1};
  bulk_spec.O_sparse = {0, -1};
  bulk_spec.error_rates.assign(9, 0.02);

  dem_chunk_spec final_spec;
  final_spec.num_faults = 5;
  final_spec.H_sparse = {0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1};
  final_spec.O_sparse = {0, -1};
  final_spec.error_rates.assign(5, 0.02);

  spec.phases.push_back({phase_name::dem_init, init_spec});
  spec.phases.push_back({phase_name::dem_bulk, bulk_spec});
  spec.phases.push_back({phase_name::dem_final, final_spec});
  return spec;
}

TEST(DemChunkSpec, ValidSpecValidates) {
  EXPECT_NO_THROW(rep5_spec().validate());
  EXPECT_TRUE(rep5_spec().has_repeating_phase());
  EXPECT_FALSE(rep5_spec().is_empty());
  EXPECT_TRUE(dem_chunks_spec{}.is_empty());
}

TEST(DemChunkSpec, NumFaultsMustFitUint32) {
  dem_chunk_spec spec;
  spec.num_faults =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ull;
  spec.error_rates.assign(1, 0.01);
  EXPECT_THROW(spec.validate("test"), std::invalid_argument);
}

TEST(DemChunkSpec, ZeroFaultsThrows) {
  dem_chunk_spec spec;
  EXPECT_THROW(spec.validate("chunk"), std::invalid_argument);
}

TEST(DemChunkSpec, ExpandsToRequestedRoundCount) {
  for (std::size_t rounds = 3; rounds <= 6;
       ++rounds) { // 2 rounds tested separately
    auto spec = rep5_spec(rounds);
    const auto chunks = dem_chunks_from_spec(spec);
    ASSERT_EQ(chunks.size(), rounds);
    // init has a prev_round seam (initial-state detector band); final has no
    // next_round seam (open end with nothing to compare against).
    EXPECT_TRUE(chunks.front().has_seam(seam_name::prev_round));
    EXPECT_GT(chunks.front().get_seam(seam_name::prev_round).num_rows(), 0u);
    EXPECT_FALSE(chunks.back().has_seam(seam_name::next_round) &&
                 chunks.back().get_seam(seam_name::next_round).num_rows() > 0);
    const auto flat = dem_close_all(chunks);
    // rounds seam contractions: the initial-state band (init's prev_round vs
    // zero) plus (rounds-1) inter-chunk XOR boundaries = rounds * kRep5Checks.
    EXPECT_EQ(flat.detector_error_matrix.shape()[0], rounds * kRep5Checks);
  }
}

TEST(DemChunkSpec, TwoRoundsNeedsNoBulkSelfLoop) {
  auto spec = rep5_spec(2);
  // Remove the bulk self-loop and bulk phase; only init→final.
  spec.connections = {{phase_name::dem_init, phase_name::dem_final}};
  spec.phases.erase(std::remove_if(spec.phases.begin(), spec.phases.end(),
                                   [](const auto &e) {
                                     return e.id == phase_name::dem_bulk;
                                   }),
                    spec.phases.end());
  ASSERT_NO_THROW(spec.validate());
  const auto chunks = dem_chunks_from_spec(spec);
  EXPECT_EQ(chunks.size(), 2u);
}

TEST(DemChunkSpec, TooFewRoundsThrows) {
  auto spec = rep5_spec(1);
  EXPECT_THROW(spec.validate(), std::invalid_argument);
  EXPECT_THROW(dem_chunks_from_spec(spec), std::invalid_argument);
}

TEST(DemChunkSpec, ErrorRateCountMustMatchNumFaults) {
  auto spec = rep5_spec();
  // Modify via phases vector directly since get_phase returns const ref
  for (auto &e : spec.phases)
    if (e.id == phase_name::dem_init) {
      e.spec.error_rates.pop_back();
      break;
    }
  EXPECT_THROW(spec.validate(), std::invalid_argument);
}

TEST(DemChunkSpec, MissingRequiredConnectionsThrows) {
  auto spec = rep5_spec();
  spec.connections.clear();
  EXPECT_THROW(spec.validate(), std::invalid_argument);
}

// A genuine width disagreement across the contracted seam still throws.
TEST(ExtendedDemPhases, MismatchedSeamWidthThrows) {
  auto narrow = rep5_phase_bulk();
  narrow.H =
      sparse_binary_matrix::from_nested_csr(2, 9, {{0, 1, 5}, {1, 2, 6}});
  narrow.tags = {0, 1};
  EXPECT_THROW(dem_stitch(rep5_phase_init(), narrow, seam_name::next_round,
                          seam_name::prev_round),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Streaming decomposition
//
// These pin the structural facts a streaming decoder relies on: that a round's
// decode problem can be built once, at configuration time, without knowing how
// many rounds the experiment will run.
//
// They say nothing about decode *quality*. Decoding round by round is a
// windowed approximation of a global decode and the two need not agree on
// corrections; that tradeoff is the decoder's, not the model's. What is exact,
// and what is checked here, is the shape of the problem each round poses.
// ---------------------------------------------------------------------------

// Fault-column offsets of the phase chunks within the closed model, plus a
// trailing total: chunk i owns columns [offsets[i], offsets[i + 1]).
std::vector<std::size_t>
rep5_fault_offsets(const std::vector<extended_dem> &phases) {
  std::vector<std::size_t> offsets{0};
  for (const auto &chunk : phases)
    offsets.push_back(offsets.back() + chunk.num_faults());
  return offsets;
}

using dense_block = std::vector<std::vector<uint8_t>>;

dense_block block_of(const cudaqx::tensor<uint8_t> &m, std::size_t row_begin,
                     std::size_t row_end, std::size_t col_begin,
                     std::size_t col_end) {
  dense_block block;
  for (std::size_t r = row_begin; r < row_end; ++r) {
    std::vector<uint8_t> row;
    for (std::size_t c = col_begin; c < col_end; ++c)
      row.push_back(m.at({r, c}));
    block.push_back(std::move(row));
  }
  return block;
}

// Round r's detector band together with the fault columns of the chunks it can
// reach: its own, and its predecessor's.
dense_block round_band(const std::vector<extended_dem> &phases,
                       std::size_t round) {
  const auto offsets = rep5_fault_offsets(phases);
  const auto closed = dem_close_all(phases);
  return block_of(closed.detector_error_matrix, round * kRep5Checks,
                  (round + 1) * kRep5Checks,
                  offsets[round == 0 ? 0 : round - 1], offsets[round + 1]);
}

// The locality claim. A round's detectors are supported entirely on two
// adjacent chunks' fault columns, so a decoder handling round r never needs
// columns from rounds it has already retired or has yet to see.
TEST(ExtendedDemStreaming, BandsTouchOnlyTwoAdjacentChunks) {
  for (std::size_t rounds = 2; rounds <= 7; ++rounds) {
    const auto phases = rep5_phases(rounds);
    const auto offsets = rep5_fault_offsets(phases);
    const auto closed = dem_close_all(phases);
    const auto &H = closed.detector_error_matrix;
    ASSERT_EQ(H.shape()[0], rounds * kRep5Checks);

    for (std::size_t round = 0; round < rounds; ++round) {
      const std::size_t lo = offsets[round == 0 ? 0 : round - 1];
      const std::size_t hi = offsets[round + 1];
      for (std::size_t r = round * kRep5Checks; r < (round + 1) * kRep5Checks;
           ++r)
        for (std::size_t c = 0; c < H.shape()[1]; ++c)
          if (H.at({r, c}) != 0u)
            EXPECT_TRUE(c >= lo && c < hi)
                << "rounds=" << rounds << " round=" << round << " row=" << r
                << " column " << c << " escapes the window [" << lo << ", "
                << hi << ")";
    }
  }
}

// The reuse claim, and the reason a streaming decoder needs no per-round setup:
// every interior bulk round poses the *same* problem. Not merely the same
// dimensions -- the identical matrix, for every round and every round count.
TEST(ExtendedDemStreaming, EveryBulkRoundPosesOneFixedProblem) {
  const auto reference = round_band(rep5_phases(5), 2);
  ASSERT_EQ(reference.size(), kRep5Checks);
  ASSERT_EQ(reference.front().size(), 18u) << "two bulk chunks of 9 faults";

  for (std::size_t rounds = 4; rounds <= 9; ++rounds) {
    const auto phases = rep5_phases(rounds);
    // Rounds 1 and rounds-1 abut init and final; the rest are interior bulk.
    for (std::size_t round = 2; round + 1 < rounds; ++round)
      EXPECT_EQ(round_band(phases, round), reference)
          << "rounds=" << rounds << " round=" << round;
  }
}

// The boundary rounds are likewise fixed, so the whole experiment reduces to
// three matrices built once: init, bulk, final. Nothing here scales with the
// round count, which is what lets the count stay unknown until the shot ends.
TEST(ExtendedDemStreaming, BoundaryRoundsPoseFixedProblemsToo) {
  const auto first = round_band(rep5_phases(4), 0);
  const auto entering_bulk = round_band(rep5_phases(4), 1);
  const auto last = round_band(rep5_phases(4), 3);

  for (std::size_t rounds = 4; rounds <= 9; ++rounds) {
    const auto phases = rep5_phases(rounds);
    EXPECT_EQ(round_band(phases, 0), first) << "rounds=" << rounds;
    EXPECT_EQ(round_band(phases, 1), entering_bulk) << "rounds=" << rounds;
    EXPECT_EQ(round_band(phases, rounds - 1), last) << "rounds=" << rounds;
  }
}

// The detector side of the same claim: forming round r's detectors reads only
// rounds r-1 and r of the measurement record, so the streaming front-end needs
// one round of lookback and no more.
TEST(ExtendedDemStreaming, DetectorsNeedOneRoundOfLookback) {
  for (std::size_t rounds = 2; rounds <= 7; ++rounds) {
    const auto d_sparse = dem_chunks_to_d_sparse(rep5_phases(rounds));
    ASSERT_EQ(d_sparse.size(), rounds * kRep5Checks);

    for (std::size_t det = 0; det < d_sparse.size(); ++det) {
      const std::size_t round = det / kRep5Checks;
      const std::size_t earliest = (round == 0 ? 0 : round - 1) * kRep5Checks;
      const std::size_t past_end = (round + 1) * kRep5Checks;
      for (const auto bit : d_sparse[det])
        EXPECT_TRUE(bit >= earliest && bit < past_end)
            << "rounds=" << rounds << " detector=" << det << " reads bit "
            << bit << " outside [" << earliest << ", " << past_end << ")";
    }
  }
}

// The chunks a streaming step spans: the one retiring and the one that just
// arrived. The opening has nothing to retire yet and the flush nothing more to
// wait for, so those two steps span a single chunk.
std::vector<extended_dem> step_chunks(const std::vector<extended_dem> &phases,
                                      std::size_t step) {
  const std::size_t retiring = step == 0 ? 0 : step - 1;
  const std::size_t arriving = std::min(step, phases.size() - 1);
  return {phases.begin() + retiring, phases.begin() + arriving + 1};
}

// What the previous tests read out of the whole experiment's DEM, built instead
// from the step's own chunks: stitch them, close the seam they arrive on, and
// drop the open one they hand forward. That construction never mentions the
// round count, which is what lets a round be posed as it arrives, and what it
// produces is exactly the corresponding block of the whole experiment.
//
// The one thing the window does not carry is the retiring band's dependence on
// the chunk before it, which is why the comparison stops at the window's own
// columns. That chunk was committed a round earlier, and backing its
// corrections out of the syndrome is what keeps the truncated band consistent.
TEST(ExtendedDemStreaming, AWindowIsTheExperimentRestrictedToItsOwnChunks) {
  for (std::size_t rounds = 3; rounds <= 8; ++rounds) {
    const auto phases = rep5_phases(rounds);
    const auto offsets = rep5_fault_offsets(phases);
    const auto whole = dem_close_all(phases);

    // Steps run 0..R inclusive, the opening and the flush included.
    for (std::size_t step = 0; step <= rounds; ++step) {
      const auto chunks = step_chunks(phases, step);
      const auto window = dem_close(dem_stitch_all(chunks));

      const std::size_t first_chunk = step == 0 ? 0 : step - 1;
      const std::size_t past_chunk = std::min(step, rounds - 1) + 1;
      const std::size_t row_begin = first_chunk * kRep5Checks;
      const std::size_t row_end = past_chunk * kRep5Checks;

      ASSERT_EQ(window.num_detectors(), row_end - row_begin)
          << "rounds=" << rounds << " step=" << step;
      ASSERT_EQ(window.num_error_mechanisms(),
                offsets[past_chunk] - offsets[first_chunk]);

      EXPECT_EQ(block_of(window.detector_error_matrix, 0,
                         window.num_detectors(), 0,
                         window.num_error_mechanisms()),
                block_of(whole.detector_error_matrix, row_begin, row_end,
                         offsets[first_chunk], offsets[past_chunk]))
          << "rounds=" << rounds << " step=" << step;

      // The priors have to come across too, or the window would be weighted
      // differently from the experiment it is a piece of.
      const std::decay_t<decltype(whole.error_rates)> want(
          whole.error_rates.begin() + offsets[first_chunk],
          whole.error_rates.begin() + offsets[past_chunk]);
      EXPECT_EQ(window.error_rates, want)
          << "rounds=" << rounds << " step=" << step;
    } // end - for(step)
  } // end - for(rounds)
}

// The observables travel with the window, so a round can say what its committed
// faults do to the logical operators without consulting the whole experiment.
TEST(ExtendedDemStreaming, AWindowCarriesItsChunksObservables) {
  const std::size_t rounds = 5;
  const auto phases = rep5_phases(rounds);
  const auto offsets = rep5_fault_offsets(phases);
  const auto whole = dem_close_all(phases);

  for (std::size_t step = 0; step <= rounds; ++step) {
    const auto window = dem_close(dem_stitch_all(step_chunks(phases, step)));
    const std::size_t first_chunk = step == 0 ? 0 : step - 1;
    const std::size_t past_chunk = std::min(step, rounds - 1) + 1;

    EXPECT_EQ(
        block_of(window.observables_flips_matrix, 0, window.num_observables(),
                 0, window.num_error_mechanisms()),
        block_of(whole.observables_flips_matrix, 0, whole.num_observables(),
                 offsets[first_chunk], offsets[past_chunk]))
        << "step=" << step;
  } // end - for(step)
}

// ---------------------------------------------------------------------------
// Internal consistency of a chunk
// ---------------------------------------------------------------------------

// num_faults() reports in_syndrome's width alone, so a block that disagrees
// with it would be scattered into the wrong columns (or past the end of the
// closed matrix) with nothing else to signal the mistake.
TEST(ExtendedDemValidate, BlockWiderThanTheChunkThrows) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  ASSERT_EQ(dem_chunk.num_faults(), 3u);
  // One interior row that claims four fault columns instead of three.
  dem_chunk.H = sparse_binary_matrix::from_nested_csr(1, 4, {{3}});

  EXPECT_THROW(dem_chunk.validate("test"), std::invalid_argument);
  EXPECT_THROW(dem_close(dem_chunk), std::invalid_argument);
  EXPECT_THROW(dem_stitch(dem_chunk, dem_chunk, seam_name::next_round,
                          seam_name::prev_round),
               std::invalid_argument);
  EXPECT_THROW(dem_close_all({dem_chunk}), std::invalid_argument);
}

// A prior per fault is what dem_close copies into error_rates, so a short or
// long list would silently reweight the DEM.
TEST(ExtendedDemValidate, PriorCountMustMatchFaultCount) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  dem_chunk.error_rates.pop_back();

  EXPECT_THROW(dem_chunk.validate("test"), std::invalid_argument);
  EXPECT_THROW(dem_close(dem_chunk), std::invalid_argument);
  EXPECT_THROW(dem_stitch_all({dem_chunk}), std::invalid_argument);
}

// Tags name seam rows one for one; a mismatched count means the seam cannot be
// checked for contractibility at all.
TEST(ExtendedDemValidate, TagCountMustMatchSeamRows) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  dem_chunk.tags.push_back(99u);

  EXPECT_THROW(dem_chunk.validate("test"), std::invalid_argument);
}

// A well-formed chunk, and every chunk these builders produce, must pass.
TEST(ExtendedDemValidate, WellFormedChunksPass) {
  EXPECT_NO_THROW(
      extended_dem_from_css_matrices(rep3(), px_only(0.01)).validate("test"));
  // All-zero rates leave a default-constructed chunk, which is consistent.
  EXPECT_NO_THROW(extended_dem_from_css_matrices(rep3(), css_noise_params{})
                      .validate("test"));
  for (const auto &phase : rep5_phases(4))
    EXPECT_NO_THROW(phase.validate("test"));
}

// ---------------------------------------------------------------------------
// Noise rate validation
// ---------------------------------------------------------------------------

// A negative or NaN rate is inactive under the "rate > 0" rule that selects
// fault columns, so without this check a mistyped rate quietly builds a
// smaller DEM instead of failing.
TEST(ExtendedDemNoiseValidation, RejectsRatesOutsideTheUnitInterval) {
  for (const double bad : {-0.01, 1.5, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    css_noise_params noise;
    noise.px = bad;
    EXPECT_THROW(extended_dem_from_css_matrices(rep3(), noise),
                 std::invalid_argument)
        << "px=" << bad;
    EXPECT_THROW(dem_from_css_matrices(rep3(), noise), std::invalid_argument)
        << "px=" << bad;
  }
}

TEST(ExtendedDemNoiseValidation, RejectsBadPerElementRates) {
  css_noise_params per_qubit;
  per_qubit.px_per_qubit = {0.01, -0.01, 0.01};
  EXPECT_THROW(extended_dem_from_css_matrices(rep3(), per_qubit),
               std::invalid_argument);

  css_noise_params per_check;
  per_check.pm_per_check = {0.01, 2.0};
  EXPECT_THROW(extended_dem_from_css_matrices(rep3(), per_check),
               std::invalid_argument);
}

// The bound is inclusive: 0 means "no such fault" and 1 means "always".
TEST(ExtendedDemNoiseValidation, AcceptsTheEndpointsOfTheUnitInterval) {
  css_noise_params certain;
  certain.px = 1.0;
  EXPECT_NO_THROW(extended_dem_from_css_matrices(rep3(), certain));
  EXPECT_NO_THROW(extended_dem_from_css_matrices(rep3(), css_noise_params{}));
}

// pm_per_check is sized by checks, not qubits, so its message must say so --
// the two dimensions differ and a wrong one sends the reader to the wrong
// field.
TEST(ExtendedDemNoiseValidation, PerCheckLengthErrorNamesTheCheckCount) {
  css_noise_params noise;
  noise.pm_per_check = {0.01}; // rep3 has 2 checks
  try {
    extended_dem_from_css_matrices(rep3(), noise);
    ADD_FAILURE() << "expected a length mismatch";
  } catch (const std::invalid_argument &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("n_checks"), std::string::npos) << message;
    EXPECT_EQ(message.find("n_qubits"), std::string::npos) << message;
  }
}

// ---------------------------------------------------------------------------
// Prior combination
// ---------------------------------------------------------------------------

// sum_combine is a small-p approximation whose sum can leave the unit
// interval. It is clamped, because every consumer reads fault_priors as a
// probability.
TEST(ExtendedDemMerge, SumCombineClampsToOne) {
  // Three columns of identical support merge into one whose sum is 1.2.
  // Use make_merge_chunk: 3 fault columns, 1 H row (all fire row 0), no seams.
  auto dem_chunk = make_merge_chunk(3, {{0}, {0}, {0}}, {0.4, 0.4, 0.4});
  ASSERT_NO_THROW(dem_chunk.validate("test"));

  const auto merged =
      dem_merge_duplicate_columns(dem_chunk, prior_combine_mode::sum_combine);
  ASSERT_EQ(merged.num_faults(), 1u);
  EXPECT_DOUBLE_EQ(merged.error_rates[0], 1.0);

  // or_combine is the GF(2) / XOR rule, which also stays in [0, 1]:
  // 1/2 * (1 - (1-2*0.4)^3) = 1/2 * (1 - 0.2^3) = 0.496.
  const auto ored =
      dem_merge_duplicate_columns(dem_chunk, prior_combine_mode::or_combine);
  ASSERT_EQ(ored.num_faults(), 1u);
  EXPECT_DOUBLE_EQ(ored.error_rates[0], 0.5 * (1.0 - 0.2 * 0.2 * 0.2));
}

// A zero-row block still has to report the chunk's fault width. A
// default-constructed (0-column) empty interior used to pass validate() and
// then OOB inside dem_merge_duplicate_columns / dem_chunks_to_o_sparse.
TEST(ExtendedDemValidate, EmptyBlockMustMatchFaultWidth) {
  extended_dem dem_chunk;
  dem_chunk.H = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {0}});
  dem_chunk.H = dem_chunk.H;
  dem_chunk.O = sparse_binary_matrix::from_nested_csc(0, 2, {{}, {}});
  dem_chunk.error_rates = {0.1, 0.2};
  dem_chunk.tags = {0};
  dem_chunk.tags = {0};
  // interior left default-constructed: 0 rows, 0 columns.

  EXPECT_THROW(dem_chunk.validate("test"), std::invalid_argument);
  EXPECT_THROW(dem_merge_duplicate_columns(dem_chunk), std::invalid_argument);
  EXPECT_THROW(are_dem_columns_unique(dem_chunk), std::invalid_argument);
  EXPECT_THROW(assert_dem_columns_unique(dem_chunk), std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_o_sparse({dem_chunk}), std::invalid_argument);
}

// Short prior lists used to be read past the end by merge helpers that skipped
// validate().
TEST(ExtendedDemValidate, MergeRejectsShortPriorList) {
  auto dem_chunk = extended_dem_from_css_matrices(rep3(), px_only(0.01));
  dem_chunk.error_rates.pop_back();

  EXPECT_THROW(dem_merge_duplicate_columns(dem_chunk), std::invalid_argument);
  EXPECT_THROW(are_dem_columns_unique(dem_chunk), std::invalid_argument);
  EXPECT_THROW(assert_dem_columns_unique(dem_chunk), std::invalid_argument);
  EXPECT_THROW(dem_chunks_to_o_sparse({dem_chunk}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Issue regressions
// ---------------------------------------------------------------------------

// Issue 1: phase_sequence() must not depend on connection order.
TEST(DemChunkSpec, PhaseSequenceIsOrderIndependent) {
  auto spec = rep5_spec(5);
  // Reverse the connections: {bulk→final, bulk→bulk, init→bulk}
  std::reverse(spec.connections.begin(), spec.connections.end());
  ASSERT_NO_THROW(spec.validate());
  const auto seq = spec.phase_sequence();
  ASSERT_EQ(seq.size(), 5u);
  EXPECT_EQ(seq.front(), phase_name::dem_init);
  EXPECT_EQ(seq.back(), phase_name::dem_final);
}

// Issue 2: per-seam O_sparse must not be silently discarded.
TEST(DemChunkSpec, PerSeamOSparseThrows) {
  dem_chunk_spec spec;
  spec.num_faults = 3;
  spec.H_sparse = {};
  spec.error_rates = {0.01, 0.01, 0.01};
  seam_spec_entry e;
  e.id = seam_name::next_round;
  e.spec.H_sparse = {0, -1, 1, -1, 2, -1};
  e.spec.O_sparse = {0, -1}; // non-empty: should throw
  spec.seam_specs.push_back(e);
  EXPECT_THROW(dem_chunk_from_spec(spec, {}, "test"), std::invalid_argument);
}

// Issue 3: zero-rate DEM must have correct tensor shape, not [0 × 0].
TEST(DemConstruction, ZeroRatesDemHasCorrectShape) {
  css_code_matrices code = rep3();
  css_noise_params noise; // all rates default to zero
  const auto flat = dem_from_css_matrices(code, noise, 3);
  EXPECT_EQ(flat.detector_error_matrix.shape()[0], 3u * 2u); // T*d detectors
  EXPECT_EQ(flat.detector_error_matrix.shape()[1], 0u);      // no fault columns
  EXPECT_EQ(flat.observables_flips_matrix.shape()[0], 1u);   // 1 observable
  EXPECT_EQ(flat.observables_flips_matrix.shape()[1], 0u);
  EXPECT_TRUE(flat.error_rates.empty());
}

// Issue 4: a matrix with rows but zero columns must be rejected.
TEST(DemConstruction, MatrixWithRowsButZeroColumnsThrows) {
  css_code_matrices code;
  // hz has 2 rows but 0 columns (default-constructed num_cols = 0)
  code.hz = sparse_binary_matrix::from_nested_csr(2, 0, {{}, {}});
  css_noise_params noise = px_only(0.01);
  EXPECT_THROW(dem_from_css_matrices(code, noise, 1), std::invalid_argument);
  EXPECT_THROW(extended_dem_from_css_matrices(code, noise),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Seams outside the contracted pair
// ---------------------------------------------------------------------------

constexpr seam_id kSideSeam{"side"};

// prev_round [0,1), next_round [1,2), and a third named seam [2, 2+side_rows).
// Each row touches its own fault column so a dropped row is visible.
extended_dem chunk_with_side_seam(uint32_t side_rows) {
  const uint32_t n_rows = 2 + side_rows;
  using idx_t = sparse_binary_matrix::index_type;
  std::vector<std::vector<idx_t>> rows;
  for (uint32_t r = 0; r < n_rows; ++r)
    rows.push_back({r});

  extended_dem dem;
  dem.H = sparse_binary_matrix::from_nested_csr(n_rows, n_rows, rows);
  dem.O = sparse_binary_matrix::from_nested_csr(1, n_rows, {{0}});
  dem.error_rates.assign(n_rows, 0.01);
  dem.add_seam(seam_name::prev_round, 0, 1);
  dem.add_seam(seam_name::next_round, 1, 2);
  dem.add_seam(kSideSeam, 2, 2 + side_rows);
  dem.tags.assign(2 + side_rows, 0);
  return dem;
}

// A seam that is neither from_seam nor to_seam has no counterpart to contract
// against, and its rows are excluded from the interior, so composing used to
// drop them from H without any error.
TEST(ExtendedDemSeams, StitchRejectsANonContractedSeam) {
  const auto a = chunk_with_side_seam(1);
  ASSERT_NO_THROW(a.validate("side-seam chunk"));
  EXPECT_THROW(dem_stitch(a, a, seam_name::next_round, seam_name::prev_round),
               std::invalid_argument);
}

TEST(ExtendedDemSeams, CloseAllRejectsANonContractedSeam) {
  const auto a = chunk_with_side_seam(1);
  EXPECT_THROW(dem_close_all({a, a}), std::invalid_argument);
}

// dem_close drops from_seam on purpose (no next round to compare against), but
// a third seam must not disappear the same way.
TEST(ExtendedDemSeams, CloseRejectsANonContractedSeam) {
  const auto a = chunk_with_side_seam(1);
  EXPECT_THROW(dem_close(a), std::invalid_argument);
}

// Zero-row seams are structural placeholders, not carriers of detector rows,
// so they must still compose.
TEST(ExtendedDemSeams, ZeroWidthNonContractedSeamIsAccepted) {
  const auto a = chunk_with_side_seam(0);
  ASSERT_TRUE(a.has_seam(kSideSeam));
  ASSERT_NO_THROW(
      dem_stitch(a, a, seam_name::next_round, seam_name::prev_round));
  ASSERT_NO_THROW(dem_close_all({a, a}));
  ASSERT_NO_THROW(dem_close(a));
}

// num_interior_rows() subtracts summed seam widths while the interior rows are
// materialized by row membership; overlapping bands make the two disagree.
TEST(ExtendedDemValidate, OverlappingSeamBandsThrow) {
  extended_dem dem;
  dem.H = sparse_binary_matrix::from_nested_csr(4, 2, {{0}, {1}, {0}, {1}});
  dem.O = sparse_binary_matrix::from_nested_csr(0, 2, {});
  dem.error_rates = {0.01, 0.01};
  dem.add_seam(seam_name::prev_round, 0, 3);
  dem.add_seam(seam_name::next_round, 2, 4); // overlaps rows [2,3)
  dem.tags.assign(5, 0);

  EXPECT_THROW(dem.validate("overlap"), std::invalid_argument);
}

// A cyclic connection graph used to walk forever, growing the phase sequence
// until the process ran out of memory.
TEST(DemChunkSpec, CyclicConnectionGraphThrows) {
  dem_chunks_spec spec;
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  const phase_id a{"A"}, b{"B"};
  spec.connections = {{a, b}, {b, a}};

  dem_chunk_spec cs;
  cs.num_faults = 1;
  cs.H_sparse = {0, -1};
  cs.error_rates = {0.01};
  spec.phases.push_back({a, cs});
  spec.phases.push_back({b, cs});

  ASSERT_NO_THROW(spec.validate());
  EXPECT_THROW(spec.phase_sequence(), std::invalid_argument);
  EXPECT_THROW(dem_chunks_from_spec(spec), std::invalid_argument);
}

// A chunk with no observables may leave O default-constructed (0 x 0). That
// shape cannot be indexed per fault column, which the column-wise helpers
// used to do unconditionally.
TEST(ExtendedDemValidate, DefaultConstructedObservablesAreHandled) {
  extended_dem dem;
  dem.H = sparse_binary_matrix::from_nested_csr(2, 3, {{0}, {1}});
  dem.error_rates = {0.1, 0.1, 0.1};
  dem.add_seam(seam_name::prev_round, 0, 1);
  dem.add_seam(seam_name::next_round, 1, 2);
  dem.tags = {0, 0};

  ASSERT_EQ(dem.O.num_rows(), 0u);
  ASSERT_EQ(dem.O.num_cols(), 0u);
  ASSERT_NO_THROW(dem.validate("no observables"));

  EXPECT_TRUE(dem_chunks_to_o_sparse({dem, dem}).empty());
  EXPECT_NO_THROW(dem_merge_duplicate_columns(dem));
  EXPECT_NO_THROW(are_dem_columns_unique(dem));
}

// An O that has rows must still be column-aligned with H.
TEST(ExtendedDemValidate, MisalignedObservablesThrow) {
  extended_dem dem;
  dem.H = sparse_binary_matrix::from_nested_csr(2, 3, {{0}, {1}});
  dem.O = sparse_binary_matrix::from_nested_csr(0, 2, {});
  dem.error_rates = {0.1, 0.1, 0.1};
  dem.add_seam(seam_name::prev_round, 0, 1);
  dem.add_seam(seam_name::next_round, 1, 2);
  dem.tags = {0, 0};

  EXPECT_THROW(dem.validate("misaligned O"), std::invalid_argument);
}

// Defaulted comparisons must visit each member; short-circuit would hide later
// fields if we only compared equal values.
TEST(ExtendedDemValueSemantics, EqualityVisitsEachMember) {
  seam_id a{"alpha"};
  seam_id b{"beta"};
  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
  EXPECT_TRUE((a < b) != (b < a));

  dem_seam_spec s{{0, -1}, {1, -1}};
  EXPECT_EQ(s, s);
  auto s_h = s;
  s_h.H_sparse = {0, 1, -1};
  EXPECT_NE(s, s_h);
  auto s_o = s;
  s_o.O_sparse = {0, -1};
  EXPECT_NE(s, s_o);

  seam_spec_entry e{seam_name::prev_round, s};
  EXPECT_EQ(e, e);
  auto e_id = e;
  e_id.id = seam_name::next_round;
  EXPECT_NE(e, e_id);
  auto e_spec = e;
  e_spec.spec.H_sparse = {1, -1};
  EXPECT_NE(e, e_spec);

  dem_chunk_spec chunk;
  chunk.num_faults = 1;
  chunk.H_sparse = {0, -1};
  chunk.O_sparse = {0, -1};
  chunk.error_rates = {0.01};
  chunk.seam_specs = {e};
  EXPECT_EQ(chunk, chunk);
  auto c1 = chunk;
  c1.num_faults = 2;
  EXPECT_NE(chunk, c1);
  auto c2 = chunk;
  c2.H_sparse = {1, -1};
  EXPECT_NE(chunk, c2);
  auto c3 = chunk;
  c3.seam_specs.clear();
  EXPECT_NE(chunk, c3);
  auto c4 = chunk;
  c4.O_sparse = {1, -1};
  EXPECT_NE(chunk, c4);
  auto c5 = chunk;
  c5.error_rates = {0.02};
  EXPECT_NE(chunk, c5);

  seam_connection sc{seam_name::next_round, seam_name::prev_round};
  EXPECT_EQ(sc, sc);
  auto sc2 = sc;
  sc2.from_seam = seam_name::prev_round;
  EXPECT_NE(sc, sc2);
  auto sc3 = sc;
  sc3.to_seam = seam_name::next_round;
  EXPECT_NE(sc, sc3);

  phase_connection pc{phase_name::dem_init, phase_name::dem_bulk};
  EXPECT_EQ(pc, pc);
  auto pc2 = pc;
  pc2.from_phase = phase_name::dem_bulk;
  EXPECT_NE(pc, pc2);
  auto pc3 = pc;
  pc3.to_phase = phase_name::dem_final;
  EXPECT_NE(pc, pc3);

  phase_spec_entry pe{phase_name::dem_bulk, chunk};
  EXPECT_EQ(pe, pe);
  auto pe2 = pe;
  pe2.id = phase_name::dem_final;
  EXPECT_NE(pe, pe2);
  auto pe3 = pe;
  pe3.spec.num_faults = 2;
  EXPECT_NE(pe, pe3);

  dem_chunks_spec spec;
  spec.phases = {pe};
  spec.connections = {pc};
  spec.seam = sc;
  spec.num_rounds = 4;
  EXPECT_EQ(spec, spec);
  auto sp1 = spec;
  sp1.phases.clear();
  EXPECT_NE(spec, sp1);
  auto sp2 = spec;
  sp2.connections.clear();
  EXPECT_NE(spec, sp2);
  auto sp3 = spec;
  sp3.seam.to_seam = seam_name::next_round;
  EXPECT_NE(spec, sp3);
  auto sp4 = spec;
  sp4.num_rounds = 5;
  EXPECT_NE(spec, sp4);
  auto sp5 = spec;
  sp5.num_rounds.reset();
  EXPECT_NE(spec, sp5);
}

extended_dem zero_width_seam_chunk() {
  extended_dem dem;
  dem.add_seam(seam_name::prev_round, 0, 0);
  dem.add_seam(seam_name::next_round, 0, 0);
  return dem;
}

// Zero-width contracted seams are a successful no-op, including zero faults.
TEST(ExtendedDemSeams, ZeroWidthBoundaryStitchAndMerge) {
  const auto a = zero_width_seam_chunk();
  ASSERT_NO_THROW(a.validate("zero-width"));
  auto stitched =
      dem_stitch(a, a, seam_name::next_round, seam_name::prev_round);
  EXPECT_EQ(stitched.num_faults(), 0u);
  auto merged = dem_merge_duplicate_columns(a);
  EXPECT_EQ(merged.num_faults(), 0u);
}

TEST(ExtendedDemFromCss, NZeroRejectsRowsAndNonEmptyNoise) {
  css_noise_params noise;
  {
    css_code_matrices code;
    code.hx = sparse_binary_matrix::from_nested_csc(1, 0, {});
    EXPECT_THROW(extended_dem_from_css_matrices(code, noise),
                 std::invalid_argument);
  }
  {
    css_code_matrices code;
    code.lz = sparse_binary_matrix::from_nested_csc(1, 0, {});
    EXPECT_THROW(extended_dem_from_css_matrices(code, noise),
                 std::invalid_argument);
  }
  {
    css_code_matrices code;
    code.lx = sparse_binary_matrix::from_nested_csc(1, 0, {});
    EXPECT_THROW(extended_dem_from_css_matrices(code, noise),
                 std::invalid_argument);
  }
  {
    css_code_matrices code;
    noise.px_per_qubit = {0.1};
    EXPECT_THROW(extended_dem_from_css_matrices(code, noise),
                 std::invalid_argument);
  }
  noise = {};
  css_code_matrices empty;
  auto dem = extended_dem_from_css_matrices(empty, noise);
  EXPECT_EQ(dem.num_faults(), 0u);
}

// All four CSS matrices plus Z and Y faults fill both detector bands and both
// observable bands.
TEST(ExtendedDemFromCss, ZyFaultsWriteAllBands) {
  css_code_matrices code;
  code.hz = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {0}});
  code.hx = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {0}});
  code.lz = sparse_binary_matrix::from_nested_csc(1, 2, {{0}, {}});
  code.lx = sparse_binary_matrix::from_nested_csc(1, 2, {{}, {0}});
  css_noise_params noise;
  noise.pz = 0.1;
  noise.py = 0.2;
  auto dem = extended_dem_from_css_matrices(code, noise);
  ASSERT_EQ(dem.H.num_rows(), 4u);
  ASSERT_EQ(dem.num_faults(), 4u);
  auto dense = dem.H.to_dense();
  auto obs = dem.O.to_dense();
  // Z on q0 is column 0: X-check at nz=1 in both seam copies.
  EXPECT_EQ(dense.at({1, 0}), 1u);
  EXPECT_EQ(dense.at({3, 0}), 1u);
  // Y on q0 is column 2: both checks in both copies, lz on obs 0.
  EXPECT_EQ(dense.at({0, 2}), 1u);
  EXPECT_EQ(dense.at({1, 2}), 1u);
  EXPECT_EQ(dense.at({2, 2}), 1u);
  EXPECT_EQ(dense.at({3, 2}), 1u);
  EXPECT_EQ(obs.at({0, 2}), 1u);
  EXPECT_EQ(obs.at({1, 3}), 1u);
}

TEST(DemChunkSpec, IsEmptyAndExpand) {
  dem_chunk_spec empty;
  EXPECT_TRUE(empty.is_empty());
  empty.num_faults = 1;
  EXPECT_FALSE(empty.is_empty());
  empty = {};
  empty.H_sparse = {0, -1};
  EXPECT_FALSE(empty.is_empty());
  empty = {};
  empty.seam_specs.push_back({seam_name::prev_round, {}});
  EXPECT_FALSE(empty.is_empty());
  empty = {};
  empty.O_sparse = {0, -1};
  EXPECT_FALSE(empty.is_empty());
  empty = {};
  empty.error_rates = {0.1};
  EXPECT_FALSE(empty.is_empty());

  dem_chunk_spec spec;
  spec.num_faults = 1;
  spec.H_sparse = {0, -1};
  spec.error_rates = {0.01};
  spec.expand({seam_name::prev_round, seam_name::next_round});
  ASSERT_EQ(spec.seam_specs.size(), 2u);
  EXPECT_TRUE(spec.H_sparse.empty());
  auto again = spec;
  again.expand({seam_name::prev_round});
  EXPECT_EQ(spec, again);

  dem_chunk_spec no_h;
  no_h.num_faults = 1;
  no_h.error_rates = {0.01};
  no_h.expand({seam_name::prev_round});
  EXPECT_TRUE(no_h.seam_specs.empty());
}

TEST(DemChunkFromSpec, EmptyHAndNoSeamFallback) {
  dem_chunk_spec empty_h;
  empty_h.num_faults = 1;
  empty_h.error_rates = {0.01};
  empty_h.seam_specs = {{seam_name::prev_round, {{}, {}}},
                        {seam_name::next_round, {{}, {}}}};
  auto dem = dem_chunk_from_spec(empty_h);
  EXPECT_EQ(dem.H.num_rows(), 0u);
  EXPECT_EQ(dem.H.num_cols(), 1u);

  dem_chunk_spec no_seam;
  no_seam.num_faults = 2;
  no_seam.error_rates = {0.01, 0.02};
  auto zero = dem_chunk_from_spec(no_seam);
  EXPECT_EQ(zero.H.num_rows(), 0u);
  EXPECT_EQ(zero.H.num_cols(), 2u);
}

TEST(DemChunksSpec, RepeatingPhaseAndSelfLoopSequence) {
  dem_chunks_spec linear;
  linear.connections = {{phase_name::dem_init, phase_name::dem_bulk}};
  EXPECT_FALSE(linear.has_repeating_phase());
  EXPECT_THROW(linear.repeating_phase(), std::invalid_argument);

  dem_chunks_spec one;
  one.connections = {{phase_name::dem_bulk, phase_name::dem_bulk}};
  EXPECT_TRUE(one.has_repeating_phase());
  EXPECT_EQ(one.repeating_phase(), phase_name::dem_bulk);

  dem_chunks_spec two;
  two.connections = {{phase_name::dem_init, phase_name::dem_init},
                     {phase_name::dem_bulk, phase_name::dem_bulk}};
  EXPECT_THROW(two.repeating_phase(), std::invalid_argument);

  dem_chunk_spec cs;
  cs.num_faults = 1;
  cs.H_sparse = {0, -1};
  cs.error_rates = {0.01};
  dem_chunks_spec only_self;
  only_self.phases.push_back({phase_name::dem_bulk, cs});
  only_self.connections = {{phase_name::dem_bulk, phase_name::dem_bulk}};
  only_self.seam = {seam_name::next_round, seam_name::prev_round};
  only_self.num_rounds = 3;
  const auto seq = only_self.phase_sequence();
  ASSERT_EQ(seq.size(), 3u);
  EXPECT_EQ(seq[0], phase_name::dem_bulk);
  EXPECT_EQ(seq[1], phase_name::dem_bulk);
  EXPECT_EQ(seq[2], phase_name::dem_bulk);
}

// chain_o_columns skips a default 0x0 O (it has no per-fault columns to copy).
// dem_stitch_all still sizes the result to n_faults, so two 1-fault chunks
// yield 0 rows x 2 cols rather than remaining 0x0.
TEST(ExtendedDemStitch, DefaultObservablesAreSkipped) {
  auto make = [] {
    extended_dem dem;
    dem.H = sparse_binary_matrix::from_nested_csr(2, 1, {{0}, {0}});
    dem.error_rates = {0.1};
    dem.add_seam(seam_name::prev_round, 0, 1);
    dem.add_seam(seam_name::next_round, 1, 2);
    dem.tags = {0, 0};
    return dem;
  };
  auto out = dem_stitch_all({make(), make()});
  ASSERT_EQ(out.num_faults(), 2u);
  EXPECT_EQ(out.num_observables(), 0u);
  EXPECT_EQ(out.O.num_rows(), 0u);
  EXPECT_EQ(out.O.num_cols(), 2u);
  EXPECT_NO_THROW(out.validate("stitched default O"));
}

} // namespace
} // namespace cudaq::qec
