/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Tests for css_matrices_from_code() and the dem_from_css_matrices(code&,...)
// overload. These require the full cudaq-qec library (code objects depend on
// the CUDA-Q framework), so they live in a separate executable from the
// matrix-level tests in test_dem_construction.cpp.

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/dem_construction.h"
#include "cudaq/qec/dem_construction_code.h"
#include "cudaq/qec/detector_error_model.h"
#include <cstdint>
#include <gtest/gtest.h>

namespace cudaq::qec {
namespace {

// css_matrices_from_code() on the d=3 repetition code must return:
//   hz: 2x3 sparse matrix matching H_Z = [[1,1,0],[0,1,1]]
//   hx: 0x3 (no X stabilizers)
//   lz: 1x3 matching L_Z = [[1,0,0]]
//   lx: 1x3 (all zeros for this pure-Z code)
TEST(DemConstructionCode, CssMatricesFromCode_Repetition3_Dimensions) {
  cudaqx::heterogeneous_map opts;
  opts.insert("distance", std::size_t{3});
  auto rep = code::get("repetition", opts);

  auto m = css_matrices_from_code(*rep);

  EXPECT_EQ(m.hz.num_rows(), 2u);
  EXPECT_EQ(m.hz.num_cols(), 3u);
  EXPECT_EQ(m.hx.num_rows(), 0u);
  EXPECT_EQ(m.lz.num_rows(), 1u);
  EXPECT_EQ(m.lz.num_cols(), 3u);
}

// The hz matrix from the repetition code must encode Z0Z1 and Z1Z2.
// hz[:,0] = {0}, hz[:,1] = {0,1}, hz[:,2] = {1}.
TEST(DemConstructionCode, CssMatricesFromCode_Repetition3_HzEntries) {
  cudaqx::heterogeneous_map opts;
  opts.insert("distance", std::size_t{3});
  auto rep = code::get("repetition", opts);

  auto m = css_matrices_from_code(*rep);
  auto hz_dense = m.hz.to_dense();

  EXPECT_EQ(hz_dense.at({0, 0}), 1u); // Z0Z1: qubit 0
  EXPECT_EQ(hz_dense.at({0, 1}), 1u); // Z0Z1: qubit 1
  EXPECT_EQ(hz_dense.at({0, 2}), 0u);
  EXPECT_EQ(hz_dense.at({1, 0}), 0u);
  EXPECT_EQ(hz_dense.at({1, 1}), 1u); // Z1Z2: qubit 1
  EXPECT_EQ(hz_dense.at({1, 2}), 1u); // Z1Z2: qubit 2
}

// The full dem_from_css_matrices(code, noise) overload must produce the same
// DEM as dem_from_css_matrices(css_matrices_from_code(code), noise).
TEST(DemConstructionCode, CodeOverload_MatchesMatrixOverload) {
  cudaqx::heterogeneous_map opts;
  opts.insert("distance", std::size_t{3});
  auto rep = code::get("repetition", opts);

  css_noise_params noise;
  noise.px = 0.01;

  auto from_code = dem_from_css_matrices(*rep, noise);
  auto from_matrices =
      dem_from_css_matrices(css_matrices_from_code(*rep), noise);

  ASSERT_EQ(from_code.num_detectors(), from_matrices.num_detectors());
  ASSERT_EQ(from_code.num_observables(), from_matrices.num_observables());
  ASSERT_EQ(from_code.num_error_mechanisms(),
            from_matrices.num_error_mechanisms());

  const auto &d1 = from_code.detector_error_matrix;
  const auto &d2 = from_matrices.detector_error_matrix;
  for (std::size_t r = 0; r < from_code.num_detectors(); ++r)
    for (std::size_t c = 0; c < from_code.num_error_mechanisms(); ++c)
      EXPECT_EQ(d1.at({r, c}), d2.at({r, c})) << "r=" << r << " c=" << c;

  for (std::size_t i = 0; i < from_code.num_error_mechanisms(); ++i)
    EXPECT_DOUBLE_EQ(from_code.error_rates[i], from_matrices.error_rates[i]);
} // end - CodeOverload_MatchesMatrixOverload

} // namespace
} // namespace cudaq::qec
