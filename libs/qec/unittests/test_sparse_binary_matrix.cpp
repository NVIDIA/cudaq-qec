/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

namespace cudaq::qec {
namespace {

using index_type = sparse_binary_matrix::index_type;

bool dense_pcm_equal(const cudaqx::tensor<std::uint8_t> &a,
                     const cudaqx::tensor<std::uint8_t> &b) {
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

// -----------------------------------------------------------------------------
// Dense <-> sparse_binary_matrix (CSC)
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, DenseToCscToDense_Small) {
  // 3x4 matrix: rows x cols
  std::vector<std::uint8_t> data = {
      1, 0, 1, 0, // row 0
      0, 1, 1, 0, // row 1
      1, 1, 0, 1  // row 2
  };
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.layout(), sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_rows(), 3);
  EXPECT_EQ(sp.num_cols(), 4);
  EXPECT_EQ(sp.num_nnz(), 7);

  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, DenseToCsrToDense_Small) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  EXPECT_EQ(sp.layout(), sparse_binary_matrix_layout::csr);
  EXPECT_EQ(sp.num_rows(), 3);
  EXPECT_EQ(sp.num_cols(), 4);
  EXPECT_EQ(sp.num_nnz(), 7);

  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, FromCscToDense) {
  // 2x3 matrix: col0 has rows {0,1}, col1 has {}, col2 has {1}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> col_ptrs = {0, 2, 2, 3};
  std::vector<index_type> row_indices = {0, 1, 1};

  auto sp =
      sparse_binary_matrix::from_csc(num_rows, num_cols, col_ptrs, row_indices);
  auto dense = sp.to_dense();

  EXPECT_EQ(dense.shape()[0], 2);
  EXPECT_EQ(dense.shape()[1], 3);
  EXPECT_EQ(dense.at({0, 0}), 1);
  EXPECT_EQ(dense.at({1, 0}), 1);
  EXPECT_EQ(dense.at({0, 1}), 0);
  EXPECT_EQ(dense.at({1, 1}), 0);
  EXPECT_EQ(dense.at({0, 2}), 0);
  EXPECT_EQ(dense.at({1, 2}), 1);
}

TEST(SparseBinaryMatrix, FromCsrToDense) {
  // 2x3 matrix: row0 has cols {0}, row1 has cols {0, 2}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> row_ptrs = {0, 1, 3};
  std::vector<index_type> col_indices = {0, 0, 2};

  auto sp =
      sparse_binary_matrix::from_csr(num_rows, num_cols, row_ptrs, col_indices);
  auto dense = sp.to_dense();

  EXPECT_EQ(dense.shape()[0], 2);
  EXPECT_EQ(dense.shape()[1], 3);
  EXPECT_EQ(dense.at({0, 0}), 1);
  EXPECT_EQ(dense.at({0, 1}), 0);
  EXPECT_EQ(dense.at({0, 2}), 0);
  EXPECT_EQ(dense.at({1, 0}), 1);
  EXPECT_EQ(dense.at({1, 1}), 0);
  EXPECT_EQ(dense.at({1, 2}), 1);
}

// -----------------------------------------------------------------------------
// CSC <-> CSR conversion round-trip
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, CscToCsrToCsc_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 1, 1, 0, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({2, 5});
  dense.copy(data.data(), {2, 5});

  sparse_binary_matrix csc(dense, sparse_binary_matrix_layout::csc);
  sparse_binary_matrix csr = csc.to_csr();
  sparse_binary_matrix csc2 = csr.to_csc();

  EXPECT_EQ(csc2.layout(), sparse_binary_matrix_layout::csc);
  auto back = csc2.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, CsrToCscToCsr_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 1, 0, 0, 1, 0, 1, 1};
  cudaqx::tensor<std::uint8_t> dense({2, 4});
  dense.copy(data.data(), {2, 4});

  sparse_binary_matrix csr(dense, sparse_binary_matrix_layout::csr);
  sparse_binary_matrix csc = csr.to_csc();
  sparse_binary_matrix csr2 = csc.to_csr();

  EXPECT_EQ(csr2.layout(), sparse_binary_matrix_layout::csr);
  auto back = csr2.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

// -----------------------------------------------------------------------------
// Edge cases: empty, 1x1, all zeros
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, EmptyMatrix) {
  cudaqx::tensor<std::uint8_t> dense({0, 0});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_rows(), 0);
  EXPECT_EQ(sp.num_cols(), 0);
  EXPECT_EQ(sp.num_nnz(), 0);
  auto back = sp.to_dense();
  EXPECT_EQ(back.shape()[0], 0);
  EXPECT_EQ(back.shape()[1], 0);
}

TEST(SparseBinaryMatrix, SingleElementZero) {
  cudaqx::tensor<std::uint8_t> dense({1, 1});
  dense.at({0, 0}) = 0;
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_nnz(), 0);
  auto back = sp.to_dense();
  EXPECT_EQ(back.at({0, 0}), 0);
}

TEST(SparseBinaryMatrix, SingleElementOne) {
  cudaqx::tensor<std::uint8_t> dense({1, 1});
  dense.at({0, 0}) = 1;
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_nnz(), 1);
  auto back = sp.to_dense();
  EXPECT_EQ(back.at({0, 0}), 1);
}

TEST(SparseBinaryMatrix, Small2x2) {
  std::vector<std::uint8_t> data = {1, 1, 1, 0};
  cudaqx::tensor<std::uint8_t> dense({2, 2});
  dense.copy(data.data(), {2, 2});

  for (auto layout :
       {sparse_binary_matrix_layout::csc, sparse_binary_matrix_layout::csr}) {
    sparse_binary_matrix sp(dense, layout);
    auto back = sp.to_dense();
    EXPECT_TRUE(dense_pcm_equal(dense, back));
  }
}

// -----------------------------------------------------------------------------
// Random PCM via generate_random_pcm
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, RandomPcm_CscRoundTrip) {
  std::mt19937_64 rng(12345);
  auto dense = generate_random_pcm(2, 3, 4, 2, std::move(rng));
  ASSERT_EQ(dense.rank(), 2);

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, RandomPcm_CsrRoundTrip) {
  std::mt19937_64 rng(67890);
  auto dense = generate_random_pcm(3, 2, 3, 2, std::move(rng));
  ASSERT_EQ(dense.rank(), 2);

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, RandomPcm_CscToCsrToDense) {
  std::mt19937_64 rng(42);
  auto dense = generate_random_pcm(2, 4, 3, 2, std::move(rng));
  sparse_binary_matrix csc(dense, sparse_binary_matrix_layout::csc);
  sparse_binary_matrix csr = csc.to_csr();
  auto back = csr.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, RandomPcm_CsrToCscToDense) {
  std::mt19937_64 rng(99);
  auto dense = generate_random_pcm(4, 3, 2, 2, std::move(rng));
  sparse_binary_matrix csr(dense, sparse_binary_matrix_layout::csr);
  sparse_binary_matrix csc = csr.to_csc();
  auto back = csc.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

// -----------------------------------------------------------------------------
// Nested CSC / CSR
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, ToNestedCsc_FromCsc) {
  // 2x3 matrix: col0 rows {0,1}, col1 rows {}, col2 rows {1}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> col_ptrs = {0, 2, 2, 3};
  std::vector<index_type> row_indices = {0, 1, 1};

  auto sp =
      sparse_binary_matrix::from_csc(num_rows, num_cols, col_ptrs, row_indices);
  auto nested = sp.to_nested_csc();

  ASSERT_EQ(nested.size(), 3);
  EXPECT_EQ(nested[0], (std::vector<index_type>{0, 1}));
  EXPECT_TRUE(nested[1].empty());
  EXPECT_EQ(nested[2], (std::vector<index_type>{1}));
}

TEST(SparseBinaryMatrix, ToNestedCsr_FromCsr) {
  // 2x3 matrix: row0 cols {0}, row1 cols {0, 2}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> row_ptrs = {0, 1, 3};
  std::vector<index_type> col_indices = {0, 0, 2};

  auto sp =
      sparse_binary_matrix::from_csr(num_rows, num_cols, row_ptrs, col_indices);
  auto nested = sp.to_nested_csr();

  ASSERT_EQ(nested.size(), 2);
  EXPECT_EQ(nested[0], (std::vector<index_type>{0}));
  EXPECT_EQ(nested[1], (std::vector<index_type>{0, 2}));
}

TEST(SparseBinaryMatrix, ToNestedCsc_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  for (auto layout :
       {sparse_binary_matrix_layout::csc, sparse_binary_matrix_layout::csr}) {
    sparse_binary_matrix sp(dense, layout);
    auto nested = sp.to_nested_csc();
    ASSERT_EQ(nested.size(), sp.num_cols());
    index_type nnz = 0;
    std::vector<index_type> col_ptrs(sp.num_cols() + 1);
    col_ptrs[0] = 0;
    std::vector<index_type> row_indices;
    for (index_type j = 0; j < sp.num_cols(); ++j) {
      nnz += static_cast<index_type>(nested[j].size());
      col_ptrs[j + 1] = col_ptrs[j] + static_cast<index_type>(nested[j].size());
      row_indices.insert(row_indices.end(), nested[j].begin(), nested[j].end());
    }
    EXPECT_EQ(nnz, sp.num_nnz());
    auto sp2 = sparse_binary_matrix::from_csc(sp.num_rows(), sp.num_cols(),
                                              std::move(col_ptrs),
                                              std::move(row_indices));
    EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
  }
}

TEST(SparseBinaryMatrix, ToNestedCsr_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  for (auto layout :
       {sparse_binary_matrix_layout::csc, sparse_binary_matrix_layout::csr}) {
    sparse_binary_matrix sp(dense, layout);
    auto nested = sp.to_nested_csr();
    ASSERT_EQ(nested.size(), sp.num_rows());
    index_type nnz = 0;
    std::vector<index_type> row_ptrs(sp.num_rows() + 1);
    row_ptrs[0] = 0;
    std::vector<index_type> col_indices;
    for (index_type i = 0; i < sp.num_rows(); ++i) {
      nnz += static_cast<index_type>(nested[i].size());
      row_ptrs[i + 1] = row_ptrs[i] + static_cast<index_type>(nested[i].size());
      col_indices.insert(col_indices.end(), nested[i].begin(), nested[i].end());
    }
    EXPECT_EQ(nnz, sp.num_nnz());
    auto sp2 = sparse_binary_matrix::from_csr(sp.num_rows(), sp.num_cols(),
                                              std::move(row_ptrs),
                                              std::move(col_indices));
    EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
  }
}

TEST(SparseBinaryMatrix, FromNestedCsc_MatchesFromCsc) {
  index_type num_rows = 2, num_cols = 3;
  std::vector<std::vector<index_type>> nested = {{0, 1}, {}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csc(num_rows, num_cols, nested);
  EXPECT_EQ(sp.num_rows(), num_rows);
  EXPECT_EQ(sp.num_cols(), num_cols);
  EXPECT_EQ(sp.num_nnz(), 3);
  std::vector<index_type> col_ptrs = {0, 2, 2, 3};
  std::vector<index_type> row_indices = {0, 1, 1};
  auto sp_ref =
      sparse_binary_matrix::from_csc(num_rows, num_cols, col_ptrs, row_indices);
  EXPECT_TRUE(dense_pcm_equal(sp.to_dense(), sp_ref.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsr_MatchesFromCsr) {
  index_type num_rows = 2, num_cols = 3;
  std::vector<std::vector<index_type>> nested = {{0}, {0, 2}};
  auto sp = sparse_binary_matrix::from_nested_csr(num_rows, num_cols, nested);
  EXPECT_EQ(sp.num_rows(), num_rows);
  EXPECT_EQ(sp.num_cols(), num_cols);
  EXPECT_EQ(sp.num_nnz(), 3);
  std::vector<index_type> row_ptrs = {0, 1, 3};
  std::vector<index_type> col_indices = {0, 0, 2};
  auto sp_ref =
      sparse_binary_matrix::from_csr(num_rows, num_cols, row_ptrs, col_indices);
  EXPECT_TRUE(dense_pcm_equal(sp.to_dense(), sp_ref.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsc_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  const auto nested = sp.to_nested_csc();
  auto sp2 = sparse_binary_matrix::from_nested_csc(sp.num_rows(), sp.num_cols(),
                                                   nested);
  EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsr_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  const auto nested = sp.to_nested_csr();
  auto sp2 = sparse_binary_matrix::from_nested_csr(sp.num_rows(), sp.num_cols(),
                                                   nested);
  EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsc_InvalidSizeThrows) {
  std::vector<std::vector<index_type>> nested = {{0}, {1}};
  EXPECT_THROW(sparse_binary_matrix::from_nested_csc(2, 3, nested),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, FromNestedCsr_InvalidSizeThrows) {
  std::vector<std::vector<index_type>> nested = {{0}, {1}, {0}};
  EXPECT_THROW(sparse_binary_matrix::from_nested_csr(2, 2, nested),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparseMatchesDenseSmall) {
  std::size_t seed = 77;
  int weight = 3;
  auto dense = cudaq::qec::generate_random_pcm(/*n_rounds=*/4, /*n_errs=*/5,
                                               /*syn_per_round=*/3, weight,
                                               std::mt19937_64(seed));
  auto sparse = cudaq::qec::generate_random_pcm_sparse(4, 5, 3, weight,
                                                       std::mt19937_64(seed));
  ASSERT_TRUE(dense_pcm_equal(dense, sparse.to_dense()));
}

TEST(SparseBinaryMatrix, FromCsc_InvalidMonotonePtrThrows) {
  sparse_binary_matrix::index_type nr = 2, nc = 2;
  std::vector<sparse_binary_matrix::index_type> bad_ptrs = {0, 3, 1};
  std::vector<sparse_binary_matrix::index_type> row_indices{};
  EXPECT_THROW(sparse_binary_matrix::from_csc(nr, nc, std::move(bad_ptrs),
                                              std::move(row_indices)),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, FromCsc_IndexOutOfRangeThrows) {
  sparse_binary_matrix::index_type nr = 2, nc = 2;
  std::vector<sparse_binary_matrix::index_type> col_ptrs = {0, 1, 1};
  std::vector<sparse_binary_matrix::index_type> row_indices = {
      99}; // >= num_rows
  EXPECT_THROW(sparse_binary_matrix::from_csc(nr, nc, std::move(col_ptrs),
                                              std::move(row_indices)),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, ValidateSortedUniqueIndicesAcceptsCanonicalCsc) {
  std::vector<std::vector<index_type>> nested = {{0, 2}, {}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 3, nested);
  EXPECT_NO_THROW(sp.validate_sorted_unique_indices("test"));
}

TEST(SparseBinaryMatrix, ValidateSortedUniqueIndicesRejectsDuplicateCsc) {
  std::vector<std::vector<index_type>> nested = {{0, 0}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csc(2, 2, nested);
  EXPECT_THROW(sp.validate_sorted_unique_indices("test"),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, ValidateSortedUniqueIndicesRejectsDuplicateCsr) {
  std::vector<std::vector<index_type>> nested = {{0, 0}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csr(2, 2, nested);
  EXPECT_THROW(sp.validate_sorted_unique_indices("test"),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, ValidateSortedUniqueIndicesRejectsUnsortedCsc) {
  std::vector<std::vector<index_type>> nested = {{2, 0}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 2, nested);
  EXPECT_THROW(sp.validate_sorted_unique_indices("test"),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, ValidateSortedUniqueIndicesRejectsUnsortedCsr) {
  std::vector<std::vector<index_type>> nested = {{2, 0}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csr(2, 3, nested);
  EXPECT_THROW(sp.validate_sorted_unique_indices("test"),
               std::invalid_argument);
}

// -----------------------------------------------------------------------------
// sparse_binary_matrix::canonicalize (GF(2) merge)
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, Canonicalize_NoOpOnAlreadyCanonical) {
  // Column 0: {0,1}, column 1: {1,2}. No duplicates → output identical.
  std::vector<std::vector<index_type>> nested = {{0, 1}, {1, 2}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 2, nested);
  auto canon = sp.canonicalize();
  EXPECT_TRUE(dense_pcm_equal(sp.to_dense(), canon.to_dense()));
  EXPECT_EQ(canon.num_nnz(), 4u);
}

TEST(SparseBinaryMatrix, Canonicalize_EvenCountDropsEntry) {
  // Column 0 has row 1 listed twice; GF(2) merge → drop both.
  std::vector<std::vector<index_type>> nested = {{1, 1}, {0}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 2, nested);
  auto canon = sp.canonicalize();
  EXPECT_EQ(canon.num_nnz(), 1u);
  auto canon_nested = canon.to_nested_csc();
  EXPECT_TRUE(canon_nested[0].empty());
  EXPECT_EQ(canon_nested[1], (std::vector<index_type>{0}));
}

TEST(SparseBinaryMatrix, Canonicalize_OddCountKeepsOne) {
  // Row 2 listed three times in column 0 → odd count → one entry survives.
  std::vector<std::vector<index_type>> nested = {{2, 2, 2}, {}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 2, nested);
  auto canon = sp.canonicalize();
  auto canon_nested = canon.to_nested_csc();
  EXPECT_EQ(canon_nested[0], (std::vector<index_type>{2}));
}

TEST(SparseBinaryMatrix, Canonicalize_SortsAscending) {
  // Unsorted, no duplicates → output sorted, same nnz.
  std::vector<std::vector<index_type>> nested = {{2, 0, 1}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 1, nested);
  auto canon = sp.canonicalize();
  EXPECT_EQ(canon.to_nested_csc()[0], (std::vector<index_type>{0, 1, 2}));
}

TEST(SparseBinaryMatrix, Canonicalize_Idempotent) {
  std::vector<std::vector<index_type>> nested = {{2, 0, 2, 1, 1, 1}};
  auto sp = sparse_binary_matrix::from_nested_csc(3, 1, nested);
  auto once = sp.canonicalize();
  auto twice = once.canonicalize();
  EXPECT_TRUE(dense_pcm_equal(once.to_dense(), twice.to_dense()));
  EXPECT_EQ(once.num_nnz(), twice.num_nnz());
}

TEST(SparseBinaryMatrix, CanonicalizePcm_DefaultConstructedIsEmpty) {
  sparse_binary_matrix sp;
  auto canon = sp.canonicalize();
  EXPECT_EQ(canon.num_rows(), 0u);
  EXPECT_EQ(canon.num_cols(), 0u);
  EXPECT_EQ(canon.num_nnz(), 0u);
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparse_ProducesSortedColumns) {
  // Generator must sort per column for downstream .front()/.back() min/max.
  auto sp = cudaq::qec::generate_random_pcm_sparse(/*n_rounds=*/4,
                                                   /*n_errs_per_round=*/8,
                                                   /*syn_per_round=*/5,
                                                   /*weight=*/3,
                                                   std::mt19937_64(31415));
  auto nested = sp.to_nested_csc();
  for (std::size_t j = 0; j < nested.size(); ++j) {
    EXPECT_TRUE(std::is_sorted(nested[j].begin(), nested[j].end()))
        << "column " << j << " is not sorted";
  }
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparse_WeightZeroIsEmpty) {
  // weight == 0 yields an all-zero PCM.
  auto sp =
      cudaq::qec::generate_random_pcm_sparse(/*n_rounds=*/3,
                                             /*n_errs_per_round=*/4,
                                             /*syn_per_round=*/5,
                                             /*weight=*/0, std::mt19937_64(0));
  EXPECT_EQ(sp.num_rows(), 15u);
  EXPECT_EQ(sp.num_cols(), 12u);
  EXPECT_EQ(sp.num_nnz(), 0u);
}

TEST(SparseBinaryMatrix, GenerateRandomPcm_WeightZeroIsEmpty) {
  // Dense path: same corner case.
  auto dense = cudaq::qec::generate_random_pcm(3, 4, 5, /*weight=*/0,
                                               std::mt19937_64(0));
  ASSERT_EQ(dense.shape()[0], 15u);
  ASSERT_EQ(dense.shape()[1], 12u);
  for (std::size_t r = 0; r < dense.shape()[0]; ++r)
    for (std::size_t c = 0; c < dense.shape()[1]; ++c)
      EXPECT_EQ(dense.at({r, c}), 0u);
}

TEST(SparseBinaryMatrix, GetPcmForRoundsSparse_GF2Duplicates) {
  // Column 0 has row 1 listed twice (GF(2)-cancels); column 1 survives.
  std::vector<std::vector<index_type>> nested = {{1, 1}, {0}};
  auto sp = sparse_binary_matrix::from_nested_csc(4, 2, nested);
  auto [sub, first_col, last_col] = cudaq::qec::get_pcm_for_rounds(
      sp, /*num_syndromes_per_round=*/2, /*start_round=*/0, /*end_round=*/0,
      /*straddle_start_round=*/true, /*straddle_end_round=*/true);
  ASSERT_EQ(sub.shape()[0], 2u);
  ASSERT_EQ(sub.shape()[1], 1u);
  EXPECT_EQ(sub.at({0, 0}), 1u); // row 0 of surviving column
  EXPECT_EQ(sub.at({1, 0}), 0u);
  EXPECT_EQ(first_col, 1u);
  EXPECT_EQ(last_col, 1u);
}

TEST(SparseBinaryMatrix, SparseReorderCanCropRows) {
  std::vector<std::vector<index_type>> nested = {{0, 2, 3}, {1}, {2}};
  auto sp = sparse_binary_matrix::from_nested_csc(4, 3, nested);
  auto reordered = cudaq::qec::reorder_pcm_columns(
      sp, std::vector<std::uint32_t>{2, 0}, /*row_begin=*/1,
      /*row_end=*/2);

  EXPECT_EQ(reordered.layout(), sparse_binary_matrix_layout::csc);
  EXPECT_EQ(reordered.num_rows(), 2u);
  EXPECT_EQ(reordered.num_cols(), 2u);
  auto out = reordered.to_nested_csc();
  EXPECT_EQ(out[0], (std::vector<index_type>{1}));
  EXPECT_EQ(out[1], (std::vector<index_type>{1}));
}

TEST(SparseBinaryMatrix, SparseReorderMatchesDense) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
                                    0, 1, 0, 0, 1, 1, 1, 0, 0, 0};
  auto dense = cudaqx::tensor<std::uint8_t>({4, 5});
  dense.copy(data.data(), {4, 5});
  auto sparse = sparse_binary_matrix(dense, sparse_binary_matrix_layout::csr);
  std::vector<std::uint32_t> column_order = {4, 2, 0, 3};

  auto dense_reordered =
      cudaq::qec::reorder_pcm_columns(dense, column_order, /*row_begin=*/1,
                                      /*row_end=*/3);
  auto sparse_reordered =
      cudaq::qec::reorder_pcm_columns(sparse, column_order, /*row_begin=*/1,
                                      /*row_end=*/3);

  EXPECT_TRUE(dense_pcm_equal(dense_reordered, sparse_reordered.to_dense()));
}

TEST(SparseBinaryMatrix, SparseShufflePreservesShapeAndNnz) {
  auto sp = cudaq::qec::generate_random_pcm_sparse(
      /*n_rounds=*/2, /*n_errs_per_round=*/4,
      /*n_syndromes_per_round=*/3, /*weight=*/2, std::mt19937_64(7));
  auto shuffled = cudaq::qec::shuffle_pcm_columns(sp, std::mt19937_64(11));
  EXPECT_EQ(shuffled.num_rows(), sp.num_rows());
  EXPECT_EQ(shuffled.num_cols(), sp.num_cols());
  EXPECT_EQ(shuffled.num_nnz(), sp.num_nnz());
}

TEST(SparseBinaryMatrix, SparseShuffleMatchesDense) {
  auto dense = cudaq::qec::generate_random_pcm(
      /*n_rounds=*/3, /*n_errs_per_round=*/6,
      /*n_syndromes_per_round=*/4, /*weight=*/2, std::mt19937_64(17));
  auto sparse = sparse_binary_matrix(dense, sparse_binary_matrix_layout::csc);

  auto dense_shuffled =
      cudaq::qec::shuffle_pcm_columns(dense, std::mt19937_64(23));
  auto sparse_shuffled =
      cudaq::qec::shuffle_pcm_columns(sparse, std::mt19937_64(23));

  EXPECT_TRUE(dense_pcm_equal(dense_shuffled, sparse_shuffled.to_dense()));
}

TEST(SparseBinaryMatrix, SparseSerializationMatchesDenseSerialization) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0};
  auto dense = cudaqx::tensor<std::uint8_t>({3, 4});
  dense.copy(data.data(), {3, 4});
  auto sparse = sparse_binary_matrix(dense);

  EXPECT_EQ(cudaq::qec::pcm_to_sparse_string(dense),
            cudaq::qec::pcm_to_sparse_string(sparse));
  EXPECT_EQ(cudaq::qec::pcm_to_sparse_vec(dense),
            cudaq::qec::pcm_to_sparse_vec(sparse));
}

TEST(SparseBinaryMatrix, SparseSerializationCanonicalizesStoredEntries) {
  auto sp = sparse_binary_matrix::from_nested_csr(
      2, 4, std::vector<std::vector<index_type>>{{2, 1, 1}, {3, 0, 3}});

  EXPECT_EQ(cudaq::qec::pcm_to_sparse_string(sp), "2,-1,0,-1");
  EXPECT_EQ(cudaq::qec::pcm_to_sparse_vec(sp),
            (std::vector<std::int64_t>{2, -1, 0, -1}));
}

TEST(SparseBinaryMatrix,
     GetPcmForRoundsSparse_CanonicalSkipMatchesCanonicalize) {
  // On canonical input, the pcm_is_canonical=true skip path must match the
  // default canonicalize-on-entry path byte-for-byte.
  std::mt19937_64 rng(20260521);
  auto dense = cudaq::qec::generate_random_pcm(/*n_rounds=*/4,
                                               /*n_errs_per_round=*/6,
                                               /*n_syndromes_per_round=*/3,
                                               /*weight=*/2, std::move(rng));
  auto sorted =
      cudaq::qec::sort_pcm_columns(dense, /*num_syndromes_per_round=*/3);
  auto canonical = sparse_binary_matrix(sorted).canonicalize().to_csc();

  constexpr std::uint32_t kSp = 3;
  for (bool straddle_start : {false, true}) {
    for (bool straddle_end : {false, true}) {
      auto [a, fc_a, lc_a] = cudaq::qec::get_pcm_for_rounds(
          canonical, kSp, /*start_round=*/1, /*end_round=*/2, straddle_start,
          straddle_end, /*pcm_is_canonical=*/false);
      auto [b, fc_b, lc_b] = cudaq::qec::get_pcm_for_rounds(
          canonical, kSp, 1, 2, straddle_start, straddle_end,
          /*pcm_is_canonical=*/true);
      EXPECT_TRUE(dense_pcm_equal(a, b));
      EXPECT_EQ(fc_a, fc_b);
      EXPECT_EQ(lc_a, lc_b);
    }
  }
}

TEST(SparseBinaryMatrix, Canonicalize_PreservesCsrLayout) {
  std::vector<std::vector<index_type>> nested_csr = {{0, 0, 1}, {1, 2}};
  auto sp = sparse_binary_matrix::from_nested_csr(2, 3, nested_csr);
  auto canon = sp.canonicalize();
  EXPECT_EQ(canon.layout(), sparse_binary_matrix_layout::csr);
  EXPECT_EQ(canon.to_nested_csr()[0], (std::vector<index_type>{1}));
  EXPECT_EQ(canon.to_nested_csr()[1], (std::vector<index_type>{1, 2}));
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparse_RejectsOverflowingCols) {
  // n_rounds * n_errs_per_round = 2^33 overflows uint32_t; guard must fire
  // before any allocation.
  EXPECT_THROW(cudaq::qec::generate_random_pcm_sparse(
                   /*n_rounds=*/std::size_t{1} << 17,
                   /*n_errs_per_round=*/std::size_t{1} << 16,
                   /*n_syndromes_per_round=*/2,
                   /*weight=*/1, std::mt19937_64(0)),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparse_RejectsOverflowingRows) {
  // n_rounds * n_syndromes_per_round = 2^33 overflows uint32_t. Same guard,
  // symmetric on the row dimension.
  EXPECT_THROW(cudaq::qec::generate_random_pcm_sparse(
                   /*n_rounds=*/std::size_t{1} << 17,
                   /*n_errs_per_round=*/1,
                   /*n_syndromes_per_round=*/std::size_t{1} << 16,
                   /*weight=*/1, std::mt19937_64(0)),
               std::invalid_argument);
}

// Large sparse PCM (nnz ~ 2e6): sparse path must succeed and produce a
// well-formed CSC matrix.
TEST(SparseBinaryMatrix, GenerateRandomPcmSparse_LargeMatrix) {
  constexpr std::size_t n_rounds = 2;
  constexpr std::size_t n_errs_per_round = 250'000;
  constexpr std::size_t n_syndromes_per_round = 1'000;
  constexpr int weight = 4;

  const std::size_t n_cols = n_rounds * n_errs_per_round;
  const std::size_t n_rows = n_rounds * n_syndromes_per_round;

  auto sp = cudaq::qec::generate_random_pcm_sparse(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(0xC0DEull));
  EXPECT_EQ(sp.layout(), sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_rows(), static_cast<index_type>(n_rows));
  EXPECT_EQ(sp.num_cols(), static_cast<index_type>(n_cols));
  EXPECT_EQ(static_cast<std::size_t>(sp.num_nnz()), n_cols * weight);

  // Each column has exactly `weight` strictly-ascending in-range row indices.
  const auto &ptr = sp.ptr();
  const auto &idx = sp.indices();
  ASSERT_EQ(ptr.size(), n_cols + 1);
  for (std::size_t c = 0; c < n_cols; ++c) {
    const auto begin = ptr[c];
    const auto end = ptr[c + 1];
    ASSERT_EQ(end - begin, static_cast<index_type>(weight));
    for (auto p = begin + 1; p < end; ++p)
      ASSERT_LT(idx[p - 1], idx[p]);
    for (auto p = begin; p < end; ++p)
      ASSERT_LT(idx[p], static_cast<index_type>(n_rows));
  }

  // canonicalize must be a content-preserving no-op on canonical input.
  auto canon = sp.canonicalize();
  EXPECT_EQ(canon.num_rows(), sp.num_rows());
  EXPECT_EQ(canon.num_cols(), sp.num_cols());
  EXPECT_EQ(canon.num_nnz(), sp.num_nnz());
  EXPECT_EQ(canon.ptr(), sp.ptr());
  EXPECT_EQ(canon.indices(), sp.indices());
}

TEST(SparseBinaryMatrix, EmptyDefaultMatrixValidatesAsCanonical) {
  // Default-constructed sparse matrices are valid empty matrices; validation
  // should return before requiring a pointer array.
  sparse_binary_matrix sp;
  EXPECT_NO_THROW(sp.validate_sorted_unique_indices("empty"));
}

TEST(SparseBinaryMatrix, DegenerateCsrDenseConstructorInitializesPointers) {
  // A CSR matrix with zero columns still needs one row pointer per row plus the
  // trailing sentinel, and to_dense must preserve the degenerate shape.
  cudaqx::tensor<std::uint8_t> dense({2, 0});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  EXPECT_EQ(sp.layout(), sparse_binary_matrix_layout::csr);
  EXPECT_EQ(sp.ptr(), (std::vector<index_type>{0, 0, 0}));
  auto back = sp.to_dense();
  EXPECT_EQ(back.shape()[0], 2u);
  EXPECT_EQ(back.shape()[1], 0u);
}

TEST(SparseBinaryMatrix, ToCsrOnCsrReturnsEquivalentCopy) {
  // Calling to_csr() on an already-CSR matrix should use the direct copy path;
  // the copied structure must retain both layout and sparse storage.
  std::vector<std::vector<index_type>> nested = {{0, 2}, {1}};
  auto csr = sparse_binary_matrix::from_nested_csr(2, 3, nested);
  auto csr_copy = csr.to_csr();
  EXPECT_EQ(csr_copy.layout(), sparse_binary_matrix_layout::csr);
  EXPECT_EQ(csr_copy.ptr(), csr.ptr());
  EXPECT_EQ(csr_copy.indices(), csr.indices());
}

TEST(SparseBinaryMatrix, PcmIsSortedRejectsOutOfOrderSparseColumns) {
  // Column 0 starts in a later round than column 1, so sorted order would swap
  // them and pcm_is_sorted must report false.
  std::vector<std::vector<std::uint32_t>> sparse_pcm = {{2}, {0}};
  EXPECT_FALSE(cudaq::qec::pcm_is_sorted(sparse_pcm,
                                         /*num_syndromes_per_round=*/2));
}

TEST(SparseBinaryMatrix, GenerateRandomPcmWeightZeroWithNoSyndromes) {
  // With zero syndromes per round and zero weight, both dense and sparse
  // generators skip row sampling and produce empty-row matrices.
  auto dense =
      cudaq::qec::generate_random_pcm(/*n_rounds=*/3,
                                      /*n_errs_per_round=*/4,
                                      /*n_syndromes_per_round=*/0,
                                      /*weight=*/0, std::mt19937_64(0));
  EXPECT_EQ(dense.shape()[0], 0u);
  EXPECT_EQ(dense.shape()[1], 12u);

  auto sparse =
      cudaq::qec::generate_random_pcm_sparse(/*n_rounds=*/3,
                                             /*n_errs_per_round=*/4,
                                             /*n_syndromes_per_round=*/0,
                                             /*weight=*/0, std::mt19937_64(0));
  EXPECT_EQ(sparse.num_rows(), 0u);
  EXPECT_EQ(sparse.num_cols(), 12u);
  EXPECT_EQ(sparse.num_nnz(), 0u);
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparseRejectsInvalidWeight) {
  // Weight cannot exceed the number of syndromes per round; the validation
  // should fail before allocation or random sampling.
  EXPECT_THROW(cudaq::qec::generate_random_pcm_sparse(
                   /*n_rounds=*/1, /*n_errs_per_round=*/2,
                   /*n_syndromes_per_round=*/1, /*weight=*/2,
                   std::mt19937_64(0)),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, GenerateRandomPcmSparseRejectsOverflowingNnz) {
  // Dimensions may fit uint32_t while nnz = n_cols * weight does not; this must
  // be rejected before building the nested sparse representation.
  constexpr std::size_t kMaxIndex = std::numeric_limits<index_type>::max();
  EXPECT_THROW(cudaq::qec::generate_random_pcm_sparse(
                   /*n_rounds=*/1, /*n_errs_per_round=*/kMaxIndex,
                   /*n_syndromes_per_round=*/2, /*weight=*/2,
                   std::mt19937_64(0)),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, GenerateRandomPcmRejectsOverflowingDenseDimensions) {
  // Dense generation must reject size_t multiplication overflow before it tries
  // to allocate the tensor backing store.
  constexpr std::size_t kTooLarge =
      std::numeric_limits<std::size_t>::max() / 2 + 1;
  EXPECT_THROW(cudaq::qec::generate_random_pcm(
                   /*n_rounds=*/kTooLarge, /*n_errs_per_round=*/2,
                   /*n_syndromes_per_round=*/1, /*weight=*/0,
                   std::mt19937_64(0)),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::generate_random_pcm(
                   /*n_rounds=*/1, /*n_errs_per_round=*/kTooLarge,
                   /*n_syndromes_per_round=*/3, /*weight=*/0,
                   std::mt19937_64(0)),
               std::invalid_argument);
}

// Empty 0 x N PCMs reorder to 0 x |column_order|; row_begin past the empty
// matrix is a bounds error.
TEST(SparseBinaryMatrix, ReorderEmptyPcmRows) {
  const std::vector<std::uint32_t> column_order{2, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({std::size_t{0}, std::size_t{3}});
  auto dense_out =
      cudaq::qec::reorder_pcm_columns(dense, column_order, /*row_begin=*/0);
  ASSERT_EQ(dense_out.rank(), 2u);
  EXPECT_EQ(dense_out.shape()[0], 0u);
  EXPECT_EQ(dense_out.shape()[1], column_order.size());

  auto sparse = sparse_binary_matrix::from_nested_csc(0, 3, {{}, {}, {}});
  auto sparse_out =
      cudaq::qec::reorder_pcm_columns(sparse, column_order, /*row_begin=*/0);
  EXPECT_EQ(sparse_out.num_rows(), 0u);
  EXPECT_EQ(sparse_out.num_cols(),
            static_cast<index_type>(column_order.size()));

  EXPECT_THROW(cudaq::qec::reorder_pcm_columns(dense, column_order,
                                               /*row_begin=*/1),
               std::invalid_argument);
  EXPECT_THROW(cudaq::qec::reorder_pcm_columns(sparse, column_order,
                                               /*row_begin=*/1),
               std::invalid_argument);
}

} // namespace
} // namespace cudaq::qec
