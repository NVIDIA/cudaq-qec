/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"
#include "sparse_binary_matrix.h"
#include <limits>
#include <random>

namespace cudaq::qec {

/// @brief Return a sparse representation of the PCM.
/// @param pcm The PCM to convert to a sparse representation.
/// @return A vector of vectors that sparsely represents the PCM. The size of
/// the outer vector is the number of columns in the PCM, and the i-th element
/// contains an inner vector of the row indices of the non-zero elements in the
/// i-th column of the PCM.
std::vector<std::vector<std::uint32_t>>
dense_to_sparse(const cudaqx::tensor<uint8_t> &pcm);

/// @brief Return a sparse representation of the PCM as a string.
/// @param pcm The PCM to convert to a sparse representation.
/// @return A string that represents the PCM in a sparse format.
std::string pcm_to_sparse_string(const cudaqx::tensor<uint8_t> &pcm);

/// @brief Sparse overload of pcm_to_sparse_string.
/// Duplicate stored entries are canonicalized with GF(2) cancellation before
/// serialization.
std::string pcm_to_sparse_string(const sparse_binary_matrix &pcm);

/// @brief Return a PCM from a sparse representation.
/// @param sparse_str The sparse representation of the PCM.
/// @param num_rows The number of rows in the PCM.
/// @param num_cols The number of columns in the PCM.
/// @return A PCM tensor.
cudaqx::tensor<uint8_t> pcm_from_sparse_string(const std::string &sparse_str,
                                               std::size_t num_rows,
                                               std::size_t num_cols);

/// @brief Return a PCM from a sparse representation.
/// @param sparse_vec The sparse representation of the PCM, where -1 separates
/// rows.
/// @param num_rows The number of rows in the PCM.
/// @param num_cols The number of columns in the PCM.
/// @return A PCM tensor.
cudaqx::tensor<uint8_t>
pcm_from_sparse_vec(const std::vector<std::int64_t> &sparse_vec,
                    std::size_t num_rows, std::size_t num_cols);

/// @brief Return a sparse representation of the PCM.
/// @param pcm The PCM to convert to a sparse representation.
/// @return A vector of integers that represents the PCM in a sparse format,
/// where -1 separates rows.
std::vector<std::int64_t> pcm_to_sparse_vec(const cudaqx::tensor<uint8_t> &pcm);

/// @brief Sparse overload of pcm_to_sparse_vec.
/// Duplicate stored entries are canonicalized with GF(2) cancellation before
/// serialization.
std::vector<std::int64_t> pcm_to_sparse_vec(const sparse_binary_matrix &pcm);

/// @brief Generate a sparse detector matrix for a given number of syndromes per
/// round and number of rounds. Timelike here means that each round of syndrome
/// bits are xor'd against the preceding round.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param num_rounds The number of rounds.
/// @param include_first_round Whether to include the first round in the
/// detector matrix.
/// @return The detector matrix format is CSR-like, with -1 values
/// indicating the end of a row.
std::vector<std::int64_t>
generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round,
                                         std::uint32_t num_rounds,
                                         bool include_first_round = false);

/// @brief Generate a sparse detector matrix for a given number of syndromes per
/// round and number of rounds. Timelike here means that each round of syndrome
/// bits are xor'd against the preceding round. The first round is supplied by
/// the user, to allow for a mixture of detectors and non-detectors.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param num_rounds The number of rounds.
/// @param first_round_matrix User specified detector matrix for the first
/// round.
/// @return The detector matrix format is CSR-like, with -1 values
/// indicating the end of a row.
std::vector<std::int64_t> generate_timelike_sparse_detector_matrix(
    std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds,
    std::vector<std::int64_t> first_round_matrix);

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
/// @details This function tries to make a matrix that is close to a block
/// diagonal matrix from its input. Columns are first sorted by the index of the
/// first non-zero entry in the column, and if those match, then they are sorted
/// by the index of the last non-zero entry in the column. This ping pong
/// continues for the indices of the second non-zero element and the
/// second-to-last non-zero element, and so forth.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    std::uint32_t num_syndromes_per_round = 0);

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param pcm The PCM to sort.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm,
                              std::uint32_t num_syndromes_per_round = 0);

/// @brief Boundary-aware overload of the above: the first and last
/// @p num_boundary_syndromes rows form the boundary rounds and each interior
/// round spans @p num_syndromes_per_round rows. throws std::invalid_argument if
/// @p num_syndromes_per_round is zero or @p num_boundary_syndromes > @p
/// num_syndromes_per_round.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    std::uint32_t num_syndromes_per_round,
    std::uint32_t num_boundary_syndromes);

/// @brief Check if a PCM is sorted.
/// @param pcm The PCM to check.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @return True if the PCM is sorted, false otherwise.
bool pcm_is_sorted(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round = 0);

/// @brief Check if a PCM is sorted.
/// @param sparse_pcm The sparse PCM to check (in the same format as
/// dense_to_sparse())
/// @param num_syndromes_per_round The number of syndromes per round.
/// @return True if the PCM is sorted, false otherwise.
bool pcm_is_sorted(const std::vector<std::vector<std::uint32_t>> &sparse_pcm,
                   std::uint32_t num_syndromes_per_round = 0);

/// @brief Boundary-aware overload of pcm_is_sorted: the first and last
/// @p num_boundary_syndromes rows form the boundary rounds and each interior
/// round spans @p num_syndromes_per_round rows (a [B | K*S | B] layout).
/// @param sparse_pcm The sparse PCM to check (in the same format as
/// dense_to_sparse())
/// @param num_syndromes_per_round The interior-round width.
/// @param num_boundary_syndromes The width of the first/last boundary rounds.
/// @return True if the PCM is sorted for this boundary layout, false otherwise.
bool pcm_is_sorted(const std::vector<std::vector<std::uint32_t>> &sparse_pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t num_boundary_syndromes);

/// @brief Reorder the columns of a PCM according to the given column order.
/// Note: this may return a subset of the columns in the original PCM if the
/// \p column_order does not contain all of the columns in the original PCM.
/// @param pcm The PCM to reorder.
/// @param column_order The column order to use for reordering.
/// @param row_begin The first row to include in the reordering. Leave at the
/// default value to include all rows.
/// @param row_end The last row to include in the reordering. Leave at the
/// default value to include all rows.
/// @return A new PCM with the columns reordered according to the given column
/// order.
cudaqx::tensor<uint8_t>
reorder_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                    const std::vector<std::uint32_t> &column_order,
                    uint32_t row_begin = 0,
                    uint32_t row_end = std::numeric_limits<uint32_t>::max());

/// @brief Sparse overload of reorder_pcm_columns. Returns CSC.
sparse_binary_matrix
reorder_pcm_columns(const sparse_binary_matrix &pcm,
                    const std::vector<std::uint32_t> &column_order,
                    uint32_t row_begin = 0,
                    uint32_t row_end = std::numeric_limits<uint32_t>::max());

/// @brief Sort the columns of a PCM in topological order.
/// @param pcm The PCM to sort.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
/// @return A new PCM with the columns sorted in topological order.
cudaqx::tensor<uint8_t>
sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                 std::uint32_t num_syndromes_per_round = 0);

/// @brief Simplify a PCM by removing duplicate columns and 0-weight columns,
/// and combine the probability weight vectors accordingly.
/// @param pcm The PCM to simplify.
/// @param weights The probability weight vectors to combine.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
/// @return A new PCM with the columns sorted in topological order, and the
/// probability weight vectors combined accordingly.
std::pair<cudaqx::tensor<uint8_t>, std::vector<double>>
simplify_pcm(const cudaqx::tensor<uint8_t> &pcm,
             const std::vector<double> &weights,
             std::uint32_t num_syndromes_per_round = 0);

/// @brief Get a sub-PCM for a range of rounds. It is recommended (but not
/// required) that you call sort_pcm_columns() before calling this function.
/// @param pcm The PCM to get a sub-PCM for.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param start_round The start round (0-based).
/// @param end_round The end round (0-based).
/// @param straddle_start_round Whether to include columns that straddle the
/// start_round (defaults to false)
/// @param straddle_end_round Whether to include columns that straddle the
/// end_round (defaults to false)
/// @param num_boundary_syndromes Width of the narrower first/last boundary
/// rounds for a non-uniform [B | K*S | B] detector layout (0 == uniform).
/// @return A tuple with the new PCM with the columns in the range [start_round,
/// end_round], the first column included, and the last column included.
std::tuple<cudaqx::tensor<uint8_t>, std::uint32_t, std::uint32_t>
get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t start_round, std::uint32_t end_round,
                   bool straddle_start_round = false,
                   bool straddle_end_round = false,
                   std::uint32_t num_boundary_syndromes = 0);

/// @brief Same semantics as the overload taking a dense tensor \p pcm, but
/// reads from \p pcm as ``sparse_binary_matrix`` so the full dense PCM is not
/// required (only the returned sub-matrix is dense). Parameter meanings match
/// the dense overload.
///
/// @param pcm The PCM to get a sub-PCM for.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param start_round The start round (0-based).
/// @param end_round The end round (0-based).
/// @param straddle_start_round Whether to include columns that straddle the
/// start_round (defaults to false)
/// @param straddle_end_round Whether to include columns that straddle the
/// end_round (defaults to false)
/// @param pcm_is_canonical If true, the caller asserts \p pcm has
/// sorted-unique per-group indices (i.e. is the output of
/// `sparse_binary_matrix::canonicalize`
/// or was constructed canonically). In that case the per-call
/// canonicalization step is skipped — useful for callers like
/// `sliding_window` that canonicalize once at construction and then call
/// this function in a per-window loop on the same matrix. Default `false`
/// preserves the original "canonicalize on entry" behavior.
///
/// @warning When \p pcm_is_canonical is `true` the precondition is *not*
/// checked. `select_pcm_columns_for_round_range` reads `.front()` / `.back()`
/// of each CSC column to derive first/last round; those are only the true
/// min/max row if the column list is sorted-unique. Passing a non-canonical
/// PCM with this flag (e.g. a raw DEM decomposition, a hand-built sparse
/// matrix with duplicate or unsorted indices, or a canonical CSR that the
/// caller forgot to re-canonicalize after conversion) silently produces
/// wrong round assignments. If unsure, leave the flag `false`.
/// @return A tuple with the new PCM with the columns in the range [start_round,
/// end_round], the first column included, and the last column included.
/// @param num_boundary_syndromes Width of the narrower first/last boundary
/// rounds for a non-uniform [B | K*S | B] detector layout (0 == uniform).
std::tuple<cudaqx::tensor<uint8_t>, std::uint32_t, std::uint32_t>
get_pcm_for_rounds(const sparse_binary_matrix &pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t start_round, std::uint32_t end_round,
                   bool straddle_start_round = false,
                   bool straddle_end_round = false,
                   bool pcm_is_canonical = false,
                   std::uint32_t num_boundary_syndromes = 0);

/// @brief Generate a random PCM with the given parameters.
///
/// The PCM has shape `(n_rounds * n_syndromes_per_round) × (n_rounds *
/// n_errs_per_round)`.
/// @param n_rounds The number of rounds in the PCM.
/// @param n_errs_per_round The number of errors per round in the PCM.
/// @param n_syndromes_per_round The number of syndromes per round in the PCM.
/// @param weight The column weight of the PCM.
/// @param rng The random number generator to use (e.g.
/// std::mt19937_64(your_seed))
/// @return A random PCM with the given parameters.
cudaqx::tensor<uint8_t> generate_random_pcm(std::size_t n_rounds,
                                            std::size_t n_errs_per_round,
                                            std::size_t n_syndromes_per_round,
                                            int weight, std::mt19937_64 &&rng);

/// @brief Same distribution as generate_random_pcm, but constructs a CSC
/// sparse_binary_matrix directly without allocating a dense rank-2 tensor.
/// Intended for large PCMs whose dense form would be impractical or impossible
/// to allocate.
sparse_binary_matrix
generate_random_pcm_sparse(std::size_t n_rounds, std::size_t n_errs_per_round,
                           std::size_t n_syndromes_per_round, int weight,
                           std::mt19937_64 &&rng);

/// @brief Randomly permute the columns of a PCM.
/// @param pcm The PCM to permute.
/// @param rng The random number generator to use (e.g.
/// std::mt19937_64(your_seed))
/// @return A new PCM with the columns permuted randomly.
cudaqx::tensor<uint8_t> shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                                            std::mt19937_64 &&rng);

/// @brief Sparse overload of shuffle_pcm_columns.
sparse_binary_matrix shuffle_pcm_columns(const sparse_binary_matrix &pcm,
                                         std::mt19937_64 &&rng);

/// @brief Extend a PCM to the given number of rounds.
/// @param pcm The PCM to extend.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param n_rounds The number of rounds to extend the PCM to.
/// @return A pair of the new PCM and the list of column indices from the
/// original PCM that were used to form the new PCM.
std::pair<cudaqx::tensor<uint8_t>, std::vector<std::uint32_t>>
pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &pcm,
                       std::size_t num_syndromes_per_round,
                       std::uint32_t n_rounds);

} // namespace cudaq::qec
