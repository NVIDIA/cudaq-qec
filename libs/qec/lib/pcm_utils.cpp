/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/pcm_utils.h"
#include "round_layout.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cudaq::qec::details {

round_layout::round_layout(std::size_t num_syndromes_per_round,
                           std::size_t num_boundary_syndromes,
                           std::size_t total_rows)
    : S(num_syndromes_per_round), B(num_boundary_syndromes),
      num_rows(total_rows) {
  boundary = (B != 0 && B != S);
  // Guard the boundary subtraction: a malformed layout (num_rows < 2*B) would
  // otherwise underflow. Callers validate row consistency and reject it.
  if (!boundary)
    num_rounds = num_rows / S;
  else if (num_rows >= 2 * B)
    num_rounds = (num_rows - 2 * B) / S + 2;
  else
    num_rounds = 0;
}

std::size_t round_layout::round_start(std::size_t r) const {
  if (!boundary)
    return r * S;
  if (r == 0)
    return 0;
  if (r >= num_rounds)
    return num_rows;
  return B + (r - 1) * S;
}

std::size_t round_layout::round_width(std::size_t r) const {
  return round_start(r + 1) - round_start(r);
}

std::size_t round_layout::row_to_round(std::size_t row) const {
  if (!boundary)
    return row / S;
  if (row < B)
    return 0;
  if (row >= num_rows - B)
    return num_rounds - 1;
  return 1 + (row - B) / S;
}

} // namespace cudaq::qec::details

namespace {

/// Validate parameters for generate_random_pcm_sparse: weight bounds and
/// index_type overflow on the product dimensions. Throws std::invalid_argument
/// on violation.
void validate_random_pcm_sparse_params(std::size_t n_rounds,
                                       std::size_t n_errs_per_round,
                                       std::size_t n_syndromes_per_round,
                                       int weight) {
  if (weight < 0 || static_cast<std::size_t>(weight) > n_syndromes_per_round) {
    throw std::invalid_argument(
        "generate_random_pcm_sparse: weight must be in [0, "
        "n_syndromes_per_round]; got weight=" +
        std::to_string(weight) +
        ", n_syndromes_per_round=" + std::to_string(n_syndromes_per_round));
  }
  // Reject products that wouldn't fit in index_type before they can wrap.
  constexpr auto kMaxDim = static_cast<std::size_t>(
      std::numeric_limits<cudaq::qec::sparse_binary_matrix::index_type>::max());
  if (n_rounds != 0 && (n_errs_per_round > kMaxDim / n_rounds ||
                        n_syndromes_per_round > kMaxDim / n_rounds)) {
    throw std::invalid_argument(
        "generate_random_pcm_sparse: n_rounds * n_errs_per_round and "
        "n_rounds * n_syndromes_per_round must each fit in "
        "sparse_binary_matrix::index_type (uint32_t); got n_rounds=" +
        std::to_string(n_rounds) +
        ", n_errs_per_round=" + std::to_string(n_errs_per_round) +
        ", n_syndromes_per_round=" + std::to_string(n_syndromes_per_round));
  }
  // nnz = n_cols * weight must fit in index_type; otherwise `nested`
  // allocation OOMs before from_nested_csc validates.
  const std::size_t n_cols = n_rounds * n_errs_per_round;
  const std::size_t w = static_cast<std::size_t>(weight);
  if (n_cols != 0 && w > kMaxDim / n_cols) {
    throw std::invalid_argument(
        "generate_random_pcm_sparse: total nnz (n_rounds * n_errs_per_round "
        "* weight) must fit in sparse_binary_matrix::index_type (uint32_t); "
        "got n_rounds=" +
        std::to_string(n_rounds) +
        ", n_errs_per_round=" + std::to_string(n_errs_per_round) +
        ", weight=" + std::to_string(weight));
  }
}

std::vector<std::vector<cudaq::qec::sparse_binary_matrix::index_type>>
to_canonical_nested_csr(const cudaq::qec::sparse_binary_matrix &pcm) {
  return pcm.canonicalize().to_nested_csr();
}

struct reorder_pcm_columns_bounds {
  std::uint32_t row_end;
  std::size_t num_rows_to_copy;
};

reorder_pcm_columns_bounds
validate_reorder_pcm_columns_rows(std::size_t num_rows, std::uint32_t row_begin,
                                  std::uint32_t row_end) {
  if (row_begin > row_end) {
    throw std::invalid_argument(
        "reorder_pcm_columns: row_begin must be less than or equal to row_end");
  }

  if (num_rows == 0) {
    if (row_begin != 0) {
      throw std::invalid_argument(
          "reorder_pcm_columns: row_begin is greater than the number of rows "
          "in PCM");
    }
    return {0, 0};
  }

  if (static_cast<std::size_t>(row_begin) >= num_rows) {
    throw std::invalid_argument(
        "reorder_pcm_columns: row_begin is greater than the number of rows in "
        "PCM");
  }

  constexpr auto max_row = std::numeric_limits<std::uint32_t>::max();
  const auto last_row = num_rows - 1 > max_row
                            ? max_row
                            : static_cast<std::uint32_t>(num_rows - 1);
  const auto clamped_row_end = std::min(row_end, last_row);
  return {clamped_row_end,
          static_cast<std::size_t>(clamped_row_end - row_begin + 1)};
}

void validate_reorder_pcm_columns_column_order(
    const std::vector<std::uint32_t> &column_order, std::size_t num_cols) {
  for (auto c : column_order) {
    if (static_cast<std::size_t>(c) >= num_cols) {
      throw std::invalid_argument(
          "reorder_pcm_columns: column_order contains a column index that is "
          "greater than the number of columns in PCM");
    }
  }
}

/// Validate the round-related parameters shared by both get_pcm_for_rounds
/// overloads. Throws std::invalid_argument on violation.
void validate_pcm_for_rounds_params(std::uint32_t num_syndromes_per_round,
                                    std::uint32_t num_boundary_syndromes,
                                    std::uint32_t start_round,
                                    std::uint32_t end_round,
                                    std::size_t num_rows) {
  if (num_syndromes_per_round == 0) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: num_syndromes_per_round must be greater than 0");
  }
  if (num_boundary_syndromes > num_syndromes_per_round) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: num_boundary_syndromes must be <= "
        "num_syndromes_per_round");
  }
  const cudaq::qec::details::round_layout layout(
      num_syndromes_per_round, num_boundary_syndromes, num_rows);
  if (layout.boundary &&
      (num_rows < 2 * num_boundary_syndromes ||
       (num_rows - 2 * num_boundary_syndromes) % num_syndromes_per_round !=
           0)) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: number of PCM rows is inconsistent with the given "
        "num_syndromes_per_round and num_boundary_syndromes");
  }
  if (end_round >= layout.num_rounds) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: end_round is greater than the last round index");
  }
  const std::size_t first_row_to_keep = layout.round_start(start_round);
  if (first_row_to_keep >= num_rows) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: first_row_to_keep is greater than the number of "
        "rows in PCM");
  }
}

/// Select PCM column indices with any nonzero in rows belonging to rounds
/// [start_round, end_round], honoring straddle flags (same logic as dense
/// `get_pcm_for_rounds`).
void select_pcm_columns_for_round_range(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    const cudaq::qec::details::round_layout &layout, std::uint32_t start_round,
    std::uint32_t end_round, bool straddle_start_round, bool straddle_end_round,
    std::vector<std::uint32_t> &columns_in_range_out,
    std::uint32_t &first_column, std::uint32_t &last_column) {
  columns_in_range_out.clear();
  for (std::size_t c = 0; c < row_indices.size(); c++) {
    const auto &col = row_indices[c];
    if (col.empty())
      continue;
    auto first_round = layout.row_to_round(col.front());
    auto last_round = layout.row_to_round(col.back());
    if (last_round < start_round || first_round > end_round)
      continue;

    const bool fully_inside =
        first_round >= start_round && last_round <= end_round;
    if (fully_inside ||
        (straddle_start_round && first_round <= start_round &&
         last_round >= start_round) ||
        (straddle_end_round && first_round <= end_round &&
         last_round >= end_round))
      columns_in_range_out.push_back(static_cast<std::uint32_t>(c));
  }

  first_column = std::numeric_limits<std::uint32_t>::max();
  last_column = 0;
  if (!columns_in_range_out.empty()) {
    first_column = columns_in_range_out.front();
    last_column = columns_in_range_out.back();
  }
}

} // namespace

namespace cudaq::qec {

namespace {
/// Topological column sort shared by the scalar and boundary-aware
/// get_sorted_pcm_column_indices overloads. @p round_of maps a detector row to
/// its round; when @p use_rounds is false the round key is skipped.
///
/// This function tries to make a matrix that is close to a block diagonal
/// matrix from its input. If @p use_rounds is true, then the columns are first
/// sorted by the round numbers in which the checks are performed. Columns are
/// then sorted by the index of the first non-zero entry in the column, and if
/// those match, then they are sorted by the index of the last non-zero entry
/// in the column. This ping pong continues for the indices of the second
/// non-zero element and the second-to-last non-zero element, and so forth.
template <class RoundOf>
std::vector<std::uint32_t> sorted_pcm_column_indices_impl(
    const std::vector<std::vector<std::uint32_t>> &row_indices, bool use_rounds,
    RoundOf round_of) {
  std::vector<std::uint32_t> column_order(row_indices.size());
  std::iota(column_order.begin(), column_order.end(), 0);
  std::sort(column_order.begin(), column_order.end(),
            [&row_indices, use_rounds, &round_of](const std::uint32_t &a,
                                                  const std::uint32_t &b) {
              const auto &a_vec = row_indices[a];
              const auto &b_vec = row_indices[b];

              if (a_vec.size() == 0 && b_vec.size() != 0)
                return true;
              if (a_vec.size() != 0 && b_vec.size() == 0)
                return false;
              if (a_vec.size() == 0 && b_vec.size() == 0)
                return a < b; // stable sort.

              // Now we know both vectors have at least one element.

              // Have a and b iterators, both head and tail versions of both.
              auto a_it_head = a_vec.begin();
              auto a_it_tail = a_vec.end() - 1;
              auto b_it_head = b_vec.begin();
              auto b_it_tail = b_vec.end() - 1;

              // First sort by the span of rounds that the errors appear in.
              if (use_rounds) {
                auto a_first_round = round_of(*a_it_head);
                auto a_last_round = round_of(*a_it_tail);
                auto b_first_round = round_of(*b_it_head);
                auto b_last_round = round_of(*b_it_tail);
                if (a_first_round != b_first_round)
                  return a_first_round < b_first_round;
                if (a_last_round != b_last_round)
                  return a_last_round < b_last_round;
              }

              // Now we sort the columns corresponding to errors that occur in
              // the same rounds.
              do {
                // Compare the head elements.
                if (*a_it_head != *b_it_head)
                  return *a_it_head < *b_it_head;

                // Before checking the tail iterators, make sure they are not
                // aliased to the head elements that we just compared. If so,
                // we've exhausted one of the vectors and will return
                // accordingly.

                // Check if we ran out of "a" elements.
                if (a_it_head == a_it_tail && b_it_head != b_it_tail)
                  return true;
                // Check if we ran out of "b" elements.
                if (a_it_head != a_it_tail && b_it_head == b_it_tail)
                  return false;
                if (a_it_head == a_it_tail && b_it_head == b_it_tail)
                  return a < b; // stable sort.

                // Compare the tail elements.
                if (*a_it_tail != *b_it_tail)
                  return *a_it_tail < *b_it_tail;

                // Advance the head iterators.
                a_it_head++;
                b_it_head++;

                // Check to see if the new head iterators match the tail
                // iterators that we just compared. If so, we've exhausted one
                // of the vectors and will return accordingly.
                if (a_it_head == a_it_tail && b_it_head != b_it_tail)
                  return true;
                if (a_it_head != a_it_tail && b_it_head == b_it_tail)
                  return false;
                if (a_it_head == a_it_tail && b_it_head == b_it_tail)
                  return a < b; // stable sort.

                // Decrement the tail iterators.
                a_it_tail--;
                b_it_tail--;
              } while (true);

              // Unreachable.
              return a < b;
            });

  return column_order;
}
} // namespace

std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    std::uint32_t num_syndromes_per_round) {
  return sorted_pcm_column_indices_impl(
      row_indices, /*use_rounds=*/num_syndromes_per_round > 0,
      // Only invoked when use_rounds is true, so the divisor is non-zero.
      [num_syndromes_per_round](std::uint32_t r) {
        return r / num_syndromes_per_round;
      });
}

std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    std::uint32_t num_syndromes_per_round,
    std::uint32_t num_boundary_syndromes) {
  if (num_syndromes_per_round == 0)
    throw std::invalid_argument(
        "get_sorted_pcm_column_indices: num_syndromes_per_round must be "
        "non-zero when num_boundary_syndromes is specified");
  if (num_boundary_syndromes > num_syndromes_per_round)
    throw std::invalid_argument(
        "get_sorted_pcm_column_indices: num_boundary_syndromes (" +
        std::to_string(num_boundary_syndromes) +
        ") must be <= num_syndromes_per_round (" +
        std::to_string(num_syndromes_per_round) + ")");
  // Detector rows map to rounds as: the first num_boundary_syndromes rows
  // are round 0, then each num_syndromes_per_round block is an interior round.
  return sorted_pcm_column_indices_impl(
      row_indices, /*use_rounds=*/true,
      [num_syndromes_per_round,
       num_boundary_syndromes](std::uint32_t r) -> std::uint32_t {
        if (r < num_boundary_syndromes)
          return 0;
        return 1 + (r - num_boundary_syndromes) / num_syndromes_per_round;
      });
}

bool pcm_is_sorted(const std::vector<std::vector<std::uint32_t>> &sparse_pcm,
                   std::uint32_t num_syndromes_per_round) {
  auto column_indices =
      get_sorted_pcm_column_indices(sparse_pcm, num_syndromes_per_round);
  auto num_cols = sparse_pcm.size();
  for (std::size_t c = 0; c < num_cols; c++)
    if (column_indices[c] != c)
      return false;
  return true;
}

bool pcm_is_sorted(const std::vector<std::vector<std::uint32_t>> &sparse_pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t num_boundary_syndromes) {
  auto column_indices = get_sorted_pcm_column_indices(
      sparse_pcm, num_syndromes_per_round, num_boundary_syndromes);
  auto num_cols = sparse_pcm.size();
  for (std::size_t c = 0; c < num_cols; c++)
    if (column_indices[c] != c)
      return false;
  return true;
}

/// @brief Check if a PCM is sorted.
/// @param pcm The PCM to check.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @return True if the PCM is sorted, false otherwise.
bool pcm_is_sorted(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("pcm_is_sorted: PCM must be a 2D tensor");
  }

  auto column_indices =
      get_sorted_pcm_column_indices(pcm, num_syndromes_per_round);
  auto num_cols = pcm.shape()[1];
  for (std::size_t c = 0; c < num_cols; c++)
    if (column_indices[c] != c)
      return false;
  return true;
}

std::vector<std::vector<std::uint32_t>>
dense_to_sparse(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("dense_to_sparse: PCM must be a 2D tensor");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];

  std::vector<std::vector<std::uint32_t>> row_indices(num_cols);
  for (std::size_t r = 0; r < num_rows; r++) {
    auto *row = &pcm.at({r, 0});
    for (std::size_t c = 0; c < num_cols; c++)
      if (row[c])
        row_indices[c].push_back(static_cast<std::uint32_t>(r));
  }

  return row_indices;
}

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm,
                              std::uint32_t num_syndromes_per_round) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument(
        "get_sorted_pcm_column_indices: PCM must be a 2D tensor");
  }

  auto row_indices = dense_to_sparse(pcm);

  return get_sorted_pcm_column_indices(row_indices, num_syndromes_per_round);
}

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
                    uint32_t row_begin, uint32_t row_end) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("reorder_pcm_columns: PCM must be a 2D tensor");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];
  auto new_num_cols = column_order.size();
  const auto bounds =
      validate_reorder_pcm_columns_rows(num_rows, row_begin, row_end);
  validate_reorder_pcm_columns_column_order(column_order, num_cols);

  if (bounds.num_rows_to_copy == 0)
    return cudaqx::tensor<uint8_t>(std::vector<std::size_t>{0, new_num_cols});

  auto transposed_pcm = pcm.transpose();
  cudaqx::tensor<uint8_t> new_pcm_t(
      std::vector<std::size_t>{new_num_cols, bounds.num_rows_to_copy});
  for (std::size_t c = 0; c < new_num_cols; c++) {
    auto *orig_col = &transposed_pcm.at({column_order[c], row_begin});
    auto *new_col = &new_pcm_t.at({c, 0});
    std::memcpy(new_col, orig_col, bounds.num_rows_to_copy * sizeof(uint8_t));
  }

  return new_pcm_t.transpose();
}

sparse_binary_matrix
reorder_pcm_columns(const sparse_binary_matrix &pcm,
                    const std::vector<std::uint32_t> &column_order,
                    uint32_t row_begin, uint32_t row_end) {
  using index_type = sparse_binary_matrix::index_type;
  const index_type num_rows = pcm.num_rows();
  const index_type num_cols = pcm.num_cols();
  if (column_order.size() >
      static_cast<std::size_t>(std::numeric_limits<index_type>::max())) {
    throw std::invalid_argument(
        "reorder_pcm_columns: output column count exceeds index_type range");
  }

  const auto bounds =
      validate_reorder_pcm_columns_rows(num_rows, row_begin, row_end);
  validate_reorder_pcm_columns_column_order(column_order, num_cols);

  if (bounds.num_rows_to_copy == 0) {
    return sparse_binary_matrix::from_nested_csc(
        0, static_cast<index_type>(column_order.size()),
        std::vector<std::vector<index_type>>(column_order.size()));
  }

  const auto num_rows_to_copy =
      static_cast<index_type>(bounds.num_rows_to_copy);

  auto csc = pcm.to_csc();
  const auto &ptr = csc.ptr();
  const auto &idx = csc.indices();
  std::vector<std::vector<index_type>> nested(column_order.size());
  for (std::size_t out_c = 0; out_c < column_order.size(); ++out_c) {
    const auto orig_c = column_order[out_c];
    for (auto p = ptr[orig_c]; p < ptr[orig_c + 1]; ++p) {
      const auto row = idx[p];
      if (row >= row_begin && row <= bounds.row_end)
        nested[out_c].push_back(static_cast<index_type>(row - row_begin));
    }
  }

  return sparse_binary_matrix::from_nested_csc(
      num_rows_to_copy, static_cast<index_type>(column_order.size()), nested);
}

/// @brief Sort the columns of a PCM in topological order.
/// @param pcm The PCM to sort.
/// @return A new PCM with the columns sorted in topological order.
cudaqx::tensor<uint8_t>
sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                 std::uint32_t num_syndromes_per_round) {
  auto column_order =
      get_sorted_pcm_column_indices(pcm, num_syndromes_per_round);
  return reorder_pcm_columns(pcm, column_order);
}

/// @brief Simplify a PCM by removing duplicate columns and 0-weight columns,
/// and combine the probability weight vectors accordingly.
/// @param pcm The PCM to simplify.
/// @param weights The probability weight vectors to combine. This assumes all
/// error mechanisms are independent of each other.
/// @return A new PCM with the columns sorted in topological order, and the
/// probability weight vectors combined accordingly.
std::pair<cudaqx::tensor<uint8_t>, std::vector<double>>
simplify_pcm(const cudaqx::tensor<uint8_t> &pcm,
             const std::vector<double> &weights,
             std::uint32_t num_syndromes_per_round) {
  auto row_indices = dense_to_sparse(pcm);
  auto column_order =
      get_sorted_pcm_column_indices(pcm, num_syndromes_per_round);
  // March through the columns in topological order, and combine the probability
  // weight vectors if the columns have the same row indices.
  std::vector<std::vector<std::uint32_t>> new_row_indices;
  std::vector<double> new_weights;
  const auto num_cols = column_order.size();
  for (std::size_t c = 0; c < num_cols; c++) {
    auto column_index = column_order[c];
    auto &curr_row_indices = row_indices[column_index];
    // If the column has no non-zero elements, or a weight of 0, then we skip
    // it.
    if (curr_row_indices.size() == 0 || weights[column_index] == 0)
      continue;
    if (new_row_indices.empty()) {
      new_row_indices.push_back(curr_row_indices);
      new_weights.push_back(weights[column_index]);
    } else {
      auto &prev_row_indices = new_row_indices.back();
      if (prev_row_indices == curr_row_indices) {
        // The current column has the same row indices as the previous column,
        // so we update the weights and do NOT add the duplicate column.
        auto prev_weight = new_weights.back();
        auto curr_weight = weights[column_index];
        auto new_weight =
            prev_weight + curr_weight - 2.0 * prev_weight * curr_weight;
        new_weights.back() = new_weight;
      } else {
        // The current column has different row indices than the previous
        // column. So we add the current column to the new PCM, and update the
        // weights.
        new_row_indices.push_back(curr_row_indices);
        new_weights.push_back(weights[column_index]);
      }
    }
  }

  // The new PCM may have fewer columns than the original PCM.
  cudaqx::tensor<uint8_t> new_pcm(
      std::vector<std::size_t>{pcm.shape()[0], new_row_indices.size()});
  for (std::size_t c = 0; c < new_row_indices.size(); c++)
    for (auto r : new_row_indices[c])
      new_pcm.at({r, c}) = 1;

  return std::make_pair(new_pcm, new_weights);
}

std::tuple<cudaqx::tensor<uint8_t>, std::uint32_t, std::uint32_t>
get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t start_round, std::uint32_t end_round,
                   bool straddle_start_round, bool straddle_end_round,
                   std::uint32_t num_boundary_syndromes) {
  validate_pcm_for_rounds_params(num_syndromes_per_round,
                                 num_boundary_syndromes, start_round, end_round,
                                 pcm.shape()[0]);
  const details::round_layout layout(num_syndromes_per_round,
                                     num_boundary_syndromes, pcm.shape()[0]);
  auto first_row_to_keep = layout.round_start(start_round);
  auto last_row_to_keep = layout.round_start(end_round + 1) - 1;

  // Get a sparse representation of the PCM.
  auto row_indices = dense_to_sparse(pcm);
  std::vector<std::uint32_t> columns_in_range;
  std::uint32_t first_column = 0, last_column = 0;
  select_pcm_columns_for_round_range(
      row_indices, layout, start_round, end_round, straddle_start_round,
      straddle_end_round, columns_in_range, first_column, last_column);

  return std::make_tuple(reorder_pcm_columns(pcm, columns_in_range,
                                             first_row_to_keep,
                                             last_row_to_keep),
                         first_column, last_column);
}

std::tuple<cudaqx::tensor<uint8_t>, std::uint32_t, std::uint32_t>
get_pcm_for_rounds(const sparse_binary_matrix &pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t start_round, std::uint32_t end_round,
                   bool straddle_start_round, bool straddle_end_round,
                   bool pcm_is_canonical,
                   std::uint32_t num_boundary_syndromes) {
  validate_pcm_for_rounds_params(num_syndromes_per_round,
                                 num_boundary_syndromes, start_round, end_round,
                                 pcm.num_rows());
  const details::round_layout layout(num_syndromes_per_round,
                                     num_boundary_syndromes, pcm.num_rows());
  auto first_row_to_keep =
      static_cast<std::uint32_t>(layout.round_start(start_round));
  auto last_row_to_keep =
      static_cast<std::uint32_t>(layout.round_start(end_round + 1) - 1);

  // select_pcm_columns_for_round_range reads .front()/.back() as min/max row,
  // which requires sorted-per-column input. Canonicalize unless the caller
  // opts out via \p pcm_is_canonical.
  auto row_indices = pcm_is_canonical ? pcm.to_nested_csc()
                                      : pcm.canonicalize().to_nested_csc();
  std::vector<std::uint32_t> columns_in_range;
  std::uint32_t first_column = 0, last_column = 0;
  select_pcm_columns_for_round_range(
      row_indices, layout, start_round, end_round, straddle_start_round,
      straddle_end_round, columns_in_range, first_column, last_column);

  const std::uint32_t num_rows_to_copy =
      last_row_to_keep - first_row_to_keep + 1;
  cudaqx::tensor<uint8_t> dense_sub(std::vector<std::size_t>{
      static_cast<std::size_t>(num_rows_to_copy), columns_in_range.size()});

  // row_indices is canonical here, so direct assignment ≡ XOR-flip.
  for (std::size_t out_c = 0; out_c < columns_in_range.size(); ++out_c) {
    auto orig_col = columns_in_range[out_c];
    for (auto row : row_indices[orig_col]) {
      if (row >= first_row_to_keep && row <= last_row_to_keep)
        dense_sub.at(
            {static_cast<std::size_t>(row - first_row_to_keep), out_c}) = 1;
    }
  }

  return std::make_tuple(std::move(dense_sub), first_column, last_column);
}

/// @brief Generate a random PCM with the given parameters.
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
                                            int weight, std::mt19937_64 &&rng) {
  // Negative or oversized weight would spin the rejection loop forever.
  if (weight < 0 || static_cast<std::size_t>(weight) > n_syndromes_per_round) {
    throw std::invalid_argument(
        "generate_random_pcm: weight must be in [0, n_syndromes_per_round]; "
        "got weight=" +
        std::to_string(weight) +
        ", n_syndromes_per_round=" + std::to_string(n_syndromes_per_round));
  }
  constexpr auto kSizeMax = std::numeric_limits<std::size_t>::max();
  // Overflow-safe cap check in (a > limit / b) form; otherwise size_t wrap
  // could bypass the cap.
  auto exceeds_cap = [](std::size_t a, std::size_t b) {
    return a != 0 && b > kSizeMax / a;
  };
  if (exceeds_cap(n_rounds, n_errs_per_round) ||
      exceeds_cap(n_rounds, n_syndromes_per_round)) {
    throw std::invalid_argument(
        "generate_random_pcm: n_rounds * n_errs_per_round and "
        "n_rounds * n_syndromes_per_round must not overflow size_t; got "
        "n_rounds=" +
        std::to_string(n_rounds) +
        ", n_errs_per_round=" + std::to_string(n_errs_per_round) +
        ", n_syndromes_per_round=" + std::to_string(n_syndromes_per_round) +
        ". Use cudaq::qec::generate_random_pcm_sparse(...) instead.");
  }
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  if (exceeds_cap(n_rows, n_cols)) {
    throw std::invalid_argument(
        std::string("generate_random_pcm: total dense size n_rows * n_cols (") +
        std::to_string(n_rows) + " × " + std::to_string(n_cols) +
        ") must not overflow size_t; use "
        "cudaq::qec::generate_random_pcm_sparse(...) instead.");
  }

  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{n_rows, n_cols});

  // Generate a random bit (either a 0 or 1) for each element of the PCM.
  std::uniform_int_distribution<> dis(0, 1);

  for (std::size_t r = 0; r < n_rounds; ++r) {
    for (std::size_t c = 0; c < n_errs_per_round; ++c) {
      auto c_ix = r * n_errs_per_round + c;
      // Randomly decide if this column has all errors appear within this round
      // or if they should also appear in the next round too.
      bool all_errors_in_this_round = dis(rng) ? true : false;
      if (r == n_rounds - 1)
        all_errors_in_this_round = true;
      std::size_t row_max = all_errors_in_this_round
                                ? n_syndromes_per_round
                                : 2 * n_syndromes_per_round;
      // row_max==0 (only reachable with weight==0) would UB row_dis(0, -1).
      if (row_max == 0)
        continue;
      // Default-int template would silently truncate row_max - 1 above INT_MAX.
      std::uniform_int_distribution<std::size_t> row_dis(0, row_max - 1);
      for (std::size_t i = 0; i < weight; ++i) {
        auto row_ix = row_dis(rng);
        // Loop until we find a row that has not been set yet
        while (pcm.at({r * n_syndromes_per_round + row_ix, c_ix}) == 1)
          row_ix = row_dis(rng);
        pcm.at({r * n_syndromes_per_round + row_ix, c_ix}) = 1;
      }
    }
  }

  return pcm;
}

sparse_binary_matrix
generate_random_pcm_sparse(std::size_t n_rounds, std::size_t n_errs_per_round,
                           std::size_t n_syndromes_per_round, int weight,
                           std::mt19937_64 &&rng) {
  validate_random_pcm_sparse_params(n_rounds, n_errs_per_round,
                                    n_syndromes_per_round, weight);
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;

  std::uniform_int_distribution<> dis(0, 1);

  std::vector<std::vector<sparse_binary_matrix::index_type>> nested(n_cols);

  // Version-stamped bitmap: row drawn iff stamp[row_ix] == cur_stamp, bumped
  // per column to skip clearing. The n_cols guard above caps bumps < 2^32.
  const std::size_t stamp_size =
      n_syndromes_per_round == 0 ? 0 : 2 * n_syndromes_per_round;
  std::vector<std::uint32_t> stamp(stamp_size, 0);
  std::uint32_t cur_stamp = 0;

  for (std::size_t r = 0; r < n_rounds; ++r) {
    for (std::size_t c = 0; c < n_errs_per_round; ++c) {
      auto c_ix = r * n_errs_per_round + c;
      bool all_errors_in_this_round = dis(rng) ? true : false;
      if (r == n_rounds - 1)
        all_errors_in_this_round = true;
      std::size_t row_max = all_errors_in_this_round
                                ? n_syndromes_per_round
                                : 2 * n_syndromes_per_round;
      // See generate_random_pcm re: row_max==0 row_dis UB.
      if (row_max == 0)
        continue;
      // index_type, not int default: row_max can reach UINT32_MAX.
      std::uniform_int_distribution<sparse_binary_matrix::index_type> row_dis(
          0, row_max - 1);
      ++cur_stamp;
      for (int i = 0; i < weight; ++i) {
        std::size_t row_ix = row_dis(rng);
        while (stamp[row_ix] == cur_stamp)
          row_ix = row_dis(rng);
        stamp[row_ix] = cur_stamp;
        std::size_t global_row = r * n_syndromes_per_round + row_ix;
        nested[c_ix].push_back(
            static_cast<sparse_binary_matrix::index_type>(global_row));
      }
    }
  }

  // Sort per column so downstream consumers reading .front()/.back() as
  // min/max row get correct round assignments.
  for (auto &col : nested)
    std::sort(col.begin(), col.end());

  return sparse_binary_matrix::from_nested_csc(
      static_cast<sparse_binary_matrix::index_type>(n_rows),
      static_cast<sparse_binary_matrix::index_type>(n_cols), nested);
}

/// @brief Randomly permute the columns of a PCM.
/// @param pcm The PCM to permute.
/// @return A new PCM with the columns permuted randomly.
cudaqx::tensor<uint8_t> shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                                            std::mt19937_64 &&rng) {
  std::vector<std::uint32_t> column_order(pcm.shape()[1]);
  std::iota(column_order.begin(), column_order.end(), 0);
  std::shuffle(column_order.begin(), column_order.end(), rng);
  return reorder_pcm_columns(pcm, column_order);
}

sparse_binary_matrix shuffle_pcm_columns(const sparse_binary_matrix &pcm,
                                         std::mt19937_64 &&rng) {
  std::vector<std::uint32_t> column_order(pcm.num_cols());
  std::iota(column_order.begin(), column_order.end(), 0);
  std::shuffle(column_order.begin(), column_order.end(), rng);
  return reorder_pcm_columns(pcm, column_order);
}

/// @brief Extend a PCM to the given number of rounds.
/// @param pcm The PCM to extend.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param n_rounds The number of rounds to extend the PCM to.
/// @return A pair of the new PCM and the list of column indices from the
/// original PCM that were used to form the new PCM.
std::pair<cudaqx::tensor<uint8_t>, std::vector<std::uint32_t>>
pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &pcm,
                       std::size_t num_syndromes_per_round,
                       std::uint32_t n_rounds) {
  // Current number of rounds
  auto orig_num_rounds = pcm.shape()[0] / num_syndromes_per_round;

  if (orig_num_rounds > n_rounds) {
    throw std::invalid_argument(
        "extend_pcm_to_n_rounds: original number of "
        "rounds must be less than or equal to n_rounds");
  }

  // Verify the PCM is already sorted.
  if (!pcm_is_sorted(pcm, num_syndromes_per_round)) {
    throw std::invalid_argument(
        "extend_pcm_to_n_rounds: input PCM is not sorted");
  }

  // Find out the number of non-zero columns in the PCM.
  auto num_orig_non_zero_cols = [&]() -> std::uint32_t {
    auto [sub_pcm, first_column, last_column] = get_pcm_for_rounds(
        pcm, num_syndromes_per_round, 0, orig_num_rounds - 1, true, true);
    return static_cast<std::uint32_t>(sub_pcm.shape()[1]);
  }();

  // Traverse the PCM, fetching one round at a time. Save the first and
  // last time the PCM matches the previous round. Once it mismatches after it
  // had started matching, then exit the loop.
  auto [prior_sub_pcm, prior_first_column, prior_last_column] =
      get_pcm_for_rounds(pcm, num_syndromes_per_round, 0, 0, true, true);
  std::uint32_t first_match = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t last_match = std::numeric_limits<std::uint32_t>::min();
  std::vector<std::uint32_t> first_columns;
  std::vector<std::uint32_t> last_columns;
  first_columns.push_back(prior_first_column);
  last_columns.push_back(prior_last_column);
  for (std::uint32_t w = 1; w < orig_num_rounds; w++) {
    auto [sub_pcm, first_column, last_column] =
        get_pcm_for_rounds(pcm, num_syndromes_per_round, w, w, true, true);
    first_columns.push_back(first_column);
    last_columns.push_back(last_column);
    if (sub_pcm.shape()[0] == prior_sub_pcm.shape()[0] &&
        sub_pcm.shape()[1] == prior_sub_pcm.shape()[1] &&
        memcmp(sub_pcm.data(), prior_sub_pcm.data(),
               sub_pcm.size() * sizeof(uint8_t)) == 0) {
      // The PCM matches the previous round, so we start (or continue) tracking
      // first_match and last_match.
      first_match = std::min(first_match, w);
      last_match = std::max(last_match, w);
    } else if (first_match != std::numeric_limits<std::uint32_t>::max()) {
      // We've found a mismatch after we started matching, so we can stop here.
      break;
    }
    prior_sub_pcm = std::move(sub_pcm);
  }

  if (first_match == std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "extend_pcm_to_n_rounds: PCM round analysis determined that "
        "consecutive rounds never matched each other, so we don't know how to "
        "insert more rounds");
  }

  // prior_sub_pcm contains the sub-PCM that we are going to insert
  // (potentially multiple times).
  std::uint32_t num_overlap_cols_per_round =
      last_columns[first_match - 1] + 1 - first_columns[first_match];
  std::uint32_t num_cols_per_inserted_round =
      prior_sub_pcm.shape()[1] - num_overlap_cols_per_round;

  // Calculate the size of the new PCM.
  std::uint32_t num_added_rounds = n_rounds - orig_num_rounds;
  std::uint32_t num_added_cols = num_cols_per_inserted_round * num_added_rounds;
  std::uint32_t num_added_rows = num_syndromes_per_round * num_added_rounds;

  // Now we insert the new rounds into the PCM.

  auto copy_sub_pcm_into_big_pcm = [](const cudaqx::tensor<uint8_t> &sub_pcm,
                                      cudaqx::tensor<uint8_t> &big_pcm,
                                      std::uint32_t starting_row,
                                      std::uint32_t starting_col) {
    assert(starting_row + sub_pcm.shape()[0] <= big_pcm.shape()[0]);
    assert(starting_col + sub_pcm.shape()[1] <= big_pcm.shape()[1]);
    for (std::uint32_t r = 0; r < sub_pcm.shape()[0]; r++) {
      auto *sub_pcm_row = &sub_pcm.at({r, 0});
      auto *big_pcm_row = &big_pcm.at({starting_row + r, starting_col});
      std::memcpy(big_pcm_row, sub_pcm_row,
                  sub_pcm.shape()[1] * sizeof(uint8_t));
    }
  };

  auto new_pcm_size = std::vector<std::size_t>{
      pcm.shape()[0] + num_added_rows, num_orig_non_zero_cols + num_added_cols};
  cudaqx::tensor<uint8_t> new_pcm(new_pcm_size);
  std::vector<std::uint32_t> column_list;
  column_list.reserve(new_pcm_size[1]);
  std::uint32_t num_rows_populated = 0;
  std::uint32_t num_cols_populated = 0;
  // Copy the beginning.
  {
    auto [sub_pcm, first_column, last_column] = get_pcm_for_rounds(
        pcm, num_syndromes_per_round, 0, last_match - 1, true, true);
    copy_sub_pcm_into_big_pcm(sub_pcm, new_pcm, 0, 0);
    for (std::uint32_t c = 0; c <= last_column; c++)
      column_list.push_back(c);
    num_rows_populated = sub_pcm.shape()[0];
    num_cols_populated = sub_pcm.shape()[1];
  }
  // Do the extension.
  {
    auto [sub_pcm, first_column, last_column] = get_pcm_for_rounds(
        pcm, num_syndromes_per_round, first_match, first_match, true, true);
    for (std::uint32_t w = 0; w < num_added_rounds; w++) {
      copy_sub_pcm_into_big_pcm(sub_pcm, new_pcm, num_rows_populated,
                                num_cols_populated -
                                    num_overlap_cols_per_round);
      for (std::uint32_t c = first_column + num_overlap_cols_per_round;
           c <= last_column; c++)
        column_list.push_back(c);
      num_rows_populated += num_syndromes_per_round;
      num_cols_populated += num_cols_per_inserted_round;
    }
  }
  // Copy the end.
  {
    auto [sub_pcm, first_column, last_column] =
        get_pcm_for_rounds(pcm, num_syndromes_per_round, last_match,
                           orig_num_rounds - 1, true, true);
    std::uint32_t col_insert_start =
        new_pcm.shape()[1] - (pcm.shape()[1] - first_column);
    std::uint32_t first_orig_col = first_column + num_overlap_cols_per_round;
    copy_sub_pcm_into_big_pcm(sub_pcm, new_pcm, num_rows_populated,
                              col_insert_start);
    for (std::uint32_t c = first_orig_col; c <= last_column; c++)
      column_list.push_back(c);
    num_cols_populated += last_column - first_orig_col + 1;
    num_rows_populated += num_syndromes_per_round;
  }

  if (column_list.size() != new_pcm.shape()[1]) {
    throw std::runtime_error(
        "extend_pcm_to_n_rounds: column_list.size() [value: " +
        std::to_string(column_list.size()) +
        "] != new_pcm.shape()[1] [value: " +
        std::to_string(new_pcm.shape()[1]) + "]");
  }

  return std::make_pair(new_pcm, column_list);
}

std::string pcm_to_sparse_string(const cudaqx::tensor<uint8_t> &pcm) {
  // Output the string like:
  // 2,5,7,-1,2,5,32,-1,...
  std::stringstream ss;
  for (std::size_t r = 0; r < pcm.shape()[0]; r++) {
    const uint8_t *row = &pcm.at({r, 0});
    for (std::size_t c = 0; c < pcm.shape()[1]; c++) {
      if (row[c] == 1) {
        ss << c << ",";
      }
    }
    ss << "-1,";
  }
  std::string result = ss.str();
  result.pop_back(); // trim the final comma
  return result;
}

std::string pcm_to_sparse_string(const sparse_binary_matrix &pcm) {
  auto rows = to_canonical_nested_csr(pcm);
  std::stringstream ss;
  for (const auto &row : rows) {
    for (auto col : row)
      ss << col << ",";
    ss << "-1,";
  }
  std::string result = ss.str();
  if (!result.empty())
    result.pop_back(); // trim the final comma
  return result;
}

std::vector<std::int64_t>
pcm_to_sparse_vec(const cudaqx::tensor<uint8_t> &pcm) {
  std::vector<std::int64_t> sparse_vec;
  for (std::size_t r = 0; r < pcm.shape()[0]; r++) {
    for (std::size_t c = 0; c < pcm.shape()[1]; c++) {
      if (pcm.at({r, c}) == 1) {
        sparse_vec.push_back(static_cast<std::int64_t>(c));
      }
    }
    sparse_vec.push_back(-1);
  }
  return sparse_vec;
}

std::vector<std::int64_t> pcm_to_sparse_vec(const sparse_binary_matrix &pcm) {
  auto rows = to_canonical_nested_csr(pcm);
  std::vector<std::int64_t> sparse_vec;
  sparse_vec.reserve(static_cast<std::size_t>(pcm.num_nnz()) + rows.size());
  for (const auto &row : rows) {
    for (auto col : row)
      sparse_vec.push_back(static_cast<std::int64_t>(col));
    sparse_vec.push_back(-1);
  }
  return sparse_vec;
}

cudaqx::tensor<uint8_t> pcm_from_sparse_string(const std::string &sparse_str,
                                               std::size_t num_rows,
                                               std::size_t num_cols) {
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{num_rows, num_cols});
  std::stringstream ss(sparse_str);
  std::string item;
  std::uint32_t row = 0;
  while (std::getline(ss, item, ',')) {
    if (item == "-1") {
      row++;
      continue;
    }
    std::uint32_t col = std::stoul(item);
    pcm.at({row, col}) = 1;
  }
  return pcm;
}

cudaqx::tensor<uint8_t>
pcm_from_sparse_vec(const std::vector<std::int64_t> &sparse_vec,
                    std::size_t num_rows, std::size_t num_cols) {
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{num_rows, num_cols});
  std::uint64_t row = 0;
  for (std::int64_t col : sparse_vec) {
    if (col < 0) {
      row++;
      continue;
    }
    pcm.at({row, static_cast<uint64_t>(col)}) = 1;
  }
  return pcm;
}

std::vector<std::int64_t>
generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round,
                                         std::uint32_t num_rounds,
                                         bool include_first_round) {
  std::vector<std::int64_t> detector_matrix;
  if (include_first_round) {
    for (std::uint32_t i = 0; i < num_syndromes_per_round; i++) {
      detector_matrix.push_back(i);
      detector_matrix.push_back(-1);
    }
  }
  // Every round after this is a XOR of the prior round's syndrome with the
  // current round's syndrome.
  for (std::uint32_t i = 1; i < num_rounds; i++) {
    for (std::uint32_t j = 0; j < num_syndromes_per_round; j++) {
      detector_matrix.push_back((i - 1) * num_syndromes_per_round + j);
      detector_matrix.push_back(i * num_syndromes_per_round + j);
      detector_matrix.push_back(-1);
    }
  }

  return detector_matrix;
}

std::vector<std::int64_t> generate_timelike_sparse_detector_matrix(
    std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds,
    std::vector<std::int64_t> first_round_matrix) {
  if (first_round_matrix.size() == 0) {
    throw std::invalid_argument("generate_timelike_sparse_detector_matrix: "
                                "first_round_matrix must be non-empty");
  }

  for (std::uint32_t i = 0; i < first_round_matrix.size(); i++) {
    bool index_parity = (i % 2 == 0);
    // even elements should be >= 0
    if (index_parity && (first_round_matrix[i] < 0)) {
      throw std::invalid_argument(
          "generate_timelike_sparse_detector_matrix: first_round_matrix should "
          "have one index per row (row end indicated by -1)");
    }
    // odd elements should be -1
    if (!index_parity && (first_round_matrix[i] != -1)) {
      throw std::invalid_argument(
          "generate_timelike_sparse_detector_matrix: first_round_matrix should "
          "have one index per row (row end indicated by -1)");
    }
  }

  std::vector<std::int64_t> detector_matrix(first_round_matrix);

  // Every round after this is a XOR of the prior round's syndrome with the
  // current round's syndrome.
  for (std::uint32_t i = 1; i < num_rounds; i++) {
    for (std::uint32_t j = 0; j < num_syndromes_per_round; j++) {
      detector_matrix.push_back((i - 1) * num_syndromes_per_round + j);
      detector_matrix.push_back(i * num_syndromes_per_round + j);
      detector_matrix.push_back(-1);
    }
  }

  return detector_matrix;
}
} // namespace cudaq::qec
