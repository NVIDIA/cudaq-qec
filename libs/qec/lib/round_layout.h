/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

// Internal helpers shared between pcm_utils.cpp and the
// sliding_window decoder. Not part of the public API.

namespace cudaq::qec::details {

/// @brief Maps between detector rounds and detector rows for a (possibly
/// non-uniform) round layout. The rows are laid out as [B | S | ... | S | B],
/// where B == num_boundary_syndromes is the width of the first/last boundary
/// rounds and S == num_syndromes_per_round is the interior width. B == 0 or
/// B == S is the uniform layout (every round has width S).
struct round_layout {
  ///  interior round width
  std::size_t S = 0;
  /// boundary round width
  std::size_t B = 0;
  /// total number of detectors
  std::size_t num_rows = 0;
  /// number of rounds (detector layers)
  std::size_t num_rounds = 0;
  /// true iff B is a distinct boundary width
  bool boundary = false;

  round_layout() = default;
  round_layout(std::size_t num_syndromes_per_round,
               std::size_t num_boundary_syndromes, std::size_t total_rows);

  /// Global row index at which round @p r begins; round_start(num_rounds) is
  /// the trailing sentinel (== num_rows).
  std::size_t round_start(std::size_t r) const;

  /// Number of detector rows in round @p r (B for first/last, else S).
  std::size_t round_width(std::size_t r) const;

  /// The round that global detector row @p row belongs to.
  std::size_t row_to_round(std::size_t row) const;
};

} // namespace cudaq::qec::details
