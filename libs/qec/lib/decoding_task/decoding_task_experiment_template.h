/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: experiment_template, a dem_chunks_spec viewed as a shot of
// any length. It gives the phase, chunk and row, column and measurement
// offsets of any round without expanding the whole shot, for the decoding
// windows built on it. Never included from a public header.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace cudaq::qec {

/// A dem_chunks_spec as a shot of R rounds: prefix phases, then the repeating
/// phase, then suffix phases. Round r is chunk r; its detector rows are that
/// chunk's to_seam band, so a fault touches at most its own round and the
/// next. A spec without a repeating phase admits only its own chain length.
class experiment_template {
public:
  /// @param spec The experiment. num_rounds, when set on a repeating spec,
  /// must be one the spec admits; queries take their own shot length.
  /// @throws std::invalid_argument if the spec is invalid, if num_rounds is
  /// below min_rounds(), if a chunk spans other than one round, or if a
  /// suffix phase other than the last differs from bulk in measurement width.
  explicit experiment_template(dem_chunks_spec spec);

  /// The validated spec.
  const dem_chunks_spec &spec() const { return spec_; }
  /// Whether the spec has a repeating phase, so admits any R >= min_rounds().
  bool repeats() const { return repeats_; }
  /// Rounds before the repeating phase; every round when it does not repeat.
  std::size_t prefix_rounds() const { return prefix_.size(); }
  /// Rounds after the repeating phase.
  std::size_t suffix_rounds() const { return suffix_.size(); }
  /// Shortest shot the spec admits.
  std::size_t min_rounds() const { return min_rounds_; }
  /// 1 if some fault touches two rounds, else 0.
  std::size_t max_span() const { return max_span_; }
  /// Observables every phase carries.
  std::size_t num_observables() const { return num_observables_; }
  /// The last phase's m2o_sparse; empty if it has none.
  const std::vector<std::int64_t> &readout() const {
    return spec_.get_phase(last_).m2o_sparse;
  }

  // The rounds of an R-round shot are [0, R). phase() and chunk() take r < R;
  // the offsets take r <= R, where R gives the shot's total. All throw
  // std::invalid_argument for other r, or if the spec does not admit R.

  /// Phase of round r.
  phase_id phase(std::size_t r, std::size_t R) const;
  /// Chunk r, carrying from_seam unless it ends the shot.
  const extended_dem &chunk(std::size_t r, std::size_t R) const;
  /// Detector rows before round r.
  std::size_t row_offset(std::size_t r, std::size_t R) const;
  /// Fault columns before round r.
  std::size_t column_offset(std::size_t r, std::size_t R) const;
  /// Measurements before round r. Throws if a phase has no measurement count.
  std::size_t measurement_offset(std::size_t r, std::size_t R) const;

private:
  /// A phase's chunk inside the shot and at its end, with its sizes.
  struct phase_chunks {
    extended_dem inner;
    extended_dem last;
    std::size_t rows = 0;
    std::size_t faults = 0;
    std::optional<std::size_t> measurements;
  };
  /// Throws if the spec does not admit an R-round shot, or if r > last.
  void check_round(std::size_t r, std::size_t last, std::size_t R) const;
  /// Sum of size(phase) over rounds [0, r) of an R-round shot.
  template <typename Size>
  std::size_t sum_before(std::size_t r, std::size_t R, Size size) const;

  dem_chunks_spec spec_;
  bool repeats_ = false;
  std::vector<phase_id> prefix_;
  std::vector<phase_id> suffix_;
  phase_id bulk_;
  phase_id last_;
  std::size_t min_rounds_ = 0;
  std::size_t max_span_ = 0;
  std::size_t num_observables_ = 0;
  std::map<phase_id, phase_chunks> phases_;
};

} // namespace cudaq::qec
