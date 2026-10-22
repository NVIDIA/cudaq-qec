/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Unit tests for the experiment template and decoding windows built from
// dem_chunks: per-round phases, chunks and offsets against an expanded shot.

#include "task_graph_test_fixtures.h"
#include "../lib/decoding_task/decoding_task_experiment_template.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {
namespace {

// Only the repeating phase: no prefix or suffix.
dem_chunks_spec repeating_only_spec() {
  auto spec = arbitrary_memory_spec(5);
  spec.phases = {spec.phases[1]};
  spec.connections = {{phase_name::dem_bulk, phase_name::dem_bulk}};
  return spec;
}

// Every fault touches its own round only: the outgoing seam band is empty.
dem_chunks_spec single_round_spec() {
  auto spec = arbitrary_memory_spec(5);
  for (auto &phase : spec.phases) {
    phase.spec.seam_specs = {{seam_name::prev_round, {phase.spec.H_sparse, {}}},
                             {seam_name::next_round, {{-1, -1, -1, -1}, {}}}};
    phase.spec.H_sparse.clear();
  }
  return spec;
}

// A surface-code-like tail: a penultimate phase whose outgoing seam reaches 2
// of its 4 detectors, then a readout phase with 2 detectors and its own H.
dem_chunks_spec narrow_readout_spec() {
  auto spec = arbitrary_memory_spec(5);
  const auto after = phase_id::from_name("after0");
  dem_chunk_spec penultimate;
  dem_chunk_spec readout;
  penultimate.num_faults = 4;
  penultimate.error_rates.assign(4, 0.02);
  penultimate.O_sparse = {0, -1};
  penultimate.seam_specs = {
      {seam_name::prev_round, {{0, -1, 1, -1, 2, -1, 3, -1}, {}}},
      {seam_name::next_round, {{0, -1, 1, -1}, {}}}};
  readout.num_faults = 3;
  readout.error_rates.assign(3, 0.03);
  readout.O_sparse = {0, 2, -1};
  readout.seam_specs = {{seam_name::prev_round, {{0, 2, -1, 1, 2, -1}, {}}}};
  spec.phases.back().spec = readout;
  spec.phases.push_back({after, penultimate});
  spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_bulk},
                      {phase_name::dem_bulk, after},
                      {after, phase_name::dem_final}};
  return spec;
} // end - narrow_readout_spec()

// The rep-5 chain without its self-loop: a fixed three-round shot.
dem_chunks_spec linear_spec() {
  auto spec = rep5_spec();
  spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_final}};
  spec.num_rounds.reset();
  return spec;
}

// The mutable spec of phase id.
dem_chunk_spec &phase_of(dem_chunks_spec &spec, phase_id id) {
  for (auto &entry : spec.phases)
    if (entry.id == id)
      return entry.spec;
  throw std::out_of_range("no phase " + id.name());
}

// Explicit D, 4 measurements a round, and a 5-measurement readout round whose
// map reads 3 of them.
dem_chunks_spec wide_readout_spec() {
  auto spec = arbitrary_memory_spec(5, true);
  auto &last = phase_of(spec, phase_name::dem_final);
  last.num_measurements = 5;
  last.m2o_sparse = {0, 2, 4, -1};
  return spec;
}

// Expect check() to throw std::invalid_argument mentioning text.
template <typename Check>
void expect_error(Check &&check, const std::string &text) {
  try {
    check();
    ADD_FAILURE() << "expected: " << text;
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
  }
}

// The first row of each round in an R-round shot, and the shot's row count
// at index R, from the expanded chain's to_seam bands. The total is checked
// against the closed model.
std::vector<std::size_t> expanded_row_offsets(dem_chunks_spec spec,
                                              std::size_t R) {
  std::vector<std::size_t> offsets(R + 1, 0);
  spec.num_rounds = R;
  const auto chunks = dem_chunks_from_spec(spec);
  for (std::size_t r = 0; r < R; ++r)
    offsets[r + 1] = offsets[r] +
                     chunks[r].get_seam(spec.seam.to_seam).num_rows() +
                     chunks[r].num_interior_rows();
  EXPECT_EQ(offsets[R], dem_close_all(chunks).num_detectors());
  return offsets;
}

// Phases, chunks and offsets agree with the expanded shot for every length.
TEST(ExperimentTemplate, MatchesExpandedShot) {
  for (const auto &spec :
       {rep5_spec(), arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3),
        repeating_only_spec(), narrow_readout_spec()}) {
    const experiment_template tmpl(spec);
    EXPECT_TRUE(tmpl.repeats());
    EXPECT_EQ(tmpl.max_span(), 1u);
    for (auto R = tmpl.min_rounds(); R <= 12; ++R) {
      auto fixed = spec;
      SCOPED_TRACE("R=" + std::to_string(R));
      fixed.num_rounds = R;
      const auto sequence = fixed.phase_sequence();
      const auto rows = expanded_row_offsets(spec, R);
      const auto columns =
          dem_chunks_to_fault_offsets(dem_chunks_from_spec(fixed));
      for (std::size_t r = 0; r <= R; ++r) {
        EXPECT_EQ(tmpl.row_offset(r, R), rows[r]);
        EXPECT_EQ(tmpl.column_offset(r, R), columns[r]);
        if (r == R)
          continue;
        EXPECT_TRUE(tmpl.phase(r, R) == sequence[r]);
        EXPECT_EQ(tmpl.chunk(r, R).has_seam(spec.seam.from_seam), r + 1 < R);
      }
      // Rounds are [0, R); offsets also take R, the shot's end.
      EXPECT_THROW(tmpl.phase(R, R), std::invalid_argument);
      EXPECT_THROW(tmpl.chunk(R, R), std::invalid_argument);
      EXPECT_THROW(tmpl.row_offset(R + 1, R), std::invalid_argument);
      EXPECT_THROW(tmpl.column_offset(R + 1, R), std::invalid_argument);
      EXPECT_THROW(tmpl.measurement_offset(R + 1, R), std::invalid_argument);
    } // end - for(R)
  } // end - for(spec)
} // end - TEST(ExperimentTemplate, MatchesExpandedShot)

// Prefix, suffix and minimum lengths, and the span of a fault in rounds.
TEST(ExperimentTemplate, ReportsItsShape) {
  const experiment_template tail(multi_terminal_memory_spec(5, 3));
  const experiment_template bare(repeating_only_spec());
  EXPECT_EQ(tail.prefix_rounds(), 1u);
  EXPECT_EQ(tail.suffix_rounds(), 3u);
  EXPECT_EQ(tail.min_rounds(), 5u);
  EXPECT_EQ(tail.num_observables(), 1u);
  EXPECT_THROW(tail.row_offset(0, 4), std::invalid_argument);
  EXPECT_EQ(bare.prefix_rounds() + bare.suffix_rounds(), 0u);
  EXPECT_EQ(bare.min_rounds(), 2u);
  EXPECT_EQ(experiment_template(single_round_spec()).max_span(), 0u);
}

// A spec without a repeating phase supports exactly its own length.
TEST(ExperimentTemplate, LinearChainHasOneLength) {
  const experiment_template tmpl(linear_spec());
  const auto sequence = linear_spec().phase_sequence();
  const auto rows = expanded_row_offsets(linear_spec(), 3);
  EXPECT_FALSE(tmpl.repeats());
  EXPECT_EQ(tmpl.min_rounds(), 3u);
  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_TRUE(tmpl.phase(r, 3) == sequence[r]);
    EXPECT_EQ(tmpl.row_offset(r, 3), rows[r]);
  }
  expect_error([&] { tmpl.phase(0, 4); }, "only 3-round shots");
}

// A spec that cannot be expanded is rejected up front, not on first use.
TEST(ExperimentTemplate, RejectsSpecsItCannotExpand) {
  auto missing = rep5_spec();
  missing.connections.push_back(
      {phase_name::dem_final, phase_id::from_name("absent")});
  // Three phases need at least three rounds.
  expect_error([] { experiment_template{rep5_spec(2)}; }, "num_rounds 2");
  expect_error([&] { experiment_template{missing}; }, "'absent'");
}

// Measurement offsets follow each phase's width, the readout map is the last
// phase's, and only the last suffix phase may differ from bulk in width.
TEST(ExperimentTemplate, MeasurementWidthsAndReadout) {
  const experiment_template tmpl(wide_readout_spec());
  const experiment_template plain(arbitrary_memory_spec(5));
  auto tail = multi_terminal_memory_spec(5, 2);
  for (auto R = tmpl.min_rounds(); R <= 8; ++R) {
    EXPECT_EQ(tmpl.measurement_offset(R - 1, R), 4 * (R - 1));
    EXPECT_EQ(tmpl.measurement_offset(R, R), 4 * (R - 1) + 5);
  }
  EXPECT_EQ(tmpl.readout(), (std::vector<std::int64_t>{0, 2, 4, -1}));
  EXPECT_TRUE(plain.readout().empty());
  expect_error([&] { plain.measurement_offset(1, 5); }, "no measurement count");

  // A surface-code memory has two suffix phases: a bulk-width round, then a
  // wider readout. Only the readout, which FINAL identifies, may differ.
  phase_of(tail, phase_name::dem_final).num_measurements = 5;
  EXPECT_EQ(experiment_template(tail).measurement_offset(5, 5), 21u);
  phase_of(tail, phase_id::from_name("terminal0")).num_measurements = 5;
  expect_error([&] { experiment_template{tail}; }, "precedes FINAL");
} // end - TEST(ExperimentTemplate, MeasurementWidthsAndReadout)

} // namespace
} // namespace cudaq::qec
