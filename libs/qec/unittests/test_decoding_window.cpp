/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the experiment template and decoding windows built from
// dem_chunks: per-round phases, chunks and offsets against an expanded shot;
// window rows, columns, ports, sends and measurement maps against a dense
// slice of the whole shot; the window spec's nine-SEC example; the spec's
// invalid constructions; the column-handling options and parallel-column
// conflicts; variant keys and translation invariance; the sparse model in
// the kernel's forms, and a window whose dense model would take gigabytes
// building in little memory; the window schedule and commitment validation:
// row finalization (F1-F4) checked against the built windows, each failure's
// own message, send and receive matching, and ambiguous parallel commit
// columns; the serial parallel-window reference: its windows match the
// schedule and its models the production ones, single faults decode exactly,
// each send is needed, sampled shots are sound and the production windows
// commit, send and update the frame as it does, the production windows,
// received and decoded serially, decode as it does, and the logical error
// rate tracks monolithic matching; receiving a send, a window with nothing
// to decode, the decoder a window takes, and a decode step that allocates
// nothing per shot; the ingest core: shots streamed in records cut at
// random, whose every window view is checked against the detectors the
// closed model's D forms from the whole shot, with the raw logical value,
// FINAL and record errors, a ring of fixed size that keeps what views out
// need, and records that allocate nothing; the decode_window core and the
// serial pipeline: ingest and the cores, run on raw measurements with
// windows from the store, decode as the reference does, bit for bit,
// fires allocate nothing beyond their decoder, a missing residual leaves
// the imports zero, and what is not a core's window is rejected; and the
// data artifact store:
// sha256, canonical content, shared entries, templates as content, and
// lookups that never build.

#include "allocation_counter.h"
#include "parallel_window_reference.h"
#include "task_graph_test_fixtures.h"
#include "../lib/decoding_task/decoding_task_commitment.h"
#include "../lib/decoding_task/decoding_task_data_artifacts.h"
#include "../lib/decoding_task/decoding_task_decode_window_core.h"
#include "../lib/decoding_task/decoding_task_decoding_window.h"
#include "../lib/decoding_task/decoding_task_experiment_template.h"
#include "../lib/decoding_task/decoding_task_ingest_core.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/dem_construction_code.h"
#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace cudaq::qec {
namespace {

// Only the repeating phase: no prefix or suffix.
dem_chunks_spec repeating_only_spec() {
  auto spec = arbitrary_memory_spec(5);
  spec.phases = {spec.phases[1]};
  spec.connections = {{phase_name::dem_bulk, phase_name::dem_bulk}};
  return spec;
}

// A surface-code-like tail: a penultimate phase whose outgoing seam reaches 2
// of its 4 detectors, then a readout phase with 2 detectors and its own H.
// Rounds of different widths need D_sparse rows: 4 measurements a round, and
// 2 in the readout round, each detector differencing the round before.
dem_chunks_spec narrow_readout_spec() {
  auto spec = arbitrary_memory_spec(5, true);
  const auto after = phase_id::from_name("after0");
  dem_chunk_spec penultimate;
  dem_chunk_spec readout;
  penultimate.num_faults = 4;
  penultimate.error_rates.assign(4, 0.02);
  penultimate.O_sparse = {0, -1};
  penultimate.seam_specs = {
      {seam_name::prev_round, {{0, -1, 1, -1, 2, -1, 3, -1}, {}}},
      {seam_name::next_round, {{0, -1, 1, -1}, {}}}};
  penultimate.num_measurements = 4;
  penultimate.D_sparse = {0, 4, -1, 1, 5, -1, 2, 6, -1, 3, 7, -1};
  readout.num_faults = 3;
  readout.error_rates.assign(3, 0.03);
  readout.O_sparse = {0, 2, -1};
  readout.seam_specs = {{seam_name::prev_round, {{0, 2, -1, 1, 2, -1}, {}}}};
  readout.num_measurements = 2;
  readout.D_sparse = {0, 4, -1, 1, 5, -1};
  spec.phases.back().spec = readout;
  spec.phases.push_back({after, penultimate});
  spec.connections = {{phase_name::dem_init, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_bulk},
                      {phase_name::dem_bulk, after},
                      {after, phase_name::dem_final}};
  return spec;
} // end - narrow_readout_spec()

// A first round with 2 detectors, then rounds with 4. Each later detector k
// differences measurement k of its round against measurement k of the round
// before, where that round has one; the ramp phase reads the 2-wide first
// round. Without D_sparse the rows and counts are cleared.
dem_chunks_spec widening_spec(bool explicit_d) {
  auto spec = arbitrary_memory_spec(5, true);
  const auto ramp_id = phase_id::from_name("ramp");
  auto ramp = spec.phases[1].spec;
  auto &init = spec.phases[0].spec;
  init.H_sparse.clear();
  init.seam_specs = {
      {seam_name::prev_round, {{0, -1, 1, -1}, {}}},
      {seam_name::next_round, {{0, -1, 1, -1, 2, -1, 3, -1}, {}}}};
  init.num_measurements = 2;
  init.D_sparse = {0, -1, 1, -1};
  ramp.D_sparse = {0, 2, -1, 1, 3, -1, 4, -1, 5, -1};
  spec.phases.push_back({ramp_id, ramp});
  spec.connections = {{phase_name::dem_init, ramp_id},
                      {ramp_id, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_bulk},
                      {phase_name::dem_bulk, phase_name::dem_final}};
  if (!explicit_d)
    for (auto &phase : spec.phases) {
      phase.spec.D_sparse.clear();
      phase.spec.num_measurements.reset();
    }
  return spec;
} // end - widening_spec()

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

// Expect check() to throw std::invalid_argument mentioning every text.
template <typename Check, typename... Texts>
void expect_error(Check &&check, const Texts &...texts) {
  const std::vector<std::string> expected{std::string(texts)...};
  try {
    check();
    ADD_FAILURE() << "expected: " << expected.front();
  } catch (const std::invalid_argument &e) {
    for (const auto &text : expected)
      EXPECT_NE(std::string(e.what()).find(text), std::string::npos)
          << e.what();
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
  // A fault that flips only the next round's detectors still reaches it.
  EXPECT_EQ(experiment_template(next_only_spec()).max_span(), 1u);
}

// A phase no connection names is not part of any shot, so it is neither
// built nor checked: one of another width, without D_sparse, whose faults
// reach the next round changes neither the checks nor max_span.
TEST(ExperimentTemplate, IgnoresPhasesTheChainDoesNotUse) {
  auto spec = single_round_spec();
  auto spare = spec.phases.at(1);
  spare.id = phase_id::from_name("spare");
  spare.spec.seam_specs = {{seam_name::prev_round, {{0, -1, 1, -1}, {}}},
                           {seam_name::next_round, {{0, -1, 1, -1}, {}}}};
  spec.phases.push_back(spare);
  const experiment_template tmpl(spec);
  const experiment_template plain(single_round_spec());
  EXPECT_EQ(tmpl.max_span(), 0u);
  EXPECT_EQ(tmpl.min_rounds(), plain.min_rounds());
  EXPECT_EQ(tmpl.row_offset(6, 6), plain.row_offset(6, 6));
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

// Measurement offsets follow each phase's width in the stream, the readout
// map is the last phase's, and only the last suffix phase may differ from
// bulk in width. Without D_sparse a round measures one bit per detector,
// the memory convention, whatever a declared count says.
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
  for (auto spec : {arbitrary_memory_spec(5), rep5_spec()}) {
    spec.measurements_per_round = 7;
    const experiment_template counted(spec);
    for (std::size_t r = 0; r <= 5; ++r)
      EXPECT_EQ(counted.measurement_offset(r, 5), counted.row_offset(r, 5));
  }

  // A surface-code memory has two suffix phases: a bulk-width round, then a
  // wider readout. Only the readout, which FINAL identifies, may differ.
  phase_of(tail, phase_name::dem_final).num_measurements = 5;
  EXPECT_EQ(experiment_template(tail).measurement_offset(5, 5), 21u);
  phase_of(tail, phase_id::from_name("terminal0")).num_measurements = 5;
  expect_error([&] { experiment_template{tail}; }, "precedes FINAL");
} // end - TEST(ExperimentTemplate, MeasurementWidthsAndReadout)

// Without D_sparse, detector k of a round differences measurement k of that
// round and of the one before, so every round needs the same detectors, as
// the whole-shot map dem_chunks_to_d_sparse() requires. A wider round would
// read its own measurements as the previous round's, and a narrower one
// leaves the whole-shot map undefined. With D_sparse rows both are fine.
TEST(ExperimentTemplate, MemoryConventionNeedsUniformRounds) {
  auto widening = widening_spec(false);
  auto narrowing = narrow_readout_spec();
  for (auto &phase : narrowing.phases) {
    phase.spec.D_sparse.clear();
    phase.spec.num_measurements.reset();
  }
  widening.num_rounds = 6;
  narrowing.num_rounds = 6;
  expect_error([&] { experiment_template{widening}; },
               "experiment_template: without D_sparse rows",
               "phase 'init' has 2 detectors a round, but its outgoing seam "
               "reaches 4");
  EXPECT_THROW(dem_chunks_to_d_sparse(widening), std::invalid_argument);
  expect_error([&] { experiment_template{narrowing}; },
               "experiment_template: without D_sparse rows");
  EXPECT_THROW(dem_chunks_to_d_sparse(narrowing), std::invalid_argument);
  EXPECT_NO_THROW(experiment_template{widening_spec(true)});
  EXPECT_NO_THROW(experiment_template{narrow_readout_spec()});
} // end - TEST(ExperimentTemplate, MemoryConventionNeedsUniformRounds)

// ---------------------------------------------------------------------------
// Decoding windows
// ---------------------------------------------------------------------------

// Every layer of every layout the window tests sweep.
std::vector<window_request> sweep_requests() {
  std::vector<window_request> all;
  for (const auto &[c_a, b, c_b] : std::vector<std::array<std::int64_t, 3>>{
           {1, 1, 1}, {2, 1, 3}, {3, 2, 2}, {2, 3, 1}}) {
    const auto layers = layered(c_a, b, c_b);
    all.push_back(layers.a);
    all.push_back(layers.b);
  }
  all.push_back(sliding(2, 5));
  return all;
}

// The closed model of a whole R-round shot, and where each round's rows and
// each chunk's columns start in it, built without the template.
struct whole_shot {
  std::size_t rounds = 0;
  detector_error_model model;
  std::vector<std::size_t> row;
  std::vector<std::size_t> column;
};

// Expands spec to R rounds and closes it.
whole_shot make_whole_shot(dem_chunks_spec spec, std::size_t R) {
  whole_shot s;
  spec.num_rounds = R;
  const auto chunks = dem_chunks_from_spec(spec);
  s.rounds = R;
  s.model = dem_close_all(chunks);
  s.row = expanded_row_offsets(spec, R);
  s.column = dem_chunks_to_fault_offsets(chunks);
  return s;
}

// Entries (r, c) of the shot's H, for rows and columns given as lists.
std::vector<std::vector<uint8_t>> h_block(const whole_shot &s,
                                          const std::vector<std::size_t> &rows,
                                          std::size_t col_begin,
                                          std::size_t col_end) {
  std::vector<std::vector<uint8_t>> out;
  for (const auto r : rows) {
    auto &line = out.emplace_back();
    for (auto c = col_begin; c < col_end; ++c)
      line.push_back(s.model.detector_error_matrix.at({r, c}));
  }
  return out;
}

// The window's model as dense matrices, H padded with the logical-error
// columns it leaves out.
detector_error_model dense(const decoding_window &w) {
  return dense_model(w.model);
}

// Entries of the window model's H, for its rows and columns [begin, end).
std::vector<std::vector<uint8_t>>
window_h(const decoding_window &w, std::size_t col_begin, std::size_t col_end) {
  std::vector<std::vector<uint8_t>> out;
  const auto model = dense(w);
  const auto &h = model.detector_error_matrix;
  for (std::size_t r = 0; r < h.shape()[0]; ++r) {
    auto &line = out.emplace_back();
    for (auto c = col_begin; c < col_end; ++c)
      line.push_back(h.at({r, c}));
  }
  return out;
}

// The shot rows of rounds [begin, end).
std::vector<std::size_t> rows_of(const whole_shot &s, round_range rounds) {
  std::vector<std::size_t> out;
  for (auto r = s.row[rounds.begin]; r < s.row[rounds.end]; ++r)
    out.push_back(r);
  return out;
}

// Each window of every layer, for every shot length, has the shot's rows,
// imports and fault columns, and commits exactly its commit zone's chunks.
TEST(DecodingWindow, MatchesTheWholeShot) {
  for (const auto &spec :
       {rep5_spec(), arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3),
        narrow_readout_spec()}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (auto R = tmpl->min_rounds(); R <= 40; ++R) {
      const auto shot = make_whole_shot(spec, R);
      SCOPED_TRACE("R=" + std::to_string(R));
      for (const auto &request : sweep_requests()) {
        const auto instances = window_instances(request, R);
        for (auto k = instances.begin; k < instances.end; ++k) {
          const auto w = window_from_dem_chunks(tmpl, request, k, R);
          const auto model = dense(w);
          auto rows = rows_of(shot, w.rounds);
          const auto imports = rows_of(shot, w.output.imports);
          SCOPED_TRACE("k=" + std::to_string(k));
          rows.insert(rows.end(), imports.begin(), imports.end());
          ASSERT_EQ(w.num_own_rows + imports.size(), rows.size());
          ASSERT_EQ(w.num_fault_columns,
                    shot.column[w.rounds.end] - shot.column[w.rounds.begin]);
          EXPECT_EQ(window_h(w, 0, w.num_fault_columns),
                    h_block(shot, rows, shot.column[w.rounds.begin],
                            shot.column[w.rounds.end]));
          for (std::size_t c = 0; c < w.num_fault_columns; ++c) {
            const auto shot_c = shot.column[w.rounds.begin] + c;
            EXPECT_EQ(w.model.priors[c], shot.model.error_rates[shot_c]);
            for (std::size_t o = 0; o < tmpl->num_observables(); ++o)
              EXPECT_EQ(model.observables_flips_matrix.at({o, c}),
                        shot.model.observables_flips_matrix.at({o, shot_c}));
          }
          EXPECT_EQ(w.commit_columns.begin,
                    shot.column[w.commit.begin] - shot.column[w.rounds.begin]);
          EXPECT_EQ(w.commit_columns.end,
                    shot.column[w.commit.end] - shot.column[w.rounds.begin]);
        } // end - for(k)
      } // end - for(request)
    } // end - for(R)
  } // end - for(spec)
} // end - TEST(DecodingWindow, MatchesTheWholeShot)

// The A and B commit zones tile every shot exactly once.
TEST(DecodingWindow, CommitZonesTileEveryShot) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto layers = layered(2, 2, 3);
  for (std::size_t R = 3; R <= 30; ++R) {
    std::vector<int> covered(R, 0);
    for (const auto &request : {layers.a, layers.b}) {
      const auto instances = window_instances(request, R);
      for (auto k = instances.begin; k < instances.end; ++k) {
        const auto w = window_from_dem_chunks(tmpl, request, k, R);
        EXPECT_FALSE(w.commit.empty());
        for (auto r = w.commit.begin; r < w.commit.end; ++r)
          ++covered[r];
      }
    }
    EXPECT_EQ(covered, std::vector<int>(R, 1)) << "R=" << R;
  } // end - for(R)
}

// An open input adds one virtual-outcome column per seam detector, then one
// logical-error column per observable, all with the port's prior and never
// committed. The inner decoder sees only the virtual-outcome columns.
TEST(DecodingWindow, OpenInputAddsBoundaryColumns) {
  const auto spec = narrow_readout_spec();
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const auto a = layered(2, 2, 3).a;
  const std::size_t R = 12;
  const auto w = window_from_dem_chunks(tmpl, a, 1, R);
  const auto model = dense(w);
  const auto &h = model.detector_error_matrix;
  const auto &o = model.observables_flips_matrix;
  const auto first = w.num_fault_columns;
  const auto seam = tmpl->row_offset(w.rounds.begin + 1, R) -
                    tmpl->row_offset(w.rounds.begin, R);
  ASSERT_TRUE(w.input.exists);
  ASSERT_EQ(w.num_virtual_outcome_columns, seam);
  ASSERT_EQ(w.num_logical_error_columns, tmpl->num_observables());
  ASSERT_EQ(h.shape()[1], first + seam + tmpl->num_observables());
  EXPECT_EQ(w.num_decoder_columns(), first + seam);
  for (std::size_t j = 0; j < seam; ++j)
    for (std::size_t r = 0; r < h.shape()[0]; ++r) {
      EXPECT_EQ(h.at({r, first + j}), r == j ? 1 : 0);
      for (std::size_t i = 0; i < o.shape()[0]; ++i)
        EXPECT_EQ(o.at({i, first + j}), 0);
    }
  for (std::size_t i = 0; i < tmpl->num_observables(); ++i) {
    const auto c = first + seam + i;
    for (std::size_t r = 0; r < h.shape()[0]; ++r)
      EXPECT_EQ(h.at({r, c}), 0);
    for (std::size_t j = 0; j < o.shape()[0]; ++j)
      EXPECT_EQ(o.at({j, c}), j == i ? 1 : 0);
  }
  for (auto c = first; c < h.shape()[1]; ++c)
    EXPECT_EQ(w.model.priors[c], 0.01);
  EXPECT_LE(w.commit_columns.end, first);
} // end - TEST(DecodingWindow, OpenInputAddsBoundaryColumns)

// A window's chain through the chunk adapter that presents dem_chunks as
// EDEMs, written from its equations as sorted column lists. With to-seam
// rows P, interior rows I, from-seam rows N and observable rows O, chunk r
// is D = [P; I] F + [Id; 0] V_in, V_out = N F and L_out = L_in + O F, and
// wiring chunk r's output to chunk r + 1's input gives
// D_(r+1) = P_(r+1) F_(r+1) + N_r F_r. The chain has an input code block
// when its input is open, and none at round 0. Template chunks have no
// interior rows, so [P; I] is P.
struct adapter_chain {
  std::vector<std::vector<std::size_t>> d_from_f; ///< Per row: fault columns
  std::vector<std::vector<std::size_t>> d_from_v; ///< Per row: V_in entries
  std::vector<std::vector<std::size_t>> l_from_f; ///< Per observable
  std::size_t num_v_in = 0;
};

// Reads each chunk's P, N and O bands; fault columns count from the first
// chunk's.
adapter_chain chunk_adapter(const experiment_template &t, round_range rounds,
                            bool input_block, std::size_t R) {
  adapter_chain out;
  std::vector<std::vector<std::size_t>> incoming;
  std::size_t first = 0;
  const auto &seam = t.spec().seam;
  out.l_from_f.resize(t.num_observables());
  for (auto r = rounds.begin; r < rounds.end; ++r) {
    const auto &chunk = t.chunk(r, R);
    const auto h = chunk.H.to_nested_csr();
    const auto o = chunk.O.to_nested_csr();
    const auto &p = chunk.get_seam(seam.to_seam);
    std::vector<std::vector<std::size_t>> outgoing;
    EXPECT_EQ(chunk.num_interior_rows(), 0u);
    for (auto i = p.row_begin; i < p.row_end; ++i) {
      const auto j = i - p.row_begin;
      auto &row = out.d_from_f.emplace_back();
      auto &v = out.d_from_v.emplace_back();
      for (const auto c : h[i])
        row.push_back(first + c);
      if (r > rounds.begin)
        row.insert(row.end(), incoming[j].begin(), incoming[j].end());
      if (r == rounds.begin && input_block)
        v.push_back(j);
      std::sort(row.begin(), row.end());
    }
    if (r == rounds.begin && input_block)
      out.num_v_in = p.row_end - p.row_begin;
    if (chunk.has_seam(seam.from_seam)) {
      const auto &n = chunk.get_seam(seam.from_seam);
      for (auto i = n.row_begin; i < n.row_end; ++i) {
        auto &row = outgoing.emplace_back();
        for (const auto c : h[i])
          row.push_back(first + c);
      }
    }
    for (std::size_t i = 0; i < o.size(); ++i)
      for (const auto c : o[i])
        out.l_from_f[i].push_back(first + c);
    incoming = std::move(outgoing);
    first += chunk.num_faults();
  } // end - for(r)
  return out;
} // end - chunk_adapter()

// The columns in [begin, end) where row r of a dense matrix is set,
// counted from begin.
std::vector<std::size_t> row_columns(const cudaqx::tensor<uint8_t> &m,
                                     std::size_t r, std::size_t begin,
                                     std::size_t end) {
  std::vector<std::size_t> out;
  for (auto c = begin; c < end; ++c)
    if (m.at({r, c}))
      out.push_back(c - begin);
  return out;
}

// Adapter equivalence, for every variant of every layer the window tests
// sweep: the open input's boundary columns are the adapter's input code
// block, V_in through an identity onto the first round's rows and L_in
// straight to L_out, and the window's O over its fault columns is the
// chain's F -> L_out block, the frame update. Own rows match D over the
// fault columns too.
TEST(DecodingWindow, MatchesTheChunkAdapter) {
  for (const auto &spec :
       {rep5_spec(), arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3),
        narrow_readout_spec()}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    const auto K = tmpl->num_observables();
    for (auto R = tmpl->min_rounds(); R <= 25; ++R) {
      SCOPED_TRACE("R=" + std::to_string(R));
      for (const auto &request : sweep_requests()) {
        const auto instances = window_instances(request, R);
        for (auto k = instances.begin; k < instances.end; ++k) {
          const auto w = window_from_dem_chunks(tmpl, request, k, R);
          // Truncation has no adapter counterpart; the sweep has none.
          const bool block =
              w.input.exists && w.input.disposition == port_disposition::open;
          ASSERT_EQ(w.num_crossing_columns, 0u);
          const auto adapter = chunk_adapter(*tmpl, w.rounds, block, R);
          const auto model = dense(w);
          const auto &h = model.detector_error_matrix;
          const auto &o = model.observables_flips_matrix;
          const auto F = w.num_fault_columns;
          const auto V = w.num_virtual_outcome_columns;
          SCOPED_TRACE("k=" + std::to_string(k));
          ASSERT_EQ(w.num_own_rows, adapter.d_from_f.size());
          ASSERT_EQ(V, adapter.num_v_in);
          ASSERT_EQ(w.num_logical_error_columns, block ? K : 0);
          for (std::size_t r = 0; r < w.num_own_rows; ++r) {
            EXPECT_EQ(row_columns(h, r, 0, F), adapter.d_from_f[r]) << r;
            EXPECT_EQ(row_columns(h, r, F, F + V), adapter.d_from_v[r]) << r;
          }
          for (std::size_t r = 0; r < h.shape()[0]; ++r)
            EXPECT_TRUE(row_columns(h, r, F + V, h.shape()[1]).empty());
          for (std::size_t i = 0; i < K; ++i) {
            EXPECT_EQ(row_columns(o, i, 0, F), adapter.l_from_f[i]) << i;
            EXPECT_EQ(row_columns(o, i, F, h.shape()[1]),
                      block ? std::vector<std::size_t>{V + i}
                            : std::vector<std::size_t>{});
          }
        } // end - for(k)
      } // end - for(request)
    } // end - for(R)
  } // end - for(spec)
} // end - TEST(DecodingWindow, MatchesTheChunkAdapter)

// A closed input adds no columns: the previous chunk's straddlers arrive as
// deltas. An open output keeps the last chunk's straddlers with their outside
// rows omitted. Together they are get_pcm_for_rounds() without the start
// straddlers and with the end ones; a window ending at R has no end
// straddlers, and there the oracle's end flag would also take start ones.
TEST(DecodingWindow, ClosedInputAndOpenOutputMatchGetPcmForRounds) {
  const auto spec = arbitrary_memory_spec(5);
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const std::size_t R = 15;
  const auto shot = make_whole_shot(spec, R);
  const auto request = sliding(2, 5);
  const auto instances = window_instances(request, R);
  for (auto k = instances.begin; k < instances.end; ++k) {
    const auto w = window_from_dem_chunks(tmpl, request, k, R);
    const auto model = dense(w);
    const auto [pcm, first, last] = get_pcm_for_rounds(
        shot.model.detector_error_matrix, rep5_checks, w.rounds.begin,
        w.rounds.end - 1, false, w.output.exists);
    SCOPED_TRACE("k=" + std::to_string(k));
    EXPECT_EQ(w.num_virtual_outcome_columns + w.num_logical_error_columns, 0u);
    ASSERT_EQ(pcm.shape()[1], w.num_fault_columns);
    for (std::size_t r = 0; r < pcm.shape()[0]; ++r)
      for (std::size_t c = 0; c < pcm.shape()[1]; ++c)
        EXPECT_EQ(pcm.at({r, c}), model.detector_error_matrix.at({r, c}));
  } // end - for(k)
} // end - TEST(DecodingWindow, ClosedInputAndOpenOutputMatchGetPcmForRounds)

// A closed output imports M rounds past its wire by default, or the named
// rounds, against the window's own columns; the shot's ends are physical
// edges, not ports.
TEST(DecodingWindow, PortsAtTheShotEdgesAreNotPorts) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  auto b = layered(2, 2, 3).b;
  const std::size_t R = 12;
  const auto inner = window_from_dem_chunks(tmpl, b, 2, R);
  const auto a = layered(2, 2, 3).a;
  const auto first = window_from_dem_chunks(tmpl, a, 0, R);
  ASSERT_TRUE(inner.output.exists);
  EXPECT_EQ(inner.output.imports,
            (round_range{inner.rounds.end, inner.rounds.end + 1}));
  b.output.imports = round_range{3, 5};
  EXPECT_EQ(window_from_dem_chunks(tmpl, b, 2, R).output.imports,
            (round_range{inner.rounds.end, inner.rounds.end + 2}));

  // A[0] starts at round 0: no input port, so no boundary columns.
  EXPECT_FALSE(first.input.exists);
  EXPECT_EQ(first.num_virtual_outcome_columns, 0u);
  // In a 10-round shot the last B window ends at R: no output port, so no
  // imports.
  const auto instances = window_instances(b, 10);
  const auto last = window_from_dem_chunks(tmpl, b, instances.end - 1, 10);
  EXPECT_EQ(last.rounds.end, 10);
  EXPECT_FALSE(last.output.exists);
  EXPECT_TRUE(last.output.imports.empty());
} // end - TEST(DecodingWindow, PortsAtTheShotEdgesAreNotPorts)

// One detector per round: a measurement fault flips its round and the next,
// and a data fault only its own.
dem_chunks_spec one_check_spec() {
  auto spec = arbitrary_memory_spec(5);
  for (auto &phase : spec.phases) {
    const bool last = phase.id == phase_name::dem_final;
    phase.spec.H_sparse.clear();
    phase.spec.num_faults = last ? 1 : 2;
    phase.spec.O_sparse = {0, -1};
    phase.spec.error_rates.assign(phase.spec.num_faults, 0.01);
    phase.spec.seam_specs = {{seam_name::prev_round,
                              {last ? std::vector<std::int64_t>{0, -1}
                                    : std::vector<std::int64_t>{0, 1, -1},
                               {}}}};
    if (!last)
      phase.spec.seam_specs.push_back({seam_name::next_round, {{0, -1}, {}}});
  }
  return spec;
} // end - one_check_spec()

// The window spec's nine-SEC example. SEC i is round i; layer A buffers SEC
// 1-3 and 7-9 around commit SEC 4-6. With detectors 3, 4, 6 and 7 firing and
// e_3 = e_6 = 1, commitment keeps only e_6; the forward delta over 7-9 is
// [1, 0, 0] and the backward residual over 4-6 is [1, 0, 0].
TEST(DecodingWindow, NineSecExample) {
  const auto tmpl =
      std::make_shared<const experiment_template>(one_check_spec());
  const std::size_t R = 11;
  window_request a;
  a.period = 12;
  a.commit = round_range{4, 7};
  a.buffers = {{1, 4}, {7, 10}};
  a.input = {port_disposition::open, 0.01};
  a.output = {port_disposition::open, std::nullopt};
  a.sends = {{"to_b", send_mode::delta, {{7, 10}}},
             {"to_prev_b", send_mode::residual, {{4, 7}}}};
  const auto w = window_from_dem_chunks(tmpl, a, 0, R);
  ASSERT_EQ(w.rounds, (round_range{1, 10}));
  ASSERT_EQ(w.num_own_rows, 9u);
  // Each chunk has a measurement fault (column 0) and a data fault.
  const auto meas = [&](std::int64_t sec) {
    return static_cast<std::size_t>(2 * (sec - w.rounds.begin));
  };
  EXPECT_LT(meas(3), w.commit_columns.begin);
  EXPECT_GE(meas(6), w.commit_columns.begin);
  EXPECT_LT(meas(6), w.commit_columns.end);

  std::vector<uint8_t> d(w.model.h.num_rows(), 0);
  std::vector<uint8_t> e(w.model.o.num_cols(), 0);
  for (const std::int64_t sec : {3, 4, 6, 7})
    d[sec - w.rounds.begin] = 1;
  e[meas(3)] = e[meas(6)] = 1;
  const auto out = commit_window(w, d, e);
  std::vector<uint8_t> kept(e.size(), 0);
  kept[meas(6)] = 1;
  EXPECT_EQ(out.committed, kept);
  ASSERT_EQ(out.sends.size(), 2u);
  EXPECT_EQ(out.sends[0], (std::vector<uint8_t>{1, 0, 0}));
  EXPECT_EQ(out.sends[1], (std::vector<uint8_t>{1, 0, 0}));
  EXPECT_EQ(out.frame, (std::vector<uint8_t>{1}));
} // end - TEST(DecodingWindow, NineSecExample)

// Each delta is the shot's H on the send's rows times the committed faults,
// and each residual adds the window's detectors. Sends may overlap, and may
// cover rows no commit fault flips.
TEST(DecodingWindow, SendsMatchTheWholeShot) {
  const auto spec = arbitrary_memory_spec(5);
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const std::size_t R = 14;
  const auto shot = make_whole_shot(spec, R);
  auto a = layered(3, 2, 3).a;
  std::mt19937 rng(7);
  a.sends.push_back({"wide", send_mode::delta, {{-2, 5}}});
  a.sends.push_back({"far", send_mode::residual, {{-2, -1}, {4, 5}}});
  const auto instances = window_instances(a, R);
  for (auto k = instances.begin; k < instances.end; ++k) {
    const auto w = window_from_dem_chunks(tmpl, a, k, R);
    const auto model = dense(w);
    const auto &h = model.detector_error_matrix;
    std::vector<uint8_t> d(h.shape()[0]);
    std::vector<uint8_t> e(h.shape()[1]);
    SCOPED_TRACE("k=" + std::to_string(k));
    for (auto &bit : d)
      bit = rng() & 1;
    for (auto &bit : e)
      bit = rng() & 1;
    const auto out = commit_window(w, d, e);
    ASSERT_EQ(out.sends.size(), w.sends.size());
    for (std::size_t s = 0; s < w.sends.size(); ++s) {
      const auto &send = w.sends[s];
      ASSERT_EQ(out.sends[s].size(), send.rows.size());
      for (std::size_t i = 0; i < send.rows.size(); ++i) {
        const auto row = shot.row[w.rounds.begin] + send.rows[i];
        uint8_t delta = 0;
        for (auto c = w.commit_columns.begin; c < w.commit_columns.end; ++c)
          delta ^= shot.model.detector_error_matrix.at(
                       {row, shot.column[w.rounds.begin] + c}) &
                   e[c];
        const uint8_t want = send.mode == send_mode::residual
                                 ? (d[send.rows[i]] ^ delta)
                                 : delta;
        EXPECT_EQ(out.sends[s][i], want) << send.id << " row " << i;
      }
    } // end - for(s)
    // The default sends cover the first M rounds of a zone, clipped.
    EXPECT_EQ(w.sends[0].id, "forward");
    EXPECT_EQ(w.sends[1].id, "backward");
    EXPECT_EQ(w.sends[1].rounds,
              (std::vector<round_range>{{w.commit.begin, w.commit.begin + 1}}));
  } // end - for(k)
} // end - TEST(DecodingWindow, SendsMatchTheWholeShot)

// Applies an affine map to an input vector over GF(2).
std::vector<uint8_t> apply_map(const affine_map &map,
                               const std::vector<uint8_t> &input) {
  std::vector<uint8_t> out = map.offset;
  const auto rows = map.matrix.to_nested_csr();
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (const auto c : rows[r])
      out[r] ^= input[c];
  return out;
}

// m2d applied to a shot's measurements gives the shot's detectors on the
// window's own rows, then the virtual outcomes on the commit zone's outgoing
// wire and on the output port, with or without explicit D rows, and across
// rounds that widen or narrow.
TEST(DecodingWindow, M2dMatchesTheWholeShot) {
  std::mt19937 rng(11);
  for (const auto &spec :
       {arbitrary_memory_spec(5, true), wide_readout_spec(), rep5_spec(),
        narrow_readout_spec(), widening_spec(true)}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (std::size_t R = tmpl->min_rounds(); R <= 16; ++R) {
      auto fixed = spec;
      fixed.num_rounds = R;
      const auto d_rows = dem_chunks_to_d_sparse(fixed);
      const auto shot = make_whole_shot(spec, R);
      // Without explicit D, each round measures its detectors.
      const auto moff = [&](std::int64_t r) {
        return spec.has_D_sparse() ? tmpl->measurement_offset(r, R)
                                   : shot.row[r];
      };
      std::vector<uint8_t> m(moff(R));
      for (auto &bit : m)
        bit = rng() & 1;
      for (const auto &request : sweep_requests()) {
        const auto instances = window_instances(request, R);
        for (auto k = instances.begin; k < instances.end; ++k) {
          const auto w = window_from_dem_chunks(tmpl, request, k, R);
          const auto s = w.rounds.begin;
          const auto e = w.rounds.end;
          std::vector<uint8_t> input;
          SCOPED_TRACE("R=" + std::to_string(R) + " k=" + std::to_string(k));
          if (s > 0)
            input.assign(m.begin() + moff(s - 1), m.begin() + moff(s));
          ASSERT_EQ(input.size(), w.num_virtual_outcomes);
          input.insert(input.end(), m.begin() + moff(s), m.begin() + moff(e));
          const auto out = apply_map(w.m2d, input);
          std::vector<uint8_t> want;
          for (auto row = shot.row[s]; row < shot.row[e]; ++row) {
            uint8_t bit = 0;
            for (const auto c : d_rows[row])
              bit ^= m[c];
            want.push_back(bit);
          }
          // The wire after round r - 1 carries round r - 1's measurements:
          // first the commit zone's outgoing wire, unless it is the shot's
          // end, then the output port's wire if it is another one.
          std::vector<std::int64_t> wires;
          if (w.commit.end < static_cast<std::int64_t>(R))
            wires.push_back(w.commit.end);
          if (w.output.exists && e != w.commit.end)
            wires.push_back(e);
          for (const auto wire : wires)
            want.insert(want.end(), m.begin() + moff(wire - 1),
                        m.begin() + moff(wire));
          EXPECT_EQ(out, want);
        } // end - for(k)
      } // end - for(request)
    } // end - for(R)
  } // end - for(spec)
} // end - TEST(DecodingWindow, M2dMatchesTheWholeShot)

// Running v <- v_out(m, v) on one-round windows from round 0 gives every
// round's incoming virtual outcomes, as reading the previous round does.
TEST(DecodingWindow, M2dRecurrenceGivesTheIncomingVirtualOutcomes) {
  const auto spec = arbitrary_memory_spec(5, true);
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const std::size_t R = 9;
  window_request round;
  std::vector<uint8_t> m(tmpl->measurement_offset(R, R));
  std::vector<uint8_t> v;
  std::mt19937 rng(3);
  round.period = 1;
  round.commit = round_range{0, 1};
  round.input = {port_disposition::closed, std::nullopt};
  round.output = {port_disposition::open, std::nullopt};
  for (auto &bit : m)
    bit = rng() & 1;
  for (std::size_t r = 0; r + 1 < R; ++r) {
    const auto w = window_from_dem_chunks(tmpl, round, r, R);
    const auto begin = m.begin() + tmpl->measurement_offset(r, R);
    const auto end = m.begin() + tmpl->measurement_offset(r + 1, R);
    std::vector<uint8_t> input = v;
    input.insert(input.end(), begin, end);
    const auto out = apply_map(w.m2d, input);
    v.assign(out.begin() + w.num_own_rows, out.end());
    EXPECT_EQ(v, std::vector<uint8_t>(begin, end)) << "round " << r;
  }
} // end - TEST(DecodingWindow, M2dRecurrenceGivesTheIncomingVirtualOutcomes)

// m2o, when requested, reads the readout records of the last round.
TEST(DecodingWindow, M2oReadsTheReadoutRound) {
  const auto tmpl =
      std::make_shared<const experiment_template>(wide_readout_spec());
  const std::size_t R = 9;
  const auto b = layered(2, 2, 3).b;
  const auto instances = window_instances(b, R);
  const auto w = window_from_dem_chunks(tmpl, b, instances.end - 1, R, true);
  std::vector<uint8_t> m(w.num_measurements, 0);
  ASSERT_TRUE(w.m2o.has_value());
  ASSERT_EQ(w.rounds.end, static_cast<std::int64_t>(R));
  const auto readout = tmpl->measurement_offset(R - 1, R) -
                       tmpl->measurement_offset(w.rounds.begin, R);
  // The map reads records 0, 2 and 4 of the readout round.
  m[readout + 2] = 1;
  EXPECT_EQ(apply_map(*w.m2o, m), (std::vector<uint8_t>{1}));
  m[readout + 1] = 1;
  EXPECT_EQ(apply_map(*w.m2o, m), (std::vector<uint8_t>{1}));
  m[readout + 4] = 1;
  EXPECT_EQ(apply_map(*w.m2o, m), (std::vector<uint8_t>{0}));
  EXPECT_FALSE(window_from_dem_chunks(tmpl, b, 1, R).m2o.has_value());
  expect_error([&] { window_from_dem_chunks(tmpl, b, 1, R, true); },
               "readout round");
} // end - TEST(DecodingWindow, M2oReadsTheReadoutRound)

// Without D_sparse the stream is detector-packed, so m2o reads the readout
// round at its detector offsets, which is what the whole-shot memory
// convention gives. A readout map past the round's detectors is rejected
// when the spec is validated, even under a wider num_measurements, so
// construction never meets one.
TEST(DecodingWindow, M2oWithoutDSparseReadsTheDetectorPackedRound) {
  auto spec = arbitrary_memory_spec(5);
  phase_of(spec, phase_name::dem_final).m2o_sparse = {0, 3, -1};
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const std::size_t R = 9;
  const auto b = layered(2, 2, 3).b;
  const auto w =
      window_from_dem_chunks(tmpl, b, window_instances(b, R).end - 1, R, true);
  const auto first = tmpl->row_offset(w.rounds.begin, R);
  const auto readout = tmpl->row_offset(R - 1, R) - first;
  std::vector<uint8_t> m(w.num_measurements, 0);
  ASSERT_TRUE(w.m2o.has_value());
  ASSERT_EQ(w.num_measurements, tmpl->row_offset(R, R) - first);
  ASSERT_EQ(readout + 4, w.num_measurements);
  m[readout] = 1;
  EXPECT_EQ(apply_map(*w.m2o, m), (std::vector<uint8_t>{1}));
  m[readout + 3] = 1;
  EXPECT_EQ(apply_map(*w.m2o, m), (std::vector<uint8_t>{0}));
  m[readout + 1] = 1;
  EXPECT_EQ(apply_map(*w.m2o, m), (std::vector<uint8_t>{0}));

  spec.measurements_per_round = 4;
  phase_of(spec, phase_name::dem_final).num_measurements = 5;
  phase_of(spec, phase_name::dem_final).m2o_sparse = {0, 4, -1};
  expect_error([&] { experiment_template{spec}; },
               "m2o_sparse reads measurement 4, past the 4 measurements its "
               "round has; without D_sparse");
} // end - TEST(DecodingWindow, M2oWithoutDSparseReadsTheDetectorPackedRound)

// The spec's invalid constructions are each rejected with their own message.
// The measurement maps index a window's measurements with 32 bits, so a
// window whose rounds hold more is rejected before any index is formed,
// rather than wrapping.
TEST(DecodingWindow, RejectsMoreMeasurementsThanTheMapsIndex) {
  auto spec = arbitrary_memory_spec(5, true);
  for (auto &phase : spec.phases)
    phase.spec.num_measurements = (std::uint64_t{1} << 32) + 1;
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  expect_error([&] { window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, 12); },
               "window_from_dem_chunks: window 1 holds",
               "measurements, more than its maps can index");
}

TEST(DecodingWindow, RejectsInvalidRequests) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto good = layered(2, 2, 3).a;
  const auto rejects = [&](window_request request, const std::string &text) {
    expect_error([&] { window_from_dem_chunks(tmpl, request, 1, 12); }, text);
  };
  auto r = good;
  EXPECT_NO_THROW(window_from_dem_chunks(tmpl, good, 1, 12));

  r = good;
  r.period = 0;
  rejects(r, "period");
  r = good;
  r.commit.reset();
  rejects(r, "no commit zone");
  r = good;
  r.commit = round_range{1, 1};
  rejects(r, "empty");
  r = good;
  r.buffers[1] = {1, 4};
  rejects(r, "overlap");
  r = good;
  r.buffers[1] = {3, 5};
  rejects(r, "contiguous");
  r = good;
  r.input.disposition.reset();
  rejects(r, "input port has no disposition");
  r = good;
  r.output.disposition.reset();
  rejects(r, "output port has no disposition");
  for (const double prior : {0.0, 1.0, -0.5}) {
    r = good;
    r.input.prior = prior;
    rejects(r, "strictly between 0 and 1");
  }
  r = good;
  r.input.prior.reset();
  rejects(r, "needs a prior");
  r = layered(2, 2, 3).b;
  r.input.prior = 0.01;
  rejects(r, "closed input takes no prior");
  r = good;
  r.output.imports = round_range{4, 5};
  rejects(r, "open output imports nothing");
  r = layered(2, 2, 3).b;
  r.output.imports = round_range{4, 6};
  rejects(r, "start at the output wire");
  r = good;
  r.sends.push_back({"forward", std::nullopt, {}});
  rejects(r, "send 'forward' is repeated");
  r = good;
  r.sends.push_back({"extra", std::nullopt, {}});
  rejects(r, "send 'extra' needs a mode and rounds");
  r = good;
  r.sends.push_back({"twice", send_mode::delta, {{0, 2}, {1, 3}}});
  rejects(r, "send 'twice' repeats round 1");
  r = good;
  r.sends.push_back({"outside", send_mode::delta, {{3, 6}}});
  rejects(r, "send 'outside' reaches past the window");
  r = layered(2, 2, 3).b;
  r.sends = {{"forward", std::nullopt, {}}};
  rejects(r, "forward send needs a buffer zone after the commit zone");
  // Rounds and offsets lie within 2^31 of the anchor, so that the window
  // arithmetic stays within int64.
  constexpr std::int64_t far = std::int64_t{1} << 40;
  r = good;
  r.period = far;
  rejects(r, "period " + std::to_string(far) + " is beyond");
  r = good;
  r.offset = -far;
  rejects(r, "offset " + std::to_string(-far) + " is beyond");
  r = good;
  r.commit = round_range{0, far};
  r.buffers = {};
  rejects(r, "round " + std::to_string(far) + " is beyond");
  r = good;
  r.sends.push_back({"far", send_mode::delta, {{-far, 0}}});
  rejects(r, "round " + std::to_string(-far) + " is beyond");

  // Only the instances window_instances() lists exist.
  const auto instances = window_instances(good, 12);
  for (const auto k : {instances.begin - 1, instances.end})
    expect_error([&] { window_from_dem_chunks(tmpl, good, k, 12); },
                 "has no commit round");
} // end - TEST(DecodingWindow, RejectsInvalidRequests)

// ---------------------------------------------------------------------------
// Column-handling options
// ---------------------------------------------------------------------------

// Four detectors a round, per seam, with eight faults a chunk:
//   0, 1: detector 0 in both rounds, no observable (identical pair);
//   2, 3: detector 1 in both rounds; 2 flips the observable, 3 does not;
//   4, 5: detector 2; 4 also flips it in the next round, 5 does not;
//   6:    detector 3 in both rounds;
//   7:    silent: no detector, no observable.
dem_chunks_spec duplicate_columns_spec() {
  auto spec = arbitrary_memory_spec(5);
  const std::vector<std::int64_t> prev{0, 1, -1, 2, 3, -1, 4, 5, -1, 6, -1};
  const std::vector<std::int64_t> next{0, 1, -1, 2, 3, -1, 4, -1, 6, -1};
  for (auto &phase : spec.phases) {
    phase.spec.H_sparse.clear();
    phase.spec.num_faults = 8;
    phase.spec.O_sparse = {2, -1};
    phase.spec.error_rates = {0.01, 0.02, 0.03, 0.01, 0.02, 0.02, 0.02, 0.005};
    phase.spec.seam_specs = {{seam_name::prev_round, {prev, {}}}};
    if (phase.id != phase_name::dem_final)
      phase.spec.seam_specs.push_back({seam_name::next_round, {next, {}}});
  }
  return spec;
} // end - duplicate_columns_spec()

// Column c of a window's H, over its rows, and its O.
std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
column_of(const decoding_window &w, std::size_t c) {
  std::pair<std::vector<uint8_t>, std::vector<uint8_t>> out;
  const auto model = dense(w);
  const auto &h = model.detector_error_matrix;
  const auto &o = model.observables_flips_matrix;
  for (std::size_t r = 0; r < h.shape()[0]; ++r)
    out.first.push_back(h.at({r, c}));
  for (std::size_t i = 0; i < o.shape()[0]; ++i)
    out.second.push_back(o.at({i, c}));
  return out;
}

// The zone of an unmerged fault column: 0 for the commit zone, else 1 plus
// the buffer's index.
std::size_t zone_of(const decoding_window &w, std::size_t column,
                    std::size_t faults_per_round) {
  const auto round =
      w.rounds.begin + static_cast<std::int64_t>(column / faults_per_round);
  if (round >= w.commit.begin && round < w.commit.end)
    return 0;
  for (std::size_t b = 0; b < w.buffers.size(); ++b)
    if (round >= w.buffers[b].begin && round < w.buffers[b].end)
      return b + 1;
  return w.buffers.size() + 1;
}

// merge_identical merges same-zone columns with identical full rows and
// observable effect, with the combined prior, and records every source with
// the most likely first. It never merges across zones, a pair that differs
// only in rows omitted at the open output, or the boundary columns.
TEST(DecodingWindowColumns, MergeIdenticalMergesWithinAZone) {
  const auto tmpl =
      std::make_shared<const experiment_template>(duplicate_columns_spec());
  auto request = layered(2, 2, 3).a;
  const std::size_t R = 14;
  const auto plain = window_from_dem_chunks(tmpl, request, 1, R);
  request.columns.merge_identical = true;
  const auto w = window_from_dem_chunks(tmpl, request, 1, R);
  std::vector<int> seen(plain.num_fault_columns, 0);
  ASSERT_EQ(w.rounds, (round_range{3, 9}));
  ASSERT_EQ(plain.num_fault_columns, 48u);
  // Per zone of two chunks: 0 and 1 merge in each chunk, and the two silent
  // columns merge, leaving 16 - 2 - 1 = 13.
  EXPECT_EQ(w.num_fault_columns, 39u);
  ASSERT_EQ(w.fault_column_sources.size(), w.num_fault_columns);
  for (std::size_t c = 0; c < w.num_fault_columns; ++c) {
    const auto &sources = w.fault_column_sources[c];
    double prior = 0.0;
    ASSERT_FALSE(sources.empty());
    for (const auto source : sources) {
      ++seen[source];
      EXPECT_EQ(column_of(w, c), column_of(plain, source)) << c;
      EXPECT_EQ(zone_of(plain, source, 8), zone_of(plain, sources[0], 8));
      EXPECT_LE(plain.model.priors[source], plain.model.priors[sources[0]]);
      prior = prior + plain.model.priors[source] -
              2 * prior * plain.model.priors[source];
    }
    EXPECT_NEAR(w.model.priors[c], prior, 1e-12);
    const bool in_commit = zone_of(plain, sources[0], 8) == 0;
    EXPECT_EQ(in_commit,
              c >= w.commit_columns.begin && c < w.commit_columns.end);
  } // end - for(c)
  EXPECT_EQ(seen, std::vector<int>(plain.num_fault_columns, 1));
  // The commit zone's first chunk: 0 and 1 merge, reported against 1.
  const auto merged = w.commit_columns.begin;
  EXPECT_EQ(w.fault_column_sources[merged], (std::vector<std::size_t>{17, 16}));
  EXPECT_NEAR(w.model.priors[merged], 0.01 + 0.02 - 2 * 0.01 * 0.02, 1e-12);
  // In the last chunk, 4 and 5 agree inside the window; round 9 differs.
  EXPECT_EQ(column_of(plain, 44), column_of(plain, 45));
  for (const auto &sources : w.fault_column_sources)
    for (const auto source : sources)
      if (source == 44)
        EXPECT_EQ(sources, (std::vector<std::size_t>{44}));
  EXPECT_EQ(w.num_virtual_outcome_columns, plain.num_virtual_outcome_columns);
  EXPECT_EQ(w.num_logical_error_columns, plain.num_logical_error_columns);
} // end - TEST(DecodingWindowColumns, MergeIdenticalMergesWithinAZone)

// Without an option, every built column is its own source.
TEST(DecodingWindowColumns, DefaultKeepsEveryColumn) {
  const auto tmpl =
      std::make_shared<const experiment_template>(duplicate_columns_spec());
  const auto w = window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, 14);
  ASSERT_EQ(w.fault_column_sources.size(), w.num_fault_columns);
  for (std::size_t c = 0; c < w.num_fault_columns; ++c)
    EXPECT_EQ(w.fault_column_sources[c], (std::vector<std::size_t>{c}));
}

// conflicting_parallel: most_likely merges a same-zone class with identical
// full rows but different observable effects into one column with the most
// likely member's response; ties keep the first. Identical pairs stay unless
// merge_identical is also set.
TEST(DecodingWindowColumns, MostLikelyKeepsTheMoreLikelyResponse) {
  auto spec = duplicate_columns_spec();
  auto request = layered(2, 2, 3).a;
  const std::size_t R = 14;
  request.columns.conflicting_parallel = parallel_resolution::most_likely;
  const auto check = [&](const dem_chunks_spec &s, std::size_t reported,
                         double prior) {
    const auto tmpl = std::make_shared<const experiment_template>(s);
    const auto plain = window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, R);
    const auto w = window_from_dem_chunks(tmpl, request, 1, R);
    // Each chunk's 2 and 3 become one column: 48 - 6.
    ASSERT_EQ(w.num_fault_columns, 42u);
    // The commit zone's first chunk is unmerged columns 16..23.
    const auto c = w.commit_columns.begin + 2;
    EXPECT_EQ(w.fault_column_sources[c],
              (std::vector<std::size_t>{reported, 18 + 19 - reported}));
    EXPECT_EQ(column_of(w, c), column_of(plain, reported));
    EXPECT_NEAR(w.model.priors[c], prior, 1e-12);
    EXPECT_EQ(w.fault_column_sources[w.commit_columns.begin],
              (std::vector<std::size_t>{16}));
  };
  check(spec, 18, 0.03 + 0.01 - 2 * 0.03 * 0.01);
  for (auto &phase : spec.phases)
    phase.spec.error_rates[3] = 0.03;
  check(spec, 18, 0.03 + 0.03 - 2 * 0.03 * 0.03);
  for (auto &phase : spec.phases)
    phase.spec.error_rates[3] = 0.04;
  check(spec, 19, 0.03 + 0.04 - 2 * 0.03 * 0.04);
} // end - TEST(DecodingWindowColumns, MostLikelyKeepsTheMoreLikelyResponse)

// most_likely keeps the observable effect whose faults together are most
// likely, not the single most likely fault's: faults 2 and 8 flip the
// observable at 0.01 each, about 0.0198 together, and outweigh fault 3,
// which does not, at 0.015. The merged column reports against fault 2, the
// first of the winning effect's most likely faults.
TEST(DecodingWindowColumns, MostLikelyWeighsEachEffectsFaultsTogether) {
  auto spec = duplicate_columns_spec();
  const std::vector<std::int64_t> prev{0, 1, -1, 2, 3, 8, -1, 4, 5, -1, 6, -1};
  const std::vector<std::int64_t> next{0, 1, -1, 2, 3, 8, -1, 4, -1, 6, -1};
  for (auto &phase : spec.phases) {
    auto &p = phase.spec;
    p.num_faults = 9;
    p.error_rates = {0.01, 0.02, 0.01, 0.015, 0.02, 0.02, 0.02, 0.005, 0.01};
    p.O_sparse = {2, 8, -1};
    p.seam_specs = {{seam_name::prev_round, {prev, {}}}};
    if (phase.id != phase_name::dem_final)
      p.seam_specs.push_back({seam_name::next_round, {next, {}}});
  }
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  auto request = layered(2, 2, 3).a;
  request.columns.conflicting_parallel = parallel_resolution::most_likely;
  const auto plain = window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, 14);
  const auto w = window_from_dem_chunks(tmpl, request, 1, 14);
  // The commit zone's first chunk is unmerged columns 18..26: faults 2, 3
  // and 8 of it are columns 20, 21 and 26.
  const auto merged = std::find_if(
      w.fault_column_sources.begin(), w.fault_column_sources.end(),
      [](const auto &sources) {
        return std::find(sources.begin(), sources.end(), 20) != sources.end();
      });
  ASSERT_NE(merged, w.fault_column_sources.end());
  const auto c =
      static_cast<std::size_t>(merged - w.fault_column_sources.begin());
  const double x = 0.01 + 0.01 - 2 * 0.01 * 0.01;
  EXPECT_EQ(*merged, (std::vector<std::size_t>{20, 26, 21}));
  EXPECT_EQ(column_of(w, c), column_of(plain, 20));
  EXPECT_NEAR(w.model.priors[c], x + 0.015 - 2 * x * 0.015, 1e-12);
} // end - TEST(DecodingWindowColumns,
  // MostLikelyWeighsEachEffectsFaultsTogether)

// Each commit column's first conflicting parallel column. In the commit zone
// of rounds 5 and 6, faults 2 and 3 have the same full rows, but only 2
// flips the observable; 0 and 1 are identical, which is no conflict.
// most_likely merges the pairs, and merge_identical alone does not.
TEST(DecodingWindowColumns, ParallelConflictsNameTheObservableCase) {
  const auto tmpl =
      std::make_shared<const experiment_template>(duplicate_columns_spec());
  const auto observable = parallel_conflict::observable;
  const std::vector<parallel_columns> expected{{18, 19, observable},
                                               {19, 18, observable},
                                               {26, 27, observable},
                                               {27, 26, observable}};
  auto request = layered(2, 2, 3).a;
  const auto unmerged = [](const decoding_window &w) {
    std::vector<parallel_columns> out;
    for (const auto &p : w.parallel_conflicts)
      out.push_back({w.fault_column_sources[p.commit_column].front(),
                     w.fault_column_sources[p.other_column].front(),
                     p.conflict});
    return out;
  };
  EXPECT_EQ(window_from_dem_chunks(tmpl, request, 1, 14).parallel_conflicts,
            expected);
  request.columns.merge_identical = true;
  EXPECT_EQ(unmerged(window_from_dem_chunks(tmpl, request, 1, 14)), expected);
  request.columns.conflicting_parallel = parallel_resolution::most_likely;
  EXPECT_TRUE(
      window_from_dem_chunks(tmpl, request, 1, 14).parallel_conflicts.empty());
} // end - TEST(DecodingWindowColumns, ParallelConflictsNameTheObservableCase)

// Without a trailing buffer, the commit zone ends at the open output:
// faults 4 and 5 of round 6 agree on the window's rows, but 4 also flips
// round 7's detector 2.
TEST(DecodingWindowColumns, ParallelConflictsNameRowsOutsideTheWindow) {
  const auto tmpl =
      std::make_shared<const experiment_template>(duplicate_columns_spec());
  const auto observable = parallel_conflict::observable;
  const auto outside = parallel_conflict::outside_rows;
  auto request = layered(2, 2, 3).a;
  request.buffers = {{-2, 0}};
  request.sends = {{"backward", std::nullopt, {}}};
  const auto w = window_from_dem_chunks(tmpl, request, 1, 14);
  ASSERT_EQ(w.rounds, (round_range{3, 7}));
  EXPECT_EQ(w.parallel_conflicts,
            (std::vector<parallel_columns>{{18, 19, observable},
                                           {19, 18, observable},
                                           {26, 27, observable},
                                           {27, 26, observable},
                                           {28, 29, outside},
                                           {29, 28, outside}}));
}

// boundary: truncate replaces the open input's boundary columns with the
// previous chunk's crossing faults, restricted to the window's rows, with
// their own priors and never committed: get_pcm_for_rounds() with the start
// straddlers.
TEST(DecodingWindowColumns, TruncateMatchesGetPcmForRounds) {
  const auto spec = arbitrary_memory_spec(5);
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const std::size_t R = 15;
  const auto shot = make_whole_shot(spec, R);
  auto request = layered(2, 2, 3).a;
  request.input = {port_disposition::open, std::nullopt,
                   input_boundary::truncate};
  const auto instances = window_instances(request, R);
  for (auto k = instances.begin; k < instances.end; ++k) {
    const auto w = window_from_dem_chunks(tmpl, request, k, R);
    const auto model = dense(w);
    const auto [pcm, first, last] = get_pcm_for_rounds(
        shot.model.detector_error_matrix, rep5_checks, w.rounds.begin,
        w.rounds.end - 1, true, w.output.exists);
    const auto F = w.num_fault_columns;
    const auto C = w.num_crossing_columns;
    SCOPED_TRACE("k=" + std::to_string(k));
    EXPECT_EQ(w.num_virtual_outcome_columns + w.num_logical_error_columns, 0u);
    EXPECT_EQ(C, w.rounds.begin > 0 ? 4u : 0u);
    EXPECT_EQ(w.num_decoder_columns(), F + C);
    ASSERT_EQ(pcm.shape()[1], F + C);
    // get_pcm_for_rounds() keeps shot order: the crossing columns first.
    for (std::size_t r = 0; r < pcm.shape()[0]; ++r) {
      for (std::size_t c = 0; c < C; ++c)
        EXPECT_EQ(pcm.at({r, c}), model.detector_error_matrix.at({r, F + c}));
      for (std::size_t c = 0; c < F; ++c)
        EXPECT_EQ(pcm.at({r, C + c}), model.detector_error_matrix.at({r, c}));
    }
    for (std::size_t c = 0; c < C; ++c)
      EXPECT_EQ(w.model.priors[F + c],
                shot.model.error_rates[shot.column[w.rounds.begin - 1] + c]);
    EXPECT_LE(w.commit_columns.end, F);
  } // end - for(k)
} // end - TEST(DecodingWindowColumns, TruncateMatchesGetPcmForRounds)

// Truncated crossing columns never merge, even when identical in the window.
TEST(DecodingWindowColumns, TruncatedColumnsNeverMerge) {
  const auto tmpl =
      std::make_shared<const experiment_template>(duplicate_columns_spec());
  auto request = layered(2, 2, 3).a;
  request.input = {port_disposition::open, std::nullopt,
                   input_boundary::truncate};
  request.columns.merge_identical = true;
  const auto w = window_from_dem_chunks(tmpl, request, 1, 14);
  const auto F = w.num_fault_columns;
  // Faults 0, 1, 2, 3, 4 and 6 of round 2 reach round 3; 5 and 7 do not.
  ASSERT_EQ(w.num_crossing_columns, 6u);
  EXPECT_EQ(column_of(w, F), column_of(w, F + 1));
}

// The options' own invalid combinations.
TEST(DecodingWindowColumns, RejectsInvalidOptions) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  auto truncated = layered(2, 2, 3).a;
  auto closed = layered(2, 2, 3).b;
  truncated.input.boundary = input_boundary::truncate;
  closed.input.boundary = input_boundary::columns;
  expect_error([&] { window_from_dem_chunks(tmpl, truncated, 1, 12); },
               "a truncated input takes no prior");
  expect_error([&] { window_from_dem_chunks(tmpl, closed, 1, 12); },
               "a closed input takes no boundary");
}

// ---------------------------------------------------------------------------
// Variants and hot-path forms
// ---------------------------------------------------------------------------

// The distinct variant keys of a layer over shots [min_rounds, end).
std::set<variant_key> keys_up_to(const experiment_template &tmpl,
                                 const window_request &request,
                                 std::size_t end) {
  std::set<variant_key> keys;
  for (auto R = tmpl.min_rounds(); R < end; ++R) {
    const auto instances = window_instances(request, R);
    for (auto k = instances.begin; k < instances.end; ++k)
      keys.insert(window_variant(tmpl, request, k, R));
  }
  return keys;
}

// The layer's keys, counted by hand for arbitrary_memory_spec (one prefix
// and one suffix round, max_span 1). A[k] (period 5, rounds [-2, 4) of its
// anchor 5k) meets the prefix only at k = 0 and the shot's end while
// R - 5k <= 5, so 1 prefix key, 5 terminal keys, 3 short-shot pairs
// (R = 3, 4, 5) and the bulk. B[k] (rounds [5k - 3, 5k), importing round 5k)
// never meets the prefix and meets the end while R - (5k - 3) <= 4: 4
// terminal keys and the bulk. Longer probes add none.
TEST(WindowVariants, CountMatchesTheLayout) {
  const experiment_template tmpl(arbitrary_memory_spec(5));
  const auto layers = layered(2, 2, 3);
  for (const auto &[request, count] :
       {std::pair{layers.a, 10u}, std::pair{layers.b, 5u}}) {
    const auto end = variant_probe_rounds(tmpl, request);
    const auto keys = keys_up_to(tmpl, request, end);
    EXPECT_EQ(keys.size(), count);
    EXPECT_EQ(keys_up_to(tmpl, request, end + 50), keys);
    EXPECT_EQ(keys.count(variant_key{}), 1u);
  }
}

// Windows with equal keys are the same problem up to a shift of rounds.
TEST(WindowVariants, EqualKeysBuildTheSameWindow) {
  for (const auto &spec : {rep5_spec(), arbitrary_memory_spec(5, true),
                           multi_terminal_memory_spec(5, 3),
                           narrow_readout_spec(), duplicate_columns_spec()}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (const auto &request : sweep_requests()) {
      std::map<variant_key, decoding_window> first;
      const auto end = variant_probe_rounds(*tmpl, request) + 2 * 12;
      for (auto R = tmpl->min_rounds(); R < end; ++R) {
        const auto instances = window_instances(request, R);
        for (auto k = instances.begin; k < instances.end; ++k) {
          const auto key = window_variant(*tmpl, request, k, R);
          auto w = window_from_dem_chunks(tmpl, request, k, R);
          const auto [it, inserted] = first.emplace(key, w);
          if (!inserted)
            EXPECT_NO_THROW(check_same_window_up_to_shift(it->second, w))
                << "R=" << R << " k=" << k;
        }
      } // end - for(R)
    } // end - for(request)
  } // end - for(spec)
} // end - TEST(WindowVariants, EqualKeysBuildTheSameWindow)

// The bulk windows k_bulk and k_bulk + 1 agree for every layer, and the
// comparison names the first field two windows differ in.
TEST(WindowVariants, BulkWindowsAreTranslationInvariant) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto a = layered(2, 2, 3).a;
  auto other_prior = a;
  for (const auto &spec : {rep5_spec(), arbitrary_memory_spec(5, true)}) {
    const auto t = std::make_shared<const experiment_template>(spec);
    for (const auto &request : sweep_requests())
      EXPECT_NO_THROW(check_translation_invariance(t, request));
  }
  // No dem_chunks spec has a varying bulk, so compare windows that differ.
  expect_error(
      [&] {
        check_same_window_up_to_shift(window_from_dem_chunks(tmpl, a, 0, 20),
                                      window_from_dem_chunks(tmpl, a, 1, 20));
      },
      "windows 0 and 1 differ in their rounds");
  other_prior.input.prior = 0.02;
  expect_error(
      [&] {
        check_same_window_up_to_shift(
            window_from_dem_chunks(tmpl, a, 1, 20),
            window_from_dem_chunks(tmpl, other_prior, 2, 20));
      },
      "windows 1 and 2 differ in their input port");
} // end - TEST(WindowVariants, BulkWindowsAreTranslationInvariant)

// The model is in the kernel's forms: H over the decoder's columns and O
// over every column, both canonical CSC, and a prior per column. The dense
// view pads H with the logical-error columns, which touch no detector.
TEST(WindowModel, HoldsTheKernelForms) {
  const auto tmpl =
      std::make_shared<const experiment_template>(narrow_readout_spec());
  auto request = layered(2, 2, 3).a;
  for (const auto truncate : {false, true}) {
    if (truncate)
      request.input = {port_disposition::open, std::nullopt,
                       input_boundary::truncate};
    const auto w = window_from_dem_chunks(tmpl, request, 1, 14);
    const auto model = dense(w);
    const auto total = w.num_decoder_columns() + w.num_logical_error_columns;
    const std::vector<std::size_t> shape{w.model.h.num_rows(), total};
    SCOPED_TRACE(truncate ? "truncate" : "columns");
    EXPECT_EQ(w.num_logical_error_columns, truncate ? 0u : 1u);
    EXPECT_EQ(w.model.h.layout(), sparse_binary_matrix_layout::csc);
    EXPECT_EQ(w.model.o.layout(), sparse_binary_matrix_layout::csc);
    EXPECT_NO_THROW(w.model.h.validate_sorted_unique_indices());
    EXPECT_NO_THROW(w.model.o.validate_sorted_unique_indices());
    EXPECT_EQ(w.model.h.num_cols(), w.num_decoder_columns());
    EXPECT_EQ(w.model.o.num_cols(), total);
    EXPECT_EQ(w.model.priors.size(), total);
    ASSERT_EQ(model.detector_error_matrix.shape(), shape);
    EXPECT_EQ(model.error_rates, w.model.priors);
    for (std::size_t r = 0; r < shape[0]; ++r)
      for (auto c = w.num_decoder_columns(); c < total; ++c)
        EXPECT_EQ(model.detector_error_matrix.at({r, c}), 0);
  } // end - for(truncate)
} // end - TEST(WindowModel, HoldsTheKernelForms)

// The large-window test caps its own address space, which needs Linux's
// /proc to measure what the process already maps.
#if defined(__linux__)

// A capped address space proves nothing under a sanitizer, whose allocator
// aborts on a refused allocation and reserves address space of its own.
constexpr bool sanitized = QEC_TEST_SANITIZED;

// A repetition-code memory over n bits: n - 1 detectors a round.
dem_chunks_spec repetition_memory_spec(std::size_t n) {
  using index = sparse_binary_matrix::index_type;
  std::vector<std::vector<index>> checks(n - 1);
  for (std::size_t i = 0; i + 1 < n; ++i)
    checks[i] = {static_cast<index>(i), static_cast<index>(i + 1)};
  const auto hz = sparse_binary_matrix::from_nested_csr(
      static_cast<index>(n - 1), static_cast<index>(n), checks);
  const auto lz =
      sparse_binary_matrix::from_nested_csr(1, static_cast<index>(n), {{0}});
  return phenomenological_memory_spec(hz, lz, 0.01, 0.01);
}

// Bytes of address space the process maps.
std::size_t mapped_bytes() {
  std::ifstream statm("/proc/self/statm");
  std::size_t pages = 0;
  statm >> pages;
  return pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
}

// An A window over a 3000-bit repetition memory: 10 rounds of 2999 rows and
// 5999 fault columns, then 2999 boundary columns and 1 logical-error column.
// That is about 2 GB as a dense H and a few megabytes sparse, and it builds
// within a little address space beyond what the process maps. The child
// lowers only its soft limit, which needs no privilege, and proves the limit
// refuses a dense H before building, by reserving that much address space:
// an mmap the compiler cannot remove, as it may an unused malloc. Its exit
// code says which step failed: 1 the shape, 2 setting the limit, 3 a limit
// that admits the dense H.
TEST(DecodingWindowDeathTest, BuildsLargeWindowsSparsely) {
  if (sanitized)
    GTEST_SKIP() << "a sanitizer's allocator aborts on a refused allocation";
  constexpr std::size_t bits = 3000;          // 2999 detectors a round
  constexpr std::size_t headroom = 256 << 20; // under a seventh of dense H
  constexpr std::size_t rows = 10 * (bits - 1);
  constexpr std::size_t columns = 10 * (2 * bits - 1) + bits;
  const auto tmpl =
      std::make_shared<const experiment_template>(repetition_memory_spec(bits));
  const auto request = layered(4, 3, 6).a;
  const auto build = [&] {
    rlimit limit{};
    void *dense = MAP_FAILED;
    if (getrlimit(RLIMIT_AS, &limit) != 0)
      std::exit(2);
    limit.rlim_cur =
        std::min<rlim_t>(mapped_bytes() + headroom, limit.rlim_max);
    if (setrlimit(RLIMIT_AS, &limit) != 0)
      std::exit(2);
    dense = mmap(nullptr, rows * columns, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (dense != MAP_FAILED) {
      munmap(dense, rows * columns);
      std::exit(3);
    }
    const auto w = window_from_dem_chunks(tmpl, request, 1, 40);
    std::exit(w.model.h.num_rows() == rows && w.model.o.num_cols() == columns
                  ? 0
                  : 1);
  };
  EXPECT_EXIT(build(), testing::ExitedWithCode(0), "");
} // end - TEST(DecodingWindowDeathTest, BuildsLargeWindowsSparsely)

#endif // defined(__linux__)

// ---------------------------------------------------------------------------
// Window schedule and commitment validation
//
// With one-round chunks, max_span() is at most 1 and every zone has a round,
// so the layout's size conditions (b, c_A and c_B at least max_span()) hold
// for every valid request. The failure tests break the same conditions with
// explicit sends and ports instead.
// ---------------------------------------------------------------------------

// Both layers resolve observable conflicts by the more likely column.
window_graph with_most_likely(window_graph g) {
  for (auto &layer : g.layers)
    layer.request.columns.conflicting_parallel =
        parallel_resolution::most_likely;
  return g;
}

// Calls validate_commitment() on a fresh template of spec.
void validate(const dem_chunks_spec &spec, const window_graph &g) {
  validate_commitment(std::make_shared<const experiment_template>(spec),
                      g.layers, g.edges);
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

// Whether window k of request has a commit round in an R-round shot,
// computed from the anchor alone.
bool has_commit(const window_request &request, std::int64_t k, std::int64_t R) {
  const auto anchor = request.offset + k * request.period;
  return std::max<std::int64_t>(0, anchor + request.commit->begin) <
         std::min<std::int64_t>(R, anchor + request.commit->end);
}

// For every shot length, the schedule lists exactly the windows with a
// commit round, by layer then k, with the built window's zones, ports and
// sends, its variant, and one dependency per incoming edge on producer
// instance k + index_offset.
TEST(WindowSchedule, MatchesAnIndependentEnumeration) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  for (const auto &g :
       {parallel_graph(layered(2, 2, 3)), sliding_graph(2, 5)}) {
    for (auto R = tmpl->min_rounds(); R <= 30; ++R) {
      const auto schedule = schedule_windows(tmpl, g.layers, g.edges, R);
      const auto shot = static_cast<std::int64_t>(R);
      std::size_t i = 0;
      SCOPED_TRACE("R=" + std::to_string(R));
      EXPECT_EQ(schedule.num_rounds, R);
      for (const auto &layer : g.layers) {
        for (std::int64_t k = -10; k < 40; ++k) {
          std::vector<window_dependency> inputs;
          if (!has_commit(layer.request, k, shot))
            continue;
          ASSERT_LT(i, schedule.windows.size());
          const auto &s = schedule.windows[i++];
          const auto w = window_from_dem_chunks(tmpl, layer.request, k, R);
          EXPECT_EQ(s.node, layer.node);
          EXPECT_EQ(s.k, k);
          EXPECT_EQ(s.variant, window_variant(*tmpl, layer.request, k, R));
          EXPECT_EQ(s.rounds, w.rounds);
          EXPECT_EQ(s.commit, w.commit);
          EXPECT_EQ(s.buffers, w.buffers);
          EXPECT_EQ(s.input, w.input);
          EXPECT_EQ(s.output, w.output);
          EXPECT_EQ(s.sends, w.sends);
          for (const auto &e : g.edges) {
            if (e.to_node != layer.node)
              continue;
            const auto producer = std::find_if(
                g.layers.begin(), g.layers.end(),
                [&](const window_layer &l) { return l.node == e.from_node; });
            const auto j = k + e.index_offset;
            inputs.push_back({e.from_node, e.from_output, j, e.to_input,
                              has_commit(producer->request, j, shot)});
          }
          EXPECT_EQ(s.inputs, inputs);
        } // end - for(k)
      } // end - for(layer)
      EXPECT_EQ(i, schedule.windows.size());
    } // end - for(R)
  } // end - for(g)
} // end - TEST(WindowSchedule, MatchesAnIndependentEnumeration)

// A repeated node, an edge naming a node that is not a layer or a send its
// producer lacks, and an invalid request are rejected.
TEST(WindowSchedule, RejectsUnknownNamesAndInvalidRequests) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  auto g = parallel_graph(layered(2, 2, 3));
  const auto schedule = [&] { schedule_windows(tmpl, g.layers, g.edges, 10); };
  g.layers[1].node = "A";
  expect_error(schedule, "schedule_windows", "node 'A' is repeated");
  g = parallel_graph(layered(2, 2, 3));
  g.edges[0].from_node = "ingest";
  expect_error(schedule, "schedule_windows",
               "edge ingest.forward -> B.receive_0", "not a window layer");
  g = parallel_graph(layered(2, 2, 3));
  g.edges[1].from_output = "sideways";
  expect_error(schedule, "schedule_windows", "A has no send 'sideways'");
  g = parallel_graph(layered(2, 2, 3));
  g.layers[1].request.commit.reset();
  expect_error(schedule, "window request: no commit zone");
}

// The probe shots hold every variant key that any longer shot has, for
// every layer.
TEST(WindowSchedule, ProbeRoundsCoverEveryVariant) {
  for (const auto &spec :
       {arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3)}) {
    const experiment_template tmpl(spec);
    for (const auto &requests :
         {layered(2, 1, 3), layered(3, 2, 2), interior_commit(3, 2, 2)}) {
      const auto g = parallel_graph(requests);
      const auto probe = commitment_probe_rounds(tmpl, g.layers, g.edges);
      const auto period = static_cast<std::size_t>(requests.a.period);
      for (const auto &layer : g.layers) {
        std::set<variant_key> early;
        std::set<variant_key> late;
        for (auto R = tmpl.min_rounds(); R < probe + 4 * period; ++R) {
          const auto instances = window_instances(layer.request, R);
          for (auto k = instances.begin; k < instances.end; ++k) {
            const auto key = window_variant(tmpl, layer.request, k, R);
            late.insert(key);
            if (R < probe)
              early.insert(key);
          }
        }
        EXPECT_EQ(early, late) << layer.node;
      } // end - for(layer)
    } // end - for(requests)
  } // end - for(spec)
} // end - TEST(WindowSchedule, ProbeRoundsCoverEveryVariant)

// Layers of one graph share a period; a chain without a repeating phase
// admits one shot length.
TEST(WindowSchedule, ProbeRoundsNeedOnePeriod) {
  const experiment_template tmpl(arbitrary_memory_spec(5));
  auto g = parallel_graph(layered(2, 2, 3));
  auto chain = rep5_spec();
  g.layers[1].request.period = 4;
  expect_error([&] { commitment_probe_rounds(tmpl, g.layers, g.edges); },
               "commitment_probe_rounds", "periods 5 and 4");
  chain.connections.erase(chain.connections.begin() + 1);
  chain.num_rounds.reset();
  const experiment_template fixed(chain);
  EXPECT_EQ(commitment_probe_rounds(
                fixed, parallel_graph(layered(1, 1, 1)).layers, {}),
            fixed.min_rounds() + 1);
}

// ---------------------------------------------------------------------------
// Row finalization
// ---------------------------------------------------------------------------

// For every probe shot, read off the built windows' models: every row that
// a fault touches has exactly one finalizer, a window holding the row whose
// columns on it are all commit columns, and every detectable fault column
// of the shot is committed exactly once. validate_commitment() accepts it.
void expect_sound(const dem_chunks_spec &spec, const window_graph &g) {
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  const auto probe = commitment_probe_rounds(*tmpl, g.layers, g.edges);
  EXPECT_NO_THROW(validate_commitment(tmpl, g.layers, g.edges));
  for (auto R = tmpl->min_rounds(); R < probe; ++R) {
    auto fixed = spec;
    fixed.num_rounds = R;
    const auto shot = dem_close_all(dem_chunks_from_spec(fixed));
    const auto &h = shot.detector_error_matrix;
    std::vector<int> finalizers(h.shape()[0], 0);
    std::vector<int> commits(h.shape()[1], 0);
    SCOPED_TRACE("R=" + std::to_string(R));
    for (const auto &layer : g.layers) {
      const auto instances = window_instances(layer.request, R);
      for (auto k = instances.begin; k < instances.end; ++k) {
        const auto w = window_from_dem_chunks(tmpl, layer.request, k, R);
        const auto full = dense(w);
        const auto &model = full.detector_error_matrix;
        const auto first_row = tmpl->row_offset(w.rounds.begin, R);
        const auto first_column = tmpl->column_offset(w.rounds.begin, R);
        for (std::size_t i = 0; i < model.shape()[0]; ++i) {
          std::size_t columns = 0;
          bool all_commit = true;
          for (std::size_t c = 0; c < w.num_decoder_columns(); ++c) {
            if (!model.at({i, c}))
              continue;
            ++columns;
            all_commit = all_commit && c >= w.commit_columns.begin &&
                         c < w.commit_columns.end;
          }
          const auto row = i < w.num_own_rows
                               ? first_row + i
                               : tmpl->row_offset(w.output.imports.begin, R) +
                                     i - w.num_own_rows;
          if (columns > 0 && all_commit)
            ++finalizers[row];
        } // end - for(i)
        for (auto c = w.commit_columns.begin; c < w.commit_columns.end; ++c)
          for (const auto source : w.fault_column_sources[c])
            ++commits[first_column + source];
      } // end - for(k)
    } // end - for(layer)
    for (std::size_t r = 0; r < h.shape()[0]; ++r) {
      bool touched = false;
      for (std::size_t c = 0; c < h.shape()[1]; ++c)
        touched = touched || h.at({r, c});
      if (touched)
        EXPECT_EQ(finalizers[r], 1) << "row " << r;
    }
    for (std::size_t c = 0; c < h.shape()[1]; ++c) {
      bool detectable = false;
      for (std::size_t r = 0; r < h.shape()[0]; ++r)
        detectable = detectable || h.at({r, c});
      if (detectable)
        EXPECT_EQ(commits[c], 1) << "column " << c;
    }
  } // end - for(R)
} // end - expect_sound()

// F1-F4 hold for the default A/B layout and the interior-commit layout,
// with and without truncate, on the memory fixtures.
TEST(Commitment, ParallelWindowLayoutsFinalizeEveryRowOnce) {
  for (const auto &spec :
       {arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3)}) {
    for (const auto &requests :
         {layered(2, 1, 3), layered(3, 2, 2), layered(2, 3, 1),
          interior_commit(3, 2, 2), interior_commit(4, 2, 1)}) {
      SCOPED_TRACE("c_a=" + std::to_string(requests.a.commit->end) +
                   " period=" + std::to_string(requests.a.period));
      expect_sound(spec, parallel_graph(requests));
      expect_sound(spec, parallel_graph(with_truncate(requests)));
    }
  }
}

// A sliding window finalizes its commit zone, its forward delta reaching
// the next instance's first round.
TEST(Commitment, SlidingWindowFinalizesEveryRowOnce) {
  for (const auto &[step, size] :
       std::vector<std::pair<std::int64_t, std::int64_t>>{
           {1, 2}, {2, 5}, {3, 4}})
    expect_sound(arbitrary_memory_spec(5), sliding_graph(step, size));
}

// F1: B with an open input has a boundary column on its first round, where
// A[k-1] has its trailing buffer, so no window finalizes the round.
TEST(Commitment, RejectsARowWithoutAFinalizer) {
  auto requests = layered(2, 2, 3);
  requests.b.input = {port_disposition::open, 0.01};
  expect_error(
      [&] { validate(arbitrary_memory_spec(5), parallel_graph(requests)); },
      "validate_commitment: F1: row", "has no finalizer; held by A[");
}

// F1: without a leading buffer, and with a closed input, A[k] finalizes its
// first commit round, which B[k] imports and finalizes too.
TEST(Commitment, RejectsARowWithTwoFinalizers) {
  auto requests = layered(2, 2, 3);
  requests.a.buffers = {{2, 4}};
  requests.a.input = {port_disposition::closed, std::nullopt};
  expect_error(
      [&] { validate(arbitrary_memory_spec(5), parallel_graph(requests)); },
      "validate_commitment: F1: row", "has 2 finalizers: A[1], B[1]");
}

// A round no window commits is reported as F4, before the rows it leaves
// unsettled: here a layer importing it would finalize its rows, with the
// round's own faults on them uncommitted. F4 implies F2, which therefore
// needs no check of its own.
TEST(Commitment, RejectsAFaultNoWindowCommits) {
  window_graph g;
  window_request w;
  w.period = 3;
  w.commit = round_range{0, 2};
  w.input = {port_disposition::closed, std::nullopt};
  w.output = {port_disposition::closed, round_range{2, 3}};
  g.layers = {{"W", "w", w}};
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: F4: fault 8 (round 2) of the 3-round "
               "shot is committed by no window");
}

// F3: without the receive_1 edge, A[k] commits faults on the rows B[k]
// imports and finalizes but sends it nothing; without receive_0, the same
// holds for A[k-1] and B[k]'s first round.
TEST(Commitment, RejectsAMissingSendEdge) {
  auto g = parallel_graph(layered(2, 2, 3));
  g.edges.pop_back();
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: F3: row", "finalized by B[1]",
               "A[1] commits fault", "no send of A[1] to B[1] covers round 5");
  g = parallel_graph(layered(2, 2, 3));
  g.edges.erase(g.edges.begin());
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: F3: row", "finalized by B[1]",
               "A[0] commits fault", "no send of A[0] to B[1] covers round 2");
}

// F4: a second layer commits a round the sliding window commits, and sends
// it a delta there, so every row still has one finalizer and sees every
// commit on it. Every fault touches its own round only.
TEST(Commitment, RejectsAFaultCommittedTwice) {
  auto g = sliding_graph(2, 3);
  window_request d;
  d.period = 2;
  d.offset = 1;
  d.commit = round_range{0, 1};
  d.input = {port_disposition::open, 0.01};
  d.output = {port_disposition::open, std::nullopt};
  d.sends = {{"delta", send_mode::delta, {{0, 1}}}};
  g.layers.push_back({"D", "d", d});
  g.edges.push_back({"D", "delta", "W", "receive_1", 0});
  expect_error([&] { validate(single_round_spec(), g); },
               "validate_commitment: F4: fault 4 (round 1)",
               "committed by 2 windows: W[0], D[0]");
  // Without the delta edge, the rows would also fail F3; the double commit
  // is reported first, as the cause.
  g.edges.pop_back();
  expect_error([&] { validate(single_round_spec(), g); },
               "validate_commitment: F4: fault 4 (round 1)",
               "committed by 2 windows: W[0], D[0]");
}

// ---------------------------------------------------------------------------
// Sends and receives
// ---------------------------------------------------------------------------

// A forward delta moved into A's commit zone lands before B[k+1]'s rounds.
TEST(CommitmentSends, RejectsADeltaOutsideTheReceiversRounds) {
  auto requests = layered(2, 2, 3);
  requests.a.sends[0] = {"forward", send_mode::delta, {{1, 2}}};
  expect_error(
      [&] { validate(arbitrary_memory_spec(5), parallel_graph(requests)); },
      "validate_commitment: edge A.forward -> B.receive_0 (index_offset -1)",
      "A[0]'s delta 'forward' reaches round 1, outside B[1]'s rounds [2, 3)");
}

// Without index_offset: -1, B[k] reads A[k]'s forward delta, which lies
// past B[k]'s rounds.
TEST(CommitmentSends, RejectsAForwardEdgeWithoutItsIndexOffset) {
  auto g = parallel_graph(layered(2, 2, 3));
  g.edges[0].index_offset = 0;
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: edge A.forward -> B.receive_0",
               "A[1]'s delta 'forward' reaches round 7, outside B[1]'s "
               "rounds [2, 5)");
}

// A backward residual must cover exactly the receiver's imports: one moved
// into A's leading buffer, and one shorter than the imports, are rejected.
TEST(CommitmentSends, RejectsAResidualThatDiffersFromTheImports) {
  auto requests = layered(2, 2, 3);
  requests.a.sends[1] = {"backward", send_mode::residual, {{-1, 0}}};
  expect_error(
      [&] { validate(arbitrary_memory_spec(5), parallel_graph(requests)); },
      "validate_commitment: edge A.backward -> B.receive_1",
      "A[1]'s residual 'backward' covers rounds [4, 5), but B[1] imports "
      "[5, 6); round 4 differs");
  requests = layered(2, 2, 3);
  requests.b.output = {port_disposition::closed, round_range{3, 5}};
  expect_error(
      [&] { validate(arbitrary_memory_spec(5), parallel_graph(requests)); },
      "A[1]'s residual 'backward' covers rounds [5, 6), but B[1] imports "
      "[5, 7); round 6 differs");
}

// Imports take their values from exactly one residual: a second residual
// send over the same rounds is rejected.
TEST(CommitmentSends, RejectsTwoResidualsForOneImport) {
  auto requests = layered(2, 2, 3);
  requests.a.sends.push_back({"again", send_mode::residual, {{0, 1}}});
  auto g = parallel_graph(requests);
  g.edges.push_back({"A", "again", "B", "receive_2", 0});
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: B[1] of the",
               "imports rounds [5, 6) but receives 2 residuals");
}

// The probe shots reach every edge's producer: with an extra edge seven
// instances back, the range grows by six periods, and that edge's delta,
// which lands before its receiver's rounds, is rejected.
TEST(CommitmentSends, ProbesReachAProducerSeveralPeriodsBack) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  auto g = sliding_graph(2, 5);
  const auto probe = commitment_probe_rounds(*tmpl, g.layers, g.edges);
  g.edges.push_back({"W", "forward", "W", "receive_1", -7});
  EXPECT_EQ(commitment_probe_rounds(*tmpl, g.layers, g.edges), probe + 6 * 2);
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: edge W.forward -> W.receive_1 "
               "(index_offset -7)",
               "outside W[");
}

// The probe range and the schedule hold only offsets a shot can: an edge
// reaching back more than 2^31 instances, the most negative int64 included,
// is rejected instead of overflowing.
TEST(CommitmentSends, RejectsAnIndexOffsetBeyondAnyShot) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  for (const auto offset :
       {std::numeric_limits<std::int64_t>::min(), -(std::int64_t{1} << 40)}) {
    auto g = sliding_graph(2, 5);
    const auto text = "edge W.forward -> W.receive_1 (index_offset " +
                      std::to_string(offset) + ") reaches back beyond";
    g.edges.push_back({"W", "forward", "W", "receive_1", offset});
    expect_error([&] { commitment_probe_rounds(*tmpl, g.layers, g.edges); },
                 "commitment_probe_rounds: " + text);
    expect_error([&] { schedule_windows(tmpl, g.layers, g.edges, 20); },
                 "schedule_windows: " + text);
  }
}

// One producer's send reaching a receiver twice, through two edges, would
// cancel itself, so it is rejected naming both inputs.
TEST(CommitmentSends, RejectsASendDeliveredTwice) {
  auto g = parallel_graph(layered(2, 2, 3));
  g.edges.push_back({"A", "forward", "B", "receive_2", -1});
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: B[1] of the",
               "receives A[0]'s delta 'forward' twice, on receive_0 and "
               "receive_2");
}

// ---------------------------------------------------------------------------
// Parallel columns
// ---------------------------------------------------------------------------

// Fault 4 flips only the next round's detector 3, and fault 5 only its own
// round's: fault 4 of round r and fault 5 of round r + 1 are twins with the
// same full rows. Without twins, only fault 5 is added.
dem_chunks_spec twin_spec(bool twins) {
  auto spec = arbitrary_memory_spec(5);
  const std::vector<std::int64_t> own{0, -1, 1, -1, 2, -1, 3, 5, -1};
  const std::vector<std::int64_t> next{0, -1, 1, -1, 2, -1, 3, 4, -1};
  const std::vector<std::int64_t> own_only{0, -1, 1, -1, 2, -1, 3, 4, -1};
  for (auto &phase : spec.phases) {
    phase.spec.H_sparse.clear();
    phase.spec.num_faults = twins ? 6 : 5;
    phase.spec.error_rates.assign(phase.spec.num_faults, 0.02);
    phase.spec.seam_specs = {
        {seam_name::prev_round, {twins ? own : own_only, {}}}};
    if (phase.id != phase_name::dem_final)
      phase.spec.seam_specs.push_back(
          {seam_name::next_round,
           {twins ? next
                  : std::vector<std::int64_t>{0, -1, 1, -1, 2, -1, 3, -1},
            {}}});
  }
  return spec;
} // end - twin_spec()

// A commit column parallel to a buffer column with the same rows is the
// spec's merge case: the decoder's choice between them changes what is
// committed.
TEST(CommitmentParallel, RejectsATwinInAnotherZone) {
  const auto tmpl =
      std::make_shared<const experiment_template>(twin_spec(true));
  const auto w = window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, 14);
  // Rounds 3-8, six faults a round: round 5's fault 5 is round 4's fault
  // 4, and round 6's fault 4 is round 7's fault 5. Round 5's fault 4 and
  // round 6's fault 5 are twins in the commit zone, which is allowed.
  EXPECT_EQ(
      w.parallel_conflicts,
      (std::vector<parallel_columns>{{17, 10, parallel_conflict::other_zone},
                                     {22, 29, parallel_conflict::other_zone}}));
  expect_error(
      [&] { validate(twin_spec(true), parallel_graph(layered(2, 2, 3))); },
      "validate_commitment: A[0]", "commit column", "in another zone");
}

// Boundary-column twins of buffer columns are accepted. Without a leading
// buffer, a commit column's boundary or truncated twin is rejected.
TEST(CommitmentParallel, AcceptsBoundaryTwinsOfBufferColumns) {
  const auto tmpl =
      std::make_shared<const experiment_template>(twin_spec(false));
  auto requests = layered(2, 2, 3);
  const auto w = window_from_dem_chunks(tmpl, requests.a, 1, 14);
  const auto model = dense(w);
  const auto &h = model.detector_error_matrix;
  // Round 3's fault 4 has the same rows as boundary column 3.
  for (std::size_t r = 0; r < h.shape()[0]; ++r)
    EXPECT_EQ(h.at({r, 4}), h.at({r, w.num_fault_columns + 3}));
  EXPECT_TRUE(w.parallel_conflicts.empty());
  expect_sound(twin_spec(false), parallel_graph(requests));

  requests.a.buffers = {{2, 4}};
  for (const auto &a : {requests.a, with_truncate(requests).a}) {
    const auto edge = window_from_dem_chunks(tmpl, a, 1, 14);
    EXPECT_EQ(edge.parallel_conflicts,
              (std::vector<parallel_columns>{{4, edge.num_fault_columns + 3,
                                              parallel_conflict::boundary}}));
    expect_error(
        [&] { validate(twin_spec(false), parallel_graph({a, requests.b})); },
        "validate_commitment: A[", "a boundary or truncated column");
  }
} // end - TEST(CommitmentParallel, AcceptsBoundaryTwinsOfBufferColumns)

// Rep-5's faults 0 and 5 have the same full rows, but 0 flips the
// observable: reject fails, and most_likely keeps the more likely.
TEST(CommitmentParallel, MostLikelyResolvesAnObservableConflict) {
  const auto g = parallel_graph(layered(2, 2, 3));
  expect_error([&] { validate(rep5_spec(), g); }, "validate_commitment: A[0]",
               "different observable effect",
               "conflicting_parallel: most_likely");
  expect_sound(rep5_spec(), with_most_likely(g));
}

// ---------------------------------------------------------------------------
// Serial parallel-window reference
// ---------------------------------------------------------------------------

using pw_reference::make_decoder;
using pw_reference::observables_of;
using pw_reference::parallel_window_reference;
using pw_reference::syndrome_of;

// Phenomenological memory over the distance-d surface code's Z checks.
dem_chunks_spec surface_memory_spec(std::size_t distance, double p) {
  cudaqx::heterogeneous_map options;
  options.insert("distance", distance);
  const auto matrices =
      css_matrices_from_code(*code::get("surface_code", options));
  return phenomenological_memory_spec(matrices.hz, matrices.lz, p, p);
}

// Detection events of one fault column.
std::vector<std::uint8_t> detectors_of(const pw_reference::shot &s,
                                       std::size_t column) {
  std::vector<std::uint8_t> e(s.h.size(), 0);
  e[column] = 1;
  return syndrome_of(s, e);
}

// Row i of a sampled tensor.
std::vector<std::uint8_t> row_of(const cudaqx::tensor<std::uint8_t> &t,
                                 std::size_t i) {
  std::vector<std::uint8_t> row(t.shape()[1]);
  for (std::size_t j = 0; j < row.size(); ++j)
    row[j] = t.at({i, j});
  return row;
}

// Whether a correction is equivalent to a fault: the same detection events
// and the same observables, so they differ by at most a stabilizer.
bool equivalent(const pw_reference::shot &s,
                const std::vector<std::uint8_t> &correction,
                const std::vector<std::uint8_t> &fault) {
  return syndrome_of(s, correction) == syndrome_of(s, fault) &&
         observables_of(s, correction) == observables_of(s, fault);
}

// pymatching keeps the lighter of parallel edges, which boundary columns
// can create.
const cudaqx::heterogeneous_map matching{
    {"merge_strategy", std::string{"smallest_weight"}}};

// Every layout the single-fault test sweeps: the default A/B sizes, the
// interior-commit layout of the same sizes, and both with truncate.
std::vector<window_graph> swept_layouts() {
  return {parallel_graph(layered(2, 1, 3)),
          parallel_graph(layered(3, 2, 2)),
          parallel_graph(interior_commit(3, 2, 2)),
          parallel_graph(with_truncate(layered(2, 1, 3))),
          parallel_graph(with_truncate(interior_commit(3, 2, 2))),
          sliding_graph(2, 5)};
}

// Every window of g, at every shot length up to 30, as the reference and
// the commitment schedule enumerate them: same instances, zones, ports,
// imports, sends and producer instances.
void expect_schedule_matches(const dem_chunks_spec &spec,
                             const window_graph &g) {
  const auto tmpl = std::make_shared<const experiment_template>(spec);
  for (auto R = tmpl->min_rounds(); R <= 30; ++R) {
    const parallel_window_reference reference(spec, g, R, "single_error_lut");
    const auto schedule = schedule_windows(tmpl, g.layers, g.edges, R);
    const auto &windows = reference.windows();
    SCOPED_TRACE("R=" + std::to_string(R));
    ASSERT_EQ(windows.size(), schedule.windows.size());
    for (std::size_t i = 0; i < windows.size(); ++i) {
      const auto &w = windows[i];
      const auto &s = schedule.windows[i];
      SCOPED_TRACE(w.node + "[" + std::to_string(w.k) + "]");
      EXPECT_EQ(w.node, s.node);
      EXPECT_EQ(w.k, s.k);
      EXPECT_EQ(w.rounds, s.rounds);
      EXPECT_EQ(w.commit, s.commit);
      EXPECT_EQ(w.buffers, s.buffers);
      EXPECT_EQ(w.input_exists, s.input.exists);
      EXPECT_EQ(w.output_exists, s.output.exists);
      EXPECT_EQ(w.imports, s.output.imports);
      EXPECT_EQ(w.inputs, s.inputs);
      ASSERT_EQ(w.sends.size(), s.sends.size());
      for (std::size_t j = 0; j < w.sends.size(); ++j) {
        EXPECT_EQ(w.sends[j].id, s.sends[j].id);
        EXPECT_EQ(w.sends[j].mode, s.sends[j].mode);
        EXPECT_EQ(w.sends[j].rounds, s.sends[j].rounds);
      }
    } // end - for(i)
  } // end - for(R)
} // end - expect_schedule_matches()

// The reference's windows and dependencies are the commitment schedule's,
// which is built independently. next_only_spec has faults that reach the
// next round without touching their own, which still sets the width of the
// default sends and imports.
TEST(ParallelWindowReference, WindowsMatchTheSchedule) {
  for (const auto &spec : {arbitrary_memory_spec(5), next_only_spec()})
    for (const auto &g : swept_layouts())
      expect_schedule_matches(spec, g);
}

// The request of g's layer for node.
const window_request &request_of(const window_graph &g,
                                 const std::string &node) {
  return std::find_if(g.layers.begin(), g.layers.end(),
                      [&](const window_layer &l) { return l.node == node; })
      ->request;
}

// The rows of column c of a dense matrix.
std::vector<std::uint32_t> column_rows(const cudaqx::tensor<uint8_t> &m,
                                       std::size_t c) {
  std::vector<std::uint32_t> rows;
  for (std::size_t r = 0; r < m.shape()[0]; ++r)
    if (m.at({r, c}))
      rows.push_back(static_cast<std::uint32_t>(r));
  return rows;
}

// Every production window of g in an R-round shot, built by
// window_from_dem_chunks(), against the reference's slice of the whole shot:
// its rows, the decoder's columns over them with their priors, every shot
// column's observables, and which of its own faults it commits.
void expect_models_match(const std::shared_ptr<const experiment_template> &t,
                         const window_graph &g, std::size_t R) {
  const parallel_window_reference reference(t->spec(), g, R,
                                            "single_error_lut");
  const auto &s = reference.model();
  const auto &windows = reference.windows();
  SCOPED_TRACE("R=" + std::to_string(R));
  for (std::size_t i = 0; i < windows.size(); ++i) {
    const auto &rw = windows[i];
    const auto w = window_from_dem_chunks(t, request_of(g, rw.node), rw.k, R);
    const auto model = dense(w);
    const auto &h = model.detector_error_matrix;
    const auto &o = model.observables_flips_matrix;
    const auto &columns = reference.decoder_columns(i);
    SCOPED_TRACE(rw.node + "[" + std::to_string(rw.k) + "]");
    ASSERT_EQ(w.num_own_rows, rw.num_own_rows);
    ASSERT_EQ(h.shape()[0], rw.rows.size());
    ASSERT_EQ(w.num_decoder_columns(), columns.size());
    for (std::size_t c = 0; c < columns.size(); ++c) {
      EXPECT_EQ(column_rows(h, c), columns[c]) << "column " << c;
      EXPECT_EQ(w.model.priors[c], reference.priors(i)[c]) << c;
    }
    for (std::size_t c = 0; c < rw.columns.size(); ++c)
      EXPECT_EQ(column_rows(o, c), s.o[rw.columns[c]]) << "column " << c;
    for (std::size_t c = 0; c < rw.num_faults; ++c)
      EXPECT_EQ(rw.committed[c] != 0,
                c >= w.commit_columns.begin && c < w.commit_columns.end)
          << "column " << c;
  } // end - for(i)
} // end - expect_models_match()

// The production models agree with the reference's, which slices the whole
// shot independently, for every layout and shot length up to 30. The
// reference does not model the column options; the DecodingWindowColumns
// tests check merged windows against these unmerged ones instead.
TEST(ParallelWindowReference, ModelsMatchTheWindows) {
  for (const auto &spec :
       {arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3),
        surface_memory_spec(3, 0.01)}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (const auto &g : swept_layouts())
      for (auto R = tmpl->min_rounds(); R <= 30; ++R)
        expect_models_match(tmpl, g, R);
  }
}

// Every single detectable fault decodes to an equivalent correction, at
// every shot length, with one single_error_lut per window, and every window
// explains its input. This covers every straddler and every terminal clip.
// rep5_spec has two faults with the same rows and different observables,
// which no decoder can tell apart, so the sweep uses memories without them.
TEST(ParallelWindowReference, SingleFaultsDecodeExactly) {
  for (const auto &spec :
       {arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3),
        surface_memory_spec(3, 0.01)}) {
    for (const auto &g : swept_layouts()) {
      for (auto R = experiment_template(spec).min_rounds(); R <= 30; ++R) {
        parallel_window_reference reference(spec, g, R, "single_error_lut");
        const auto &s = reference.model();
        SCOPED_TRACE("R=" + std::to_string(R));
        for (std::size_t c = 0; c < s.h.size(); ++c) {
          std::vector<std::uint8_t> fault(s.h.size(), 0);
          if (s.h[c].empty())
            continue;
          fault[c] = 1;
          const auto got = reference.decode(detectors_of(s, c));
          EXPECT_TRUE(equivalent(s, got.errors, fault)) << "column " << c;
          EXPECT_TRUE(got.satisfied) << "column " << c;
        }
      } // end - for(R)
    } // end - for(g)
  } // end - for(spec)
} // end - TEST(ParallelWindowReference, SingleFaultsDecodeExactly)

// Without either send, some single fault decodes wrongly in each layout, so
// each send is needed for single faults to decode exactly. Without the
// forward delta, a measurement error in A's last commit round reaches B's
// first round, where a boundary data fault of B explains it a second time.
// Without the backward residual, B's imports keep detection events A's
// commit already explains, and B explains them again. PyMatching explains
// whatever it is given; single_error_lut commits nothing for a pattern no
// single column explains, which hides the missing residual.
TEST(ParallelWindowReference, SendsAreNeeded) {
  constexpr std::size_t R = 20; // two periods of each layout
  const auto spec = surface_memory_spec(3, 0.01);
  for (const auto &g : {parallel_graph(layered(2, 1, 3)),
                        parallel_graph(interior_commit(3, 2, 2))}) {
    parallel_window_reference reference(spec, g, R, "pymatching", matching);
    const auto &s = reference.model();
    for (const std::string send : {"forward", "backward"}) {
      std::size_t wrong = 0;
      for (std::size_t c = 0; c < s.h.size(); ++c) {
        std::vector<std::uint8_t> fault(s.h.size(), 0);
        const auto detectors = detectors_of(s, c);
        fault[c] = 1;
        EXPECT_TRUE(equivalent(s, reference.decode(detectors).errors, fault))
            << "column " << c;
        wrong +=
            !equivalent(s, reference.decode(detectors, {send}).errors, fault);
      }
      EXPECT_GT(wrong, 0u) << "without " << send;
    } // end - for(send)
  } // end - for(g)
} // end - TEST(ParallelWindowReference, SendsAreNeeded)

// commit_window() on the production window, from the reference's detectors
// and estimate, commits the reference's faults and gives its sends and frame
// update, which the reference takes from the whole shot.
void expect_commit_matches(const decoding_window &w,
                           const pw_reference::reference_window &rw,
                           const pw_reference::window_decode &d) {
  auto estimate = d.estimate;
  std::vector<std::size_t> committed;
  estimate.resize(w.num_decoder_columns() + w.num_logical_error_columns, 0);
  const auto out = commit_window(w, d.detectors, estimate);
  for (std::size_t c = 0; c < rw.num_faults; ++c)
    if (out.committed[c])
      committed.push_back(rw.columns[c]);
  ASSERT_EQ(committed, d.committed);
  ASSERT_EQ(out.sends, d.sends);
  ASSERT_EQ(out.frame, d.frame);
}

// On sampled shots where every window converged, the committed faults
// reproduce every detection event of the shot (the soundness invariant),
// and every window's solution explains its input (what verify_residual
// checks). From the reference's detectors and estimate, every production
// window commits the same faults and gives the same sends and frame update.
TEST(ParallelWindowReference, SampledShotsAreSound) {
  constexpr std::size_t R = 20;        // two periods of each layout
  constexpr std::size_t shots = 10000; // enough to reach every window variant
  for (const auto &spec :
       {arbitrary_memory_spec(5), surface_memory_spec(3, 0.02)}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (const auto &g : {parallel_graph(layered(3, 3, 9)),
                          parallel_graph(interior_commit(3, 3, 9)),
                          parallel_graph(with_truncate(layered(3, 3, 9)))}) {
      parallel_window_reference reference(spec, g, R, "pymatching", matching);
      const auto &s = reference.model();
      const auto &windows = reference.windows();
      const auto [detectors, errors] = dem_sampler::cpu::sample_dem(
          s.dem.detector_error_matrix, shots, s.dem.error_rates, 11);
      std::vector<decoding_window> production;
      std::size_t checked = 0;
      for (const auto &rw : windows)
        production.push_back(
            window_from_dem_chunks(tmpl, request_of(g, rw.node), rw.k, R));
      for (std::size_t i = 0; i < shots; ++i) {
        const auto syndrome = row_of(detectors, i);
        const auto got = reference.decode(syndrome);
        if (!got.converged)
          continue;
        ++checked;
        ASSERT_EQ(syndrome_of(s, got.errors), syndrome) << "shot " << i;
        ASSERT_TRUE(got.satisfied) << "shot " << i;
        for (std::size_t j = 0; j < windows.size(); ++j)
          ASSERT_NO_FATAL_FAILURE(
              expect_commit_matches(production[j], windows[j], got.windows[j]))
              << "shot " << i << ", window " << j;
      }
      EXPECT_EQ(checked, shots);
    } // end - for(g)
  } // end - for(spec)
} // end - TEST(ParallelWindowReference, SampledShotsAreSound)

// Windowed decoding agrees with monolithic matching on every shot with at
// most one fault, and the two are right on the same shots but for a few: the
// logical error rates, near 0.11 at d = 3 and 0.06 at d = 5, then differ by
// at most 0.0033. The bound leaves room for samples that differ between
// standard libraries; on this one the two disagree on at most 2 shots.
TEST(ParallelWindowReference, TracksMonolithicMatching) {
  constexpr std::size_t shots = 3000;     // a few hundred logical errors
  constexpr std::size_t disagreeing = 10; // shots one decoder alone gets right
  constexpr double p = 0.01;              // well below threshold
  for (const std::size_t d : {3, 5}) {
    const auto spec = surface_memory_spec(d, p);
    const auto c = static_cast<std::int64_t>(d);
    const std::size_t R = 8 * d + 1; // two periods of the layouts below
    for (const auto &g : {parallel_graph(layered(c, c, 3 * c)),
                          parallel_graph(interior_commit(c, c, 3 * c))}) {
      parallel_window_reference reference(spec, g, R, "pymatching", matching);
      const auto &s = reference.model();
      const auto monolithic = make_decoder(s.h, s.row.back(), s.dem.error_rates,
                                           "pymatching", matching);
      const auto [detectors, errors] = dem_sampler::cpu::sample_dem(
          s.dem.detector_error_matrix, shots, s.dem.error_rates, 13);
      std::size_t windowed_wrong = 0;
      std::size_t monolithic_wrong = 0;
      std::size_t differ = 0;
      for (std::size_t i = 0; i < shots; ++i) {
        const auto syndrome = row_of(detectors, i);
        const auto fault = row_of(errors, i);
        const auto truth = observables_of(s, fault);
        const auto windowed = reference.decode(syndrome).observables;
        const auto out = monolithic->decode(
            std::vector<float_t>(syndrome.begin(), syndrome.end()));
        std::vector<std::uint8_t> matched(out.result.size());
        for (std::size_t j = 0; j < matched.size(); ++j)
          matched[j] = convert_soft_to_hard(out.result[j]);
        const auto whole = observables_of(s, matched);
        if (std::accumulate(fault.begin(), fault.end(), 0u) <= 1)
          EXPECT_EQ(windowed, truth) << "d=" << d << " shot " << i;
        windowed_wrong += windowed != truth;
        monolithic_wrong += whole != truth;
        differ += (windowed != truth) != (whole != truth);
      } // end - for(i)
      EXPECT_LE(differ, disagreeing)
          << "d=" << d << ": windowed " << windowed_wrong << ", monolithic "
          << monolithic_wrong << " of " << shots;
    } // end - for(g)
  } // end - for(d)
} // end - TEST(ParallelWindowReference, TracksMonolithicMatching)

// The shot rows of a window's model rows: its own rounds' rows, then its
// imports', as the template lays out the shot.
std::vector<std::size_t> shot_rows(const decoding_window &w) {
  const auto &t = *w.context;
  const auto R = w.shot_rounds;
  std::vector<std::size_t> rows;
  for (auto r = t.row_offset(w.rounds.begin, R);
       r < t.row_offset(w.rounds.end, R); ++r)
    rows.push_back(r);
  if (!w.output.imports.empty())
    for (auto r = t.row_offset(w.output.imports.begin, R);
         r < t.row_offset(w.output.imports.end, R); ++r)
      rows.push_back(r);
  return rows;
}

// One production window's decode, with the detectors it decoded.
struct production_decode {
  std::vector<std::uint8_t> detectors;
  window_solution solution;
};

// Every window of g in an R-round shot, built, received and decoded by the
// production code alone, one at a time in the reference's order: a window's
// detectors come from the shot by the template's offsets, and what its
// producers sent arrives through receive_send(). Each window keeps one
// workspace for every shot, as a node would.
class production_windows {
public:
  production_windows(const std::shared_ptr<const experiment_template> &t,
                     const window_graph &g, std::size_t R,
                     const std::vector<std::size_t> &order,
                     const std::string &decoder_name,
                     const cudaqx::heterogeneous_map &params = {})
      : schedule_(schedule_windows(t, g.layers, g.edges, R)), order_(order) {
    for (std::size_t i = 0; i < schedule_.windows.size(); ++i) {
      const auto &s = schedule_.windows[i];
      index_[{s.node, s.k}] = i;
      windows_.push_back(
          window_from_dem_chunks(t, request_of(g, s.node), s.k, R));
      decoders_.push_back(
          make_window_decoder(windows_.back(), decoder_name, params));
      workspaces_.push_back(make_window_workspace(windows_.back()));
    }
  }

  // Each window's decode of one shot's detection events, in schedule order.
  std::vector<production_decode> decode(const std::vector<std::uint8_t> &shot) {
    std::vector<production_decode> out(windows_.size());
    for (const auto i : order_) {
      auto &d = out[i];
      for (const auto row : shot_rows(windows_[i]))
        d.detectors.push_back(shot[row]);
      for (const auto &input : schedule_.windows[i].inputs) {
        if (!input.exists)
          continue;
        const auto p = index_.at({input.from_node, input.k});
        const auto &sends = windows_[p].sends;
        const auto send = static_cast<std::size_t>(
            std::find_if(sends.begin(), sends.end(),
                         [&](const window_send &x) {
                           return x.id == input.from_output;
                         }) -
            sends.begin());
        receive_send(windows_[i], windows_[p], send,
                     out[p].solution.outputs.sends.at(send), d.detectors);
      }
      solve_window(windows_[i], decoders_[i].get(), d.detectors,
                   workspaces_[i]);
      d.solution = workspaces_[i].solution;
    } // end - for(i)
    return out;
  } // end - decode()

  const std::vector<decoding_window> &windows() const { return windows_; }

private:
  window_schedule schedule_;
  std::vector<std::size_t> order_;
  std::map<std::pair<std::string, std::int64_t>, std::size_t> index_;
  std::vector<decoding_window> windows_;
  std::vector<std::unique_ptr<decoder>> decoders_;
  std::vector<window_workspace> workspaces_;
};

// Each production window's decode equals the reference's bit for bit: its
// detectors after receiving, its estimate over the decoder's columns,
// convergence and H e = d, and the faults it commits, its sends and its frame
// update.
void expect_decodes_match(const parallel_window_reference &reference,
                          const production_windows &production,
                          const std::vector<production_decode> &got,
                          const pw_reference::result &want) {
  for (std::size_t i = 0; i < got.size(); ++i) {
    const auto &rw = reference.windows()[i];
    const auto &w = production.windows()[i];
    const auto &mine = got[i].solution;
    const auto &theirs = want.windows[i];
    std::vector<std::size_t> committed;
    SCOPED_TRACE(rw.node + "[" + std::to_string(rw.k) + "]");
    ASSERT_EQ(mine.outputs.committed.size(), w.model.o.num_cols());
    for (std::size_t c = 0; c < rw.num_faults; ++c)
      if (mine.outputs.committed[c])
        committed.push_back(rw.columns[c]);
    ASSERT_EQ(got[i].detectors, theirs.detectors);
    ASSERT_EQ(std::vector<std::uint8_t>(mine.estimate.begin(),
                                        mine.estimate.begin() +
                                            w.num_decoder_columns()),
              theirs.estimate);
    ASSERT_EQ(mine.converged, theirs.converged);
    ASSERT_EQ(mine.satisfied, theirs.satisfied);
    ASSERT_EQ(committed, theirs.committed);
    ASSERT_EQ(mine.outputs.sends, theirs.sends);
    ASSERT_EQ(mine.outputs.frame, theirs.frame);
  } // end - for(i)
} // end - expect_decodes_match()

// The production windows, decoded serially with their own decoders, decode
// every single fault as the reference does, window by window, for every
// layout and shot length up to 30.
TEST(ParallelWindowReference, ProductionDecodesSingleFaultsAlike) {
  for (const auto &spec :
       {arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3),
        surface_memory_spec(3, 0.01)}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (const auto &g : swept_layouts()) {
      for (auto R = tmpl->min_rounds(); R <= 30; ++R) {
        parallel_window_reference reference(spec, g, R, "single_error_lut");
        production_windows production(tmpl, g, R, reference.order(),
                                      "single_error_lut");
        const auto &s = reference.model();
        SCOPED_TRACE("R=" + std::to_string(R));
        for (std::size_t c = 0; c < s.h.size(); ++c) {
          const auto detectors = detectors_of(s, c);
          ASSERT_NO_FATAL_FAILURE(expect_decodes_match(
              reference, production, production.decode(detectors),
              reference.decode(detectors)))
              << "column " << c;
        }
      } // end - for(R)
    } // end - for(g)
  } // end - for(spec)
} // end - TEST(ParallelWindowReference, ProductionDecodesSingleFaultsAlike)

// The same on sampled shots with PyMatching, where windows commit several
// faults and send what they leave.
TEST(ParallelWindowReference, ProductionDecodesSampledShotsAlike) {
  constexpr std::size_t R = 20;       // two periods of each layout
  constexpr std::size_t shots = 2000; // a few hundred with several faults
  for (const auto &spec :
       {arbitrary_memory_spec(5), surface_memory_spec(3, 0.02)}) {
    const auto tmpl = std::make_shared<const experiment_template>(spec);
    for (const auto &g : {parallel_graph(layered(3, 3, 9)),
                          parallel_graph(interior_commit(3, 3, 9)),
                          parallel_graph(with_truncate(layered(3, 3, 9)))}) {
      parallel_window_reference reference(spec, g, R, "pymatching", matching);
      production_windows production(tmpl, g, R, reference.order(), "pymatching",
                                    matching);
      const auto &s = reference.model();
      const auto [detectors, errors] = dem_sampler::cpu::sample_dem(
          s.dem.detector_error_matrix, shots, s.dem.error_rates, 17);
      for (std::size_t i = 0; i < shots; ++i) {
        const auto syndrome = row_of(detectors, i);
        ASSERT_NO_FATAL_FAILURE(expect_decodes_match(
            reference, production, production.decode(syndrome),
            reference.decode(syndrome)))
            << "shot " << i;
      }
    } // end - for(g)
  } // end - for(spec)
} // end - TEST(ParallelWindowReference, ProductionDecodesSampledShotsAlike)

// A send reaches its receiver only on the rows it is meant for: a delta on
// its own rows and a residual on its imports, in the same shot, one value per
// row. Delivered to the wrong window, or with the wrong values, it is
// rejected.
TEST(DecodingWindow, ReceiveSendRejectsRowsItCannotTake) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto layers = layered(2, 2, 3);
  const std::size_t R = 20;
  const auto a = window_from_dem_chunks(tmpl, layers.a, 1, R);
  const auto b = window_from_dem_chunks(tmpl, layers.b, 1, R);
  const auto next_b = window_from_dem_chunks(tmpl, layers.b, 2, R);
  const auto other_shot = window_from_dem_chunks(tmpl, layers.b, 2, R + 1);
  const std::vector<std::uint8_t> forward(a.sends[0].rows.size(), 1);
  const std::vector<std::uint8_t> backward(a.sends[1].rows.size(), 1);
  auto detectors = std::vector<std::uint8_t>(next_b.model.h.num_rows(), 0);
  ASSERT_EQ(a.sends[0].id, "forward");
  ASSERT_EQ(a.sends[1].id, "backward");
  // A[1].forward reaches B[2]'s own rows; A[1].backward is B[1]'s imports.
  receive_send(next_b, a, 0, forward, detectors);
  EXPECT_EQ(std::count(detectors.begin(), detectors.end(), 1),
            static_cast<std::ptrdiff_t>(forward.size()));
  detectors.assign(b.model.h.num_rows(), 0);
  receive_send(b, a, 1, backward, detectors);
  EXPECT_EQ(std::count(detectors.begin(), detectors.end(), 1),
            static_cast<std::ptrdiff_t>(backward.size()));
  EXPECT_THROW(receive_send(b, a, 0, forward, detectors),
               std::invalid_argument);
  EXPECT_THROW(receive_send(next_b, a, 1, backward, detectors),
               std::invalid_argument);
  EXPECT_THROW(receive_send(next_b, a, 0, {1}, detectors),
               std::invalid_argument);
  EXPECT_THROW(receive_send(other_shot, a, 0, forward, detectors),
               std::invalid_argument);
  EXPECT_THROW(receive_send(next_b, a, 2, forward, detectors),
               std::invalid_argument);
} // end - TEST(DecodingWindow, ReceiveSendRejectsRowsItCannotTake)

// A window with nothing to decode has no decoder: its estimate is zero, it
// commits nothing, and it explains its detectors only when they are zero.
TEST(DecodingWindow, AWindowWithoutADecoderCommitsNothing) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto w = window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, 20);
  std::vector<std::uint8_t> detectors(w.model.h.num_rows(), 0);
  auto ws = make_window_workspace(w);
  solve_window(w, nullptr, detectors, ws);
  const auto quiet = ws.solution;
  detectors[0] = 1;
  solve_window(w, nullptr, detectors, ws);
  const auto loud = ws.solution;
  EXPECT_EQ(quiet.estimate, std::vector<std::uint8_t>(w.model.o.num_cols(), 0));
  EXPECT_TRUE(quiet.satisfied);
  EXPECT_FALSE(loud.satisfied);
  EXPECT_EQ(loud.outputs.frame,
            std::vector<std::uint8_t>(w.model.o.num_rows(), 0));
  EXPECT_NE(make_window_decoder(w, "single_error_lut"), nullptr);
}

// A decoder whose params make it misreport what a window decoder must be:
// observables instead of fault columns, or one detector or column more than
// the window has.
class window_misfit_probe final : public decoder {
public:
  window_misfit_probe(decoder_init inputs,
                      const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs),
                decoder_output_request{params.get<bool>("observables", false)
                                           ? decode_result_type::observables
                                           : decode_result_type::errors}) {
    syndrome_size += params.get<std::size_t>("extra_rows", 0);
    block_size += params.get<std::size_t>("extra_columns", 0);
  }

  decoder_result decode(const std::vector<float_t> &) override {
    return {true, std::vector<float_t>(block_size, 0.0)};
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      window_misfit_probe,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decoder_output_request>,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<window_misfit_probe>(std::move(inputs), params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(window_misfit_probe)

// A window's decoder must return fault columns, take the window's rows and
// return its decoder columns; any other is rejected when it is made.
TEST(DecodingWindow, MakeWindowDecoderRejectsADecoderThatDoesNotFit) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto w = window_from_dem_chunks(tmpl, layered(2, 2, 3).a, 1, 20);
  const auto rows = std::to_string(w.model.h.num_rows());
  const auto columns = std::to_string(w.num_decoder_columns());
  const auto made = [&](const cudaqx::heterogeneous_map &params) {
    return
        [&w, params] { make_window_decoder(w, "window_misfit_probe", params); };
  };
  EXPECT_NE(make_window_decoder(w, "window_misfit_probe"), nullptr);
  expect_error(made({{"observables", true}}),
               "make_window_decoder: decoder 'window_misfit_probe' returns "
               "observables");
  expect_error(made({{"extra_rows", std::size_t{1}}}),
               "make_window_decoder: decoder 'window_misfit_probe' takes " +
                   std::to_string(w.model.h.num_rows() + 1) +
                   " detectors and returns " + columns,
               "window 1 has " + rows + " rows and " + columns);
  expect_error(made({{"extra_columns", std::size_t{1}}}),
               "takes " + rows + " detectors and returns " +
                   std::to_string(w.num_decoder_columns() + 1));
} // end - TEST(DecodingWindow, MakeWindowDecoderRejectsADecoderThatDoesNotFit)

// A workspace serves only a window of its own size: solving or committing on
// another window's is rejected before anything is written.
TEST(DecodingWindow, AWorkspaceServesOnlyItsOwnWindow) {
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto layers = layered(2, 2, 3);
  const auto a = window_from_dem_chunks(tmpl, layers.a, 1, 20);
  const auto b = window_from_dem_chunks(tmpl, layers.b, 1, 20);
  const std::vector<std::uint8_t> detectors(a.model.h.num_rows(), 0);
  auto a_ws = make_window_workspace(a);
  auto b_ws = make_window_workspace(b);
  ASSERT_NE(a.model.h.num_rows(), b.model.h.num_rows());
  expect_error([&] { solve_window(a, nullptr, detectors, b_ws); },
               "solve_window: the workspace is not sized for window 1");
  expect_error([&] { commit_window(a, detectors, b_ws); },
               "commit_window: the workspace is not sized for window 1");
  // One value short of a send.
  a_ws.solution.outputs.sends.back().pop_back();
  expect_error([&] { commit_window(a, detectors, a_ws); },
               "commit_window: the workspace is not sized for window 1");
}

// Decodes with @p inner, counting what each of its decode() calls
// allocates, so a test can tell its caller's allocations from the decoder's
// own in the same call.
class allocation_counting_decoder final : public decoder {
public:
  allocation_counting_decoder(const decoding_window &window,
                              std::unique_ptr<decoder> inner)
      : decoder(decoder_init(window.model.h)), inner_(std::move(inner)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    last_decode_allocations_ =
        allocation_counter::during([&] { result = inner_->decode(syndrome); });
    return result;
  }

  std::size_t last_decode_allocations() const {
    return last_decode_allocations_;
  }

private:
  std::unique_ptr<decoder> inner_;
  std::size_t last_decode_allocations_ = 0;
};

// On a workspace sized beforehand, receiving a send, solving a window
// without a decoder and committing allocate nothing, and solving with a
// decoder allocates only what the decoder's own decode() does in that call.
// The results match a fresh workspace's, so nothing is left over from the
// last shot.
TEST(DecodingWindow, TheDecodeStepAllocatesNothingPerShot) {
  if (QEC_TEST_SANITIZED)
    GTEST_SKIP() << "a sanitizer's allocator replaces the counting operator "
                    "new";
  const auto tmpl =
      std::make_shared<const experiment_template>(arbitrary_memory_spec(5));
  const auto layers = layered(2, 2, 3);
  const std::size_t R = 20;
  const auto a = window_from_dem_chunks(tmpl, layers.a, 1, R);
  const auto next_b = window_from_dem_chunks(tmpl, layers.b, 2, R);
  allocation_counting_decoder dec(a,
                                  make_window_decoder(a, "single_error_lut"));
  const std::vector<std::uint8_t> forward(a.sends[0].rows.size(), 1);
  std::vector<std::uint8_t> b_detectors(next_b.model.h.num_rows(), 0);
  std::vector<std::uint8_t> a_detectors(a.model.h.num_rows(), 0);
  auto a_ws = make_window_workspace(a);
  auto b_ws = make_window_workspace(next_b);
  const auto allocations = [](const auto &run) {
    return allocation_counter::during(run);
  };
  // The detectors fault column c of A flips.
  const auto fault = [&](std::size_t c) {
    std::vector<std::uint8_t> d(a.model.h.num_rows(), 0);
    for (auto p = a.model.h.ptr()[c]; p < a.model.h.ptr()[c + 1]; ++p)
      d[a.model.h.indices()[p]] = 1;
    return d;
  };
  // ws holds what a fresh workspace's decode of the same detectors gives.
  const auto expect_fresh = [](const decoding_window &w, decoder *d,
                               const std::vector<std::uint8_t> &detectors,
                               const window_workspace &ws) {
    auto fresh = make_window_workspace(w);
    solve_window(w, d, detectors, fresh);
    EXPECT_EQ(ws.solution.estimate, fresh.solution.estimate);
    EXPECT_EQ(ws.solution.satisfied, fresh.solution.satisfied);
    EXPECT_EQ(ws.solution.outputs.committed, fresh.solution.outputs.committed);
    EXPECT_EQ(ws.solution.outputs.sends, fresh.solution.outputs.sends);
    EXPECT_EQ(ws.solution.outputs.frame, fresh.solution.outputs.frame);
  };
  // The counter sees plain and aligned allocations alike.
  ASSERT_EQ(allocations([] {
              constexpr std::align_val_t line{64}; // a cache line
              ::operator delete(::operator new(8));
              ::operator delete(::operator new(8, line), line);
            }),
            2u);
  // A first shot commits A's first commit column, leaving its estimate,
  // sends and frame in the workspace.
  a_detectors = fault(a.commit_columns.begin);
  solve_window(a, &dec, a_detectors, a_ws);
  ASSERT_EQ(a_ws.solution.outputs.committed[a.commit_columns.begin], 1);
  // The next shot commits another.
  a_detectors = fault(a.commit_columns.begin + 1);
  EXPECT_EQ(
      allocations([&] { receive_send(next_b, a, 0, forward, b_detectors); }),
      0u);
  EXPECT_EQ(
      allocations([&] { solve_window(next_b, nullptr, b_detectors, b_ws); }),
      0u);
  EXPECT_EQ(allocations([&] { commit_window(next_b, b_detectors, b_ws); }), 0u);
  // Counted first, so that the decoder's count read next is this call's.
  const auto solved =
      allocations([&] { solve_window(a, &dec, a_detectors, a_ws); });
  EXPECT_EQ(solved, dec.last_decode_allocations());
  EXPECT_EQ(a_ws.solution.outputs.committed[a.commit_columns.begin], 0);
  expect_fresh(a, &dec, a_detectors, a_ws);
  expect_fresh(next_b, nullptr, b_detectors, b_ws);
} // end - TEST(DecodingWindow, TheDecodeStepAllocatesNothingPerShot)

// ---------------------------------------------------------------------------
// Ingest core
// ---------------------------------------------------------------------------

// multi_terminal_memory_spec with three suffix phases, the first of which
// forms its detectors from its own measurements alone, unlike bulk, so a
// suffix round taken for bulk would form other detectors.
dem_chunks_spec distinct_suffix_spec() {
  auto spec = multi_terminal_memory_spec(5, 3);
  for (auto &phase : spec.phases)
    if (phase.id == phase_id::from_name("terminal0"))
      phase.spec.D_sparse = {4, -1, 5, -1, 6, -1, 7, -1};
  return spec;
}

// Expect check() to throw E mentioning text.
template <typename E, typename Check>
void expect_throw(Check &&check, const std::string &text) {
  try {
    check();
    ADD_FAILURE() << "expected: " << text;
  } catch (const E &e) {
    EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
  }
}

// A shot of R rounds on spec: random measurements, and the detectors the
// closed model's D forms from them, independently of ingest.
struct shot {
  std::shared_ptr<const experiment_template> context;
  std::size_t R = 0;
  std::vector<std::uint8_t> measurements;
  std::vector<std::uint8_t> detectors;

  shot(const dem_chunks_spec &spec, std::size_t rounds, std::mt19937 &rng)
      : context(std::make_shared<const experiment_template>(spec)), R(rounds) {
    auto expanded = spec;
    expanded.num_rounds = R;
    const auto d = dem_chunks_to_d_sparse(expanded);
    measurements.resize(context->measurement_offset(R, R));
    for (auto &m : measurements)
      m = static_cast<std::uint8_t>(rng() & 1);
    detectors.assign(d.size(), 0);
    for (std::size_t r = 0; r < d.size(); ++r)
      for (const auto c : d[r])
        detectors[r] ^= measurements.at(c);
  }

  // Round r's detectors.
  std::vector<std::uint8_t> round(std::int64_t r) const {
    const auto i = static_cast<std::size_t>(r);
    return {detectors.begin() + context->row_offset(i, R),
            detectors.begin() + context->row_offset(i + 1, R)};
  }

  // The readout round's measurements through the last phase's m2o_sparse.
  std::vector<std::uint8_t> logical() const {
    std::vector<std::uint8_t> out;
    std::uint8_t value = 0;
    const auto base = context->measurement_offset(R - 1, R);
    for (const auto i : context->readout()) {
      if (i == -1) {
        out.push_back(value);
        value = 0;
      } else {
        value ^= measurements.at(base + static_cast<std::size_t>(i));
      }
    }
    return out;
  }
};

// Ingest's layers for requests, each with a pool of three views.
std::vector<ingest_layer> layers_of(const std::vector<window_request> &reqs) {
  std::vector<ingest_layer> out;
  for (const auto &request : reqs)
    out.push_back({request, 3});
  return out;
}

// The view is the shot's window k of its layer, rows and variant alike.
void expect_view(const ingest_core &core, const shot &s,
                 const std::vector<window_request> &requests,
                 const window_view &view) {
  const auto &request = requests.at(view.layer);
  const auto placed = place_window(s.context, request, view.k, s.R);
  SCOPED_TRACE("layer " + std::to_string(view.layer) + ", window " +
               std::to_string(view.k));
  EXPECT_EQ(view.rounds, placed.rounds);
  EXPECT_EQ(
      view.first_row,
      s.context->row_offset(static_cast<std::size_t>(view.rounds.begin), s.R));
  EXPECT_EQ(view.source, &core);
  EXPECT_TRUE(view.variant == window_variant(*s.context, request, view.k, s.R));
  for (auto r = view.rounds.begin; r < view.rounds.end; ++r) {
    const auto got = core.detectors(r);
    EXPECT_EQ(std::vector<std::uint8_t>(got.begin(), got.end()), s.round(r))
        << "round " << r;
  }
}

// Streams shot s into core in records cut at random, no longer than its
// limit, ending with a FINAL record that starts at or before the readout
// round. Every publication is checked against the shot, its views held for
// a while, checked again and released. @p before_final counts the indices
// published before the FINAL record.
void expect_streams(ingest_core &core, const shot &s,
                    const std::vector<window_request> &requests,
                    std::mt19937 &rng, std::size_t &before_final) {
  const auto total = s.measurements.size();
  const auto limit = core.record_limit();
  const auto readout = s.context->measurement_offset(s.R - 1, s.R);
  const auto final_start = std::uniform_int_distribution<std::size_t>(
      total > limit ? total - limit : 0, readout)(rng);
  std::deque<const window_view *> held;
  std::set<std::pair<std::size_t, std::int64_t>> published;
  std::int64_t last_k = std::numeric_limits<std::int64_t>::min();
  bool final_taken = false;
  before_final = 0;
  // Publications ready so far, checked, with views held up to two indices.
  const auto drain = [&] {
    while (const auto *p = core.poll()) {
      EXPECT_GT(p->k, last_k);
      last_k = p->k;
      before_final += final_taken ? 0 : 1;
      for (std::size_t l = 0; l < p->views.size(); ++l) {
        const auto instances = window_instances(requests[l], s.R);
        EXPECT_EQ(p->views[l] != nullptr,
                  p->k >= instances.begin && p->k < instances.end)
            << "layer " << l << ", window " << p->k;
        if (!p->views[l])
          continue;
        expect_view(core, s, requests, *p->views[l]);
        EXPECT_TRUE(published.insert({l, p->k}).second);
        held.push_back(p->views[l]);
      }
      while (held.size() > 2 * requests.size()) {
        expect_view(core, s, requests, *held.front());
        core.release(*held.front());
        held.pop_front();
      }
    } // end - while(p)
  };
  // Feeds one record, releasing every held view if the ring defers it.
  const auto feed = [&](std::size_t begin, std::size_t end, bool final) {
    auto status =
        core.enqueue(s.measurements.data() + begin, end - begin, final);
    if (status == ingest_status::deferred) {
      for (const auto *view : held)
        core.release(*view);
      held.clear();
      drain();
      status = core.enqueue(s.measurements.data() + begin, end - begin, final);
    }
    ASSERT_EQ(status, ingest_status::accepted);
  };
  for (std::size_t at = 0; at < final_start;) {
    const auto len = std::uniform_int_distribution<std::size_t>(
        1, std::min(limit, final_start - at))(rng);
    ASSERT_NO_FATAL_FAILURE(feed(at, at + len, false));
    at += len;
    drain();
  }
  final_taken = true;
  ASSERT_NO_FATAL_FAILURE(feed(final_start, total, true));
  drain();
  for (const auto *view : held)
    core.release(*view);
  held.clear();
  EXPECT_TRUE(core.done());
  EXPECT_EQ(core.shot_rounds(), s.R);
  for (std::size_t l = 0; l < requests.size(); ++l) {
    const auto instances = window_instances(requests[l], s.R);
    for (auto k = instances.begin; k < instances.end; ++k)
      EXPECT_TRUE(published.count({l, k}))
          << "layer " << l << ", window " << k << " was never published";
  }
} // end - expect_streams()

// Every layout the streaming tests run: the parallel-window layers at two
// sizes, and a sequential sliding window.
std::vector<std::vector<window_request>> ingest_layouts() {
  const auto large = layered(3, 3, 9);
  const auto small = layered(2, 2, 3);
  return {{large.a, large.b}, {small.a, small.b}, {sliding(2, 5)}};
}

// Every view of every window, in shots of every length up to 40 cut into
// records at random, holds the rows the whole shot's D forms, on specs with
// D_sparse and without, with one suffix phase or three, one of them unlike
// bulk, and with a wider readout round. Windows are handed over before FINAL
// once complete, and the raw logical value is the readout round's through m2o.
TEST(IngestCore, StreamsShotsCutAnywhere) {
  std::mt19937 rng(7);
  for (const auto &spec : {arbitrary_memory_spec(5, true), rep5_spec(),
                           multi_terminal_memory_spec(5, 3),
                           distinct_suffix_spec(), wide_readout_spec()}) {
    const auto context = std::make_shared<const experiment_template>(spec);
    for (const auto &requests : ingest_layouts()) {
      ingest_core core(context, layers_of(requests),
                       !context->readout().empty());
      for (auto R = context->min_rounds(); R <= 40; ++R) {
        SCOPED_TRACE("R=" + std::to_string(R) + ", " +
                     std::to_string(requests.size()) + " layers");
        const shot s(spec, R, rng);
        std::size_t early = 0;
        expect_streams(core, s, requests, rng, early);
        if (HasFatalFailure())
          return;
        if (R >= 30)
          EXPECT_GT(early, 0u) << "nothing was handed over before FINAL";
        EXPECT_EQ(core.logical(), s.logical());
        core.reset();
      } // end - for(R)
    } // end - for(requests)
  } // end - for(spec)
} // end - TEST(IngestCore, StreamsShotsCutAnywhere)

// A record longer than one period of rounds is rejected, and the core takes
// no more input until it is reset.
TEST(IngestCore, RejectsARecordLongerThanAPeriod) {
  const auto spec = arbitrary_memory_spec(5, true);
  const auto context = std::make_shared<const experiment_template>(spec);
  const auto layers = layered(3, 3, 9);
  ingest_core core(context, layers_of({layers.a, layers.b}), false);
  const std::vector<std::uint8_t> record(core.record_limit() + 1, 0);
  EXPECT_EQ(core.record_limit(), 12u * 4u);
  expect_throw<std::invalid_argument>(
      [&] { core.enqueue(record.data(), record.size(), false); },
      "ingest: a record of 49 measurements is longer than one period of "
      "rounds, 48");
  expect_throw<std::logic_error>([&] { core.enqueue(record.data(), 4, false); },
                                 "ingest: reset() is needed after a failure");
  core.reset();
  EXPECT_EQ(core.enqueue(record.data(), 4, false), ingest_status::accepted);
} // end - TEST(IngestCore, RejectsARecordLongerThanAPeriod)

// A FINAL record must end a shot the template admits: its readout round is
// whole, the rounds before it are whole, and the shot is long enough. Input
// after FINAL waits for reset().
TEST(IngestCore, RejectsAFinalRecordThatEndsNoShot) {
  const auto spec = arbitrary_memory_spec(5, true);
  const auto context = std::make_shared<const experiment_template>(spec);
  const auto layers = layered(3, 3, 9);
  const std::vector<std::uint8_t> record(48, 0);
  ingest_core core(context, layers_of({layers.a, layers.b}), false);
  const auto rejects = [&](std::size_t before, std::size_t final_len,
                           const std::string &text) {
    core.reset();
    ASSERT_EQ(core.enqueue(record.data(), before, false),
              ingest_status::accepted);
    expect_throw<std::invalid_argument>(
        [&] { core.enqueue(record.data(), final_len, true); }, text);
  };
  rejects(0, 3,
          "ingest: the FINAL record and the partial round before it hold 3 "
          "measurements, fewer than the readout round's 4");
  rejects(0, 10,
          "ingest: the FINAL record ends mid-round: round 1 has 2 of its 4 "
          "measurements before the readout round");
  rejects(1, 9,
          "ingest: the FINAL record ends mid-round: round 1 has 2 of its 4 "
          "measurements before the readout round");
  rejects(0, 4,
          "ingest: the FINAL record ends the shot at 1 round, but the "
          "template's shots have at least " +
              std::to_string(context->min_rounds()));
  core.reset();
  ASSERT_EQ(core.enqueue(record.data(), 2, false), ingest_status::accepted);
  ASSERT_EQ(core.enqueue(record.data(), 34, true), ingest_status::accepted);
  EXPECT_EQ(core.shot_rounds(), 9u);
  expect_throw<std::logic_error>(
      [&] { core.enqueue(record.data(), 4, false); },
      "ingest: input after FINAL; reset() starts the next shot");
} // end - TEST(IngestCore, RejectsAFinalRecordThatEndsNoShot)

// The ring is sized at construction and never grows: while views are out,
// it keeps their rounds intact and defers what it has no room for, and a
// full pool holds back the next index. Releasing them lets the shot go on.
TEST(IngestCore, HoldsEveryRoundAViewNeeds) {
  const auto spec = arbitrary_memory_spec(5, true);
  const auto layers = layered(3, 3, 9);
  const std::vector<window_request> requests{layers.a, layers.b};
  std::mt19937 rng(11);
  const shot s(spec, 200, rng);
  ingest_core core(s.context, layers_of(requests), false);
  const auto ring = core.ring_rounds();
  const auto period = 12u * 4u;
  std::vector<const window_view *> out;
  std::size_t at = 0;
  // Records of one period until the ring defers one.
  for (;;) {
    ASSERT_LT(at + period, s.measurements.size());
    if (core.enqueue(s.measurements.data() + at, period, false) ==
        ingest_status::deferred)
      break;
    at += period;
    while (const auto *p = core.poll())
      for (const auto *view : p->views)
        if (view)
          out.push_back(view);
  }
  EXPECT_EQ(core.ring_rounds(), ring);
  // A's pool of three is out, with B's views of the same indices: B has no
  // window 0.
  EXPECT_EQ(out.size(), 3u + 2u);
  EXPECT_EQ(core.poll(), nullptr);
  for (const auto *view : out)
    expect_view(core, s, requests, *view);
  for (const auto *view : out)
    core.release(*view);
  expect_throw<std::logic_error>([&] { core.release(*out.front()); },
                                 "ingest: release of a view that is not out");
  // Released, the rest of the shot streams through.
  out.clear();
  for (;;) {
    const auto len = std::min<std::size_t>(period, s.measurements.size() - at);
    const bool final = at + len == s.measurements.size();
    ASSERT_EQ(core.enqueue(s.measurements.data() + at, len, final),
              ingest_status::accepted);
    at += len;
    while (const auto *p = core.poll())
      for (const auto *view : p->views)
        if (view) {
          expect_view(core, s, requests, *view);
          core.release(*view);
        }
    if (final)
      break;
  }
  EXPECT_TRUE(core.done());
  EXPECT_EQ(core.ring_rounds(), ring);
  expect_throw<std::out_of_range>([&] { core.detectors(0); },
                                  "ingest: round 0's detectors are not in "
                                  "the ring");
} // end - TEST(IngestCore, HoldsEveryRoundAViewNeeds)

// A view out holds its shot: reset() waits for it, and the core reads the
// logical value only from a last phase with m2o_sparse.
TEST(IngestCore, ChecksResetAndReadout) {
  const auto layers = layered(3, 3, 9);
  const std::vector<window_request> requests{layers.a, layers.b};
  std::mt19937 rng(3);
  const shot s(arbitrary_memory_spec(5, true), 40, rng);
  ingest_core core(s.context, layers_of(requests), false);
  const ingest_publication *p = nullptr;
  for (std::size_t at = 0; !p; at += 48) {
    ASSERT_LE(at + 48, s.measurements.size());
    ASSERT_EQ(core.enqueue(s.measurements.data() + at, 48, false),
              ingest_status::accepted);
    p = core.poll();
  }
  expect_throw<std::logic_error>([&] { core.reset(); },
                                 "ingest: reset() with views out");
  EXPECT_TRUE(core.logical().empty());
  auto no_readout = arbitrary_memory_spec(5, true);
  no_readout.phases.back().spec.m2o_sparse.clear();
  expect_throw<std::invalid_argument>(
      [&] {
        ingest_core(std::make_shared<const experiment_template>(no_readout),
                    layers_of(requests), true);
      },
      "ingest: read_logical needs the last phase's m2o_sparse");
  expect_throw<std::invalid_argument>(
      [&] {
        ingest_core(s.context, layers_of({layers.a, sliding(2, 5)}), false);
      },
      "ingest: the layers' periods differ: 12 and 2");
} // end - TEST(IngestCore, ChecksResetAndReadout)

// Records, polls and releases allocate nothing once the core is built, with
// three suffix phases held back, and reading the logical value.
TEST(IngestCore, AllocatesNothingPerRecord) {
  if (QEC_TEST_SANITIZED)
    GTEST_SKIP() << "a sanitizer's allocator replaces the counting operator "
                    "new";
  const auto layers = layered(3, 3, 9);
  const std::vector<window_request> requests{layers.a, layers.b};
  std::mt19937 rng(5);
  for (const auto &spec : {distinct_suffix_spec(), wide_readout_spec()}) {
    const shot s(spec, 120, rng);
    ingest_core core(s.context, layers_of(requests),
                     !s.context->readout().empty());
    std::size_t allocations = 0;
    for (std::size_t at = 0; at < s.measurements.size();) {
      const auto len = std::min<std::size_t>(17, s.measurements.size() - at);
      const bool final = at + len == s.measurements.size();
      auto status = ingest_status::deferred;
      allocations += allocation_counter::during([&] {
        status = core.enqueue(s.measurements.data() + at, len, final);
        while (const auto *p = core.poll())
          for (const auto *view : p->views)
            if (view)
              core.release(*view);
      });
      ASSERT_EQ(status, ingest_status::accepted);
      at += len;
    }
    EXPECT_TRUE(core.done());
    EXPECT_EQ(core.logical(), s.logical());
    EXPECT_EQ(allocations, 0u);
  } // end - for(spec)
} // end - TEST(IngestCore, AllocatesNothingPerRecord)

// ---------------------------------------------------------------------------
// Decode window core and the serial pipeline
// ---------------------------------------------------------------------------

// Measurements whose detectors are @p detectors where a round's detector j
// is its measurement j, XOR the round before's, as in the memory specs:
// each round's measurement j is detector j XOR the round before's
// measurement j, and measurements past a round's detectors are 0.
std::vector<std::uint8_t>
memory_measurements(const experiment_template &t, std::size_t R,
                    const std::vector<std::uint8_t> &detectors) {
  std::vector<std::uint8_t> m(t.measurement_offset(R, R), 0);
  for (std::size_t r = 0; r < R; ++r) {
    const auto row = t.row_offset(r, R);
    const auto at = t.measurement_offset(r, R);
    const auto before = r == 0 ? 0 : t.measurement_offset(r - 1, R);
    for (std::size_t j = 0; j < t.row_offset(r + 1, R) - row; ++j)
      m[at + j] = detectors[row + j] ^ (r == 0 ? 0 : m[before + j]);
  }
  return m;
} // end - memory_measurements()

// The readout round's measurements through the last phase's m2o_sparse.
std::vector<std::uint8_t> raw_logical(const experiment_template &t,
                                      std::size_t R,
                                      const std::vector<std::uint8_t> &m) {
  std::vector<std::uint8_t> out;
  std::uint8_t value = 0;
  for (const auto i : t.readout()) {
    if (i == -1) {
      out.push_back(value);
      value = 0;
    } else {
      value ^= m.at(t.measurement_offset(R - 1, R) + i);
    }
  }
  return out;
} // end - raw_logical()

// single_error_lut, counting what each decode() allocates into
// decode_allocations, so a test can tell a fire's own allocations from its
// decoder's.
class allocation_counting_lut final : public decoder {
public:
  static inline std::size_t decode_allocations = 0;

  allocation_counting_lut(decoder_init inputs, decoder_output_request output,
                          const cudaqx::heterogeneous_map &params)
      : decoder(inputs, output),
        inner_(get_decoder("single_error_lut", inputs, output, params)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    decode_allocations +=
        allocation_counter::during([&] { result = inner_->decode(syndrome); });
    return result;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      allocation_counting_lut,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decoder_output_request> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<allocation_counting_lut>(
            std::move(inputs),
            output.value_or(decoder_output_request{decode_result_type::errors}),
            params);
      })

private:
  std::unique_ptr<decoder> inner_;
};
CUDAQ_EXT_PT_REGISTER_TYPE(allocation_counting_lut)

// A windowed graph run without the executor, for shots of any length, with
// its windows from a data artifact store: ingest takes a shot's
// measurements in records of one period, and each index it hands over
// fires its layers in g's order, each receiving the sends of the producer
// instances its edges name. Every window's detectors and solution are kept.
class serial_pipeline {
public:
  serial_pipeline(const dem_chunks_spec &spec, const window_graph &g,
                  const std::string &decoder_name,
                  const cudaqx::heterogeneous_map &params = {})
      : graph_(g), context_(store_.add_template(spec)) {
    std::vector<ingest_layer> layers;
    for (const auto &layer : g.layers) {
      store_.add_windows(context_, layer.request);
      layers.push_back({layer.request, 2});
      cores_.push_back(std::make_unique<decode_window_core>(
          store_, context_, layer.request, decoder_name, params, 4));
    }
    ingest_ = std::make_unique<ingest_core>(context_, std::move(layers),
                                            !context_->readout().empty());
  } // end - serial_pipeline()

  // Decodes one shot of measurements.
  void run(const std::vector<std::uint8_t> &measurements) {
    ingest_->reset();
    decodes_.clear();
    commits_.clear();
    for (std::size_t at = 0; at < measurements.size();) {
      const auto len =
          std::min(ingest_->record_limit(), measurements.size() - at);
      const bool final = at + len == measurements.size();
      ASSERT_EQ(ingest_->enqueue(measurements.data() + at, len, final),
                ingest_status::accepted);
      at += len;
      while (const auto *p = ingest_->poll())
        fire_index(*p);
    }
    ASSERT_TRUE(ingest_->done());
  } // end - run()

  const std::shared_ptr<const experiment_template> &context() const {
    return context_;
  }
  const ingest_core &ingest() const { return *ingest_; }
  const decode_window_core &core(std::size_t l) const { return *cores_[l]; }
  // Every window's decode, by node and instance.
  const std::map<std::pair<std::string, std::int64_t>, production_decode> &
  decodes() const {
    return decodes_;
  }
  // Every window's commit zone, by node and instance.
  const std::map<std::pair<std::string, std::int64_t>, round_range> &
  commits() const {
    return commits_;
  }
  // What the fires themselves allocated, the decoder's included.
  std::size_t fire_allocations() const { return fire_allocations_; }

private:
  // Fires each layer's window of the index, in g's order, and releases its
  // view once its rows are copied.
  void fire_index(const ingest_publication &p) {
    for (std::size_t l = 0; l < graph_.layers.size(); ++l) {
      const auto *view = p.views[l];
      std::vector<const window_send_value *> receives;
      if (!view)
        continue;
      for (const auto &e : graph_.edges) {
        if (e.to_node != graph_.layers[l].node)
          continue;
        const auto port = std::stoul(e.to_input.substr(8)); // "receive_k"
        const auto &producer = *cores_[layer_of(e.from_node)];
        const auto *f = producer.find(view->k + e.index_offset);
        receives.resize(std::max(receives.size(), port + 1), nullptr);
        receives[port] =
            f ? &f->sends[producer.send_index(e.from_output)] : nullptr;
      }
      const auto slots = static_cast<std::int64_t>(cores_[l]->num_slots());
      const auto slot =
          static_cast<std::size_t>((view->k % slots + slots) % slots);
      const window_fire *f = nullptr;
      fire_allocations_ += allocation_counter::during(
          [&] { f = &cores_[l]->fire(slot, *view, receives); });
      decodes_[{graph_.layers[l].node, view->k}] = {
          {f->detectors.begin(), f->detectors.end()}, *f->solution};
      commits_[{graph_.layers[l].node, view->k}] = f->commit;
      ingest_->release(*view);
    } // end - for(l)
  } // end - fire_index()

  // The layer of node id.
  std::size_t layer_of(const std::string &id) const {
    for (std::size_t l = 0; l < graph_.layers.size(); ++l)
      if (graph_.layers[l].node == id)
        return l;
    throw std::logic_error("no layer " + id);
  }

  window_graph graph_;
  data_artifact_store store_;
  std::shared_ptr<const experiment_template> context_;
  std::unique_ptr<ingest_core> ingest_;
  std::vector<std::unique_ptr<decode_window_core>> cores_;
  std::map<std::pair<std::string, std::int64_t>, production_decode> decodes_;
  std::map<std::pair<std::string, std::int64_t>, round_range> commits_;
  std::size_t fire_allocations_ = 0;
};

// The pipeline, run on the measurements behind @p detectors, decodes as the
// reference does, window by window and bit for bit, over the same commit
// zones, and the frames it folds are the reference's.
void expect_pipeline_matches(serial_pipeline &pipeline,
                             parallel_window_reference &reference,
                             const production_windows &production,
                             const std::vector<std::uint8_t> &detectors,
                             std::size_t R) {
  const auto want = reference.decode(detectors);
  std::vector<production_decode> got;
  std::vector<std::uint8_t> frame(want.frame.size(), 0);
  ASSERT_NO_FATAL_FAILURE(
      pipeline.run(memory_measurements(*pipeline.context(), R, detectors)));
  ASSERT_EQ(pipeline.decodes().size(), reference.windows().size());
  for (const auto &rw : reference.windows()) {
    got.push_back(pipeline.decodes().at({rw.node, rw.k}));
    EXPECT_EQ(pipeline.commits().at({rw.node, rw.k}), rw.commit)
        << rw.node << "[" << rw.k << "]";
  }
  ASSERT_NO_FATAL_FAILURE(
      expect_decodes_match(reference, production, got, want));
  for (const auto &d : got)
    for (std::size_t j = 0; j < frame.size(); ++j)
      frame[j] ^= d.solution.outputs.frame[j];
  EXPECT_EQ(frame, want.frame);
} // end - expect_pipeline_matches()

// @p requests with every anchor @p by rounds later, so other windows
// straddle round 0.
layered_requests shifted(layered_requests requests, std::int64_t by) {
  requests.a.offset += by;
  requests.b.offset += by;
  return requests;
}

// Ingest and the decode_window cores, run serially on raw measurements with
// windows from the store, decode every single fault as the reference does,
// window by window, for every layout and shot length up to 24, and with
// the default layout a round earlier, whose first A window's commit zone
// starts before the shot. One pipeline serves every length.
TEST(DecodeWindowCore, PipelineDecodesSingleFaultsAsTheReference) {
  auto layouts = swept_layouts();
  layouts.push_back(parallel_graph(shifted(layered(2, 1, 3), -1)));
  for (const auto &spec :
       {arbitrary_memory_spec(5), multi_terminal_memory_spec(5, 3)}) {
    for (const auto &g : layouts) {
      serial_pipeline pipeline(spec, g, "single_error_lut");
      for (auto R = pipeline.context()->min_rounds(); R <= 24; ++R) {
        parallel_window_reference reference(spec, g, R, "single_error_lut");
        const production_windows production(
            pipeline.context(), g, R, reference.order(), "single_error_lut");
        const auto &s = reference.model();
        SCOPED_TRACE("R=" + std::to_string(R));
        for (std::size_t c = 0; c < s.h.size(); ++c)
          ASSERT_NO_FATAL_FAILURE(expect_pipeline_matches(
              pipeline, reference, production, detectors_of(s, c), R))
              << "column " << c;
      } // end - for(R)
    } // end - for(g)
  } // end - for(spec)
} // end - TEST(DecodeWindowCore, PipelineDecodesSingleFaultsAsTheReference)

// The same on sampled shots with PyMatching, where windows commit several
// faults and send what they leave; the raw logical value is the readout
// round's through m2o.
TEST(DecodeWindowCore, PipelineDecodesSampledShotsAsTheReference) {
  constexpr std::size_t R = 20;      // two periods of each layout
  constexpr std::size_t shots = 500; // a hundred or so with several faults
  for (const auto &spec : {arbitrary_memory_spec(5), wide_readout_spec()}) {
    for (const auto &g : {parallel_graph(layered(3, 3, 9)),
                          parallel_graph(with_truncate(layered(3, 3, 9)))}) {
      serial_pipeline pipeline(spec, g, "pymatching", matching);
      parallel_window_reference reference(spec, g, R, "pymatching", matching);
      const production_windows production(
          pipeline.context(), g, R, reference.order(), "pymatching", matching);
      const auto &s = reference.model();
      const auto [detectors, errors] = dem_sampler::cpu::sample_dem(
          s.dem.detector_error_matrix, shots, s.dem.error_rates, 23);
      for (std::size_t i = 0; i < shots; ++i) {
        const auto syndrome = row_of(detectors, i);
        ASSERT_NO_FATAL_FAILURE(expect_pipeline_matches(
            pipeline, reference, production, syndrome, R))
            << "shot " << i;
        EXPECT_EQ(
            pipeline.ingest().logical(),
            raw_logical(*pipeline.context(), R,
                        memory_measurements(*pipeline.context(), R, syndrome)));
      }
    } // end - for(g)
  } // end - for(spec)
} // end - TEST(DecodeWindowCore, PipelineDecodesSampledShotsAsTheReference)

// A fire allocates nothing beyond what its decoder's decode() does, over a
// long shot with every kind of variant.
TEST(DecodeWindowCore, FiresAllocateNothingBeyondTheDecoder) {
  if (QEC_TEST_SANITIZED)
    GTEST_SKIP() << "a sanitizer's allocator replaces the counting operator "
                    "new";
  const auto spec = arbitrary_memory_spec(5);
  const auto g = parallel_graph(layered(3, 3, 9));
  constexpr std::size_t R = 60; // five periods: prefix, bulk and terminal
  serial_pipeline pipeline(spec, g, "allocation_counting_lut");
  const parallel_window_reference reference(spec, g, R, "single_error_lut");
  allocation_counting_lut::decode_allocations = 0;
  ASSERT_NO_FATAL_FAILURE(pipeline.run(memory_measurements(
      *pipeline.context(), R, detectors_of(reference.model(), 37))));
  EXPECT_GT(allocation_counting_lut::decode_allocations, 0u);
  EXPECT_EQ(pipeline.fire_allocations(),
            allocation_counting_lut::decode_allocations);
} // end - TEST(DecodeWindowCore, FiresAllocateNothingBeyondTheDecoder)

// An input from a producer instance that does not exist is empty: a window
// fired without its residual has zero imported rows, whatever its slot
// held before.
TEST(DecodeWindowCore, AMissingResidualLeavesTheImportsZero) {
  const auto spec = arbitrary_memory_spec(5);
  const auto g = parallel_graph(layered(2, 1, 3));
  constexpr std::size_t R = 30;
  std::mt19937 rng(13);
  data_artifact_store store;
  const auto context = store.add_template(spec);
  for (const auto &layer : g.layers)
    store.add_windows(context, layer.request);
  decode_window_core a(store, context, g.layers[0].request, "single_error_lut",
                       {}, 2);
  decode_window_core b(store, context, g.layers[1].request, "single_error_lut",
                       {}, 2);
  ingest_core ingest(
      context, {{g.layers[0].request, 4}, {g.layers[1].request, 4}}, false);
  std::vector<std::uint8_t> m(context->measurement_offset(R, R));
  for (auto &x : m)
    x = static_cast<std::uint8_t>(rng() & 1);
  const ingest_publication *p = nullptr;
  for (std::size_t at = 0; !p || !p->views[1]; at += ingest.record_limit()) {
    ASSERT_LT(at + ingest.record_limit(), m.size());
    ASSERT_EQ(ingest.enqueue(m.data() + at, ingest.record_limit(), false),
              ingest_status::accepted);
    while ((p = ingest.poll()) && !p->views[1])
      ;
  }
  const auto &fired = a.fire(0, *p->views[0], {});
  const auto &residual = fired.sends[a.send_index("backward")];
  const window_send_value *with[] = {nullptr, &residual};
  const window_send_value *without[] = {nullptr, nullptr};
  const auto imported = [&](const window_fire &f) {
    return std::vector<std::uint8_t>(
        f.detectors.end() - static_cast<std::ptrdiff_t>(residual.values.size()),
        f.detectors.end());
  };
  ASSERT_NE(std::count(residual.values.begin(), residual.values.end(), 1), 0);
  EXPECT_EQ(imported(b.fire(0, *p->views[1], with)),
            std::vector<std::uint8_t>(residual.values.begin(),
                                      residual.values.end()));
  EXPECT_EQ(imported(b.fire(0, *p->views[1], without)),
            std::vector<std::uint8_t>(residual.values.size(), 0));
} // end - TEST(DecodeWindowCore, AMissingResidualLeavesTheImportsZero)

// A fire takes only its own layer's windows into existing slots, and a
// send only onto the rows its mode reaches in the receiving window.
TEST(DecodeWindowCore, RejectsWhatIsNotItsWindow) {
  const auto spec = arbitrary_memory_spec(5);
  const auto g = parallel_graph(layered(2, 1, 3));
  constexpr std::size_t R = 30;
  data_artifact_store store;
  const auto context = store.add_template(spec);
  for (const auto &layer : g.layers)
    store.add_windows(context, layer.request);
  decode_window_core a(store, context, g.layers[0].request, "single_error_lut",
                       {}, 2);
  decode_window_core b(store, context, g.layers[1].request, "single_error_lut",
                       {}, 2);
  ingest_core ingest(
      context, {{g.layers[0].request, 4}, {g.layers[1].request, 4}}, false);
  const auto m = memory_measurements(
      *context, R, std::vector<std::uint8_t>(context->row_offset(R, R), 0));
  std::vector<const window_view *> a_views;
  std::vector<const window_view *> b_views;
  // Records until A has three views out and B two, held.
  for (std::size_t at = 0; a_views.size() < 3 || b_views.size() < 2;
       at += ingest.record_limit()) {
    ASSERT_LT(at + ingest.record_limit(), m.size());
    ASSERT_EQ(ingest.enqueue(m.data() + at, ingest.record_limit(), false),
              ingest_status::accepted);
    while (const auto *p = ingest.poll()) {
      if (p->views[0])
        a_views.push_back(p->views[0]);
      if (p->views[1])
        b_views.push_back(p->views[1]);
    }
  }
  auto unknown = *a_views[0];
  unknown.variant.position = 1000;
  expect_error([&] { a.fire(2, *a_views[0], {}); },
               "decode_window: slot 2 does not exist; there are 2");
  expect_error([&] { a.fire(0, unknown, {}); },
               "decode_window: window 0's variant is none of this layer's");
  expect_error([&] { a.fire(0, *b_views.back(), {}); },
               "decode_window: window " + std::to_string(b_views.back()->k) +
                   "'s view holds other rows than the");
  // A[k].backward is B[k]'s imports, before B[k + 1]'s rows.
  ASSERT_EQ(b_views[1]->k, b_views[0]->k + 1);
  const auto &fired =
      a.fire(0, *a_views[static_cast<std::size_t>(b_views[0]->k)], {});
  const window_send_value *misdirected[] = {
      nullptr, &fired.sends[a.send_index("backward")]};
  expect_error([&] { b.fire(0, *b_views[1], misdirected); },
               "send 'backward' reaches shot row");
  expect_error([&] { a.send_index("sideways"); },
               "decode_window: the layer has no send 'sideways'");
} // end - TEST(DecodeWindowCore, RejectsWhatIsNotItsWindow)

// ---------------------------------------------------------------------------
// Data artifact store
// ---------------------------------------------------------------------------

// Expect check() to throw std::logic_error mentioning text.
template <typename Check>
void expect_logic_error(Check &&check, const std::string &text) {
  try {
    check();
    ADD_FAILURE() << "expected: " << text;
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
  }
}

// sha256_hex() gives the FIPS 180-2 test vectors, and the lengths either side
// of the block boundary where the length field moves to a second block, after
// no whole block and after one or two.
TEST(DataArtifactStore, Sha256MatchesTheStandardVectors) {
  EXPECT_EQ(sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  // A default view may hold a null pointer; offsetting it by zero is defined.
  EXPECT_EQ(sha256_hex(std::string_view{}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(
      sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(sha256_hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  EXPECT_EQ(sha256_hex(std::string(55, 'a')),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  EXPECT_EQ(sha256_hex(std::string(56, 'a')),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  EXPECT_EQ(sha256_hex(std::string(64, 'a')),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
  EXPECT_EQ(sha256_hex(std::string(63, 'a')),
            "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34");
  EXPECT_EQ(sha256_hex(std::string(119, 'a')),
            "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb");
  EXPECT_EQ(sha256_hex(std::string(120, 'a')),
            "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c");
  EXPECT_EQ(sha256_hex(std::string(127, 'a')),
            "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a");
  EXPECT_EQ(sha256_hex(std::string(128, 'a')),
            "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e");
}

// Changing any one field of a template or a window request changes its
// canonical bytes, so no field is left out of the content.
TEST(DataArtifactStore, EveryFieldIsContent) {
  using spec_change = std::function<void(dem_chunks_spec &)>;
  using request_change = std::function<void(window_request &)>;
  const auto spec = arbitrary_memory_spec(5, true);
  const auto request = layered(2, 2, 3).a;
  const std::vector<spec_change> spec_changes{
      [](auto &s) { s.num_rounds = 7; },
      [](auto &s) { s.num_rounds.reset(); },
      [](auto &s) { s.measurements_per_round = 4; },
      [](auto &s) { s.phases_supply_D_sparse = true; },
      [](auto &s) { s.observable_names = {"Z_L"}; },
      [](auto &s) { s.connections.pop_back(); },
      [](auto &s) { std::swap(s.seam.from_seam, s.seam.to_seam); },
      [](auto &s) { s.phases[0].id = phase_name::dem_final; },
      [](auto &s) { s.phases[0].spec.num_faults = 5; },
      [](auto &s) { s.phases[0].spec.H_sparse = {0, 1, -1}; },
      [](auto &s) {
        s.phases[0].spec.seam_specs = {{seam_name::prev_round, {{0, -1}, {}}}};
      },
      [](auto &s) {
        s.phases[0].spec.seam_specs = {
            {seam_name::prev_round, {{0, -1}, {0, -1}}}};
      },
      [](auto &s) { s.phases[0].spec.O_sparse = {1, -1}; },
      [](auto &s) { s.phases[0].spec.error_rates[0] = 0.03; },
      [](auto &s) { s.phases[0].spec.D_sparse = {0, -1}; },
      [](auto &s) { s.phases[0].spec.num_measurements = 5; },
      [](auto &s) { s.phases[2].spec.m2o_sparse = {0, -1}; }};
  const std::vector<request_change> request_changes{
      [](auto &r) { r.period = 6; },
      [](auto &r) { r.offset = 1; },
      [](auto &r) { r.commit = round_range{0, 1}; },
      [](auto &r) { r.buffers.pop_back(); },
      [](auto &r) { r.input.disposition = port_disposition::closed; },
      [](auto &r) { r.input.prior = 0.02; },
      [](auto &r) { r.input.boundary = input_boundary::columns; },
      [](auto &r) { r.output.disposition = port_disposition::closed; },
      [](auto &r) { r.output.imports = round_range{4, 5}; },
      [](auto &r) { r.sends[0].id = "ahead"; },
      [](auto &r) { r.sends[0].mode = send_mode::delta; },
      [](auto &r) { r.sends[0].rounds = {{2, 3}}; },
      [](auto &r) { r.columns.merge_identical = true; },
      [](auto &r) {
        r.columns.conflicting_parallel = parallel_resolution::most_likely;
      }};
  std::set<std::string> seen{canonical_bytes(spec), canonical_bytes(request)};
  for (const auto &change : spec_changes) {
    auto changed = spec;
    change(changed);
    EXPECT_TRUE(seen.insert(canonical_bytes(changed)).second);
  }
  for (const auto &change : request_changes) {
    auto changed = request;
    change(changed);
    EXPECT_TRUE(seen.insert(canonical_bytes(changed)).second);
  }
  EXPECT_EQ(canonical_bytes(spec), canonical_bytes(dem_chunks_spec(spec)));
} // end - TEST(DataArtifactStore, EveryFieldIsContent)

// Equal content is one entry and one handle; its record gives the kind and
// the size and sha256 of the canonical bytes. Lookups return what was added.
TEST(DataArtifactStore, EqualTemplatesShareOneEntry) {
  data_artifact_store store;
  const auto spec = arbitrary_memory_spec(5);
  const auto first = store.add_template(spec);
  const auto again = store.add_template(dem_chunks_spec(spec));
  const auto other = store.add_template(rep5_spec());
  const auto bytes = canonical_bytes(spec);
  EXPECT_EQ(first, again);
  EXPECT_NE(first, other);
  EXPECT_EQ(store.find_template(spec), first);
  ASSERT_EQ(store.records().size(), 2u);
  EXPECT_EQ(
      store.records()[0],
      (data_artifact_record{"dem_chunks", bytes.size(), sha256_hex(bytes)}));
  EXPECT_NE(store.records()[1].sha256, store.records()[0].sha256);
  expect_logic_error(
      [&] { store.find_template(single_round_spec()); },
      "data_artifact_store::find_template: no template with this content was "
      "added");
}

// A Stim DEM is keyed by its text, whether the store hashes it or is given
// the hash a caller has already computed.
TEST(DataArtifactStore, StimDemsAreKeyedByTheirText) {
  data_artifact_store store;
  const std::string text = "error(0.01) D0\n";
  store.add_stim_dem(text);
  store.add_stim_dem(std::string(text));
  store.add_stim_dem(text, sha256_hex(text));
  store.add_stim_dem("error(0.02) D0\n");
  ASSERT_EQ(store.records().size(), 2u);
  EXPECT_EQ(store.records()[0],
            (data_artifact_record{"stim_dem", text.size(), sha256_hex(text)}));
}

// Every variant of a request is built once, at the first shot length that
// has it, and equal requests from other nodes reuse it. Lookups never build:
// a variant, request or template the store did not build is a logic error.
TEST(DataArtifactStore, VariantsBuildOnceAndLookupsNeverBuild) {
  data_artifact_store store;
  data_artifact_store other;
  const auto tmpl = store.add_template(arbitrary_memory_spec(5));
  const auto request = layered(2, 2, 3).a;
  std::set<variant_key> keys;
  store.add_windows(tmpl, request);
  const auto added = store.records().size();
  store.add_windows(tmpl, window_request(request));
  EXPECT_EQ(store.records().size(), added);
  for (auto R = tmpl->min_rounds();
       R < variant_probe_rounds(*tmpl, request) + 2 * request.period; ++R) {
    const auto instances = window_instances(request, R);
    for (auto k = instances.begin; k < instances.end; ++k) {
      const auto key = window_variant(*tmpl, request, k, R);
      const auto w = store.find_window(*tmpl, request, key);
      keys.insert(key);
      ASSERT_NE(w, nullptr);
      EXPECT_EQ(store.find_window(*tmpl, request, key), w);
      EXPECT_NO_THROW(check_same_window_up_to_shift(
          *w, window_from_dem_chunks(tmpl, request, k, R)));
    }
  } // end - for(R)
  EXPECT_EQ(added, 1 + keys.size());
  for (std::size_t i = 1; i < added; ++i)
    EXPECT_EQ(store.records()[i].kind, "window");
  expect_logic_error(
      [&] { store.find_window(*tmpl, layered(2, 2, 3).b, *keys.begin()); },
      "was not built at configuration time");
  expect_logic_error(
      [&] { store.find_window(*tmpl, request, variant_key{999, 999}); },
      "was not built at configuration time");
} // end - TEST(DataArtifactStore, VariantsBuildOnceAndLookupsNeverBuild)

// A template is its content: an equal one built elsewhere finds the store's
// windows, a zero's sign is not content, and a store that holds no template
// with that content rejects it.
TEST(DataArtifactStore, TemplatesAreTheirContent) {
  data_artifact_store store;
  data_artifact_store other;
  auto spec = arbitrary_memory_spec(5);
  auto negative = spec;
  spec.phases[0].spec.error_rates[0] = 0.0;
  negative.phases[0].spec.error_rates[0] = -0.0;
  const auto tmpl = store.add_template(spec);
  const auto twin = std::make_shared<const experiment_template>(negative);
  const auto request = layered(2, 2, 3).a;
  const auto key = window_variant(*tmpl, request, 1, 14);
  store.add_windows(twin, request);
  EXPECT_EQ(canonical_bytes(negative), canonical_bytes(spec));
  EXPECT_EQ(store.add_template(negative), tmpl);
  EXPECT_EQ(store.find_template(negative), tmpl);
  EXPECT_EQ(store.find_window(*twin, request, key),
            store.find_window(*tmpl, request, key));
  EXPECT_EQ(store.records()[0].kind, "dem_chunks");
  for (std::size_t i = 1; i < store.records().size(); ++i)
    EXPECT_EQ(store.records()[i].kind, "window");
  expect_logic_error([&] { other.add_windows(tmpl, request); },
                     "no template with this content was added");
  expect_logic_error([&] { other.find_window(*tmpl, request, key); },
                     "no template with this content was added");
} // end - TEST(DataArtifactStore, TemplatesAreTheirContent)

// A stored window that holds the shot's readout round carries the
// template's readout map, which a variant fixes since its key fixes where it
// ends; every other window has none. wide_readout_spec's last phase has
// m2o_sparse.
TEST(DataArtifactStore, WindowsHoldingTheReadoutCarryM2o) {
  data_artifact_store store;
  const auto tmpl = store.add_template(wide_readout_spec());
  std::size_t with_m2o = 0;
  for (const auto &request : {layered(2, 2, 3).a, layered(2, 2, 3).b}) {
    store.add_windows(tmpl, request);
    for (auto R = tmpl->min_rounds(); R < variant_probe_rounds(*tmpl, request);
         ++R) {
      const auto instances = window_instances(request, R);
      for (auto k = instances.begin; k < instances.end; ++k) {
        const auto w = store.find_window(*tmpl, request,
                                         window_variant(*tmpl, request, k, R));
        const bool readout =
            w->rounds.end == static_cast<std::int64_t>(w->shot_rounds);
        SCOPED_TRACE("R=" + std::to_string(R) + " k=" + std::to_string(k));
        EXPECT_EQ(w->m2o.has_value(), readout);
        with_m2o += w->m2o.has_value();
        EXPECT_NO_THROW(check_same_window_up_to_shift(
            *w, window_from_dem_chunks(tmpl, request, k, R, readout)));
      }
    } // end - for(R)
  } // end - for(request)
  EXPECT_GT(with_m2o, 0u);
} // end - TEST(DataArtifactStore, WindowsHoldingTheReadoutCarryM2o)

} // namespace
} // namespace cudaq::qec
