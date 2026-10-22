/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Unit tests for the experiment template and decoding windows built from
// dem_chunks: per-round phases, chunks and offsets against an expanded shot;
// window rows, columns, ports, sends and measurement maps against a dense
// slice of the whole shot; the window spec's nine-SEC example; the spec's
// invalid constructions; the column-handling options; variant keys and
// translation invariance; and the kernel's forms.

#include "task_graph_test_fixtures.h"
#include "../lib/decoding_task/decoding_task_decoding_window.h"
#include "../lib/decoding_task/decoding_task_experiment_template.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <random>
#include <set>
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

// ---------------------------------------------------------------------------
// Decoding windows
// ---------------------------------------------------------------------------

// The parallel-window layers of the plan: A[k] buffers b rounds on each side
// of c_a commit rounds, and B[k] is the c_b-round gap before A[k].
struct layered_requests {
  window_request a;
  window_request b;
};

// A and B windows for commit c_a, buffer b and gap c_b.
layered_requests layered(std::int64_t c_a, std::int64_t b, std::int64_t c_b) {
  layered_requests out;
  const std::int64_t period = c_a + c_b;
  out.a.period = period;
  out.a.commit = round_range{0, c_a};
  out.a.buffers = {{-b, 0}, {c_a, c_a + b}};
  out.a.input = {port_disposition::open, 0.01};
  out.a.output = {port_disposition::open, std::nullopt};
  out.a.sends = {{"forward", std::nullopt, {}}, {"backward", std::nullopt, {}}};
  out.b.period = period;
  out.b.offset = -c_b;
  out.b.commit = round_range{0, c_b};
  out.b.input = {port_disposition::closed, std::nullopt};
  out.b.output = {port_disposition::closed, std::nullopt};
  return out;
} // end - layered()

// A sequential sliding window: commit step rounds, then look ahead.
window_request sliding(std::int64_t step, std::int64_t size) {
  window_request w;
  w.period = step;
  w.commit = round_range{0, step};
  w.buffers = {{step, size}};
  w.input = {port_disposition::closed, std::nullopt};
  w.output = {port_disposition::open, std::nullopt};
  w.sends = {{"forward", std::nullopt, {}}};
  return w;
}

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

// Entries of the window model's H, for its rows and columns [begin, end).
std::vector<std::vector<uint8_t>>
window_h(const decoding_window &w, std::size_t col_begin, std::size_t col_end) {
  std::vector<std::vector<uint8_t>> out;
  const auto &h = w.model.detector_error_matrix;
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
            EXPECT_EQ(w.model.error_rates[c], shot.model.error_rates[shot_c]);
            for (std::size_t o = 0; o < tmpl->num_observables(); ++o)
              EXPECT_EQ(w.model.observables_flips_matrix.at({o, c}),
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
  const auto &h = w.model.detector_error_matrix;
  const auto &o = w.model.observables_flips_matrix;
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
    EXPECT_EQ(w.model.error_rates[c], 0.01);
  EXPECT_LE(w.commit_columns.end, first);
} // end - TEST(DecodingWindow, OpenInputAddsBoundaryColumns)

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
    const auto [pcm, first, last] = get_pcm_for_rounds(
        shot.model.detector_error_matrix, rep5_checks, w.rounds.begin,
        w.rounds.end - 1, false, w.output.exists);
    SCOPED_TRACE("k=" + std::to_string(k));
    EXPECT_EQ(w.num_virtual_outcome_columns + w.num_logical_error_columns, 0u);
    ASSERT_EQ(pcm.shape()[1], w.num_fault_columns);
    for (std::size_t r = 0; r < pcm.shape()[0]; ++r)
      for (std::size_t c = 0; c < pcm.shape()[1]; ++c)
        EXPECT_EQ(pcm.at({r, c}), w.model.detector_error_matrix.at({r, c}));
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

  std::vector<uint8_t> d(w.model.detector_error_matrix.shape()[0], 0);
  std::vector<uint8_t> e(w.model.detector_error_matrix.shape()[1], 0);
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
    const auto &h = w.model.detector_error_matrix;
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
// wire and on the output port, with or without explicit D rows.
TEST(DecodingWindow, M2dMatchesTheWholeShot) {
  std::mt19937 rng(11);
  for (const auto &spec :
       {arbitrary_memory_spec(5, true), wide_readout_spec(), rep5_spec()}) {
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

// The spec's invalid constructions are each rejected with their own message.
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
  const auto &h = w.model.detector_error_matrix;
  const auto &o = w.model.observables_flips_matrix;
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
      EXPECT_LE(plain.model.error_rates[source],
                plain.model.error_rates[sources[0]]);
      prior = prior + plain.model.error_rates[source] -
              2 * prior * plain.model.error_rates[source];
    }
    EXPECT_NEAR(w.model.error_rates[c], prior, 1e-12);
    const bool in_commit = zone_of(plain, sources[0], 8) == 0;
    EXPECT_EQ(in_commit,
              c >= w.commit_columns.begin && c < w.commit_columns.end);
  } // end - for(c)
  EXPECT_EQ(seen, std::vector<int>(plain.num_fault_columns, 1));
  // The commit zone's first chunk: 0 and 1 merge, reported against 1.
  const auto merged = w.commit_columns.begin;
  EXPECT_EQ(w.fault_column_sources[merged], (std::vector<std::size_t>{17, 16}));
  EXPECT_NEAR(w.model.error_rates[merged], 0.01 + 0.02 - 2 * 0.01 * 0.02,
              1e-12);
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
    EXPECT_NEAR(w.model.error_rates[c], prior, 1e-12);
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
        EXPECT_EQ(pcm.at({r, c}), w.model.detector_error_matrix.at({r, F + c}));
      for (std::size_t c = 0; c < F; ++c)
        EXPECT_EQ(pcm.at({r, C + c}), w.model.detector_error_matrix.at({r, c}));
    }
    for (std::size_t c = 0; c < C; ++c)
      EXPECT_EQ(w.model.error_rates[F + c],
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

// The kernel's forms: H over the decoder's columns as canonical CSC, O and
// the priors over every column; converting back loses nothing.
TEST(WindowKernelForms, RoundTripWithoutLoss) {
  const auto tmpl =
      std::make_shared<const experiment_template>(narrow_readout_spec());
  auto request = layered(2, 2, 3).a;
  for (const auto truncate : {false, true}) {
    if (truncate)
      request.input = {port_disposition::open, std::nullopt,
                       input_boundary::truncate};
    const auto w = window_from_dem_chunks(tmpl, request, 1, 14);
    const auto forms = kernel_forms(w);
    const auto back = model_from_kernel_forms(forms);
    const auto csc = forms.h.to_nested_csc();
    EXPECT_EQ(forms.h.num_cols(), w.num_decoder_columns());
    EXPECT_EQ(forms.o.num_cols(), w.model.num_error_mechanisms());
    for (const auto &column : csc)
      EXPECT_TRUE(std::is_sorted(column.begin(), column.end()));
    EXPECT_EQ(back.error_rates, w.model.error_rates);
    ASSERT_EQ(back.detector_error_matrix.shape(),
              w.model.detector_error_matrix.shape());
    ASSERT_EQ(back.observables_flips_matrix.shape(),
              w.model.observables_flips_matrix.shape());
    for (std::size_t c = 0; c < w.model.num_error_mechanisms(); ++c)
      EXPECT_EQ(column_of(w, c), column_of(decoding_window{.model = back}, c));
  } // end - for(truncate)
} // end - TEST(WindowKernelForms, RoundTripWithoutLoss)
} // namespace
} // namespace cudaq::qec
