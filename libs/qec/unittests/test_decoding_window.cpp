/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the experiment template and decoding windows built from
// dem_chunks: per-round phases, chunks and offsets against an expanded shot;
// window rows, columns, ports, sends and measurement maps against a dense
// slice of the whole shot; the window spec's nine-SEC example; the spec's
// invalid constructions; the column-handling options and parallel-column
// conflicts; variant keys and translation invariance; the kernel's forms;
// and the window schedule and commitment validation: row finalization
// (F1-F4) checked against the built windows, each failure's own message,
// send and receive matching, and ambiguous parallel commit columns.

#include "task_graph_test_fixtures.h"
#include "../lib/decoding_task/decoding_task_commitment.h"
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
#include <utility>
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
          const auto &h = w.model.detector_error_matrix;
          const auto &o = w.model.observables_flips_matrix;
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

// ---------------------------------------------------------------------------
// Window schedule and commitment validation
//
// With one-round chunks, max_span() is at most 1 and every zone has a round,
// so the layout's size conditions (b, c_A and c_B at least max_span()) hold
// for every valid request. The failure tests break the same conditions with
// explicit sends and ports instead.
// ---------------------------------------------------------------------------

// Window layers and the send edges between them.
struct window_graph {
  std::vector<window_layer> layers;
  std::vector<task_graph_edge> edges;
};

// The parallel-window graph: A[k-1].forward to B[k].receive_0 and
// A[k].backward to B[k].receive_1.
window_graph parallel_graph(const layered_requests &requests) {
  window_graph g;
  g.layers = {{"A", "a", requests.a}, {"B", "b", requests.b}};
  g.edges = {{"A", "forward", "B", "receive_0", -1},
             {"A", "backward", "B", "receive_1", 0}};
  return g;
}

// A sliding window whose forward delta feeds its next instance.
window_graph sliding_graph(std::int64_t step, std::int64_t size) {
  window_graph g;
  g.layers = {{"W", "w", sliding(step, size)}};
  g.edges = {{"W", "forward", "W", "receive_0", -1}};
  return g;
}

// The interior-commit layout for the default sizes, with M = 1: A commits
// c_a - 2 rounds between b + 1 buffer rounds, and B's gap grows to c_b + 2.
layered_requests interior_commit(std::int64_t c_a, std::int64_t b,
                                 std::int64_t c_b) {
  return layered(c_a - 2, b + 1, c_b + 2);
}

// Both layers resolve observable conflicts by the more likely column.
window_graph with_most_likely(window_graph g) {
  for (auto &layer : g.layers)
    layer.request.columns.conflicting_parallel =
        parallel_resolution::most_likely;
  return g;
}

// A's input truncates the previous round's crossing faults.
layered_requests with_truncate(layered_requests requests) {
  requests.a.input = {port_disposition::open, std::nullopt,
                      input_boundary::truncate};
  return requests;
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
      const auto probe = commitment_probe_rounds(tmpl, g.layers);
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
  expect_error([&] { commitment_probe_rounds(tmpl, g.layers); },
               "commitment_probe_rounds", "periods 5 and 4");
  chain.connections.erase(chain.connections.begin() + 1);
  chain.num_rounds.reset();
  const experiment_template fixed(chain);
  EXPECT_EQ(
      commitment_probe_rounds(fixed, parallel_graph(layered(1, 1, 1)).layers),
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
  const auto probe = commitment_probe_rounds(*tmpl, g.layers);
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
        const auto &model = w.model.detector_error_matrix;
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

// F2: a layer importing a round that no window commits finalizes the round's
// rows, but the round's own faults touch them uncommitted.
TEST(Commitment, RejectsAFaultNoWindowCommits) {
  window_graph g;
  window_request w;
  w.period = 3;
  w.commit = round_range{0, 2};
  w.input = {port_disposition::closed, std::nullopt};
  w.output = {port_disposition::closed, round_range{2, 3}};
  g.layers = {{"W", "w", w}};
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: F2: row 8 (round 2) of the 3-round shot, "
               "finalized by W[0]: fault 8 (round 2)",
               "committed by no window");
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

// Imports take their values from exactly one residual.
TEST(CommitmentSends, RejectsTwoResidualsForOneImport) {
  auto g = parallel_graph(layered(2, 2, 3));
  g.edges.push_back({"A", "backward", "B", "receive_2", 0});
  expect_error([&] { validate(arbitrary_memory_spec(5), g); },
               "validate_commitment: B[1] of the",
               "imports rounds [5, 6) but receives 2 residuals");
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
  const auto &h = w.model.detector_error_matrix;
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

} // namespace
} // namespace cudaq::qec
