/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The surface-code app examples' per-round DEM chunker: it turns a memory
// experiment's characterized whole-shot model into a dem_chunks spec with a
// repeating bulk round, as --dem_chunks and --parallel_window need.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

/// Split a characterized memory model into one DEM chunk per measurement
/// round, so a single config serves any number of rounds. Round r's chunk
/// holds the detectors whose latest measurement lies in round r, and the
/// faults whose first detector lies there; a fault must reach no further
/// than the next round's detectors, and a detector no further back than the
/// previous round's measurements. The rounds equal to the middle one are the
/// repeating bulk.
/// @param h_sparse Detector rows of fault columns, -1-terminated.
/// @param o_sparse Observable rows of fault columns, -1-terminated.
/// @param error_rates One prior per fault column.
/// @param d_sparse Detector rows of measurements, -1-terminated.
/// @param num_rounds Syndrome rounds, before the data readout.
/// @param num_syndromes_per_round Measurements of each syndrome round.
/// @param num_data Measurements of the data readout.
/// @throws std::runtime_error if the model does not split into rounds.
inline cudaq::qec::dem_chunks_spec dem_chunks_from_rounds(
    const std::vector<std::int64_t> &h_sparse,
    const std::vector<std::int64_t> &o_sparse,
    const std::vector<double> &error_rates,
    const std::vector<std::int64_t> &d_sparse, std::size_t num_rounds,
    std::size_t num_syndromes_per_round, std::size_t num_data) {
  using namespace cudaq::qec;
  const auto rows_of = [](const std::vector<std::int64_t> &flat) {
    std::vector<std::vector<std::int64_t>> rows(1);
    for (const auto value : flat)
      if (value < 0)
        rows.emplace_back();
      else
        rows.back().push_back(value);
    rows.pop_back();
    return rows;
  };
  const std::size_t S = num_syndromes_per_round, R = num_rounds + 1;
  const auto count = [&](std::size_t r) {
    return r < num_rounds ? S : num_data;
  };
  const auto round_of = [&](std::int64_t m) {
    return std::min(static_cast<std::size_t>(m) / S, num_rounds);
  };

  // Detector rounds, local indices, and D rows in round-local measurements:
  // the previous round's first, then the detector's own round's.
  const auto d_rows = rows_of(d_sparse);
  std::vector<std::size_t> det_round(d_rows.size()), local(d_rows.size());
  std::vector<std::size_t> width(R + 1, 0);
  std::vector<std::vector<std::int64_t>> d_local(R);
  for (std::size_t k = 0; k < d_rows.size(); ++k) {
    if (d_rows[k].empty())
      throw std::runtime_error("dem_chunks_from_rounds: detector " +
                               std::to_string(k) + " reads no measurement");
    std::size_t r = 0;
    for (const auto m : d_rows[k])
      r = std::max(r, round_of(m));
    det_round[k] = r;
    local[k] = width[r]++;
    for (const auto m : d_rows[k]) {
      const std::size_t at = round_of(m);
      const auto offset = static_cast<std::int64_t>(at * S);
      if (at + 1 == r)
        d_local[r].push_back(m - offset);
      else if (at == r)
        d_local[r].push_back(
            m - offset + static_cast<std::int64_t>(r > 0 ? count(r - 1) : 0));
      else
        throw std::runtime_error("dem_chunks_from_rounds: detector " +
                                 std::to_string(k) +
                                 " reads a measurement more than one round "
                                 "before its last");
    }
    d_local[r].push_back(-1);
  }

  // Round-major numbering makes a column's lowest detector index its earliest
  // round, and makes the closed chunks renumber detectors as this DEM does.
  if (!std::is_sorted(det_round.begin(), det_round.end()))
    throw std::runtime_error(
        "dem_chunks_from_rounds: needs detectors numbered round by round");

  struct fault {
    double p = 0.0;
    std::vector<std::int64_t> rows, next_rows;
    std::uint64_t obs = 0;
    auto key() const { return std::tie(rows, next_rows, obs); }
  };
  const std::size_t num_columns = error_rates.size();
  std::vector<std::vector<std::size_t>> columns(num_columns);
  const auto h_rows = rows_of(h_sparse);
  for (std::size_t det = 0; det < h_rows.size(); ++det)
    for (const auto col : h_rows[det])
      columns[col].push_back(det);
  std::vector<std::uint64_t> column_obs(num_columns, 0);
  const auto o_rows = rows_of(o_sparse);
  if (o_rows.size() > 64)
    throw std::runtime_error(
        "dem_chunks_from_rounds: supports at most 64 observables");
  for (std::size_t o = 0; o < o_rows.size(); ++o)
    for (const auto col : o_rows[o])
      column_obs[col] |= std::uint64_t{1} << o;

  std::vector<std::vector<fault>> rounds(R);
  for (std::size_t col = 0; col < num_columns; ++col) {
    if (columns[col].empty())
      throw std::runtime_error(
          "dem_chunks_from_rounds: needs every column to flip a "
          "detector");
    const std::size_t r = det_round[columns[col].front()];
    fault f;
    f.p = error_rates[col];
    f.obs = column_obs[col];
    for (const auto det : columns[col]) {
      const std::size_t at = det_round[det];
      // Both directions: an earlier round would land on the outgoing seam,
      // which is the wrong boundary rather than a wider one.
      if (at != r && at != r + 1)
        throw std::runtime_error(
            "dem_chunks_from_rounds: needs every column within two "
            "consecutive rounds");
      (at == r ? f.rows : f.next_rows)
          .push_back(static_cast<std::int64_t>(local[det]));
    }
    rounds[r].push_back(std::move(f));
  }
  for (std::size_t r = 0; r < R; ++r) {
    auto &faults = rounds[r];
    std::sort(faults.begin(), faults.end(),
              [](const fault &a, const fault &b) { return a.key() < b.key(); });
    std::vector<fault> merged;
    for (auto &f : faults) {
      if (!merged.empty() && merged.back().key() == f.key()) {
        double &p = merged.back().p;
        p = p * (1 - f.p) + f.p * (1 - p);
      } else {
        merged.push_back(std::move(f));
      }
    }
    // Every fault here starts in round r, so ordering the round's columns as
    // pcm_is_sorted() would, with the next round's rows one block later,
    // leaves the closed model sorted, as sliding_window requires.
    const auto block = static_cast<std::uint32_t>(
        std::max<std::size_t>({width[r], width[r + 1], 1}));
    std::vector<std::vector<std::uint32_t>> local_rows;
    for (const auto &f : merged) {
      auto &rows = local_rows.emplace_back(f.rows.begin(), f.rows.end());
      for (const auto row : f.next_rows)
        rows.push_back(block + static_cast<std::uint32_t>(row));
    }
    faults.clear();
    for (const auto i : get_sorted_pcm_column_indices(local_rows, block))
      faults.push_back(std::move(merged[i]));
  }

  const auto same = [&](std::size_t a, std::size_t b) {
    if (width[a] != width[b] || width[a + 1] != width[b + 1] ||
        count(a) != count(b) || (a > 0) != (b > 0) ||
        (a > 0 && count(a - 1) != count(b - 1)) || d_local[a] != d_local[b] ||
        rounds[a].size() != rounds[b].size())
      return false;
    for (std::size_t i = 0; i < rounds[a].size(); ++i) {
      const fault &x = rounds[a][i], &y = rounds[b][i];
      if (x.key() != y.key() || std::abs(x.p - y.p) > 1e-9 * std::max(x.p, y.p))
        return false;
    }
    return true;
  };
  const std::size_t middle = R / 2;
  std::size_t bulk_first = middle, bulk_last = middle;
  while (bulk_first > 0 && same(bulk_first - 1, middle))
    --bulk_first;
  while (bulk_last + 1 < R && same(bulk_last + 1, middle))
    ++bulk_last;
  if (bulk_last == bulk_first)
    throw std::runtime_error(
        "dem_chunks_from_rounds: found no repeating round in " +
        std::to_string(num_rounds) + " rounds; raise num_rounds");

  const auto chunk_of = [&](std::size_t r) {
    const auto &faults = rounds[r];
    const auto seam_rows = [&](std::size_t rows, bool next) {
      std::vector<std::vector<std::int64_t>> by_row(rows);
      for (std::size_t i = 0; i < faults.size(); ++i)
        for (const auto row : next ? faults[i].next_rows : faults[i].rows)
          by_row[row].push_back(static_cast<std::int64_t>(i));
      std::vector<std::int64_t> sparse;
      for (const auto &row : by_row) {
        sparse.insert(sparse.end(), row.begin(), row.end());
        sparse.push_back(-1);
      }
      return sparse;
    };
    dem_chunk_spec spec;
    spec.num_faults = faults.size();
    for (const auto &f : faults)
      spec.error_rates.push_back(f.p);
    spec.seam_specs.push_back(
        {seam_name::prev_round, {seam_rows(width[r], false), {}}});
    if (r + 1 < R)
      spec.seam_specs.push_back(
          {seam_name::next_round, {seam_rows(width[r + 1], true), {}}});
    for (std::size_t o = 0; o < o_rows.size(); ++o) {
      for (std::size_t i = 0; i < faults.size(); ++i)
        if (faults[i].obs >> o & 1)
          spec.O_sparse.push_back(static_cast<std::int64_t>(i));
      spec.O_sparse.push_back(-1);
    }
    spec.D_sparse = d_local[r];
    if (count(r) != S)
      spec.num_measurements = count(r);
    return spec;
  };

  dem_chunks_spec spec;
  spec.seam = {seam_name::next_round, seam_name::prev_round};
  spec.num_rounds = R;
  spec.measurements_per_round = S;
  std::vector<phase_id> chain;
  const auto add_phase = [&](const std::string &name, std::size_t r) {
    const phase_id id = phase_id::from_name(name);
    spec.phases.push_back({id, chunk_of(r)});
    chain.push_back(id);
  };
  for (std::size_t r = 0; r < bulk_first; ++r)
    add_phase("before" + std::to_string(r), r);
  add_phase("bulk", middle);
  for (std::size_t r = bulk_last + 1; r < R; ++r)
    add_phase("after" + std::to_string(r - bulk_last - 1), r);
  for (std::size_t i = 0; i + 1 < chain.size(); ++i)
    spec.connections.push_back({chain[i], chain[i + 1]});
  spec.connections.push_back({chain[bulk_first], chain[bulk_first]});
  printf("DEM chunks: %zu rounds before the repeating round, %zu after\n",
         bulk_first, R - 1 - bulk_last);
  return spec;
}
