/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaq::qec {
namespace {

constexpr std::size_t max_observables =
    std::numeric_limits<std::uint64_t>::digits;

[[noreturn]] void invalid(const std::string &reason) {
  throw std::invalid_argument("memory DEM chunks: " + reason);
}

void validate_measurement_map(const std::vector<std::vector<std::size_t>> &rows,
                              std::size_t measurements, const char *name) {
  for (const auto &row : rows) {
    if (!std::is_sorted(row.begin(), row.end()) ||
        std::adjacent_find(row.begin(), row.end()) != row.end())
      invalid(std::string(name) + " rows must have sorted unique indices");
    for (const auto measurement : row)
      if (measurement >= measurements)
        invalid(std::string(name) + " measurement index is out of range");
  }
}

struct local_fault {
  double probability = 0.0;
  std::vector<std::int64_t> rows, next_rows;
  std::uint64_t observables = 0;

  auto key() const { return std::tie(rows, next_rows, observables); }
};

} // namespace

dem_chunks_spec memory_dem_chunks_from_decoder_inputs(
    const decoder_inputs &inputs, std::size_t extraction_rounds,
    std::size_t measurements_per_round, std::size_t final_measurements) {
  const auto size_max = std::numeric_limits<std::size_t>::max();
  const auto index_max =
      std::numeric_limits<sparse_binary_matrix::index_type>::max();
  if (extraction_rounds == 0 || measurements_per_round == 0 ||
      final_measurements == 0)
    invalid("extraction rounds and measurement widths must be positive");
  if (extraction_rounds >= size_max - 1 ||
      extraction_rounds >
          (size_max - final_measurements) / measurements_per_round)
    throw std::overflow_error("memory DEM chunks: measurement count overflow");
  const auto measurement_count =
      extraction_rounds * measurements_per_round + final_measurements;
  if (measurement_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("memory DEM chunks: measurement index overflow");
  const auto round_count = extraction_rounds + 1;
  const auto &dem = inputs.dem;
  const auto &h = dem.detector_error_matrix;
  const auto &o = dem.observables_flips_matrix;
  if (h.rank() != 2 || o.rank() != 2)
    invalid("H and O must be rank-two matrices");
  const auto h_shape = h.shape(), o_shape = o.shape();
  const auto detectors = h_shape[0], faults = h_shape[1];
  const auto observables = o_shape[0];
  if (detectors > index_max || faults > index_max)
    throw std::overflow_error(
        "memory DEM chunks: DEM dimensions exceed uint32");
  if (detectors == 0 || inputs.m2d.rows.size() != detectors ||
      o_shape[1] != faults || dem.error_rates.size() != faults ||
      inputs.m2o.rows.size() != observables ||
      inputs.m2d.num_measurements != measurement_count ||
      inputs.m2o.num_measurements != measurement_count)
    invalid("H, O, rates and measurement-map dimensions disagree");
  if (observables > max_observables)
    invalid("at most 64 observables are supported");
  if (faults != 0 &&
      (detectors > size_max / faults || observables > size_max / faults))
    throw std::overflow_error("memory DEM chunks: dense matrix size overflow");
  const auto h_elements = detectors * faults;
  const auto o_elements = observables * faults;
  if (h.size() != h_elements || o.size() != o_elements)
    invalid("dense matrix storage size disagrees with its dimensions");
  const auto *h_data = h.data();
  const auto *o_data = o.data();
  if ((h_elements != 0 && h_data == nullptr) ||
      (o_elements != 0 && o_data == nullptr))
    invalid("a nonempty dense matrix has no storage");
  if (faults == 0)
    invalid("zero-fault phases are not supported by extended_dem on this "
            "framework revision");
  validate_measurement_map(inputs.m2d.rows, measurement_count, "m2d");
  validate_measurement_map(inputs.m2o.rows, measurement_count, "m2o");
  for (const auto probability : dem.error_rates)
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
      invalid("fault probabilities must be finite and in [0,1]");
  if (dem.error_ids) {
    if (dem.error_ids->size() != faults)
      invalid("error_ids count differs from the number of faults");
    const std::set<std::size_t> ids(dem.error_ids->begin(),
                                    dem.error_ids->end());
    if (ids.size() != faults)
      invalid("correlated error_ids are not supported");
  }

  const auto measurements_of = [&](std::size_t round) {
    return round < extraction_rounds ? measurements_per_round
                                     : final_measurements;
  };
  const auto measurement_round = [&](std::size_t measurement) {
    return std::min(measurement / measurements_per_round, extraction_rounds);
  };
  std::vector<std::size_t> detector_round(detectors), local_detector(detectors);
  // The trailing sentinel is the nonexistent next detector round of FINAL.
  std::vector<std::size_t> widths(round_count + 1, 0);
  std::vector<std::vector<std::int64_t>> local_d(round_count);
  for (std::size_t detector = 0; detector < detectors; ++detector) {
    const auto &row = inputs.m2d.rows[detector];
    if (row.empty())
      invalid("a detector reads no measurements");
    const auto round = measurement_round(row.back());
    detector_round[detector] = round;
    local_detector[detector] = widths[round]++;
    for (const auto measurement : row) {
      const auto at = measurement_round(measurement);
      const auto offset = at * measurements_per_round;
      if (at + 1 == round)
        local_d[round].push_back(measurement - offset);
      else if (at == round)
        local_d[round].push_back(measurement - offset +
                                 (round > 0 ? measurements_of(round - 1) : 0));
      else
        invalid("a detector reads measurements more than one round before "
                "its latest measurement");
    }
    local_d[round].push_back(-1);
  }
  if (!std::is_sorted(detector_round.begin(), detector_round.end()))
    invalid("detectors must be numbered round by round");

  std::vector<std::vector<std::size_t>> fault_detectors(faults);
  // cudaqx::tensor uses contiguous row-major storage, including transpose()
  // results. As in sparse_binary_matrix's dense conversion, borrow each row
  // once instead of constructing a tensor accessor for every matrix element.
  for (std::size_t detector = 0; detector < detectors; ++detector) {
    const auto *row = h_data + detector * faults;
    for (std::size_t fault = 0; fault < faults; ++fault) {
      const auto value = row[fault];
      if (value > 1)
        invalid("H must be binary");
      if (value)
        fault_detectors[fault].push_back(detector);
    }
  }
  std::vector<std::uint64_t> fault_observables(faults, 0);
  for (std::size_t observable = 0; observable < observables; ++observable) {
    const auto *row = o_data + observable * faults;
    for (std::size_t fault = 0; fault < faults; ++fault) {
      const auto value = row[fault];
      if (value > 1)
        invalid("O must be binary");
      if (value)
        fault_observables[fault] |= std::uint64_t{1} << observable;
    }
  }

  std::vector<std::vector<local_fault>> rounds(round_count);
  for (std::size_t fault = 0; fault < faults; ++fault) {
    if (fault_detectors[fault].empty())
      invalid("every fault must flip a detector");
    const auto round = detector_round[fault_detectors[fault].front()];
    local_fault local;
    local.probability = dem.error_rates[fault];
    local.observables = fault_observables[fault];
    for (const auto detector : fault_detectors[fault]) {
      const auto at = detector_round[detector];
      if (at != round && at != round + 1)
        invalid("a fault spans more than two consecutive detector rounds");
      (at == round ? local.rows : local.next_rows)
          .push_back(static_cast<std::int64_t>(local_detector[detector]));
    }
    rounds[round].push_back(std::move(local));
  }
  for (std::size_t round = 0; round < round_count; ++round) {
    auto &local_faults = rounds[round];
    if (local_faults.empty())
      invalid("phase " + std::to_string(round) +
              " has no faults; zero-fault phases are not supported by "
              "extended_dem on this framework revision");
    std::sort(local_faults.begin(), local_faults.end(),
              [](const local_fault &a, const local_fault &b) {
                return a.key() < b.key();
              });
    std::vector<local_fault> merged;
    for (auto &fault : local_faults) {
      if (!merged.empty() && merged.back().key() == fault.key()) {
        auto &probability = merged.back().probability;
        probability = probability * (1 - fault.probability) +
                      fault.probability * (1 - probability);
      } else {
        merged.push_back(std::move(fault));
      }
    }
    const auto block = static_cast<std::uint32_t>(
        std::max<std::size_t>({widths[round], widths[round + 1], 1}));
    std::vector<std::vector<std::uint32_t>> local_rows;
    local_rows.reserve(merged.size());
    for (const auto &fault : merged) {
      auto &rows =
          local_rows.emplace_back(fault.rows.begin(), fault.rows.end());
      for (const auto row : fault.next_rows) {
        if (static_cast<std::uint64_t>(block) + row > index_max)
          throw std::overflow_error(
              "memory DEM chunks: local fault index exceeds uint32");
        rows.push_back(block + static_cast<std::uint32_t>(row));
      }
    }
    local_faults.clear();
    for (const auto index : get_sorted_pcm_column_indices(local_rows, block))
      local_faults.push_back(std::move(merged[index]));
  }

  const auto same_phase = [&](std::size_t a, std::size_t b) {
    if (widths[a] != widths[b] || widths[a + 1] != widths[b + 1] ||
        measurements_of(a) != measurements_of(b) || (a > 0) != (b > 0) ||
        (a > 0 && measurements_of(a - 1) != measurements_of(b - 1)) ||
        local_d[a] != local_d[b] || rounds[a].size() != rounds[b].size())
      return false;
    for (std::size_t i = 0; i < rounds[a].size(); ++i) {
      const auto &x = rounds[a][i], &y = rounds[b][i];
      if (x.key() != y.key() || !details::memory_dem_probabilities_equivalent(
                                    x.probability, y.probability))
        return false;
    }
    return true;
  };
  const auto middle = round_count / 2;
  auto bulk_first = middle, bulk_last = middle;
  while (bulk_first > 0 && same_phase(bulk_first - 1, middle))
    --bulk_first;
  while (bulk_last + 1 < round_count && same_phase(bulk_last + 1, middle))
    ++bulk_last;
  if (bulk_first == bulk_last)
    invalid("found no repeating round; characterize more extraction "
            "rounds or use a circuit with a stationary bulk");

  const auto chunk_of = [&](std::size_t round) {
    const auto &local_faults = rounds[round];
    const auto seam_rows = [&](std::size_t count, bool next) {
      std::vector<std::vector<std::int64_t>> by_row(count);
      for (std::size_t i = 0; i < local_faults.size(); ++i)
        for (const auto row :
             next ? local_faults[i].next_rows : local_faults[i].rows)
          by_row[row].push_back(static_cast<std::int64_t>(i));
      std::vector<std::int64_t> sparse;
      for (const auto &row : by_row) {
        sparse.insert(sparse.end(), row.begin(), row.end());
        sparse.push_back(-1);
      }
      return sparse;
    };
    dem_chunk_spec chunk;
    chunk.num_faults = local_faults.size();
    chunk.error_rates.reserve(local_faults.size());
    for (const auto &fault : local_faults)
      chunk.error_rates.push_back(fault.probability);
    chunk.seam_specs.push_back(
        {seam_name::prev_round, {seam_rows(widths[round], false), {}}});
    if (round + 1 < round_count)
      chunk.seam_specs.push_back(
          {seam_name::next_round, {seam_rows(widths[round + 1], true), {}}});
    for (std::size_t observable = 0; observable < observables; ++observable) {
      for (std::size_t i = 0; i < local_faults.size(); ++i)
        if ((local_faults[i].observables >> observable) & 1)
          chunk.O_sparse.push_back(static_cast<std::int64_t>(i));
      chunk.O_sparse.push_back(-1);
    }
    chunk.D_sparse = local_d[round];
    if (measurements_of(round) != measurements_per_round)
      chunk.num_measurements = measurements_of(round);
    return chunk;
  };

  dem_chunks_spec result;
  result.seam = {seam_name::next_round, seam_name::prev_round};
  result.num_rounds = round_count;
  result.measurements_per_round = measurements_per_round;
  std::vector<phase_id> chain;
  const auto add_phase = [&](const std::string &name, std::size_t round) {
    const auto id = phase_id::from_name(name);
    result.phases.push_back({id, chunk_of(round)});
    chain.push_back(id);
  };
  for (std::size_t round = 0; round < bulk_first; ++round)
    add_phase("before" + std::to_string(round), round);
  add_phase("bulk", middle);
  for (auto round = bulk_last + 1; round < round_count; ++round)
    add_phase("after" + std::to_string(round - bulk_last - 1), round);
  for (std::size_t i = 0; i + 1 < chain.size(); ++i)
    result.connections.push_back({chain[i], chain[i + 1]});
  result.connections.push_back({chain[bulk_first], chain[bulk_first]});
  result.validate();
  return result;
}

} // namespace cudaq::qec
