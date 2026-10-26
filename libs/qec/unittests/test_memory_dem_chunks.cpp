/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/experiments.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudaq::qec {
namespace {

constexpr std::size_t regular_measurements = 2;
constexpr std::size_t readout_measurements = 3;
constexpr std::size_t representative_rounds = 8;

TEST(MemoryDemChunks, ProbabilityComparisonRetainsSmallMechanisms) {
  EXPECT_TRUE(memory_dem_probabilities_equivalent(0, 0));
  EXPECT_TRUE(memory_dem_probabilities_equivalent(0.01, 0.01 + 1e-12));
  EXPECT_FALSE(memory_dem_probabilities_equivalent(0.01, 0.01 + 1e-8));
  const auto smallest = std::numeric_limits<double>::denorm_min();
  EXPECT_TRUE(memory_dem_probabilities_equivalent(smallest, smallest));
  EXPECT_FALSE(memory_dem_probabilities_equivalent(0, smallest));
  EXPECT_FALSE(memory_dem_probabilities_equivalent(smallest, 2 * smallest));
}

struct fixture_fault {
  std::vector<std::size_t> detectors;
  std::vector<std::size_t> observables;
  double probability;
};

// Two distinct prefixes and suffixes, a stationary interior, unequal endpoint
// detector counts, and a terminal record wider than the extraction records.
// This is an independent sparse construction, not a round-trip through the
// phase splitter being tested. Each phase starts at least one fault.
decoder_inputs fixture(std::size_t extraction_rounds = representative_rounds,
                       bool duplicate_faults = false,
                       std::size_t observables = 2) {
  const auto rounds = extraction_rounds + 1;
  std::vector<std::size_t> widths(rounds, regular_measurements);
  widths.front() = widths.back() = 1;
  std::vector<std::size_t> offsets(rounds + 1, 0);
  std::partial_sum(widths.begin(), widths.end(), offsets.begin() + 1);

  decoder_inputs result;
  const auto measurements =
      extraction_rounds * regular_measurements + readout_measurements;
  result.m2d.num_measurements = result.m2o.num_measurements = measurements;
  for (std::size_t round = 0; round < rounds; ++round)
    for (std::size_t detector = 0; detector < widths[round]; ++detector) {
      auto &row = result.m2d.rows.emplace_back();
      if (round > 0 && detector < widths[round - 1])
        row.push_back((round - 1) * regular_measurements + detector);
      row.push_back(round * regular_measurements + detector);
    }
  // O maps fault columns; m2o instead selects raw final-readout support. Keep
  // both nontrivial so accidentally substituting one for the other is visible.
  for (std::size_t observable = 0; observable < observables; ++observable)
    result.m2o.rows.push_back({extraction_rounds * regular_measurements +
                               observable % readout_measurements});

  std::vector<fixture_fault> faults;
  for (std::size_t round = 0; round < rounds; ++round) {
    const double probability = round == 0            ? 0.01
                               : round == 1          ? 0.02
                               : round + 2 == rounds ? 0.04
                               : round + 1 == rounds ? 0.05
                                                     : 0.03;
    for (std::size_t detector = 0; detector < widths[round]; ++detector) {
      const auto observable = detector % observables;
      faults.push_back(
          {{offsets[round] + detector}, {observable}, probability});
      if (duplicate_faults)
        faults.push_back({{offsets[round] + detector}, {observable}, 0.2});
      if (round + 1 < rounds && detector < widths[round + 1])
        faults.push_back(
            {{offsets[round] + detector, offsets[round + 1] + detector},
             {},
             probability / 2});
    }
  }

  auto &dem = result.dem;
  dem.detector_error_matrix =
      cudaqx::tensor<std::uint8_t>({offsets.back(), faults.size()});
  dem.observables_flips_matrix =
      cudaqx::tensor<std::uint8_t>({observables, faults.size()});
  std::fill_n(dem.detector_error_matrix.data(),
              dem.detector_error_matrix.size(), 0);
  std::fill_n(dem.observables_flips_matrix.data(),
              dem.observables_flips_matrix.size(), 0);
  for (std::size_t column = 0; column < faults.size(); ++column) {
    for (const auto detector : faults[column].detectors)
      dem.detector_error_matrix.at({detector, column}) = 1;
    for (const auto observable : faults[column].observables)
      dem.observables_flips_matrix.at({observable, column}) = 1;
    dem.error_rates.push_back(faults[column].probability);
  }
  return result;
}

dem_chunks_spec split(const decoder_inputs &inputs,
                      std::size_t rounds = representative_rounds) {
  return memory_dem_chunks_from_decoder_inputs(
      inputs, rounds, regular_measurements, readout_measurements);
}

using fault_signature =
    std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;

std::map<fault_signature, double>
canonical_faults(const detector_error_model &dem) {
  std::map<fault_signature, double> result;
  for (std::size_t fault = 0; fault < dem.num_error_mechanisms(); ++fault) {
    fault_signature signature;
    for (std::size_t detector = 0; detector < dem.num_detectors(); ++detector)
      if (dem.detector_error_matrix.at({detector, fault}))
        signature.first.push_back(detector);
    for (std::size_t observable = 0; observable < dem.num_observables();
         ++observable)
      if (dem.observables_flips_matrix.at({observable, fault}))
        signature.second.push_back(observable);
    auto &probability = result[signature];
    const auto next = dem.error_rates[fault];
    probability = probability * (1 - next) + next * (1 - probability);
  }
  return result;
}

void expect_equivalent(const dem_chunks_spec &spec,
                       const decoder_inputs &inputs) {
  const auto closed = dem_close(dem_stitch_all(dem_chunks_from_spec(spec)));
  ASSERT_EQ(closed.num_detectors(), inputs.dem.num_detectors());
  ASSERT_EQ(closed.num_observables(), inputs.dem.num_observables());
  const auto expected = canonical_faults(inputs.dem);
  const auto actual = canonical_faults(closed);
  ASSERT_EQ(actual.size(), expected.size());
  for (const auto &[signature, probability] : expected) {
    const auto found = actual.find(signature);
    ASSERT_NE(found, actual.end());
    EXPECT_NEAR(found->second, probability, 1e-14);
  }
  std::vector<std::vector<std::uint32_t>> expected_d;
  expected_d.reserve(inputs.m2d.rows.size());
  for (const auto &row : inputs.m2d.rows)
    expected_d.emplace_back(row.begin(), row.end());
  EXPECT_EQ(dem_chunks_to_d_sparse(spec), expected_d);
  EXPECT_EQ(dem_chunks_measurement_count(spec), inputs.m2d.num_measurements);
}

TEST(MemoryDemChunks, PreservesFaultsMappingsAndTransientEndpoints) {
  const auto inputs = fixture();
  const auto spec = split(inputs);
  EXPECT_EQ(spec.num_rounds, representative_rounds + 1);
  EXPECT_EQ(spec.phases.size(), 5);
  EXPECT_EQ(spec.minimum_rounds(), 5);
  const auto span = spec.repeating_rounds();
  EXPECT_EQ(span.prefix, 2);
  EXPECT_EQ(span.length, 5);
  EXPECT_EQ(span.suffix, 2);
  EXPECT_TRUE(spec.has_D_sparse());
  const auto sequence = spec.phase_sequence();
  EXPECT_EQ(spec.measurements_of(spec.get_phase(sequence.front())), 2);
  EXPECT_EQ(spec.measurements_of(spec.get_phase(sequence.back())), 3);
  expect_equivalent(spec, inputs);
}

TEST(MemoryDemChunks, RetargetingRepeatsBulkWithoutGrowingStoredPhases) {
  auto spec = split(fixture());
  const auto phases = spec.phases;
  const auto connections = spec.connections;
  for (std::size_t rounds : {std::size_t{26}, std::size_t{40}}) {
    spec.num_rounds = rounds + 1;
    EXPECT_EQ(spec.phases, phases);
    EXPECT_EQ(spec.connections, connections);
    expect_equivalent(spec, fixture(rounds));
  }
  spec.num_rounds = std::uint64_t{1} << 40;
  EXPECT_NO_THROW(spec.validate());
  EXPECT_EQ(spec.phases, phases);
  EXPECT_EQ(spec.repeating_rounds().prefix, 2);
  EXPECT_EQ(spec.repeating_rounds().suffix, 2);
}

TEST(MemoryDemChunks, MergesOnlyIdenticalFaultAndObservableSignatures) {
  const auto inputs = fixture(representative_rounds, true);
  const auto spec = split(inputs);
  expect_equivalent(spec, inputs);
  const auto closed = dem_close(dem_stitch_all(dem_chunks_from_spec(spec)));
  EXPECT_LT(closed.num_error_mechanisms(), inputs.dem.num_error_mechanisms());
  const auto &bulk = spec.get_phase(spec.repeating_phase());
  EXPECT_TRUE(std::any_of(
      bulk.error_rates.begin(), bulk.error_rates.end(), [](double value) {
        return std::abs(value - (0.03 * 0.8 + 0.2 * 0.97)) < 1e-14;
      }));
}

TEST(MemoryDemChunks, KeepsEqualDetectorSignaturesWithDifferentLogicalEffects) {
  auto inputs = fixture(representative_rounds, true);
  const auto &h = inputs.dem.detector_error_matrix;
  auto &o = inputs.dem.observables_flips_matrix;
  for (std::size_t fault = 1; fault < inputs.dem.num_error_mechanisms();
       ++fault) {
    bool duplicate = true;
    for (std::size_t detector = 0; detector < inputs.dem.num_detectors();
         ++detector)
      duplicate &= h.at({detector, fault - 1}) == h.at({detector, fault});
    if (duplicate) {
      o.at({0, fault}) ^= 1;
      o.at({1, fault}) ^= 1;
    }
  }
  const auto spec = split(inputs);
  expect_equivalent(spec, inputs);
  const auto closed = dem_close(dem_stitch_all(dem_chunks_from_spec(spec)));
  EXPECT_EQ(closed.num_error_mechanisms(), inputs.dem.num_error_mechanisms());
}

TEST(MemoryDemChunks, AcceptsUniqueErrorIdsAndHighestObservableBit) {
  auto inputs = fixture(representative_rounds, false, 64);
  // A bit at index 63 verifies unsigned shifts rather than an int-width mask.
  for (std::size_t fault = 0; fault < inputs.dem.num_error_mechanisms();
       ++fault)
    if (inputs.dem.observables_flips_matrix.at({0, fault}))
      inputs.dem.observables_flips_matrix.at({63, fault}) = 1;
  inputs.dem.error_ids.emplace(inputs.dem.num_error_mechanisms());
  std::iota(inputs.dem.error_ids->begin(), inputs.dem.error_ids->end(), 100);
  expect_equivalent(split(inputs), inputs);
  EXPECT_THROW(split(fixture(representative_rounds, false, 65)),
               std::invalid_argument);
}

TEST(MemoryDemChunks, RejectsDimensionAndNonbinaryMatrixErrors) {
  auto inputs = fixture();
  inputs.dem.detector_error_matrix = cudaqx::tensor<std::uint8_t>({2});
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.dem.observables_flips_matrix = cudaqx::tensor<std::uint8_t>({2});
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.dem.error_rates.pop_back();
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.dem.observables_flips_matrix = cudaqx::tensor<std::uint8_t>({2, 1});
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.m2d.rows.pop_back();
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.m2o.rows.pop_back();
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.m2d.num_measurements++;
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.m2o.num_measurements++;
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.dem.detector_error_matrix.at({0, 0}) = 2;
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.dem.observables_flips_matrix.at({0, 0}) = 2;
  EXPECT_THROW(split(inputs), std::invalid_argument);
}

TEST(MemoryDemChunks, RejectsInvalidMeasurementMapsBeforeIndexing) {
  auto inputs = fixture();
  inputs.m2d.rows[0].clear();
  EXPECT_THROW(split(inputs), std::invalid_argument);
  for (bool logical_map : {false, true}) {
    inputs = fixture();
    auto &rows = logical_map ? inputs.m2o.rows : inputs.m2d.rows;
    rows[0] = {0, inputs.m2d.num_measurements};
    EXPECT_THROW(split(inputs), std::invalid_argument);
    rows[0] = {1, 0};
    EXPECT_THROW(split(inputs), std::invalid_argument);
    rows[0] = {0, 0};
    EXPECT_THROW(split(inputs), std::invalid_argument);
  }
  inputs = fixture();
  std::swap(inputs.m2d.rows.front(), inputs.m2d.rows.back());
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.m2d.rows[5] = {0, 6};
  EXPECT_THROW(split(inputs), std::invalid_argument);
}

TEST(MemoryDemChunks, RejectsNonemptyMatrixWithNullBorrowedStorage) {
  auto inputs = fixture();
  const auto h_shape = inputs.dem.detector_error_matrix.shape();
  inputs.dem.detector_error_matrix = cudaqx::tensor<std::uint8_t>();
  inputs.dem.detector_error_matrix.borrow(nullptr, h_shape);
  EXPECT_THROW(split(inputs), std::invalid_argument);

  inputs = fixture();
  const auto o_shape = inputs.dem.observables_flips_matrix.shape();
  inputs.dem.observables_flips_matrix = cudaqx::tensor<std::uint8_t>();
  inputs.dem.observables_flips_matrix.borrow(nullptr, o_shape);
  EXPECT_THROW(split(inputs), std::invalid_argument);
}

TEST(MemoryDemChunks, RejectsInvalidProbabilitiesAndCorrelations) {
  for (const auto value : {-0.1, 1.1, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    auto inputs = fixture();
    inputs.dem.error_rates[0] = value;
    EXPECT_THROW(split(inputs), std::invalid_argument);
  }
  auto inputs = fixture();
  inputs.dem.error_ids.emplace(inputs.dem.num_error_mechanisms(), 0);
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs.dem.error_ids->pop_back();
  EXPECT_THROW(split(inputs), std::invalid_argument);
}

TEST(MemoryDemChunks, RejectsUnsupportedFaultsRatherThanDroppingThem) {
  auto inputs = fixture();
  for (std::size_t detector = 0; detector < inputs.dem.num_detectors();
       ++detector)
    inputs.dem.detector_error_matrix.at({detector, 0}) = 0;
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  inputs.dem.detector_error_matrix.at({5, 0}) = 1;
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  // Give every starting fault of the final round its preceding detector, so
  // FINAL has detectors but no locally owned faults. Never fabricate one.
  const auto final_detector = inputs.dem.num_detectors() - 1;
  for (std::size_t fault = 0; fault < inputs.dem.num_error_mechanisms();
       ++fault)
    if (inputs.dem.detector_error_matrix.at({final_detector, fault}))
      inputs.dem.detector_error_matrix.at({final_detector - 1, fault}) = 1;
  EXPECT_THROW(split(inputs), std::invalid_argument);
}

TEST(MemoryDemChunks, RejectsNonstationaryBulkAndInvalidRecordCounts) {
  auto inputs = fixture();
  for (std::size_t fault = 0; fault < inputs.dem.num_error_mechanisms();
       ++fault)
    inputs.dem.error_rates[fault] = 0.001 * (fault + 1);
  EXPECT_THROW(split(inputs), std::invalid_argument);
  inputs = fixture();
  EXPECT_THROW(memory_dem_chunks_from_decoder_inputs(inputs, 0, 2, 3),
               std::invalid_argument);
  EXPECT_THROW(memory_dem_chunks_from_decoder_inputs(inputs, 8, 0, 3),
               std::invalid_argument);
  EXPECT_THROW(memory_dem_chunks_from_decoder_inputs(inputs, 8, 2, 0),
               std::invalid_argument);
  EXPECT_THROW(memory_dem_chunks_from_decoder_inputs(
                   inputs, std::numeric_limits<std::size_t>::max(), 2, 3),
               std::overflow_error);
  EXPECT_THROW(memory_dem_chunks_from_decoder_inputs(
                   inputs, 2, std::numeric_limits<std::size_t>::max(), 3),
               std::overflow_error);
}

} // namespace
} // namespace cudaq::qec
