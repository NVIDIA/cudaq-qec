/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/
#include "edem_memory_gadgets.h"
#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/edem_composer.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <set>
#include <stdexcept>

namespace {
using edem_example::bytes;
using edem_example::check;
using edem_example::construct;
using edem_example::from_bytes;

cudaq::edem::PackedEDEM compose(std::span<const cudaq_edem_view> occurrences) {
  cudaq::edem::PackedEDEM result;
  check(cudaq::edem::compose_chain(
      occurrences.data(), static_cast<uint32_t>(occurrences.size()), result));
  return result;
}

bool is_dem(const cudaq::edem::PackedEDEM &model) {
  uint32_t closed;
  check(cudaq_edem_is_dem(&model.view(), &closed));
  return closed != 0;
}

cudaq::edem::PackedEDEM project_dem(const cudaq::edem::PackedEDEM &model) {
  cudaq::edem::PackedEDEM result;
  check(cudaq::edem::project_DEM(model, result));
  return result;
}

void check_samples(const cudaq::qec::detector_error_model &dem, size_t shots) {
  auto [checks, errors] = cudaq::qec::dem_sampler::cpu::sample_dem(
      dem.detector_error_matrix, shots, dem.error_rates, 7);
  ASSERT_EQ(checks.shape(), (std::vector<size_t>{shots, dem.num_detectors()}));
  ASSERT_EQ(errors.shape(),
            (std::vector<size_t>{shots, dem.num_error_mechanisms()}));
  for (size_t s = 0; s < shots; ++s)
    for (size_t d = 0; d < dem.num_detectors(); ++d) {
      uint8_t expected = 0;
      for (size_t f = 0; f < dem.num_error_mechanisms(); ++f)
        expected ^= errors.at({s, f}) & dem.detector_error_matrix.at({d, f});
      ASSERT_EQ(checks.at({s, d}), expected)
          << "shot " << s << ", detector " << d;
    }
}

TEST(EDEMComposer, MemoryChain) {
  auto prep = edem_memory_demo::prep();
  auto sec = edem_memory_demo::sec();
  auto mz = edem_memory_demo::mz();
  for (size_t rounds : {0, 1, 3}) {
    SCOPED_TRACE(rounds);
    auto experiment =
        edem_memory_demo::memory_experiment(prep, sec, mz, rounds);
    auto composed = compose(experiment);
    ASSERT_TRUE(is_dem(composed));
    auto projected = project_dem(composed);
    cudaq_edem_dem_view projected_view{projected.view()};
    auto dem = cudaq::qec::dem_from_edem({projected.view()});
    size_t faults = rounds + 2, detectors = rounds + 1;
    ASSERT_EQ(dem.num_error_mechanisms(), faults);
    ASSERT_EQ(dem.num_detectors(), detectors);
    ASSERT_EQ(dem.num_observables(), 1);
    ASSERT_TRUE(dem.error_ids.has_value());
    EXPECT_EQ(std::set(dem.error_ids->begin(), dem.error_ids->end()).size(),
              faults)
        << "repeated SECs shared a source fault identity";
    for (size_t f = 0; f < faults; ++f) {
      EXPECT_DOUBLE_EQ(dem.error_rates[f], (f == 0            ? .01
                                            : f + 1 == faults ? .03
                                                              : .02));
      for (size_t d = 0; d < detectors; ++d)
        EXPECT_EQ(dem.detector_error_matrix.at({d, f}), f == d || f == d + 1);
      EXPECT_EQ(dem.observables_flips_matrix.at({0, f}),
                f == 0 || f + 1 == faults);
    }
    // Compare every fault pattern to an independently specified recurrence.
    for (unsigned pattern = 0; pattern < (1u << faults); ++pattern) {
      std::vector<uint8_t> input(faults), output(detectors + 1);
      for (size_t f = 0; f < faults; ++f)
        input[f] = (pattern >> f) & 1;
      ASSERT_EQ(cudaq_gf2_map_apply(&projected_view.edem.map, input.data(),
                                    input.size(), output.data(), output.size()),
                CUDAQ_GF2_OK);
      for (size_t d = 0; d < detectors; ++d)
        EXPECT_EQ(output[d], input[d] ^ input[d + 1]);
      EXPECT_EQ(output.back(), input.front() ^ input.back());
    }
    check_samples(dem, 4096);
    check_samples(dem, 0);
  }
}

TEST(EDEMComposer, RejectsIncompleteExperiments) {
  auto prep = edem_memory_demo::prep();
  auto sec = edem_memory_demo::sec();
  auto mz = edem_memory_demo::mz();
  EXPECT_FALSE(is_dem(prep));
  EXPECT_FALSE(is_dem(sec));
  EXPECT_FALSE(is_dem(mz));
  EXPECT_THROW(project_dem(prep), std::invalid_argument);
  EXPECT_THROW(project_dem(sec), std::invalid_argument);
  EXPECT_THROW(project_dem(mz), std::invalid_argument);
  EXPECT_THROW(edem_memory_demo::memory_experiment(sec, sec, mz),
               std::invalid_argument);
  EXPECT_THROW(edem_memory_demo::memory_experiment(prep, prep, mz),
               std::invalid_argument);
  EXPECT_THROW(edem_memory_demo::memory_experiment(prep, sec, sec),
               std::invalid_argument);
  EXPECT_THROW(edem_memory_demo::memory_experiment(prep, sec, mz, UINT32_MAX),
               std::invalid_argument);
  std::array incomplete{prep.view(), sec.view()};
  auto open = compose(incomplete);
  EXPECT_FALSE(is_dem(open));
  EXPECT_THROW(project_dem(open), std::invalid_argument);
  EXPECT_THROW(cudaq::qec::dem_from_edem({open.view()}), std::invalid_argument);
}

TEST(EDEMComposer, OpenCompositionPreservesBoundaries) {
  auto sec = edem_memory_demo::sec();
  // Preserve actual exposed V/S/L inputs and outputs when composing open SECs.
  std::array pair{sec.view(), sec.view()};
  auto joined = compose(pair);
  for (unsigned pattern = 0; pattern < 64; ++pattern) {
    std::array<uint8_t, 6> bits{}, response{};
    for (size_t i = 0; i < bits.size(); ++i)
      bits[i] = (pattern >> i) & 1;
    std::array<uint8_t, 5> first_input{bits[0], bits[2], bits[3], bits[4],
                                       bits[5]},
        first_output{}, second_output{};
    ASSERT_EQ(cudaq_gf2_map_apply(&sec.view().map, first_input.data(), 5,
                                  first_output.data(), 5),
              CUDAQ_GF2_OK);
    auto second_input = first_output;
    second_input[0] = bits[1];
    ASSERT_EQ(cudaq_gf2_map_apply(&sec.view().map, second_input.data(), 5,
                                  second_output.data(), 5),
              CUDAQ_GF2_OK);
    ASSERT_EQ(cudaq_gf2_map_apply(&joined.view().map, bits.data(), 6,
                                  response.data(), 6),
              CUDAQ_GF2_OK);
    EXPECT_EQ(response[0], first_output[0]);
    EXPECT_EQ(response[1], second_output[0]);
    for (size_t i = 1; i < second_output.size(); ++i)
      EXPECT_EQ(response[i + 1], second_output[i]);
  }
}

TEST(EDEMComposer, CompositionIsAssociative) {
  auto prep = edem_memory_demo::prep();
  auto sec = edem_memory_demo::sec();
  auto mz = edem_memory_demo::mz();
  std::array prefix{prep.view(), sec.view()};
  auto open = compose(prefix);
  std::array nested_occurrences{open.view(), sec.view(), sec.view(), mz.view()};
  auto nested = compose(nested_occurrences);
  auto experiment = edem_memory_demo::memory_experiment(prep, sec, mz);
  auto flat = compose(experiment);
  ASSERT_EQ(bytes(nested).size(), bytes(flat).size());
  EXPECT_TRUE(std::equal(bytes(nested).begin(), bytes(nested).end(),
                         bytes(flat).begin()));
}

cudaq::edem::PackedEDEM law_fixture() {
  uint32_t independent_id = 1, exclusive_ids[]{0, 2}, other_id = 0;
  double independent_p = 1, exclusive_p[]{.25, .75}, other_p = .5;
  cudaq_edem_fault_group groups[]{
      {CUDAQ_EDEM_EXCLUSIVE, 2, exclusive_ids, exclusive_p},
      {CUDAQ_EDEM_INDEPENDENT, 1, &independent_id, &independent_p}};
  cudaq_edem_fault_group other_group{CUDAQ_EDEM_INDEPENDENT, 1, &other_id,
                                     &other_p};
  cudaq_edem_fault_distribution faults[]{{3, 2, groups}, {1, 1, &other_group}};
  uint32_t detector_size = 3, observable_size = 1;
  uint64_t offsets[]{0, 1, 2, 3}, observable_offsets[]{0, 1, 1, 2};
  uint32_t rows[]{0, 1, 2}, observable_rows[]{0, 0};
  cudaq_edem_block blocks[]{{{CUDAQ_EDEM_DETECTOR, 0},
                             {CUDAQ_EDEM_FAULT, 0},
                             {3, 3, 3, offsets, rows}},
                            {{CUDAQ_EDEM_OBSERVABLE, 0},
                             {CUDAQ_EDEM_FAULT, 0},
                             {1, 3, 2, observable_offsets, observable_rows}}};
  cudaq_edem_description d{
      {0, 0, 0, 2, 1, 1, 0, 2}, nullptr,          nullptr, nullptr, faults,
      &detector_size,           &observable_size, nullptr, blocks};
  return construct(d);
}

TEST(EDEMComposer, IndependentFaultApproximation) {
  auto model = law_fixture();
  ASSERT_TRUE(is_dem(model));
  auto projected = project_dem(model);
  cudaq_edem_dem_view native{projected.view()};
  cudaq_edem_fault_group source_group;
  check(cudaq_edem_get_group(&native.edem, 0, 1, &source_group));
  EXPECT_EQ(source_group.kind, CUDAQ_EDEM_EXCLUSIVE);
  EXPECT_EQ(source_group.size, 2);
  auto dem = cudaq::qec::dem_from_edem({projected.view()});
  EXPECT_EQ(dem.error_rates, (std::vector<double>{.25, 1, .75, .5}));
  ASSERT_EQ(dem.num_error_mechanisms(), 4) << "silent fault column was removed";
  ASSERT_TRUE(dem.error_ids.has_value());
  const auto &ids = *dem.error_ids;
  EXPECT_EQ(std::set(ids.begin(), ids.end()).size(), 4);
  auto [checks, errors] = cudaq::qec::dem_sampler::cpu::sample_dem(
      dem.detector_error_matrix, 8192, dem.error_rates, 31);
  size_t first = 0, both = 0, neither = 0;
  for (size_t shot = 0; shot < 8192; ++shot) {
    ASSERT_EQ(errors.at({shot, 1}), 1) << "probability-one fault did not occur";
    first += errors.at({shot, 0});
    both += errors.at({shot, 0}) && errors.at({shot, 2});
    neither += !errors.at({shot, 0}) && !errors.at({shot, 2});
  }
  EXPECT_GT(first, 1600);
  EXPECT_LT(first, 2500);
  // Each has probability 0.25 * 0.75 after factorization; both are impossible
  // in this fixture's exact categorical law. This checks the documented loss.
  EXPECT_GT(both, 1200);
  EXPECT_LT(both, 1900);
  EXPECT_GT(neither, 1200);
  EXPECT_LT(neither, 1900);
  check(cudaq_edem_get_group(&native.edem, 0, 1, &source_group));
  EXPECT_EQ(source_group.kind, CUDAQ_EDEM_EXCLUSIVE);
  EXPECT_EQ(source_group.size, 2);
  check_samples(dem, 128);
  check_samples(dem, 0);
}

TEST(EDEMComposer, OwnsConvertedData) {
  auto projected = [] {
    auto prep = edem_memory_demo::prep(), sec = edem_memory_demo::sec(),
         mz = edem_memory_demo::mz();
    auto experiment = edem_memory_demo::memory_experiment(prep, sec, mz);
    auto composed = compose(experiment);
    return project_dem(composed);
  }();
  // All definitions and the composed EDEM are gone before conversion.
  auto dem = cudaq::qec::dem_from_edem({projected.view()});
  check_samples(dem, 16);
  std::vector<std::byte> unaligned(bytes(projected).size() + 1);
  std::copy(bytes(projected).begin(), bytes(projected).end(),
            unaligned.begin() + 1);
  auto copy = from_bytes(std::span(unaligned).subspan(1));
  auto first_byte = bytes(copy).front();
  ASSERT_TRUE(is_dem(copy));
  unaligned[1] = std::byte{0};
  EXPECT_EQ(bytes(copy).front(), first_byte);
  EXPECT_TRUE(is_dem(copy));
  EXPECT_THROW(from_bytes({}), std::invalid_argument);
  std::array<std::byte, 8> invalid{};
  EXPECT_THROW(from_bytes(invalid), std::invalid_argument);
  // QEC owns its conversion even after all packed input storage is released.
  projected.reset();
  copy.reset();
  check_samples(dem, 16);
}

TEST(EDEMComposer, RejectsSuccessOutputs) {
  uint32_t size = 1;
  cudaq_edem_description d{};
  d.shape.success_ports = 1;
  d.success_sizes = &size;
  auto heralded = construct(d);
  ASSERT_TRUE(is_dem(heralded));
  auto success_dem = project_dem(heralded);
  EXPECT_THROW(cudaq::qec::dem_from_edem({success_dem.view()}),
               std::invalid_argument);
}

TEST(EDEMComposer, EmptyModel) {
  cudaq_edem_description empty{};
  auto empty_dem = project_dem(construct(empty));
  auto qec_empty = cudaq::qec::dem_from_edem({empty_dem.view()});
  check_samples(qec_empty, 0);
  check_samples(qec_empty, 4);
}

} // namespace
