/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "raw_replay.h"
#include "cudaq/platform.h"
#include "cudaq/qec/noise_model.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cudaq::qec::memory_experiment {
namespace {

preparation_request replay_request(memory_basis basis = memory_basis::z) {
  return {{"surface_code", 3, 26, 1, basis}};
}

std::vector<raw_replay_shot> collect(const preparation_request &request,
                                     const prepared_model &model,
                                     std::uint64_t shots = 2,
                                     std::uint64_t seed = 19) {
  std::vector<raw_replay_shot> result;
  sample_raw_replay(
      request, model, shots, seed,
      [&](const raw_replay_shot &shot) { result.push_back(shot); });
  return result;
}

TEST(MemoryExperimentReplay, BothBasesPreserveRawRecordLayoutAndLogicalMap) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const auto request = replay_request(basis);
    const auto model = prepare_model(request);
    const auto samples = collect(request, model);
    ASSERT_EQ(samples.size(), 2);
    ASSERT_EQ(model.regular_measurements, 8);
    ASSERT_EQ(model.final_measurements, 9);
    for (std::size_t shot = 0; shot < samples.size(); ++shot) {
      const auto &sample = samples[shot];
      EXPECT_EQ(sample.patch, 0);
      EXPECT_EQ(sample.shot, shot);
      EXPECT_EQ(sample.seed, 19 + shot);
      ASSERT_EQ(sample.measurements.size(), 217);
      EXPECT_EQ(sample.measurements.find_first_not_of("01"), std::string::npos);
      ASSERT_EQ(sample.raw_logical.size(), model.final_m2o.size());
      for (std::size_t logical = 0; logical < model.final_m2o.size();
           ++logical) {
        bool parity = false;
        for (const auto data : model.final_m2o[logical])
          parity ^= sample.measurements[208 + data] == '1';
        EXPECT_EQ(sample.raw_logical[logical], parity ? '1' : '0');
      }
    }
  }
}

TEST(MemoryExperimentReplay, RepeatedSeedsReproduceBothBases) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const auto request = replay_request(basis);
    const auto model = prepare_model(request);
    const auto first = collect(request, model);
    const auto repeated = collect(request, model);
    ASSERT_EQ(first.size(), repeated.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
      EXPECT_EQ(first[i].seed, repeated[i].seed);
      EXPECT_EQ(first[i].measurements, repeated[i].measurements);
      EXPECT_EQ(first[i].raw_logical, repeated[i].raw_logical);
    }
  }
}

TEST(MemoryExperimentReplay, SamplingCannotChangePreparedNoiseOrNormalization) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    auto request = replay_request(basis);
    const auto model = prepare_model(request);
    request.cnot_error_rate = 0.002;
    EXPECT_THROW(collect(request, model, 1), std::invalid_argument);
    request.cnot_error_rate = 0.0;
    EXPECT_THROW(collect(request, model, 1), std::invalid_argument);
    request.cnot_error_rate = 0.001;
    request.decompose_errors = false;
    EXPECT_THROW(collect(request, model, 1), std::invalid_argument);
  }
}

TEST(MemoryExperimentReplay, SamplingReplacesAndClearsExistingGlobalNoise) {
  auto request = replay_request();
  const auto model = prepare_model(request);
  const auto baseline = collect(request, model, 1);
  cudaq::noise_model inherited;
  inherited.add_all_qubit_channel("x", two_qubit_depolarization(0.25),
                                  /*num_controls=*/1);
  cudaq::get_platform().set_noise(&inherited);
  ASSERT_NE(cudaq::get_platform().get_noise(), nullptr);
  const auto samples = collect(request, model, 1);
  ASSERT_EQ(samples.size(), 1);
  ASSERT_EQ(baseline.size(), 1);
  EXPECT_EQ(samples.front().measurements, baseline.front().measurements);
  EXPECT_EQ(samples.front().raw_logical, baseline.front().raw_logical);
  EXPECT_EQ(cudaq::get_platform().get_noise(), nullptr);
}

TEST(MemoryExperimentReplay, IndependentPatchesReceiveDistinctSeeds) {
  auto request = replay_request();
  request.experiment.logical_qubits = 2;
  const auto model = prepare_model(request);
  const auto samples = collect(request, model, 2, 23);
  ASSERT_EQ(samples.size(), 4);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    EXPECT_EQ(samples[index].patch, index / 2);
    EXPECT_EQ(samples[index].shot, index % 2);
    EXPECT_EQ(samples[index].seed, 23 + index);
  }
  request.experiment.logical_qubits = 1;
  const auto second_patch = collect(request, model, 2, 25);
  ASSERT_EQ(second_patch.size(), 2);
  EXPECT_EQ(samples[2].measurements, second_patch[0].measurements);
  EXPECT_EQ(samples[3].measurements, second_patch[1].measurements);
}

TEST(MemoryExperimentReplay, ZeroShotsDoesNotRequireSamplingOrConsumer) {
  EXPECT_NO_THROW(sample_raw_replay({}, {}, 0, 0, {}));
  EXPECT_NO_THROW(validate_raw_replay_options(0, 0, 0));
  EXPECT_THROW(validate_raw_replay_options(0, 1, 1), std::invalid_argument);
}

TEST(MemoryExperimentReplay,
     MissingConsumerAndSeedOverflowRejectBeforeSampling) {
  const auto request = replay_request();
  const auto consumer = [](const raw_replay_shot &) {
    FAIL() << "invalid replay must not produce a sample";
  };
  EXPECT_THROW(sample_raw_replay(request, {}, 1, 0, {}), std::invalid_argument);
  EXPECT_THROW(sample_raw_replay(request, {}, 1, 0, consumer),
               std::invalid_argument);
  EXPECT_THROW(sample_raw_replay(request, {}, 2,
                                 std::numeric_limits<std::uint64_t>::max(),
                                 consumer),
               std::overflow_error);
  auto many_patches = request;
  many_patches.experiment.logical_qubits = 2;
  EXPECT_THROW(sample_raw_replay(many_patches, {},
                                 std::numeric_limits<std::uint64_t>::max(), 1,
                                 consumer),
               std::overflow_error);
}

TEST(MemoryExperimentReplay, InvalidNoiseAndRecordCountsRejectBeforeSampling) {
  auto request = replay_request();
  const auto consumer = [](const raw_replay_shot &) {
    FAIL() << "invalid replay must not produce a sample";
  };
  for (const auto probability :
       {-0.01, 1.0, 1.01, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    request.cnot_error_rate = probability;
    EXPECT_THROW(sample_raw_replay(request, {}, 1, 1, consumer),
                 std::invalid_argument);
  }
  request.cnot_error_rate = 0.001;
  request.experiment.rounds = std::numeric_limits<std::uint64_t>::max() - 1;
  prepared_model model;
  model.regular_measurements = 8;
  model.final_measurements = 9;
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1, consumer),
               std::overflow_error);
}

TEST(MemoryExperimentReplay, InconsistentModelRejectedBeforeSampling) {
  const auto request = replay_request();
  const auto valid_model = prepare_model(request);
  const auto consumer = [](const raw_replay_shot &) {
    FAIL() << "invalid replay must not produce a sample";
  };
  auto model = valid_model;
  ++model.regular_measurements;
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1, consumer),
               std::invalid_argument);
  model = valid_model;
  ++model.final_measurements;
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1, consumer),
               std::invalid_argument);
  model = valid_model;
  ++model.physical_qubits;
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1, consumer),
               std::invalid_argument);
  model = valid_model;
  ++*model.chunks.num_rounds;
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1, consumer),
               std::invalid_argument);
  model = valid_model;
  model.final_m2o[0].push_back(model.final_measurements);
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1, consumer),
               std::invalid_argument);
}

TEST(MemoryExperimentReplay, ConsumerExceptionsAreNotSwallowed) {
  const auto request = replay_request();
  const auto model = prepare_model(request);
  EXPECT_THROW(sample_raw_replay(request, model, 1, 1,
                                 [](const raw_replay_shot &) {
                                   throw std::runtime_error("write failed");
                                 }),
               std::runtime_error);
  EXPECT_EQ(cudaq::get_platform().get_noise(), nullptr);
}

} // namespace
} // namespace cudaq::qec::memory_experiment
