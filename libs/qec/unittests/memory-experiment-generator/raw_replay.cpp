/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "raw_replay.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/noise_model.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudaq::qec::memory_experiment {
namespace {

// Qualification sampling shares the CUDA-Q process-wide random state.
std::mutex sampling_mutex;

} // namespace

void validate_raw_replay_options(std::uint64_t patches, std::uint64_t shots,
                                 std::uint64_t base_seed) {
  if (shots == 0)
    return;
  if (patches == 0)
    throw std::invalid_argument(
        "memory experiment: replay patch count must be positive");
  if (base_seed == 0)
    throw std::invalid_argument(
        "memory experiment: replay seed must be positive; CUDA-Q seed 0 "
        "requests unseeded sampling");
  const auto maximum =
      std::min<std::uint64_t>(std::numeric_limits<std::uint64_t>::max(),
                              std::numeric_limits<std::size_t>::max());
  if (patches > maximum / shots || base_seed > maximum ||
      patches * shots - 1 > maximum - base_seed)
    throw std::overflow_error(
        "memory experiment: replay seed range exceeds the CUDA-Q seed type");
}

void sample_raw_replay(
    const preparation_request &request, const prepared_model &model,
    std::uint64_t shots_per_patch, std::uint64_t base_seed,
    const std::function<void(const raw_replay_shot &)> &consume) {
  if (shots_per_patch == 0)
    return;
  if (!consume)
    throw std::invalid_argument(
        "memory experiment: replay consumer must be supplied");
  validate_preparation_request(request);
  validate_raw_replay_options(request.experiment.logical_qubits,
                              shots_per_patch, base_seed);
  validate_prepared_model(request, model);
  const memory_record_layout layout(request.experiment.rounds,
                                    model.regular_measurements,
                                    model.final_measurements);
  if (layout.raw_measurements() > std::string{}.max_size() ||
      request.experiment.rounds > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error(
        "memory experiment: one replay shot exceeds addressable storage");

  auto surface = create_experiment_code(request.experiment);
  const bool measure_x = request.experiment.basis == memory_basis::x;
  const auto prep_kind = measure_x ? operation::prepp : operation::prep0;
  const auto observables =
      measure_x ? surface->get_observables_x() : surface->get_observables_z();
  if (model.final_m2o != final_readout_support(observables))
    throw std::invalid_argument(
        "memory experiment: replay raw-logical map disagrees with the code");
  cudaq::noise_model noise;
  if (request.cnot_error_rate != 0.0)
    noise.add_all_qubit_channel(
        "x", two_qubit_depolarization(request.cnot_error_rate),
        /*num_controls=*/1);
  raw_replay_shot output{0, 0, base_seed, {}, {}};
  const auto num_measurements =
      static_cast<std::size_t>(layout.raw_measurements());
  output.raw_logical.resize(model.final_m2o.size());
  const auto final_begin = num_measurements - model.final_measurements;
  for (std::uint64_t patch = 0; patch < request.experiment.logical_qubits;
       ++patch)
    for (std::uint64_t shot = 0; shot < shots_per_patch; ++shot) {
      output.patch = patch;
      output.shot = shot;
      output.seed = base_seed + patch * shots_per_patch + shot;
      // CUDA-Q sampling and seed mutation share global state. Do not run
      // different seeded shots concurrently or hold this lock in the consumer.
      auto samples = [&] {
        const std::lock_guard lock(sampling_mutex);
        cudaq::set_random_seed(static_cast<std::size_t>(output.seed));
        return sample_memory_circuit_raw(
            *surface, prep_kind, 1,
            static_cast<std::size_t>(request.experiment.rounds), noise);
      }();
      if (samples.size() != 1 || samples.front().size() != num_measurements ||
          samples.front().find_first_not_of("01") != std::string::npos)
        throw std::runtime_error(
            "memory experiment: sampled raw records violate the model layout");
      output.measurements = std::move(samples.front());
      for (std::size_t logical = 0; logical < model.final_m2o.size();
           ++logical) {
        bool parity = false;
        for (const auto data : model.final_m2o[logical])
          parity ^= output.measurements[final_begin + data] == '1';
        output.raw_logical[logical] = parity ? '1' : '0';
      }
      consume(output);
    }
}

} // namespace cudaq::qec::memory_experiment
