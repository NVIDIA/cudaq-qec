/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "memory_circuit.h"
#include <numeric>

namespace cudaq::qec {

__qpu__ void
memory_circuit(const code::stabilizer_round &stabilizer_round,
               const code::one_qubit_encoding &statePrep, std::size_t num_data,
               std::size_t numAncx, std::size_t numAncz, std::size_t num_rounds,
               const std::vector<std::size_t> &x_stabilizer_schedule,
               const std::vector<std::size_t> &z_stabilizer_schedule,
               const std::vector<std::size_t> &obs_matrix_flat,
               std::size_t num_observables, bool measure_in_x_basis) {
  // Allocate the data and ancilla qubits
  cudaq::qvector data(num_data), xstab_anc(numAncx), zstab_anc(numAncz);

  // Create the logical patch
  patch logical(data, xstab_anc, zstab_anc);

  // Prepare the initial state
  statePrep({data, xstab_anc, zstab_anc});

  // The "off-basis" detectors will be non-deterministic after the first
  // stabilizer round.
  auto final_syndrome =
      stabilizer_round(logical, x_stabilizer_schedule, z_stabilizer_schedule);
  std::size_t num_fixed_measurements =
      measure_in_x_basis ? xstab_anc.size() : zstab_anc.size();
  std::size_t fixed_offset =
      measure_in_x_basis ? final_syndrome.size() - num_fixed_measurements : 0;
  for (std::size_t i = 0; i < num_fixed_measurements; ++i) {
    cudaq::detector(final_syndrome[fixed_offset + i]);
  }

  // Generate syndrome data
  for (std::size_t round = 1; round < num_rounds; ++round) {
    auto syndrome =
        stabilizer_round(logical, x_stabilizer_schedule, z_stabilizer_schedule);
    cudaq::detectors(final_syndrome, syndrome);
    final_syndrome = syndrome;
  }

  if (measure_in_x_basis) {
    h(data);
  }
  auto data_results = mz(data);

  // Emit one logical_observable per row of the observable matrix.
  for (std::size_t obs = 0; obs < num_observables; ++obs) {
    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (obs_matrix_flat[obs * num_data + q] != 0)
        support_weight++;
    }
    std::vector<cudaq::measure_result> obs_support(support_weight);
    std::size_t idx = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (obs_matrix_flat[obs * num_data + q] != 0)
        obs_support[idx++] = data_results[q];
    }
    cudaq::logical_observable(obs_support);
  }

  // For each stabilizer, form detectors from data qubit readout connected with
  // final stabilizer round. Any nonzero schedule entry marks support.
  const std::vector<size_t> &stabilizer_schedule =
      measure_in_x_basis ? x_stabilizer_schedule : z_stabilizer_schedule;

  for (std::size_t x = 0; x < num_fixed_measurements; ++x) {
    std::size_t row_base = x * num_data;

    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (stabilizer_schedule[row_base + q] != 0) {
        support_weight++;
      }
    }

    std::vector<cudaq::measure_result> support(support_weight + 1);
    support[0] = final_syndrome[fixed_offset + x];
    std::size_t support_idx = 1;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (stabilizer_schedule[row_base + q] != 0) {
        support[support_idx++] = data_results[q];
      }
    }

    cudaq::detector(support);
  }
}

} // namespace cudaq::qec
