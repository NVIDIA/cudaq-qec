/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "device/memory_circuit.h"
#include "cudaq/platform.h"
#include <cstddef>
#include <string>
#include <vector>

namespace cudaq::qec::testing {

// Test-only direct sampling of the existing compiled kernel. Callers provide
// valid circuit inputs and manage the process-wide seed; this is not an SDK
// validation or sampling interface.
inline std::vector<std::string>
sample_raw_memory_records(const code &qec_code, operation preparation,
                          std::size_t shots, std::size_t rounds,
                          cudaq::noise_model &noise) {
  const bool is_z_preparation =
      preparation == operation::prep0 || preparation == operation::prep1;
  const auto schedule_x = qec_code.get_stabilizer_schedule_x();
  const auto schedule_z = qec_code.get_stabilizer_schedule_z();
  const std::vector<std::size_t> x_schedule(
      schedule_x.data(), schedule_x.data() + schedule_x.size());
  const std::vector<std::size_t> z_schedule(
      schedule_z.data(), schedule_z.data() + schedule_z.size());
  const auto observables = is_z_preparation ? qec_code.get_observables_z()
                                            : qec_code.get_observables_x();
  const std::vector<std::size_t> observable_map(
      observables.data(), observables.data() + observables.size());
  const cudaq::sample_options options{
      .shots = shots, .noise = noise, .explicit_measurements = true};
  try {
    auto result = cudaq::sample(
        options, memory_circuit,
        qec_code.get_operation<code::stabilizer_round>(
            operation::stabilizer_round),
        qec_code.get_operation<code::one_qubit_encoding>(preparation),
        qec_code.get_num_data_qubits(), qec_code.get_num_ancilla_x_qubits(),
        qec_code.get_num_ancilla_z_qubits(), rounds, x_schedule, z_schedule,
        observable_map, observables.shape()[0], !is_z_preparation);
    return result.sequential_data();
  } catch (...) {
    // CUDA-Q clears noise on success only; do not leave a borrowed pointer to
    // options.noise on the platform when unwinding destroys the options.
    cudaq::get_platform().reset_noise();
    throw;
  }
}

} // namespace cudaq::qec::testing
