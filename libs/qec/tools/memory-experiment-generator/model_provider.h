/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "experiment_spec.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/extended_dem.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cudaq::qec::memory_experiment {

struct preparation_request {
  experiment_spec experiment;
  double cnot_error_rate = 0.001;
  bool decompose_errors = true;
};

struct prepared_model {
  /// Preparation identity, excluding the number of replicas at reuse time.
  /// An unbound model cannot be published or sampled as prepared evidence.
  std::optional<preparation_request> preparation;
  dem_chunks_spec chunks;
  /// Raw-logical support relative to the beginning of the final readout.
  /// This is m2o, not the fault-observable O stored in chunks.
  std::vector<std::vector<std::uint64_t>> final_m2o;
  std::uint64_t regular_measurements = 0;
  std::uint64_t final_measurements = 0;
  /// Data and syndrome qubits in one independent patch, not all replicas.
  std::uint64_t physical_qubits = 0;
};

/// Validate supported inputs and checked record arithmetic without constructing
/// a code, characterizing a circuit, or allocating duration-dependent storage.
void validate_preparation_request(const preparation_request &request);

/// Reject accidental reuse of a model prepared for different code geometry,
/// basis, duration, noise or normalization. Replica counts may differ.
void validate_prepared_model(const preparation_request &request,
                             const prepared_model &model);

/// Check the public terminal-phase readout against independently validated
/// raw-logical support. Terminal metadata is required. Does not expand the
/// horizon or assume a particular endpoint phase name.
void validate_final_readout(
    const dem_chunks_spec &chunks,
    const std::vector<std::vector<std::uint64_t>> &readout);

/// Construct the selected code after checking its supported geometry and raw
/// record arithmetic. This does not characterize a circuit or expand its DEM.
std::unique_ptr<code> create_experiment_code(const experiment_spec &spec);

/// Canonical final-readout support from the code's basis-specific observable
/// matrix. Shared by preparation, sampling and model validation; these are
/// measurement indices, not fault-observable columns.
std::vector<std::vector<std::uint64_t>>
final_readout_support(const cudaqx::tensor<std::uint8_t> &observables);

/// Prepare one patch through the public circuit-characterization API, validate
/// repeating phases at two fixed depths, and retarget the compact spec to the
/// requested duration without expanding its rounds. Independent patches reuse
/// this model. The active CUDA-Q backend must support circuit DEM analysis.
///
/// The initial model uses two-qubit depolarizing noise on every CNOT. Zero
/// noise is rejected until the public framework supports zero-fault phases.
/// Durations shorter than the retained prefix/bulk/suffix chain are rejected,
/// not rounded or replaced by a different decoding algorithm.
prepared_model prepare_model(const preparation_request &request);

} // namespace cudaq::qec::memory_experiment
