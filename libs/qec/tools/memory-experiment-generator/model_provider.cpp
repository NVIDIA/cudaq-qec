/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "model_provider.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/memory_dem_chunks.h"
#include "cudaq/qec/noise_model.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::qec::memory_experiment {
namespace {

void validate_code_geometry(const experiment_spec &spec) {
  validate_experiment_spec(spec);
  const auto distance = spec.distance;
  const auto index_max = std::numeric_limits<std::uint32_t>::max();
  // The Surface Code implementation indexes its (distance + 1)^2 grid with
  // uint32-compatible coordinates. Validate before constructing that grid.
  if (distance >= index_max || distance + 1 > index_max / (distance + 1) ||
      distance > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error(
        "memory experiment: surface-code grid exceeds uint32 dimensions");
  const auto data = distance * distance;
  (void)memory_record_layout(spec.rounds, data - 1, data);
}

bool same_phase(const dem_chunk_spec &left, const dem_chunk_spec &right) {
  if (left.num_faults != right.num_faults || left.H_sparse != right.H_sparse ||
      left.seam_specs != right.seam_specs || left.O_sparse != right.O_sparse ||
      left.D_sparse != right.D_sparse ||
      left.num_measurements != right.num_measurements ||
      left.error_rates.size() != right.error_rates.size())
    return false;
  for (std::size_t i = 0; i < left.error_rates.size(); ++i)
    if (!memory_dem_probabilities_equivalent(left.error_rates[i],
                                             right.error_rates[i]))
      return false;
  return true;
}

bool same_phases(const dem_chunks_spec &left, const dem_chunks_spec &right) {
  if (left.seam != right.seam || left.connections != right.connections ||
      left.measurements_per_round != right.measurements_per_round ||
      left.phases_supply_D_sparse != right.phases_supply_D_sparse ||
      left.phases.size() != right.phases.size())
    return false;
  // num_rounds intentionally differs; every retained phase must agree.
  for (std::size_t i = 0; i < left.phases.size(); ++i)
    if (left.phases[i].id != right.phases[i].id ||
        !same_phase(left.phases[i].spec, right.phases[i].spec))
      return false;
  return true;
}

prepared_model characterize(const code &surface, operation prep,
                            std::uint64_t rounds,
                            const preparation_request &request,
                            cudaq::noise_model &noise) {
  const auto inputs = decoder_context_from_memory_circuit(
                          surface, prep, static_cast<std::size_t>(rounds),
                          noise, request.decompose_errors)
                          .full_component();
  prepared_model result;
  result.regular_measurements =
      surface.get_num_ancilla_x_qubits() + surface.get_num_ancilla_z_qubits();
  result.final_measurements = surface.get_num_data_qubits();
  result.physical_qubits =
      surface.get_num_ancilla_qubits() + result.final_measurements;
  result.characterization_rounds = rounds;
  result.chunks = memory_dem_chunks_from_decoder_inputs(
      inputs, rounds, result.regular_measurements, result.final_measurements);
  const auto final_begin = rounds * result.regular_measurements;
  result.final_m2o.reserve(inputs.m2o.rows.size());
  for (const auto &row : inputs.m2o.rows) {
    auto &relative = result.final_m2o.emplace_back();
    relative.reserve(row.size());
    for (const auto measurement : row) {
      if (measurement < final_begin ||
          measurement - final_begin >= result.final_measurements)
        throw std::invalid_argument(
            "memory experiment: raw logical map must read only the final "
            "data-qubit measurements");
      relative.push_back(measurement - final_begin);
    }
  }
  return result;
}

} // namespace

std::unique_ptr<code> create_experiment_code(const experiment_spec &spec) {
  validate_code_geometry(spec);
  return get_code(spec.code_family,
                  cudaqx::heterogeneous_map{
                      {"distance", static_cast<std::size_t>(spec.distance)}});
}

std::vector<std::vector<std::uint64_t>>
final_readout_support(const cudaqx::tensor<std::uint8_t> &observables) {
  if (observables.shape().size() != 2)
    throw std::invalid_argument(
        "memory experiment: logical observables must form a matrix");
  std::vector<std::vector<std::uint64_t>> result(observables.shape()[0]);
  for (std::size_t logical = 0; logical < result.size(); ++logical)
    for (std::size_t data = 0; data < observables.shape()[1]; ++data)
      if (observables.at({logical, data}))
        result[logical].push_back(data);
  return result;
}

void validate_preparation_request(const preparation_request &request) {
  validate_code_geometry(request.experiment);
  if (!std::isfinite(request.cnot_error_rate) ||
      request.cnot_error_rate < 0.0 || request.cnot_error_rate >= 1.0)
    throw std::invalid_argument(
        "memory experiment: CNOT error rate must be finite and in [0,1); "
        "the circuit-analysis backend does not support probability 1");
  if (request.cnot_error_rate == 0.0)
    throw std::invalid_argument(
        "memory experiment: noiseless preparation requires public framework "
        "support for zero-fault DEM phases");
}

void validate_prepared_model(const preparation_request &request,
                             const prepared_model &model) {
  validate_preparation_request(request);
  if (!model.preparation)
    throw std::invalid_argument(
        "memory experiment: prepared model has no preparation identity");
  const auto &prepared = *model.preparation;
  const auto &expected = prepared.experiment;
  const auto &actual = request.experiment;
  if (expected.code_family != actual.code_family ||
      expected.distance != actual.distance ||
      expected.rounds != actual.rounds || expected.basis != actual.basis ||
      prepared.cnot_error_rate != request.cnot_error_rate ||
      prepared.decompose_errors != request.decompose_errors)
    throw std::invalid_argument(
        "memory experiment: request disagrees with prepared model identity");
  const auto surface = create_experiment_code(actual);
  const auto data = surface->get_num_data_qubits();
  const auto regular =
      surface->get_num_ancilla_x_qubits() + surface->get_num_ancilla_z_qubits();
  const auto observables = actual.basis == memory_basis::x
                               ? surface->get_observables_x()
                               : surface->get_observables_z();
  const auto readout_support = final_readout_support(observables);
  if (readout_support.size() != 1)
    throw std::invalid_argument(
        "memory experiment: one logical observable per independent patch "
        "is currently supported");
  if (model.regular_measurements != regular ||
      model.final_measurements != data ||
      model.physical_qubits != surface->get_num_ancilla_qubits() + data ||
      model.final_m2o.size() != readout_support.size() ||
      model.chunks.num_rounds != actual.rounds + 1)
    throw std::invalid_argument("memory experiment: prepared model geometry or "
                                "horizon is inconsistent");
  validate_characterization_provenance(model.chunks, regular, data,
                                       model.characterization_rounds,
                                       model.validation_rounds);
}

void validate_characterization_provenance(const dem_chunks_spec &chunks,
                                          std::uint64_t regular_measurements,
                                          std::uint64_t final_measurements,
                                          std::uint64_t characterization_rounds,
                                          std::uint64_t validation_rounds) {
  // minimum_rounds includes one bulk visit and the terminal readout. A
  // characterization with that many extraction cycles adds a second bulk
  // visit without tying a schema version to the producer's chosen depths.
  const auto minimum_extraction_rounds = chunks.minimum_rounds();
  if (characterization_rounds < minimum_extraction_rounds ||
      validation_rounds < minimum_extraction_rounds ||
      characterization_rounds == validation_rounds)
    throw std::invalid_argument(
        "memory experiment: characterization depths must be distinct and "
        "cover prefix, repeated bulk and suffix");
  for (const auto rounds : {characterization_rounds, validation_rounds})
    (void)memory_record_layout(rounds, regular_measurements,
                               final_measurements);
}

prepared_model prepare_model(const preparation_request &request) {
  validate_preparation_request(request);
  auto surface = create_experiment_code(request.experiment);
  // Reject record-count overflow before the more expensive characterization.
  const memory_record_layout layout(request.experiment.rounds,
                                    surface->get_num_ancilla_x_qubits() +
                                        surface->get_num_ancilla_z_qubits(),
                                    surface->get_num_data_qubits());
  const auto prep = request.experiment.basis == memory_basis::z
                        ? operation::prep0
                        : operation::prepp;
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x",
                              two_qubit_depolarization(request.cnot_error_rate),
                              /*num_controls=*/1);
  auto representative = characterize(
      *surface, prep, representative_extraction_rounds, request, noise);
  const auto verification = characterize(
      *surface, prep, validation_extraction_rounds, request, noise);
  const auto logical_observables = request.experiment.basis == memory_basis::x
                                       ? surface->get_observables_x()
                                       : surface->get_observables_z();
  if (representative.regular_measurements !=
          verification.regular_measurements ||
      representative.final_measurements != verification.final_measurements ||
      representative.physical_qubits != verification.physical_qubits ||
      representative.final_m2o != verification.final_m2o ||
      representative.final_m2o != final_readout_support(logical_observables) ||
      !same_phases(representative.chunks, verification.chunks))
    throw std::invalid_argument(
        "memory experiment: bounded characterization did not establish "
        "matching prefix, repeating bulk, suffix and raw-logical mappings");

  // Neither retargeting nor validation materializes D or a delivery schedule.
  representative.chunks.num_rounds = layout.input_rounds();
  if (layout.input_rounds() < representative.chunks.minimum_rounds())
    throw std::invalid_argument(
        "memory experiment: requested duration is shorter than the "
        "characterized prefix/bulk/suffix chain (minimum " +
        std::to_string(representative.chunks.minimum_rounds() - 1) +
        " extraction cycles); no duration was rounded");
  representative.chunks.validate();
  // repeating_rounds validates the chain using its shortest expansion, so
  // this topology check is bounded even when the requested duration is huge.
  (void)representative.chunks.repeating_rounds();
  representative.preparation = request;
  representative.validation_rounds = verification.characterization_rounds;
  return representative;
}

} // namespace cudaq::qec::memory_experiment
