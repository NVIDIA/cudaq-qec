/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "model_provider.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/dem_chunks_memory.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/noise_model.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if __has_include("../../lib/decoding_task/decoding_task_commitment.h") &&     \
                  __has_include("../task_graph_test_fixtures.h")
#include "../../lib/decoding_task/decoding_task_commitment.h"
#include "../task_graph_test_fixtures.h"
#endif

namespace cudaq::qec::memory_experiment {
namespace {

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

void compare_with_full_circuit(std::uint64_t distance, memory_basis basis,
                               std::uint64_t extraction_rounds = 26) {
  const preparation_request request{
      {"surface_code", distance, extraction_rounds, 1, basis}};
  const auto prepared = prepare_model(request);
  ASSERT_EQ(prepared.chunks.num_rounds, extraction_rounds + 1);
  ASSERT_EQ(prepared.characterization_rounds, representative_extraction_rounds);
  ASSERT_EQ(prepared.validation_rounds, validation_extraction_rounds);

  auto surface = get_code(
      "surface_code", cudaqx::heterogeneous_map{
                          {"distance", static_cast<std::size_t>(distance)}});
  const auto prep =
      basis == memory_basis::z ? operation::prep0 : operation::prepp;
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x",
                              two_qubit_depolarization(request.cnot_error_rate),
                              /*num_controls=*/1);
  // This independently characterizes the complete requested experiment. It
  // never calls the splitter, and must agree with the bounded representative
  // model expanded to the same horizon, including both endpoint mappings.
  const auto full =
      decoder_context_from_memory_circuit(*surface, prep, extraction_rounds,
                                          noise, request.decompose_errors)
          .full_component();
  const auto closed =
      dem_close(dem_stitch_all(dem_chunks_from_spec(prepared.chunks)));
  ASSERT_EQ(closed.num_detectors(), full.dem.num_detectors());
  ASSERT_EQ(closed.num_observables(), full.dem.num_observables());
  const auto expected = canonical_faults(full.dem);
  const auto actual = canonical_faults(closed);
  ASSERT_EQ(actual.size(), expected.size());
  for (const auto &[signature, probability] : expected) {
    const auto found = actual.find(signature);
    ASSERT_NE(found, actual.end());
    EXPECT_NEAR(found->second, probability, 1e-12);
  }
  std::vector<std::vector<std::uint32_t>> expected_d;
  for (const auto &row : full.m2d.rows)
    expected_d.emplace_back(row.begin(), row.end());
  EXPECT_EQ(dem_chunks_to_d_sparse(prepared.chunks), expected_d);
  EXPECT_EQ(dem_chunks_measurement_count(prepared.chunks),
            full.m2d.num_measurements);
  EXPECT_EQ(prepared.regular_measurements,
            surface->get_num_ancilla_x_qubits() +
                surface->get_num_ancilla_z_qubits());
  EXPECT_EQ(prepared.final_measurements, surface->get_num_data_qubits());
  EXPECT_EQ(prepared.physical_qubits,
            prepared.regular_measurements + prepared.final_measurements);

  const auto logical = basis == memory_basis::z ? surface->get_observables_z()
                                                : surface->get_observables_x();
  std::vector<std::vector<std::uint64_t>> expected_m2o(logical.shape()[0]);
  for (std::size_t observable = 0; observable < logical.shape()[0];
       ++observable)
    for (std::size_t data = 0; data < logical.shape()[1]; ++data)
      if (logical.at({observable, data}))
        expected_m2o[observable].push_back(data);
  EXPECT_EQ(prepared.final_m2o, expected_m2o);
  ASSERT_EQ(full.m2o.rows.size(), expected_m2o.size());
  const auto final_begin = extraction_rounds * prepared.regular_measurements;
  for (std::size_t observable = 0; observable < expected_m2o.size();
       ++observable) {
    auto expected_absolute = expected_m2o[observable];
    for (auto &measurement : expected_absolute)
      measurement += final_begin;
    EXPECT_EQ(full.m2o.rows[observable], expected_absolute);
  }
}

TEST(MemoryExperimentProvider, BoundedModelMatchesCompleteZMemoryCircuit) {
  compare_with_full_circuit(3, memory_basis::z);
}

TEST(MemoryExperimentProvider, BoundedModelMatchesCompleteXMemoryCircuit) {
  compare_with_full_circuit(3, memory_basis::x);
}

TEST(MemoryExperimentProvider, PyMatchingFaultFramesCommitReferenceWindows) {
#if __has_include("../../lib/decoding_task/decoding_task_commitment.h") &&     \
                  __has_include("../task_graph_test_fixtures.h")
  if (!decoder::is_registered("pymatching"))
    GTEST_SKIP() << "Requires the installed PyMatching plugin";
  // Reuse the framework's windows, not a second scheduler. This checks output
  // and commit contracts, not logical accuracy or parallel runtime execution.
  constexpr std::uint64_t extraction_rounds = 26;
  constexpr std::size_t detector_rounds = extraction_rounds + 1;
  const auto parity = [](const auto &rows,
                         const std::vector<std::uint8_t> &input) {
    std::vector<std::uint8_t> result(rows.size(), 0);
    for (std::size_t row = 0; row < result.size(); ++row)
      for (const auto column : rows[row])
        result[row] ^= input[column];
    return result;
  };
  const auto requests = layered(3, 3, 9);
  const std::vector<window_layer> layers{{"A", "a", requests.a},
                                         {"B", "b", requests.b}};
  const std::vector<task_graph_edge> edges{
      {"A", "forward", "B", "receive_0", -1},
      {"A", "backward", "B", "receive_1", 0}};
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    SCOPED_TRACE(basis == memory_basis::x ? "X" : "Z");
    const auto prepared =
        prepare_model({{"surface_code", 3, extraction_rounds, 1, basis}});
    const auto context =
        std::make_shared<const experiment_template>(prepared.chunks);
    const auto schedule =
        schedule_windows(context, layers, edges, detector_rounds);
    ASSERT_EQ(schedule.windows.size(), 5);
    for (const auto &entry : schedule.windows) {
      SCOPED_TRACE(entry.node);
      SCOPED_TRACE(entry.k);
      const auto &request = entry.node == "A" ? requests.a : requests.b;
      const auto window =
          window_from_dem_chunks(context, request, entry.k, detector_rounds);
      const auto forms = kernel_forms(window);
      const auto h = forms.h.to_nested_csr();
      const auto full_h =
          sparse_binary_matrix(window.model.detector_error_matrix)
              .to_nested_csr();
      const auto o = sparse_binary_matrix(window.model.observables_flips_matrix)
                         .to_nested_csr();
      const auto columns = forms.h.to_nested_csc();
      ASSERT_GT(columns.size(), 1);
      ASSERT_EQ(columns.size(), window.num_decoder_columns());
      ASSERT_GE(window.model.num_error_mechanisms(), columns.size());
      ASSERT_GE(forms.priors.size(), columns.size());
      const std::vector<double> priors(forms.priors.begin(),
                                       forms.priors.begin() + columns.size());
      for (const std::string policy : {"smallest_weight", "independent"}) {
        SCOPED_TRACE(policy);
        auto matching = decoder::get(
            "pymatching", decoder_init(forms.h, std::nullopt, priors),
            decoder_output_request{decode_result_type::errors},
            cudaqx::heterogeneous_map{{"merge_strategy", policy}});
        ASSERT_EQ(matching->get_result_type(), decode_result_type::errors);
        const auto check = [&](const std::vector<std::uint8_t> &detectors) {
          const std::vector<float_t> syndrome(detectors.begin(),
                                              detectors.end());
          const auto decoded = matching->decode(syndrome);
          ASSERT_TRUE(decoded.converged);
          ASSERT_EQ(decoded.result.size(), columns.size());
          std::vector<std::uint8_t> estimate(
              window.model.num_error_mechanisms(), 0);
          for (std::size_t column = 0; column < decoded.result.size();
               ++column) {
            ASSERT_TRUE(std::isfinite(decoded.result[column]));
            ASSERT_TRUE(decoded.result[column] == 0 ||
                        decoded.result[column] == 1);
            estimate[column] =
                static_cast<std::uint8_t>(decoded.result[column]);
          }
          const std::vector<std::uint8_t> error_frame(
              estimate.begin(), estimate.begin() + columns.size());
          EXPECT_EQ(parity(h, error_frame), detectors);
          const auto committed = commit_window(window, detectors, estimate);
          auto expected = estimate;
          for (std::size_t column = 0; column < expected.size(); ++column)
            if (column < window.commit_columns.begin ||
                column >= window.commit_columns.end)
              expected[column] = 0;
          EXPECT_EQ(committed.committed, expected);
          EXPECT_EQ(committed.frame, parity(o, expected));
          const auto delta = parity(full_h, expected);
          ASSERT_EQ(committed.sends.size(), window.sends.size());
          for (std::size_t send = 0; send < window.sends.size(); ++send) {
            std::vector<std::uint8_t> values;
            for (const auto row : window.sends[send].rows)
              values.push_back(window.sends[send].mode == send_mode::residual
                                   ? detectors[row] ^ delta[row]
                                   : delta[row]);
            EXPECT_EQ(committed.sends[send], values);
          }
        };
        std::vector<std::uint8_t> detectors(forms.h.num_rows(), 0);
        check(detectors);
        for (const auto &column : columns) {
          std::fill(detectors.begin(), detectors.end(), 0);
          for (const auto row : column)
            detectors[row] ^= 1;
          check(detectors);
        }
        std::fill(detectors.begin(), detectors.end(), 0);
        for (const auto column : {std::size_t{0}, columns.size() - 1})
          for (const auto row : columns[column])
            detectors[row] ^= 1;
        check(detectors);
      }
    }
  }
#else
  GTEST_SKIP() << "Requires the public parallel-window model implementation";
#endif
}

TEST(MemoryExperimentProvider, DistanceFiveMatchesCompleteMemoryCircuit) {
  compare_with_full_circuit(5, memory_basis::z);
  compare_with_full_circuit(5, memory_basis::x);
}

TEST(MemoryExperimentProvider,
     MinimumAndAwkwardDurationsMatchCompleteCircuits) {
  for (const auto basis : {memory_basis::x, memory_basis::z})
    for (const auto rounds : {4, 5, 25, 27}) {
      SCOPED_TRACE(rounds);
      SCOPED_TRACE(basis == memory_basis::x ? "X" : "Z");
      compare_with_full_circuit(3, basis, rounds);
    }
}

TEST(MemoryExperimentProvider, RequestedDurationAndPatchCountDoNotGrowModel) {
  preparation_request request{{"surface_code", 3, 26, 1, memory_basis::z}};
  const auto small = prepare_model(request);
  for (const auto rounds : {std::uint64_t{10000}, std::uint64_t{1000000000}}) {
    request.experiment.rounds = rounds;
    request.experiment.logical_qubits = 4;
    const auto large = prepare_model(request);
    EXPECT_EQ(large.chunks.num_rounds, rounds + 1);
    EXPECT_EQ(large.chunks.phases, small.chunks.phases);
    EXPECT_EQ(large.chunks.connections, small.chunks.connections);
    EXPECT_EQ(large.chunks.measurements_per_round,
              small.chunks.measurements_per_round);
    EXPECT_EQ(large.final_m2o, small.final_m2o);
    EXPECT_EQ(large.characterization_rounds, small.characterization_rounds);
    EXPECT_EQ(large.validation_rounds, small.validation_rounds);
    EXPECT_EQ(large.physical_qubits, small.physical_qubits);
  }
}

TEST(MemoryExperimentProvider, InvalidNoiseAndGeometryFailBeforeCircuitWork) {
  preparation_request request{{"surface_code", 3, 26, 1, memory_basis::z}};
  for (const auto probability :
       {0.0, -0.01, 1.0, 1.01, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    request.cnot_error_rate = probability;
    EXPECT_THROW(prepare_model(request), std::invalid_argument);
  }
  request.cnot_error_rate = 0.001;
  request.experiment.distance = std::numeric_limits<std::uint32_t>::max();
  EXPECT_THROW(prepare_model(request), std::overflow_error);
  request.experiment.distance = 3;
  request.experiment.rounds = std::numeric_limits<std::uint64_t>::max() - 1;
  EXPECT_THROW(prepare_model(request), std::overflow_error);
}

TEST(MemoryExperimentProvider, TooShortDurationIsRejectedWithoutRounding) {
  const preparation_request request{{"surface_code", 3, 1, 1, memory_basis::z}};
  EXPECT_THROW(prepare_model(request), std::invalid_argument);
}

TEST(MemoryExperimentProvider, PreparedGeometryIsCheckedAgainstCode) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const preparation_request request{{"surface_code", 3, 26, 1, basis}};
    const auto model = prepare_model(request);
    for (const auto field : {0, 1, 2, 3}) {
      auto invalid = model;
      switch (field) {
      case 0:
        ++invalid.regular_measurements;
        break;
      case 1:
        ++invalid.final_measurements;
        break;
      case 2:
        ++invalid.physical_qubits;
        break;
      case 3:
        invalid.final_m2o.emplace_back();
        break;
      }
      EXPECT_THROW(validate_prepared_model(request, invalid),
                   std::invalid_argument);
    }
  }
}

TEST(MemoryExperimentProvider, PublicReadoutAgreesWithTerminalMeasurements) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const preparation_request request{{"surface_code", 3, 26, 1, basis}};
    const auto model = prepare_model(request);
    EXPECT_NO_THROW(validate_final_readout(model.chunks, model.final_m2o));
    // Member-dependent checks keep internal model tests useful with public
    // builds predating terminal readout metadata, without inventing a schema.
    [&]<typename Spec>(Spec chunks) {
      if constexpr (requires { chunks.phases.front().spec.m2o_sparse; }) {
        ASSERT_NO_THROW(validate_final_readout(chunks, model.final_m2o, true));
        const auto found = std::find_if(
            chunks.phases.begin(), chunks.phases.end(),
            [](const auto &phase) { return !phase.spec.m2o_sparse.empty(); });
        ASSERT_NE(found, chunks.phases.end());
        const auto old_id = found->id;
        const auto renamed = phase_id::from_name("readout_endpoint");
        found->id = renamed;
        for (auto &connection : chunks.connections) {
          if (connection.from_phase == old_id)
            connection.from_phase = renamed;
          if (connection.to_phase == old_id)
            connection.to_phase = renamed;
        }
        EXPECT_NO_THROW(validate_final_readout(chunks, model.final_m2o, true));
        const auto expected = found->spec.m2o_sparse;
        auto with_unused_phase = chunks;
        auto unused = with_unused_phase.phases.front();
        unused.id = phase_id::from_name("unused_phase");
        unused.spec.m2o_sparse.clear();
        with_unused_phase.phases.insert(with_unused_phase.phases.begin(),
                                        std::move(unused));
        EXPECT_NO_THROW(
            validate_final_readout(with_unused_phase, model.final_m2o, true));
        found->spec.m2o_sparse.clear();
        // Optional readout metadata is insufficient for server configuration.
        EXPECT_NO_THROW(validate_final_readout(chunks, model.final_m2o));
        EXPECT_THROW(validate_final_readout(chunks, model.final_m2o, true),
                     std::invalid_argument);
        found->spec.m2o_sparse = expected;
        found->spec.m2o_sparse.front() = model.final_measurements;
        EXPECT_THROW(validate_final_readout(chunks, model.final_m2o),
                     std::invalid_argument);
        found->spec.m2o_sparse = expected;
        auto &other = chunks.phases.front();
        ASSERT_NE(other.id, renamed);
        other.spec.m2o_sparse = expected;
        EXPECT_THROW(validate_final_readout(chunks, model.final_m2o),
                     std::invalid_argument);
      } else {
        EXPECT_THROW(validate_final_readout(chunks, model.final_m2o, true),
                     std::invalid_argument);
      }
    }(model.chunks);
  }
}

TEST(MemoryExperimentProvider, AcceptsSaneBoundedCharacterizationProvenance) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto original = prepare_model(request);
  for (const auto &[characterization, validation] :
       std::vector<std::pair<std::uint64_t, std::uint64_t>>{{16, 17},
                                                            {14, 19}}) {
    auto model = original;
    model.characterization_rounds = characterization;
    model.validation_rounds = validation;
    EXPECT_NO_THROW(validate_prepared_model(request, model));
  }
}

TEST(MemoryExperimentProvider, RejectsDegenerateCharacterizationProvenance) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto original = prepare_model(request);
  const auto too_short = original.chunks.minimum_rounds() - 1;
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  for (const auto &[characterization, validation] :
       std::vector<std::pair<std::uint64_t, std::uint64_t>>{
           {0, 13},
           {12, 0},
           {12, 12},
           {too_short, 13},
           {12, too_short},
           {maximum, 13},
           {12, maximum},
           {maximum - 1, 13}}) {
    auto model = original;
    model.characterization_rounds = characterization;
    model.validation_rounds = validation;
    EXPECT_THROW(validate_prepared_model(request, model), std::exception);
  }
}

TEST(MemoryExperimentProvider, RejectsWrongReadoutAndObservableCount) {
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const preparation_request request{{"surface_code", 3, 26, 1, basis}};
    const auto model = prepare_model(request);
    auto wrong = model;
    wrong.final_m2o = {{0}};
    ASSERT_NE(wrong.final_m2o, model.final_m2o);
    EXPECT_THROW(validate_prepared_model(request, wrong),
                 std::invalid_argument);
    wrong = model;
    for (auto &phase : wrong.chunks.phases)
      phase.spec.O_sparse.push_back(-1);
    EXPECT_THROW(validate_prepared_model(request, wrong),
                 std::invalid_argument);
  }
}

TEST(MemoryExperimentProvider, PreparationIdentityRejectsAccidentalReuse) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto model = prepare_model(request);
  std::vector<preparation_request> mismatched(5, request);
  mismatched[0].experiment.distance = 5;
  mismatched[1].experiment.rounds = 27;
  mismatched[2].experiment.basis = memory_basis::x;
  mismatched[3].cnot_error_rate = 0.002;
  mismatched[4].decompose_errors = false;
  for (const auto &other : mismatched)
    EXPECT_THROW(validate_prepared_model(other, model), std::invalid_argument);
  auto unbound = model;
  unbound.preparation.reset();
  EXPECT_THROW(validate_prepared_model(request, unbound),
               std::invalid_argument);
  auto replicated = request;
  replicated.experiment.logical_qubits = 4;
  EXPECT_NO_THROW(validate_prepared_model(replicated, model));
}

TEST(MemoryExperimentProvider, PublicModelYamlRoundtripsBothBases) {
  namespace config = cudaq::qec::decoding::config;
  for (const auto basis : {memory_basis::x, memory_basis::z}) {
    const preparation_request request{{"surface_code", 3, 26, 1, basis}};
    const auto model = prepare_model(request);
    const auto yaml = config::dem_chunks_to_yaml_str(model.chunks);
    EXPECT_EQ(config::dem_chunks_from_yaml_str(yaml), model.chunks);
  }
}

TEST(MemoryExperimentProvider, RejectsMalformedDetectorMapsBeforePublication) {
  const preparation_request request{
      {"surface_code", 3, 26, 1, memory_basis::z}};
  const auto original = prepare_model(request);
  constexpr std::int64_t invalid_measurement_index = 1000000;
  for (const auto remove_row : {true, false}) {
    auto invalid = original;
    const auto connection = std::find_if(
        invalid.chunks.connections.begin(), invalid.chunks.connections.end(),
        [](const auto &entry) { return entry.from_phase == entry.to_phase; });
    ASSERT_NE(connection, invalid.chunks.connections.end());
    const auto phase = std::find_if(
        invalid.chunks.phases.begin(), invalid.chunks.phases.end(),
        [&](const auto &entry) { return entry.id == connection->from_phase; });
    ASSERT_NE(phase, invalid.chunks.phases.end());
    auto &rows = phase->spec.D_sparse;
    if (remove_row) {
      const auto terminator = std::find(rows.begin(), rows.end(), -1);
      ASSERT_NE(terminator, rows.end());
      rows.erase(rows.begin(), terminator + 1);
    } else {
      const auto index = std::find_if(rows.begin(), rows.end(),
                                      [](auto value) { return value >= 0; });
      ASSERT_NE(index, rows.end());
      *index = invalid_measurement_index;
    }
    EXPECT_THROW(validate_prepared_model(request, invalid), std::exception);
  }
}

} // namespace
} // namespace cudaq::qec::memory_experiment
