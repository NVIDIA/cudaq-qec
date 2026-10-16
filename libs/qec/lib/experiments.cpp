/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/experiments.h"
#include "device/memory_circuit.h"
#include "cudaq/algorithms/dem.h"
#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace cudaqx;

namespace cudaq::qec {

namespace details {
auto __sample_code_capacity(const cudaqx::tensor<uint8_t> &H,
                            std::size_t nShots, double error_probability,
                            unsigned seed) {
  // init RNG
  std::mt19937 rng(seed);
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({nShots, H.shape()[1]});
  cudaqx::tensor<uint8_t> syndromes({nShots, H.shape()[0]});

  std::vector<uint8_t> bits(nShots * H.shape()[1]);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());

  // Syn = D * H^T
  // [n,s] = [n,d]*[d,s]
  syndromes = data.dot(H.transpose()) % 2;

  return std::make_tuple(syndromes, data);
}
} // namespace details

// Single shot version
cudaqx::tensor<uint8_t> generate_random_bit_flips(size_t numBits,
                                                  double error_probability) {
  // init RNG
  std::random_device rd;
  std::mt19937 rng(rd());
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({numBits});
  std::vector<uint8_t> bits(numBits);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());
  return data;
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return details::__sample_code_capacity(H, nShots, error_probability, seed);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability) {
  return details::__sample_code_capacity(H, nShots, error_probability,
                                         std::random_device()());
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability,
                              seed);
}

} // namespace cudaq::qec

namespace cudaq::qec::dem_sampler::cpu {

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_dem(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
           const std::vector<double> &error_probabilities, unsigned seed) {
  if (check_matrix.rank() != 2)
    throw std::invalid_argument("check_matrix must be rank-2");

  size_t num_checks = check_matrix.shape()[0];
  size_t num_error_mechanisms = check_matrix.shape()[1];

  if (error_probabilities.size() != num_error_mechanisms)
    throw std::invalid_argument(
        "error_probabilities size must match number of error mechanisms");

  for (double p : error_probabilities) {
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
      throw std::invalid_argument(
          "error_probabilities entries must be finite values in [0, 1]");
  }

  if (numShots == 0) {
    cudaqx::tensor<uint8_t> errors({0, num_error_mechanisms});
    cudaqx::tensor<uint8_t> checks({0, num_checks});
    return std::make_tuple(checks, errors);
  }

  std::mt19937 rng(seed);
  std::vector<std::bernoulli_distribution> distributions;
  distributions.reserve(num_error_mechanisms);
  for (double p : error_probabilities)
    distributions.emplace_back(p);

  cudaqx::tensor<uint8_t> H_binary({num_checks, num_error_mechanisms});
  std::vector<uint8_t> h_bin(num_checks * num_error_mechanisms);
  for (size_t i = 0; i < num_checks; ++i)
    for (size_t j = 0; j < num_error_mechanisms; ++j)
      h_bin[i * num_error_mechanisms + j] =
          check_matrix.at({i, j}) & static_cast<uint8_t>(1);
  H_binary.copy(h_bin.data(), H_binary.shape());

  cudaqx::tensor<uint8_t> errors({numShots, num_error_mechanisms});
  cudaqx::tensor<uint8_t> checks({numShots, num_checks});

  std::vector<uint8_t> error_bits(numShots * num_error_mechanisms);
  for (size_t shot = 0; shot < numShots; ++shot) {
    for (size_t err = 0; err < num_error_mechanisms; ++err) {
      error_bits[shot * num_error_mechanisms + err] = distributions[err](rng);
    }
  }

  errors.copy(error_bits.data(), errors.shape());
  checks = errors.dot(H_binary.transpose()) % 2;

  return std::make_tuple(checks, errors);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_dem(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
           const std::vector<double> &error_probabilities) {
  return sample_dem(check_matrix, numShots, error_probabilities,
                    std::random_device()());
}

} // namespace cudaq::qec::dem_sampler::cpu

namespace cudaq::qec {

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
dem_sampling(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t nShots,
             const std::vector<double> &error_probabilities) {
  return dem_sampler::cpu::sample_dem(check_matrix, nShots,
                                      error_probabilities);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
dem_sampling(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t nShots,
             const std::vector<double> &error_probabilities, unsigned seed) {
  return dem_sampler::cpu::sample_dem(check_matrix, nShots, error_probabilities,
                                      seed);
}

namespace details {

/// @brief A contiguous range [start, start + length) of detector-ordered
/// values: rows for a DEM's detector_error_matrix, or per-shot columns for a
/// sample_memory_circuit syndrome tensor.
struct detector_range {
  std::size_t start;
  std::size_t length;
};

std::size_t total_length(const std::vector<detector_range> &ranges) {
  std::size_t total = 0;
  for (const auto &r : ranges)
    total += r.length;
  return total;
}

/// @brief Compute the detector-ordered ranges that correspond to the
/// requested stabilizer type(s) for a memory-circuit experiment.
///
/// Detector values produced by `memory_circuit` are laid out as: `numFixed`
/// boundary values (only the stabilizer type matching the state prep/
/// measurement basis, since that is the only type whose round-0 syndrome is
/// deterministic and the only type whose last round can be reconstructed
/// from the final data measurement), then `numRounds - 1` interior rounds
/// each containing a full [Z][X] block, then another `numFixed` boundary
/// values. This is shared by `dem_from_memory_circuit` (which slices rows of
/// a detector_error_matrix) and `sample_memory_circuit` (which slices
/// columns of a syndrome tensor), so both stay in sync with a single
/// definition of the layout.
std::vector<detector_range>
stabilizer_detector_ranges(std::size_t numRounds, std::size_t numZStabs,
                           std::size_t numXStabs, bool is_z_prep,
                           bool keep_x_stabilizers, bool keep_z_stabilizers,
                           std::size_t numTotalDetectors) {
  if (!keep_x_stabilizers && !keep_z_stabilizers)
    throw std::runtime_error(
        "stabilizer_detector_ranges error - no stabilizers to keep.");

  const std::size_t numSyndromesPerRound = numZStabs + numXStabs;

  // Round-to-round detector values are laid out as [Z][X]. Keep a contiguous
  // sub-range of that layout: skip past the Z values when only X is wanted,
  // and keep however many values belong to the requested type(s).
  const std::size_t offset = keep_z_stabilizers ? 0 : numZStabs;
  const std::size_t numReturnSynPerRound =
      (keep_z_stabilizers ? numZStabs : 0) +
      (keep_x_stabilizers ? numXStabs : 0);

  const std::size_t numFixed = is_z_prep ? numZStabs : numXStabs;
  const bool keep_fixed =
      (is_z_prep && keep_z_stabilizers) || (!is_z_prep && keep_x_stabilizers);
  const std::size_t numBoundaryDets = keep_fixed ? numFixed : 0;
  const std::size_t numInteriorRounds = numRounds > 0 ? numRounds - 1 : 0;

  if (2 * numFixed + numInteriorRounds * numSyndromesPerRound !=
      numTotalDetectors)
    throw std::runtime_error("stabilizer_detector_ranges error - unexpected "
                             "number of detector values.");

  std::vector<detector_range> ranges;

  // Initial boundary detectors: the first `numFixed` values.
  if (numBoundaryDets > 0)
    ranges.push_back({0, numBoundaryDets});

  // Interior round-to-round transitions, each sliced down to the requested
  // stabilizer type(s).
  for (std::size_t round = 0; round < numInteriorRounds; ++round)
    ranges.push_back({numFixed + round * numSyndromesPerRound + offset,
                      numReturnSynPerRound});

  // Final boundary detectors: the last `numFixed` values.
  if (numBoundaryDets > 0)
    ranges.push_back({numTotalDetectors - numFixed, numBoundaryDets});

  return ranges;
}

/// @brief Given a memory circuit setup, sample syndrome and data qubit
/// measurements. This is the main driver function that all of the
/// sample_memory_circuit overloads invoke.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise, bool keep_x_stabilizers,
                      bool keep_z_stabilizers) {
  if (!keep_x_stabilizers && !keep_z_stabilizers)
    throw std::runtime_error(
        "sample_memory_circuit error - no stabilizers to keep.");
  if (numRounds == 0)
    throw std::runtime_error(
        "sample_memory_circuit error - numRounds must be >= 1. The memory "
        "circuit always performs at least one stabilizer measurement round. ");
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);
  if (!(statePrep == operation::prep0 || statePrep == operation::prep1 ||
        statePrep == operation::prepp || statePrep == operation::prepm))
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");

  bool is_z_prep =
      statePrep == operation::prep0 || statePrep == operation::prep1;

  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("sample_memory_circuit error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);

  // Schedule matrices, not plain parity matrices: nonzero entries carry the
  // per-stabilizer interaction timestep (see code::get_stabilizer_schedule_x).
  auto sched_x = code.get_stabilizer_schedule_x();
  auto sched_z = code.get_stabilizer_schedule_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();
  auto numXStabs = code.get_num_x_stabilizers();
  auto numZStabs = code.get_num_z_stabilizers();

  std::vector<std::size_t> xVec(sched_x.data(),
                                sched_x.data() + sched_x.size());
  std::vector<std::size_t> zVec(sched_z.data(),
                                sched_z.data() + sched_z.size());
  auto logical_obs =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                    logical_obs.data() + logical_obs.size());

  const std::size_t numCols = numAncx + numAncz;

  // Obtain the Measurement-to-Detector (M2D) sparse matrix.
  // m2d.rows[d] = set of chronological measurement indices whose XOR = detector
  // d.
  cudaq::M2DSparseMatrix m2d;
  cudaq::M2OSparseMatrix m2o;
  cudaq::dem_from_kernel(memory_circuit, &noise, /*options=*/{}, m2d, m2o,
                         stabRound, prep, numData, numAncx, numAncz, numRounds,
                         xVec, zVec, obs_flat, num_obs, !is_z_prep);

  // Sample the memory circuit and collect all raw measurements.
  cudaq::sample_options opts{
      .shots = numShots, .noise = noise, .explicit_measurements = true};
  auto result = cudaq::sample(opts, memory_circuit, stabRound, prep, numData,
                              numAncx, numAncz, numRounds, xVec, zVec, obs_flat,
                              num_obs, !is_z_prep);

  // mzTable[shot, meas_idx] = raw 0/1 outcome; shape (numShots,
  // numMeasPerShot). Measurement layout per shot: numRounds*numCols ancilla,
  // then numData qubits.
  cudaqx::tensor<uint8_t> mzTable(result.sequential_data());

  // Data results: tail numData measurements of each shot.
  cudaqx::tensor<uint8_t> dataResults({numShots, numData});
  for (std::size_t shot = 0; shot < numShots; ++shot)
    std::memcpy(&dataResults.at({shot, 0}),
                &mzTable.at({shot, numCols * numRounds}), numData);

  // syndromeTensor = M2D @ mzTable   (per shot).
  // Output shape: (numShots, k) where k = number of detectors
  std::size_t k = m2d.rows.size();

  cudaqx::tensor<uint8_t> syndromeTensor({numShots, k});
  for (std::size_t shot = 0; shot < numShots; ++shot) {
    const uint8_t *measRow = &mzTable.at({shot, 0});
    std::size_t d = 0;
    for (const auto &rowIdx : m2d.rows) {
      uint8_t val = 0;
      for (std::size_t m : rowIdx) {
        val ^= measRow[m];
      }
      syndromeTensor.at({shot, d++}) = val;
    }
  }

  if (!keep_x_stabilizers || !keep_z_stabilizers) {
    auto ranges =
        stabilizer_detector_ranges(numRounds, numZStabs, numXStabs, is_z_prep,
                                   keep_x_stabilizers, keep_z_stabilizers, k);
    // Build a new [numShots, total_length(ranges)] tensor by copying the
    // requested column ranges out of syndromeTensor, per shot, to restrict it
    // to the requested stabilizer type(s).
    const auto numOutCols = total_length(ranges);
    cudaqx::tensor<uint8_t> selectedSyndromes({numShots, numOutCols});
    // No columns to keep (e.g. a single-round experiment where the requested
    // stabilizer type has no detectors at all): nothing to copy, and indexing
    // column 0 of a zero-column tensor below would be out of bounds.
    if (numOutCols > 0) {
      for (std::size_t shot = 0; shot < numShots; ++shot) {
        const auto *src_row = &syndromeTensor.at({shot, 0});
        auto *dst_row = &selectedSyndromes.at({shot, 0});
        std::size_t col = 0;
        for (const auto &r : ranges) {
          std::memcpy(dst_row + col, src_row + r.start,
                      r.length * sizeof(uint8_t));
          col += r.length;
        }
      }
    }
    syndromeTensor = std::move(selectedSyndromes);
  }

  return std::make_tuple(syndromeTensor, dataResults);
}
} // namespace details

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise) {
  return details::sample_memory_circuit(code, statePrep, numShots, numRounds,
                                        noise, /*keep_x_stabilizers=*/true,
                                        /*keep_z_stabilizers=*/true);
}

// For CSS codes, may want to partition x vs z decoding.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
x_sample_memory_circuit(const code &code, operation statePrep,
                        std::size_t numShots, std::size_t numRounds,
                        cudaq::noise_model &noise) {
  return details::sample_memory_circuit(code, statePrep, numShots, numRounds,
                                        noise, /*keep_x_stabilizers=*/true,
                                        /*keep_z_stabilizers=*/false);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
z_sample_memory_circuit(const code &code, operation statePrep,
                        std::size_t numShots, std::size_t numRounds,
                        cudaq::noise_model &noise) {
  return details::sample_memory_circuit(code, statePrep, numShots, numRounds,
                                        noise, /*keep_x_stabilizers=*/false,
                                        /*keep_z_stabilizers=*/true);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation op, std::size_t numShots,
                      std::size_t numRounds) {
  cudaq::noise_model noise;
  return sample_memory_circuit(code, op, numShots, numRounds, noise);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds, cudaq::noise_model &noise) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds,
                               noise);
}

namespace details {

/// @brief Select detector rows and m2d rows, canonicalize once, and return
/// (dem, m2d, m2o). For the full (both-basis) case the boundary-aware
/// canonicalization is used; for a single-type selection the layout is uniform
/// so the simpler overload suffices.
static decoder_inputs
make_component(detector_error_model dem, cudaq::M2DSparseMatrix m2d,
               cudaq::M2OSparseMatrix m2o, std::size_t num_rounds,
               std::size_t num_x_stabilizers, std::size_t num_z_stabilizers,
               bool fixed_basis_is_z, bool keep_x, bool keep_z) {
  if (keep_x && keep_z) {
    const uint32_t numBoundary = fixed_basis_is_z
                                     ? static_cast<uint32_t>(num_z_stabilizers)
                                     : static_cast<uint32_t>(num_x_stabilizers);
    dem.canonicalize_for_rounds_with_boundary(
        static_cast<uint32_t>(num_x_stabilizers + num_z_stabilizers),
        numBoundary, /*remove_zero_syndrome_errors=*/true);
    return {std::move(dem), std::move(m2d), std::move(m2o)};
  }

  const std::size_t numDetectors = dem.detector_error_matrix.shape()[0];
  const std::size_t numErrorMechanisms = dem.detector_error_matrix.shape()[1];
  const auto ranges = stabilizer_detector_ranges(
      num_rounds, num_z_stabilizers, num_x_stabilizers, fixed_basis_is_z,
      keep_x, keep_z, numDetectors);
  const std::size_t length = total_length(ranges);

  if (length == 0) {
    // No detectors for this stabilizer type: return an empty result that
    // carries the observable map and measurement count.
    cudaq::M2DSparseMatrix empty_m2d;
    empty_m2d.num_measurements = m2d.num_measurements;
    const std::size_t numObs = dem.num_observables();
    detector_error_model empty_dem;
    empty_dem.detector_error_matrix = cudaqx::tensor<uint8_t>({0, 0});
    empty_dem.observables_flips_matrix = cudaqx::tensor<uint8_t>({numObs, 0});
    return {std::move(empty_dem), std::move(empty_m2d), std::move(m2o)};
  }

  // Select the detector rows.
  cudaqx::tensor<uint8_t> selected_dem({length, numErrorMechanisms});
  auto *dst = selected_dem.data();
  const auto *src = dem.detector_error_matrix.data();
  for (const auto &r : ranges) {
    std::memcpy(dst, src + r.start * numErrorMechanisms,
                r.length * numErrorMechanisms * sizeof(uint8_t));
    dst += r.length * numErrorMechanisms;
  }
  dem.detector_error_matrix = std::move(selected_dem);

  // Select the m2d rows.
  cudaq::M2DSparseMatrix out_m2d;
  out_m2d.num_measurements = m2d.num_measurements;
  out_m2d.rows.reserve(length);
  for (const auto &r : ranges)
    for (std::size_t i = 0; i < r.length; ++i)
      out_m2d.rows.push_back(std::move(m2d.rows[r.start + i]));

  const std::size_t numReturnSynPerRound =
      (keep_z ? num_z_stabilizers : 0) + (keep_x ? num_x_stabilizers : 0);
  dem.canonicalize_for_rounds(static_cast<uint32_t>(numReturnSynPerRound),
                              /*remove_zero_syndrome_errors=*/true);
  return {std::move(dem), std::move(out_m2d), std::move(m2o)};
}

} // namespace details

std::vector<std::int64_t> d_sparse(const cudaq::M2DSparseMatrix &m2d) {
  std::vector<std::int64_t> out;
  out.reserve(m2d.rows.size() * 2); // rough estimate
  for (const auto &row : m2d.rows) {
    for (auto meas : row)
      out.push_back(static_cast<std::int64_t>(meas));
    out.push_back(-1);
  }
  return out;
}

/// Bridge from CUDA-Q circuit analysis to the QEC-owned sparse matrix a
/// decoder is constructed from.
sparse_binary_matrix m2d_to_sparse(const cudaq::M2DSparseMatrix &m2d) {
  using index_type = sparse_binary_matrix::index_type;
  if (m2d.rows.size() > std::numeric_limits<index_type>::max() ||
      m2d.num_measurements > std::numeric_limits<index_type>::max())
    throw std::overflow_error(
        "measurement-to-detector map exceeds uint32_t dimensions");

  std::vector<std::vector<index_type>> rows;
  rows.reserve(m2d.rows.size());
  for (const auto &source_row : m2d.rows) {
    auto &row = rows.emplace_back();
    row.reserve(source_row.size());
    for (const auto measurement : source_row) {
      if (measurement > std::numeric_limits<index_type>::max())
        throw std::overflow_error(
            "measurement-to-detector index exceeds uint32_t range");
      row.push_back(static_cast<index_type>(measurement));
    }
  }
  return sparse_binary_matrix::from_nested_csr(
      static_cast<index_type>(rows.size()),
      static_cast<index_type>(m2d.num_measurements), rows);
}

decoder_context decoder_context_from_memory_circuit(const code &code,
                                                    operation statePrep,
                                                    std::size_t numRounds,
                                                    cudaq::noise_model &noise,
                                                    bool decompose_errors) {
  if (numRounds == 0)
    throw std::runtime_error(
        "dem_from_memory_circuit error - numRounds must be >= 1. The memory "
        "circuit always performs at least one stabilizer measurement round.");
  if (!code.contains_operation(statePrep))
    throw std::runtime_error("dem_from_memory_circuit error - requested state "
                             "prep kernel not found.");
  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("dem_from_memory_circuit error - no stabilizer "
                             "round kernel for this code.");
  if (!(statePrep == operation::prep0 || statePrep == operation::prep1 ||
        statePrep == operation::prepp || statePrep == operation::prepm))
    throw std::runtime_error(
        "dem_from_memory_circuit - invalid requested state prep kernel.");

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);
  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);
  // Schedule matrices, not plain parity matrices: nonzero entries carry the
  // per-stabilizer interaction timestep (see code::get_stabilizer_schedule_x).
  auto sched_x = code.get_stabilizer_schedule_x();
  auto sched_z = code.get_stabilizer_schedule_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();
  const bool is_z_prep =
      statePrep == operation::prep0 || statePrep == operation::prep1;
  std::vector<std::size_t> xVec(sched_x.data(),
                                sched_x.data() + sched_x.size());
  std::vector<std::size_t> zVec(sched_z.data(),
                                sched_z.data() + sched_z.size());

  auto logical_obs =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                    logical_obs.data() + logical_obs.size());

  cudaq::dem_options dem_opts;
  dem_opts.decompose_errors = decompose_errors;
  decoder_context result;
  auto dem_text = cudaq::dem_from_kernel(
      memory_circuit, &noise, dem_opts, result.m2d_, result.m2o_, stabRound,
      prep, numData, numAncx, numAncz, numRounds, xVec, zVec, obs_flat, num_obs,
      !is_z_prep);
  result.dem_ = cudaq::qec::dem_from_stim_text(dem_text, decompose_errors);
  result.num_rounds_ = numRounds;
  result.num_x_stabilizers_ = code.get_num_x_stabilizers();
  result.num_z_stabilizers_ = code.get_num_z_stabilizers();
  result.fixed_basis_is_z_ = is_z_prep;
  return result;
}

std::size_t decoder_context::num_measurements() const {
  return m2d_.num_measurements;
}

decoder_inputs decoder_context::x_component() const {
  return details::make_component(dem_, m2d_, m2o_, num_rounds_,
                                 num_x_stabilizers_, num_z_stabilizers_,
                                 fixed_basis_is_z_, /*keep_x=*/true,
                                 /*keep_z=*/false);
}

decoder_inputs decoder_context::z_component() const {
  return details::make_component(dem_, m2d_, m2o_, num_rounds_,
                                 num_x_stabilizers_, num_z_stabilizers_,
                                 fixed_basis_is_z_, /*keep_x=*/false,
                                 /*keep_z=*/true);
}

decoder_inputs decoder_context::full_component() const {
  return details::make_component(dem_, m2d_, m2o_, num_rounds_,
                                 num_x_stabilizers_, num_z_stabilizers_,
                                 fixed_basis_is_z_, /*keep_x=*/true,
                                 /*keep_z=*/true);
}

/// @brief Given a memory circuit setup, generate a DEM
detector_error_model dem_from_memory_circuit(const code &code,
                                             operation statePrep,
                                             std::size_t numRounds,
                                             cudaq::noise_model &noise,
                                             bool decompose_errors) {
  return decoder_context_from_memory_circuit(code, statePrep, numRounds, noise,
                                             decompose_errors)
      .full_component()
      .dem;
}

// For CSS codes, may want to partition x vs z decoding
detector_error_model x_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise,
                                               bool decompose_errors) {
  return decoder_context_from_memory_circuit(code, statePrep, numRounds, noise,
                                             decompose_errors)
      .x_component()
      .dem;
}

detector_error_model z_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise,
                                               bool decompose_errors) {
  return decoder_context_from_memory_circuit(code, statePrep, numRounds, noise,
                                             decompose_errors)
      .z_component()
      .dem;
}

} // namespace cudaq::qec
