/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// For full test script: surface_code-3-test.sh
//
// Surface-code memory experiment with two realtime decoders per logical qubit
// (separate X- and Z-error decoding), runnable in either basis (prep0 or
// prep+). Unique to this example: it builds a separate DEM per decoder with two
// `cudaq::dem_from_kernel` passes, selected by the `declare_detectors_z`/`_x`
// flags that partition the cross-round detectors by stabilizer type.

#include "cudaq.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/codes/surface_code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <common/CustomOp.h>
#include <common/NoiseModel.h>
#include <cudaq/algorithms/dem.h>
#include <fstream>

// Whether or not to put calls to debug functions in the QIR program. You cannot
// set this to 1 if you are submitting to hardware.
#ifndef PER_SHOT_DEBUG
#define PER_SHOT_DEBUG 0
#endif

// Uncomment this to manually inject errors.
// #define MANUALLY_INJECT_ERRORS

// + Helper function to create a decoder config from a DEM
cudaq::qec::decoding::config::decoder_config
create_decoder_config(uint64_t id, const cudaq::qec::detector_error_model &dem,
                      const std::vector<int64_t> &det_mat,
                      uint64_t numSyndromesPerRound) {
  cudaq::qec::decoding::config::decoder_config config;
  config.id = id;
  config.type = "multi_error_lut";
  config.block_size = dem.num_error_mechanisms();
  config.syndrome_size = dem.num_detectors();
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);
  config.D_sparse = det_mat;
  cudaqx::heterogeneous_map lut_args;
  lut_args.insert("lut_error_depth", 2);
  config.decoder_custom_args = lut_args;
  return config;
}

// *** Now takes two DEMs to save a combined config
void save_dem_to_file(const cudaq::qec::detector_error_model &dem_x,
                      const cudaq::qec::detector_error_model &dem_z,
                      const std::vector<int64_t> &det_mat_x,
                      const std::vector<int64_t> &det_mat_z,
                      std::string dem_filename, uint64_t numSyndromesPerRound_x,
                      uint64_t numSyndromesPerRound_z, uint64_t numLogical) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  for (uint64_t i = 0; i < numLogical; i++) {
    // Decoder for X errors (from Z stabilizers)
    multi_config.decoders.push_back(
        create_decoder_config(2 * i, dem_z, det_mat_z, numSyndromesPerRound_z));
    // Decoder for Z errors (from X stabilizers)
    multi_config.decoders.push_back(create_decoder_config(
        2 * i + 1, dem_x, det_mat_x, numSyndromesPerRound_x));
  }
  std::string config_str = multi_config.to_yaml_str(200);
  std::ofstream config_file(dem_filename);
  config_file << config_str;
  config_file.close();
  printf("Saved config to file: %s\n", dem_filename.c_str());
}

// *** Now loads a config for 2*numLogical decoders and initializes dem_z and
// dem_x
void load_dem_from_file(const std::string &dem_filename,
                        std::vector<cudaq::qec::detector_error_model> &dem_z,
                        std::vector<cudaq::qec::detector_error_model> &dem_x,
                        uint64_t numLogical) {
  printf("load_dem_from_file: Loading dem from file: %s\n",
         dem_filename.c_str());

  // Read dem_filename into a std::string
  std::ifstream dem_file(dem_filename);
  std::string dem_str((std::istreambuf_iterator<char>(dem_file)),
                      std::istreambuf_iterator<char>());

  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          dem_str);
  if (2 * numLogical != config.decoders.size()) {
    printf("ERROR: 2 * numLogical [%lu] != config.decoders.size() [%lu]\n",
           2 * numLogical, config.decoders.size());
    exit(1);
  }

  dem_z.resize(numLogical);
  dem_x.resize(numLogical);

  for (uint64_t i = 0; i < numLogical; ++i) {
    const auto &decoder_config_z = config.decoders[2 * i];
    const auto &decoder_config_x = config.decoders[2 * i + 1];

    // Z stab decoder
    dem_z[i].detector_error_matrix = cudaq::qec::pcm_from_sparse_vec(
        decoder_config_z.H_sparse, decoder_config_z.syndrome_size,
        decoder_config_z.block_size);

    size_t num_observables_z = std::count(decoder_config_z.O_sparse.begin(),
                                          decoder_config_z.O_sparse.end(), -1);
    dem_z[i].observables_flips_matrix = cudaq::qec::pcm_from_sparse_vec(
        decoder_config_z.O_sparse, num_observables_z,
        decoder_config_z.block_size);

    // X stab decoder
    dem_x[i].detector_error_matrix = cudaq::qec::pcm_from_sparse_vec(
        decoder_config_x.H_sparse, decoder_config_x.syndrome_size,
        decoder_config_x.block_size);

    size_t num_observables_x = std::count(decoder_config_x.O_sparse.begin(),
                                          decoder_config_x.O_sparse.end(), -1);
    dem_x[i].observables_flips_matrix = cudaq::qec::pcm_from_sparse_vec(
        decoder_config_x.O_sparse, num_observables_x,
        decoder_config_x.block_size);
  }

  printf("Loaded %lu Z and %lu X DEMs from file: %s\n", dem_z.size(),
         dem_x.size(), dem_filename.c_str());

  // Now configure the decoders globally (if needed for inference later)
  cudaq::qec::decoding::config::configure_decoders(config);
}

std::vector<size_t> get_stab_cnot_schedule(char stab_type, int distance) {
  cudaq::qec::surface_code::stabilizer_grid grid(distance);
  if (stab_type != 'X' && stab_type != 'Z') {
    throw std::runtime_error(
        "get_stab_cnot_schedule: Invalid stabilizer type. Must be 'X' or 'Z'.");
  }
  // CNOT pairs ordered by timestep within each stabilizer, so that mid-round
  // ancilla (hook) errors land perpendicular to the logical operators.
  // Stabilizer indices match the sorted parity-matrix rows and hence the
  // ancilla indexing.
  return stab_type == 'X' ? grid.get_cnot_schedule_pairs_x()
                          : grid.get_cnot_schedule_pairs_z();
}

void debug_print_syndromes(int64_t syndrome_x_int, int64_t syndrome_z_int) {
  printf("syndrome_x_int: %ld, syndrome_z_int: %ld\n", syndrome_x_int,
         syndrome_z_int);
}

void debug_print_applying_correction(const std::string &type,
                                     int64_t correction) {
  printf("Applying %s correction: %ld\n", type.c_str(), correction);
}

void debug_start_shot() { printf("Starting shot\n"); }

namespace cudaq::qec::qpu {

// Transversal CNOT gate
__qpu__ void logical_cnot(cudaq::qview<> ctrl_data, cudaq::qview<> tgt_data) {
  for (std::size_t i = 0; i < ctrl_data.size(); i++) {
    cudaq::x<cudaq::ctrl>(ctrl_data[i], tgt_data[i]);
  }
}
__qpu__ void spam_error(cudaq::qec::patch logicalQubit, double p_spam_data,
                        double p_spam_ancx, double p_spam_ancz) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_data,
                                               logicalQubit.data[i]);
  }
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_ancx,
                                               logicalQubit.ancx[i]);
  }
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_ancz,
                                               logicalQubit.ancz[i]);
  }
}

// Z basis measurement -- Z syndromes
__qpu__ std::vector<cudaq::measure_result>
se_z_ft(cudaq::qec::patch logicalQubit,
        const std::vector<std::size_t> &cnot_sched) {
  for (std::size_t i = 0; i < cnot_sched.size(); i += 2) {
    cudaq::x<cudaq::ctrl>(logicalQubit.data[cnot_sched[i + 1]],
                          logicalQubit.ancz[cnot_sched[i]]);
  }
  auto results = mz(logicalQubit.ancz);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);
  return results;
}

// X basis measurement -- X syndromes
__qpu__ std::vector<cudaq::measure_result>
se_x_ft(cudaq::qec::patch logicalQubit,
        const std::vector<std::size_t> &cnot_sched) {
  h(logicalQubit.ancx);
  for (std::size_t i = 0; i < cnot_sched.size(); i += 2) {
    cudaq::x<cudaq::ctrl>(logicalQubit.ancx[cnot_sched[i]],
                          logicalQubit.data[cnot_sched[i + 1]]);
  }
  h(logicalQubit.ancx);
  auto results = mz(logicalQubit.ancx);
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  return results;
}

// Runs the stabilizer measurement rounds and returns the last round's
// combined [Z..., X...] syndrome so the caller can chain it into a later
// call. When declare_detectors_z (or _x) is set (DEM generation of that
// stabilizer type's decoder model), every round declares that type's
// detectors: cross-round pairs against the previous round or, when there is
// no reference round yet (the lock-in round), one single-measurement
// detector per stabilizer if the type matches the prepared basis - the only
// type that is deterministic right after state prep. prev_syndrome supplies
// the reference round (pass an empty vector for none).
__qpu__ std::vector<cudaq::measure_result> custom_memory_circuit_stabs(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    std::size_t numRounds, const std::vector<std::size_t> &cnot_schedX_flat,
    const std::vector<std::size_t> &cnot_schedZ_flat, bool enqueue_syndromes,
    bool do_errors_after_non_last_rounds, double p_spam, int logical_qubit_idx,
    bool is_on_Z_basis, bool declare_detectors_z, bool declare_detectors_x,
    const std::vector<cudaq::measure_result> &prev_syndrome) {
  patch logical(data, xstab_anc, zstab_anc);

  // Local copy of the reference syndrome (kernel vector parameters are
  // read-only, and the round loop below reassigns this as it advances). The
  // local is always full-size so the reassignment never changes its length;
  // have_prev says whether it currently holds a valid reference round.
  bool have_prev = prev_syndrome.size() == xstab_anc.size() + zstab_anc.size();
  std::vector<cudaq::measure_result> prev(xstab_anc.size() + zstab_anc.size());
  if (have_prev) {
    for (std::size_t k = 0; k < prev.size(); ++k)
      prev[k] = prev_syndrome[k];
  }

  for (std::size_t round = 0; round < numRounds; round++) {
    auto syndrome_z = se_z_ft(logical, cnot_schedZ_flat);
    auto syndrome_x = se_x_ft(logical, cnot_schedX_flat);

    // *** Enqueue syndromes to separate decoders
    if (enqueue_syndromes) {
      // Enqueue Z-stabilizer results to the X-error decoder
      cudaq::qec::decoding::enqueue_syndromes(
          /*decoder_id=*/2 * logical_qubit_idx, syndrome_z);

      // Enqueue X-stabilizer results to the Z-error decoder
      cudaq::qec::decoding::enqueue_syndromes(
          /*decoder_id=*/2 * logical_qubit_idx + 1, syndrome_x);
    }

    if (have_prev) {
      if (declare_detectors_z) {
        for (std::size_t k = 0; k < zstab_anc.size(); ++k)
          cudaq::detector(prev[k], syndrome_z[k]);
      }
      if (declare_detectors_x) {
        for (std::size_t k = 0; k < xstab_anc.size(); ++k)
          cudaq::detector(prev[zstab_anc.size() + k], syndrome_x[k]);
      }
    } else {
      // Lock-in round: only the prepared-basis stabilizers are deterministic.
      if (declare_detectors_z && is_on_Z_basis) {
        for (std::size_t k = 0; k < zstab_anc.size(); ++k)
          cudaq::detector(syndrome_z[k]);
      }
      if (declare_detectors_x && !is_on_Z_basis) {
        for (std::size_t k = 0; k < xstab_anc.size(); ++k)
          cudaq::detector(syndrome_x[k]);
      }
    }
    std::vector<cudaq::measure_result> combined_syndrome(xstab_anc.size() +
                                                         zstab_anc.size());
    for (std::size_t k = 0; k < zstab_anc.size(); ++k)
      combined_syndrome[k] = syndrome_z[k];
    for (std::size_t k = 0; k < xstab_anc.size(); ++k)
      combined_syndrome[zstab_anc.size() + k] = syndrome_x[k];
    prev = combined_syndrome;
    have_prev = true;

    if (do_errors_after_non_last_rounds && round < numRounds - 1) {
      // spam_error(logical, p_spam, p_spam, p_spam);
      spam_error(logical, /*p_spam_data=*/p_spam, /*p_spam_ancx=*/0.001,
                 /*p_spam_ancz=*/0.001);
      // Uncomment the following to force a single error that should likely be
      // correctable.
#if MANUALLY_INJECT_ERRORS
      if (round == 0) {
        // Inject a single error
        if (is_on_Z_basis)
          cudaq::x(logical.data[3]);
        else
          cudaq::z(logical.data[3]);
      }
#endif
    }
  }
  return prev;
}

// When declare_detectors_z (or _x) is set (only meaningful with
// numLogical = 1 and allow_device_calls = false), the kernel annotates
// itself for DEM generation of that stabilizer type's decoder model via
// cudaq::dem_from_kernel - one dem_from_kernel pass per decoder. The lock-in
// round declares the deterministic prepared-basis stabilizer singles, every
// later round declares cross-round detectors against the previous round,
// and, on the pass whose type matches the prepared basis, the logical
// observable is declared over the final data measurements at obs_indices
// (the data qubits are already rotated into that basis before the final
// measurement; the other basis' observable is not deterministic).
__qpu__ std::int64_t demo_circuit_qpu(
    bool allow_device_calls, bool declare_detectors_z, bool declare_detectors_x,
    const cudaq::qec::code::one_qubit_encoding &statePrep, bool is_on_Z_basis,
    std::size_t numData, std::size_t numAncx, std::size_t numAncz,
    std::size_t numRounds, std::size_t numLogical,
    const std::vector<std::size_t> &cnot_schedX_flat,
    const std::vector<std::size_t> &cnot_schedZ_flat, double p_spam,
    bool apply_corrections, const std::vector<std::size_t> &obs_indices) {
#if PER_SHOT_DEBUG
  debug_start_shot();
#endif
  std::uint64_t num_corrections = 0;

  if (allow_device_calls) {
    // *** Reset 2 decoders per logical qubit
    for (int logical_qubit_idx = 0; logical_qubit_idx < numLogical;
         logical_qubit_idx++) {
      cudaq::qec::decoding::reset_decoder(/*decoder_id=*/2 * logical_qubit_idx);
      cudaq::qec::decoding::reset_decoder(/*decoder_id=*/2 * logical_qubit_idx +
                                          1);
    }
  }

  cudaq::qvector data(numLogical * numData), xstab_anc(numLogical * numAncx),
      zstab_anc(numLogical * numAncz);

  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
    auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);
    patch logical(subData, subXstab_anc, subZstab_anc);
    statePrep(logical);
  }

  // For each logical qubit: lock in the stabilizers, inject SPAM, then run the
  // measurement rounds. Each qubit's own lock-in syndrome is the reference
  // round for its first cross-round detectors (DEM generation always uses
  // numLogical = 1).
  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
    auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);

    // Lock-in round (no reference round yet).
    std::vector<cudaq::measure_result> no_prev(0);
    auto lockin_syndrome = custom_memory_circuit_stabs(
        subData, subXstab_anc, subZstab_anc,
        /*numRounds=*/1, cnot_schedX_flat, cnot_schedZ_flat,
        /*enqueue_syndromes=*/allow_device_calls,
        /*do_errors_after_non_last_rounds=*/false, p_spam, i, is_on_Z_basis,
        declare_detectors_z, declare_detectors_x, no_prev);

    // Inject errors.
    patch logical(subData, subXstab_anc, subZstab_anc);
    spam_error(logical, p_spam, 0.001, 0.001);

    // Measurement rounds, referenced to this qubit's lock-in syndrome.
    custom_memory_circuit_stabs(
        subData, subXstab_anc, subZstab_anc, numRounds, cnot_schedX_flat,
        cnot_schedZ_flat, /*enqueue_syndromes=*/allow_device_calls,
        /*do_errors_after_non_last_rounds=*/true, p_spam, i, is_on_Z_basis,
        declare_detectors_z, declare_detectors_x, lockin_syndrome);
  }

  std::uint16_t num_x_corrections = 0;
  std::uint16_t num_z_corrections = 0;

  // *** Apply corrections from separate X and Z decoders
  if (allow_device_calls && apply_corrections) {
    for (int i = 0; i < numLogical; i++) {
      auto subData = data.slice(i * numData, numData);

      if (is_on_Z_basis) {
        // Get X correction from Z-stabilizer decoder (ID: 2*i)
        auto x_correction_result = cudaq::qec::decoding::get_corrections(
            /*decoder_id=*/2 * i, /*return_size=*/1, /*reset=*/false);
        if (x_correction_result[0] != 0) {
          num_x_corrections++;
          cudaq::x(subData); // Apply transversal X, in this version we
          // assume the correction outside
        }
      }

      if (!is_on_Z_basis) {
        // Get Z correction from X-stabilizer decoder (ID: 2*i + 1)
        auto z_correction_result = cudaq::qec::decoding::get_corrections(
            /*decoder_id=*/2 * i + 1, /*return_size=*/1, /*reset=*/false);
        if (z_correction_result[0] != 0) {
          num_z_corrections++;
          cudaq::z(subData); // Apply transversal Z, in this version we
          // assume the correction outside
        }
      }
    }
  }

  std::uint64_t packed_counts = ((std::uint64_t)num_z_corrections << 16) |
                                (std::uint64_t)num_x_corrections;
  std::uint64_t ret = 0;
  for (int i = 0; i < numLogical; i++) {
    if (i > 0)
      ret <<= numData;
    auto subData = data.slice(i * numData, numData);

    if (!is_on_Z_basis) {
      // this is meant to turn the mz into an mx when needed
      // but it works only for very basic types of state preps
      h(subData);
    }

    auto subMeas = mz(subData);

    // The observable belongs to the pass declaring the prepared-basis
    // detectors; the other basis' observable is not deterministic.
    bool declare_observable =
        is_on_Z_basis ? declare_detectors_z : declare_detectors_x;
    if (declare_observable && i == 0) {
      std::vector<cudaq::measure_result> obs_log(obs_indices.size());
      for (std::size_t k = 0; k < obs_indices.size(); ++k)
        obs_log[k] = subMeas[obs_indices[k]];
      cudaq::logical_observable(obs_log, /*observable_index=*/0);
    }

    int bit_offset = (numLogical - 1 - i) * numData;

    // Pack the measured bits branch-free. Routing the measurement results
    // through a branch (or a call like to_integer) would tag this kernel
    // with qubitMeasurementFeedback, which cudaq::dem_from_kernel rejects.
    for (std::size_t j = 0; j < subMeas.size(); j++) {
      std::uint64_t bitval = subMeas[j];
      ret |= bitval << (bit_offset + j);
    }
  }
  ret |= packed_counts << (numData * numLogical);
  return ret;
}

} // namespace cudaq::qec::qpu

void demo_circuit_host(const cudaq::qec::code &code, int distance,
                       double p_spam, cudaq::qec::operation statePrep,
                       std::size_t numShots, std::size_t numRounds,
                       std::size_t numLogical, std::string dem_filename,
                       bool save_dem, bool load_dem) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep =
      code.get_operation<cudaq::qec::code::one_qubit_encoding>(statePrep);

  if (!code.contains_operation(cudaq::qec::operation::stabilizer_round))
    throw std::runtime_error("demo_circuit_host error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound = code.get_operation<cudaq::qec::code::stabilizer_round>(
      cudaq::qec::operation::stabilizer_round);

  bool is_on_Z_basis;
  if (statePrep == cudaq::qec::operation::prepp) {
    is_on_Z_basis = false;
  } else if (statePrep == cudaq::qec::operation::prep0) {
    is_on_Z_basis = true;
  }

  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  auto cnot_schedX_flat = get_stab_cnot_schedule('X', distance);
  auto cnot_schedZ_flat = get_stab_cnot_schedule('Z', distance);

  printf("cnot_schedX_flat: ");
  for (std::size_t i = 0; i < cnot_schedX_flat.size(); i += 2)
    printf("%lu %lu, ", cnot_schedX_flat[i], cnot_schedX_flat[i + 1]);
  printf("\n");
  printf("cnot_schedZ_flat: ");
  for (std::size_t i = 0; i < cnot_schedZ_flat.size(); i += 2)
    printf("%lu %lu, ", cnot_schedZ_flat[i], cnot_schedZ_flat[i + 1]);
  printf("\n");

  // ------------------------------------------------------------------------

  cudaq::noise_model noise;
  std::vector<cudaq::qec::detector_error_model> dem_z_vec, dem_x_vec;
  cudaq::qec::detector_error_model dem_z, dem_x;

  // The prepared-basis logical observable's data-qubit support (row 0 of the
  // matching observables matrix); demo_circuit_qpu declares the corresponding
  // logical_observable over the final data measurements when generating the
  // DEM. Only the prepared basis is deterministic, so only it is declared.
  auto obs_matrix_on_basis =
      is_on_Z_basis ? code.get_observables_z() : code.get_observables_x();
  std::vector<std::size_t> obs_indices;
  for (std::size_t col = 0; col < obs_matrix_on_basis.shape()[1]; ++col)
    if (obs_matrix_on_basis.at({0, col}))
      obs_indices.push_back(col);

  if (load_dem) {
    // *** Pass numLogical to load_dem_from_file
    load_dem_from_file(dem_filename, dem_z_vec, dem_x_vec, numLogical);
    dem_z = dem_z_vec.at(0);
    dem_x = dem_x_vec.at(0);
  } else {
    if (p_spam == 0.0) {
      printf("p_spam is 0.0, cannot generate the DEM\n");
      exit(0);
    }
    std::size_t numSyndromesPerRound = distance * distance - 1;

    // in the case of the surface code, where each should be
    // numSyndromesPerRound / 2
    auto numSyndromesPerRound_x = numAncx;
    auto numSyndromesPerRound_z = numAncz;

    if (numSyndromesPerRound_x != numSyndromesPerRound / 2 ||
        numSyndromesPerRound_z != numSyndromesPerRound / 2) {
      throw std::runtime_error("Num of X (or Z) syndrome data [" +
                               std::to_string(numSyndromesPerRound_x) + ", " +
                               std::to_string(numSyndromesPerRound_z) +
                               "] is not equal to the half of the total number "
                               "of syndromes per round + [" +
                               std::to_string(numSyndromesPerRound / 2) + "]");
    }

    // ------------------------------------------------------------------------
    // One dem_from_kernel pass per decoder: the Z pass declares only the
    // Z-stabilizer detectors (and, on a Z-basis prep, the observable), the X
    // pass only the X-stabilizer ones. Each pass directly yields that
    // decoder's model.
    cudaq::M2DSparseMatrix m2d_z, m2d_x;
    cudaq::M2OSparseMatrix m2o_z, m2o_x;
    std::string dem_text_z = cudaq::dem_from_kernel(
        cudaq::qec::qpu::demo_circuit_qpu, &noise, m2d_z, m2o_z,
        /*allow_device_calls=*/false,
        /*declare_detectors_z=*/true, /*declare_detectors_x=*/false, prep,
        is_on_Z_basis, numData, numAncx, numAncz, numRounds,
        /*numLogical=*/1, cnot_schedX_flat, cnot_schedZ_flat, p_spam,
        /*apply_corrections=*/false, obs_indices);
    dem_z = cudaq::qec::dem_from_stim_text(dem_text_z);

    std::string dem_text_x = cudaq::dem_from_kernel(
        cudaq::qec::qpu::demo_circuit_qpu, &noise, m2d_x, m2o_x,
        /*allow_device_calls=*/false,
        /*declare_detectors_z=*/false, /*declare_detectors_x=*/true, prep,
        is_on_Z_basis, numData, numAncx, numAncz, numRounds,
        /*numLogical=*/1, cnot_schedX_flat, cnot_schedZ_flat, p_spam,
        /*apply_corrections=*/false, obs_indices);
    dem_x = cudaq::qec::dem_from_stim_text(dem_text_x);

    // Only the prepared-basis pass declares an observable (the other basis'
    // observable is not deterministic), so the off-basis DEM comes back with
    // zero observables. Give its decoder an explicit all-zero observables
    // row; its corrections are never queried in this experiment.
    auto &off_basis_dem = is_on_Z_basis ? dem_x : dem_z;
    off_basis_dem.observables_flips_matrix = cudaqx::tensor<uint8_t>(
        {std::size_t{1}, off_basis_dem.detector_error_matrix.shape()[1]});

    printf("numSyndromesPerRound_z: %ld\n", numSyndromesPerRound_z);
    printf("numSyndromesPerRound_x: %ld\n", numSyndromesPerRound_x);

    // Canonicalize both DEMs with their respective syndrome counts
    dem_z.canonicalize_for_rounds(numSyndromesPerRound_z,
                                  /*remove_zero_syndrome_errors=*/true);
    dem_x.canonicalize_for_rounds(numSyndromesPerRound_x,
                                  /*remove_zero_syndrome_errors=*/true);

    printf("dem_z.detector_error_matrix:\n");
    dem_z.detector_error_matrix.dump_bits();
    printf("dem_z.observables_flips_matrix:\n");
    dem_z.observables_flips_matrix.dump_bits();

    printf("dem_x.detector_error_matrix:\n");
    dem_x.detector_error_matrix.dump_bits();
    printf("dem_x.observables_flips_matrix:\n");
    dem_x.observables_flips_matrix.dump_bits();

    // ------------------------------------------------------------------------
    // Per-decoder runtime detector matrices, straight from each pass'
    // measurements-to-detectors map. m2d indexes the kernel's full
    // chronological measurement stream ([Z..., X...] per round), while each
    // decoder only receives its own stabilizer type, so remap every index
    // into the decoder's own (Z-only or X-only) enqueue stream.
    auto build_det_mat = [&](const cudaq::M2DSparseMatrix &m2d, bool z_type) {
      std::vector<int64_t> out;
      for (const auto &row : m2d.rows) {
        for (auto g : row) {
          const std::size_t round = g / numSyndromesPerRound;
          const std::size_t pos = g % numSyndromesPerRound;
          out.push_back(static_cast<int64_t>(
              z_type ? round * numSyndromesPerRound_z + pos
                     : round * numSyndromesPerRound_x +
                           (pos - numSyndromesPerRound_z)));
        }
        out.push_back(-1);
      }
      return out;
    };
    std::vector<int64_t> det_mat_z = build_det_mat(m2d_z, /*z_type=*/true);
    std::vector<int64_t> det_mat_x = build_det_mat(m2d_x, /*z_type=*/false);

    // ------------------------------------------------------------------------

    if (save_dem) {
      // *** Call the new save function
      save_dem_to_file(dem_x, dem_z, det_mat_x, det_mat_z, dem_filename,
                       numSyndromesPerRound_z, numSyndromesPerRound_x,
                       numLogical);
      return;
    }
  }
  // ------------------------------------------------------------------------
  size_t numSyndromesPerRound = distance * distance - 1;
  auto chosen_dem = dem_x;
  if (!is_on_Z_basis) {
    chosen_dem = dem_z;
  }
  size_t numRoundsOfSyndromData =
      chosen_dem.detector_error_matrix.shape()[0] /
      (numSyndromesPerRound /
       2); // it depends on which on the two is "on basis" for the state prep
  if (numRoundsOfSyndromData != numRounds) {
    throw std::runtime_error("Num rounds of syndrome data [" +
                             std::to_string(numRoundsOfSyndromData) +
                             "] is not equal to the number of rounds [" +
                             std::to_string(numRounds) + "]");
  }
  // ------------------------------------------------------------------------

  // If this is a remote platform (not local sim nor emulation), don't use the
  // noise model.
  bool declare_detectors = false;
  bool allow_device_calls = true;
  auto run_result =
      cudaq::get_platform().is_remote()
          ? cudaq::run(numShots, cudaq::qec::qpu::demo_circuit_qpu,
                       allow_device_calls, declare_detectors, declare_detectors,
                       prep, is_on_Z_basis, numData, numAncx, numAncz,
                       numRounds, numLogical, cnot_schedX_flat,
                       cnot_schedZ_flat, p_spam,
                       /*apply_corrections=*/true, obs_indices)
          : cudaq::run(numShots, noise, cudaq::qec::qpu::demo_circuit_qpu,
                       allow_device_calls, declare_detectors, declare_detectors,
                       prep, is_on_Z_basis, numData, numAncx, numAncz,
                       numRounds, numLogical, cnot_schedX_flat,
                       cnot_schedZ_flat, p_spam,
                       /*apply_corrections=*/true, obs_indices);

  // ------------------------------------------------------------------------
  // Collecting data on experiment

  printf("Result size: %ld\n", run_result.size());
  std::vector<std::vector<uint8_t>> logical_results;
  auto obs_matrix_z = code.get_observables_z();
  auto obs_matrix_x = code.get_observables_x();
  int num_logical_x_errors = 0;
  int num_logical_z_errors = 0;

  std::int64_t total_x_corrections = 0;
  std::int64_t total_z_corrections = 0;

  for (int i = 0; i < run_result.size(); i++) {
    std::uint64_t packed_counts =
        (std::uint64_t)run_result[i] >> (numData * numLogical);

    // Unpack the correction counts
    total_x_corrections += packed_counts & 0xFFFF;
    total_z_corrections += packed_counts >> 16;

    for (int j = 0; j < numLogical; j++) {
      std::vector<double> result_vec(numData);

      // Explicit bit-by-bit unpacking to match the packing
      int bit_offset =
          (numLogical - 1 - j) * numData; // Account for left-shifting

      for (int l = 0; l < numData; l++) {
        result_vec[l] = (run_result[i] & (1ul << (bit_offset + l))) ? 1.0 : 0.0;
      }

      cudaqx::tensor<uint8_t> result_tensor;
      cudaq::qec::convert_vec_soft_to_tensor_hard(result_vec, result_tensor);

      // Check for logical errors
      uint8_t logical_x_result = (obs_matrix_z.dot(result_tensor) % 2).at({0});
      uint8_t logical_z_result = (obs_matrix_x.dot(result_tensor) % 2).at({0});

      if (logical_x_result != 0 && is_on_Z_basis)
        num_logical_x_errors++;
      if (logical_z_result != 0 && !is_on_Z_basis)
        num_logical_z_errors++;
    }
  }
  printf("Number of logical X errors measured: %d\n", num_logical_x_errors);
  printf("Number of logical Z errors measured: %d\n", num_logical_z_errors);

  printf("Total X corrections decoder found: %ld\n", total_x_corrections);
  printf("Total Z corrections decoder found: %ld\n", total_z_corrections);

  // =======================
  // COVERAGE:
  //   - Fraction of samples where the decoder applied a correction.
  //   - High coverage = the process is very noisy, decoder must act a lot.
  //     This raises the risk of *overcorrection* (fixing states that were
  //     actually fine), which goes against the QEC principle of minimal,
  //     necessary intervention.
  //   - Low or balanced coverage is more intuitive and suggests healthier QEC.
  //   - This is a heuristic, rule-of-thumb metric of the "intuitive goodness"
  //     of the correction process.
  // =======================

  double coverage_x = (double)total_x_corrections / numShots;
  double coverage_z = (double)total_z_corrections / numShots;

  // ======================
  // FINAL ERROR RATE:
  //   - Fraction of samples still wrong after correction.
  //   - Direct measure of effectiveness: lower = better.
  //   - This is the main bottom-line result we care about.
  // ======================

  double e_final_x = (double)num_logical_x_errors / numShots;
  double e_final_z = (double)num_logical_z_errors / numShots;

  // ======================
  // UNSAFETY:
  //   - Defined as: Final Error Rate / (1 - Coverage).
  //   - Purpose: indicates how *trustworthy* the final error rate is.
  //   - If coverage is high, most samples were corrected, so we lose a stable
  //     baseline for what "uncorrected errors" would look like,
  //     and how reliable were the choices made by the decoder.
  //   - If coverage = 1, we cap unsafety = 1.0 (cannot meaningfully interpret).
  //   - This is a heuristic safety check: rule-of-thumb guidance on whether
  //     the error rate can be taken at face value in our experiments.
  // ======================

  double unsafety_x = (coverage_x < 1.0) ? e_final_x / (1.0 - coverage_x) : 1.0;
  double unsafety_z = (coverage_z < 1.0) ? e_final_z / (1.0 - coverage_z) : 1.0;

  printf("X coverage: %.4f\n", coverage_x);
  printf("X final error rate: %.4f\n", e_final_x);
  printf("X Unsafety: %.4f\n", unsafety_x);

  printf("Z coverage: %.4f\n", coverage_z);
  printf("Z final error rate: %.4f\n", e_final_z);
  printf("Z Unsafety: %.4f\n", unsafety_z);
}

void show_help() {
  printf("Usage: qec-test4 [options]\n");
  printf("Options:\n");
  printf("  --distance <int>    Distance of the surface code. Default: 5\n");
  printf("  --num_shots <int>   Number of shots. Default: 10\n");
  printf(
      "  --p_spam <double>   SPAM probability. Range[0, 1]. Default: 0.01\n");
  printf("  --num_logical <int> Number of logical qubits. Default: 1\n");
  printf("  --save_dem <string> Save the detector error model to a file.\n");
  printf("  --load_dem <string> Load the detector error model from a file. "
         "(Cannot be used with --save_dem)\n");
  printf("  --help              Show this help message\n");
}

int main(int argc, char **argv) {
  int num_shots = 10;
  int distance = 3; // Defaulting to 3 for faster testing
  double p_spam = 0.01;
  int num_logical = 1;
  bool save_dem = false;
  bool load_dem = false;
  std::string dem_filename;
  std::string state_prep;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--distance") {
      distance = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--num_shots") {
      num_shots = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--p_spam") {
      p_spam = std::stod(argv[i + 1]);
      i++;
    } else if (arg == "--help" || arg == "-h") {
      show_help();
      return 0;
    } else if (arg == "--num_logical") {
      num_logical = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--save_dem") {
      save_dem = true;
      dem_filename = argv[i + 1];
      i++;
    } else if (arg == "--load_dem") {
      load_dem = true;
      dem_filename = argv[i + 1];
      i++;
    } else if (arg == "--state_prep") {
      state_prep = argv[i + 1];
      i++;
    } else {
      printf("Unknown argument: %s\n", arg.c_str());
      show_help();
      return 1;
    }
  }

  if (!load_dem && !save_dem) {
    printf("Neither --save_dem nor --load_dem was specified. This is not a "
           "valid use case for this version of this program.\n");
    show_help();
    return 1;
  }

  int num_rounds = distance;
  if (num_logical * distance * distance >= 64) {
    printf("num_logical * distance * distance >= 64 is not supported.\n");
    return 1;
  }

  printf("Running with p_spam = %f, distance = %d, num_shots = %d\n", p_spam,
         distance, num_shots);
  auto code = cudaq::qec::get_code(
      "surface_code", cudaqx::heterogeneous_map{{"distance", distance}});

  if (state_prep == "prep0")
    demo_circuit_host(*code, distance, p_spam, cudaq::qec::operation::prep0,
                      num_shots, num_rounds, num_logical, dem_filename,
                      save_dem, load_dem);

  if (state_prep == "prepp")
    demo_circuit_host(*code, distance, p_spam, cudaq::qec::operation::prepp,
                      num_shots, num_rounds, num_logical, dem_filename,
                      save_dem, load_dem);

  cudaq::qec::decoding::config::finalize_decoders();

  return 0;
}
