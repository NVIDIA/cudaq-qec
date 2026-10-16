/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cudaq/qec/code.h"

namespace cudaq::qec {

/// \entry_point_kernel
///
/// @brief Execute a memory circuit for quantum error correction
/// @param stabilizer_round Function pointer to the stabilizer round
/// implementation
/// @param statePrep Function pointer to the state preparation implementation
/// @param numData Number of data qubits in the code
/// @param numAncx Number of ancilla x qubits in the code
/// @param numAncz Number of ancilla z qubits in the code
/// @param numRounds Number of rounds to execute the memory circuit
/// @param x_stabilizer_schedule Row-major flattened X-stabilizer schedule
///        matrix (see code::get_stabilizer_schedule_x): entry 0 = no support,
///        entry k >= 1 = interaction at timestep k.
/// @param z_stabilizer_schedule Row-major flattened Z-stabilizer schedule
///        matrix (see code::get_stabilizer_schedule_z).
/// @param obs_matrix_flat Row-major flattened logical observable matrix
///        (num_observables × numData entries, values 0/1).
/// @param num_observables Number of rows in the observable matrix (k).
/// @param measure_in_x_basis Performing X- or Z-memory circuit
__qpu__ void
memory_circuit(const code::stabilizer_round &stabilizer_round,
               const code::one_qubit_encoding &statePrep, std::size_t numData,
               std::size_t numAncx, std::size_t numAncz, std::size_t numRounds,
               const std::vector<std::size_t> &x_stabilizer_schedule,
               const std::vector<std::size_t> &z_stabilizer_schedule,
               const std::vector<std::size_t> &obs_matrix_flat,
               std::size_t num_observables, bool measure_in_x_basis);
} // namespace cudaq::qec
