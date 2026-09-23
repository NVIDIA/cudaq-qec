# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
import cudaq
import cudaq_qec as qec
import numpy as np

# Set target simulator (Stim) for fast stabilizer circuit simulation
cudaq.set_target("stim")

distance = 3  # Code distance (number of physical qubits for repetition code)
nRounds = 6  # Number of syndrome measurement rounds
nShots = 10000  # Number of circuit samples to run

# Set verbosity based on shot count
verbose = nShots <= 10


def vprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# Retrieve a 3-qubit repetition code instance
three_qubit_repetition_code = qec.get_code("repetition", distance=distance)

# Z logical observable (for repetition codes, only Z matters)
logical_single_round = three_qubit_repetition_code.get_observables_z()

# Use predefined state preparation (|1⟩ for logical '1')
statePrep = qec.operation.prep1

# Create a noise model instance
noise_model = cudaq.NoiseModel()

# Define physical gate error probability
p = 0.01
# Define measurement error probability (not activated by default)
p_per_mz = 0.001

# Inject depolarizing noise on CX gates
noise_model.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
# noise_model.add_all_qubit_channel("mz", cudaq.BitFlipChannel(p_per_mz))  # Optional: measurement noise

# === Decoder Setup ===

# Generate full detector error model (DEM), tracking all observables
dem_rep_full = qec.dem_from_memory_circuit(three_qubit_repetition_code,
                                           statePrep, nRounds, noise_model)

# Generate Z-only detector error model (sufficient for repetition code)
dem_rep_z = qec.z_dem_from_memory_circuit(three_qubit_repetition_code,
                                          statePrep, nRounds, noise_model)

# Extract multi-round parity check matrix (H matrix)
H_pcm_from_dem_full = dem_rep_full.detector_error_matrix
H_pcm_from_dem_z = dem_rep_z.detector_error_matrix

# Sanity check: for repetition codes, full and Z-only matrices should match
assert (H_pcm_from_dem_z == H_pcm_from_dem_full).all()

# Retrieve observable flips matrix: maps physical errors to logical flips
Lz_observables_flips_matrix = dem_rep_z.observables_flips_matrix

# Instantiate a decoder: single-error lookup table (fast and sufficient for small codes)
decoder = qec.get_decoder("single_error_lut", H_pcm_from_dem_z)

# === Simulation ===

# Sample noisy executions of the code circuit
syndromes, data = qec.sample_memory_circuit(three_qubit_repetition_code,
                                            statePrep, nShots, nRounds,
                                            noise_model)

syndromes = syndromes.reshape((nShots, -1))

# Expected logical measurement (we prepared |1⟩)
expected_value = 1

# Counters for statistics
nLogicalErrorsWithoutDecoding = 0
nLogicalErrorsWDecoding = 0
nCorrections = 0

# === Loop over shots ===
for i in range(nShots):
    vprint(f"shot: {i}")

    data_i = data[i, :]  # Final data measurement
    vprint(f"data: {data_i}")

    results = decoder.decode(syndromes[i, :])
    convergence = results.converged
    result = results.result
    error_prediction = np.array(result, dtype=np.uint8)
    vprint(f"error_prediction: {error_prediction}")

    predicted_observable_flip = Lz_observables_flips_matrix @ error_prediction % 2
    vprint(f"predicted_observable_flip: {predicted_observable_flip}")

    measured_observable = logical_single_round @ data_i % 2
    vprint(f"measured_observable: {measured_observable}")

    if measured_observable != expected_value:
        nLogicalErrorsWithoutDecoding += 1

    predicted_observable = predicted_observable_flip ^ measured_observable
    vprint(f"predicted_observable: {predicted_observable}")

    if predicted_observable != expected_value:
        nLogicalErrorsWDecoding += 1

    nCorrections += int(predicted_observable_flip[0])

# === Summary statistics ===
print(
    f"{nLogicalErrorsWithoutDecoding} logical errors without decoding in {nShots} shots\n"
)
print(
    f"{nLogicalErrorsWDecoding} logical errors with decoding in {nShots} shots\n"
)
print(f"{nCorrections} corrections applied in {nShots} shots\n")
