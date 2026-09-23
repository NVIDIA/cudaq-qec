# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
import numpy as np
import cudaq
import cudaq_qec as qec

# Get a QEC code
cudaq.set_target("stim")
distance = 5
surface_code = qec.get_code("surface_code", distance=distance)

# Get the Z observables.
Lz = surface_code.get_observables_z()
print(f"Lz:\n{Lz}")

nShots = 1000
nRounds = distance

# Uncomment for repeatability
# cudaq.set_random_seed(13)

# error probability
p = 0.001
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)

# prepare logical |0> state, tells the sampler to do z-basis experiment
statePrep = qec.operation.prep0
# our expected measurement in this state is 0
expected_value = 0

# Get the detector error model for this circuit.
dem = qec.z_dem_from_memory_circuit(surface_code, statePrep, nRounds, noise)

# For large runs, set verbose to False to suppress output
verbose = nShots <= 10

# Sample the surface code memory circuit with noise on each cx gate.
syndromes, data = qec.z_sample_memory_circuit(surface_code, statePrep, nShots,
                                              nRounds, noise)

if verbose:
    print("From sample function:\n")
    print("syndromes:\n", syndromes)
    print("data:\n", data)

# Get a decoder
decoder = qec.get_decoder("single_error_lut", dem.detector_error_matrix)
nLogicalErrors = 0

# Logical Mz each shot (use Lx if preparing in X-basis)
logical_measurements = (Lz @ data.transpose()) % 2
# only one logical qubit, so do not need the second axis
logical_measurements = logical_measurements.flatten()
if verbose:
    print("LMz:\n", logical_measurements)

dr = decoder.decode_batch(syndromes)
error_predictions = np.array([e.result for e in dr], dtype=np.uint8)
data_predictions = (dem.observables_flips_matrix @ error_predictions.T) % 2

nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
print(
    f'Number of logical errors without decoding (out of {nShots} shots): {nLogicalErrorsWithoutDecoding}'
)
print(
    f'Number of logical errors with decoding    (out of {nShots} shots): {nLogicalErrorsWithDecoding}'
)
