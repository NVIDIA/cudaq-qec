# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [Begin Documentation]
import cudaq
import cudaq_qec as qec
from cudaq_qec import patch
import numpy as np

# Use Stim for fast stabilizer simulation
cudaq.set_target("stim")

# Repetition code parameters
distance = 3  # Number of physical qubits (also sets number of data qubits)
nRounds = 6  # Number of stabilizer measurement rounds
nShots = 10000  # Number of noisy circuit executions to simulate

# Define logical operations as CUDA-Q kernels


@cudaq.kernel
def x_logical(logicalQubit: patch):
    # Apply a logical X: bit-flip all data qubits
    for i in range(len(logicalQubit.data)):
        x(logicalQubit.data[i])


@cudaq.kernel
def prep0(logicalQubit: patch):
    # Initialize all qubits in |0⟩
    reset(logicalQubit.data)
    reset(logicalQubit.ancz)


@cudaq.kernel
def prep1(logicalQubit: patch):
    # Prepare logical |1⟩: apply X to logical |0⟩
    prep0(logicalQubit)
    x_logical(logicalQubit)


@cudaq.kernel
def stabilizer_round(logicalQubit: patch) -> list[cudaq.measure_handle]:
    # Run one round of stabilizer measurements for the Z-type repetition code

    num_ancilla = len(logicalQubit.ancz)
    num_data = len(logicalQubit.data)

    # Apply depolarizing noise to each data qubit
    for i in range(num_data):
        cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1,
                          logicalQubit.data[i])
        # It is possible to have even more control over the noise.
        # cudaq.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, logicalQubit.data[i]) # in order pX, pY, and pZ errors.

    # Measure each ZZ stabilizer using CNOTs from data to ancilla
    for i in range(num_ancilla):
        x.ctrl(logicalQubit.data[i], logicalQubit.ancz[i])
        x.ctrl(logicalQubit.data[i + 1], logicalQubit.ancz[i])

    # Measure ancilla qubits to extract the syndrome
    measurements = mz(logicalQubit.ancz)

    # Reset ancillas for the next round
    reset(logicalQubit.ancz)

    return measurements


# Define the custom repetition code using the @qec.code decorator
@qec.code('custom_repetition_code')
class MyRepetitionCode:

    def __init__(self, **kwargs):
        qec.Code.__init__(self)  # Without this it won't work
        self.distance = kwargs.get("distance", 3)

        # Create ZZ stabilizer generators
        stabilizers_str = self.__make_stabilizers_strings()
        self.stabilizers = [
            cudaq.SpinOperator.from_word(s) for s in stabilizers_str
        ]

        # Define logical Z observable
        obs_str = self.__make_pauli_observables()
        self.pauli_observables = [
            cudaq.SpinOperator.from_word(p) for p in obs_str
        ]

        # Register logical operations used by the simulator
        self.operation_encodings = {
            qec.operation.prep0: prep0,
            qec.operation.prep1: prep1,
            qec.operation.x: x_logical,
            qec.operation.stabilizer_round: stabilizer_round
        }

    def __make_stabilizers_strings(self):
        # Create ZZ stabilizer strings: e.g., "ZZI", "IZZ", etc.
        d = self.distance
        return ['I' * i + 'ZZ' + 'I' * (d - i - 2) for i in range(d - 1)]

    def __make_pauli_observables(self):
        # Logical Z is Z on all data qubits
        d = self.distance
        return ["Z" * d]

    # --- Required methods for the QEC backend ---

    def get_num_data_qubits(self):
        return self.distance

    def get_num_ancilla_x_qubits(self):
        return 0  # Not needed for Z-type repetition code

    def get_num_ancilla_z_qubits(self):
        return self.distance - 1

    def get_num_ancilla_qubits(self):
        return self.get_num_ancilla_z_qubits() + self.get_num_ancilla_x_qubits()

    def get_num_x_stabilizers(self):
        return 0

    def get_num_z_stabilizers(self):
        return self.distance - 1


# Instantiate the custom repetition code
my_repetition_code = qec.get_code("custom_repetition_code", distance=distance)
print(f"\n Created custom repetition code with distance {distance}.")

all_codes = qec.get_available_codes()
print("\n  Available QEC codes both in the library and in Python:", all_codes)

# Let's check some propreties to verify that code is correctly created
# Display the code's stabilizer generators
stabilizers = my_repetition_code.get_stabilizers()
print(f"\n The code has {len(stabilizers)} stabilizers:")
for s in stabilizers:
    print(" ", s)

logical_single_round = my_repetition_code.get_observables_z()

# Define and register a noise model
noise_model = cudaq.NoiseModel()
p = 0.01  # depolarizing noise strength

noise_model.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)

# Set initial logical state to |0⟩
statePrep = qec.operation.prep0

# Generate a full detector error model (DEM)
print("\n Generating detector error model...")
dem_rep = qec.dem_from_memory_circuit(my_repetition_code, statePrep, nRounds,
                                      noise_model)

# Extract H matrix (syndrome parity checks)
H_pcm = dem_rep.detector_error_matrix

# Extract observable flips matrix (maps physical errors to logical flips)
Lz_observables_flips_matrix = dem_rep.observables_flips_matrix

# Sample noisy executions of the full memory circuit
print("\n Sampling noisy memory circuit executions...")
syndromes, data = qec.sample_memory_circuit(my_repetition_code, statePrep,
                                            nShots, nRounds, noise_model)

print(f"\n Showing first {min(nShots, 5)} of {nShots} sampled results:")
for i in range(min(nShots, 5)):
    print(
        f"Shot {i+1:>2}: Logical measurement = {data[i]}, Syndromes = {syndromes[i]}"
    )
