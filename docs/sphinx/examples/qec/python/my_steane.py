# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [Begin Documentation1]
import cudaq
import cudaq_qec as qec
from cudaq_qec import patch
# [End Documentation1]


# [Begin Documentation2]
@cudaq.kernel
def prep0(logicalQubit: patch):
    h(logicalQubit.data[0], logicalQubit.data[4], logicalQubit.data[6])
    x.ctrl(logicalQubit.data[0], logicalQubit.data[1])
    x.ctrl(logicalQubit.data[4], logicalQubit.data[5])
    # ... additional initialization gates ...


@cudaq.kernel
def stabilizer(logicalQubit: patch, x_stabilizers: list[int],
               z_stabilizers: list[int]) -> list[cudaq.measure_handle]:
    # Measure X stabilizers
    h(logicalQubit.ancx)
    for xi in range(len(logicalQubit.ancx)):
        for di in range(len(logicalQubit.data)):
            if x_stabilizers[xi * len(logicalQubit.data) + di] == 1:
                x.ctrl(logicalQubit.ancx[xi], logicalQubit.data[di])
    h(logicalQubit.ancx)

    # Measure Z stabilizers
    for zi in range(len(logicalQubit.ancz)):
        for di in range(len(logicalQubit.data)):
            if z_stabilizers[zi * len(logicalQubit.data) + di] == 1:
                x.ctrl(logicalQubit.data[di], logicalQubit.ancz[zi])

    # Get and reset ancillas
    results = mz([*logicalQubit.ancz, *logicalQubit.ancx])
    reset(logicalQubit.ancx)
    reset(logicalQubit.ancz)
    return results


# [End Documentation2]


# [Begin Documentation3]
@qec.code('py-steane-example')
class MySteaneCodeImpl:

    def __init__(self, **kwargs):
        qec.Code.__init__(self, **kwargs)

        # Define stabilizer generators
        stabilizers_str = [
            "XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI", "IIZZIZZ"
        ]
        self.stabilizers = [
            cudaq.SpinOperator.from_word(s) for s in stabilizers_str
        ]

        # Define observables
        obs_str = ["IIIIXXX", "IIIIZZZ"]
        self.pauli_observables = [
            cudaq.SpinOperator.from_word(p) for p in obs_str
        ]

        # Register quantum kernels
        self.operation_encodings = {
            qec.operation.prep0: prep0,
            qec.operation.stabilizer_round: stabilizer
        }

    def get_num_data_qubits(self):
        return 7

    def get_num_ancilla_x_qubits(self):
        return 3

    def get_num_ancilla_z_qubits(self):
        return 3

    def get_num_ancilla_qubits(self):
        return 6

    def get_num_x_stabilizers(self):
        return 3

    def get_num_z_stabilizers(self):
        return 3


# [End Documentation3]
