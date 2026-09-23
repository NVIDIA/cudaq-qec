# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudaq_qec as qec
import cudaq
from cudaq_qec import patch


@cudaq.kernel
def prep0(logicalQubit: patch):
    h(logicalQubit.data[0], logicalQubit.data[4], logicalQubit.data[6])
    x.ctrl(logicalQubit.data[0], logicalQubit.data[1])
    x.ctrl(logicalQubit.data[4], logicalQubit.data[5])
    x.ctrl(logicalQubit.data[6], logicalQubit.data[3])
    x.ctrl(logicalQubit.data[6], logicalQubit.data[5])
    x.ctrl(logicalQubit.data[4], logicalQubit.data[2])
    x.ctrl(logicalQubit.data[0], logicalQubit.data[3])
    x.ctrl(logicalQubit.data[4], logicalQubit.data[1])
    x.ctrl(logicalQubit.data[3], logicalQubit.data[2])


@cudaq.kernel
def stabilizer(logicalQubit: patch, x_stabilizers: list[int],
               z_stabilizers: list[int]) -> list[cudaq.measure_handle]:
    h(logicalQubit.ancx)
    for xi in range(len(logicalQubit.ancx)):
        for di in range(len(logicalQubit.data)):
            if x_stabilizers[xi * len(logicalQubit.data) + di] == 1:
                x.ctrl(logicalQubit.ancx[xi], logicalQubit.data[di])

    h(logicalQubit.ancx)
    for zi in range(len(logicalQubit.ancx)):
        for di in range(len(logicalQubit.data)):
            if z_stabilizers[zi * len(logicalQubit.data) + di] == 1:
                x.ctrl(logicalQubit.data[di], logicalQubit.ancz[zi])

    results = mz([*logicalQubit.ancz, *logicalQubit.ancx])

    reset(logicalQubit.ancx)
    reset(logicalQubit.ancz)
    return results


@qec.code('py-steane-example')
class MySteaneCodeImpl:

    def __init__(self, **kwargs):
        qec.Code.__init__(self, **kwargs)
        self.stabilizers = [
            cudaq.SpinOperator.from_word(word) for word in
            ["XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI", "IIZZIZZ"]
        ]
        self.pauli_observables = [
            cudaq.SpinOperator.from_word(p) for p in ["IIIIXXX", "IIIIZZZ"]
        ]
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
