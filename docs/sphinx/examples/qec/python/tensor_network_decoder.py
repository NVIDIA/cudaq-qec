# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
# Check Python version
# Remove this check once the pipeline advances to Python 3.11
if sys.version_info < (3, 11):
    print(
        "Warning: The tensor network decoder requires Python 3.11 or higher. Exiting..."
    )
    sys.exit(0)

# [Begin Documentation]
"""
Example usage of tensor_network_decoder from cudaq-qec.

This script demonstrates how to instantiate and use the tensor network decoder
to decode a circuit level noise problem derived from a Stim surface code experiment.

This example requires the `cudaq-qec` package and the optional tensor-network-decoder dependencies.
To install the required dependencies, run:

pip install cudaq-qec[tensor-network-decoder]

Additionaly, in this example, you will need `stim` and `beliefmatching` packages:
pip install stim beliefmatching

"""
import cudaq_qec as qec
import numpy as np

import platform
if platform.machine().lower() in ("arm64", "aarch64"):
    print(
        "Warning: stim is not supported on manylinux ARM64/aarch64. Skipping this example..."
    )
    sys.exit(0)

import stim

from beliefmatching.belief_matching import detector_error_model_to_check_matrices


def parse_detector_error_model(detector_error_model):
    matrices = detector_error_model_to_check_matrices(detector_error_model)

    out_H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=out_H)
    out_L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=out_L)

    return out_H, out_L, [float(p) for p in matrices.priors]


def main():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                     rounds=3,
                                     distance=3,
                                     after_clifford_depolarization=0.001,
                                     after_reset_flip_probability=0.01,
                                     before_measure_flip_probability=0.01,
                                     before_round_data_depolarization=0.01)

    detector_error_model = circuit.detector_error_model(decompose_errors=True)

    H, logicals, noise_model = parse_detector_error_model(detector_error_model)

    decoder = qec.get_decoder(
        "tensor_network_decoder",
        H,
        logical_obs=logicals,
        noise_model=noise_model,
        contract_noise_model=True,
    )

    num_shots = 5
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True)

    res = decoder.decode_batch(detection_events)

    print("Tensor network prediction: ", [r.result[0] > 0.5 for r in res])
    print("Actual observable flips: ", [bool(o[0]) for o in observable_flips])


if __name__ == "__main__":
    main()
