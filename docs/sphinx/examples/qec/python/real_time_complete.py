# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]

#!/usr/bin/env python3
"""
Simple 3-qubit repetition code with real-time decoding.
This is the most basic QEC example possible.
"""

import os

os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec


# Prepare logical |0⟩
@cudaq.kernel
def prep0(logical: qec.patch):
    for i in range(logical.data.size()):
        reset(logical.data[i])


# Measure ZZ stabilizers for 3-qubit repetition code
@cudaq.kernel
def measure_stabilizers(logical: qec.patch) -> list[cudaq.measure_handle]:
    for i in range(logical.ancz.size()):
        reset(logical.ancz[i])

    # Z0Z1 stabilizer
    cx(logical.data[0], logical.ancz[0])
    cx(logical.data[1], logical.ancz[0])

    # Z1Z2 stabilizer
    cx(logical.data[1], logical.ancz[1])
    cx(logical.data[2], logical.ancz[1])

    return [mz(logical.ancz[0]), mz(logical.ancz[1])]


# [Begin QEC Circuit]
# QEC circuit with real-time decoding
@cudaq.kernel
def qec_circuit() -> int:
    qec.reset_decoder(0)

    data = cudaq.qvector(3)
    ancz = cudaq.qvector(2)
    ancx = cudaq.qvector(0)
    logical = patch(data, ancx, ancz)

    prep0(logical)

    # 3 rounds of syndrome measurement
    for _ in range(3):
        syndromes = measure_stabilizers(logical)
        qec.enqueue_syndromes(0, syndromes, 0)
    # Final data readout
    data_meas = [mz(data[0]), mz(data[1]), mz(data[2])]
    qec.enqueue_syndromes(0, data_meas, 0)
    result = cudaq.to_bools(data_meas)

    # Get corrections and apply them (single logical observable)
    corrections = qec.get_corrections(0, 1, False)
    result[0] ^= corrections[0]
    if corrections[0]:
        for i in range(3):
            x(data[i])

    return cudaq.to_integer(result)


# [End QEC Circuit]


def main():
    # Get 3-qubit repetition code
    code = qec.get_code("repetition", distance=3)

    # [Begin DEM Generation]
    # Step 1: Generate detector error model
    print("Step 1: Generating DEM...")
    cudaq.set_target("stim")

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)

    ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, 3,
                                                  noise)
    dem, m2d, m2o = ctx.full_component()
    # [End DEM Generation]

    # [Begin Save DEM]
    # Save decoder config
    config = qec.decoder_config()
    config.id = 0
    config.type = "multi_error_lut"
    config.block_size = dem.detector_error_matrix.shape[1]
    config.syndrome_size = dem.detector_error_matrix.shape[0]
    config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
    config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)

    config.D_sparse = qec.d_sparse(m2d)
    # Decoder parameters are a plain dict; keys are governed by the parameter
    # schema the decoder registered (see qec.decoder_param_schema).
    config.decoder_custom_args = {"lut_error_depth": 2}
    # Check the dict against the decoder's schema (unknown keys, missing
    # required keys, decoder-specific constraints) before using the config.
    config.validate_custom_args()

    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [config]

    with open("config.yaml", 'w') as f:
        f.write(multi_config.to_yaml_str(200))
    print("Saved config to config.yaml")
    # [End Save DEM]

    # Step 2: Load config and run circuit
    print("\nStep 2: Running circuit with decoding...")
    # [Begin Load DEM]
    qec.configure_decoders_from_file("config.yaml")
    # [End Load DEM]

    run_result = cudaq.run(qec_circuit, shots_count=10, noise_model=noise)
    print("Ran 10 shots")

    qec.finalize_decoders()

    print("\nDone!")


if __name__ == "__main__":
    main()
# [End Documentation]
