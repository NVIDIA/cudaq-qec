# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [Begin Documentation]
import numpy as np
import cudaq_qec as qec

# Get a QEC code
steane = qec.get_code("steane")

# Get the parity check matrix of a code
# Can get the full code, or for CSS codes
# just the X or Z component
Hz = steane.get_parity_z()
print(f"Hz:\n{Hz}")
observable = steane.get_observables_z()
print(f"observable:\n{observable}")

# error probabily
p = 0.1
# Get a decoder
decoder = qec.get_decoder("single_error_lut", Hz)

# Perform a code capacity noise model numerical experiment
nShots = 10
nLogicalErrors = 0
for i in range(nShots):
    print(f"shot: {i}")

    # Generate noisy data
    data = qec.generate_random_bit_flips(Hz.shape[1], p)
    print(f"data: {data}")

    # Calculate which syndromes are flagged.
    syndrome = Hz @ data % 2
    print(f"syndrome: {syndrome}")

    # Decode the syndrome to predict what happened to the data
    results = decoder.decode(syndrome)
    convergence = results.converged
    result = results.result
    data_prediction = np.array(result, dtype=np.uint8)
    print(f"data_prediction: {data_prediction}")

    # See if this prediction flipped the observable
    predicted_observable = observable @ data_prediction % 2
    print(f"predicted_observable: {predicted_observable}")

    # See if the observable was actually flipped
    actual_observable = observable @ data % 2
    print(f"actual_observable: {actual_observable}")
    if (predicted_observable != actual_observable):
        nLogicalErrors += 1

# Count how many shots the decoder failed to correct the errors
print(f"{nLogicalErrors} logical errors in {nShots} shots\n")

# Can also generate syndromes and data from a single line with:
syndromes, data = qec.sample_code_capacity(Hz, nShots, p)
print("From sample function:")
print("syndromes:\n", syndromes)
print("data:\n", data)
