# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [Begin Documentation]
import numpy as np
import cudaq_qec as qec
import matplotlib.pyplot as plt

# Get a QEC code
# steane = qec.get_code("repetition", distance=9)
steane = qec.get_code("steane")

# Get the parity check matrix of a code
# Can get the full code, or for CSS codes
# just the X or Z component
Hz = steane.get_parity_z()
observable = steane.get_observables_z()

# Get a decoder
decoder = qec.get_decoder("single_error_lut", Hz)

# Perform a code capacity noise model numerical experiment
nShots = 100000
LERates = []
# PERates = np.linspace(0.1, 0.50, num=20)
PERates = np.logspace(-2.0, -0.5, num=25)

for p in PERates:
    nLogicalErrors = 0
    for i in range(nShots):
        data = qec.generate_random_bit_flips(Hz.shape[1], p)
        # Calculate which syndromes are flagged.
        syndrome = Hz @ data % 2

        # Decode the syndrome
        results = decoder.decode(syndrome)
        convergence = results.converged
        result = results.result
        data_prediction = np.array(result)

        predicted_observable = observable @ data_prediction % 2

        actual_observable = observable @ data % 2
        if (predicted_observable != actual_observable):
            nLogicalErrors += 1
    LERates.append(nLogicalErrors / nShots)

# Count how many shots the decoder failed to correct the errors
print("PERates:", PERates)
print("LERates:", LERates)

# Create a figure and an axes object
fig, ax = plt.subplots()

# Plot the data
ax.loglog(PERates, LERates)
ax.loglog(PERates, PERates, 'r--', label='y=x')

# Add a title and labels
ax.set_title("Steane Code")
ax.set_xlabel("Physical Error Rate")
ax.set_ylabel("Logical Error Rate")

# Show the plot
# plt.show()
# plt.savefig("myplot.png")
