# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [Begin Documentation]
import numpy as np
import cudaq_qec as qec

# Each row compares a neighboring pair of bits in a three-bit repetition code.
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
decoder = qec.get_decoder("single_error_lut", H)

# Start from 000 and inject no error or one bit flip at each position.
errors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

for error in errors:
    syndrome = (H @ error) % 2
    result = decoder.decode(syndrome)
    correction = np.array(result.result, dtype=np.uint8)
    corrected = (error + correction) % 2

    assert result.converged, "The decoder did not find a correction."
    assert np.array_equal(corrected,
                          [0, 0, 0]), "The original bits were not recovered."
    print(f"error={error.tolist()}  syndrome={syndrome.tolist()}  "
          f"corrected={corrected.tolist()}")

print("Installation check passed: all 4 cases recovered.")
