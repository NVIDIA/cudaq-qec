# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
import numpy as np
import cudaq_qec as qec

dem_text = """\
error(0.1) D0 L0
error(0.1) D1 L0
error(0.05) D0 D1
error(0.02) D0 ^ D1
"""

# Decoder construction uses default parsing (use_decomp_suggestions=False):
# '^' hints in the DEM text are ignored.
decoder = qec.get_decoder("single_error_lut", dem_text)
dem = qec.dem_from_stim_text(dem_text)

print("detectors:", dem.num_detectors())
print("error mechanisms (used by decoder):", dem.num_error_mechanisms())
print("observables:", dem.num_observables())

# Inspection only: show how many columns the matrix would have if '^'
# hints were honored. This does not affect the decoder above.
dem_decomposed = qec.dem_from_stim_text(dem_text, use_decomp_suggestions=True)

print("error mechanisms (if ^ hints honored):",
      dem_decomposed.num_error_mechanisms())

syndromes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
results = decoder.decode_batch(syndromes)
error_predictions = np.array([r.result for r in results], dtype=np.uint8)
observable_predictions = (
    dem.observables_flips_matrix @ error_predictions.T) % 2

for syndrome, error, observable in zip(syndromes, error_predictions,
                                       observable_predictions.T):
    print(f"syndrome {syndrome.tolist()} -> "
          f"error {error.tolist()} -> "
          f"observable flip {observable.tolist()}")
