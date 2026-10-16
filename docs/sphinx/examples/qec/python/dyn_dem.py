# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# [Begin Documentation]
import cudaq_qec as qec

# Build a T-round DEM from a CSS code and phenomenological noise — no Stim
# circuit required.
code = qec.get_code("repetition", distance=3)
noise = qec.CssNoise()
noise.px = 0.01
noise.pm = 0.005

num_rounds = 5
flat = qec.dem_from_css_matrices(code, noise, num_rounds)
print(f"flat DEM: {flat.num_detectors()} detectors, "
      f"{flat.num_error_mechanisms()} faults, "
      f"{flat.num_observables()} observables")

# The same experiment as composable one-round chunks. Stitch-and-close (or
# dem_close_all) recovers the flat DEM, which is useful when rounds are
# streamed or when init/bulk/final phases differ.
matrices = qec.css_matrices_from_code(code)
chunk = qec.extended_dem_from_css_matrices(matrices, noise)
chunks = [chunk] * num_rounds
closed = qec.dem_close_all(chunks)

assert closed.num_detectors() == flat.num_detectors()
assert closed.num_error_mechanisms() == flat.num_error_mechanisms()
print("dem_close_all matches dem_from_css_matrices")

# Streaming helpers: detector→round map and D_sparse for realtime configs.
detector_round = qec.dem_chunks_to_detector_round(chunks)
d_sparse = qec.dem_chunks_to_d_sparse(chunks)
print(f"detector_round length {len(detector_round)}, "
      f"D_sparse rows {len(d_sparse)}")
# [End Documentation]
