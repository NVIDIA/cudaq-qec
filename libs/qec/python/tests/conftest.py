# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-Q is an optional extra of cudaq-qec, so the suite runs in two modes.

With CUDA-Q (the default when `cudaq` is installed) everything runs. Without
it, or with CUDAQ_QEC_DISABLE_CUDAQ=1, the modules below are not collected
and tests marked `requires_cudaq` are skipped; what remains is the
CUDA-Q-free surface of the package (decoders, decoding config, task graphs,
DEM construction) and must pass with no `cudaq` in the process.
"""
import pytest

import cudaq_qec as qec

# Test modules that need CUDA-Q throughout: they import cudaq at module level
# or build every fixture from a cudaq_qec code object.
_CUDAQ_ONLY_MODULES = [
    "test_code.py",
    "test_dem.py",
    "test_dem_sampling.py",
    "test_device_call.py",
    "test_playback_emulator.py",
    "test_realtime_decoding_negative.py",
    "test_sliding_window.py",
    "test_surface_code.py",
]

if not qec.has_cudaq():
    collect_ignore = list(_CUDAQ_ONLY_MODULES)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_cudaq: the test needs CUDA-Q (cudaq_qec.has_cudaq()); "
        "skipped when the package runs without it")


def pytest_collection_modifyitems(config, items):
    if qec.has_cudaq():
        return
    skip = pytest.mark.skip(
        reason="requires CUDA-Q (pip install cudaq-qec[cudaq])")
    for item in items:
        if item.get_closest_marker("requires_cudaq"):
            item.add_marker(skip)
