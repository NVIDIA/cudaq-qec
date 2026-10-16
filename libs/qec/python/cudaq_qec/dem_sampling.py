# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
"""DEM sampling via the C++ pybind11 binding (GPU with cuStabilizer + CPU fallback).

Public API:
    dem_sampling(
        check_matrix, num_shots, error_probabilities, seed=None, backend="auto"
    )

The GPU path uses cuStabilizer for accelerated sampling. The CPU path is
always available as a fallback.

Inputs are NumPy arrays (primary) or PyTorch CUDA tensors (requires
user-installed torch via ``pip install torch``). PyTorch CPU tensors are
moved to CUDA under ``backend="gpu"`` but rejected under ``"auto"``/``"cpu"``;
convert them to NumPy arrays first for the CPU path.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

__all__ = ["dem_sampling"]


def dem_sampling(
    check_matrix,
    num_shots: int,
    error_probabilities,
    seed: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[object, object]:
    """Sample errors and syndromes from a Detector Error Model.

    Args:
        check_matrix: Binary matrix [num_checks x num_error_mechanisms],
            as a NumPy uint8 array or PyTorch CUDA tensor.
        num_shots: Number of independent Monte-Carlo shots.
        error_probabilities: 1-D array of length num_error_mechanisms with
            independent Bernoulli probabilities for each mechanism.
            Accepts NumPy float64 array or PyTorch CUDA tensor.
        seed: Optional RNG seed for reproducibility.
        backend: Backend selection policy:
            - "auto" (default): try GPU, fall back to CPU.
            - "cpu": force CPU implementation.
            - "gpu": force GPU implementation and raise if unavailable.

    Returns:
        (syndromes, errors) where
          syndromes: uint8 array/tensor [num_shots x num_checks]
          errors:    uint8 array/tensor [num_shots x num_error_mechanisms]

    The GPU path uses cuStabilizer for accelerated sampling. When PyTorch
    CUDA tensors are provided, outputs are returned as CUDA tensors.
    Otherwise, outputs are NumPy arrays.

    PyTorch CPU tensors are moved to CUDA under ``backend="gpu"`` (raising a
    GPU-unavailable error if there is no GPU) but rejected under
    ``"auto"``/``"cpu"``; convert to NumPy first for the CPU path.
    PyTorch is an optional dependency; install with ``pip install torch``.
    """
    for obj in (check_matrix, error_probabilities):
        if hasattr(obj, 'data_ptr'):
            try:
                import torch  # noqa: F401
            except ImportError:
                warnings.warn(
                    "Input appears to be a PyTorch tensor but torch is not "
                    "installed. Install it with: pip install torch",
                    stacklevel=2,
                )
            break

    from . import _pycudaqx_qec_the_suffix_matters_cudaq_qec as _qecmod

    return _qecmod.qecrt.dem_sampling(check_matrix, num_shots,
                                      error_probabilities, seed, backend)
