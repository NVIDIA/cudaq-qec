# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cudaq import qvector
from dataclasses import dataclass


@dataclass(slots=True)
class patch:
    """A logical qubit patch representation for surface code quantum error correction.
    
    This class represents a logical qubit encoded in a 2D patch, containing
    data qubits and both X and Z ancilla qubits arranged in a 2D lattice pattern.
    The patch structure is fundamental to implementing quantum error correction
    and fault-tolerant quantum computation in CUDA-Q.
    
    Attributes
    ----------
    data : qvector
        The collection of data qubits that encode the logical qubit state.
        These qubits store the actual quantum information being protected.
        
    ancx : qvector
        The X-basis ancilla qubits used for syndrome measurement.
        These qubits are used to detect and correct bit-flip (X) errors
        on the data qubits through stabilizer measurements.
        
    ancz : qvector
        The Z-basis ancilla qubits used for syndrome measurement.
        These qubits are used to detect and correct phase-flip (Z) errors
        on the data qubits through stabilizer measurements.
        
    Notes
    -----
    The patch layout follows the standard surface code arrangement where:
    - Data qubits are placed at the vertices
    - X ancillas are placed on horizontal edges
    - Z ancillas are placed on vertical edges
    
    This structure enables the implementation of weight-4 stabilizer
    measurements required for surface code error correction.
    """
    data: qvector
    ancx: qvector
    ancz: qvector
