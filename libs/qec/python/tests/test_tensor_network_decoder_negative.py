# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import numpy as np
import cudaq_qec as qec


# Relevant to Python3.10 only
# Update this once the CI advances to Python3.11
# TODO: Should still perform negative tests for Python>=3.11 which may require modifications of test scripts
@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Only meaningful on Python < 3.11 when dependencies are missing")
def test_tensor_network_decoder_missing_dependencies():
    """Test that when tensor_network_decoder dependencies are missing, 
    we get the proper C++ error message instead of a Python ImportError."""

    # Create a simple test case
    H = np.array([[1, 1, 0], [0, 1, 1]])
    logical = np.array([[1, 0, 1]])
    noise = [0.1, 0.2, 0.3]

    # Test that we get the specific C++ error message when dependencies are missing
    with pytest.raises(RuntimeError) as excinfo:
        qec.get_decoder("tensor_network_decoder",
                        H,
                        logical_obs=logical,
                        noise_model=noise)

    # Verify the error message contains the expected C++ message and installation instructions
    error_msg = str(excinfo.value)
    assert "Decoder 'tensor_network_decoder' is not available" in error_msg and "pip install cudaq-qec[tensor-network-decoder]" in error_msg, f"Unexpected error message: {error_msg}"

    # Verify this is NOT a Python ImportError (which would indicate the graceful failure didn't work)
    assert not isinstance(
        excinfo.value, ImportError
    ), "Should not be an ImportError, should be a RuntimeError from C++"
