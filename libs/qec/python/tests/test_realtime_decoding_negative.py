# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq
import cudaq_qec as qec


def _max_syndromes_from_sparse(sparse):
    max_col = max((val for val in sparse if val >= 0), default=-1)
    return max_col + 1


def _make_decoder_config():
    config = qec.decoder_config()
    config.id = 0
    config.type = "single_error_lut"
    config.block_size = 2
    config.syndrome_size = 1

    H = np.zeros((config.syndrome_size, config.block_size), dtype=np.uint8)
    O = np.zeros((1, config.block_size), dtype=np.uint8)
    config.H_sparse = qec.pcm_to_sparse_vec(H)
    config.O_sparse = qec.pcm_to_sparse_vec(O)
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)
    return config


@pytest.fixture
def configured_decoder():
    cudaq.reset_target()
    multi_config = qec.multi_decoder_config()
    config = _make_decoder_config()
    multi_config.decoders = [config]
    status = qec.configure_decoders(multi_config)
    assert status == 0
    try:
        yield config
    finally:
        qec.finalize_decoders()


# qec.enqueue_syndromes uses conditional feedback so we must use cudaq.run, and
# cudaq.run requires that kernels return non-void values.
@cudaq.kernel
def enqueue_kernel(decoder_id: int, num_syndromes: int) -> int:
    q = cudaq.qvector(num_syndromes)
    syndromes = mz(q)
    qec.enqueue_syndromes(decoder_id, syndromes, 0)
    return 0


@cudaq.kernel
def get_corrections_kernel(return_size: int):
    q = cudaq.qvector(1)
    mz(q)
    qec.get_corrections(0, return_size, False)


def test_get_corrections_rejects_bad_return_size(configured_decoder):
    # User-facing kernel call: invalid return sizes should raise.
    with pytest.raises(ValueError, match="correction_length must be greater"):
        cudaq.sample(get_corrections_kernel, 0, shots_count=1)
    with pytest.raises(ValueError,
                       match="does not match number of observables"):
        cudaq.sample(get_corrections_kernel, 2, shots_count=1)


def test_enqueue_syndromes_rejects_bad_args(configured_decoder):
    # User-facing kernel call: invalid decoder_id and oversize syndromes.
    max_syndromes = _max_syndromes_from_sparse(configured_decoder.D_sparse)
    with pytest.raises(ValueError, match="Decoder 1 not found"):
        cudaq.run(enqueue_kernel, 1, max_syndromes, shots_count=1)
    with pytest.raises(ValueError,
                       match="exceeds configured measurement count"):
        cudaq.run(enqueue_kernel, 0, max_syndromes + 1, shots_count=1)


def test_configure_decoders_rejects_invalid_o_sparse():
    # User-facing config API: invalid O_sparse indices should fail to configure.
    multi_config = qec.multi_decoder_config()
    config = _make_decoder_config()
    config.O_sparse = [config.block_size + 1, -1]
    multi_config.decoders = [config]
    try:
        with pytest.raises(RuntimeError, match="out of range"):
            qec.configure_decoders(multi_config)
    finally:
        qec.finalize_decoders()
