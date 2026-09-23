# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest

import numpy as np

torch = pytest.importorskip(
    "torch", reason="torch not installed; skipping TN decoder tests")
import cudaq_qec as qec

if sys.version_info >= (3, 11):
    from quimb.tensor import TensorNetwork

    from cudaq_qec.plugins.decoders.tensor_network_utils.tensor_network_factory import (
        tensor_network_from_parity_check, tensor_network_from_single_syndrome,
        prepare_syndrome_data_batch, tensor_network_from_syndrome_batch,
        tensor_network_from_logical_observable)
    from cudaq_qec.plugins.decoders.tensor_network_utils.contractors import (
        optimize_path, cutn_contractor, ContractorConfig, contractor)
    from cudaq_qec.plugins.decoders.tensor_network_utils.noise_models import factorized_noise_model, error_pairs_noise_model

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11),
                                reason="Requires Python >= 3.11")


def is_nvidia_gpu_available():
    return torch.cuda.is_available()


def make_simple_code():
    # [[1, 1, 0], [0, 1, 1]] parity check, 1 logical, depolarizing noise
    H = np.array([[1, 1, 0], [0, 1, 1]])
    logical = np.array([[1, 0, 1]])
    noise = [0.1, 0.2, 0.3]
    return H, logical, noise


def test_decoder_init_and_attributes():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    assert isinstance(decoder.code_tn, TensorNetwork)
    assert isinstance(decoder.logical_tn, TensorNetwork)
    assert isinstance(decoder.syndrome_tn, TensorNetwork)
    assert isinstance(decoder.full_tn, TensorNetwork)
    assert hasattr(decoder, "noise_model")

    if is_nvidia_gpu_available():
        assert decoder.contractor_config.contractor_name == "cutensornet"
        assert decoder.contractor_config.backend == "numpy"
        assert decoder.contractor_config.device == "cuda"
    else:
        assert decoder.contractor_config.contractor_name == "torch"
        assert decoder.contractor_config.backend == "torch"
        assert decoder.contractor_config.device == "cpu"
    assert decoder._dtype == "float64"


def test_decoder_replace_logical_observable():
    H, logical, noise = make_simple_code()
    import cudaq_qec as qec

    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # New logical observable and indices
    new_logical = np.array([[0, 1, 1]])
    new_logical_inds = ["l_1"]
    new_logical_tags = ["LOG_1"]

    decoder.replace_logical_observable(logical_obs=new_logical,
                                       logical_inds=new_logical_inds,
                                       logical_tags=new_logical_tags)

    # Check that the logical observable and indices are updated
    assert np.array_equal(decoder.logical_obs, new_logical)
    assert decoder.logical_inds == new_logical_inds
    assert decoder.logical_tags == new_logical_tags


def test_decoder_replace_logical_observable_shape_error():
    H, logical, noise = make_simple_code()
    import cudaq_qec as qec

    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # new_logical with wrong shape: first dimension != 1
    new_logical = np.array([[0, 1, 1], [1, 0, 0]])  # shape (2, 3)
    new_logical_inds = ["l_1"]
    new_logical_obs_inds = ["e_0", "e_1", "e_2"]
    new_logical_tags = ["LOG_1"]

    with pytest.raises(Exception):
        decoder.replace_logical_observable(
            logical_obs=new_logical,
            logical_inds=new_logical_inds,
            logical_obs_inds=new_logical_obs_inds,
            logical_tags=new_logical_tags)


def test_decoder_flip_syndromes():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    new_syndromes = [1.0] * H.shape[0]
    decoder.flip_syndromes(new_syndromes)
    for i, t in enumerate(decoder.syndrome_tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, -1.0]))

    new_syndromes = [0.0] * H.shape[0]
    decoder.flip_syndromes(new_syndromes)
    for i, t in enumerate(decoder.syndrome_tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, 1.0]))


def test_decoder_decode_single():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    syndrome = [1.0, 0.0]
    res = decoder.decode(syndrome)
    assert hasattr(res, "converged")
    assert hasattr(res, "result")
    assert isinstance(res.result, np.ndarray)
    assert 0.0 <= res.result[0] <= 1.0


def test_decoder_decode_batch():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    batch = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    res = decoder.decode_batch(batch)
    print(res.result)
    assert isinstance(res, qec.BatchDecoderResult)
    assert isinstance(res.result, np.ndarray)
    assert res.result.shape == (3, 1)
    assert res.converged.shape == (3,)
    assert np.all(res.converged)
    np.testing.assert_allclose(res.result[:, 0], [1.0, 1.0, 0.0],
                               atol=1e-12,
                               rtol=1e-12)


def test_decoder_set_contractor_invalid():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    with pytest.raises(ValueError):
        decoder._set_contractor("not_a_contractor", "cpu", "numpy")
    with pytest.raises(ValueError):
        decoder._set_contractor("numpy", "not_a_device", "numpy")
    with pytest.raises(ValueError):
        decoder._set_contractor("numpy", "cpu", "not_a_backend")


def test_TensorNetworkDecoder_optimize_path_all_variants():
    import cotengra
    from cuquantum import tensornet as cutn
    from opt_einsum.contract import PathInfo
    from cuquantum.tensornet.configuration import OptimizerInfo

    # Simple code setup
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical = np.array([[1, 0, 1]], dtype=np.uint8)
    noise = [0.1, 0.2, 0.3]
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # optimize="auto" (opt_einsum)
    info = decoder.optimize_path(optimize="auto")
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info, PathInfo)

    if not is_nvidia_gpu_available():
        pytest.skip("No GPUs available, skip cuQuantum test.")

    # optimize=cuQuantum OptimizerOptions
    opt = cutn.OptimizerOptions()
    info2 = decoder.optimize_path(optimize=opt)
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info2, OptimizerInfo)

    # optimize=cotengra.HyperOptimizer()
    hyper = cotengra.HyperOptimizer()
    info3 = decoder.optimize_path(optimize=hyper)
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info3, PathInfo)


def test_decoder_batch_vs_single_and_expected_results_with_contractors():
    np.random.seed(42)
    n_checks = 5
    n_errors = 8
    n_logical = 1
    n_batch = 10

    # Generate random binary parity check matrix and logical
    H = np.random.randint(0, 2, size=(n_checks, n_errors)).astype(np.float64)
    logical = np.random.randint(0, 2,
                                size=(n_logical, n_errors)).astype(np.float64)
    noise = np.random.uniform(0.01, 0.2, size=n_errors).tolist()

    import cudaq_qec as qec
    import logging
    from cudaq_qec.plugins.decoders.tensor_network_decoder import TensorNetworkDecoder

    # Provided expected results
    expected = [
        0.9604944927882665, 0.9796816612788876, 0.020709125507417103,
        0.35314051570803995, 0.3616138088105539, 0.01979825044290266,
        0.01979825044290266, 0.6381641010485968, 0.01979825044290266,
        0.3616795232730325
    ]

    contractors = [
        ("numpy", "float64", "cpu", "numpy"),
        ("torch", "float64", "cpu", "torch"),
        ("cutensornet", "float32", "cuda:0", "numpy"),
    ]

    try:
        decoder = qec.get_decoder("tensor_network_decoder",
                                  H,
                                  logical_obs=logical,
                                  noise_model=noise)
        assert isinstance(decoder, TensorNetworkDecoder)
    except Exception as e:
        logging.error(f"Test failed due to: {e}")
        pytest.fail(f"Operation failed: {e}")

    # Generate a batch of random syndromes
    batch = np.random.choice([False, True], size=(n_batch, n_checks))
    batch = batch.astype(np.float64, copy=False)  # Ensure float64 dtype

    for contractor, dtype, device, backend in contractors:
        if "cuda" in device and not is_nvidia_gpu_available():
            # Skip cutensornet tests if no GPU is available
            print(
                f"Skipping contractor {contractor} ({dtype}, {device}): No GPU available."
            )
            continue
        try:
            decoder._set_contractor(contractor, device, backend, dtype=dtype)
        except Exception as e:
            print(f"Skipping contractor {contractor} ({dtype}, {device}): {e}")
            continue

        # Decode each syndrome individually
        single_results = []
        for syndrome in batch:

            try:
                res = decoder.decode(syndrome.tolist())
                assert isinstance(res, qec.DecoderResult)
                assert hasattr(res, "converged")
            except Exception as e:
                logging.error(f"Test failed due to: {e}")
                pytest.fail(f"Operation failed: {e}")
            # Use float32 for float32 contractors, float64 otherwise
            if dtype == "float32":
                single_results.append(np.float32(res.result[0]))
            else:
                single_results.append(np.float64(res.result[0]))

        # Decode the batch
        try:
            res_batch = decoder.decode_batch(batch)
            assert isinstance(res_batch, qec.BatchDecoderResult)
            assert np.all(res_batch.converged)
        except Exception as e:
            logging.error(f"Test failed due to: {e}")
            pytest.fail(f"Operation failed: {e}")

        if dtype == "float32":
            batch_results = res_batch.result[:, 0].astype(np.float32)
            expected_cast = np.array(expected, dtype=np.float32)
            rtol = 1e-5
            atol = 1e-5
        else:
            batch_results = res_batch.result[:, 0].astype(np.float64)
            expected_cast = np.array(expected, dtype=np.float64)
            rtol = 1e-5
            atol = 1e-5

        # Compare single and batch results
        np.testing.assert_allclose(single_results,
                                   batch_results,
                                   rtol=rtol,
                                   atol=atol)

        # Compare single and batch results
        np.testing.assert_allclose(single_results,
                                   expected_cast,
                                   rtol=rtol,
                                   atol=atol)

        # Compare to expected results
        np.testing.assert_allclose(batch_results,
                                   expected_cast,
                                   rtol=rtol,
                                   atol=atol)


def test_tensor_network_from_parity_check_basic():
    mat = np.array([[1, 1, 0], [0, 1, 1]])
    row_inds = ['r0', 'r1']
    col_inds = ['c0', 'c1', 'c2']
    tags = ['tag0', 'tag1', 'tag2', 'tag3']
    tn = tensor_network_from_parity_check(mat, row_inds, col_inds, tags=tags)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 4
    expected_inds = [('r0', 'c0'), ('r0', 'c1'), ('r1', 'c1'), ('r1', 'c2')]
    inds = [t.inds for t in tn.tensors]
    assert set(inds) == set(expected_inds)
    for i, t in enumerate(tn.tensors):
        assert t.tags.pop() in tags
        assert t.inds in expected_inds
        np.testing.assert_array_equal(t.data, np.array([[1.0, 1.0], [1.0,
                                                                     -1.0]]))


def test_tensor_network_from_parity_check_no_tags():
    mat = np.array([[1, 0], [0, 1]])
    row_inds = ['r0', 'r1']
    col_inds = ['c0', 'c1']
    tn = tensor_network_from_parity_check(mat, row_inds, col_inds)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 2
    for t in tn.tensors:
        assert len(t.tags) == 0


def test_tensor_network_from_parity_check_empty():
    mat = np.zeros((2, 2), dtype=int)
    row_inds = ['r0', 'r1']
    col_inds = ['c0', 'c1']
    tn = tensor_network_from_parity_check(mat, row_inds, col_inds)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 0


def test_tensor_network_from_single_syndrome_all_flipped():
    syndrome = [1.0, 1.0, 1.0]
    check_inds = ['c0', 'c1', 'c2']
    tn = tensor_network_from_single_syndrome(syndrome, check_inds)
    assert len(tn.tensors) == 3
    for i, t in enumerate(tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, -1.0]))
        assert t.inds == (check_inds[i],)
        assert f"SYN_{i}" in t.tags
        assert "SYNDROME" in t.tags


def test_tensor_network_from_single_syndrome_mixed():
    syndrome = [0.0, 1.0, 0.0]
    check_inds = ['a', 'b', 'c']
    tn = tensor_network_from_single_syndrome(syndrome, check_inds)
    assert len(tn.tensors) == 3
    for i, t in enumerate(tn.tensors):
        expected = np.array([1.0, -1.0]) if syndrome[i] else np.array(
            [1.0, 1.0])
        np.testing.assert_array_equal(t.data, expected)
        assert t.inds == (check_inds[i],)
        assert f"SYN_{i}" in t.tags
        assert "SYNDROME" in t.tags


def test_prepare_syndrome_data_batch_shape_and_values_randomized():
    np.random.seed(123)
    data = np.random.choice([1.0, 0.0],
                            size=(4, 5))  # syndrome length 4, 5 syndromes
    arr = prepare_syndrome_data_batch(data)
    assert arr.shape == (5, 4, 2)
    # Check that each entry is [1, 1] if False, [1, -1] if True
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            expected = np.array([1, -1]) if data[i, j] else np.array([1, 1])
            np.testing.assert_array_equal(arr[j, i], expected)


def test_tensor_network_from_syndrome_batch_tags_and_inds_randomized():
    np.random.seed(42)
    batch_size = 5
    n_synd = 4
    detection_events = np.random.choice([1.0, 0.0], size=(batch_size, n_synd))
    detection_events = detection_events.astype(np.float32,
                                               copy=False)  # Ensure int8 type
    syndrome_inds = [f's{i}' for i in range(n_synd)]
    tags = [f'tag{i}' for i in range(n_synd)]
    tn = tensor_network_from_syndrome_batch(detection_events,
                                            syndrome_inds,
                                            batch_index="batch",
                                            tags=tags)
    assert len(tn.tensors) == n_synd
    for i, t in enumerate(tn.tensors):
        assert t.inds == (syndrome_inds[i], "batch")
        assert tags[i] in t.tags
        assert "SYNDROME" in t.tags
        # Check tensor data for each batch
        for b in range(batch_size):
            expected = detection_events[b, i] * np.array([1.0, -1.0])
            expected += (1 - detection_events[b, i]) * np.array([1.0, 1.0])
            np.testing.assert_array_equal(t.data[:, b], expected)


def test_tensor_network_from_logical_observable():
    obs = np.array([[0.0, 1.0, 0.0]])
    obs_inds = ['o0']
    tn = tensor_network_from_logical_observable(obs,
                                                obs_inds, ["l0"],
                                                logical_tags=["OBS_0"])
    assert len(tn.tensors) == obs.shape[0]
    for i, t in enumerate(tn.tensors):
        expected = np.array([[1.0, 1.0], [1.0, -1.0]])
        np.testing.assert_array_equal(t.data, expected)
        assert t.inds == (obs_inds[i], "l0")
        assert f"OBS_{i}" in t.tags


def test_optimize_path_numpy_variants():
    from quimb.tensor import TensorNetwork, Tensor
    from cuquantum import tensornet as cutn
    from opt_einsum.contract import PathInfo

    tn = TensorNetwork([
        Tensor(np.ones((2, 2)), inds=("a", "b")),
        Tensor(np.ones((2, 2)), inds=("b", "c")),
        Tensor(np.ones((2, 2)), inds=("c", "a")),
    ])

    # Case 1: optimize="auto"
    path, info = optimize_path("auto", output_inds=("a",), tn=tn)
    assert isinstance(path, (list, tuple))
    assert isinstance(info, PathInfo)

    if not is_nvidia_gpu_available():
        pytest.skip("No GPUs available, skip cuQuantum test.")

    # Case 2: optimize=None (should use cuQuantum path finder)
    path2, info2 = optimize_path(None, output_inds=("a",), tn=tn)
    assert path2 is not None
    from cuquantum.tensornet.configuration import OptimizerInfo
    assert isinstance(info2, OptimizerInfo)

    # Case 3: optimize=OptimizerOptions
    opt = cutn.OptimizerOptions()
    path3, info3 = optimize_path(opt, output_inds=("a",), tn=tn)
    assert path3 is not None
    assert isinstance(info3, OptimizerInfo)


def test_factorized_noise_model_basic():
    error_indices = ['e0', 'e1', 'e2']
    error_probabilities = [0.1, 0.5, 0.9]
    tags = ['tag0', 'tag1', 'tag2']
    tn = factorized_noise_model(error_indices,
                                error_probabilities,
                                tensors_tags=tags)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 3
    for i, t in enumerate(tn.tensors):
        np.testing.assert_array_equal(
            t.data,
            np.array([1.0 - error_probabilities[i], error_probabilities[i]]))
        assert t.inds == (error_indices[i],)
        assert tags[i] in t.tags


def test_factorized_noise_model_default_tags():
    error_indices = ['e0', 'e1']
    error_probabilities = [0.2, 0.8]
    tn = factorized_noise_model(error_indices, error_probabilities)
    for t in tn.tensors:
        assert "NOISE" in t.tags


def test_error_pairs_noise_model_basic():
    error_index_pairs = [('e0', 'e1'), ('e2', 'e3')]
    error_probabilities = [
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[0.7, 0.3], [0.4, 0.6]])
    ]
    tags = ['tagA', 'tagB']
    tn = error_pairs_noise_model(error_index_pairs,
                                 error_probabilities,
                                 tensors_tags=tags)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 2
    for i, t in enumerate(tn.tensors):
        np.testing.assert_array_equal(t.data, error_probabilities[i])
        assert t.inds == error_index_pairs[i]
        assert tags[i] in t.tags


def test_error_pairs_noise_model_default_tags():
    error_index_pairs = [('x', 'y')]
    error_probabilities = [np.array([[0.6, 0.5], [0.3, 0.7]])]
    tn = error_pairs_noise_model(error_index_pairs, error_probabilities)
    for t in tn.tensors:
        assert "NOISE" in t.tags


def test_valid_numpy_cpu():
    cfg = ContractorConfig("numpy", "numpy", "cpu")
    assert cfg.contractor_name == "numpy"
    assert cfg.backend == "numpy"
    assert cfg.device == "cpu"
    assert cfg.device_id == 0
    assert cfg.contractor is contractor


def test_valid_torch_cpu():
    cfg = ContractorConfig("torch", "torch", "cpu")
    assert cfg.contractor_name == "torch"
    assert cfg.backend == "torch"
    assert cfg.device == "cpu"
    assert cfg.device_id == 0
    assert cfg.contractor is contractor


def test_valid_cutensornet_numpy_cuda():
    cfg = ContractorConfig("cutensornet", "numpy", "cuda")
    assert cfg.contractor_name == "cutensornet"
    assert cfg.backend == "numpy"
    assert cfg.device == "cuda"
    assert cfg.device_id == 0
    assert cfg.contractor is cutn_contractor


def test_valid_cutensornet_torch_cuda():
    cfg = ContractorConfig("cutensornet", "torch", "cuda")
    assert cfg.contractor_name == "cutensornet"
    assert cfg.backend == "torch"
    assert cfg.device == "cuda"
    assert cfg.device_id == 0
    assert cfg.contractor is cutn_contractor


def test_cuda_device_id_parsing():
    cfg = ContractorConfig("cutensornet", "torch", "cuda:3")
    assert cfg.device == "cuda:3"
    assert cfg.device_id == 3


def test_invalid_contractor_name():
    with pytest.raises(ValueError):
        ContractorConfig("invalid", "numpy", "cpu")


def test_invalid_backend():
    with pytest.raises(ValueError):
        ContractorConfig("numpy", "invalid", "cpu")


def test_invalid_combo():
    with pytest.raises(ValueError):
        ContractorConfig("torch", "numpy", "cpu")


# ---------------------------------------------------------------------------
# Focused tests for _safe_posterior / mass-validation logic, using a mocked
# contractor so these run without a GPU and without a real contraction.
# ContractorConfig is a frozen dataclass; its contractor property reads from
# the class-level _contractors dict, which is the correct mock surface.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_decoder(monkeypatch):
    """Fixture that returns a factory for decoders with a mocked contractor.

    Usage:
        decoder = mock_decoder(np.array([m0, m1]))           # single decode
        decoder = mock_decoder(np.array([[...],...]), rows=N) # batch decode
    The factory patches ContractorConfig._contractors for the active backend
    and pre-sets path_single/path_batch so no path optimisation is triggered.
    """

    def _make(mock_output, rows=None, dtype="float64"):
        H, logical, noise = make_simple_code()
        decoder = qec.get_decoder("tensor_network_decoder",
                                  H,
                                  logical_obs=logical,
                                  noise_model=noise,
                                  dtype=dtype)
        decoder.path_single = "auto"
        decoder.path_batch = "auto"
        if rows is not None:
            decoder._batch_size = rows
        name = decoder.contractor_config.contractor_name
        monkeypatch.setitem(ContractorConfig._contractors, name,
                            lambda *a, **kw: mock_output)
        return decoder

    return _make


def test_default_dtype_is_float64():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    assert decoder._dtype == "float64"


_BIG = np.finfo(np.float64).max


@pytest.mark.parametrize(
    "masses",
    [
        [-0.1, 1.1],  # negative mass
        [0.0, 0.0],  # zero denominator
        [float("nan"), 0.5],  # NaN
        [float("inf"), 0.5],  # infinite mass
        [_BIG, _BIG],  # finite masses whose sum overflows to inf
    ])
def test_decode_invalid_masses_marks_unconverged(mock_decoder, masses):
    res = mock_decoder(np.array(masses)).decode([0.0, 0.0])
    assert not res.converged
    assert res.result[0] == pytest.approx(0.5)


def test_decode_valid_masses_converged(mock_decoder):
    res = mock_decoder(np.array([0.3, 0.7])).decode([0.0, 0.0])
    assert res.converged
    assert res.result[0] == pytest.approx(0.7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_decode_roundoff_negative_mass_converged(mock_decoder, dtype):
    eps = np.finfo(np.dtype(dtype)).eps
    res = mock_decoder(np.array([-eps, 1.0]), dtype=dtype).decode([0.0, 0.0])
    assert res.converged
    assert res.result[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_row",
    [
        [-0.1, 1.1],  # negative mass
        [float("nan"), 0.5],  # NaN
        [0.0, 0.0],  # zero denominator
    ])
def test_decode_batch_invalid_row_marks_unconverged(mock_decoder, bad_row):
    mock = np.array([[0.3, 0.7], bad_row])
    res = mock_decoder(mock, rows=2).decode_batch(np.zeros((2, 2)))
    assert res.converged[0]
    assert not res.converged[1]
    assert res.result[0, 0] == pytest.approx(0.7)
    assert res.result[1, 0] == pytest.approx(0.5)


def test_decode_batch_all_valid_all_converged(mock_decoder):
    mock = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    res = mock_decoder(mock, rows=3).decode_batch(np.zeros((3, 2)))
    assert np.all(res.converged)
    np.testing.assert_allclose(res.result[:, 0], [0.8, 0.4, 0.9])


def test_error_pairs_noise_model_ndarray_batch():
    # Ndarray batch must be (N, 2, 2): one 2x2 joint-probability matrix per pair.
    error_index_pairs = [('e0', 'e1'), ('e2', 'e3')]
    error_probabilities = np.array([[[0.9, 0.1], [0.2, 0.8]],
                                    [[0.7, 0.3], [0.4, 0.6]]])
    tn = error_pairs_noise_model(error_index_pairs, error_probabilities)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 2
    for i, t in enumerate(tn.tensors):
        # Each slice is a 2-index tensor, not a length-2 vector.
        np.testing.assert_array_equal(t.data, error_probabilities[i])
        assert t.inds == error_index_pairs[i]
        assert "NOISE" in t.tags


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.9, 0.1], [0.7, 0.3]]),  # (N, 2): one vector per pair
        np.ones((2, 2, 3)),  # trailing size is not 2x2
        np.ones((2, 3, 2)),  # middle size is not 2
        np.ones((2, 2, 2, 2)),  # extra dimension
    ])
def test_error_pairs_noise_model_ndarray_rejects_bad_shapes(bad):
    # Unusable ndarray ranks/shapes must fail in this helper, not later in Quimb.
    error_index_pairs = [('e0', 'e1'), ('e2', 'e3')]
    with pytest.raises(AssertionError, match=r"\(N, 2, 2\)"):
        error_pairs_noise_model(error_index_pairs, bad)


def test_factorized_noise_model_rejects_2d_ndarray():
    # A 2-D batch whose first axis matches the indices is still the wrong rank.
    error_indices = ['e0', 'e1', 'e2']
    bad = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    with pytest.raises(AssertionError, match="1D array"):
        factorized_noise_model(error_indices, bad)


def test_factorized_noise_model_rejects_tuple():
    with pytest.raises(TypeError, match="list or numpy array"):
        factorized_noise_model(['e0', 'e1'], (0.1, 0.2))


def test_error_pairs_noise_model_rejects_tuple():
    pairs = [('e0', 'e1')]
    mats = (np.array([[0.9, 0.1], [0.2, 0.8]]),)
    with pytest.raises(TypeError, match="list or numpy array"):
        error_pairs_noise_model(pairs, mats)


def test_decoder_cpu_fallback_custom_inds_and_tn_noise(monkeypatch):
    # Without CUDA, construction uses the Torch/CPU contractor; custom check
    # and error names are stored, and a TensorNetwork noise model is reindexed
    # onto those error names.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    H, logical, noise = make_simple_code()
    check_inds = ["chk_a", "chk_b"]
    error_inds = ["err_x", "err_y", "err_z"]
    tn = factorized_noise_model(["n0", "n1", "n2"], noise)
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=tn,
                              check_inds=check_inds,
                              error_inds=error_inds)
    assert decoder.contractor_config.contractor_name == "torch"
    assert decoder.contractor_config.backend == "torch"
    assert decoder.contractor_config.device == "cpu"
    assert decoder.check_inds == check_inds
    assert decoder.error_inds == error_inds
    assert list(decoder.noise_model.outer_inds()) == error_inds


@pytest.mark.parametrize("ind_name, inds, n_got", [
    ("check_inds", ["s0"], 1),
    ("check_inds", ["s0", "s1", "s2"], 3),
    ("error_inds", ["e0", "e1"], 2),
    ("error_inds", ["e0", "e1", "e2", "e3"], 4),
])
def test_decoder_custom_inds_length(ind_name, inds, n_got):
    H, logical, noise = make_simple_code()
    n_exp = H.shape[0] if ind_name == "check_inds" else H.shape[1]
    with pytest.raises(
            AssertionError,
            match=rf"{ind_name} must have length {n_exp}, but got {n_got}"):
        qec.get_decoder("tensor_network_decoder",
                        H,
                        logical_obs=logical,
                        noise_model=noise,
                        **{ind_name: inds})


def test_decoder_noise_tn_open_index_count():
    # A TensorNetwork noise model must expose one open index per error.
    H, logical, _ = make_simple_code()
    tn = factorized_noise_model(["n0", "n1"], [0.1, 0.2])
    with pytest.raises(AssertionError,
                       match=r"Noise model has 2 open indices, but expected 3"):
        qec.get_decoder("tensor_network_decoder",
                        H,
                        logical_obs=logical,
                        noise_model=tn)


if __name__ == "__main__":
    pytest.main()
