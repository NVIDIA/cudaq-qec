# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cudaq_qec.dem_sampling (GPU with cuStabilizer + CPU fallback)."""

import numpy as np
import pytest

from cudaq_qec.dem_sampling import dem_sampling


def _compute_syndrome(H, errors):
    """Reference: syndromes = errors @ H^T mod 2."""
    return (errors @ H.T) % 2


def _has_runtime_gpu_backend():
    """True only when GPU backend is usable at runtime."""
    H = np.array([[1]], dtype=np.uint8)
    probs = np.array([0.0], dtype=np.float64)
    try:
        dem_sampling(H, 1, probs, seed=0, backend="gpu")
        return True
    except RuntimeError:
        return False


_HAS_RUNTIME_GPU_BACKEND = _has_runtime_gpu_backend()

# ---------------------------------------------------------------------------
# Deterministic tests (numpy input)
# ---------------------------------------------------------------------------


class TestAllZeroProbabilities:

    def test_all_zeros(self):
        H = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
        probs = np.zeros(4)
        syndromes, errors = dem_sampling(H, 10, probs, seed=0)

        assert syndromes.shape == (10, 3)
        assert errors.shape == (10, 4)
        np.testing.assert_array_equal(errors, 0)
        np.testing.assert_array_equal(syndromes, 0)


class TestAllOneProbabilities:

    def test_all_ones(self):
        H = np.array([[1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
        probs = np.ones(4)
        syndromes, errors = dem_sampling(H, 5, probs, seed=0)

        np.testing.assert_array_equal(errors, 1)
        for shot in range(5):
            assert syndromes[shot, 0] == 0
            assert syndromes[shot, 1] == 1
            assert syndromes[shot, 2] == 0


class TestMixedDeterministicProbs:

    def test_mixed(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0])
        syndromes, errors = dem_sampling(H, 8, probs, seed=42)

        expected_errors = np.array([1, 0, 1], dtype=np.uint8)
        expected_syn = np.array([0, 1], dtype=np.uint8)

        for shot in range(8):
            np.testing.assert_array_equal(errors[shot], expected_errors)
            np.testing.assert_array_equal(syndromes[shot], expected_syn)


class TestIdentityMatrix:

    def test_identity_all_ones(self):
        H = np.eye(5, dtype=np.uint8)
        probs = np.ones(5)
        syndromes, errors = dem_sampling(H, 3, probs, seed=7)

        np.testing.assert_array_equal(errors, 1)
        np.testing.assert_array_equal(syndromes, errors)

    def test_identity_syndrome_equals_errors(self):
        H = np.eye(10, dtype=np.uint8)
        probs = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        syndromes, errors = dem_sampling(H, 4, probs, seed=0)

        np.testing.assert_array_equal(syndromes, errors)
        expected = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        for shot in range(4):
            np.testing.assert_array_equal(errors[shot], expected)


class TestAllOnesMatrix:

    def test_even_columns(self):
        H = np.ones((3, 4), dtype=np.uint8)
        probs = np.ones(4)
        syndromes, errors = dem_sampling(H, 4, probs, seed=0)

        np.testing.assert_array_equal(errors, 1)
        np.testing.assert_array_equal(syndromes, 0)

    def test_odd_columns(self):
        H = np.ones((3, 3), dtype=np.uint8)
        probs = np.ones(3)
        syndromes, errors = dem_sampling(H, 4, probs, seed=0)

        np.testing.assert_array_equal(errors, 1)
        np.testing.assert_array_equal(syndromes, 1)


class TestSingleColumnMatrix:

    def test_single_column(self):
        H = np.array([[1], [0], [1]], dtype=np.uint8)
        probs = np.array([1.0])
        syndromes, errors = dem_sampling(H, 6, probs, seed=0)

        np.testing.assert_array_equal(errors, 1)
        expected_syn = np.array([1, 0, 1], dtype=np.uint8)
        for shot in range(6):
            np.testing.assert_array_equal(syndromes[shot], expected_syn)


class TestSingleRowMatrix:

    def test_single_row(self):
        H = np.array([[1, 1, 0, 1, 0]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        syndromes, errors = dem_sampling(H, 4, probs, seed=0)

        expected_errors = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        for shot in range(4):
            np.testing.assert_array_equal(errors[shot], expected_errors)
            assert syndromes[shot, 0] == 1


class TestSingleShot:

    def test_single_shot(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([1.0, 1.0, 0.0])
        syndromes, errors = dem_sampling(H, 1, probs, seed=0)

        assert errors.shape == (1, 3)
        np.testing.assert_array_equal(errors[0], [1, 1, 0])
        np.testing.assert_array_equal(syndromes[0], [0, 1])


class TestRepetitionCode:

    def test_single_error(self):
        H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
        probs = np.array([0.0, 1.0, 0.0, 0.0])
        syndromes, errors = dem_sampling(H, 3, probs, seed=0)

        expected_errors = np.array([0, 1, 0, 0], dtype=np.uint8)
        expected_syn = np.array([1, 1, 0], dtype=np.uint8)

        for shot in range(3):
            np.testing.assert_array_equal(errors[shot], expected_errors)
            np.testing.assert_array_equal(syndromes[shot], expected_syn)


class TestSyndromeConsistency:

    def test_fixed_matrix(self):
        H = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
        ],
                     dtype=np.uint8)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        syndromes, errors = dem_sampling(H, 500, probs, seed=7)

        expected = _compute_syndrome(H, errors)
        np.testing.assert_array_equal(syndromes, expected)


class TestSeedReproducibility:

    def test_same_seed(self):
        H = np.eye(5, dtype=np.uint8)
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        s1, e1 = dem_sampling(H, 100, probs, seed=42)
        s2, e2 = dem_sampling(H, 100, probs, seed=42)

        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(e1, e2)

    def test_different_seeds_differ(self):
        H = np.eye(5, dtype=np.uint8)
        probs = np.full(5, 0.5)

        _, e1 = dem_sampling(H, 1000, probs, seed=1)
        _, e2 = dem_sampling(H, 1000, probs, seed=2)

        assert not np.array_equal(e1, e2)


class TestOutputProperties:

    def test_shapes_and_dtypes(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([0.1, 0.2, 0.3])
        syndromes, errors = dem_sampling(H, 10, probs, seed=0)

        assert syndromes.shape == (10, 2)
        assert errors.shape == (10, 3)
        assert syndromes.dtype == np.uint8
        assert errors.dtype == np.uint8

    def test_binary_output(self):
        H = np.eye(5, dtype=np.uint8)
        probs = np.full(5, 0.5)
        syndromes, errors = dem_sampling(H, 100, probs, seed=42)

        assert set(np.unique(syndromes)).issubset({0, 1})
        assert set(np.unique(errors)).issubset({0, 1})


class TestInputValidation:

    def test_non_2d_matrix(self):
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(np.zeros(5, dtype=np.uint8), 10, np.zeros(5))

    def test_mismatched_probs(self):
        H = np.eye(5, dtype=np.uint8)
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, np.zeros(3))

    def test_non_1d_probs(self):
        H = np.eye(5, dtype=np.uint8)
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, np.zeros((5, 1)))

    def test_probability_below_zero(self):
        H = np.eye(3, dtype=np.uint8)
        probs = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, probs, backend="cpu")

    def test_probability_above_one(self):
        H = np.eye(3, dtype=np.uint8)
        probs = np.array([0.1, 1.2, 0.3], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, probs, backend="cpu")

    def test_probability_nan(self):
        H = np.eye(3, dtype=np.uint8)
        probs = np.array([0.1, np.nan, 0.3], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, probs, backend="cpu")

    @pytest.mark.skipif(not _HAS_RUNTIME_GPU_BACKEND,
                        reason="GPU backend unavailable in this environment")
    def test_force_gpu_rejects_invalid_probability_numpy(self):
        H = np.eye(3, dtype=np.uint8)
        probs = np.array([0.1, 1.1, 0.3], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, probs, backend="gpu")

    @pytest.mark.skipif(not _HAS_RUNTIME_GPU_BACKEND,
                        reason="GPU backend unavailable in this environment")
    def test_force_gpu_rejects_invalid_probability_torch_cuda(self):
        torch_mod = pytest.importorskip("torch")
        if not torch_mod.cuda.is_available():
            pytest.skip("PyTorch CUDA unavailable")

        H = torch_mod.eye(3, dtype=torch_mod.uint8, device="cuda")
        probs = torch_mod.tensor([0.1, 1.1, 0.3],
                                 dtype=torch_mod.float64,
                                 device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            dem_sampling(H, 10, probs, backend="gpu")


class TestBackendSelection:

    def test_invalid_backend(self):
        H = np.eye(2, dtype=np.uint8)
        probs = np.array([0.1, 0.2], dtype=np.float64)
        with pytest.raises(RuntimeError):
            dem_sampling(H, 4, probs, seed=1, backend="not-a-backend")

    def test_force_cpu_backend(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        syndromes, errors = dem_sampling(H, 5, probs, seed=11, backend="cpu")

        expected_errors = np.array([1, 0, 1], dtype=np.uint8)
        expected_syn = np.array([0, 1], dtype=np.uint8)
        for shot in range(5):
            np.testing.assert_array_equal(errors[shot], expected_errors)
            np.testing.assert_array_equal(syndromes[shot], expected_syn)

    @pytest.mark.skipif(not _HAS_RUNTIME_GPU_BACKEND,
                        reason="GPU backend unavailable in this environment")
    def test_force_gpu_backend(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        syndromes, errors = dem_sampling(H, 5, probs, seed=11, backend="gpu")

        expected_errors = np.array([1, 0, 1], dtype=np.uint8)
        expected_syn = np.array([0, 1], dtype=np.uint8)
        for shot in range(5):
            np.testing.assert_array_equal(errors[shot], expected_errors)
            np.testing.assert_array_equal(syndromes[shot], expected_syn)


# ---------------------------------------------------------------------------
# PyTorch tensor input tests
# ---------------------------------------------------------------------------

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchInput:

    def test_cpu_tensor_rejected(self):
        """PyTorch CPU tensors should be rejected (no silent conversion)."""
        H = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
        probs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        with pytest.raises((RuntimeError, TypeError)):
            dem_sampling(H, 5, probs, seed=42)

    def test_cpu_tensor_with_cpu_backend_rejected(self):
        """PyTorch CPU tensors with backend='cpu' should also be rejected."""
        H = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
        probs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        with pytest.raises((RuntimeError, TypeError)):
            dem_sampling(H, 4, probs, seed=7, backend="cpu")

    @pytest.mark.skipif(not _HAS_RUNTIME_GPU_BACKEND,
                        reason="GPU backend unavailable in this environment")
    def test_force_gpu_backend(self):
        H = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
        probs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        syndromes, errors = dem_sampling(H, 4, probs, seed=7, backend="gpu")

        assert isinstance(syndromes, torch.Tensor)
        assert isinstance(errors, torch.Tensor)
        assert syndromes.is_cuda
        assert errors.is_cuda
        assert syndromes.data_ptr() > 0
        assert errors.data_ptr() > 0

        expected_errors = np.array([1, 0, 1], dtype=np.uint8)
        expected_syn = np.array([0, 1], dtype=np.uint8)
        for shot in range(4):
            np.testing.assert_array_equal(errors[shot].cpu().numpy(),
                                          expected_errors)
            np.testing.assert_array_equal(syndromes[shot].cpu().numpy(),
                                          expected_syn)

    @pytest.mark.skipif(not (_HAS_TORCH and torch.cuda.is_available()),
                        reason="CUDA not available for PyTorch")
    def test_cuda_tensors(self):
        """CUDA tensors should stay on device in torch GPU path."""
        H = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8).cuda()
        probs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64).cuda()
        syndromes, errors = dem_sampling(H, 4, probs, seed=7)

        assert isinstance(syndromes, torch.Tensor)
        assert isinstance(errors, torch.Tensor)
        assert syndromes.is_cuda
        assert errors.is_cuda
        assert tuple(syndromes.shape) == (4, 2)
        assert tuple(errors.shape) == (4, 3)
        assert syndromes.data_ptr() > 0
        assert errors.data_ptr() > 0

        expected_errors = np.array([1, 0, 1], dtype=np.uint8)
        expected_syn = np.array([0, 1], dtype=np.uint8)
        for shot in range(4):
            np.testing.assert_array_equal(errors[shot].cpu().numpy(),
                                          expected_errors)
            np.testing.assert_array_equal(syndromes[shot].cpu().numpy(),
                                          expected_syn)

    @pytest.mark.skipif(
        not (_HAS_RUNTIME_GPU_BACKEND and _HAS_TORCH and
             torch.cuda.is_available()),
        reason="GPU backend unavailable for CUDA stream test",
    )
    def test_non_default_cuda_stream(self):
        """GPU path must honor torch's current (non-default) CUDA stream."""
        device = torch.device("cuda")
        H = torch.tensor([[1, 0, 1], [0, 1, 1]],
                         dtype=torch.uint8,
                         device=device)
        probs = torch.tensor([1.0, 0.0, 1.0],
                             dtype=torch.float64,
                             device=device)
        stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream):
            syndromes, errors = dem_sampling(H, 4, probs, seed=7, backend="gpu")
            assert isinstance(syndromes, torch.Tensor)
            assert isinstance(errors, torch.Tensor)
            assert syndromes.device.type == "cuda"
            assert errors.device.type == "cuda"

            expected_errors = torch.tensor([1, 0, 1],
                                           dtype=torch.uint8,
                                           device=device)
            expected_syn = torch.tensor([0, 1],
                                        dtype=torch.uint8,
                                        device=device)
            for shot in range(4):
                assert torch.equal(errors[shot], expected_errors)
                assert torch.equal(syndromes[shot], expected_syn)

        stream.synchronize()

    @pytest.mark.skipif(
        not (_HAS_RUNTIME_GPU_BACKEND and _HAS_TORCH and
             torch.cuda.device_count() > 1),
        reason="Requires runtime GPU backend and at least two CUDA devices",
    )
    def test_multi_gpu_device_routing(self):
        """GPU path should execute on and return tensors from the input device."""
        device = torch.device("cuda:1")
        H = torch.tensor([[1, 0, 1], [0, 1, 1]],
                         dtype=torch.uint8,
                         device=device)
        probs = torch.tensor([1.0, 0.0, 1.0],
                             dtype=torch.float64,
                             device=device)

        syndromes, errors = dem_sampling(H, 4, probs, seed=7, backend="gpu")
        assert isinstance(syndromes, torch.Tensor)
        assert isinstance(errors, torch.Tensor)
        assert syndromes.device == device
        assert errors.device == device

        expected_errors = torch.tensor([1, 0, 1],
                                       dtype=torch.uint8,
                                       device=device)
        expected_syn = torch.tensor([0, 1], dtype=torch.uint8, device=device)
        for shot in range(4):
            assert torch.equal(errors[shot], expected_errors)
            assert torch.equal(syndromes[shot], expected_syn)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestZeroShots:

    def test_zero_shots_numpy(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([0.1, 0.2, 0.3])
        syndromes, errors = dem_sampling(H, 0, probs, seed=0)

        assert syndromes.shape == (0, 2)
        assert errors.shape == (0, 3)
        assert syndromes.dtype == np.uint8
        assert errors.dtype == np.uint8

    def test_zero_shots_cpu_backend(self):
        H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        probs = np.array([0.5, 0.5])
        syndromes, errors = dem_sampling(H, 0, probs, seed=0, backend="cpu")

        assert syndromes.shape == (0, 2)
        assert errors.shape == (0, 2)

    @pytest.mark.skipif(not _HAS_RUNTIME_GPU_BACKEND,
                        reason="GPU backend unavailable")
    def test_zero_shots_gpu_backend(self):
        H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        probs = np.array([0.5, 0.5])
        syndromes, errors = dem_sampling(H, 0, probs, seed=0, backend="gpu")

        assert syndromes.shape == (0, 2)
        assert errors.shape == (0, 2)


class TestNonBinaryCheckMatrix:

    def test_non_binary_entries_masked(self):
        """H entries > 1 should be treated as H & 1 (binary)."""
        H = np.array([[2, 3], [1, 0]], dtype=np.uint8)
        probs = np.array([1.0, 1.0])
        syndromes, errors = dem_sampling(H, 4, probs, seed=0, backend="cpu")

        np.testing.assert_array_equal(errors, 1)
        for shot in range(4):
            assert syndromes[shot, 0] == 1, "Binarized [0,1]: sum=1 mod 2=1"
            assert syndromes[shot, 1] == 1, "Binarized [1,0]: sum=1 mod 2=1"

    @pytest.mark.skipif(not _HAS_RUNTIME_GPU_BACKEND,
                        reason="GPU backend unavailable")
    def test_non_binary_cpu_gpu_match(self):
        """CPU and GPU must agree on non-binary H after binarization."""
        H = np.array([[2, 3], [1, 0]], dtype=np.uint8)
        probs = np.array([1.0, 1.0])

        syn_cpu, err_cpu = dem_sampling(H, 4, probs, seed=0, backend="cpu")
        syn_gpu, err_gpu = dem_sampling(H, 4, probs, seed=0, backend="gpu")

        np.testing.assert_array_equal(err_cpu, err_gpu)
        np.testing.assert_array_equal(syn_cpu, syn_gpu)


class TestSeedlessPath:

    def test_seedless_runs(self):
        """Calling without seed should succeed and produce valid output."""
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([0.3, 0.5, 0.7])
        syndromes, errors = dem_sampling(H, 50, probs)

        assert syndromes.shape == (50, 2)
        assert errors.shape == (50, 3)
        assert set(np.unique(syndromes)).issubset({0, 1})
        assert set(np.unique(errors)).issubset({0, 1})

    def test_seedless_nondeterministic(self):
        """Two seedless calls should very likely produce different results."""
        H = np.eye(10, dtype=np.uint8)
        probs = np.full(10, 0.5)
        _, e1 = dem_sampling(H, 500, probs)
        _, e2 = dem_sampling(H, 500, probs)
        assert not np.array_equal(e1, e2)


# ---------------------------------------------------------------------------
# Tests for missing-torch print and invalid types
# ---------------------------------------------------------------------------


class TestTorchNotInstalledWarning:

    def test_warns_install_message(self):
        """When torch is absent and input has data_ptr, emit install hint."""
        import sys

        class FakeTensor:
            """Object that looks like a torch tensor but torch is absent."""

            def data_ptr(self):
                return 0

        real_torch = sys.modules.get("torch")
        sys.modules["torch"] = None  # force ImportError on `import torch`
        try:
            with pytest.warns(UserWarning, match="pip install torch"):
                try:
                    dem_sampling(FakeTensor(), 1, FakeTensor(), seed=0)
                except Exception:
                    pass
        finally:
            if real_torch is not None:
                sys.modules["torch"] = real_torch
            else:
                sys.modules.pop("torch", None)


class TestRandomObjectRejected:

    def test_list_accepted(self):
        """Plain lists are auto-converted via numpy and should work."""
        syndromes, errors = dem_sampling([[1, 0], [0, 1]],
                                         4, [0.5, 0.5],
                                         seed=0)
        assert syndromes.shape == (4, 2)
        assert errors.shape == (4, 2)

    def test_string_rejected(self):
        """Strings are not valid inputs."""
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            dem_sampling("not a matrix", 4, "not probs", seed=0)
