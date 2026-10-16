# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from typing import Any
import cudaq_qec as qec
import numpy as np

import numpy.typing as npt
from quimb.tensor import TensorNetwork
import torch
from autoray import do, to_backend_dtype
from .tensor_network_utils.contractors import ContractorConfig, optimize_path
from .tensor_network_utils.tensor_network_factory import (
    tensor_network_from_parity_check, tensor_network_from_single_syndrome,
    tensor_network_from_syndrome_batch, tensor_network_from_logical_observable)


def _safe_posterior(mass0, mass1, dtype: str) -> tuple[float, bool]:
    """Validate contraction masses and compute the posterior P(logical flip).

    Tiny negative values within floating-point round-off of zero are normalized
    to zero. On any materially non-physical input (negative, NaN, infinite, or
    zero-sum masses), return ``(0.5, False)`` so callers can mark the result
    unconverged rather than silently propagating a bad value.
    """
    m0, m1 = float(mass0), float(mass1)
    if not np.isfinite(m0) or not np.isfinite(m1):
        return 0.5, False

    scale = max(abs(m0), abs(m1))
    tolerance = 64.0 * np.finfo(np.dtype(dtype)).eps * scale
    if m0 < -tolerance or m1 < -tolerance:
        return 0.5, False

    # Signed contractions can leave an exact zero a few ULPs below zero.
    m0 = max(m0, 0.0)
    m1 = max(m1, 0.0)
    denom = m0 + m1
    if not np.isfinite(denom) or denom == 0.0:
        return 0.5, False
    return m1 / denom, True


@qec.decoder("tensor_network_decoder")
class TensorNetworkDecoder:
    r"""A general class for tensor network decoders.

    Constructs a tensor network with the following tensors:
    Hadamard matrices for each edge of the Tanner graph:
        - Hadamard tensors for each check-error pair.
        - Hadamard tensors for each logical observable-error pair.
    Delta tensors for each vertex of the Tanner graph:
        - Delta tensors for each error represented as indices.
        - Delta tensors for each logical check represented as indices.

    The core, tensor-network graph is identical to a Tanner graph.
    The tensor network is a bipartite graph with two types of nodes:
    - Check nodes (c_i's) and error nodes (e_j's), delta tensors.
    - Hadamard matrix (H) for each check-error pair.

    Then, the tensor network is extended by the noise model, the logical observables,
    and the product state / batch of syndromes.

    For example,

                  open leg      < logical observable
                  --------
                     |
        s1      s2   |     s3   < syndromes               : product state of zeros/ones
        |       |    |     |                        ----|
        c1      c2  l1     c3   < checks / logical      | : delta tensors
        |     / |   | \    |                            |
        H   H   H   H  H   H    < Hadamard matrices     | TANNER (bipartite) GRAPH
          \ |   |  /   |  /                             |
            e1  e2     e3       < errors                | : delta tensors
            |   |     /                            -----|
             \ /     /
            P(e1, e2, e3)       < noise / error model     : classical probability density

    ci, ej, lk are delta tensors represented sparsely as indices.


    Attributes:
        code_tn (TensorNetwork): The tensor network for the code (parity check matrix).
        logical_tn (TensorNetwork): The tensor network for the logical observables.
        syndrome_tn (TensorNetwork): The tensor network for the syndrome.
        noise_model (TensorNetwork): The noise model tensor network.
        full_tn (TensorNetwork): The full tensor network including code, logical, syndrome, and noise model.
        check_inds (list[str]): The check indices.
        error_inds (list[str]): The error indices.
        logical_inds (list[str]): The logical indices.
        logical_obs_inds (list[str]): The logical observable indices.
        logical_tags (list[str]): The logical tags.
        _contractor_name (str): The contractor to use.
        _backend (str): The backend used for tensor operations ("numpy" or "torch").
        _dtype (str): The data type of the tensors.
        _device (str): The device for tensor operations ("cpu" or "cuda:X").
        path_single (Any): The contraction path for single syndrome decoding.
        path_batch (Any): The contraction path for batch decoding.
        slicing_single (Any): Slicing specification for single syndrome contraction.
        slicing_batch (Any): Slicing specification for batch contraction.
    """
    code_tn: TensorNetwork
    logical_tn: TensorNetwork
    syndrome_tn: TensorNetwork
    noise_model: TensorNetwork
    full_tn: TensorNetwork
    check_inds: list[str]
    error_inds: list[str]
    logical_inds: list[str]
    logical_obs_inds: list[str]
    logical_tags: list[str]
    _contractor_name: str
    _backend: str
    _dtype: str
    _device: str
    path_single: Any
    path_batch: Any
    slicing_single: Any
    slicing_batch: Any

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        noise_model: TensorNetwork | list[float],
        check_inds: list[str] | None = None,
        error_inds: list[str] | None = None,
        logical_inds: list[str] | None = None,
        logical_tags: list[str] | None = None,
        contract_noise_model: bool = True,
        dtype: str = "float64",
        device: str = "cuda",
    ) -> None:
        """Initialize a sparse representation of a tensor network decoder for an arbitrary code
        given by its parity check matrix, logical observables and noise model.

        Args:
            H (np.ndarray): The parity check matrix. First dimension is the number of checks, second is the number of errors.
            logical_obs (np.ndarray): The logical. First dimension is one, second is the number of errors.
            noise_model (Union[TensorNetwork, list[float]]): The noise model to use. Can be a tensor network or a list of probabilities.
                If a tensor network, it must have exactly parity_check_matrix.shape[1] open indices.
                The same ordering is assumed as in the parity check matrix.
                If a list, it must have the same length as parity_check_matrix.shape[1].
                A product state noise model will be constructed from it.
            check_inds (Optional[list[str]], optional): The check indices. If None, defaults to [c_0, c_1, ...].
            error_inds (Optional[list[str]], optional): The error indices. If None, defaults to [e_0, e_1, ...].
            logical_inds (Optional[list[str]], optional): The index of the logical. If None, defaults to [l_0].
            logical_tags (Optional[list[str]], optional): The logical tags. If None, defaults to [LOG_0, LOG_1, ...].
            contract_noise_model (bool, optional): Whether to contract the noise model with the tensor network at initialization.
            contractor_name (Optional[str], optional): The contractor to use. If None, defaults to "numpy".
            dtype (str, optional): The data type of the tensors in the tensor network. Defaults to "float64".
            device (str, optional): The device to use for the tensors in the tensor network. Defaults to "gpu".
                Options are "cpu", "cuda", or "cuda:X", where X is the target cuda device.
        """

        qec.Decoder.__init__(self, H)

        gpu_available = torch.cuda.is_available()

        if gpu_available and "cuda" in device:
            contractor_name = "cutensornet"
            backend = "numpy"
        else:
            print("Warning: CUDA is not available. "
                  "Using CPU for tensor network operations.")
            contractor_name = "torch"
            device = "cpu"
            backend = "torch"

        num_checks, num_errs = H.shape
        if check_inds is None:
            self.check_inds = [f"s_{j}" for j in range(num_checks)]
        else:
            assert len(check_inds) == num_checks, (
                f"check_inds must have length {num_checks}, "
                f"but got {len(check_inds)}.")
            self.check_inds = check_inds
        if error_inds is None:
            self.error_inds = [f"e_{j}" for j in range(num_errs)]
        else:
            assert len(error_inds) == num_errs, (
                f"error_inds must have length {num_errs}, "
                f"but got {len(error_inds)}.")
            self.error_inds = error_inds

        self.logical_obs_inds = ["obs"]  # Open logical index

        # Construct the tensor network of the code
        self.parity_check_matrix = H.copy()
        self.code_tn = tensor_network_from_parity_check(
            self.parity_check_matrix,
            col_inds=self.error_inds,
            row_inds=self.check_inds,
        )

        self.replace_logical_observable(
            logical_obs,
            logical_inds=logical_inds,
            logical_tags=logical_tags,
        )

        # Initialize the syndrome tensor network with no errors.
        self.syndrome_tn = tensor_network_from_single_syndrome(
            [0.0] * len(self.check_inds), self.check_inds)

        # Construct the tensor network of code + logical + syndromes
        # The noise model is added later
        self.full_tn = TensorNetwork()
        self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)

        # Default values for the path finders
        self.path_single = None if contractor_name == "cutensornet" else "auto"
        self.path_batch = None if contractor_name == "cutensornet" else "auto"
        self.slicing_batch = tuple()
        self.slicing_single = tuple()
        self._batch_size = 1

        self._set_contractor(contractor_name, device, backend, dtype)

        # Initialize the noise model
        if isinstance(noise_model, TensorNetwork):
            old_inds = noise_model._outer_inds
            assert len(old_inds) == len(self.error_inds), (
                f"Noise model has {len(old_inds)} open indices, "
                f"but expected {len(self.error_inds)} for the error indices.")
            # Reindex the noise model to match the error indices
            ind_map = {oi: ni for oi, ni in zip(old_inds, self.error_inds)}
            noise_model = noise_model.reindex(ind_map)
        else:
            from .tensor_network_utils.noise_models import factorized_noise_model
            noise_model = factorized_noise_model(self.error_inds, noise_model)
        self.init_noise_model(noise_model, contract=contract_noise_model)

    def replace_logical_observable(
            self,
            logical_obs: npt.NDArray[Any],
            logical_inds: list[str] | None = None,
            logical_tags: list[str] | None = None) -> None:
        """Add logical observables to the tensor network.
        Args:
            logical_obs (np.ndarray): The logical matrix.
            logical_inds (Optional[list[str]], optional): The logical indices. If None, defaults to [l_0, l_1, ...].
            logical_obs_inds (Optional[list[str]], optional): The logical observable indices. If None, defaults to [l_obs_0, l_obs_1, ...].
            logical_tags (Optional[list[str]], optional): The logical tags. If None, defaults to [LOG_0, LOG_1, ...].
        """
        assert logical_obs.shape == (1, len(self.error_inds)), (
            "logical must be a single row matrix, shape (1, n), where n is the number of errors."
            "Only single logical are supported for now.")
        if logical_inds is None:
            self.logical_inds = ["l_0"]  # Index before the Hadamard tensor
        else:
            assert len(logical_inds) == 1, (
                "logical_inds must be a list of length 1, "
                "as only single logical observables are supported for now.")
            self.logical_inds = logical_inds

        if logical_tags is None:
            self.logical_tags = ["LOG_0"]
        else:
            self.logical_tags = logical_tags

        # Construct the tensor network of the logical observables
        self.logical_obs = logical_obs.copy()
        self.logical_tn = tensor_network_from_parity_check(
            self.logical_obs,
            col_inds=self.error_inds,
            row_inds=self.logical_inds,
            tags=self.logical_tags,
        )

        # Add a Hadamard tensor for each logical observable for its outer leg
        self.logical_tn = self.logical_tn.combine(
            tensor_network_from_logical_observable(self.logical_obs,
                                                   self.logical_inds,
                                                   self.logical_obs_inds,
                                                   self.logical_tags),
            virtual=True)

        if hasattr(self, "full_tn"):
            self.full_tn = TensorNetwork()
            self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
            self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
            self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)
            if hasattr(self, "noise_model"):
                self.full_tn = self.full_tn.combine(self.noise_model,
                                                    virtual=True)

            self._set_tensor_type(self.full_tn)

    def init_noise_model(self,
                         noise_model: TensorNetwork,
                         contract: bool = False) -> None:
        """Initialize the noise model.

        Args:
            noise_model (TensorNetwork): The noise model tensor network.
            contract (bool, optional): Whether to contract the noise model with the tensor network. Defaults to False.
        """
        assert isinstance(noise_model, TensorNetwork), (
            "noise_model must be an instance of quimb.tensor.TensorNetwork.")
        assert sorted(noise_model.outer_inds()) == sorted(self.error_inds), (
            f"Noise model has {len(noise_model.outer_inds())} open indices, "
            f"but expected {len(self.error_inds)} for the error indices.")
        self.noise_model = noise_model
        self._set_tensor_type(self.noise_model)
        self.full_tn = TensorNetwork()
        self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.noise_model, virtual=True)

        if contract:
            for ie in self.error_inds:
                self.full_tn.contract_ind(ie)

    def flip_syndromes(self, values: list[float]) -> None:
        """Modify the tensor network in place to represent a given syndrome.

        Args:
            values (list): A list of float values for the syndrome.
                The probability an observable was flipped.
        """

        # Below we use autoray.do to ensure that the data is
        # defined via the correct backend: numpy, torch, etc.
        assert len(values) == len(self.check_inds), (
            f"Values length {len(values)} does not match the number of checks {len(self.check_inds)}."
        )
        assert all(isinstance(v, float)
                   for v in values), "Syndrome values must be float."

        dtype = to_backend_dtype(self._dtype,
                                 like=self.contractor_config.backend)
        array_args = {"like": self.contractor_config.backend, "dtype": dtype}

        minus = do("array", (1.0, -1.0), **array_args)
        plus = do("array", (1.0, 1.0), **array_args)

        for ind in range(len(self.check_inds)):
            t = self.syndrome_tn.tensors[next(
                iter(self.syndrome_tn.tag_map[f"SYN_{ind}"]))]
            t.modify(data=values[ind] * minus + (1.0 - values[ind]) * plus,)

    def _set_contractor(
        self,
        contractor: str,
        device: str,
        backend: str,
        dtype: str | None = None,
    ) -> None:
        """Set the contractor for the tensor network.

        Args:
            contractor (str): The contractor to use.
            dtype (str, optional): The data type to use. If None, keeps the current dtype.
            device (str, optional): The device to use. If None, keeps the current device.

        Raises:
            ValueError: If the contractor is not found or device is invalid for the contractor.
        """
        print(
            f"(Re-)set contractor: {contractor} on device: {device} with backend: {backend}"
        )
        self.contractor_config = ContractorConfig(
            contractor_name=contractor,
            backend=backend,
            device=device,
        )

        # Reset only if specified
        if dtype is not None:
            self._dtype = dtype
        print(f"Using dtype: {self._dtype}")

        self._set_tensor_type(self.full_tn)

        is_cutensornet = contractor == "cutensornet"

        def _adjust_default_path_value(val):
            if is_cutensornet:
                return None if val == "auto" else val
            else:
                return "auto" if val is None else val

        self.path_batch = _adjust_default_path_value(self.path_batch)
        self.path_single = _adjust_default_path_value(self.path_single)

    def decode(
        self,
        syndrome: list[float],
    ) -> "qec.DecoderResult":
        """
        Decode the syndrome by contracting exactly the full tensor network.

        Args:
            syndrome (list[float]): 
                The syndrome soft decision probabilities ordered as the check indices.

        Returns:
            qec.DecoderResult: The result of the decoding.
                The probability that the logical observable flipped.
        """
        assert hasattr(self, "noise_model")
        assert len(syndrome) == len(self.check_inds), (
            f"Syndrome length {len(syndrome)} does not match the number of checks {len(self.check_inds)}."
        )
        assert all(isinstance(s, float)
                   for s in syndrome), "Syndrome values must be float."

        # adjust the values of the syndromes
        self.flip_syndromes(syndrome)

        if self.path_single is None:
            # If the path is not set, we need to optimize it
            self.optimize_path(optimize=self.path_single,)

        contraction_value = self.contractor_config.contractor(
            self.full_tn.get_equation(output_inds=(self.logical_obs_inds[0],)),
            self.full_tn.arrays,
            optimize=self.path_single,
            slicing=self.slicing_single,
            device_id=self.contractor_config.device_id,
        )

        posterior, valid = _safe_posterior(contraction_value[0],
                                           contraction_value[1], self._dtype)
        res = qec.DecoderResult()
        res.converged = valid
        res.result = [posterior]
        return res

    def decode_batch(
        self,
        syndrome_batch: npt.NDArray[Any],
    ) -> "qec.BatchDecoderResult":
        """Decode a batch of detection events.

        Args:
            syndrome_batch (np.ndarray): A numpy array of shape (batch_size, syndrome_length) where each row is a detection event.

        Returns:
            qec.BatchDecoderResult: batched results for each detection event in
                the batch. The `result` field has shape (batch_size, 1) and
                contains the probabilities that the logical observable flipped
                for each syndrome.
        """

        assert hasattr(self, "noise_model")
        syndrome_length = syndrome_batch.shape[1]
        assert syndrome_length == len(self.check_inds)

        # Remove the syndrome tensors from the full tensor network
        tn = TensorNetwork(
            [t for t in self.full_tn.tensors if "SYNDROME" not in t.tags])

        tn = tn.combine(tensor_network_from_syndrome_batch(
            syndrome_batch, self.check_inds, batch_index="batch_index"),
                        virtual=True)
        # Set the tensor type for the new tensor network
        self._set_tensor_type(tn)

        if self.path_batch is None or syndrome_batch.shape[
                0] != self._batch_size:
            # If the path is not set, we need to optimize it
            self.optimize_path(
                optimize=self.path_batch,
                batch_size=syndrome_batch.shape[0],
            )
            self._batch_size = syndrome_batch.shape[0]

        contraction_value = self.contractor_config.contractor(
            tn.get_equation(output_inds=("batch_index",
                                         self.logical_obs_inds[0])),
            tn.arrays,
            optimize=self.path_batch,
            slicing=self.slicing_batch,
            device_id=self.contractor_config.device_id,
        )

        probabilities = []
        converged = []
        for r in range(syndrome_batch.shape[0]):
            posterior, valid = _safe_posterior(contraction_value[r, 0],
                                               contraction_value[r, 1],
                                               self._dtype)
            probabilities.append(posterior)
            converged.append(valid)

        # Python `decode_batch` override: construct a BatchDecoderResult
        # directly, bypassing the native decoder aggregation path. This is
        # the only sanctioned caller of the BatchDecoderResult constructor;
        # see its docstring for the supported construction surface.
        return qec.BatchDecoderResult(
            np.asarray(probabilities, dtype=np.float64).reshape((-1, 1)),
            np.asarray(converged, dtype=bool),
        )

    def optimize_path(
        self,
        optimize: Any = None,
        batch_size: int = -1,
    ) -> Any:
        """Optimize the contraction path of the tensor network.

        Args:
            optimize (cuquantum.tensornet.OptimizerOptions | None, optional):
                The optimization options to use.
                If None or ``cuquantum.tensornet.OptimizerOptions``, we use cuquantum.tensornet.
                Else, Quimb interface at 
                https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_core/index.html#quimb.tensor.tensor_core.TensorNetwork.contraction_info
            batch_size (int, optional): The batch size for the optimization. Defaults to -1, which means no batching.
        Returns:
            Any: The optimizer info object.
        """
        assert isinstance(batch_size, int), ("batch_size must be an integer, "
                                             "or -1 to indicate no batching.")

        is_batch = batch_size > 0

        # Build the tensor network
        if is_batch:
            import numpy as np
            self._batch_size = batch_size
            tn = TensorNetwork(
                [t for t in self.full_tn.tensors if "SYNDROME" not in t.tags])
            fake_batch = np.ones((batch_size, len(self.check_inds)),
                                 dtype=self._dtype)
            tn = tn.combine(tensor_network_from_syndrome_batch(
                fake_batch, self.check_inds, batch_index="batch_index"),
                            virtual=True)
            self._set_tensor_type(tn)
            output_inds = ("batch_index", self.logical_obs_inds[0])
        else:
            tn = self.full_tn
            output_inds = (self.logical_obs_inds[0],)

        self._set_tensor_type(tn)

        # Optimize the path
        path, info = optimize_path(optimize, output_inds, tn)
        slices = info.slices if hasattr(info, "slices") else tuple()

        # Assign result
        target = "path_batch" if is_batch else "path_single"
        setattr(self, target, path)

        target = "slicing_batch" if is_batch else "slicing_single"
        setattr(self, target, slices)

        return info

    def _set_tensor_type(self, tn: TensorNetwork) -> None:
        """Set the backend for the tensor network."""
        assert isinstance(tn, TensorNetwork), (
            "The tensor network must be an instance of quimb.tensor.TensorNetwork."
        )
        tn.apply_to_arrays(lambda x: do(
            "array",
            x,
            like=self.contractor_config.backend,
            dtype=to_backend_dtype(self._dtype,
                                   like=self.contractor_config.backend),
        ))
