# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import opt_einsum as oe
from quimb.tensor import TensorNetwork


def contractor(subscripts: str,
               tensors: list[Any],
               optimize: str = "auto",
               **_: Any) -> Any:
    """
    Perform einsum contraction using opt_einsum.

    Args:
        subscripts (str): The einsum subscripts.
        tensors (list[Any]): list of tensors to contract.
        optimize (str, optional): Optimization strategy. Defaults to "auto".

    Returns:
        Any: The contracted tensor.
    """
    return oe.contract(subscripts, *tensors, optimize=optimize)


def cutn_contractor(subscripts: str,
                    tensors: list[Any],
                    optimize: Any | None = None,
                    slicing: tuple = tuple(),
                    device_id: int = 0) -> Any:
    """
    Perform contraction using cuQuantum's tensornet contractor.

    Args:
        subscripts (str): The einsum subscripts.
        tensors (list[Any]): list of tensors to contract.
        optimize (Optional[Any], optional): cuQuantum optimizer options or path. Defaults to None.
            If None, uses default optimization. Else, cuquantum.tensornet.OptimizerOptions.
        slicing (tuple, optional): Slicing specification. Defaults to empty tuple.
        device_id (int, optional): Device ID for the contraction. Defaults to 0.

    Returns:
        Any: The contracted tensor.
    """
    from cuquantum import tensornet as cutn
    return cutn.contract(
        subscripts,
        *tensors,
        optimize=cutn.OptimizerOptions(path=optimize, slicing=slicing),
        options={'device_id': device_id},
    )


def optimize_path(optimize: Any, output_inds: tuple[str, ...],
                  tn: TensorNetwork) -> tuple[Any, Any]:
    """
    Optimize the contraction path for a tensor network.

    Args:
        optimize (Any): The optimization options to use. 
            If None or cuquantum.tensornet.OptimizerOptions, we use cuquantum.tensornet.
            Else, Quimb interface at 
            https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_core/index.html#quimb.tensor.tensor_core.TensorNetwork.contraction_info
        output_inds (tuple[str, ...]): Output indices for the contraction.
        tn (TensorNetwork): The tensor network.

    Returns:
        tuple[Any, Any]: The contraction path and optimizer info.
    """
    use_cutn = optimize is None or (
        type(optimize).__module__.startswith("cuquantum") and
        type(optimize).__name__ == "OptimizerOptions")
    if use_cutn:
        from cuquantum import tensornet as cutn
        path, info = cutn.contract_path(
            tn.get_equation(output_inds=output_inds),
            *tn.arrays,
            optimize=optimize,
        )
        return path, info

    ci = tn.contraction_info(output_inds=output_inds, optimize=optimize)
    return ci.path, ci


@dataclass(frozen=True)
class ContractorConfig:
    """
    Configuration for a tensor network contractor.
    This class encapsulates the contractor name, backend, and device
    to be used for tensor network contractions.
    It validates the configuration against allowed combinations and provides
    the appropriate contractor function based on the configuration."""
    contractor_name: str
    backend: str
    device: str
    device_id: int = field(init=False)

    _allowed_configs: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("numpy", "numpy", "cpu"),
        ("torch", "torch", "cpu"),
        ("cutensornet", "numpy", "cuda"),
        ("cutensornet", "torch", "cuda"),
    )
    _allowed_backends: ClassVar[list[str]] = ["numpy", "torch"]
    _contractors: ClassVar[dict[str, Callable]] = {
        "numpy": contractor,
        "torch": contractor,
        "cutensornet": cutn_contractor,
    }

    def __post_init__(self):
        """Validate the contractor configuration."""
        if self.contractor_name not in self._contractors:
            raise ValueError(
                f"Invalid contractor name: {self.contractor_name}. "
                f"Allowed contractor names are: {list(self._contractors.keys())}."
            )
        if self.backend not in self._allowed_backends:
            raise ValueError(f"Invalid backend: {self.backend}. "
                             f"Allowed backends are: {self._allowed_backends}.")

        if "cuda" in self.device:
            dev = "cuda"
        elif "cpu" in self.device:
            dev = "cpu"
        else:
            dev = self.device
        if (self.contractor_name, self.backend,
                dev) not in self._allowed_configs:
            raise ValueError(
                f"Invalid contractor configuration: "
                f"{self.contractor_name}, {self.backend}, {self.device}. "
                f"Allowed configurations are: {self._allowed_configs}.")
        object.__setattr__(
            self, "device_id",
            int(self.device.split(":")[-1]) if "cuda:" in self.device else 0)

    @property
    def contractor(self) -> Callable:
        """Return the contractor function for this configuration."""
        return self._contractors[self.contractor_name]
