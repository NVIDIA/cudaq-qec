# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named option bundles for the nv-qldpc decoder.

A preset is a plain dict of ``cudaq_qec.get_decoder("nv-qldpc-decoder", H,
...)`` options. ``preset(name, **overrides)`` returns a copy with the
overrides applied; pass it to the regular ``get_decoder``.

    import cudaq_qec as qec
    from cudaq_qec.plugins import nv_qldpc
    dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates,
                          **nv_qldpc.preset("bp10-minllr-osd-cs1000"))
    model = nv_qldpc.init_tcbp_model(H)                     # once per code family
    dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates,
                          **nv_qldpc.preset("tcbp-jobs", tcbp_model=model))

Names are read left to right: the BP front end, the OSD column ordering, the
post-processor, then modifiers.

Front end (sparse GPU BP, ``bp_method=0``, ``clip_value=15``, fp32):

- ``bp60``: 60 BP iterations, columns ordered by the final LLRs.
- ``bp10-minllr``: 10 BP iterations, columns ordered by the minimum marginal
  LLR over the iterations (``osd_init_method="min_llr"``).
- ``tcbp``: 10 BP iterations, columns ordered by a TCBP model
  (``osd_init_method="tcbp"``). These are the settings ``init_tcbp_model``
  fits at by default, so a model made with default parameters is accepted.
  Needs a ``tcbp_model`` override.
- ``relay-bp``: Relay-BP, 80 pre-iterations and 300 relay legs.

Post-processor:

- none (``bp60``, ``relay-bp``): BP's hard decision.
- none (``tcbp``): the TCBP model's soft output (see ``init_tcbp_model``).
- ``osd0``: OSD-0 (``osd_method=1``).
- ``osd-cs1000``: combination-sweep OSD of order 1000 (``osd_method=3``,
  ``osd_order=1000``).
- ``jobs``: JOBS (``osd_method=4``).

Modifiers:

- ``sc``: ``jobs_search_converged=True``.
- ``gap``: ``opt_results={"jobs_gap": True}``; the decoder must be
  constructed with observables (``O=``).

Presets are starting points. They set no error rates (pass
``error_rate_vec``, ``error_rate`` or a ``DetectorErrorModel`` as usual) and
leave every other option (``osd_sparsify``, the ``jobs_*`` options, the
Relay-BP ``gamma0`` / ``gamma_dist`` / ``srelay_config`` legs,
``bp_batch_size``, ``bp_method``) at the plugin's defaults; override them to
tune. The ``tcbp*`` presets must keep the ``bp_method``, ``max_iterations``,
``clip_value`` and ``proc_float`` the model was fitted at, so change those on
the model side (``init_tcbp_model`` parameters), not on the decoder.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Union

import numpy as np

_FRONT: Dict[str, Any] = dict(use_sparsity=True,
                              clip_value=15.0,
                              bp_method=0,
                              proc_float="fp32",
                              repeatable=True)
_BP60 = dict(_FRONT, max_iterations=60)
_BP10 = dict(_FRONT, max_iterations=10)
_MINLLR = dict(_BP10, osd_init_method="min_llr")
_TCBP = dict(_BP10, osd_init_method="tcbp")

_OSD0 = dict(use_osd=True, osd_method=1)
_CS1000 = dict(use_osd=True, osd_method=3, osd_order=1000)
_JOBS = dict(use_osd=True, osd_method=4)

PRESETS: Dict[str, Dict[str, Any]] = {
    # 60-iteration BP, final-LLR ordering
    "bp60":
        dict(_BP60),
    "bp60-osd0":
        dict(_BP60, **_OSD0),
    "bp60-osd-cs1000":
        dict(_BP60, **_CS1000),
    # 10-iteration BP, minimum-LLR ordering
    "bp10-minllr-osd0":
        dict(_MINLLR, **_OSD0),
    "bp10-minllr-osd-cs1000":
        dict(_MINLLR, **_CS1000),
    "bp10-minllr-jobs":
        dict(_MINLLR, **_JOBS),
    # Relay-BP
    "relay-bp":
        dict(use_sparsity=True,
             repeatable=True,
             scale_factor=1.0,
             proc_float="fp32",
             bp_seed=4242,
             iter_per_check=1,
             bp_method=3,
             composition=1,
             gamma0=0.125,
             max_iterations=60,
             clip_value=200.0,
             gamma_dist=[-0.24, 0.66],
             use_osd=False,
             srelay_config={
                 "pre_iter": 80,
                 "num_sets": 300,
                 "stopping_criterion": "NConv",
                 "stop_nconv": 1,
             }),
    # TCBP: soft predecoder, or TCBP-ordered OSD / JOBS
    "tcbp":
        dict(_BP10, use_osd=False),
    "tcbp-osd-cs1000":
        dict(_TCBP, **_CS1000),
    "tcbp-jobs":
        dict(_TCBP, **_JOBS),
    "tcbp-jobs-sc":
        dict(_TCBP, **_JOBS, jobs_search_converged=True),
    "tcbp-jobs-sc-gap":
        dict(_TCBP,
             **_JOBS,
             jobs_search_converged=True,
             opt_results={"jobs_gap": True}),
}

# Presets that read a TCBP model (soft predecoder, or OSD / JOBS ordering).
REQUIRES_TCBP_MODEL = frozenset(n for n in PRESETS if n.startswith("tcbp"))

# Presets whose decoder must be constructed with observables (O=...).
REQUIRES_OBSERVABLES = frozenset(n for n in PRESETS if n.endswith("-gap"))

TcbpModelLike = Union[bytes, bytearray, np.ndarray, str, "os.PathLike[str]"]


def names() -> list:
    """The preset names, in documentation order."""
    return list(PRESETS)


def _tcbp_model_option(model: TcbpModelLike):
    """The ``tcbp_model`` option value for ``model``: a path string stays a
    path; bytes, a bytearray or a uint8 array become the writable 1-D uint8
    array the decoder expects."""
    if isinstance(model, (str, os.PathLike)):
        return os.fspath(model)
    if isinstance(model, (bytes, bytearray)):
        return np.frombuffer(bytes(model), dtype=np.uint8).copy()
    arr = np.ascontiguousarray(model, dtype=np.uint8).reshape(-1)
    return arr if arr.flags.writeable else arr.copy()


def preset(name: str, **overrides) -> Dict[str, Any]:
    """A copy of preset ``name`` with ``overrides`` applied on top, ready for
    ``cudaq_qec.get_decoder("nv-qldpc-decoder", H, **options)``.

    The result is a starting point: nested dicts (``srelay_config``,
    ``opt_results``) are copied, not shared, so it can be edited freely, and
    the decoder validates whatever is changed. A ``tcbp_model`` override may be the
    bytes returned by ``init_tcbp_model``, a uint8 array or a file path; it is
    converted to the form the decoder accepts. Raises ``KeyError`` for an
    unknown name and ``ValueError`` for a ``tcbp*`` preset without a
    ``tcbp_model``.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown nv-qldpc preset {name!r}; "
                       f"known presets: {', '.join(PRESETS)}")
    out = {
        k: (dict(v)
            if isinstance(v, dict) else list(v) if isinstance(v, list) else v
           ) for k, v in PRESETS[name].items()
    }
    out.update(overrides)
    if out.get("tcbp_model") is None:
        out.pop("tcbp_model", None)  # an explicit None means "not given"
    if "tcbp_model" in out:
        out["tcbp_model"] = _tcbp_model_option(out["tcbp_model"])
    elif name in REQUIRES_TCBP_MODEL:
        raise ValueError(
            f"nv-qldpc preset {name!r} reads a TCBP model; pass "
            "tcbp_model=<bytes from init_tcbp_model(H)> (or a uint8 array or "
            "a file path)")
    return out
