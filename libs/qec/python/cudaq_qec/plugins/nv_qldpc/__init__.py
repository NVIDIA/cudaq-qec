# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""nv-qldpc-decoder plugin utilities: in-plugin calibration and noise learning.

Pure Python over the plugin's C ABI (nv_qldpc_c_api.h) through ctypes. ``calibrate``, ``learn_noise`` and
``inspect`` need only numpy; the ``tying`` / ``p25`` / ``io`` helpers and the command-line tools need stim and
are imported on first use.

    from cudaq_qec.plugins import nv_qldpc
    H, _, _ = nv_qldpc.tying.dem_arrays(nv_qldpc.tying.circuit_dem(circuit))   # structure only
    calibration = nv_qldpc.calibrate(H, seed=1)                   # calibration data (bytes), reusable
    nv_qldpc.inspect(calibration)                                 # dict: stored report + fingerprints
    rates, report = nv_qldpc.learn_noise(H, syndromes, calibration)            # float64 [n], dict
    ids = nv_qldpc.tying.round_classes(circuit).class_ids         # for mode="tied"

``H`` is the detector error matrix, uint8 [m detectors, n error mechanisms]. Error rates are not an
input anywhere: ``calibrate`` trains on rates it draws itself (``min_p`` / ``max_p`` / ``seed``, or
``calibrate_p``) and ``learn_noise`` learns them from the syndromes. A ``DetectorErrorModel`` or an
``(H, rates)`` tuple is rejected.

The plugin library is located, in this order, from ``$NVQLDPC_PLUGIN_LIB`` (full path), from the
installed layout next to this package (``<site-packages>/cudaq_qec/lib/decoder-plugins/``), and from
the build-tree layout (``<build>/lib/decoder-plugins/``); ``plugin_path()`` tells which.
"""
from __future__ import annotations

import ctypes as C
import json
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np

import importlib

_LAZY = ("io", "p25", "tying"
        )  # need stim; imported on first use so the core works with numpy alone


def __getattr__(name):
    if name in _LAZY:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "calibrate", "inspect", "learn_noise", "plugin_path", "tying", "p25", "io"
]

_LIB_NAME = "libcudaq-qec-nv-qldpc-decoder.so"


class _Matrix(C.Structure):
    _fields_ = [("m", C.c_uint32), ("n", C.c_uint32),
                ("H", C.POINTER(C.c_uint8))]


class _Design(C.Structure):
    _fields_ = [("n_params", C.c_uint32), ("col_ptr", C.POINTER(C.c_int32)),
                ("param_idx", C.POINTER(C.c_int32)),
                ("mult", C.POINTER(C.c_float))]


def plugin_path() -> str:
    """Path of the plugin library the module uses (see the module docstring for the search order)."""
    env = os.environ.get("NVQLDPC_PLUGIN_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(
            here, "..", "..", "lib", "decoder-plugins",
            _LIB_NAME),  # <site-packages>/cudaq_qec/lib/decoder-plugins
        os.path.join(
            here, "..", "..", "..", "..", "lib", "decoder-plugins", _LIB_NAME
        ),  # <build>/python/cudaq_qec/plugins/nv_qldpc -> <build>/lib
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    raise FileNotFoundError(
        f"{_LIB_NAME} not found next to {here}; set NVQLDPC_PLUGIN_LIB to its full path"
    )


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = C.CDLL(plugin_path())
    lib.nvqldpc_calibrate.restype = C.c_int
    lib.nvqldpc_calibrate.argtypes = [
        C.POINTER(_Matrix), C.c_char_p,
        C.POINTER(C.POINTER(C.c_uint8)),
        C.POINTER(C.c_size_t),
        C.POINTER(C.c_char_p)
    ]
    lib.nvqldpc_learn_noise.restype = C.c_int
    lib.nvqldpc_learn_noise.argtypes = [
        C.POINTER(_Matrix),
        C.POINTER(C.c_uint8), C.c_size_t, C.c_char_p,
        C.POINTER(C.c_uint8), C.c_size_t,
        C.POINTER(C.c_int32),
        C.POINTER(_Design),
        C.POINTER(C.c_double),
        C.POINTER(C.c_double),
        C.POINTER(C.c_char_p)
    ]
    lib.nvqldpc_inspect.restype = C.c_int
    lib.nvqldpc_inspect.argtypes = [
        C.POINTER(C.c_uint8), C.c_size_t,
        C.POINTER(C.c_char_p)
    ]
    lib.nvqldpc_free.restype = None
    lib.nvqldpc_free.argtypes = [C.c_void_p]
    lib.nvqldpc_last_error.restype = C.c_char_p
    lib.nvqldpc_last_error.argtypes = []
    _lib = lib
    return lib


def _check(rc: int) -> None:
    if rc != 0:
        raise ValueError(_load().nvqldpc_last_error().decode())


def _take_json(cp) -> Dict:
    s = C.cast(cp, C.c_char_p).value.decode()
    _load().nvqldpc_free(cp)
    return json.loads(s)


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _json(params: Dict) -> bytes:
    """Flat JSON for the C ABI; an ``advanced={...}`` dict is flattened to the top level."""
    out = {}
    for k, v in params.items():
        if k == "advanced" and isinstance(v, dict):
            for kk, vv in v.items():
                out[kk] = _plain(vv)
        else:
            out[k] = _plain(v)
    return json.dumps(out).encode()


def _matrix(H) -> Tuple[_Matrix, list]:
    """ctypes view of the detector error matrix. Rates are refused on purpose: calibrate() draws its own training
    rates and learn_noise() learns them, so a DEM object or an (H, rates) tuple is a mistake here."""
    looks_like_dem = hasattr(H, "detector_error_matrix") or hasattr(
        H, "error_rates")
    if isinstance(H, (tuple, list)) and 2 <= len(H) <= 3 and np.ndim(
            H[0]) == 2 and np.ndim(H[1]) == 1:
        looks_like_dem = True  # the old (H, rates[, O]) form
    if looks_like_dem:
        raise TypeError(
            "pass the detector error matrix H only (uint8 [detectors, mechanisms]): error rates are not an input"
        )
    H = np.ascontiguousarray(H, dtype=np.uint8)
    if H.ndim != 2 or H.shape[0] == 0 or H.shape[1] == 0:
        raise ValueError(
            "H must be a non-empty 2-D uint8 array [detectors, mechanisms]")
    d = _Matrix(H.shape[0], H.shape[1], H.ctypes.data_as(C.POINTER(C.c_uint8)))
    return d, [H]


def calibrate(H,
              return_report: bool = False,
              **params) -> Union[bytes, Tuple[bytes, Dict]]:
    """Calibrate for the detector error matrix ``H`` (uint8 [detectors, mechanisms]; no rates); returns the
    calibration data (bytes) that ``learn_noise`` requires, or ``(calibration, report)`` with ``return_report=True``.
    Training rates are drawn per column, log-uniformly in [``min_p``, ``max_p``] with ``seed``; ``calibrate_p``
    (one rate per column) replaces the draw. Other keywords: shots, steps, device_memory_budget_bytes, verbosity."""
    d, keep = _matrix(H)
    lib = _load()
    cal = C.POINTER(C.c_uint8)()
    cal_len = C.c_size_t()
    rep = C.c_char_p()
    _check(
        lib.nvqldpc_calibrate(C.byref(d), _json(params), C.byref(cal),
                              C.byref(cal_len), C.byref(rep)))
    out = bytes(C.string_at(cal, cal_len.value))
    lib.nvqldpc_free(cal)
    report = _take_json(rep)
    return (out, report) if return_report else out


def inspect(calibration: bytes) -> Dict:
    """Stored report, version and fingerprints of calibration data (ValueError if malformed)."""
    b = np.frombuffer(calibration, np.uint8)
    lib = _load()
    rep = C.c_char_p()
    _check(
        lib.nvqldpc_inspect(b.ctypes.data_as(C.POINTER(C.c_uint8)), len(b),
                            C.byref(rep)))
    return _take_json(rep)


def learn_noise(H,
                syndromes: np.ndarray,
                calibration: bytes,
                class_ids=None,
                design: Optional[Dict] = None,
                **params) -> Tuple[np.ndarray, Dict]:
    """Learn the error rate of every column of ``H`` (uint8 [detectors, mechanisms]; no rates are taken in) from
    detector data ``syndromes`` (uint8 [N, m]) with the ``calibration`` data of ``calibrate()`` (required); returns
    ``(rates, report)``. Keywords: mode ("per_mechanism" |
    "tied" | "parametric"), class_ids (tied), design (parametric, from nv_qldpc.p25), passes (maximum),
    tol (0.1; the change a pass must stay under to stop: median column |log change|, or the largest parameter |log change| in parametric mode; 0 = fixed count), reference_shots, seed, device_memory_budget_bytes, verbosity, advanced={rate_min, rate_max}. A note is printed when most learned rates fall outside the calibration's
    training range (``report["rates_outside_training_range"]``)."""
    if calibration is None:
        raise ValueError(
            "learn_noise: calibration data is required (see calibrate())")
    d, keep = _matrix(H)
    lib = _load()
    syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
    if syn.ndim != 2 or syn.shape[1] != d.m:
        raise ValueError(f"syndromes must be [N, {d.m}] uint8")
    sb = np.ascontiguousarray(np.frombuffer(calibration, np.uint8) if
                              isinstance(calibration,
                                         (bytes, bytearray)) else calibration,
                              dtype=np.uint8)
    cls = np.ascontiguousarray(
        class_ids, dtype=np.int32) if class_ids is not None else None
    if cls is not None and cls.shape != (d.n,):
        raise ValueError(f"class_ids must have {d.n} entries")
    des = None
    q = np.zeros(1, np.float64)
    if design is not None:
        cp = np.ascontiguousarray(design["col_ptr"], dtype=np.int32)
        pi = np.ascontiguousarray(design["param_idx"], dtype=np.int32)
        mu = np.ascontiguousarray(design["mult"], dtype=np.float32)
        des = _Design(int(design["n_params"]),
                      cp.ctypes.data_as(C.POINTER(C.c_int32)),
                      pi.ctypes.data_as(C.POINTER(C.c_int32)),
                      mu.ctypes.data_as(C.POINTER(C.c_float)))
        keep += [cp, pi, mu]
        q = np.zeros(int(design["n_params"]), np.float64)
    rates = np.zeros(d.n, np.float64)
    rep = C.c_char_p()
    _check(
        lib.nvqldpc_learn_noise(
            C.byref(d), syn.ctypes.data_as(C.POINTER(C.c_uint8)), syn.shape[0],
            _json(params),
            sb.ctypes.data_as(C.POINTER(C.c_uint8)) if sb is not None else None,
            len(sb) if sb is not None else 0,
            cls.ctypes.data_as(C.POINTER(C.c_int32))
            if cls is not None else None,
            C.byref(des) if des is not None else None,
            rates.ctypes.data_as(C.POINTER(C.c_double)),
            q.ctypes.data_as(C.POINTER(C.c_double)), C.byref(rep)))
    report = _take_json(rep)
    if design is not None:
        report["q_hat"] = q.tolist()
    return rates, report
