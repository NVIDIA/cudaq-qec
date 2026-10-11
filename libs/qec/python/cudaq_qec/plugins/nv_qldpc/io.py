# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Detector-sample files: ``samples_{basis}.dets`` (stim shot data, detectors then observables) + ``metadata_{basis}.json``
(``distance``, ``n_rounds``, ``basis``, ``code_rotation``, ``num_detectors``, ``num_observables``, ``num_shots``, ``format``,
``append_observables``). ``read_detector_samples`` returns the uint8 [N, n_det] syndromes ``learn_noise`` takes and checks the
structural fields against the circuit the learner runs on (``num_detectors`` from H, ``num_observables`` from O, ``expect=`` for
distance / rounds / basis / rotation from ``circuit_structure``), so a file from another experiment fails loudly. Only numpy and
stim. Files in the packed ``b8`` and text ``01`` formats are read in slabs (no whole-file copy in memory beyond the result)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import stim

PathLike = Union[str, os.PathLike]
SUPPORTED_FORMATS = ("dets", "b8", "01", "hits", "r8")
STRUCTURAL_FIELDS = ("distance", "n_rounds", "basis", "code_rotation",
                     "num_detectors", "num_observables", "num_shots")
_ROTATION_ALIASES = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}
_MEASURE_Z = frozenset({"M", "MZ", "MR", "MRZ"})
_MEASURE_X = frozenset({"MX", "MRX"})


class DetectorSampleMismatch(ValueError):
    """Metadata of a detector-sample pair disagrees with the circuit; ``fields`` names the offending structural fields."""

    def __init__(self, problems):
        self.problems = list(problems)
        self.fields = [p.split()[0] for p in self.problems]
        super().__init__(
            "detector-sample metadata does not match the circuit:\n- " +
            "\n- ".join(self.problems))


def circuit_structure(circuit: stim.Circuit) -> Dict[str, Any]:
    """basis / distance / n_rounds / counts of a memory circuit, from the circuit itself (measurement layers open after a
    DETECTOR; the last layer is the data readout: its basis and d*d qubits). ``code_rotation`` is not recoverable."""
    layers = []
    after_detector = True
    for inst in circuit.flattened():
        name = inst.name
        if name in _MEASURE_Z or name in _MEASURE_X:
            if after_detector:
                layers.append((set(), 0))
                after_detector = False
            names, n = layers[-1]
            names.add(name)
            layers[-1] = (names, n + len(inst.targets_copy()))
        elif name == "DETECTOR":
            after_detector = True
    out: Dict[str, Any] = dict(num_detectors=int(circuit.num_detectors),
                               num_observables=int(circuit.num_observables))
    if not layers:
        return out
    names, n_data = layers[-1]
    if names <= _MEASURE_X:
        out["basis"] = "X"
    elif names <= _MEASURE_Z:
        out["basis"] = "Z"
    d = int(round(n_data**0.5))
    if d * d == n_data and d > 0:
        out["distance"] = d
    out["n_rounds"] = len(layers) - 1
    return out


def normalize_code_rotation(value: Any) -> str:
    rot = str(value).strip().upper()
    return _ROTATION_ALIASES.get(rot, rot)


def read_metadata(path: PathLike) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    problems = [
        f"missing field {k!r}" for k in STRUCTURAL_FIELDS if k not in meta
    ]
    for key in ("distance", "n_rounds", "num_detectors", "num_observables",
                "num_shots"):
        if key in meta:
            try:
                if int(meta[key]) != meta[key] or int(meta[key]) < 0:
                    problems.append(
                        f"{key} must be a non-negative integer, got {meta[key]!r}"
                    )
            except (TypeError, ValueError):
                problems.append(f"{key} must be an integer, got {meta[key]!r}")
    fmt = str(meta.get("format", "")).strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        problems.append(
            f"format {meta.get('format')!r} not in {SUPPORTED_FORMATS}")
    if meta.get("append_observables", True) is not True:
        problems.append(
            "append_observables must be true (observable bits follow the detectors)"
        )
    if problems:
        raise ValueError(f"{path}: invalid detector-sample metadata:\n- " +
                         "\n- ".join(problems))
    return meta


def validate_metadata(meta: Mapping[str, Any],
                      *,
                      H=None,
                      O=None,
                      expect: Optional[Mapping[str, Any]] = None) -> None:
    problems = []
    if H is not None and int(meta["num_detectors"]) != int(np.shape(H)[0]):
        problems.append(
            f"num_detectors {meta['num_detectors']} vs H rows {np.shape(H)[0]}")
    if O is not None and int(meta["num_observables"]) != int(np.shape(O)[0]):
        problems.append(
            f"num_observables {meta['num_observables']} vs O rows {np.shape(O)[0]}"
        )
    for key, val in (expect or {}).items():
        if key not in STRUCTURAL_FIELDS:
            raise KeyError(f"expect: unknown structural field {key!r}")
        if key == "code_rotation":
            same = normalize_code_rotation(
                meta[key]) == normalize_code_rotation(val)
        elif key == "basis":
            same = str(meta[key]).strip().upper() == str(val).strip().upper()
        else:
            same = int(meta[key]) == int(val)
        if not same:
            problems.append(f"{key} {meta[key]!r} vs expected {val!r}")
    if problems:
        raise DetectorSampleMismatch(problems)


def _read_shots(path: PathLike, fmt: str, n_det: int, n_obs: int, n_shots: int,
                slab: int) -> np.ndarray:
    """uint8 [n_shots, n_det + n_obs] (observable bits after the detectors); b8 and 01 are read in slabs of ``slab``
    shots, the other stim formats through stim."""
    n_bits = n_det + n_obs
    if fmt == "b8":
        row = (n_bits + 7) // 8
        out = np.empty((n_shots, n_bits), np.uint8)
        with open(path, "rb") as f:
            done = 0
            while done < n_shots:
                k = min(slab, n_shots - done)
                buf = np.frombuffer(f.read(k * row), np.uint8)
                if buf.size != k * row:
                    raise ValueError(
                        f"{path}: file holds fewer than {n_shots} shots")
                out[done:done + k] = np.unpackbits(
                    buf.reshape(k, row), axis=1, bitorder="little")[:, :n_bits]
                done += k
            if f.read(1):
                raise ValueError(
                    f"{path}: file holds more than {n_shots} shots")
        return out
    if fmt == "01":
        out = np.empty((n_shots, n_bits), np.uint8)
        with open(path, "r", encoding="utf-8") as f:
            done = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if done >= n_shots:
                    raise ValueError(
                        f"{path}: file holds more than {n_shots} shots")
                if len(line) != n_bits:
                    raise ValueError(
                        f"{path}: shot {done} has {len(line)} bits, expected {n_bits}"
                    )
                out[done] = np.frombuffer(line.encode(), np.uint8) - ord("0")
                done += 1
        if done != n_shots:
            raise ValueError(
                f"{path}: {done} shots in the file, metadata says {n_shots}")
        return out
    data = stim.read_shot_data_file(path=str(path),
                                    format=fmt,
                                    num_detectors=n_det,
                                    num_observables=n_obs)
    arr = np.asarray(data, dtype=np.uint8)
    if arr.shape[0] != n_shots:
        raise ValueError(
            f"{path}: {arr.shape[0]} shots in the file, metadata says {n_shots}"
        )
    return arr


def read_detector_samples(dets_path: PathLike,
                          metadata_path: PathLike,
                          *,
                          H=None,
                          O=None,
                          expect=None,
                          with_observables: bool = False,
                          slab: int = 65536):
    """``(syndromes, meta)`` (uint8 [N, n_det]) or ``(syndromes, observables, meta)``; raises DetectorSampleMismatch /
    ValueError as documented in the module docstring."""
    meta = read_metadata(metadata_path)
    validate_metadata(meta, H=H, O=O, expect=expect)
    n_det, n_obs, n_shots = int(meta["num_detectors"]), int(
        meta["num_observables"]), int(meta["num_shots"])
    fmt = str(meta["format"]).strip().lower()
    arr = _read_shots(dets_path, fmt, n_det, n_obs, n_shots, slab)
    if arr.ndim != 2 or arr.shape[1] != n_det + n_obs:
        raise ValueError(
            f"{dets_path}: shot file has shape {arr.shape}, metadata says {n_det} detectors + {n_obs} observables"
        )
    syn = np.ascontiguousarray(arr[:, :n_det])
    if with_observables:
        return syn, np.ascontiguousarray(arr[:, n_det:]), dict(meta)
    return syn, dict(meta)


def resolve_sample_paths(root: PathLike, basis: str) -> Tuple[Path, Path]:
    root = Path(root)
    b = str(basis).strip().upper()
    candidates = ((root / f"samples_{b}.dets", root / f"metadata_{b}.json"),
                  (root / b / "samples.dets", root / b / "metadata.json"),
                  (root / "samples.dets", root / "metadata.json"))
    for dets, meta in candidates:
        if dets.exists() and meta.exists():
            return dets, meta
    raise FileNotFoundError(
        f"no detector-sample pair for basis {b} under {root}; tried " +
        ", ".join(f"{d} + {m}" for d, m in candidates))
