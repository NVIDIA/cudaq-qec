# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Design maps for ``learn_noise(mode="parametric")``: rates = A q.

Two ways to say which physical parameter each DEM column is made of (numpy
and stim only):

* ``design_map_by_gate(circuit)``: one parameter per distinct noise
  instruction ``(gate, rounded args)`` of the circuit, weighted 1/3 per
  component for DEPOLARIZE1 and 1/15 for DEPOLARIZE2. Right for stim's
  generated circuits (one strength per gate family).
* ``design_map(tag_circuit, tags, target_circuit)``: the 25-parameter
  surface-code noise model (two preparation, two measurement, six idle and
  fifteen CNOT rates). The caller builds the memory circuit a second time
  from ``tag_noise_model()`` (25 distinct sentinel probabilities); every
  fault location then names its parameter through the sentinel in the
  instruction's args, and the rows are permuted into the target circuit's
  DEM order. Circuits whose idle / CNOT noise is written as ``DEPOLARIZE1``
  / ``DEPOLARIZE2`` carry one probability per instruction, so their tag
  circuit names a parameter *family* instead (``tag_noise_model(families=
  True)`` adds ``p_idle_cnot``, ``p_idle_spam``, ``p_cnot``); the Pauli that
  occurred selects the member. The parameters keep their meaning (the
  probability of that Pauli at that location): a depolarizing strength
  ``p`` shows up as ``p / 3`` per single-qubit component and ``p / 15`` per
  two-qubit component.

Both return a dict ``{"col_ptr", "param_idx", "mult", "n_params"}`` for the
``design=`` keyword of ``learn_noise`` plus ``"names"`` (parameter labels).
These helpers are pure Python.
``learn_noise`` reports the design's rank and which parameters are
individually identifiable (a Z memory alone determines 23 of the 25 and
identifies ``p_prep_Z`` and ``p_meas_Z`` only as a sum).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import stim

from .tying import DEM_KWARGS, circuit_dem

CNOT_ERROR_TYPES: List[str] = [
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI",
    "ZX", "ZY", "ZZ"
]
P25_PARAM_NAMES: List[str] = [
    "p_prep_X", "p_prep_Z", "p_meas_X", "p_meas_Z", "p_idle_cnot_X",
    "p_idle_cnot_Y", "p_idle_cnot_Z", "p_idle_spam_X", "p_idle_spam_Y",
    "p_idle_spam_Z"
] + [f"p_cnot_{k}" for k in CNOT_ERROR_TYPES]
P25_FAMILY_NAMES: List[str] = ["p_idle_cnot", "p_idle_spam", "p_cnot"]
_FAMILY_GATES = {
    "p_idle_cnot": "DEPOLARIZE1",
    "p_idle_spam": "DEPOLARIZE1",
    "p_cnot": "DEPOLARIZE2"
}
_PC1_INDEX = {"X": 0, "Y": 1, "Z": 2}
_PC2_INDEX = {k: i for i, k in enumerate(CNOT_ERROR_TYPES)}
# Named by the Pauli applied, as the circuit builder's NoiseModel does:
# p_prep_X is an X_ERROR after a preparation, p_meas_X an X_ERROR before a
# measurement.
_ALLOWED_BY_GATE = {
    "X_ERROR": {"p_prep_X", "p_meas_X", "p_idle_cnot_X", "p_idle_spam_X"},
    "Z_ERROR": {"p_prep_Z", "p_meas_Z", "p_idle_cnot_Z", "p_idle_spam_Z"}
}
_COMP_FRAC = {"DEPOLARIZE1": 1 / 3, "DEPOLARIZE2": 1 / 15}


def _pack(rows: List[List[Tuple[int, float]]], n_params: int,
          names: Sequence[str]) -> Dict[str, object]:
    col_ptr = np.zeros(len(rows) + 1, np.int32)
    idx, mult = [], []
    for k, comps in enumerate(rows):
        acc: Dict[int, float] = {}
        for j, w in comps:
            acc[j] = acc.get(j, 0.0) + w
        for j in sorted(acc):
            idx.append(j)
            mult.append(acc[j])
        col_ptr[k + 1] = len(idx)
    return dict(col_ptr=col_ptr,
                param_idx=np.asarray(idx, np.int32),
                mult=np.asarray(mult, np.float32),
                n_params=int(n_params),
                names=list(names))


def design_map_by_gate(
        circuit: stim.Circuit,
        dem: Optional[stim.DetectorErrorModel] = None) -> Dict[str, object]:
    """One parameter per distinct noise instruction ``(gate, args rounded to 12 decimals)``."""
    if dem is None:
        dem = circuit_dem(circuit)
    expl = circuit.explain_detector_error_model_errors(
        dem_filter=dem, reduce_to_one_representative_error=False)
    classes: Dict[Tuple[str, Tuple[float, ...]], int] = {}
    rows: List[List[Tuple[int, float]]] = []
    for ex in expl:
        comps = []
        for loc in ex.circuit_error_locations:
            ti = loc.instruction_targets
            key = (ti.gate, tuple(round(a, 12) for a in ti.args))
            j = classes.setdefault(key, len(classes))
            comps.append((j, _COMP_FRAC.get(ti.gate, 1.0)))
        rows.append(comps)
    if any(not r for r in rows):
        raise ValueError("a DEM mechanism has no circuit explanation")
    names = [f"{g} {a}" for (g, a) in classes]
    return _pack(rows, len(classes), names)


def tag_noise_model(base: float = 1e-3,
                    step: float = 1e-2,
                    families: bool = False) -> Dict[str, float]:
    """Distinct sentinel probabilities ``base * (1 + j * step)`` for building the tag circuit: the 25 parameters in
    ``P25_PARAM_NAMES`` order and, with ``families=True``, the three family tags (``P25_FAMILY_NAMES``) for circuits
    whose idle / CNOT noise is written as ``DEPOLARIZE1`` / ``DEPOLARIZE2``."""
    names = P25_PARAM_NAMES + (P25_FAMILY_NAMES if families else [])
    return {n: base * (1.0 + j * step) for j, n in enumerate(names)}


def _component(loc: stim.CircuitErrorLocation) -> Tuple[str, str, float]:
    ti = loc.instruction_targets
    gate, args = ti.gate, list(ti.args)
    flipped = {
        t.gate_target.value: t.gate_target.pauli_type
        for t in loc.flipped_pauli_product
    }
    if gate in ("X_ERROR", "Z_ERROR", "Y_ERROR"):
        return gate, gate[0], args[0]
    if gate in ("PAULI_CHANNEL_1", "DEPOLARIZE1"):
        (pauli,) = flipped.values()
        return gate, pauli, args[
            _PC1_INDEX[pauli]] if gate == "PAULI_CHANNEL_1" else args[0]
    if gate in ("PAULI_CHANNEL_2", "DEPOLARIZE2"):
        qubits = [t.gate_target.value for t in ti.targets_in_range]
        if len(qubits) != 2:
            raise ValueError(f"{gate} location with targets {qubits}")
        label = "".join(flipped.get(q, "I") for q in qubits)
        return gate, label, args[
            _PC2_INDEX[label]] if gate == "PAULI_CHANNEL_2" else args[0]
    raise ValueError(f"unsupported noise instruction {gate} in the tag circuit")


def _consistent(gate: str, label: str, name: str) -> bool:
    if gate == "PAULI_CHANNEL_2":
        return name == f"p_cnot_{label}"
    if gate == "PAULI_CHANNEL_1":
        return name.startswith("p_idle_") and name.endswith("_" + label)
    return name in _ALLOWED_BY_GATE.get(gate, set())


def _resolve(gate: str, label: str, tag_name: str) -> str:
    """The parameter a tagged fault location belongs to; family tags (DEPOLARIZE instructions) are resolved by the
    Pauli that occurred. Raises on a tag that cannot sit on this instruction."""
    if tag_name in _FAMILY_GATES:
        if gate != _FAMILY_GATES[tag_name]:
            raise ValueError(
                f"{gate} tagged as {tag_name}: that family tag belongs on {_FAMILY_GATES[tag_name]}"
            )
        return f"{tag_name}_{label}"
    if gate in ("DEPOLARIZE1", "DEPOLARIZE2"):
        raise ValueError(
            f"{gate} tagged as {tag_name}: a depolarizing instruction carries one probability, so it must carry a "
            f"family tag ({', '.join(P25_FAMILY_NAMES)}; tag_noise_model(families=True))"
        )
    if not _consistent(gate, label, tag_name):
        raise ValueError(
            f"{gate} component {label} tagged as {tag_name}: inconsistent with the circuit builder's conventions"
        )
    return tag_name


def _mechanism_keys(dem: stim.DetectorErrorModel):
    keys = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        d, o = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                d.symmetric_difference_update([t.val])
            elif t.is_logical_observable_id():
                o.symmetric_difference_update([t.val])
        keys.append((frozenset(d), frozenset(o)))
    return keys


def align_mechanisms(dem_a: stim.DetectorErrorModel,
                     dem_b: stim.DetectorErrorModel) -> np.ndarray:
    """``perm`` with mechanism k of ``dem_b`` == mechanism ``perm[k]`` of ``dem_a`` (matched by detector + observable set)."""
    ka, kb = _mechanism_keys(dem_a), _mechanism_keys(dem_b)
    pos: Dict[object, int] = {}
    for i, k in enumerate(ka):
        if k in pos:
            raise ValueError("dem_a has a duplicated mechanism")
        pos[k] = i
    if len(set(kb)) != len(kb):
        raise ValueError("dem_b has a duplicated mechanism")
    missing = [k for k in kb if k not in pos]
    if missing or len(ka) != len(kb):
        raise ValueError(
            f"mechanism sets differ: {len(ka)} vs {len(kb)} mechanisms, {len(missing)} only in dem_b; "
            "is a parameter zero in one circuit, or are the circuits different?"
        )
    return np.array([pos[k] for k in kb], np.int64)


def design_map(tag_circuit: stim.Circuit,
               tags: Dict[str, float],
               target_circuit: Optional[stim.Circuit] = None,
               target_dem: Optional[stim.DetectorErrorModel] = None,
               rtol: float = 1e-6) -> Dict[str, object]:
    """The 25-parameter design from a tag circuit (built with ``tag_noise_model()``), rows in the target DEM's column order.
    ``tags`` may also carry the family tags of ``tag_noise_model(families=True)`` for ``DEPOLARIZE1`` / ``DEPOLARIZE2``
    instructions."""
    names = P25_PARAM_NAMES
    unknown = sorted(set(tags) - set(P25_PARAM_NAMES) - set(P25_FAMILY_NAMES))
    if unknown:
        raise ValueError(f"unknown tag names {unknown}")
    tag_names = P25_PARAM_NAMES + [n for n in P25_FAMILY_NAMES if n in tags]
    vals = np.array([float(tags[n]) for n in tag_names])
    if np.any(vals <= 0):
        raise ValueError("tag values must be > 0")
    s = np.sort(vals)
    if len(s) > 1 and ((s[1:] - s[:-1]) / s[:-1]).min() <= 2 * rtol:
        raise ValueError(f"tag values not separable at rtol={rtol}")
    lookup = {round(v, 10): j for j, v in enumerate(vals)}
    tag_dem = circuit_dem(tag_circuit)
    expl = tag_circuit.explain_detector_error_model_errors(
        dem_filter=tag_dem, reduce_to_one_representative_error=False)
    rows: List[List[Tuple[int, float]]] = []
    for k, ex in enumerate(expl):
        comps = []
        for loc in ex.circuit_error_locations:
            gate, label, value = _component(loc)
            j = lookup.get(round(value, 10))
            if j is None:
                jj = int(np.argmin(np.abs(vals - value)))
                if abs(vals[jj] - value) > rtol * vals[jj]:
                    raise ValueError(
                        f"{gate} arg {value!r} matches no tag (mechanism {k}); was the circuit built from tag_noise_model()?"
                    )
                j = jj
            param = _resolve(gate, label, tag_names[j])
            comps.append((names.index(param), 1.0))
        rows.append(comps)
    if any(not r for r in rows):
        raise ValueError("a DEM mechanism has no circuit explanation")
    if target_dem is None and target_circuit is not None:
        target_dem = circuit_dem(target_circuit)
    if target_dem is not None:
        perm = align_mechanisms(tag_dem, target_dem)
        rows = [rows[i] for i in perm]
    return _pack(rows, len(names), names)


def rates_from_q(design: Dict[str, object], q: Sequence[float]) -> np.ndarray:
    """``clip(A q, 1e-12, 0.45)`` per DEM column, the rates ``learn_noise`` induces from ``q_hat``."""
    q = np.asarray(q, np.float64)
    cp, idx, mult = design["col_ptr"], design["param_idx"], design["mult"]
    out = np.add.reduceat(mult * q[idx],
                          cp[:-1]) if len(idx) else np.zeros(len(cp) - 1)
    return np.clip(out, 1e-12, 0.45)


def export_p25(q: Sequence[float],
               names: Sequence[str] = P25_PARAM_NAMES) -> Dict[str, float]:
    """Named 25-key dict in ``P25_PARAM_NAMES`` order (ValueError on missing / unexpected names)."""
    q = np.asarray(q, np.float64).reshape(-1)
    names = list(names)
    if len(q) != len(names):
        raise ValueError(f"q has {len(q)} entries for {len(names)} names")
    named = dict(zip(names, q))
    missing = [k for k in P25_PARAM_NAMES if k not in named]
    extra = [k for k in named if k not in P25_PARAM_NAMES]
    if missing or extra:
        raise ValueError(
            f"noise-model parameters: missing {missing}, unexpected {extra}")
    return {k: float(named[k]) for k in P25_PARAM_NAMES}


def to_yaml_fragment(named: Dict[str, float],
                     identifiable: Optional[Sequence[int]] = None,
                     indent: int = 2) -> str:
    """``data: noise_model:`` YAML fragment with the 25 keys (``%.6e``); parameters flagged as not individually identifiable
    (``learn_noise`` report ``q_identifiable``) carry a trailing comment."""
    ordered = export_p25(list(named.values()), list(named.keys()))
    pad1, pad2 = " " * indent, " " * (2 * indent)
    lines = [
        "# 25-parameter circuit-level noise model learned by nv_qldpc.learn_noise (mode=parametric).",
        "# CNOT pair order (p_cnot_*): stim PAULI_CHANNEL_2 ordering, \"AB\" = Pauli A on the control, B on the target.",
        "data:", f"{pad1}noise_model:"
    ]
    for j, key in enumerate(P25_PARAM_NAMES):
        note = "" if identifiable is None or int(
            identifiable[j]
        ) else "  # not individually identifiable (read only in the combinations the data fix)"
        lines.append(f"{pad2}{key}: {ordered[key]:.6e}{note}")
    return "\n".join(lines) + "\n"
