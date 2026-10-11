# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Time-translation classes of a circuit's DEM columns for ``learn_noise(mode="tied")``.

A mechanism that repeats every round of a memory experiment should carry the
same rate in every round. ``round_classes`` groups the DEM columns by their
detector footprint shifted to round 0 (plus observable set and distance to the
prep / readout boundary) and returns one class id per column; ``learn_noise``
pools the data of each class. Only numpy and stim are used.

    class_ids = nv_qldpc.tying.round_classes(circuit).class_ids
    rates, report = nv_qldpc.learn_noise(H, syndromes, calibration, mode="tied", class_ids=class_ids)

The DEM columns are those of ``circuit.detector_error_model(flatten_loops=True,
decompose_errors=False, approximate_disjoint_errors=True,
ignore_decomposition_failures=True)`` in instruction order (the plugin's DEM
convention; pass ``dem=`` to use a DEM you already built with those flags).
The round of a detector comes from its ``DETECTOR(x, y, ..., t)`` coordinates
when the circuit has them, else from the measurement layer the detector
follows. Classes are validated against the circuit's nominal rates (members of
one class must share the nominal rate); violating classes are split into
singletons, so the tying can never merge mechanisms the circuit itself
distinguishes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import stim

DEM_KWARGS = dict(flatten_loops=True,
                  decompose_errors=False,
                  approximate_disjoint_errors=True,
                  ignore_decomposition_failures=True)
_MEASURE_GATES = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"})


def circuit_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """The DEM whose columns ``learn_noise`` learns (plugin convention)."""
    return circuit.detector_error_model(**DEM_KWARGS)


def dem_columns(
    dem: stim.DetectorErrorModel
) -> Tuple[List[List[int]], List[List[int]], np.ndarray]:
    """Per ``error`` instruction of ``dem.flattened()``: sorted detector ids
    (repeated targets XOR-cancel), sorted observable ids, and the probability."""
    dets_l, obs_l, pr = [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        d, o = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                d.symmetric_difference_update([t.val])
            elif t.is_logical_observable_id():
                o.symmetric_difference_update([t.val])
        dets_l.append(sorted(d))
        obs_l.append(sorted(o))
        pr.append(inst.args_copy()[0])
    return dets_l, obs_l, np.asarray(pr, np.float64)


def dem_arrays(dem: stim.DetectorErrorModel):
    """Dense uint8 H [m, n], O [k, n] and the rates [n] of a flattened DEM (columns = error instructions in order)."""
    dets_l, obs_l, rates = dem_columns(dem)
    n = len(dets_l)
    H = np.zeros((dem.num_detectors, n), np.uint8)
    O = np.zeros((dem.num_observables, n), np.uint8)
    for j, (d, o) in enumerate(zip(dets_l, obs_l)):
        H[d, j] = 1
        O[o, j] = 1
    return H, O, rates


@dataclass
class DetectorLayout:
    """``round_index[d]`` (0-based, contiguous) and ``spatial_id[d]`` (names the same stabiliser across rounds)."""
    round_index: np.ndarray
    spatial_id: np.ndarray
    source: str

    @property
    def n_rounds(self) -> int:
        return int(self.round_index.max()) + 1 if len(self.round_index) else 0

    @property
    def round_sizes(self) -> np.ndarray:
        return np.bincount(self.round_index, minlength=self.n_rounds)


def _contiguous(values: Sequence[Hashable]) -> np.ndarray:
    ids: Dict[Hashable, int] = {}
    out = np.empty(len(values), np.int64)
    for i, v in enumerate(values):
        out[i] = ids.setdefault(v, len(ids))
    return out


def layout_from_coordinates(circuit: stim.Circuit) -> DetectorLayout:
    """Round = distinct last coordinate (time) in increasing order, spatial id = the other coordinates."""
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    if any(len(coords.get(d, ())) < 2 for d in range(n)):
        raise ValueError("some detectors have no (x, ..., t) coordinates")
    t = np.array([coords[d][-1] for d in range(n)], np.float64)
    r = np.searchsorted(np.unique(t), t)
    sid = _contiguous([tuple(coords[d][:-1]) for d in range(n)])
    return DetectorLayout(r, sid, "coordinates")


def layout_from_measurements(circuit: stim.Circuit) -> DetectorLayout:
    """For circuits without coordinates: measurement instructions that follow a
    DETECTOR open a new layer (= round); a detector's spatial id is the set of
    qubits whose measurement records it compares."""
    meas_layer: List[int] = []
    meas_qubit: List[int] = []
    det_layer: List[int] = []
    det_qubits: List[FrozenSet[int]] = []
    layer = -1
    after_detector = True
    for inst in circuit.flattened():
        name = inst.name
        if name in _MEASURE_GATES:
            if after_detector:
                layer += 1
                after_detector = False
            for tgt in inst.targets_copy():
                if not tgt.is_qubit_target:
                    raise ValueError(f"{name} with non-qubit target {tgt}")
                meas_layer.append(layer)
                meas_qubit.append(tgt.value)
        elif inst.num_measurements:
            raise ValueError(f"unsupported measurement instruction {name}")
        elif name == "DETECTOR":
            after_detector = True
            qs = set()
            for tgt in inst.targets_copy():
                if not tgt.is_measurement_record_target:
                    raise ValueError(f"DETECTOR with target {tgt}")
                k = len(meas_layer) + tgt.value
                if k < 0:
                    raise ValueError(
                        "DETECTOR references a measurement before the circuit start"
                    )
                qs.add(meas_qubit[k])
            det_layer.append(layer)
            det_qubits.append(frozenset(qs))
    if len(det_layer) != circuit.num_detectors:
        raise ValueError(
            f"walked {len(det_layer)} detectors, circuit has {circuit.num_detectors}"
        )
    r = np.asarray(det_layer, np.int64)
    r = np.searchsorted(np.unique(r), r)
    return DetectorLayout(r, _contiguous(det_qubits), "measurements")


def _record_qubit_sets(circuit: stim.Circuit) -> List[FrozenSet[int]]:
    """Per detector, the set of qubits whose measurement records it references (the stabiliser it
    watches); works for any circuit whose DETECTORs use rec[-k] targets, whatever their placement."""
    recq: List[int] = []
    out: List[FrozenSet[int]] = []
    for inst in circuit.flattened():
        if inst.name in _MEASURE_GATES:
            recq.extend(t.value for t in inst.targets_copy())
        elif inst.name == "DETECTOR":
            qs = set()
            for tgt in inst.targets_copy():
                if not tgt.is_measurement_record_target:
                    raise ValueError(f"DETECTOR with target {tgt}")
                k = len(recq) + tgt.value
                if k < 0:
                    raise ValueError(
                        "DETECTOR references a measurement before the circuit start"
                    )
                qs.add(recq[k])
            out.append(frozenset(qs))
    if len(out) != circuit.num_detectors:
        raise ValueError(
            f"walked {len(out)} detectors, circuit has {circuit.num_detectors}")
    return out


def layout_from_rounds(
        circuit: stim.Circuit,
        round_index: Optional[Sequence[int]] = None,
        round_sizes: Optional[Sequence[int]] = None) -> DetectorLayout:
    """Explicit rounds: ``round_index`` (one id per detector, relabelled to contiguous 0-based) or
    ``round_sizes`` (consecutive blocks of detectors in index order). The spatial id is the set of
    qubits whose measurement records the detector references (the same stabiliser across rounds);
    when the circuit does not expose that (no rec targets), the position within the round is used,
    which is right only when every round lists the same stabilisers in the same order."""
    n = circuit.num_detectors
    if (round_index is None) == (round_sizes is None):
        raise ValueError("give exactly one of round_index or round_sizes")
    if round_sizes is not None:
        sizes = [int(x) for x in round_sizes]
        if sum(sizes) != n or any(x <= 0 for x in sizes):
            raise ValueError(
                f"round_sizes must be positive and sum to the detector count {n} (got sum {sum(sizes)})"
            )
        r = np.repeat(np.arange(len(sizes)), sizes)
    else:
        r = np.asarray(round_index, np.int64)
        if r.shape != (n,):
            raise ValueError(
                f"round_index must have one entry per detector ({n})")
        r = np.searchsorted(np.unique(r), r)
    try:
        sid = _contiguous(_record_qubit_sets(circuit))
    except ValueError:
        pos = np.zeros(n, np.int64)
        for g in range(int(r.max()) + 1):
            idx = np.where(r == g)[0]
            pos[idx] = np.arange(len(idx))
        sid = pos
    return DetectorLayout(r, sid, "rounds")


def detector_layout(circuit: stim.Circuit) -> DetectorLayout:
    """Coordinates when the circuit has them, else the measurement-record walk."""
    try:
        return layout_from_coordinates(circuit)
    except ValueError:
        return layout_from_measurements(circuit)


@dataclass
class RoundClasses:
    """Result of ``round_classes``: ``class_ids`` (int32 [n], -1 never occurs), sizes and validation."""
    class_ids: np.ndarray
    n_rounds: int
    layout_source: str
    boundary_layers: Tuple[int, int]
    n_untied: int = 0  #: classes split into singletons because their nominal rates differed
    max_spread: float = 0.0  #: largest within-class log(max/min) of the nominal rates before untying

    @property
    def n_classes(self) -> int:
        return int(self.class_ids.max()) + 1

    @property
    def sizes(self) -> np.ndarray:
        return np.bincount(self.class_ids, minlength=self.n_classes)

    def __str__(self) -> str:
        s = self.sizes
        return (
            f"{len(self.class_ids)} mechanisms -> {self.n_classes} time-translation classes (max size {s.max()}, "
            f"{int((s == 1).sum())} singletons, {int((s[self.class_ids] > 1).sum())} mechanisms tied); "
            f"{self.n_rounds} rounds ({self.layout_source}), boundary_layers={self.boundary_layers}, "
            f"{self.n_untied} classes untied (nominal spread {self.max_spread:.2e})"
        )


def round_classes(circuit: stim.Circuit,
                  boundary_layers=2,
                  dem: Optional[stim.DetectorErrorModel] = None,
                  round_index: Optional[Sequence[int]] = None,
                  round_sizes: Optional[Sequence[int]] = None,
                  tol: float = 1e-9) -> RoundClasses:
    """Group the DEM columns of ``circuit`` into time-translation classes.

    The round of a detector comes from its coordinates (last entry = time) or from the measurement
    layers; circuits that expose neither (all DETECTORs after the measurements, 1-D coordinates, e.g.
    the relay bicycle circuits) take the rounds explicitly: ``round_index`` (one round id per detector)
    or ``round_sizes`` (detectors per round in index order, e.g. ``[72] + [144] * 11 + [72]``); the
    stabiliser identity across rounds then comes from the measurement records each detector references.
    The classes are validated against the circuit's nominal rates either way (a class whose members
    differ in nominal rate is split), so a wrong round assignment shows up as ``n_untied`` > 0.

    For a column with detector rows ``rr``: ``fr`` = its earliest round; the
    signature is the sorted tuple of ``(round - fr, spatial_id)``; the class
    key adds the observable set and the boundary flags ``(min(fr, b_prep),
    min(last - lr, b_read))``: a column within ``b_prep`` rounds of the prep
    boundary (``b_read`` of the readout boundary) keeps its distance to that
    boundary, everything else is bulk. ``boundary_layers`` is an int (both
    sides) or a ``(prep, readout)`` pair; 2 is right for surface-code memory
    circuits (the first detector layer has one stabiliser type only, and the
    readout side has a noiseless final round). Columns without detectors are
    keyed by their observables. The classes are validated against the
    circuit's nominal rates: a class whose members differ by more than
    ``tol`` in log rate is split into singletons (``n_untied``).
    """
    if dem is None:
        dem = circuit_dem(circuit)
    dets_l, obs_l, p_ref = dem_columns(dem)
    n = len(dets_l)
    layout = (layout_from_rounds(circuit, round_index, round_sizes) if
              (round_index is not None or
               round_sizes is not None) else detector_layout(circuit))
    if len(layout.round_index) != dem.num_detectors:
        raise ValueError(
            f"layout has {len(layout.round_index)} detectors, the DEM {dem.num_detectors}"
        )
    if layout.n_rounds < 2:
        raise ValueError(
            "round tying needs a round structure (>= 2 detector layers) but the circuit's coordinates / "
            "measurement layers give one: pass round_index= (one round id per detector) or round_sizes= "
            "(detectors per round in index order), or use mode='per_mechanism'")
    last = layout.n_rounds - 1
    if isinstance(boundary_layers, (tuple, list)):
        b_prep, b_read = int(boundary_layers[0]), int(boundary_layers[1])
    else:
        b_prep = b_read = int(boundary_layers)
    keys: Dict[Hashable, int] = {}
    cls = np.empty(n, np.int64)
    rounds, sids = layout.round_index, layout.spatial_id
    for c in range(n):
        rr = dets_l[c]
        obs = tuple(obs_l[c])
        if not rr:
            key = ((), obs, None)
        else:
            r = rounds[rr]
            fr, lr = int(r.min()), int(r.max())
            sig = tuple(sorted(zip((r - fr).tolist(), sids[rr].tolist())))
            key = (sig, obs, (min(fr, b_prep), min(last - lr, b_read)))
        cls[c] = keys.setdefault(key, len(keys))
    # validation against the nominal rates; violating classes -> singletons
    G = len(keys)
    mx = np.full(G, -np.inf)
    mn = np.full(G, np.inf)
    np.maximum.at(mx, cls, p_ref)
    np.minimum.at(mn, cls, p_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.log(mx / mn)
    spread = np.where(np.isfinite(spread), spread, np.inf)
    viol = np.where(spread > tol)[0]
    nxt = G
    for g in viol:
        idx = np.where(cls == g)[0]
        for j in idx[1:]:
            cls[j] = nxt
            nxt += 1
    return RoundClasses(cls.astype(np.int32),
                        layout.n_rounds,
                        layout.source, (b_prep, b_read),
                        n_untied=int(len(viol)),
                        max_spread=float(spread.max()) if G else 0.0)
