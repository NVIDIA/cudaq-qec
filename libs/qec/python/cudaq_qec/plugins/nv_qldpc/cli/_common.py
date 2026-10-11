# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared plumbing of the CLI commands: the argument parser, the inputs (circuit -> DEM matrices and nominal rates; device data
from a .dets + metadata pair, a .npy / .npz array, or a DEM simulation; the calibration data), the learn_noise keyword set,
per-pass records and the JSON writer."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import stim

from .. import learn_noise  # the package API
from ..io import circuit_structure, read_detector_samples
from ..tying import DEM_KWARGS, dem_arrays

TRUTH_EXACT, TRUTH_NOMINAL = "err", "deviation from nominal"


def make_parser(description: str,
                circuit_required: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("circuit",
                    nargs=None if circuit_required else "?",
                    default=None,
                    help="noisy stim circuit file (.stim)")
    ap.add_argument(
        "--dets",
        default=None,
        help=
        "device detector data: a stim shot file (needs --metadata) or a [N, n_det] array "
        "(.npy, or .npz with key 'dets'); default: simulate the circuit's DEM")
    ap.add_argument(
        "--metadata",
        default=None,
        help=
        "metadata_{basis}.json of --dets; its distance, rounds, basis and counts are checked against the circuit"
    )
    ap.add_argument(
        "--code-rotation",
        default=None,
        help=
        "orientation (XV/XH/ZV/ZH or O1-O4) the circuit was built with; checked against --metadata when given"
    )
    ap.add_argument(
        "--calibration",
        required=True,
        help="calibration data file written by the calibrate command")
    ap.add_argument("-N",
                    "--shots",
                    type=int,
                    default=16384,
                    help="device shots to use (simulated, or prefix of --dets)")
    ap.add_argument("--passes", type=int, default=4, help="maximum passes")
    ap.add_argument(
        "--tol",
        type=float,
        default=0.1,
        help=
        "stop when a pass moves the rates by less than this median |log change| (0 = always run --passes)"
    )
    ap.add_argument("--chunk",
                    type=int,
                    default=1024,
                    help="shots per device chunk (memory)")
    ap.add_argument("--seed",
                    type=int,
                    default=5550,
                    help="seed of the simulated device stream")
    ap.add_argument("--ref-seed",
                    type=int,
                    default=0,
                    help="seed of the simulated shots")
    ap.add_argument("--n-ref",
                    type=int,
                    default=None,
                    help="simulated shots per pass; default max(N, 32768)")
    ap.add_argument(
        "--truth",
        default="auto",
        choices=["auto", "exact", "nominal", "none"],
        help=
        "what the learned rates are compared with: 'exact' = the circuit's rates (simulated device only), 'nominal' = the circuit's "
        "rates as a reference model (reported as deviation_from_nominal), 'none'; auto = exact without --dets, none with --dets"
    )
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="output JSON (default: <circuit stem>.<scenario>.json)")
    return ap


def simulate_dem(dem: stim.DetectorErrorModel,
                 n_shots: int,
                 seed: int,
                 slab: int = 65536) -> np.ndarray:
    """Detector samples of the DEM (independent mechanisms through H), uint8 [n_shots, n_det]."""
    sampler = dem.compile_sampler(seed=seed)
    out = np.empty((n_shots, dem.num_detectors), np.uint8)
    done = 0
    while done < n_shots:
        k = min(slab, n_shots - done)
        d, _, _ = sampler.sample(k)
        out[done:done + k] = d
        done += k
    return out


class Inputs:
    """Circuit, DEM structure, device data, calibration data. The circuit's own rates are kept as ``truth_rates``
    for scoring only; they are never passed to learning."""

    def __init__(self,
                 args: argparse.Namespace,
                 scenario: str,
                 *,
                 circuit: Optional[str] = None,
                 dets: Optional[str] = None,
                 metadata: Optional[str] = None,
                 seed: Optional[int] = None,
                 label: str = ""):
        self.args = args
        self.scenario = scenario
        circuit = args.circuit if circuit is None else circuit
        if circuit is None:
            raise SystemExit("a circuit file is required")
        dets = args.dets if dets is None else dets
        metadata = args.metadata if metadata is None else metadata
        seed = args.seed if seed is None else seed
        self.circuit_path = circuit
        self.label = label
        t0 = time.perf_counter()
        self.circuit = stim.Circuit.from_file(circuit)
        self.dem = self.circuit.detector_error_model(**DEM_KWARGS)
        self.H, self.O, self.truth_rates = dem_arrays(
            self.dem)  # evaluation only
        self.truth_mass = float(self.truth_rates.sum())
        self.calibration_path = Path(args.calibration)
        self.calibration = open(self.calibration_path, "rb").read()
        if dets and (metadata or str(dets).endswith(".dets")):
            if not metadata:
                raise SystemExit(
                    "--dets <file>.dets needs --metadata <metadata.json>")
            expect = circuit_structure(self.circuit)
            if args.code_rotation:
                expect["code_rotation"] = args.code_rotation
            data, meta = read_detector_samples(dets,
                                               metadata,
                                               H=self.H,
                                               O=self.O,
                                               expect=expect)
            self.dets = np.ascontiguousarray(data[:args.shots], np.uint8)
            self.source = f"file {dets} (d{meta['distance']} r{meta['n_rounds']} {meta['basis']} {meta['code_rotation']}, {meta['num_shots']} shots)"
        elif dets:
            z = np.load(dets)
            data = z["dets"] if hasattr(z, "files") else z
            if data.shape[1] != self.H.shape[0]:
                raise SystemExit(
                    f"--dets has {data.shape[1]} detectors, the circuit {self.H.shape[0]}"
                )
            self.dets = np.ascontiguousarray(data[:args.shots], np.uint8)
            self.source = f"file {dets}"
        else:
            self.dets = simulate_dem(self.dem, args.shots, seed)
            self.source = f"simulated from the circuit's DEM, seed {seed}"
        truth = args.truth
        if truth == "auto":
            truth = "none" if dets else "exact"
        if truth == "exact" and dets:
            raise SystemExit(
                "--truth exact: the true rates of device data are unknown; use --truth nominal to report the deviation from the nominal model instead"
            )
        self.truth = truth
        self.p_true = self.truth_rates if truth in ("exact",
                                                    "nominal") else None
        self.truth_label = TRUTH_NOMINAL if truth == "nominal" else TRUTH_EXACT
        self.t_setup = time.perf_counter() - t0
        tag = f"[{label}] " if label else ""
        print(
            f"{tag}circuit {circuit}: {self.H.shape[0]} detectors, {self.H.shape[1]} mechanisms, circuit mass {self.truth_mass:.3f} (evaluation only); device data: "
            f"{self.dets.shape[0]} shots ({self.source}); calibration {self.calibration_path}; setup {self.t_setup:.1f}s",
            flush=True)

    def em_kwargs(self) -> Dict[str, Any]:
        a = self.args
        kw = dict(passes=a.passes,
                  tol=a.tol,
                  seed=a.ref_seed,
                  verbosity=1,
                  advanced={
                      "chunk": a.chunk,
                      "holdout_report": False
                  })
        if a.n_ref is not None:
            kw["reference_shots"] = a.n_ref
        kw["calibration"] = np.frombuffer(self.calibration, np.uint8).copy()
        return kw

    def learn(self, **extra):
        """learn_noise on this input's H and shots (no rates); returns (rates, report, q_hat or None)."""
        out = learn_noise(self.H, self.dets, **self.em_kwargs(), **extra)
        rates, report = out
        q_hat = report.get("q_hat")
        return np.asarray(rates), report, (np.asarray(q_hat)
                                           if q_hat is not None else None)

    def error_line(self, rates) -> str:
        if self.p_true is None:
            return ""
        e = np.abs(np.log(rates / self.p_true))
        return f"  med|log {self.truth_label}| {np.median(e):.4f}  p90 {np.percentile(e, 90):.4f}"

    def truth_fields(self, rates) -> Dict[str, Any]:
        out = dict(truth=self.truth)
        if self.p_true is not None:
            e = np.abs(np.log(rates / self.p_true))
            key = "err" if self.truth == "exact" else "deviation_from_nominal"
            out[f"med_log_{key}"] = float(np.median(e))
            out[f"p90_log_{key}"] = float(np.percentile(e, 90))
        return out

    def out_path(self) -> Path:
        return Path(self.args.out) if self.args.out else Path(
            Path(self.circuit_path).stem + f".{self.scenario}.json")


def pass_records(report: Dict[str, Any]) -> list:
    """One record per pass from the learn_noise report (the rates the passes went through are not kept, so the
    comparison with the truth is reported once, at the end)."""
    n = int(report["passes_run"])
    keys = ("pass_mass_dev", "pass_mass_ref", "pass_med_log_change",
            "pass_max_log_change_q")
    return [
        dict(it=i,
             n_dev=int(report["device_shots"]),
             n_ref=int(report["reference_shots"]),
             **{
                 k[5:]: float(report[k][i])
                 for k in keys
                 if k in report and len(report[k]) > i
             })
        for i in range(n)
    ]


def write_json(path: Path, payload: Dict[str, Any]) -> None:

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=conv)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)", flush=True)
