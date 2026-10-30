# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set up the TCBP model of a code family: stim circuit in, TCBP model out (init_tcbp_model()).

    python -m cudaq_qec.plugins.nv_qldpc.cli.init_tcbp_model circuit.stim --out surface.tcbp
    python -m ...cli.init_tcbp_model circuit.stim --out new.tcbp --validate --reference old.tcbp

--validate runs one learn_noise pass on a simulated device stream with the new TCBP model and, with --reference, with a previous one,
and compares the mass per shot (they should agree within 5%; exit code 2 otherwise).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import stim

from .. import init_tcbp_model, learn_noise
from ..tying import DEM_KWARGS
from ._common import dem_arrays, simulate_dem

MASS_BIAS_TOL = 0.05


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("circuit", help="stim circuit of the code family (.stim)")
    ap.add_argument("--out",
                    required=True,
                    help="destination TCBP model file (.tcbp)")
    ap.add_argument("--seed", type=int, default=1, help="training seed")
    ap.add_argument(
        "--min-p",
        type=float,
        default=None,
        help=
        "lower end of the training-rate range (default: the package default)")
    ap.add_argument("--max-p",
                    type=float,
                    default=None,
                    help="upper end of the training-rate range")
    ap.add_argument("--n-shots",
                    type=int,
                    default=None,
                    help="training shots (default: error-rate aware)")
    ap.add_argument("--steps",
                    type=int,
                    default=None,
                    help="fitting steps (default 25000)")
    ap.add_argument(
        "--validate",
        action="store_true",
        help=
        "run one learn_noise pass with the new TCBP model on a simulated stream"
    )
    ap.add_argument(
        "--reference",
        default=None,
        help="previous TCBP model to compare against under --validate")
    ap.add_argument("-N",
                    "--shots",
                    type=int,
                    default=8192,
                    help="device shots of the --validate pass")
    ap.add_argument("--dev-seed",
                    type=int,
                    default=5550,
                    help="seed of the simulated device stream")
    ap.add_argument("--quiet", action="store_true")
    return ap


def one_pass(H, truth_rates, tcbp_model, dets):
    t0 = time.perf_counter()
    rates, report = learn_noise(H,
                                dets,
                                np.frombuffer(tcbp_model, np.uint8).copy(),
                                passes=1,
                                tol=0,
                                advanced={"holdout_report": False})
    e = np.abs(np.log(np.asarray(rates) / truth_rates))  # scoring only
    return dict(mass_dev=report["pass_mass_dev"][0],
                mass_ref=report["pass_mass_ref"][0],
                med_log_err=float(np.median(e)),
                p90_log_err=float(np.percentile(e, 90)),
                wall_s=time.perf_counter() - t0)


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    circuit = stim.Circuit.from_file(args.circuit)
    dem = circuit.detector_error_model(**DEM_KWARGS)
    H, _, truth_rates = dem_arrays(dem)  # rates: evaluation only
    print(
        f"circuit {args.circuit}: {H.shape[0]} detectors, {H.shape[1]} mechanisms; fitting the TCBP model, seed {args.seed}",
        flush=True)
    kw = dict(seed=args.seed, verbosity=0 if args.quiet else 1)
    if args.min_p is not None:
        kw["min_p"] = args.min_p
    if args.max_p is not None:
        kw["max_p"] = args.max_p
    if args.n_shots:
        kw["shots"] = args.n_shots
    if args.steps:
        kw["steps"] = args.steps
    tcbp_model, report = init_tcbp_model(H, return_report=True, **kw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "wb").write(tcbp_model)
    print(
        f"wrote {args.out}: {len(tcbp_model)} bytes, BCE {report['bce_model']:.5f} (baseline {report['bce_baseline']:.5f}), "
        f"{report['shots_used']} shots, {report['wall_s']:.1f}s",
        flush=True)
    if not args.validate:
        return 0
    dets = simulate_dem(dem, args.shots, args.dev_seed)
    new = one_pass(H, truth_rates, tcbp_model, dets)
    print(
        f"validate (one pass, {args.shots} simulated shots): new model mass/shot {new['mass_dev']:.4f} (circuit {truth_rates.sum():.4f}), predicted mass/shot "
        f"{new['mass_ref']:.4f}, med|log err| {new['med_log_err']:.4f} (p90 {new['p90_log_err']:.4f}), {new['wall_s']:.1f}s",
        flush=True)
    if args.reference is None:
        return 0
    ref = one_pass(H, truth_rates, open(args.reference, "rb").read(), dets)
    bias = new["mass_dev"] / ref["mass_dev"] - 1.0
    verdict = "ok" if abs(
        bias) <= MASS_BIAS_TOL else f"MISMATCH (> {MASS_BIAS_TOL:.0%})"
    print(
        f"reference {args.reference}: mass/shot {ref['mass_dev']:.4f}, med|log err| {ref['med_log_err']:.4f}; new/ref mass bias {bias:+.2%} -> {verdict}",
        flush=True)
    return 0 if abs(bias) <= MASS_BIAS_TOL else 2


if __name__ == "__main__":
    raise SystemExit(main())
