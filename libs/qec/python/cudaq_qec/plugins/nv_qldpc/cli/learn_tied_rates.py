# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""One shared rate update per time-translation class (learn_noise mode tied; classes from tying.round_classes).

    python -m cudaq_qec.plugins.nv_qldpc.cli.learn_tied_rates circuit.stim -N 16384 [--boundary-layers 2] [--round-sizes 72 144 ...]
"""
from __future__ import annotations

import time

import numpy as np

from ..tying import round_classes
from ._common import Inputs, make_parser, pass_records, write_json


def main(argv=None) -> int:
    ap = make_parser(
        "Learn round-tied DEM rates (shared ratio per time-translation class).")
    ap.add_argument(
        "--boundary-layers",
        type=int,
        nargs="+",
        default=[2],
        help=
        "detector layers near the prep / readout boundary that keep their position (one value, or 'prep readout')"
    )
    ap.add_argument(
        "--round-sizes",
        type=int,
        nargs="+",
        default=None,
        help=
        "detectors per round in index order, for circuits whose coordinates carry no round structure"
    )
    args = ap.parse_args(argv)
    inp = Inputs(args, "tied_rates")
    bl = tuple(args.boundary_layers) if len(
        args.boundary_layers) == 2 else args.boundary_layers[0]
    t0 = time.perf_counter()
    rc = round_classes(inp.circuit,
                       boundary_layers=bl,
                       dem=inp.dem,
                       round_sizes=args.round_sizes)
    t_cls = time.perf_counter() - t0
    print(f"{rc} ({t_cls:.1f}s)", flush=True)
    t0 = time.perf_counter()
    rates, report, _ = inp.learn(mode="tied", class_ids=rc.class_ids)
    wall = time.perf_counter() - t0
    sizes = rc.sizes
    print(
        f"{report['passes_run']} passes in {wall:.1f}s{inp.error_line(rates)}; drift check over {int((sizes > 1).sum())} tied classes: "
        f"|residual| p90 {report['tied_resid_p90']:.3f} max {report['tied_resid_max']:.3f}",
        flush=True)
    write_json(
        inp.out_path(),
        dict(scenario="tied_rates",
             circuit=args.circuit,
             device_data=inp.source,
             n_shots=int(inp.dets.shape[0]),
             n_mechanisms=int(inp.H.shape[1]),
             truth_mass=inp.truth_mass,
             wall_s=wall,
             classes_s=t_cls,
             setup_s=inp.t_setup,
             layout=dict(source=rc.layout_source, n_rounds=rc.n_rounds),
             classes=dict(
                 n_classes=rc.n_classes,
                 boundary_layers=rc.boundary_layers,
                 n_untied=rc.n_untied,
                 sizes_hist={
                     int(k): int(v)
                     for k, v in zip(*np.unique(sizes, return_counts=True))
                 }),
             passes=pass_records(report),
             report=report,
             **inp.truth_fields(rates),
             p_hat=rates,
             truth_rates=inp.truth_rates,
             class_id=rc.class_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
