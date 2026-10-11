# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""One rate per DEM mechanism (learn_noise mode per_mechanism).

    python -m cudaq_qec.plugins.nv_qldpc.cli.learn_dem_rates circuit.stim -N 16384
"""
from __future__ import annotations

import time

from ._common import Inputs, make_parser, pass_records, write_json


def main(argv=None) -> int:
    args = make_parser("Learn one rate per DEM mechanism.").parse_args(argv)
    inp = Inputs(args, "dem_rates")
    t0 = time.perf_counter()
    rates, report, _ = inp.learn()
    wall = time.perf_counter() - t0
    print(
        f"{report['passes_run']} passes in {wall:.1f}s{inp.error_line(rates)}",
        flush=True)
    write_json(
        inp.out_path(),
        dict(scenario="dem_rates",
             circuit=args.circuit,
             device_data=inp.source,
             n_shots=int(inp.dets.shape[0]),
             n_mechanisms=int(inp.H.shape[1]),
             truth_mass=inp.truth_mass,
             wall_s=wall,
             setup_s=inp.t_setup,
             passes=pass_records(report),
             report=report,
             **inp.truth_fields(rates),
             p_hat=rates,
             truth_rates=inp.truth_rates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
