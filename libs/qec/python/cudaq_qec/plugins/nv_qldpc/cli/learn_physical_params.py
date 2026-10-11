# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Physical noise parameters, rates = A q (learn_noise mode parametric).

    python -m ...learn_physical_params --print-tags > tags.json          # sentinel values for the tag circuit
    python -m ...learn_physical_params circuit.stim --tag-circuit tag.stim --yaml noise_model.yaml -N 16384
    python -m ...learn_physical_params circuit.stim -N 16384              # generic design: one parameter per (gate, args)

The tag design (25 parameters of the surface-code noise model) needs the memory circuit built once from the tag noise
model.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import stim

from .. import p25
from ._common import Inputs, make_parser, pass_records, write_json


def _names(design):
    return list(design["names"])


def _report_line(report) -> str:
    ident = report.get("q_identifiable", [])
    return f"identifiability: rank {report['design_rank']} / {report['n_params']}, {int(np.sum(ident))} of {report['n_params']} parameters individually identifiable"


def main(argv=None) -> int:
    ap = make_parser("Learn physical noise parameters (rates = A q).",
                     circuit_required=False)
    ap.add_argument(
        "--tag-circuit",
        default=None,
        help=
        "the memory circuit built from the tag noise model (--print-tags); enables the 25-parameter design"
    )
    ap.add_argument(
        "--tags",
        default=None,
        help=
        "JSON of the tag values used to build --tag-circuit (default: p25.tag_noise_model())"
    )
    ap.add_argument(
        "--yaml",
        default=None,
        help=
        "write the 25 parameters as a data.noise_model YAML fragment (tag design only)"
    )
    ap.add_argument("--print-tags",
                    action="store_true",
                    help="print p25.tag_noise_model() as JSON and exit")
    ap.add_argument(
        "--family-tags",
        action="store_true",
        help=
        "with --print-tags: also print the family tags for circuits whose idle / CNOT noise is DEPOLARIZE1 / DEPOLARIZE2"
    )
    argv = sys.argv[1:] if argv is None else argv
    if "--print-tags" in argv:
        print(
            json.dumps(p25.tag_noise_model(families="--family-tags" in argv),
                       indent=1))
        return 0
    args = ap.parse_args(argv)
    if not args.circuit:
        ap.error("a circuit file is required")
    inp = Inputs(args, "physical_params")
    t0 = time.perf_counter()
    if args.tag_circuit:
        tags = json.load(open(
            args.tags)) if args.tags else p25.tag_noise_model()
        design = p25.design_map(stim.Circuit.from_file(args.tag_circuit),
                                tags,
                                target_dem=inp.dem)
        kind = "p25 tag circuit"
    else:
        design = p25.design_map_by_gate(inp.circuit, inp.dem)
        kind = "generic (gate, args)"
    t_design = time.perf_counter() - t0
    print(
        f"design {kind}: {inp.H.shape[1]} mechanisms x {design['n_params']} parameters, {t_design:.1f}s",
        flush=True)
    t0 = time.perf_counter()
    rates, report, q_hat = inp.learn(mode="parametric", design=design)
    wall = time.perf_counter() - t0
    named = dict(zip(_names(design), q_hat.tolist()))
    print(
        f"{_report_line(report)}\n{report['passes_run']} passes in {wall:.1f}s{inp.error_line(rates)}",
        flush=True)
    for n, v in named.items():
        print(f"  {n:32s} {v:.6g}")
    payload = dict(scenario="physical_params",
                   design=kind,
                   circuit=args.circuit,
                   tag_circuit=args.tag_circuit,
                   device_data=inp.source,
                   n_shots=int(inp.dets.shape[0]),
                   n_mechanisms=int(inp.H.shape[1]),
                   truth_mass=inp.truth_mass,
                   wall_s=wall,
                   design_s=t_design,
                   setup_s=inp.t_setup,
                   param_names=_names(design),
                   q_hat=q_hat,
                   q_named=named,
                   rank=report["design_rank"],
                   q_identifiable=report["q_identifiable"],
                   passes=pass_records(report),
                   report=report,
                   **inp.truth_fields(rates),
                   p_hat=rates,
                   truth_rates=inp.truth_rates)
    if args.tag_circuit:
        exported = p25.export_p25(q_hat, _names(design))
        payload["noise_model"] = exported
        if args.yaml:
            with open(args.yaml, "w") as f:
                f.write(
                    p25.to_yaml_fragment(exported,
                                         identifiable=report["q_identifiable"]))
            print(f"wrote {args.yaml}", flush=True)
    elif args.yaml:
        print("--yaml needs --tag-circuit (25-parameter design); skipped",
              flush=True)
    write_json(inp.out_path(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
