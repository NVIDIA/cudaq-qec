# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This demo demonstrates decoding color-code memory experiments from the
# Ising-Decoder repository using the CUDA-Q QEC decoding server. It calculates
# logical error rate (LER) per round vs. decode runtime per round, for raw
# Chromobius and for the Ising predecoder + Chromobius, on color-code memory
# experiments of distance d with d rounds.
#
# Steps:
#   * prepare: per distance, write the detector error model (DEM), export the
#              Ising predecoder to ONNX, and build its TensorRT engine.
#   * run:     per distance, replay one schedule of sampled shots through each
#              decoder and session: inproc (decoder in the emulator process)
#              and server (a decoding_server over UDP).
#   * plot:    draw the figure from the collected results.

# [Begin Documentation]
"""Ising color-code decoder: LER per round vs. runtime on the playback emulator.

Usage: ./run_color_code_demo.sh [options]
"""
import argparse, contextlib, json, os, re, shutil, subprocess, sys, warnings
import numpy as np
import cudaq_qec as qec

pb = qec.playback

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ising",
                    required=True,
                    help="Ising-Decoding checkout code.")
parser.add_argument("--weights",
                    help="Ising-Decoder-ColorCode-1-Fast .safetensors file "
                    "(needed by the prepare step)")
parser.add_argument("--steps",
                    default="prepare,run,plot",
                    help="comma-separated list of: prepare, run, plot")
parser.add_argument("--distances", default="5,7,9,11,13")
parser.add_argument("--p", type=float, default=1e-3, help="physical error rate")
parser.add_argument("--shots",
                    type=int,
                    default=100000,
                    help="shots per distance")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--decoding-server",
                    default=shutil.which("decoding_server"),
                    help="decoding_server binary (default: from PATH)")
parser.add_argument("--server-timeout",
                    type=int,
                    default=7200,
                    help="decoding_server --timeout, in s")
args = parser.parse_args()

steps = args.steps.split(",")
distances = [int(d) for d in args.distances.split(",")]
if "run" in steps and not args.decoding_server:
    parser.error("decoding_server not found; pass --decoding-server")
if "prepare" in steps and not args.weights:
    parser.error("the prepare step needs --weights")
sys.path.insert(0, os.path.join(args.ising, "code"))

from qec.color_code.color_code import ColorCode
from qec.color_code.reference_superdense_noise import build_color_memory_circuit

# Shots in flight: shot i starts once shot i - DEPTH has its correction.
DEPTH = 4
MEASURE_OPS = {"M", "MZ", "MX", "MY", "MR", "MRX", "MRY", "MRZ"}


def color_circuit(d):
    """Z-basis color-code memory circuit with d rounds, as in the model card."""
    return build_color_memory_circuit(distance=d,
                                      n_rounds=d,
                                      basis="Z",
                                      p_error=args.p,
                                      noise_model_family="legacy",
                                      noise_instruction_semantics="current",
                                      gidney_style_noise=True,
                                      schedule="nearest-neighbor",
                                      add_boundary_detectors=True).stim_circuit


def measurement_maps(circuit, d):
    """Returns (m2d, support): the measurements each detector reads, and the
    data qubits the observable reads. Stim's converter reveals both when
    measurements are flipped one at a time."""
    n = circuit.num_measurements
    flips = np.vstack([np.zeros((1, n), bool), np.eye(n, dtype=bool)])
    dets, obs = circuit.compile_m2d_converter().convert(
        measurements=flips, separate_observables=True)
    dets, obs = dets[1:] ^ dets[0], obs[1:, 0] ^ obs[0, 0]
    data_start = n - ColorCode(d).num_data
    return ([np.flatnonzero(col).tolist() for col in dets.T],
            np.flatnonzero(obs[data_start:]).tolist())


def export_onnx(d, model, cfg, support, onnx_path):
    """Exports the Ising predecoder, with the observable's support baked in.
    `cfg` is the model's Ising config; its test section sets the sampling."""
    import torch
    from qec.color_code.detector_input import ColorDetectorInputTransform
    from benchmarks.export_detector_input_trtexec import (
        DetectorInputColorEval, DetectorInputModel)
    from evaluation.logical_error_rate_color import (
        PreDecoderColorEvalModule, _build_color_code_parity_maps)
    xform = ColorDetectorInputTransform(distance=d, rounds=d, basis="Z")
    maps = _build_color_code_parity_maps(d)
    obs_support = torch.zeros(int(maps["num_data"]))
    obs_support[support] = 1.0
    pipeline = DetectorInputColorEval(
        DetectorInputModel(xform, model),
        PreDecoderColorEvalModule(model,
                                  cfg,
                                  maps,
                                  basis="Z",
                                  obs_support=obs_support,
                                  num_boundary_dets=int(
                                      xform.num_stabs))).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(pipeline,
                          torch.zeros(1, xform.detector_width),
                          onnx_path,
                          opset_version=17,
                          input_names=["dets"],
                          output_names=["L_and_residual_dets"],
                          dynamic_axes={
                              "dets": {
                                  0: "batch"
                              },
                              "L_and_residual_dets": {
                                  0: "batch"
                              }
                          },
                          dynamo=False)


def decoder_config(name, d, m2d, engine_args=None):
    """A one-decoder config. The decoder reads its matrices from the DEM file;
    D_sparse maps the raw measurements to detectors."""
    dc = qec.decoder_config()
    dc.type = "chromobius" if name == "chromobius" else "trt_decoder"
    dc.D_sparse = qec.d_sparse(m2d)
    dc.stim_dem_path = "dem_d%d.txt" % d
    if name == "composed":
        # The predecoder returns its observable flips and residual detectors;
        # Chromobius decodes the residual.
        dc.decoder_custom_args = {
            "batch_size": 1,
            "engine_output_format": "observables_and_residual_detectors",
            "global_decoder": "chromobius",
            **(engine_args or {
                "engine_load_path": "engine_d%d.trt" % d
            })
        }
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc


def write_schedule(circuit, shots, sched_path, chunk=10000):
    """Samples `circuit` and writes a playback schedule.

    Each shot is a reset, one enqueue per measurement layer, and a get_corrections 
    that expects the shot's true observable flips; the emulator flags replies that 
    differ. Shot i waits (after=) on the signal raised by shot i - DEPTH, so at 
    most DEPTH shots are in flight. Every line's trigger is '-': send as soon as
    the line before it has been sent.
    """
    layers, offset = [], 0
    for inst in circuit.flattened():
        if inst.name in MEASURE_OPS:
            width = len(inst.targets_copy())
            layers.append((offset, width))
            offset += width
    sampler = circuit.compile_sampler(seed=args.seed)
    converter = circuit.compile_m2d_converter()
    with open(sched_path, "w") as f:
        for start in range(0, shots, chunk):
            meas = sampler.sample(min(chunk, shots - start))
            obs = converter.convert(measurements=meas,
                                    separate_observables=True)[1]
            text = (meas.astype(np.uint8) + ord("0")).view("S1")
            obs_text = (obs.astype(np.uint8) + ord("0")).view("S1")
            for i, row in enumerate(text):
                s = start + i
                after = " after=s%d" % (s - DEPTH) if s >= DEPTH else ""
                f.write("- reset%s\n" % after)
                for off, w in layers:
                    f.write("- enqueue source=0b%s\n" %
                            row[off:off + w].tobytes().decode())
                f.write("- get_corrections %s signal=s%d\n" %
                        (obs_text[i].tobytes().decode(), s))


def per_round(ler, rounds):
    """Per-round logical error rate from the rate over `rounds` rounds."""
    return float(1 - (1 - np.clip(ler, 1e-12, 1 - 1e-12))**(1 / rounds))


@contextlib.contextmanager
def decoding_server(mdc):
    """Runs a decoding_server for `mdc`; yields the port it reports."""
    cfg_path = "server_config.yaml"
    with open(cfg_path, "w") as f:
        f.write(mdc.to_yaml_str())
    proc = subprocess.Popen([
        args.decoding_server, "--config=" + cfg_path, "--transport=udp",
        "--port=0",
        "--timeout=%d" % args.server_timeout
    ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    try:
        port = next((
            int(m.group(1))
            for line in proc.stdout
            for m in [re.search(r"QEC_DECODING_SERVER_READY port=(\d+)", line)]
            if m), None)
        if port is None:
            raise RuntimeError("decoding_server did not start")
        yield port
    finally:
        proc.terminate()
        proc.wait()


def run_checked(schedule, shots, **session):
    """Runs `schedule`; fails unless every shot returned its correction."""
    result = pb.run(schedule, tick_ns=0, sources={}, **session)
    gc = [r for r in result.records if r.op == pb.operation.get_corrections]
    bad = [
        str(r.status)
        for r in gc
        if str(r.status) != "OK" or r.correction_count == 0
    ]
    if len(gc) != shots or bad:
        raise RuntimeError("%d/%d corrections returned, failures: %s" %
                           (len(gc), shots, sorted(set(bad))))
    return result, gc


@contextlib.contextmanager
def session(name, mdc):
    """Yields the pb.run() arguments that route requests through `name`."""
    if name == "inproc":
        yield dict(decoders=mdc)
    else:
        with decoding_server(mdc) as port:
            yield dict(udp_endpoints={0: "127.0.0.1:%d" % port},
                       udp_timeout_ms=5000)


results_path = "results.json"
results = json.load(open(results_path)) if os.path.exists(results_path) else {}

if "prepare" in steps:
    from export.safetensors_utils import load_safetensors, _build_minimal_cfg
    model, metadata = load_safetensors(args.weights, device="cpu")
    model = model.float().eval()
    cfg = _build_minimal_cfg(metadata["model_id"])
    for d in distances:
        circuit = color_circuit(d)
        m2d, support = measurement_maps(circuit, d)
        with open("dem_d%d.txt" % d, "w") as f:
            f.write(str(circuit.detector_error_model()))
        if not os.path.exists("pre_d%d.onnx" % d):
            export_onnx(d, model, cfg, support, "pre_d%d.onnx" % d)
        if not os.path.exists("engine_d%d.trt" % d):
            # Build and cache the TensorRT engine in a short in-process run.
            write_schedule(circuit, 100, "schedule.txt")
            run_checked(open("schedule.txt").read(),
                        100,
                        decoders=decoder_config(
                            "composed", d, m2d, {
                                "onnx_load_path": "pre_d%d.onnx" % d,
                                "engine_save_path": "engine_d%d.trt" % d
                            }))
        print("d=%-2d %d detectors" % (d, circuit.num_detectors), flush=True)

if "run" in steps:
    for d in distances:
        circuit = color_circuit(d)
        m2d = measurement_maps(circuit, d)[0]
        write_schedule(circuit, args.shots, "schedule.txt")
        schedule = open("schedule.txt").read()
        for dec in ("chromobius", "composed"):
            for sess in ("inproc", "server"):
                with session(sess, decoder_config(dec, d, m2d)) as kw:
                    result, gc = run_checked(schedule, args.shots, **kw)
                # Runtime per round: the time until the last reply, divided by
                # the rounds decoded.
                runtime = max(
                    r.return_ns for r in result.records) / 1e3 / args.shots / d
                errors = sum(r.correction_mismatch for r in gc)
                # Per-round rate from the d-round rate, with a binomial error
                # bar.
                total = min(max(errors / args.shots, 0.5 / args.shots),
                            1 - 1e-12)
                sigma = np.sqrt(total * (1 - total) / args.shots)
                lo, mid, hi = (per_round(x, d)
                               for x in (total - sigma, total, total + sigma))
                point = results.setdefault(str(d), {}).setdefault(dec, {})
                point[sess] = dict(runtime=runtime,
                                   ler=mid,
                                   ler_unc=max(hi - mid, mid - lo),
                                   errors=errors)
                print("d=%-2d %-10s %-6s %8.3f us/round  LER/round %.3g +/- "
                      "%.1g (%d/%d)" %
                      (d, dec, sess, runtime, mid, point[sess]["ler_unc"],
                       errors, args.shots),
                      flush=True)
        json.dump(results, open(results_path, "w"), indent=1)

if "plot" in steps:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {
        "chromobius": "Raw Chromobius",
        "composed": "Ising-Decoder-ColorCode-1-Fast"
    }
    colors = {"chromobius": "#808080", "composed": "#76b900"}
    styles = {
        "server": ("--", "o", "server, UDP"),
        "inproc": (":", "o", "inproc")
    }
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for dec in labels:
        for sess, (style, marker, sess_label) in styles.items():
            pts = sorted((int(d), r[dec][sess])
                         for d, r in results.items()
                         if sess in r.get(dec, {}))
            if not pts:
                continue
            x = [p["runtime"] for _, p in pts]
            y = [p["ler"] for _, p in pts]
            ax.errorbar(x,
                        y,
                        yerr=[p["ler_unc"] for _, p in pts],
                        color=colors[dec],
                        linestyle=style,
                        marker=marker,
                        markersize=7,
                        linewidth=2.5,
                        capsize=3,
                        label="%s (%s)" % (labels[dec], sess_label))
            # Hollow markers: fewer than 25 logical errors observed.
            thin = [(xi, yi)
                    for (_, p), xi, yi in zip(pts, x, y)
                    if p["errors"] < 25]
            if thin:
                ax.scatter(*zip(*thin),
                           s=52,
                           zorder=4,
                           facecolors="white",
                           edgecolors=colors[dec],
                           linewidths=2.5)
            if sess == "server":
                for (d, _), xi, yi in zip(pts, x, y):
                    ax.annotate("d=%d" % d, (xi, yi),
                                textcoords="offset points",
                                xytext=(9, 4),
                                color=colors[dec])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Runtime ($\mu$s / round)")
    ax.set_ylabel("Logical error rate per round")
    ax.set_title(r"$p = %g$ (Z basis)" % args.p)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    xlo, xhi = ax.get_xlim()
    ylo, _ = ax.get_ylim()
    ax.set_xlim(xlo / 1.5, xhi * 2.6)
    xlo = ax.get_xlim()[0]
    ax.annotate("Faster",
                xy=(xlo * 1.35, ylo * 2.0),
                xytext=(xlo * 5.5, ylo * 2.0),
                va="center",
                arrowprops=dict(arrowstyle="->", lw=1.6))
    fig.tight_layout()
    fig.savefig("color_code_demo.png", dpi=150)
    print("wrote", "color_code_demo.png")
# [End Documentation]
