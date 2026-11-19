#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tesseract long beam (unsparsified, results from the tesseract-decoder repository) vs TCBP-JOBS on the Tesseract SI1000 benchmark.

Three steps, each a subcommand:

  python3 tradeoff.py fetch                      # circuits + aggregated_results.jsonl from quantumlib/tesseract-decoder
  python3 tradeoff.py run    --gpus 0,1,3        # TCBP-JOBS batched on every circuit, the repository's shot count per circuit
  python3 tradeoff.py serial --gpu 1             # TCBP-JOBS one shot at a time on a subset, for the x axis (idle machine)
  python3 tradeoff.py plot                       # plots/tradeoff_longbeam_vs_tcbp_jobs_p0.002_Z.{pdf,png}

Tesseract side: the ``sparsify_errors=false`` rows of ``benchmarking/sparsify_errors/aggregated_results.jsonl``
(long beam: --beam 20 --beam-climbing --num-det-orders 21 --pqlimit 1e6 --no-revisit-dets --det-order-index).
``total_time_seconds`` there is Tesseract's per-shot decode time summed over worker threads, so divided by
``num_shots`` it is the time per shot on one CPU thread. The CPU model is not recorded in that data.

TCBP-JOBS side: ``nv-qldpc-decoder`` from CUDA-Q QEC 0.9.0, preset ``tcbp-jobs-sc`` (sparse sum-product BP, 10
iterations, clip 15, fp32, repeatable; JOBS with the TCBP ordering, 16 draws, pad 8, beam 32/40/8; every shot is
searched, including the BP-converged ones), TCBP model ``init_tcbp_model(H, seed=1)``, the circuit's own undecomposed DEM. Shots are sampled here with
Stim (seed 2026) so they match the repository's in number and distribution, not identity. The LER and the GPU-hours in the
figure's panel come from this batched run (``decode_batch`` of 16,384 shots). The x position of each TCBP-JOBS point is
the mean serial latency from ``serial``: one shot per ``decode()`` call (``bp_batch_size=1``) on the first 4,096 shots
of a second seeded stream of the same circuit, on an otherwise idle machine. Without a serial record the batched time
is used and the footnote says so.

Needs: numpy, scipy, stim, matplotlib, and ``cudaq_qec`` with the nv-qldpc plugin. With a decoders-only build that
has no installed package, pass ``--build <build dir> --src <cuda-qx checkout>`` to ``run`` and the plugin is loaded
from there instead.
"""
import argparse, csv, glob, hashlib, io, json, math, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESS_REPO = "quantumlib/tesseract-decoder"
TESS_COMMIT = "cf340cb008db2fc3d541d65f51226707e13e4e03"  # circuits (as recorded in the jsonl provenance)
JSONL_URL = f"https://raw.githubusercontent.com/{TESS_REPO}/main/benchmarking/sparsify_errors/aggregated_results.jsonl"
PRESET = "tcbp-jobs-sc"
VERSION = "CUDA-Q QEC 0.9.0"
CHUNK = 16384


# ----------------------------------------------------------------------------- data selection
def google_rows(jsonl, p, basis):
    rows = [json.loads(l) for l in open(jsonl)]
    sel = [
        r for r in rows if r["physical_error_rate"] == p and r["basis"] == basis
    ]
    plain = {r["circuit_path"]: r for r in sel if not r["sparsify_errors"]}
    return sel, plain


def tag_of(circuit_path):
    name = os.path.basename(circuit_path)[:-5]
    kv = dict(x.split("=", 1) for x in name.split(",") if "=" in x)
    c = kv["c"]
    basis = c[-1]
    if "bicycle" in c:
        return f"bb_d{kv['d']}_{basis}_{kv['nkd'].strip('[]').replace(',', '_')}"
    return f"{'surface' if 'surface' in c else 'color'}_d{kv['d']}_{basis}"


# ----------------------------------------------------------------------------- fetch
def cmd_fetch(a):
    out = HERE / "circuits"
    out.mkdir(exist_ok=True)
    jsonl = out / "aggregated_results.jsonl"
    if not jsonl.exists():
        urllib.request.urlretrieve(JSONL_URL, jsonl)
    _, plain = google_rows(jsonl, a.p, a.basis)
    for cp, r in sorted(plain.items()):
        dst = out / cp  # keeps testdata/<family>/<name>.stim
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            url = f"https://raw.githubusercontent.com/{TESS_REPO}/{TESS_COMMIT}/{urllib.request.quote(cp)}"
            urllib.request.urlretrieve(url, dst)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        assert sha == r["circuit_sha256"], f"sha256 mismatch for {cp}"
        print(f"{tag_of(cp):16} {r['num_shots']:>10,} shots  ok")
    print(
        f"{len(plain)} circuits, {sum(r['num_shots'] for r in plain.values()):,} shots in the repository's unsparsified rows"
    )


# ----------------------------------------------------------------------------- run
def load_plugin(a):
    """Public path: the installed cudaq_qec package. Fallback: a decoders-only build directory."""
    try:
        import cudaq_qec as qec
        from cudaq_qec.plugins import nv_qldpc
        return qec.get_decoder, nv_qldpc
    except ImportError:
        if not (a.build and a.src):
            raise SystemExit(
                "cudaq_qec is not importable; pass --build <build dir> --src <cuda-qx checkout>"
            )
    import importlib.util
    sys.path.insert(0, str(Path(a.build) / "python"))
    os.environ.setdefault(
        "NVQLDPC_PLUGIN_LIB",
        str(
            Path(a.build) /
            "lib/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so"))
    import _qec_decoders_standalone as m
    pkg = Path(a.src) / "libs/qec/python/cudaq_qec/plugins/nv_qldpc"

    def load(name, path):
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(path.parent)]
            if path.name == "__init__.py" else None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    nvq = load("nv_qldpc", pkg / "__init__.py")
    load("nv_qldpc.tying", pkg / "tying.py")
    return m.qecrt.get_decoder, nvq


def dem_arrays(circuit):
    """Undecomposed DEM -> (H [detectors x mechanisms], O [observables x mechanisms], rates), one column per error."""
    import numpy as np, scipy.sparse as sp
    dem = circuit.detector_error_model(flatten_loops=True,
                                       approximate_disjoint_errors=True)
    rows_h, cols_h, rows_o, cols_o, rates = [], [], [], [], []
    col = 0
    for inst in dem:
        if inst.type != "error":
            continue
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        for d in dets:
            rows_h.append(d)
            cols_h.append(col)
        for o in obs:
            rows_o.append(o)
            cols_o.append(col)
        rates.append(inst.args_copy()[0])
        col += 1
    m, k = circuit.num_detectors, circuit.num_observables
    H = sp.csr_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)),
                      shape=(m, col))
    O = sp.csr_matrix((np.ones(len(rows_o), np.uint8), (rows_o, cols_o)),
                      shape=(k, col))
    return H, O, np.asarray(rates, np.float64)


def run_one(a, cp, g):
    import numpy as np, scipy.sparse as sp, stim
    get_decoder, nvq = load_plugin(a)
    tag = tag_of(cp)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    rec_path = out / f"{tag}.json"
    if rec_path.exists() and not json.load(open(rec_path))["checkpoint"]:
        print(f"{tag}: done, skipping")
        return
    circuit = stim.Circuit.from_file(str(HERE / "circuits" / cp))
    H, O, rates = dem_arrays(circuit)
    nshots = a.shots or g["num_shots"]
    t0 = time.time()
    model = nvq.init_tcbp_model(H, seed=a.model_seed)
    model_s = time.time() - t0
    opts = nvq.preset(PRESET,
                      tcbp_model=np.frombuffer(model, np.uint8).copy(),
                      error_rate_vec=rates,
                      bp_batch_size=CHUNK)
    t0 = time.time()
    dec = get_decoder("nv-qldpc-decoder", H, **opts)
    ctor_s = time.time() - t0
    gpu = subprocess.run([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
        os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ],
                         capture_output=True,
                         text=True).stdout.strip()
    rec = dict(
        circuit_path=cp,
        circuit_sha256=g["circuit_sha256"],
        tag=tag,
        basis=g["basis"],
        code_family=g["code_family"],
        distance=g["distance"],
        rounds=g["rounds"],
        num_qubits=g["num_qubits"],
        num_detectors=H.shape[0],
        num_errors_dem=H.shape[1],
        num_observables=O.shape[0],
        physical_error_rate=g["physical_error_rate"],
        decoder="nv-qldpc-decoder",
        preset=PRESET,
        options={
            k: (v if not isinstance(v, np.ndarray) else f"<{v.dtype}[{v.size}]>"
               ) for k, v in opts.items()
        },
        tcbp_model_seed=a.model_seed,
        stim_seed=a.seed,
        stim_version=stim.__version__,
        software=VERSION,
        host=socket.gethostname(),
        gpu=gpu,
        chunk=CHUNK,
        target_shots=nshots,
        google_plain_row={
            k: g[k] for k in ("num_shots", "num_errors", "num_low_confidence",
                             "total_time_seconds")
        },
        num_shots=0,
        num_errors=0,
        invalid=0,
        converged=0,
        total_time_seconds=0.0,
        chunk_us_per_shot=[],
        init_tcbp_model_s=model_s,
        ctor_s=ctor_s,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checkpoint=True)
    Hs, Os = sp.csr_matrix(H), sp.csr_matrix(O)
    sampler = circuit.compile_detector_sampler(seed=a.seed)
    fail_bits, conv_bits = [], []
    while rec["num_shots"] < nshots:
        nb = min(CHUNK, nshots - rec["num_shots"])
        det, obs = sampler.sample(nb, separate_observables=True)
        det = det.astype(np.uint8)
        t0 = time.perf_counter()
        res = dec.decode_batch(det)
        dt = time.perf_counter() - t0
        X = sp.csr_matrix((np.asarray(res.result) > 0.5).astype(np.uint8))
        valid = ((X @ Hs.T).toarray() % 2 == det).all(1)
        pred = ((X @ Os.T).toarray() % 2).astype(np.uint8)
        fails = (pred != obs.astype(np.uint8)).any(1)
        conv = np.asarray(res.converged, dtype=bool)
        rec["num_shots"] += nb
        rec["num_errors"] += int(fails.sum())
        rec["invalid"] += int((~valid).sum())
        rec["converged"] += int(conv.sum())
        rec["total_time_seconds"] += dt
        rec["chunk_us_per_shot"].append(round(dt / nb * 1e6, 1))
        rec["us_per_shot_batched"] = rec["total_time_seconds"] / rec[
            "num_shots"] * 1e6
        rec["ler"] = rec["num_errors"] / rec["num_shots"]
        fail_bits.append(fails)
        conv_bits.append(conv)
        json.dump(rec, open(rec_path, "w"), indent=1)
        print(
            f"{tag}: {rec['num_shots']:,}/{nshots:,} fails {rec['num_errors']} conv {rec['converged']/rec['num_shots']:.3f} "
            f"{dt/nb*1e6:.0f} us/shot",
            flush=True)
    np.savez_compressed(out / f"{tag}.shots.npz",
                        fail=np.concatenate(fail_bits),
                        converged=np.concatenate(conv_bits))
    rec["checkpoint"] = False
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(rec_path, "w"), indent=1)


SERIAL_SHOTS = 4096
SERIAL_WARMUP = 64


def cmd_serial(a):
    """Serial decode() latency on SERIAL_SHOTS shots per circuit, written into each batched record as rec["serial"]."""
    import numpy as np, stim
    get_decoder, nvq = load_plugin(a)
    _, plain = google_rows(HERE / "circuits" / "aggregated_results.jsonl", a.p,
                           a.basis)
    load1 = os.getloadavg()[0]
    for cp, g in sorted(plain.items()):
        tag = tag_of(cp)
        rec_path = HERE / "results" / f"{tag}.json"
        if not rec_path.exists():
            print(f"{tag}: no batched record, run `run` first")
            continue
        rec = json.load(open(rec_path))
        if "serial" in rec and not a.force:
            print(f"{tag}: serial done, skipping")
            continue
        circuit = stim.Circuit.from_file(str(HERE / "circuits" / cp))
        H, O, rates = dem_arrays(circuit)
        model = nvq.init_tcbp_model(H, seed=a.model_seed)
        opts = nvq.preset(PRESET,
                          tcbp_model=np.frombuffer(model, np.uint8).copy(),
                          error_rate_vec=rates,
                          bp_batch_size=1)
        dec = get_decoder("nv-qldpc-decoder", H, **opts)
        n = a.shots or SERIAL_SHOTS
        det, obs = circuit.compile_detector_sampler(seed=a.seed + 1).sample(
            n + SERIAL_WARMUP, separate_observables=True)
        det = det.astype(np.uint8)
        for i in range(SERIAL_WARMUP):
            dec.decode(det[i])
        ts = np.empty(n)
        fails = 0
        Os = O.tocsr()
        for j in range(n):
            i = SERIAL_WARMUP + j
            t0 = time.perf_counter()
            r = dec.decode(det[i])
            ts[j] = time.perf_counter() - t0
            x = (np.asarray(r.result) > 0.5).astype(np.uint8)
            fails += int(((Os @ x) % 2 != obs[i].astype(np.uint8)).any())
        rec["serial"] = dict(
            n=n,
            mean_us=float(ts.mean() * 1e6),
            p50_us=float(np.median(ts) * 1e6),
            p99_us=float(np.percentile(ts, 99) * 1e6),
            max_us=float(ts.max() * 1e6),
            fails=fails,
            stim_seed=a.seed + 1,
            warmup=SERIAL_WARMUP,
            bp_batch_size=1,
            load1_at_start=load1,
            gpu=subprocess.run([
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            ],
                               capture_output=True,
                               text=True).stdout.strip(),
            timer=
            "time.perf_counter around decode(), host result conversion excluded",
            when_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        json.dump(rec, open(rec_path, "w"), indent=1)
        print(
            f"{tag}: serial mean {ts.mean()*1e3:.3f} ms  p50 {np.median(ts)*1e3:.3f}  p99 {np.percentile(ts,99)*1e3:.2f}  "
            f"fails {fails}/{n}",
            flush=True)


def cmd_run(a):
    _, plain = google_rows(HERE / "circuits" / "aggregated_results.jsonl", a.p,
                           a.basis)
    if a.circuit:  # worker: one circuit on the GPU already set in CUDA_VISIBLE_DEVICES
        run_one(a, a.circuit, plain[a.circuit])
        return
    # driver: spread circuits over the GPUs by expected cost (shots x mechanisms), heaviest first
    import stim
    cost = {
        cp:
            g["num_shots"] *
            stim.Circuit.from_file(str(HERE / "circuits" / cp)).num_detectors
        for cp, g in plain.items()
    }
    gpus = a.gpus.split(",")
    queues = {g_: [] for g_ in gpus}
    load = {g_: 0.0 for g_ in gpus}
    for cp in sorted(cost, key=cost.get, reverse=True):
        g_ = min(load, key=load.get)
        queues[g_].append(cp)
        load[g_] += cost[cp]
    (HERE / "results" / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for g_, cps in queues.items():
        script = " && ".join(
            f"{sys.executable} {__file__} run --p {a.p} --basis {a.basis} --seed {a.seed} --model-seed {a.model_seed}"
            + (f" --build {a.build} --src {a.src}" if a.build else "") +
            f" --circuit '{cp}'" for cp in cps)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g_)
        log = open(HERE / "results" / "logs" / f"gpu{g_}.log", "a")
        procs.append(
            subprocess.Popen(["bash", "-c", script],
                             env=env,
                             stdout=log,
                             stderr=subprocess.STDOUT))
        print(f"GPU {g_}: {len(cps)} circuits ->",
              ", ".join(tag_of(c) for c in cps))
    for p_ in procs:
        p_.wait()
    print("all done; exit codes", [p_.returncode for p_ in procs])


# ----------------------------------------------------------------------------- plot
COLOR_MAP = {
    "surfacecodes": "#5D95E8",
    "colorcodes": "#F6C644",
    "bivariatebicyclecodes": "fuchsia"
}
DISPLAY = {
    "surfacecodes": "Surface Codes",
    "colorcodes": "Color Codes",
    "bivariatebicyclecodes": "Bicycle Codes"
}
JOBS_INK = "#5A8F00"
CONF = 0.95


def point(shots, failures, rounds, seconds_per_shot):
    from scipy.stats import binomtest
    upper = failures == 0
    if upper:
        plotted, lo, hi = 1 - (1 - CONF)**(1 / shots), 0.0, 0.0
    else:
        ci = binomtest(k=failures, n=shots).proportion_ci(confidence_level=CONF)
        plotted = failures / shots
        lo, hi = plotted - ci.low, ci.high - plotted
    return dict(failures=failures,
                shots=shots,
                is_upper_limit=upper,
                ler=plotted / rounds,
                err_low=lo / rounds,
                err_high=hi / rounds,
                time_per_round=seconds_per_shot / rounds)


def rr_label(bp, ap):
    if bp["is_upper_limit"] or ap["is_upper_limit"]:
        return "censored (95% bound)"
    ba, aa, bt, at = bp["failures"] + 0.5, ap["failures"] + 0.5, bp[
        "shots"] + 1.0, ap["shots"] + 1.0
    rr = (aa / at) / (ba / bt)
    se = math.sqrt(max(0, 1 / aa - 1 / at + 1 / ba - 1 / bt))
    lo, hi = rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se)
    return f"{lo:.2f}x err" if round(lo, 2) == round(
        hi, 2) else f"{lo:.2f}-{hi:.2f}x err"


def speed_label(r):
    return f"{r:.2f}x spd" if r < 1 else (
        f"{r:.1f}x spd" if r < 10 else f"{r:.0f}x spd")


def marker_map(rows):
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "<", ">", "P"]
    out = {}
    for fam in sorted({r["code_family"] for r in rows}):
        ids = sorted({(r["code_family"], r["distance"], r["num_qubits"],
                       r["basis"]) for r in rows if r["code_family"] == fam})
        out.update(zip(ids, markers))
    return out


def arrow_with_label(ax,
                     x0,
                     y0,
                     x1,
                     y1,
                     color,
                     text,
                     frac=0.82,
                     box_edge="none"):
    ax.annotate("",
                xy=(x1, y1),
                xytext=(x0, y0),
                zorder=3,
                arrowprops=dict(arrowstyle="->",
                                color=color,
                                lw=1.5,
                                ls=":",
                                shrinkA=8,
                                shrinkB=8))
    mx = math.exp(math.log(x0) * (1 - frac) + math.log(x1) * frac)
    my = math.exp(math.log(y0) * (1 - frac) + math.log(y1) * frac)
    t = ax.text(mx,
                my * 1.05,
                text,
                fontsize=6,
                color="black",
                ha="center",
                va="bottom",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white",
                          ec=box_edge,
                          lw=0.8,
                          alpha=0.85))
    t._arrow = (x0, y0, x1, y1)
    return t


def declutter(fig, ax, movable, points, step_px=4, max_steps=24):
    """Move each arrow label to the first collision-free spot along its arrow."""
    import matplotlib.transforms as mtrans
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    obstacles = []
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        obstacles.append(
            mtrans.Bbox.from_extents(px - 9, py - 9, px + 9, py + 9))
    heads = [
        ax.transData.transform((t._arrow[2], t._arrow[3]))[1] for t in movable
    ]
    for i, t in enumerate(movable):
        x0, y0, x1, y1 = t._arrow
        near = [
            h for j, h in enumerate(heads) if j != i and abs(h - heads[i]) < 45
        ]
        first_below = bool(near) and all(h > heads[i] for h in near)
        best, best_cost = None, None
        for frac in (0.82, 0.62, 0.45, 0.3, 0.95):
            mx = math.exp(math.log(x0) * (1 - frac) + math.log(x1) * frac)
            my = math.exp(math.log(y0) * (1 - frac) + math.log(y1) * frac)
            for below in (first_below, not first_below):
                t.set_va("top" if below else "bottom")
                bx, by = ax.transData.transform(
                    (mx, my * (0.95 if below else 1.05)))
                for k in range(max_steps + 1):
                    dy = (k + 1) // 2 * step_px * (1 if k %
                                                   2 else -1) if k else 0
                    t.set_position(ax.transData.inverted().transform(
                        (bx, by + dy)))
                    bb = t.get_window_extent(r).expanded(1.05, 1.15)
                    if bb.y0 < axbb.y0 or bb.y1 > axbb.y1 or bb.x0 < axbb.x0 or bb.x1 > axbb.x1:
                        continue
                    cost = sum(
                        max(0,
                            min(bb.x1, q.x1) - max(bb.x0, q.x0)) *
                        max(0,
                            min(bb.y1, q.y1) - max(bb.y0, q.y0))
                        for q in obstacles) + 0.02 * abs(dy) + 30 * abs(frac -
                                                                        0.82)
                    if best_cost is None or cost < best_cost:
                        best, best_cost = (t.get_position(), t.get_va()), cost
                    if cost < 1.0:
                        break
                if best_cost is not None and best_cost < 1.0:
                    break
            if best_cost is not None and best_cost < 1.0:
                break
        t.set_position(best[0])
        t.set_va(best[1])
        obstacles.append(t.get_window_extent(r).expanded(1.05, 1.15))


def sketch_outline(ax,
                   verts,
                   color,
                   label=None,
                   label_at=None,
                   lw=2.2,
                   fill_alpha=0.07):
    import numpy as np
    from scipy.interpolate import splprep, splev
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MPath
    v = np.asarray(verts)
    if len(v) >= 4:
        tck, _ = splprep([v[:, 0], v[:, 1]], s=0, per=True)
        xs, ys = splev(np.linspace(0, 1, 200), tck)
        v = np.c_[xs, ys]
    path = MPath(np.vstack([v, v[:1]]), closed=True)
    ax.add_patch(
        mpatches.PathPatch(path,
                           transform=ax.transAxes,
                           facecolor=color,
                           alpha=fill_alpha,
                           edgecolor="none",
                           zorder=1.5))
    line = mpatches.PathPatch(path,
                              transform=ax.transAxes,
                              facecolor="none",
                              edgecolor=color,
                              lw=lw,
                              zorder=1.6,
                              capstyle="round")
    line.set_sketch_params(scale=1.3, length=120.0, randomness=4.0)
    line.set_clip_on(True)
    ax.add_patch(line)
    if label:
        ax.text(*label_at,
                label,
                transform=ax.transAxes,
                color=color,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white",
                          edgecolor="none",
                          alpha=0.85))
    return v


def padded_hull(pts, pad):
    import numpy as np
    from scipy.spatial import ConvexHull
    P = np.asarray(pts)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    disc = np.c_[np.cos(th), np.sin(th)] * pad
    Q = (P[:, None, :] + disc[None, :, :]).reshape(-1, 2)
    return [tuple(q) for q in Q[ConvexHull(Q).vertices]]


def cmd_plot(a):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.lines as mlines
    from matplotlib.path import Path as MPath
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
    sel, plain = google_rows(HERE / "circuits" / "aggregated_results.jsonl",
                             a.p, a.basis)
    mmap = marker_map(sel)
    recs = {}
    for f in glob.glob(str(HERE / "results" / "*.json")):
        r = json.load(open(f))
        if not r["checkpoint"] and r["physical_error_rate"] == a.p and r[
                "basis"] == a.basis:
            recs[r["circuit_path"]] = r
    missing = sorted(set(plain) - set(recs))
    if missing:
        print("WARNING: no finished TCBP-JOBS record for",
              [tag_of(c) for c in missing])
    fig, ax = plt.subplots(figsize=(10, 8))
    tess_hours = jobs_hours = 0.0
    tess_shots = jobs_shots = 0
    tess_pts, jobs_pts, pts, texts, all_ler = [], [], [], [], []
    any_upper = False
    n_serial = 0
    for cp, g in sorted(plain.items()):
        fam, color = g["code_family"], COLOR_MAP[g["code_family"]]
        marker = mmap[(fam, g["distance"], g["num_qubits"], g["basis"])]
        rounds = g["rounds"]
        tess_hours += g["total_time_seconds"] / 3600
        tess_shots += g["num_shots"]
        bp = point(g["num_shots"], g["num_errors"] + g["num_low_confidence"],
                   rounds, g["total_time_seconds"] / g["num_shots"])
        x0, y0 = bp["time_per_round"], bp["ler"]
        all_ler.append(y0)
        any_upper |= bp["is_upper_limit"]
        if not bp["is_upper_limit"]:
            ax.errorbar([x0], [y0],
                        yerr=[[bp["err_low"]], [bp["err_high"]]],
                        fmt="none",
                        ecolor="black",
                        alpha=0.3,
                        zorder=1)
        ax.scatter([x0], [y0],
                   facecolors="white",
                   edgecolors="black",
                   marker="v" if bp["is_upper_limit"] else marker,
                   s=80,
                   linewidths=1.5,
                   zorder=4)
        pts.append((x0, y0))
        tess_pts.append((x0, y0))
        r = recs.get(cp)
        if r is None:
            continue
        jobs_hours += r["total_time_seconds"] / 3600
        jobs_shots += r["num_shots"]
        ser = r.get("serial")
        sec = ser["mean_us"] * 1e-6 if ser else r["total_time_seconds"] / r[
            "num_shots"]
        n_serial += bool(ser)
        jp = point(r["num_shots"], r["num_errors"], rounds, sec)
        x2, y2 = jp["time_per_round"], jp["ler"]
        all_ler.append(y2)
        any_upper |= jp["is_upper_limit"]
        if not jp["is_upper_limit"]:
            ax.errorbar([x2], [y2],
                        yerr=[[jp["err_low"]], [jp["err_high"]]],
                        fmt="none",
                        ecolor=color,
                        alpha=0.9,
                        zorder=2)
        ax.scatter([x2], [y2],
                   facecolors=color,
                   edgecolors="black",
                   marker="v" if jp["is_upper_limit"] else marker,
                   s=90,
                   linewidths=1.0,
                   zorder=5)
        texts.append(
            arrow_with_label(ax,
                             x0,
                             y0,
                             x2,
                             y2,
                             color,
                             f"{speed_label(x0 / x2)}\n{rr_label(bp, jp)}",
                             box_edge=color))
        pts.append((x2, y2))
        jobs_pts.append((x2, y2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=max(min(all_ler) / 5.0, 1e-6))
    ax.set_xlabel("Time per round (seconds)")
    ax.set_ylabel("Shot failure probability ÷ rounds")
    ax.text(0.5,
            0.03,
            "← lower is better",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="dimgray",
            zorder=0)
    ax.text(0.02,
            0.5,
            "← lower is better",
            transform=ax.transAxes,
            ha="left",
            va="center",
            rotation=90,
            fontsize=10,
            color="dimgray",
            zorder=0)
    ax.set_title(f"Tesseract long beam vs TCBP-JOBS (p={a.p}, {a.basis} basis)")
    fig.text(
        0.01,
        -0.015,
        "Tesseract long beam: --beam 20 --beam-climbing --num-det-orders 21 --pqlimit 1e6 --no-revisit-dets, no sparsification, "
        "CPU time per shot on one thread (results on the tesseract-decoder repository)\n"
        f"TCBP-JOBS ({VERSION}, preset {PRESET}): BP10 sum-product + JOBS search, TCBP ordering, converged shots searched; one GB200.\n"
        +
        (f"    x = mean serial decode() latency, one shot per call, {SERIAL_SHOTS:,} shots per circuit; LER and GPU-hours from the "
         f"batched run (decode_batch of {CHUNK:,} shots)"
         if n_serial == len(recs) else
         f"x = batched decode_batch ({CHUNK:,} shots) wall time per shot" +
         (f" ({n_serial}/{len(recs)} circuits have serial records; "
          "run `serial` for all)" if n_serial else "")),
        fontsize=7,
        color="dimgray",
        ha="left",
        va="top")
    # compute panel
    tp = lambda c: dict(color=c, fontsize=8.5, fontweight="semibold")
    name_box = lambda c: dict(
        boxstyle="square,pad=0.25", facecolor="white", edgecolor=c, lw=1.4)
    tess_name = TextArea("Tesseract long beam",
                         textprops=dict(**tp("black"), bbox=name_box("black")))
    jobs_name = TextArea("TCBP-JOBS",
                         textprops=dict(**tp(JOBS_INK),
                                        bbox=name_box(JOBS_INK)))
    rows_ = [
        TextArea("Compute behind the points shown",
                 textprops=dict(color="black", fontsize=8.5,
                                fontweight="bold")),
        HPacker(children=[
            jobs_name,
            TextArea(
                f": {jobs_hours:.2f} GPU-hours, one GB200 ({jobs_shots / 1e6:.1f}M shots, batched"
                + ("; points at serial latency)"
                   if n_serial == len(recs) else ")"),
                textprops=tp(JOBS_INK))
        ],
                pad=0,
                sep=2,
                align="baseline"),
        HPacker(children=[
            tess_name,
            TextArea(
                f": {tess_hours:,.0f} CPU core-hours ({tess_shots / 1e6:.1f}M shots)",
                textprops=tp("black"))
        ],
                pad=0,
                sep=2,
                align="baseline")
    ]
    ab = AnchoredOffsetbox(loc="upper right",
                           child=VPacker(children=rows_,
                                         align="left",
                                         pad=2,
                                         sep=9),
                           pad=0.5,
                           borderpad=0.6,
                           frameon=True)
    ab.patch.set(facecolor="white", edgecolor="#AAAAAA", lw=0.8, alpha=0.95)
    ab.set_zorder(10)
    ax.add_artist(ab)
    # legend
    legend = []
    for fam in ["surfacecodes", "colorcodes", "bivariatebicyclecodes"]:
        ids = sorted({(r["distance"], r["num_qubits"], r["basis"])
                      for r in sel
                      if r["code_family"] == fam})
        if not ids:
            continue
        legend.append(
            mlines.Line2D([0], [0], color="none", label=f"  {DISPLAY[fam]}"))
        for d, q, b in ids:
            legend.append(
                mlines.Line2D([0], [0],
                              color="none",
                              marker=mmap[(fam, d, q, b)],
                              markerfacecolor=COLOR_MAP[fam],
                              markeredgecolor="none",
                              markersize=8,
                              label=f"d={d}, q={q}, {b}"))
    legend += [
        mlines.Line2D([0], [0], color="none", label=""),
        mlines.Line2D([0], [0],
                      color="none",
                      marker="o",
                      linestyle="None",
                      markerfacecolor="white",
                      markeredgecolor="black",
                      markersize=8,
                      markeredgewidth=1.5,
                      label="Tesseract long beam (CPU)"),
        mlines.Line2D([0], [0],
                      color="none",
                      marker="o",
                      linestyle="None",
                      markerfacecolor="gray",
                      markeredgecolor="black",
                      markersize=8,
                      label="TCBP-JOBS (GPU)")
    ]
    if any_upper:
        legend.append(
            mlines.Line2D([0], [0],
                          color="gray",
                          marker="v",
                          linestyle="None",
                          markersize=8,
                          label="95% upper limit (0 failures)"))
    ax.legend(handles=legend,
              loc="center left",
              bbox_to_anchor=(1, 0.5),
              fontsize=8,
              labelspacing=0.6)
    fig.tight_layout()
    declutter(fig, ax, texts, pts)
    # outlines + arrows from the panel
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 10**(0.22 * math.log10(hi / lo)))
    to_ax = lambda ps: [
        tuple(ax.transAxes.inverted().transform(ax.transData.transform(p)))
        for p in ps
    ]
    T, J = np.asarray(to_ax(tess_pts)), np.asarray(to_ax(jobs_pts))
    pad = 0.045
    jhull = padded_hull(J, pad)
    jpath = MPath(np.vstack([jhull, jhull[:1]]), closed=True)
    inside = np.array(
        [jpath.contains_point(q) or q[0] < J[:, 0].max() + pad for q in T])
    thull = padded_hull(T[~inside], pad)
    th = np.asarray(thull)
    jh = np.asarray(jhull)
    tout = sketch_outline(ax, thull, "black", "Tesseract long beam (CPU)",
                          (float(th[:, 0].mean()), float(th[:, 1].mean())))
    jout = sketch_outline(
        ax, jhull, JOBS_INK, "TCBP-JOBS (GPU)",
        (float(jh[:, 0].mean()), float(jh[:, 1].min()) - 0.035))
    for q in T[inside]:
        rr = 0.022
        sketch_outline(ax,
                       [(q[0] + rr * np.cos(t), q[1] + 1.25 * rr * np.sin(t))
                        for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)],
                       "black",
                       lw=1.6,
                       fill_alpha=0.0)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    for area, hull, col, mode in ((tess_name, tout, "black", "down"),
                                  (jobs_name, jout, JOBS_INK, "side")):
        bb = area._text.get_bbox_patch().get_window_extent(rend).transformed(
            inv)
        Hh = np.asarray(hull)
        if mode == "down":
            st = ((bb.x0 + bb.x1) / 2, bb.y0)
            tg = tuple(Hh[np.argmin(
                np.hypot(Hh[:, 0] - st[0], 3 * (Hh[:, 1] - st[1])))])
        else:
            st = (bb.x0, (bb.y0 + bb.y1) / 2)
            crown = Hh[Hh[:, 1] > Hh[:, 1].max() - 0.015]
            tg = tuple(crown[np.argmax(crown[:, 0])])
        ax.annotate("",
                    xy=tg,
                    xytext=st,
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>",
                                    color=col,
                                    lw=1.8,
                                    shrinkA=0,
                                    shrinkB=2),
                    zorder=11)
    out = HERE / "plots" / f"tradeoff_longbeam_vs_tcbp_jobs_p{a.p}_{a.basis}"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    # per-circuit CSV next to the plot
    with open(str(out) + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tag", "shots", "tess_fails", "tess_ler", "tess_s_per_shot",
            "jobs_fails", "jobs_ler", "jobs_us_per_shot_batched",
            "jobs_us_per_shot_serial_mean", "jobs_serial_shots",
            "ler_ratio_tess_over_jobs", "time_ratio_tess_over_jobs_x_axis"
        ])
        for cp, g in sorted(plain.items()):
            r = recs.get(cp)
            if r is None:
                continue
            tf = g["num_errors"] + g["num_low_confidence"]
            ser = r.get("serial")
            xs = ser["mean_us"] * 1e-6 if ser else r["total_time_seconds"] / r[
                "num_shots"]
            w.writerow([
                tag_of(cp), r["num_shots"], tf, tf / g["num_shots"],
                g["total_time_seconds"] / g["num_shots"], r["num_errors"],
                r["ler"], r["us_per_shot_batched"],
                ser["mean_us"] if ser else "", ser["n"] if ser else "",
                (tf / g["num_shots"]) / r["ler"] if r["ler"] else float("inf"),
                (g["total_time_seconds"] / g["num_shots"]) / xs
            ])
    print(
        f"wrote {out}.pdf/.png/.csv  Tesseract {tess_hours:,.0f} core-h, TCBP-JOBS {jobs_hours:.2f} GPU-h, "
        f"{len(recs)}/{len(plain)} circuits")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "run", "serial", "plot"])
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--gpus",
                    default="0",
                    help="run: comma-separated CUDA device indices")
    ap.add_argument(
        "--shots",
        type=int,
        default=None,
        help=
        "run: override the repository's shot count (testing); serial: shots per circuit"
    )
    ap.add_argument(
        "--gpu",
        default=None,
        help="serial: CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="serial: redo circuits that already have a serial record")
    ap.add_argument("--seed",
                    type=int,
                    default=2026,
                    help="run: Stim sampler seed")
    ap.add_argument("--model-seed",
                    type=int,
                    default=1,
                    help="run: init_tcbp_model seed")
    ap.add_argument(
        "--build",
        default=None,
        help=
        "run: cuda-qx build dir (decoders-only build without an installed package)"
    )
    ap.add_argument("--src",
                    default=None,
                    help="run: cuda-qx checkout (with --build)")
    ap.add_argument("--circuit", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cmd == "serial" and a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    {
        "fetch": cmd_fetch,
        "run": cmd_run,
        "serial": cmd_serial,
        "plot": cmd_plot
    }[a.cmd](a)


if __name__ == "__main__":
    main()
