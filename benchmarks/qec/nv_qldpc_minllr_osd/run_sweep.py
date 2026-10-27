# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LER-vs-physical-error-rate sweep: BP+OSD with the default final-LLR OSD
# ordering versus BP + osd_init_method="min_llr". See README.md for usage.
#
# Every arm decodes the same sampled syndromes (paired comparison). An arm
# stops after it has accumulated STOP_FAILS logical failures or MAX_SHOTS
# shots, whichever comes first; results are appended as JSON lines and as
# per-shot flag arrays so the plots can recompute Wilson intervals. Re-running
# with a different --seed appends an independent sample that plot_sweep.py
# pools with the earlier one (used to extend low-LER points).
import argparse, glob, json, os, sys, time
import numpy as np, stim, cudaq_qec as qec

ARMS = {
    # name: (max_iterations, osd_method, osd_order, osd_init_method)
    "BP60+OSD-CS10": (60, 3, 10, "final_llr"),
    "BP10-minLLR+OSD-CS10": (10, 3, 10, "min_llr"),
    "BP60+OSD-0": (60, 1, 0, "final_llr"),
    "BP10-minLLR+OSD-0": (10, 1, 0, "min_llr"),
    # matched-iteration controls: each changes only one factor relative to the
    # two arms above, so the effect of osd_init_method can be separated from
    # the effect of max_iterations.
    "BP10+OSD-CS10": (10, 3, 10, "final_llr"),  # BP10, default ordering
    "BP60-minLLR+OSD-CS10": (60, 3, 10, "min_llr"),  # BP60, min-LLR ordering
}


def load_circuit(spec, p):
    """spec is either a path to a stim file whose noise arguments are all the
    literal (0.002) (the assets/benchmarks Z-only circuits) -- rescaled to p by
    substitution -- or a glob with {p} for a per-rate circuit family."""
    if "{p}" in spec:
        fs = glob.glob(spec.format(p=p))
        if len(fs) != 1:
            raise SystemExit(f"expected one circuit for {spec} p={p}, got {fs}")
        return stim.Circuit.from_file(fs[0])
    txt = open(spec).read()
    if "(0.002)" not in txt:
        raise SystemExit(f"{spec}: expected literal (0.002) noise arguments")
    return stim.Circuit(txt.replace("(0.002)", f"({p})"))


def dem_mats(c):
    dem = c.detector_error_model()
    H = np.zeros((dem.num_detectors, dem.num_errors), np.uint8)
    L = np.zeros((dem.num_observables, dem.num_errors), np.uint8)
    er = np.zeros(dem.num_errors)
    e = 0
    for ins in dem.flattened():
        if ins.type == "error":
            er[e] = ins.args_copy()[0]
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    H[t.val, e] = 1
                elif t.is_logical_observable_id():
                    L[t.val, e] = 1
            e += 1
    return H, L, er


def make_decoder(H, L, er, arm):
    max_iter, osd_method, osd_order, init = ARMS[arm]
    return qec.get_decoder("nv-qldpc-decoder",
                           H,
                           error_rate_vec=er,
                           use_sparsity=True,
                           use_osd=True,
                           osd_method=osd_method,
                           osd_order=osd_order,
                           osd_init_method=init,
                           max_iterations=max_iter,
                           bp_batch_size=2048,
                           O=L,
                           output="observables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="DEM label used in output")
    ap.add_argument(
        "--circuit",
        required=True,
        help="stim file with literal (0.002) noise, or glob with {p}")
    ap.add_argument("--rates", required=True, help="comma-separated p values")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated")
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--max-shots", type=int, required=True)
    ap.add_argument("--stop-fails", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arms = a.arms.split(",")
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; choose from {list(ARMS)}")
    jsonl = os.path.join(a.out, f"{a.label}.jsonl")

    for p in a.rates.split(","):
        c = load_circuit(a.circuit, p)
        H, L, er = dem_mats(c)
        sampler = c.compile_detector_sampler(seed=a.seed)
        n_batches = -(-a.max_shots // a.batch)
        dets, obss = [], []
        decoders = {arm: make_decoder(H, L, er, arm) for arm in arms}
        flags = {arm: [] for arm in arms}
        conv = {arm: [] for arm in arms}
        secs = {arm: 0.0 for arm in arms}
        active = set(arms)
        for b in range(n_batches):
            if not active:
                break
            det, obs = sampler.sample(a.batch, separate_observables=True)
            det = det.astype(np.uint8)
            obs = obs.astype(np.uint8)
            for arm in list(active):
                t = time.time()
                res = decoders[arm].decode_batch(det)
                secs[arm] += time.time() - t
                pred = np.array([r.result for r in res], np.uint8)
                flags[arm].append(np.any(pred != obs, axis=1))
                conv[arm].append(np.array([r.converged for r in res], bool))
                fails = int(sum(f.sum() for f in flags[arm]))
                shots = sum(len(f) for f in flags[arm])
                print(
                    f"{a.label} p={p} {arm}: {fails}/{shots} fails, "
                    f"{secs[arm]:.0f}s",
                    flush=True)
                if fails >= a.stop_fails:
                    active.discard(arm)
        for arm in arms:
            f = np.concatenate(flags[arm])
            cv = np.concatenate(conv[arm])
            rec = dict(label=a.label,
                       p=float(p),
                       arm=arm,
                       shots=int(len(f)),
                       fails=int(f.sum()),
                       ler=float(f.mean()),
                       bp_converged=int(cv.sum()),
                       decode_sec=round(secs[arm], 1),
                       num_detectors=int(H.shape[0]),
                       num_errors=int(H.shape[1]),
                       num_observables=int(L.shape[0]),
                       seed=a.seed,
                       max_shots=a.max_shots,
                       stop_fails=a.stop_fails)
            with open(jsonl, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            np.savez_compressed(os.path.join(
                a.out, f"{a.label}_p{p}_{arm}_seed{a.seed}.npz"),
                                fail=f,
                                bp_converged=cv)
        del decoders


if __name__ == "__main__":
    main()
