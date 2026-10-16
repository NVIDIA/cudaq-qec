# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Plots LER vs physical error rate from the JSON lines written by run_sweep.py
# and prints the per-point table (fails/shots, LER, Wilson 95% interval, ratio
# to the min_llr arm). See README.md.
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "report_data"
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Categorical palette slots 1-4, fixed order.
STYLE = {
    "BP60+OSD-CS10":
        dict(color="#eb6834", ls="-", lw=2.0, alpha=1.0, marker="o"),
    "BP10-minLLR+OSD-CS10":
        dict(color="#2a78d6", ls="-", lw=2.4, alpha=1.0, marker="o"),
    "BP60+OSD-0":
        dict(color="#eda100", ls="-", lw=2.0, alpha=1.0, marker="o"),
    "BP10-minLLR+OSD-0":
        dict(color="#1baf7a", ls="-", lw=2.4, alpha=1.0, marker="o"),
    # matched-iteration controls (dashed, in the hue of the arm sharing their
    # OSD ordering); plotted only in the controls figure
    "BP10+OSD-CS10":
        dict(color="#eb6834", ls="--", lw=1.8, alpha=0.6, marker="s"),
    "BP60-minLLR+OSD-CS10":
        dict(color="#2a78d6", ls="--", lw=1.8, alpha=0.6, marker="s"),
}
FIGURES = {
    "minllr_osd_cs10_ler": ["BP60+OSD-CS10", "BP10-minLLR+OSD-CS10"],
    "minllr_osd_0_ler": ["BP60+OSD-0", "BP10-minLLR+OSD-0"],
}
# Only these labels are plotted (in this order); other labels in the data
# directory are still tabulated.
TITLES = {
    "bb72_fullZ": "[[72,12,6]] joint XYZ",
    "bb144_fullZ": "[[144,12,12]] joint XYZ",
    "bb72_Zonly": "[[72,12,6]] split X/Z",
    "bb144_Zonly": "[[144,12,12]] split X/Z",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(c - h, 0.0), min(c + h, 1.0))


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*.jsonl"))):
    rows += [json.loads(l) for l in open(f) if l.strip()]
# label -> arm -> list of (p, fails, shots); records for the same (label, arm,
# p) -- e.g. extension runs with different seeds -- are pooled.
acc = defaultdict(lambda: [0, 0])
for r in rows:
    acc[(r["label"], r["arm"], r["p"])][0] += r["fails"]
    acc[(r["label"], r["arm"], r["p"])][1] += r["shots"]
data = defaultdict(dict)
for (label, arm, p), (k, n) in acc.items():
    data[label].setdefault(arm, []).append((p, k, n))
labels = [l for l in TITLES if l in data]
extra = sorted(set(data) - set(TITLES))

# ---- table -----------------------------------------------------------------
ref = "BP10-minLLR+OSD-CS10"
for label in labels + extra:
    print(f"\n== {label}")
    print(
        f"{'p':>7} {'arm':24} {'fails/shots':>14} {'LER':>9} {'Wilson 95%':>22} {'x vs min_llr CS10':>18}"
    )
    refd = {p: (k, n) for p, k, n in data[label].get(ref, [])}
    for arm in list(STYLE) + sorted(set(data[label]) - set(STYLE)):
        for p, k, n in sorted(data[label].get(arm, [])):
            lo, hi = wilson(k, n)
            ratio = ""
            if p in refd and arm != ref:
                rk, rn = refd[p]
                ratio = f"{(k / n) / (rk / rn):.1f}x" if rk else f">{(k / n) / (1 / rn):.0f}x"
            print(f"{p:>7} {arm:24} {f'{k}/{n}':>14} {k / n:>9.2e} "
                  f"{f'[{lo:.1e}, {hi:.1e}]':>22} {ratio:>18}")

# ---- figures -----------------------------------------------------------------
ROWS = [("OSD-CS10", ["BP60+OSD-CS10", "BP10-minLLR+OSD-CS10"]),
        ("OSD-0", ["BP60+OSD-0", "BP10-minLLR+OSD-0"])]
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "legend.handlelength": 4.0
})

ZERO_FAIL_POINTS = []


def draw(ax, label, arm, name, ls=None, alpha=None):
    """One LER-vs-p curve with a Wilson band; zero-failure points become open
    down-triangles at the 95% upper bound."""
    pts = sorted(data[label].get(arm, []))
    if not pts:
        return []
    st = STYLE[arm]
    ls = st["ls"] if ls is None else ls
    alpha = st["alpha"] if alpha is None else alpha
    ps = np.array([p for p, _, _ in pts])
    ler = np.array([k / n_ for _, k, n_ in pts])
    ci = np.array([wilson(k, n_) for _, k, n_ in pts])
    pos = ler > 0
    ax.plot(ps[pos],
            ler[pos],
            color=st["color"],
            ls=ls,
            lw=st["lw"],
            alpha=alpha,
            marker=st["marker"],
            ms=6,
            label=name)
    ax.fill_between(ps[pos],
                    ci[pos, 0],
                    ci[pos, 1],
                    color=st["color"],
                    alpha=0.12 * alpha,
                    lw=0)
    if (~pos).any():
        ax.plot(ps[~pos],
                ci[~pos, 1],
                color=st["color"],
                ls="none",
                marker="v",
                ms=8,
                alpha=alpha,
                mfc="none")
        ZERO_FAIL_POINTS.append((label, arm))
    return list(ps)


def finish(ax, title, allp):
    ax.set_xscale("log")
    ax.set_yscale("log")
    allp = sorted(set(allp))
    ax.set_xticks(allp)
    ax.set_xticklabels([f"{p:g}" for p in allp], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(title)
    ax.grid(True, which="both", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False)


def save(fig, axes, name):
    for axrow in axes:
        axrow[0].set_ylabel("logical error rate per shot")
    for ax in axes[-1]:
        ax.set_xlabel("physical error rate p")
    if ZERO_FAIL_POINTS:
        fig.text(
            0.995,
            0.005,
            "open triangles: 95% Wilson upper bound at zero observed failures",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=160)
    print("wrote", out)


# Figure 1: one panel per DEM, OSD-CS10 arms (used on the docs page).
# Figure 1b: the same with the OSD-0 arms (collected, kept in report_data only).
ncol = len(labels)
for (row_name,
     arms), fname in zip(ROWS, ("minllr_osd_ler.png", "minllr_osd_0_ler.png")):
    fig, axes = plt.subplots(1,
                             ncol,
                             figsize=(4.2 * ncol, 4.6),
                             sharey=True,
                             squeeze=False)
    for ax, label in zip(axes[0], labels):
        allp = []
        for arm in arms:
            allp += draw(ax, label, arm, arm)
        finish(ax, f"{TITLES.get(label, label)} -- {row_name}", allp)
    save(fig, axes, fname)

# Figure 1c: matched-iteration controls, one panel per DEM, OSD-CS10 only.
# Legend order = curve order at low p: BP60 / BP10 default ordering, then
# BP10 / BP60 min-LLR.
CTRL = [
    "BP60+OSD-CS10", "BP10+OSD-CS10", "BP10-minLLR+OSD-CS10",
    "BP60-minLLR+OSD-CS10"
]
if any(a in data[l] for l in labels for a in CTRL[1::2]):
    fig, axes = plt.subplots(1,
                             ncol,
                             figsize=(4.4 * ncol, 4.8),
                             sharey=True,
                             squeeze=False)
    for ax, label in zip(axes[0], labels):
        allp = []
        for arm in CTRL:
            allp += draw(ax, label, arm, arm)
        finish(ax, f"{TITLES.get(label, label)} -- OSD-CS10", allp)
    save(fig, axes, "minllr_osd_controls.png")

# Figure 2: correlated (joint XYZ) vs uncorrelated (split X/Z) decoding of the
# same memory-Z circuit, per code, OSD-CS10 only; one column per code.
CODES = [("[[72,12,6]]", "bb72_fullZ", "bb72_Zonly"),
         ("[[144,12,12]]", "bb144_fullZ", "bb144_Zonly")]
CODES = [c for c in CODES if c[1] in data or c[2] in data]
if CODES:
    fig, axes = plt.subplots(1,
                             len(CODES),
                             figsize=(5.2 * len(CODES), 5.0),
                             sharey=True,
                             squeeze=False)
    for (row_name, arms), axrow in zip(ROWS[:1], axes):
        for ax, (code, joint, split) in zip(axrow, CODES):
            allp = []
            # legend order = top-to-bottom order of the curves at low p:
            # baseline joint, baseline split, min-LLR split, min-LLR joint
            base, minllr = arms
            allp += draw(ax, joint, base, f"{base}, joint XYZ")
            allp += draw(ax,
                         split,
                         base,
                         f"{base}, split X/Z",
                         ls="--",
                         alpha=0.5)
            allp += draw(ax,
                         split,
                         minllr,
                         f"{minllr}, split X/Z",
                         ls="--",
                         alpha=0.5)
            allp += draw(ax, joint, minllr, f"{minllr}, joint XYZ")
            finish(ax, f"{code} -- {row_name}", allp)
    save(fig, axes, "minllr_osd_joint_vs_split.png")
