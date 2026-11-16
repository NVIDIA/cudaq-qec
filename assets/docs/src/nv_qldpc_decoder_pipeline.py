# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block diagram of the nv-qldpc decoder pipeline for the API docs.

Regenerate ../nv_qldpc_decoder_pipeline.png (referenced from
docs/sphinx/api/qec/nv_qldpc_decoder_api.rst) with

    python3 assets/docs/src/nv_qldpc_decoder_pipeline.py

Hand-laid on a fixed grid (inputs, BP stage, OSD post-processing, outputs)
so the edges stay orthogonal and the labels never collide; matplotlib only.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 18.0, 5.8
fig, ax = plt.subplots(figsize=(W, H), dpi=110)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

FONT = "DejaVu Sans"
C_STAGE, C_TCBP, C_OUT, EDGE = "#eef2f8", "#fdf1dc", "#e6f3e6", "#333333"
C_BORDER = "#7a869a"


def box(x, y, w, h, title, lines=(), fc=C_STAGE, double=False, dashed=False):
    kw = dict(boxstyle="round,pad=0.02,rounding_size=0.12",
              fc=fc,
              ec=C_BORDER,
              lw=1.1)
    if dashed:
        kw.update(fc="none", ls=(0, (4, 3)), ec="#9aa3b2")
    ax.add_patch(FancyBboxPatch((x, y), w, h, **kw))
    if double:
        ax.add_patch(
            FancyBboxPatch((x + 0.06, y + 0.06),
                           w - 0.12,
                           h - 0.12,
                           boxstyle="round,pad=0.02,rounding_size=0.09",
                           fc="none",
                           ec=C_BORDER,
                           lw=0.9))
    if title:
        cy = y + h / 2 + 0.11 * len(lines)
        ax.text(x + w / 2,
                cy,
                title,
                ha="center",
                va="center",
                fontsize=9.8,
                fontname=FONT,
                weight="bold",
                color="#1f2430")
        for i, l in enumerate(lines):
            ax.text(x + w / 2,
                    cy - 0.26 * (i + 1),
                    l,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    fontname=FONT,
                    color="#2d3440")
    return (x, y, w, h)


def port(b, side, frac=0.5):
    x, y, w, h = b
    return {
        "l": (x, y + h * frac),
        "r": (x + w, y + h * frac),
        "t": (x + w * frac, y + h),
        "b": (x + w * frac, y)
    }[side]


def arrow(p, q, label=None, lab_dy=0.14, via=None, lab_dx=0.0):
    pts = [p] + (via or []) + [q]
    for a, b in zip(pts[:-1], pts[1:]):
        last = b is q
        ax.add_patch(
            FancyArrowPatch(a,
                            b,
                            arrowstyle="-|>" if last else "-",
                            mutation_scale=12,
                            lw=1.1,
                            color=EDGE,
                            shrinkA=0,
                            shrinkB=0))
    if label:
        mid = pts[len(pts) // 2 - 1] if len(pts) > 2 else p
        nxt = pts[len(pts) // 2] if len(pts) > 2 else q
        mx, my = (mid[0] + nxt[0]) / 2, (mid[1] + nxt[1]) / 2
        ax.text(mx + lab_dx,
                my + lab_dy,
                label,
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontname=FONT,
                color="#2d3440",
                bbox=dict(fc="white", ec="none", pad=1.0))


# column x positions
X0, X1, X2, X3 = 0.3, 4.3, 8.7, 13.15
BW1, BW2, BW3 = 3.55, 3.9, 4.45

# inputs
syn = box(X0, 3.95, BW1, 0.55, "syndromes", fc="white")
model = box(X0,
            1.45,
            BW1,
            1.05,
            "TCBP model",
            ["init_tcbp_model(H), once per code family", "tcbp_model=..."],
            fc=C_TCBP)
# BP stage
bp = box(X1, 3.55, BW2, 1.2, "BP front end", [
    "bp_method, max_iterations, clip_value",
    "memory / relay: composition, gamma*, srelay_config"
])
scores = box(X1,
             1.45,
             BW2,
             1.05,
             "TCBP calibrated scores",
             ["P(error) per mechanism,", "from the LLR trajectory"],
             fc=C_TCBP)
# post-processing cluster
CLW = (X3 + BW3 + 0.3) - (X2 - 0.3)
cl = box(X2 - 0.3, 0.95, CLW, 3.15, "", dashed=True)
ax.text(X2 - 0.3 + CLW / 2,
        3.88,
        "OSD post-processing (use_osd=true)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontname=FONT,
        color="#4a5262")
ordb = box(X2 + 0.15, 1.95, 2.45, 1.15, "column ordering",
           ["osd_init_method:", "final_llr | min_llr | tcbp"])
osd = box(X3,
          2.55,
          BW3,
          1.2,
          "OSD-0 / exhaustive / combination sweep",
          ["osd_method 1 / 2 / 3, osd_order", "osd_sparsify, osd_sparsify_pad"],
          fc=C_OUT,
          double=True)
jobs = box(X3,
           1.2,
           BW3,
           1.05,
           "JOBS", ["osd_method=4, jobs_*", "(+ jobs_gap in opt_results)"],
           fc=C_OUT,
           double=True)
# other outputs
hard = box(X3,
           4.5,
           BW3,
           0.85,
           "hard decision", ["BP converged, or use_osd=false"],
           fc=C_OUT,
           double=True)
soft = box(X3,
           0.1,
           BW3,
           0.75,
           "soft probabilities", ["use_osd=false"],
           fc=C_OUT,
           double=True)

# edges
arrow(port(syn, "r"), port(bp, "l", 0.56))
arrow(port(model, "r"), port(scores, "l"))
arrow(port(bp, "b", 0.5), port(scores, "t", 0.5))
ax.text(port(bp, "b", 0.5)[0] + 0.1,
        (port(bp, "b", 0.5)[1] + port(scores, "t", 0.5)[1]) / 2,
        "LLR trajectory",
        ha="left",
        va="center",
        fontsize=8.6,
        fontname=FONT,
        color="#2d3440")
# bp -> hard: orthogonal, over the cluster
hy = port(hard, "l")[1]
top = port(bp, "t", 0.82)
arrow(top, port(hard, "l"), via=[(top[0], hy)])
# bp -> ordering (final/min llr) and scores -> ordering (tcbp)
arrow(port(bp, "r", 0.25),
      port(ordb, "l", 0.72),
      "final_llr | min_llr",
      lab_dy=0.1,
      lab_dx=0.25)
arrow(port(scores, "r", 0.5),
      port(ordb, "l", 0.3),
      "tcbp",
      lab_dy=0.1,
      lab_dx=-0.15)
# ordering -> osd / jobs
arrow(port(ordb, "r", 0.7),
      port(osd, "l", 0.5),
      "osd_method 1 / 2 / 3",
      lab_dy=0.1,
      lab_dx=-0.35)
arrow(port(ordb, "r", 0.3), port(jobs, "l", 0.5), "osd_method 4", lab_dy=0.08)
# scores -> soft: orthogonal, under the cluster
sy = port(soft, "l")[1]
arrow(port(scores, "b", 0.85),
      port(soft, "l"),
      via=[(port(scores, "b", 0.85)[0], sy)])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   "nv_qldpc_decoder_pipeline.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print("wrote", os.path.normpath(out))
