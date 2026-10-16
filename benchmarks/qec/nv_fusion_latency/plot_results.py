#!/usr/bin/env python3
"""Plot the NV-Fusion latency study from benchmark-qec-decoder CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

COLUMNS = [
    "csv",
    "mode",
    "id",
    "decoder",
    "distance",
    "rounds",
    "detector_rounds",
    "instances",
    "shots",
    "total_shots",
    "noise",
    "p50",
    "p90",
    "p99",
    "tail_p50",
    "tail_p99",
    "rounds_per_second",
    "ler",
]
INDEX = {name: i for i, name in enumerate(COLUMNS)}
NUMERIC_FIELDS = {
    "p50",
    "p90",
    "p99",
    "tail_p50",
    "tail_p99",
    "rounds_per_second",
    "ler",
}
PLOT_DISTANCES = (5, 7, 13)
PM_COLOR = "#76B900"
NVF_COLOR = "#1F77B4"
DISTANCE_COLORS = {
    5: "#1f77b4",
    7: "#ff7f0e",
    13: "#2ca02c",
    17: "#d62728",
    21: "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def parse_logs(log_directory: Path) -> list[dict]:
    stream: dict[tuple, dict] = {}

    paths = [
        *sorted(log_directory.glob("log*.txt")),
        *sorted(log_directory.glob("log*.out")),
    ]
    for path in paths:
        with path.open(newline="") as handle:
            for raw in csv.reader(handle):
                if len(raw) < len(COLUMNS) or raw[0].strip() != "csv":
                    continue
                if not raw[INDEX["distance"]].strip().isdigit():
                    continue
                row = {name: raw[i].strip() for i, name in enumerate(COLUMNS)}
                for field in ("distance", "rounds", "instances", "shots"):
                    row[field] = int(row[field])
                for field in NUMERIC_FIELDS:
                    row[field] = float(row[field])
                row["source"] = path.name

                if row["mode"] == "stream":
                    key = (
                        row["decoder"],
                        row["distance"],
                        row["rounds"],
                        row["instances"],
                    )
                    # Prefer a dedicated per-run log; consolidated scheduler
                    # output supplements keys whose log file is missing.
                    stream.setdefault(key, row)

    return list(stream.values())


def configure_axis(ax: plt.Axes, rounds: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(rounds)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, which="both", alpha=0.25)


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_blocks(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    width = 0.82
    height = 0.58

    def node(x: float, y: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, zorder=3)

    def fused_pair(left_x: float, right_x: float, child_y: float,
                   parent_x: float, parent_y: float) -> None:
        child_top = child_y + height / 2
        parent_bottom = parent_y - height / 2
        join_y = (child_top + parent_bottom) / 2
        for child_x, target_x in (
            (left_x, parent_x - 0.15),
            (right_x, parent_x + 0.15),
        ):
            ax.plot(
                [child_x, child_x, target_x],
                [child_top, join_y, join_y],
                color="#666666",
                linewidth=1.2,
                zorder=1,
            )
            ax.annotate(
                "",
                xy=(target_x, parent_bottom),
                xytext=(target_x, join_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#666666",
                    "linewidth": 1.2,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=1,
            )

    leaf_y, layer1_y, layer2_y = 0.25, 1.65, 3.05
    for index in range(6):
        node(index, leaf_y, f"Block {index}\nr{index}–r{index + 1}", "#DCEEFF")

    layer1_x = (0.5, 2.5, 4.5)
    for index, x in enumerate(layer1_x):
        node(x, layer1_y, f"Layer-1 fuse {index}", "#DDF3D3")
        fused_pair(2 * index, 2 * index + 1, leaf_y, x, layer1_y)

    layer2_x = (1.5, 3.5)
    for index, x in enumerate(layer2_x):
        node(x, layer2_y, f"Layer-2 fuse {index}", "#FFF0C9")
        fused_pair(
            layer1_x[index],
            layer1_x[index + 1],
            layer1_y,
            x,
            layer2_y,
        )

    ax.text(-0.62,
            leaf_y,
            "Leaf MWPMs",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold")
    ax.text(-0.62,
            layer1_y,
            "Adjacent pairs",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold")
    ax.text(-0.62,
            layer2_y,
            "Overlapping\nfour-block windows",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold")
    for y in (leaf_y, layer1_y, layer2_y):
        ax.text(5.72,
                y,
                "⋯",
                ha="center",
                va="center",
                fontsize=24,
                color="#555555")
    ax.annotate(
        "Syndrome arrival / time — schedule continues",
        xy=(6.2, -0.58),
        xytext=(-0.4, -0.58),
        ha="center",
        va="center",
        fontsize=10,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.3,
            "color": "#333333"
        },
    )
    ax.set_xlim(-1.5, 6.35)
    ax.set_ylim(-0.9, 3.65)
    ax.axis("off")
    fig.suptitle(
        "NV-Fusion two-layer brickwall schedule\n"
        "The pattern repeats as new syndrome blocks arrive",
        fontsize=13,
    )
    save(fig, output, "nv_fusion_block_schedule.png")


def stream_value(
    rows: list[dict],
    decoder: str,
    distance: int,
    rounds: int,
    field: str,
    instances: int = 1,
) -> float | None:
    for row in rows:
        if (row["decoder"] == decoder and row["distance"] == distance and
                row["rounds"] == rounds and row["instances"] == instances):
            return row[field]
    return None


def plot_stream_tail(rows: list[dict], output: Path) -> None:
    distances = [
        distance for distance in PLOT_DISTANCES
        if any(row["distance"] == distance and
               row["decoder"] == "nv-fusion-decoder" for row in rows)
    ]
    fig, axes = plt.subplots(1,
                             len(distances),
                             figsize=(5 * len(distances), 4.6),
                             squeeze=False)
    flat = axes.ravel()

    for ax, distance in zip(flat, distances):
        rounds = sorted({
            row["rounds"]
            for row in rows
            if row["decoder"] == "nv-fusion-decoder" and
            row["distance"] == distance and row["instances"] == 1
        })
        for decoder, color, label in (
            ("pymatching", PM_COLOR, "PyMatching"),
            ("nv-fusion-decoder", NVF_COLOR, "NV-Fusion"),
        ):
            p50 = [
                stream_value(rows, decoder, distance, r, "tail_p50")
                for r in rounds
            ]
            p99 = [
                stream_value(rows, decoder, distance, r, "tail_p99")
                for r in rounds
            ]
            valid = [(r, median, tail)
                     for r, median, tail in zip(rounds, p50, p99)
                     if median is not None and tail is not None]
            x = [item[0] for item in valid]
            median = [item[1] for item in valid]
            tail = [item[2] for item in valid]
            ax.fill_between(
                x,
                median,
                tail,
                color=color,
                alpha=0.18,
                label=f"{label} p50–p99",
            )
            ax.plot(
                x,
                median,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=f"{label} p50",
            )
        configure_axis(ax, rounds)
        ax.axvline(192, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Distance {distance}")
        ax.set_xlabel("Stabilizer rounds")
        ax.set_ylabel("Streaming tail latency (µs)")
        ax.legend(fontsize=7)

    for ax in flat[len(distances):]:
        ax.set_visible(False)
    fig.suptitle(
        "NVIDIA Vera CPU — Streaming Tail Latency, one decoder instance\n"
        "Final detector round through observable correction; 1 µs round interval",
        fontsize=13,
    )
    save(fig, output, "nv_fusion_streaming_tail_latency.png")


def plot_instance_tail(rows: list[dict], output: Path) -> None:
    distances = sorted({
        distance for distance in PLOT_DISTANCES if len({
            row["instances"] for row in rows
            if row["decoder"] == "nv-fusion-decoder" and
            row["distance"] == distance
        }) > 1
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, round_count in zip(axes, (1000, 10000)):
        for distance in distances:
            instances = sorted({
                row["instances"]
                for row in rows
                if row["decoder"] == "nv-fusion-decoder" and
                row["distance"] == distance and row["rounds"] == round_count
            })
            values = [
                stream_value(
                    rows,
                    "nv-fusion-decoder",
                    distance,
                    round_count,
                    "tail_p99",
                    instance,
                ) for instance in instances
            ]
            valid = [(instance, value)
                     for instance, value in zip(instances, values)
                     if value is not None]
            if not valid:
                continue
            valid_instances, valid_values = zip(*valid)
            ax.plot(
                valid_instances,
                valid_values,
                color=DISTANCE_COLORS[distance],
                marker="o",
                linewidth=1.8,
                label=f"d={distance}",
            )
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xlabel("Concurrent decoder instances")
        ax.set_ylabel("Streaming tail p99 (µs)")
        ax.set_title(f"{round_count:,} rounds")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "NVIDIA Vera CPU — NV-Fusion latency under concurrent decoder load\n"
        "Instances are pinned to non-overlapping CPU sets",
        fontsize=13,
    )
    save(fig, output, "nv_fusion_streaming_instance_latency.png")


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    stream = parse_logs(args.log_directory)
    if not stream:
        raise RuntimeError("Expected streaming CSV records")
    plot_fusion_blocks(args.output_directory)
    plot_stream_tail(stream, args.output_directory)
    plot_instance_tail(stream, args.output_directory)


if __name__ == "__main__":
    main()
