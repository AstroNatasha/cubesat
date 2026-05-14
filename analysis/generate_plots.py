#!/usr/bin/env python3
"""
Generate publication-quality figures for the CubeSat FL paper.

Run from the project root:
    .venv/bin/python analysis/generate_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Overleaf-ready academic style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.75",
    "figure.figsize":     (6.5, 4.0),
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.28,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,   # TrueType in PDF — required for Overleaf
    "ps.fonttype":        42,
})

# ── experiment registry ────────────────────────────────────────────────────
# (key, label, color, linestyle, marker, markersize, zorder)
_REG = [
    ("federated_baseline", "FedAvg (IID)",     "#1976D2", "-",   "o",  7, 6),
    ("federated_non_iid",  "FedAvg (Non-IID)", "#E53935", "--",  "s",  6, 4),
    ("federated_int8",     "FedAvg + INT8",    "#2E7D32", "-.",  "^",  7, 7),
    ("federated_topk_50",  "TopK-50%",         "#F57C00", ":",   "D",  5, 3),
    ("federated_topk_25",  "TopK-25%",         "#6A1B9A", ":",   "v",  5, 3),
    ("federated_topk_10",  "TopK-10%",         "#4E342E", ":",   "P",  5, 3),
]


def load_data() -> dict:
    data = {}
    for key, label, color, ls, marker, ms, zo in _REG:
        rp = RESULTS / key / "round_metrics.json"
        fp = RESULTS / key / "final_test_metrics.json"
        if not rp.exists() or not fp.exists():
            print(f"  [skip] {key} — results not found")
            continue
        data[key] = dict(
            label=label, color=color, ls=ls, marker=marker, ms=ms, zo=zo,
            rounds=json.loads(rp.read_text()),
            final=json.loads(fp.read_text()),
        )
    return data


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  → figures/{name}.{{pdf,png}}")


# ── Figure 1: convergence curves ──────────────────────────────────────────

def plot_convergence(data: dict):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for key, d in data.items():
        rounds = d["rounds"]
        x = [r["round"] for r in rounds]
        y = [r["test_accuracy"] * 100 for r in rounds]
        ax.plot(
            x, y,
            label=d["label"],
            color=d["color"],
            linestyle=d["ls"],
            marker=d["marker"],
            markersize=d["ms"],
            markevery=4,
            linewidth=1.6,
            zorder=d["zo"],
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    legend = ax.legend(
        loc="lower right",
        ncol=2,
        handlelength=2.2,
        columnspacing=1.0,
    )
    ax.set_title("Convergence of Global Test Accuracy over FL Rounds")
    fig.tight_layout()
    _save(fig, "convergence_curves")


# ── Figure 2: accuracy vs total communication ──────────────────────────────

def plot_accuracy_vs_communication(data: dict):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for key, d in data.items():
        f = d["final"]
        x = f["total_communication_MB"]
        y = f["accuracy"] * 100
        ax.scatter(x, y,
                   color=d["color"], marker=d["marker"],
                   s=80, zorder=d["zo"], edgecolors="white", linewidths=0.5)
        offset_x = -10 if x > 800 else 12
        offset_y = 0.5
        ax.annotate(
            d["label"],
            xy=(x, y), xytext=(x + offset_x, y + offset_y),
            fontsize=8.5, color=d["color"],
            ha="right" if x > 800 else "left",
        )

    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. Communication Cost")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    fig.tight_layout()
    _save(fig, "accuracy_vs_communication")


# ── Figure 3: communication reduction bar chart ────────────────────────────

def plot_communication_reduction(data: dict):
    keys   = list(data.keys())
    labels = [data[k]["label"] for k in keys]
    comms  = [data[k]["final"]["total_communication_MB"] for k in keys]
    colors = [data[k]["color"] for k in keys]

    # sort ascending by total communication
    order  = sorted(range(len(comms)), key=lambda i: comms[i])
    labels = [labels[i] for i in order]
    comms  = [comms[i]  for i in order]
    colors = [colors[i] for i in order]

    baseline_comm = data.get("federated_baseline", {}).get("final", {}).get(
        "total_communication_MB", None
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.barh(labels, comms, color=colors, edgecolor="white", height=0.55)

    # value labels
    for bar, val in zip(bars, comms):
        ax.text(
            bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} MB",
            va="center", ha="left", fontsize=8.5,
        )

    # baseline reference line
    if baseline_comm:
        ax.axvline(baseline_comm, color="#1976D2", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"FedAvg baseline ({baseline_comm:.0f} MB)")
        ax.legend(fontsize=8.5, loc="lower right")

    ax.set_xlabel("Total Communication over 20 Rounds (MB)")
    ax.set_title("Communication Cost by Method")
    ax.set_xlim(0, max(comms) * 1.22)
    fig.tight_layout()
    _save(fig, "communication_reduction")


# ── Figure 4: accuracy drop relative to IID baseline ──────────────────────

def plot_accuracy_drop(data: dict):
    if "federated_baseline" not in data:
        print("  [skip] accuracy_drop — baseline missing")
        return

    baseline_acc = data["federated_baseline"]["final"]["accuracy"] * 100

    other = {k: d for k, d in data.items() if k != "federated_baseline"}
    # sort by drop (smallest first = closest to baseline)
    order = sorted(other.keys(), key=lambda k: data[k]["final"]["accuracy"], reverse=True)

    labels = [other[k]["label"]  for k in order]
    drops  = [baseline_acc - other[k]["final"]["accuracy"] * 100 for k in order]
    colors = [other[k]["color"]  for k in order]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.barh(labels, drops, color=colors, edgecolor="white", height=0.55)

    for bar, drop in zip(bars, drops):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"−{drop:.2f} pp",
            va="center", ha="left", fontsize=8.5,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Drop vs. FedAvg IID Baseline (percentage points)")
    ax.set_title("Accuracy Cost of Each Method Relative to IID Baseline")
    ax.set_xlim(-0.5, max(drops) * 1.35)
    fig.tight_layout()
    _save(fig, "accuracy_drop")


# ── Figure 5: Pareto frontier ──────────────────────────────────────────────

def _pareto_mask(xs, ys):
    """Return boolean mask of Pareto-optimal points (min x, max y)."""
    n = len(xs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break
    return [not d for d in dominated]


def plot_pareto_frontier(data: dict):
    keys   = list(data.keys())
    xs     = [data[k]["final"]["total_communication_MB"] for k in keys]
    ys     = [data[k]["final"]["accuracy"] * 100          for k in keys]
    labels = [data[k]["label"]  for k in keys]
    colors = [data[k]["color"]  for k in keys]
    markers= [data[k]["marker"] for k in keys]

    pareto = _pareto_mask(xs, ys)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # non-Pareto points (dimmer)
    for i, (x, y, label, color, marker, on_front) in enumerate(
        zip(xs, ys, labels, colors, markers, pareto)
    ):
        if not on_front:
            ax.scatter(x, y, color=color, marker=marker,
                       s=60, alpha=0.45, zorder=3, edgecolors="white")

    # Pareto-optimal points (highlighted)
    pareto_xs, pareto_ys = [], []
    for i, (x, y, label, color, marker, on_front) in enumerate(
        zip(xs, ys, labels, colors, markers, pareto)
    ):
        if on_front:
            ax.scatter(x, y, color=color, marker=marker,
                       s=110, zorder=6, edgecolors="black", linewidths=0.8)
            pareto_xs.append(x)
            pareto_ys.append(y)

    # Pareto frontier step line
    if pareto_xs:
        order = sorted(zip(pareto_xs, pareto_ys))
        pfx   = [p[0] for p in order]
        pfy   = [p[1] for p in order]
        ax.step(pfx, pfy, where="post", color="gray", linewidth=1.0,
                linestyle="-", alpha=0.6, zorder=2)

    # annotations for all points
    for x, y, label, color in zip(xs, ys, labels, colors):
        ax.annotate(
            label, xy=(x, y),
            xytext=(8, 4), textcoords="offset points",
            fontsize=8, color=color,
        )

    # legend patch for Pareto indicator
    pareto_patch = mpatches.Patch(facecolor="none", edgecolor="black",
                                   linewidth=0.8, label="Pareto-optimal")
    ax.legend(handles=[pareto_patch], fontsize=8.5, loc="lower right")

    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy–Communication Pareto Frontier")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    fig.tight_layout()
    _save(fig, "pareto_frontier")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    data = load_data()
    if not data:
        print("No results found. Run experiments first.")
        return
    print(f"Loaded {len(data)} experiments: {list(data.keys())}\n")

    print("Generating figures...")
    plot_convergence(data)
    plot_accuracy_vs_communication(data)
    plot_communication_reduction(data)
    plot_accuracy_drop(data)
    plot_pareto_frontier(data)
    print(f"\nAll figures saved to: {FIGURES}/")


if __name__ == "__main__":
    main()
