#!/usr/bin/env python3
"""
Publication-quality figures for the CubeSat Federated EO Learning study.

Run from the project root:
    .venv/bin/python analysis/generate_plots.py

All experiments are defined in results/experiment_registry.md.
Missing experiments are skipped gracefully.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DAILY_CAPACITY_MB = 2.16   # 9.6 kbps × 600 s/contact × 3 contacts/day

# ── Global academic rcParams ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      12,
    "axes.labelweight":    "bold",
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.fontsize":     9,
    "legend.framealpha":   0.93,
    "legend.edgecolor":    "0.70",
    "legend.borderpad":    0.5,
    "figure.dpi":          150,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linestyle":      "--",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.08,
    "pdf.fonttype":        42,   # embed TrueType — required for Overleaf
    "ps.fonttype":         42,
})

# ── Canonical experiment registry ──────────────────────────────────────────
# (key, legend_label, annot_label, color, marker, lineStyle, markerSize, zOrder, group)
_REG = [
    # Group A — Unrestricted baselines
    ("federated_baseline",   "FedAvg, FP32 (IID)",         "FP32·IID",       "#1565C0", "o",  "-",   7, 6, "A"),
    ("federated_non_iid",    "FedAvg, FP32 (Non-IID)",      "FP32·NonIID",    "#E53935", "s",  "--",  6, 4, "A"),
    ("federated_int8",       "FedAvg, INT8 (IID)",          "INT8·IID",       "#2E7D32", "^",  "-.",  7, 7, "A"),
    ("federated_topk_50",    "FedAvg, TopK-50% (IID)",      "TopK-50%",       "#E65100", "D",  ":",   5, 3, "A"),
    ("federated_topk_25",    "FedAvg, TopK-25% (IID)",      "TopK-25%",       "#6A1B9A", "v",  ":",   5, 3, "A"),
    ("federated_topk_10",    "FedAvg, TopK-10% (IID)",      "TopK-10%",       "#4E342E", "P",  ":",   5, 3, "A"),
    # Group B — Budget-constrained scenarios
    ("federated_cubesat_constrained",         "Constrained INT8, IID (50 MB)",        "CS-IID",    "#00897B", "H",  "-.", 7, 5, "B"),
    ("federated_cubesat_constrained_non_iid", "Constrained INT8, Non-IID (50 MB)",    "CS-NonIID", "#C62828", "X",  ":",  6, 4, "B"),
    ("federated_cubesat_sparse_contacts",     "Partial Partic., Non-IID (50 MB, 3/10 clients)", "CS-Partial", "#AD1457", "*",  "--", 7, 5, "B"),
    # Group C — Budget sweep (Non-IID, 3/10 clients, INT8)
    ("federated_budget_25",  "Budget Sweep, 25 MB",   "BS-25 MB",  "#90A4AE", "<",  ":", 5, 2, "C"),
    ("federated_budget_50",  "Budget Sweep, 50 MB",   "BS-50 MB",  "#607D8B", ">",  ":", 5, 2, "C"),
    ("federated_budget_100", "Budget Sweep, 100 MB",  "BS-100 MB", "#455A64", "p",  ":", 5, 2, "C"),
    ("federated_budget_250", "Budget Sweep, 250 MB",  "BS-250 MB", "#263238", "h",  ":", 5, 2, "C"),
]

# Fast lookups
_KEY_TO_META = {r[0]: r for r in _REG}


def _meta(key):
    return _KEY_TO_META.get(key, (key, key, key, "#888", "o", "-", 5, 1, "?"))


def load_data(keys=None):
    targets = keys or [r[0] for r in _REG]
    data = {}
    for key in targets:
        rp = RESULTS / key / "round_metrics.json"
        fp = RESULTS / key / "final_test_metrics.json"
        if not rp.exists() or not fp.exists():
            continue
        _, leg, ann, color, marker, ls, ms, zo, grp = _meta(key)
        final = json.loads(fp.read_text())
        if "estimated_days_to_transmit" not in final:
            final["estimated_days_to_transmit"] = (
                final["total_communication_MB"] / DAILY_CAPACITY_MB
            )
        data[key] = dict(
            leg=leg, ann=ann, color=color, marker=marker,
            ls=ls, ms=ms, zo=zo, grp=grp,
            rounds=json.loads(rp.read_text()),
            final=final,
        )
    return data


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  → figures/{name}.{{pdf,png}}")


# ── Annotation helpers ─────────────────────────────────────────────────────

def _smart_annotations(ax, xs, ys, labels, colors, fontsize=8.0,
                        default_offset=(8, 4)):
    """Place per-point text avoiding obvious collisions using quadrant offsets."""
    x_mid = float(np.median(xs))
    y_mid = float(np.median(ys))

    # Compute initial offsets based on quadrant
    offsets = []
    for x, y in zip(xs, ys):
        dx = default_offset[0] if x <= x_mid else -default_offset[0] - 10
        dy = default_offset[1] if y <= y_mid else -default_offset[1] - 6
        offsets.append((dx, dy))

    # Nudge duplicates in the same quadrant apart
    used = {}
    final_offsets = []
    for i, ((dx, dy), x, y) in enumerate(zip(offsets, xs, ys)):
        key = (dx > 0, dy > 0)
        count = used.get(key, 0)
        dy_adj = dy + count * 11
        used[key] = count + 1
        final_offsets.append((dx, dy_adj))

    for (dx, dy), x, y, label, color in zip(final_offsets, xs, ys, labels, colors):
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            label, xy=(x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color=color, ha=ha,
            arrowprops=dict(
                arrowstyle="-",
                color=color, lw=0.45, alpha=0.7,
                shrinkA=2, shrinkB=2,
            ) if abs(dx) > 10 else None,
        )


# ── Figure 1: Accuracy vs. Total Communication ─────────────────────────────

def plot_accuracy_vs_communication(data):
    """
    Scatter plot: final test accuracy (y) vs. total communication cost (x).
    Excludes federated_budget_50 (duplicate of sparse_contacts).
    Groups experiments by visual cluster and uses a legend instead of
    per-point text for the dense cluster around 993 MB.
    """
    exclude = {"federated_budget_50", "federated_budget_25",
               "federated_budget_100", "federated_budget_250"}
    sub = {k: v for k, v in data.items() if k not in exclude}
    if not sub:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    handles = []
    for key, d in sub.items():
        x = d["final"]["total_communication_MB"]
        y = d["final"]["accuracy"] * 100
        sc = ax.scatter(x, y, color=d["color"], marker=d["marker"],
                        s=65, zorder=d["zo"],
                        edgecolors="white", linewidths=0.5)
        handles.append(mlines.Line2D([], [], color=d["color"], marker=d["marker"],
                                     linestyle="None", markersize=7,
                                     label=d["leg"]))

    # Annotate only the most important / outlier points
    annotate_keys = {
        "federated_int8":                       (12,  3),
        "federated_topk_10":                    (10, -9),
        "federated_cubesat_constrained":         (10,  3),
        "federated_cubesat_constrained_non_iid": (10, -8),
        "federated_cubesat_sparse_contacts":     (-8,  5),
    }
    for key, (dx, dy) in annotate_keys.items():
        if key not in sub:
            continue
        d = sub[key]
        x = d["final"]["total_communication_MB"]
        y = d["final"]["accuracy"] * 100
        ha = "left" if dx > 0 else "right"
        ax.annotate(d["ann"], xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.0, color=d["color"], ha=ha, style="italic",
                    arrowprops=dict(arrowstyle="-", color=d["color"],
                                    lw=0.5, alpha=0.6, shrinkB=3))

    # Group separators (light vertical lines)
    ax.axvline(300, color="0.82", linewidth=0.7, linestyle="--", zorder=0)
    ax.text(310, 74, "Compression\nthreshold", fontsize=7.5, color="0.55",
            va="bottom")

    leg = ax.legend(handles=handles, loc="lower right", fontsize=8.2,
                    ncol=1, title="Experiment", title_fontsize=8.5,
                    handletextpad=0.4, borderpad=0.6)

    ax.set_xlabel("Total Communication Cost (MB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Final Accuracy vs. Total Communication Cost\n"
                 r"EuroSAT · BaselineCNN · 9.6 kbps UHF link model")
    ax.set_ylim(68, 96)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.tight_layout()
    _save(fig, "accuracy_vs_communication")


# ── Figure 2: Budget vs. Accuracy ──────────────────────────────────────────

def plot_budget_vs_accuracy(data):
    """
    Continuous accuracy curve derived from INT8 (IID, 20R) round-by-round data,
    plus discrete budget-sweep (Non-IID, PartC) results.
    Shows how much accuracy one can buy for a given link budget.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Continuous reference curve from INT8 full run
    if "federated_int8" in data:
        d = data["federated_int8"]
        comm = [r["cumulative_communication_MB"] for r in d["rounds"]]
        acc  = [r["test_accuracy"] * 100          for r in d["rounds"]]
        ax.plot(comm, acc, color=d["color"], linewidth=2.0, linestyle="-.",
                marker="^", markersize=4, markevery=3,
                label="FedAvg, INT8, IID (10/10 clients, 20 rounds)",
                zorder=6, alpha=0.85)

    # Budget sweep: discrete markers
    sweep = ["federated_budget_25", "federated_budget_50",
             "federated_budget_100", "federated_budget_250"]
    sweep_avail = [k for k in sweep if k in data]
    if sweep_avail:
        sx = [data[k]["final"]["total_communication_MB"] for k in sweep_avail]
        sy = [data[k]["final"]["accuracy"] * 100          for k in sweep_avail]
        sc = [data[k]["color"]                             for k in sweep_avail]
        ax.scatter(sx, sy, c=sc, s=75, zorder=8,
                   edgecolors="black", linewidths=0.7,
                   label="Budget sweep (INT8, Non-IID, 3/10 clients)",
                   marker="D")
        for k, x, y in zip(sweep_avail, sx, sy):
            mb  = data[k]["final"]["total_communication_MB"]
            rds = data[k]["final"].get("rounds_completed", "?")
            ax.annotate(f"{mb:.0f} MB\n({rds}R)",
                        xy=(x, y), xytext=(6, 4), textcoords="offset points",
                        fontsize=7.5, color="#263238", ha="left")

    # Reference: IID and Non-IID unconstrained baselines
    for ref_key, kw in [
        ("federated_baseline", dict(color="#1565C0", ls="--",
                                    label="FedAvg, FP32, IID — accuracy ceiling")),
        ("federated_non_iid",  dict(color="#E53935", ls=":",
                                    label="FedAvg, FP32, Non-IID — ceiling")),
    ]:
        if ref_key in data:
            acc = data[ref_key]["final"]["accuracy"] * 100
            ax.axhline(acc, linewidth=1.2, alpha=0.6, **kw)

    ax.set_xlabel("Total Communication Budget (MB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy as a Function of Communication Budget\n"
                 r"INT8 compression · EuroSAT · BaselineCNN")
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(loc="lower right", fontsize=8.2, ncol=1)
    fig.tight_layout()
    _save(fig, "budget_vs_accuracy")


# ── Figure 3: Days-to-Transmit vs. Accuracy ────────────────────────────────

def plot_days_vs_accuracy(data):
    """
    Scatter: x = estimated days to transmit all model updates at 9.6 kbps;
    y = final test accuracy. Feasibility zones shown as background regions.
    """
    # Exclude budget sweep duplicates for clarity
    exclude = {"federated_budget_50"}
    sub = {k: v for k, v in data.items() if k not in exclude}
    if not sub:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    x_max = max(d["final"]["estimated_days_to_transmit"]
                for d in sub.values()) * 1.12

    # Feasibility zones
    ax.axvspan(0,    15,     alpha=0.07, color="#4CAF50", zorder=0)
    ax.axvspan(15,   30,     alpha=0.07, color="#FFC107", zorder=0)
    ax.axvspan(30,   x_max,  alpha=0.07, color="#F44336", zorder=0)
    ax.text(1,    94.5, "Feasible\n(≤15 days)",  fontsize=8,  color="#2E7D32", va="top")
    ax.text(15.5, 94.5, "Tight\n(15–30 d)",      fontsize=8,  color="#E65100", va="top")
    ax.text(32,   94.5, "Infeasible\n(>30 days)", fontsize=8,  color="#B71C1C", va="top")

    handles = []
    for key, d in sub.items():
        x = d["final"]["estimated_days_to_transmit"]
        y = d["final"]["accuracy"] * 100
        ax.scatter(x, y, color=d["color"], marker=d["marker"],
                   s=70, zorder=5, edgecolors="white", linewidths=0.5)
        handles.append(mlines.Line2D([], [], color=d["color"], marker=d["marker"],
                                     linestyle="None", markersize=7, label=d["leg"]))

    # Annotate key points
    annotate_keys = {
        "federated_int8":                       ( 10,  3),
        "federated_topk_10":                    ( 10, -8),
        "federated_cubesat_constrained":         (-8,  5),
        "federated_cubesat_constrained_non_iid": (-8, -8),
        "federated_cubesat_sparse_contacts":     (-8,  5),
        "federated_budget_250":                  ( 8,  3),
    }
    for key, (dx, dy) in annotate_keys.items():
        if key not in sub:
            continue
        d = sub[key]
        x = d["final"]["estimated_days_to_transmit"]
        y = d["final"]["accuracy"] * 100
        ha = "left" if dx > 0 else "right"
        ax.annotate(d["ann"], xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.0, color=d["color"], ha=ha, style="italic",
                    arrowprops=dict(arrowstyle="-", color=d["color"],
                                    lw=0.5, alpha=0.6, shrinkB=3))

    leg = ax.legend(handles=handles, loc="lower right", fontsize=8,
                    title="Experiment", title_fontsize=8.5, ncol=1,
                    handletextpad=0.4)

    ax.set_xlabel("Estimated Days to Transmit at 9.6 kbps / 2.16 MB·day⁻¹")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. Downlink Time Requirement\n"
                 r"EuroSAT · BaselineCNN · UHF link model (9.6 kbps)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(68, 97)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.tight_layout()
    _save(fig, "days_to_transmit_vs_accuracy")


# ── Figure 4a/b: Convergence Curves (two separate panels) ──────────────────

def plot_convergence_baselines(data):
    """Convergence of the 6 unrestricted methods (Group A). Single panel."""
    keys = ["federated_baseline", "federated_non_iid", "federated_int8",
            "federated_topk_50", "federated_topk_25", "federated_topk_10"]
    sub = {k: data[k] for k in keys if k in data}
    if not sub:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, d in sub.items():
        rds = d["rounds"]
        x = [r["round"]        for r in rds]
        y = [r["test_accuracy"] * 100 for r in rds]
        me = max(1, len(x) // 5)
        ax.plot(x, y, label=d["leg"], color=d["color"],
                linestyle=d["ls"], marker=d["marker"],
                markersize=d["ms"], markevery=me, linewidth=1.7, zorder=d["zo"])

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Convergence: Unrestricted FL Methods (Group A)\n"
                 "EuroSAT · 10 clients · 20 rounds · no budget cap")
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(loc="lower right", fontsize=8.2, ncol=1)
    fig.tight_layout()
    _save(fig, "convergence_curves_baselines")


def plot_convergence_constrained(data):
    """Convergence of budget-constrained and budget-sweep scenarios (Groups B+C)."""
    keys = ["federated_cubesat_constrained",
            "federated_cubesat_constrained_non_iid",
            "federated_cubesat_sparse_contacts",
            "federated_budget_25",
            "federated_budget_100",
            "federated_budget_250"]
    sub = {k: data[k] for k in keys if k in data}
    if not sub:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, d in sub.items():
        rds = d["rounds"]
        x = [r["round"]        for r in rds]
        y = [r["test_accuracy"] * 100 for r in rds]
        me = max(1, len(x) // 6)
        ax.plot(x, y, label=d["leg"], color=d["color"],
                linestyle=d["ls"], marker=d["marker"],
                markersize=d["ms"], markevery=me, linewidth=1.7, zorder=d["zo"])

    # Budget ceiling marker for 4-round experiments
    ax.axvline(4,  color="0.70", linewidth=0.8, linestyle=":", zorder=1)
    ax.axvline(13, color="0.70", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(4.2,  12, "4R\n(50 MB, AllC)", fontsize=7.5, color="0.45", va="bottom")
    ax.text(13.2, 12, "13R\n(50 MB, PartC)", fontsize=7.5, color="0.45", va="bottom")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Convergence: Budget-Constrained Scenarios (Groups B & C)\n"
                 "EuroSAT · INT8 compression · 50–250 MB budget cap")
    ax.set_xlim(1, None)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(loc="lower right", fontsize=8.0, ncol=1)
    fig.tight_layout()
    _save(fig, "convergence_curves_constrained")


def plot_convergence_curves_all(data):
    """
    Two-panel composite figure: (a) baselines, (b) constrained.
    Used as the main convergence figure in the paper.
    """
    keys_a = ["federated_baseline", "federated_non_iid", "federated_int8",
              "federated_topk_50", "federated_topk_25", "federated_topk_10"]
    keys_b = ["federated_cubesat_constrained",
              "federated_cubesat_constrained_non_iid",
              "federated_cubesat_sparse_contacts",
              "federated_budget_25",
              "federated_budget_100",
              "federated_budget_250"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), sharey=True,
                             constrained_layout=True)

    configs = [(axes[0], keys_a, "(a) Unrestricted FL methods (Group A)"),
               (axes[1], keys_b, "(b) Budget-constrained scenarios (Groups B & C)")]

    for ax, keys, title in configs:
        sub = {k: data[k] for k in keys if k in data}
        for key, d in sub.items():
            rds = d["rounds"]
            x   = [r["round"]        for r in rds]
            y   = [r["test_accuracy"] * 100 for r in rds]
            me  = max(1, len(x) // 5)
            ax.plot(x, y, label=d["leg"], color=d["color"],
                    linestyle=d["ls"], marker=d["marker"],
                    markersize=d["ms"] - 1, markevery=me, linewidth=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Round")
        ax.set_xlim(1, None)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.legend(loc="lower right", fontsize=7.8, ncol=1)

    axes[0].set_ylabel("Test Accuracy (%)")
    fig.suptitle("Federated Learning Convergence on EuroSAT · BaselineCNN",
                 fontsize=12, y=1.01)
    _save(fig, "convergence_curves_all")


# ── Figure 5: Mission Feasibility Regions ─────────────────────────────────

def plot_mission_feasibility(data):
    """
    2D scatter: x = estimated downlink days, y = accuracy.
    Background regions indicate operational feasibility at 9.6 kbps.
    """
    exclude = {"federated_budget_50"}
    sub = {k: v for k, v in data.items() if k not in exclude}
    if not sub:
        return

    x_max = max(d["final"]["estimated_days_to_transmit"]
                for d in sub.values()) * 1.15

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.axvspan(0,    15,    alpha=0.08, color="#4CAF50", zorder=0)
    ax.axvspan(15,   30,    alpha=0.08, color="#FFC107", zorder=0)
    ax.axvspan(30, x_max,   alpha=0.08, color="#F44336", zorder=0)

    # Accuracy threshold lines
    ax.axhline(90, color="#1B5E20", ls="--", lw=0.9, alpha=0.7,
               label="Excellent accuracy threshold (90%)")
    ax.axhline(80, color="#E65100", ls="--", lw=0.9, alpha=0.7,
               label="Acceptable accuracy threshold (80%)")

    handles = []
    for key, d in sub.items():
        x = d["final"]["estimated_days_to_transmit"]
        y = d["final"]["accuracy"] * 100
        ax.scatter(x, y, color=d["color"], marker=d["marker"],
                   s=80, zorder=5, edgecolors="white", linewidths=0.5)
        handles.append(mlines.Line2D([], [], color=d["color"], marker=d["marker"],
                                     linestyle="None", markersize=7, label=d["leg"]))

    # Selective annotation: only key points
    annotate = {
        "federated_int8":                       ( 8,  4),
        "federated_cubesat_constrained":         ( 8, -8),
        "federated_cubesat_constrained_non_iid": (-8, -8),
        "federated_cubesat_sparse_contacts":     (-8,  4),
        "federated_budget_250":                  ( 8,  4),
        "federated_budget_25":                   ( 8, -8),
    }
    for key, (dx, dy) in annotate.items():
        if key not in sub:
            continue
        d = sub[key]
        x = d["final"]["estimated_days_to_transmit"]
        y = d["final"]["accuracy"] * 100
        ha = "left" if dx > 0 else "right"
        ax.annotate(d["ann"], xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=d["color"], ha=ha, style="italic",
                    arrowprops=dict(arrowstyle="-", color=d["color"],
                                    lw=0.45, alpha=0.6, shrinkB=3))

    # Zone labels
    for x_pos, label, col in [(1, "Feasible\n(≤15 d)", "#2E7D32"),
                               (16, "Tight\n(15–30 d)", "#E65100"),
                               (32, "Infeasible\n(>30 d)", "#B71C1C")]:
        ax.text(x_pos, 70.5, label, fontsize=7.5, color=col, va="bottom")

    # Legend patches for zones
    zone_patches = [
        mpatches.Patch(facecolor="#4CAF50", alpha=0.3, label="Feasible (≤15 days)"),
        mpatches.Patch(facecolor="#FFC107", alpha=0.3, label="Tight (15–30 days)"),
        mpatches.Patch(facecolor="#F44336", alpha=0.3, label="Infeasible (>30 days)"),
    ]
    leg1 = ax.legend(handles=zone_patches + handles[:2],
                     loc="upper right", fontsize=8, ncol=1,
                     title="Feasibility zones & thresholds", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=handles[2:], loc="lower right", fontsize=7.8, ncol=1,
              title="Experiments", title_fontsize=8)

    ax.set_xlabel("Estimated Days to Transmit at 9.6 kbps (2.16 MB·day⁻¹)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Mission Feasibility Analysis\n"
                 "CubeSat UHF link model: 9.6 kbps · 3 contacts/day × 10 min")
    ax.set_xlim(0, x_max)
    ax.set_ylim(68, 97)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.tight_layout()
    _save(fig, "mission_feasibility_regions")


# ── Figure 6: Pareto Frontier ─────────────────────────────────────────────

def _pareto_mask(xs, ys):
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


def plot_pareto_frontier(data):
    """
    Accuracy vs. total communication with Pareto frontier highlighted.
    Excludes federated_budget_50 (duplicate of sparse_contacts).
    """
    exclude = {"federated_budget_50"}
    sub = {k: v for k, v in data.items() if k not in exclude}
    if not sub:
        return

    keys   = list(sub.keys())
    xs     = [sub[k]["final"]["total_communication_MB"] for k in keys]
    ys     = [sub[k]["final"]["accuracy"] * 100          for k in keys]
    pareto = _pareto_mask(xs, ys)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    handles = []
    for i, key in enumerate(keys):
        d   = sub[key]
        x, y = xs[i], ys[i]
        on_front = pareto[i]
        ec = "black" if on_front else "white"
        sz = 110 if on_front else 55
        al = 1.0  if on_front else 0.38
        ax.scatter(x, y, color=d["color"], marker=d["marker"],
                   s=sz, alpha=al, zorder=6 if on_front else 3,
                   edgecolors=ec, linewidths=0.9)
        marker_style = dict(linestyle="None", markersize=8 if on_front else 6,
                            markeredgecolor=ec, markeredgewidth=0.6,
                            alpha=al, label=d["leg"])
        handles.append(mlines.Line2D([], [], color=d["color"],
                                     marker=d["marker"], **marker_style))

    # Draw Pareto step line
    front_pts = sorted((xs[i], ys[i]) for i in range(len(keys)) if pareto[i])
    if len(front_pts) > 1:
        fx, fy = zip(*front_pts)
        ax.step(fx, fy, where="post", color="#78909C", lw=1.0,
                linestyle="-", alpha=0.55, label="_nolegend_")

    # Annotate Pareto-optimal points only
    for i, key in enumerate(keys):
        if not pareto[i]:
            continue
        d = sub[key]
        x, y = xs[i], ys[i]
        ax.annotate(d["ann"], xy=(x, y),
                    xytext=(10, 4), textcoords="offset points",
                    fontsize=8.5, color=d["color"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=d["color"],
                                    lw=0.5, shrinkB=3))

    # Legend items
    pareto_sym = mlines.Line2D([], [], color="black", marker="o",
                               linestyle="None", markersize=8,
                               markeredgecolor="black", markerfacecolor="none",
                               markeredgewidth=1.0,
                               label="Pareto-optimal (bold border)")
    dominated_sym = mlines.Line2D([], [], color="0.60", marker="o",
                                  linestyle="None", markersize=6,
                                  label="Dominated (faded)")
    legend_all = [pareto_sym, dominated_sym] + handles
    ax.legend(handles=legend_all, loc="lower right", fontsize=8, ncol=1,
              title="Methods", title_fontsize=8.5)

    ax.set_xlabel("Total Communication Cost (MB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy–Communication Pareto Frontier\n"
                 "EuroSAT · BaselineCNN · all evaluated configurations")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    fig.tight_layout()
    _save(fig, "pareto_frontier")


# ── Figure 7: Mission Utility Comparison ──────────────────────────────────

MISSION_PROFILES = {
    "Low-Bandwidth\n(λ_c=0.40, λ_d=0.30)": dict(lam_comm=0.40, lam_days=0.30, color="#1565C0"),
    "Balanced\n(λ_c=0.15, λ_d=0.15)":       dict(lam_comm=0.15, lam_days=0.15, color="#2E7D32"),
    "High-Accuracy\n(λ_c=0.05, λ_d=0.05)":  dict(lam_comm=0.05, lam_days=0.05, color="#6A1B9A"),
}


def _compute_utility(data):
    comms = [d["final"]["total_communication_MB"]     for d in data.values()]
    days  = [d["final"]["estimated_days_to_transmit"] for d in data.values()]
    max_c, max_d = max(comms) or 1, max(days) or 1
    scores = {}
    for key, d in data.items():
        nc = d["final"]["total_communication_MB"]     / max_c
        nd = d["final"]["estimated_days_to_transmit"] / max_d
        acc = d["final"]["accuracy"]
        scores[key] = {p: acc - cfg["lam_comm"] * nc - cfg["lam_days"] * nd
                       for p, cfg in MISSION_PROFILES.items()}
    return scores


def plot_mission_utility(data):
    """
    Grouped bar chart of mission utility U = acc − λ_c·c̃ − λ_d·d̃ for
    three mission profiles.  Excludes budget_50 (duplicate).
    """
    exclude = {"federated_budget_50"}
    sub = {k: v for k, v in data.items() if k not in exclude}
    if len(sub) < 2:
        return

    keys   = list(sub.keys())
    labels = [sub[k]["ann"] for k in keys]
    scores = _compute_utility(sub)
    profs  = list(MISSION_PROFILES.keys())
    colors = [MISSION_PROFILES[p]["color"] for p in profs]

    n  = len(keys)
    x  = np.arange(n)
    w  = 0.24
    offsets = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    bars_handles = []
    for i, (prof, color, offset) in enumerate(zip(profs, colors, offsets)):
        vals = [scores[k][prof] for k in keys]
        bars = ax.bar(x + offset, vals, w * 0.90,
                      color=color, alpha=0.80, edgecolor="white", linewidth=0.5,
                      label=prof.replace("\n", " "))
        bars_handles.append(bars)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Mission Utility Score  U = acc − λ_c·c̃ − λ_d·d̃")
    ax.set_title("Mission Utility by Operational Profile\n"
                 "Higher score = better trade-off for that profile")
    ax.legend(loc="upper right", fontsize=8.5, ncol=1,
              title="Mission profiles", title_fontsize=8.5)

    # Group separators
    group_boundaries = [
        (0, 5.5,  "Group A — Unrestricted"),
        (6, 8.5,  "Group B — Constrained (50 MB)"),
        (9, 12.5, "Group C — Budget Sweep"),
    ]
    y_min = ax.get_ylim()[0]
    for x0, x1, grp_label in group_boundaries:
        if x1 < n:
            ax.axvline(x1 + 0.5, color="0.75", lw=0.7, linestyle="-")
        ax.text((x0 + min(x1, n - 1)) / 2, y_min * 0.88,
                grp_label, ha="center", fontsize=7.5, color="0.45")

    fig.tight_layout()
    _save(fig, "mission_utility_comparison")


# ── Figure 8: Communication Reduction (bar) ────────────────────────────────

def plot_communication_reduction(data):
    """
    Horizontal bar chart: total communication cost per experiment.
    Sorted ascending by total communication.
    """
    exclude = {"federated_budget_50"}
    sub = {k: v for k, v in data.items() if k not in exclude}
    if not sub:
        return

    pairs = sorted(sub.items(),
                   key=lambda kv: kv[1]["final"]["total_communication_MB"])
    labels = [d["ann"]                              for _, d in pairs]
    comms  = [d["final"]["total_communication_MB"]  for _, d in pairs]
    colors = [d["color"]                            for _, d in pairs]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    bars = ax.barh(labels, comms, color=colors, edgecolor="white",
                   height=0.55)

    # Value labels
    for bar, val in zip(bars, comms):
        ax.text(bar.get_width() + 8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} MB", va="center", ha="left", fontsize=8.5)

    # Reference: FP32 baseline
    fp32_comm = None
    for k, d in sub.items():
        if k == "federated_baseline":
            fp32_comm = d["final"]["total_communication_MB"]
            break
    if fp32_comm:
        ax.axvline(fp32_comm, color="#1565C0", linestyle="--", lw=1.1,
                   alpha=0.7, label=f"FP32 baseline ({fp32_comm:.0f} MB)")
        ax.legend(fontsize=8.5, loc="lower right")

    ax.set_xlabel("Total Communication over All Rounds (MB)")
    ax.set_title("Total Communication Cost by Experiment\n"
                 "Sorted ascending — lower is fewer link-minutes on UHF")
    ax.set_xlim(0, max(comms) * 1.20)
    fig.tight_layout()
    _save(fig, "communication_reduction")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading results …")
    data = load_data()
    if not data:
        print("No results found. Run at least one experiment first.")
        return
    print(f"Loaded {len(data)} experiments.\n")

    print("Generating figures …")
    plot_accuracy_vs_communication(data)
    plot_budget_vs_accuracy(data)
    plot_days_vs_accuracy(data)
    plot_convergence_baselines(data)
    plot_convergence_constrained(data)
    plot_convergence_curves_all(data)
    plot_mission_feasibility(data)
    plot_pareto_frontier(data)
    plot_mission_utility(data)
    plot_communication_reduction(data)
    print(f"\nAll figures saved to: {FIGURES}/")


if __name__ == "__main__":
    main()
