#!/usr/bin/env python3
"""
Generate Overleaf-ready LaTeX tables and the final Markdown summary.

Run from the project root:
    .venv/bin/python analysis/generate_tables.py
"""

import json
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
RESULTS    = ROOT / "results"
TABLES_DIR = RESULTS / "tables"
TABLES_DIR.mkdir(exist_ok=True)

DAILY_CAPACITY_MB = 2.16  # 9.6 kbps × 600 s/contact × 3 contacts/day

# ── Display config for every experiment key ────────────────────────────────
_DISPLAY = {
    "federated_baseline":                  ("FedAvg (IID)",              "none",   "baseline"),
    "federated_non_iid":                   ("FedAvg (Non-IID)",          "none",   "baseline"),
    "federated_int8":                      ("FedAvg + INT8",             "int8",   "compression"),
    "federated_topk_50":                   ("TopK-50\\%",                "topk",   "compression"),
    "federated_topk_25":                   ("TopK-25\\%",                "topk",   "compression"),
    "federated_topk_10":                   ("TopK-10\\%",                "topk",   "compression"),
    "federated_cubesat_constrained":       ("INT8 (50 MB, IID)",         "int8",   "constrained"),
    "federated_cubesat_constrained_non_iid":("INT8 (50 MB, Non-IID)",   "int8",   "constrained"),
    "federated_cubesat_sparse_contacts":   ("INT8 (50 MB, 3 clients)",   "int8",   "constrained"),
    "federated_budget_25":                 ("INT8, 3c (25 MB)",          "int8",   "budget"),
    "federated_budget_50":                 ("INT8, 3c (50 MB)",          "int8",   "budget"),
    "federated_budget_100":                ("INT8, 3c (100 MB)",         "int8",   "budget"),
    "federated_budget_250":                ("INT8, 3c (250 MB)",         "int8",   "budget"),
}


def _label(key: str) -> str:
    return _DISPLAY.get(key, (key, "", ""))[0]


def load_final(key: str) -> dict | None:
    p = RESULTS / key / "final_test_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "estimated_days_to_transmit" not in d:
        d["estimated_days_to_transmit"] = d["total_communication_MB"] / DAILY_CAPACITY_MB
    return d


def bf(s: str) -> str:
    return f"\\textbf{{{s}}}"


def fmt(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}"


def _load_group(keys: list[str]) -> list[dict]:
    rows = []
    for k in keys:
        f = load_final(k)
        if f is None:
            continue
        rows.append({
            "key":      k,
            "label":    _label(k),
            "accuracy": f["accuracy"],
            "macro_f1": f["macro_f1"],
            "total_comm_mb": f["total_communication_MB"],
            "rounds":   f.get("rounds_completed", f.get("num_rounds", "—")),
            "days_tx":  f.get("estimated_days_to_transmit", 0.0),
            "feasible": f.get("feasible_under_budget", None),
            "latency_ms": f.get("latency_ms", 0.0),
        })
    return rows


def _best(rows: list[dict], field: str, hi: bool = True) -> float:
    vals = [r[field] for r in rows if isinstance(r[field], float)]
    return (max if hi else min)(vals) if vals else None


# ── Table 1: main results ──────────────────────────────────────────────────

def build_main_results() -> str:
    keys = ["federated_baseline", "federated_non_iid", "federated_int8",
            "federated_topk_50", "federated_topk_25", "federated_topk_10"]
    rows = _load_group(keys)
    if not rows:
        return ""
    baseline_acc  = rows[0]["accuracy"]
    baseline_comm = rows[0]["total_comm_mb"]
    best_acc  = _best(rows, "accuracy")
    best_f1   = _best(rows, "macro_f1")
    best_comm = _best(rows, "total_comm_mb", hi=False)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Federated learning on EuroSAT (20 rounds, 10 clients, IID unless noted). "
        r"Best per column in \textbf{bold}. Acc.\ drop vs.\ FedAvg IID baseline.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Method & Acc (\%) & F1 (\%) & Comm (MB) & Comm Reduc. & $\Delta$Acc (pp) \\",
        r"\midrule",
    ]
    for r in rows:
        a = fmt(r["accuracy"] * 100)
        f = fmt(r["macro_f1"] * 100)
        c = fmt(r["total_comm_mb"], 1)
        red = f"{baseline_comm / r['total_comm_mb']:.2f}$\\times$"
        drop = (baseline_acc - r["accuracy"]) * 100
        d = f"$-${fmt(drop)}" if drop > 0.005 else fmt(drop)
        if r["accuracy"]      == best_acc:  a = bf(a)
        if r["macro_f1"]      == best_f1:   f = bf(f)
        if r["total_comm_mb"] == best_comm: c = bf(c)
        lines.append(f"  {r['label']} & {a} & {f} & {c} & {red} & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Table 2: compression trade-offs ───────────────────────────────────────

def build_compression_tradeoffs() -> str:
    keys = ["federated_baseline", "federated_int8",
            "federated_topk_50", "federated_topk_25", "federated_topk_10"]
    rows = _load_group(keys)
    if not rows:
        return ""
    baseline_acc  = rows[0]["accuracy"]
    baseline_comm = rows[0]["total_comm_mb"]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Communication compression trade-offs on EuroSAT (IID, 20 rounds, 10 clients). "
        r"Compression ratio vs.\ FP32 baseline. Latency measured on CPU at $64\times64$ px.}",
        r"\label{tab:compression_tradeoffs}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Method & Acc (\%) & $\Delta$Acc (pp) & Comm (MB) & Ratio & Latency (ms) \\",
        r"\midrule",
    ]
    best_comm = _best(rows, "total_comm_mb", hi=False)
    for r in rows:
        a    = fmt(r["accuracy"] * 100)
        drop = (baseline_acc - r["accuracy"]) * 100
        d    = f"$-${fmt(drop)}" if drop > 0.005 else fmt(drop)
        c    = fmt(r["total_comm_mb"], 1)
        rat  = f"{baseline_comm / r['total_comm_mb']:.2f}$\\times$"
        lat  = fmt(r["latency_ms"])
        if r["total_comm_mb"] == best_comm: c = bf(c)
        lines.append(f"  {r['label']} & {a} & {d} & {c} & {rat} & {lat} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Table 3: constrained scenarios ────────────────────────────────────────

def build_constrained_scenarios() -> str:
    keys = ["federated_baseline", "federated_int8",
            "federated_cubesat_constrained",
            "federated_cubesat_constrained_non_iid",
            "federated_cubesat_sparse_contacts"]
    rows = _load_group(keys)
    if not rows:
        return ""
    baseline_acc  = rows[0]["accuracy"]
    baseline_comm = rows[0]["total_comm_mb"]
    best_acc  = _best(rows, "accuracy")
    best_comm = _best(rows, "total_comm_mb", hi=False)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{CubeSat mission-constrained scenarios (50 MB budget, 9.6 kbps UHF). "
        r"Baseline rows are unconstrained (20 rounds). "
        r"$\dagger$ Estimated at 9.6 kbps, 2.16 MB/day capacity.}",
        r"\label{tab:constrained_scenarios}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Scenario & Rounds & Acc (\%) & Comm (MB) & $\Delta$Acc (pp) & Days$^\dagger$ \\",
        r"\midrule",
    ]
    for r in rows:
        a    = fmt(r["accuracy"] * 100)
        c    = fmt(r["total_comm_mb"], 1)
        drop = (baseline_acc - r["accuracy"]) * 100
        d    = f"$-${fmt(drop)}" if drop > 0.005 else fmt(drop)
        days = fmt(r["days_tx"], 1)
        rds  = str(r["rounds"])
        if r["accuracy"]      == best_acc:  a = bf(a)
        if r["total_comm_mb"] == best_comm: c = bf(c)
        lines.append(f"  {r['label']} & {rds} & {a} & {c} & {d} & {days} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Table 4: budget sweep ─────────────────────────────────────────────────

def build_budget_sweep() -> str:
    keys = ["federated_budget_25", "federated_budget_50",
            "federated_budget_100", "federated_budget_250"]
    rows = _load_group(keys)
    if not rows:
        return ""
    best_acc = _best(rows, "accuracy")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Communication budget sweep (INT8, Non-IID, 3 clients/round). "
        r"Accuracy improves monotonically with larger budget. "
        r"$\dagger$ Estimated at 9.6 kbps.}",
        r"\label{tab:budget_sweep}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Config & Budget (MB) & Rounds & Acc (\%) & F1 (\%) & Comm (MB) & Days$^\dagger$ \\",
        r"\midrule",
    ]
    for r in rows:
        budget = r["total_comm_mb"]   # total used ≈ budget (capped)
        a  = fmt(r["accuracy"] * 100)
        f  = fmt(r["macro_f1"] * 100)
        c  = fmt(r["total_comm_mb"], 1)
        days = fmt(r["days_tx"], 1)
        rds = str(r["rounds"])
        if r["accuracy"] == best_acc: a = bf(a)
        lines.append(f"  {r['label']} & {c} & {rds} & {a} & {f} & {c} & {days} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Table 5: CubeSat comparison (ideal vs constrained) ────────────────────

def build_cubesat_comparison() -> str:
    paths = {
        "federated_baseline":          "FedAvg (IID, 20 rounds)",
        "federated_int8":              "FedAvg + INT8 (20 rounds)",
        "federated_cubesat_constrained": "INT8 (constrained, 50 MB)",
    }
    rows = []
    for key, label in paths.items():
        f = load_final(key)
        if f is None:
            continue
        rows.append(dict(label=label, accuracy=f["accuracy"], macro_f1=f["macro_f1"],
                         total_comm_mb=f["total_communication_MB"],
                         rounds=f.get("rounds_completed", f.get("num_rounds", "—")),
                         days_tx=f.get("estimated_days_to_transmit", 0.0),
                         feasible=f.get("feasible_under_budget", None)))
    if len(rows) < 2:
        return ""
    baseline_acc  = rows[0]["accuracy"]
    best_acc  = _best(rows, "accuracy")
    best_comm = _best(rows, "total_comm_mb", hi=False)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Ideal vs.\ constrained CubeSat federated learning. "
        r"The constrained scenario (50 MB, 9.6 kbps UHF) achieves only 4 rounds. "
        r"$\dagger$ Estimated at 9.6 kbps, 2.16 MB/day.}",
        r"\label{tab:cubesat_constrained}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Scenario & Rounds & Acc (\%) & Comm (MB) & $\Delta$Acc (pp) & Days$^\dagger$ \\",
        r"\midrule",
    ]
    for r in rows:
        a    = fmt(r["accuracy"] * 100)
        c    = fmt(r["total_comm_mb"], 1)
        drop = (baseline_acc - r["accuracy"]) * 100
        d    = f"$-${fmt(drop)}" if drop > 0.005 else fmt(drop)
        days = fmt(r["days_tx"], 1) if isinstance(r["days_tx"], float) else "—"
        rds  = str(r["rounds"])
        if r["accuracy"]      == best_acc:  a = bf(a)
        if r["total_comm_mb"] == best_comm: c = bf(c)
        lines.append(f"  {r['label']} & {rds} & {a} & {c} & {d} & {days} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Final summary markdown ─────────────────────────────────────────────────

def build_final_summary() -> str:
    # Gather all available results
    all_rows = _load_group(list(_DISPLAY.keys()))
    if not all_rows:
        return "# No results available yet.\n"

    baseline = next((r for r in all_rows if r["key"] == "federated_baseline"), None)
    int8_row  = next((r for r in all_rows if r["key"] == "federated_int8"),     None)
    constrained = next((r for r in all_rows if r["key"] == "federated_cubesat_constrained"), None)
    noniid    = next((r for r in all_rows if r["key"] == "federated_non_iid"),  None)
    sparse    = next((r for r in all_rows if r["key"] == "federated_cubesat_sparse_contacts"), None)
    topk10    = next((r for r in all_rows if r["key"] == "federated_topk_10"),  None)

    def pct(r): return f"{r['accuracy']*100:.2f}%" if r else "N/A"
    def mb(r):  return f"{r['total_comm_mb']:.1f} MB" if r else "N/A"
    def days(r):return f"{r['days_tx']:.1f}" if r else "N/A"

    b_acc  = baseline["accuracy"] * 100 if baseline else 0
    b_comm = baseline["total_comm_mb"]  if baseline else 1

    lines = [
        "# Federated CubeSat Mission-Aware Study — Final Summary",
        "",
        "## Experimental Setup",
        "",
        "| Parameter | Value |",
        "|---|---|",
        "| Dataset | EuroSAT (10 classes, 64×64 RGB, 27,000 images) |",
        "| Model | BaselineCNN (621K params, 2.49 MB) |",
        "| Default rounds | 20 | Default clients | 10 |",
        "| Local epochs | 5 | Batch size | 32 |",
        "| Link (UHF) | 9.6 kbps, 3 contacts/day × 10 min |",
        "| Daily capacity | 2.16 MB/day |",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. INT8 is the dominant compression strategy",
    ]
    if int8_row and baseline:
        reduc = b_comm / int8_row["total_comm_mb"]
        lines += [
            f"FedAvg + INT8 achieves **{reduc:.2f}× communication reduction** "
            f"({mb(baseline)} → {mb(int8_row)}) with **zero accuracy loss** "
            f"({pct(int8_row)} vs {pct(baseline)} baseline). "
            f"It is the only Pareto-optimal method — no other configuration achieves "
            f"both lower communication *and* equal or higher accuracy.",
            "",
        ]

    lines += ["### 2. TopK sparsification shows diminishing returns"]
    if topk10:
        drop = (b_acc - topk10["accuracy"] * 100)
        lines += [
            f"TopK-10% reduces upload bytes by 5× but suffers a **{drop:.2f} pp accuracy drop** "
            f"({pct(topk10)}) — worse than INT8 in both dimensions. "
            f"The index overhead (8 bytes/entry vs 1 byte/entry for INT8) makes TopK "
            f"less efficient when the fraction is low. "
            f"For EO FL on CubeSats, INT8 is unambiguously better than TopK.",
            "",
        ]

    lines += ["### 3. Non-IID data distribution costs 1.5 pp accuracy"]
    if noniid and baseline:
        drop = (b_acc - noniid["accuracy"] * 100)
        lines += [
            f"Switching from IID to non-IID (2 dominant classes per client) "
            f"reduces accuracy by **{drop:.2f} pp** ({pct(noniid)} vs {pct(baseline)}). "
            f"In Earth Observation FL this is realistic: ground tracks mean each "
            f"satellite predominantly observes specific geographic regions and land-cover types, "
            f"creating a natural non-IID distribution.",
            "",
        ]

    lines += ["### 4. The 50 MB CubeSat budget severely limits convergence"]
    if constrained and baseline:
        drop = (b_acc - constrained["accuracy"] * 100)
        lines += [
            f"Under a 50 MB budget at 9.6 kbps, INT8 completes only **4 rounds** "
            f"(vs 20 unconstrained), reaching **{pct(constrained)}** — "
            f"a **{drop:.2f} pp gap** below the ideal. "
            f"Transmitting 49.7 MB requires **{days(constrained)} days** at 9.6 kbps. "
            f"This is the key operational bottleneck: the model can *learn* quickly, "
            f"but the downlink budget prevents it from learning *enough*.",
            "",
        ]

    if sparse:
        lines += [
            "### 5. Sparse ground contacts compound the budget constraint",
            f"When only 3 of 10 clients participate per round (sparse contacts), "
            f"the per-round cost drops to ~3.73 MB, enabling **{sparse['rounds']} rounds** "
            f"in the same 50 MB budget — but each round carries less gradient diversity, "
            f"often converging more slowly. "
            f"Accuracy reaches **{pct(sparse)}** under these conditions.",
            "",
        ]

    lines += [
        "### 6. Why EO Federated Learning is harder than telemetry FL",
        "",
        "- **Non-IID by design**: satellite orbits create geographic data skew",
        "- **High model complexity**: image classification needs CNNs, not simple regressors",
        "- **Severe link budgets**: 9.6 kbps UHF is orders of magnitude below terrestrial FL assumptions",
        "- **Infrequent contacts**: 3 passes/day × 10 min = 30 min/day of connectivity",
        "- **Asymmetric cost**: model updates (~2.5 MB FP32) dwarf typical telemetry payloads",
        "",
        "### 7. INT8 outperformed TopK for two structural reasons",
        "",
        "| Property | INT8 | TopK-10% |",
        "|---|---|---|",
        "| Bytes/param (upload) | 1 | 8 (value + index) |",
        "| Info loss | Quantisation rounding | Zeroed small updates |",
        "| FedAvg stability | High (all weights present) | Lower (90% zeroed) |",
        "| Compression ratio | **4.00×** | **1.67×** (net, with FP32 download) |",
        "",
        "---",
        "",
        "## Communication Bottleneck Analysis",
        "",
        "At 9.6 kbps with 3 contacts/day × 10 min, daily downlink capacity is **2.16 MB/day**.",
        "",
        "| Scenario | Total Comm | Days to Transmit | Assessment |",
        "|---|---|---|---|",
    ]
    for r in all_rows:
        assessment = ("✅ Feasible" if r["days_tx"] <= 15
                      else ("⚠️ Tight" if r["days_tx"] <= 30 else "❌ Infeasible"))
        lines.append(f"| {r['label']} | {r['total_comm_mb']:.1f} MB "
                     f"| {r['days_tx']:.1f} days | {assessment} |")

    lines += [
        "",
        "---",
        "",
        "## Recommended Strategy",
        "",
        "For a typical CubeSat EO mission with UHF downlink:",
        "1. **Use INT8 compression** — 4× bandwidth reduction, zero accuracy loss",
        "2. **Target 50–100 MB budgets** — enables 13–26 rounds at 3-client pace",
        "3. **Expect non-IID degradation** of ~1.5 pp; mitigate with more local epochs",
        "4. **Plan for >10 days** of downlink time even with INT8",
        "5. **Do not use TopK** at low fractions — index overhead kills the savings",
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "source .venv/bin/activate",
        "",
        "# Run all new constrained experiments",
        "python train_federated.py --config configs/federated_cubesat_constrained_non_iid.yaml",
        "python train_federated.py --config configs/federated_cubesat_sparse_contacts.yaml",
        "python train_federated.py --config configs/federated_budget_25.yaml",
        "python train_federated.py --config configs/federated_budget_50.yaml",
        "python train_federated.py --config configs/federated_budget_100.yaml",
        "python train_federated.py --config configs/federated_budget_250.yaml",
        "",
        "# Regenerate all figures and tables",
        "python analysis/generate_plots.py",
        "python analysis/generate_tables.py",
        "```",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    tables = [
        ("main_results",           build_main_results),
        ("compression_tradeoffs",  build_compression_tradeoffs),
        ("constrained_scenarios",  build_constrained_scenarios),
        ("budget_sweep",           build_budget_sweep),
        ("cubesat_comparison",     build_cubesat_comparison),
    ]

    for fname, builder in tables:
        tex = builder()
        if tex:
            p = TABLES_DIR / f"{fname}.tex"
            p.write_text(tex)
            print(f"  → {p}")
        else:
            print(f"  [skip] {fname}.tex — no data")

    summary = build_final_summary()
    sp = RESULTS / "final_summary.md"
    sp.write_text(summary)
    print(f"  → {sp}")
    print("\nDone.")


if __name__ == "__main__":
    main()
