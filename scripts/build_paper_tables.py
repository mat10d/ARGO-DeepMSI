"""F2 — freeze the manuscript deliverables: head-to-head-vs-MSIntuit table + figures.

No new modeling. Reads the frozen leaderboard + per-scorer metrics + R1 (κ) + T3 (fairness)
and writes:

    results/comparison/head_to_head_msintuit.csv   ARGO vs MSIntuit vs FM-MSI benchmark
    results/comparison/figure_msintuit_gap.png     spec@sens operating points vs MSIntuit
    results/comparison/figure_leaderboard.png      clean patient AUROC, all scorers
    results/comparison/figure_per_site.png         per-site AUROC, champion vs fusion
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argo_deepmsi.eval.screening import FM_BENCHMARK_CONCH, MSINTUIT_TARGET

COMP = Path("results/comparison")
plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _lb() -> pd.DataFrame:
    return pd.read_csv(COMP / "leaderboard.csv")


def _row(lb, scorer):
    r = lb[lb["scorer"] == scorer]
    return r.iloc[0] if len(r) else None


def _r1_kappa() -> float | None:
    p = Path("results/scorers/clam_tilemil_stainaug/metrics.json")
    return json.loads(p.read_text()).get("inter_condition_kappa") if p.exists() else None


def _t3() -> dict:
    p = Path("results/analysis/fairness/fairness_report.json")
    return json.loads(p.read_text()) if p.exists() else {}


def head_to_head():
    lb = _lb()
    champ = _row(lb, "calibrated_pool")
    fusion = _row(lb, "fusion_top3")
    kappa = _r1_kappa()
    rows = []
    for name, r, ck in [("ARGO champion (calibrated_pool)", champ, kappa),
                        ("ARGO fusion (top-3 stack)", fusion, None)]:
        if r is None:
            continue
        rows.append({
            "method": name, "cohort": "Nigerian CRC (ours, n=181 pt)",
            "sensitivity": 0.95,
            "spec_at_sens90": round(float(r["spec_at_sens90"]), 3),
            "spec_at_sens95": round(float(r["spec_at_sens95"]), 3),
            "spec_at_sens96": round(float(r["spec_at_sens96"]), 3),
            "npv_at_sens95": round(float(r["npv_at_sens95"]), 3),
            "kappa": round(ck, 3) if ck is not None else "",
            "auroc": round(float(r["patient_auroc_clean"]), 3),
            "notes": "zero-param max/√n" if "champion" in name else "LR stack of 3 signals",
        })
    # MSIntuit (verified from paper abstract)
    rows.append({
        "method": "MSIntuit CRC (Owkin, Nat Commun 2023)", "cohort": "external (PRECISE, ~600 slides)",
        "sensitivity": "0.96-0.98", "spec_at_sens90": "", "spec_at_sens95": "",
        "spec_at_sens96": "0.46-0.47", "npv_at_sens95": "high (rule-out)", "kappa": 0.82,
        "auroc": "", "notes": "reference target (spec @ sens 0.96-0.98)",
    })
    if FM_BENCHMARK_CONCH:
        rows.append({
            "method": "FM-MSI benchmark (CONCH, CMIG 2025)",
            "cohort": "external (TCGA/PAIP)", "sensitivity": "0.90 / 0.94",
            "spec_at_sens90": FM_BENCHMARK_CONCH["spec_at_sens90"], "spec_at_sens95": "",
            "spec_at_sens96": "", "npv_at_sens95": "", "kappa": "",
            "auroc": "", "notes": f"spec 0.45 @ sens 0.94; {FM_BENCHMARK_CONCH['source']}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMP / "head_to_head_msintuit.csv", index=False)
    return df


def fig_msintuit_gap():
    lb = _lb()
    champ = _row(lb, "calibrated_pool")
    fusion = _row(lb, "fusion_top3")
    labels = ["ARGO\nchampion", "ARGO\nfusion", "MSIntuit\ntarget"]
    spec95 = [float(champ["spec_at_sens95"]), float(fusion["spec_at_sens95"]) if fusion is not None else np.nan, np.nan]
    spec96 = [float(champ["spec_at_sens96"]), float(fusion["spec_at_sens96"]) if fusion is not None else np.nan,
              np.mean(MSINTUIT_TARGET["specificity"])]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(x - w / 2, spec95, w, label="spec @ sens 0.95", color="#4C72B0")
    ax.bar(x + w / 2, spec96, w, label="spec @ sens 0.96", color="#C44E52")
    ax.axhline(np.mean(MSINTUIT_TARGET["specificity"]), ls="--", c="gray", lw=1)
    ax.annotate("MSIntuit 0.46-0.47 @ sens 0.96", (0, np.mean(MSINTUIT_TARGET["specificity"]) + 0.01),
                fontsize=7, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("specificity"); ax.set_ylim(0, 0.55)
    ax.set_title("Rule-out operating point vs MSIntuit", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(COMP / "figure_msintuit_gap.png"); plt.close(fig)


def fig_leaderboard():
    lb = _lb().sort_values("patient_auroc_clean")
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#C44E52" if s in ("calibrated_pool", "wagner_zeroshot") else "#4C72B0"
              for s in lb["scorer"]]
    ax.barh(lb["scorer"], lb["patient_auroc_clean"], color=colors)
    ax.axvline(0.5, ls=":", c="gray", lw=1)
    ax.set_xlabel("patient AUROC (clean cohort)"); ax.set_xlim(0.4, 0.75)
    ax.set_title("MSI scorers on the clean cohort (champion in red)", fontsize=9)
    fig.savefig(COMP / "figure_leaderboard.png"); plt.close(fig)


def fig_per_site():
    ps = pd.read_csv(COMP / "per_site_clean.csv")
    champ = ps[ps["scorer"] == "calibrated_pool"].set_index("site")["auroc"]
    fus = ps[ps["scorer"] == "fusion_top3"].set_index("site")["auroc"] if \
        (ps["scorer"] == "fusion_top3").any() else None
    sites = list(champ.index)
    x = np.arange(len(sites))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    if fus is not None:
        w = 0.38
        ax.bar(x - w / 2, champ.values, w, label="champion", color="#C44E52")
        ax.bar(x + w / 2, [fus.get(s, np.nan) for s in sites], w, label="fusion", color="#4C72B0")
    else:
        ax.bar(x, champ.values, 0.6, label="champion", color="#C44E52")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(sites, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("patient AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-site AUROC (OAUTHC is the floor)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(COMP / "figure_per_site.png"); plt.close(fig)


def main():
    df = head_to_head()
    fig_msintuit_gap()
    fig_leaderboard()
    fig_per_site()
    print("head_to_head_msintuit.csv:")
    print(df.to_string(index=False))
    t3 = _t3()
    if t3:
        print(f"\nFairness gate: {'PASS' if t3['fairness_gate']['pass'] else 'FAIL'} "
              f"(outliers: {t3['fairness_gate']['disqualifying_outliers']})")
    print(f"\nfigures: {sorted(p.name for p in COMP.glob('figure_*.png'))}")


if __name__ == "__main__":
    main()
