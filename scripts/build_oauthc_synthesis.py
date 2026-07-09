"""A6 — OAUTHC-recovery synthesis: leaderboard + figure + head-to-head-vs-MSIntuit.

Assembles the A-phase OAUTHC-recovery story from the per-scorer metrics + the leaderboard:
  - results/comparison/oauthc_recovery_leaderboard.csv
  - results/comparison/figure_oauthc_recovery.png
  - results/comparison/head_to_head_msintuit.csv   (updated: full cohort WITH OAUTHC)

No modeling — pure assembly of already-computed results. Run on a compute node.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COMP = Path("results/comparison")
SCORERS = Path("results/scorers")
RETRO_OAU_CEILING = 0.80  # same-institution ceiling (D0)
CHAMPION_OAUTHC = 0.579186  # calibrated_pool / wagner OAUTHC AUROC (per_site_clean)

# A-phase scorers (label, results dir) in narrative order.
A_ROWS = [
    ("A0 Harmony (base)", "harmony_probe"),
    ("A1 batch-correction", "batch_corrected_probe"),
    ("A2 stain-norm (image)", "stainnorm_probe"),
    ("A3 tile-MIL OAUTHC-adapt", "tilemil_oauthc_adapt"),
]


def _metrics(name: str) -> dict:
    p = SCORERS / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def build_recovery_leaderboard() -> pd.DataFrame:
    persite = pd.read_csv(COMP / "per_site_clean.csv")
    oa = persite[persite["site"] == "OAUTHC"].set_index("scorer")["auroc"]
    lb = pd.read_csv(COMP / "leaderboard.csv").set_index("scorer")["patient_auroc_clean"]
    rows = []
    for label, name in A_ROWS:
        m = _metrics(name)
        rows.append({
            "label": label, "scorer": name,
            "oauthc_auroc": float(oa.get(name, float("nan"))),
            "oauthc_spec_at_sens95": m.get("oauthc_spec_at_sens95", float("nan")),
            "oauthc_spec_at_sens96": m.get("oauthc_spec_at_sens96", float("nan")),
            "overall_auroc": float(lb.get(name, float("nan"))),
            "gap_to_retrooau_ceiling": RETRO_OAU_CEILING - float(oa.get(name, float("nan"))),
        })
    # Baselines for context.
    for label, name in [("raw CONCH-TITAN probe", "slidefm_linearprobe"),
                        ("champion (calibrated_pool)", "calibrated_pool")]:
        rows.append({
            "label": label, "scorer": name,
            "oauthc_auroc": float(oa.get(name, float("nan"))),
            "oauthc_spec_at_sens95": float("nan"),
            "oauthc_spec_at_sens96": float("nan"),
            "overall_auroc": float(lb.get(name, float("nan"))),
            "gap_to_retrooau_ceiling": RETRO_OAU_CEILING - float(oa.get(name, float("nan"))),
        })
    df = pd.DataFrame(rows).sort_values("oauthc_auroc", ascending=False).reset_index(drop=True)
    df.to_csv(COMP / "oauthc_recovery_leaderboard.csv", index=False)
    return df


def build_figure(df: pd.DataFrame) -> None:
    d = df.sort_values("oauthc_auroc")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b8cbe" if "A0" in l else ("#a6bddb" if l.startswith("A") else "#cccccc")
              for l in d["label"]]
    ax.barh(d["label"], d["oauthc_auroc"], color=colors)
    ax.axvline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.axvline(CHAMPION_OAUTHC, color="#e34a33", ls="--", lw=1.2, label=f"champion {CHAMPION_OAUTHC:.2f}")
    ax.axvline(RETRO_OAU_CEILING, color="#31a354", ls="--", lw=1.2, label=f"retro-OAU ceiling {RETRO_OAU_CEILING:.2f}")
    ax.set_xlabel("OAUTHC held-out patient AUROC (n=64 pt)")
    ax.set_title("A-phase OAUTHC recovery — Harmony leads, ceiling still 0.12 away")
    ax.set_xlim(0.4, 0.85)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(COMP / "figure_oauthc_recovery.png", dpi=150)
    plt.close(fig)


def update_head_to_head() -> None:
    """Refresh head_to_head_msintuit.csv to foreground the full-cohort (OAUTHC-included)
    A-phase result + the OAUTHC-specific operating point (A5)."""
    harmony = _metrics("harmony_probe")
    a5 = pd.read_csv("results/analysis/per_site_calibration/oauthc_operating_points.csv")
    a5_glob96 = a5[(a5["method"] == "global") & (a5["target_sens"] == 0.96)].iloc[0]

    rows = [
        {"method": "ARGO champion (calibrated_pool)", "cohort": "Nigerian CRC full (ours, n=181 pt, OAUTHC incl.)",
         "sensitivity": 0.95, "spec_at_sens90": 0.193, "spec_at_sens95": 0.179, "spec_at_sens96": 0.079,
         "npv_at_sens95": 0.926, "kappa": 0.927, "auroc": 0.713, "notes": "zero-param max/sqrt(n); OAUTHC AUROC 0.579"},
        {"method": "ARGO A0 Harmony (OAUTHC-recovery base)", "cohort": "Nigerian CRC full (ours, n=181 pt, OAUTHC incl.)",
         "sensitivity": 0.95, "spec_at_sens90": harmony.get("spec_at_sens90"), "spec_at_sens95": harmony.get("spec_at_sens95"),
         "spec_at_sens96": harmony.get("spec_at_sens96"), "npv_at_sens95": harmony.get("npv_at_sens95"),
         "kappa": "", "auroc": harmony.get("auroc"), "notes": "batch-corrected CONCH-TITAN; OAUTHC AUROC 0.683 (best on board)"},
        {"method": "ARGO A0 Harmony @ OAUTHC operating point (A5)", "cohort": "OAUTHC only (ours, n=64 pt)",
         "sensitivity": float(a5_glob96["sensitivity"]), "spec_at_sens90": "", "spec_at_sens95": "",
         "spec_at_sens96": float(a5_glob96["specificity"]), "npv_at_sens95": float(a5_glob96["npv"]),
         "kappa": "", "auroc": 0.683, "notes": "global threshold safe on OAUTHC: sens/NPV 1.0, spec 0.275 (A5)"},
        {"method": "MSIntuit CRC (Owkin, Nat Commun 2023)", "cohort": "external (PRECISE, ~600 slides)",
         "sensitivity": "0.96-0.98", "spec_at_sens90": "", "spec_at_sens95": "", "spec_at_sens96": "0.46-0.47",
         "npv_at_sens95": "high (rule-out)", "kappa": 0.82, "auroc": "", "notes": "reference target (spec @ sens 0.96-0.98)"},
        {"method": "FM-MSI benchmark (CONCH; PII S0895611125001892)", "cohort": "external (TCGA/PAIP)",
         "sensitivity": "", "spec_at_sens90": "", "spec_at_sens95": "", "spec_at_sens96": "", "npv_at_sens95": "",
         "kappa": "", "auroc": "", "notes": "closed-access; full text not obtained; operating points NOT transcribed -- named comparator only"},
    ]
    pd.DataFrame(rows).to_csv(COMP / "head_to_head_msintuit.csv", index=False)


def main() -> None:
    df = build_recovery_leaderboard()
    build_figure(df)
    update_head_to_head()
    print("=== OAUTHC recovery leaderboard ===")
    print(df.to_string(index=False))
    print(f"\nwrote {COMP/'oauthc_recovery_leaderboard.csv'}, figure_oauthc_recovery.png, head_to_head_msintuit.csv")


if __name__ == "__main__":
    main()
