"""Summarize the A1 paired-staining sweep across embedding models.

Reads every `results/analysis/paired_staining/*/metrics.json` and produces
one summary CSV + one comparison figure (5 panels, one per embedding).
Run after `paired_staining.py` has been executed for each model.

Usage:
    python scripts/paired_staining_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("results/analysis/paired_staining")
OUT_CSV = ROOT / "summary.csv"
OUT_FIG = ROOT / "summary_paired.png"


def load_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir():
            continue
        mj = mdir / "metrics.json"
        if not mj.exists():
            continue
        d = json.loads(mj.read_text())
        d["embedding"] = mdir.name
        rows.append(d)
    return pd.DataFrame(rows).set_index("embedding")


def summary_figure(root: Path, outfile: Path) -> None:
    dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "paired_scores.csv").exists()])
    n = len(dirs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, dirs):
        paired = pd.read_csv(d / "paired_scores.csv")
        metrics = json.loads((d / "metrics.json").read_text())
        colors = paired["y"].map({1: "tab:red", 0: "tab:blue"})
        ax.scatter(paired["MSK"], paired["Nigeria"], c=colors, s=25, alpha=0.7, edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
        ax.axhline(0.5, c="lightgray", lw=0.5)
        ax.axvline(0.5, c="lightgray", lw=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("P(MSI-H) — MSK stain")
        subtitle = (
            f"{d.name}\n"
            f"N={metrics['n_patients']}  r={metrics['pearson_r']:.2f}  "
            f"κ={metrics['cohen_kappa_binary@0.5']:.2f}"
        )
        ax.set_title(subtitle, fontsize=10)
    axes[0].set_ylabel("P(MSI-H) — Nigeria stain")
    for label, color in [("MSI-H", "tab:red"), ("MSS", "tab:blue")]:
        axes[-1].scatter([], [], c=color, label=label, s=25)
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("A1 — Within-patient paired staining (retrospective patients)", y=1.02)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_metrics(ROOT)
    cols = [
        "n_patients",
        "cohen_kappa_binary@0.5",
        "pearson_r",
        "pearson_p",
        "wilcoxon_p",
        "mean_abs_diff",
        "mean_msk_minus_nigeria",
    ]
    df = df[cols].round(4)
    df.to_csv(OUT_CSV)
    summary_figure(ROOT, OUT_FIG)

    print(f"\nSummary ({len(df)} embeddings):")
    print(df.to_string())
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
