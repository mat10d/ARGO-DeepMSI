"""Publication-style plots: ROC overlay, AUROC bar chart, per-site breakdown.

Style settings target matplotlib defaults that look clean in print:
Arial fallback, 300 dpi, no chart junk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
}


def apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def roc_overlay(
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    outpath: Path,
    baseline_name: str | None = None,
) -> None:
    """Overlay ROC curves for many scorers on one axis.

    Args:
        scores: name → (y_true, y_score).
        baseline_name: name to draw as a thicker dashed reference line.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    items = sorted(
        scores.items(),
        key=lambda kv: roc_auc_score(kv[1][0], kv[1][1]),
        reverse=True,
    )
    for name, (y, s) in items:
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        kw = dict(lw=1.5)
        if name == baseline_name:
            kw = dict(lw=2.5, ls="--", color="black", alpha=0.7)
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})", **kw)
    ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(outpath)
    plt.close(fig)


def auroc_bar(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    title: str,
    outpath: Path,
    baseline: float | None = None,
    color_col: str | None = None,
) -> None:
    """Bar chart of AUROC values, sorted descending.

    ``df`` rows are (scorer, AUROC, optionally site). If ``baseline`` is
    given, draw a horizontal dashed line at that value (e.g. Wagner).
    """
    apply_style()
    df = df.copy().sort_values(score_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.45 * len(df)), 4))
    colors = None
    if color_col is not None and color_col in df.columns:
        cats = df[color_col].astype("category")
        palette = plt.get_cmap("tab10")(np.linspace(0, 1, max(cats.cat.codes.max() + 1, 1)))
        colors = [palette[c] for c in cats.cat.codes]
    ax.bar(np.arange(len(df)), df[score_col], color=colors)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df[label_col], rotation=35, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    if baseline is not None:
        ax.axhline(baseline, ls="--", c="black", lw=1, alpha=0.5)
        ax.text(len(df) - 1, baseline + 0.01, f"baseline={baseline:.3f}",
                ha="right", va="bottom", fontsize=8, color="black", alpha=0.7)
    fig.savefig(outpath)
    plt.close(fig)


def per_site_grid(
    rows: list[dict],
    title: str,
    outpath: Path,
    baseline_site_map: dict[str, float] | None = None,
) -> None:
    """Heatmap-like grid: scorers (rows) × sites (cols), value = AUROC."""
    apply_style()
    df = pd.DataFrame(rows).pivot(index="scorer", columns="site", values="auroc")
    fig, ax = plt.subplots(figsize=(1.1 * len(df.columns) + 2, 0.45 * len(df) + 1))
    im = ax.imshow(df.values, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if (np.isnan(v) or 0.3 < v < 0.7) else "white",
                    fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label="AUROC")
    fig.savefig(outpath)
    plt.close(fig)
