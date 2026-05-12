"""Evaluation metrics: AUROC, AUPRC, per-site breakdown, patient aggregation.

The canonical evaluation flow:

    slide_scores → aggregate_to_patient(max/√n) → roc_auc(patient)
                ↘ roc_auc(slide)
                ↘ per_site_breakdown(slide)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PATIENT_AGG: dict[str, Callable[[pd.Series], float]] = {
    "max_sqrtn": lambda v: float(v.max() / np.sqrt(len(v))),
    "max": lambda v: float(v.max()),
    "mean": lambda v: float(v.mean()),
    "median": lambda v: float(v.median()),
    "top3_mean": lambda v: float(np.sort(v)[-min(3, len(v)):].mean()),
}


def aggregate_to_patient(
    slide_df: pd.DataFrame,
    score_col: str,
    agg: str = "max_sqrtn",
    group_col: str = "patient_id",
    label_col: str = "y",
    site_col: str = "site",
) -> pd.DataFrame:
    """Collapse slide scores → patient scores.

    Returns a DataFrame with one row per patient and columns
    ``patient_id, y, site, n_slides, score``.
    """
    fn = PATIENT_AGG[agg]
    rows = []
    for pid, g in slide_df.groupby(group_col):
        sub = g.dropna(subset=[score_col])
        if len(sub) == 0:
            continue
        rows.append(
            {
                group_col: pid,
                label_col: int(sub[label_col].iloc[0]),
                site_col: sub[site_col].iloc[0],
                "n_slides": int(len(sub)),
                "score": fn(sub[score_col]),
            }
        )
    return pd.DataFrame(rows)


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def per_site_breakdown(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "y",
    site_col: str = "site",
) -> dict[str, dict]:
    """AUROC/AUPRC per site at whatever resolution ``df`` is at.

    Sites with a single class are reported with ``note='single-class'``.
    """
    out: dict[str, dict] = {}
    for s, sub in df.groupby(site_col):
        if sub[label_col].nunique() == 2:
            out[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub[label_col].mean()),
                "auroc": _safe_auroc(sub[label_col].to_numpy(), sub[score_col].to_numpy()),
                "auprc": _safe_auprc(sub[label_col].to_numpy(), sub[score_col].to_numpy()),
            }
        else:
            out[s] = {
                "n": int(len(sub)),
                "prevalence": float(sub[label_col].mean()),
                "note": "single-class",
            }
    return out


def evaluate_scorer(
    slide_df: pd.DataFrame,
    score_col: str,
    agg: str = "max_sqrtn",
    label_col: str = "y",
    site_col: str = "site",
    group_col: str = "patient_id",
) -> dict:
    """End-to-end evaluation: slide + patient AUROC/AUPRC + per-site.

    Returns a dict ready to dump to JSON.
    """
    sd = slide_df.dropna(subset=[score_col]).copy()
    pat = aggregate_to_patient(
        sd, score_col, agg=agg, group_col=group_col, label_col=label_col, site_col=site_col
    )
    return {
        "n_slides": int(len(sd)),
        "n_patients": int(len(pat)),
        "prevalence_slide": float(sd[label_col].mean()) if len(sd) else float("nan"),
        "prevalence_patient": float(pat[label_col].mean()) if len(pat) else float("nan"),
        "patient_agg": agg,
        "slide_auroc": _safe_auroc(sd[label_col].to_numpy(), sd[score_col].to_numpy()),
        "slide_auprc": _safe_auprc(sd[label_col].to_numpy(), sd[score_col].to_numpy()),
        "patient_auroc": _safe_auroc(pat[label_col].to_numpy(), pat["score"].to_numpy()),
        "patient_auprc": _safe_auprc(pat[label_col].to_numpy(), pat["score"].to_numpy()),
        "per_site_slide": per_site_breakdown(sd, score_col, label_col=label_col, site_col=site_col),
        "per_site_patient": per_site_breakdown(pat, "score", label_col=label_col, site_col=site_col),
    }
