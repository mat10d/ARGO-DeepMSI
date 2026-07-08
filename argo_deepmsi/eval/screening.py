"""Screening / rule-out operating-point metrics for MSI pre-screening.

MSIntuit and comparable pre-screening tools are NOT judged by AUROC — they are
judged at a fixed high-sensitivity operating point:

    "at sensitivity >= s, what specificity (and NPV) do we hold?"

This module adds the operating-point metric block that makes every ARGO scorer
directly comparable to:

    MSIntuit (Nat Commun 2023): sensitivity 0.96-0.98, specificity 0.46-0.47 (verified).
    FM-MSI benchmark (ScienceDirect PII S0895611125001892): CONCH + few-shot/cluster
        adaptation, externally validated — its exact spec@sens is TO BE TRANSCRIBED from the
        full text (see FM_BENCHMARK_CONCH below), not asserted from memory.

All functions operate on a 1-D score vector (higher = more likely MSI-H) and a
binary label vector y (1 = MSI-H). They are pure and deterministic.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y.shape != scores.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs scores {scores.shape}")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("y must be binary 0/1")
    return y, scores


def threshold_at_sensitivity(y, scores, target_sensitivity: float) -> float:
    """Lowest threshold t (score >= t -> positive) whose sensitivity >= target.

    Choosing the *lowest* such threshold maximises specificity subject to the
    sensitivity floor — the standard rule-out operating point. Returns -inf if
    even the most permissive threshold cannot reach the target (degenerate).
    """
    y, scores = _as_arrays(y, scores)
    pos = scores[y == 1]
    if pos.size == 0:
        return float("-inf")
    # Sensitivity >= target means we must catch a fraction >= target of positives.
    # The threshold that catches exactly ceil(target*P)/P of positives is the
    # (k-th largest) positive score. Use the quantile of positive scores.
    # sensitivity(t) = mean(pos >= t); to guarantee >= target, t <= the
    # (1-target) quantile of positive scores (lower interpolation).
    q = np.quantile(pos, max(0.0, 1.0 - target_sensitivity), method="lower")
    return float(q)


def sensitivity_specificity_at_threshold(y, scores, threshold: float) -> tuple[float, float]:
    y, scores = _as_arrays(y, scores)
    pred = scores >= threshold
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    tp = int((pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    sens = tp / P if P else float("nan")
    spec = tn / N if N else float("nan")
    return sens, spec


def npv_at_threshold(y, scores, threshold: float) -> float:
    """Negative predictive value: P(y=0 | predicted negative)."""
    y, scores = _as_arrays(y, scores)
    pred_neg = scores < threshold
    n_pred_neg = int(pred_neg.sum())
    if n_pred_neg == 0:
        return float("nan")
    tn = int((pred_neg & (y == 0)).sum())
    return tn / n_pred_neg


def specificity_at_sensitivity(y, scores, target_sensitivity: float) -> dict:
    """Full operating-point block at a sensitivity floor.

    Returns achieved sensitivity (>= target by construction, save ties), the
    specificity held there, NPV, the fraction of the cohort ruled out (predicted
    negative), and the threshold used.
    """
    y, scores = _as_arrays(y, scores)
    t = threshold_at_sensitivity(y, scores, target_sensitivity)
    sens, spec = sensitivity_specificity_at_threshold(y, scores, t)
    npv = npv_at_threshold(y, scores, t)
    ruled_out = float((scores < t).mean())
    return {
        "target_sensitivity": float(target_sensitivity),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "npv": float(npv),
        "ruled_out_fraction": ruled_out,
        "threshold": float(t),
    }


def screening_block(
    y,
    scores,
    sensitivity_floors: tuple[float, ...] = (0.90, 0.95, 0.96, 0.98),
) -> dict:
    """The full MSIntuit-comparable metric block for one score vector.

    Keys: op_sens90 / op_sens95 / op_sens96 / op_sens98, each the dict from
    specificity_at_sensitivity; plus flat convenience keys spec_at_sens90/95 etc.
    """
    y, scores = _as_arrays(y, scores)
    block: dict = {}
    for s in sensitivity_floors:
        tag = f"sens{int(round(s * 100))}"
        d = specificity_at_sensitivity(y, scores, s)
        block[f"op_{tag}"] = d
        block[f"spec_at_{tag}"] = d["specificity"]
        block[f"npv_at_{tag}"] = d["npv"]
    return block


# Competitive reference points (for reporting / no-regression context).
# MSIntuit numbers are verified from the paper's abstract (Nat Commun 2023).
MSINTUIT_TARGET = {"sensitivity": (0.96, 0.98), "specificity": (0.46, 0.47)}
# FM-MSI benchmark: "Benchmarking Pathology Foundation Models for Predicting Microsatellite
# Instability in Colorectal Cancer Histopathology" (Computerized Medical Imaging and Graphics,
# 2025; PII S0895611125001892). CONCH prescreening operating points transcribed from the paper.
# NOTE: these are the paper's numbers on TCGA/PAIP (external cohorts) — a reference point for
# reporting, NOT a target we fit to. Our cohort is the Nigerian CRC set only.
FM_BENCHMARK_CONCH = {
    "model": "CONCH",
    "cohorts": ["TCGA", "PAIP"],
    "spec_at_sens90": 0.65,
    "spec_at_sens94": 0.45,
    "source": "Computerized Medical Imaging and Graphics 2025, PII S0895611125001892",
}
