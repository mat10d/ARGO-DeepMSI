"""A5 — OAUTHC-specific operating-point calibration for the rule-out screener.

A pooled rule-out threshold set for sensitivity 0.95/0.96 on the WHOLE cohort is not
guaranteed to hold that sensitivity ON OAUTHC — if OAUTHC MSI-H scores sit lower than
the pooled positives, the global threshold silently under-catches OAUTHC positives
(unsafe rule-out). This module fits an OAUTHC-specific operating point (threshold, and
optionally isotonic recalibration) on held-out OAUTHC patients and reports the honest
OAUTHC spec@sens / NPV vs the global-threshold baseline.

Everything is patient-level (one score per patient, e.g. Harmony max/√n) and uses
patient-grouped CV over OAUTHC to avoid threshold-selection optimism. No new data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .screening import (
    npv_at_threshold,
    sensitivity_specificity_at_threshold,
    threshold_at_sensitivity,
)

SEED = 42
N_SPLITS = 5


def patient_scores_from_slide_csv(path, score_col: str = "p_msih") -> pd.DataFrame:
    """Aggregate a scorer's slide_scores.csv to one patient row (max/√n)."""
    sdf = pd.read_csv(path)
    rows = []
    for pid, g in sdf.groupby("patient_id"):
        rows.append({
            "patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
            "score": float(g[score_col].max() / np.sqrt(len(g))),
        })
    return pd.DataFrame(rows)


def _op_at_threshold(y, s, t) -> dict:
    sens, spec = sensitivity_specificity_at_threshold(y, s, t)
    return {"sensitivity": float(sens), "specificity": float(spec),
            "npv": float(npv_at_threshold(y, s, t))}


def global_threshold_on_site(pat: pd.DataFrame, site: str, target_sens: float) -> dict:
    """Global threshold (set for target_sens on the FULL cohort) evaluated on `site`."""
    t = threshold_at_sensitivity(pat["y"], pat["score"], target_sens)
    sub = pat[pat["site"] == site]
    return {"threshold": float(t), "target_sens": target_sens, "method": "global",
            **_op_at_threshold(sub["y"].to_numpy(), sub["score"].to_numpy(), t)}


def _cv_site_operating_point(
    sub: pd.DataFrame, target_sens: float, isotonic: bool, n_splits: int, seed: int
) -> dict:
    """Patient-grouped CV over `sub`: fit threshold (+ optional isotonic) on train,
    collect OOF predictions on test, report the aggregate operating point."""
    y = sub["y"].to_numpy()
    s = sub["score"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "npv": float("nan"), "n_splits": 0}
    n_splits = int(min(n_splits, (y == 1).sum(), (y == 0).sum()))
    if n_splits < 2:
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "npv": float("nan"), "n_splits": 0}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=bool)
    oof_cal = np.full(len(y), np.nan, dtype=float)
    thresholds = []
    for tr, te in skf.split(s, y):
        s_tr, s_te = s[tr], s[te]
        if isotonic:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(s_tr, y[tr])
            s_tr_c, s_te_c = iso.transform(s_tr), iso.transform(s_te)
        else:
            s_tr_c, s_te_c = s_tr, s_te
        t = threshold_at_sensitivity(y[tr], s_tr_c, target_sens)
        thresholds.append(t)
        oof_pred[te] = s_te_c >= t
        oof_cal[te] = s_te_c
    P = int((y == 1).sum()); N = int((y == 0).sum())
    tp = int((oof_pred & (y == 1)).sum()); tn = int((~oof_pred & (y == 0)).sum())
    pred_neg = ~oof_pred
    npv = (tn / int(pred_neg.sum())) if pred_neg.sum() else float("nan")
    return {
        "threshold": float(np.mean(thresholds)), "target_sens": target_sens,
        "method": "per_site_isotonic" if isotonic else "per_site_threshold",
        "sensitivity": tp / P if P else float("nan"),
        "specificity": tn / N if N else float("nan"),
        "npv": float(npv), "n_splits": n_splits,
    }


def per_site_operating_points(
    pat: pd.DataFrame, site: str = "OAUTHC",
    target_sens: tuple[float, ...] = (0.95, 0.96),
    n_splits: int = N_SPLITS, seed: int = SEED,
) -> pd.DataFrame:
    """Full A5 table: global vs per-site-threshold vs per-site-isotonic operating
    points on `site`, at each target sensitivity."""
    sub = pat[pat["site"] == site].reset_index(drop=True)
    rows = []
    for ts in target_sens:
        rows.append(global_threshold_on_site(pat, site, ts))
        r_thr = _cv_site_operating_point(sub, ts, isotonic=False, n_splits=n_splits, seed=seed)
        r_iso = _cv_site_operating_point(sub, ts, isotonic=True, n_splits=n_splits, seed=seed)
        rows.append({"target_sens": ts, **r_thr})
        rows.append({"target_sens": ts, **r_iso})
    df = pd.DataFrame(rows)
    df["site"] = site
    df["n_site_patients"] = int(len(sub))
    df["n_site_positives"] = int((sub["y"] == 1).sum())
    return df
