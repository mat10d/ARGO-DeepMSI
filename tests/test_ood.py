"""Tests for the T2 OOD characterization module (eval/ood.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.eval.ood import (
    knn_ood,
    leave_site_out_ood,
    mahalanobis_ood,
    ood_report,
)


def _synthetic():
    rng = np.random.default_rng(0)
    # 3 in-distribution sites near origin, 1 shifted OOD site far away
    Xin = rng.normal(0, 1, (90, 8)).astype(np.float32)
    sin = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
    Xood = rng.normal(8, 1, (20, 8)).astype(np.float32)
    sood = np.array(["OOD"] * 20)
    X = np.vstack([Xin, Xood])
    sites = np.concatenate([sin, sood])
    return X, sites


def test_mahalanobis_flags_shifted_points():
    X, sites = _synthetic()
    d = mahalanobis_ood(X[sites != "OOD"], X)
    assert d[sites == "OOD"].mean() > d[sites != "OOD"].mean()
    assert np.all(d >= 0)


def test_knn_flags_shifted_points():
    X, sites = _synthetic()
    d = knn_ood(X[sites != "OOD"], X, k=5)
    assert d[sites == "OOD"].mean() > d[sites != "OOD"].mean()


def test_leave_site_out_detects_ood_site():
    X, sites = _synthetic()
    out = leave_site_out_ood(X, sites, method="knn", k=5)
    assert out["OOD"]["ood_auroc"] > 0.9  # the shifted site is easily detected
    # the genuinely-shifted site is far more detectable than an in-distribution site
    # (an in-site can look somewhat OOD here because the leave-out "in" pool still
    # contains the far OOD cluster — the relative ordering is the robust check)
    assert out["OOD"]["ood_auroc"] > out["A"]["ood_auroc"]


def test_ood_report_structure():
    X, sites = _synthetic()
    meta = pd.DataFrame({
        "slide_id": [f"s{i}" for i in range(len(X))],
        "patient_id": [f"p{i // 2}" for i in range(len(X))],
        "site": sites,
        "y": (np.arange(len(X)) % 2),
    })
    rep = ood_report(X, meta, champion_patient=None, k=5)
    assert set(rep["methods"]) == {"knn", "mahalanobis"}
    assert "per_site_ood_auroc" in rep["methods"]["knn"]
    assert "OOD" in rep["methods"]["knn"]["per_site_ood_auroc"]
