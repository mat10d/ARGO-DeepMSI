"""Tests for the T3 fairness evaluation module (eval/fairness.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.eval.fairness import (
    _max_safe_tau,
    fairness_report,
    group_conditional_conformal,
    per_site_metrics,
)


def _cohort(good_sep=True):
    rng = np.random.default_rng(0)
    rows = []
    # two "good" sites where score separates classes, one "bad" site at chance
    for site, sep in [("good1", 2.5), ("good2", 2.5), ("bad", 0.0)]:
        n = 40
        y = (rng.random(n) < 0.25).astype(int)
        base = rng.normal(0, 1, n)
        score = 1 / (1 + np.exp(-(base + sep * y)))  # informative if sep>0
        for i in range(n):
            rows.append({"patient_id": f"{site}_{i}", "y": int(y[i]), "site": site,
                         "n_slides": 1, "p_msih": float(score[i])})
    return pd.DataFrame(rows)


def test_max_safe_tau_controls_for():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(int)
    score = np.clip(y * 0.5 + rng.normal(0.3, 0.3, 200), 0, 1)
    tau = _max_safe_tau(y, score, target_for=0.10)
    keep = score <= tau
    assert np.mean(y[keep]) <= 0.10 + 1e-9  # covered FOR within target


def test_per_site_metrics_keys():
    pat = _cohort()
    m = per_site_metrics(pat, global_tau=0.3)
    assert set(m) == {"good1", "good2", "bad"}
    for d in m.values():
        assert {"n", "auroc", "for_at_global_tau", "coverage_at_global_tau"} <= set(d)


def test_group_conditional_conformal_runs():
    pat = _cohort()
    gcc = group_conditional_conformal(pat, target_for=0.10)
    assert set(gcc) == {"good1", "good2", "bad"}
    for d in gcc.values():
        assert "coverage" in d


def test_fairness_gate_verdict_present():
    pat = _cohort()
    rep = fairness_report(pat, target_for=0.10)
    assert "fairness_gate" in rep
    assert isinstance(rep["fairness_gate"]["pass"], bool)
    assert "coverage_adjusted_gap" in rep
    assert "per_site" in rep and "group_conditional_conformal" in rep
