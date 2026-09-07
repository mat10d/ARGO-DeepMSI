"""Contract test for the T1 selective-abstention screener.

Interface assertions always run. Unit tests cover the risk-coverage curve and split-conformal
logic on synthetic scores. The compute smoke needs the champion (Wagner) scores and is skipped
when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.selective_abstention import (
    WAGNER_CSV,
    SelectiveAbstention,
    risk_coverage_curve,
    split_conformal,
)

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "selective_abstention" in list_scorers()
    assert isinstance(get_scorer("selective_abstention"), SelectiveAbstention)


def test_contract_attributes():
    s = get_scorer("selective_abstention")
    assert s.resolution == "patient"
    assert s.needs_training_on_our_data is False
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "patient_scores.csv"


def test_risk_coverage_monotone_ish():
    # Perfectly ranked scores: abstaining on the least confident should not increase risk.
    rng = np.random.default_rng(0)
    y = (np.arange(100) < 30).astype(int)
    score = np.where(y == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100))
    thr = 0.5
    pts = risk_coverage_curve(y, score, thr)
    full_risk = [p["risk"] for p in pts if abs(p["coverage"] - 1.0) < 0.03][0]
    low_cov_risk = pts[0]["risk"]  # most-confident 10%
    assert low_cov_risk <= full_risk + 1e-9
    assert all(0.0 <= p["risk"] <= 1.0 for p in pts)


def test_split_conformal_bounds_risk():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    score = np.clip(y + rng.normal(0, 0.4, n), 0, 1)  # informative but noisy
    site = np.where(np.arange(n) < n // 2, "A", "B")
    out = split_conformal(y, score, site, thr=0.5, target_risk=0.10)
    assert 0.0 <= out["guaranteed_coverage"] <= 1.0
    assert "per_site_coverage" in out and set(out["per_site_coverage"]) == {"A", "B"}


@pytest.mark.integration
@pytest.mark.skipif(
    not WAGNER_CSV.exists() or not CLEAN_CSV.exists(),
    reason="Wagner scores / cohort_clean.csv not present",
)
def test_compute_batch_smoke(tmp_path):
    import pandas as pd

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    s = SelectiveAbstention()
    s.score_path = tmp_path / "patient_scores.csv"
    pat = s.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids)
    for col in ("patient_id", "y", "site", "p_msih"):
        assert col in pat.columns
    assert len(pat) > 0
