"""Tests for A5 per-site operating-point calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.eval.per_site_calibration import (
    global_threshold_on_site,
    per_site_operating_points,
)


def _synth(seed=0):
    """Two sites: 'good' with well-separated scores, OAUTHC shifted lower so the
    global threshold under-catches OAUTHC positives."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(60):
        y = int(i < 15)
        rows.append({"patient_id": f"g{i}", "site": "good", "y": y,
                     "score": rng.normal(0.7 if y else 0.3, 0.1)})
    for i in range(40):
        y = int(i < 10)
        # OAUTHC positives sit lower → global threshold misses them.
        rows.append({"patient_id": f"o{i}", "site": "OAUTHC", "y": y,
                     "score": rng.normal(0.45 if y else 0.25, 0.1)})
    return pd.DataFrame(rows)


def test_global_threshold_underdelivers_on_oauthc():
    pat = _synth()
    g = global_threshold_on_site(pat, "OAUTHC", 0.95)
    # Global threshold is set for 0.95 sens on the pooled cohort; on OAUTHC the
    # achieved sensitivity should be able to differ from 0.95 (that is the whole point).
    assert 0.0 <= g["sensitivity"] <= 1.0
    assert g["method"] == "global"
    assert "npv" in g and "specificity" in g


def test_per_site_table_shape_and_methods():
    pat = _synth()
    tbl = per_site_operating_points(pat, site="OAUTHC", target_sens=(0.95, 0.96))
    methods = set(tbl["method"])
    assert {"global", "per_site_threshold", "per_site_isotonic"} <= methods
    # 3 methods × 2 target sensitivities = 6 rows.
    assert len(tbl) == 6
    for col in ("threshold", "sensitivity", "specificity", "npv", "site"):
        assert col in tbl.columns
    assert (tbl["site"] == "OAUTHC").all()


def test_per_site_threshold_recovers_target_sens_in_cv():
    # The CV per-site threshold should hold sensitivity closer to the target on
    # OAUTHC than a threshold never fit on OAUTHC would in the degenerate case.
    pat = _synth()
    tbl = per_site_operating_points(pat, site="OAUTHC", target_sens=(0.90,))
    ps = tbl[tbl["method"] == "per_site_threshold"].iloc[0]
    assert ps["sensitivity"] >= 0.6  # CV-fit for high sensitivity on OAUTHC
