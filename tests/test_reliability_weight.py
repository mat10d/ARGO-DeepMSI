"""Tests for D2 reliability weighting + weighted patient aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.eval.reliability_weight import (
    FLOOR,
    patient_max_sqrtn,
    reliability_weight,
)


def test_weight_bounds_and_monotonicity():
    df = pd.DataFrame({
        "artifact_fraction": [0.0, 0.0, 1.0, np.nan],
        "tumor_fraction": [0.5, 0.0, 0.5, 0.5],
        "n_tiles": [1000, 1000, 1000, 1000],
    })
    w = reliability_weight(df)
    assert (w >= FLOOR - 1e-9).all() and (w <= 1.0 + 1e-9).all()
    # clean+tumor-rich slide (row 0) is the most reliable of the set
    assert w.iloc[0] == w.max()
    # zero tumor (row 1) and full artifact (row 2) both collapse toward the floor
    assert w.iloc[1] <= 0.1 and w.iloc[2] <= 0.1


def test_weighted_vs_unweighted_downweights_bad_high_slide():
    # A patient with one reliable low slide and one UNRELIABLE high slide.
    slide = pd.DataFrame({
        "patient_id": ["p", "p"], "y": [0, 0], "site": ["OAUTHC", "OAUTHC"],
        "score": [0.2, 0.9], "w": [1.0, FLOOR],
    })
    unw = patient_max_sqrtn(slide, "score", weight_col=None)["score"].iloc[0]
    wtd = patient_max_sqrtn(slide, "score", weight_col="w")["score"].iloc[0]
    # Unweighted is dominated by the 0.9 false-high; weighting pulls it down.
    assert wtd < unw


def test_unweighted_matches_max_over_sqrtn():
    slide = pd.DataFrame({
        "patient_id": ["p"] * 4, "y": [1] * 4, "site": ["msk"] * 4,
        "score": [0.1, 0.4, 0.8, 0.3],
    })
    out = patient_max_sqrtn(slide, "score")
    assert np.isclose(out["score"].iloc[0], 0.8 / np.sqrt(4))
    assert out["n_eff"].iloc[0] == 4
