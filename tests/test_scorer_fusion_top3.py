"""Contract test for the F1 fusion scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.fusion_top3 import FusionTop3, _agg_patient, _stack_oof

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "fusion_top3" in list_scorers()
    assert isinstance(get_scorer("fusion_top3"), FusionTop3)


def test_contract_attributes():
    s = get_scorer("fusion_top3")
    assert s.resolution == "patient"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "patient_scores.csv"


def test_agg_patient_max_sqrtn():
    df = pd.DataFrame({
        "patient_id": ["p", "p", "q"], "y": [1, 1, 0], "site": ["A", "A", "B"],
        "p_msih": [0.2, 0.8, 0.5],
    })
    out = _agg_patient(df)
    p = out[out["patient_id"] == "p"].iloc[0]
    assert p["n_slides"] == 2
    assert abs(p["score"] - 0.8 / np.sqrt(2)) < 1e-6


def test_stack_oof_learns_from_components():
    # 3 informative base columns → the stacked OOF should rank classes well.
    rng = np.random.default_rng(0)
    n = 80
    y = (rng.random(n) < 0.4).astype(int)
    wide = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "y": y, "site": "A", "n_slides": 1,
        "calibrated_pool": np.clip(y * 0.4 + rng.normal(0.3, 0.2, n), 0, 1),
        "slidefm_linearprobe": np.clip(y * 0.3 + rng.normal(0.3, 0.25, n), 0, 1),
        "flex_bottleneck": np.clip(y * 0.35 + rng.normal(0.3, 0.25, n), 0, 1),
    })
    oof = _stack_oof(wide)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, oof) > 0.7
    assert np.all((oof >= 0) & (oof <= 1))
