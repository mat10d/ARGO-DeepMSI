"""Contract test for the S3 Tip-Adapter scorer.

Interface assertions always run. The compute smoke needs cached TITAN embeddings +
text prototypes and is skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.tip_adapter import PROTO_NPY, TITAN_DIR, TipAdapter, _cache_delta

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "tip_adapter" in list_scorers()
    assert isinstance(get_scorer("tip_adapter"), TipAdapter)


def test_contract_attributes():
    s = get_scorer("tip_adapter")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def test_cache_delta_prefers_own_class():
    # Support: MSI cluster near +e0, MSS cluster near +e1. A test point near the MSI
    # cluster should get a positive cache delta.
    Fsup = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]], dtype=float)
    Fsup = Fsup / np.linalg.norm(Fsup, axis=1, keepdims=True)
    ysup = np.array([1, 1, 0, 0])
    f = np.array([[1, 0]], dtype=float)
    assert _cache_delta(f, Fsup, ysup)[0] > 0


@pytest.mark.skipif(
    not (TITAN_DIR / "embeddings.npy").exists() or not PROTO_NPY.exists() or not CLEAN_CSV.exists(),
    reason="TITAN embeddings / text prototypes / cohort_clean.csv not present",
)
def test_compute_batch_smoke(tmp_path):
    import pandas as pd

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    s = TipAdapter()
    s.score_path = tmp_path / "slide_scores.csv"
    df = s.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=False)
    for col in ("slide_id", "patient_id", "site", "y", "p_msih", "coop_lite", "zero_shot"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
    assert len(df) > 0
