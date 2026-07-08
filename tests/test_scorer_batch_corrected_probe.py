"""Contract test for the A1 batch-correction-sweep scorer.

Interface assertions always run (registration + Scorer contract). The compute
smoke exercises the real raw CONCH-TITAN embedding on the clean cohort with a
cheap subset of methods, skipped when artifacts are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.batch_corrected_probe import (
    RAW_EMB,
    BatchCorrectedProbe,
    _oauthc_target,
)

EMB_ROOT = Path("results/embeddings")
CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "batch_corrected_probe" in list_scorers()
    s = get_scorer("batch_corrected_probe")
    assert isinstance(s, BatchCorrectedProbe)


def test_contract_attributes():
    s = get_scorer("batch_corrected_probe")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def test_oauthc_target_only_transforms_oauthc():
    # OAUTHC rows are re-centered to the reference; other sites are untouched.
    X = np.array([[0.0, 0.0], [10.0, 10.0], [11.0, 9.0], [1.0, -1.0]], dtype=np.float32)
    sites = np.array(["OAUTHC", "retrospective_oau", "retrospective_oau", "OAUTHC"])
    out = _oauthc_target(X, sites, sites == "retrospective_oau")
    # retro rows unchanged
    assert np.allclose(out[1], X[1]) and np.allclose(out[2], X[2])
    # OAUTHC rows moved toward the retro mean
    assert not np.allclose(out[0], X[0])


@pytest.mark.skipif(
    not CLEAN_CSV.exists() or not (EMB_ROOT / RAW_EMB / "embeddings.npy").exists(),
    reason="raw CONCH-TITAN embedding / cohort_clean.csv not present in this checkout",
)
def test_compute_batch_smoke(tmp_path):
    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    s = BatchCorrectedProbe()
    s.score_path = tmp_path / "slide_scores.csv"
    df = s.compute_batch(
        pd.DataFrame(),
        clean_slide_ids=clean_ids,
        methods=("raw", "oauthc_target_retrooau"),
        write_outputs=True,
    )
    for col in ("slide_id", "patient_id", "site", "y", "p_msih"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
    assert (tmp_path / "sweep.csv").exists()
