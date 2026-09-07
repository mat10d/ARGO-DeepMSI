"""Contract test for the A2 stain-norm probe scorer.

Interface assertions always run. The compute smoke needs the re-extracted stain-norm
embedding (produced by the GPU array + merge) and the base CONCH-TITAN embedding; it
is skipped when either is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.stainnorm_probe import BASE_EMB, STAINNORM_DIR, StainNormProbe

EMB_ROOT = Path("results/embeddings")
CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "stainnorm_probe" in list_scorers()
    assert isinstance(get_scorer("stainnorm_probe"), StainNormProbe)


def test_contract_attributes():
    s = get_scorer("stainnorm_probe")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


@pytest.mark.integration
@pytest.mark.skipif(
    not CLEAN_CSV.exists()
    or not (EMB_ROOT / BASE_EMB / "embeddings.npy").exists()
    or not (STAINNORM_DIR / "embeddings.npy").exists(),
    reason="stain-norm / base CONCH-TITAN embedding not present in this checkout",
)
def test_compute_batch_smoke(tmp_path):
    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    s = StainNormProbe()
    s.score_path = tmp_path / "slide_scores.csv"
    df = s.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, write_outputs=True)
    for col in ("slide_id", "patient_id", "site", "y", "p_msih"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
    assert s._last["n_replaced"] > 0
