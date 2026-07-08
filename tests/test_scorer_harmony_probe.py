"""Contract test for the A0 Harmony-corrected CONCH-TITAN linear-probe scorer.

Interface assertions always run (registration + Scorer contract). The compute
smoke exercises the real Harmony embedding on the clean cohort and is skipped
when the embedding / cohort artifacts are not present in this checkout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.harmony_probe import HARMONY_EMB, HarmonyProbe

EMB_ROOT = Path("results/embeddings")
CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "harmony_probe" in list_scorers()
    s = get_scorer("harmony_probe")
    assert isinstance(s, HarmonyProbe)


def test_contract_attributes():
    s = get_scorer("harmony_probe")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


@pytest.mark.skipif(
    not CLEAN_CSV.exists() or not (EMB_ROOT / HARMONY_EMB / "embeddings.npy").exists(),
    reason="Harmony CONCH-TITAN embedding / cohort_clean.csv not present in this checkout",
)
def test_compute_batch_smoke(tmp_path):
    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    s = HarmonyProbe()
    s.score_path = tmp_path / "slide_scores.csv"  # don't clobber canonical output
    df = s.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, write_outputs=True)
    for col in ("slide_id", "patient_id", "site", "y", "p_msih"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
    assert len(df) > 0
    assert s.score_path.exists()
