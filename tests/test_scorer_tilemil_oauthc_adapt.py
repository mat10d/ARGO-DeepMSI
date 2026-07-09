"""Contract test for the A3 OAUTHC-adaptive tile-MIL scorer.

Interface assertions always run. The compute smoke needs the S4 tumor-CONCH bags and
runs a tiny cached-path check; the full train is exercised by the runner, not the gate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.clam_tilemil import BAG_DIR
from argo_deepmsi.scorers.tilemil_oauthc_adapt import TileMILOAUTHCAdapt

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "tilemil_oauthc_adapt" in list_scorers()
    assert isinstance(get_scorer("tilemil_oauthc_adapt"), TileMILOAUTHCAdapt)


def test_contract_attributes():
    s = get_scorer("tilemil_oauthc_adapt")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


@pytest.mark.skipif(
    not CLEAN_CSV.exists() or not (BAG_DIR / "bags_concat.npy").exists(),
    reason="S4 tumor-CONCH bags / cohort_clean.csv not present in this checkout",
)
def test_cached_read(tmp_path):
    # If a canonical slide_scores.csv exists, the cache-first path returns it filtered.
    s = get_scorer("tilemil_oauthc_adapt")
    if s.score_path is None or not s.score_path.exists():
        pytest.skip("no cached slide_scores yet (runner not yet executed)")
    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(x) for x in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    df = s.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids)
    for col in ("slide_id", "patient_id", "site", "y", "p_msih"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
