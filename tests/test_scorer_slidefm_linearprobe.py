"""Contract test for the S1 slide-FM linear-probe scorer.

Interface assertions always run (registration + Scorer contract). The compute
smoke exercises one real embedding on the clean cohort and is skipped when the
embeddings / cohort artifacts are not present in this checkout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.slidefm_linearprobe import EMBEDDINGS, SlideFMLinearProbe

EMB_ROOT = Path("results/embeddings")
CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "slidefm_linearprobe" in list_scorers()
    s = get_scorer("slidefm_linearprobe")
    assert isinstance(s, SlideFMLinearProbe)


def test_contract_attributes():
    s = get_scorer("slidefm_linearprobe")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def _first_available_embedding() -> str | None:
    for emb_name, _ in EMBEDDINGS:
        if (EMB_ROOT / emb_name / "embeddings.npy").exists():
            return emb_name
    return None


@pytest.mark.skipif(
    not CLEAN_CSV.exists() or _first_available_embedding() is None,
    reason="slide-FM embeddings / cohort_clean.csv not present in this checkout",
)
def test_compute_batch_smoke(tmp_path):
    emb = _first_available_embedding()
    disp = dict(EMBEDDINGS)[emb]
    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}

    s = SlideFMLinearProbe()
    s.score_path = tmp_path / "slide_scores.csv"  # don't clobber canonical output
    df = s.compute_batch(
        pd.DataFrame(),
        clean_slide_ids=clean_ids,
        embeddings=((emb, disp),),
        full_curve=False,
        write_outputs=True,
    )
    for col in ("slide_id", "patient_id", "site", "y", "p_msih"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
    assert len(df) > 0
    assert s.score_path.exists()
