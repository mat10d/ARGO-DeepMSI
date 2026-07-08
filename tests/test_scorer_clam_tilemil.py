"""Contract test for the S4 attention-MIL scorer.

Interface assertions always run. A tiny end-to-end ABMIL train/predict smoke uses
synthetic bags (no disk deps). The full compute smoke needs the built bags and is
skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.clam_tilemil import BAG_DIR, CLAMTileMIL, _make_model, _predict, _train_one

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "clam_tilemil" in list_scorers()
    assert isinstance(get_scorer("clam_tilemil"), CLAMTileMIL)


def test_contract_attributes():
    s = get_scorer("clam_tilemil")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def test_abmil_learns_separable_bags():
    # Positive bags carry a +signal tile; negatives don't. ABMIL should separate them.
    rng = np.random.default_rng(0)
    dim = 16
    bags, y = [], []
    for i in range(24):
        label = i % 2
        b = rng.normal(0, 0.5, (20, dim)).astype(np.float32)
        if label == 1:
            b[0] += 4.0  # a discriminative instance
        bags.append(b)
        y.append(label)
    y = np.array(y)
    tr = np.arange(0, 20)
    te = np.arange(20, 24)
    model = _train_one(bags, y, tr, dim, seed=0, epochs=40, pos_weight=1.0)
    p = _predict(model, bags, te)
    # test bags alternate label starting at index 20 (even=0, odd=1)
    assert p[1] > p[0]  # a positive test bag scores above a negative one
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.skipif(
    not (BAG_DIR / "bags_concat.npy").exists() or not CLEAN_CSV.exists(),
    reason="tile bags / cohort_clean.csv not present in this checkout",
)
def test_compute_batch_smoke(tmp_path):
    import pandas as pd

    from argo_deepmsi.scorers.clam_tilemil import _load_bags

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    loaded = _load_bags(clean_ids)
    assert loaded is not None
    bag_list, idx = loaded
    assert len(bag_list) == len(idx) > 0
    assert bag_list[0].ndim == 2
