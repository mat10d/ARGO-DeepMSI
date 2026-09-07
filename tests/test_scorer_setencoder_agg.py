"""Contract test for the S5 Deep-Sets patient-bag aggregator.

Interface assertions always run. A tiny Deep-Sets learn smoke on synthetic bags checks
the model + train/predict path. The full compute smoke needs Wagner scores + TITAN
embeddings and is skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.setencoder_agg import (
    TITAN_DIR,
    WAGNER_CSV,
    SetEncoderAgg,
    _predict,
    _train,
)

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "setencoder_agg" in list_scorers()
    assert isinstance(get_scorer("setencoder_agg"), SetEncoderAgg)


def test_contract_attributes():
    s = get_scorer("setencoder_agg")
    assert s.resolution == "patient"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "patient_scores.csv"


def test_deepsets_learns_separable_bags():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    dim = 8
    bags, y = [], []
    for i in range(24):
        label = i % 2
        n = rng.integers(1, 4)
        b = rng.normal(2.0 if label == 1 else -2.0, 0.3, (n, dim)).astype(np.float32)
        bags.append(b)
        y.append(label)
    y = np.array(y)
    tr = np.arange(0, 20)
    te = np.arange(20, 24)
    scaler = StandardScaler().fit(np.vstack([bags[i] for i in tr]))
    model = _train(bags, y, tr, dim, seed=0, scaler=scaler, pos_weight=1.0)
    p = _predict(model, bags, te, scaler)
    assert p[1] > p[0]  # a positive test bag scores above a negative one
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.integration
@pytest.mark.skipif(
    not WAGNER_CSV.exists() or not (TITAN_DIR / "embeddings.npy").exists() or not CLEAN_CSV.exists(),
    reason="Wagner scores / TITAN embeddings / cohort_clean.csv not present",
)
def test_load_bags_smoke():
    # Exercise bag assembly on the real cohort without the (slow) full training,
    # which the runner does on a compute node.
    import pandas as pd

    from argo_deepmsi.scorers.setencoder_agg import _load_bags

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    loaded = _load_bags(clean_ids)
    assert loaded is not None
    bags, y, sites, pids, dim = loaded
    assert len(bags) == len(y) == len(pids) > 0
    assert bags[0].ndim == 2 and bags[0].shape[1] == dim
    assert set(np.unique(y)).issubset({0, 1})
