"""Contract test for the R2 FLEX bottleneck scorer.

Interface assertions always run. A tiny train/predict smoke on synthetic features checks
the multi-loss (MSI + GRL site + text anchor) path. The load smoke needs TITAN
embeddings + text prototypes and is skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.flex_bottleneck import (
    PROTO_NPY,
    TITAN_DIR,
    FlexBottleneck,
    _predict,
    _train,
)

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "flex_bottleneck" in list_scorers()
    assert isinstance(get_scorer("flex_bottleneck"), FlexBottleneck)


def test_contract_attributes():
    s = get_scorer("flex_bottleneck")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def test_flex_learns_separable_features():
    rng = np.random.default_rng(0)
    dim = 32
    n = 60
    y = (np.arange(n) % 2).astype(int)
    X = rng.normal(0, 0.3, (n, dim)).astype(np.float32)
    X[y == 1] += 2.0  # class signal
    site = (np.arange(n) % 3).astype(int)  # a nuisance site variable
    protos = np.stack([np.ones(dim), -np.ones(dim)]).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    tr = np.arange(0, 50)
    te = np.arange(50, 60)
    model = _train(X, y, site, protos, tr, dim, 3, seed=0, epochs=60, pos_weight=1.0)
    p = _predict(model, X, te)
    auroc_ok = ((p[y[te] == 1].mean()) > (p[y[te] == 0].mean()))
    assert auroc_ok
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.integration
@pytest.mark.skipif(
    not (TITAN_DIR / "embeddings.npy").exists() or not PROTO_NPY.exists() or not CLEAN_CSV.exists(),
    reason="TITAN embeddings / text prototypes / cohort_clean.csv not present",
)
def test_load_smoke():
    import pandas as pd

    from argo_deepmsi.scorers.flex_bottleneck import _load

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    loaded = _load(clean_ids)
    assert loaded is not None
    X, meta, protos, n_sites = loaded
    assert X.shape[0] == len(meta) > 0
    assert protos.shape == (2, X.shape[1])
    assert n_sites >= 2
