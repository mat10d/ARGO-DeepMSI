"""Contract test for the R3 fmMAP probe scorer.

Interface assertions always run. A tiny fold smoke on synthetic features checks the
site-residualize + supervised-UMAP + LR path. The load smoke needs TITAN embeddings and
is skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.fmmap_probe import TITAN_DIR, FmmapProbe, _fmmap_fold, _site_residualize

CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "fmmap_probe" in list_scorers()
    assert isinstance(get_scorer("fmmap_probe"), FmmapProbe)


def test_contract_attributes():
    s = get_scorer("fmmap_probe")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def test_site_residualize_removes_site_shift():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.1, (40, 5)).astype(np.float32)
    site = np.array(["A"] * 20 + ["B"] * 20)
    X[site == "A"] += 5.0  # big per-site shift
    Rtr, Rte = _site_residualize(X, site, X[:4], site[:4])
    # after residualization each site's mean is ~0
    assert abs(Rtr[site == "A"].mean()) < 1e-4
    assert abs(Rtr[site == "B"].mean()) < 1e-4


def test_fmmap_fold_separates_classes():
    rng = np.random.default_rng(0)
    dim = 20
    n = 80
    y = (np.arange(n) % 2).astype(int)
    X = rng.normal(0, 0.4, (n, dim)).astype(np.float32)
    X[y == 1] += 2.0
    site = np.where(np.arange(n) < n // 2, "A", "B")
    tr = np.arange(0, 64)
    te = np.arange(64, 80)
    p = _fmmap_fold(X[tr], y[tr], site[tr], X[te], site[te], seed=0)
    assert p.shape == (len(te),)
    assert np.all((p >= 0) & (p <= 1))
    assert p[y[te] == 1].mean() > p[y[te] == 0].mean()


@pytest.mark.integration
@pytest.mark.skipif(
    not (TITAN_DIR / "embeddings.npy").exists() or not CLEAN_CSV.exists(),
    reason="TITAN embeddings / cohort_clean.csv not present",
)
def test_load_smoke():
    import pandas as pd

    from argo_deepmsi.scorers.fmmap_probe import _load

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    loaded = _load(clean_ids)
    assert loaded is not None
    X, meta = loaded
    assert X.shape[0] == len(meta) > 0
