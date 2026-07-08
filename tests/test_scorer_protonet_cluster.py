"""Contract test for the S2 ProtoNet-cluster scorer.

Interface assertions always run. The compute smoke needs the built cluster
features (scripts/build_protonet_features.py) and is skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argo_deepmsi.scorers import get_scorer, list_scorers
from argo_deepmsi.scorers.protonet_cluster import ProtoNetCluster, _proto_scores

FEAT = Path("results/scorers/protonet_cluster/cluster_features.npy")
CLEAN_CSV = Path("results/data/cohort_clean.csv")


def test_registered():
    assert "protonet_cluster" in list_scorers()
    assert isinstance(get_scorer("protonet_cluster"), ProtoNetCluster)


def test_contract_attributes():
    s = get_scorer("protonet_cluster")
    assert s.resolution == "slide"
    assert s.needs_training_on_our_data is True
    assert s.primary_score == "p_msih"
    assert s.score_path is not None and s.score_path.name == "slide_scores.csv"


def test_proto_scores_separates_classes():
    # Two well-separated clusters → test points score toward their own class.
    rng = np.random.default_rng(0)
    Xtr = np.vstack([rng.normal(-3, 0.1, (10, 4)), rng.normal(3, 0.1, (10, 4))])
    ytr = np.array([0] * 10 + [1] * 10)
    Xte = np.array([[3, 3, 3, 3], [-3, -3, -3, -3]], dtype=float)
    p = _proto_scores(Xtr, ytr, Xte)
    assert p[0] > 0.5 > p[1]
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.skipif(
    not FEAT.exists() or not CLEAN_CSV.exists(),
    reason="cluster features / cohort_clean.csv not present in this checkout",
)
def test_compute_batch_smoke(tmp_path):
    import pandas as pd

    cc = pd.read_csv(CLEAN_CSV)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    s = ProtoNetCluster()
    s.score_path = tmp_path / "slide_scores.csv"
    df = s.compute_batch(pd.DataFrame(), clean_slide_ids=clean_ids, full_curve=False)
    for col in ("slide_id", "patient_id", "site", "y", "p_msih"):
        assert col in df.columns
    assert df["p_msih"].between(0.0, 1.0).all()
    assert len(df) > 0
