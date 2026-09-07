from pathlib import Path

import numpy as np
import pandas as pd

import argo_deepmsi.scorers.nested_linear_probe as nested_module
from argo_deepmsi.scorers.nested_linear_probe import NestedLinearProbe
from argo_deepmsi.scorers.registry import get_scorer


def test_nested_probe_contract():
    scorer = get_scorer("nested_linear_probe")
    assert isinstance(scorer, NestedLinearProbe)
    assert scorer.needs_training_on_our_data is True
    assert scorer.resolution == "slide"
    assert scorer.patient_aggregation == "mean"
    assert scorer.primary_score == "p_msih"


def test_canonical_cache_matches_slide_ids_not_parameter_tuple(tmp_path: Path, monkeypatch):
    cohort_path = tmp_path / "cohort.csv"
    cohort = pd.DataFrame(
        {
            "slide_id": ["s2", "s1"],
            "patient_id": ["p2", "p1"],
            "site": ["b", "a"],
            "y": [1, 0],
        }
    )
    cohort.to_csv(cohort_path, index=False)
    embedding_root = tmp_path / "embeddings"
    for name in nested_module.EMBEDDINGS:
        directory = embedding_root / name
        directory.mkdir(parents=True)
        cohort[["slide_id"]].to_csv(directory / "metadata.csv", index=False)
        np.save(directory / "embeddings.npy", np.ones((2, 3), dtype=np.float32))

    cache = tmp_path / "slide_scores.csv"
    pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "patient_id": ["p1", "p2"],
            "site": ["a", "b"],
            "y": [0, 1],
            "p_msih": [0.1, 0.9],
        }
    ).to_csv(cache, index=False)
    monkeypatch.setattr(nested_module, "COHORT", cohort_path)
    monkeypatch.setattr(nested_module, "EMB_ROOT", embedding_root)
    scorer = NestedLinearProbe()
    scorer.score_path = cache

    result = scorer.compute_batch(
        pd.DataFrame(),
        embedding_root=embedding_root,
        cohort_file=cohort_path,
        write_outputs=False,
    )
    assert result["slide_id"].tolist() == ["s1", "s2"]
    assert result["p_msih"].tolist() == [0.1, 0.9]
