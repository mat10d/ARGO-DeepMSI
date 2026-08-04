from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from argo_deepmsi.eval.validation import nested_grouped_oof


def _factory():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))


def test_nested_grouped_oof_has_no_patient_overlap():
    rng = np.random.default_rng(42)
    n_patients = 40
    slides_per_patient = 2
    patient_y = np.tile([0, 1], n_patients // 2)
    y = np.repeat(patient_y, slides_per_patient)
    frame = pd.DataFrame({
        "slide_id": [f"s{i}" for i in range(len(y))],
        "patient_id": np.repeat([f"p{i}" for i in range(n_patients)], slides_per_patient),
        "site": np.repeat(["A", "B"], len(y) // 2),
        "y": y,
    })
    signal = y[:, None] + rng.normal(0, 0.5, size=(len(y), 3))
    noise = rng.normal(size=(len(y), 3))

    scored, folds = nested_grouped_oof(
        frame,
        {"signal": signal, "noise": noise},
        {"signal": _factory, "noise": _factory},
        outer_splits=4,
        inner_splits=3,
        repeats=2,
    )

    assert len(scored) == len(frame)
    assert scored["p_msih"].between(0, 1).all()
    assert (folds["patient_overlap"] == 0).all()
    assert len(folds) == 8


def test_nested_grouped_oof_rejects_misaligned_candidates():
    frame = pd.DataFrame({
        "slide_id": ["s1", "s2"], "patient_id": ["p1", "p2"],
        "site": ["A", "B"], "y": [0, 1],
    })
    with pytest.raises(ValueError, match="keys must match"):
        nested_grouped_oof(frame, {"x": np.zeros((2, 1))}, {"y": _factory})
