from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argo_deepmsi.w1 import make_patient_fold_manifest


def _synthetic_cohort() -> pd.DataFrame:
    rows = []
    for patient in range(20):
        for slide in range(1 + patient % 3):
            rows.append(
                {
                    "slide_id": f"s{patient}_{slide}",
                    "patient_id": f"p{patient}",
                    "site": f"site{patient % 2}",
                    "patient_cohort": f"site{patient % 2}",
                    "y": int(patient % 4 == 0),
                    "in_primary_set": 1,
                }
            )
    return pd.DataFrame(rows)


def test_fold_manifest_is_patient_unique_complete_and_reproducible():
    cohort = _synthetic_cohort()
    folds1, contract1 = make_patient_fold_manifest(cohort, n_splits=5, seed=42)
    folds2, contract2 = make_patient_fold_manifest(cohort, n_splits=5, seed=42)
    pd.testing.assert_frame_equal(folds1, folds2)
    assert contract1 == contract2
    assert len(folds1) == cohort["patient_id"].nunique()
    assert set(folds1["outer_fold"]) == set(range(5))
    assert folds1["patient_id"].is_unique
    assert folds1["n_slides"].sum() == len(cohort)


def test_fold_manifest_rejects_inconsistent_patient_labels():
    cohort = _synthetic_cohort()
    cohort.loc[cohort["patient_id"] == "p1", "y"] = np.arange(
        (cohort["patient_id"] == "p1").sum()
    )
    with pytest.raises(ValueError, match="inconsistent labels"):
        make_patient_fold_manifest(cohort, n_splits=5, seed=42)


def test_fold_manifest_requires_enough_positive_patients():
    cohort = _synthetic_cohort()
    cohort["y"] = 0
    with pytest.raises(ValueError):
        make_patient_fold_manifest(cohort, n_splits=5, seed=42)
