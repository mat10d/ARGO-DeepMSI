from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.w1_training import _patient_split


def test_inner_split_is_patient_disjoint_and_outer_test_absent():
    rows = []
    for patient in range(40):
        fold = patient % 5
        for slide in range(1 + patient % 2):
            rows.append(
                {
                    "slide_id": f"s{patient}_{slide}",
                    "patient_id": f"p{patient}",
                    "y": int(patient % 4 == 0),
                    "outer_fold": fold,
                }
            )
    index = pd.DataFrame(rows)
    outer_train = np.flatnonzero(index["outer_fold"].to_numpy() != 0)
    train, validation = _patient_split(index, outer_train, seed=42, n_splits=4)
    train_patients = set(index.iloc[train]["patient_id"])
    validation_patients = set(index.iloc[validation]["patient_id"])
    test_patients = set(index[index["outer_fold"] == 0]["patient_id"])
    assert not train_patients & validation_patients
    assert not train_patients & test_patients
    assert not validation_patients & test_patients
    assert train_patients | validation_patients == set(
        index.iloc[outer_train]["patient_id"]
    )
