from __future__ import annotations

import numpy as np
import pandas as pd

from argo_deepmsi.w1_peft_training import _one_slide_per_patient


def test_peft_epoch_samples_exactly_one_slide_per_patient_reproducibly():
    index = pd.DataFrame(
        {
            "slide_id": ["a1", "a2", "b1", "c1", "c2", "c3"],
            "patient_id": ["a", "a", "b", "c", "c", "c"],
        }
    )
    rows = np.arange(len(index))
    selected = _one_slide_per_patient(index, rows, seed=42, epoch=1)
    repeat = _one_slide_per_patient(index, rows, seed=42, epoch=1)
    np.testing.assert_array_equal(selected, repeat)
    assert index.iloc[selected]["patient_id"].nunique() == 3
    assert len(selected) == 3
