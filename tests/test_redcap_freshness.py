from __future__ import annotations

import pandas as pd

from scripts.audit_redcap_freshness import _field_delta, stable_patient_key


def test_stable_patient_key_collapses_retrospective_records():
    frame = pd.DataFrame({
        "batch_number": ["1", "2", "3"],
        "record_id": ["r1", "r2", "r3"],
        "crc_redcap_number": ["crc1", "crc1", None],
    })
    assert stable_patient_key(frame).tolist() == [
        "retro:crc1",
        "retro:crc1",
        "pros:r3",
    ]


def test_field_delta_normalises_numeric_scores():
    local = pd.DataFrame({
        "_key": ["pros:r1", "pros:r2"],
        "isMSIH": ["MSS", "MSS"],
        "cmo_msi_score": ["20.0", None],
    })
    live = pd.DataFrame({
        "_key": ["pros:r1", "pros:r2"],
        "isMSIH": ["MSS", "MSI-H"],
        "cmo_msi_score": ["20", "4.5"],
    })
    delta = _field_delta(local, live, set(local["_key"]))
    assert delta["cmo_msi_score"]["changed"] == 1
    assert delta["cmo_msi_score"]["newly_populated"] == 1
    assert delta["isMSIH"]["changed"] == 1
