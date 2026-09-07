from __future__ import annotations

import json

import pandas as pd

from argo_deepmsi.eval.cohort import (
    build_feature_complete_cohort,
    freeze_manifest,
    rebuild_full_feature_cohort,
)
from argo_deepmsi.eval.metrics import aggregate_to_patient, stratified_patient_bootstrap
from argo_deepmsi.eval.qc_comparison import _cache_matches_estimand


def test_rebuild_full_cohort_preserves_qc_as_sensitivity(tmp_path):
    cohort_path = tmp_path / "cohort.csv"
    clinical_path = tmp_path / "clinical.csv"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame({
        "slide_id": ["a", "b", "c"],
        "patient_id": ["p1", "p1", "p2"],
        "site": ["retrospective_msk", "retrospective_oau", "OAUTHC"],
        "y": [0, 0, 1],
        "n_tiles": [10, 0, 5],
        "in_clean_set": [1, 0, 0],
        "passes_ecrf_qc": [True, False, False],
    }).to_csv(cohort_path, index=False)
    pd.DataFrame({
        "PATIENT": ["p1", "p2"],
        "cmo_msi_status": [None, "Instable"],
        "cmo_msi_score": [None, 20.0],
        "msi_status_mmr": [2, None],
    }).to_csv(clinical_path, index=False)
    manifest_path.write_text(json.dumps({"no_regression_floor": 0.7}))

    cohort, manifest = rebuild_full_feature_cohort(
        cohort_path, clinical_path, manifest_path
    )

    assert cohort["in_primary_set"].tolist() == [1, 0, 1]
    assert cohort["in_qc_sensitivity_set"].tolist() == [1, 0, 0]
    assert cohort.loc[cohort.patient_id == "p1", "patient_cohort"].eq("retrospective").all()
    assert manifest["primary_estimand"]["n_patients"] == 2
    assert manifest["qc_sensitivity_estimand"]["n_patients"] == 1


def test_feature_complete_cohort_is_rebuilt_from_fresh_embeddings(tmp_path):
    slide_table = tmp_path / "slides.csv"
    clinical = tmp_path / "clinical.csv"
    embeddings = tmp_path / "embeddings" / "encoder_mean"
    embeddings.mkdir(parents=True)
    pd.DataFrame(
        {
            "PATIENT": ["p1", "p1", "p2"],
            "FILENAME": ["/slides/a.svs", "/slides/b.svs", "/slides/c.svs"],
            "SITE": ["retrospective_msk", "retrospective_oau", "OAUTHC"],
        }
    ).to_csv(slide_table, index=False)
    pd.DataFrame(
        {"PATIENT": ["p1", "p2"], "isMSIH": ["MSS", "MSI-H"]}
    ).to_csv(clinical, index=False)
    pd.DataFrame(
        {"slide_id": ["a", "b", "c"], "n_tiles": [10, 0, 5]}
    ).to_csv(embeddings / "metadata.csv", index=False)

    cohort, manifest = build_feature_complete_cohort(slide_table, clinical, tmp_path / "embeddings")

    assert cohort["in_primary_set"].tolist() == [1, 1, 0]
    assert cohort.loc[cohort.patient_id == "p1", "patient_cohort"].eq("retrospective").all()
    assert manifest["n_slides_primary"] == 2
    assert manifest["n_patients_primary"] == 2
    assert manifest["n_positive_patients"] == 1


def test_patient_aggregation_uses_stable_retrospective_cohort():
    slides = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "site": ["retrospective_msk", "retrospective_oau", "OAUTHC"],
        "y": [0, 0, 1],
        "score": [0.1, 0.2, 0.9],
    })
    patients = aggregate_to_patient(slides, "score")
    assert patients.set_index("patient_id").loc["p1", "site"] == "retrospective"


def test_stratified_bootstrap_is_deterministic():
    patients = pd.DataFrame({
        "y": [0, 0, 0, 1, 1, 1],
        "score": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
    })
    first = stratified_patient_bootstrap(patients, n_boot=100, seed=7)
    second = stratified_patient_bootstrap(patients, n_boot=100, seed=7)
    assert first == second
    assert first["estimate"] == 1.0


def test_score_cache_must_match_estimand_population():
    full = pd.DataFrame({
        "slide_id": ["s1", "s2", "s3"],
        "patient_id": ["p1", "p2", "p3"],
    })
    sensitivity = full.iloc[:2]
    assert _cache_matches_estimand(
        full,
        resolution="slide",
        eligible_slide_ids={"s1", "s2", "s3"},
        eligible_patient_ids={"p1", "p2", "p3"},
    )
    assert not _cache_matches_estimand(
        sensitivity,
        resolution="slide",
        eligible_slide_ids={"s1", "s2", "s3"},
        eligible_patient_ids={"p1", "p2", "p3"},
    )
    assert not _cache_matches_estimand(
        full,
        resolution="slide",
        eligible_slide_ids={"s1", "s2"},
        eligible_patient_ids={"p1", "p2"},
    )


def test_freeze_ignores_nonconfirmatory_high_score(tmp_path):
    cohort_path = tmp_path / "cohort.csv"
    manifest_path = tmp_path / "manifest.json"
    leaderboard_path = tmp_path / "leaderboard.csv"
    pd.DataFrame({
        "slide_id": ["s1", "s2"],
        "patient_id": ["p1", "p2"],
        "site": ["A", "B"],
        "y": [0, 1],
        "in_primary_set": [1, 1],
    }).to_csv(cohort_path, index=False)
    manifest_path.write_text(json.dumps({"layers": []}))
    pd.DataFrame([
        {
            "scorer": "valid",
            "patient_auroc_clean": 0.7,
            "comparable_primary": True,
            "confirmatory_valid": True,
        },
        {
            "scorer": "selected_on_oof",
            "patient_auroc_clean": 0.99,
            "comparable_primary": True,
            "confirmatory_valid": False,
        },
    ]).to_csv(leaderboard_path, index=False)

    manifest = freeze_manifest(cohort_path, manifest_path, leaderboard_path)
    assert manifest["no_regression_champion"] == "valid"
    assert manifest["no_regression_floor"] == 0.7
