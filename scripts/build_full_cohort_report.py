"""Freeze the v2 full-cohort baseline and uncertainty/sensitivity analyses."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from argo_deepmsi.eval.metrics import (
    aggregate_to_patient,
    paired_bootstrap_auroc_delta,
    per_site_breakdown,
    stratified_patient_bootstrap,
)
from argo_deepmsi.eval.screening import screening_block

COHORT = Path("results/data/cohort_clean.csv")
WAGNER = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
OUT = Path("results/analysis/full_cohort_reset")


def _patient_scores(cohort: pd.DataFrame, scores: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    slides = cohort.loc[mask, ["slide_id", "patient_id", "patient_cohort", "y"]].merge(
        scores[["slide_id", "p_msih"]], on="slide_id", how="inner"
    )
    slides = slides.rename(columns={"patient_cohort": "site"})
    return aggregate_to_patient(slides, "p_msih", agg="max_sqrtn")


def _summary(name: str, patients: pd.DataFrame) -> dict:
    interval = stratified_patient_bootstrap(patients)
    block = screening_block(patients["y"].to_numpy(), patients["score"].to_numpy())
    return {
        "estimand": name,
        "n_patients": int(len(patients)),
        "n_positive": int(patients["y"].sum()),
        "auroc": interval["estimate"],
        "auroc_ci_low": interval["ci_low"],
        "auroc_ci_high": interval["ci_high"],
        "spec_at_sens90": block["spec_at_sens90"],
        "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"],
        "npv_at_sens95": block["npv_at_sens95"],
    }


def main() -> None:
    cohort = pd.read_csv(COHORT)
    scores = pd.read_csv(WAGNER)
    primary = _patient_scores(cohort, scores, cohort["in_primary_set"] == 1)
    qc = _patient_scores(cohort, scores, cohort["in_qc_sensitivity_set"] == 1)
    definite_ids = set(
        cohort.loc[
            (cohort["in_primary_set"] == 1) & (cohort["label_certainty"] == "definite"),
            "patient_id",
        ]
    )
    definite = primary[primary["patient_id"].isin(definite_ids)].reset_index(drop=True)

    rows = [_summary("primary_full", primary), _summary("qc_sensitivity", qc),
            _summary("definite_labels", definite)]
    paired = paired_bootstrap_auroc_delta(primary, qc)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "estimand_summary.csv", index=False)
    primary.to_csv(OUT / "primary_patient_scores.csv", index=False)
    per_site = per_site_breakdown(primary, "score")
    pd.DataFrame([{"site": site, **metrics} for site, metrics in per_site.items()]).to_csv(
        OUT / "primary_per_site.csv", index=False
    )
    (OUT / "uncertainty.json").write_text(json.dumps({
        "bootstrap": "patient-stratified, 2000 draws, seed 42",
        "primary_vs_qc_paired_delta": paired,
        "estimands": rows,
    }, indent=2))

    print(pd.DataFrame(rows).to_string(index=False))
    print("primary minus QC AUROC:", paired)


if __name__ == "__main__":
    main()
