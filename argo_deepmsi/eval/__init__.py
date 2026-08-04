"""Shared evaluation primitives: cohort prep, metrics, plotting."""

from .cohort import build_cohort, load_clinical, load_qc_exclusion, patient_folds
from .metrics import (
    PATIENT_AGG,
    aggregate_to_patient,
    canonical_patient_site,
    evaluate_scorer,
    paired_bootstrap_auroc_delta,
    per_site_breakdown,
    stratified_patient_bootstrap,
)
from .plotting import apply_style, auroc_bar, per_site_grid, roc_overlay
from .validation import nested_grouped_oof

__all__ = [
    "build_cohort",
    "load_clinical",
    "load_qc_exclusion",
    "patient_folds",
    "PATIENT_AGG",
    "aggregate_to_patient",
    "canonical_patient_site",
    "evaluate_scorer",
    "paired_bootstrap_auroc_delta",
    "per_site_breakdown",
    "stratified_patient_bootstrap",
    "apply_style",
    "auroc_bar",
    "per_site_grid",
    "roc_overlay",
    "nested_grouped_oof",
]
