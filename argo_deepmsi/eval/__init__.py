"""Shared evaluation primitives: cohort prep, metrics, plotting."""

from .cohort import build_cohort, load_clinical, load_qc_exclusion, patient_folds
from .metrics import (
    PATIENT_AGG,
    aggregate_to_patient,
    evaluate_scorer,
    per_site_breakdown,
)
from .plotting import apply_style, auroc_bar, per_site_grid, roc_overlay

__all__ = [
    "build_cohort",
    "load_clinical",
    "load_qc_exclusion",
    "patient_folds",
    "PATIENT_AGG",
    "aggregate_to_patient",
    "evaluate_scorer",
    "per_site_breakdown",
    "apply_style",
    "auroc_bar",
    "per_site_grid",
    "roc_overlay",
]
