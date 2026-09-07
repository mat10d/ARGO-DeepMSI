"""Shared evaluation primitives, loaded only when requested."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_cohort": "cohort",
    "load_clinical": "cohort",
    "load_qc_exclusion": "cohort",
    "patient_folds": "cohort",
    "PATIENT_AGG": "metrics",
    "aggregate_to_patient": "metrics",
    "canonical_patient_site": "metrics",
    "evaluate_scorer": "metrics",
    "paired_bootstrap_auroc_delta": "metrics",
    "per_site_breakdown": "metrics",
    "stratified_patient_bootstrap": "metrics",
    "apply_style": "plotting",
    "auroc_bar": "plotting",
    "per_site_grid": "plotting",
    "roc_overlay": "plotting",
    "nested_grouped_oof": "validation",
}

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


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
