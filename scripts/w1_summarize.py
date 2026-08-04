"""Build the W1 frozen-reference and cached-candidate comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from argo_deepmsi.eval.metrics import (
    aggregate_to_patient,
    evaluate_scorer,
    paired_bootstrap_auroc_delta,
    stratified_patient_bootstrap,
)
from argo_deepmsi.eval.screening import screening_block
from argo_deepmsi.models.w1_heads import CACHED_CANDIDATES
from argo_deepmsi.w1_peft_training import PEFT_CANDIDATES

W1_DIR = Path("results/experiments/w1_ctranspath")
REFERENCE = Path("results/scorers/wagner_zeroshot/slide_scores.csv")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate dependency."""
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    rows = [
        "| " + " | ".join(map(str, formatted.columns)) + " |",
        "| " + " | ".join("---" for _ in formatted.columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _reference_row(slide_ids: set[str]) -> dict:
    slides = pd.read_csv(REFERENCE)
    slides = slides[slides["slide_id"].astype(str).isin(slide_ids)].copy()
    metrics = evaluate_scorer(slides, "p_msih", agg="max_sqrtn")
    patient = aggregate_to_patient(slides, "p_msih", agg="max_sqrtn")
    interval = stratified_patient_bootstrap(patient)
    screening = screening_block(patient["y"], patient["score"])
    capped_path = W1_DIR / "candidates" / "W1-0c" / "patient_scores.csv"
    capped_delta = (
        paired_bootstrap_auroc_delta(patient, pd.read_csv(capped_path))
        if capped_path.exists()
        else {}
    )
    return {
        "candidate_id": "W1-0",
        "description": "Frozen external Wagner/CTransPath reference",
        "status": "complete",
        "n_patients": len(patient),
        "patient_auroc": metrics["patient_auroc"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "delta_vs_w1_0": 0.0,
        "delta_ci_low": 0.0,
        "delta_ci_high": 0.0,
        "probability_delta_le_zero": np.nan,
        "delta_vs_w1_0c": capped_delta.get("estimate", np.nan),
        "delta_vs_w1_0c_ci_low": capped_delta.get("ci_low", np.nan),
        "delta_vs_w1_0c_ci_high": capped_delta.get("ci_high", np.nan),
        "spec_at_sens95": screening["spec_at_sens95"],
        "spec_at_sens96": screening["spec_at_sens96"],
        "oauthc_auroc": metrics["per_site_patient"].get("OAUTHC", {}).get(
            "auroc", np.nan
        ),
        "trainable_parameters": 0,
        "outer_gpu_hours": 0.0,
        "developmental": False,
    }


def _candidate_row(candidate_id: str) -> dict:
    directory = W1_DIR / "candidates" / candidate_id
    metrics_path = directory / "metrics.json"
    metadata_path = directory / "metadata.json"
    audit_path = directory / "fold_audit.json"
    base = {
        "candidate_id": candidate_id,
        "description": (
            CACHED_CANDIDATES[candidate_id].description
            if candidate_id in CACHED_CANDIDATES
            else PEFT_CANDIDATES[candidate_id].description
        ),
        "status": "missing",
        "developmental": True,
    }
    if not metrics_path.exists():
        if directory.exists():
            base["status"] = "running_or_incomplete"
        return base
    metrics = json.loads(metrics_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    audit = json.loads(audit_path.read_text())
    interval = metrics["patient_bootstrap"]
    delta = metrics["paired_delta_vs_frozen_wagner"]
    screening = metrics["screening"]
    capped_path = W1_DIR / "candidates" / "W1-0c" / "patient_scores.csv"
    capped_delta = (
        paired_bootstrap_auroc_delta(
            pd.read_csv(directory / "patient_scores.csv"), pd.read_csv(capped_path)
        )
        if capped_path.exists()
        else {}
    )
    base.update(
        {
            "status": "complete",
            "n_patients": metrics["n_patients"],
            "patient_auroc": metrics["patient_auroc"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "delta_vs_w1_0": delta["estimate"],
            "delta_ci_low": delta["ci_low"],
            "delta_ci_high": delta["ci_high"],
            "probability_delta_le_zero": delta["probability_le_zero"],
            "delta_vs_w1_0c": capped_delta.get("estimate", np.nan),
            "delta_vs_w1_0c_ci_low": capped_delta.get("ci_low", np.nan),
            "delta_vs_w1_0c_ci_high": capped_delta.get("ci_high", np.nan),
            "spec_at_sens95": screening["spec_at_sens95"],
            "spec_at_sens96": screening["spec_at_sens96"],
            "oauthc_auroc": metrics["per_site_patient"].get("OAUTHC", {}).get(
                "auroc", np.nan
            ),
            "trainable_parameters": metadata["parameter_count"]["trainable"],
            "outer_gpu_hours": sum(row["elapsed_seconds"] for row in audit) / 3600,
        }
    )
    return base


def _capped_reference_row() -> dict:
    directory = W1_DIR / "candidates" / "W1-0c"
    metrics_path = directory / "metrics.json"
    base = {
        "candidate_id": "W1-0c",
        "description": "Frozen Wagner on deterministic 1024-tile reservoirs",
        "status": "missing",
        "developmental": False,
    }
    if not metrics_path.exists():
        return base
    metrics = json.loads(metrics_path.read_text())
    interval = metrics["patient_bootstrap"]
    delta = metrics["paired_delta_vs_frozen_wagner"]
    screening = metrics["screening"]
    base.update(
        {
            "status": "complete",
            "n_patients": metrics["n_patients"],
            "patient_auroc": metrics["patient_auroc"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "delta_vs_w1_0": delta["estimate"],
            "delta_ci_low": delta["ci_low"],
            "delta_ci_high": delta["ci_high"],
            "probability_delta_le_zero": delta["probability_le_zero"],
            "delta_vs_w1_0c": 0.0,
            "delta_vs_w1_0c_ci_low": 0.0,
            "delta_vs_w1_0c_ci_high": 0.0,
            "spec_at_sens95": screening["spec_at_sens95"],
            "spec_at_sens96": screening["spec_at_sens96"],
            "oauthc_auroc": metrics["per_site_patient"].get("OAUTHC", {}).get(
                "auroc", np.nan
            ),
            "trainable_parameters": 0,
            "outer_gpu_hours": np.nan,
        }
    )
    return base


def main() -> None:
    index = pd.read_csv(W1_DIR / "slide_index.csv")
    slide_ids = set(index["slide_id"].astype(str))
    rows = [_reference_row(slide_ids), _capped_reference_row()]
    rows.extend(_candidate_row(candidate_id) for candidate_id in CACHED_CANDIDATES)
    rows.append(
        {
            "candidate_id": "W1-8",
            "description": "Last-stage CTransPath LoRA",
            "status": "blocked_current_torchscript",
            "developmental": True,
            "note": (
                "Current LazySlide CTransPath weights are a RecursiveScriptModule, "
                "which cannot accept module replacement or forward hooks. Rehydrate "
                "the checkpoint into an eager architecture after transition."
            ),
        }
    )
    rows.extend(_candidate_row(candidate_id) for candidate_id in PEFT_CANDIDATES)
    table = pd.DataFrame(rows)

    complete = table[table["status"] == "complete"].copy()
    reference = complete[complete["candidate_id"] == "W1-0"].iloc[0]
    complete["passes_oauthc_gate"] = (
        complete["oauthc_auroc"] >= float(reference["oauthc_auroc"]) - 0.03
    )
    complete["passes_high_sensitivity_gate"] = (
        complete["spec_at_sens95"] >= float(reference["spec_at_sens95"])
    )
    complete["passes_overall_gate"] = (
        complete["patient_auroc"] >= float(reference["patient_auroc"])
    )
    complete["eligible_for_lock_review"] = (
        complete["passes_oauthc_gate"]
        & complete["passes_high_sensitivity_gate"]
        & complete["passes_overall_gate"]
    )
    gate_columns = [
        "passes_oauthc_gate",
        "passes_high_sensitivity_gate",
        "passes_overall_gate",
        "eligible_for_lock_review",
    ]
    gates = complete.set_index("candidate_id")[gate_columns]
    for column in gate_columns:
        table[column] = table["candidate_id"].map(gates[column])
    table.to_csv(W1_DIR / "comparison.csv", index=False)
    display_columns = [
        "candidate_id",
        "status",
        "patient_auroc",
        "ci_low",
        "ci_high",
        "delta_vs_w1_0",
        "delta_vs_w1_0c",
        "spec_at_sens95",
        "spec_at_sens96",
        "oauthc_auroc",
        "trainable_parameters",
        "outer_gpu_hours",
        "eligible_for_lock_review",
    ]
    markdown = [
        "# W1 CTransPath development comparison",
        "",
        "All learned rows are developmental. Selection among them is performed on the current",
        "cohort and requires confirmation on sealed temporal patients.",
        "",
        _markdown_table(complete[display_columns]),
        "",
    ]
    (W1_DIR / "comparison.md").write_text("\n".join(markdown))
    print(complete[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
