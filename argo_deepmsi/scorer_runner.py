"""Configurable execution boundary for all registered MSI scorers."""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .eval.metrics import evaluate_scorer, per_site_breakdown, stratified_patient_bootstrap
from .reproducibility import environment_snapshot, utc_now, write_json
from .scorers import get_scorer

if TYPE_CHECKING:
    from .scorers.base import Scorer

RESERVED_PARAMETERS = {"clean_slide_ids", "write_outputs"}


def parse_parameters(assignments: Iterable[str]) -> dict[str, Any]:
    """Parse repeatable ``KEY=JSON`` command-line parameters.

    JSON covers numbers, booleans, null, lists, and dictionaries. Unquoted
    strings remain convenient for values such as ``embedding=phaet_mean``.
    """
    parsed: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected KEY=VALUE, got {assignment!r}")
        key, raw_value = assignment.split("=", 1)
        key = key.strip()
        if not key or key in RESERVED_PARAMETERS:
            raise ValueError(f"Parameter name {key!r} is empty or reserved")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parsed[key] = value
    return parsed


def scorer_contract(name: str) -> dict[str, Any]:
    scorer = get_scorer(name)
    signature = inspect.signature(scorer.compute_batch)
    parameters = {}
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "slide_table", "clean_slide_ids", "write_outputs"}:
            continue
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        parameters[parameter.name] = (
            None if parameter.default is inspect.Parameter.empty else parameter.default
        )
    return {
        "name": scorer.name,
        "description": scorer.description,
        "resolution": scorer.resolution,
        "needs_training_on_our_data": scorer.needs_training_on_our_data,
        "patient_aggregation": scorer.patient_aggregation,
        "primary_score": scorer.primary_score,
        "score_columns": [column.name for column in scorer.score_columns],
        "parameters": parameters,
        "cache": str(scorer.score_path) if scorer.score_path is not None else None,
    }


def _validate_parameters(scorer: Scorer, parameters: dict[str, Any]) -> None:
    signature = inspect.signature(scorer.compute_batch)
    accepted = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in {"self", "slide_table", *RESERVED_PARAMETERS}
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }
    unknown = set(parameters) - accepted
    if unknown:
        raise ValueError(
            f"Unknown parameters for {scorer.name}: {sorted(unknown)}. "
            f"Use `argo scorers show {scorer.name}` to list tunables."
        )


def load_cohort(cohort_csv: Path) -> tuple[pd.DataFrame, set[str]]:
    cohort = pd.read_csv(cohort_csv)
    required = {"slide_id", "patient_id"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"{cohort_csv} is missing columns: {sorted(missing)}")
    inclusion = "in_primary_set" if "in_primary_set" in cohort else "in_clean_set"
    if inclusion not in cohort:
        raise ValueError(f"{cohort_csv} needs in_primary_set or in_clean_set")
    primary = cohort[cohort[inclusion] == 1].copy()
    primary["slide_id"] = primary["slide_id"].astype(str)
    primary["patient_id"] = primary["patient_id"].astype(str)
    return primary, set(primary["slide_id"].astype(str))


def _add_cohort_columns(scores: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Fill identifiers/labels from the canonical cohort without duplicating columns."""
    scores = scores.copy()
    join_key = "slide_id" if "slide_id" in scores else "patient_id"
    if join_key not in scores:
        raise ValueError("Scorer output needs slide_id or patient_id")
    eligible = (
        ("patient_id", "site", "patient_cohort", "y")
        if join_key == "slide_id"
        else ("site", "patient_cohort", "y")
    )
    available = [column for column in eligible if column in cohort and column not in scores]
    if not available:
        return scores
    reference = cohort[[join_key, *available]].drop_duplicates(join_key)
    return scores.merge(reference, on=join_key, how="left", validate="many_to_one")


def _filter_primary(scores: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Enforce the configured primary estimand even if a scorer/cache is broader."""
    if "slide_id" in scores:
        scores = scores.copy()
        scores["slide_id"] = scores["slide_id"].astype(str)
        allowed = set(cohort["slide_id"].astype(str))
        return scores[scores["slide_id"].astype(str).isin(allowed)].copy()
    if "patient_id" in scores:
        scores = scores.copy()
        scores["patient_id"] = scores["patient_id"].astype(str)
        allowed = set(cohort["patient_id"].astype(str))
        return scores[scores["patient_id"].astype(str).isin(allowed)].copy()
    raise ValueError("Scorer output needs slide_id or patient_id")


def _evaluate_patient_scores(scores: pd.DataFrame, score_col: str) -> dict[str, Any]:
    frame = scores.dropna(subset=[score_col, "y"]).copy()
    frame = frame.rename(columns={score_col: "score"})
    if "patient_cohort" in frame:
        frame["site"] = frame["patient_cohort"].fillna(frame.get("site"))
    has_classes = frame["y"].nunique() == 2
    ci = stratified_patient_bootstrap(frame)
    return {
        "n_patients": int(len(frame)),
        "prevalence_patient": float(frame["y"].mean()) if len(frame) else float("nan"),
        "patient_auroc": (
            float(roc_auc_score(frame["y"], frame["score"])) if has_classes else float("nan")
        ),
        "patient_auprc": (
            float(average_precision_score(frame["y"], frame["score"]))
            if has_classes
            else float("nan")
        ),
        "patient_auroc_ci95": [ci["ci_low"], ci["ci_high"]],
        "per_site_patient": per_site_breakdown(frame, "score"),
    }


def run_scorer(
    name: str,
    *,
    cohort_csv: Path,
    slide_table_csv: Path,
    output_dir: Path,
    parameters: dict[str, Any] | None = None,
    use_cache: bool = False,
    publish: bool = False,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Execute one scorer and persist scores, metrics, and exact parameters."""
    parameters = dict(parameters or {})
    conflict = RESERVED_PARAMETERS & parameters.keys()
    if conflict:
        raise ValueError(f"Reserved scorer parameters: {sorted(conflict)}")
    if use_cache and parameters:
        raise ValueError("Cached scorer runs cannot claim new parameters; use recompute instead")

    workspace = (workspace or Path.cwd()).resolve()
    scorer = get_scorer(name)
    _validate_parameters(scorer, parameters)
    cohort, clean_slide_ids = load_cohort(cohort_csv)
    slide_table = pd.read_csv(slide_table_csv)

    if use_cache:
        if scorer.score_path is None or not scorer.score_path.exists():
            raise FileNotFoundError(f"No canonical cache for {name}: {scorer.score_path}")
        scores = pd.read_csv(scorer.score_path)
        source = "cache"
    else:
        compute_parameters = inspect.signature(scorer.compute_batch).parameters
        if "retrain" in compute_parameters and "retrain" not in parameters:
            parameters["retrain"] = True
        scores = scorer.compute_batch(
            slide_table,
            clean_slide_ids=clean_slide_ids,
            write_outputs=False,
            **parameters,
        )
        source = "computed"

    scores = _filter_primary(scores, cohort)
    scores = _add_cohort_columns(scores, cohort)
    expected_columns = {column.name for column in scorer.score_columns}
    missing = expected_columns - set(scores.columns)
    if missing:
        raise ValueError(f"{name} did not produce declared score columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "patient_scores.csv" if scorer.resolution == "patient" else "slide_scores.csv"
    scores_path = output_dir / filename
    scores.to_csv(scores_path, index=False)
    scorer.write_run_artifacts(output_dir)

    required = {"patient_id", "site", "y", scorer.primary_score}
    metrics = None
    if required <= set(scores):
        if scorer.resolution == "patient":
            metrics = _evaluate_patient_scores(scores, scorer.primary_score)
        else:
            metrics = evaluate_scorer(
                scores,
                scorer.primary_score,
                agg=scorer.patient_aggregation,
            )
        write_json(output_dir / "metrics.json", metrics)

    run_metadata = {
        **scorer_contract(name),
        "created_at": utc_now(),
        "source": source,
        "parameters": parameters,
        "cohort": str(cohort_csv.resolve()),
        "slide_table": str(slide_table_csv.resolve()),
        "n_rows": len(scores),
        "environment": environment_snapshot(workspace),
    }
    write_json(output_dir / "run.json", run_metadata)

    if publish:
        if scorer.score_path is None:
            raise ValueError(f"{name} has no canonical score path")
        scorer.score_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scores_path, scorer.score_path)
        scorer.write_metadata(scorer.score_path.parent)
        scorer.write_run_artifacts(scorer.score_path.parent)
        write_json(scorer.score_path.parent / "run.json", run_metadata)

    return {
        "scores": scores_path,
        "metrics": output_dir / "metrics.json" if metrics is not None else None,
        "metadata": output_dir / "run.json",
        "n_rows": len(scores),
    }
