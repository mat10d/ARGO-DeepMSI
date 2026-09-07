"""Machine-readable schema and semantic validation for experiment TOML files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

STRATEGIES = {
    "frozen_foundation_trained_head": (
        "Freeze foundation-model features and fit a Nigeria-cohort readout, "
        "including linear probes, MIL, and newly initialized transformers."
    ),
    "pretrained_end_to_end": (
        "Use a pretrained slide-to-prediction system without fitting on the evaluation cohort."
    ),
    "adapted_pretrained_head": (
        "Adapt a pretrained prediction head or lightweight adapters using only "
        "training folds from the target cohort."
    ),
    "backbone_finetune": (
        "Update some or all foundation-model backbone parameters inside the training folds."
    ),
    "mixed": "Compare explicitly labelled stages from more than one strategy family.",
}

TOP_LEVEL_KEYS = {
    "run",
    "ingest",
    "pyramidal",
    "extract",
    "aggregate",
    "cohort",
    "bag",
    "train",
    "scorer",
    "comparison",
}
SECTION_KEYS = {
    "run": {
        "name",
        "workspace",
        "results_dir",
        "slide_table",
        "clinical_table",
        "cohort",
        "seed",
        "device",
        "strategy",
        "budget",
    },
    "ingest": {
        "enabled",
        "output_dir",
        "metadata_dir",
        "halo_base_dir",
        "api_url_env",
        "api_token_env",
        "expected_patients",
        "expected_slides",
    },
    "pyramidal": {
        "enabled",
        "slide_table",
        "output",
        "slide_column",
        "tile_size",
        "quality",
        "allow_failures",
    },
    "extract": {
        "enabled",
        "engine",
        "models",
        "slide_table",
        "slide_column",
        "device",
        "allow_failures",
        "tile_px",
        "mpp",
        "amp",
        "batch_size",
        "num_workers",
        "tiling_policy",
        "overwrite",
        "max_slides",
        "partition",
        "min_workers",
        "max_workers",
        "walltime",
        "cores",
        "memory",
        "conda_env",
    },
    "aggregate": {
        "id",
        "enabled",
        "models",
        "method",
        "slide_table",
        "output_dir",
        "device",
        "write_h5ad",
    },
    "cohort": {
        "enabled",
        "slide_table",
        "clinical_table",
        "embeddings_dir",
        "output",
        "manifest",
        "label_column",
        "positive_label",
        "expected_patients",
        "expected_positive_patients",
    },
    "bag": {
        "id",
        "enabled",
        "models",
        "variants",
        "cohort",
        "slide_table",
        "tumor_tiles_dir",
        "max_tiles",
        "seed",
        "slide_column",
    },
    "train": {
        "id",
        "enabled",
        "strategy",
        "embedding",
        "clinical_table",
        "label_column",
        "positive_label",
        "group_column",
        "n_splits",
        "seed",
        "classifiers",
        "parameters",
    },
    "scorer": {
        "id",
        "enabled",
        "strategy",
        "name",
        "cohort",
        "slide_table",
        "parameters",
        "use_cache",
        "publish",
    },
    "comparison": {"enabled"},
}
BUDGET_KEYS = {
    "max_models",
    "max_stages",
    "max_trainable_stages",
    "max_epochs",
    "max_repeats",
    "allow_network",
    "allow_model_downloads",
}


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a table")
    return value


def _require_job_list(value: Any, location: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(job, Mapping) for job in value):
        raise ValueError(f"{location} must be an array of tables")
    return value


def _unknown_keys(table: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {location}: {sorted(unknown)}")


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _model_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{location} must be a non-empty array of model names")
    if len(set(value)) != len(value):
        raise ValueError(f"{location} contains duplicate model names")
    return value


def _strategy(value: Any, location: str) -> str:
    if not isinstance(value, str) or value not in STRATEGIES:
        raise ValueError(f"{location} must be one of {sorted(STRATEGIES)}")
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Reject ambiguous or misspelled experiment configuration before imports/run."""
    _unknown_keys(config, TOP_LEVEL_KEYS, "the document root")
    run = _require_mapping(config.get("run"), "[run]")
    _unknown_keys(run, SECTION_KEYS["run"], "[run]")
    if not isinstance(run.get("name"), str) or not run["name"]:
        raise ValueError("[run].name must be a non-empty string")
    run_strategy = _strategy(run["strategy"], "[run].strategy") if "strategy" in run else None

    budget = _require_mapping(run.get("budget", {}), "[run].budget")
    _unknown_keys(budget, BUDGET_KEYS, "[run].budget")
    for key in BUDGET_KEYS - {"allow_network", "allow_model_downloads"}:
        if key in budget:
            _positive_int(budget[key], f"[run].budget.{key}")
    for key in ("allow_network", "allow_model_downloads"):
        if key in budget and not isinstance(budget[key], bool):
            raise ValueError(f"[run].budget.{key} must be true or false")

    extract = _require_mapping(config.get("extract", {}), "[extract]")
    _unknown_keys(extract, SECTION_KEYS["extract"], "[extract]")
    if extract.get("enabled", False):
        models = _model_list(extract.get("models"), "[extract].models")
        if "slide_table" not in extract and "slide_table" not in run:
            raise ValueError("Enabled extraction needs [run].slide_table or [extract].slide_table")
        if extract.get("engine", "local") not in {"local", "dask"}:
            raise ValueError("[extract].engine must be 'local' or 'dask'")
        if extract.get("tiling_policy", "require-current") not in {"reuse", "require-current"}:
            raise ValueError("[extract].tiling_policy must be 'reuse' or 'require-current'")
        if "max_models" in budget and len(models) > budget["max_models"]:
            raise ValueError(
                f"Extraction requests {len(models)} models, exceeding "
                f"[run].budget.max_models={budget['max_models']}"
            )

    ingest = _require_mapping(config.get("ingest", {}), "[ingest]")
    _unknown_keys(ingest, SECTION_KEYS["ingest"], "[ingest]")
    if ingest.get("enabled", False):
        if "metadata_dir" in ingest and "halo_base_dir" in ingest:
            raise ValueError("[ingest] cannot set both metadata_dir and legacy halo_base_dir")
        for key in ("expected_patients", "expected_slides"):
            if key in ingest:
                _positive_int(ingest[key], f"[ingest].{key}")

    pyramidal = _require_mapping(config.get("pyramidal", {}), "[pyramidal]")
    _unknown_keys(pyramidal, SECTION_KEYS["pyramidal"], "[pyramidal]")
    if pyramidal.get("enabled", False):
        if "slide_table" not in pyramidal:
            raise ValueError("Enabled [pyramidal] needs slide_table")
        if "output" not in pyramidal and "slide_table" not in run:
            raise ValueError("Enabled [pyramidal] needs output or [run].slide_table")

    cohort = _require_mapping(config.get("cohort", {}), "[cohort]")
    _unknown_keys(cohort, SECTION_KEYS["cohort"], "[cohort]")
    if cohort.get("enabled", False):
        if "slide_table" not in cohort and "slide_table" not in run:
            raise ValueError("Enabled [cohort] needs [run].slide_table or [cohort].slide_table")
        if "clinical_table" not in cohort and "clinical_table" not in run:
            raise ValueError(
                "Enabled [cohort] needs [run].clinical_table or [cohort].clinical_table"
            )
        if "output" not in cohort and "cohort" not in run:
            raise ValueError("Enabled [cohort] needs output or [run].cohort")
        for key in ("expected_patients", "expected_positive_patients"):
            if key in cohort:
                _positive_int(cohort[key], f"[cohort].{key}")

    jobs_by_section: dict[str, Sequence[Mapping[str, Any]]] = {}
    for section in ("aggregate", "bag", "train", "scorer"):
        jobs = _require_job_list(config.get(section, []), f"[[{section}]]")
        jobs_by_section[section] = jobs
        for index, job in enumerate(jobs, 1):
            location = f"[[{section}]] job {index}"
            _unknown_keys(job, SECTION_KEYS[section], location)
            if not isinstance(job.get("enabled", True), bool):
                raise ValueError(f"{location}.enabled must be true or false")
            if not job.get("enabled", True):
                continue
            if section in {"aggregate", "bag"}:
                _model_list(job.get("models"), f"{location}.models")
            if (
                section in {"aggregate", "bag", "scorer"}
                and "slide_table" not in job
                and "slide_table" not in run
            ):
                raise ValueError(f"{location} needs slide_table here or in [run]")
            if section in {"bag", "scorer"} and "cohort" not in job and "cohort" not in run:
                raise ValueError(f"{location} needs cohort here or in [run]")
            if section == "train" and not isinstance(job.get("embedding"), str):
                raise ValueError(f"{location}.embedding must be a string")
            if section == "train" and "clinical_table" not in job and "clinical_table" not in run:
                raise ValueError(f"{location} needs clinical_table here or in [run]")
            if section == "train" and "classifiers" in job:
                classifiers = job["classifiers"]
                if (
                    not isinstance(classifiers, list)
                    or not classifiers
                    or any(not isinstance(item, str) for item in classifiers)
                ):
                    raise ValueError(f"{location}.classifiers must be a non-empty string array")
            if section == "scorer" and not isinstance(job.get("name"), str):
                raise ValueError(f"{location}.name must be a string")
            if section in {"train", "scorer"} and "strategy" in job:
                job_strategy = _strategy(job["strategy"], f"{location}.strategy")
                if run_strategy not in {None, "mixed", job_strategy}:
                    raise ValueError(
                        f"{location}.strategy={job_strategy!r} conflicts with "
                        f"[run].strategy={run_strategy!r}"
                    )

    comparison = _require_mapping(config.get("comparison", {}), "[comparison]")
    _unknown_keys(comparison, SECTION_KEYS["comparison"], "[comparison]")
    if "enabled" in comparison and not isinstance(comparison["enabled"], bool):
        raise ValueError("[comparison].enabled must be true or false")

    labelled_jobs = [
        (section, index, job)
        for section in ("train", "scorer")
        for index, job in enumerate(jobs_by_section[section], 1)
        if job.get("enabled", True)
    ]
    if run_strategy == "mixed":
        missing = [
            f"[[{section}]] job {index}"
            for section, index, job in labelled_jobs
            if "strategy" not in job
        ]
        if missing:
            raise ValueError(
                "Mixed experiments require a strategy on every enabled train/scorer job: "
                + ", ".join(missing)
            )

    n_stages = (
        int(bool(ingest.get("enabled", False)))
        + int(bool(pyramidal.get("enabled", False)))
        + int(bool(extract.get("enabled", False)))
        + sum(job.get("enabled", True) for jobs in jobs_by_section.values() for job in jobs)
        + int(bool(cohort.get("enabled", False)))
        + int(bool(comparison.get("enabled", False)))
    )
    if "max_stages" in budget and n_stages > budget["max_stages"]:
        raise ValueError(
            f"Experiment has {n_stages} stages, exceeding "
            f"[run].budget.max_stages={budget['max_stages']}"
        )
    n_trainable = len(labelled_jobs)
    if "max_trainable_stages" in budget and n_trainable > budget["max_trainable_stages"]:
        raise ValueError(
            f"Experiment has {n_trainable} train/scorer stages, exceeding "
            f"[run].budget.max_trainable_stages={budget['max_trainable_stages']}"
        )
    for section, index, job in labelled_jobs:
        parameters = job.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"[[{section}]] job {index}.parameters must be a table")
        for parameter, limit in (("epochs", "max_epochs"), ("repeats", "max_repeats")):
            if parameter in parameters:
                requested = _positive_int(
                    parameters[parameter], f"[[{section}]] job {index}.parameters.{parameter}"
                )
            else:
                continue
            if limit in budget and requested > budget[limit]:
                raise ValueError(
                    f"[[{section}]] job {index} requests {parameter}={requested}, "
                    f"exceeding [run].budget.{limit}={budget[limit]}"
                )


def experiment_json_schema() -> dict[str, Any]:
    """Return a compact JSON Schema for editors and autonomous tooling."""
    strategy = {"$ref": "#/$defs/strategy"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/mat10d/ARGO-DeepMSI/experiment.schema.json",
        "title": "ARGO-DeepMSI experiment",
        "type": "object",
        "additionalProperties": False,
        "required": ["run"],
        "properties": {
            "run": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$"},
                    "strategy": strategy,
                    "budget": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            key: (
                                {"type": "boolean"}
                                if key.startswith("allow_")
                                else {"type": "integer", "minimum": 1}
                            )
                            for key in sorted(BUDGET_KEYS)
                        },
                    },
                    **{
                        key: {"type": "string"}
                        for key in sorted(
                            SECTION_KEYS["run"] - {"name", "strategy", "budget", "seed"}
                        )
                    },
                    "seed": {"type": "integer"},
                },
            },
            **{
                section: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            key: (strategy if key == "strategy" else {})
                            for key in sorted(SECTION_KEYS[section])
                        },
                    },
                }
                for section in ("aggregate", "bag", "train", "scorer")
            },
            **{
                section: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {key: {} for key in sorted(SECTION_KEYS[section])},
                }
                for section in ("ingest", "pyramidal", "extract", "cohort")
            },
            "comparison": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"enabled": {"type": "boolean"}},
            },
        },
        "$defs": {
            "strategy": {
                "type": "string",
                "enum": sorted(STRATEGIES),
            }
        },
        "x-strategy-descriptions": STRATEGIES,
    }
