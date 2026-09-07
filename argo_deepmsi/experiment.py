"""TOML-driven, resumable ARGO experiment runner.

The runner deliberately composes the same public functions as the individual
CLI commands.  It adds ordering, exact parameter capture, and stage-level
restartability; it does not create a second implementation of the pipeline.
"""

from __future__ import annotations

import os
import re
import shutil
import tomllib
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from .experiment_schema import validate_experiment_config
from .reproducibility import environment_snapshot, file_sha256, utc_now, write_json

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DASK_ONLY_EXTRACT_OPTIONS = {
    "partition",
    "min_workers",
    "max_workers",
    "walltime",
    "cores",
    "memory",
    "conda_env",
}


def load_experiment(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_experiment_config(config)
    run = config.get("run", {})
    name = run.get("name")
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
        raise ValueError("[run].name must contain only letters, numbers, '.', '_' or '-'")
    return config


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _job_id(job: dict[str, Any], default: str) -> str:
    value = str(job.get("id", default))
    if not RUN_NAME.fullmatch(value):
        raise ValueError(f"Invalid job id: {value!r}")
    return value


def experiment_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the validated ordered plan without importing ML dependencies."""
    plan: list[dict[str, Any]] = []
    run_strategy = config.get("run", {}).get("strategy")
    if config.get("ingest", {}).get("enabled", False):
        plan.append({"id": "ingest", "kind": "ingest"})
    if config.get("pyramidal", {}).get("enabled", False):
        plan.append({"id": "pyramidal", "kind": "pyramidal"})
    extraction = config.get("extract", {})
    if extraction.get("enabled", False):
        models = extraction.get("models", [])
        if not models:
            raise ValueError("[extract].models cannot be empty when extraction is enabled")
        plan.append({"id": "extract", "kind": "extract", "models": models})
    for index, job in enumerate(config.get("aggregate", []), 1):
        if job.get("enabled", True):
            plan.append(
                {
                    "id": f"aggregate:{_job_id(job, str(index))}",
                    "kind": "aggregate",
                    "models": job.get("models", []),
                    "method": job.get("method", "mean"),
                }
            )
    if config.get("cohort", {}).get("enabled", False):
        plan.append({"id": "cohort", "kind": "cohort"})
    for index, job in enumerate(config.get("bag", []), 1):
        if job.get("enabled", True):
            models = job.get("models", [])
            if not models:
                raise ValueError(f"[[bag]] job {index} needs models")
            plan.append(
                {
                    "id": f"bag:{_job_id(job, str(index))}",
                    "kind": "bag",
                    "models": models,
                    "variants": job.get("variants"),
                }
            )
    for index, job in enumerate(config.get("train", []), 1):
        if job.get("enabled", True):
            embedding = job.get("embedding")
            if not embedding:
                raise ValueError(f"[[train]] job {index} needs embedding")
            plan.append(
                {
                    "id": f"train:{_job_id(job, Path(str(embedding)).name)}",
                    "kind": "train",
                    "embedding": embedding,
                    "classifiers": job.get("classifiers", ["logistic", "random_forest", "svm"]),
                    "strategy": job.get("strategy", run_strategy),
                }
            )
    for index, job in enumerate(config.get("scorer", []), 1):
        if job.get("enabled", True):
            name = job.get("name")
            if not name:
                raise ValueError(f"[[scorer]] job {index} needs name")
            plan.append(
                {
                    "id": f"scorer:{_job_id(job, str(name))}",
                    "kind": "scorer",
                    "name": name,
                    "parameters": job.get("parameters", {}),
                    "strategy": job.get("strategy", run_strategy),
                }
            )
    if config.get("comparison", {}).get("enabled", False):
        plan.append({"id": "comparison", "kind": "comparison"})
    if len({stage["id"] for stage in plan}) != len(plan):
        raise ValueError("Every enabled stage needs a unique id")
    return plan


def select_experiment_plan(
    plan: list[dict[str, Any]],
    *,
    from_stage: str | None = None,
    until_stage: str | None = None,
) -> list[dict[str, Any]]:
    """Select an inclusive stage range using stable manifest stage IDs."""
    ids = [stage["id"] for stage in plan]
    if from_stage is not None and from_stage not in ids:
        raise ValueError(f"Unknown --from-stage {from_stage!r}; choose one of {ids}")
    if until_stage is not None and until_stage not in ids:
        raise ValueError(f"Unknown --until-stage {until_stage!r}; choose one of {ids}")
    if not plan:
        return []
    start = ids.index(from_stage) if from_stage is not None else 0
    stop = ids.index(until_stage) + 1 if until_stage is not None else len(plan)
    if start >= stop:
        raise ValueError("--from-stage must not come after --until-stage")
    return plan[start:stop]


def _stage_jobs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map stable stage IDs to job tables without relying on iterator order."""
    result: dict[str, dict[str, Any]] = {}
    for kind in ("aggregate", "bag", "train", "scorer"):
        for index, job in enumerate(config.get(kind, []), 1):
            if not job.get("enabled", True):
                continue
            if kind == "train":
                default = Path(str(job["embedding"])).name
            elif kind == "scorer":
                default = str(job["name"])
            else:
                default = str(index)
            result[f"{kind}:{_job_id(job, default)}"] = job
    return result


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    previous_workspace = os.environ.get("ARGO_WORKSPACE")
    os.chdir(path)
    os.environ["ARGO_WORKSPACE"] = str(path)
    try:
        yield
    finally:
        if previous_workspace is None:
            os.environ.pop("ARGO_WORKSPACE", None)
        else:
            os.environ["ARGO_WORKSPACE"] = previous_workspace
        os.chdir(previous)


@contextmanager
def _run_policy(config: dict[str, Any]) -> Iterator[None]:
    """Apply process-local offline policy requested by the experiment budget."""
    budget = config.get("run", {}).get("budget", {})
    offline = not budget.get("allow_network", True) or not budget.get("allow_model_downloads", True)
    if not offline:
        yield
        return
    variables = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    previous = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_ingest(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Create source tables after protected slides have been acquired locally."""
    from .data_ingestion import process_redcap_data
    from dotenv import load_dotenv

    options = dict(config["ingest"])
    options.pop("enabled", None)
    expected_patients = options.pop("expected_patients", None)
    expected_slides = options.pop("expected_slides", None)
    api_url_env = str(options.pop("api_url_env", "REDCAP_API_URL"))
    api_token_env = str(options.pop("api_token_env", "REDCAP_API_TOKEN"))
    output_dir = _resolve(workspace, options.pop("output_dir", "results/data"))
    metadata_value = options.pop("metadata_dir", options.pop("halo_base_dir", "data"))
    metadata_dir = _resolve(workspace, metadata_value)
    if options:
        raise ValueError(f"Unsupported [ingest] options: {sorted(options)}")
    load_dotenv(workspace / ".env", override=False)
    api_url = os.environ.get(api_url_env)
    api_token = os.environ.get(api_token_env)
    if not api_url or not api_token:
        raise ValueError(
            f"Ingestion needs environment variables {api_url_env} and {api_token_env}"
        )
    clinical, slides = process_redcap_data(
        output_dir=output_dir,
        api_url=api_url,
        api_token=api_token,
        metadata_dir=metadata_dir,
    )
    if expected_patients is not None and len(clinical) != int(expected_patients):
        raise RuntimeError(
            f"Ingestion produced {len(clinical)} patients; expected {expected_patients}"
        )
    if expected_slides is not None and len(slides) != int(expected_slides):
        raise RuntimeError(f"Ingestion found {len(slides)} slides; expected {expected_slides}")
    return {
        "clinical_table": output_dir / "clinical_table.csv",
        "clinical_table_sha256": file_sha256(output_dir / "clinical_table.csv"),
        "slide_table": output_dir / "slide_table.csv",
        "slide_table_sha256": file_sha256(output_dir / "slide_table.csv"),
        "n_patients": len(clinical),
        "n_slides": len(slides),
    }


def _run_pyramidal(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    from .slide_prep import convert_non_pyramidal_slides, summarize

    run = config["run"]
    options = dict(config["pyramidal"])
    options.pop("enabled", None)
    allow_failures = bool(options.pop("allow_failures", False))
    slide_table = _resolve(workspace, options.pop("slide_table"))
    output = _resolve(workspace, options.pop("output", run["slide_table"]))
    results = convert_non_pyramidal_slides(
        slide_table=slide_table,
        output_table=output,
        **options,
    )
    counts = summarize(results)
    failures = counts.get("failed", 0) + counts.get("unreadable", 0)
    if failures and not allow_failures:
        raise RuntimeError(f"Pyramidal preparation failed for {failures} slides")
    return {"slide_table": output, "slide_table_sha256": file_sha256(output), "counts": counts}


def _run_extract(config: dict[str, Any], workspace: Path, run_dir: Path) -> dict[str, Any]:
    run = config["run"]
    options = dict(config["extract"])
    options.pop("enabled", None)
    engine = options.pop("engine", "local")
    allow_failures = bool(options.pop("allow_failures", False))
    slide_table_path = _resolve(workspace, options.pop("slide_table", run["slide_table"]))
    options.setdefault("device", run.get("device", "cuda"))
    if engine == "dask":
        from .dask_extraction import run_dask_extraction

        report = run_dask_extraction(
            slide_table=slide_table_path,
            workspace=workspace,
            output_dir=run_dir / "extraction",
            **options,
        )
        if report["counts"]["failed"] and not allow_failures:
            raise RuntimeError(
                f"Extraction failed for {report['counts']['failed']} slides; "
                f"see {run_dir / 'extraction' / 'extraction.json'}"
            )
        return {"slide_table": slide_table_path, **report["counts"]}
    if engine != "local":
        raise ValueError("[extract].engine must be 'local' or 'dask'")
    from .feature_extraction import extract_features_batch

    ignored_scheduler_options = sorted(DASK_ONLY_EXTRACT_OPTIONS & options.keys())
    for key in ignored_scheduler_options:
        options.pop(key)
    slide_table = pd.read_csv(slide_table_path)
    result = extract_features_batch(slide_table=slide_table, **options)
    failures = int((~result["success"]).sum())
    if failures and not allow_failures:
        raise RuntimeError(f"Extraction failed for {failures} slides")
    return {
        "slide_table": slide_table_path,
        "n_slides": len(result),
        "n_success": int(result["success"].sum()),
        "ignored_scheduler_options": ignored_scheduler_options,
    }


def _run_aggregate(config: dict[str, Any], job: dict[str, Any], workspace: Path) -> dict[str, Any]:
    from .feature_extraction import aggregate_features

    run = config["run"]
    options = dict(job)
    options.pop("id", None)
    options.pop("enabled", None)
    slide_table = _resolve(workspace, options.pop("slide_table", run["slide_table"]))
    if "output_dir" in options:
        options["output_dir"] = _resolve(workspace, options["output_dir"])
    options.setdefault("device", run.get("device", "cuda"))
    result = aggregate_features(slide_table=slide_table, **options)
    return {"slide_table": slide_table, "n_embeddings": {k: len(v) for k, v in result.items()}}


def _run_cohort(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    from .eval.cohort import build_feature_complete_cohort

    run = config["run"]
    options = dict(config["cohort"])
    options.pop("enabled", None)
    slide_table = _resolve(workspace, options.pop("slide_table", run["slide_table"]))
    clinical_table = _resolve(
        workspace, options.pop("clinical_table", run["clinical_table"])
    )
    embeddings_dir = _resolve(workspace, options.pop("embeddings_dir", "results/embeddings"))
    output = _resolve(workspace, options.pop("output", run["cohort"]))
    manifest_path = _resolve(
        workspace,
        options.pop("manifest", output.with_name("cohort_manifest.json")),
    )
    expected_patients = options.pop("expected_patients", None)
    expected_positive = options.pop("expected_positive_patients", None)
    cohort, cohort_manifest = build_feature_complete_cohort(
        slide_table,
        clinical_table,
        embeddings_dir,
        label_col=options.pop("label_column", "isMSIH"),
        positive=options.pop("positive_label", "MSI-H"),
    )
    if options:
        raise ValueError(f"Unsupported [cohort] options: {sorted(options)}")
    if expected_patients is not None and cohort_manifest["n_patients_primary"] != int(
        expected_patients
    ):
        raise RuntimeError(
            f"Primary cohort has {cohort_manifest['n_patients_primary']} patients; "
            f"expected {expected_patients}"
        )
    if expected_positive is not None and cohort_manifest["n_positive_patients"] != int(
        expected_positive
    ):
        raise RuntimeError(
            f"Primary cohort has {cohort_manifest['n_positive_patients']} positive patients; "
            f"expected {expected_positive}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    write_json(manifest_path, cohort_manifest)
    return {
        "cohort": output,
        "cohort_sha256": file_sha256(output),
        "manifest": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "n_slides": cohort_manifest["n_slides_primary"],
        "n_patients": cohort_manifest["n_patients_primary"],
        "n_positive_patients": cohort_manifest["n_positive_patients"],
    }


def _run_train(
    config: dict[str, Any], job: dict[str, Any], workspace: Path, run_dir: Path
) -> dict[str, Any]:
    from .training import compare_classifiers, load_training_data

    run = config["run"]
    job_id = _job_id(job, Path(str(job["embedding"])).name)
    embedding_value = Path(str(job["embedding"]))
    if embedding_value.is_absolute() or len(embedding_value.parts) > 1:
        embedding_dir = _resolve(workspace, embedding_value)
    else:
        embedding_dir = workspace / "results" / "embeddings" / embedding_value
    clinical = _resolve(workspace, job.get("clinical_table", run["clinical_table"]))
    X, y, frame = load_training_data(
        embedding_dir,
        clinical,
        label_column=job.get("label_column", "isMSIH"),
        positive_label=job.get("positive_label", "MSI-H"),
    )
    group_column = job.get("group_column", "patient_id")
    if group_column not in frame:
        raise ValueError(f"Training metadata has no group column {group_column!r}")
    results = compare_classifiers(
        X,
        y,
        groups=frame[group_column].to_numpy(),
        n_splits=int(job.get("n_splits", 5)),
        random_state=int(job.get("seed", run.get("seed", 42))),
        classifiers=job.get("classifiers", ["logistic", "random_forest", "svm"]),
        classifier_params=job.get("parameters", {}),
    )
    output = run_dir / "training" / job_id
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "classifier_comparison.csv", index=False)
    frame.to_csv(output / "training_data.csv", index=False)
    write_json(
        output / "run.json",
        {
            "embedding": str(embedding_dir),
            "clinical_table": str(clinical),
            "parameters": job,
            "n_samples": len(X),
            "n_features": X.shape[1],
        },
    )
    return {"output": output, "n_samples": len(X), "n_features": X.shape[1]}


def _run_bag(
    config: dict[str, Any], job: dict[str, Any], workspace: Path, run_dir: Path
) -> dict[str, Any]:
    from .bags import build_tile_bags

    run = config["run"]
    job_id = _job_id(job, "bags")
    tumor_dir = job.get("tumor_tiles_dir")
    return build_tile_bags(
        models=job["models"],
        variants=job.get("variants"),
        cohort_csv=_resolve(workspace, job.get("cohort", run["cohort"])),
        slide_table_csv=_resolve(workspace, job.get("slide_table", run["slide_table"])),
        tumor_tiles_dir=_resolve(workspace, tumor_dir) if tumor_dir else None,
        output_dir=run_dir / "bags" / job_id,
        max_tiles=job.get("max_tiles", 500),
        seed=int(job.get("seed", run.get("seed", 42))),
        slide_column=job.get("slide_column", "FILENAME"),
    )


def _expand_run_values(value: Any, workspace: Path, run_dir: Path) -> Any:
    if isinstance(value, str):
        return value.replace("$WORKSPACE", str(workspace)).replace("$RUN_DIR", str(run_dir))
    if isinstance(value, list):
        return [_expand_run_values(item, workspace, run_dir) for item in value]
    if isinstance(value, dict):
        return {key: _expand_run_values(item, workspace, run_dir) for key, item in value.items()}
    return value


def _run_scorer(
    config: dict[str, Any], job: dict[str, Any], workspace: Path, run_dir: Path
) -> dict[str, Any]:
    from .scorer_runner import run_scorer

    run = config["run"]
    job_id = _job_id(job, str(job["name"]))
    output = run_dir / "scorers" / job_id
    return run_scorer(
        str(job["name"]),
        cohort_csv=_resolve(workspace, job.get("cohort", run["cohort"])),
        slide_table_csv=_resolve(workspace, job.get("slide_table", run["slide_table"])),
        output_dir=output,
        parameters=_expand_run_values(job.get("parameters", {}), workspace, run_dir),
        use_cache=bool(job.get("use_cache", False)),
        publish=bool(job.get("publish", False)),
        workspace=workspace,
    )


def _run_comparison(run_dir: Path, scorer_stages: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for stage in scorer_stages:
        if stage.get("status") != "completed":
            continue
        metrics_path = (
            Path(stage["outputs"]["metrics"]) if stage["outputs"].get("metrics") else None
        )
        if metrics_path is None or not metrics_path.exists():
            continue
        import json

        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "run": stage["id"].split(":", 1)[1],
                "scorer": stage["outputs"].get("scorer", stage["id"].split(":", 1)[1]),
                "n_patients": metrics.get("n_patients"),
                "patient_auroc": metrics.get("patient_auroc"),
                "patient_auprc": metrics.get("patient_auprc"),
            }
        )
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values("patient_auroc", ascending=False)
    output = run_dir / "comparison.csv"
    comparison.to_csv(output, index=False)
    return {"output": output, "n_scorers": len(comparison)}


def run_experiment(
    config_path: Path,
    *,
    dry_run: bool = False,
    resume: bool = True,
    from_stage: str | None = None,
    until_stage: str | None = None,
    on_stage: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run a TOML experiment, atomically checkpointing after every stage."""
    config_path = config_path.resolve()
    config = load_experiment(config_path)
    full_plan = experiment_plan(config)
    plan = select_experiment_plan(
        full_plan,
        from_stage=from_stage,
        until_stage=until_stage,
    )
    run = config["run"]
    workspace = _resolve(config_path.parent, run.get("workspace", "."))
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    results_root = _resolve(workspace, run.get("results_dir", "results/runs"))
    run_dir = results_root / run["name"]
    if dry_run:
        return {
            "config": config_path,
            "workspace": workspace,
            "run_dir": run_dir,
            "plan": plan,
            "full_plan": full_plan,
        }

    manifest_path = run_dir / "manifest.json"
    config_digest = file_sha256(config_path)
    manifest: dict[str, Any]
    if resume and manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_sha256") != config_digest:
            raise ValueError(
                "The config changed since this run started. Choose a new [run].name "
                "instead of mixing configurations in one manifest."
            )
    else:
        if not resume and run_dir.exists():
            raise FileExistsError(
                f"Fresh run directory already exists: {run_dir}. Choose a new [run].name."
            )
        inputs = {}
        for key in ("slide_table", "clinical_table", "cohort"):
            value = run.get(key)
            if value is None:
                continue
            path = _resolve(workspace, value)
            inputs[key] = {
                "path": path,
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        manifest = {
            "name": run["name"],
            "strategy": run.get("strategy"),
            "budget": run.get("budget", {}),
            "status": "running",
            "started_at": utc_now(),
            "config": str(config_path),
            "config_sha256": config_digest,
            "workspace": str(workspace),
            "inputs": inputs,
            "environment": environment_snapshot(workspace),
            "stages": [],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, run_dir / "config.toml")
        write_json(manifest_path, manifest)

    completed = {stage["id"] for stage in manifest["stages"] if stage["status"] == "completed"}
    if from_stage is not None:
        start = next(i for i, item in enumerate(full_plan) if item["id"] == from_stage)
        unfinished = [item["id"] for item in full_plan[:start] if item["id"] not in completed]
        if unfinished:
            raise ValueError(
                f"Cannot start at {from_stage!r}; prerequisite stages are incomplete: {unfinished}"
            )
    jobs = _stage_jobs(config)
    manifest.pop("paused_at", None)
    manifest.pop("resume_from", None)
    manifest.update(status="running")
    write_json(manifest_path, manifest)

    with _working_directory(workspace), _run_policy(config):
        for item in plan:
            stage_id, kind = item["id"], item["kind"]
            job = jobs.get(stage_id)
            if stage_id in completed:
                if on_stage:
                    on_stage(stage_id, "skipped")
                continue
            stage = {
                "id": stage_id,
                "kind": kind,
                "strategy": item.get("strategy"),
                "status": "running",
                "started_at": utc_now(),
            }
            manifest["stages"].append(stage)
            write_json(manifest_path, manifest)
            if on_stage:
                on_stage(stage_id, "running")
            try:
                if kind == "ingest":
                    outputs = _run_ingest(config, workspace)
                elif kind == "pyramidal":
                    outputs = _run_pyramidal(config, workspace)
                elif kind == "extract":
                    outputs = _run_extract(config, workspace, run_dir)
                elif kind == "aggregate":
                    assert job is not None
                    outputs = _run_aggregate(config, job, workspace)
                elif kind == "cohort":
                    outputs = _run_cohort(config, workspace)
                elif kind == "train":
                    assert job is not None
                    outputs = _run_train(config, job, workspace, run_dir)
                elif kind == "bag":
                    assert job is not None
                    outputs = _run_bag(config, job, workspace, run_dir)
                elif kind == "scorer":
                    assert job is not None
                    outputs = _run_scorer(config, job, workspace, run_dir)
                    outputs["scorer"] = job["name"]
                else:
                    scorer_stages = [s for s in manifest["stages"] if s["kind"] == "scorer"]
                    outputs = _run_comparison(run_dir, scorer_stages)
                stage.update(status="completed", finished_at=utc_now(), outputs=outputs)
                completed.add(stage_id)
                write_json(manifest_path, manifest)
                if on_stage:
                    on_stage(stage_id, "completed")
            except Exception as error:
                stage.update(
                    status="failed",
                    finished_at=utc_now(),
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
                manifest.update(status="failed", finished_at=utc_now())
                write_json(manifest_path, manifest)
                if on_stage:
                    on_stage(stage_id, "failed")
                raise

    remaining = [item["id"] for item in full_plan if item["id"] not in completed]
    if remaining:
        manifest.update(status="paused", paused_at=utc_now(), resume_from=remaining[0])
        manifest.pop("finished_at", None)
    else:
        manifest.update(status="completed", finished_at=utc_now())
    write_json(manifest_path, manifest)
    return manifest
