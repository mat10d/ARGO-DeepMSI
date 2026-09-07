from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argo_deepmsi.bags import build_tile_bags
from argo_deepmsi.dask_extraction import process_slide
from argo_deepmsi.experiment import _run_extract, experiment_plan, run_experiment
from argo_deepmsi.reproducibility import write_json
from argo_deepmsi.scorer_runner import parse_parameters, run_scorer
from argo_deepmsi.scorers.base import Scorer, ScoreColumn
from argo_deepmsi.scorers.bag_transformer import BagTransformer
from argo_deepmsi.scorers.clam_tilemil import _load_bags
from argo_deepmsi.scorers.registry import register


def test_parse_parameters_uses_json_and_plain_strings():
    assert parse_parameters(
        ["repeats=5", 'embeddings=["phaet_mean","mascaret_mean"]', "device=cuda"]
    ) == {
        "repeats": 5,
        "embeddings": ["phaet_mean", "mascaret_mean"],
        "device": "cuda",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_parameters(["broken"])
    with pytest.raises(ValueError, match="reserved"):
        parse_parameters(["clean_slide_ids=[]"])


def test_json_provenance_is_strict_and_deterministic(tmp_path: Path):
    output = tmp_path / "metadata.json"
    write_json(output, {"nonfinite": float("nan"), "unordered": {"b", "a"}})
    assert json.loads(output.read_text()) == {"nonfinite": None, "unordered": ["a", "b"]}


def test_experiment_dry_run_and_resume(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("slides.csv", "clinical.csv", "cohort.csv"):
        (workspace / name).write_text("id\n1\n")
    config = tmp_path / "experiment.toml"
    config.write_text("""
[run]
name = "smoke"
workspace = "workspace"
slide_table = "slides.csv"
clinical_table = "clinical.csv"
cohort = "cohort.csv"

[comparison]
enabled = true
""")
    dry = run_experiment(config, dry_run=True)
    assert [stage["id"] for stage in dry["plan"]] == ["comparison"]
    assert not dry["run_dir"].exists()

    first = run_experiment(config)
    assert first["status"] == "completed"
    assert first["stages"][0]["status"] == "completed"
    assert len(first["inputs"]["cohort"]["sha256"]) == 64
    resumed = run_experiment(config)
    assert len(resumed["stages"]) == 1
    assert (workspace / "results/runs/smoke/config.toml").exists()
    with pytest.raises(FileExistsError, match="Fresh run directory"):
        run_experiment(config, resume=False)

    config.write_text(config.read_text().replace("enabled = true", "enabled = false"))
    with pytest.raises(ValueError, match="config changed"):
        run_experiment(config)


def test_experiment_can_checkpoint_at_embeddings_and_resume_post_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pd.DataFrame({"FILENAME": ["slide.svs"]}).to_csv(workspace / "slides.csv", index=False)
    (workspace / "clinical.csv").write_text("PATIENT,isMSIH\np1,MSI-H\n")
    (workspace / "cohort.csv").write_text(
        "slide_id,patient_id,site,y,in_primary_set\ns1,p1,a,1,1\n"
    )
    config = tmp_path / "checkpoint.toml"
    config.write_text("""
[run]
name = "checkpoint"
workspace = "workspace"
slide_table = "slides.csv"
clinical_table = "clinical.csv"
cohort = "cohort.csv"
device = "cpu"

[extract]
enabled = true
models = ["synthetic"]
device = "cpu"

[[aggregate]]
id = "mean"
models = ["synthetic"]
method = "mean"

[comparison]
enabled = true
""")
    monkeypatch.setattr(
        "argo_deepmsi.experiment._run_extract", lambda *_args: {"n_success": 1}
    )
    monkeypatch.setattr(
        "argo_deepmsi.experiment._run_aggregate", lambda *_args: {"n_embeddings": 1}
    )

    embedded = run_experiment(config, until_stage="extract")
    assert embedded["status"] == "paused"
    assert embedded["resume_from"] == "aggregate:mean"
    assert [stage["id"] for stage in embedded["stages"]] == ["extract"]

    resumed = run_experiment(config, from_stage="aggregate:mean")
    assert resumed["status"] == "completed"
    assert [stage["id"] for stage in resumed["stages"]] == [
        "extract",
        "aggregate:mean",
        "comparison",
    ]


def test_experiment_rejects_skipping_unfinished_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pd.DataFrame({"FILENAME": ["slide.svs"]}).to_csv(workspace / "slides.csv", index=False)
    config = tmp_path / "skip.toml"
    config.write_text("""
[run]
name = "skip"
workspace = "workspace"
slide_table = "slides.csv"

[extract]
enabled = true
models = ["synthetic"]
device = "cpu"

[[aggregate]]
id = "mean"
models = ["synthetic"]
method = "mean"
""")
    monkeypatch.setattr(
        "argo_deepmsi.experiment._run_extract", lambda *_args: {"n_success": 1}
    )
    with pytest.raises(ValueError, match="prerequisite stages are incomplete"):
        run_experiment(config, from_stage="aggregate:mean")


def test_plan_covers_fresh_machine_preprocessing_before_modeling():
    plan = experiment_plan(
        {
            "run": {},
            "ingest": {"enabled": True},
            "pyramidal": {"enabled": True},
            "extract": {"enabled": True, "models": ["a"]},
            "aggregate": [{"id": "mean", "models": ["a"]}],
            "cohort": {"enabled": True},
            "bag": [{"id": "bags", "models": ["a"]}],
        }
    )
    assert [stage["id"] for stage in plan] == [
        "ingest",
        "pyramidal",
        "extract",
        "aggregate:mean",
        "cohort",
        "bag:bags",
    ]


def test_local_extraction_ignores_dask_scheduler_options(tmp_path: Path, monkeypatch):
    slide_table = tmp_path / "slides.csv"
    pd.DataFrame({"FILENAME": ["slide.svs"]}).to_csv(slide_table, index=False)
    observed = {}

    def fake_extract(*, slide_table, **options):
        observed.update(options)
        return pd.DataFrame({"success": [True] * len(slide_table)})

    monkeypatch.setattr("argo_deepmsi.feature_extraction.extract_features_batch", fake_extract)
    outcome = _run_extract(
        {
            "run": {"slide_table": str(slide_table), "device": "cuda"},
            "extract": {
                "enabled": True,
                "engine": "local",
                "models": ["uni2"],
                "partition": "gpu",
                "min_workers": 1,
                "max_workers": 8,
                "memory": "256 GB",
            },
        },
        tmp_path,
        tmp_path / "run",
    )

    assert observed == {"models": ["uni2"], "device": "cuda"}
    assert outcome["ignored_scheduler_options"] == [
        "max_workers",
        "memory",
        "min_workers",
        "partition",
    ]


def test_generic_bag_builder_supports_concatenated_encoders(tmp_path: Path, monkeypatch):
    cohort = pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "patient_id": ["p1", "p2"],
            "site": ["UITH", "OAUTHC"],
            "y": [0, 1],
            "in_primary_set": [1, 1],
        }
    )
    cohort_path = tmp_path / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    slide_table = pd.DataFrame({"FILENAME": [tmp_path / "s1.svs", tmp_path / "s2.svs"]})
    slide_path = tmp_path / "slides.csv"
    slide_table.to_csv(slide_path, index=False)
    arrays = {
        ("s1", "a"): np.arange(15, dtype=np.float32).reshape(5, 3),
        ("s1", "b"): np.arange(10, dtype=np.float32).reshape(5, 2),
        ("s2", "a"): np.ones((4, 3), dtype=np.float32),
        ("s2", "b"): np.ones((4, 2), dtype=np.float32),
    }

    def fake_read(zarr_path: Path, model: str) -> np.ndarray:
        return arrays[(zarr_path.stem, model)]

    monkeypatch.setattr("argo_deepmsi.bags._read_tile_features", fake_read)
    output = tmp_path / "bags"
    summary = build_tile_bags(
        models=["a", "b"],
        variants={"a": ["a"], "both": ["a", "b"]},
        cohort_csv=cohort_path,
        slide_table_csv=slide_path,
        output_dir=output,
        max_tiles=3,
        seed=7,
    )
    assert summary["a"]["dimension"] == 3
    assert summary["both"]["dimension"] == 5
    archive = np.load(output / "both_bags.npz")
    assert archive["lengths"].tolist() == [3, 3]
    assert archive["concat"].shape == (6, 5)
    assert pd.read_csv(output / "both_index.csv")["y"].tolist() == [0, 1]
    loaded = _load_bags(
        {"s2"}, bag_file=output / "both_bags.npz", index_file=output / "both_index.csv"
    )
    assert loaded is not None
    bags, index = loaded
    assert len(bags) == 1 and bags[0].shape == (3, 5)
    assert index["slide_id"].tolist() == ["s2"]


def test_dask_worker_checks_tiling_policy_even_when_features_exist(tmp_path: Path, monkeypatch):
    slide = tmp_path / "s1.svs"
    slide.touch()
    (tmp_path / "s1.zarr" / "tables" / "a_tiles").mkdir(parents=True)
    calls = []

    def reject_legacy(*args, **kwargs):
        calls.append(kwargs["tiling_policy"])
        return None

    monkeypatch.setattr(
        "argo_deepmsi.feature_extraction.extract_features_single_slide", reject_legacy
    )
    with pytest.raises(RuntimeError, match="extraction failed"):
        process_slide(
            str(slide),
            ["a"],
            tile_px=256,
            mpp=0.5,
            amp=False,
            device="cpu",
            num_workers=0,
            batch_size=1,
            overwrite=False,
            tiling_policy="require-current",
        )
    assert calls == ["require-current"]


def test_bag_transformer_runs_small_nested_grouped_cv(tmp_path: Path):
    rng = np.random.default_rng(3)
    labels = np.tile([0, 1], 6)
    bags = [rng.normal(loc=label, size=(3, 4)).astype(np.float32) for label in labels]
    concatenated = np.concatenate(bags)
    bag_file = tmp_path / "tiny_bags.npz"
    np.savez(bag_file, concat=concatenated, lengths=np.repeat(3, len(bags)))
    index = pd.DataFrame(
        {
            "slide_id": [f"s{i}" for i in range(len(bags))],
            "patient_id": [f"p{i}" for i in range(len(bags))],
            "site": np.tile(["a", "b"], 6),
            "y": labels,
        }
    )
    index_file = tmp_path / "tiny_index.csv"
    index.to_csv(index_file, index=False)

    scorer = BagTransformer()
    scores = scorer.compute_batch(
        pd.DataFrame(),
        clean_slide_ids=set(index["slide_id"]),
        bag_file=bag_file,
        index_file=index_file,
        epochs=1,
        device="cpu",
        outer_splits=2,
        inner_splits=2,
        dim=8,
        depth=1,
        heads=2,
        mlp_dim=8,
        dim_head=4,
        capture_attention=True,
        write_outputs=False,
    )
    assert len(scores) == len(index)
    assert scores["p_msih"].between(0, 1).all()
    scorer.write_run_artifacts(tmp_path / "artifacts")
    attention = np.load(tmp_path / "artifacts/attention.npz")
    assert attention["lengths"].tolist() == [3] * len(index)
    assert len(BagTransformer().score_columns) == 1


class _DummyScorer(Scorer):
    name = "test_reproducible_dummy"
    description = "Synthetic scorer for the run boundary."
    score_columns = [ScoreColumn("p_msih", "synthetic score", primary=True)]
    patient_aggregation = "mean"

    def compute_batch(self, slide_table, *, clean_slide_ids=None, write_outputs=True, offset=0.0):
        # Deliberately return a broader set; the runner owns estimand filtering.
        frame = slide_table.copy()
        frame["p_msih"] = frame["raw_score"] + offset
        return frame


def test_scorer_runner_writes_scores_metrics_and_environment(tmp_path: Path):
    try:
        register(_DummyScorer.name, _DummyScorer)
    except KeyError:
        pass
    cohort = pd.DataFrame(
        {
            "slide_id": ["s1", "s2", "s3", "s4"],
            "patient_id": ["p1", "p2", "p3", "p4"],
            "site": ["a", "a", "b", "b"],
            "y": [0, 1, 0, 1],
            "in_primary_set": [1, 1, 1, 1],
        }
    )
    cohort_path = tmp_path / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    slides = cohort[["slide_id", "patient_id", "site", "y"]].copy()
    slides["raw_score"] = [0.1, 0.8, 0.2, 0.9]
    slides.loc[len(slides)] = ["outside", "outside", "c", 1, 0.99]
    slides_path = tmp_path / "slides.csv"
    slides.to_csv(slides_path, index=False)
    output = tmp_path / "run"

    outcome = run_scorer(
        _DummyScorer.name,
        cohort_csv=cohort_path,
        slide_table_csv=slides_path,
        output_dir=output,
        parameters={"offset": 0.01},
        workspace=tmp_path,
    )
    assert outcome["n_rows"] == 4
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["patient_auroc"] == 1.0
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["parameters"] == {"offset": 0.01}
    assert "lazyslide" in metadata["environment"]["packages"]
    with pytest.raises(ValueError, match="Unknown parameters"):
        run_scorer(
            _DummyScorer.name,
            cohort_csv=cohort_path,
            slide_table_csv=slides_path,
            output_dir=tmp_path / "typo",
            parameters={"offest": 0.01},
            workspace=tmp_path,
        )
