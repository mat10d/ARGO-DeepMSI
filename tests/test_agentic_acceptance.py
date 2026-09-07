from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argo_deepmsi.cli import app
from argo_deepmsi.doctor import run_doctor
from argo_deepmsi.experiment import load_experiment
from argo_deepmsi.experiment_schema import STRATEGIES, experiment_json_schema
from argo_deepmsi.synthetic import run_synthetic_acceptance


runner = CliRunner()


def test_schema_requires_labels_for_mixed_strategy(tmp_path: Path):
    config = tmp_path / "mixed.toml"
    config.write_text("""
[run]
name = "mixed"
strategy = "mixed"
clinical_table = "clinical.csv"

[[train]]
embedding = "uni2_mean"
""")
    with pytest.raises(ValueError, match="require a strategy"):
        load_experiment(config)


def test_schema_enforces_resource_budgets(tmp_path: Path):
    config = tmp_path / "budget.toml"
    config.write_text("""
[run]
name = "bounded"
strategy = "frozen_foundation_trained_head"
cohort = "cohort.csv"
slide_table = "slides.csv"

[run.budget]
max_epochs = 2

[[scorer]]
name = "bag_transformer"
strategy = "frozen_foundation_trained_head"
parameters = { epochs = 3 }
""")
    with pytest.raises(ValueError, match="exceeding"):
        load_experiment(config)


def test_generated_schema_exposes_every_strategy():
    jsonschema = pytest.importorskip("jsonschema")
    schema = experiment_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    encoded = json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(STRATEGIES) == set(schema["x-strategy-descriptions"])
    assert set(STRATEGIES) == set(schema["$defs"]["strategy"]["enum"])
    assert all(strategy in encoded for strategy in STRATEGIES)


def test_doctor_is_machine_readable_and_does_not_expose_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "uv.lock").touch()
    config = tmp_path / "experiment.toml"
    config.write_text("""
[run]
name = "doctor"
workspace = "."
device = "cpu"
strategy = "pretrained_end_to_end"

[comparison]
enabled = true
""")
    versions = {"lazyslide": "0.12.0", "lazyslide-models": "0.0.4", "wsidata": "0.11.0"}
    monkeypatch.setattr("argo_deepmsi.doctor._package_version", versions.get)
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")

    report = run_doctor(config_path=config)

    assert report["counts"]["error"] == 0
    assert "must-not-appear" not in json.dumps(report)
    assert {check["name"] for check in report["checks"]} >= {
        "config",
        "workspace",
        "dependencies",
        "accelerator",
    }


def test_doctor_json_cli_is_valid_json(monkeypatch: pytest.MonkeyPatch):
    report = {
        "status": "ok",
        "workspace": "/a/long/workspace/path",
        "config": None,
        "counts": {"ok": 1, "warning": 0, "error": 0},
        "checks": [
            {
                "name": "workspace",
                "status": "ok",
                "message": "a deliberately long machine-readable message " * 5,
                "details": None,
            }
        ],
    }
    monkeypatch.setattr("argo_deepmsi.doctor.run_doctor", lambda **_kwargs: report)

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == report


def test_synthetic_acceptance_exercises_every_experiment_stage(tmp_path: Path):
    report = run_synthetic_acceptance(tmp_path / "acceptance")

    assert report["status"] == "completed"
    assert [stage["kind"] for stage in report["stages"]] == [
        "extract",
        "aggregate",
        "bag",
        "train",
        "scorer",
        "comparison",
    ]
    assert all(
        stage["strategy"] == "frozen_foundation_trained_head"
        for stage in report["stages"]
        if stage["kind"] in {"train", "scorer"}
    )
    assert (tmp_path / "acceptance/results/runs/synthetic-acceptance/comparison.csv").is_file()
