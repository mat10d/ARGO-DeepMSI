"""Fast synthetic acceptance run for the complete experiment boundary."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd

from .experiment import run_experiment


def _write_fixture(workspace: Path, seed: int) -> Path:
    data_dir = workspace / "results" / "data"
    data_dir.mkdir(parents=True)
    slide_rows = []
    cohort_rows = []
    clinical_rows = []
    for index in range(24):
        patient_id = f"P{index:03d}"
        slide_id = f"synthetic_{index:03d}"
        label = index % 2
        site = "site_a" if index % 4 < 2 else "site_b"
        slide_path = workspace / "slides" / f"{slide_id}.svs"
        slide_path.parent.mkdir(exist_ok=True)
        slide_path.touch()
        slide_path.with_suffix(".zarr").mkdir()
        slide_rows.append({"PATIENT": patient_id, "FILENAME": slide_path, "SITE": site})
        cohort_rows.append(
            {
                "slide_id": slide_id,
                "patient_id": patient_id,
                "site": site,
                "y": label,
                "in_primary_set": 1,
            }
        )
        clinical_rows.append({"PATIENT": patient_id, "isMSIH": "MSI-H" if label else "MSS"})
    pd.DataFrame(slide_rows).to_csv(data_dir / "slide_table.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(data_dir / "cohort_clean.csv", index=False)
    pd.DataFrame(clinical_rows).to_csv(data_dir / "clinical_table.csv", index=False)

    config = workspace / "synthetic-acceptance.toml"
    config.write_text(
        dedent("""
            [run]
            name = "synthetic-acceptance"
            workspace = "."
            slide_table = "results/data/slide_table.csv"
            clinical_table = "results/data/clinical_table.csv"
            cohort = "results/data/cohort_clean.csv"
            seed = 7
            device = "cpu"
            strategy = "frozen_foundation_trained_head"

            [run.budget]
            max_models = 1
            max_stages = 6
            max_trainable_stages = 2
            max_epochs = 1
            max_repeats = 1
            allow_network = false
            allow_model_downloads = false

            [extract]
            enabled = true
            engine = "local"
            models = ["synthetic"]
            device = "cpu"
            allow_failures = false

            [[aggregate]]
            id = "mean"
            models = ["synthetic"]
            method = "mean"
            write_h5ad = false

            [[bag]]
            id = "tiles"
            models = ["synthetic"]
            max_tiles = 4
            seed = 7

            [[train]]
            id = "linear"
            strategy = "frozen_foundation_trained_head"
            embedding = "synthetic_mean"
            classifiers = ["logistic"]
            n_splits = 2

            [[scorer]]
            id = "nested"
            strategy = "frozen_foundation_trained_head"
            name = "nested_linear_probe"
            parameters = { embeddings = ["synthetic_mean"], embedding_root = "$WORKSPACE/results/embeddings", cohort_file = "$WORKSPACE/results/data/cohort_clean.csv", repeats = 1, outer_splits = 2, inner_splits = 2, seed = 7 }

            [comparison]
            enabled = true
            """).strip()
        + "\n"
    )
    return config


def run_synthetic_acceptance(
    workspace: Path,
    *,
    seed: int = 7,
    on_stage: Callable[[str, str], None] | None = None,
) -> dict:
    """Exercise every orchestration stage without slide I/O or model downloads.

    The extraction call is replaced only at the external WSI/model boundary;
    aggregation, bagging, training, nested scoring, comparison, provenance,
    budgeting, and resume machinery are the production implementations.
    """
    workspace = workspace.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"Synthetic acceptance workspace must be empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    config = _write_fixture(workspace, seed)

    def synthetic_features(path: Path) -> np.ndarray:
        index = int(path.stem.rsplit("_", 1)[-1])
        label = index % 2
        rng = np.random.default_rng(seed + index)
        return rng.normal(loc=0.6 * label, scale=1.0, size=(8, 12)).astype(np.float32)

    def synthetic_extract(*, slide_table: pd.DataFrame, **_options) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "slide": slide_table["FILENAME"].map(lambda value: Path(str(value)).name),
                "success": True,
                "zarr_path": slide_table["FILENAME"].map(
                    lambda value: str(Path(str(value)).with_suffix(".zarr"))
                ),
            }
        )

    def synthetic_store(zarr_path: Path):
        return {"tables": {"synthetic_tiles": {"X": synthetic_features(zarr_path)}}}

    def synthetic_bag(zarr_path: Path, _model: str) -> np.ndarray:
        return synthetic_features(zarr_path)

    with (
        patch(
            "argo_deepmsi.feature_extraction.extract_features_batch",
            side_effect=synthetic_extract,
        ),
        patch(
            "argo_deepmsi.feature_extraction._open_feature_store",
            side_effect=synthetic_store,
        ),
        patch("argo_deepmsi.bags._read_tile_features", side_effect=synthetic_bag),
    ):
        manifest = run_experiment(config, resume=False, on_stage=on_stage)
    expected = {
        "extract",
        "aggregate:mean",
        "bag:tiles",
        "train:linear",
        "scorer:nested",
        "comparison",
    }
    completed = {stage["id"] for stage in manifest["stages"] if stage["status"] == "completed"}
    if manifest["status"] != "completed" or completed != expected:
        raise RuntimeError(
            f"Synthetic acceptance did not complete the expected stages: {sorted(completed)}"
        )
    manifest["acceptance_workspace"] = str(workspace)
    return manifest
