"""Uniform interface for every MSI scorer.

Each approach implements a ``Scorer`` subclass with three primary methods:

- ``predict_slide(zarr_path)``  : score one slide → float
- ``predict_batch(slide_table)``: score many slides → DataFrame
- ``fit(train_df)``             : (optional) train internal head on our data

The ``needs_training_on_our_data`` flag separates zero-shot scorers
(``vl_text_cosine``, ``wagner_zeroshot``) from scorers that fit a small
head on our cohort (``slide_attention_mil``, ``nuclear_morphology``,
``calibrated_pool``).

Constraint: NO scorer is allowed to train on external slide sets —
fitting on our 200-patient cohort is always permitted, fitting on
TCGA / external pretraining sets is not.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_ROOT = Path("results/scorers")


@dataclass
class ScoreColumn:
    """One column produced by a scorer's slide_scores.csv.

    A scorer may emit multiple score variants (e.g. tile_mean / tile_max /
    slide_score for ``vl_text_cosine``); each variant is one ``ScoreColumn``.
    The ``primary`` variant is what the unified comparison ranks by.
    """

    name: str            # column name in slide_scores.csv
    description: str     # one-line human description
    primary: bool = False


class Scorer(abc.ABC):
    """Base class for any MSI scoring approach."""

    name: str = ""                                 # snake_case identifier
    description: str = ""                          # one-paragraph mechanism
    needs_training_on_our_data: bool = False       # fit() does meaningful work?
    score_columns: list[ScoreColumn] = []          # what slide_scores.csv contains
    resolution: str = "slide"                      # "slide" or "patient"
    patient_aggregation: str = "max_sqrtn"         # slide -> patient operator

    def __init__(self) -> None:
        if self.name and self.score_path is None:
            fname = "patient_scores.csv" if self.resolution == "patient" else "slide_scores.csv"
            self.score_path = RESULTS_ROOT / self.name / fname

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> None:  # noqa: B027
        """Optional: train an internal head on our cohort. Default no-op."""

    score_path: Path | None = None
    """results/scorers/<name>/slide_scores.csv — set by subclasses."""

    def predict_batch(
        self, slide_table: pd.DataFrame | None = None, *, use_cache: bool = True, **kwargs: Any
    ) -> pd.DataFrame:
        """Score a batch of slides.

        Default behaviour: read the cached ``slide_scores.csv`` at
        ``self.score_path``. Subclasses override ``compute_batch`` to
        recompute from raw data when ``use_cache=False`` or the cache
        is missing.

        Returns a DataFrame containing at minimum:
            slide_id, patient_id, site, <every score_column.name>
        """
        if use_cache and self.score_path is not None and self.score_path.exists():
            return pd.read_csv(self.score_path)
        if slide_table is None:
            raise ValueError(
                f"{self.name}: no cache at {self.score_path} and no slide_table provided"
            )
        return self.compute_batch(slide_table, **kwargs)

    def compute_batch(self, slide_table: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Recompute scores from scratch. Default: not implemented."""
        raise NotImplementedError(
            f"{self.name} does not support recomputation; provide cached "
            f"scores at {self.score_path}"
        )

    @property
    def primary_score(self) -> str:
        for c in self.score_columns:
            if c.primary:
                return c.name
        if self.score_columns:
            return self.score_columns[0].name
        raise ValueError(f"Scorer {self.name} declares no score columns")

    def publish(self, slide_table: pd.DataFrame | None = None, **kwargs: Any) -> Path:
        """Run compute_batch and write the canonical slide_scores.csv.

        Returns the path written. Used by ``argo score <name>`` and by the
        comparison harness to ensure the canonical cache is up to date.
        """
        assert self.score_path is not None, f"{self.name} has no score_path"
        df = self.compute_batch(slide_table if slide_table is not None else pd.DataFrame(), **kwargs)
        self.score_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.score_path, index=False)
        self.write_metadata(self.score_path.parent)
        self.write_run_artifacts(self.score_path.parent)
        return self.score_path

    def write_metadata(self, outdir: Path) -> None:
        """Drop a metadata.json so downstream tooling knows what this is."""
        outdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "description": self.description,
            "needs_training_on_our_data": self.needs_training_on_our_data,
            "resolution": self.resolution,
            "score_columns": [
                {"name": c.name, "description": c.description, "primary": c.primary}
                for c in self.score_columns
            ],
            "primary_score": self.primary_score,
            "patient_aggregation": self.patient_aggregation,
        }
        (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def write_run_artifacts(self, outdir: Path) -> None:  # noqa: B027
        """Write optional fold audits/curves produced by the most recent run."""
