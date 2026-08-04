"""Nested-CV linear probe across frozen mean-pooled pathology encoders.

Unlike ``simple_grid``, encoder selection occurs only inside each outer training
fold. The reported probabilities therefore never participate in choosing the
encoder that produced them. Scaling and logistic-regression fitting are also
fold-local. This is the valid cohort-trained reference for the v2 full cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.validation import nested_grouped_oof
from .base import Scorer, ScoreColumn
from .registry import register

COHORT = Path("results/data/cohort_clean.csv")
EMB_ROOT = Path("results/embeddings")
EMBEDDINGS = (
    "conch_v1.5_mean",
    "ctranspath_mean",
    "uni2_mean",
    "virchow2_mean",
)
SEED = 42


def _factory():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
            solver="liblinear",
        ),
    )


class NestedLinearProbe(Scorer):
    name = "nested_linear_probe"
    description = (
        "Repeated nested patient-grouped CV selects among four frozen mean-pooled "
        "encoders inside each outer training fold. Scaling and the class-balanced "
        "logistic head are fold-local; outer predictions are untouched by model "
        "selection. Patient aggregation is pre-specified as mean."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    patient_aggregation = "mean"
    score_columns = [
        ScoreColumn("p_msih", "repeated nested-CV p(MSI-H)", primary=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[str, ...], pd.DataFrame] = {}

    def write_metadata(self, outdir: Path) -> None:
        super().write_metadata(outdir)
        path = outdir / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["validation"] = {
            "design": "repeated nested StratifiedGroupKFold(patient_id)",
            "outer_splits": 5,
            "inner_splits": 4,
            "repeats": 3,
            "seed": SEED,
            "candidates": list(EMBEDDINGS),
            "selection_metric": "inner patient AUROC",
            "preprocessing": "fold-local StandardScaler",
        }
        path.write_text(json.dumps(metadata, indent=2))

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        embeddings: tuple[str, ...] = EMBEDDINGS,
        repeats: int = 3,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        cohort = pd.read_csv(COHORT)
        if clean_slide_ids is not None:
            cohort = cohort[cohort["slide_id"].isin(clean_slide_ids)]

        metadata: dict[str, pd.DataFrame] = {}
        matrices_full: dict[str, np.ndarray] = {}
        common = set(cohort["slide_id"].astype(str))
        for name in embeddings:
            emb_dir = EMB_ROOT / name
            if not (emb_dir / "embeddings.npy").exists():
                continue
            meta = pd.read_csv(emb_dir / "metadata.csv").reset_index(drop=True)
            meta["_row"] = np.arange(len(meta))
            metadata[name] = meta
            matrices_full[name] = np.load(emb_dir / "embeddings.npy", mmap_mode="r")
            common &= set(meta["slide_id"].astype(str))
        if not metadata:
            raise RuntimeError(f"{self.name}: none of {embeddings} is available")

        frame = (
            cohort[cohort["slide_id"].isin(common)]
            .sort_values("slide_id")
            .drop_duplicates("slide_id")
            [["slide_id", "patient_id", "site", "y"]]
            .reset_index(drop=True)
        )
        cache_key = tuple(frame["slide_id"].astype(str))
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        if self.score_path is not None and self.score_path.exists():
            cached = pd.read_csv(self.score_path)
            if set(cached["slide_id"].astype(str)) == set(cache_key):
                cached = cached.set_index("slide_id").loc[frame["slide_id"]].reset_index()
                self._cache[cache_key] = cached.copy()
                if write_outputs and not (self.score_path.parent / "metadata.json").exists():
                    self.write_metadata(self.score_path.parent)
                return cached

        candidate_matrices: dict[str, np.ndarray] = {}
        factories = {}
        for name, meta in metadata.items():
            row_map = meta.set_index("slide_id")["_row"]
            rows = row_map.loc[frame["slide_id"]].to_numpy()
            candidate_matrices[name] = np.asarray(matrices_full[name][rows])
            factories[name] = _factory

        scored, folds = nested_grouped_oof(
            frame,
            candidate_matrices,
            factories,
            patient_agg=self.patient_aggregation,
            outer_splits=5,
            inner_splits=4,
            repeats=repeats,
            seed=SEED,
        )
        self._cache[cache_key] = scored.copy()
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(self.score_path, index=False)
            self.write_metadata(self.score_path.parent)
            folds.to_json(
                self.score_path.parent / "nested_fold_audit.json",
                orient="records",
                indent=2,
            )
        return scored


register("nested_linear_probe", NestedLinearProbe)
