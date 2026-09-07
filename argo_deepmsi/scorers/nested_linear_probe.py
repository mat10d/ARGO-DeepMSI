"""Nested-CV linear probe across frozen mean-pooled pathology encoders.

Unlike ``simple_grid``, encoder selection occurs only inside each outer training
fold. The reported probabilities therefore never participate in choosing the
encoder that produced them. Scaling and logistic-regression fitting are also
fold-local. This is the valid cohort-trained reference for the v2 full cohort.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

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


def _factory(*, seed: int, C: float, solver: str, max_iter: int):
    def create():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight="balanced",
                random_state=seed,
                solver=solver,
            ),
        )

    return create


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
        self._cache: dict[tuple, pd.DataFrame] = {}

    def write_run_artifacts(self, outdir: Path) -> None:
        if hasattr(self, "_last_fold_audit"):
            self._last_fold_audit.to_json(
                outdir / "nested_fold_audit.json",
                orient="records",
                indent=2,
            )

    def write_metadata(self, outdir: Path) -> None:
        super().write_metadata(outdir)
        path = outdir / "metadata.json"
        metadata = json.loads(path.read_text())
        parameters = getattr(
            self,
            "_last_parameters",
            {
                "outer_splits": 5,
                "inner_splits": 4,
                "repeats": 3,
                "seed": SEED,
                "embeddings": EMBEDDINGS,
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
            },
        )
        metadata["validation"] = {
            "design": "repeated nested StratifiedGroupKFold(patient_id)",
            "outer_splits": parameters["outer_splits"],
            "inner_splits": parameters["inner_splits"],
            "repeats": parameters["repeats"],
            "seed": parameters["seed"],
            "candidates": list(parameters["embeddings"]),
            "selection_metric": "inner patient AUROC",
            "preprocessing": "fold-local StandardScaler",
            "classifier": {
                "type": "LogisticRegression",
                "C": parameters["C"],
                "solver": parameters["solver"],
                "max_iter": parameters["max_iter"],
                "class_weight": "balanced",
            },
        }
        path.write_text(json.dumps(metadata, indent=2))

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        embeddings: Sequence[str] = EMBEDDINGS,
        embedding_root: str | Path = EMB_ROOT,
        cohort_file: str | Path = COHORT,
        repeats: int = 3,
        outer_splits: int = 5,
        inner_splits: int = 4,
        seed: int = SEED,
        C: float = 1.0,
        solver: str = "liblinear",
        max_iter: int = 1000,
        retrain: bool = False,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        if isinstance(embeddings, str):
            embeddings = (embeddings,)
        else:
            embeddings = tuple(embeddings)
        if not embeddings:
            raise ValueError("At least one candidate embedding is required")
        embedding_root = Path(embedding_root)
        cohort_file = Path(cohort_file)
        self._last_parameters = {
            "embeddings": embeddings,
            "embedding_root": embedding_root,
            "cohort_file": cohort_file,
            "repeats": repeats,
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "seed": seed,
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
        }
        cohort = pd.read_csv(cohort_file)
        if clean_slide_ids is not None:
            cohort = cohort[cohort["slide_id"].isin(clean_slide_ids)]

        metadata: dict[str, pd.DataFrame] = {}
        matrices_full: dict[str, np.ndarray] = {}
        common = set(cohort["slide_id"].astype(str))
        for name in embeddings:
            emb_dir = embedding_root / name
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
            .drop_duplicates("slide_id")[["slide_id", "patient_id", "site", "y"]]
            .reset_index(drop=True)
        )
        slide_ids = tuple(frame["slide_id"].astype(str))
        cache_key = (
            slide_ids,
            embeddings,
            repeats,
            outer_splits,
            inner_splits,
            seed,
            C,
            solver,
            max_iter,
            str(embedding_root.resolve()),
            str(cohort_file.resolve()),
        )
        if not retrain and cache_key in self._cache:
            return self._cache[cache_key].copy()
        is_canonical = (
            embeddings == EMBEDDINGS
            and repeats == 3
            and outer_splits == 5
            and inner_splits == 4
            and seed == SEED
            and C == 1.0
            and solver == "liblinear"
            and max_iter == 1000
            and embedding_root.resolve() == EMB_ROOT.resolve()
            and cohort_file.resolve() == COHORT.resolve()
        )
        if (
            not retrain
            and is_canonical
            and self.score_path is not None
            and self.score_path.exists()
        ):
            cached = pd.read_csv(self.score_path)
            if set(cached["slide_id"].astype(str)) == set(slide_ids):
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
            factories[name] = _factory(seed=seed, C=C, solver=solver, max_iter=max_iter)

        scored, folds = nested_grouped_oof(
            frame,
            candidate_matrices,
            factories,
            patient_agg=self.patient_aggregation,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            repeats=repeats,
            seed=seed,
        )
        self._last_fold_audit = folds
        self._cache[cache_key] = scored.copy()
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(self.score_path, index=False)
            self.write_metadata(self.score_path.parent)
            self.write_run_artifacts(self.score_path.parent)
        return scored


register("nested_linear_probe", NestedLinearProbe)
