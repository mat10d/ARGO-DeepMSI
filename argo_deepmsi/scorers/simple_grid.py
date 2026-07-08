"""Grid sweep of lightweight classifier heads on frozen embeddings.

For every combination of:

  classifier  ∈ {logistic_regression, random_forest, xgboost}
  embedding   ∈ {conch_v1.5_mean, conch_v1.5_titan, ctranspath_mean,
                 uni2_mean, virchow2_mean, virchow2_prism}
  representation ∈ {raw, pca100}

fit a class-balanced head 5-fold patient-grouped on our cohort and
score it by OOF patient AUROC (max/√n). The Scorer exposes the best
configuration as the primary slide score; the full grid table is
written alongside for inspection.

"Simple" because everything here is a small head on frozen features —
no foundation-model finetuning, no torch, no GPU. Fit time is minutes
on CPU.

Resolution: slide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import Scorer, ScoreColumn
from .registry import register

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

EMB_ROOT = Path("results/embeddings")
SLIDE_TABLE_CSV = Path("results/data/slide_table_pyramidal.csv")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42

EMBEDDINGS = (
    "conch_v1.5_mean",
    "conch_v1.5_titan",
    "ctranspath_mean",
    "uni2_mean",
    "virchow2_mean",
    "virchow2_prism",
    # Harmony batch-corrected variants (A-phase; skipped automatically if absent).
    "conch_v1.5_mean_harmony",
    "conch_v1.5_titan_harmony",
    "ctranspath_mean_harmony",
    "uni2_mean_harmony",
    "virchow2_mean_harmony",
    "virchow2_prism_harmony",
)
REPRESENTATIONS = ("raw", "pca100")


def _classifiers() -> dict:
    grid = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
    }
    if HAS_XGB:
        # ~19% MSI-H → scale_pos_weight ≈ 4.26
        grid["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=4.26, n_jobs=-1, random_state=SEED,
            eval_metric="auc", tree_method="hist",
        )
    return grid


def _build_pipeline(repr_kind: str, clf_factory) -> Pipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if repr_kind == "pca100":
        steps.append(("pca", PCA(n_components=100, random_state=SEED)))
    steps.append(("clf", clf_factory()))
    return Pipeline(steps)


def _patient_cv_oof(pipe: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in cv.split(X, y, groups=groups):
        from sklearn.base import clone
        p = clone(pipe).fit(X[tr], y[tr])
        oof[te] = p.predict_proba(X[te])[:, 1]
    return oof


def _patient_max_sqrtn_auroc(slide_df: pd.DataFrame, score_col: str) -> float:
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        rows.append({"y": int(g["y"].iloc[0]), "score": float(g[score_col].max() / np.sqrt(len(g)))})
    pat = pd.DataFrame(rows)
    if pat["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(pat["y"], pat["score"]))


class SimpleGrid(Scorer):
    name = "simple_grid"
    description = (
        "Sweeps logistic regression / random forest / xgboost over 6 "
        "frozen embeddings × {raw, PCA-100} on our cohort with 5-fold "
        "patient-grouped CV. Picks the config that maximises patient "
        "AUROC (max/√n) and exposes its OOF as the primary score. The "
        "full sweep table is written to grid.csv for inspection."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "OOF p(MSI-H) of best-grid config", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        embeddings: tuple[str, ...] = EMBEDDINGS,
        representations: tuple[str, ...] = REPRESENTATIONS,
        write_grid: bool = True,
        **_,
    ) -> pd.DataFrame:
        # Shared per-slide labels
        cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)

        clf_factories = _classifiers()

        grid_rows: list[dict] = []
        oof_store: dict[tuple[str, str, str], tuple[pd.DataFrame, np.ndarray]] = {}

        for emb_name in embeddings:
            emb_dir = EMB_ROOT / emb_name
            if not (emb_dir / "embeddings.npy").exists():
                continue
            X_full = np.load(emb_dir / "embeddings.npy")
            meta = pd.read_csv(emb_dir / "metadata.csv")
            if "slide_id" not in meta.columns:
                continue
            meta = meta.copy()
            meta["_row"] = np.arange(len(meta))
            # Embedding metadata already has patient_id + site; join labels only.
            base = meta.merge(
                cl[["PATIENT", "y"]],
                left_on="patient_id", right_on="PATIENT", how="inner",
            )
            if clean_slide_ids is not None:
                base = base[base["slide_id"].isin(clean_slide_ids)]
            base = base.drop_duplicates("slide_id").reset_index(drop=True)
            if len(base) == 0:
                continue
            X = X_full[base["_row"].to_numpy()]
            y = base["y"].to_numpy()
            groups = base["patient_id"].to_numpy()

            for repr_kind in representations:
                if repr_kind == "pca100" and X.shape[1] <= 100:
                    continue
                for clf_name, clf_factory in clf_factories.items():
                    pipe = _build_pipeline(repr_kind, clf_factory)
                    oof = _patient_cv_oof(pipe, X, y, groups)
                    slide_df = pd.DataFrame({
                        "slide_id": base["slide_id"],
                        "patient_id": base["patient_id"],
                        "site": base["site"],
                        "y": base["y"],
                        "p_msih": oof,
                    })
                    pat_auroc = _patient_max_sqrtn_auroc(slide_df, "p_msih")
                    slide_auroc = float(roc_auc_score(slide_df["y"], slide_df["p_msih"])) \
                        if slide_df["y"].nunique() == 2 else float("nan")
                    grid_rows.append({
                        "embedding": emb_name,
                        "classifier": clf_name,
                        "representation": repr_kind,
                        "n_slides": int(len(slide_df)),
                        "n_patients": int(slide_df["patient_id"].nunique()),
                        "slide_auroc": slide_auroc,
                        "patient_auroc": pat_auroc,
                    })
                    oof_store[(emb_name, clf_name, repr_kind)] = (slide_df, oof)

        if not grid_rows:
            raise RuntimeError(f"{self.name}: grid produced no rows")

        grid_df = pd.DataFrame(grid_rows).sort_values("patient_auroc", ascending=False).reset_index(drop=True)
        best_key = (
            grid_df.iloc[0]["embedding"],
            grid_df.iloc[0]["classifier"],
            grid_df.iloc[0]["representation"],
        )
        best_slide_df, _ = oof_store[best_key]

        if write_grid and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            grid_df.to_csv(self.score_path.parent / "grid.csv", index=False)
            (self.score_path.parent / "best_config.txt").write_text(
                f"embedding={best_key[0]}  classifier={best_key[1]}  representation={best_key[2]}  "
                f"patient_auroc={grid_df.iloc[0]['patient_auroc']:.4f}\n"
            )
            # Persist canonical slide_scores.csv (matches whatever pass
            # wrote grid.csv — the comparison harness runs clean last, so
            # the canonical CSV ends up reflecting the clean cohort).
            best_slide_df.to_csv(self.score_path, index=False)

        return best_slide_df


register("simple_grid", SimpleGrid)
