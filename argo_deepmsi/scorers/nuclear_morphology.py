"""NuLite nuclear segmentation + LR head — interpretable morphology.

Per slide: NuLite cell-type segmentation on a 100-tile random subset
(at 256×256, 0.5 mpp, ``amp=False``). Computes 11 features:

    fractions      : neoplastic / inflammatory / connective / dead / epithelial
    derived ratios : til_density, stroma_ratio, immune_density,
                     necrosis_fraction, mitotic_density, cells_per_tile

Logistic-regression head fit under StratifiedGroupKFold(patient_id, k=5),
class-balanced, StandardScaler-piped. The LR head is trained on *our*
cohort only — no external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import Scorer, ScoreColumn
from .registry import register

FEATURES_PATH = Path("results/scorers/nuclear_morphology/features.csv")
FEATURE_COLS = [
    "frac_neoplastic", "frac_inflammatory", "frac_connective",
    "frac_dead", "frac_epithelial",
    "til_density", "stroma_ratio", "immune_density",
    "necrosis_fraction", "mitotic_density", "cells_per_tile",
]
SEED = 42


class NuclearMorphology(Scorer):
    name = "nuclear_morphology"
    description = (
        "NuLite cell-type fractions (tumor / lymphocyte / stroma / dead / "
        "epithelial) + derived ratios (TIL density, stroma ratio, ...) fed "
        "to a class-balanced LR head trained 5-fold patient-grouped on our "
        "cohort. Refit on whatever slide set the harness hands in."
    )
    needs_training_on_our_data = True
    score_columns = [
        ScoreColumn("p_msih", "LR p(MSI-H) on morphology features", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        **_,
    ) -> pd.DataFrame:
        feats = pd.read_csv(FEATURES_PATH)
        cl = pd.read_csv("results/data/clinical_table.csv")[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
        df = feats.merge(cl, left_on="patient_id", right_on="PATIENT", how="inner")

        if clean_slide_ids is not None:
            df = df[df["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)

        # Drop any feature columns not present (forward-compatible)
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feat_cols].fillna(0.0).to_numpy()
        y = df["y"].to_numpy()
        groups = df["patient_id"].to_numpy()

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(df), dtype=np.float64)
        for tr, te in cv.split(X, y, groups):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            )
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]

        out = pd.DataFrame({
            "slide_id": df["slide_id"],
            "patient_id": df["patient_id"],
            "site": df["site"],
            "y": df["y"],
            "p_msih": oof,
        })
        return out


register("nuclear_morphology", NuclearMorphology)
