"""Attention-MIL: learned slide-level attention pooling, trained on our cohort.

Per-patient bag of all that patient's slide-level features (e.g.
``virchow2_mean``); attention head pools tiles → slide → patient and
emits ``p_msih``. Trained 5-fold (StratifiedGroupKFold on patient_id).

Resolution: patient (the architecture produces a single p_msih per bag
which here is a patient, not a slide).

This is the C5/Phase-2 model — small attention head trained on our
cohort. No external pretraining beyond the frozen embedder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Scorer, ScoreColumn
from .registry import register

LEGACY_PATH = Path("results/analysis/c5_phase2/oof_predictions.csv")


class SlideAttentionMIL(Scorer):
    name = "slide_attention_mil"
    description = (
        "Attention-MIL head trained on our cohort: bag = all of a patient's "
        "slide-level frozen-embedding features; small attention pooler emits "
        "p(MSI-H). Trained 5-fold patient-grouped CV."
    )
    needs_training_on_our_data = True
    resolution = "patient"
    score_columns = [
        ScoreColumn("p_msih", "attention-MIL p(MSI-H)", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        **_,
    ) -> pd.DataFrame:
        """Read cached patient_scores.csv produced by the GPU retrain.

        When ``clean_slide_ids`` is provided we read the clean-cohort
        cache (``patient_scores.csv``) if it exists; otherwise fall back
        to the dirty cohort cache.
        """
        # Prefer the canonical clean cache if a QC retrain has happened.
        clean_cache = Path("results/scorers/slide_attention_mil/patient_scores_clean.csv")
        if clean_slide_ids is not None and clean_cache.exists():
            df = pd.read_csv(clean_cache)
        elif LEGACY_PATH.exists():
            df = pd.read_csv(LEGACY_PATH)
        else:
            raise FileNotFoundError(
                f"{self.name}: legacy output missing at {LEGACY_PATH}. "
                "Re-run scripts/slide_attention_mil.py to populate it."
            )
        st = pd.read_csv("results/data/slide_table_pyramidal.csv")
        site_map = st.groupby("PATIENT")["SITE"].first().to_dict()
        if "site" not in df.columns:
            df["site"] = df["patient_id"].map(site_map)
        if "n_slides" not in df.columns:
            df["n_slides"] = df["patient_id"].map(st.groupby("PATIENT").size().to_dict())

        # When clean_slide_ids is given but the clean cache is missing,
        # filter patients whose every slide is excluded (post-hoc; coarse
        # since the head was fit on dirty CV folds).
        if clean_slide_ids is not None and not clean_cache.exists():
            kept_patients = (
                st.assign(slide_id=st["FILENAME"].apply(lambda p: Path(p).stem))
                  .groupby("PATIENT")["slide_id"]
                  .apply(lambda ids: any(s in clean_slide_ids for s in ids))
            )
            keep = set(kept_patients[kept_patients].index)
            df = df[df["patient_id"].isin(keep)].reset_index(drop=True)

        return df[["patient_id", "y", "site", "n_slides", "p_msih"]].copy()


register("slide_attention_mil", SlideAttentionMIL)
