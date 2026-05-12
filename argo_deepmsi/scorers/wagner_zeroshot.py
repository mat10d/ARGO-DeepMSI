"""Wagner et al. (HistoBistro, Cancer Cell 2023) MSI transformer — zero-shot.

The published transformer MSI classifier (trained on ~13K Western CRC slides
using CTransPath features) applied to our Nigerian cohort with no fine-tuning.

Purely measures Western → Africa generalization. The Wagner architecture +
weights are an external pretrained artifact; no training happens here.

Cache: ``results/scorers/wagner_zeroshot/slide_scores.csv``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Scorer, ScoreColumn
from .registry import register

LEGACY_PATH = Path("results/analysis/wagner_zeroshot/slide_scores.csv")


class WagnerZeroShot(Scorer):
    name = "wagner_zeroshot"
    description = (
        "Wagner et al. (HistoBistro, Cancer Cell 2023) transformer MSI head, "
        "trained on ~13K Western CRC slides with CTransPath features. Applied "
        "zero-shot to the Nigerian cohort: per-slide p_msih via mean-attention "
        "pool over all ctranspath tiles."
    )
    needs_training_on_our_data = False
    score_columns = [
        ScoreColumn("p_msih", "Wagner transformer p(MSI-H)", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        **_,
    ) -> pd.DataFrame:
        """Adapt the legacy slide_scores.csv into canonical schema.

        Wagner is zero-shot — ``clean_slide_ids`` is a filter, not a refit.
        """
        if not LEGACY_PATH.exists():
            raise FileNotFoundError(
                f"{self.name}: legacy output missing at {LEGACY_PATH}. "
                f"Re-run scripts/wagner_zeroshot.py to populate it."
            )
        df = pd.read_csv(LEGACY_PATH)
        keep = ["slide_id", "patient_id", "site", "y", "p_msih"]
        if "stain_location" in df.columns:
            keep.append("stain_location")
        df = df[keep].copy()
        if clean_slide_ids is not None:
            df = df[df["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)
        return df


register("wagner_zeroshot", WagnerZeroShot)
