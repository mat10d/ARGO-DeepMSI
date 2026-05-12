"""Vision-language zero-shot MSI scoring via TITAN's CONCH v1.5 text encoder.

Per slide: compute per-tile cosine similarity between L2-normalised
``conch_v1.5_tiles`` features and prompt-set prototypes
(``MSI-H ‒ MSS``). Aggregate to slide score via mean / top-k / max.
Optionally compare the TITAN slide embedding directly.

No model training. Existing prompts in ``scripts/vl_text_cosine.py``.
"""

from __future__ import annotations

import pandas as pd

from .base import Scorer, ScoreColumn
from .registry import register


class VLTextCosine(Scorer):
    name = "vl_text_cosine"
    description = (
        "Zero-shot MSI scoring via TITAN (CONCH v1.5) text encoder. Per-tile "
        "cosine to curated MSI-H / MSS prompt prototypes, aggregated by "
        "mean / top-k / max. Slide-level variant uses the TITAN slide "
        "embedding directly. No training."
    )
    needs_training_on_our_data = False
    score_columns = [
        ScoreColumn("tile_max", "max over per-tile (MSI−MSS) cos", primary=True),
        ScoreColumn("tile_topk", "mean of top-k tiles (k=64)"),
        ScoreColumn("tile_mean", "mean over all tiles"),
        ScoreColumn("slide_score", "(MSI−MSS) cos on TITAN slide emb"),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        **_,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.score_path)
        if clean_slide_ids is not None:
            df = df[df["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)
        return df


register("vl_text_cosine", VLTextCosine)
