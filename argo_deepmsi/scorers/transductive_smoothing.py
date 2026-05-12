"""kNN label propagation over tile VL scores (Histo-TransCLIP style).

Per slide: build cosine kNN graph (k=10) on ``conch_v1.5_tiles`` features;
iterate row-stochastic transition matrix W: ``s ← (1−α) s_0 + α W s``
for T iterations (α=0.7, T=5); aggregate smoothed s to slide score.

Reference: Zanella et al. 2024 — Histo-TransCLIP. No training.
"""

from __future__ import annotations

import pandas as pd

from .base import Scorer, ScoreColumn
from .registry import register


class TransductiveSmoothing(Scorer):
    name = "transductive_smoothing"
    description = (
        "kNN-graph label propagation over per-tile (MSI−MSS) VL scores. "
        "Smooths the raw vl_text_cosine output through a within-slide tile "
        "affinity graph. No training."
    )
    needs_training_on_our_data = False
    score_columns = [
        ScoreColumn("smooth_max", "max over graph-smoothed tile scores", primary=True),
        ScoreColumn("smooth_topk", "mean of top-k smoothed tiles"),
        ScoreColumn("smooth_mean", "mean over all smoothed tiles"),
        ScoreColumn("raw_topk", "mean top-k of unsmoothed tiles (control)"),
        ScoreColumn("raw_max", "max of unsmoothed tiles (control)"),
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


register("transductive_smoothing", TransductiveSmoothing)
