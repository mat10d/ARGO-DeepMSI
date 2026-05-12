"""Calibrated patient-level aggregation of Wagner per-slide scores.

Aggregators (per patient, over that patient's Wagner per-slide p_msih):

- ``max_over_sqrtn``  : max / √n_slides (primary; the C5-best baseline)
- ``raw_max``         : max
- ``raw_mean``        : mean
- ``top3_mean``       : mean of top 3
- ``max_minus_Emax_MSS``: max minus the empirical MSS Emax at this n

The MSS null curve is built from the Wagner per-slide p_msih of every
MSS slide in the (post-QC) cohort: for each ``n``, draw N bootstrap
samples of size n with replacement from the MSS slide pool and take
the mean of max-over-bag. No learned parameters; the null is closed
form once you fix the cohort.

Resolution: patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Scorer, ScoreColumn
from .registry import register
from .wagner_zeroshot import WagnerZeroShot

N_BOOT = 1000
MAX_N = 60
SEED = 42


def _mss_null_curve(mss_slide_p: np.ndarray, max_n: int = MAX_N) -> pd.DataFrame:
    """E[max] over n-bootstrap-sized MSS slide bags, n=1..max_n."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n in range(1, max_n + 1):
        draws = rng.choice(mss_slide_p, size=(N_BOOT, n), replace=True)
        rows.append({"n": n, "Emax": float(draws.max(axis=1).mean())})
    return pd.DataFrame(rows).set_index("n")


def _aggregate_patient(g: pd.DataFrame, null_df: pd.DataFrame) -> dict:
    p = g["p_msih"].to_numpy()
    n = len(p)
    sorted_desc = np.sort(p)[::-1]
    return {
        "n_slides": n,
        "raw_mean": float(p.mean()),
        "raw_max": float(p.max()),
        "top3_mean": float(sorted_desc[: min(3, n)].mean()),
        "max_over_sqrtn": float(p.max() / np.sqrt(n)),
        "max_minus_Emax_MSS": float(p.max() - null_df.loc[min(n, MAX_N), "Emax"]),
    }


class CalibratedPool(Scorer):
    name = "calibrated_pool"
    description = (
        "Patient-level aggregation operators applied to Wagner per-slide "
        "p_msih. Headline is max/√n; calibrated variants subtract the "
        "empirical MSS Emax(n) bootstrap curve. The null curve is rebuilt "
        "from whatever slide set the harness hands in (so QC exclusion "
        "propagates correctly)."
    )
    needs_training_on_our_data = False
    resolution = "patient"
    score_columns = [
        ScoreColumn("max_over_sqrtn", "max p(MSI-H) / √n_slides", primary=True),
        ScoreColumn("raw_max", "max p(MSI-H) over slides"),
        ScoreColumn("raw_mean", "mean p(MSI-H) over slides"),
        ScoreColumn("top3_mean", "mean of top-3 slide p(MSI-H)"),
        ScoreColumn("max_minus_Emax_MSS", "max − empirical MSS Emax at this n"),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        **_,
    ) -> pd.DataFrame:
        # Source per-slide scores from the canonical Wagner output (no
        # dirty legacy CSVs).
        wagner = WagnerZeroShot().compute_batch(pd.DataFrame())
        if clean_slide_ids is not None:
            wagner = wagner[wagner["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)

        mss_pool = wagner.loc[wagner["y"] == 0, "p_msih"].to_numpy()
        if len(mss_pool) == 0:
            raise ValueError(f"{self.name}: empty MSS slide pool — can't build null curve")
        null_df = _mss_null_curve(mss_pool)

        rows = []
        for pid, g in wagner.groupby("patient_id"):
            row = {
                "patient_id": pid,
                "y": int(g["y"].iloc[0]),
                "site": g["site"].iloc[0],
            }
            row.update(_aggregate_patient(g, null_df))
            rows.append(row)
        return pd.DataFrame(rows)


register("calibrated_pool", CalibratedPool)
