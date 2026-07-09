"""D2 — per-slide reliability weight + weighted patient aggregation.

Hard QC exclusion (in_clean_set) is a 0/1 slide gate that, because QC-flagging is ~99%
OAUTHC vs ~20% MSK, effectively shrinks OAUTHC. The SOFT alternative keeps every slide
but discounts unreliable ones in the max/√n patient aggregation via a continuous weight
w ∈ [FLOOR, 1] built from the same signals the hard gate used:

    w = FLOOR + (1-FLOOR) * (1 - artifact_fraction) * min(tumor_fraction/TUMOR_REF, 1)
                                                     * min(n_tiles/NTILES_REF, 1)

A small FLOOR keeps every slide contributing something (that is what makes it "soft"
rather than another exclusion). Missing QC fields are imputed conservatively (unknown
artifact → 0.5; no confirmed tumor → 0). Weighted aggregation:

    patient_score = max_i(w_i · s_i) / sqrt(Σ_i w_i)

so an unreliable high-scoring slide is both pulled down (w·s) and contributes less to the
soft bag size (Σw). Pure functions; deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TUMOR_REF = 0.10      # tumor_fraction that saturates the tumor term
NTILES_REF = 200      # n_tiles that saturates the size-confidence term
FLOOR = 0.05          # minimum weight — no slide is fully zeroed (vs hard exclusion)


def reliability_weight(df: pd.DataFrame) -> pd.Series:
    """Per-slide reliability weight in [FLOOR, 1] from artifact/tumor/n_tiles."""
    af = pd.to_numeric(df.get("artifact_fraction"), errors="coerce").fillna(0.5).clip(0, 1)
    tf = pd.to_numeric(df.get("tumor_fraction"), errors="coerce").fillna(0.0).clip(0, 1)
    nt = pd.to_numeric(df.get("n_tiles"), errors="coerce")
    nt = nt.fillna(nt.median() if nt.notna().any() else NTILES_REF)
    w_art = 1.0 - af
    w_tum = (tf / TUMOR_REF).clip(0, 1)
    w_nt = (nt / NTILES_REF).clip(0, 1)
    w = FLOOR + (1.0 - FLOOR) * (w_art * w_tum * w_nt)
    return w.clip(FLOOR, 1.0).rename("w")


def patient_max_sqrtn(
    slide_df: pd.DataFrame, score_col: str = "score", weight_col: str | None = None
) -> pd.DataFrame:
    """Aggregate slides to one patient score via (weighted) max/√n.

    weight_col=None → plain max(s)/√n. Otherwise → max(w·s)/√(Σw), a reliability-
    discounted rule-out score.
    """
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        s = g[score_col].to_numpy(dtype=float)
        if weight_col is None:
            n = len(s)
            score = float(s.max() / np.sqrt(n)) if n else float("nan")
            n_eff = float(n)
        else:
            w = g[weight_col].to_numpy(dtype=float)
            W = float(w.sum())
            score = float((w * s).max() / np.sqrt(W)) if W > 0 else float("nan")
            n_eff = W
        rows.append({
            "patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
            "n_slides": int(len(g)), "n_eff": n_eff, "score": score,
        })
    return pd.DataFrame(rows)
