"""A0 — LR probe on Harmony-corrected CONCH-TITAN slide embeddings.

Promotes the D0 root-cause finding (docs/experiments/D0-oauthc-batch-rootcause.md)
into a registered scorer and makes it the evaluated baseline for the OAUTHC-recovery
(A-phase). D0 established that OAUTHC's MSI deficit is a processing-pipeline batch
effect, not biology: raw ``conch_v1.5_titan`` separates the OAUTHC-prospective vs
retrospective_oau pipelines at site-pred AUROC 1.00, and Harmony batch-correction
(``conch_v1.5_titan_harmony``) drops that to 0.82 while lifting OAUTHC MSI-AUROC
0.61 -> 0.69.

This scorer is exactly S1's linear probe (class-balanced logistic regression,
5-fold patient-grouped, max/√n patient aggregation) fixed to the single Harmony-
corrected embedding, so the board carries a clean apples-to-apples "Harmony
baseline" row. Frozen embeddings only — Harmony was fit on OUR cohort's features
(no external data), the FM weights are frozen feature extractors.

Resolution: slide.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ..eval.metrics import _safe_auprc, _safe_auroc
from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register
from .slidefm_linearprobe import (
    CLINICAL_CSV,
    EMB_ROOT,
    _load_embedding,
    _oof_full,
    _patient_auroc,
    _patient_max_sqrtn,
)

# The Harmony-corrected CONCH-TITAN embedding (D0 base model for the A-phase).
HARMONY_EMB = "conch_v1.5_titan_harmony"
# The prospective-OAUTHC site the A-phase must recover (48% of patients).
OAUTHC_SITE = "OAUTHC"


class HarmonyProbe(Scorer):
    name = "harmony_probe"
    description = (
        "Class-balanced logistic-regression linear probe on the Harmony batch-"
        "corrected CONCH-TITAN slide embedding (conch_v1.5_titan_harmony), 5-fold "
        "patient-grouped, max/√n patient aggregation. The D0-promoted OAUTHC-"
        "recovery baseline: Harmony deflates the OAUTHC-prospective vs retro-OAU "
        "processing-batch axis (site-pred AUROC 1.00 -> 0.82) that the raw embedding "
        "latches onto instead of MSI morphology. Frozen features; Harmony fit on our "
        "own cohort (no external data)."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    # D3 found that smooth OOF probe scores benefit from averaging, unlike the
    # Wagner max/√n score. Keep this pre-specified in the scorer contract.
    patient_aggregation = "mean"
    score_columns = [
        ScoreColumn("p_msih", "full-shot OOF p(MSI-H), Harmony CONCH-TITAN", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        embedding: str = HARMONY_EMB,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)

        base = _load_embedding(embedding, cl, clean_slide_ids)
        if base is None:
            raise RuntimeError(f"{self.name}: embedding {embedding} unavailable under {EMB_ROOT}")

        X = np.load(EMB_ROOT / embedding / "embeddings.npy")[base["_row"].to_numpy()]
        oof = _oof_full(X, base["y"].to_numpy(), base["patient_id"].to_numpy())
        slide_df = pd.DataFrame({
            "slide_id": base["slide_id"],
            "patient_id": base["patient_id"],
            "site": base["site"],
            "y": base["y"],
            "p_msih": oof,
        })

        self._last = {
            "embedding": embedding,
            "patient_auroc": _patient_auroc(slide_df),
        }

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)

        return slide_df


register("harmony_probe", HarmonyProbe)


def _by_site_block(pat: pd.DataFrame) -> dict:
    out = {}
    for site, sub in pat.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["score"].to_numpy()
        out[str(site)] = {
            "n": int(len(sub)),
            "prevalence": float(sub["y"].mean()),
            "auroc": _safe_auroc(yy, ss),
            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
            if sub["y"].nunique() == 2 else float("nan"),
            "spec_at_sens96": screening_block(yy, ss, (0.96,))["spec_at_sens96"]
            if sub["y"].nunique() == 2 else float("nan"),
        }
    return out


def _write_metrics(scorer: "HarmonyProbe", slide_df: pd.DataFrame) -> dict:
    """Assemble + write the FULL metric block, OAUTHC operating point FIRST."""
    pat = _patient_max_sqrtn(slide_df)
    y = pat["y"].to_numpy()
    s = pat["score"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))
    by_site = _by_site_block(pat)

    bins = [0, 1, 2, 4, np.inf]
    labels = ["1", "2", "3-4", "5+"]
    pat = pat.assign(_bucket=pd.cut(pat["n_slides"], bins=bins, labels=labels))
    by_bagsize = {
        str(bucket): {
            "n": int(len(sub)),
            "auroc": _safe_auroc(sub["y"].to_numpy(), sub["score"].to_numpy()),
        }
        for bucket, sub in pat.groupby("_bucket", observed=True)
    }

    oauthc = by_site.get(OAUTHC_SITE, {})
    metrics = {
        "scorer": scorer.name,
        "embedding": scorer._last["embedding"],
        # ---- OAUTHC-first (A-phase mission) ----
        "oauthc_auroc": oauthc.get("auroc", float("nan")),
        "oauthc_spec_at_sens95": oauthc.get("spec_at_sens95", float("nan")),
        "oauthc_spec_at_sens96": oauthc.get("spec_at_sens96", float("nan")),
        "oauthc_n_patients": oauthc.get("n", 0),
        "oauthc_prevalence": oauthc.get("prevalence", float("nan")),
        # ---- overall ----
        "n_slides": int(len(slide_df)),
        "n_patients": int(len(pat)),
        "prevalence_patient": float(pat["y"].mean()),
        "auroc": _safe_auroc(y, s),
        "auprc": _safe_auprc(y, s),
        "sensitivity": block["op_sens95"]["sensitivity"],
        "spec_at_sens90": block["spec_at_sens90"],
        "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"],
        "npv_at_sens95": block["npv_at_sens95"],
        "operating_threshold": block["op_sens95"]["threshold"],
        "by_site": by_site,
        "by_bagsize": by_bagsize,
        "screening_clean": block,
    }
    out = HarmonyProbe().score_path.parent / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
