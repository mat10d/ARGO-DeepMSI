"""A2 — LR probe on stain-normalized OAUTHC + original CONCH-TITAN elsewhere.

Image-level counterpart to A1's feature-space batch correction. The OAUTHC-prospective
slides are re-extracted through CONCH v1.5 -> TITAN after Macenko normalization to a
retro-OAU reference (scripts/stain_norm_oauthc.py); every other site keeps its original
`conch_v1.5_titan` vector. The probe (class-balanced logistic regression, 5-fold
patient-grouped, max/√n patient aggregation) then runs on the merged matrix.

Tests whether correcting the batch effect at the PIXEL level — upstream of the frozen
encoder — recovers more OAUTHC MSI signal than the feature-space corrections (A0 Harmony
0.683, A1 batch-correction 0.615). Frozen FM weights; no external data.

Resolution: slide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.metrics import _safe_auprc, _safe_auroc
from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .harmony_probe import OAUTHC_SITE, _by_site_block
from .registry import register
from .slidefm_linearprobe import (
    CLINICAL_CSV,
    EMB_ROOT,
    _load_embedding,
    _oof_full,
    _patient_auroc,
    _patient_max_sqrtn,
)

BASE_EMB = "conch_v1.5_titan"
STAINNORM_DIR = EMB_ROOT / "conch_v1.5_titan_stainnorm"


def _load_stainnorm() -> tuple[np.ndarray, pd.DataFrame] | None:
    npy, csv = STAINNORM_DIR / "embeddings.npy", STAINNORM_DIR / "metadata.csv"
    if not (npy.exists() and csv.exists()):
        return None
    return np.load(npy), pd.read_csv(csv)


class StainNormProbe(Scorer):
    name = "stainnorm_probe"
    description = (
        "LR probe on CONCH-TITAN with OAUTHC-prospective slides replaced by their "
        "Macenko-stain-normalized (retro-OAU reference) re-extraction; all other sites "
        "keep the original conch_v1.5_titan vector. Tests pixel-level batch correction "
        "against feature-space correction (A0/A1). 5-fold patient-grouped, max/√n. "
        "Frozen features, no external data."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "OOF p(MSI-H), stain-normalized OAUTHC + original elsewhere", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
        base = _load_embedding(BASE_EMB, cl, clean_slide_ids)
        if base is None:
            raise RuntimeError(f"{self.name}: base embedding {BASE_EMB} unavailable")
        X = np.load(EMB_ROOT / BASE_EMB / "embeddings.npy")[base["_row"].to_numpy()].astype(np.float32).copy()

        sn = _load_stainnorm()
        if sn is None:
            raise RuntimeError(
                f"{self.name}: stain-norm embedding missing under {STAINNORM_DIR} — "
                "run scripts/stain_norm_oauthc.sh then scripts/stain_norm_oauthc_merge.py"
            )
        sn_X, sn_meta = sn
        sn_row = {str(s): i for i, s in enumerate(sn_meta["slide_id"])}

        n_replaced = 0
        for i, (sid, site) in enumerate(zip(base["slide_id"], base["site"])):
            if site == OAUTHC_SITE and str(sid) in sn_row:
                X[i] = sn_X[sn_row[str(sid)]]
                n_replaced += 1

        oof = _oof_full(X, base["y"].to_numpy(), base["patient_id"].to_numpy())
        slide_df = pd.DataFrame({
            "slide_id": base["slide_id"], "patient_id": base["patient_id"],
            "site": base["site"], "y": base["y"], "p_msih": oof,
        })
        n_oauthc = int((base["site"] == OAUTHC_SITE).sum())
        self._last = {
            "n_replaced": n_replaced,
            "n_oauthc_slides": n_oauthc,
            "patient_auroc": _patient_auroc(slide_df),
        }

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)

        return slide_df


register("stainnorm_probe", StainNormProbe)


def _write_metrics(scorer: "StainNormProbe", slide_df: pd.DataFrame) -> dict:
    pat = _patient_max_sqrtn(slide_df)
    y, s = pat["y"].to_numpy(), pat["score"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))
    by_site = _by_site_block(pat)

    bins = [0, 1, 2, 4, np.inf]
    labels = ["1", "2", "3-4", "5+"]
    pat = pat.assign(_bucket=pd.cut(pat["n_slides"], bins=bins, labels=labels))
    by_bagsize = {
        str(b): {"n": int(len(sub)), "auroc": _safe_auroc(sub["y"].to_numpy(), sub["score"].to_numpy())}
        for b, sub in pat.groupby("_bucket", observed=True)
    }

    oauthc = by_site.get(OAUTHC_SITE, {})
    metrics = {
        "scorer": scorer.name,
        "n_oauthc_slides_replaced": scorer._last["n_replaced"],
        "n_oauthc_slides_total": scorer._last["n_oauthc_slides"],
        "oauthc_auroc": oauthc.get("auroc", float("nan")),
        "oauthc_spec_at_sens95": oauthc.get("spec_at_sens95", float("nan")),
        "oauthc_spec_at_sens96": oauthc.get("spec_at_sens96", float("nan")),
        "oauthc_n_patients": oauthc.get("n", 0),
        "oauthc_prevalence": oauthc.get("prevalence", float("nan")),
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
    out = StainNormProbe().score_path.parent / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
