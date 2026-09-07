"""A1 — batch-correction sweep on frozen CONCH-TITAN (OAUTHC-targeted).

D0 established the OAUTHC deficit is a processing-pipeline batch effect and A0's
Harmony probe recovered part of it (OAUTHC AUROC 0.608 -> 0.683) but leaves a large
residual batch axis (site-pred OAUTHC-prosp vs retro-OAU AUROC 0.82). This scorer
sweeps STRONGER / TARGETED feature-space corrections on the raw ``conch_v1.5_titan``
embedding and keeps the one with the highest **OAUTHC held-out patient AUROC**:

  - ``raw``                  : no correction (reference).
  - ``combat_global``        : parametric-EB ComBat (inmoose ``pycombat_norm``) with
                               batch = site — global location/scale equalization.
  - ``oauthc_target_retrooau``: OAUTHC-prospective location/scale aligned to the
                               retrospective_oau reference (SAME OAU patients, tissue
                               cut at MSKCC) — one-site target, other sites untouched.
  - ``oauthc_target_pooled`` : OAUTHC-prospective aligned to the pooled non-OAUTHC
                               reference.

R2/R3 proved GLOBAL site-invariance equalization regresses to the mean, so the two
``oauthc_target_*`` methods correct ONLY the OAUTHC-prospective batch (the odd-one-out
in the cut-location split), leaving the already-good sites alone. Metric per method:
OAUTHC held-out AUROC + residual batch separability (site-pred AUROC on corrected
features). The primary score is the winning method's 5-fold patient-grouped OOF.

Corrections are unsupervised w.r.t. MSI (they use only per-slide features + site) and
fit transductively on the cohort being scored — the same protocol A0/Harmony used.
No external data; frozen features.

Resolution: slide.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..eval.metrics import _safe_auprc, _safe_auroc
from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .harmony_probe import OAUTHC_SITE, _by_site_block
from .registry import register
from .slidefm_linearprobe import (
    CLINICAL_CSV,
    EMB_ROOT,
    _load_embedding,
    _lr_pipeline,
    _oof_full,
    _patient_auroc,
    _patient_max_sqrtn,
)

RAW_EMB = "conch_v1.5_titan"
SEED = 42
N_SPLITS = 5
EPS = 1e-8
# retro-OAU = same OAU patients cut at MSKCC → the natural same-population reference.
RETRO_OAU = "retrospective_oau"

METHODS = ("raw", "combat_global", "oauthc_target_retrooau", "oauthc_target_pooled")


def _combat_global(X: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Parametric-EB ComBat across sites. X is (n_slides, n_feat)."""
    from inmoose.pycombat import pycombat_norm

    # pycombat_norm wants features × samples; batch = per-sample label.
    corrected = pycombat_norm(X.T.astype(np.float64), sites)
    return np.asarray(corrected).T.astype(np.float32)


def _oauthc_target(X: np.ndarray, sites: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Location/scale align OAUTHC-prospective rows to the reference distribution.

    Only OAUTHC rows are transformed; every other site is left untouched (targeted,
    not global). x' = (x - mu_o)/sd_o * sd_ref + mu_ref, per feature.
    """
    oauthc_mask = sites == OAUTHC_SITE
    if oauthc_mask.sum() == 0 or ref_mask.sum() < 2:
        return X.astype(np.float32)
    mu_o = X[oauthc_mask].mean(0)
    sd_o = X[oauthc_mask].std(0)
    mu_r = X[ref_mask].mean(0)
    sd_r = X[ref_mask].std(0)
    out = X.astype(np.float32).copy()
    out[oauthc_mask] = ((X[oauthc_mask] - mu_o) / (sd_o + EPS) * sd_r + mu_r).astype(np.float32)
    return out


def _apply_method(method: str, X: np.ndarray, sites: np.ndarray) -> np.ndarray:
    if method == "raw":
        return X.astype(np.float32)
    if method == "combat_global":
        return _combat_global(X, sites)
    if method == "oauthc_target_retrooau":
        return _oauthc_target(X, sites, sites == RETRO_OAU)
    if method == "oauthc_target_pooled":
        return _oauthc_target(X, sites, sites != OAUTHC_SITE)
    raise ValueError(f"unknown batch-correction method {method!r}")


def _residual_batch_auroc(X: np.ndarray, sites: np.ndarray) -> float:
    """Site-pred AUROC (OAUTHC-prospective vs retro-OAU) from corrected features,
    5-fold slide-level LR. Lower = better batch removal (0.5 = indistinguishable)."""
    mask = np.isin(sites, [OAUTHC_SITE, RETRO_OAU])
    if mask.sum() < 10:
        return float("nan")
    Xs = X[mask]
    ys = (sites[mask] == OAUTHC_SITE).astype(int)
    if len(np.unique(ys)) != 2:
        return float("nan")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    # group by nothing meaningful across the 2 sites → use slide index groups so
    # StratifiedGroupKFold behaves like stratified k-fold at slide resolution.
    groups = np.arange(len(ys))
    oof = np.full(len(ys), np.nan, dtype=np.float32)
    for tr, te in cv.split(Xs, ys, groups=groups):
        clf = clone(_lr_pipeline()).fit(Xs[tr], ys[tr])
        oof[te] = clf.predict_proba(Xs[te])[:, 1]
    return float(roc_auc_score(ys, oof))


def _eval_method(
    method: str, X_raw: np.ndarray, base: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    sites = base["site"].to_numpy()
    Xc = _apply_method(method, X_raw, sites)
    oof = _oof_full(Xc, base["y"].to_numpy(), base["patient_id"].to_numpy())
    slide_df = pd.DataFrame({
        "slide_id": base["slide_id"], "patient_id": base["patient_id"],
        "site": base["site"], "y": base["y"], "p_msih": oof,
    })
    pat = _patient_max_sqrtn(slide_df)
    oauthc = pat[pat["site"] == OAUTHC_SITE]
    oauthc_auroc = (
        float(roc_auc_score(oauthc["y"], oauthc["score"]))
        if oauthc["y"].nunique() == 2 else float("nan")
    )
    row = {
        "method": method,
        "overall_auroc": _patient_auroc(slide_df),
        "oauthc_auroc": oauthc_auroc,
        "oauthc_n": int(len(oauthc)),
        "residual_batch_auroc": _residual_batch_auroc(Xc, sites),
    }
    return row, slide_df


class BatchCorrectedProbe(Scorer):
    name = "batch_corrected_probe"
    description = (
        "LR probe on frozen CONCH-TITAN after an OAUTHC-targeted feature-space "
        "batch correction, chosen by OAUTHC held-out AUROC. Sweeps ComBat (global) "
        "vs one-site OAUTHC->reference location/scale alignment (retro-OAU / pooled) "
        "against the raw baseline; keeps the winner. Corrections are unsupervised "
        "w.r.t. MSI and fit transductively on the scored cohort. Frozen features, "
        "no external data."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "OOF p(MSI-H), best OAUTHC-targeted batch correction", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        methods: tuple[str, ...] = METHODS,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
        base = _load_embedding(RAW_EMB, cl, clean_slide_ids)
        if base is None:
            raise RuntimeError(f"{self.name}: raw embedding {RAW_EMB} unavailable")
        X_raw = np.load(EMB_ROOT / RAW_EMB / "embeddings.npy")[base["_row"].to_numpy()]

        rows, slide_dfs = [], {}
        for m in methods:
            row, sdf = _eval_method(m, X_raw, base)
            rows.append(row)
            slide_dfs[m] = sdf

        sweep = pd.DataFrame(rows)
        # Winner = highest OAUTHC AUROC (tie-break: overall AUROC).
        sweep = sweep.sort_values(
            ["oauthc_auroc", "overall_auroc"], ascending=False
        ).reset_index(drop=True)
        best_method = sweep.iloc[0]["method"]
        best_slide_df = slide_dfs[best_method]

        self._last = {"best_method": best_method, "sweep": sweep}

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            best_slide_df.to_csv(self.score_path, index=False)
            sweep.to_csv(self.score_path.parent / "sweep.csv", index=False)

        return best_slide_df


register("batch_corrected_probe", BatchCorrectedProbe)


def _write_metrics(scorer: "BatchCorrectedProbe", slide_df: pd.DataFrame) -> dict:
    """Full metric block for the winning method, OAUTHC operating point FIRST."""
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
    sweep = scorer._last["sweep"]
    metrics = {
        "scorer": scorer.name,
        "best_method": scorer._last["best_method"],
        "sweep": sweep.to_dict("records"),
        # ---- OAUTHC-first ----
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
    out = BatchCorrectedProbe().score_path.parent / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
