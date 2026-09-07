"""A3 — attention-MIL with OAUTHC-specific few-shot adaptation.

S4 (clam_tilemil) was the ONLY learned recipe that beat the champion on OAUTHC
(0.629 vs 0.579), but its few-shot curve subsampled K patients per class from the
WHOLE pool — a GLOBAL few-shot that (correctly) failed at low N. This scorer gives
few-shot the OAUTHC-SPECIFIC test it never got: train a good-site ABMIL head, then
ADAPT it on K held-out OAUTHC patients and evaluate on the remaining OAUTHC patients.

Two evaluations (both reuse S4's frozen tumor-CONCH bags + ABMIL):

1. **Full-cohort adapted OOF (primary score).** Standard 5-fold patient-grouped CV
   with a base ABMIL ensemble trained on each fold's train patients; for OAUTHC test
   patients the fold's base models are additionally fine-tuned on that fold's OAUTHC
   train patients before predicting. Non-OAUTHC test uses the base. No leakage —
   adaptation only ever sees train-fold patients.

2. **OAUTHC K-shot adaptation curve (headline).** A base ensemble is trained ONCE on
   ALL non-OAUTHC (good-site) patients; then 5-fold patient-grouped CV over OAUTHC only:
   for each OAUTHC fold, the base is adapted on K OAUTHC support patients (per class,
   averaged over draws) and evaluated on the OAUTHC held-out fold. K=0 is zero-shot
   transfer of the good-site head. Reports OAUTHC patient AUROC vs K.

Frozen features; adaptation fits on our own OAUTHC patients (no external data).
Resolution: slide.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..eval.metrics import _safe_auprc, _safe_auroc
from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .clam_tilemil import (
    BAG_DIR,
    _load_bags,
    _patient_max_sqrtn,
    _predict,
    _torch,
    _train_one,
)
from .harmony_probe import OAUTHC_SITE
from .registry import register

SEED = 42
N_SPLITS = 5
M_BASE = 3           # base ensemble size
EPOCHS_BASE = 12
ADAPT_EPOCHS = 8
ADAPT_LR = 5e-4      # half the base lr — gentle fine-tune
OAUTHC_K: tuple[int | str, ...] = (0, 2, 4, 8, 16, "all")
N_DRAW = 3           # random OAUTHC support draws averaged per K


def _finetune(model, bag_list, y, idxs, epochs, lr, pos_weight, seed):
    """Continue-train an existing ABMIL on the support bags (in place)."""
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    order = list(idxs)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            opt.zero_grad()
            logit = model(torch.from_numpy(bag_list[i]))
            loss = lossf(logit, torch.tensor([float(y[i])]))
            loss.backward()
            opt.step()
    model.eval()
    return model


def _train_base_ensemble(bag_list, y, base_idx, dim, pos_weight, m=M_BASE, epochs=EPOCHS_BASE):
    return [
        _train_one(bag_list, y, base_idx, dim, SEED + 7 * k, epochs, pos_weight)
        for k in range(m)
    ]


def _ensemble_predict(models, bag_list, idxs) -> np.ndarray:
    return np.mean([_predict(mdl, bag_list, idxs) for mdl in models], axis=0)


def _adapt_and_predict(base_models, bag_list, y, support_idx, test_idx, pos_weight, seed):
    """Fine-tune a copy of each base model on support_idx, average predictions on test_idx."""
    preds = []
    for j, base in enumerate(base_models):
        mdl = copy.deepcopy(base)
        _finetune(mdl, bag_list, y, support_idx, ADAPT_EPOCHS, ADAPT_LR, pos_weight, seed + j)
        preds.append(_predict(mdl, bag_list, test_idx))
    return np.mean(preds, axis=0)


def _full_cohort_adapted_oof(bag_list, idx) -> np.ndarray:
    """Eval 1: base ensemble per fold; OAUTHC test additionally adapted on OAUTHC-train."""
    y = idx["y"].to_numpy()
    sites = idx["site"].to_numpy()
    groups = idx["patient_id"].to_numpy()
    dim = bag_list[0].shape[1]
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    oof = np.full(len(y), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        base = _train_base_ensemble(bag_list, y, tr, dim, pos_weight)
        te_oauthc = te[sites[te] == OAUTHC_SITE]
        te_other = te[sites[te] != OAUTHC_SITE]
        if len(te_other):
            oof[te_other] = _ensemble_predict(base, bag_list, te_other)
        if len(te_oauthc):
            oauthc_support = tr[sites[tr] == OAUTHC_SITE]
            if len(oauthc_support) >= 2 and len(np.unique(y[oauthc_support])) == 2:
                oof[te_oauthc] = _adapt_and_predict(
                    base, bag_list, y, oauthc_support, te_oauthc, pos_weight, SEED + 100 * fold
                )
            else:
                oof[te_oauthc] = _ensemble_predict(base, bag_list, te_oauthc)
    return oof


def _oauthc_kshot_curve(bag_list, idx) -> list[dict]:
    """Eval 2: good-site base trained once; adapt on K OAUTHC support, test OAUTHC held-out."""
    y = idx["y"].to_numpy()
    sites = idx["site"].to_numpy()
    groups = idx["patient_id"].to_numpy()
    dim = bag_list[0].shape[1]
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))

    base_idx = np.where(sites != OAUTHC_SITE)[0]
    base = _train_base_ensemble(bag_list, y, base_idx, dim, pos_weight)

    oa = np.where(sites == OAUTHC_SITE)[0]
    oa_y, oa_groups = y[oa], groups[oa]
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def _auroc_from_oof(oof_oa):
        sd = idx.iloc[oa].assign(p_msih=oof_oa).dropna(subset=["p_msih"])
        if len(sd) == 0:
            return float("nan")
        pat = _patient_max_sqrtn(sd)
        if len(pat) == 0 or pat["y"].nunique() != 2:
            return float("nan")
        return float(roc_auc_score(pat["y"], pat["score"]))

    curve = []
    for k in OAUTHC_K:
        draw_aurocs = []
        n_draws = 1 if k in (0, "all") else N_DRAW
        for d in range(n_draws):
            rng = np.random.default_rng(SEED + d)
            oof_oa = np.full(len(oa), np.nan, dtype=np.float32)
            for fold, (tr_rel, te_rel) in enumerate(cv.split(np.zeros(len(oa)), oa_y, groups=oa_groups)):
                te_abs = oa[te_rel]
                if k == 0:
                    oof_oa[te_rel] = _ensemble_predict(base, bag_list, te_abs)
                    continue
                tr_abs = oa[tr_rel]
                if k == "all":
                    support = tr_abs
                else:
                    tr_pat = idx.iloc[tr_abs].drop_duplicates("patient_id")
                    pos = tr_pat[tr_pat["y"] == 1]["patient_id"].to_numpy()
                    neg = tr_pat[tr_pat["y"] == 0]["patient_id"].to_numpy()
                    if len(pos) < k or len(neg) < k:
                        continue
                    keep = set(rng.choice(pos, k, replace=False)) | set(rng.choice(neg, k, replace=False))
                    support = tr_abs[idx.iloc[tr_abs]["patient_id"].isin(keep).to_numpy()]
                if len(np.unique(y[support])) < 2:
                    oof_oa[te_rel] = _ensemble_predict(base, bag_list, te_abs)
                    continue
                oof_oa[te_rel] = _adapt_and_predict(
                    base, bag_list, y, support, te_abs, pos_weight, SEED + 100 * fold + d
                )
            au = _auroc_from_oof(oof_oa)
            if not np.isnan(au):
                draw_aurocs.append(au)
        curve.append({
            "K": ("all" if k == "all" else int(k)),
            "oauthc_patient_auroc": float(np.mean(draw_aurocs)) if draw_aurocs else float("nan"),
            "n_draws": len(draw_aurocs),
        })
    return curve


class TileMILOAUTHCAdapt(Scorer):
    name = "tilemil_oauthc_adapt"
    description = (
        "Attention-MIL (ABMIL) on tumor CONCH bags with OAUTHC-specific few-shot "
        "adaptation: a good-site-trained head is fine-tuned on held-out OAUTHC support "
        "patients and evaluated on the rest of OAUTHC. Primary score is a full-cohort "
        "5-fold OOF where OAUTHC test patients get within-fold OAUTHC adaptation. Reports "
        "the OAUTHC K-shot adaptation curve. Frozen features; no external data."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "fold OOF p(MSI-H); OAUTHC adapted on OAUTHC support", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        full_curve: bool = False,
        write_outputs: bool = True,
        retrain: bool = False,
        **_,
    ) -> pd.DataFrame:
        # Cache-first (like clam_tilemil): the leaderboard re-race reads the cached OOF
        # instead of retraining the MIL twice per pass. Only the runner (retrain=True) trains.
        if not retrain and self.score_path is not None and self.score_path.exists():
            df = pd.read_csv(self.score_path)
            if clean_slide_ids is not None:
                df = df[df["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)
            return df

        loaded = _load_bags(clean_slide_ids)
        if loaded is None:
            raise RuntimeError(f"{self.name}: tile bags missing under {BAG_DIR} — run build_tilemil_bags.py")
        bag_list, idx = loaded
        oof = _full_cohort_adapted_oof(bag_list, idx)
        slide_df = pd.DataFrame({
            "slide_id": idx["slide_id"], "patient_id": idx["patient_id"],
            "site": idx["site"], "y": idx["y"], "p_msih": oof,
        }).dropna(subset=["p_msih"]).reset_index(drop=True)

        self._last = {}
        if full_curve:
            self._last["oauthc_curve"] = _oauthc_kshot_curve(bag_list, idx)

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["oauthc_curve"]).to_csv(
                    self.score_path.parent / "oauthc_kshot_curve.csv", index=False)
        return slide_df


register("tilemil_oauthc_adapt", TileMILOAUTHCAdapt)


def _write_metrics(scorer: "TileMILOAUTHCAdapt", slide_df: pd.DataFrame) -> dict:
    pat = _patient_max_sqrtn(slide_df, "p_msih")
    y, s = pat["y"].to_numpy(), pat["score"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))

    by_site = {}
    for site, sub in pat.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["score"].to_numpy()
        by_site[str(site)] = {
            "n": int(len(sub)), "prevalence": float(sub["y"].mean()),
            "auroc": _safe_auroc(yy, ss),
            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
            if sub["y"].nunique() == 2 else float("nan"),
            "spec_at_sens96": screening_block(yy, ss, (0.96,))["spec_at_sens96"]
            if sub["y"].nunique() == 2 else float("nan"),
        }

    bins = [0, 1, 2, 4, np.inf]
    labels = ["1", "2", "3-4", "5+"]
    pat = pat.assign(_bucket=pd.cut(pat["n_slides"], bins=bins, labels=labels))
    by_bagsize = {
        str(b): {"n": int(len(sub)), "auroc": _safe_auroc(sub["y"].to_numpy(), sub["score"].to_numpy())}
        for b, sub in pat.groupby("_bucket", observed=True)
    }

    oauthc = by_site.get(OAUTHC_SITE, {})
    curve = scorer._last.get("oauthc_curve", [])
    metrics = {
        "scorer": scorer.name,
        "oauthc_auroc": oauthc.get("auroc", float("nan")),
        "oauthc_spec_at_sens95": oauthc.get("spec_at_sens95", float("nan")),
        "oauthc_spec_at_sens96": oauthc.get("spec_at_sens96", float("nan")),
        "oauthc_n_patients": oauthc.get("n", 0),
        "oauthc_prevalence": oauthc.get("prevalence", float("nan")),
        "oauthc_kshot_curve": curve,
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
    out = TileMILOAUTHCAdapt().score_path.parent / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
