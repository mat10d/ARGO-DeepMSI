"""R1 — 5-crop + stain-augmentation test-time robustness on the S4 attention-MIL base.

MSIntuit reports an inter-scanner Cohen's κ = 0.82: its MSI call is stable when the same
slide is imaged on a different scanner. R1 measures the analogous stability of our best
tile-level scorer (S4 `clam_tilemil`, the most site-uniform) under test-time augmentation.

IMPORTANT — feature-space proxy. True H&E stain augmentation perturbs raw pixels and
re-extracts foundation-model features; on this cohort that is ~55 h of GPU re-extraction
(428 slides × conditions × an 11 h CONCH pass), which is infeasible in the autorun loop.
The pipeline retains only frozen CONCH tile features, so R1 applies augmentation in
feature space and documents it as an approximation:

  - "5-crop" TTA  → 5 tile-subset views per slide (each a 70% bootstrap of the slide's
                    tumor tiles), emulating spatial crop / field-of-view variation.
  - "stain" TTA   → per-view feature-space perturbation: a per-dimension log-normal gain
                    plus small Gaussian jitter, emulating the embedding shift a stain /
                    scanner change induces (calibrated to per-dimension feature std).

For every condition the trained S4 fold-model scores each held-out slide; slide scores are
aggregated to patients (max/√n) and binarized at the identity condition's sens-95 operating
threshold. inter_condition_kappa is the mean pairwise Cohen's κ across conditions — directly
comparable to MSIntuit's inter-scanner κ.

Output: results/scorers/clam_tilemil_stainaug/metrics.json  (full block + inter_condition_kappa)
CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from argo_deepmsi.eval.metrics import _safe_auprc, _safe_auroc
from argo_deepmsi.eval.screening import screening_block, threshold_at_sensitivity
from argo_deepmsi.scorers.clam_tilemil import (
    EPOCHS_FULL,
    M_FULL,
    N_SPLITS,
    SEED,
    TOPK_FULL,
    _load_bags,
    _patient_max_sqrtn,
    _torch,
    _train_one,
)

OUTDIR = Path("results/scorers/clam_tilemil_stainaug")
N_CROP = 5
CROP_FRAC = 0.70
STAIN_GAIN_SIGMA = 0.10   # log-normal per-dim gain
STAIN_JITTER = 0.05       # gaussian jitter × per-dim std


def _augment_bag(bag: np.ndarray, feat_std: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Return an augmented view of a bag. kind='identity' | 'cropstain'."""
    if kind == "identity":
        return bag
    n = len(bag)
    take = max(1, int(round(CROP_FRAC * n)))
    idx = rng.choice(n, take, replace=False)
    view = bag[idx].copy()
    gain = np.exp(rng.normal(0.0, STAIN_GAIN_SIGMA, size=(1, bag.shape[1]))).astype(np.float32)
    jitter = (rng.normal(0.0, STAIN_JITTER, size=view.shape) * feat_std[None, :]).astype(np.float32)
    return view * gain + jitter


def _predict_cond(model, bags, idxs, scaler, feat_std, kind, seed) -> np.ndarray:
    torch = _torch()
    rng = np.random.default_rng(seed)
    out = np.zeros(len(idxs), dtype=np.float32)
    with torch.no_grad():
        for j, i in enumerate(idxs):
            view = _augment_bag(bags[i], feat_std, kind, rng)
            out[j] = torch.sigmoid(model(torch.from_numpy(scaler.transform(view).astype(np.float32)))).item()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    a = p.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cc = pd.read_csv(a.clean_csv)
    clean_ids = {str(s) for s in cc.loc[cc["in_clean_set"] == 1, "slide_id"]}
    loaded = _load_bags(clean_ids)
    if loaded is None:
        raise RuntimeError("clam_tilemil bags missing — run scripts/build_tilemil_bags.py first")
    bag_list, idx = loaded
    y = idx["y"].to_numpy()
    groups = idx["patient_id"].to_numpy()
    dim = bag_list[0].shape[1]
    feat_std = np.concatenate(bag_list, axis=0).std(axis=0).astype(np.float32) + 1e-6
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))

    conditions = ["identity"] + [f"cropstain{v}" for v in range(N_CROP)]
    cond_slide = {c: np.full(len(y), np.nan, dtype=np.float32) for c in conditions}

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups)):
        inner = StratifiedGroupKFold(n_splits=min(4, max(2, len(np.unique(groups[tr])) // 2)),
                                     shuffle=True, random_state=SEED)
        itr_rel, ival_rel = next(inner.split(np.zeros(len(tr)), y[tr], groups=groups[tr]))
        itr = tr[itr_rel] if len(np.unique(y[tr[itr_rel]])) == 2 else tr
        scaler = StandardScaler().fit(np.vstack([bag_list[i] for i in itr]))
        # multi-fidelity: train M, keep top-k by inner-val identity AUROC
        cands = []
        for m in range(M_FULL):
            model = _train_one(bag_list, y, itr, dim, SEED + 100 * fold + m, EPOCHS_FULL, pos_weight)
            vpred = _predict_cond(model, bag_list, tr[ival_rel], scaler, feat_std, "identity", 0)
            vdf = idx.iloc[tr[ival_rel]].assign(p_msih=vpred)
            vpat = _patient_max_sqrtn(vdf)
            va = _safe_auroc(vpat["y"].to_numpy(), vpat["score"].to_numpy()) if vpat["y"].nunique() == 2 else -1
            cands.append((va, model))
        cands.sort(key=lambda t: t[0], reverse=True)
        chosen = [mdl for _, mdl in cands[:TOPK_FULL]] or [cands[0][1]]
        for ci, c in enumerate(conditions):
            preds = np.mean([_predict_cond(mdl, bag_list, te, scaler, feat_std, c, SEED + fold * 10 + ci)
                             for mdl in chosen], axis=0)
            cond_slide[c][te] = preds

    # aggregate each condition to patients; binarize at identity's sens95 threshold
    def patient(cond):
        sd = idx.assign(p_msih=cond_slide[cond]).dropna(subset=["p_msih"])
        return _patient_max_sqrtn(sd).sort_values("patient_id").reset_index(drop=True)

    pat_id = patient("identity")
    y_pat = pat_id["y"].to_numpy()
    thr = threshold_at_sensitivity(y_pat, pat_id["score"].to_numpy(), 0.95)
    cond_bin = {}
    cond_auroc = {}
    for c in conditions:
        pc = patient(c)
        cond_bin[c] = (pc["score"].to_numpy() >= thr).astype(int)
        cond_auroc[c] = _safe_auroc(pc["y"].to_numpy(), pc["score"].to_numpy())

    kappas = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            kappas.append(cohen_kappa_score(cond_bin[conditions[i]], cond_bin[conditions[j]]))
    inter_kappa = float(np.mean(kappas))

    # full block on the identity condition (should ≈ S4)
    s = pat_id["score"].to_numpy()
    block = screening_block(y_pat, s, (0.90, 0.95, 0.96, 0.98))
    by_site = {}
    for site, sub in pat_id.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["score"].to_numpy()
        by_site[str(site)] = {"n": int(len(sub)), "prevalence": float(sub["y"].mean()),
                              "auroc": _safe_auroc(yy, ss),
                              "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
                              if sub["y"].nunique() == 2 else float("nan")}
    pat_b = pat_id.assign(_b=pd.cut(pat_id["n_slides"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]))
    by_bagsize = {str(b): {"n": int(len(g)), "auroc": _safe_auroc(g["y"].to_numpy(), g["score"].to_numpy())}
                  for b, g in pat_b.groupby("_b", observed=True)}

    metrics = {
        "scorer": "clam_tilemil_stainaug", "base": "clam_tilemil",
        "method": "feature-space TTA (5 crop-subset views + per-dim stain gain/jitter)",
        "note": "feature-space proxy; image-level stain-aug re-extraction (~55h GPU) infeasible",
        "n_conditions": len(conditions), "conditions": conditions,
        "inter_condition_kappa": inter_kappa,
        "per_condition_auroc": {c: cond_auroc[c] for c in conditions},
        "n_patients": int(len(pat_id)), "prevalence_patient": float(y_pat.mean()),
        "auroc": _safe_auroc(y_pat, s), "auprc": _safe_auprc(y_pat, s),
        "sensitivity": block["op_sens95"]["sensitivity"],
        "spec_at_sens90": block["spec_at_sens90"], "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"], "npv_at_sens95": block["npv_at_sens95"],
        "operating_threshold": float(thr), "by_site": by_site, "by_bagsize": by_bagsize,
        "screening_clean": block,
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"clam_tilemil_stainaug: inter_condition_kappa={inter_kappa:.3f} "
          f"(base AUROC {metrics['auroc']:.3f}); per-condition AUROC "
          f"{[round(cond_auroc[c], 3) for c in conditions]}")


if __name__ == "__main__":
    main()
