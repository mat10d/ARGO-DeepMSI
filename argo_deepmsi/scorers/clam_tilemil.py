"""S4 — attention-MIL (ABMIL) on tumor-only CONCH tile bags, with multi-fidelity fusion.

Base MIL is gated-attention MIL (Ilse et al. 2018): a small projection + gated
attention pools each slide's tumor-only CONCH tiles into a bag embedding, and a linear
head predicts MSI-H. On frozen tile features this is a tiny model — trained on CPU.

Variance reduction is the **Multi-Fidelity Model Fusion** of Mammadov et al. (MICCAI
2025 / arXiv 2507.00292): MIL runs vary by 10–15 AUC points across weight init / batch
order / lr, so instead of a single run we train M models (different seeds) for a few
epochs on an inner-train split, select the top-k by inner-validation patient AUROC, and
average their predictions. This is applied inside every patient-grouped outer fold, so
the reported OOF is a fused, lower-variance estimate.

Bags are built by ``scripts/build_tilemil_bags.py`` (tumor-only CONCH tiles, ≤500/slide).
Few-shot curve K∈{1,2,4,8,16,all} trains the fusion on K patients per class.

Resolution: slide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register

BAG_DIR = Path("results/scorers/clam_tilemil")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42
N_SPLITS = 5
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")

# Multi-fidelity fusion hyperparameters. Kept lean: ABMIL on frozen features
# converges in a handful of epochs, and batch-1 training is Python-loop bound, so
# more epochs/models buy little beyond the variance-reduction fusion itself.
M_FULL, EPOCHS_FULL, TOPK_FULL = 3, 12, 2
M_FEW, EPOCHS_FEW, TOPK_FEW, REPEAT_FEW = 2, 10, 1, 2
HIDDEN, PROJ, LR, DROPOUT = 128, 192, 1e-3, 0.25


def _torch():
    import torch
    torch.manual_seed(SEED)
    # Tiny per-bag matmuls: over-subscribing intra-op threads (e.g. os.cpu_count()
    # on a 128-core shared node) makes every op pathologically slow. Cap small.
    try:
        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001
        pass
    return torch


def _load_bags(clean_slide_ids: set[str] | None):
    if not (BAG_DIR / "bags_concat.npy").exists() or not (BAG_DIR / "bags_index.csv").exists():
        return None
    bags = np.load(BAG_DIR / "bags_concat.npy").astype(np.float32)
    idx = pd.read_csv(BAG_DIR / "bags_index.csv")
    cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    idx = idx.merge(cl[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")
    if clean_slide_ids is not None:
        idx = idx[idx["slide_id"].isin(clean_slide_ids)]
    idx = idx.drop_duplicates("slide_id").reset_index(drop=True)
    if len(idx) == 0:
        return None
    bag_list = [bags[int(r.start):int(r.start) + int(r.length)] for r in idx.itertuples()]
    return bag_list, idx


def _make_model(dim: int, seed: int):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)

    class ABMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(dim, PROJ), nn.ReLU(), nn.Dropout(DROPOUT))
            self.attn_V = nn.Linear(PROJ, HIDDEN)
            self.attn_U = nn.Linear(PROJ, HIDDEN)
            self.attn_w = nn.Linear(HIDDEN, 1)
            self.head = nn.Linear(PROJ, 1)

        def forward(self, H):  # H: (n_tiles, dim)
            h = self.proj(H)
            a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))  # (n,1)
            a = torch.softmax(a, dim=0)
            z = (a * h).sum(0, keepdim=True)  # (1, PROJ)
            return self.head(z).squeeze(1)  # (1,)

    return ABMIL()


def _train_one(bag_list, y, tr_idx, dim, seed, epochs, pos_weight):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)
    model = _make_model(dim, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    order = list(tr_idx)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            opt.zero_grad()
            H = torch.from_numpy(bag_list[i])
            logit = model(H)
            loss = lossf(logit, torch.tensor([float(y[i])]))
            loss.backward()
            opt.step()
    model.eval()
    return model


def _predict(model, bag_list, idxs) -> np.ndarray:
    torch = _torch()
    out = np.zeros(len(idxs), dtype=np.float32)
    with torch.no_grad():
        for j, i in enumerate(idxs):
            out[j] = torch.sigmoid(model(torch.from_numpy(bag_list[i]))).item()
    return out


def _patient_max_sqrtn(df: pd.DataFrame, score_col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("patient_id"):
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
                     "n_slides": int(len(g)), "score": float(g[score_col].max() / np.sqrt(len(g)))})
    return pd.DataFrame(rows)


def _patient_auroc_from_scores(idx: pd.DataFrame, scores: np.ndarray, rows) -> float:
    sd = idx.iloc[rows].assign(p_msih=scores)
    pat = _patient_max_sqrtn(sd)
    if pat["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(pat["y"], pat["score"]))


def _fused_oof(bag_list, idx, support_fn, m, epochs, topk) -> np.ndarray:
    """Multi-fidelity fused OOF over patient-grouped folds."""
    y = idx["y"].to_numpy()
    groups = idx["patient_id"].to_numpy()
    dim = bag_list[0].shape[1]
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    oof = np.full(len(y), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups)):
        sup = tr if support_fn is None else support_fn(tr)
        if sup is None or len(np.unique(y[sup])) < 2:
            continue
        # inner train/val split (patient-grouped) on the support set
        inner = StratifiedGroupKFold(n_splits=min(4, max(2, len(np.unique(groups[sup])) // 2)),
                                     shuffle=True, random_state=SEED)
        gi = groups[sup]
        try:
            itr_rel, ival_rel = next(inner.split(np.zeros(len(sup)), y[sup], groups=gi))
        except ValueError:
            itr_rel, ival_rel = np.arange(len(sup)), np.arange(len(sup))
        itr, ival = sup[itr_rel], sup[ival_rel]
        if len(np.unique(y[itr])) < 2:
            itr = sup
        cands = []
        for mm in range(m):
            model = _train_one(bag_list, y, itr, dim, SEED + 100 * fold + mm, epochs, pos_weight)
            val_scores = _predict(model, bag_list, ival)
            val_auroc = _patient_auroc_from_scores(idx, val_scores, ival)
            cands.append((val_auroc if not np.isnan(val_auroc) else -1.0, model))
        cands.sort(key=lambda t: t[0], reverse=True)
        chosen = [mdl for _, mdl in cands[:topk]] or [cands[0][1]]
        preds = np.mean([_predict(mdl, bag_list, te) for mdl in chosen], axis=0)
        oof[te] = preds
    return oof


def _fewshot_curve(bag_list, idx) -> list[dict]:
    y = idx["y"].to_numpy()
    groups = idx["patient_id"].to_numpy()
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _fused_oof(bag_list, idx, None, M_FULL, EPOCHS_FULL, TOPK_FULL)
            sd = idx.assign(p_msih=oof).dropna(subset=["p_msih"])
            pat = _patient_max_sqrtn(sd)
            curve.append({"K": "all", "n_train_per_class": int(min((y == 1).sum(), (y == 0).sum())),
                          "patient_auroc": float(roc_auc_score(pat["y"], pat["score"]))
                          if pat["y"].nunique() == 2 else float("nan")})
        else:
            aurocs = []
            for rep in range(REPEAT_FEW):
                rng = np.random.default_rng(SEED + rep)

                def pick(tr, _rng=rng, _k=int(k)):
                    tr_pat = idx.iloc[tr].drop_duplicates("patient_id")
                    pos = tr_pat[tr_pat["y"] == 1]["patient_id"].to_numpy()
                    neg = tr_pat[tr_pat["y"] == 0]["patient_id"].to_numpy()
                    if len(pos) < _k or len(neg) < _k:
                        return None
                    keep = set(_rng.choice(pos, _k, replace=False)) | set(_rng.choice(neg, _k, replace=False))
                    return tr[idx.iloc[tr]["patient_id"].isin(keep).to_numpy()]

                oof = _fused_oof(bag_list, idx, pick, M_FEW, EPOCHS_FEW, TOPK_FEW)
                sd = idx.assign(p_msih=oof).dropna(subset=["p_msih"])
                pat = _patient_max_sqrtn(sd)
                if pat["y"].nunique() == 2:
                    aurocs.append(float(roc_auc_score(pat["y"], pat["score"])))
            curve.append({"K": int(k), "n_train_per_class": int(k),
                          "patient_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan")})
    return curve


class CLAMTileMIL(Scorer):
    name = "clam_tilemil"
    description = (
        "Gated attention-MIL (ABMIL) on tumor-only CONCH tile bags with Multi-Fidelity "
        "Model Fusion (Mammadov et al., MICCAI 2025) for variance reduction: per outer "
        "fold, M models are trained on an inner-train split, the top-k by inner-val "
        "patient AUROC are averaged. Few-shot curve K∈{1,2,4,8,16,all}. Frozen features."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "fused ABMIL OOF p(MSI-H) on tumor CONCH bags", primary=True),
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
        # Cache-first: training the fused ABMIL is expensive (~15 min), so the
        # leaderboard re-race reads the cached OOF (like vl_text_cosine) rather than
        # retraining twice per pass. Only the runner (retrain=True) trains.
        if not retrain and self.score_path is not None and self.score_path.exists():
            df = pd.read_csv(self.score_path)
            if clean_slide_ids is not None:
                df = df[df["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)
            return df

        loaded = _load_bags(clean_slide_ids)
        if loaded is None:
            raise RuntimeError(
                f"{self.name}: tile bags missing under {BAG_DIR} — "
                "run scripts/build_tilemil_bags.py first"
            )
        bag_list, idx = loaded
        oof = _fused_oof(bag_list, idx, None, M_FULL, EPOCHS_FULL, TOPK_FULL)
        slide_df = pd.DataFrame({
            "slide_id": idx["slide_id"], "patient_id": idx["patient_id"],
            "site": idx["site"], "y": idx["y"], "p_msih": oof,
        }).dropna(subset=["p_msih"]).reset_index(drop=True)
        self._last = {}
        if full_curve:
            self._last["curve"] = _fewshot_curve(bag_list, idx)

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["curve"]).to_csv(
                    self.score_path.parent / "few_shot_curve.csv", index=False)
        return slide_df


register("clam_tilemil", CLAMTileMIL)


def _write_metrics(scorer: "CLAMTileMIL", slide_df: pd.DataFrame) -> dict:
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc

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
        }

    pat = pat.assign(_bucket=pd.cut(pat["n_slides"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]))
    by_bagsize = {str(b): {"n": int(len(sub)), "auroc": _safe_auroc(sub["y"].to_numpy(), sub["score"].to_numpy())}
                  for b, sub in pat.groupby("_bucket", observed=True)}

    info = json.loads((BAG_DIR / "bags_info.json").read_text()) if (BAG_DIR / "bags_info.json").exists() else {}
    metrics = {
        "scorer": scorer.name, "backbone": "ABMIL+multifidelity-fusion",
        "tile_model": info.get("model"), "max_tiles": info.get("max_tiles"),
        "fusion": {"m_full": M_FULL, "epochs_full": EPOCHS_FULL, "topk_full": TOPK_FULL},
        "n_slides": int(len(slide_df)), "n_patients": int(len(pat)),
        "prevalence_patient": float(pat["y"].mean()),
        "auroc": _safe_auroc(y, s), "auprc": _safe_auprc(y, s),
        "sensitivity": block["op_sens95"]["sensitivity"],
        "spec_at_sens90": block["spec_at_sens90"],
        "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"],
        "npv_at_sens95": block["npv_at_sens95"],
        "operating_threshold": block["op_sens95"]["threshold"],
        "by_site": by_site, "by_bagsize": by_bagsize,
        "few_shot_curve": scorer._last.get("curve", []),
        "screening_clean": block,
    }
    out = CLAMTileMIL().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
