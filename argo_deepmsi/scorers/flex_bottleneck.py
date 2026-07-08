"""R2 — FLEX-style knowledge-guided information bottleneck for site-invariance.

FLEX (Feature-Level Enhancement for Cross-domain generalization; Nat Commun 2025,
s41467-025-66300-y) learns a task-specific information bottleneck guided by textual
concepts from a pathology FM's text encoder. Because text concepts originate in the text
domain, they are free of site-specific signatures and demographic bias present in images;
aligning image features to them disentangles robust pathological signal (MSI) from site
signal — the published fix for our diagnosed OAUTHC domain-shift failure (see R1).

This scorer implements a compact FLEX on frozen TITAN (CONCH v1.5) slide features:

  encoder φ : 768 → 128 bottleneck z
  MSI head  : z → p(MSI-H)                              (retain task signal)
  site head : GRL(z) → site   (gradient-reversal)       (remove site signal — the bottleneck)
  text anchor: g(z) → 768, cosine-aligned to the class's site-free text prototype
               (MSI / MSS concept from vl_text_cosine)  (knowledge guidance)

Loss = BCE(MSI) + λ_site · CE(site | GRL) + λ_text · (1 − cos(g(z), text_proto[y])).
Patient-grouped CV; OOF aggregated to patients (max/√n). Reports per-site AUROC and the
delta vs the champion, the headline for a site-invariance method. Frozen features only.

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

TITAN_DIR = Path("results/embeddings/conch_v1.5_titan")
PROTO_NPY = Path("results/scorers/vl_text_cosine/text_embeddings.npy")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42
N_SPLITS = 5
EPOCHS = 200
Z_DIM = 128
LR = 1e-3
LAMBDA_SITE = 0.5
LAMBDA_TEXT = 0.3
GRL_LAMBDA = 1.0
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")


def _torch():
    import torch
    torch.manual_seed(SEED)
    try:
        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001
        pass
    return torch


def _l2(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def _load(clean_slide_ids: set[str] | None):
    if not (TITAN_DIR / "embeddings.npy").exists() or not PROTO_NPY.exists():
        return None
    X = _l2(np.load(TITAN_DIR / "embeddings.npy").astype(np.float32))
    meta = pd.read_csv(TITAN_DIR / "metadata.csv").copy()
    meta["_row"] = np.arange(len(meta))
    cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    meta = meta.merge(cl[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")
    if clean_slide_ids is not None:
        meta = meta[meta["slide_id"].isin(clean_slide_ids)]
    meta = meta.drop_duplicates("slide_id").reset_index(drop=True)
    if len(meta) == 0:
        return None
    X = X[meta["_row"].to_numpy()]
    protos = _l2(np.load(PROTO_NPY).astype(np.float32))  # (2,768): [MSI, MSS]
    sites = sorted(meta["site"].unique())
    site_code = {s: i for i, s in enumerate(sites)}
    meta = meta.assign(site_code=meta["site"].map(site_code))
    return X, meta, protos, len(sites)


def _make_model(dim: int, n_sites: int, seed: int):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = lambd
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lambd * g, None

    class FLEX(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(256, Z_DIM), nn.ReLU())
            self.msi = nn.Linear(Z_DIM, 1)
            self.site = nn.Sequential(nn.Linear(Z_DIM, 64), nn.ReLU(), nn.Linear(64, n_sites))
            self.text = nn.Linear(Z_DIM, dim)

        def forward(self, x, grl_lambda=GRL_LAMBDA):
            z = self.phi(x)
            return self.msi(z).squeeze(1), self.site(GRL.apply(z, grl_lambda)), self.text(z)

    return FLEX()


def _train(X, y, site, protos, tr, dim, n_sites, seed, epochs, pos_weight):
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = _make_model(dim, n_sites, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    ce = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X[tr])
    yt = torch.from_numpy(y[tr].astype(np.float32))
    st = torch.from_numpy(site[tr].astype(np.int64))
    proto_t = torch.from_numpy(protos)  # (2,768)
    target_text = proto_t[y[tr].astype(np.int64)]  # (n,768) each row = class text concept
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        msi_logit, site_logit, text_pred = model(Xt)
        loss = bce(msi_logit, yt) + LAMBDA_SITE * ce(site_logit, st)
        loss = loss + LAMBDA_TEXT * (1 - F.cosine_similarity(text_pred, target_text, dim=1)).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _predict(model, X, idxs) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        logit, _, _ = model(torch.from_numpy(X[idxs]))
        return torch.sigmoid(logit).numpy().astype(np.float32)


def _patient_max_sqrtn(df: pd.DataFrame, col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("patient_id"):
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
                     "n_slides": int(len(g)), "score": float(g[col].max() / np.sqrt(len(g)))})
    return pd.DataFrame(rows)


def _patient_auroc(df: pd.DataFrame) -> float:
    pat = _patient_max_sqrtn(df)
    return float(roc_auc_score(pat["y"], pat["score"])) if pat["y"].nunique() == 2 else float("nan")


def _oof(X, meta, protos, n_sites, support_fn=None) -> np.ndarray:
    y = meta["y"].to_numpy()
    site = meta["site_code"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    dim = X.shape[1]
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    oof = np.full(len(y), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
        sup = tr if support_fn is None else support_fn(tr)
        if sup is None or len(np.unique(y[sup])) < 2:
            continue
        model = _train(X, y, site, protos, sup, dim, n_sites, SEED + fold, EPOCHS, pos_weight)
        oof[te] = _predict(model, X, te)
    return oof


def _fewshot_curve(X, meta, protos, n_sites) -> list[dict]:
    y = meta["y"].to_numpy()
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _oof(X, meta, protos, n_sites, None)
            curve.append({"K": "all", "n_train_per_class": int(min((y == 1).sum(), (y == 0).sum())),
                          "patient_auroc": _patient_auroc(meta.assign(p_msih=oof).dropna(subset=["p_msih"]))})
        else:
            aurocs = []
            for rep in range(3):
                rng = np.random.default_rng(SEED + rep)

                def pick(tr, _rng=rng, _k=int(k)):
                    tp = meta.iloc[tr].drop_duplicates("patient_id")
                    pos = tp[tp["y"] == 1]["patient_id"].to_numpy()
                    neg = tp[tp["y"] == 0]["patient_id"].to_numpy()
                    if len(pos) < _k or len(neg) < _k:
                        return None
                    keep = set(_rng.choice(pos, _k, replace=False)) | set(_rng.choice(neg, _k, replace=False))
                    return tr[meta.iloc[tr]["patient_id"].isin(keep).to_numpy()]

                oof = _oof(X, meta, protos, n_sites, pick)
                sd = meta.assign(p_msih=oof).dropna(subset=["p_msih"])
                if sd["y"].nunique() == 2:
                    aurocs.append(_patient_auroc(sd))
            curve.append({"K": int(k), "n_train_per_class": int(k),
                          "patient_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan")})
    return curve


class FlexBottleneck(Scorer):
    name = "flex_bottleneck"
    description = (
        "FLEX-style knowledge-guided information bottleneck (Nat Commun 2025): a 768→128 "
        "bottleneck on TITAN slide features with an MSI head, a gradient-reversal site "
        "adversary (removes site signal), and cosine alignment to site-free MSI/MSS text "
        "concepts. Targets cross-site (OAUTHC) generalization. Patient-grouped CV; few-shot "
        "curve K∈{1,2,4,8,16,all}. Frozen features."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "FLEX bottleneck OOF p(MSI-H)", primary=True),
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
        if not retrain and self.score_path is not None and self.score_path.exists():
            df = pd.read_csv(self.score_path)
            if clean_slide_ids is not None:
                df = df[df["slide_id"].isin(clean_slide_ids)].reset_index(drop=True)
            return df
        loaded = _load(clean_slide_ids)
        if loaded is None:
            raise RuntimeError(f"{self.name}: TITAN embeddings / text prototypes missing")
        X, meta, protos, n_sites = loaded
        oof = _oof(X, meta, protos, n_sites, None)
        slide_df = pd.DataFrame({
            "slide_id": meta["slide_id"], "patient_id": meta["patient_id"],
            "site": meta["site"], "y": meta["y"], "p_msih": oof,
        }).dropna(subset=["p_msih"]).reset_index(drop=True)
        self._last = {}
        if full_curve:
            self._last["curve"] = _fewshot_curve(X, meta, protos, n_sites)
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["curve"]).to_csv(
                    self.score_path.parent / "few_shot_curve.csv", index=False)
        return slide_df


register("flex_bottleneck", FlexBottleneck)


def _write_metrics(scorer: "FlexBottleneck", slide_df: pd.DataFrame) -> dict:
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc

    pat = _patient_max_sqrtn(slide_df)
    y, s = pat["y"].to_numpy(), pat["score"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))

    # champion per-site AUROC for the delta (site-invariance headline)
    champ = {}
    ps = Path("results/comparison/per_site_clean.csv")
    if ps.exists():
        cdf = pd.read_csv(ps)
        cdf = cdf[cdf["scorer"] == "calibrated_pool"]
        champ = {str(r.site): float(r.auroc) for r in cdf.itertuples()}

    by_site = {}
    for site, sub in pat.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["score"].to_numpy()
        a = _safe_auroc(yy, ss)
        by_site[str(site)] = {
            "n": int(len(sub)), "prevalence": float(sub["y"].mean()), "auroc": a,
            "champion_auroc": champ.get(str(site)),
            "delta_vs_champion": (a - champ[str(site)]) if str(site) in champ and not np.isnan(a) else None,
            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
            if sub["y"].nunique() == 2 else float("nan"),
        }

    pat = pat.assign(_b=pd.cut(pat["n_slides"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]))
    by_bagsize = {str(b): {"n": int(len(g)), "auroc": _safe_auroc(g["y"].to_numpy(), g["score"].to_numpy())}
                  for b, g in pat.groupby("_b", observed=True)}

    metrics = {
        "scorer": scorer.name, "method": "FLEX bottleneck (GRL site-adversary + text-concept anchor)",
        "lambda_site": LAMBDA_SITE, "lambda_text": LAMBDA_TEXT, "z_dim": Z_DIM,
        "n_slides": int(len(slide_df)), "n_patients": int(len(pat)),
        "prevalence_patient": float(pat["y"].mean()),
        "auroc": _safe_auroc(y, s), "auprc": _safe_auprc(y, s),
        "sensitivity": block["op_sens95"]["sensitivity"],
        "spec_at_sens90": block["spec_at_sens90"], "spec_at_sens95": block["spec_at_sens95"],
        "spec_at_sens96": block["spec_at_sens96"], "npv_at_sens95": block["npv_at_sens95"],
        "operating_threshold": block["op_sens95"]["threshold"],
        "by_site": by_site, "by_bagsize": by_bagsize,
        "few_shot_curve": scorer._last.get("curve", []), "screening_clean": block,
    }
    out = FlexBottleneck().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
