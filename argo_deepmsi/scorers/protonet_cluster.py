"""S2 — ProtoNet few-shot on cluster-aggregated CONCH tumor-tile embeddings.

The FM-MSI benchmark recipe (Computerized Medical Imaging and Graphics 2025, PII
S0895611125001892): CONCH patch embeddings are clustered into groups, each group's
embedding averaged, and the group means concatenated into a WSI embedding; a
few-shot ProtoNet head then classifies WSIs by distance to class prototypes.

Here the WSI cluster embeddings are built on TUMOR-ONLY tiles (Q3 filter) by
``scripts/build_protonet_features.py`` (global unsupervised k-means — no label
leakage). This scorer runs the label-aware ProtoNet head strictly train-only per
patient-grouped CV fold: prototypes are the per-class mean of the training WSI
embeddings (in a per-fold StandardScaler space), and each test WSI's p(MSI-H) is
the softmax over negative squared distances to the two prototypes. The softmax
temperature is monotonic, so AUROC and sensitivity-quantile thresholds are
temperature-invariant.

Reports a few-shot curve K ∈ {1,2,4,8,16,all}: K patients per class per fold form
the prototypes (averaged over random draws). Frozen features only.

Resolution: slide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register

FEAT_DIR = Path("results/scorers/protonet_cluster")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42
N_SPLITS = 5
N_REPEAT = 10
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")


def _load_features(clean_slide_ids: set[str] | None) -> tuple[np.ndarray, pd.DataFrame] | None:
    fnpy = FEAT_DIR / "cluster_features.npy"
    fmeta = FEAT_DIR / "cluster_metadata.csv"
    if not fnpy.exists() or not fmeta.exists():
        return None
    X = np.load(fnpy)
    meta = pd.read_csv(fmeta)
    cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    meta = meta.copy()
    meta["_row"] = np.arange(len(meta))
    meta = meta.merge(cl[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")
    if clean_slide_ids is not None:
        meta = meta[meta["slide_id"].isin(clean_slide_ids)]
    meta = meta.drop_duplicates("slide_id").reset_index(drop=True)
    if len(meta) == 0:
        return None
    return X[meta["_row"].to_numpy()], meta


def _proto_scores(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """ProtoNet p(MSI-H) for test rows: softmax over -||x-c_k||^2 / D."""
    c1 = Xtr[ytr == 1].mean(axis=0)
    c0 = Xtr[ytr == 0].mean(axis=0)
    d = Xte.shape[1]
    d1 = ((Xte - c1) ** 2).sum(axis=1) / d
    d0 = ((Xte - c0) ** 2).sum(axis=1) / d
    m = np.maximum(-d1, -d0)
    e1, e0 = np.exp(-d1 - m), np.exp(-d0 - m)
    return (e1 / (e1 + e0)).astype(np.float32)


def _patient_max_sqrtn(slide_df: pd.DataFrame, score_col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        rows.append({
            "patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
            "n_slides": int(len(g)), "score": float(g[score_col].max() / np.sqrt(len(g))),
        })
    return pd.DataFrame(rows)


def _patient_auroc(slide_df: pd.DataFrame) -> float:
    pat = _patient_max_sqrtn(slide_df)
    if pat["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(pat["y"], pat["score"]))


def _oof_full(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    y = meta["y"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in cv.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        oof[te] = _proto_scores(sc.transform(X[tr]), y[tr], sc.transform(X[te]))
    return oof


def _fewshot_auroc(X: np.ndarray, meta: pd.DataFrame, k: int) -> float:
    y = meta["y"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aurocs = []
    for rep in range(N_REPEAT):
        rng = np.random.default_rng(SEED + rep)
        oof = np.full(len(y), np.nan, dtype=np.float32)
        ok = True
        for tr, te in cv.split(X, y, groups=groups):
            tr_pat = meta.iloc[tr].drop_duplicates("patient_id")
            pos = tr_pat[tr_pat["y"] == 1]["patient_id"].to_numpy()
            neg = tr_pat[tr_pat["y"] == 0]["patient_id"].to_numpy()
            if len(pos) < k or len(neg) < k:
                ok = False
                break
            pick = set(rng.choice(pos, k, replace=False)) | set(rng.choice(neg, k, replace=False))
            sel = meta.iloc[tr]["patient_id"].isin(pick).to_numpy()
            tr_sel = tr[sel]
            if len(np.unique(y[tr_sel])) < 2:
                ok = False
                break
            sc = StandardScaler().fit(X[tr_sel])
            oof[te] = _proto_scores(sc.transform(X[tr_sel]), y[tr_sel], sc.transform(X[te]))
        if not ok:
            continue
        sd = meta.assign(p_msih=oof).dropna(subset=["p_msih"])
        aurocs.append(_patient_auroc(sd))
    return float(np.nanmean(aurocs)) if aurocs else float("nan")


def _fewshot_curve(X: np.ndarray, meta: pd.DataFrame) -> list[dict]:
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _oof_full(X, meta)
            sd = meta.assign(p_msih=oof)
            curve.append({
                "K": "all",
                "n_train_per_class": int(min((meta["y"] == 1).sum(), (meta["y"] == 0).sum())),
                "patient_auroc": _patient_auroc(sd),
            })
        else:
            curve.append({
                "K": int(k), "n_train_per_class": int(k),
                "patient_auroc": _fewshot_auroc(X, meta, int(k)),
            })
    return curve


class ProtoNetCluster(Scorer):
    name = "protonet_cluster"
    description = (
        "Few-shot ProtoNet on cluster-aggregated CONCH tumor-tile embeddings "
        "(FM-MSI benchmark recipe). WSI embeddings = concatenated per-group means "
        "of a global k-means over tumor-only CONCH tiles; ProtoNet prototypes are "
        "per-class training means, scored by softmax over negative squared distance. "
        "Reports a few-shot curve K∈{1,2,4,8,16,all}. Frozen features only."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "ProtoNet full-shot OOF p(MSI-H) on cluster-agg CONCH", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        full_curve: bool = False,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        loaded = _load_features(clean_slide_ids)
        if loaded is None:
            raise RuntimeError(
                f"{self.name}: cluster features missing under {FEAT_DIR} — "
                "run scripts/build_protonet_features.py first"
            )
        X, meta = loaded
        oof = _oof_full(X, meta)
        slide_df = pd.DataFrame({
            "slide_id": meta["slide_id"], "patient_id": meta["patient_id"],
            "site": meta["site"], "y": meta["y"], "p_msih": oof,
        })
        self._last = {"patient_auroc": _patient_auroc(slide_df)}
        if full_curve:
            self._last["curve"] = _fewshot_curve(X, meta)

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["curve"]).to_csv(
                    self.score_path.parent / "few_shot_curve.csv", index=False)

        return slide_df


register("protonet_cluster", ProtoNetCluster)


def _write_metrics(scorer: "ProtoNetCluster", slide_df: pd.DataFrame) -> dict:
    """Assemble + write the FULL metric block on the clean cohort."""
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc

    pat = _patient_max_sqrtn(slide_df)
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
    by_bagsize = {}
    for bucket, sub in pat.groupby("_bucket", observed=True):
        by_bagsize[str(bucket)] = {
            "n": int(len(sub)), "auroc": _safe_auroc(sub["y"].to_numpy(), sub["score"].to_numpy())}

    info = json.loads((FEAT_DIR / "cluster_info.json").read_text()) \
        if (FEAT_DIR / "cluster_info.json").exists() else {}
    metrics = {
        "scorer": scorer.name,
        "recipe": info.get("recipe"),
        "n_clusters": info.get("n_clusters"),
        "wsi_dim": info.get("wsi_dim"),
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
        "few_shot_curve": scorer._last.get("curve", []),
        "screening_clean": block,
    }
    out = ProtoNetCluster().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
