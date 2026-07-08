"""S3 — training-free Tip-Adapter (+ a CoOp-lite learned-prompt variant) on CONCH.

Extends the underperforming zero-shot ``vl_text_cosine`` (hand prompts) with a
few-shot cache model, using cached artifacts only — no GPU, no text-encoder reload:

  - image features : TITAN (CONCH v1.5) slide embeddings, ``conch_v1.5_titan`` (768-d)
  - text prototypes: the MSI / MSS prompt prototypes already encoded by
    ``vl_text_cosine`` (``text_embeddings.npy``, 2 × 768)

**Tip-Adapter (training-free, primary).** For a test slide feature f (L2-normalised):

  text_delta  = cos(f, MSI_proto) − cos(f, MSS_proto)                    # zero-shot
  cache_delta = mean_{s∈MSI} exp(−β(1−cos(f,F_s)))                       # few-shot cache
              − mean_{s∈MSS} exp(−β(1−cos(f,F_s)))                       # class-balanced
  score       = sigmoid( z(text_delta) + z(cache_delta) )               # α=1, β=5.5

The z(·) are per-fold StandardScalers fit train-only, so the two terms combine at
equal weight without a learned coefficient — the Tip-Adapter "training-free"
property (no gradient; β fixed to the paper default). The cache is built strictly
from the training fold; a few-shot curve K∈{1,2,4,8,16,all} rebuilds it from K
patients per class. At K=0 the score reduces to the zero-shot text_delta.

**CoOp-lite (learned, secondary).** A logistic-regression head on the two
prompt-channel cosines [cos(f,MSI), cos(f,MSS)] — a minimal learned re-weighting of
the hand prompts (a lightweight stand-in for CoOp prompt-context tuning, which would
require differentiating through TITAN's text transformer; scoped out here in favour of
the training-free cache).

Resolution: slide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register

TITAN_DIR = Path("results/embeddings/conch_v1.5_titan")
PROTO_NPY = Path("results/scorers/vl_text_cosine/text_embeddings.npy")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42
N_SPLITS = 5
N_REPEAT = 10
BETA = 5.5  # Tip-Adapter sharpness (paper default)
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")


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
    protos = _l2(np.load(PROTO_NPY).astype(np.float32))  # (2, 768): [MSI, MSS]
    sims = X @ protos.T  # (n, 2)
    meta = meta.assign(msi_sim=sims[:, 0], mss_sim=sims[:, 1], text_delta=sims[:, 0] - sims[:, 1])
    return X, meta


def _cache_delta(f: np.ndarray, Fsup: np.ndarray, ysup: np.ndarray) -> np.ndarray:
    """Class-balanced Tip-Adapter cache delta for each row of f (n_test × dim)."""
    aff = np.exp(-BETA * (1.0 - f @ Fsup.T))  # (n_test, n_sup)
    msi = aff[:, ysup == 1].mean(axis=1) if (ysup == 1).any() else np.zeros(len(f))
    mss = aff[:, ysup == 0].mean(axis=1) if (ysup == 0).any() else np.zeros(len(f))
    return (msi - mss).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _patient_max_sqrtn(slide_df: pd.DataFrame, score_col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        rows.append({
            "patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
            "n_slides": int(len(g)), "score": float(g[score_col].max() / np.sqrt(len(g))),
        })
    return pd.DataFrame(rows)


def _patient_auroc(slide_df: pd.DataFrame, score_col: str = "p_msih") -> float:
    pat = _patient_max_sqrtn(slide_df, score_col)
    if pat["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(pat["y"], pat["score"]))


def _tip_oof(X: np.ndarray, meta: pd.DataFrame, support_idx_fn=None) -> np.ndarray:
    """Tip-Adapter OOF. support_idx_fn(tr, rng) -> subset of tr for the cache;
    None = use the whole training fold."""
    y = meta["y"].to_numpy()
    td = meta["text_delta"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in cv.split(X, y, groups=groups):
        sup = tr if support_idx_fn is None else support_idx_fn(tr)
        if sup is None or len(np.unique(y[sup])) < 2:
            continue
        cd_tr = _cache_delta(X[tr], X[sup], y[sup])
        cd_te = _cache_delta(X[te], X[sup], y[sup])
        sc = StandardScaler().fit(np.column_stack([td[tr], cd_tr]))
        zte = sc.transform(np.column_stack([td[te], cd_te]))
        oof[te] = _sigmoid(zte.sum(axis=1))
    return oof


def _coop_oof(meta: pd.DataFrame) -> np.ndarray:
    y = meta["y"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    F = meta[["msi_sim", "mss_sim"]].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=SEED))])
    for tr, te in cv.split(F, y, groups=groups):
        p = clone(pipe).fit(F[tr], y[tr])
        oof[te] = p.predict_proba(F[te])[:, 1]
    return oof


def _fewshot_curve(X: np.ndarray, meta: pd.DataFrame) -> list[dict]:
    y = meta["y"].to_numpy()
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _tip_oof(X, meta, support_idx_fn=None)
            curve.append({"K": "all",
                          "n_train_per_class": int(min((y == 1).sum(), (y == 0).sum())),
                          "patient_auroc": _patient_auroc(meta.assign(p_msih=oof))})
        else:
            aurocs = []
            for rep in range(N_REPEAT):
                rng = np.random.default_rng(SEED + rep)

                def pick(tr, _rng=rng, _k=int(k)):
                    tr_pat = meta.iloc[tr].drop_duplicates("patient_id")
                    pos = tr_pat[tr_pat["y"] == 1]["patient_id"].to_numpy()
                    neg = tr_pat[tr_pat["y"] == 0]["patient_id"].to_numpy()
                    if len(pos) < _k or len(neg) < _k:
                        return None
                    keep = set(_rng.choice(pos, _k, replace=False)) | set(_rng.choice(neg, _k, replace=False))
                    sel = meta.iloc[tr]["patient_id"].isin(keep).to_numpy()
                    return tr[sel]

                oof = _tip_oof(X, meta, support_idx_fn=pick)
                sd = meta.assign(p_msih=oof).dropna(subset=["p_msih"])
                if sd["y"].nunique() == 2:
                    aurocs.append(_patient_auroc(sd))
            curve.append({"K": int(k), "n_train_per_class": int(k),
                          "patient_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan")})
    return curve


class TipAdapter(Scorer):
    name = "tip_adapter"
    description = (
        "Training-free Tip-Adapter on TITAN (CONCH v1.5) slide embeddings: few-shot "
        "class-balanced cache affinity combined at equal weight (per-fold z-scored) "
        "with the zero-shot MSI/MSS text delta. Secondary CoOp-lite variant learns a "
        "logistic head on the two prompt-channel cosines. Few-shot curve K∈{1,2,4,8,16,all}."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "training-free Tip-Adapter score (full cache)", primary=True),
        ScoreColumn("coop_lite", "learned LR on [MSI,MSS] prompt cosines"),
        ScoreColumn("zero_shot", "zero-shot text_delta (vl baseline)"),
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
        loaded = _load(clean_slide_ids)
        if loaded is None:
            raise RuntimeError(
                f"{self.name}: TITAN embeddings / text prototypes missing "
                f"({TITAN_DIR}, {PROTO_NPY})"
            )
        X, meta = loaded
        tip = _tip_oof(X, meta, support_idx_fn=None)
        coop = _coop_oof(meta)
        slide_df = pd.DataFrame({
            "slide_id": meta["slide_id"], "patient_id": meta["patient_id"],
            "site": meta["site"], "y": meta["y"],
            "p_msih": tip, "coop_lite": coop,
            "zero_shot": _sigmoid((meta["text_delta"].to_numpy() - meta["text_delta"].mean())
                                  / (meta["text_delta"].std() + 1e-8)),
        })
        self._last = {
            "patient_auroc_tip": _patient_auroc(slide_df, "p_msih"),
            "patient_auroc_coop": _patient_auroc(slide_df, "coop_lite"),
            "patient_auroc_zeroshot": _patient_auroc(slide_df, "zero_shot"),
        }
        if full_curve:
            self._last["curve"] = _fewshot_curve(X, meta)

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["curve"]).to_csv(
                    self.score_path.parent / "few_shot_curve.csv", index=False)
        return slide_df


register("tip_adapter", TipAdapter)


def _write_metrics(scorer: "TipAdapter", slide_df: pd.DataFrame) -> dict:
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

    metrics = {
        "scorer": scorer.name,
        "variant_patient_auroc": {
            "tip_adapter": scorer._last["patient_auroc_tip"],
            "coop_lite": scorer._last["patient_auroc_coop"],
            "zero_shot": scorer._last["patient_auroc_zeroshot"],
        },
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
    out = TipAdapter().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
