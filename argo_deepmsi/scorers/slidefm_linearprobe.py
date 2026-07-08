"""S1 — linear probe (logistic regression) on slide-level FM embeddings.

Fits a class-balanced logistic-regression head, 5-fold patient-grouped, on the
already-extracted slide-level foundation-model embeddings:

  - TITAN  (``conch_v1.5_titan``,  768-d)  — CONCH v1.5 slide encoder
  - PRISM  (``virchow2_prism``,   1280-d)  — Virchow2 slide encoder

Slide-level FM embeddings are one vector per slide (no tile aggregation), so the
probe is a genuine low-N recipe: a linear head on frozen features, nothing more.
Reports a few-shot learning curve K ∈ {1, 2, 4, 8, 16, all} — for each K the head
is trained on only K patients per class (drawn from the fold's training patients,
averaged over ``N_REPEAT`` random draws) and evaluated on the held-out fold. The
primary score is the full-shot (K=all) OOF probability of the better embedding.

Frozen embeddings only — no FM finetuning, no external fitting. Fit time is
seconds on CPU.

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

EMB_ROOT = Path("results/embeddings")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42
N_SPLITS = 5
N_REPEAT = 10  # random few-shot draws averaged per (fold, K)

# (embedding dir, display name)
EMBEDDINGS: tuple[tuple[str, str], ...] = (
    ("conch_v1.5_titan", "TITAN"),
    ("virchow2_prism", "PRISM"),
)
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")


def _lr_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def _load_embedding(
    emb_name: str, clinical: pd.DataFrame, clean_slide_ids: set[str] | None
) -> pd.DataFrame | None:
    """Return a per-slide frame with columns slide_id, patient_id, site, y and an
    ``_row`` index into the embedding matrix; None if the embedding is absent."""
    emb_dir = EMB_ROOT / emb_name
    if not (emb_dir / "embeddings.npy").exists():
        return None
    meta = pd.read_csv(emb_dir / "metadata.csv")
    if "slide_id" not in meta.columns:
        return None
    meta = meta.copy()
    meta["_row"] = np.arange(len(meta))
    base = meta.merge(clinical[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")
    if clean_slide_ids is not None:
        base = base[base["slide_id"].isin(clean_slide_ids)]
    base = base.drop_duplicates("slide_id").reset_index(drop=True)
    return base if len(base) else None


def _patient_max_sqrtn(slide_df: pd.DataFrame, score_col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        rows.append({
            "patient_id": pid,
            "y": int(g["y"].iloc[0]),
            "site": g["site"].iloc[0],
            "n_slides": int(len(g)),
            "score": float(g[score_col].max() / np.sqrt(len(g))),
        })
    return pd.DataFrame(rows)


def _patient_auroc(slide_df: pd.DataFrame, score_col: str = "p_msih") -> float:
    pat = _patient_max_sqrtn(slide_df, score_col)
    if pat["y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(pat["y"], pat["score"]))


def _oof_full(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Full-shot 5-fold patient-grouped OOF probabilities."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in cv.split(X, y, groups=groups):
        p = clone(_lr_pipeline()).fit(X[tr], y[tr])
        oof[te] = p.predict_proba(X[te])[:, 1]
    return oof


def _fewshot_patient_auroc(
    X: np.ndarray, base: pd.DataFrame, k: int
) -> float:
    """Few-shot curve point: train on k patients/class per fold, averaged over
    ``N_REPEAT`` draws; return patient AUROC of the concatenated OOF."""
    y = base["y"].to_numpy()
    groups = base["patient_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aurocs = []
    for rep in range(N_REPEAT):
        rng = np.random.default_rng(SEED + rep)
        oof = np.full(len(y), np.nan, dtype=np.float32)
        ok = True
        for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
            tr_pat = base.iloc[tr].drop_duplicates("patient_id")
            pos = tr_pat[tr_pat["y"] == 1]["patient_id"].to_numpy()
            neg = tr_pat[tr_pat["y"] == 0]["patient_id"].to_numpy()
            if len(pos) < k or len(neg) < k:
                ok = False
                break
            pick = set(rng.choice(pos, k, replace=False)) | set(rng.choice(neg, k, replace=False))
            sel = base.iloc[tr]["patient_id"].isin(pick).to_numpy()
            tr_sel = tr[sel]
            if len(np.unique(y[tr_sel])) < 2:
                ok = False
                break
            p = clone(_lr_pipeline()).fit(X[tr_sel], y[tr_sel])
            oof[te] = p.predict_proba(X[te])[:, 1]
        if not ok:
            continue
        sd = base.assign(p_msih=oof).dropna(subset=["p_msih"])
        aurocs.append(_patient_auroc(sd))
    return float(np.nanmean(aurocs)) if aurocs else float("nan")


def _fewshot_curve(X: np.ndarray, base: pd.DataFrame) -> list[dict]:
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _oof_full(X, base["y"].to_numpy(), base["patient_id"].to_numpy())
            sd = base.assign(p_msih=oof)
            pat = _patient_max_sqrtn(sd)
            blk = screening_block(pat["y"].to_numpy(), pat["score"].to_numpy(), (0.95,)) \
                if pat["y"].nunique() == 2 else {"spec_at_sens95": float("nan")}
            curve.append({
                "K": "all",
                "n_train_per_class": int(min((base["y"] == 1).sum(), (base["y"] == 0).sum())),
                "patient_auroc": _patient_auroc(sd),
                "spec_at_sens95": float(blk["spec_at_sens95"]),
            })
        else:
            curve.append({
                "K": int(k),
                "n_train_per_class": int(k),
                "patient_auroc": _fewshot_patient_auroc(X, base, int(k)),
                "spec_at_sens95": float("nan"),
            })
    return curve


class SlideFMLinearProbe(Scorer):
    name = "slidefm_linearprobe"
    description = (
        "Class-balanced logistic-regression linear probe on frozen slide-level "
        "FM embeddings (TITAN / PRISM), 5-fold patient-grouped. Exposes the "
        "full-shot OOF p(MSI-H) of the better embedding as the primary score, "
        "and reports a few-shot learning curve K∈{1,2,4,8,16,all} (K patients "
        "per class per fold, averaged over random draws)."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "full-shot OOF p(MSI-H), best slide-FM embedding", primary=True),
    ]

    def compute_batch(
        self,
        slide_table: pd.DataFrame,
        *,
        clean_slide_ids: set[str] | None = None,
        embeddings: tuple[tuple[str, str], ...] = EMBEDDINGS,
        full_curve: bool = False,
        write_outputs: bool = True,
        **_,
    ) -> pd.DataFrame:
        cl = pd.read_csv(CLINICAL_CSV)[["PATIENT", "isMSIH"]]
        cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)

        per_emb: list[dict] = []
        for emb_name, disp in embeddings:
            base = _load_embedding(emb_name, cl, clean_slide_ids)
            if base is None:
                continue
            X = np.load(EMB_ROOT / emb_name / "embeddings.npy")[base["_row"].to_numpy()]
            oof = _oof_full(X, base["y"].to_numpy(), base["patient_id"].to_numpy())
            slide_df = pd.DataFrame({
                "slide_id": base["slide_id"], "patient_id": base["patient_id"],
                "site": base["site"], "y": base["y"], "p_msih": oof,
            })
            per_emb.append({
                "embedding": emb_name, "display": disp, "X": X, "base": base,
                "slide_df": slide_df, "patient_auroc": _patient_auroc(slide_df),
            })

        if not per_emb:
            raise RuntimeError(f"{self.name}: no slide-FM embeddings available under {EMB_ROOT}")

        per_emb.sort(key=lambda d: (np.nan_to_num(d["patient_auroc"], nan=-1.0)), reverse=True)
        best = per_emb[0]
        best_slide_df = best["slide_df"]

        # Stash for the standalone runner (metrics.json + doc).
        self._last = {
            "best_embedding": best["embedding"],
            "best_display": best["display"],
            "per_embedding_auroc": {d["display"]: d["patient_auroc"] for d in per_emb},
        }
        if full_curve:
            self._last["curve"] = {
                d["display"]: _fewshot_curve(d["X"], d["base"]) for d in per_emb
            }

        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            best_slide_df.to_csv(self.score_path, index=False)
            pd.DataFrame(
                [{"embedding": d["embedding"], "display": d["display"],
                  "patient_auroc": d["patient_auroc"]} for d in per_emb]
            ).to_csv(self.score_path.parent / "embedding_ranking.csv", index=False)
            if full_curve:
                rows = []
                for disp, curve in self._last["curve"].items():
                    for pt in curve:
                        rows.append({"embedding": disp, **pt})
                pd.DataFrame(rows).to_csv(self.score_path.parent / "few_shot_curve.csv", index=False)

        return best_slide_df


register("slidefm_linearprobe", SlideFMLinearProbe)


def _write_metrics(scorer: "SlideFMLinearProbe", slide_df: pd.DataFrame) -> dict:
    """Assemble + write the FULL metric block on the clean cohort (Q4 in_clean_set)."""
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc

    pat = _patient_max_sqrtn(slide_df)
    y = pat["y"].to_numpy()
    s = pat["score"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))

    by_site = {}
    for site, sub in pat.groupby("site"):
        yy, ss = sub["y"].to_numpy(), sub["score"].to_numpy()
        by_site[str(site)] = {
            "n": int(len(sub)),
            "prevalence": float(sub["y"].mean()),
            "auroc": _safe_auroc(yy, ss),
            "spec_at_sens95": screening_block(yy, ss, (0.95,))["spec_at_sens95"]
            if sub["y"].nunique() == 2 else float("nan"),
        }

    bins = [0, 1, 2, 4, np.inf]
    labels = ["1", "2", "3-4", "5+"]
    pat = pat.assign(_bucket=pd.cut(pat["n_slides"], bins=bins, labels=labels))
    by_bagsize = {}
    for bucket, sub in pat.groupby("_bucket", observed=True):
        by_bagsize[str(bucket)] = {
            "n": int(len(sub)),
            "auroc": _safe_auroc(sub["y"].to_numpy(), sub["score"].to_numpy()),
        }

    metrics = {
        "scorer": scorer.name,
        "best_embedding": scorer._last["best_embedding"],
        "per_embedding_patient_auroc": scorer._last["per_embedding_auroc"],
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
        "few_shot_curve": scorer._last.get("curve", {}),
        "screening_clean": block,
    }
    out = SlideFMLinearProbe().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
