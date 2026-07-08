"""R3 — fmMAP: supervised-UMAP site-suppressing projection + linear probe.

R2 showed adversarial site-invariance (FLEX) equalizes sites *downward*. R3 tries the
gentler, projection-based alternative: suppress site as a preprocessing constraint, then
learn a low-dimensional MSI-guided manifold and probe it.

Per patient-grouped fold:

1. **Site residualization** — subtract each site's train-set mean TITAN feature (a first-order
   batch correction that removes the dominant per-site shift; test slides use their site's
   train mean, or the global train mean for a site unseen in the fold).
2. **Supervised UMAP (fmMAP)** — fit `umap.UMAP(target_metric='categorical', y=MSI)` on the
   residualized train features → a low-dim embedding shaped by MSI biology; `transform` the
   test features (out-of-sample, no test labels used → no leakage).
3. **Linear probe** — class-balanced logistic regression on the UMAP coordinates.

Reports per-site AUROC and the delta vs the champion, the site-invariance headline. Frozen
features only (CPU).

Resolution: slide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..eval.screening import screening_block
from .base import Scorer, ScoreColumn
from .registry import register

TITAN_DIR = Path("results/embeddings/conch_v1.5_titan")
CLINICAL_CSV = Path("results/data/clinical_table.csv")
SEED = 42
N_SPLITS = 5
UMAP_COMPONENTS = 15
UMAP_NEIGHBORS = 15
FEWSHOT_K: tuple[int | str, ...] = (1, 2, 4, 8, 16, "all")


def _load(clean_slide_ids: set[str] | None):
    if not (TITAN_DIR / "embeddings.npy").exists():
        return None
    X = np.load(TITAN_DIR / "embeddings.npy").astype(np.float32)
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
    return X[meta["_row"].to_numpy()], meta


def _site_residualize(Xtr, site_tr, Xte, site_te):
    """Subtract per-site train mean (global fallback for unseen sites)."""
    gmean = Xtr.mean(axis=0)
    means = {s: Xtr[site_tr == s].mean(axis=0) for s in np.unique(site_tr)}
    Rtr = Xtr - np.vstack([means[s] for s in site_tr])
    Rte = Xte - np.vstack([means.get(s, gmean) for s in site_te])
    return Rtr.astype(np.float32), Rte.astype(np.float32)


def _fmmap_fold(Xtr, ytr, site_tr, Xte, site_te, seed):
    import umap

    Rtr, Rte = _site_residualize(Xtr, site_tr, Xte, site_te)
    sc = StandardScaler().fit(Rtr)
    Rtr, Rte = sc.transform(Rtr), sc.transform(Rte)
    n_nb = min(UMAP_NEIGHBORS, max(2, len(Rtr) - 1))
    reducer = umap.UMAP(n_components=UMAP_COMPONENTS, n_neighbors=n_nb, min_dist=0.1,
                        target_metric="categorical", random_state=seed, verbose=False)
    Ztr = reducer.fit_transform(Rtr, y=ytr)
    Zte = reducer.transform(Rte)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1].astype(np.float32)


def _oof(X, meta, support_fn=None) -> np.ndarray:
    y = meta["y"].to_numpy()
    site = meta["site"].to_numpy()
    groups = meta["patient_id"].to_numpy()
    oof = np.full(len(y), np.nan, dtype=np.float32)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
        sup = tr if support_fn is None else support_fn(tr)
        if sup is None or len(np.unique(y[sup])) < 2 or len(sup) <= UMAP_COMPONENTS + 2:
            continue
        try:
            oof[te] = _fmmap_fold(X[sup], y[sup], site[sup], X[te], site[te], SEED + fold)
        except Exception:  # noqa: BLE001 — UMAP can fail on degenerate tiny supports
            continue
    return oof


def _patient_max_sqrtn(df: pd.DataFrame, col: str = "p_msih") -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("patient_id"):
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
                     "n_slides": int(len(g)), "score": float(g[col].max() / np.sqrt(len(g)))})
    return pd.DataFrame(rows)


def _patient_auroc(df: pd.DataFrame) -> float:
    pat = _patient_max_sqrtn(df)
    return float(roc_auc_score(pat["y"], pat["score"])) if pat["y"].nunique() == 2 else float("nan")


def _fewshot_curve(X, meta) -> list[dict]:
    y = meta["y"].to_numpy()
    curve = []
    for k in FEWSHOT_K:
        if k == "all":
            oof = _oof(X, meta, None)
            sd = meta.assign(p_msih=oof).dropna(subset=["p_msih"])
            curve.append({"K": "all", "n_train_per_class": int(min((y == 1).sum(), (y == 0).sum())),
                          "patient_auroc": _patient_auroc(sd) if len(sd) else float("nan")})
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

                oof = _oof(X, meta, pick)
                sd = meta.assign(p_msih=oof).dropna(subset=["p_msih"])
                if sd["y"].nunique() == 2 and len(sd):
                    aurocs.append(_patient_auroc(sd))
            curve.append({"K": int(k), "n_train_per_class": int(k),
                          "patient_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan")})
    return curve


class FmmapProbe(Scorer):
    name = "fmmap_probe"
    description = (
        "fmMAP: per-fold site residualization + supervised UMAP (MSI-guided, "
        "target_metric='categorical') on TITAN slide features, then a class-balanced "
        "logistic-regression probe on the projected coordinates. A projection-based "
        "site-suppression alternative to R2's adversary. Patient-grouped CV; few-shot "
        "curve K∈{1,2,4,8,16,all}. Frozen features."
    )
    needs_training_on_our_data = True
    resolution = "slide"
    score_columns = [
        ScoreColumn("p_msih", "fmMAP UMAP-probe OOF p(MSI-H)", primary=True),
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
            raise RuntimeError(f"{self.name}: TITAN embeddings missing under {TITAN_DIR}")
        X, meta = loaded
        oof = _oof(X, meta, None)
        slide_df = pd.DataFrame({
            "slide_id": meta["slide_id"], "patient_id": meta["patient_id"],
            "site": meta["site"], "y": meta["y"], "p_msih": oof,
        }).dropna(subset=["p_msih"]).reset_index(drop=True)
        self._last = {}
        if full_curve:
            self._last["curve"] = _fewshot_curve(X, meta)
        if write_outputs and self.score_path is not None:
            self.score_path.parent.mkdir(parents=True, exist_ok=True)
            slide_df.to_csv(self.score_path, index=False)
            if full_curve:
                pd.DataFrame(self._last["curve"]).to_csv(
                    self.score_path.parent / "few_shot_curve.csv", index=False)
        return slide_df


register("fmmap_probe", FmmapProbe)


def _write_metrics(scorer: "FmmapProbe", slide_df: pd.DataFrame) -> dict:
    import json

    from ..eval.metrics import _safe_auprc, _safe_auroc

    pat = _patient_max_sqrtn(slide_df)
    y, s = pat["y"].to_numpy(), pat["score"].to_numpy()
    block = screening_block(y, s, (0.90, 0.95, 0.96, 0.98))

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
        "scorer": scorer.name, "method": "site-residualize + supervised UMAP + LR probe",
        "umap_components": UMAP_COMPONENTS, "umap_neighbors": UMAP_NEIGHBORS,
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
    out = FmmapProbe().score_path.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    return metrics
