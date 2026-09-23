"""Domain-shift quantification, mitigation and calibration on the primary cohort.

Implements, on existing slide embeddings and the Wagner reference scores, the three
steps of the proposed domain-shift aim:

1. **Quantify** — Fréchet distance between site Gaussians in each foundation-model
   space (with a patient-level permutation null), patient-grouped site
   predictability, and per-site baseline performance of the Wagner reference. The
   image-level counterparts (colour-histogram KL, Macenko stain vectors, Inception
   FID/KID) summarise the shards written by ``scripts/domain_shift/image_stats.py``.
2. **Mitigate** — multi-source domain-adversarial training (gradient reversal over
   site) of a small slide head, with the adversarial weight selected in the inner
   loop of nested patient-grouped CV, and a leave-OAUTHC-out variant with and without
   unlabelled target slides in the adversary.
3. **Lock and calibrate** — nested comparison of no calibration, Platt scaling and
   isotonic regression: ECE, Brier, calibration slope and realised sensitivity of a
   train-fold sens-0.95 threshold, pooled and per site.

Three natural contrasts isolate processing factors (all slides imaged in Nigeria):
``stain`` = retrospective_msk vs retrospective_oau (same patients, same MSKCC cut,
MSKCC vs OAUTHC staining); ``cut`` = retrospective_oau vs OAUTHC (same staining lab,
MSKCC vs OAUTHC sectioning, different patients).

Usage:
    python -m argo_deepmsi.eval.domain_shift embed
    python -m argo_deepmsi.eval.domain_shift dann
    python -m argo_deepmsi.eval.domain_shift calibration
    python -m argo_deepmsi.eval.domain_shift image
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .metrics import aggregate_to_patient, stratified_patient_bootstrap
from .screening import threshold_at_sensitivity

COHORT_CSV = Path("results/data/cohort_clean.csv")
EMB_ROOT = Path("results/embeddings")
WAGNER_CSV = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
OUT_DIR = Path("results/analysis/domain_shift")
IMAGE_CACHE = Path("results/domain_shift_cache/image_stats")
REFERENCE_SITE = "retrospective_msk"
ENCODERS = [
    "ctranspath_mean",
    "uni2_mean",
    "virchow2_mean",
    "conch_v1.5_titan",
    "phaet_mean",
    "mascaret_mean",
    "virchow2_prism",
]
HARMONY = ["conch_v1.5_titan_harmony", "uni2_mean_harmony", "virchow2_mean_harmony"]
CONTRASTS = {
    "stain (retro_msk vs retro_oau, same patients)": ("retrospective_msk", "retrospective_oau"),
    "cut (retro_oau vs OAUTHC, same stain lab)": ("retrospective_oau", "OAUTHC"),
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_cohort(cohort_csv: Path = COHORT_CSV) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_csv)
    cohort = cohort[cohort["in_primary_set"] == 1]
    return cohort[["slide_id", "patient_id", "site", "y"]].reset_index(drop=True)


def load_embedding(name: str, cohort: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows of ``cohort`` that have this embedding, aligned with its matrix."""
    X = np.load(EMB_ROOT / name / "embeddings.npy").astype(np.float64)
    meta = pd.read_csv(EMB_ROOT / name / "metadata.csv")[["slide_id"]]
    meta["row"] = np.arange(len(meta))
    frame = cohort.merge(meta, on="slide_id", how="inner")
    return frame.drop(columns="row").reset_index(drop=True), X[frame["row"].to_numpy()]


# --------------------------------------------------------------------------
# Step 1 — distances and site predictability
# --------------------------------------------------------------------------


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fit (Ledoit-Wolf) to two samples."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = LedoitWolf().fit(a).covariance_
    cov_b = LedoitWolf().fit(b).covariance_
    covmean = linalg.sqrtm(cov_a @ cov_b)
    covmean = np.real(covmean)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))


def permutation_frechet(
    X: np.ndarray,
    in_a: np.ndarray,
    in_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Observed FD between A and B plus a null that permutes group membership.

    Membership is permuted at the level of ``groups`` (patients) among the units in
    A ∪ B, so the null keeps within-patient correlation and group sizes. A unit
    present in both A and B (paired scans) keeps one draw for all its slides.
    """
    obs = frechet_distance(X[in_a], X[in_b])
    keep = in_a | in_b
    Xk, ga, gk = X[keep], in_a[keep], groups[keep]
    units = np.unique(gk)
    frac_a = np.mean([ga[gk == u].mean() for u in units])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        draw = dict(zip(units, rng.random(len(units)) < frac_a, strict=True))
        pa = np.array([draw[g] for g in gk])
        if pa.sum() < 3 or (~pa).sum() < 3:
            null[i] = np.nan
            continue
        null[i] = frechet_distance(Xk[pa], Xk[~pa])
    null = null[~np.isnan(null)]
    return {
        "fd": obs,
        "null_mean": float(null.mean()),
        "excess_fd": float(obs - null.mean()),
        "fd_ratio": float(obs / null.mean()),
        "p": float((1 + (null >= obs).sum()) / (1 + len(null))),
    }


def site_predictability(frame: pd.DataFrame, X: np.ndarray, *, seed: int = 0) -> dict:
    """Patient-grouped CV: how well a linear model identifies the site of a slide."""
    y_site = frame["site"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    classes = np.unique(y_site)
    proba = np.zeros((len(frame), len(classes)))
    for tr, te in cv.split(X, y_site, groups):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y_site[tr])
        proba[np.ix_(te, np.searchsorted(classes, clf.classes_))] = clf.predict_proba(
            scaler.transform(X[te])
        )
    out = {
        "site_macro_auroc": float(
            roc_auc_score(y_site, proba, multi_class="ovr", average="macro", labels=classes)
        ),
        "site_balanced_acc": float(balanced_accuracy_score(y_site, classes[proba.argmax(1)])),
    }
    for label, (a, b) in CONTRASTS.items():
        m = frame["site"].isin([a, b]).to_numpy()
        yb = (frame.loc[m, "site"] == b).to_numpy().astype(int)
        pb = np.zeros(m.sum())
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            X[m], yb, groups[m]
        ):
            scaler = StandardScaler().fit(X[m][tr])
            clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
            clf.fit(scaler.transform(X[m][tr]), yb[tr])
            pb[te] = clf.predict_proba(scaler.transform(X[m][te]))[:, 1]
        out[f"auroc_{label.split()[0]}"] = float(roc_auc_score(yb, pb))
    return out


def embed_shift(encoders: list[str], n_pca: int = 32, n_perm: int = 200) -> pd.DataFrame:
    cohort = load_cohort()
    rows = []
    for name in encoders:
        if not (EMB_ROOT / name / "embeddings.npy").exists():
            continue
        frame, X = load_embedding(name, cohort)
        Z = PCA(n_components=min(n_pca, X.shape[1]), random_state=0).fit_transform(
            StandardScaler().fit_transform(X)
        )
        sites = frame["site"].to_numpy()
        groups = frame["patient_id"].to_numpy()
        comparisons = {
            f"{s} vs {REFERENCE_SITE}": (sites == s, sites == REFERENCE_SITE)
            for s in np.unique(sites)
            if s != REFERENCE_SITE
        }
        comparisons |= {k: (sites == a, sites == b) for k, (a, b) in CONTRASTS.items()}
        y = frame["y"].to_numpy() == 1
        comparisons["biology: MSI-H vs MSS (all sites)"] = (y, ~y)
        oau = sites == "OAUTHC"
        comparisons["biology: MSI-H vs MSS (OAUTHC)"] = (y & oau, ~y & oau)
        for label, (a, b) in comparisons.items():
            rows.append(
                {"encoder": name, "comparison": label, "n_a": int(a.sum()), "n_b": int(b.sum())}
                | permutation_frechet(Z, a, b, groups, n_perm=n_perm)
            )
        rows.append({"encoder": name, "comparison": "site_predictability"} | site_predictability(
            frame, X
        ))
        print(f"[embed] {name} done", flush=True)
    return pd.DataFrame(rows)


def baseline_per_site(score_csv: Path = WAGNER_CSV, score_col: str = "p_msih") -> pd.DataFrame:
    """Per-site patient AUROC (bootstrap CI) and sens-0.95 operating point of a scorer."""
    cohort = load_cohort()
    slides = pd.read_csv(score_csv)[["slide_id", score_col]].merge(cohort, on="slide_id")
    patients = aggregate_to_patient(slides, score_col)
    thr = threshold_at_sensitivity(patients["y"], patients["score"], 0.95)
    rows = []
    for site, sub in [("ALL", patients), *patients.groupby("site")]:
        ci = stratified_patient_bootstrap(sub) if sub["y"].nunique() == 2 else {}
        called = sub["score"] >= thr
        pos, neg = sub["y"] == 1, sub["y"] == 0
        rows.append(
            {
                "site": site,
                "n_patients": len(sub),
                "n_msih": int(pos.sum()),
                "auroc": ci.get("estimate", np.nan),
                "ci_low": ci.get("ci_low", np.nan),
                "ci_high": ci.get("ci_high", np.nan),
                "sens_at_global_thr": float(called[pos].mean()) if pos.any() else np.nan,
                "spec_at_global_thr": float((~called[neg]).mean()) if neg.any() else np.nan,
                "mss_score_median": float(sub.loc[neg, "score"].median()),
            }
        )
    return pd.DataFrame(rows)


def slide_count_audit(score_csv: Path = WAGNER_CSV, score_col: str = "p_msih") -> pd.DataFrame:
    """Separate image signal from slides-per-patient in a pooled patient score.

    For each site: patient AUROC of max/√n pooling, of mean pooling (no count term), of
    one random slide per patient (300 draws), of the slide count alone (−n), and the
    slide-level AUROC. If −n rivals max/√n, the pooled score is exploiting how many slides
    were submitted rather than morphology.
    """
    cohort = load_cohort()
    slides = pd.read_csv(score_csv)[["slide_id", score_col]].merge(cohort, on="slide_id")
    rng = np.random.default_rng(0)
    rows = []
    for site, d in [("ALL", slides), *slides.groupby("site")]:
        if d["y"].nunique() < 2:
            continue
        g = d.groupby("patient_id").agg(y=("y", "first"), mx=(score_col, "max"),
                                        mean=(score_col, "mean"), n=(score_col, "size"))
        draws = [roc_auc_score(s["y"], s[score_col]) for s in (
            d.groupby("patient_id").sample(1, random_state=int(rng.integers(1e9)))
            for _ in range(300))]
        rows.append({
            "site": site, "n_patients": len(g), "n_msih": int(g["y"].sum()),
            "slides_per_patient_mss": float(g.loc[g.y == 0, "n"].mean()),
            "slides_per_patient_msih": float(g.loc[g.y == 1, "n"].mean()),
            "auroc_max_sqrtn": roc_auc_score(g["y"], g["mx"] / np.sqrt(g["n"])),
            "auroc_mean": roc_auc_score(g["y"], g["mean"]),
            "auroc_one_random_slide": float(np.mean(draws)),
            "auroc_slide_level": roc_auc_score(d["y"], d[score_col]),
            "auroc_minus_n_slides": roc_auc_score(g["y"], -g["n"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 2 — multi-source domain-adversarial head
# --------------------------------------------------------------------------


def fit_predict_dann(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    d_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    lam: float,
    X_unlab: np.ndarray | None = None,
    d_unlab: np.ndarray | None = None,
    hidden: int = 128,
    epochs: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Train an MLP MSI head with a gradient-reversed site adversary; return P(MSI).

    ``lam=0`` is the identical architecture without adversarial pressure (the
    matched control). Unlabelled slides (e.g. a held-out target site) contribute
    only to the adversary.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_tr if X_unlab is None else np.vstack([X_tr, X_unlab]))
    xt = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
    xe = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    d_all = d_tr if d_unlab is None else np.concatenate([d_tr, d_unlab])
    domains = {d: i for i, d in enumerate(np.unique(d_all))}
    xd = xt if X_unlab is None else torch.cat(
        [xt, torch.tensor(scaler.transform(X_unlab), dtype=torch.float32)]
    )
    dd = torch.tensor([domains[d] for d in d_all])

    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -lam * g

    feat = nn.Sequential(nn.Dropout(0.2), nn.Linear(X_tr.shape[1], hidden), nn.GELU(),
                         nn.Dropout(0.3))
    head = nn.Linear(hidden, 1)
    adv = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, len(domains)))
    params = [*feat.parameters(), *head.parameters(), *adv.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    pos_weight = torch.tensor((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Balance domains so the large site does not own the adversary.
    counts = torch.bincount(dd, minlength=len(domains)).float()
    ce = nn.CrossEntropyLoss(weight=counts.sum() / (len(domains) * counts.clamp(min=1)))
    for epoch in range(epochs):
        feat.train(), head.train(), adv.train()
        opt.zero_grad()
        loss = bce(head(feat(xt)).squeeze(1), yt)
        if lam > 0:
            ramp = 2 / (1 + np.exp(-10 * epoch / epochs)) - 1
            loss = loss + ramp * ce(adv(Reverse.apply(feat(xd))), dd)
        loss.backward()
        opt.step()
    feat.eval(), head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(feat(xe)).squeeze(1)).numpy()


def _patient_auc(frame: pd.DataFrame, p: np.ndarray) -> float:
    pt = aggregate_to_patient(frame.assign(p=p), "p")
    return float(roc_auc_score(pt["y"], pt["score"])) if pt["y"].nunique() == 2 else np.nan


def _patient_split(frame: pd.DataFrame, n_splits: int, seed: int):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(frame))
    yield from cv.split(idx, frame["y"].to_numpy(), frame["patient_id"].to_numpy())


def dann_nested(
    frame: pd.DataFrame,
    X: np.ndarray,
    *,
    lams: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0),
    repeats: int = 2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested patient-grouped CV; lambda chosen on inner folds. Returns OOF + audit."""
    y = frame["y"].to_numpy().astype(float)
    d = frame["site"].to_numpy()
    oof = {k: np.zeros(len(frame)) for k in ("selected", "lam0")}
    audit = []
    for r in range(repeats):
        for k, (tr, te) in enumerate(_patient_split(frame, 5, seed + r)):
            inner_scores = {}
            for lam in lams:
                aucs = []
                for itr, ite in _patient_split(frame.iloc[tr].reset_index(drop=True), 3, seed):
                    p = fit_predict_dann(X[tr][itr], y[tr][itr], d[tr][itr], X[tr][ite],
                                         lam=lam, seed=seed)
                    aucs.append(_patient_auc(frame.iloc[tr].iloc[ite], p))
                inner_scores[lam] = float(np.nanmean(aucs))
            best = max(inner_scores, key=inner_scores.get)
            oof["selected"][te] += fit_predict_dann(X[tr], y[tr], d[tr], X[te], lam=best,
                                                    seed=seed) / repeats
            oof["lam0"][te] += fit_predict_dann(X[tr], y[tr], d[tr], X[te], lam=0.0,
                                                seed=seed) / repeats
            audit.append({"repeat": r, "fold": k, "selected_lam": best} | {
                f"inner_auc_lam{lam}": v for lam, v in inner_scores.items()
            })
    out = frame.assign(p_dann=oof["selected"], p_mlp=oof["lam0"])
    return out, pd.DataFrame(audit)


def dann_leave_site_out(
    frame: pd.DataFrame,
    X: np.ndarray,
    target: str = "OAUTHC",
    *,
    lams: tuple[float, ...] = (0.1, 0.3, 1.0),
    seed: int = 0,
) -> pd.DataFrame:
    """Train on every patient without a ``target`` slide; score the target site.

    Variants: source-only MLP; multi-source DANN (adversary over source sites);
    DANN + unlabelled target slides in the adversary (target labels never used).
    The adversarial weight is selected by inner leave-one-source-site-out AUROC.
    """
    tgt_pts = set(frame.loc[frame["site"] == target, "patient_id"])
    src = ~frame["patient_id"].isin(tgt_pts).to_numpy()
    te = (frame["site"] == target).to_numpy()
    fs = frame[src].reset_index(drop=True)
    Xs, ys, ds = X[src], fs["y"].to_numpy().astype(float), fs["site"].to_numpy()
    big_sources = [s for s in fs["site"].unique() if fs.loc[fs.site == s, "y"].nunique() == 2
                   and (fs.site == s).sum() >= 20]

    def inner_select(use_target: bool) -> float:
        scores = {}
        for lam in lams:
            aucs = []
            for hold in big_sources:
                hold_pts = set(fs.loc[fs.site == hold, "patient_id"])
                itr = ~fs["patient_id"].isin(hold_pts).to_numpy()
                ite = (fs["site"] == hold).to_numpy()
                p = fit_predict_dann(
                    Xs[itr], ys[itr], ds[itr], Xs[ite], lam=lam, seed=seed,
                    X_unlab=Xs[ite] if use_target else None,
                    d_unlab=np.array([f"target:{hold}"] * ite.sum()) if use_target else None,
                )
                aucs.append(_patient_auc(fs[ite], p))
            scores[lam] = float(np.nanmean(aucs))
        return max(scores, key=scores.get)

    rows = []
    variants = {
        "source-only MLP": dict(lam=0.0),
        "multi-source DANN": dict(lam=inner_select(False)),
        "DANN + unlabelled target": dict(lam=inner_select(True), X_unlab=X[te],
                                         d_unlab=np.array(["target"] * te.sum())),
    }
    lr = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")
    scaler = StandardScaler().fit(Xs)
    p_lr = lr.fit(scaler.transform(Xs), ys).predict_proba(scaler.transform(X[te]))[:, 1]
    preds = {"source-only LR": p_lr}
    for name, kw in variants.items():
        preds[name] = np.mean([fit_predict_dann(Xs, ys, ds, X[te], seed=s, **kw)
                               for s in range(3)], axis=0)
    for name, p in preds.items():
        pt = aggregate_to_patient(frame[te].assign(p=p), "p")
        ci = stratified_patient_bootstrap(pt)
        rows.append({"variant": name, "lam": variants.get(name, {}).get("lam", np.nan),
                     "target": target, "n_patients": len(pt), "n_msih": int(pt.y.sum()),
                     "auroc": ci["estimate"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 3 — nested calibration
# --------------------------------------------------------------------------


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> float:
    """Equal-mass-bin ECE (few bins: 47 positives cannot support more)."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def nested_calibration(patients: pd.DataFrame, *, repeats: int = 50, seed: int = 0) -> tuple[
    pd.DataFrame, pd.DataFrame
]:
    """Outer 5-fold (stratified by label) CV of calibration maps fit on train folds.

    ``patients`` has one row per patient with ``y``, ``site`` and a raw ``score``.
    Returns pooled metrics per method and per-site realised operating points.
    """
    from sklearn.model_selection import StratifiedKFold

    y = patients["y"].to_numpy()
    s = patients["score"].to_numpy()
    methods = ("uncalibrated", "platt", "isotonic")
    pooled, sites = [], []
    for r in range(repeats):
        preds = {m: np.zeros(len(y)) for m in methods}
        called = {m: np.zeros(len(y), bool) for m in methods}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(s, y):
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(s[tr, None], y[tr])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[tr], y[tr])
            fitted = {
                "uncalibrated": (s[tr], s[te]),
                "platt": (platt.predict_proba(s[tr, None])[:, 1],
                          platt.predict_proba(s[te, None])[:, 1]),
                "isotonic": (iso.predict(s[tr]), iso.predict(s[te])),
            }
            for m, (p_tr, p_te) in fitted.items():
                preds[m][te] = p_te
                thr = threshold_at_sensitivity(y[tr], p_tr, 0.95)
                called[m][te] = p_te >= thr
        for m in methods:
            slope, intercept = calibration_slope(y, preds[m]) if m != "uncalibrated" else (
                np.nan, np.nan)
            pooled.append({
                "repeat": r, "method": m,
                "auroc": roc_auc_score(y, preds[m]),
                "ece": expected_calibration_error(y, preds[m]) if m != "uncalibrated" else np.nan,
                "brier": float(np.mean((preds[m] - y) ** 2)) if m != "uncalibrated" else np.nan,
                "slope": slope, "intercept": intercept,
                "n_distinct_probs": len(np.unique(np.round(preds[m], 6))),
                "sens": called[m][y == 1].mean(), "spec": (~called[m][y == 0]).mean(),
            })
            for site, idx in patients.groupby("site").indices.items():
                yy, cc = y[idx], called[m][idx]
                sites.append({"repeat": r, "method": m, "site": site,
                              "sens": cc[yy == 1].mean() if (yy == 1).any() else np.nan,
                              "spec": (~cc[yy == 0]).mean() if (yy == 0).any() else np.nan})
    pooled = pd.DataFrame(pooled).groupby("method").agg(["mean", "std"]).drop(columns="repeat")
    pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
    per_site = pd.DataFrame(sites).groupby(["method", "site"])[["sens", "spec"]].mean()
    return pooled.reset_index(), per_site.reset_index()


# --------------------------------------------------------------------------
# Step 1 (image level) — summarise image_stats shards
# --------------------------------------------------------------------------


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


def kernel_inception_distance(a: np.ndarray, b: np.ndarray, *, n_sub: int = 500,
                              n_rep: int = 20, seed: int = 0) -> float:
    """Unbiased polynomial-kernel MMD^2 (KID), averaged over equal-size subsets."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    for _ in range(n_rep):
        m = min(n_sub, len(a), len(b))
        x = a[rng.choice(len(a), m, replace=False)]
        z = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kzz = (z @ z.T / d + 1) ** 3
        kxz = (x @ z.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kzz.sum() - np.trace(kzz)) / (m * (m - 1)) - 2 * kxz.mean())
    return float(np.mean(vals))


def image_summary(cache: Path = IMAGE_CACHE) -> dict[str, pd.DataFrame]:
    cohort = load_cohort()
    meta = pd.concat([pd.read_csv(f) for f in sorted(cache.glob("shard*.csv"))])
    errors = meta[meta.get("error").notna()] if "error" in meta else meta.iloc[:0]
    npz = [np.load(f) for f in sorted(cache.glob("shard*.npz"))]
    ids = np.concatenate([z["slide_id"] for z in npz])
    hist = np.concatenate([z["hist"] for z in npz])
    feats, tile_slide, offset = [], [], 0
    for z in npz:
        feats.append(z["inception"].astype(np.float32))
        tile_slide.append(z["tile_slide"] + offset)
        offset += len(z["slide_id"])
    feats, tile_slide = np.concatenate(feats), np.concatenate(tile_slide)
    frame = pd.DataFrame({"slide_id": ids, "row": np.arange(len(ids))}).merge(
        cohort, on="slide_id")
    frame = frame.merge(meta.drop(columns=["site", "patient_id"], errors="ignore"),
                        on="slide_id", how="left")
    site_of_row = dict(zip(frame["row"], frame["site"], strict=True))
    y_of_row = dict(zip(frame["row"], frame["y"], strict=True))
    tile_site = np.array([site_of_row.get(r) for r in tile_slide])
    tile_y = np.array([y_of_row.get(r, -1) for r in tile_slide])

    parts = {"rgb_joint": slice(0, 512), "hematoxylin_od": slice(512, 544),
             "eosin_od": slice(544, 576)}
    groups = {s: frame.loc[frame.site == s, "row"].to_numpy() for s in frame.site.unique()}
    comparisons = {f"{s} vs {REFERENCE_SITE}": (s, REFERENCE_SITE) for s in groups
                   if s != REFERENCE_SITE}
    comparisons |= CONTRASTS
    kl_rows, fid_rows = [], []
    rng = np.random.default_rng(0)
    for label, (a, b) in comparisons.items():
        ra, rb = groups[a], groups[b]
        row = {"comparison": label, "n_slides_a": len(ra), "n_slides_b": len(rb)}
        for pname, part in parts.items():
            row[f"kl_{pname}"] = _kl(hist[ra][:, part].sum(0), hist[rb][:, part].sum(0))
            # Null: split the pooled slides of both sites at random (same sizes).
            pool = np.concatenate([ra, rb])
            null = []
            for _ in range(100):
                perm = rng.permutation(pool)
                null.append(_kl(hist[perm[: len(ra)]][:, part].sum(0),
                                hist[perm[len(ra):]][:, part].sum(0)))
            row[f"kl_{pname}_null"] = float(np.mean(null))
        kl_rows.append(row)
        fa, fb = feats[tile_site == a], feats[tile_site == b]
        n = min(len(fa), len(fb), 2000)
        fid_rows.append({
            "comparison": label, "n_tiles": n,
            "fid": frechet_distance(fa[rng.choice(len(fa), n, replace=False)],
                                    fb[rng.choice(len(fb), n, replace=False)]),
            "kid_x1000": 1000 * kernel_inception_distance(fa, fb),
        })
    pos, neg = feats[tile_y == 1], feats[tile_y == 0]
    n = min(len(pos), len(neg), 2000)
    fid_rows.append({"comparison": "biology: MSI-H vs MSS tiles (all sites)", "n_tiles": n,
                     "fid": frechet_distance(pos[rng.choice(len(pos), n, replace=False)],
                                             neg[rng.choice(len(neg), n, replace=False)]),
                     "kid_x1000": 1000 * kernel_inception_distance(pos, neg)})
    # Same-site split-half FID: the noise floor for a given tile budget.
    oau = feats[tile_site == "OAUTHC"]
    perm = rng.permutation(len(oau))[: 4000]
    fid_rows.append({"comparison": "noise floor: OAUTHC split-half", "n_tiles": 2000,
                     "fid": frechet_distance(oau[perm[:2000]], oau[perm[2000:4000]]),
                     "kid_x1000": 1000 * kernel_inception_distance(oau[perm[:2000]],
                                                                   oau[perm[2000:4000]])})
    stain_cols = ["mean_r", "mean_g", "mean_b", "h_r", "h_g", "h_b", "e_r", "e_g", "e_b",
                  "h_max", "e_max", "tissue_fraction", "mpp"]
    stain = frame.groupby("site")[stain_cols].median().reset_index()
    # Does colour alone identify site? (patient-grouped, 8 stain descriptors)
    cf = frame.dropna(subset=stain_cols)
    colour_pred = site_predictability(cf.reset_index(drop=True),
                                      cf[stain_cols[:-2]].to_numpy())
    # Does stain intensity move the Wagner score among MSS slides?
    wag = pd.read_csv(WAGNER_CSV)[["slide_id", "p_msih"]]
    cw = cf.merge(wag, on="slide_id")
    assoc = []
    for col in ("h_max", "e_max", "mean_r", "mean_b", "tissue_fraction"):
        for site, sub in [("ALL", cw), *cw.groupby("site")]:
            mss = sub[sub.y == 0]
            if len(mss) >= 10:
                assoc.append({"feature": col, "site": site, "n_mss_slides": len(mss),
                              "spearman_vs_wagner": mss[[col, "p_msih"]].corr("spearman").iloc[
                                  0, 1]})
    return {
        "image_kl": pd.DataFrame(kl_rows),
        "image_fid": pd.DataFrame(fid_rows),
        "image_stain_by_site": stain,
        "image_colour_site_pred": pd.DataFrame([colour_pred]),
        "image_colour_vs_wagner": pd.DataFrame(assoc),
        "image_errors": errors,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {out_dir / name}.csv ({len(df)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["embed", "dann", "calibration", "image", "slidecount"])
    p.add_argument("--encoders", nargs="*", default=ENCODERS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-perm", type=int, default=200)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.step == "embed":
        _write({"embed_shift": embed_shift(a.encoders + HARMONY, n_perm=a.n_perm),
                "baseline_per_site_wagner": baseline_per_site()}, a.out_dir)
    elif a.step == "dann":
        cohort = load_cohort()
        nested_rows, loso = [], []
        for name in a.encoders:
            frame, X = load_embedding(name, cohort)
            oof, audit = dann_nested(frame, X)
            oof.to_csv(a.out_dir / f"dann_oof_{name}.csv", index=False)
            for col in ("p_mlp", "p_dann"):
                pt = aggregate_to_patient(oof, col)
                ci = stratified_patient_bootstrap(pt)
                per_site = {f"auroc_{s}": roc_auc_score(g.y, g.score)
                            for s, g in pt.groupby("site") if g.y.nunique() == 2}
                nested_rows.append({"encoder": name, "head": col, "auroc": ci["estimate"],
                                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                                    "lams_selected": ",".join(map(str, audit.selected_lam))}
                                   | per_site)
            loso.append(dann_leave_site_out(frame, X).assign(encoder=name))
            print(f"[dann] {name} done", flush=True)
            _write({"dann_nested": pd.DataFrame(nested_rows),
                    "dann_leave_oauthc_out": pd.concat(loso)}, a.out_dir)
    elif a.step == "calibration":
        cohort = load_cohort()
        tables = {}
        slides = pd.read_csv(WAGNER_CSV)[["slide_id", "p_msih"]].merge(cohort, on="slide_id")
        pooled, per_site = nested_calibration(aggregate_to_patient(slides, "p_msih"))
        tables["calibration_wagner"] = pooled
        tables["calibration_wagner_per_site"] = per_site
        for f in sorted(a.out_dir.glob("**/dann_oof_*.csv")):
            name = f.stem.removeprefix("dann_oof_")
            oof = pd.read_csv(f)
            pooled, _ = nested_calibration(aggregate_to_patient(oof, "p_mlp"))
            tables[f"calibration_mlp_{name}"] = pooled
        _write(tables, a.out_dir)
    elif a.step == "image":
        _write(image_summary(), a.out_dir)
    elif a.step == "slidecount":
        _write({"slide_count_audit_wagner": slide_count_audit()}, a.out_dir)
    (a.out_dir / f"{a.step}.json").write_text(json.dumps(vars(a), default=str, indent=2))


if __name__ == "__main__":
    main()
