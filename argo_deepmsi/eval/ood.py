"""T2 — out-of-distribution (covariate-shift) characterization.

R1–R3 established that the OAUTHC failure is a domain-shift problem, and T1 showed a global
abstention threshold is unsafe because a low champion score on OAUTHC means "no signal", not
"confidently MSS". T2 builds the missing piece: a *site-aware* OOD score that measures how far
each slide is from the in-distribution feature manifold, so the abstention gate (T1/T3) can
key on genuine covariate shift rather than raw model confidence.

Two OOD scores on frozen TITAN (CONCH v1.5) slide features:

- **Mahalanobis** — distance to the in-distribution mean under a shrunk covariance
  (Lee et al. 2018 style): the classic parametric OOD detector.
- **kNN** — mean distance to the k nearest in-distribution neighbours (Sun et al. 2022):
  non-parametric, robust to the long-tailed, multi-cluster structure of a multi-site cohort.

Reported quantities (`ood_report`):

- **per-site OOD-AUROC** (leave-site-out): treat each site as OOD and the rest as
  in-distribution; AUROC of the OOD score separating that site's slides from the in slides.
  High ⇒ the site is a detectable covariate shift.
- **OOD↔error correlation**: does a higher OOD score predict a wrong champion call? Reported as
  the AUROC of the OOD score predicting patient-level error, and Spearman ρ.

Frozen features only. No training, no external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


def mahalanobis_ood(X_in: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each query row to the in-distribution (shrunk covariance)."""
    mu = X_in.mean(axis=0)
    cov = LedoitWolf().fit(X_in)
    prec = cov.precision_
    d = X_query - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d)).astype(np.float64)


def knn_ood(X_in: np.ndarray, X_query: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean Euclidean distance from each query row to its k nearest in-distribution rows."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(X_in))
    nn = NearestNeighbors(n_neighbors=k).fit(X_in)
    dist, _ = nn.kneighbors(X_query)
    return dist.mean(axis=1).astype(np.float64)


def _safe_auroc(y, s) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def leave_site_out_ood(X: np.ndarray, sites: np.ndarray, method: str = "knn", k: int = 10) -> dict:
    """For each site, OOD-AUROC of `method` separating that site (OOD) from the rest (in)."""
    score_fn = knn_ood if method == "knn" else mahalanobis_ood
    out = {}
    for s in np.unique(sites):
        is_ood = sites == s
        if is_ood.sum() == 0 or (~is_ood).sum() < 5:
            continue
        scores = score_fn(X[~is_ood], X, **({"k": k} if method == "knn" else {}))
        out[str(s)] = {"n": int(is_ood.sum()), "ood_auroc": _safe_auroc(is_ood, scores)}
    return out


def global_ood_scores(X: np.ndarray, sites: np.ndarray, method: str = "knn", k: int = 10) -> np.ndarray:
    """Per-slide OOD score: each slide scored against all OTHER sites' slides (leave-its-site-out)."""
    score_fn = knn_ood if method == "knn" else mahalanobis_ood
    scores = np.full(len(X), np.nan)
    for s in np.unique(sites):
        m = sites == s
        if (~m).sum() < 5:
            continue
        scores[m] = score_fn(X[~m], X[m], **({"k": k} if method == "knn" else {}))
    return scores


def ood_report(X: np.ndarray, meta: pd.DataFrame, champion_patient: pd.DataFrame | None = None,
               k: int = 10) -> dict:
    """Full OOD characterization. meta needs columns slide_id, patient_id, site, y."""
    sites = meta["site"].to_numpy()
    report: dict = {"n_slides": int(len(X)), "k": k, "methods": {}}
    for method in ("knn", "mahalanobis"):
        per_site = leave_site_out_ood(X, sites, method=method, k=k)
        gscore = global_ood_scores(X, sites, method=method, k=k)
        blk = {"per_site_ood_auroc": per_site,
               "mean_ood_auroc": float(np.nanmean([d["ood_auroc"] for d in per_site.values()]))
               if per_site else float("nan")}
        # OOD ↔ error correlation at the patient level (aggregate slide OOD by mean per patient)
        if champion_patient is not None:
            mdf = meta.assign(_ood=gscore).dropna(subset=["_ood"])
            pat_ood = mdf.groupby("patient_id")["_ood"].mean()
            cp = champion_patient.set_index("patient_id")
            common = pat_ood.index.intersection(cp.index)
            if len(common) > 5:
                thr = float(np.quantile(cp.loc[common, "p_msih"], 0.5))
                pred = (cp.loc[common, "p_msih"].to_numpy() >= thr).astype(int)
                err = (pred != cp.loc[common, "y"].to_numpy()).astype(int)
                oo = pat_ood.loc[common].to_numpy()
                blk["ood_predicts_error_auroc"] = _safe_auroc(err, oo)
                rho, _ = spearmanr(oo, err)
                blk["ood_error_spearman"] = float(rho)
        report["methods"][method] = blk
    return report


def _load_titan(clean_csv: Path, titan_dir: Path):
    X = np.load(titan_dir / "embeddings.npy").astype(np.float32)
    meta = pd.read_csv(titan_dir / "metadata.csv").copy()
    meta["_row"] = np.arange(len(meta))
    cc = pd.read_csv(clean_csv)
    clean = set(cc.loc[cc["in_clean_set"] == 1, "slide_id"])
    cl = pd.read_csv("results/data/clinical_table.csv")[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    meta = meta.merge(cl[["PATIENT", "y"]], left_on="patient_id", right_on="PATIENT", how="inner")
    meta = meta[meta["slide_id"].isin(clean)].drop_duplicates("slide_id").reset_index(drop=True)
    return X[meta["_row"].to_numpy()], meta


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--clean-csv", default="results/data/cohort_clean.csv", type=Path)
    p.add_argument("--titan-dir", default="results/embeddings/conch_v1.5_titan", type=Path)
    p.add_argument("--outdir", default="results/analysis/ood", type=Path)
    p.add_argument("--k", default=10, type=int)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    X, meta = _load_titan(a.clean_csv, a.titan_dir)
    champ = None
    cp = Path("results/scorers/selective_abstention/patient_scores.csv")
    if cp.exists():
        champ = pd.read_csv(cp)[["patient_id", "y", "p_msih"]]
    rep = ood_report(X, meta, champion_patient=champ, k=a.k)
    (a.outdir / "ood_report.json").write_text(json.dumps(rep, indent=2))

    knn = rep["methods"]["knn"]
    print(f"OOD (kNN): mean per-site OOD-AUROC {knn['mean_ood_auroc']:.3f}")
    for s, d in sorted(knn["per_site_ood_auroc"].items(), key=lambda kv: -kv[1]["ood_auroc"]):
        print(f"  {s:<20s} OOD-AUROC {d['ood_auroc']:.3f}  (n={d['n']})")
    if "ood_predicts_error_auroc" in knn:
        print(f"  OOD predicts champion error: AUROC {knn['ood_predicts_error_auroc']:.3f}, "
              f"Spearman {knn['ood_error_spearman']:+.3f}")


if __name__ == "__main__":
    main()
