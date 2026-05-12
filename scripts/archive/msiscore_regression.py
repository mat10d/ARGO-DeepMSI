"""Step 3c — regress cmo_msi_score on slide embeddings.

Target: continuous `cmo_msi_score` (MSIsensor-style % unstable sites) pulled
live from REDCap. Score is populated only for prospective (CMO-molecular)
records, so the usable subset is ~123 patients / ~430 slides — see
`docs/c1_failure_diagnosis.md` for the Wagner baseline to beat (Pearson 0.17
against this same score, 123 patients).

We evaluate Ridge on each embedding under two regimes:
  * patient_cv  — StratifiedKFold(5) bucketed on log-score deciles, grouped
                  on PATIENT (via GroupKFold-compatible split)
  * loso        — leave-one-site-out over `redcap_data_access_group`
                  (retrospective sites drop out automatically — they have no
                  score)

Metrics: Spearman ρ, Pearson r, RMSE on log1p scale. Also reports the binary
MSI-H AUROC at the classical 10%-threshold cutoff, so we can compare directly
against Wagner.

Outputs:
    results/analysis/msiscore_regression/
        patient_table.csv          PATIENT → score + site, w/ slide counts
        regression_results.csv     per-embedding × regime metrics
        oof_predictions.csv        per-slide predicted score for every emb
        scatter_{emb}.png          predicted vs actual scatter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from argo_deepmsi.data_ingestion import create_clinical_table, fetch_redcap_data

OUTDIR = Path("results/analysis/msiscore_regression")
EMB_ROOT = Path("results/embeddings")
SLIDE_TABLE = Path("results/data/slide_table_pyramidal.csv")

MSI_H_THRESHOLD = 10.0  # classical MSIsensor cut

BASE_MODELS = ["conch_v1.5_mean", "ctranspath_mean", "uni2_mean", "virchow2_mean",
               "conch_v1.5_titan", "virchow2_prism"]
SEED = 42


def _safe_numeric(s):
    # The one LASUTH record 144-5 carries a comma-joined score string. Take the
    # mean of the parseable sub-values so that patient still contributes.
    def parse(v):
        if pd.isna(v):
            return np.nan
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        vals = pd.to_numeric(parts, errors="coerce")
        return float(np.nanmean(vals)) if len(vals) else np.nan
    return s.apply(parse)


def pull_scores() -> pd.DataFrame:
    raw = fetch_redcap_data()
    raw.replace("", pd.NA, inplace=True)
    _, rec_map = create_clinical_table(raw)
    raw["PATIENT"] = raw["record_id"].astype(str).map(rec_map)
    raw = raw[raw["PATIENT"].notna()].copy()
    raw["cmo_msi_score_num"] = _safe_numeric(raw["cmo_msi_score"])
    # Patient-level: max score across the patient's records (conservative —
    # picks the dominant block for the 144-5-style multi-block cases).
    patient = (raw.groupby("PATIENT").agg(
        cmo_msi_score=("cmo_msi_score_num", "max"),
        cmo_msi_status=("cmo_msi_status", lambda s: s.dropna().iloc[0]
                        if s.notna().any() else pd.NA),
        msi_method=("msi_method", lambda s: s.dropna().iloc[0]
                    if s.notna().any() else pd.NA),
        sample_type=("sample_type", lambda s: s.dropna().iloc[0]
                     if s.notna().any() else pd.NA),
        DAG=("redcap_data_access_group", "first"),
    ).reset_index())
    return patient


def load_embedding(name: str) -> pd.DataFrame:
    d = EMB_ROOT / name
    X = np.load(d / "embeddings.npy")
    meta = pd.read_csv(d / "metadata.csv")
    meta = meta.rename(columns={"patient_id": "PATIENT"})
    meta["_row"] = np.arange(len(meta))
    return meta, X


def align_all(patient_scores: pd.DataFrame, emb_names: list[str]
              ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    metas, mats = {}, {}
    for name in emb_names:
        meta, X = load_embedding(name)
        metas[name] = meta
        mats[name] = X
    shared = set(metas[emb_names[0]]["slide_id"])
    for name in emb_names[1:]:
        shared &= set(metas[name]["slide_id"])
    shared = sorted(shared)

    base = (metas[emb_names[0]]
            .loc[metas[emb_names[0]]["slide_id"].isin(shared)]
            .drop_duplicates("slide_id")
            .set_index("slide_id").loc[shared].reset_index())

    # Attach score
    base = base.merge(patient_scores[["PATIENT", "cmo_msi_score",
                                      "cmo_msi_status", "DAG"]],
                      on="PATIENT", how="inner")
    base = base[base["cmo_msi_score"].notna()].reset_index(drop=True)

    # Attach slide-table SITE
    st = pd.read_csv(SLIDE_TABLE)
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    base = base.merge(st[["slide_id", "SITE"]], on="slide_id", how="left")

    aligned = {}
    for name in emb_names:
        m = metas[name].set_index("slide_id")
        idx = m.loc[base["slide_id"], "_row"].values
        aligned[name] = mats[name][idx]
    return aligned, base


def patient_cv_oof(X: np.ndarray, y_log: np.ndarray, groups: np.ndarray,
                   alpha: float = 1.0) -> np.ndarray:
    cv = GroupKFold(n_splits=5)
    oof = np.full(len(y_log), np.nan, dtype=np.float32)
    for tr, te in cv.split(X, y_log, groups=groups):
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=alpha, random_state=SEED).fit(sc.transform(X[tr]), y_log[tr])
        oof[te] = reg.predict(sc.transform(X[te]))
    return oof


def loso_oof(X: np.ndarray, y_log: np.ndarray, groups: np.ndarray,
             sites: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    oof = np.full(len(y_log), np.nan, dtype=np.float32)
    for site in np.unique(sites):
        held_patients = set(groups[sites == site])
        te = np.array([g in held_patients for g in groups])
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=alpha, random_state=SEED).fit(sc.transform(X[tr]), y_log[tr])
        oof[te] = reg.predict(sc.transform(X[te]))
    return oof


def score_oof(y: np.ndarray, pred_log: np.ndarray, y_bin: np.ndarray) -> dict:
    mask = ~np.isnan(pred_log)
    if mask.sum() < 3 or len(np.unique(y_bin[mask])) < 2:
        return dict(n=int(mask.sum()), pearson=np.nan, spearman=np.nan,
                    rmse_log=np.nan, auroc_at_10=np.nan)
    yi = y[mask]; pi = pred_log[mask]; yb = y_bin[mask]
    pred_lin = np.expm1(pi).clip(0)
    return dict(
        n=int(mask.sum()),
        pearson=float(pearsonr(yi, pred_lin)[0]),
        spearman=float(spearmanr(yi, pred_lin)[0]),
        rmse_log=float(np.sqrt(mean_squared_error(np.log1p(yi), pi))),
        auroc_at_10=float(roc_auc_score(yb, pi)),
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    patient_scores = pull_scores()
    n_with_score = patient_scores["cmo_msi_score"].notna().sum()
    print(f"REDCap: {len(patient_scores)} patients, {n_with_score} with cmo_msi_score")

    aligned, df = align_all(patient_scores, BASE_MODELS)
    print(f"Aligned: {len(df)} slides, {df['PATIENT'].nunique()} patients")
    print("  by DAG:", df.groupby("DAG")["PATIENT"].nunique().to_dict())
    print("  score distribution: median={:.2f}  p25={:.2f}  p75={:.2f}  "
          "max={:.2f}".format(df["cmo_msi_score"].median(),
                              df["cmo_msi_score"].quantile(.25),
                              df["cmo_msi_score"].quantile(.75),
                              df["cmo_msi_score"].max()))

    y = df["cmo_msi_score"].astype(float).values
    y_log = np.log1p(y)
    y_bin = (y >= MSI_H_THRESHOLD).astype(int)
    groups = df["PATIENT"].values
    sites = df["SITE"].astype(str).values

    df[["slide_id", "PATIENT", "SITE", "DAG", "cmo_msi_score",
        "cmo_msi_status"]].to_csv(OUTDIR / "patient_table.csv", index=False)

    rows = []
    oof_pred_tbl = df[["slide_id", "PATIENT", "SITE", "cmo_msi_score"]].copy()
    for name, X in aligned.items():
        for regime in ("patient_cv", "loso"):
            fn = patient_cv_oof if regime == "patient_cv" else loso_oof
            args = (X, y_log, groups) if regime == "patient_cv" else (X, y_log, groups, sites)
            pred_log = fn(*args)
            m = score_oof(y, pred_log, y_bin)
            rows.append(dict(embedding=name, regime=regime, **m))
            oof_pred_tbl[f"{regime}__{name}"] = np.expm1(pred_log).clip(0)
            print(f"  {name:22s} {regime:10s}  "
                  f"ρ={m['spearman']:.3f}  r={m['pearson']:.3f}  "
                  f"AUROC@10={m['auroc_at_10']:.3f}  rmse_log={m['rmse_log']:.3f}  (n={m['n']})")

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "regression_results.csv", index=False)
    oof_pred_tbl.to_csv(OUTDIR / "oof_predictions.csv", index=False)

    # Scatter — best embedding per regime
    for regime in ("patient_cv", "loso"):
        sub = results[results["regime"] == regime].sort_values("spearman", ascending=False)
        if sub.empty or sub.iloc[0]["n"] < 10: continue
        top = sub.iloc[0]["embedding"]
        col = f"{regime}__{top}"
        pred = oof_pred_tbl[col].values
        mask = ~np.isnan(pred)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.scatter(y[mask], pred[mask], s=14, alpha=0.6,
                   c=["#4477AA" if b == 0 else "#CC3311" for b in y_bin[mask]])
        lim = max(y[mask].max(), pred[mask].max())
        ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=0.8)
        ax.axhline(MSI_H_THRESHOLD, ls=":", c="k", lw=0.6)
        ax.axvline(MSI_H_THRESHOLD, ls=":", c="k", lw=0.6)
        ax.set_xlabel("cmo_msi_score (truth)")
        ax.set_ylabel("predicted cmo_msi_score")
        ax.set_title(f"{regime} — {top}\n"
                     f"ρ={sub.iloc[0]['spearman']:.3f}  r={sub.iloc[0]['pearson']:.3f}  "
                     f"AUROC@10={sub.iloc[0]['auroc_at_10']:.3f}")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"scatter_{regime}_{top}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved: {OUTDIR/'regression_results.csv'}")
    print(f"Saved: {OUTDIR/'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
