"""D3 — tumor-filter x aggregation 2-D ablation on champion + A0 Harmony.

Two inherited choices tested as a grid on the same 181 patients (D2 base = ALL slides,
no hard QC gate):

  tumor_floor  in {off(0.0), 0.01(current), 0.05, 0.10}  — slide-level tumor_fraction gate
  aggregator   in {max_sqrtn, mean, top3_mean, learned_lr}

  learned_lr = logistic regression over per-patient summary features [max, mean, top3,
               1/sqrt(n)], 5-fold patient-grouped OOF (a learned aggregation baseline).

Champion per-slide score = Wagner p_msih; Harmony per-slide score = OOF probe refit on
each tumor-floor slide set. Reports per-site AUROC + spec@sens95, OAUTHC-first. CPU-only.
Output: results/analysis/tumor_agg_ablation/grid_by_site.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from argo_deepmsi.eval.screening import screening_block
from argo_deepmsi.scorers.slidefm_linearprobe import _oof_full

COHORT = Path("results/data/cohort_clean.csv")
WAGNER = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
HARM = Path("results/embeddings/conch_v1.5_titan_harmony")
OUT = Path("results/analysis/tumor_agg_ablation")
OAUTHC = "OAUTHC"
FLOORS = [("off", 0.0), ("0.01", 0.01), ("0.05", 0.05), ("0.10", 0.10)]
AGGS = ["max_sqrtn", "mean", "top3_mean", "learned_lr"]
SEED = 42


def _safe_auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _patient_features(slide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in slide_df.groupby("patient_id"):
        s = np.sort(g["score"].to_numpy())[::-1]
        n = len(s)
        rows.append({
            "patient_id": pid, "y": int(g["y"].iloc[0]), "site": g["site"].iloc[0],
            "n_slides": n, "f_max": float(s.max()), "f_mean": float(s.mean()),
            "f_top3": float(s[: min(3, n)].mean()), "f_invsqrtn": float(1 / np.sqrt(n)),
        })
    return pd.DataFrame(rows)


def _aggregate(slide_df: pd.DataFrame, agg: str) -> pd.DataFrame:
    feat = _patient_features(slide_df)
    if agg == "max_sqrtn":
        feat["patient_score"] = feat["f_max"] * feat["f_invsqrtn"]
    elif agg == "mean":
        feat["patient_score"] = feat["f_mean"]
    elif agg == "top3_mean":
        feat["patient_score"] = feat["f_top3"]
    elif agg == "learned_lr":
        X = feat[["f_max", "f_mean", "f_top3", "f_invsqrtn"]].to_numpy()
        y = feat["y"].to_numpy()
        oof = np.full(len(y), np.nan)
        if len(np.unique(y)) == 2:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for tr, te in skf.split(X, y):
                lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
                lr.fit(X[tr], y[tr])
                oof[te] = lr.predict_proba(X[te])[:, 1]
        feat["patient_score"] = oof
    else:
        raise ValueError(agg)
    return feat.rename(columns={"patient_score": "score"})[["patient_id", "y", "site", "n_slides", "score"]]


def _site_rows(scorer, floor_label, agg, pat, n_slides_total, oauthc_slides):
    def block(sub):
        y, s = sub["y"].to_numpy(), sub["score"].to_numpy()
        m = sub.dropna(subset=["score"])
        b = screening_block(m["y"].to_numpy(), m["score"].to_numpy(), (0.95,)) \
            if m["y"].nunique() == 2 else {}
        return {"auroc": _safe_auroc(m["y"], m["score"]), "spec_at_sens95": b.get("spec_at_sens95", float("nan"))}
    rows = []
    for site, sub in pat.groupby("site"):
        rows.append({"scorer": scorer, "tumor_floor": floor_label, "aggregator": agg, "site": site,
                     "n_patients": int(len(sub)), "n_slides_total": n_slides_total,
                     "n_oauthc_slides": oauthc_slides, **block(sub)})
    rows.append({"scorer": scorer, "tumor_floor": floor_label, "aggregator": agg, "site": "OVERALL",
                 "n_patients": int(len(pat)), "n_slides_total": n_slides_total,
                 "n_oauthc_slides": oauthc_slides, **block(pat)})
    return rows


def main() -> None:
    cc = pd.read_csv(COHORT)
    clean_pat = set(cc.loc[cc["in_clean_set"] == 1, "patient_id"])
    base = cc[cc["patient_id"].isin(clean_pat)].copy()
    canon = base.groupby("patient_id")["site"].agg(lambda s: s.mode().iloc[0])
    base["site"] = base["patient_id"].map(canon)
    base["tf"] = pd.to_numeric(base["tumor_fraction"], errors="coerce").fillna(0.0)

    wag = pd.read_csv(WAGNER)[["slide_id", "p_msih"]]
    Xf = np.load(HARM / "embeddings.npy")
    meta = pd.read_csv(HARM / "metadata.csv"); meta["_row"] = np.arange(len(meta))

    rows = []
    for floor_label, floor in FLOORS:
        sub = base[base["tf"] >= floor] if floor > 0 else base
        n_total = int(len(sub))
        n_oa = int((sub["site"] == OAUTHC).sum())

        champ = sub.merge(wag, on="slide_id", how="inner").rename(columns={"p_msih": "score"})
        hb = sub.merge(meta[["slide_id", "_row"]], on="slide_id", how="inner").reset_index(drop=True)
        h_oof = _oof_full(Xf[hb["_row"].to_numpy()], hb["y"].to_numpy(), hb["patient_id"].to_numpy())
        harm = hb.assign(score=h_oof)

        for agg in AGGS:
            rows += _site_rows("champion", floor_label, agg, _aggregate(champ, agg), n_total, n_oa)
            rows += _site_rows("harmony", floor_label, agg, _aggregate(harm, agg), n_total, n_oa)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "grid_by_site.csv", index=False)

    print("=== D3 tumor x aggregation ablation — OAUTHC then OVERALL (AUROC) ===")
    for site in (OAUTHC, "OVERALL"):
        print(f"\n[{site}]")
        for scorer in ("harmony", "champion"):
            print(f"  {scorer}: aggregator ->")
            hdr = "    floor  " + "".join(f"{a:>12}" for a in AGGS)
            print(hdr)
            for fl, _ in FLOORS:
                cells = []
                for agg in AGGS:
                    r = df[(df.scorer == scorer) & (df.tumor_floor == fl) & (df.aggregator == agg) & (df.site == site)]
                    cells.append(f"{r['auroc'].iloc[0]:.3f}" if len(r) else "  -  ")
                print(f"    {fl:>5}  " + "".join(f"{c:>12}" for c in cells))
    print(f"\nsaved -> {OUT/'grid_by_site.csv'}")


if __name__ == "__main__":
    main()
