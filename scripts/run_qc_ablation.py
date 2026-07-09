"""D2 — QC-exclusion ablation: hard vs none vs soft reliability-weight.

On the SAME 181 in_clean patients, aggregate the champion (Wagner max/√n) and the A0
Harmony probe under three slide-inclusion regimes:

  hard  : only in_clean_set==1 slides (current default)
  none  : every slide of those patients (no QC exclusion)
  soft  : every slide, reliability-weighted in the max/√n aggregation

Harmony's OOF probe is refit on each regime's slide set (hard set vs full set; soft
shares the full-set fit and only changes aggregation). Reports per-site AUROC +
spec@sens95/96 and how many OAUTHC slides each regime keeps. CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from argo_deepmsi.eval.reliability_weight import patient_max_sqrtn, reliability_weight
from argo_deepmsi.eval.screening import screening_block
from argo_deepmsi.scorers.slidefm_linearprobe import _oof_full

COHORT = Path("results/data/cohort_clean.csv")
WAGNER = Path("results/scorers/wagner_zeroshot/slide_scores.csv")
HARM = Path("results/embeddings/conch_v1.5_titan_harmony")
OUT = Path("results/analysis/qc_ablation")
OAUTHC = "OAUTHC"


def _safe_auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _per_site_rows(scorer: str, regime: str, pat: pd.DataFrame, slide_set: pd.DataFrame) -> list[dict]:
    rows = []
    def _block(sub):
        y, s = sub["y"].to_numpy(), sub["score"].to_numpy()
        b = screening_block(y, s, (0.95, 0.96)) if sub["y"].nunique() == 2 else {}
        return {
            "auroc": _safe_auroc(y, s),
            "spec_at_sens95": b.get("spec_at_sens95", float("nan")),
            "spec_at_sens96": b.get("spec_at_sens96", float("nan")),
        }
    site_slidecount = slide_set.groupby("site").size().to_dict()
    for site, sub in pat.groupby("site"):
        rows.append({
            "scorer": scorer, "regime": regime, "site": site,
            "n_patients": int(len(sub)), "prevalence": float(sub["y"].mean()),
            "n_site_slides_kept": int(site_slidecount.get(site, 0)),
            **_block(sub),
        })
    rows.append({
        "scorer": scorer, "regime": regime, "site": "OVERALL",
        "n_patients": int(len(pat)), "prevalence": float(pat["y"].mean()),
        "n_site_slides_kept": int(len(slide_set)),
        **_block(pat),
    })
    return rows


def main() -> None:
    cc = pd.read_csv(COHORT)
    clean_pat = set(cc.loc[cc["in_clean_set"] == 1, "patient_id"])
    base = cc[cc["patient_id"].isin(clean_pat)].copy()
    base["w"] = reliability_weight(base)
    # Fix a CANONICAL per-patient site (mode over the patient's slides) so the per-site
    # partition is identical across regimes — otherwise a multi-site patient's site would
    # flip with the slide set and the per-site n_patients would wobble.
    canon = base.groupby("patient_id")["site"].agg(lambda s: s.mode().iloc[0])
    base["site"] = base["patient_id"].map(canon)

    # ---- champion: Wagner per-slide p_msih ----
    wag = pd.read_csv(WAGNER)[["slide_id", "p_msih"]]
    champ = base.merge(wag, on="slide_id", how="inner").rename(columns={"p_msih": "score"})

    # ---- harmony: refit OOF probe per fit-set ----
    Xf = np.load(HARM / "embeddings.npy")
    meta = pd.read_csv(HARM / "metadata.csv").reset_index(drop=True)
    meta["_row"] = np.arange(len(meta))
    hbase = base.merge(meta[["slide_id", "_row"]], on="slide_id", how="inner")

    def _harmony_oof(fit_df: pd.DataFrame) -> pd.DataFrame:
        X = Xf[fit_df["_row"].to_numpy()]
        oof = _oof_full(X, fit_df["y"].to_numpy(), fit_df["patient_id"].to_numpy())
        return fit_df.assign(score=oof)

    harm_hard = _harmony_oof(hbase[hbase["in_clean_set"] == 1].reset_index(drop=True))
    harm_all = _harmony_oof(hbase.reset_index(drop=True))

    rows = []
    # champion
    ch_hard = champ[champ["in_clean_set"] == 1]
    rows += _per_site_rows("champion", "hard", patient_max_sqrtn(ch_hard, "score"), ch_hard)
    rows += _per_site_rows("champion", "none", patient_max_sqrtn(champ, "score"), champ)
    rows += _per_site_rows("champion", "soft", patient_max_sqrtn(champ, "score", "w"), champ)
    # harmony
    rows += _per_site_rows("harmony", "hard", patient_max_sqrtn(harm_hard, "score"), harm_hard)
    rows += _per_site_rows("harmony", "none", patient_max_sqrtn(harm_all, "score"), harm_all)
    rows += _per_site_rows("harmony", "soft", patient_max_sqrtn(harm_all, "score", "w"), harm_all)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "qc_regime_by_site.csv", index=False)

    # OAUTHC-first console report
    print("=== D2 QC-exclusion ablation (OAUTHC first) ===")
    for scorer in ("harmony", "champion"):
        print(f"\n{scorer}:")
        print(f"  {'regime':>5} {'site':>18} {'n_pt':>5} {'slides':>7} {'auroc':>6} {'spec95':>7} {'spec96':>7}")
        sub = df[df["scorer"] == scorer]
        order = [OAUTHC, "OVERALL"]
        for site in order + sorted(set(sub["site"]) - set(order)):
            for regime in ("hard", "none", "soft"):
                r = sub[(sub["regime"] == regime) & (sub["site"] == site)]
                if len(r):
                    r = r.iloc[0]
                    print(f"  {regime:>5} {site:>18} {r['n_patients']:>5} {r['n_site_slides_kept']:>7} "
                          f"{r['auroc']:>6.3f} {r['spec_at_sens95']:>7.3f} {r['spec_at_sens96']:>7.3f}")
    print(f"\nsaved -> {OUT/'qc_regime_by_site.csv'}")


if __name__ == "__main__":
    main()
