"""Waiv OAUTHC race — honest nested test of the acquisition-robust encoders.

Runs the SAME repeated nested patient-grouped LR probe used for the V1 reset
(``NestedLinearProbe``) over three encoder sets, so the Waiv numbers are directly
comparable to the honest nested 4-encoder baseline (0.530), not a select-on-OOF value:

- base4      : conch_v1.5, ctranspath, uni2, virchow2 (reproduces V1)
- waiv       : phaet, mascaret
- all6       : base4 + waiv

Reports patient AUROC overall + per-site (OAUTHC-first) with a patient-bootstrap CI.
CPU-only. Output: results/analysis/error_anatomy/waiv_race/waiv_race.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import argo_deepmsi  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from argo_deepmsi.eval.metrics import aggregate_to_patient
from argo_deepmsi.eval.validation import nested_grouped_oof
from argo_deepmsi.scorers.nested_linear_probe import _factory

EMB_ROOT = Path("results/embeddings")
COHORT = "results/data/cohort_clean.csv"


def _nested_scores(emb_names):
    """Run the nested probe over the given embedding set (no scorer caching)."""
    cohort = pd.read_csv(COHORT)
    common = set(cohort["slide_id"].astype(str))
    meta, mats = {}, {}
    for name in emb_names:
        d = EMB_ROOT / name
        m = pd.read_csv(d / "metadata.csv").reset_index(drop=True)
        m["_row"] = np.arange(len(m))
        meta[name] = m
        mats[name] = np.load(d / "embeddings.npy", mmap_mode="r")
        common &= set(m["slide_id"].astype(str))
    frame = (cohort[cohort["slide_id"].isin(common)].sort_values("slide_id")
             .drop_duplicates("slide_id")[["slide_id", "patient_id", "site", "y"]]
             .reset_index(drop=True))
    cand, facs = {}, {}
    for name, m in meta.items():
        rows = m.set_index("slide_id")["_row"].loc[frame["slide_id"]].to_numpy()
        cand[name] = np.asarray(mats[name][rows])
        facs[name] = _factory
    scored, _ = nested_grouped_oof(frame, cand, facs, patient_agg="mean",
                                   outer_splits=5, inner_splits=4, repeats=3, seed=42)
    return scored

SETS = {
    "base4": ("conch_v1.5_mean", "ctranspath_mean", "uni2_mean", "virchow2_mean"),
    "waiv": ("phaet_mean", "mascaret_mean"),
    "all6": ("conch_v1.5_mean", "ctranspath_mean", "uni2_mean", "virchow2_mean",
             "phaet_mean", "mascaret_mean"),
}


def _patient_auc_by_site(scored: pd.DataFrame) -> dict:
    pat = aggregate_to_patient(scored[["patient_id", "y", "site", "p_msih"]], "p_msih", agg="mean")
    out = {"overall": float(roc_auc_score(pat["y"], pat["score"])) if pat["y"].nunique() == 2 else float("nan")}
    # patient-bootstrap CI on overall
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(1000):
        samp = pat.sample(len(pat), replace=True, random_state=int(rng.integers(1 << 31)))
        if samp["y"].nunique() == 2:
            aucs.append(roc_auc_score(samp["y"], samp["score"]))
    out["ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else [float("nan")] * 2
    by_site = {}
    for s, sub in pat.groupby("site"):
        by_site[s] = float(roc_auc_score(sub["y"], sub["score"])) if sub["y"].nunique() == 2 else None
    out["by_site"] = by_site
    return out


def main():
    results = {}
    for label, emb in SETS.items():
        missing = [e for e in emb if not (EMB_ROOT / e / "embeddings.npy").exists()]
        if missing:
            results[label] = {"error": f"missing embeddings: {missing}"}
            print(f"{label}: SKIP missing {missing}")
            continue
        scored = _nested_scores(emb)
        results[label] = _patient_auc_by_site(scored)
        r = results[label]
        oau = r["by_site"].get("OAUTHC")
        print(f"{label:6s} overall={r['overall']:.3f} CI{tuple(round(x,3) for x in r['ci95'])} "
              f"OAUTHC={oau if oau is None else round(oau,3)}")

    out = Path("results/analysis/error_anatomy/waiv_race")
    out.mkdir(parents=True, exist_ok=True)
    (out / "waiv_race.json").write_text(json.dumps(results, indent=2))
    print("\nReferences: nested base4 V1=0.530 | Wagner overall 0.717 / OAUTHC 0.616 | "
          "Harmony A0 OAUTHC 0.683 (non-nested)")


if __name__ == "__main__":
    main()
