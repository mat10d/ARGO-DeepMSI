"""Unified comparison of every registered scorer on the QC-clean cohort.

Loads every scorer's canonical scores, applies the pathologist QC
exclusion (``problem_slides.csv``), evaluates with the same metrics +
per-site breakdown, and writes a single ranked table + headline plots.

This is the deliverable that answers "which approach should we pick?".

Outputs (under ``results/comparison/``):
    comparison_all.csv     # all scorers × all variants × clean / dirty
    leaderboard.csv        # primary score per scorer, clean cohort, ranked
    roc_overlay.png        # ROC curves of top-N scorers
    auroc_bar_overall.png  # bar chart of overall patient AUROC
    auroc_grid_per_site.png# heatmap scorer × site

Usage:
    python -m argo_deepmsi.eval.qc_comparison \\
        --qc-csv results/data/problem_slides.csv \\
        --slide-table results/data/slide_table_pyramidal.csv \\
        --clinical results/data/clinical_table.csv \\
        --outdir results/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..scorers import get_scorer, list_scorers
from .cohort import load_clean_exclusion, load_qc_exclusion
from .metrics import aggregate_to_patient, evaluate_scorer
from .plotting import auroc_bar, per_site_grid, roc_overlay
from .screening import screening_block


def _apply_qc_slide_level(df: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    if not excluded:
        return df
    return df[~df["slide_id"].isin(excluded)].reset_index(drop=True)


def _apply_qc_patient_level(
    df: pd.DataFrame,
    excluded_slides: set[str],
    slide_table: pd.DataFrame,
) -> pd.DataFrame:
    """For patient-resolution scorers: drop patients whose every slide is excluded."""
    if not excluded_slides:
        return df
    st = slide_table.assign(slide_id=slide_table["FILENAME"].apply(lambda p: Path(p).stem))
    bad_per_patient = st.groupby("PATIENT")["slide_id"].apply(
        lambda ids: all(s in excluded_slides for s in ids)
    )
    drop = set(bad_per_patient[bad_per_patient].index)
    return df[~df["patient_id"].isin(drop)].reset_index(drop=True)


def _safe_auroc(y, s) -> float:
    if len(np.unique(y)) != 2 or len(y) == 0:
        return float("nan")
    return float(roc_auc_score(y, s))


def evaluate_one_scorer(
    name: str,
    excluded: set[str],
    slide_table: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Evaluate one scorer dirty and clean. Returns:

    - summary row dict (one per primary variant)
    - per-site long-form DataFrame
    - ROC payload (y, scores) clean — patient resolution

    Dirty pass = compute on the full cohort.
    Clean pass = compute with ``clean_slide_ids`` restricted to the QC-kept
    slides — this triggers an actual refit for scorers whose
    ``compute_batch`` retrains a head on the provided slide set.
    """
    s = get_scorer(name)
    st = slide_table.assign(slide_id=slide_table["FILENAME"].apply(lambda p: Path(p).stem))
    all_slide_ids = set(st["slide_id"])
    kept_slide_ids = all_slide_ids - excluded

    df = s.compute_batch(pd.DataFrame())
    df_clean = s.compute_batch(pd.DataFrame(), clean_slide_ids=kept_slide_ids)

    if s.resolution == "slide":
        pat_dirty = aggregate_to_patient(df, score_col=s.primary_score)
        pat_clean = aggregate_to_patient(df_clean, score_col=s.primary_score)
    else:
        pat_dirty = df.rename(columns={s.primary_score: "score"})[["patient_id", "y", "site", "score"]]
        pat_clean = df_clean.rename(columns={s.primary_score: "score"})[["patient_id", "y", "site", "score"]]

    summary = {
        "scorer": name,
        "primary_variant": s.primary_score,
        "resolution": s.resolution,
        "needs_training_on_our_data": s.needs_training_on_our_data,
        "n_patients_dirty": int(len(pat_dirty)),
        "n_patients_clean": int(len(pat_clean)),
        "patient_auroc_dirty": _safe_auroc(pat_dirty["y"], pat_dirty["score"]),
        "patient_auroc_clean": _safe_auroc(pat_clean["y"], pat_clean["score"]),
    }
    if s.resolution == "slide":
        summary["slide_auroc_dirty"] = _safe_auroc(df["y"], df[s.primary_score])
        summary["slide_auroc_clean"] = _safe_auroc(df_clean["y"], df_clean[s.primary_score])

    # MSIntuit-comparable operating points on the clean patient scores.
    if pat_clean["y"].nunique() == 2:
        block = screening_block(pat_clean["y"].to_numpy(), pat_clean["score"].to_numpy())
        for k in ("spec_at_sens90", "spec_at_sens95", "spec_at_sens96", "npv_at_sens95"):
            summary[k] = float(block[k])
    else:
        for k in ("spec_at_sens90", "spec_at_sens95", "spec_at_sens96", "npv_at_sens95"):
            summary[k] = float("nan")

    # Per-site rows
    per_site_rows = []
    for site, sub in pat_clean.groupby("site"):
        per_site_rows.append({
            "scorer": name,
            "site": site,
            "n": int(len(sub)),
            "prevalence": float(sub["y"].mean()),
            "auroc": _safe_auroc(sub["y"], sub["score"]),
        })

    # ROC payload (patient-level, clean)
    roc_payload = {}
    if pat_clean["y"].nunique() == 2:
        roc_payload[name] = (pat_clean["y"].to_numpy(), pat_clean["score"].to_numpy())

    return summary, pd.DataFrame(per_site_rows), roc_payload


def run(
    qc_csv: Path,
    slide_table_csv: Path,
    outdir: Path,
    only: list[str] | None = None,
    clean_csv: Path | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Prefer the canonical clean cohort (all QC layers) when given; fall back to
    # the eCRF-only pathologist list for backward compatibility.
    if clean_csv is not None:
        excluded = load_clean_exclusion(clean_csv)
        exclusion_source = str(clean_csv)
        print(f"Clean-cohort exclusion: {len(excluded)} slides not in_clean_set ({clean_csv})")
    else:
        excluded = load_qc_exclusion(qc_csv)
        exclusion_source = str(qc_csv)
        print(f"QC exclusion list: {len(excluded)} slides flagged in {qc_csv}")
    slide_table = pd.read_csv(slide_table_csv)

    names = list_scorers() if only is None else only
    summaries, per_site_dfs, roc_all = [], [], {}
    for n in names:
        try:
            summary, per_site_df, roc_p = evaluate_one_scorer(n, excluded, slide_table)
        except Exception as e:
            print(f"  {n}: SKIP ({e})")
            continue
        summaries.append(summary)
        per_site_dfs.append(per_site_df)
        roc_all.update(roc_p)

    leaderboard = pd.DataFrame(summaries).sort_values(
        "patient_auroc_clean", ascending=False
    ).reset_index(drop=True)
    leaderboard.to_csv(outdir / "leaderboard.csv", index=False)
    print("\n=== Leaderboard (patient AUROC, QC-clean cohort) ===")
    cols = ["scorer", "patient_auroc_clean", "patient_auroc_dirty",
            "n_patients_clean", "needs_training_on_our_data"]
    print(leaderboard[cols].to_string(index=False))

    per_site_df = pd.concat(per_site_dfs, ignore_index=True)
    per_site_df.to_csv(outdir / "per_site_clean.csv", index=False)

    # Headline plots
    wagner_auc = float(
        leaderboard.loc[leaderboard["scorer"] == "wagner_zeroshot", "patient_auroc_clean"].iloc[0]
    ) if (leaderboard["scorer"] == "wagner_zeroshot").any() else None

    auroc_bar(
        leaderboard.rename(columns={"patient_auroc_clean": "auroc"}),
        score_col="auroc",
        label_col="scorer",
        title="MSI scorers — patient AUROC on QC-clean cohort",
        outpath=outdir / "auroc_bar_overall.png",
        baseline=wagner_auc,
    )

    # Top 5 ROC overlay
    top = leaderboard.head(5)["scorer"].tolist()
    roc_subset = {k: v for k, v in roc_all.items() if k in top}
    if roc_subset:
        roc_overlay(
            roc_subset,
            title="Top-5 scorers — patient ROC, QC-clean cohort",
            outpath=outdir / "roc_overlay.png",
            baseline_name="wagner_zeroshot",
        )

    per_site_grid(
        per_site_df.to_dict("records"),
        title="Per-site patient AUROC, QC-clean cohort",
        outpath=outdir / "auroc_grid_per_site.png",
    )

    (outdir / "metadata.json").write_text(json.dumps({
        "scorers_evaluated": [s["scorer"] for s in summaries],
        "qc_excluded_slides": len(excluded),
        "exclusion_source": exclusion_source,
        "clean_cohort": clean_csv is not None,
    }, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qc-csv", default="results/data/problem_slides.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--clean-csv", default=None, type=Path,
                   help="canonical cohort_clean.csv; race on in_clean_set instead of --qc-csv")
    p.add_argument("--outdir", default="results/comparison", type=Path)
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to a subset of scorer names")
    a = p.parse_args()
    run(a.qc_csv, a.slide_table, a.outdir, only=a.only, clean_csv=a.clean_csv)


if __name__ == "__main__":
    main()
