"""Unified comparison of every registered scorer on the canonical primary cohort.

Loads every scorer's canonical scores, evaluates the feature-complete primary
cohort plus the historical QC subset, and writes a single ranked table with
uncertainty, coverage, per-cohort breakdowns, and headline plots.

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
from .metrics import aggregate_to_patient, stratified_patient_bootstrap
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


def _cache_matches_estimand(
    scores: pd.DataFrame,
    *,
    resolution: str,
    eligible_slide_ids: set[str],
    eligible_patient_ids: set[str],
    minimum_patient_coverage: float = 0.98,
) -> bool:
    """Whether a cached score table is a valid near-complete estimand table."""
    if "patient_id" not in scores or not eligible_patient_ids:
        return False
    scored_patients = set(scores["patient_id"].astype(str))
    coverage = len(scored_patients & eligible_patient_ids) / len(eligible_patient_ids)
    if coverage < minimum_patient_coverage or not scored_patients <= eligible_patient_ids:
        return False
    if resolution == "slide":
        if "slide_id" not in scores:
            return False
        return set(scores["slide_id"].astype(str)) <= eligible_slide_ids
    return True


def evaluate_one_scorer(
    name: str,
    excluded: set[str],
    slide_table: pd.DataFrame,
    *,
    sensitivity_excluded: set[str] | None = None,
    patient_site_map: dict[str, str] | None = None,
    expected_primary_patients: set[str] | None = None,
    cache_only: bool = False,
) -> tuple[dict, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Evaluate one scorer dirty and clean. Returns:

    - summary row dict (one per primary variant)
    - per-site long-form DataFrame
    - ROC payload (y, scores) clean — patient resolution

    A cache is reused only when its IDs are contained in the requested estimand
    and it covers at least 98% of that estimand's patients. This lets the reset
    reuse exact historical QC-refit predictions while recomputing missing full-
    cohort predictions, without confusing the two populations.
    """
    s = get_scorer(name)
    st = slide_table.assign(slide_id=slide_table["FILENAME"].apply(lambda p: Path(p).stem))
    all_slide_ids = set(st["slide_id"])
    kept_slide_ids = all_slide_ids - excluded
    primary_patients = set(st.loc[st["slide_id"].isin(kept_slide_ids), "PATIENT"].astype(str))
    cached = pd.read_csv(s.score_path) if s.score_path is not None and s.score_path.exists() else None
    primary_from_cache = cached is not None and _cache_matches_estimand(
        cached,
        resolution=s.resolution,
        eligible_slide_ids=kept_slide_ids,
        eligible_patient_ids=primary_patients,
    )
    primary_was_recomputed = False
    if cache_only and cached is not None:
        df = cached.copy()
    elif cache_only and s.needs_training_on_our_data:
        raise RuntimeError("no cached scores; cache-only mode forbids learned-model fitting")
    else:
        if primary_from_cache:
            df = cached.copy()
        else:
            df = s.compute_batch(pd.DataFrame())
            primary_was_recomputed = True
    df_clean = df.copy()
    df_sensitivity = None
    sensitivity_from_cache = False
    if sensitivity_excluded is not None:
        sensitivity_ids = all_slide_ids - sensitivity_excluded
        sensitivity_patients = set(
            st.loc[st["slide_id"].isin(sensitivity_ids), "PATIENT"].astype(str)
        )
        sensitivity_from_cache = cached is not None and _cache_matches_estimand(
            cached,
            resolution=s.resolution,
            eligible_slide_ids=sensitivity_ids,
            eligible_patient_ids=sensitivity_patients,
        )
        if sensitivity_from_cache:
            df_sensitivity = cached.copy()
        elif not cache_only or not s.needs_training_on_our_data:
            df_sensitivity = s.compute_batch(
                pd.DataFrame(), clean_slide_ids=sensitivity_ids, write_outputs=False
            )

    if s.resolution == "slide":
        pat_dirty = aggregate_to_patient(
            df, score_col=s.primary_score, agg=s.patient_aggregation
        )
        pat_clean = aggregate_to_patient(
            df_clean, score_col=s.primary_score, agg=s.patient_aggregation
        )
        pat_sensitivity = (
            aggregate_to_patient(
                df_sensitivity, score_col=s.primary_score, agg=s.patient_aggregation
            ) if df_sensitivity is not None else None
        )
    else:
        pat_dirty = df.rename(columns={s.primary_score: "score"})[["patient_id", "y", "site", "score"]]
        pat_clean = df_clean.rename(columns={s.primary_score: "score"})[["patient_id", "y", "site", "score"]]
        pat_sensitivity = (
            df_sensitivity.rename(columns={s.primary_score: "score"})[
                ["patient_id", "y", "site", "score"]
            ] if df_sensitivity is not None else None
        )

    if patient_site_map:
        for patient_df in (pat_dirty, pat_clean, pat_sensitivity):
            if patient_df is not None:
                patient_df["site"] = patient_df["patient_id"].map(patient_site_map).fillna(
                    patient_df["site"]
                )

    expected_n = len(expected_primary_patients or set(pat_clean["patient_id"]))
    observed_primary = set(pat_clean["patient_id"])
    coverage = len(observed_primary & (expected_primary_patients or observed_primary)) / max(expected_n, 1)
    ci = stratified_patient_bootstrap(pat_clean, n_boot=2000)

    summary = {
        "scorer": name,
        "primary_variant": s.primary_score,
        "resolution": s.resolution,
        "needs_training_on_our_data": s.needs_training_on_our_data,
        "patient_aggregation": s.patient_aggregation if s.resolution == "slide" else "native",
        "n_patients_dirty": int(len(pat_dirty)),
        "n_patients_clean": int(len(pat_clean)),
        "patient_auroc_dirty": _safe_auroc(pat_dirty["y"], pat_dirty["score"]),
        "patient_auroc_clean": _safe_auroc(pat_clean["y"], pat_clean["score"]),
        "primary_auroc_ci_low": ci["ci_low"],
        "primary_auroc_ci_high": ci["ci_high"],
        "primary_patient_coverage": coverage,
        "comparable_primary": coverage >= 0.98,
        "confirmatory_valid": (
            name == "nested_linear_probe"
            or (
                not s.needs_training_on_our_data
                and name != "selective_abstention"
            )
        ),
        "validation_design": (
            "nested_patient_grouped"
            if name == "nested_linear_probe"
            else (
                "pretrained_or_fixed"
                if not s.needs_training_on_our_data
                else "historical_non_nested"
            )
        ),
        "primary_score_source": (
            "cache"
            if primary_from_cache
            else (
                "recomputed"
                if primary_was_recomputed
                else "historical_partial_cache"
            )
        ),
    }
    if pat_sensitivity is not None:
        summary["n_patients_qc_sensitivity"] = int(len(pat_sensitivity))
        summary["patient_auroc_qc_sensitivity"] = _safe_auroc(
            pat_sensitivity["y"], pat_sensitivity["score"]
        )
        summary["qc_sensitivity_score_source"] = (
            "cache" if sensitivity_from_cache else "recomputed"
        )
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
    cache_only: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Prefer the canonical clean cohort (all QC layers) when given; fall back to
    # the eCRF-only pathologist list for backward compatibility.
    patient_site_map: dict[str, str] | None = None
    expected_primary_patients: set[str] | None = None
    sensitivity_excluded: set[str] | None = None
    if clean_csv is not None:
        excluded = load_clean_exclusion(clean_csv)
        exclusion_source = str(clean_csv)
        print(f"Primary exclusion: {len(excluded)} feature-incomplete slides ({clean_csv})")
        cohort = pd.read_csv(clean_csv)
        inclusion_col = "in_primary_set" if "in_primary_set" in cohort else "in_clean_set"
        expected_primary_patients = set(
            cohort.loc[cohort[inclusion_col] == 1, "patient_id"].astype(str)
        )
        if "patient_cohort" in cohort:
            patient_site_map = (
                cohort.drop_duplicates("patient_id").set_index("patient_id")["patient_cohort"]
                .astype(str).to_dict()
            )
        if "in_qc_sensitivity_set" in cohort:
            sensitivity_excluded = set(
                cohort.loc[cohort["in_qc_sensitivity_set"] != 1, "slide_id"].astype(str)
            )
    else:
        excluded = load_qc_exclusion(qc_csv)
        exclusion_source = str(qc_csv)
        print(f"QC exclusion list: {len(excluded)} slides flagged in {qc_csv}")
    slide_table = pd.read_csv(slide_table_csv)

    names = list_scorers() if only is None else only
    summaries, per_site_dfs, roc_all = [], [], {}
    for n in names:
        print(f"  {n}: evaluating", flush=True)
        try:
            summary, per_site_df, roc_p = evaluate_one_scorer(
                n,
                excluded,
                slide_table,
                sensitivity_excluded=sensitivity_excluded,
                patient_site_map=patient_site_map,
                expected_primary_patients=expected_primary_patients,
                cache_only=cache_only,
            )
        except Exception as e:
            print(f"  {n}: SKIP ({e})")
            continue
        summaries.append(summary)
        per_site_dfs.append(per_site_df)
        roc_all.update(roc_p)
        print(
            f"  {n}: AUROC={summary['patient_auroc_clean']:.3f}, "
            f"coverage={summary['primary_patient_coverage']:.1%}",
            flush=True,
        )

    leaderboard = pd.DataFrame(summaries).sort_values(
        ["confirmatory_valid", "comparable_primary", "patient_auroc_clean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(outdir / "leaderboard.csv", index=False)
    print("\n=== Leaderboard (patient AUROC, primary cohort) ===")
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
        title="MSI scorers — patient AUROC on primary cohort",
        outpath=outdir / "auroc_bar_overall.png",
        baseline=wagner_auc,
    )

    # Top 5 ROC overlay
    top = leaderboard[
        leaderboard["comparable_primary"] & leaderboard["confirmatory_valid"]
    ].head(5)["scorer"].tolist()
    roc_subset = {k: v for k, v in roc_all.items() if k in top}
    if roc_subset:
        roc_overlay(
            roc_subset,
            title="Top-5 scorers — patient ROC, primary cohort",
            outpath=outdir / "roc_overlay.png",
            baseline_name="wagner_zeroshot",
        )

    per_site_grid(
        per_site_df.to_dict("records"),
        title="Per-cohort patient AUROC, primary cohort",
        outpath=outdir / "auroc_grid_per_site.png",
    )

    (outdir / "metadata.json").write_text(json.dumps({
        "scorers_evaluated": [s["scorer"] for s in summaries],
        "qc_excluded_slides": len(excluded),
        "exclusion_source": exclusion_source,
        "clean_cohort": clean_csv is not None,
        "cache_only": cache_only,
    }, indent=2))


def finalize_existing(outdir: Path, clean_csv: Path) -> None:
    """Annotate and replot an already-computed board without model inference."""
    leaderboard_path = outdir / "leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard["confirmatory_valid"] = (
        leaderboard["scorer"].eq("nested_linear_probe")
        | (
            ~leaderboard["needs_training_on_our_data"].astype(bool)
            & ~leaderboard["scorer"].eq("selective_abstention")
        )
    )
    leaderboard["validation_design"] = np.where(
        leaderboard["scorer"].eq("nested_linear_probe"),
        "nested_patient_grouped",
        np.where(
            leaderboard["needs_training_on_our_data"].astype(bool),
            "historical_non_nested",
            "pretrained_or_fixed",
        ),
    )
    leaderboard = leaderboard.sort_values(
        ["confirmatory_valid", "comparable_primary", "patient_auroc_clean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(leaderboard_path, index=False)

    wagner_auc = float(
        leaderboard.loc[
            leaderboard["scorer"] == "wagner_zeroshot", "patient_auroc_clean"
        ].iloc[0]
    )
    auroc_bar(
        leaderboard.rename(columns={"patient_auroc_clean": "auroc"}),
        score_col="auroc",
        label_col="scorer",
        title="MSI scorers — patient AUROC on primary cohort",
        outpath=outdir / "auroc_bar_overall.png",
        baseline=wagner_auc,
    )
    per_site_df = pd.read_csv(outdir / "per_site_clean.csv")
    per_site_grid(
        per_site_df.to_dict("records"),
        title="Per-cohort patient AUROC, primary cohort",
        outpath=outdir / "auroc_grid_per_site.png",
    )

    cohort = pd.read_csv(clean_csv)
    primary_ids = set(
        cohort.loc[cohort["in_primary_set"] == 1, "slide_id"].astype(str)
    )
    roc_payload = {}
    top = leaderboard[
        leaderboard["comparable_primary"] & leaderboard["confirmatory_valid"]
    ].head(5)["scorer"]
    for name in top:
        scorer = get_scorer(name)
        if scorer.score_path is None or not scorer.score_path.exists():
            continue
        scores = pd.read_csv(scorer.score_path)
        if scorer.resolution == "slide":
            scores = scores[scores["slide_id"].astype(str).isin(primary_ids)]
            patients = aggregate_to_patient(
                scores, scorer.primary_score, agg=scorer.patient_aggregation
            )
        else:
            patients = scores.rename(columns={scorer.primary_score: "score"})
        if patients["y"].nunique() == 2:
            roc_payload[name] = (
                patients["y"].to_numpy(),
                patients["score"].to_numpy(),
            )
    if roc_payload:
        roc_overlay(
            roc_payload,
            title="Confirmatory scorers — patient ROC, primary cohort",
            outpath=outdir / "roc_overlay.png",
            baseline_name="wagner_zeroshot",
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qc-csv", default="results/data/problem_slides.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--clean-csv", default=Path("results/data/cohort_clean.csv"), type=Path,
                   help="canonical cohort_clean.csv; race on in_clean_set instead of --qc-csv")
    p.add_argument("--outdir", default="results/comparison", type=Path)
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to a subset of scorer names")
    p.add_argument("--finalize-only", action="store_true",
                   help="annotate/replot the existing leaderboard without scorer inference")
    p.add_argument("--cache-only", action="store_true",
                   help="assemble cached rows; never fit a cohort-trained scorer")
    a = p.parse_args()
    if a.finalize_only:
        finalize_existing(a.outdir, a.clean_csv)
        return
    run(
        a.qc_csv,
        a.slide_table,
        a.outdir,
        only=a.only,
        clean_csv=a.clean_csv,
        cache_only=a.cache_only,
    )


if __name__ == "__main__":
    main()
