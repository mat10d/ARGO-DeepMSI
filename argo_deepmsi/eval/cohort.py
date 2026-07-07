"""Cohort manipulation: label join, QC exclusion, patient splits.

The MSI cohort lives in two CSV files:

- ``slide_table_pyramidal.csv`` : one row per slide
- ``clinical_table.csv``       : one row per patient, with ``isMSIH``

Optionally, a pathologist QC list (``problem_slides.csv``) excludes slides
that failed manual review. Apply the exclusion early so every scorer
operates on the same clean cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_clinical(
    clinical_csv: Path, label_col: str = "isMSIH", positive: str = "MSI-H"
) -> pd.DataFrame:
    """Return ``[PATIENT, y]`` with binary MSI-H labels."""
    cl = pd.read_csv(clinical_csv)[["PATIENT", label_col]]
    cl["y"] = (cl[label_col] == positive).astype(int)
    return cl[["PATIENT", "y"]]


def load_qc_exclusion(qc_csv: Path | None) -> set[str]:
    """Return the set of slide filenames flagged ``passes_qc=False``.

    The pathologist QC list uses the slide filename (``B24_096.svs``) in
    the ``Name`` column. We match against slide_id (the file stem) by
    stripping the extension.
    """
    if qc_csv is None or not Path(qc_csv).exists():
        return set()
    df = pd.read_csv(qc_csv)
    bad = df[df["passes_qc"] == False]["Name"].astype(str)
    return {Path(n).stem for n in bad}


def build_cohort(
    slide_table_csv: Path,
    clinical_csv: Path,
    qc_csv: Path | None = None,
    label_col: str = "isMSIH",
    positive: str = "MSI-H",
) -> pd.DataFrame:
    """Slide-level table with labels, optionally filtered by QC.

    Columns guaranteed: ``slide_id, patient_id, site, y, stain_location?``.
    Other columns from the slide table are passed through.
    """
    st = pd.read_csv(slide_table_csv)
    cl = load_clinical(clinical_csv, label_col=label_col, positive=positive)
    df = st.merge(cl, left_on="PATIENT", right_on="PATIENT", how="inner")
    df["slide_id"] = df["FILENAME"].apply(lambda p: Path(p).stem)
    df = df.rename(columns={"PATIENT": "patient_id", "SITE": "site"})

    excluded = load_qc_exclusion(qc_csv)
    if excluded:
        before = len(df)
        df = df[~df["slide_id"].isin(excluded)].reset_index(drop=True)
        print(f"QC exclusion: {before} → {len(df)} slides ({before - len(df)} dropped)")
    return df


def load_tile_counts(embeddings_dir: Path | None = None) -> pd.DataFrame:
    """Return ``[slide_id, n_tiles]`` from a cached embeddings metadata.csv.

    Tile counts are model-independent, so any per-model ``metadata.csv`` under
    ``results/embeddings/`` carries the same ``n_tiles``. We prefer the widest
    (most slides) available so no scorable slide is missed, and default to the
    ``results/embeddings/`` tree next to this repo.
    """
    if embeddings_dir is None:
        embeddings_dir = Path(__file__).resolve().parents[2] / "results" / "embeddings"
    metas = sorted(Path(embeddings_dir).glob("*/metadata.csv"))
    if not metas:
        return pd.DataFrame(columns=["slide_id", "n_tiles"])
    best = max(
        (pd.read_csv(m)[["slide_id", "n_tiles"]] for m in metas),
        key=len,
    )
    return best.drop_duplicates("slide_id").reset_index(drop=True)


def build_clean_cohort(
    slide_table_csv: Path,
    clinical_csv: Path,
    ecrf_qc_csv: Path,
    embeddings_dir: Path | None = None,
    label_col: str = "isMSIH",
    positive: str = "MSI-H",
) -> tuple[pd.DataFrame, dict]:
    """Build the canonical clean-cohort table (v0: eCRF QC layer only).

    Joins the slide table to clinical labels, attaches per-slide tile counts,
    and flags the pathologist eCRF exclusion (``problem_slides.csv``,
    ``passes_qc=False``). ``in_clean_set`` is the eCRF-passing set restricted to
    slides that actually have extracted features (n_tiles > 0) — an unreadable
    slide can never enter training/scoring.

    Returns ``(cohort_df, manifest_dict)``. Columns of ``cohort_df``:
    ``slide_id, patient_id, site, y, n_tiles, passes_ecrf_qc, in_clean_set``.
    """
    df = build_cohort(slide_table_csv, clinical_csv, qc_csv=None,
                       label_col=label_col, positive=positive)
    tiles = load_tile_counts(embeddings_dir)
    df = df.merge(tiles, on="slide_id", how="left")

    excluded = load_qc_exclusion(ecrf_qc_csv)
    df["passes_ecrf_qc"] = ~df["slide_id"].isin(excluded)
    has_features = df["n_tiles"].fillna(0) > 0
    df["in_clean_set"] = (df["passes_ecrf_qc"] & has_features).astype(int)

    df["n_tiles"] = df["n_tiles"].astype("Int64")
    cols = ["slide_id", "patient_id", "site", "y", "n_tiles",
            "passes_ecrf_qc", "in_clean_set"]
    out = df[cols].sort_values(["site", "slide_id"]).reset_index(drop=True)

    per_site = {}
    for site, sub in df.groupby("site"):
        per_site[str(site)] = {
            "n_total": int(len(sub)),
            "n_dropped_ecrf": int((~sub["passes_ecrf_qc"]).sum()),
            "n_missing_features": int((sub["n_tiles"].fillna(0) == 0).sum()),
            "n_in_clean_set": int(sub["in_clean_set"].sum()),
        }
    clean = out[out["in_clean_set"] == 1]
    manifest = {
        "cohort_version": "v0-ecrf",
        "layers": ["ecrf_qc"],
        "ecrf_qc_csv": str(ecrf_qc_csv),
        "n_slides_total": int(len(out)),
        "n_slides_ecrf_pass": int(out["passes_ecrf_qc"].sum()),
        "n_slides_missing_features": int((out["n_tiles"].fillna(0) == 0).sum()),
        "n_slides_in_clean_set": int(out["in_clean_set"].sum()),
        "n_patients_total": int(out["patient_id"].nunique()),
        "n_patients_in_clean_set": int(clean["patient_id"].nunique()),
        "clean_prevalence": float(clean["y"].mean()) if len(clean) else float("nan"),
        "by_site": per_site,
    }
    return out, manifest


def patient_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = "y",
    group_col: str = "patient_id",
):
    """Yield (train_idx, test_idx) under StratifiedGroupKFold on patient_id.

    Always use patient-grouped folds — slide-level CV leaks across slides
    from the same patient.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(len(df)), df[label_col].to_numpy(), df[group_col].to_numpy()):
        yield tr, te


def main() -> None:
    """Build results/data/cohort_clean.csv + cohort_manifest.json (eCRF layer)."""
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--clinical", default="results/data/clinical_table.csv", type=Path)
    p.add_argument("--ecrf-qc", default="results/data/problem_slides.csv", type=Path)
    p.add_argument("--outdir", default="results/data", type=Path)
    a = p.parse_args()

    cohort, manifest = build_clean_cohort(a.slide_table, a.clinical, a.ecrf_qc)
    a.outdir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(a.outdir / "cohort_clean.csv", index=False)
    (a.outdir / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"cohort_clean.csv: {len(cohort)} slides, "
          f"{manifest['n_slides_in_clean_set']} in clean set "
          f"({manifest['n_patients_in_clean_set']} patients).")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
