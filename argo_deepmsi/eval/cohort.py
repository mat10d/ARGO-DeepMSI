"""Cohort manipulation: label join, QC exclusion, patient splits.

The MSI cohort lives in two CSV files:

- ``slide_table_pyramidal.csv`` : one row per slide
- ``clinical_table.csv``       : one row per patient, with ``isMSIH``

Optionally, a pathologist QC list (``problem_slides.csv``) excludes slides
that failed manual review. Apply the exclusion early so every scorer
operates on the same clean cohort.
"""

from __future__ import annotations

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
