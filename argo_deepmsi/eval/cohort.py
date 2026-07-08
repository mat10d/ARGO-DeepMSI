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


def load_clean_exclusion(clean_csv: Path | None) -> set[str]:
    """Return the slide_ids to EXCLUDE per the canonical clean cohort.

    Reads ``cohort_clean.csv`` (all 808 slides, ``in_clean_set`` 0/1 after every
    QC layer) and returns the complement of the clean set — the slides any board
    re-race must drop. This is a superset of ``load_qc_exclusion`` (eCRF only):
    it also carries the artifact-missing and tumor-filter drops.
    """
    if clean_csv is None or not Path(clean_csv).exists():
        return set()
    df = pd.read_csv(clean_csv)
    return {str(s) for s in df.loc[df["in_clean_set"] != 1, "slide_id"]}


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


def add_artifact_qc(
    cohort_csv: Path,
    artifact_qc_dir: Path,
    manifest_csv: Path,
    floor: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Merge GrandQC artifact-QC shard CSVs into cohort_clean.csv (Q2 layer).

    Reads every ``artifact_qc.part*.csv`` under ``artifact_qc_dir`` (written by
    ``scripts/artifact_qc.py``), attaches ``artifact_fraction`` per slide, and
    flags ``passes_artifact_qc = artifact_fraction <= floor`` (a slide whose seg
    failed → NaN fraction → flagged False). This layer is FLAG-only:
    ``in_clean_set`` is left untouched (dropping is deferred to Q3/Q4).

    Rewrites ``cohort_clean.csv`` in place and returns the updated
    ``(cohort_df, manifest_dict)`` with an ``artifact_qc`` block appended.
    """
    cohort = pd.read_csv(cohort_csv)
    parts = sorted(Path(artifact_qc_dir).glob("artifact_qc.part*.csv"))
    if not parts:
        raise FileNotFoundError(f"no artifact_qc.part*.csv under {artifact_qc_dir}")
    aq = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    aq = aq.drop_duplicates("slide_id", keep="last")

    cohort = cohort.drop(
        columns=[c for c in ("artifact_fraction", "passes_artifact_qc") if c in cohort.columns]
    )
    cohort = cohort.merge(
        aq[["slide_id", "artifact_fraction"]], on="slide_id", how="left"
    )
    cohort["passes_artifact_qc"] = (
        cohort["artifact_fraction"].notna() & (cohort["artifact_fraction"] <= floor)
    )
    cohort.to_csv(cohort_csv, index=False)

    manifest = json.loads(Path(manifest_csv).read_text())
    clean_mask = cohort["in_clean_set"] == 1
    n_scored = int(cohort.loc[clean_mask, "artifact_fraction"].notna().sum())
    per_site = {}
    for site, sub in cohort[clean_mask].groupby("site"):
        per_site[str(site)] = {
            "n_in_clean_set": int(len(sub)),
            "n_artifact_scored": int(sub["artifact_fraction"].notna().sum()),
            "n_flagged_artifact": int((~sub["passes_artifact_qc"]).sum()),
        }
    manifest.setdefault("layers", [])
    if "artifact_qc" not in manifest["layers"]:
        manifest["layers"].append("artifact_qc")
    manifest["artifact_qc"] = {
        "model": "grandqc-artifact",
        "reduce": "polygon_area_fraction",
        "flag_floor": floor,
        "note": "flag-only (in_clean_set unchanged); drop deferred to Q3/Q4",
        "n_clean_scored": n_scored,
        "n_clean_flagged": int((~cohort.loc[clean_mask, "passes_artifact_qc"]).sum()),
        "by_site": per_site,
    }
    Path(manifest_csv).write_text(json.dumps(manifest, indent=2))
    return cohort, manifest


def add_tumor_filter(
    cohort_csv: Path,
    tumor_tiles_dir: Path,
    manifest_csv: Path,
    method: str,
    floor: float,
    smokeoff_json: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge the Q3 tumor-tile filter into cohort_clean.csv and DROP by floor.

    Reads ``tumor_fraction.csv`` (written by ``scripts/tumor_tiles_apply.py``),
    attaches ``n_tumor_tiles`` and ``tumor_fraction`` per slide, and removes
    slides from the clean set whose ``tumor_fraction < floor`` (a slide with no
    tumor tiles carries no MSI signal). Unlike the artifact layer, this layer
    DROPS: it flips ``in_clean_set`` to 0 for below-floor slides.

    Rewrites ``cohort_clean.csv`` in place and returns ``(cohort_df, manifest)``
    with a ``tumor_filter`` block (chosen method + floor + per-site drops + the
    smoke-off metrics) appended.
    """
    cohort = pd.read_csv(cohort_csv)
    tf = pd.read_csv(Path(tumor_tiles_dir) / "tumor_fraction.csv")
    tf = tf.drop_duplicates("slide_id", keep="last")

    cohort = cohort.drop(
        columns=[c for c in ("n_tumor_tiles", "tumor_fraction") if c in cohort.columns]
    )
    cohort = cohort.merge(
        tf[["slide_id", "n_tumor_tiles", "tumor_fraction"]], on="slide_id", how="left"
    )

    was_clean = cohort["in_clean_set"] == 1
    below = was_clean & (cohort["tumor_fraction"].fillna(0.0) < floor)
    cohort.loc[below, "in_clean_set"] = 0

    per_site = {}
    for site, sub in cohort[was_clean].groupby("site"):
        s_below = sub["tumor_fraction"].fillna(0.0) < floor
        per_site[str(site)] = {
            "n_before": int(len(sub)),
            "n_dropped_tumor": int(s_below.sum()),
            "n_after": int((~s_below).sum()),
        }
    cohort.to_csv(cohort_csv, index=False)

    manifest = json.loads(Path(manifest_csv).read_text())
    manifest.setdefault("layers", [])
    if "tumor_filter" not in manifest["layers"]:
        manifest["layers"].append("tumor_filter")
    clean = cohort[cohort["in_clean_set"] == 1]
    block = {
        "method": method,
        "floor": floor,
        "reduce": "per_tile_argmax_TUM_fraction",
        "n_clean_before": int(was_clean.sum()),
        "n_dropped": int(below.sum()),
        "n_clean_after": int((cohort["in_clean_set"] == 1).sum()),
        "n_patients_after": int(clean["patient_id"].nunique()),
        "clean_prevalence_after": float(clean["y"].mean()) if len(clean) else float("nan"),
        "by_site": per_site,
    }
    if smokeoff_json is not None and Path(smokeoff_json).exists():
        block["smokeoff"] = json.loads(Path(smokeoff_json).read_text())
    manifest["tumor_filter"] = block
    Path(manifest_csv).write_text(json.dumps(manifest, indent=2))
    return cohort, manifest


def freeze_manifest(
    cohort_csv: Path,
    manifest_csv: Path,
    leaderboard_csv: Path,
) -> dict:
    """Freeze cohort_manifest.json after all QC layers land (Q4).

    Records the final clean-cohort layer counts and the no-regression floor read
    from the re-raced leaderboard (best ``patient_auroc_clean``). ``no_regression.py``
    reads ``no_regression_floor`` from here; the gate ratchets from this value.
    """
    manifest = json.loads(Path(manifest_csv).read_text())
    cohort = pd.read_csv(cohort_csv)
    clean = cohort[cohort["in_clean_set"] == 1]

    lb = pd.read_csv(leaderboard_csv)
    best_auroc = float(lb["patient_auroc_clean"].max())
    champion = str(lb.sort_values("patient_auroc_clean", ascending=False)["scorer"].iloc[0])

    per_site = {}
    for site, sub in cohort.groupby("site"):
        sub_clean = sub[sub["in_clean_set"] == 1]
        per_site[str(site)] = {
            "n_total": int(len(sub)),
            "n_in_clean_set": int(len(sub_clean)),
            "n_patients_in_clean_set": int(sub_clean["patient_id"].nunique()),
        }

    manifest["cohort_version"] = "v1-ecrf+artifact+tumor"
    manifest["frozen"] = True
    manifest["clean_cohort"] = {
        "csv": str(cohort_csv),
        "layers": list(manifest.get("layers", [])),
        "n_slides_in_clean_set": int(len(clean)),
        "n_patients_in_clean_set": int(clean["patient_id"].nunique()),
        "clean_prevalence": float(clean["y"].mean()) if len(clean) else float("nan"),
        "by_site": per_site,
    }
    manifest["no_regression_floor"] = best_auroc
    manifest["no_regression_champion"] = champion
    Path(manifest_csv).write_text(json.dumps(manifest, indent=2))
    return manifest


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
    """Build/extend results/data/cohort_clean.csv + cohort_manifest.json.

    Default mode builds the v0 eCRF cohort. Pass ``--artifact-qc-dir`` to instead
    merge the GrandQC artifact-QC layer (Q2) into an existing cohort_clean.csv.
    """
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--clinical", default="results/data/clinical_table.csv", type=Path)
    p.add_argument("--ecrf-qc", default="results/data/problem_slides.csv", type=Path)
    p.add_argument("--outdir", default="results/data", type=Path)
    p.add_argument("--artifact-qc-dir", default=None, type=Path,
                   help="if set, merge artifact-QC shards into the existing cohort (Q2)")
    p.add_argument("--artifact-floor", default=0.5, type=float)
    p.add_argument("--tumor-tiles-dir", default=None, type=Path,
                   help="if set, merge the Q3 tumor filter + drop by floor")
    p.add_argument("--tumor-method", default="ctranspath")
    p.add_argument("--tumor-floor", default=0.0, type=float)
    p.add_argument("--smokeoff-json",
                   default="results/data/tumor_smokeoff/smokeoff_metrics.json", type=Path)
    p.add_argument("--freeze", action="store_true",
                   help="freeze the manifest (Q4): set layer counts + no_regression_floor")
    p.add_argument("--leaderboard",
                   default="results/comparison/leaderboard.csv", type=Path)
    a = p.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    cohort_csv = a.outdir / "cohort_clean.csv"
    manifest_csv = a.outdir / "cohort_manifest.json"

    if a.freeze:
        manifest = freeze_manifest(cohort_csv, manifest_csv, a.leaderboard)
        cc = manifest["clean_cohort"]
        print(f"manifest frozen ({manifest['cohort_version']}): "
              f"{cc['n_slides_in_clean_set']} slides / {cc['n_patients_in_clean_set']} patients; "
              f"no_regression_floor={manifest['no_regression_floor']:.4f} "
              f"({manifest['no_regression_champion']}).")
        return

    if a.tumor_tiles_dir is not None:
        cohort, manifest = add_tumor_filter(
            cohort_csv, a.tumor_tiles_dir, manifest_csv,
            method=a.tumor_method, floor=a.tumor_floor, smokeoff_json=a.smokeoff_json)
        blk = manifest["tumor_filter"]
        print(f"tumor-filter merged ({blk['method']}, floor={blk['floor']}): "
              f"{blk['n_clean_before']} → {blk['n_clean_after']} clean slides "
              f"({blk['n_dropped']} dropped, {blk['n_patients_after']} patients).")
        print(json.dumps(blk, indent=2))
        return

    if a.artifact_qc_dir is not None:
        cohort, manifest = add_artifact_qc(cohort_csv, a.artifact_qc_dir, manifest_csv,
                                           floor=a.artifact_floor)
        blk = manifest["artifact_qc"]
        print(f"artifact-QC merged: {blk['n_clean_scored']} clean slides scored, "
              f"{blk['n_clean_flagged']} flagged (floor={blk['flag_floor']}).")
        print(json.dumps(blk, indent=2))
        return

    cohort, manifest = build_clean_cohort(a.slide_table, a.clinical, a.ecrf_qc)
    cohort.to_csv(cohort_csv, index=False)
    manifest_csv.write_text(json.dumps(manifest, indent=2))
    print(f"cohort_clean.csv: {len(cohort)} slides, "
          f"{manifest['n_slides_in_clean_set']} in clean set "
          f"({manifest['n_patients_in_clean_set']} patients).")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
