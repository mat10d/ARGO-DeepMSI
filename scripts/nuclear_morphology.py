"""C6 / D — Morphological feature engineering for MSI prediction.

Runs NuLite (lazyslide ``zs.seg.cell_types``) for nuclear segmentation +
classification on each slide, then summarises the cell-class composition
into a small interpretable feature vector per slide:

    Classes (NuLite)
        1: Neoplastic       (tumor)
        2: Inflammatory     (TILs / lymphocytes)
        3: Connective       (stroma)
        4: Dead             (necrosis)
        5: Epithelial       (normal epithelium)

    Per-slide features
        n_cells_total
        frac_neoplastic, frac_inflammatory, frac_connective, frac_dead, frac_epithelial
        til_density        = inflammatory / (inflammatory + neoplastic + 1)
        stroma_ratio       = connective / (neoplastic + 1)
        necrosis_fraction  = dead / (total + 1)
        cells_per_tile     = total / n_tiles_processed

These features are then used in a logistic-regression MSI classifier with
StratifiedGroupKFold(patient_id) to avoid slide-level leakage.

Compute strategy: NuLite is heavy. To keep the run tractable on a single GPU
we subsample tiles per slide (default 500). A random subsample preserves
expected class fractions at the slide level — the per-feature standard error
goes as 1/√k_cells but k_cells_total stays in the thousands.

Usage:
    # Inference pass (writes morphology features per slide)
    python scripts/c6_d_morphology.py extract \\
        --slide-table results/data/slide_table_pyramidal.csv \\
        --outdir results/analysis/c6_d_morphology \\
        --tiles-per-slide 500 \\
        --batch-size 8 \\
        --device cuda

    # Train + evaluate (after extract)
    python scripts/c6_d_morphology.py train \\
        --features results/analysis/c6_d_morphology/features.csv \\
        --clinical results/data/clinical_table.csv \\
        --outdir results/analysis/c6_d_morphology

Outputs:
    features.csv          slide-level morphology features (one row per slide)
    metrics.json          slide / patient AUROC, per-site
    roc.png
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
HISTOPLUS_CLASS_NAMES = [
    "Cancer cell",
    "Lymphocytes",
    "Fibroblasts",
    "Plasmocytes",
    "Eosinophils",
    "Neutrophils",
    "Macrophages",
    "Muscle Cell",
    "Endothelial Cell",
    "Red blood cell",
    "Epithelial",
    "Apoptotic Body",
    "Mitotic Figures",
    "Minor Stromal Cell",
]


# ---------------------------------------------------------------------------
# Extract pass
# ---------------------------------------------------------------------------


def open_wsi_for_seg(svs_path: str):
    from wsidata import open_wsi

    parent = str(Path(svs_path).parent)
    return open_wsi(svs_path, store=parent, attach_thumbnail=False)


def subsample_tile_table(wsi, tile_key: str, n: int, seed: int) -> "list[int]":
    tbl = wsi.tables[tile_key]
    n_tiles = tbl.shape[0]
    if n_tiles <= n:
        return list(range(n_tiles))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_tiles, size=n, replace=False).tolist())


def _slug(c: str) -> str:
    return c.lower().replace(" ", "_")


def features_from_cell_table(
    cells_df: pd.DataFrame, n_tiles_processed: int, cell_model: str = "nulite"
) -> dict:
    use_histoplus = cell_model == "histoplus"
    class_list = HISTOPLUS_CLASS_NAMES if use_histoplus else CLASS_NAMES
    counts = {c: int((cells_df["class"] == c).sum()) for c in class_list}
    total = int(sum(counts.values()))
    out = {"n_cells_total": total, "n_tiles_processed": n_tiles_processed}
    for c in class_list:
        out[f"n_{_slug(c)}"] = counts[c]
        out[f"frac_{_slug(c)}"] = counts[c] / total if total else 0.0
    if use_histoplus:
        tumor = counts["Cancer cell"]
        lymph = counts["Lymphocytes"]
        stroma = counts["Fibroblasts"] + counts["Minor Stromal Cell"]
        immune = (
            counts["Lymphocytes"]
            + counts["Plasmocytes"]
            + counts["Macrophages"]
            + counts["Neutrophils"]
            + counts["Eosinophils"]
        )
        out["til_density"] = lymph / (lymph + tumor + 1)
        out["stroma_ratio"] = stroma / (tumor + 1)
        out["immune_density"] = immune / (immune + tumor + 1)
        out["necrosis_fraction"] = counts["Apoptotic Body"] / (total + 1)
        out["mitotic_density"] = counts["Mitotic Figures"] / (total + 1)
    else:
        out["til_density"] = counts["Inflammatory"] / (
            counts["Inflammatory"] + counts["Neoplastic"] + 1
        )
        out["stroma_ratio"] = counts["Connective"] / (counts["Neoplastic"] + 1)
        out["immune_density"] = out["til_density"]
        out["necrosis_fraction"] = counts["Dead"] / (total + 1)
        out["mitotic_density"] = 0.0
    out["cells_per_tile"] = total / max(n_tiles_processed, 1)
    return out


def extract_features_one_slide(
    svs_path: str,
    tiles_per_slide: int,
    batch_size: int,
    device: str,
    seed: int,
    cell_model: str,
    big_tile_px: int,
    big_tile_mpp: float,
) -> dict | None:
    import lazyslide as zs

    try:
        wsi = open_wsi_for_seg(svs_path)
    except Exception as e:
        print(f"  open failed: {svs_path}: {e}")
        return None

    # NuLite/HistoPLUS need ≥840×840 tiles. Build a fresh tile_key at the
    # requested big-tile geometry. Skip if it already exists.
    big_key = f"tiles_{big_tile_px}_{int(big_tile_mpp*1000)}"
    if big_key not in wsi.shapes or wsi.tile_spec(big_key) is None:
        try:
            zs.pp.find_tissues(wsi)
        except Exception:
            pass  # already done
        try:
            zs.pp.tile_tissues(
                wsi,
                tile_px=big_tile_px,
                mpp=big_tile_mpp,
                key_added=big_key,
            )
        except Exception as e:
            print(f"  tile_tissues failed on {svs_path}: {e}")
            return None

    tiles_gdf = wsi.shapes[big_key]
    n_tiles = len(tiles_gdf)
    if n_tiles == 0:
        return None

    if n_tiles > tiles_per_slide:
        rng = np.random.default_rng(seed)
        sel = sorted(
            rng.choice(n_tiles, size=tiles_per_slide, replace=False).tolist()
        )
        sub_key = f"_c6d_sub_{big_tile_px}"
        sub_gdf = tiles_gdf.iloc[sel].reset_index(drop=True)
        if "tile_id" in sub_gdf.columns:
            sub_gdf = sub_gdf.assign(tile_id=range(len(sub_gdf)))
        wsi.shapes[sub_key] = sub_gdf
        spec_dict = wsi.attrs[wsi.TILE_SPEC_KEY]
        spec_dict[sub_key] = dict(spec_dict[big_key])
        wsi.attrs[wsi.TILE_SPEC_KEY] = spec_dict
        tile_key_used = sub_key
        n_used = len(sub_gdf)
    else:
        tile_key_used = big_key
        n_used = n_tiles

    try:
        zs.seg.cell_types(
            wsi,
            model=cell_model,
            tile_key=tile_key_used,
            batch_size=batch_size,
            device=device,
            num_workers=0,
            amp=False,
            pbar=False,
            key_added="c6d_cells",
        )
    except Exception as e:
        print(f"  seg failed on {svs_path}: {e}")
        return None

    cells = wsi.shapes.get("c6d_cells")
    if cells is None or len(cells) == 0:
        return {"n_cells_total": 0, "n_tiles_processed": n_used}
    return features_from_cell_table(cells, n_used, cell_model)


def extract_run(
    slide_table: Path,
    outdir: Path,
    tiles_per_slide: int,
    batch_size: int,
    device: str,
    limit: int | None,
    seed: int,
    cell_model: str = "histoplus",
    big_tile_px: int = 1024,
    big_tile_mpp: float = 0.5,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "features.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(prev["slide_id"].astype(str).tolist())
        print(f"resuming: {len(done)} slides already extracted")

    st = pd.read_csv(slide_table)
    if limit is not None:
        st = st.head(limit)

    rows: list[dict] = []
    for _, r in tqdm(st.iterrows(), total=len(st), desc="C6/D NuLite"):
        slide_id = Path(r["FILENAME"]).stem
        if slide_id in done:
            continue
        feats = extract_features_one_slide(
            r["FILENAME"],
            tiles_per_slide,
            batch_size,
            device,
            seed,
            cell_model,
            big_tile_px,
            big_tile_mpp,
        )
        if feats is None:
            continue
        rows.append(
            {
                "slide_id": slide_id,
                "patient_id": r["PATIENT"],
                "site": r["SITE"],
                "stain_location": r.get("stain_location"),
                **feats,
            }
        )
        # Periodic flush so a crash doesn't lose everything.
        if len(rows) % 25 == 0:
            df = pd.DataFrame(rows)
            if out_csv.exists():
                df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
            df.to_csv(out_csv, index=False)
            rows = []

    if rows:
        df = pd.DataFrame(rows)
        if out_csv.exists():
            df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
        df.to_csv(out_csv, index=False)

    print(f"\nFeatures written to {out_csv}")


# ---------------------------------------------------------------------------
# Train + evaluate pass
# ---------------------------------------------------------------------------

FEATURE_COLS_DERIVED = [
    "til_density",
    "stroma_ratio",
    "immune_density",
    "necrosis_fraction",
    "mitotic_density",
    "cells_per_tile",
]


def feature_cols_for(features_df: pd.DataFrame) -> list[str]:
    """Pick raw class-fraction cols + the derived ratios that are present."""
    frac_cols = [c for c in features_df.columns if c.startswith("frac_")]
    derived = [c for c in FEATURE_COLS_DERIVED if c in features_df.columns]
    return frac_cols + derived


def train_run(features_csv: Path, clinical: Path, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    outdir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(features_csv)
    cl = pd.read_csv(clinical)[["PATIENT", "isMSIH"]]
    cl["y"] = (cl["isMSIH"] == "MSI-H").astype(int)
    df = feats.merge(cl, left_on="patient_id", right_on="PATIENT", how="inner")

    feat_cols = feature_cols_for(df)
    X = df[feat_cols].fillna(0.0).to_numpy()
    y = df["y"].to_numpy()
    groups = df["patient_id"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df), dtype=np.float64)
    for tr, te in cv.split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    df["p_msih"] = oof
    df.to_csv(outdir / "slide_scores.csv", index=False)

    pat = (
        df.groupby("patient_id")
        .agg(
            p_msih=("p_msih", lambda v: float(v.max() / np.sqrt(len(v)))),
            y=("y", "first"),
            site=("site", "first"),
            n_slides=("slide_id", "count"),
        )
        .reset_index()
    )
    pat.to_csv(outdir / "patient_scores.csv", index=False)

    metrics = {
        "n_slides": int(len(df)),
        "n_patients": int(len(pat)),
        "feature_cols": feat_cols,
        "slide_auroc": float(roc_auc_score(df["y"], df["p_msih"])),
        "slide_auprc": float(average_precision_score(df["y"], df["p_msih"])),
        "patient_auroc": float(roc_auc_score(pat["y"], pat["p_msih"])),
        "patient_auprc": float(average_precision_score(pat["y"], pat["p_msih"])),
        "per_site": {},
    }
    for s, sub in df.groupby("site"):
        if sub["y"].nunique() == 2:
            metrics["per_site"][s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "auroc": float(roc_auc_score(sub["y"], sub["p_msih"])),
            }
        else:
            metrics["per_site"][s] = {
                "n": int(len(sub)),
                "prevalence": float(sub["y"].mean()),
                "note": "single-class",
            }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    for label, sub in [("slide", df), ("patient", pat)]:
        if sub["y"].nunique() == 2:
            fpr, tpr, _ = roc_curve(sub["y"], sub["p_msih"])
            auc = roc_auc_score(sub["y"], sub["p_msih"])
            ax.plot(fpr, tpr, label=f"{label} (AUROC={auc:.3f}, n={len(sub)})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("C6/D — Morphological features (NuLite)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", bbox_inches="tight")

    print(f"\nslide AUROC={metrics['slide_auroc']:.3f}  patient AUROC={metrics['patient_auroc']:.3f}")
    print("Per-site (slide):")
    for s, m in metrics["per_site"].items():
        if "auroc" in m:
            print(f"  {s:<22s}  n={m['n']:<4d} prev={m['prevalence']:.2f}  AUROC={m['auroc']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv")
    e.add_argument("--outdir", default="results/analysis/c6_d_morphology")
    e.add_argument("--tiles-per-slide", type=int, default=500)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--cell-model", default="histoplus", choices=["histoplus", "nulite"])
    e.add_argument("--big-tile-px", type=int, default=1024)
    e.add_argument("--big-tile-mpp", type=float, default=0.5)

    t = sub.add_parser("train")
    t.add_argument("--features", default="results/analysis/c6_d_morphology/features.csv")
    t.add_argument("--clinical", default="results/data/clinical_table.csv")
    t.add_argument("--outdir", default="results/analysis/c6_d_morphology")

    a = p.parse_args()
    if a.cmd == "extract":
        extract_run(
            Path(a.slide_table),
            Path(a.outdir),
            a.tiles_per_slide,
            a.batch_size,
            a.device,
            a.limit,
            a.seed,
            a.cell_model,
            a.big_tile_px,
            a.big_tile_mpp,
        )
    else:
        train_run(Path(a.features), Path(a.clinical), Path(a.outdir))


if __name__ == "__main__":
    main()
