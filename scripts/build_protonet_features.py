"""S2 front-end — cluster-aggregated CONCH WSI embeddings on tumor-only tiles.

Implements the FM-MSI benchmark's cluster-aggregation recipe (Computerized Medical
Imaging and Graphics 2025, PII S0895611125001892): patch embeddings are clustered
into groups, each group's embedding is averaged, and the group means are
concatenated into a single WSI embedding.

Pipeline (CPU-only; reads cached ``conch_v1.5_tiles`` from each zarr — no GPU, no
re-extraction):

1. For each in_clean_set slide, load its CONCH tile features and subset to the Q3
   tumor tile indices (``results/data/tumor_tiles/<slide>.npy``) → tumor-only tiles.
2. Fit a GLOBAL MiniBatchKMeans (G clusters) on the pooled tumor tiles of the whole
   clean cohort (subsampled per slide for the fit). This is UNSUPERVISED — it uses no
   labels, so a single global clustering is a fixed feature transform, not label
   leakage; the label-aware ProtoNet head is fit strictly train-only per CV fold in
   the scorer.
3. For each slide, assign its tumor tiles to the G clusters, take the per-cluster mean
   (empty cluster → zeros), L2-normalise each group mean, and concatenate → a
   (G × 768) WSI embedding.

Writes:
    results/scorers/protonet_cluster/cluster_features.npy   (n_slides × G*768) float32
    results/scorers/protonet_cluster/cluster_metadata.csv   slide_id, patient_id, site, n_tumor_tiles
    results/scorers/protonet_cluster/cluster_info.json      G, dim, seed, n_slides
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s2_build")

SEED = 42
G_CLUSTERS = 8
FIT_TILES_PER_SLIDE = 200  # subsample cap per slide for the global k-means fit
OUTDIR = Path("results/scorers/protonet_cluster")


def _load_conch_tiles(zarr_path: Path) -> np.ndarray | None:
    try:
        return np.asarray(ad.read_zarr(str(zarr_path / "tables" / "conch_v1.5_tiles")).X)
    except Exception as e:  # noqa: BLE001
        log.warning(f"conch read failed for {zarr_path.name}: {e}")
        return None


def _tumor_tiles(sid: str, zarr_path: Path, tumor_dir: Path) -> np.ndarray | None:
    npy = tumor_dir / f"{sid}.npy"
    if not npy.exists():
        return None
    X = _load_conch_tiles(zarr_path)
    if X is None or X.size == 0:
        return None
    idx = np.load(npy)
    idx = idx[idx < len(X)]
    if len(idx) == 0:
        return None
    return X[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="results/data/cohort_clean.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--tumor-dir", default="results/data/tumor_tiles", type=Path)
    p.add_argument("--outdir", default=OUTDIR, type=Path)
    p.add_argument("--n-clusters", default=G_CLUSTERS, type=int)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    coh = pd.read_csv(a.cohort)
    coh = coh[coh["in_clean_set"] == 1]
    st = pd.read_csv(a.slide_table)
    st["slide_id"] = st["FILENAME"].apply(lambda pth: Path(pth).stem)
    src = st[["slide_id", "FILENAME"]].drop_duplicates("slide_id")
    df = coh[["slide_id", "patient_id", "site"]].merge(src, on="slide_id", how="left")

    rng = np.random.default_rng(SEED)

    # Pass 1: load tumor tiles per slide, cache in memory, accumulate a fit subsample.
    per_slide: dict[str, np.ndarray] = {}
    meta_rows: list[dict] = []
    fit_chunks: list[np.ndarray] = []
    for _, r in df.iterrows():
        sid = r["slide_id"]
        zpath = Path(r["FILENAME"]).with_suffix(".zarr") if pd.notna(r["FILENAME"]) else None
        tiles = _tumor_tiles(sid, zpath, a.tumor_dir) if zpath and zpath.exists() else None
        if tiles is None:
            log.warning(f"{sid}: no tumor CONCH tiles — skipped")
            continue
        per_slide[sid] = tiles
        meta_rows.append({"slide_id": sid, "patient_id": r["patient_id"], "site": r["site"],
                          "n_tumor_tiles": int(len(tiles))})
        take = min(FIT_TILES_PER_SLIDE, len(tiles))
        sel = rng.choice(len(tiles), take, replace=False)
        fit_chunks.append(tiles[sel])
    if not per_slide:
        raise RuntimeError("no slides yielded tumor CONCH tiles")

    fit_X = np.concatenate(fit_chunks, axis=0)
    log.info(f"fitting MiniBatchKMeans(G={a.n_clusters}) on {fit_X.shape[0]} pooled tumor tiles")
    km = MiniBatchKMeans(n_clusters=a.n_clusters, random_state=SEED, batch_size=4096, n_init=10)
    km.fit(fit_X)

    # Pass 2: per-slide cluster-aggregated WSI embedding.
    dim = fit_X.shape[1]
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    wsi = np.zeros((len(meta), a.n_clusters * dim), dtype=np.float32)
    for i, sid in enumerate(meta["slide_id"]):
        tiles = per_slide[sid]
        lab = km.predict(tiles)
        vec = np.zeros((a.n_clusters, dim), dtype=np.float32)
        for c in range(a.n_clusters):
            m = lab == c
            if m.any():
                vec[c] = tiles[m].mean(axis=0)
        vec = normalize(vec, axis=1)  # L2 per group mean
        wsi[i] = vec.reshape(-1)

    np.save(a.outdir / "cluster_features.npy", wsi)
    meta.to_csv(a.outdir / "cluster_metadata.csv", index=False)
    (a.outdir / "cluster_info.json").write_text(json.dumps({
        "recipe": "global-kmeans cluster-agg on tumor-only CONCH tiles (concat per-group L2 means)",
        "n_clusters": int(a.n_clusters), "tile_dim": int(dim),
        "wsi_dim": int(a.n_clusters * dim), "seed": SEED,
        "n_slides": int(len(meta)), "fit_tiles_per_slide": FIT_TILES_PER_SLIDE,
        "source_paper": "Computerized Medical Imaging and Graphics 2025, PII S0895611125001892",
    }, indent=2))
    log.info(f"wrote {len(meta)} WSI cluster embeddings (dim {a.n_clusters * dim}) to {a.outdir}")


if __name__ == "__main__":
    main()
