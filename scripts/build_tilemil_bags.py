"""S4 front-end — tumor-only CONCH tile bags for attention-MIL.

Reads cached ``conch_v1.5_tiles`` from each in_clean_set slide's zarr, keeps only the
Q3 tumor tiles, caps each bag at ``MAX_TILES`` random tiles (seed 42) to bound memory,
and writes a single concatenated feature matrix + per-slide offsets:

    results/scorers/clam_tilemil/bags_concat.npy   (total_tiles × 768) float32
    results/scorers/clam_tilemil/bags_index.csv    slide_id, patient_id, site, start, length
    results/scorers/clam_tilemil/bags_info.json    model, dim, max_tiles, seed, n_slides

CPU-only; run on a compute node.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s4_bags")

SEED = 42
MAX_TILES = 500
OUTDIR = Path("results/scorers/clam_tilemil")


def _tumor_conch_tiles(zarr_path: Path, tumor_npy: Path, rng: np.random.Generator) -> np.ndarray | None:
    if not tumor_npy.exists():
        return None
    try:
        X = np.asarray(ad.read_zarr(str(zarr_path / "tables" / "conch_v1.5_tiles")).X, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        log.warning(f"conch read failed for {zarr_path.name}: {e}")
        return None
    if X.size == 0:
        return None
    idx = np.load(tumor_npy)
    idx = idx[idx < len(X)]
    if len(idx) == 0:
        return None
    if len(idx) > MAX_TILES:
        idx = rng.choice(idx, MAX_TILES, replace=False)
    return X[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="results/data/cohort_clean.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--tumor-dir", default="results/data/tumor_tiles", type=Path)
    p.add_argument("--outdir", default=OUTDIR, type=Path)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    coh = pd.read_csv(a.cohort)
    coh = coh[coh["in_clean_set"] == 1]
    st = pd.read_csv(a.slide_table)
    st["slide_id"] = st["FILENAME"].apply(lambda pth: Path(pth).stem)
    src = st[["slide_id", "FILENAME"]].drop_duplicates("slide_id")
    df = coh[["slide_id", "patient_id", "site"]].merge(src, on="slide_id", how="left")

    rng = np.random.default_rng(SEED)
    chunks, index_rows, cursor = [], [], 0
    for _, r in df.iterrows():
        sid = r["slide_id"]
        zpath = Path(r["FILENAME"]).with_suffix(".zarr") if pd.notna(r["FILENAME"]) else None
        tiles = _tumor_conch_tiles(zpath, a.tumor_dir / f"{sid}.npy", rng) if zpath and zpath.exists() else None
        if tiles is None:
            log.warning(f"{sid}: no tumor CONCH tiles — skipped")
            continue
        chunks.append(tiles)
        index_rows.append({"slide_id": sid, "patient_id": r["patient_id"], "site": r["site"],
                           "start": cursor, "length": int(len(tiles))})
        cursor += len(tiles)

    if not chunks:
        raise RuntimeError("no slides yielded tumor CONCH tiles")
    bags = np.concatenate(chunks, axis=0).astype(np.float32)
    idx = pd.DataFrame(index_rows)
    np.save(a.outdir / "bags_concat.npy", bags)
    idx.to_csv(a.outdir / "bags_index.csv", index=False)
    (a.outdir / "bags_info.json").write_text(json.dumps({
        "model": "conch_v1.5", "dim": int(bags.shape[1]), "max_tiles": MAX_TILES,
        "seed": SEED, "n_slides": int(len(idx)), "total_tiles": int(len(bags)),
    }, indent=2))
    log.info(f"wrote {len(idx)} bags ({len(bags)} tiles, dim {bags.shape[1]}) to {a.outdir}")


if __name__ == "__main__":
    main()
