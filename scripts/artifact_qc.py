"""GrandQC artifact-QC fraction per slide (BACKLOG Q2-artifact-qc).

For each retained (``in_clean_set``) slide, reopen its cached zarr and run
``zs.seg.artifact`` (GrandQC) over the existing ``tiles`` set. The artifact
fraction is the artifact-polygon area divided by the tissue area. This is a
FLAG-only measurement: nothing is dropped here — Q2 only records the column.

Writes one shard CSV (``artifact_qc.part<shard>.csv``) with columns
``slide_id, artifact_fraction, n_artifact_polys, artifact_qc_ok``, where
``artifact_qc_ok`` is False only on read/segmentation failure (NaN fraction).
The pass/fail threshold is applied downstream in ``eval.cohort`` so the floor
lives in one place (the manifest).

Shardable for parallel GPU jobs: ``--shard i --nshards N`` processes slice
``[i::N]`` of the retained slide list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import lazyslide as zs
from wsidata import open_wsi


def retained_slides(cohort_csv: Path, slide_table_csv: Path) -> pd.DataFrame:
    """Return ``[slide_id, source]`` for every in-clean-set slide.

    ``source`` is the WSI path from the slide table; the cached zarr lives at
    ``source.parent / f'{source.stem}.zarr'`` (reopened via ``store=parent``).
    """
    cohort = pd.read_csv(cohort_csv)
    cohort = cohort[cohort["in_clean_set"] == 1]
    st = pd.read_csv(slide_table_csv)
    st["slide_id"] = st["FILENAME"].apply(lambda p: Path(p).stem)
    src = st[["slide_id", "FILENAME"]].drop_duplicates("slide_id")
    df = cohort[["slide_id"]].merge(src, on="slide_id", how="left")
    return df.rename(columns={"FILENAME": "source"}).reset_index(drop=True)


# GrandQC artifact model runs at a coarse mpp per variant (trained on 512px
# tiles at mpp 1/1.5/2). The cached feature tiles are mpp 0.5 and incompatible,
# so we lay down a fresh coarse tile grid inside the existing tissue polygons.
VARIANT_MPP = {"5x": 2.0, "7x": 1.5, "10x": 1.0}


def slide_artifact_fraction(source: str, variant: str, device: str) -> tuple[float, int]:
    """Run GrandQC artifact seg on one slide; return (fraction, n_polys).

    Fraction = artifact-polygon area / tissue area, clipped to [0, 1]. A fresh
    coarse tile grid (``artifact_tiles``) at the model's native mpp is built
    inside the cached ``tissues`` polygons; results are read from the in-memory
    ``artifacts`` shapes (the zarr is never mutated).
    """
    source = Path(source)
    wsi = open_wsi(str(source), store=str(source.parent), attach_thumbnail=False)
    if "tissues" not in wsi.shapes:
        raise RuntimeError("no tissues shape in cached zarr")

    # Overlapping tiles so predictions are averaged across tile seams. We force
    # constant-weight blending and pass an explicit sigma_scale: lazyslide's
    # gaussian sigma heuristic divides by the tile overlap, which is None for
    # slides whose tissue fits in a single tile row/col (crashes on those).
    zs.pp.tile_tissues(
        wsi,
        tile_px=512,
        mpp=VARIANT_MPP[variant],
        overlap=0.5,
        tissue_key="tissues",
        key_added="artifact_tiles",
    )

    # Drop tissue polygons that received no tiles: the seg runner iterates every
    # tissue and leaves its prob-mask None when a tissue has zero tiles, then
    # crashes on `prob_mask /= count` (None / float). Small fragments far from
    # the coarse tile grid trigger this on ~half the cohort.
    covered = set(wsi["artifact_tiles"]["tissue_id"].unique())
    tissues = wsi.shapes["tissues"]
    wsi.shapes["tissues"] = tissues[tissues["tissue_id"].isin(covered)].reset_index(drop=True)

    zs.seg.artifact(
        wsi,
        tile_key="artifact_tiles",
        variant=variant,
        mode="constant",
        sigma_scale=1.0,
        device=device,
        key_added="artifacts",
        pbar=False,
    )

    tissue_area = float(wsi.shapes["tissues"].geometry.area.sum())
    if "artifacts" not in wsi.shapes or len(wsi.shapes["artifacts"]) == 0:
        return 0.0, 0
    arts = wsi.shapes["artifacts"]
    art_area = float(arts.geometry.area.sum())
    frac = 0.0 if tissue_area <= 0 else float(np.clip(art_area / tissue_area, 0.0, 1.0))
    return frac, int(len(arts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="results/data/cohort_clean.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--outdir", default="results/data/artifact_qc", type=Path)
    p.add_argument("--variant", default="7x")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="debug: cap slide count")
    a = p.parse_args()

    df = retained_slides(a.cohort, a.slide_table)
    df = df.iloc[a.shard :: a.nshards].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)

    a.outdir.mkdir(parents=True, exist_ok=True)
    out = a.outdir / f"artifact_qc.part{a.shard}.csv"

    # Resume across SLURM preemption: skip slides already scored, and rewrite the
    # shard CSV after every slide so a killed job never loses completed work.
    rows = []
    done_ids = set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done_ids = set(prev["slide_id"].astype(str))
        print(f"[{a.shard}] resuming: {len(done_ids)} slides already scored", flush=True)

    for i, r in df.iterrows():
        sid = str(r["slide_id"])
        if sid in done_ids:
            continue
        try:
            frac, npoly = slide_artifact_fraction(r["source"], a.variant, a.device)
            ok = True
            print(f"[{a.shard}] {i+1}/{len(df)} {sid}: frac={frac:.4f} polys={npoly}", flush=True)
        except Exception as e:  # per-slide isolation — one bad slide never kills the shard
            import traceback

            frac, npoly, ok = float("nan"), 0, False
            print(f"[{a.shard}] {i+1}/{len(df)} {sid}: FAILED {e}", flush=True)
            traceback.print_exc()
        rows.append(
            {"slide_id": sid, "artifact_fraction": frac, "n_artifact_polys": npoly, "artifact_qc_ok": ok}
        )
        pd.DataFrame(rows).to_csv(out, index=False)

    print(f"wrote {out} ({len(rows)} slides)")


if __name__ == "__main__":
    main()
