"""A2 — Macenko stain-normalize OAUTHC-prospective slides, re-extract CONCH-TITAN.

D0/A1 pin the OAUTHC deficit to a processing-pipeline batch effect that A1 showed is
NOT a linearly-removable feature offset. This is the IMAGE-level test: stain-normalize
every OAUTHC-prospective tile to a retrospective_oau reference (Macenko), RE-EXTRACT
CONCH v1.5 patch features under that normalization, and re-run TITAN aggregation — so
the correction happens at the pixel level, upstream of the frozen encoder.

Pipeline per slide (GPU):
  1. open the existing zarr (reuses the SAME tile grid used for the original extraction).
  2. build conch_v1.5 + its transform; compose Macenko(reference=retro-OAU) in front.
  3. feature_extraction(model=conch_v1.5, transform=Macenko∘conch, key_added=
     "conch_v1.5_stainnorm") -> new patch table, 1:1 with the original tiles.
  4. feature_aggregation(feature_key="conch_v1.5_stainnorm", encoder="titan") -> slide vec.
  5. persist (wsi.write) so re-runs are incremental; collect the slide embedding.

Only OAUTHC-prospective clean slides are processed (the non-OAUTHC sites already
match the reference pipeline). The A2 scorer merges these stain-normed OAUTHC vectors
with the original conch_v1.5_titan vectors for every other site.

No external data: Macenko fits on OUR retro-OAU tiles; CONCH/TITAN weights are frozen.

Modes:
  --make-ref            build results/data/stain_ref_retrooau.png (deterministic) and exit
  --shard i --nshards k process shard i of the OAUTHC clean slides (SLURM array)
  --limit N             smoke: process only the first N slides of the shard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLEAN_CSV = Path("results/data/cohort_clean.csv")
SLIDE_TABLE = Path("results/data/slide_table_pyramidal.csv")
REF_PNG = Path("results/data/stain_ref_retrooau.png")
OUT_DIR = Path("results/embeddings/conch_v1.5_titan_stainnorm")
STAINNORM_KEY = "conch_v1.5_stainnorm"
TILE_PX = 256
MPP = 0.5
SEED = 42


def _clean_slides(site: str) -> pd.DataFrame:
    cc = pd.read_csv(CLEAN_CSV)
    cc = cc[(cc["in_clean_set"] == 1) & (cc["site"] == site)].copy()
    st = pd.read_csv(SLIDE_TABLE)
    st["slide_id"] = st["FILENAME"].map(lambda p: Path(p).stem)
    merged = cc.merge(st[["slide_id", "FILENAME", "PATIENT", "SITE"]], on="slide_id", how="inner")
    return merged.sort_values("slide_id").reset_index(drop=True)


def _tissue_tile_rgb(svs_path: str, n_probe: int = 40) -> np.ndarray | None:
    """Return one representative tissue tile (HWC uint8) from a slide: the tile with
    median optical density among probed tiles (tissue-like, not background/pen)."""
    from wsidata import open_wsi

    wsi = open_wsi(svs_path, store=str(Path(svs_path).parent), attach_thumbnail=False)
    ds = wsi.ds.tile_images(tile_key="tiles", transform=None)
    n = len(ds)
    if n == 0:
        return None
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(n, size=min(n_probe, n), replace=False)
    cands = []
    for i in idxs:
        arr = np.asarray(ds[int(i)]["image"], dtype=np.uint8)  # HWC uint8
        od = -np.log((arr.astype(np.float32) + 1) / 256.0).mean()
        # skip near-white background (low OD) and near-black/pen (very high OD)
        if 0.15 < od < 1.2:
            cands.append((od, arr))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    return cands[len(cands) // 2][1]


def make_ref() -> None:
    """Deterministically build the retro-OAU stain reference tile."""
    ro = _clean_slides("retrospective_oau")
    for _, row in ro.iterrows():
        try:
            tile = _tissue_tile_rgb(row["FILENAME"])
        except Exception as e:  # noqa: BLE001
            print(f"  ref probe failed on {row['slide_id']}: {e}")
            continue
        if tile is not None:
            REF_PNG.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(tile).save(REF_PNG)
            print(f"Wrote stain reference {REF_PNG} from {row['slide_id']} shape={tile.shape}")
            return
    raise RuntimeError("no usable retro-OAU reference tile found")


def _build_normalizer():
    import torch
    from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer

    ref = np.array(Image.open(REF_PNG).convert("RGB"), dtype=np.uint8)  # writable copy
    n = TorchMacenkoNormalizer()
    n.fit(torch.from_numpy(ref).permute(2, 0, 1).float())
    return n


def _make_transform(base_tf, normalizer):
    """Compose Macenko normalization in front of CONCH's own transform.

    LazySlide's tile dataset hands the transform a HWC uint8 ndarray (exactly what
    ``model.get_transform()`` consumed during the original extraction), so we
    normalize the ndarray and pass a same-dtype ndarray downstream. Degenerate
    tiles (background / pen / SVD failure) fall back to the identity image.
    """
    import torch

    def _t(tile):
        arr = np.asarray(tile, dtype=np.uint8)
        try:
            I = torch.from_numpy(arr).permute(2, 0, 1).float()
            Inorm = normalizer.normalize(I=I, stains=False)[0]  # HWC [0,255]
            norm_arr = Inorm.numpy().clip(0, 255).astype(np.uint8)
            if not np.isfinite(norm_arr).all():
                raise ValueError("non-finite")
        except Exception:  # noqa: BLE001 — background/degenerate tile → identity
            norm_arr = arr
        return base_tf(norm_arr)

    return _t


def process_shard(shard: int, nshards: int, limit: int | None) -> None:
    import lazyslide as zs
    from lazyslide.tools._features import load_models
    from wsidata import open_wsi

    slides = _clean_slides("OAUTHC")
    mine = slides.iloc[shard::nshards].reset_index(drop=True)
    if limit is not None:
        mine = mine.head(limit)
    print(f"[shard {shard}/{nshards}] {len(mine)} OAUTHC slides")

    normalizer = _build_normalizer()
    model, _ = load_models("conch_v1.5")
    base_tf = model.get_transform()
    transform = _make_transform(base_tf, normalizer)

    rows = []
    for _, row in mine.iterrows():
        sid, svs = row["slide_id"], row["FILENAME"]
        try:
            wsi = open_wsi(svs, store=str(Path(svs).parent), attach_thumbnail=False)
            if f"{STAINNORM_KEY}_tiles" not in wsi.tables:
                zs.tl.feature_extraction(
                    wsi, model=model, transform=transform, key_added=STAINNORM_KEY,
                    device="cuda", amp=True, num_workers=0, batch_size=64, pbar=False,
                )
                wsi.write()
            zs.tl.feature_aggregation(
                wsi, feature_key=STAINNORM_KEY, encoder="titan", device="cuda"
            )
            adata = wsi.tables[STAINNORM_KEY]
            agg = adata.uns.get("agg_ops", {}).get("agg_slide", {})
            emb = np.asarray(agg["features"]).flatten()
            rows.append({
                "slide_id": sid, "patient_id": row["PATIENT"], "site": row["SITE"],
                "n_tiles": int(adata.n_obs), "zarr_path": str(Path(svs).with_suffix(".zarr")),
                "embedding": emb,
            })
            print(f"  {sid}: OK dim={emb.shape[0]} n_tiles={adata.n_obs}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sid}: FAIL {e}")
            continue

    if not rows:
        print("[shard] no embeddings produced")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.vstack([r["embedding"] for r in rows]).astype(np.float32)
    meta = pd.DataFrame([{k: r[k] for k in ("slide_id", "patient_id", "site", "n_tiles", "zarr_path")} for r in rows])
    np.save(OUT_DIR / f"shard_{shard}_of_{nshards}.npy", X)
    meta.to_csv(OUT_DIR / f"shard_{shard}_of_{nshards}.csv", index=False)
    print(f"[shard {shard}] wrote {len(rows)} embeddings -> {OUT_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--make-ref", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    if a.make_ref:
        make_ref()
    else:
        process_shard(a.shard, a.nshards, a.limit)


if __name__ == "__main__":
    main()
