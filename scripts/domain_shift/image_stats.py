"""Per-slide image-level domain statistics for the Step-1 shift quantification.

For every slide in the pyramidal slide table, writes one shard of:

- colour histograms of tissue pixels (joint RGB 8x8x8 and per-channel HED optical
  density, 32 bins), for site-vs-reference KL divergence;
- the Macenko stain matrix (H and E optical-density vectors) and 99th-percentile
  stain concentrations, for a direct staining-protocol readout;
- ImageNet Inception-v3 pool features (2048-d) of randomly sampled tissue tiles
  (256 um^2 fields at ~1 um/px, resized to 299 px), for FID/KID between sites.

CPU-only. Run through ``scripts/domain_shift/image_stats.sh`` (SLURM array), then
``python -m argo_deepmsi.eval.domain_shift image`` to summarise.

Usage:
    python scripts/domain_shift/image_stats.py --shard 0 --n-shards 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
from skimage.color import rgb2hed, rgb2hsv
from torchvision.models import Inception_V3_Weights, inception_v3

THUMB_MAX = 2048
TILE_UM = 256.0
TILE_PX = 299
N_TILES = 32
HED_RANGE = {"H": (0.0, 0.25), "E": (0.0, 0.25), "D": (-0.05, 0.1)}


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    grey = rgb.mean(axis=-1)
    return (hsv[..., 1] > 0.07) & (grey > 30) & (grey < 235)


def macenko(pixels: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> dict:
    od = -np.log((pixels.astype(np.float64) + 1.0) / 256.0)
    od = od[(od > beta).all(axis=1)]
    if len(od) < 500:
        return {}
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, 1:3]
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    h, e = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    h, e = h * np.sign(h.sum()), e * np.sign(e.sum())
    stains = np.stack([h / np.linalg.norm(h), e / np.linalg.norm(e)], axis=1)
    conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    return {
        "h_r": stains[0, 0], "h_g": stains[1, 0], "h_b": stains[2, 0],
        "e_r": stains[0, 1], "e_g": stains[1, 1], "e_b": stains[2, 1],
        "h_max": float(np.percentile(conc[0], 99)),
        "e_max": float(np.percentile(conc[1], 99)),
    }


def slide_mpp(slide: openslide.OpenSlide) -> float:
    props = slide.properties
    for key in ("openslide.mpp-x", "aperio.MPP"):
        if key in props:
            return float(props[key])
    if props.get("tiff.ResolutionUnit") == "centimeter" and "tiff.XResolution" in props:
        return 1e4 / float(props["tiff.XResolution"])
    return 0.25


def thumbnail(slide: openslide.OpenSlide) -> tuple[np.ndarray, float]:
    w, h = slide.dimensions
    scale = max(w, h) / THUMB_MAX
    level = slide.get_best_level_for_downsample(scale)
    lw, lh = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
    img.thumbnail((THUMB_MAX, THUMB_MAX))
    arr = np.asarray(img)
    return arr, w / arr.shape[1]


def sample_tiles(slide, mask: np.ndarray, down: float, mpp: float, rng) -> list[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    size0 = TILE_UM / mpp
    level = slide.get_best_level_for_downsample(size0 / TILE_PX)
    ds = slide.level_downsamples[level]
    read = int(round(size0 / ds))
    tiles = []
    for i in rng.choice(len(ys), size=min(N_TILES * 3, len(ys)), replace=False):
        x0 = int(xs[i] * down - size0 / 2)
        y0 = int(ys[i] * down - size0 / 2)
        tile = slide.read_region((max(x0, 0), max(y0, 0)), level, (read, read)).convert("RGB")
        arr = np.asarray(tile.resize((TILE_PX, TILE_PX)))
        if tissue_mask(arr).mean() > 0.5:
            tiles.append(arr)
        if len(tiles) == N_TILES:
            break
    return tiles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--out-dir", default="results/domain_shift_cache/image_stats", type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    table = pd.read_csv(a.slide_table).iloc[a.shard :: a.n_shards]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    weights = Inception_V3_Weights.IMAGENET1K_V1
    net = inception_v3(weights=weights, aux_logits=True).eval()
    net.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    rows, hists, feats = [], [], []
    for _, r in table.iterrows():
        path = Path(r["FILENAME"])
        slide_id = path.name.removesuffix(path.suffix)
        rng = np.random.default_rng(abs(hash((a.seed, slide_id))) % 2**32)
        try:
            slide = openslide.OpenSlide(str(path))
            thumb, down = thumbnail(slide)
            mask = tissue_mask(thumb)
            px = thumb[mask]
            joint, _ = np.histogramdd(px // 32, bins=(8, 8, 8), range=((0, 8),) * 3)
            hed = rgb2hed(thumb)[mask]
            hed_h = [
                np.histogram(hed[:, i], bins=32, range=HED_RANGE[c])[0]
                for i, c in enumerate("HED")
            ]
            stain = macenko(px)
            mpp = slide_mpp(slide)
            tiles = sample_tiles(slide, mask, down, mpp, rng)
            if tiles:
                x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float() / 255.0
                with torch.no_grad():
                    f = net((x - mean) / std).numpy().astype(np.float16)
            else:
                f = np.zeros((0, 2048), dtype=np.float16)
        except Exception as exc:  # per-slide isolation: unreadable slides are recorded
            rows.append({"slide_id": slide_id, "error": repr(exc)})
            continue
        rows.append(
            {
                "slide_id": slide_id,
                "site": r["SITE"],
                "patient_id": r["PATIENT"],
                "cut_location": r["cut_location"],
                "stain_location": r["stain_location"],
                "mpp": mpp,
                "tissue_fraction": float(mask.mean()),
                "n_tiles": len(f),
                "mean_r": float(px[:, 0].mean()),
                "mean_g": float(px[:, 1].mean()),
                "mean_b": float(px[:, 2].mean()),
                **stain,
            }
        )
        hists.append(np.concatenate([joint.ravel(), *hed_h]).astype(np.float32))
        feats.append(f)
        print(f"{slide_id}: {len(f)} tiles", flush=True)

    ok = [row for row in rows if "error" not in row]
    stem = a.out_dir / f"shard{a.shard:02d}"
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    np.savez_compressed(
        f"{stem}.npz",
        slide_id=np.array([row["slide_id"] for row in ok]),
        hist=np.stack(hists) if hists else np.zeros((0, 512 + 96), np.float32),
        tile_slide=np.concatenate([[i] * len(f) for i, f in enumerate(feats)]).astype(int)
        if feats
        else np.zeros(0, int),
        inception=np.concatenate(feats) if feats else np.zeros((0, 2048), np.float16),
    )
    (a.out_dir / f"shard{a.shard:02d}.json").write_text(
        json.dumps({"n": len(rows), "ok": len(ok)}, indent=2)
    )


if __name__ == "__main__":
    main()
