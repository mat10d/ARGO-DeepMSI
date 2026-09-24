"""Locate LazySlide-vs-Mussel H-optimus-0 divergence on identical tile boxes (X1 study).

Each backend's own documented read + preprocessing path is replayed on the SAME level-0
tile boxes (sampled from Mussel's grid), in that backend's environment, without modifying
either tool:

* LazySlide (wsidata 0.11 ``TileImagesDataset``): ``reader.get_region(x, y, ops_w, ops_h,
  level=ops_level)`` via OpenSlide -> ``cv2.resize(tile, (224, 224))`` (INTER_LINEAR) when
  ``ops_w != 224`` -> ``lazyslide_models`` H-optimus-0 transform (v2 ToImage, Resize 224
  bicubic antialias [no-op at 224], CenterCrop, ToDtype, Normalize) -> ``encode_image``.
* Mussel (``WholeSlideImageTileCoordDataset``): tiffslide ``read_region(coord, 0,
  (patch_size, patch_size)).convert("RGB")`` -> ``Resize(224, BICUBIC)`` on the PIL image
  (antialiased) -> ToTensor -> Normalize -> ``OptimusModel`` forward.

Swapping reader outputs and preprocessing between the two paths, and running each model on
the other's tensors, factorises the feature difference into reader (decode), resize, and
runtime (torch/timm/kernel) effects.

Stages (run by ``scripts/backends/x1_pixel_probe.sh``):
    ls-read   (envs/lazyslide)  LazySlide raw/tensors + LazySlide-model features + self-check
    m-read    (envs/mussel)     Mussel raw/tensors + Mussel-model features on both tensor sets
    ls-cross  (envs/lazyslide)  LazySlide model on Mussel tensors and swapped preprocessing
    report    (core)            pixel/tensor/feature tables -> results/analysis/backends/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/analysis/backends")
STORES = ROOT / "stores"
PROBE = ROOT / "pixel_probe"
MEAN = (0.707223, 0.578729, 0.703617)
STD = (0.211883, 0.230117, 0.177517)


def _slides() -> pd.DataFrame:
    return pd.read_csv(ROOT / "x1_slides.csv")


def _mussel_h5(slide: Path) -> Path:
    """``argo_deepmsi.backends.mussel.output_paths`` layout (inlined: runs in envs/mussel)."""
    parent = slide.absolute().parent
    mirrored = (STORES / "mussel").absolute() / parent.relative_to(parent.anchor)
    return mirrored / f"{slide.stem}.mussel" / "hoptimus0.features.h5"


def _sample_coords(slide: Path, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    import h5py

    with h5py.File(_mussel_h5(slide), "r") as handle:
        coords = handle["coords"][:]
        attrs = {key: handle["coords"].attrs[key] for key in handle["coords"].attrs}
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(coords), size=min(k, len(coords)), replace=False))
    return idx, coords[idx], attrs


def _mussel_prep():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


# ---------------------------------------------------------------------------- LazySlide


def _ls_model():
    from argo_deepmsi.models._lazyslide import MODEL_REGISTRY

    model = MODEL_REGISTRY["h-optimus-0"]()
    model.to("cuda")
    return model


def _ls_embed(model, tensors: np.ndarray, batch: int = 32) -> np.ndarray:
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(tensors), batch):
            x = torch.from_numpy(tensors[i : i + batch]).to("cuda")
            out.append(model.encode_image(x).float().cpu().numpy())
    return np.concatenate(out)


def _ls_prep_uint8(raw: np.ndarray, ops_w: int, width: int):
    """wsidata TileImagesDataset: cv2.resize to (width, height) only if ops size differs."""
    import cv2

    return cv2.resize(raw, (width, width)) if ops_w != width else raw


def ls_read(k: int, seed: int) -> None:
    import torch
    from PIL import Image
    from wsidata import open_wsi

    from argo_deepmsi.backends.readers import lazyslide_store, read_lazyslide

    model = _ls_model()
    transform = model.get_transform()
    m_prep = _mussel_prep()
    PROBE.mkdir(parents=True, exist_ok=True)
    for row, slide in enumerate(map(Path, _slides()["FILENAME"])):
        store = lazyslide_store(slide, STORES / "lazyslide_224")
        wsi = open_wsi(str(slide), store=str(store), attach_thumbnail=False)
        spec = wsi.tile_spec("tiles")
        level, ops_w, width = int(spec.ops_level), int(spec.ops_width), int(spec.width)
        _, coords, attrs = _sample_coords(slide, k, seed)
        reader = wsi.reader
        raw = np.stack([reader.get_region(x, y, ops_w, ops_w, level=level) for x, y in coords])
        prepped = np.stack([_ls_prep_uint8(r, ops_w, width) for r in raw])
        t_ls = torch.stack([transform(p) for p in prepped]).numpy()
        t_ls_mprep = torch.stack([m_prep(Image.fromarray(r)) for r in raw]).numpy()

        # Self-check: replaying the path on LazySlide's own tiles reproduces its stored table
        native = read_lazyslide(store, "h-optimus-0")
        pick = np.random.default_rng(seed).choice(native.n_tiles, size=min(16, native.n_tiles))
        n_raw = [reader.get_region(x, y, ops_w, ops_w, level=level) for x, y in native.coords[pick]]
        n_t = torch.stack([transform(_ls_prep_uint8(r, ops_w, width)) for r in n_raw]).numpy()
        np.savez_compressed(
            PROBE / f"row{row}_ls.npz",
            coords=coords,
            raw_ls=raw,
            prep_ls=prepped,
            t_ls=t_ls,
            t_ls_mprep=t_ls_mprep,
            F_ls_ls=_ls_embed(model, t_ls),
            F_ls_lsraw_mprep=_ls_embed(model, t_ls_mprep),
            native_stored=native.features[pick],
            native_replay=_ls_embed(model, n_t),
            meta=json.dumps(
                {
                    "slide": str(slide),
                    "reader": type(reader).__name__,
                    "ops_level": level,
                    "ops_width": ops_w,
                    "width": width,
                    "base_downsample": float(spec.base_downsample),
                    "slide_mpp": wsi.properties.mpp,
                    "mussel_patch_size": int(attrs["patch_size"]),
                    "mussel_patch_level": int(attrs["patch_level"]),
                    "mussel_native_mpp": float(attrs["native_mpp"]),
                }
            ),
        )
        print(f"row {row}: ls-read done ({slide.name}, ops_w={ops_w} level={level})")


def ls_cross() -> None:
    import torch

    model = _ls_model()
    transform = model.get_transform()
    for row in range(len(_slides())):
        ls = dict(np.load(PROBE / f"row{row}_ls.npz"))
        m = dict(np.load(PROBE / f"row{row}_m.npz"))
        meta = json.loads(str(ls["meta"]))
        width = meta["width"]
        # LazySlide preprocessing applied to Mussel's raw read (size = Mussel patch_size)
        m_prepped = np.stack([_ls_prep_uint8(r, r.shape[1], width) for r in m["raw_m"]])
        t = torch.stack([transform(p) for p in m_prepped]).numpy()
        np.savez_compressed(
            PROBE / f"row{row}_cross.npz",
            prep_m_lsprep=m_prepped,
            F_ls_m=_ls_embed(model, m["t_m"]),
            F_ls_mraw_lsprep=_ls_embed(model, t),
        )
        print(f"row {row}: ls-cross done")


# ------------------------------------------------------------------------------- Mussel


def m_read(k: int, seed: int) -> None:
    import h5py
    import torch
    from mussel.models.model_factory import ModelType, get_model_factory
    from mussel.utils.wsi_backend import open_slide

    model = get_model_factory(ModelType.OPTIMUS).get_model(None, True, 0)
    preprocess = model.get_preprocessing_fun()
    model_fun = model.get_model_fun()

    def embed(tensors: np.ndarray, batch: int = 32) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch):
                out.append(np.asarray(model_fun(torch.from_numpy(tensors[i : i + batch]))))
        return np.concatenate(out).astype(np.float32)

    for row, slide in enumerate(map(Path, _slides()["FILENAME"])):
        idx, coords, attrs = _sample_coords(slide, k, seed)
        with h5py.File(_mussel_h5(slide), "r") as handle:
            stored = handle["features"][:][idx].astype(np.float32)
        wsi = open_slide(str(slide))
        ps, level = int(attrs["patch_size"]), int(attrs["patch_level"])
        imgs = [
            wsi.read_region(tuple(int(v) for v in c), level, (ps, ps)).convert("RGB")
            for c in coords
        ]
        raw = np.stack([np.asarray(img) for img in imgs])
        t_m = torch.stack([preprocess(img) for img in imgs]).numpy()
        ls = dict(np.load(PROBE / f"row{row}_ls.npz"))
        np.savez_compressed(
            PROBE / f"row{row}_m.npz",
            raw_m=raw,
            t_m=t_m,
            G_m_m=embed(t_m),
            G_m_ls=embed(ls["t_ls"]),
            stored_m=stored,
            meta=json.dumps({"reader": type(wsi).__name__}),
        )
        print(f"row {row}: m-read done ({slide.name})")


# ------------------------------------------------------------------------------- report


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _summ(prefix: str, c: np.ndarray) -> dict:
    return {f"{prefix}_median": float(np.median(c)), f"{prefix}_p05": float(np.percentile(c, 5))}


def report() -> None:
    rows = []
    for row, slide in enumerate(map(Path, _slides()["FILENAME"])):
        ls = dict(np.load(PROBE / f"row{row}_ls.npz"))
        m = dict(np.load(PROBE / f"row{row}_m.npz"))
        cross = dict(np.load(PROBE / f"row{row}_cross.npz"))
        meta = json.loads(str(ls["meta"]))
        out = {
            "row": row,
            "slide": slide.name,
            **meta,
            "m_reader": json.loads(str(m["meta"]))["reader"],
        }
        out["n_tiles"] = int(len(ls["coords"]))
        out["raw_same_shape"] = ls["raw_ls"].shape == m["raw_m"].shape
        # decoder check on the common top-left crop (boxes share the origin; sizes may
        # differ by the level-0 rounding of the tile edge)
        h = min(ls["raw_ls"].shape[1], m["raw_m"].shape[1])
        d = np.abs(
            ls["raw_ls"][:, :h, :h].astype(np.int16) - m["raw_m"][:, :h, :h].astype(np.int16)
        )
        out["raw_crop_px"] = h
        out["raw_frac_identical_px"] = float(np.mean(d.max(-1) == 0))
        out["raw_mean_abs_diff"] = float(d.mean())
        out["raw_max_abs_diff"] = int(d.max())
        # resize: same (Mussel) raw, LazySlide cv2 linear vs Mussel PIL bicubic-antialias
        denorm = (
            np.asarray(STD)[None, :, None, None] * m["t_m"] + np.asarray(MEAN)[None, :, None, None]
        )
        m_prep_u8 = np.clip(np.rint(denorm * 255), 0, 255).transpose(0, 2, 3, 1)
        rd = np.abs(cross["prep_m_lsprep"].astype(np.int16) - m_prep_u8.astype(np.int16))
        out["resize_mean_abs_diff_u8"] = float(rd.mean())
        out["tensor_max_abs_diff_ls_vs_m"] = float(np.abs(ls["t_ls"] - m["t_m"]).max())
        F_ls_ls, G_m_m = ls["F_ls_ls"], m["G_m_m"]
        out |= _summ("total_cos", _cos(F_ls_ls, G_m_m))  # LazySlide pipeline vs Mussel pipeline
        out |= _summ("mussel_replay_cos", _cos(G_m_m, m["stored_m"]))
        out |= _summ("ls_replay_cos", _cos(ls["native_replay"], ls["native_stored"]))
        # runtime: same tensors, LazySlide model/env vs Mussel model/env
        out |= _summ("runtime_cos_on_m_tensors", _cos(cross["F_ls_m"], G_m_m))
        out |= _summ("runtime_cos_on_ls_tensors", _cos(F_ls_ls, m["G_m_ls"]))
        # reader: Mussel preprocessing on each reader's raw read (LazySlide model)
        out |= _summ("reader_cos", _cos(ls["F_ls_lsraw_mprep"], cross["F_ls_m"]))
        # resize: Mussel raw, LazySlide prep vs Mussel prep (LazySlide model)
        out |= _summ("resize_cos", _cos(cross["F_ls_mraw_lsprep"], cross["F_ls_m"]))
        out["max_abs_diff_runtime"] = float(np.abs(cross["F_ls_m"] - G_m_m).max())
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "pixel_probe.csv", index=False)
    cols = [
        "slide",
        "m_reader",
        "reader",
        "ops_level",
        "ops_width",
        "mussel_patch_size",
        "raw_frac_identical_px",
        "raw_mean_abs_diff",
        "resize_mean_abs_diff_u8",
        "total_cos_median",
        "reader_cos_median",
        "resize_cos_median",
        "runtime_cos_on_m_tensors_median",
        "mussel_replay_cos_median",
        "ls_replay_cos_median",
    ]
    print(df[[c for c in cols if c in df]].to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["ls-read", "m-read", "ls-cross", "report"])
    parser.add_argument("--k", type=int, default=64, help="Tiles per slide")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "ls-read":
        ls_read(args.k, args.seed)
    elif args.stage == "m-read":
        m_read(args.k, args.seed)
    elif args.stage == "ls-cross":
        ls_cross()
    else:
        report()


if __name__ == "__main__":
    main()
