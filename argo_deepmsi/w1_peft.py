"""Raw-tile CTransPath adaptation primitives for W1.

The efficient W1 path replaces a small, aligned subset of cached tile features
with differentiably re-encoded features.  This preserves slide context while
bounding image-encoder memory and I/O during the PEFT feasibility stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from wsidata import open_wsi

from .models.ctranspath import CTransPathMode, TrainableCTransPath
from .models.wagner import configure_wagner_trainable, load_wagner
from .w1 import _stable_seed


@dataclass(frozen=True)
class MixedBagSample:
    cached_features: np.ndarray
    raw_positions: np.ndarray
    source_tile_indices: np.ndarray


@dataclass(frozen=True)
class FixedMixedBagSample:
    cached_features: np.ndarray
    raw_positions: np.ndarray
    source_tile_indices: np.ndarray
    raw_images: np.ndarray


def _tile_rows_in_feature_order(wsi) -> pd.DataFrame:
    """Return tile geometry in the row order used by LazySlide extraction.

    The source indices stored with the W1 feature reservoir are positions in the
    CTransPath feature table, whose rows follow the tile-shape table.  Older Zarr
    stores do not consistently expose a ``tile_id`` column, so row position is
    the portable alignment contract and must not be replaced by ID sorting.
    """
    tiles = wsi.shapes["tiles"]
    if not isinstance(tiles, pd.DataFrame):
        raise TypeError("WSI 'tiles' shape must be a pandas DataFrame")
    return tiles.reset_index(drop=True)


class MixedBagStore:
    """Aligned cached features and original tile indices for PEFT sampling."""

    def __init__(
        self,
        cache_dir: str | Path,
        slide_index: pd.DataFrame,
        *,
        seed: int = 42,
    ):
        cache_dir = Path(cache_dir)
        self.features = np.load(cache_dir / "bags.npy", mmap_mode="r")
        self.source_indices = np.load(
            cache_dir / "tile_indices.npy", mmap_mode="r"
        )
        cache_index = pd.read_csv(cache_dir / "bags_index.csv")
        self.index = slide_index.reset_index(drop=True).merge(
            cache_index, on="slide_id", how="left", validate="one_to_one"
        )
        if self.index[["start", "length"]].isna().any().any():
            raise ValueError("mixed PEFT cache does not cover every W1 slide")
        self.seed = int(seed)

    def read(
        self,
        row_index: int,
        *,
        epoch: int,
        bag_tiles: int,
        raw_tiles: int,
    ) -> MixedBagSample:
        row = self.index.iloc[int(row_index)]
        start, length = int(row["start"]), int(row["length"])
        positions = np.arange(length, dtype=np.int64)
        if len(positions) > bag_tiles:
            rng = np.random.default_rng(
                _stable_seed(str(row["slide_id"]), self.seed, int(epoch))
            )
            positions = np.sort(rng.choice(positions, bag_tiles, replace=False))
        n_raw = min(int(raw_tiles), len(positions))
        raw_rng = np.random.default_rng(
            _stable_seed(str(row["slide_id"]), self.seed + 17_003, int(epoch))
        )
        raw_positions = np.sort(
            raw_rng.choice(len(positions), n_raw, replace=False)
        )
        absolute = start + positions
        return MixedBagSample(
            cached_features=np.array(self.features[absolute], dtype=np.float32, copy=True),
            raw_positions=raw_positions,
            source_tile_indices=np.asarray(
                self.source_indices[absolute[raw_positions]], dtype=np.int64
            ),
        )


def _extract_raw_tile_job(job) -> np.ndarray:
    """Worker helper used by ``build_fixed_raw_tile_cache``."""
    filename, source_indices = job
    path = Path(filename)
    wsi = open_wsi(str(path), store=str(path.parent), attach_thumbnail=False)
    spec = wsi.tile_spec("tiles")
    tiles = _tile_rows_in_feature_order(wsi)
    images = []
    for tile_index in source_indices:
        tile = tiles.iloc[int(tile_index)]
        x, y = tile.geometry.bounds[:2]
        image = wsi.reader.get_region(
            x,
            y,
            spec.ops_width,
            spec.ops_height,
            level=spec.ops_level,
        )
        images.append(
            wsi.reader.resize_img(image, dsize=(spec.width, spec.height))
        )
    return np.stack(images).astype(np.uint8, copy=False)


def build_fixed_raw_tile_cache(
    slide_index: pd.DataFrame,
    *,
    feature_cache_dir: str | Path,
    outdir: str | Path,
    raw_tiles: int = 8,
    seed: int = 42,
    workers: int = 8,
) -> Path:
    """Extract a fixed aligned raw-tile subset once for efficient PEFT."""
    from concurrent.futures import ProcessPoolExecutor

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    feature_store = MixedBagStore(feature_cache_dir, slide_index, seed=seed)
    jobs = []
    rows = []
    cursor = 0
    for row_index, row in slide_index.reset_index(drop=True).iterrows():
        sample = feature_store.read(
            row_index,
            epoch=-1,
            bag_tiles=1024,
            raw_tiles=raw_tiles,
        )
        jobs.append((str(row["FILENAME"]), sample.source_tile_indices.tolist()))
        rows.append(
            {
                "slide_id": row["slide_id"],
                "start": cursor,
                "length": len(sample.source_tile_indices),
                "reservoir_positions": json.dumps(sample.raw_positions.tolist()),
                "source_tile_indices": json.dumps(
                    sample.source_tile_indices.tolist()
                ),
            }
        )
        cursor += len(sample.source_tile_indices)

    image_path = outdir / "images.npy"
    partial_image_path = outdir / "images.partial.npy"
    images = np.lib.format.open_memmap(
        partial_image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(cursor, 256, 256, 3),
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row, extracted in zip(rows, executor.map(_extract_raw_tile_job, jobs)):
            start, length = int(row["start"]), int(row["length"])
            if extracted.shape != (length, 256, 256, 3):
                raise RuntimeError(
                    f"{row['slide_id']}: unexpected raw cache shape {extracted.shape}"
                )
            images[start : start + length] = extracted
    images.flush()
    del images
    partial_index_path = outdir / "index.partial.csv"
    pd.DataFrame(rows).to_csv(partial_index_path, index=False)
    metadata = {
        "n_slides": len(rows),
        "n_images": cursor,
        "raw_tiles_per_slide": raw_tiles,
        "height": 256,
        "width": 256,
        "channels": 3,
        "dtype": "uint8",
        "seed": seed,
        "workers": workers,
        "slide_index_sha256": hashlib.sha256(
            slide_index[["slide_id", "patient_id", "y", "outer_fold"]]
            .to_csv(index=False, lineterminator="\n")
            .encode()
        ).hexdigest(),
    }
    partial_metadata_path = outdir / "metadata.partial.json"
    partial_metadata_path.write_text(json.dumps(metadata, indent=2))
    # Publish only after every slide has been extracted. Consumers require all
    # three files, so an interrupted rebuild cannot look like a complete cache.
    partial_image_path.replace(image_path)
    partial_index_path.replace(outdir / "index.csv")
    partial_metadata_path.replace(outdir / "metadata.json")
    return image_path


def fixed_raw_tile_cache_ready(raw_cache_dir: str | Path) -> bool:
    """Return whether a fixed raw cache has published all required artifacts."""
    raw_cache_dir = Path(raw_cache_dir)
    return all(
        (raw_cache_dir / filename).exists()
        for filename in ("images.npy", "index.csv", "metadata.json")
    )


class FixedRawTileStore:
    """Fast PEFT store using cached context and a fixed aligned raw subset."""

    def __init__(
        self,
        feature_cache_dir: str | Path,
        raw_cache_dir: str | Path,
        slide_index: pd.DataFrame,
        *,
        seed: int = 42,
    ):
        feature_cache_dir = Path(feature_cache_dir)
        raw_cache_dir = Path(raw_cache_dir)
        self.features = np.load(feature_cache_dir / "bags.npy", mmap_mode="r")
        self.source_indices = np.load(
            feature_cache_dir / "tile_indices.npy", mmap_mode="r"
        )
        feature_index = pd.read_csv(feature_cache_dir / "bags_index.csv").rename(
            columns={"start": "feature_start", "length": "feature_length"}
        )
        raw_index = pd.read_csv(raw_cache_dir / "index.csv").rename(
            columns={"start": "raw_start", "length": "raw_length"}
        )
        self.images = np.load(raw_cache_dir / "images.npy", mmap_mode="r")
        self.index = (
            slide_index.reset_index(drop=True)
            .merge(feature_index, on="slide_id", how="left", validate="one_to_one")
            .merge(raw_index, on="slide_id", how="left", validate="one_to_one")
        )
        if self.index[
            ["feature_start", "feature_length", "raw_start", "raw_length"]
        ].isna().any().any():
            raise ValueError("fixed raw cache does not cover every W1 slide")
        self.seed = int(seed)

    def read(
        self,
        row_index: int,
        *,
        epoch: int,
        bag_tiles: int,
        raw_tiles: int,
    ) -> FixedMixedBagSample:
        row = self.index.iloc[int(row_index)]
        feature_start = int(row["feature_start"])
        feature_length = int(row["feature_length"])
        raw_start = int(row["raw_start"])
        available_raw = int(row["raw_length"])
        n_raw = min(raw_tiles, available_raw)
        if bag_tiles < n_raw:
            raise ValueError("bag_tiles must be at least raw_tiles")
        raw_reservoir_positions = np.asarray(
            json.loads(row["reservoir_positions"]), dtype=np.int64
        )[:n_raw]
        all_positions = np.arange(feature_length, dtype=np.int64)
        other = np.setdiff1d(
            all_positions, raw_reservoir_positions, assume_unique=True
        )
        n_other = min(len(other), bag_tiles - n_raw)
        if len(other) > n_other:
            rng = np.random.default_rng(
                _stable_seed(str(row["slide_id"]), self.seed, int(epoch))
            )
            other = np.sort(rng.choice(other, n_other, replace=False))
        selected = np.sort(np.concatenate([raw_reservoir_positions, other]))
        raw_positions = np.searchsorted(selected, raw_reservoir_positions)
        absolute = feature_start + selected
        source = np.asarray(
            self.source_indices[feature_start + raw_reservoir_positions],
            dtype=np.int64,
        )
        expected_source = np.asarray(
            json.loads(row["source_tile_indices"]), dtype=np.int64
        )[:n_raw]
        if not np.array_equal(source, expected_source):
            raise RuntimeError(f"{row['slide_id']}: raw/source alignment changed")
        return FixedMixedBagSample(
            cached_features=np.array(
                self.features[absolute], dtype=np.float32, copy=True
            ),
            raw_positions=raw_positions,
            source_tile_indices=source,
            raw_images=np.array(
                self.images[raw_start : raw_start + n_raw], copy=True
            ),
        )


class RawTileReader:
    """Reconstruct current LazySlide tiles by their original Zarr row indices."""

    def __init__(self, slide_index: pd.DataFrame, *, open_slide_cache: int = 8):
        self.slide_index = slide_index.reset_index(drop=True)
        # Bound open WSI readers. Creating the cached function per instance keeps
        # separate experiment readers isolated.
        self._open = lru_cache(maxsize=open_slide_cache)(self._open_uncached)

    @staticmethod
    def _open_uncached(filename: str):
        path = Path(filename)
        return open_wsi(
            str(path), store=str(path.parent), attach_thumbnail=False
        )

    def read(self, row_index: int, tile_indices: np.ndarray) -> list[np.ndarray]:
        row = self.slide_index.iloc[int(row_index)]
        wsi = self._open(str(row["FILENAME"]))
        spec = wsi.tile_spec("tiles")
        if spec is None:
            raise ValueError(f"{row['slide_id']}: missing tiles TileSpec")
        tiles = _tile_rows_in_feature_order(wsi)
        requested = np.asarray(tile_indices, dtype=np.int64)
        if len(requested) == 0:
            raise ValueError("at least one raw tile is required")
        if requested.min() < 0 or requested.max() >= len(tiles):
            raise IndexError(f"{row['slide_id']}: raw tile index out of range")
        images = []
        for tile_index in requested:
            tile = tiles.iloc[int(tile_index)]
            x, y = tile.geometry.bounds[:2]
            image = wsi.reader.get_region(
                x,
                y,
                spec.ops_width,
                spec.ops_height,
                level=spec.ops_level,
            )
            image = wsi.reader.resize_img(
                image, dsize=(spec.width, spec.height)
            )
            images.append(image)
        return images


class MixedFeatureCTransPath(nn.Module):
    """CTransPath PEFT plus warm-started Wagner over a mixed cached/raw bag."""

    def __init__(
        self,
        ctranspath_mode: CTransPathMode,
        *,
        wagner_mode: str = "last_block",
    ):
        super().__init__()
        self.ctranspath = TrainableCTransPath(mode=ctranspath_mode)
        self.wagner = load_wagner(eval_mode=False)
        configure_wagner_trainable(self.wagner, wagner_mode)

    @property
    def transform(self):
        return self.ctranspath.transform

    def forward(
        self,
        cached_features: torch.Tensor,
        raw_images: torch.Tensor,
        replace_positions: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.ctranspath(raw_images)
        mixed = cached_features.clone()
        mixed[:, replace_positions, :] = encoded.unsqueeze(0)
        return self.wagner(mixed)


def transform_raw_tiles(
    images: list[np.ndarray], transform
) -> torch.Tensor:
    return torch.stack([transform(image) for image in images], dim=0)
