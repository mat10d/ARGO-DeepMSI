"""Build reusable tile-feature bags for configurable MIL experiments."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .reproducibility import write_json

logger = logging.getLogger(__name__)


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _read_tile_features(zarr_path: Path, model: str) -> np.ndarray:
    import anndata as ad

    table = zarr_path / "tables" / f"{model}_tiles"
    if not table.exists():
        raise FileNotFoundError(table)
    return np.asarray(ad.read_zarr(str(table)).X, dtype=np.float32)


def _normalise_variants(
    models: Sequence[str], variants: Mapping[str, str | Sequence[str]] | None
) -> dict[str, tuple[str, ...]]:
    if variants is None:
        return {model: (model,) for model in models}
    result = {}
    for name, members in variants.items():
        members = (members,) if isinstance(members, str) else tuple(members)
        if not members:
            raise ValueError(f"Bag variant {name!r} has no models")
        unknown = set(members) - set(models)
        if unknown:
            raise ValueError(f"Bag variant {name!r} references unknown models: {sorted(unknown)}")
        result[str(name)] = members
    return result


def build_tile_bags(
    *,
    models: Sequence[str],
    cohort_csv: Path,
    slide_table_csv: Path,
    output_dir: Path,
    variants: Mapping[str, str | Sequence[str]] | None = None,
    tumor_tiles_dir: Path | None = None,
    max_tiles: int | None = 500,
    seed: int = 42,
    slide_column: str = "FILENAME",
) -> dict[str, dict]:
    """Build deterministic bag files from arbitrary LazySlide tile encoders.

    Every variant gets its own ``<variant>_bags.npz`` and
    ``<variant>_index.csv``. A variant may concatenate multiple encoders; tile
    rows must align because they originate from the same LazySlide tile grid.
    """
    if not models:
        raise ValueError("At least one tile encoder is required")
    if max_tiles is not None and max_tiles < 1:
        raise ValueError("max_tiles must be positive or null")
    variants = _normalise_variants(models, variants)

    cohort = pd.read_csv(cohort_csv)
    required = {"slide_id", "patient_id", "site", "y"}
    missing = required - set(cohort)
    if missing:
        raise ValueError(f"{cohort_csv} is missing columns: {sorted(missing)}")
    inclusion = "in_primary_set" if "in_primary_set" in cohort else "in_clean_set"
    if inclusion not in cohort:
        raise ValueError(f"{cohort_csv} needs in_primary_set or in_clean_set")
    cohort = cohort[cohort[inclusion] == 1].copy()
    slide_table = pd.read_csv(slide_table_csv)
    if slide_column not in slide_table:
        raise ValueError(f"{slide_table_csv} has no {slide_column!r} column")
    slide_table = slide_table.copy()
    slide_table["slide_id"] = slide_table[slide_column].map(lambda value: Path(str(value)).stem)
    source = slide_table[["slide_id", slide_column]].drop_duplicates("slide_id")
    columns = ["slide_id", "patient_id", "site", "y"]
    frame = (
        cohort[columns]
        .drop_duplicates("slide_id")
        .merge(source, on="slide_id", how="left", validate="one_to_one")
    )

    stores: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    indices: dict[str, list[dict]] = {name: [] for name in variants}
    skipped: dict[str, list[dict[str, str]]] = {name: [] for name in variants}
    for row in frame.itertuples(index=False):
        slide_id = str(row.slide_id)
        slide_path = getattr(row, slide_column)
        if pd.isna(slide_path):
            for name in variants:
                skipped[name].append({"slide_id": slide_id, "reason": "missing slide path"})
            continue
        zarr_path = Path(str(slide_path)).with_suffix(".zarr")
        features: dict[str, np.ndarray] = {}
        for model in models:
            try:
                features[model] = _read_tile_features(zarr_path, model)
            except (FileNotFoundError, OSError, ValueError) as error:
                logger.warning("%s: could not read %s (%s)", slide_id, model, error)

        tumor_indices = None
        if tumor_tiles_dir is not None:
            tumor_path = tumor_tiles_dir / f"{slide_id}.npy"
            if tumor_path.exists():
                tumor_indices = np.asarray(np.load(tumor_path), dtype=int)

        for variant, members in variants.items():
            if any(member not in features for member in members):
                skipped[variant].append({"slide_id": slide_id, "reason": "missing features"})
                continue
            lengths = {len(features[member]) for member in members}
            if len(lengths) != 1:
                skipped[variant].append({"slide_id": slide_id, "reason": "unaligned tile rows"})
                continue
            n_tiles = lengths.pop()
            selected = np.arange(n_tiles)
            if tumor_tiles_dir is not None:
                if tumor_indices is None:
                    skipped[variant].append({"slide_id": slide_id, "reason": "missing tumor mask"})
                    continue
                selected = tumor_indices[(tumor_indices >= 0) & (tumor_indices < n_tiles)]
            if len(selected) == 0:
                skipped[variant].append({"slide_id": slide_id, "reason": "empty bag"})
                continue
            if max_tiles is not None and len(selected) > max_tiles:
                rng = np.random.default_rng(_stable_seed(seed, slide_id, variant))
                selected = np.sort(rng.choice(selected, max_tiles, replace=False))
            arrays = [features[member][selected] for member in members]
            bag = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)
            stores[variant].append(np.asarray(bag, dtype=np.float32))
            indices[variant].append(
                {
                    key: getattr(row, key)
                    for key in ("slide_id", "patient_id", "site", "y")
                    if hasattr(row, key)
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant, bags in stores.items():
        if not bags:
            raise RuntimeError(f"Bag variant {variant!r} produced no slides")
        concatenated = np.concatenate(bags, axis=0).astype(np.float32)
        lengths = np.asarray([len(bag) for bag in bags], dtype=np.int64)
        bag_path = output_dir / f"{variant}_bags.npz"
        index_path = output_dir / f"{variant}_index.csv"
        np.savez(bag_path, concat=concatenated, lengths=lengths)
        pd.DataFrame(indices[variant]).to_csv(index_path, index=False)
        summary[variant] = {
            "models": list(variants[variant]),
            "bag_file": bag_path,
            "index_file": index_path,
            "n_slides": len(bags),
            "total_tiles": len(concatenated),
            "dimension": concatenated.shape[1],
            "skipped": skipped[variant],
        }
    write_json(
        output_dir / "bags.json",
        {
            "models": list(models),
            "variants": summary,
            "cohort": cohort_csv,
            "slide_table": slide_table_csv,
            "tumor_tiles_dir": tumor_tiles_dir,
            "max_tiles": max_tiles,
            "seed": seed,
        },
    )
    return summary
