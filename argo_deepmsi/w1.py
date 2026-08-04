"""W1 data contracts and cached CTransPath experiment utilities.

The module is intentionally independent of the archived ``old/`` tree.  It
uses the current cohort manifest, current slide table, Zarr feature stores, and
project-local model implementations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_COHORT = Path("results/data/cohort_clean.csv")
DEFAULT_SLIDE_TABLE = Path("results/data/slide_table_pyramidal.csv")
DEFAULT_TUMOR_DIR = Path("results/data/tumor_tiles")
DEFAULT_OUTDIR = Path("results/experiments/w1_ctranspath")
FEATURE_DIM = 768


@dataclass(frozen=True)
class FoldContract:
    cohort_sha256: str
    seed: int
    n_splits: int
    n_patients: int
    n_slides: int
    n_positive_patients: int
    fold_counts: dict[str, dict[str, int]]


def _cohort_digest(frame: pd.DataFrame) -> str:
    columns = ["slide_id", "patient_id", "site", "y"]
    payload = (
        frame[columns]
        .sort_values(columns)
        .to_csv(index=False, lineterminator="\n")
        .encode()
    )
    return hashlib.sha256(payload).hexdigest()


def load_primary_cohort(path: str | Path = DEFAULT_COHORT) -> pd.DataFrame:
    """Load and validate the feature-complete W1 development cohort."""
    frame = pd.read_csv(path)
    required = {"slide_id", "patient_id", "site", "y", "in_primary_set"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cohort missing required columns: {sorted(missing)}")
    frame = frame[frame["in_primary_set"].astype(bool)].copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["y"] = frame["y"].astype(int)
    if frame["slide_id"].duplicated().any():
        raise ValueError("primary cohort contains duplicate slide_id values")
    label_counts = frame.groupby("patient_id")["y"].nunique()
    if (label_counts != 1).any():
        bad = label_counts[label_counts != 1].index.tolist()
        raise ValueError(f"patients have inconsistent labels: {bad[:5]}")
    return frame.sort_values("slide_id").reset_index(drop=True)


def make_patient_fold_manifest(
    cohort: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, FoldContract]:
    """Create one stable outer-fold assignment per patient."""
    label_counts = cohort.groupby("patient_id")["y"].nunique()
    if (label_counts != 1).any():
        bad = label_counts[label_counts != 1].index.tolist()
        raise ValueError(f"patients have inconsistent labels: {bad[:5]}")
    patient = (
        cohort.sort_values(["patient_id", "slide_id"])
        .groupby("patient_id", as_index=False)
        .agg(
            y=("y", "first"),
            site=("patient_cohort", "first")
            if "patient_cohort" in cohort.columns
            else ("site", "first"),
            n_slides=("slide_id", "size"),
        )
    )
    class_counts = patient["y"].value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < n_splits:
        raise ValueError(
            f"need two classes with at least {n_splits} patients each; "
            f"got {class_counts.to_dict()}"
        )
    patient["outer_fold"] = -1
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(
        cv.split(
            np.zeros(len(patient)),
            patient["y"].to_numpy(),
            groups=patient["patient_id"].to_numpy(),
        )
    ):
        patient.loc[test_idx, "outer_fold"] = fold
    if (patient["outer_fold"] < 0).any() or patient["patient_id"].duplicated().any():
        raise RuntimeError("failed to assign exactly one outer fold per patient")

    fold_counts = {}
    for fold, sub in patient.groupby("outer_fold"):
        fold_counts[str(int(fold))] = {
            "patients": int(len(sub)),
            "positive_patients": int(sub["y"].sum()),
            "slides": int(sub["n_slides"].sum()),
        }
    contract = FoldContract(
        cohort_sha256=_cohort_digest(cohort),
        seed=seed,
        n_splits=n_splits,
        n_patients=int(len(patient)),
        n_slides=int(len(cohort)),
        n_positive_patients=int(patient["y"].sum()),
        fold_counts=fold_counts,
    )
    patient["seed"] = seed
    patient["cohort_sha256"] = contract.cohort_sha256
    return patient, contract


def build_slide_index(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    slide_table_path: str | Path = DEFAULT_SLIDE_TABLE,
) -> pd.DataFrame:
    """Join cohort rows to current CTransPath Zarr arrays and outer folds."""
    slide_table = pd.read_csv(slide_table_path)
    if "FILENAME" not in slide_table:
        raise ValueError("slide table missing FILENAME")
    slide_table = slide_table.copy()
    slide_table["slide_id"] = slide_table["FILENAME"].map(
        lambda value: Path(str(value)).stem
    )
    sources = slide_table[["slide_id", "FILENAME"]].drop_duplicates("slide_id")
    if sources["slide_id"].duplicated().any():
        raise ValueError("slide table maps a slide_id to multiple files")

    index = cohort.merge(sources, on="slide_id", how="left", validate="one_to_one")
    index = index.merge(
        folds[["patient_id", "outer_fold"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
    )
    if index[["FILENAME", "outer_fold"]].isna().any().any():
        raise ValueError("some primary slides lack a source path or fold assignment")

    zarr_paths: list[str] = []
    feature_rows: list[int] = []
    missing: list[str] = []
    for row in index.itertuples():
        feature_path = (
            Path(row.FILENAME).with_suffix(".zarr")
            / "tables"
            / "ctranspath_tiles"
            / "X"
        )
        if not feature_path.exists():
            missing.append(str(row.slide_id))
            zarr_paths.append(str(feature_path))
            feature_rows.append(0)
            continue
        array = zarr.open_array(str(feature_path), mode="r")
        if len(array.shape) != 2 or array.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"{row.slide_id}: expected CTransPath (*, {FEATURE_DIM}), got {array.shape}"
            )
        zarr_paths.append(str(feature_path))
        feature_rows.append(int(array.shape[0]))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} primary slides lack CTransPath features: {missing[:5]}"
        )
    index["ctranspath_zarr"] = zarr_paths
    index["feature_rows"] = feature_rows
    if (index["feature_rows"] <= 0).any():
        raise ValueError("primary cohort contains empty CTransPath bags")
    return index.sort_values("slide_id").reset_index(drop=True)


def write_w1_data_contract(
    *,
    cohort_path: str | Path = DEFAULT_COHORT,
    slide_table_path: str | Path = DEFAULT_SLIDE_TABLE,
    outdir: str | Path = DEFAULT_OUTDIR,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, FoldContract]:
    """Write the versioned patient folds and feature index used by all W1 runs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cohort = load_primary_cohort(cohort_path)
    folds, contract = make_patient_fold_manifest(
        cohort, n_splits=n_splits, seed=seed
    )
    index = build_slide_index(cohort, folds, slide_table_path)
    folds.to_csv(outdir / "patient_folds.csv", index=False)
    index.to_csv(outdir / "slide_index.csv", index=False)
    (outdir / "fold_contract.json").write_text(json.dumps(asdict(contract), indent=2))
    return folds, index, contract


def _stable_seed(slide_id: str, seed: int, epoch: int) -> int:
    payload = f"{slide_id}\0{seed}\0{epoch}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class CTransPathBagStore:
    """Read reproducibly capped bags directly from the current Zarr feature stores."""

    def __init__(
        self,
        index: pd.DataFrame,
        *,
        max_tiles: int,
        seed: int = 42,
        tumor_dir: str | Path = DEFAULT_TUMOR_DIR,
        sampling: str = "uniform",
    ):
        if max_tiles < 1:
            raise ValueError("max_tiles must be positive")
        if sampling not in {"uniform", "tumor_enriched"}:
            raise ValueError("sampling must be 'uniform' or 'tumor_enriched'")
        self.index = index.reset_index(drop=True)
        self.max_tiles = int(max_tiles)
        self.seed = int(seed)
        self.tumor_dir = Path(tumor_dir)
        self.sampling = sampling

    def _candidate_indices(self, row: pd.Series, n_rows: int) -> np.ndarray:
        if self.sampling == "tumor_enriched":
            path = self.tumor_dir / f"{row['slide_id']}.npy"
            if path.exists():
                indices = np.asarray(np.load(path), dtype=np.int64)
                indices = np.unique(indices[(indices >= 0) & (indices < n_rows)])
                if len(indices):
                    return indices
        return np.arange(n_rows, dtype=np.int64)

    def read(self, row_index: int, *, epoch: int = 0) -> np.ndarray:
        """Return one float32 bag; changing ``epoch`` changes the sampled tiles."""
        row = self.index.iloc[int(row_index)]
        array = zarr.open_array(str(row["ctranspath_zarr"]), mode="r")
        candidates = self._candidate_indices(row, int(array.shape[0]))
        if len(candidates) > self.max_tiles:
            rng = np.random.default_rng(
                _stable_seed(str(row["slide_id"]), self.seed, int(epoch))
            )
            candidates = np.sort(
                rng.choice(candidates, self.max_tiles, replace=False)
            )
        return np.asarray(array.oindex[candidates, :], dtype=np.float32)


def build_bag_cache(
    index: pd.DataFrame,
    *,
    outdir: str | Path = DEFAULT_OUTDIR / "bag_cache_uniform_1024",
    max_tiles: int = 1024,
    seed: int = 42,
    sampling: str = "uniform",
    tumor_dir: str | Path = DEFAULT_TUMOR_DIR,
) -> tuple[Path, Path]:
    """Materialize one deterministic tile reservoir per slide for fast W1 training."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = CTransPathBagStore(
        index,
        max_tiles=max_tiles,
        seed=seed,
        tumor_dir=tumor_dir,
        sampling=sampling,
    )
    lengths = []
    for row_index in range(len(index)):
        row = index.iloc[row_index]
        n_rows = int(row["feature_rows"])
        candidates = source._candidate_indices(row, n_rows)
        lengths.append(min(len(candidates), max_tiles))
    total_rows = int(sum(lengths))
    data_path = outdir / "bags.npy"
    cache = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, FEATURE_DIM),
    )
    source_indices_path = outdir / "tile_indices.npy"
    source_indices = np.lib.format.open_memmap(
        source_indices_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_rows,),
    )
    rows = []
    cursor = 0
    for row_index, length in enumerate(lengths):
        row = index.iloc[row_index]
        n_rows = int(row["feature_rows"])
        selected = source._candidate_indices(row, n_rows)
        if len(selected) > max_tiles:
            rng = np.random.default_rng(
                _stable_seed(str(row["slide_id"]), seed, -1)
            )
            selected = np.sort(rng.choice(selected, max_tiles, replace=False))
        bag = source.read(row_index, epoch=-1)
        if len(bag) != length:
            raise RuntimeError(
                f"cache length changed for {index.iloc[row_index]['slide_id']}: "
                f"expected {length}, got {len(bag)}"
            )
        cache[cursor : cursor + length] = bag
        source_indices[cursor : cursor + length] = selected
        rows.append(
            {
                "slide_id": row["slide_id"],
                "start": cursor,
                "length": length,
            }
        )
        cursor += length
    cache.flush()
    source_indices.flush()
    index_path = outdir / "bags_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    metadata = {
        "feature": "ctranspath_tiles",
        "feature_dim": FEATURE_DIM,
        "max_tiles": int(max_tiles),
        "seed": int(seed),
        "sampling": sampling,
        "n_slides": int(len(index)),
        "total_tiles": total_rows,
        "tile_indices": source_indices_path.name,
        "slide_index_sha256": hashlib.sha256(
            index[["slide_id", "patient_id", "y", "outer_fold"]]
            .to_csv(index=False, lineterminator="\n")
            .encode()
        ).hexdigest(),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return data_path, index_path


def write_existing_cache_tile_indices(
    index: pd.DataFrame,
    *,
    cache_dir: str | Path,
    max_tiles: int,
    seed: int = 42,
    sampling: str = "uniform",
    tumor_dir: str | Path = DEFAULT_TUMOR_DIR,
) -> Path:
    """Backfill original Zarr row indices for a cache built before index tracking."""
    cache_dir = Path(cache_dir)
    cache_index = pd.read_csv(cache_dir / "bags_index.csv")
    joined = index.reset_index(drop=True).merge(
        cache_index, on="slide_id", how="left", validate="one_to_one"
    )
    source = CTransPathBagStore(
        index,
        max_tiles=max_tiles,
        seed=seed,
        tumor_dir=tumor_dir,
        sampling=sampling,
    )
    total = int((joined["start"] + joined["length"]).max())
    path = cache_dir / "tile_indices.npy"
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.int64, shape=(total,)
    )
    for row_index, row in joined.iterrows():
        candidates = source._candidate_indices(row, int(row["feature_rows"]))
        if len(candidates) > max_tiles:
            rng = np.random.default_rng(
                _stable_seed(str(row["slide_id"]), seed, -1)
            )
            candidates = np.sort(rng.choice(candidates, max_tiles, replace=False))
        start, length = int(row["start"]), int(row["length"])
        if len(candidates) != length:
            raise RuntimeError(
                f"cannot reconstruct source indices for {row['slide_id']}: "
                f"expected {length}, got {len(candidates)}"
            )
        output[start : start + length] = candidates
    output.flush()
    return path


class CachedCTransPathBagStore:
    """Memory-mapped W1 tile reservoirs with deterministic epoch subsampling."""

    def __init__(
        self,
        data_path: str | Path,
        cache_index_path: str | Path,
        slide_index: pd.DataFrame,
        *,
        max_tiles: int,
        seed: int = 42,
    ):
        self.data = np.load(data_path, mmap_mode="r")
        cache_index = pd.read_csv(cache_index_path)
        self.index = slide_index.reset_index(drop=True).merge(
            cache_index,
            on="slide_id",
            how="left",
            validate="one_to_one",
        )
        if self.index[["start", "length"]].isna().any().any():
            raise ValueError("bag cache does not cover every W1 slide")
        self.max_tiles = int(max_tiles)
        self.seed = int(seed)

    def read(self, row_index: int, *, epoch: int = 0) -> np.ndarray:
        row = self.index.iloc[int(row_index)]
        start, length = int(row["start"]), int(row["length"])
        if length <= self.max_tiles:
            selected = np.arange(length, dtype=np.int64)
        else:
            rng = np.random.default_rng(
                _stable_seed(str(row["slide_id"]), self.seed, int(epoch))
            )
            selected = np.sort(rng.choice(length, self.max_tiles, replace=False))
        return np.asarray(self.data[start + selected], dtype=np.float32)
