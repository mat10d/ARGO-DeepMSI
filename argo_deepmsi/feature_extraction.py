"""
Feature extraction module using LazySlide.

Architecture:
- Patch-level features stored in {slide}.zarr/tables/{model}_tiles (AnnData)
- Slide-level aggregation reads from zarr, outputs to results/embeddings/
- All aggregation uses zarr-based workflow (no intermediate h5ad files)

Workflow:
1. Extract features: extract_features_batch() → {slide}.zarr with multiple models
2. Aggregate: aggregate_features_new() → results/embeddings/{model}_{method}/
3. Train: Use embeddings for MSI classification

Aggregation Methods:
- Simple pooling: mean, max, median, sum (fast, dataset-level)
- Neural encoders: prism, titan, chief (slower, per-slide, needs spatial context)
"""

import logging
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import lazyslide as zs
    from wsidata import open_wsi

    from . import models as _argo_models  # noqa: F401  (registers phaet/mascaret)
    from .models._lazyslide import MODEL_REGISTRY

    LAZYSLIDE_AVAILABLE = True
except ImportError:
    LAZYSLIDE_AVAILABLE = False

from .io_utils import ensure_dir, get_embeddings_dir
from .reproducibility import write_json

logger = logging.getLogger(__name__)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _feature_store_manifest(zarr_path: Path) -> dict:
    path = zarr_path / "argo_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Ignoring unreadable feature-store manifest: %s", path)
    return {}


def _current_tiling(tile_px: int, mpp: float) -> dict:
    return {
        "tile_px": tile_px,
        "mpp": mpp,
        "lazyslide": _package_version("lazyslide"),
        "wsidata": _package_version("wsidata"),
    }


def _tiling_matches(manifest: dict, tile_px: int, mpp: float) -> bool:
    recorded = manifest.get("tiling", {})
    expected = _current_tiling(tile_px, mpp)
    return all(recorded.get(key) == value for key, value in expected.items())


def _model_provenance(model: str) -> dict:
    model_class = MODEL_REGISTRY.get(model) if LAZYSLIDE_AVAILABLE else None
    if model_class is None:
        return {"class": None}
    fields = ("description", "encode_dim", "hf_url", "github_url", "license", "is_gated")
    return {
        "class": f"{model_class.__module__}.{model_class.__qualname__}",
        **{
            field: getattr(model_class, field)
            for field in fields
            if getattr(model_class, field, None) is not None
        },
        "task": str(getattr(model_class, "task", "unknown")),
    }


# ============================================================================
# Model configurations
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for a feature extraction model."""

    name: str
    type: str  # patch, slide, qc, segmentation, or style_transfer
    requires_auth: bool = False
    tile_px: int = 256
    mpp: float = 0.5
    description: str = ""


# Known LazySlide models. Capability-specific catalogs are derived below.
_MODEL_CONFIGS = {
    # No authentication required
    "ctranspath": ModelConfig(
        "ctranspath", "patch", False, 256, 0.5, "CTransPath pathology foundation model"
    ),
    "plip": ModelConfig("plip", "patch", False, 256, 0.5, "PLIP vision-language model"),
    "phikon": ModelConfig("phikon", "patch", False, 256, 0.5, "Phikon pathology foundation model"),
    "phikonv2": ModelConfig(
        "phikonv2", "patch", False, 256, 0.5, "Phikon v2 pathology foundation model"
    ),
    # Gated models (require HuggingFace auth)
    "uni": ModelConfig("uni", "patch", True, 256, 0.5, "UNI pathology foundation model"),
    "uni2": ModelConfig("uni2", "patch", True, 256, 0.5, "UNI2 pathology foundation model"),
    "virchow": ModelConfig(
        "virchow", "patch", True, 256, 0.5, "Virchow pathology foundation model"
    ),
    "virchow2": ModelConfig(
        "virchow2", "patch", True, 256, 0.5, "Virchow2 (631M params, 2560 features)"
    ),
    "conch": ModelConfig("conch", "patch", True, 256, 0.5, "CONCH vision-language model"),
    "conch_v1.5": ModelConfig(
        "conch_v1.5", "patch", True, 256, 0.5, "CONCH v1.5 vision-language model"
    ),
    "gigapath": ModelConfig(
        "gigapath", "patch", True, 256, 0.5, "GigaPath pathology foundation model"
    ),
    "h-optimus-0": ModelConfig(
        "h-optimus-0", "patch", False, 256, 0.5, "H-Optimus-0 pathology model"
    ),
    "h-optimus-1": ModelConfig(
        "h-optimus-1", "patch", True, 256, 0.5, "H-Optimus-1 pathology model"
    ),
    "h0-mini": ModelConfig("h0-mini", "patch", True, 256, 0.5, "H-Optimus-0 mini variant"),
    "hibou-b": ModelConfig(
        "hibou-b", "patch", True, 256, 0.5, "Hibou-B pathology foundation model"
    ),
    "hibou-l": ModelConfig(
        "hibou-l", "patch", True, 256, 0.5, "Hibou-L pathology foundation model"
    ),
    "chief": ModelConfig("chief", "patch", False, 256, 0.5, "CHIEF pathology foundation model"),
    "madeleine": ModelConfig("madeleine", "slide", False, 256, 0.5, "Madeleine slide encoder"),
    "medsiglip": ModelConfig(
        "medsiglip", "patch", True, 256, 0.5, "MedSigLIP vision-language model"
    ),
    "omiclip": ModelConfig("omiclip", "patch", True, 256, 0.5, "OmiCLIP vision-language model"),
    "path_orchestra": ModelConfig(
        "path_orchestra", "patch", True, 256, 0.5, "PathOrchestra pathology model"
    ),
    "pathprofiler": ModelConfig(
        "pathprofiler", "segmentation", False, 256, 0.5, "PathProfiler segmentation model"
    ),
    "musk": ModelConfig("musk", "patch", True, 256, 0.5, "MUSK pathology foundation model"),
    "nulite": ModelConfig("nulite", "segmentation", False, 256, 0.5, "NuLite segmentation model"),
    "gpfm": ModelConfig("gpfm", "patch", False, 256, 0.5, "GPFM pathology foundation model"),
    "histoplus": ModelConfig(
        "histoplus", "segmentation", True, 256, 0.5, "HistoPlus segmentation model"
    ),
    "rosie": ModelConfig("rosie", "style_transfer", True, 256, 0.5, "Rosie virtual staining"),
    # ---- Waiv robust encoders (argo_deepmsi.models.waiv) ----
    "phaet": ModelConfig(
        "phaet", "patch", True, 256, 0.5, "Phaet: robust fine-tuned Phikon-v2 (Waiv)"
    ),
    "mascaret": ModelConfig(
        "mascaret", "patch", True, 256, 0.5, "Mascaret: robust fine-tuned Midnight-12k (Waiv)"
    ),
    # ---- Quality-control models (LazySlide) ----
    "grandqc-artifact": ModelConfig(
        "grandqc-artifact", "qc", False, 256, 0.5, "GrandQC artifact detection (bubbles/folds/pen)"
    ),
    "grandqc-tissue": ModelConfig(
        "grandqc-tissue", "qc", False, 256, 0.5, "GrandQC tissue-quality assessment"
    ),
    "pathprofilerqc": ModelConfig(
        "pathprofilerqc", "qc", False, 256, 0.5, "PathProfilerQC tile-level QC"
    ),
    "focus": ModelConfig("focus", "qc", False, 256, 0.5, "Focus/sharpness score"),
    "focuslitenn": ModelConfig("focuslitenn", "qc", False, 256, 0.5, "FocusLiteNN focus metric"),
}


def _model_has_task(model_class, *names: str) -> bool:
    tasks = getattr(model_class, "task", ())
    tasks = (tasks,) if not isinstance(tasks, (list, tuple, set)) else tasks
    return any(str(task).rsplit(".", 1)[-1] in names for task in tasks)


if LAZYSLIDE_AVAILABLE:
    # LazySlide 0.12's separate model catalog evolves faster than this project.
    # Make new vision/multimodal encoders immediately CLI-addressable while
    # retaining curated names above and suppressing duplicate registry aliases.
    _patch_classes = {
        MODEL_REGISTRY[name]
        for name, config in _MODEL_CONFIGS.items()
        if config.type == "patch" and name in MODEL_REGISTRY
    }
    for _name, _model_class in MODEL_REGISTRY.items():
        if (
            _name not in _MODEL_CONFIGS
            and _model_class not in _patch_classes
            and callable(getattr(_model_class, "encode_image", None))
            and _model_has_task(_model_class, "vision", "multimodal")
        ):
            _MODEL_CONFIGS[_name] = ModelConfig(
                _name,
                "patch",
                bool(getattr(_model_class, "is_gated", False)),
                256,
                0.5,
                str(getattr(_model_class, "description", "LazySlide model")),
            )
            _patch_classes.add(_model_class)

PATCH_MODELS = {key: value for key, value in _MODEL_CONFIGS.items() if value.type == "patch"}
QC_MODELS = {key: value for key, value in _MODEL_CONFIGS.items() if value.type == "qc"}

# Slide-level aggregation methods
SLIDE_ENCODERS = {
    # Simple pooling (fast, no GPU required)
    "mean": "Mean pooling across all tiles",
    "max": "Max pooling across all tiles",
    "median": "Median pooling across all tiles",
    "sum": "Sum pooling across all tiles",
}

if LAZYSLIDE_AVAILABLE:
    _slide_classes = set()
    for _name, _model_class in MODEL_REGISTRY.items():
        if _model_class not in _slide_classes and _model_has_task(_model_class, "slide_encoder"):
            SLIDE_ENCODERS[_name] = str(
                getattr(_model_class, "description", "LazySlide slide encoder")
            )
            _slide_classes.add(_model_class)

ALL_MODELS = {**PATCH_MODELS}


def list_available_models(model_type: Optional[str] = None) -> List[str]:
    """List available models for feature extraction."""
    if model_type == "patch":
        return list(PATCH_MODELS.keys())
    elif model_type == "encoder":
        return list(SLIDE_ENCODERS.keys())
    else:
        return list(ALL_MODELS.keys())


# ============================================================================
# Batch Feature Extraction
# ============================================================================


def extract_features_single_slide(
    slide_path: Union[str, Path],
    models: Union[str, List[str]] = "uni2",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    overwrite: bool = False,
    num_workers: int = 4,
    batch_size: int = 64,
    tiling_policy: Literal["reuse", "require-current"] = "require-current",
) -> Optional[Path]:
    """Extract patch features from a single slide using one or more models.

    Uses LazySlide's design: preprocess once, extract all models, write once.
    Saves a single Zarr next to the original slide with all features.

    **Incremental Extraction:**
    If zarr already exists, only extracts models that are not already present.
    This allows adding new models to existing zarr files without reprocessing.

    Args:
        slide_path: Path to WSI file
        models: Model name(s) for feature extraction (string or list of strings)
        tile_px: Tile size in pixels
        mpp: Microns per pixel
        amp: Use automatic mixed precision
        device: Device for inference
        overwrite: If True, re-extract the requested model features. This does
                   not regenerate an existing tile grid. If False (default),
                   only extract missing models.
        tiling_policy: ``require-current`` rejects an existing feature store
                       unless its manifest matches the installed tiling stack,
                       tile size, and MPP. ``reuse`` accepts legacy stores.

    Returns:
        Path to saved Zarr directory (next to original slide)

    Examples:
        # First run: extract plip and ctranspath
        extract_features_single_slide(slide, models=["plip", "ctranspath"])
        # Creates: slide.zarr with plip_tiles and ctranspath_tiles

        # Second run: add uni2 to existing zarr
        extract_features_single_slide(slide, models=["plip", "ctranspath", "uni2"])
        # Only extracts uni2, skips plip and ctranspath
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    slide_path = Path(slide_path)
    if not slide_path.exists():
        logger.error(f"Slide not found: {slide_path}")
        return None

    # Handle single model or list
    if isinstance(models, str):
        models = [models]

    # Zarr will be saved next to the slide
    zarr_path = slide_path.parent / f"{slide_path.stem}.zarr"
    store_existed = zarr_path.exists()
    store_manifest = _feature_store_manifest(zarr_path)
    if tiling_policy not in {"reuse", "require-current"}:
        raise ValueError("tiling_policy must be 'reuse' or 'require-current'")
    if (
        store_existed
        and tiling_policy == "require-current"
        and not _tiling_matches(store_manifest, tile_px, mpp)
    ):
        logger.error(
            "%s uses an unverified or different tiling stack. Preserve/move the old "
            "zarr and rerun into a fresh store before comparing LazySlide 0.12 features.",
            zarr_path,
        )
        return None

    # Check which models need extraction
    models_to_extract = models.copy() if isinstance(models, list) else [models]

    if zarr_path.exists() and not overwrite:
        # Zarr exists - check which models are already extracted
        existing_models = []

        if (zarr_path / "tables").exists():
            for table_dir in (zarr_path / "tables").iterdir():
                if table_dir.is_dir() and table_dir.name.endswith("_tiles"):
                    model_name = table_dir.name.replace("_tiles", "")
                    existing_models.append(model_name)

        # Filter to only models we don't have yet
        models_to_extract = [m for m in models_to_extract if m not in existing_models]

        if not models_to_extract:
            logger.info(
                f"All requested models already extracted in {zarr_path.name}: "
                f"{', '.join(existing_models)}"
            )
            return zarr_path

        logger.info(
            f"Found existing models {existing_models} in {zarr_path.name}, "
            f"will extract: {models_to_extract}"
        )

    try:
        # Open WSI (either new slide or existing zarr) — skip thumbnail for batch jobs.
        # For cached zarrs, open from the SVS path with store=parent so the reader is
        # whatever is installed locally (avoids KeyError when the recorded reader
        # — e.g. 'fastslide' — isn't available).
        if zarr_path.exists():
            logger.info(f"Loading existing zarr: {zarr_path.name}")
            wsi = open_wsi(
                str(slide_path),
                store=str(slide_path.parent),
                attach_thumbnail=False,
            )
        else:
            logger.info(f"Processing {slide_path.name} with models: {', '.join(models_to_extract)}")
            wsi = open_wsi(str(slide_path), attach_thumbnail=False)

        # Preprocess if needed (only for new slides)
        if not zarr_path.exists():
            logger.info("Preprocessing: tissue detection and tiling...")
            zs.pp.find_tissues(wsi)
            zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

        # Extract only the models we need
        for model in models_to_extract:
            logger.info(f"Extracting features with {model}...")
            zs.tl.feature_extraction(
                wsi,
                model=model,
                amp=amp,
                device=device,
                num_workers=num_workers,
                batch_size=batch_size,
                pbar=False,
            )

        # Write ONCE → saves next to original slide
        logger.info("Saving WSI with all features...")
        wsi.write()

        if not store_existed:
            store_manifest["tiling"] = {
                **_current_tiling(tile_px, mpp),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "implementation": "lazyslide.pp.tile_tissues",
            }
        elif "tiling" not in store_manifest:
            store_manifest["tiling"] = {
                "source": "preexisting-unversioned-store",
                "lazyslide": "unknown",
                "wsidata": "unknown",
                "tile_px": "unknown",
                "mpp": "unknown",
            }
        features = store_manifest.setdefault("features", {})
        for model in models_to_extract:
            features[model] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "lazyslide": _package_version("lazyslide"),
                "lazyslide-models": _package_version("lazyslide-models"),
                "torch": _package_version("torch"),
                "registry": _model_provenance(model),
            }
        write_json(zarr_path / "argo_manifest.json", store_manifest)

        # Verify features were saved
        if hasattr(wsi, "tables"):
            saved_features = list(wsi.tables.keys())
            logger.info(f"Saved {len(saved_features)} feature tables: {saved_features}")

        logger.info(f"✓ Saved complete WSI: {zarr_path}")
        return zarr_path

    except Exception as e:
        logger.error(f"Failed to process {slide_path.name}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def extract_features_batch(
    slide_table: pd.DataFrame,
    models: Union[str, List[str]] = "uni2",
    slide_column: str = "FILENAME",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    overwrite: bool = False,
    max_slides: Optional[int] = None,
    num_workers: int = 4,
    batch_size: int = 64,
    tiling_policy: Literal["reuse", "require-current"] = "require-current",
) -> pd.DataFrame:
    """Extract features from all slides using one or more models.

    Each slide is processed once with all models, creating a single Zarr
    next to the original slide containing all feature tables.
    """
    if slide_column not in slide_table.columns:
        raise ValueError(f"Column '{slide_column}' not found in slide table")

    slides = slide_table[slide_column].dropna().unique()
    if max_slides:
        slides = slides[:max_slides]

    # Handle single model or list
    if isinstance(models, str):
        models = [models]

    logger.info(f"Extracting features from {len(slides)} slides")
    logger.info(f"Models: {', '.join(models)}")

    results = []
    for slide_path in tqdm(slides, desc="Extracting features"):
        output_path = extract_features_single_slide(
            slide_path=slide_path,
            models=models,
            tile_px=tile_px,
            mpp=mpp,
            amp=amp,
            device=device,
            overwrite=overwrite,
            num_workers=num_workers,
            batch_size=batch_size,
            tiling_policy=tiling_policy,
        )
        results.append(
            {
                "slide_path": slide_path,
                "models": ",".join(models),
                "zarr_path": str(output_path) if output_path else None,
                "success": output_path is not None,
            }
        )

    results_df = pd.DataFrame(
        results,
        columns=["slide_path", "models", "zarr_path", "success"],
    )
    success_count = results_df["success"].sum()
    logger.info(f"Completed: {success_count}/{len(slides)} slides successful")

    return results_df


# ============================================================================
# Slide-Level Aggregation (DEPRECATED - Use zarr-based functions below)
# ============================================================================
#
# NOTE: The functions in this section are deprecated and kept for backwards
# compatibility only. Use the new zarr-based aggregation functions instead:
#   - aggregate_simple_pooling() for mean/max/median/sum
#   - aggregate_neural_encoders() for PRISM/TITAN/etc.
#   - aggregate_features_new() for unified interface
# ============================================================================


# ============================================================================
# Zarr-Based Aggregation (CANONICAL IMPLEMENTATION)
# ============================================================================
#
# These functions implement the new zarr-based workflow:
# 1. Read AnnData directly from {slide}.zarr/tables/{model}_tiles
# 2. Apply aggregation (simple pooling or neural encoders)
# 3. Save to results/embeddings/{model}_{method}/
#
# Functions:
# - aggregate_simple_pooling(): Fast pooling (mean/max/median/sum)
# - aggregate_neural_encoders(): Neural slide encoders (PRISM/TITAN)
# - aggregate_features_new(): Unified interface (use this from CLI)
# ============================================================================


def _save_embeddings(
    df: pd.DataFrame,
    model: str,
    method: str,
    output_dir: Optional[Path],
    write_h5ad: bool = True,
) -> Path:
    """Save embeddings in numpy + CSV format.

    Args:
        df: DataFrame with columns: slide_id, patient_id, site, n_tiles,
            zarr_path, embedding
        model: Model name
        method: Aggregation method name
        output_dir: Output directory (if None, uses default)

    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = get_embeddings_dir(f"{model}_{method}")
    ensure_dir(output_dir)

    # Extract embedding matrix
    embedding_matrix = np.vstack(df["embedding"].values)

    # Save metadata (without embedding column)
    metadata_cols = ["slide_id", "patient_id", "site", "n_tiles", "zarr_path"]
    metadata_df = df[metadata_cols].copy()
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)

    # Save embeddings as numpy array
    np.save(output_dir / "embeddings.npy", embedding_matrix)

    # Also save as AnnData for scverse interop (scanpy UMAP/leiden/etc.)
    if write_h5ad:
        try:
            import anndata as ad

            obs = metadata_df.set_index("slide_id", drop=False).astype(
                {"slide_id": str, "patient_id": str, "site": str, "zarr_path": str}
            )
            adata = ad.AnnData(X=embedding_matrix.astype(np.float32), obs=obs)
            adata.uns["model"] = model
            adata.uns["aggregation"] = method
            adata.write_h5ad(output_dir / "embeddings.h5ad")
        except Exception as e:
            logger.warning(f"Could not write AnnData output: {e}")

    logger.info(f"Saved {len(df)} embeddings ({embedding_matrix.shape[1]}D) to {output_dir}")

    return output_dir


_POOL_FNS = {
    "mean": lambda X: np.asarray(X).mean(axis=0),
    "max": lambda X: np.asarray(X).max(axis=0),
    "median": lambda X: np.median(np.asarray(X), axis=0),
    "sum": lambda X: np.asarray(X).sum(axis=0),
}


def _open_feature_store(zarr_path: Path):
    """Open the tile store behind a narrow boundary that acceptance tests can replace."""
    import zarr

    return zarr.open(str(zarr_path), mode="r")


def aggregate_simple_pooling(
    slide_table: Union[str, Path, pd.DataFrame],
    models: Union[str, List[str]],
    method: Literal["mean", "max", "median", "sum"] = "mean",
    output_dir: Optional[Path] = None,
    write_h5ad: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Aggregate features using simple pooling.

    Reads each slide's zarr store once and aggregates all requested models
    from it (slide-outer / model-inner loop). Reads the X matrix directly
    from zarr, skipping full AnnData construction.

    Args:
        slide_table: Path to slide_table.csv or DataFrame with:
                     PATIENT, FILENAME, SITE
        models: Model(s) to aggregate (e.g., "plip" or ["plip", "ctranspath"])
        method: Pooling method (mean, max, median, sum)
        output_dir: Output directory (defaults to results/embeddings/)

    Returns:
        Dict mapping model -> results DataFrame
    """
    # Load slide table
    if isinstance(slide_table, (str, Path)):
        df = pd.read_csv(slide_table)
    else:
        df = slide_table.copy()

    # Normalize models to list
    models = [models] if isinstance(models, str) else list(models)

    if method not in _POOL_FNS:
        raise ValueError(f"Unknown pooling method: {method}. Supported: {list(_POOL_FNS)}")
    pool_fn = _POOL_FNS[method]

    # Accumulator per model — open each zarr exactly once per slide
    # (outer=slide, inner=model) so we don't re-open N_models times.
    per_model: Dict[str, List[dict]] = {m: [] for m in models}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"pool:{method}"):
        svs_path = Path(row["FILENAME"])
        zarr_path = svs_path.with_suffix(".zarr")

        if not zarr_path.exists():
            logger.warning(f"Zarr not found: {zarr_path}")
            continue

        try:
            store = _open_feature_store(zarr_path)
        except Exception as e:
            logger.error(f"Failed to open zarr {zarr_path.name}: {e}")
            continue

        for model in models:
            feature_key = f"{model}_tiles"
            try:
                # Zarr layout: {zarr}/tables/{feature_key}/X
                X_group = store["tables"][feature_key]["X"]
                # X may be a sparse-backed group or a dense array
                try:
                    X = X_group[:]
                except Exception:
                    # Sparse fallback via anndata
                    import anndata as ad

                    adata = ad.read_zarr(str(zarr_path / "tables" / feature_key))
                    X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X)

                embedding = np.asarray(pool_fn(X)).flatten()
                per_model[model].append(
                    {
                        "slide_id": svs_path.stem,
                        "patient_id": row["PATIENT"],
                        "site": row["SITE"],
                        "embedding": embedding,
                        "n_tiles": int(X.shape[0]),
                        "zarr_path": str(zarr_path),
                    }
                )
            except KeyError:
                logger.warning(f"No {feature_key} in {zarr_path.name}")
                continue
            except Exception as e:
                logger.error(f"Failed to aggregate {zarr_path.name} / {model}: {e}")
                continue

    results: Dict[str, pd.DataFrame] = {}
    for model, rows in per_model.items():
        if not rows:
            logger.warning(f"No embeddings generated for {model}")
            continue
        df_result = pd.DataFrame(rows)
        _save_embeddings(df_result, model, method, output_dir, write_h5ad=write_h5ad)
        results[model] = df_result

    return results


def aggregate_neural_encoders(
    slide_table: Union[str, Path, pd.DataFrame],
    model: str,
    encoder: str,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    write_h5ad: bool = True,
) -> pd.DataFrame:
    """Aggregate features using neural slide encoders.

    Requires loading original slide files because neural encoders need
    spatial context (not just patch features).

    Args:
        slide_table: Path to slide_table.csv with PATIENT, FILENAME, SITE
        model: Model name (must match encoder requirements)
                - prism: requires virchow or virchow2
                - titan: requires conch_v1.5
                - chief: requires chief
                - madeleine: requires conch
        encoder: Neural slide encoder name
        output_dir: Output directory
        device: Device for inference

    Returns:
        DataFrame with embeddings and metadata
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    from wsidata import open_wsi

    # Validate model-encoder compatibility
    encoder_requirements = {
        "prism": ["virchow", "virchow2"],
        "titan": ["conch_v1.5"],
        "conch_v1.5": ["conch_v1.5"],
        "chief-slide-encoder": ["chief"],
        "madeleine": ["conch"],
    }

    if encoder in encoder_requirements:
        if model not in encoder_requirements[encoder]:
            raise ValueError(f"{encoder} requires {encoder_requirements[encoder]}, got {model}")

    # Load slide table
    if isinstance(slide_table, (str, Path)):
        df = pd.read_csv(slide_table)
    else:
        df = slide_table.copy()

    embeddings = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{encoder}"):
        svs_path = Path(row["FILENAME"])
        zarr_path = svs_path.with_suffix(".zarr")

        if not zarr_path.exists():
            logger.warning(f"Zarr not found: {zarr_path}")
            continue

        try:
            # Use the svs-path + store=parent pattern. Opening the zarr
            # directly KeyErrors on readers (e.g. `fastslide`) that aren't
            # installed locally but are recorded in the zarr metadata.
            wsi = open_wsi(str(svs_path), store=str(svs_path.parent), attach_thumbnail=False)
            feature_key = f"{model}_tiles"

            if feature_key not in wsi.tables:
                logger.warning(f"No {feature_key} in {zarr_path}")
                continue

            # Run neural aggregation
            zs.tl.feature_aggregation(
                wsi,
                feature_key=model,  # Base name without "_tiles"
                encoder=encoder,
                device=device,
            )

            # Extract aggregated embedding. LazySlide stores the slide
            # representation in feature_table.uns["agg_ops"]["agg_slide"]
            # (see lazyslide.tools._features.feature_aggregation). The varm
            # branch only fires when the aggregated dim matches the tile
            # feature dim (e.g. mean pooling) — PRISM/TITAN change dim, so
            # we must read from uns["agg_ops"].
            adata = wsi.tables[feature_key]
            agg_ops = adata.uns.get("agg_ops", {})

            if "agg_slide" in agg_ops and "features" in agg_ops["agg_slide"]:
                embedding = np.asarray(agg_ops["agg_slide"]["features"]).flatten()
            elif "agg_slide" in adata.varm:
                embedding = np.asarray(adata.varm["agg_slide"]).flatten()
            elif "agg_slide" in adata.uns:
                embedding = np.asarray(adata.uns["agg_slide"]).flatten()
            else:
                raise ValueError(
                    f"Aggregation result not found in uns['agg_ops'] / varm / uns "
                    f"for {svs_path.name}"
                )

            embeddings.append(
                {
                    "slide_id": svs_path.stem,
                    "patient_id": row["PATIENT"],
                    "site": row["SITE"],
                    "embedding": embedding,
                    "n_tiles": adata.n_obs,
                    "zarr_path": str(zarr_path),
                }
            )

        except Exception as e:
            logger.error(f"Failed to aggregate {svs_path.name}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Convert to DataFrame and save
    df_result = pd.DataFrame(embeddings)

    if len(df_result) > 0:
        _save_embeddings(df_result, model, encoder, output_dir, write_h5ad=write_h5ad)
    else:
        logger.warning("No embeddings generated")

    return df_result


def filter_slides_by_qc(
    slide_table: Union[str, Path, pd.DataFrame],
    qc_model: str = "grandqc-artifact",
    threshold: float = 0.5,
    reduce: Literal["mean", "max", "median"] = "mean",
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Filter slides using QC scores already extracted into the zarr.

    For each slide, loads ``wsi.tables[f'{qc_model}_tiles'].X`` and reduces
    across tiles (mean/max/median). Slides with a reduced score <= ``threshold``
    are kept (lower = cleaner for artifact-style models — flip ``threshold``
    semantics as needed for tissue-quality models).

    Returns a DataFrame with an added ``qc_score`` column and a ``passes_qc``
    boolean; optionally writes a filtered CSV.
    """
    import zarr as _zarr

    if isinstance(slide_table, (str, Path)):
        df = pd.read_csv(slide_table)
    else:
        df = slide_table.copy()

    feature_key = f"{qc_model}_tiles"
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"qc:{qc_model}"):
        zarr_path = Path(row["FILENAME"]).with_suffix(".zarr")
        if not zarr_path.exists():
            scores.append(np.nan)
            continue
        try:
            store = _zarr.open(str(zarr_path), mode="r")
            X = store["tables"][feature_key]["X"][:]
            X = np.asarray(X).astype(np.float32)
            if reduce == "mean":
                s = float(X.mean())
            elif reduce == "max":
                s = float(X.max())
            else:
                s = float(np.median(X))
            scores.append(s)
        except Exception as e:
            logger.warning(f"QC read failed for {zarr_path.name}: {e}")
            scores.append(np.nan)

    df = df.copy()
    df["qc_score"] = scores
    df["passes_qc"] = (df["qc_score"] <= threshold) & df["qc_score"].notna()

    kept = int(df["passes_qc"].sum())
    logger.info(f"QC ({qc_model}, reduce={reduce}, thr={threshold}): {kept}/{len(df)} slides pass")

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df


def aggregate_features(
    slide_table: Union[str, Path, pd.DataFrame],
    models: Union[str, List[str]],
    method: str = "mean",
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    write_h5ad: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Unified aggregation interface supporting both simple pooling and neural encoders.

    Args:
        slide_table: slide_table.csv path or DataFrame
        models: Model(s) to aggregate
        method: Aggregation method
                - Simple: "mean", "max", "median", "sum"
                - Neural: "prism", "titan", "chief", "madeleine"
        output_dir: Output directory
        device: Device for neural encoders

    Returns:
        Dict mapping model -> results DataFrame
    """
    # Simple pooling methods
    simple_methods = ["mean", "max", "median", "sum"]

    # Neural encoder methods
    neural_encoders = [name for name in SLIDE_ENCODERS if name not in simple_methods]

    if method in simple_methods:
        # Use agg_wsi() for simple pooling
        return aggregate_simple_pooling(
            slide_table=slide_table,
            models=models,
            method=method,
            output_dir=output_dir,
            write_h5ad=write_h5ad,
        )

    elif method in neural_encoders:
        # Use feature_aggregation() per slide
        # Can only process one model at a time for neural encoders
        if isinstance(models, list):
            if len(models) > 1:
                raise ValueError(
                    f"Neural encoder {method} can only process one model at a time. Got: {models}"
                )
            model = models[0]
        else:
            model = models

        df = aggregate_neural_encoders(
            slide_table=slide_table,
            model=model,
            encoder=method,
            output_dir=output_dir,
            device=device,
            write_h5ad=write_h5ad,
        )

        return {model: df}

    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. Available: {simple_methods + neural_encoders}"
        )
