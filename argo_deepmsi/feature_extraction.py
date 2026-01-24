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
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import lazyslide as zs
    from wsidata import open_wsi

    LAZYSLIDE_AVAILABLE = True
except ImportError:
    LAZYSLIDE_AVAILABLE = False

from .io_utils import get_embeddings_dir, ensure_dir

logger = logging.getLogger(__name__)


# ============================================================================
# Model configurations
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for a feature extraction model."""

    name: str
    type: str  # "patch" or "slide"
    requires_auth: bool = False
    tile_px: int = 256
    mpp: float = 0.5
    description: str = ""


# Patch-level feature extractors (tile → embedding)
PATCH_MODELS = {
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
        "h-optimus-0", "patch", True, 256, 0.5, "H-Optimus-0 pathology model"
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
    "chief": ModelConfig("chief", "patch", True, 256, 0.5, "CHIEF pathology foundation model"),
    "madeleine": ModelConfig(
        "madeleine", "patch", True, 256, 0.5, "Madeleine pathology foundation model"
    ),
    "medsiglip": ModelConfig(
        "medsiglip", "patch", True, 256, 0.5, "MedSigLIP vision-language model"
    ),
    "omiclip": ModelConfig("omiclip", "patch", True, 256, 0.5, "OmiCLIP vision-language model"),
    "path_orchestra": ModelConfig(
        "path_orchestra", "patch", True, 256, 0.5, "PathOrchestra pathology model"
    ),
    "pathprofiler": ModelConfig(
        "pathprofiler", "patch", True, 256, 0.5, "PathProfiler pathology model"
    ),
    "musk": ModelConfig("musk", "patch", True, 256, 0.5, "MUSK pathology foundation model"),
    "nulite": ModelConfig("nulite", "patch", True, 256, 0.5, "NuLite pathology foundation model"),
    "gpfm": ModelConfig("gpfm", "patch", True, 256, 0.5, "GPFM pathology foundation model"),
    "histoplus": ModelConfig(
        "histoplus", "patch", True, 256, 0.5, "HistoPlus pathology foundation model"
    ),
    "rosie": ModelConfig("rosie", "patch", True, 256, 0.5, "Rosie pathology foundation model"),
}

# Slide-level aggregation methods
SLIDE_ENCODERS = {
    # Simple pooling (fast, no GPU required)
    "mean": "Mean pooling across all tiles",
    "max": "Max pooling across all tiles",
    "median": "Median pooling across all tiles",
    "sum": "Sum pooling across all tiles",
    # Neural slide encoders (slower, GPU required, needs spatial context)
    "prism": "PRISM slide encoder (requires virchow/virchow2 features)",
    "titan": "TITAN slide encoder (requires conch_v1.5 features)",
    "chief": "CHIEF slide encoder (requires chief features)",
    "madeleine": "Madeleine slide encoder (requires conch features)",
    "gigapath-slide-encoder": "GigaPath slide-level aggregator",
}

ALL_MODELS = {**PATCH_MODELS}


def list_available_models(model_type: Optional[str] = None) -> List[str]:
    """List available models for feature extraction."""
    if model_type == "patch":
        return list(PATCH_MODELS.keys())
    elif model_type == "encoder":
        return list(SLIDE_ENCODERS.keys())
    else:
        return list(ALL_MODELS.keys())


def get_model_info(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a model."""
    return ALL_MODELS.get(model_name)


# ============================================================================
# Core WSI Processing
# ============================================================================


def process_slide(
    slide_path: Union[str, Path],
    patch_model: str = "uni2",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
):
    """Load and process a slide through LazySlide pipeline.

    Args:
        slide_path: Path to WSI file
        patch_model: Model for patch-level feature extraction
        tile_px: Tile size in pixels
        mpp: Microns per pixel
        amp: Use automatic mixed precision
        device: Device for inference

    Returns:
        LazySlide WSI object with extracted features
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed. Run: pip install lazyslide")

    slide_path = Path(slide_path)
    logger.info(f"Processing: {slide_path.name}")

    # Load slide
    wsi = open_wsi(str(slide_path))

    # Preprocessing: tissue detection and tiling
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

    # Patch-level feature extraction
    zs.tl.feature_extraction(wsi, model=patch_model, amp=amp, device=device)

    return wsi


def process_slide_with_aggregation(
    slide_path: Union[str, Path],
    patch_model: str = "virchow",
    slide_encoder: str = "prism",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
):
    """Process slide with both patch and slide-level features.

    PRISM requires Virchow features as input.

    Args:
        slide_path: Path to WSI file
        patch_model: Model for patch features (use 'virchow' for PRISM)
        slide_encoder: Slide-level aggregator ('prism', 'titan', 'mean')
        tile_px: Tile size
        mpp: Microns per pixel
        amp: Use mixed precision
        device: Device for inference

    Returns:
        Tuple of (wsi, slide_embedding)
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    # Process slide with patch features
    wsi = process_slide(
        slide_path=slide_path,
        patch_model=patch_model,
        tile_px=tile_px,
        mpp=mpp,
        amp=amp,
        device=device,
    )

    # Slide-level aggregation
    if slide_encoder == "mean":
        # Simple mean pooling
        feature_key = f"{patch_model}_tiles"
        if feature_key in wsi:
            slide_embedding = wsi[feature_key].X.mean(axis=0)
        else:
            slide_embedding = None
    else:
        # Use LazySlide's feature_aggregation with encoder
        zs.tl.feature_aggregation(
            wsi,
            feature_key=patch_model,
            encoder=slide_encoder,
            device=device,
        )
        # Get the aggregated embedding
        agg_key = f"{patch_model}_{slide_encoder}"
        if agg_key in wsi.sdata:
            slide_embedding = wsi.sdata[agg_key]
        else:
            slide_embedding = None

    return wsi, slide_embedding


# ============================================================================
# Tile-Level Analysis
# ============================================================================


def analyze_tiles(
    wsi,
    feature_key: str,
    n_neighbors: int = 15,
    resolution: float = 1.0,
):
    """Perform tile-level analysis with Leiden clustering.

    Args:
        wsi: LazySlide WSI object with extracted features
        feature_key: Key for the features (e.g., 'uni2')
        n_neighbors: Number of neighbors for graph construction
        resolution: Resolution for Leiden clustering

    Returns:
        AnnData with clustering results
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    import scanpy as sc

    # Get the feature AnnData
    tiles_key = f"{feature_key}_tiles"
    if tiles_key not in wsi:
        raise ValueError(f"Features not found: {tiles_key}")

    adata = wsi[tiles_key]

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)

    return adata


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
) -> Optional[Path]:
    """Extract patch features from a single slide using one or more models.

    Uses LazySlide's design: preprocess once, extract all models, write once.
    Saves a single Zarr next to the original slide with all features.

    Args:
        slide_path: Path to WSI file
        models: Model name(s) for feature extraction (string or list of strings)
        tile_px: Tile size in pixels
        mpp: Microns per pixel
        amp: Use automatic mixed precision
        device: Device for inference
        overwrite: Overwrite existing features

    Returns:
        Path to saved Zarr directory (next to original slide)
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

    if zarr_path.exists() and not overwrite:
        logger.info(f"Zarr exists, skipping: {zarr_path}")
        return zarr_path

    try:
        # Open WSI
        logger.info(f"Processing {slide_path.name} with models: {', '.join(models)}")
        wsi = open_wsi(str(slide_path))

        # Preprocess ONCE: tissue detection and tiling
        logger.info("Preprocessing: tissue detection and tiling...")
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

        # Extract ALL models (each adds to wsi.tables)
        for model in models:
            logger.info(f"Extracting features with {model}...")
            zs.tl.feature_extraction(wsi, model=model, amp=amp, device=device)

        # Write ONCE → saves next to original slide
        logger.info("Saving WSI with all features...")
        wsi.write()

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
        )
        results.append(
            {
                "slide_path": slide_path,
                "models": ",".join(models),
                "zarr_path": str(output_path) if output_path else None,
                "success": output_path is not None,
            }
        )

    results_df = pd.DataFrame(results)
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

    logger.info(f"Saved {len(df)} embeddings ({embedding_matrix.shape[1]}D) to {output_dir}")

    return output_dir


def aggregate_simple_pooling(
    slide_table: Union[str, Path, pd.DataFrame],
    models: Union[str, List[str]],
    method: Literal["mean", "max", "median", "sum"] = "mean",
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Aggregate features using simple pooling.

    Loads features from zarr files and applies numpy-based pooling.
    Reads AnnData objects directly from zarr to avoid reader compatibility issues.

    Args:
        slide_table: Path to slide_table.csv or DataFrame with:
                     PATIENT, FILENAME, SITE
        models: Model(s) to aggregate (e.g., "plip" or ["plip", "ctranspath"])
        method: Pooling method (mean, max, median, sum)
        output_dir: Output directory (defaults to results/embeddings/)

    Returns:
        Dict mapping model -> results DataFrame
    """
    import anndata as ad

    # Load slide table
    if isinstance(slide_table, (str, Path)):
        df = pd.read_csv(slide_table)
    else:
        df = slide_table.copy()

    # Normalize models to list
    models = [models] if isinstance(models, str) else models

    results = {}
    for model in models:
        logger.info(f"Aggregating {model} with {method}...")

        embeddings = []
        feature_key = f"{model}_tiles"

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{model} {method}"):
            svs_path = Path(row["FILENAME"])
            zarr_path = svs_path.with_suffix(".zarr")

            if not zarr_path.exists():
                logger.warning(f"Zarr not found: {zarr_path}")
                continue

            try:
                # Load AnnData directly from zarr tables subdirectory
                adata_path = zarr_path / "tables" / feature_key
                if not adata_path.exists():
                    logger.warning(f"No {feature_key} in {zarr_path.name}")
                    continue

                # Read AnnData from zarr
                adata = ad.read_zarr(str(adata_path))

                # Apply pooling method
                if method == "mean":
                    embedding = np.asarray(adata.X.mean(axis=0)).flatten()
                elif method == "max":
                    embedding = np.asarray(adata.X.max(axis=0)).flatten()
                elif method == "median":
                    embedding = np.median(np.asarray(adata.X), axis=0).flatten()
                elif method == "sum":
                    embedding = np.asarray(adata.X.sum(axis=0)).flatten()
                else:
                    embedding = np.asarray(adata.X.mean(axis=0)).flatten()

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
                logger.error(f"Failed to aggregate {zarr_path.name}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                continue

        if not embeddings:
            logger.warning(f"No embeddings generated for {model}")
            continue

        # Convert to DataFrame and save
        df_result = pd.DataFrame(embeddings)
        _save_embeddings(df_result, model, method, output_dir)
        results[model] = df_result

    return results


def aggregate_neural_encoders(
    slide_table: Union[str, Path, pd.DataFrame],
    model: str,
    encoder: Literal["prism", "titan", "chief", "madeleine", "gigapath-slide-encoder"],
    output_dir: Optional[Path] = None,
    device: str = "cuda",
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
        "chief": ["chief"],
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
            # Load original slide (needed for spatial context)
            wsi = open_wsi(str(svs_path))

            # Load pre-extracted features from Zarr (auto-detect reader)
            zarr_wsi = open_wsi(str(zarr_path))
            feature_key = f"{model}_tiles"

            if feature_key not in zarr_wsi.tables:
                logger.warning(f"No {feature_key} in {zarr_path}")
                continue

            # Copy features to avoid re-extraction
            wsi.tables[feature_key] = zarr_wsi.tables[feature_key]

            # Run neural aggregation
            zs.tl.feature_aggregation(
                wsi,
                feature_key=model,  # Base name without "_tiles"
                encoder=encoder,
                device=device,
            )

            # Extract aggregated embedding from AnnData.uns
            # LazySlide stores result in: wsi.tables['{model}_tiles'].uns['agg_slide']
            adata = wsi.tables[feature_key]

            if "agg_slide" in adata.uns:
                # Extract embedding from uns
                embedding = np.asarray(adata.uns["agg_slide"]).flatten()
            elif "agg_slide" in adata.varm:
                # Alternative storage location
                embedding = np.asarray(adata.varm["agg_slide"]).flatten()
            else:
                raise ValueError(
                    f"Aggregation result not found in .uns or .varm for {svs_path.name}"
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
        _save_embeddings(df_result, model, encoder, output_dir)
    else:
        logger.warning("No embeddings generated")

    return df_result


def aggregate_features(
    slide_table: Union[str, Path, pd.DataFrame],
    models: Union[str, List[str]],
    method: str = "mean",
    output_dir: Optional[Path] = None,
    device: str = "cuda",
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
    simple_methods = ["mean", "max", "median", "sum", "std", "var"]

    # Neural encoder methods
    neural_encoders = [
        "prism",
        "titan",
        "chief",
        "madeleine",
        "gigapath-slide-encoder",
    ]

    if method in simple_methods:
        # Use agg_wsi() for simple pooling
        return aggregate_simple_pooling(
            slide_table=slide_table,
            models=models,
            method=method,
            output_dir=output_dir,
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
        )

        return {model: df}

    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. Available: {simple_methods + neural_encoders}"
        )
