"""
Feature extraction module using LazySlide.

Supports both patch-level and slide-level feature extractors.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

try:
    import lazyslide as zs
    from lazyslide.models import list_models
    LAZYSLIDE_AVAILABLE = True
except ImportError:
    LAZYSLIDE_AVAILABLE = False

from .io_utils import get_features_dir, get_embeddings_dir, ensure_dir

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
    "resnet50": ModelConfig("resnet50", "patch", False, 256, 0.5, "ImageNet pretrained ResNet50"),
    "ctranspath": ModelConfig("ctranspath", "patch", False, 256, 0.5, "CTransPath pathology foundation model"),
    "plip": ModelConfig("plip", "patch", False, 256, 0.5, "PLIP vision-language model"),

    # Gated models (require HuggingFace auth)
    "uni": ModelConfig("uni", "patch", True, 256, 0.5, "UNI pathology foundation model"),
    "uni2": ModelConfig("uni2", "patch", True, 256, 0.5, "UNI2 pathology foundation model"),
    "virchow": ModelConfig("virchow", "patch", True, 256, 0.5, "Virchow pathology foundation model"),
    "virchow2": ModelConfig("virchow2", "patch", True, 256, 0.5, "Virchow2 (631M params, 2560 features)"),
    "conch": ModelConfig("conch", "patch", True, 256, 0.5, "CONCH vision-language model"),
    "gigapath": ModelConfig("gigapath", "patch", True, 256, 0.5, "GigaPath pathology foundation model"),
    "h-optimus-0": ModelConfig("h-optimus-0", "patch", True, 256, 0.5, "H-Optimus-0 pathology model"),
    "h-optimus-1": ModelConfig("h-optimus-1", "patch", True, 256, 0.5, "H-Optimus-1 pathology model"),
}

# Slide-level feature extractors (WSI → single embedding)
SLIDE_MODELS = {
    "prism": ModelConfig("prism", "slide", True, description="PRISM slide-level aggregator"),
    "threads": ModelConfig("threads", "slide", True, description="THREADS slide-level model"),
}

ALL_MODELS = {**PATCH_MODELS, **SLIDE_MODELS}


def list_available_models(model_type: Optional[str] = None) -> List[str]:
    """List available models for feature extraction.

    Args:
        model_type: Filter by type - "patch", "slide", or None for all

    Returns:
        List of model names
    """
    if model_type == "patch":
        return list(PATCH_MODELS.keys())
    elif model_type == "slide":
        return list(SLIDE_MODELS.keys())
    else:
        return list(ALL_MODELS.keys())


def get_model_info(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a model."""
    return ALL_MODELS.get(model_name)


# ============================================================================
# Feature extraction
# ============================================================================

def extract_features_single_slide(
    slide_path: Union[str, Path],
    model: str = "uni2",
    output_dir: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    overwrite: bool = False,
) -> Optional[Path]:
    """Extract patch features from a single slide.

    Args:
        slide_path: Path to WSI file (.svs, .ndpi, etc.)
        model: Model name for feature extraction
        output_dir: Output directory (default: results/features/{model}/)
        tile_px: Tile size in pixels
        mpp: Microns per pixel for tiling
        amp: Use automatic mixed precision
        device: Device for inference ("cuda" or "cpu")
        overwrite: Overwrite existing features

    Returns:
        Path to saved features (AnnData .h5ad file) or None if failed
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed. Run: pip install lazyslide")

    slide_path = Path(slide_path)
    if not slide_path.exists():
        logger.error(f"Slide not found: {slide_path}")
        return None

    # Setup output
    if output_dir is None:
        output_dir = get_features_dir(model)
    ensure_dir(output_dir)

    output_path = output_dir / f"{slide_path.stem}.h5ad"

    if output_path.exists() and not overwrite:
        logger.info(f"Features exist, skipping: {output_path}")
        return output_path

    try:
        logger.info(f"Processing: {slide_path.name}")

        # Load slide
        wsi = zs.WSI(str(slide_path))

        # Find tissue regions
        zs.pp.find_tissues(wsi)

        # Tile tissues
        zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

        # Extract features
        zs.tl.feature_extraction(wsi, model=model, amp=amp, device=device)

        # Save features
        feature_key = f"{model}_tiles"
        if feature_key in wsi:
            wsi[feature_key].write_h5ad(str(output_path))
            logger.info(f"Saved features: {output_path}")
            return output_path
        else:
            logger.warning(f"No features extracted for: {slide_path.name}")
            return None

    except Exception as e:
        logger.error(f"Failed to process {slide_path.name}: {e}")
        return None


def extract_features_batch(
    slide_table: pd.DataFrame,
    model: str = "uni2",
    output_dir: Optional[Path] = None,
    slide_column: str = "FILENAME",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    overwrite: bool = False,
    max_slides: Optional[int] = None,
) -> pd.DataFrame:
    """Extract features from all slides in a table.

    Args:
        slide_table: DataFrame with slide paths
        model: Model name for feature extraction
        output_dir: Output directory
        slide_column: Column containing slide paths
        tile_px: Tile size in pixels
        mpp: Microns per pixel
        amp: Use automatic mixed precision
        device: Device for inference
        overwrite: Overwrite existing features
        max_slides: Maximum number of slides to process (for testing)

    Returns:
        DataFrame with extraction status for each slide
    """
    if slide_column not in slide_table.columns:
        raise ValueError(f"Column '{slide_column}' not found in slide table")

    slides = slide_table[slide_column].dropna().unique()
    if max_slides:
        slides = slides[:max_slides]

    logger.info(f"Extracting {model} features from {len(slides)} slides")

    if output_dir is None:
        output_dir = get_features_dir(model)
    ensure_dir(output_dir)

    results = []
    for slide_path in tqdm(slides, desc=f"Extracting {model}"):
        output_path = extract_features_single_slide(
            slide_path=slide_path,
            model=model,
            output_dir=output_dir,
            tile_px=tile_px,
            mpp=mpp,
            amp=amp,
            device=device,
            overwrite=overwrite,
        )
        results.append({
            "slide_path": slide_path,
            "model": model,
            "features_path": str(output_path) if output_path else None,
            "success": output_path is not None,
        })

    results_df = pd.DataFrame(results)
    success_count = results_df["success"].sum()
    logger.info(f"Completed: {success_count}/{len(slides)} slides successful")

    return results_df


def extract_features_multi_model(
    slide_table: pd.DataFrame,
    models: List[str],
    slide_column: str = "FILENAME",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    overwrite: bool = False,
    max_slides: Optional[int] = None,
) -> pd.DataFrame:
    """Extract features from all slides using multiple models.

    Args:
        slide_table: DataFrame with slide paths
        models: List of model names
        slide_column: Column containing slide paths
        tile_px: Tile size in pixels
        mpp: Microns per pixel
        amp: Use automatic mixed precision
        device: Device for inference
        overwrite: Overwrite existing features
        max_slides: Maximum number of slides to process

    Returns:
        Combined DataFrame with extraction status for all models
    """
    all_results = []

    for model in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model}")
        logger.info(f"{'='*60}")

        results = extract_features_batch(
            slide_table=slide_table,
            model=model,
            slide_column=slide_column,
            tile_px=tile_px,
            mpp=mpp,
            amp=amp,
            device=device,
            overwrite=overwrite,
            max_slides=max_slides,
        )
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


# ============================================================================
# Slide-level aggregation
# ============================================================================

def aggregate_features(
    features_dir: Union[str, Path],
    model: str,
    method: str = "mean",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Aggregate patch features to slide-level embeddings.

    Args:
        features_dir: Directory containing .h5ad feature files
        model: Model name (for organizing outputs)
        method: Aggregation method - "mean", "max", or "attention"
        output_dir: Output directory for embeddings

    Returns:
        DataFrame with slide-level embeddings
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    import anndata as ad
    import numpy as np

    features_dir = Path(features_dir)
    feature_files = list(features_dir.glob("*.h5ad"))

    if not feature_files:
        logger.warning(f"No feature files found in {features_dir}")
        return pd.DataFrame()

    logger.info(f"Aggregating {len(feature_files)} slides with {method} pooling")

    embeddings = []
    for feature_file in tqdm(feature_files, desc="Aggregating"):
        try:
            adata = ad.read_h5ad(feature_file)

            if method == "mean":
                embedding = adata.X.mean(axis=0)
            elif method == "max":
                embedding = adata.X.max(axis=0)
            else:
                embedding = adata.X.mean(axis=0)  # fallback

            embeddings.append({
                "slide_id": feature_file.stem,
                "embedding": embedding,
                "n_tiles": adata.n_obs,
            })
        except Exception as e:
            logger.error(f"Failed to aggregate {feature_file.name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(embeddings)

    # Save embeddings
    if output_dir is None:
        output_dir = get_embeddings_dir(model)
    ensure_dir(output_dir)

    # Save as parquet with embeddings as arrays
    embedding_matrix = np.vstack(df["embedding"].values)

    metadata_df = df[["slide_id", "n_tiles"]]
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    np.save(output_dir / "embeddings.npy", embedding_matrix)

    logger.info(f"Saved {len(df)} slide embeddings to {output_dir}")

    return df
