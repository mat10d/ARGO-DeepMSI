"""
Feature extraction module using LazySlide.

Supports:
- Patch-level feature extraction (tile → embedding)
- Slide-level aggregation (PRISM, TITAN, mean pooling)
- Spatial analysis (Leiden clustering, neighborhood graphs)
"""

import logging
from pathlib import Path
from typing import Optional, List, Union, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import lazyslide as zs

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

# Slide-level aggregation encoders
SLIDE_ENCODERS = {
    "mean": "Mean pooling (default)",
    "max": "Max pooling",
    "prism": "PRISM slide encoder (requires Virchow features)",
    "titan": "TITAN slide encoder",
    "chief-slide-encoder": "CHIEF slide-level aggregator",
    "gigapath-slide-encoder": "GigaPath slide-level aggregator",
    "gigatime": "GigaTime slide-level encoder",
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
    wsi = zs.WSI(str(slide_path))

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
    model: str = "uni2",
    output_dir: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    overwrite: bool = False,
    save_wsi: bool = False,
) -> Optional[Path]:
    """Extract patch features from a single slide.

    Args:
        slide_path: Path to WSI file
        model: Model name for feature extraction
        output_dir: Output directory
        tile_px: Tile size in pixels
        mpp: Microns per pixel
        amp: Use automatic mixed precision
        device: Device for inference
        overwrite: Overwrite existing features
        save_wsi: Also save the full WSI object (for visualization)

    Returns:
        Path to saved features (.h5ad file)
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    slide_path = Path(slide_path)
    if not slide_path.exists():
        logger.error(f"Slide not found: {slide_path}")
        return None

    if output_dir is None:
        output_dir = get_features_dir(model)
    ensure_dir(output_dir)

    output_path = output_dir / f"{slide_path.stem}.h5ad"

    if output_path.exists() and not overwrite:
        logger.info(f"Features exist, skipping: {output_path}")
        return output_path

    try:
        wsi = process_slide(
            slide_path=slide_path,
            patch_model=model,
            tile_px=tile_px,
            mpp=mpp,
            amp=amp,
            device=device,
        )

        # Save features
        feature_key = f"{model}_tiles"
        if feature_key in wsi:
            wsi[feature_key].write_h5ad(str(output_path))
            logger.info(f"Saved features: {output_path}")

            # Optionally save full WSI for visualization
            if save_wsi:
                wsi_path = output_dir / f"{slide_path.stem}.wsi.zarr"
                wsi.write_zarr(str(wsi_path))

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
    """Extract features from all slides in a table."""
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
        results.append(
            {
                "slide_path": slide_path,
                "model": model,
                "features_path": str(output_path) if output_path else None,
                "success": output_path is not None,
            }
        )

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
    """Extract features from all slides using multiple models."""
    all_results = []

    for model in models:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model: {model}")
        logger.info(f"{'=' * 60}")

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
# Slide-Level Aggregation
# ============================================================================


def aggregate_features(
    features_dir: Union[str, Path],
    model: str,
    method: Literal[
        "mean", "max", "prism", "titan", "chief-slide-encoder", "gigapath-slide-encoder", "gigatime"
    ] = "mean",
    output_dir: Optional[Path] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Aggregate patch features to slide-level embeddings.

    Args:
        features_dir: Directory containing .h5ad feature files
        model: Model name (for organizing outputs)
        method: Aggregation method:
            - "mean": Simple mean pooling
            - "max": Max pooling
            - "prism": PRISM encoder (requires Virchow features)
            - "titan": TITAN encoder
            - "chief-slide-encoder": CHIEF slide aggregator
            - "gigapath-slide-encoder": GigaPath slide aggregator
            - "gigatime": GigaTime encoder
        output_dir: Output directory for embeddings
        device: Device for neural aggregators

    Returns:
        DataFrame with slide-level embeddings
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    import anndata as ad

    features_dir = Path(features_dir)
    feature_files = list(features_dir.glob("*.h5ad"))

    if not feature_files:
        logger.warning(f"No feature files found in {features_dir}")
        return pd.DataFrame()

    # Validate PRISM requirements
    if method == "prism" and model not in ["virchow", "virchow2"]:
        logger.warning(
            f"PRISM encoder works best with Virchow features. Using {model} may give suboptimal results."
        )

    logger.info(f"Aggregating {len(feature_files)} slides with {method}")

    embeddings = []
    for feature_file in tqdm(feature_files, desc=f"Aggregating ({method})"):
        try:
            adata = ad.read_h5ad(feature_file)

            if method == "mean":
                embedding = np.asarray(adata.X.mean(axis=0)).flatten()
            elif method == "max":
                embedding = np.asarray(adata.X.max(axis=0)).flatten()
            elif method in [
                "prism",
                "titan",
                "chief-slide-encoder",
                "gigapath-slide-encoder",
                "gigatime",
            ]:
                # Use LazySlide's neural aggregator
                # Need to reload the slide for this
                # For now, fall back to mean pooling
                # TODO: Implement proper neural aggregation
                logger.warning(f"{method} aggregation requires slide reload. Using mean pooling.")
                embedding = np.asarray(adata.X.mean(axis=0)).flatten()
            else:
                embedding = np.asarray(adata.X.mean(axis=0)).flatten()

            embeddings.append(
                {
                    "slide_id": feature_file.stem,
                    "embedding": embedding,
                    "n_tiles": adata.n_obs,
                }
            )
        except Exception as e:
            logger.error(f"Failed to aggregate {feature_file.name}: {e}")

    if not embeddings:
        return pd.DataFrame()

    df = pd.DataFrame(embeddings)

    # Save embeddings
    if output_dir is None:
        output_dir = get_embeddings_dir(model)
    ensure_dir(output_dir)

    embedding_matrix = np.vstack(df["embedding"].values)

    metadata_df = df[["slide_id", "n_tiles"]].copy()
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    np.save(output_dir / "embeddings.npy", embedding_matrix)
    np.save(output_dir / f"embeddings_{method}.npy", embedding_matrix)

    logger.info(f"Saved {len(df)} embeddings ({embedding_matrix.shape[1]}D) to {output_dir}")

    return df


def aggregate_with_encoder(
    slide_paths: List[Union[str, Path]],
    patch_model: str = "virchow",
    slide_encoder: str = "prism",
    output_dir: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
) -> pd.DataFrame:
    """Extract features and aggregate with neural encoder in one pass.

    This is the proper way to use PRISM/TITAN - process each slide
    end-to-end rather than loading saved features.

    Args:
        slide_paths: List of slide paths
        patch_model: Patch feature model ('virchow' for PRISM)
        slide_encoder: Slide encoder ('prism', 'titan')
        output_dir: Output directory
        tile_px: Tile size
        mpp: Microns per pixel
        amp: Use mixed precision
        device: Device for inference

    Returns:
        DataFrame with slide-level embeddings
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    if slide_encoder == "prism" and patch_model not in ["virchow", "virchow2"]:
        raise ValueError("PRISM requires Virchow features. Set patch_model='virchow'")

    logger.info(f"Processing {len(slide_paths)} slides with {patch_model} + {slide_encoder}")

    embeddings = []
    for slide_path in tqdm(slide_paths, desc=f"{patch_model}+{slide_encoder}"):
        try:
            wsi, slide_embedding = process_slide_with_aggregation(
                slide_path=slide_path,
                patch_model=patch_model,
                slide_encoder=slide_encoder,
                tile_px=tile_px,
                mpp=mpp,
                amp=amp,
                device=device,
            )

            if slide_embedding is not None:
                embeddings.append(
                    {
                        "slide_id": Path(slide_path).stem,
                        "embedding": np.asarray(slide_embedding).flatten(),
                        "patch_model": patch_model,
                        "slide_encoder": slide_encoder,
                    }
                )
        except Exception as e:
            logger.error(f"Failed: {Path(slide_path).name}: {e}")

    if not embeddings:
        return pd.DataFrame()

    df = pd.DataFrame(embeddings)

    # Save
    if output_dir is None:
        output_dir = get_embeddings_dir(f"{patch_model}_{slide_encoder}")
    ensure_dir(output_dir)

    embedding_matrix = np.vstack(df["embedding"].values)
    metadata_df = df[["slide_id", "patch_model", "slide_encoder"]].copy()

    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    np.save(output_dir / "embeddings.npy", embedding_matrix)

    logger.info(f"Saved {len(df)} slide embeddings to {output_dir}")

    return df
