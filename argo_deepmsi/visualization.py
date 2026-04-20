"""
Visualization module using LazySlide.

Provides slide visualization, feature exploration, and data analysis.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import lazyslide as zs
    from wsidata import open_wsi

    LAZYSLIDE_AVAILABLE = True
except ImportError:
    LAZYSLIDE_AVAILABLE = False

from .io_utils import get_visualizations_dir, ensure_dir

logger = logging.getLogger(__name__)


# ============================================================================
# Cached loading — avoid re-running GPU inference when zarr already exists
# ============================================================================


def _open_cached(
    slide_path: Path,
    tile_px: int = 256,
    mpp: float = 0.5,
    model: Optional[str] = None,
    ensure_tiles: bool = True,
    device: str = "cuda",
):
    """Open a WSI, preferring the cached zarr if it exists.

    - If `{slide}.zarr` exists, open it (preprocessing/features already cached).
    - Otherwise open the raw slide and run the minimum preprocessing required.
    - If `model` is given and the feature table is missing, run extraction and
      persist it back to zarr so subsequent calls are cheap.
    """
    zarr_path = slide_path.with_suffix(".zarr")
    if zarr_path.exists():
        wsi = open_wsi(str(zarr_path))
    else:
        wsi = open_wsi(str(slide_path))
        zs.pp.find_tissues(wsi)
        if ensure_tiles or model is not None:
            zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

    if model is not None:
        feature_key = f"{model}_tiles"
        if not hasattr(wsi, "tables") or feature_key not in wsi.tables:
            zs.tl.feature_extraction(
                wsi, model=model, device=device, num_workers=4, batch_size=64, pbar=False
            )
            wsi.write()

    return wsi


# ============================================================================
# Slide visualization
# ============================================================================


def visualize_slide(
    slide_path: Union[str, Path],
    output_path: Optional[Path] = None,
    show_tissues: bool = True,
    show_tiles: bool = True,
    tile_px: int = 256,
    mpp: float = 0.5,
    figsize: Tuple[int, int] = (15, 10),
) -> Optional[plt.Figure]:
    """Visualize a single slide with tissue detection and tiling.

    Args:
        slide_path: Path to WSI file
        output_path: Path to save figure (optional)
        show_tissues: Show detected tissue regions
        show_tiles: Show tile grid
        tile_px: Tile size for visualization
        mpp: Microns per pixel
        figsize: Figure size

    Returns:
        Matplotlib figure or None
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    slide_path = Path(slide_path)

    try:
        wsi = _open_cached(slide_path, tile_px=tile_px, mpp=mpp, ensure_tiles=show_tiles)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Original slide thumbnail
        axes[0].set_title("Original Slide")
        zs.pl.wsi(wsi, ax=axes[0])

        # 2. Tissue detection (already run by _open_cached)
        axes[1].set_title("Tissue Detection")
        zs.pl.tissues(wsi, ax=axes[1])

        # 3. Tiling
        if show_tiles:
            axes[2].set_title(f"Tiling ({tile_px}px @ {mpp} mpp)")
            zs.pl.tiles(wsi, ax=axes[2])
        else:
            axes[2].axis("off")

        fig.suptitle(slide_path.name, fontsize=14)
        fig.tight_layout()

        if output_path:
            ensure_dir(output_path.parent)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved visualization: {output_path}")

        return fig

    except Exception as e:
        logger.error(f"Failed to visualize {slide_path.name}: {e}")
        return None


def visualize_features(
    slide_path: Union[str, Path],
    model: str,
    feature_indices: List[int] = [0, 1],
    output_path: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    figsize: Tuple[int, int] = (15, 5),
) -> Optional[plt.Figure]:
    """Visualize extracted features on a slide.

    Args:
        slide_path: Path to WSI file
        model: Model used for feature extraction
        feature_indices: Which feature dimensions to visualize
        output_path: Path to save figure
        tile_px: Tile size
        mpp: Microns per pixel
        figsize: Figure size

    Returns:
        Matplotlib figure or None
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    slide_path = Path(slide_path)

    try:
        wsi = _open_cached(slide_path, tile_px=tile_px, mpp=mpp, model=model)

        n_features = len(feature_indices)
        fig, axes = plt.subplots(1, n_features + 1, figsize=figsize)

        # Original slide
        axes[0].set_title("Original")
        zs.pl.wsi(wsi, ax=axes[0])

        # Feature maps
        for i, feat_idx in enumerate(feature_indices):
            axes[i + 1].set_title(f"Feature {feat_idx}")
            zs.pl.tiles(wsi, feature_key=model, color=[str(feat_idx)], ax=axes[i + 1])

        fig.suptitle(f"{slide_path.name} - {model} features", fontsize=14)
        fig.tight_layout()

        if output_path:
            ensure_dir(output_path.parent)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    except Exception as e:
        logger.error(f"Failed to visualize features for {slide_path.name}: {e}")
        return None


# ============================================================================
# Embedding visualization (UMAP, t-SNE)
# ============================================================================


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    slide_ids: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Slide Embeddings (UMAP)",
    figsize: Tuple[int, int] = (10, 8),
    **umap_kwargs,
) -> plt.Figure:
    """Plot UMAP visualization of slide embeddings.

    Args:
        embeddings: Array of shape (n_slides, n_features)
        labels: Optional labels for coloring (e.g., MSI status)
        slide_ids: Optional slide identifiers for hover
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        **umap_kwargs: Additional arguments to UMAP

    Returns:
        Matplotlib figure
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")

    # Default UMAP parameters
    umap_params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42,
    }
    umap_params.update(umap_kwargs)

    logger.info(f"Computing UMAP for {len(embeddings)} slides")
    reducer = UMAP(**umap_params)
    embedding_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=50,
            )
        ax.legend(title="Label")
    else:
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=50)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    if output_path:
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved UMAP plot: {output_path}")

    return fig


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "Slide Embeddings (t-SNE)",
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
) -> plt.Figure:
    """Plot t-SNE visualization of slide embeddings.

    Args:
        embeddings: Array of shape (n_slides, n_features)
        labels: Optional labels for coloring
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        perplexity: t-SNE perplexity parameter

    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE

    logger.info(f"Computing t-SNE for {len(embeddings)} slides")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=50,
            )
        ax.legend(title="Label")
    else:
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=50)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)

    if output_path:
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved t-SNE plot: {output_path}")

    return fig


# ============================================================================
# Data exploration plots
# ============================================================================


def plot_dataset_summary(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_dir: Optional[Path] = None,
    label_column: str = "isMSIH",
) -> List[plt.Figure]:
    """Generate summary plots for the dataset.

    Args:
        clinical_table: DataFrame with patient info
        slide_table: DataFrame with slide info
        output_dir: Directory to save figures
        label_column: Column containing labels (e.g., MSI status)

    Returns:
        List of matplotlib figures
    """
    if output_dir is None:
        output_dir = get_visualizations_dir() / "data_summary"
    ensure_dir(output_dir)

    figures = []

    # 1. Label distribution
    if label_column in clinical_table.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        clinical_table[label_column].value_counts().plot(
            kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"]
        )
        ax.set_title(f"Label Distribution ({label_column})")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        fig.savefig(output_dir / "label_distribution.png", dpi=150, bbox_inches="tight")
        figures.append(fig)

    # 2. Slides per patient
    slides_per_patient = slide_table.groupby("PATIENT").size()
    fig, ax = plt.subplots(figsize=(8, 6))
    slides_per_patient.hist(bins=20, ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Slides per Patient")
    ax.set_xlabel("Number of Slides")
    ax.set_ylabel("Number of Patients")
    fig.savefig(output_dir / "slides_per_patient.png", dpi=150, bbox_inches="tight")
    figures.append(fig)

    # 3. Site distribution (if available)
    if "SITE" in slide_table.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        slide_table["SITE"].value_counts().plot(kind="bar", ax=ax, color="#9b59b6")
        ax.set_title("Slides by Site")
        ax.set_xlabel("Site")
        ax.set_ylabel("Number of Slides")
        plt.xticks(rotation=45, ha="right")
        fig.savefig(output_dir / "site_distribution.png", dpi=150, bbox_inches="tight")
        figures.append(fig)

    logger.info(f"Saved {len(figures)} summary plots to {output_dir}")
    return figures


def plot_model_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metric: str = "auroc",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot comparison of model performance.

    Args:
        results_df: DataFrame with model results (columns: model, metric)
        output_path: Path to save figure
        metric: Metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = results_df["model"].unique()
    x = np.arange(len(models))
    values = [results_df[results_df["model"] == m][metric].mean() for m in models]

    bars = ax.bar(x, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison - {metric.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================================
# LazySlide-native tile visualization
# ============================================================================


def visualize_tile_clusters(
    slide_path: Union[str, Path],
    model: str = "uni2",
    output_path: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    resolution: float = 1.0,
    figsize: Tuple[int, int] = (18, 6),
    device: str = "cuda",
) -> Optional[plt.Figure]:
    """Visualize tile-level Leiden clusters on a slide.

    This performs:
    1. Feature extraction
    2. Neighbor graph construction
    3. Leiden clustering
    4. Spatial visualization of clusters

    Args:
        slide_path: Path to WSI file
        model: Feature extraction model
        output_path: Path to save figure
        tile_px: Tile size
        mpp: Microns per pixel
        resolution: Leiden clustering resolution
        figsize: Figure size
        device: Device for inference

    Returns:
        Matplotlib figure or None
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required. Run: pip install scanpy")

    slide_path = Path(slide_path)

    try:
        # Load cached zarr if available; only run GPU extraction if truly missing
        wsi = _open_cached(slide_path, tile_px=tile_px, mpp=mpp, model=model, device=device)

        # Get features and perform clustering
        feature_key = f"{model}_tiles"
        adata = wsi[feature_key]

        # Compute neighbors and cluster
        sc.pp.neighbors(adata, n_neighbors=15)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Original slide
        axes[0].set_title("Original Slide")
        zs.pl.wsi(wsi, ax=axes[0])

        # 2. Spatial cluster map
        axes[1].set_title(f"Tile Clusters (Leiden, res={resolution})")
        zs.pl.tiles(wsi, feature_key=model, color="leiden", alpha=0.6, ax=axes[1])

        # 3. UMAP of tiles colored by cluster
        axes[2].set_title("Tile UMAP")
        sc.pl.umap(adata, color="leiden", ax=axes[2], show=False)

        fig.suptitle(f"{slide_path.name} - {model}", fontsize=14)
        fig.tight_layout()

        if output_path:
            ensure_dir(output_path.parent)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved cluster visualization: {output_path}")

        return fig

    except Exception as e:
        logger.error(f"Failed to visualize clusters for {slide_path.name}: {e}")
        return None


def visualize_feature_heatmap(
    slide_path: Union[str, Path],
    model: str = "uni2",
    feature_idx: int = 0,
    output_path: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 5),
    device: str = "cuda",
) -> Optional[plt.Figure]:
    """Visualize a single feature dimension as a spatial heatmap.

    Args:
        slide_path: Path to WSI file
        model: Feature extraction model
        feature_idx: Which feature dimension to visualize
        output_path: Path to save figure
        tile_px: Tile size
        mpp: Microns per pixel
        cmap: Colormap for heatmap
        figsize: Figure size
        device: Device for inference

    Returns:
        Matplotlib figure or None
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    slide_path = Path(slide_path)

    try:
        wsi = _open_cached(slide_path, tile_px=tile_px, mpp=mpp, model=model, device=device)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Original slide
        axes[0].set_title("Original Slide")
        zs.pl.wsi(wsi, ax=axes[0])

        # Feature heatmap
        axes[1].set_title(f"Feature {feature_idx} Heatmap")
        zs.pl.tiles(
            wsi,
            feature_key=model,
            color=[str(feature_idx)],
            style="heatmap",
            cmap=cmap,
            ax=axes[1],
        )

        fig.suptitle(f"{slide_path.name} - {model} Feature {feature_idx}", fontsize=14)
        fig.tight_layout()

        if output_path:
            ensure_dir(output_path.parent)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    except Exception as e:
        logger.error(f"Failed to create heatmap for {slide_path.name}: {e}")
        return None


def explore_slide(
    slide_path: Union[str, Path],
    model: str = "uni2",
    output_dir: Optional[Path] = None,
    tile_px: int = 256,
    mpp: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Generate comprehensive exploration visualizations for a slide.

    Creates:
    - Overview (slide + tissue + tiles)
    - Cluster visualization
    - Top 5 feature heatmaps
    - Tile UMAP

    Args:
        slide_path: Path to WSI file
        model: Feature extraction model
        output_dir: Directory to save figures
        tile_px: Tile size
        mpp: Microns per pixel
        device: Device for inference

    Returns:
        Dictionary with paths to generated figures
    """
    if not LAZYSLIDE_AVAILABLE:
        raise ImportError("LazySlide is not installed")

    slide_path = Path(slide_path)
    slide_name = slide_path.stem

    if output_dir is None:
        output_dir = get_visualizations_dir() / "slides" / slide_name
    ensure_dir(output_dir)

    logger.info(f"Exploring slide: {slide_name}")

    results = {"slide": str(slide_path), "model": model, "figures": {}}

    # 1. Overview
    try:
        fig = visualize_slide(
            slide_path=slide_path,
            output_path=output_dir / f"{slide_name}_overview.png",
            tile_px=tile_px,
            mpp=mpp,
        )
        if fig:
            results["figures"]["overview"] = str(output_dir / f"{slide_name}_overview.png")
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Overview failed: {e}")

    # 2. Cluster visualization
    try:
        fig = visualize_tile_clusters(
            slide_path=slide_path,
            model=model,
            output_path=output_dir / f"{slide_name}_clusters.png",
            tile_px=tile_px,
            mpp=mpp,
            device=device,
        )
        if fig:
            results["figures"]["clusters"] = str(output_dir / f"{slide_name}_clusters.png")
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Cluster viz failed: {e}")

    # 3. Feature heatmaps (top 5 features)
    for feat_idx in range(5):
        try:
            fig = visualize_feature_heatmap(
                slide_path=slide_path,
                model=model,
                feature_idx=feat_idx,
                output_path=output_dir / f"{slide_name}_feature_{feat_idx}.png",
                tile_px=tile_px,
                mpp=mpp,
                device=device,
            )
            if fig:
                results["figures"][f"feature_{feat_idx}"] = str(
                    output_dir / f"{slide_name}_feature_{feat_idx}.png"
                )
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Feature {feat_idx} heatmap failed: {e}")

    logger.info(f"Generated {len(results['figures'])} figures for {slide_name}")

    return results
