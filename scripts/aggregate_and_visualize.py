#!/usr/bin/env python3
"""
Aggregate features and generate visualizations for all extracted models.

This script:
1. Finds all extracted feature directories
2. Aggregates patch features to slide-level embeddings
3. Generates UMAP/t-SNE visualizations for each model
4. Creates comparison plots across models

Usage:
    python scripts/aggregate_and_visualize.py

    # With clinical data for colored plots
    python scripts/aggregate_and_visualize.py --clinical results/data/clinical_table.csv

    # Specific models only
    python scripts/aggregate_and_visualize.py --models uni2 virchow2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from argo_deepmsi.feature_extraction import aggregate_features
from argo_deepmsi.visualization import (
    plot_embedding_umap,
    plot_embedding_tsne,
    plot_model_comparison,
)
from argo_deepmsi.io_utils import (
    setup_logging,
    get_features_dir,
    get_embeddings_dir,
    get_visualizations_dir,
    ensure_dir,
)


def find_extracted_models(features_base: Path) -> List[str]:
    """Find all models that have extracted features."""
    if not features_base.exists():
        return []

    models = []
    for model_dir in features_base.iterdir():
        if model_dir.is_dir():
            h5ad_files = list(model_dir.glob("*.h5ad"))
            if h5ad_files:
                models.append(model_dir.name)

    return sorted(models)


def aggregate_all_models(
    models: List[str],
    method: str = "mean",
    overwrite: bool = False,
) -> dict:
    """Aggregate features for all models."""
    logger = setup_logging("aggregate")

    results = {}
    for model in models:
        features_dir = get_features_dir(model)
        embeddings_dir = get_embeddings_dir(model)

        # Check if already aggregated
        if (embeddings_dir / "embeddings.npy").exists() and not overwrite:
            logger.info(f"Embeddings exist for {model}, loading...")
            embeddings = np.load(embeddings_dir / "embeddings.npy")
            metadata = pd.read_csv(embeddings_dir / "metadata.csv")
            results[model] = {"embeddings": embeddings, "metadata": metadata}
            continue

        logger.info(f"Aggregating {model}...")
        try:
            df = aggregate_features(
                features_dir=features_dir,
                model=model,
                method=method,
                output_dir=embeddings_dir,
            )
            if not df.empty:
                embeddings = np.load(embeddings_dir / "embeddings.npy")
                metadata = pd.read_csv(embeddings_dir / "metadata.csv")
                results[model] = {"embeddings": embeddings, "metadata": metadata}
                logger.info(f"  Aggregated {len(df)} slides for {model}")
        except Exception as e:
            logger.error(f"  Failed to aggregate {model}: {e}")

    return results


def visualize_all_models(
    model_data: dict,
    clinical_table: Optional[pd.DataFrame] = None,
    label_column: str = "isMSIH",
    output_dir: Optional[Path] = None,
):
    """Generate visualizations for all models."""
    logger = setup_logging("visualize")

    if output_dir is None:
        output_dir = get_visualizations_dir()
    ensure_dir(output_dir)

    # Per-model visualizations
    for model, data in model_data.items():
        embeddings = data["embeddings"]
        metadata = data["metadata"]

        logger.info(f"Visualizing {model} ({len(embeddings)} slides)...")

        # Get labels if clinical data provided
        labels = None
        if clinical_table is not None and label_column in clinical_table.columns:
            merged = metadata.merge(
                clinical_table[["PATIENT", label_column]],
                left_on="slide_id",
                right_on="PATIENT",
                how="left"
            )
            if label_column in merged.columns:
                labels = merged[label_column].values

        # UMAP
        try:
            plot_embedding_umap(
                embeddings=embeddings,
                labels=labels,
                output_path=output_dir / f"umap_{model}.png",
                title=f"{model} - UMAP",
            )
        except Exception as e:
            logger.warning(f"  UMAP failed for {model}: {e}")

        # t-SNE (can be slow for large datasets)
        if len(embeddings) <= 1000:
            try:
                plot_embedding_tsne(
                    embeddings=embeddings,
                    labels=labels,
                    output_path=output_dir / f"tsne_{model}.png",
                    title=f"{model} - t-SNE",
                )
            except Exception as e:
                logger.warning(f"  t-SNE failed for {model}: {e}")

    # Combined visualization - all models on same plot
    if len(model_data) > 1:
        logger.info("Creating combined visualization...")
        create_combined_plot(model_data, clinical_table, label_column, output_dir)


def create_combined_plot(
    model_data: dict,
    clinical_table: Optional[pd.DataFrame],
    label_column: str,
    output_dir: Path,
):
    """Create a grid of UMAP plots for all models."""
    try:
        from umap import UMAP
    except ImportError:
        print("umap-learn not installed, skipping combined plot")
        return

    n_models = len(model_data)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (model, data) in enumerate(model_data.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        embeddings = data["embeddings"]
        metadata = data["metadata"]

        # Get labels
        labels = None
        if clinical_table is not None:
            merged = metadata.merge(
                clinical_table[["PATIENT", label_column]],
                left_on="slide_id",
                right_on="PATIENT",
                how="left"
            )
            if label_column in merged.columns:
                labels = merged[label_column].values

        # Compute UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

        # Plot
        if labels is not None:
            unique_labels = np.unique(labels[~pd.isna(labels)])
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                          c=[colors[i]], label=str(label), alpha=0.6, s=20)
            if idx == 0:
                ax.legend(fontsize=8)
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=20)

        ax.set_title(model, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle("UMAP Embeddings by Model", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "umap_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate and visualize features for all models")

    parser.add_argument("--models", nargs="+", help="Specific models to process (default: all found)")
    parser.add_argument("--clinical", type=Path, help="Clinical table for colored plots")
    parser.add_argument("--label", default="isMSIH", help="Label column for coloring")
    parser.add_argument("--method", default="mean", choices=["mean", "max"], help="Aggregation method")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    parser.add_argument("--output", type=Path, help="Output directory for visualizations")

    args = parser.parse_args()

    # Find models
    features_base = get_features_dir()
    available_models = find_extracted_models(features_base)

    if not available_models:
        print(f"No extracted features found in {features_base}")
        print("Run feature extraction first: argo extract slide_table.csv --model uni2")
        return

    models = args.models if args.models else available_models
    print(f"Found models: {', '.join(available_models)}")
    print(f"Processing: {', '.join(models)}")

    # Load clinical data
    clinical_table = None
    if args.clinical and args.clinical.exists():
        clinical_table = pd.read_csv(args.clinical)
        print(f"Loaded clinical table: {len(clinical_table)} records")

    # Aggregate
    print("\nAggregating features...")
    model_data = aggregate_all_models(models, method=args.method, overwrite=args.overwrite)

    if not model_data:
        print("No models successfully aggregated")
        return

    # Visualize
    print("\nGenerating visualizations...")
    visualize_all_models(
        model_data=model_data,
        clinical_table=clinical_table,
        label_column=args.label,
        output_dir=args.output,
    )

    print(f"\nDone! Visualizations saved to {args.output or get_visualizations_dir()}")


if __name__ == "__main__":
    main()
