"""
CLI entry point for ARGO-DeepMSI.

Single command interface for the entire pipeline:
    argo ingest      - Data ingestion from REDCap
    argo extract     - Feature extraction with LazySlide
    argo aggregate   - Aggregate patch features to slide embeddings
    argo visualize   - Generate visualizations
    argo train       - Train classifiers on embeddings
    argo run         - Run full pipeline
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="argo",
    help="ARGO-DeepMSI: MSI prediction from whole slide images using LazySlide",
    add_completion=False,
)
console = Console()


# ============================================================================
# Data ingestion
# ============================================================================


@app.command()
def ingest(
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="REDCap API URL"),
    api_token: Optional[str] = typer.Option(None, "--api-token", help="REDCap API token"),
):
    """Ingest data from REDCap and Halo Link exports."""
    from .io_utils import setup_logging, get_results_dir, ensure_dir
    from .data_ingestion import process_redcap_data

    setup_logging("ingest")

    if output_dir is None:
        output_dir = get_results_dir() / "data"
    ensure_dir(output_dir)

    console.print("[bold blue]ARGO-DeepMSI: Data Ingestion[/bold blue]")
    console.print(f"Output: {output_dir}")

    clinical_table, slide_table = process_redcap_data(
        output_dir=output_dir,
        api_url=api_url,
        api_token=api_token,
    )

    console.print(f"[green]Done![/green] {len(clinical_table)} patients, {len(slide_table)} slides")


# ============================================================================
# Feature extraction
# ============================================================================


@app.command()
def extract(
    slide_table: Path = typer.Argument(..., help="Path to slide table CSV"),
    models: List[str] = typer.Option(["uni2"], "--model", "-m", help="Models to use"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    tile_px: int = typer.Option(256, "--tile-px", help="Tile size in pixels"),
    mpp: float = typer.Option(0.5, "--mpp", help="Microns per pixel"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use automatic mixed precision"),
    max_slides: Optional[int] = typer.Option(None, "--max-slides", help="Max slides (for testing)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing features"),
):
    """Extract features from slides using LazySlide.

    Example:
        argo extract slide_table.csv --model uni2 --model virchow2
    """
    import pandas as pd
    from .io_utils import setup_logging
    from .feature_extraction import extract_features_multi_model, list_available_models

    setup_logging("extract")

    console.print("[bold blue]ARGO-DeepMSI: Feature Extraction[/bold blue]")
    console.print(f"Slide table: {slide_table}")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Device: {device}")

    # Validate models
    available = list_available_models()
    for model in models:
        if model not in available:
            console.print(f"[red]Unknown model: {model}[/red]")
            console.print(f"Available: {', '.join(available)}")
            raise typer.Exit(1)

    # Load slide table
    df = pd.read_csv(slide_table)
    console.print(f"Loaded {len(df)} slides")

    # Extract features
    results = extract_features_multi_model(
        slide_table=df,
        models=models,
        tile_px=tile_px,
        mpp=mpp,
        amp=amp,
        device=device,
        overwrite=overwrite,
        max_slides=max_slides,
    )

    # Summary
    for model in models:
        model_results = results[results["model"] == model]
        success = model_results["success"].sum()
        total = len(model_results)
        console.print(f"[green]{model}[/green]: {success}/{total} successful")


@app.command()
def models():
    """List available feature extraction models."""
    from .feature_extraction import PATCH_MODELS, SLIDE_MODELS

    console.print("[bold blue]Available Models[/bold blue]\n")

    # Patch models
    table = Table(title="Patch-Level Extractors")
    table.add_column("Model", style="cyan")
    table.add_column("Auth Required", style="yellow")
    table.add_column("Description")

    for name, config in PATCH_MODELS.items():
        auth = "Yes" if config.requires_auth else "No"
        table.add_row(name, auth, config.description)

    console.print(table)

    # Slide models
    table = Table(title="Slide-Level Extractors")
    table.add_column("Model", style="cyan")
    table.add_column("Auth Required", style="yellow")
    table.add_column("Description")

    for name, config in SLIDE_MODELS.items():
        auth = "Yes" if config.requires_auth else "No"
        table.add_row(name, auth, config.description)

    console.print(table)


# ============================================================================
# Aggregation
# ============================================================================


@app.command()
def aggregate(
    features_dir: Path = typer.Argument(..., help="Directory with .h5ad feature files"),
    model: str = typer.Argument(..., help="Model name (for organizing outputs)"),
    method: str = typer.Option("mean", "--method", help="Aggregation method (mean/max)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Aggregate patch features to slide-level embeddings."""
    from .io_utils import setup_logging
    from .feature_extraction import aggregate_features

    setup_logging("aggregate")

    console.print("[bold blue]ARGO-DeepMSI: Feature Aggregation[/bold blue]")
    console.print(f"Features: {features_dir}")
    console.print(f"Method: {method}")

    results = aggregate_features(
        features_dir=features_dir,
        model=model,
        method=method,
        output_dir=output_dir,
    )

    console.print(f"[green]Done![/green] Aggregated {len(results)} slides")


# ============================================================================
# Visualization
# ============================================================================


@app.command()
def visualize(
    slide_path: Optional[Path] = typer.Option(
        None, "--slide", "-s", help="Single slide to visualize"
    ),
    embeddings_dir: Optional[Path] = typer.Option(
        None, "--embeddings", "-e", help="Embeddings directory"
    ),
    clinical_table: Optional[Path] = typer.Option(
        None, "--clinical", "-c", help="Clinical table for labels"
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Generate visualizations (slides, embeddings, summaries)."""
    import pandas as pd
    import numpy as np
    from .io_utils import setup_logging, get_visualizations_dir, ensure_dir
    from . import visualization as viz

    setup_logging("visualize")

    if output_dir is None:
        output_dir = get_visualizations_dir()
    ensure_dir(output_dir)

    console.print("[bold blue]ARGO-DeepMSI: Visualization[/bold blue]")

    # Visualize single slide
    if slide_path:
        console.print(f"Visualizing slide: {slide_path}")
        viz.visualize_slide(
            slide_path=slide_path,
            output_path=output_dir / f"{slide_path.stem}_overview.png",
        )
        console.print(f"[green]Saved to {output_dir}[/green]")

    # Visualize embeddings
    if embeddings_dir:
        console.print(f"Visualizing embeddings: {embeddings_dir}")

        embeddings = np.load(embeddings_dir / "embeddings.npy")
        pd.read_csv(embeddings_dir / "metadata.csv")

        labels = None
        if clinical_table:
            pd.read_csv(clinical_table)
            # Match labels to embeddings
            # (simplified - assumes slide_id matches PATIENT)

        viz.plot_embedding_umap(
            embeddings=embeddings,
            labels=labels,
            output_path=output_dir / "umap_embeddings.png",
        )
        console.print(f"[green]Saved UMAP to {output_dir}[/green]")


# ============================================================================
# Training
# ============================================================================


@app.command()
def train(
    embeddings_dir: Path = typer.Argument(..., help="Directory with embeddings"),
    clinical_table: Path = typer.Argument(..., help="Clinical table with labels"),
    label_column: str = typer.Option("isMSIH", "--label", "-l", help="Label column"),
    n_splits: int = typer.Option(5, "--splits", help="Number of CV folds"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Train classifiers on slide embeddings."""
    import pandas as pd
    import numpy as np
    from .io_utils import setup_logging, get_models_dir, ensure_dir
    from .training import compare_classifiers

    setup_logging("train")

    if output_dir is None:
        output_dir = get_models_dir()
    ensure_dir(output_dir)

    console.print("[bold blue]ARGO-DeepMSI: Training[/bold blue]")

    # Load data
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    metadata = pd.read_csv(embeddings_dir / "metadata.csv")
    clinical = pd.read_csv(clinical_table)

    console.print(f"Loaded {len(embeddings)} embeddings")
    console.print(f"Loaded {len(clinical)} clinical records")

    # Match embeddings to labels
    # (This is simplified - in practice need proper patient-slide matching)
    merged = metadata.merge(clinical, left_on="slide_id", right_on="PATIENT", how="inner")

    if len(merged) == 0:
        console.print("[red]No matching records found between embeddings and clinical data[/red]")
        raise typer.Exit(1)

    X = embeddings[: len(merged)]  # simplified
    y = (merged[label_column] == "MSI-H").astype(int).values

    console.print(f"Training on {len(X)} samples")
    console.print(f"Label distribution: MSI-H={y.sum()}, MSS={len(y) - y.sum()}")

    # Compare classifiers
    results = compare_classifiers(X, y, n_splits=n_splits)

    # Save results
    results.to_csv(output_dir / "classifier_comparison.csv", index=False)

    # Print results table
    table = Table(title="Classifier Comparison")
    table.add_column("Classifier", style="cyan")
    table.add_column("AUROC", justify="right")
    table.add_column("Accuracy", justify="right")

    for _, row in results.iterrows():
        table.add_row(
            row["classifier"],
            f"{row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}",
            f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}",
        )

    console.print(table)


# ============================================================================
# Full pipeline
# ============================================================================


@app.command()
def run(
    slide_table: Path = typer.Argument(..., help="Path to slide table CSV"),
    clinical_table: Path = typer.Argument(..., help="Path to clinical table CSV"),
    models: List[str] = typer.Option(["uni2"], "--model", "-m", help="Models to use"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    max_slides: Optional[int] = typer.Option(None, "--max-slides", help="Max slides"),
):
    """Run the full pipeline: extract → aggregate → train."""
    import pandas as pd
    import numpy as np
    from .io_utils import setup_logging, get_features_dir, get_embeddings_dir, get_models_dir
    from .feature_extraction import extract_features_multi_model, aggregate_features
    from .training import compare_classifiers

    setup_logging("pipeline")

    console.print("[bold blue]ARGO-DeepMSI: Full Pipeline[/bold blue]")
    console.print(f"Models: {', '.join(models)}")

    # Load tables
    slide_df = pd.read_csv(slide_table)
    clinical_df = pd.read_csv(clinical_table)

    for model in models:
        console.print(f"\n[bold cyan]Processing model: {model}[/bold cyan]")

        # 1. Extract features
        console.print("Step 1: Extracting features...")
        extract_features_multi_model(
            slide_table=slide_df,
            models=[model],
            device=device,
            max_slides=max_slides,
        )

        # 2. Aggregate
        console.print("Step 2: Aggregating features...")
        features_dir = get_features_dir(model)
        aggregate_features(
            features_dir=features_dir,
            model=model,
            method="mean",
        )

        # 3. Train
        console.print("Step 3: Training classifiers...")
        embeddings_dir = get_embeddings_dir(model)
        embeddings = np.load(embeddings_dir / "embeddings.npy")
        metadata = pd.read_csv(embeddings_dir / "metadata.csv")

        # Match to labels (simplified)
        merged = metadata.merge(clinical_df, left_on="slide_id", right_on="PATIENT", how="inner")
        if len(merged) > 0:
            X = embeddings[: len(merged)]
            y = (merged["isMSIH"] == "MSI-H").astype(int).values
            results = compare_classifiers(X, y)
            results.to_csv(get_models_dir() / f"{model}_results.csv", index=False)

    console.print("\n[bold green]Pipeline complete![/bold green]")


# ============================================================================
# Version
# ============================================================================


@app.command()
def version():
    """Show version information."""
    console.print("ARGO-DeepMSI v0.2.0")
    console.print("LazySlide-based MSI prediction pipeline")


if __name__ == "__main__":
    app()
