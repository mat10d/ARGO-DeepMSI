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
    from .feature_extraction import list_available_models

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

    # Extract features (all models in one pass per slide)
    from .feature_extraction import extract_features_batch

    results = extract_features_batch(
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
    success = results["success"].sum()
    total = len(results)
    console.print(f"[green]Complete![/green] {success}/{total} slides processed")
    console.print(f"Each slide contains features from: {', '.join(models)}")


@app.command()
def models(
    check: bool = typer.Option(
        False, "--check", help="Instantiate each model (downloads weights) and report pass/fail"
    ),
    non_gated_only: bool = typer.Option(
        False, "--non-gated-only", help="With --check, skip gated models"
    ),
):
    """List available feature extraction models and aggregation methods.

    With ``--check``, also instantiate each model to verify the HF weights
    are accessible. This downloads weights (slow on first run, cached
    after) but does not run inference. Gated models are skipped if
    ``HF_TOKEN`` isn't set.
    """
    from .feature_extraction import PATCH_MODELS, SLIDE_ENCODERS

    if check:
        _models_check(PATCH_MODELS, non_gated_only=non_gated_only)
        return

    console.print("[bold blue]Available Models & Aggregation Methods[/bold blue]\n")

    # Patch models
    table = Table(title="Patch-Level Feature Extractors")
    table.add_column("Model", style="cyan")
    table.add_column("Auth Required", style="yellow")
    table.add_column("Description")

    for name, config in PATCH_MODELS.items():
        auth = "Yes" if config.requires_auth else "No"
        table.add_row(name, auth, config.description)

    console.print(table)
    console.print()

    # Aggregation methods
    table = Table(title="Slide-Level Aggregation Methods")
    table.add_column("Method", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Description")

    # Simple pooling
    for method in ["mean", "max", "median", "sum"]:
        table.add_row(
            method, "Simple Pooling", SLIDE_ENCODERS.get(method, f"{method.capitalize()} pooling")
        )

    # Neural encoders
    for name, desc in SLIDE_ENCODERS.items():
        if name not in ["mean", "max", "median", "sum"]:
            table.add_row(name, "Neural Encoder", desc)

    console.print(table)


# ============================================================================
# Aggregation
# ============================================================================


@app.command()
def aggregate(
    models: str = typer.Argument(..., help="Model(s) to aggregate (comma-separated)"),
    slide_table: Optional[Path] = typer.Option(
        None,
        "--slide-table",
        "-s",
        help="slide_table.csv with PATIENT,FILENAME,SITE",
    ),
    method: str = typer.Option("mean", "--method", "-m", help="Aggregation method"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    device: str = typer.Option("cuda", "--device", "-d"),
):
    """Aggregate patch features to slide-level embeddings.

    Simple pooling (fast):
        argo aggregate plip --method mean
        argo aggregate plip,ctranspath --method max

    Neural slide encoders (slower, more accurate):
        argo aggregate virchow --method prism --device cuda
        argo aggregate conch_v1.5 --method titan --device cuda
    """
    from .io_utils import setup_logging, get_data_dir
    from .feature_extraction import aggregate_features

    setup_logging("aggregate")

    # Default to results/data/slide_table.csv
    if slide_table is None:
        slide_table = get_data_dir().parent / "results" / "data" / "slide_table.csv"

    if not slide_table.exists():
        console.print(f"[red]Slide table not found: {slide_table}[/red]")
        console.print("Run 'argo ingest' first to create slide_table.csv")
        raise typer.Exit(1)

    # Parse models
    model_list = [m.strip() for m in models.split(",")]

    console.print("[bold blue]ARGO-DeepMSI: Feature Aggregation[/bold blue]")
    console.print(f"Slide table: {slide_table}")
    console.print(f"Models: {', '.join(model_list)}")
    console.print(f"Method: {method}")

    # Aggregate
    results = aggregate_features(
        slide_table=slide_table,
        models=model_list,
        method=method,
        output_dir=output_dir,
        device=device,
    )

    # Summary
    for model, df in results.items():
        console.print(f"[green]{model}:[/green] {len(df)} slides aggregated")


def _models_check(patch_models, non_gated_only: bool = False) -> None:
    """Walk PATCH_MODELS and try to instantiate each via LazySlide's registry."""
    import os

    try:
        import lazyslide as zs
    except ImportError:  # pragma: no cover
        console.print("[red]lazyslide not installed[/red]")
        raise typer.Exit(1)

    registry = zs.models.MODEL_REGISTRY
    hf_token_set = bool(os.environ.get("HF_TOKEN"))

    table = Table(title="Model availability check")
    table.add_column("Model", style="cyan")
    table.add_column("Gated", justify="center")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    n_ok = n_skip = n_fail = 0
    for name, cfg in patch_models.items():
        gated = cfg.requires_auth
        if non_gated_only and gated:
            continue
        if name not in registry:
            table.add_row(name, "?" if gated else "-", "[red]missing[/red]", "not in lazyslide registry")
            n_fail += 1
            continue
        if gated and not hf_token_set:
            table.add_row(
                name, "yes", "[yellow]skipped[/yellow]", "HF_TOKEN not set"
            )
            n_skip += 1
            continue
        try:
            registry[name]()
            table.add_row(name, "yes" if gated else "no", "[green]ok[/green]", "")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:120]
            table.add_row(name, "yes" if gated else "no", "[red]fail[/red]", msg)
            n_fail += 1

    console.print(table)
    console.print(f"\n[green]ok={n_ok}[/green]  [yellow]skipped={n_skip}[/yellow]  [red]fail={n_fail}[/red]")


# ============================================================================
# Quality control
# ============================================================================


@app.command()
def qc(
    slide_table: Path = typer.Argument(..., help="slide_table.csv"),
    qc_model: str = typer.Option("grandqc-artifact", "--model", "-m", help="QC model name"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Pass if reduced score <= threshold"),
    reduce: str = typer.Option("mean", "--reduce", "-r", help="Per-tile reduction: mean/max/median"),
    output_csv: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Filtered slide table output path"
    ),
):
    """Filter a slide table by QC scores already extracted into the zarr.

    Run ``argo extract <table> --model grandqc-artifact`` first to populate
    QC features, then this command to produce a filtered table for downstream
    feature extraction.
    """
    from .io_utils import setup_logging
    from .feature_extraction import filter_slides_by_qc

    setup_logging("qc")

    if output_csv is None:
        output_csv = slide_table.parent / f"{slide_table.stem}_qc_filtered.csv"

    console.print("[bold blue]ARGO-DeepMSI: QC Filter[/bold blue]")
    console.print(f"Slide table: {slide_table}")
    console.print(f"QC model: {qc_model} (reduce={reduce}, threshold<= {threshold})")

    df = filter_slides_by_qc(
        slide_table=slide_table,
        qc_model=qc_model,
        threshold=threshold,
        reduce=reduce,  # type: ignore[arg-type]
        output_csv=output_csv,
    )

    kept = int(df["passes_qc"].sum())
    console.print(f"[green]{kept}/{len(df)} slides pass QC[/green]")
    console.print(f"Filtered table: {output_csv}")


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

        labels = None
        if clinical_table:
            from .training import load_training_data

            X, y, merged = load_training_data(
                embeddings_dir=embeddings_dir,
                clinical_table=clinical_table,
            )
            embeddings = X
            labels = merged["isMSIH"].values if "isMSIH" in merged.columns else y
        else:
            embeddings = np.load(embeddings_dir / "embeddings.npy")

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
    clinical_table: Optional[Path] = typer.Option(
        None, "--clinical", "-c", help="Clinical table with labels"
    ),
    label_column: str = typer.Option("isMSIH", "--label", "-l"),
    n_splits: int = typer.Option(5, "--splits"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
):
    """Train classifiers on slide embeddings.

    Example:
        argo train results/embeddings/plip_mean \\
            --clinical results/data/clinical_table.csv
    """
    from .io_utils import setup_logging, get_models_dir, get_data_dir, ensure_dir
    from .training import compare_classifiers, load_training_data

    setup_logging("train")

    # Default to results/data/clinical_table.csv
    if clinical_table is None:
        clinical_table = get_data_dir().parent / "results" / "data" / "clinical_table.csv"

    if not clinical_table.exists():
        console.print(f"[red]Clinical table not found: {clinical_table}[/red]")
        raise typer.Exit(1)

    if output_dir is None:
        output_dir = get_models_dir() / embeddings_dir.name
    ensure_dir(output_dir)

    console.print("[bold blue]ARGO-DeepMSI: Training[/bold blue]")
    console.print(f"Embeddings: {embeddings_dir}")
    console.print(f"Clinical: {clinical_table}")

    # Load data with robust matching
    X, y, merged_df = load_training_data(
        embeddings_dir=embeddings_dir,
        clinical_table=clinical_table,
        label_column=label_column,
    )

    console.print(f"\nTraining on {len(X)} samples")
    console.print(f"Features: {X.shape[1]}D")

    # Train classifiers — group CV by patient_id to prevent leakage
    groups = merged_df["patient_id"].values
    results = compare_classifiers(X, y, groups=groups, n_splits=n_splits)

    # Save
    results.to_csv(output_dir / "classifier_comparison.csv", index=False)
    merged_df.to_csv(output_dir / "training_data.csv", index=False)

    # Display results
    table = Table(title="Classifier Performance")
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

    console.print(f"\n[green]Results saved to:[/green] {output_dir}")


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
    from .io_utils import setup_logging, get_embeddings_dir, get_models_dir
    from .feature_extraction import extract_features_batch, aggregate_features
    from .training import compare_classifiers, load_training_data

    setup_logging("pipeline")

    console.print("[bold blue]ARGO-DeepMSI: Full Pipeline[/bold blue]")
    console.print(f"Models: {', '.join(models)}")

    # Load slide table
    slide_df = pd.read_csv(slide_table)

    for model in models:
        console.print(f"\n[bold cyan]Processing model: {model}[/bold cyan]")

        # 1. Extract features
        console.print("Step 1: Extracting features...")
        extract_features_batch(
            slide_table=slide_df,
            models=[model],
            device=device,
            max_slides=max_slides,
        )

        # 2. Aggregate
        console.print("Step 2: Aggregating features...")
        aggregate_features(
            slide_table=slide_df,
            models=[model],
            method="mean",
        )

        # 3. Train
        console.print("Step 3: Training classifiers...")
        embeddings_dir = get_embeddings_dir(f"{model}_mean")

        try:
            X, y, merged = load_training_data(
                embeddings_dir=embeddings_dir,
                clinical_table=clinical_table,
            )

            console.print(f"Training on {len(X)} samples")
            results = compare_classifiers(X, y, groups=merged["patient_id"].values)

            model_output_dir = get_models_dir() / f"{model}_mean"
            model_output_dir.mkdir(parents=True, exist_ok=True)
            results.to_csv(model_output_dir / "classifier_comparison.csv", index=False)
            merged.to_csv(model_output_dir / "training_data.csv", index=False)

            console.print(f"✓ Best AUROC: {results['auroc_mean'].max():.3f}")
        except Exception as e:
            console.print(f"[red]✗ Training failed: {e}[/red]")

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
