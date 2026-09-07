"""
CLI entry point for ARGO-DeepMSI.

Single command interface for the entire pipeline:
    argo ingest      - Data ingestion from REDCap
    argo pyramidal   - Convert non-pyramidal WSIs to tiled pyramidal TIFFs
    argo extract     - Feature extraction with LazySlide
    argo qc          - Filter slides by QC scores
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

from . import __version__

app = typer.Typer(
    name="argo",
    help="ARGO-DeepMSI: MSI prediction from whole slide images using LazySlide",
    add_completion=False,
)
scorers_app = typer.Typer(help="Inspect and run registered MSI scoring methods.")
app.add_typer(scorers_app, name="scorers")
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
    from .io_utils import get_results_dir, ensure_dir
    from .data_ingestion import process_redcap_data

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
# Slide preprocessing (pyramidal conversion)
# ============================================================================


@app.command()
def pyramidal(
    slide_table: Path = typer.Argument(..., help="Path to slide table CSV"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (default: <slide_table>_pyramidal.csv)",
    ),
    slide_column: str = typer.Option(
        "FILENAME", "--slide-column", help="Column in the CSV that holds the slide path"
    ),
    tile_size: int = typer.Option(256, "--tile-size", help="Pyramid tile size (px)"),
    quality: int = typer.Option(90, "--quality", help="JPEG quality for tiles"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be converted without writing files"
    ),
):
    """Convert non-pyramidal WSIs to tiled pyramidal TIFFs.

    LazySlide's ``find_tissues`` OOMs on slides with ``n_levels == 1`` because
    it loads the full-resolution image. This command scans the slide table,
    converts any non-pyramidal slides via ``vips tiffsave --pyramid --tile``,
    and writes an updated slide table pointing at the converted files. Runs
    serially — submit via ``scripts/pyramidal.sh`` for anything cohort-sized.
    """
    from .slide_prep import convert_non_pyramidal_slides, summarize

    console.print("[bold blue]ARGO-DeepMSI: Pyramidal Conversion[/bold blue]")
    console.print(f"Slide table: {slide_table}")
    if dry_run:
        console.print("[yellow]Dry run — no files will be written.[/yellow]")

    results = convert_non_pyramidal_slides(
        slide_table=slide_table,
        output_table=output,
        slide_column=slide_column,
        tile_size=tile_size,
        quality=quality,
        dry_run=dry_run,
    )

    counts = summarize(results)
    table = Table(title="Conversion summary")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status in ("ok", "converted", "already_converted", "would_convert", "unreadable", "failed"):
        table.add_row(status, str(counts.get(status, 0)))
    console.print(table)

    for r in results:
        if r.status in {"failed", "unreadable"}:
            console.print(f"  [red]{r.status}[/red]: {r.slide} — {r.detail or ''}")

    out_table = output or slide_table.with_name(slide_table.stem + "_pyramidal.csv")
    console.print(f"Updated slide table: [green]{out_table}[/green]")


# ============================================================================
# Feature extraction
# ============================================================================


@app.command()
def extract(
    slide_table: Path = typer.Argument(..., help="Path to slide table CSV"),
    models: List[str] = typer.Option(["uni2"], "--model", "-m", help="Models to use"),
    tile_px: int = typer.Option(256, "--tile-px", help="Tile size in pixels"),
    mpp: float = typer.Option(0.5, "--mpp", help="Microns per pixel"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use automatic mixed precision"),
    num_workers: int = typer.Option(4, "--workers", "-j", min=0),
    batch_size: int = typer.Option(64, "--batch-size", min=1),
    tiling_policy: str = typer.Option(
        "require-current",
        "--tiling-policy",
        help="require-current, or reuse to opt into legacy/unversioned tile grids",
    ),
    max_slides: Optional[int] = typer.Option(None, "--max-slides", help="Max slides (for testing)"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-extract requested features; does not regenerate an existing tile grid",
    ),
):
    """Extract features from slides using LazySlide.

    Example:
        argo extract slide_table.csv --model uni2 --model virchow2
    """
    import pandas as pd
    from .feature_extraction import list_available_models

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
        num_workers=num_workers,
        batch_size=batch_size,
        tiling_policy=tiling_policy,  # type: ignore[arg-type]
    )

    # Summary
    success = results["success"].sum()
    total = len(results)
    console.print(f"[green]Complete![/green] {success}/{total} slides processed")
    console.print(f"Each slide contains features from: {', '.join(models)}")


@app.command("extract-dask")
def extract_dask_command(
    slide_table: Path = typer.Argument(..., help="Path to slide table CSV"),
    models: Optional[List[str]] = typer.Option(None, "--model", "-m"),
    partition: str = typer.Option("nvidia-A6000-20", "--partition"),
    min_workers: int = typer.Option(1, "--min-workers", min=0),
    max_workers: int = typer.Option(3, "--max-workers", min=1),
    walltime: str = typer.Option("24:00:00", "--walltime"),
    cores: int = typer.Option(16, "--cores", min=1),
    memory: str = typer.Option("256 GB", "--memory"),
    tile_px: int = typer.Option(256, "--tile-px"),
    mpp: float = typer.Option(0.5, "--mpp"),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    num_workers: int = typer.Option(2, "--workers", "-j", min=0),
    conda_env: Optional[str] = typer.Option(None, "--conda-env"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    tiling_policy: str = typer.Option("require-current", "--tiling-policy"),
):
    """Extract features on an elastic SLURM GPU cluster using the canonical extractor."""
    from .dask_extraction import DEFAULT_MODELS, run_dask_extraction

    report = run_dask_extraction(
        slide_table=slide_table,
        models=models or DEFAULT_MODELS,
        partition=partition,
        min_workers=min_workers,
        max_workers=max_workers,
        walltime=walltime,
        cores=cores,
        memory=memory,
        tile_px=tile_px,
        mpp=mpp,
        batch_size=batch_size,
        num_workers=num_workers,
        conda_env=conda_env,
        output_dir=output_dir,
        overwrite=overwrite,
        tiling_policy=tiling_policy,
    )
    counts = report["counts"]
    console.print(
        f"[green]success={counts['success']}[/green] "
        f"[yellow]skipped={counts['skipped']}[/yellow] "
        f"[red]failed={counts['failed']}[/red]"
    )


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
    from .io_utils import get_results_dir
    from .feature_extraction import aggregate_features

    # Default to results/data/slide_table.csv
    if slide_table is None:
        slide_table = get_results_dir() / "data" / "slide_table.csv"

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
        from .models._lazyslide import MODEL_REGISTRY
    except ImportError:  # pragma: no cover
        console.print("[red]lazyslide not installed[/red]")
        raise typer.Exit(1)

    registry = MODEL_REGISTRY
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
            table.add_row(
                name, "?" if gated else "-", "[red]missing[/red]", "not in lazyslide registry"
            )
            n_fail += 1
            continue
        if gated and not hf_token_set:
            table.add_row(name, "yes", "[yellow]skipped[/yellow]", "HF_TOKEN not set")
            n_skip += 1
            continue
        try:
            registry[name]()
            table.add_row(name, "yes" if gated else "no", "[green]ok[/green]", "")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            # Some errors (notably GatedRepoError) start with a blank line;
            # scan for the first non-empty line to avoid truncating to "".
            lines = [line for line in str(e).splitlines() if line.strip()]
            msg = (lines[0] if lines else type(e).__name__)[:120]
            table.add_row(name, "yes" if gated else "no", "[red]fail[/red]", msg)
            n_fail += 1

    console.print(table)
    console.print(
        f"\n[green]ok={n_ok}[/green]  [yellow]skipped={n_skip}[/yellow]  [red]fail={n_fail}[/red]"
    )


# ============================================================================
# Quality control
# ============================================================================


@app.command()
def qc(
    slide_table: Path = typer.Argument(..., help="slide_table.csv"),
    qc_model: str = typer.Option("grandqc-artifact", "--model", "-m", help="QC model name"),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t", help="Pass if reduced score <= threshold"
    ),
    reduce: str = typer.Option(
        "mean", "--reduce", "-r", help="Per-tile reduction: mean/max/median"
    ),
    output_csv: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Filtered slide table output path"
    ),
):
    """Filter a slide table by QC scores already extracted into the zarr.

    Run ``argo extract <table> --model grandqc-artifact`` first to populate
    QC features, then this command to produce a filtered table for downstream
    feature extraction.
    """
    from .feature_extraction import filter_slides_by_qc

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
    import numpy as np
    from .io_utils import get_visualizations_dir, ensure_dir
    from . import visualization as viz

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
    seed: int = typer.Option(42, "--seed"),
    classifiers: Optional[List[str]] = typer.Option(
        None,
        "--classifier",
        help="Repeat to select logistic, random_forest, and/or svm",
    ),
    classifier_params: Optional[List[str]] = typer.Option(
        None,
        "--classifier-param",
        help="Repeat CLASSIFIER.KEY=JSON, for example logistic.C=0.1",
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
):
    """Train classifiers on slide embeddings.

    Example:
        argo train results/embeddings/plip_mean \\
            --clinical results/data/clinical_table.csv
    """
    from .io_utils import ensure_dir, get_models_dir, get_results_dir
    from .reproducibility import environment_snapshot, write_json
    from .scorer_runner import parse_parameters
    from .training import compare_classifiers, load_training_data

    # Default to results/data/clinical_table.csv
    if clinical_table is None:
        clinical_table = get_results_dir() / "data" / "clinical_table.csv"

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
    selected = classifiers or ["logistic", "random_forest", "svm"]
    nested_params: dict[str, dict] = {}
    for key, value in parse_parameters(classifier_params or []).items():
        if "." not in key:
            raise typer.BadParameter(
                f"Classifier parameter {key!r} must have the form CLASSIFIER.KEY=VALUE"
            )
        classifier, parameter = key.split(".", 1)
        nested_params.setdefault(classifier, {})[parameter] = value
    results = compare_classifiers(
        X,
        y,
        groups=groups,
        n_splits=n_splits,
        random_state=seed,
        classifiers=selected,
        classifier_params=nested_params,
    )

    # Save
    results.to_csv(output_dir / "classifier_comparison.csv", index=False)
    merged_df.to_csv(output_dir / "training_data.csv", index=False)
    write_json(
        output_dir / "run.json",
        {
            "embeddings": str(embeddings_dir.resolve()),
            "clinical_table": str(clinical_table.resolve()),
            "label_column": label_column,
            "n_splits": n_splits,
            "seed": seed,
            "classifiers": selected,
            "parameters": nested_params,
            "environment": environment_snapshot(Path.cwd()),
        },
    )

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
# Registered scorers and reproducible experiment runner
# ============================================================================


@scorers_app.command("list")
def list_registered_scorers():
    """List scorer names without importing every model stack."""
    from .scorers import list_scorers

    for name in list_scorers():
        console.print(name)


@scorers_app.command("show")
def show_scorer(name: str = typer.Argument(..., help="Registered scorer name")):
    """Show a scorer's contract and configurable compute parameters."""
    from .scorer_runner import scorer_contract

    try:
        contract = scorer_contract(name)
    except KeyError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    console.print(f"[bold]{contract['name']}[/bold] — {contract['description']}")
    table = Table(title="Scorer contract")
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    for key in (
        "resolution",
        "needs_training_on_our_data",
        "primary_score",
        "patient_aggregation",
        "cache",
    ):
        table.add_row(key, str(contract[key]))
    for key, default in contract["parameters"].items():
        table.add_row(f"parameter.{key}", repr(default))
    console.print(table)


@scorers_app.command("run")
def run_registered_scorer(
    name: str = typer.Argument(..., help="Registered scorer name"),
    cohort: Optional[Path] = typer.Option(None, "--cohort"),
    slide_table: Optional[Path] = typer.Option(None, "--slide-table"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    parameters: Optional[List[str]] = typer.Option(
        None,
        "--param",
        "-p",
        help='Repeat KEY=JSON, for example embeddings=["phaet_mean","mascaret_mean"]',
    ),
    cache: bool = typer.Option(False, "--cache/--recompute"),
    publish: bool = typer.Option(False, "--publish", help="Replace the canonical scorer cache"),
):
    """Run any scorer with explicit parameters and capture its provenance."""
    from datetime import datetime, timezone

    from .io_utils import get_results_dir
    from .scorer_runner import parse_parameters, run_scorer

    results = get_results_dir()
    cohort = cohort or results / "data" / "cohort_clean.csv"
    slide_table = slide_table or results / "data" / "slide_table_pyramidal.csv"
    if output_dir is None:
        run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = results / "runs" / run_name / "scorers" / name
    try:
        parsed = parse_parameters(parameters or [])
        outcome = run_scorer(
            name,
            cohort_csv=cohort,
            slide_table_csv=slide_table,
            output_dir=output_dir,
            parameters=parsed,
            use_cache=cache,
            publish=publish,
        )
    except (KeyError, ValueError, OSError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    console.print(f"[green]Scored {outcome['n_rows']} rows[/green] → {outcome['scores']}")


@app.command("experiment")
def experiment_command(
    config: Path = typer.Argument(..., help="Experiment TOML file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the stage plan"),
    resume: bool = typer.Option(True, "--resume/--fresh"),
):
    """Run or resume a configuration-driven extraction/training/scorer experiment."""
    from .experiment import run_experiment

    def report(stage: str, status: str) -> None:
        color = {"completed": "green", "failed": "red", "skipped": "yellow"}.get(status, "cyan")
        console.print(f"[{color}]{status:9s}[/{color}] {stage}")

    try:
        result = run_experiment(config, dry_run=dry_run, resume=resume, on_stage=report)
    except (KeyError, ValueError, OSError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    if dry_run:
        console.print(f"Workspace: {result['workspace']}")
        console.print(f"Run directory: {result['run_dir']}")
        for stage in result["plan"]:
            console.print(f"  {stage['id']}")
    else:
        console.print(f"[bold green]Experiment complete[/bold green]: {result['name']}")


@app.command("experiment-schema")
def experiment_schema_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON Schema here"),
):
    """Print the experiment JSON Schema used by agents and editors."""
    import json

    from .experiment_schema import experiment_json_schema
    from .reproducibility import write_json

    schema = experiment_json_schema()
    if output is not None:
        write_json(output, schema)
        console.print(f"[green]Wrote experiment schema[/green] → {output}")
    else:
        typer.echo(json.dumps(schema, indent=2))


@app.command()
def strategies():
    """List the distinct pathology-ML training strategy families."""
    from .experiment_schema import STRATEGIES

    table = Table(title="Pathology ML strategy families")
    table.add_column("Strategy", style="cyan")
    table.add_column("Meaning")
    for name, description in STRATEGIES.items():
        table.add_row(name, description)
    console.print(table)


@app.command("self-test")
def self_test(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Keep the synthetic acceptance workspace at this empty path",
    ),
):
    """Run a small, offline acceptance experiment across every stage."""
    import tempfile

    from .synthetic import run_synthetic_acceptance

    def report(stage: str, status: str) -> None:
        console.print(f"{status:9s} {stage}")

    if output is not None:
        result = run_synthetic_acceptance(output, on_stage=report)
        console.print(
            f"[bold green]Synthetic acceptance passed[/bold green]: "
            f"{len(result['stages'])} stages → {result['acceptance_workspace']}"
        )
        return
    with tempfile.TemporaryDirectory(prefix="argo-acceptance-") as directory:
        result = run_synthetic_acceptance(Path(directory), on_stage=report)
        console.print(
            f"[bold green]Synthetic acceptance passed[/bold green]: {len(result['stages'])} stages"
        )


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
    from .io_utils import get_embeddings_dir, get_models_dir
    from .feature_extraction import extract_features_batch, aggregate_features
    from .training import compare_classifiers, load_training_data

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
    console.print(f"ARGO-DeepMSI v{__version__}")
    console.print("LazySlide-based MSI prediction pipeline")


@app.command()
def env():
    """Show effective HuggingFace cache / environment info."""
    import os
    import shutil as _shutil

    def _disk(path: str) -> str:
        try:
            total, used, free = _shutil.disk_usage(path)
            return f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB"
        except FileNotFoundError:
            return "(path does not exist)"

    table = Table(title="HuggingFace cache environment")
    table.add_column("Var", style="cyan")
    table.add_column("Value")
    table.add_column("Disk")

    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_TOKEN"):
        val = os.environ.get(var, "")
        if var == "HF_TOKEN":
            shown = "(set)" if val else "(unset)"
            table.add_row(var, shown, "")
        else:
            disk = _disk(val) if val else ""
            table.add_row(var, val or "(unset — HF defaults to ~/.cache/huggingface)", disk)

    console.print(table)


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Experiment TOML to validate and preflight"
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace override (defaults to config or cwd)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a machine-readable report instead of a table"
    ),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as a failed preflight"),
):
    """Check environment, inputs, models, GPU, tiling provenance, and budgets."""
    import json

    from .doctor import run_doctor

    report = run_doctor(workspace=workspace, config_path=config)
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        table = Table(title=f"ARGO doctor: {report['status']}")
        table.add_column("Status")
        table.add_column("Check", style="cyan")
        table.add_column("Result")
        colors = {"ok": "green", "warning": "yellow", "error": "red"}
        for check in report["checks"]:
            color = colors[check["status"]]
            table.add_row(f"[{color}]{check['status']}[/{color}]", check["name"], check["message"])
        console.print(table)
    failed = report["status"] == "error" or (strict and report["status"] == "warning")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
