"""
CLI entry point for ARGO-DeepMSI.

Single command interface for the entire pipeline:
    argo setup       - Create/sync the core, LazySlide, and Mussel environments
    argo envs        - Show environment status
    argo ingest      - Data ingestion from REDCap
    argo pyramidal   - Convert non-pyramidal WSIs to tiled pyramidal TIFFs
    argo extract     - Feature extraction with LazySlide or Mussel
    argo qc          - Filter slides by QC scores
    argo aggregate   - Aggregate patch features to slide embeddings
    argo visualize   - Generate visualizations
    argo train       - Train classifiers on embeddings
    argo run         - Run full pipeline

Commands that need LazySlide re-execute themselves in ``envs/lazyslide`` when the
current environment lacks it (see ``argo_deepmsi.envs.ensure_env``).
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


def _require_env(name: str) -> None:
    """Run the current command in environment ``name`` or exit with guidance."""
    from .envs import EnvDispatchError, ensure_env

    try:
        ensure_env(name)
    except EnvDispatchError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


# ============================================================================
# Environments and setup
# ============================================================================


def _flag(value: Optional[bool]) -> str:
    return {True: "[green]yes[/green]", False: "[red]no[/red]", None: "-"}[value]


@app.command("envs")
def envs_command(
    probe: bool = typer.Option(
        True, "--probe/--no-probe", help="Import each env's probe module in its interpreter"
    ),
    lock_check: bool = typer.Option(
        False, "--lock-check", help="Also run `uv lock --check` per uv env"
    ),
):
    """Show the status of the core, LazySlide, Mussel, and PALADIN environments."""
    from .envs import ENVS, env_status

    table = Table(title="ARGO environments")
    table.add_column("Env", style="cyan")
    table.add_column("Kind")
    table.add_column("Project")
    table.add_column("Lock")
    table.add_column("Synced")
    if lock_check:
        table.add_column("Lock check")
    table.add_column("Probe")
    table.add_column("Detail", overflow="fold")
    for name in ENVS:
        status = env_status(name, probe=probe, lock_check=lock_check)
        row = [
            name,
            status["kind"],
            _flag(status["exists"]),
            _flag(status["lock"]),
            _flag(status["venv"]),
        ]
        if lock_check:
            row.append(_flag(status["lock_check"]))
        row += [_flag(status["probe"]), status["detail"] or status["description"]]
        table.add_row(*row)
    console.print(table)


@app.command()
def setup(
    envs: Optional[List[str]] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment to sync (repeatable; default core, lazyslide, mussel)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without running"),
    hf_check: bool = typer.Option(
        True, "--hf-check/--no-hf-check", help="Report whether a Hugging Face token is present"
    ),
):
    """Create or sync every environment and check cache, credentials, and data paths.

    PALADIN is opt-in (``--env paladin``): its setup script needs access to the
    PALADIN repository and its weights are not public. Token and credential values
    are never printed, only whether they are present.
    """
    import os

    from .envs import DEFAULT_SETUP_ENVS, ENVS, REPO_ROOT, EnvSetupError, dotenv_keys
    from .envs import hf_token_source, setup_env
    from .io_utils import get_data_dir, get_results_dir, setup_huggingface_cache

    selected = list(dict.fromkeys(envs or DEFAULT_SETUP_ENVS))
    unknown = [name for name in selected if name not in ENVS]
    if unknown:
        raise typer.BadParameter(f"Unknown env(s) {unknown}; choose from {sorted(ENVS)}")

    console.print("[bold blue]ARGO-DeepMSI: Setup[/bold blue]")
    failures: list[str] = []
    for name in selected:
        try:
            commands = setup_env(name, dry_run=dry_run)
        except EnvSetupError as error:
            console.print(f"[red]fail[/red]  {error}")
            failures.append(name)
            continue
        verb = "would run" if dry_run else "[green]ok[/green]  "
        for cmd in commands:
            console.print(f"{verb} {name}: {' '.join(cmd)}")

    table = Table(title="Setup checks")
    table.add_column("Check", style="cyan")
    table.add_column("Result", overflow="fold")
    hf_home = os.environ.get("HF_HOME") or str(REPO_ROOT / ".huggingface_cache")
    if not dry_run:
        hf_home = str(setup_huggingface_cache())
    table.add_row("HF_HOME", f"{hf_home} ({'exists' if Path(hf_home).is_dir() else 'missing'})")
    if hf_check:
        source = hf_token_source()
        table.add_row(
            "HF token",
            f"[green]present[/green] via {source}"
            if source
            else "[yellow]missing[/yellow] (gated models need `hf auth login` or $HF_TOKEN)",
        )
    dotenv = REPO_ROOT / ".env"
    keys = dotenv_keys(dotenv) | {key for key in os.environ if os.environ[key]}
    for key in ("REDCAP_API_URL", "REDCAP_API_TOKEN"):
        table.add_row(
            key, "[green]set[/green]" if key in keys else f"[yellow]missing[/yellow] ({dotenv})"
        )
    results = get_results_dir()
    for label, path in (
        ("data dir", get_data_dir()),
        ("results dir", results),
        ("slide table", results / "data" / "slide_table.csv"),
        ("pyramidal slide table", results / "data" / "slide_table_pyramidal.csv"),
        ("clinical table", results / "data" / "clinical_table.csv"),
        ("cohort", results / "data" / "cohort_clean.csv"),
    ):
        state = "[green]exists[/green]" if path.exists() else "[yellow]missing[/yellow]"
        table.add_row(label, f"{state} {path}")
    console.print(table)
    if failures:
        console.print(f"[red]Environment setup failed: {', '.join(failures)}[/red]")
        raise typer.Exit(1)


# ============================================================================
# Data ingestion
# ============================================================================


@app.command()
def ingest(
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    metadata_dir: Optional[Path] = typer.Option(
        None,
        "--metadata-dir",
        help="Directory containing PathPresenter CSV/Excel exports and downloaded slides",
    ),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="REDCap API URL"),
    api_token: Optional[str] = typer.Option(None, "--api-token", help="REDCap API token"),
):
    """Ingest data from REDCap and PathPresenter spreadsheet exports."""
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
        metadata_dir=metadata_dir,
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


def _parse_indices(spec: Optional[str]) -> Optional[List[int]]:
    """Parse ``"0-7"``, ``"1,3,5"``, or ``"0-3,8"`` into sorted unique row indices."""
    if spec is None or not spec.strip():
        return None
    indices: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                low, high = (int(value) for value in part.split("-", 1))
                if high < low:
                    raise ValueError(part)
                indices.update(range(low, high + 1))
            else:
                indices.add(int(part))
    except ValueError as error:
        raise typer.BadParameter(
            f"--indices must look like '0-7' or '1,3,5', got {spec!r}"
        ) from error
    if any(index < 0 for index in indices):
        raise typer.BadParameter("--indices must be non-negative")
    return sorted(indices)


_LAZYSLIDE_DEFAULTS = {
    "tile_px": 256,
    "mpp": 0.5,
    "amp": True,
    "num_workers": 4,
    "batch_size": 64,
    "slide_encoder": None,
}


def _lazyslide_run_config(
    config: Optional[Path], models: Optional[List[str]], overrides: dict
) -> tuple[List[str], dict, object]:
    """Resolve LazySlide extraction params: explicit CLI flag > config ``[params]`` > default.

    Returns:
        ``(models, params, backend_config)`` where ``backend_config`` records the
        effective params for provenance and keeps the source TOML path.
    """
    from .backends.config import BackendConfig, load_backend_config

    cfg_params: dict = {}
    cfg = None
    if config is not None:
        if not Path(config).is_file():
            raise typer.BadParameter(f"Backend config not found: {config}")
        cfg = load_backend_config(config)
        if cfg.backend != "lazyslide":
            raise typer.BadParameter(f"{config} is a {cfg.backend} config, not lazyslide")
        cfg_params = dict(cfg.params)
    model_key = cfg_params.pop("model_key", None)
    if models and model_key and list(models) != [model_key]:
        raise typer.BadParameter(
            f"--model {models} conflicts with model_key={model_key!r} in {config}"
        )
    models = list(models) if models else ([model_key] if model_key else ["uni2"])
    params = {
        key: overrides[key] if overrides.get(key) is not None else cfg_params.get(key, default)
        for key, default in _LAZYSLIDE_DEFAULTS.items()
    }
    effective = {"model_key": models[0] if len(models) == 1 else list(models), **params}
    backend_config = BackendConfig(
        backend="lazyslide",
        model=cfg.model if cfg is not None else models[0],
        params={key: value for key, value in effective.items() if value is not None},
        source=cfg.source if cfg is not None else None,
    )
    return models, params, backend_config


@app.command()
def extract(
    slide_table: Path = typer.Argument(..., help="Path to slide table CSV"),
    models: Optional[List[str]] = typer.Option(
        None, "--model", "-m", help="Models to use (default: config model_key, else uni2)"
    ),
    backend: str = typer.Option(
        "lazyslide", "--backend", "-b", help="Extraction backend: lazyslide or mussel"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help=(
            "Backend TOML (configs/backends/<backend>-<model>.toml). Mussel defaults to "
            "mussel-<model>.toml; for LazySlide its [params] fill any flag not given"
        ),
    ),
    indices: Optional[str] = typer.Option(
        None, "--indices", help="Slide-table rows to process, e.g. 0-7 or 1,3,5 (SLURM arrays)"
    ),
    out_root: Optional[Path] = typer.Option(
        None,
        "--out-root",
        help=(
            "Write outputs under this root (slide directory mirrored) instead of next to "
            "each slide; use one root per LazySlide tiling (tile_px/mpp)"
        ),
    ),
    tile_px: Optional[int] = typer.Option(None, "--tile-px", help="Tile size in pixels [256]"),
    mpp: Optional[float] = typer.Option(None, "--mpp", help="Microns per pixel [0.5]"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    amp: Optional[bool] = typer.Option(
        None, "--amp/--no-amp", help="Use automatic mixed precision [amp]"
    ),
    num_workers: Optional[int] = typer.Option(None, "--workers", "-j", min=0, help="[4]"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", min=1, help="[64]"),
    slide_encoder: Optional[str] = typer.Option(
        None, "--slide-encoder", help="LazySlide slide encoder applied after extraction (titan)"
    ),
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
    """Extract features from slides with LazySlide (default) or Mussel.

    Example:
        argo extract slide_table.csv --model uni2 --model virchow2
        argo extract slide_table.csv --config configs/backends/lazyslide-hoptimus0.toml \
            --out-root results/analysis/backends/stores/lazyslide_256 --indices 0-7
        argo extract slide_table.csv --backend mussel --model hoptimus0 --indices 0-7
    """
    row_indices = _parse_indices(indices)
    if backend == "mussel":
        _extract_mussel(slide_table, models, config, row_indices, out_root, max_slides, overwrite)
        return
    if backend != "lazyslide":
        raise typer.BadParameter("--backend must be 'lazyslide' or 'mussel'")
    models, params, backend_config = _lazyslide_run_config(
        config,
        models,
        {
            "tile_px": tile_px,
            "mpp": mpp,
            "amp": amp,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "slide_encoder": slide_encoder,
        },
    )
    _require_env("lazyslide")

    import pandas as pd
    from .feature_extraction import list_available_models

    console.print("[bold blue]ARGO-DeepMSI: Feature Extraction[/bold blue]")
    console.print(f"Slide table: {slide_table}")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Device: {device}")
    console.print(f"Params: {params}")
    if config is not None:
        console.print(f"Config: {config}")
    if out_root is not None:
        console.print(f"Out root: {out_root}")

    # Validate models
    available = list_available_models()
    for model in models:
        if model not in available:
            console.print(f"[red]Unknown model: {model}[/red]")
            console.print(f"Available: {', '.join(available)}")
            raise typer.Exit(1)

    # Load slide table
    df = pd.read_csv(slide_table)
    if row_indices is not None:
        if row_indices[-1] >= len(df):
            raise typer.BadParameter(f"--indices out of range for {len(df)} rows")
        df = df.iloc[row_indices]
    console.print(f"Loaded {len(df)} slides")

    # Extract features (all models in one pass per slide)
    from .feature_extraction import extract_features_batch

    results = extract_features_batch(
        slide_table=df,
        models=models,
        tile_px=params["tile_px"],
        mpp=params["mpp"],
        amp=params["amp"],
        device=device,
        overwrite=overwrite,
        max_slides=max_slides,
        num_workers=params["num_workers"],
        batch_size=params["batch_size"],
        tiling_policy=tiling_policy,  # type: ignore[arg-type]
        out_root=out_root,
        slide_encoder=params["slide_encoder"],
        backend_config=backend_config,
    )

    # Summary
    success = results["success"].sum()
    total = len(results)
    console.print(f"[green]Complete![/green] {success}/{total} slides processed")
    console.print(f"Each slide contains features from: {', '.join(models)}")
    if total and success == 0:
        raise typer.Exit(1)


def _extract_mussel(
    slide_table: Path,
    models: Optional[List[str]],
    config: Optional[Path],
    row_indices: Optional[List[int]],
    out_root: Optional[Path],
    max_slides: Optional[int],
    overwrite: bool,
) -> None:
    """Run Mussel per slide in ``envs/mussel`` as a subprocess (no LazySlide needed)."""
    from .backends.config import default_config_path, load_backend_config
    from .backends.mussel import run_table
    from .envs import env_command

    if models and len(models) != 1:
        raise typer.BadParameter("--backend mussel takes exactly one --model")
    if not models and config is None:
        raise typer.BadParameter("--backend mussel needs --model or --config")
    config_path = config or default_config_path("mussel", models[0])
    if not Path(config_path).is_file():
        console.print(f"[red]Backend config not found: {config_path}[/red]")
        raise typer.Exit(1)
    cfg = load_backend_config(config_path)

    if max_slides is not None:
        if row_indices is None:
            import pandas as pd

            row_indices = list(range(len(pd.read_csv(slide_table))))
        row_indices = row_indices[:max_slides]

    console.print("[bold blue]ARGO-DeepMSI: Mussel Extraction[/bold blue]")
    console.print(f"Slide table: {slide_table}")
    console.print(f"Config: {config_path}")

    try:
        results = run_table(
            cfg,
            slide_table,
            command_prefix=env_command("mussel", []),
            indices=row_indices,
            out_root=out_root,
            force=overwrite,
        )
    except (IndexError, ValueError, OSError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    counts = {str(k): int(v) for k, v in results["status"].value_counts().items()}
    table = Table(title="Mussel extraction summary")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status, count in sorted(counts.items()):
        table.add_row(status, str(count))
    console.print(table)
    if counts and counts.get("failed", 0) == len(results):
        console.print("[red]Every slide failed[/red]")
        raise typer.Exit(1)


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
    _require_env("lazyslide")
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


# ============================================================================
# Backend comparison and PALADIN
# ============================================================================


def _select_slides(df, n: int, select: str, seed: int):
    """Pick ``n`` rows deterministically, round-robin across ``SITE`` when requested."""
    if select == "first":
        return df.head(n)
    if select == "random":
        return df.sample(n=min(n, len(df)), random_state=seed)
    if select != "sites":
        raise typer.BadParameter("--select must be sites, random, or first")
    if "SITE" not in df.columns:
        raise typer.BadParameter("--select sites needs a SITE column in the slide table")
    groups = [
        group.sample(frac=1.0, random_state=seed)
        for _, group in df.groupby(df["SITE"].astype(str), sort=True)
    ]
    chosen = []
    depth = 0
    while len(chosen) < n and any(depth < len(group) for group in groups):
        for group in groups:
            if depth < len(group) and len(chosen) < n:
                chosen.append(group.index[depth])
        depth += 1
    return df.loc[chosen]


def _parse_side(spec: str, default_roots: dict) -> tuple[str, Optional[Path]]:
    """Parse ``backend`` or ``backend=out_root`` into ``(backend, out_root)``."""
    backend, _, root = spec.partition("=")
    backend = backend.strip()
    if backend not in ("lazyslide", "mussel"):
        raise typer.BadParameter(f"--a/--b backend must be lazyslide or mussel, got {spec!r}")
    return backend, Path(root) if root else default_roots.get(backend)


def _side_label(backend: str, root: Optional[Path]) -> str:
    return f"{backend}:{root.name}" if root else backend


@app.command("compare-backends")
def compare_backends(
    slides: Path = typer.Option(..., "--slides", help="Slide table CSV with FILENAME and SITE"),
    n: int = typer.Option(8, "--n", min=1, help="Number of slides to compare"),
    model: str = typer.Option("hoptimus0", "--model", "-m", help="Backend model name"),
    lazyslide_model_key: Optional[str] = typer.Option(
        None,
        "--lazyslide-model-key",
        help="LazySlide table key (default: the readers' mapping, e.g. h-optimus-0)",
    ),
    out_root: Optional[Path] = typer.Option(
        None, "--out-root", help="Mussel output root used at extraction (default: next to slide)"
    ),
    lazyslide_out_root: Optional[Path] = typer.Option(
        None,
        "--lazyslide-out-root",
        help="LazySlide store root used at extraction (default: next to slide)",
    ),
    side_a: str = typer.Option(
        "lazyslide", "--a", help="First side: backend or backend=out_root (e.g. lazyslide=stores/x)"
    ),
    side_b: str = typer.Option("mussel", "--b", help="Second side, same format as --a"),
    anchor: str = typer.Option(
        "topleft", "--anchor", help="Match tiles on level-0 topleft or center"
    ),
    slide_embedding: bool = typer.Option(
        False,
        "--slide-embedding/--no-slide-embedding",
        help="Also compare slide-encoder outputs (e.g. TITAN) per slide",
    ),
    stem: Optional[str] = typer.Option(None, "--stem", help="Report file stem [compare_<model>]"),
    out: Path = typer.Option(
        Path("results/analysis/backends"), "--out", "-o", help="Report directory"
    ),
    select: str = typer.Option(
        "sites", "--select", help="sites (spread across SITE), random, first"
    ),
    seed: int = typer.Option(0, "--seed"),
    slide_column: str = typer.Option("FILENAME", "--slide-column"),
):
    """Compare existing features from two backends (or two LazySlide runs) on the same slides.

    Runs in core and never extracts: missing outputs are listed together with the
    commands that would produce them.

    Example:
        argo compare-backends --slides results/data/slide_table_pyramidal.csv \
            --a lazyslide=results/analysis/backends/stores/lazyslide_224 \
            --b mussel=results/analysis/backends/stores/mussel --stem compare_hoptimus0_224
    """
    import pandas as pd

    from ._environment import project_root
    from .backends.compare import compare_slides, verdict, write_report
    from .backends.readers import LAZYSLIDE_MODEL_KEYS, read_tiles

    if anchor not in ("topleft", "center"):
        raise typer.BadParameter("--anchor must be topleft or center")
    roots = {"lazyslide": lazyslide_out_root, "mussel": out_root}
    sides = [_parse_side(side_a, roots), _parse_side(side_b, roots)]
    labels = [_side_label(*side) for side in sides]
    if labels[0] == labels[1]:
        raise typer.BadParameter("--a and --b point at the same outputs")

    df = pd.read_csv(slides)
    chosen = _select_slides(df, n, select, seed)
    lazyslide_key = lazyslide_model_key or LAZYSLIDE_MODEL_KEYS.get(model, model)
    console.print("[bold blue]ARGO-DeepMSI: Backend Comparison[/bold blue]")
    console.print(
        f"{len(chosen)} slides, model {model} (LazySlide key {lazyslide_key}): "
        f"{labels[0]} vs {labels[1]}"
    )

    def load(slide_path: Path, backend: str, root: Optional[Path]):
        if backend == "lazyslide":
            return read_tiles(slide_path, backend, model, out_root=root, model_key=lazyslide_key)
        return read_tiles(slide_path, backend, model, out_root=root)

    pairs = []
    embeddings = []
    missing: list[tuple[str, str, str]] = []
    missing_rows: dict[int, list[int]] = {0: [], 1: []}
    for index, row in chosen.iterrows():
        slide_path = Path(row[slide_column])
        loaded = {}
        for side, (backend, root) in enumerate(sides):
            try:
                tiles = load(slide_path, backend, root)
                if slide_embedding:
                    tiles = (
                        tiles,
                        _slide_embedding(slide_path, backend, root, model, lazyslide_key),
                    )
                loaded[side] = tiles
            except (FileNotFoundError, KeyError) as error:
                missing.append((slide_path.name, labels[side], str(error)))
                missing_rows[side].append(int(df.index.get_loc(index)))
        if len(loaded) == 2:
            if slide_embedding:
                (tiles_a, emb_a), (tiles_b, emb_b) = loaded[0], loaded[1]
                embeddings.append((slide_path.stem, emb_a, emb_b))
                pairs.append((slide_path.stem, tiles_a, tiles_b))
            else:
                pairs.append((slide_path.stem, loaded[0], loaded[1]))

    if missing:
        table = Table(title="Missing backend outputs")
        table.add_column("Slide", style="cyan")
        table.add_column("Backend")
        table.add_column("Reason", overflow="fold")
        for slide, label, reason in missing:
            table.add_row(slide, label, reason)
        console.print(table)
        console.print("Produce them with (GPU; submit through SLURM):")
        for side, (backend, root) in enumerate(sides):
            if not missing_rows[side]:
                continue
            rows = ",".join(map(str, sorted(set(missing_rows[side]))))
            flag = f" --out-root {root}" if root else ""
            if backend == "lazyslide":
                config = Path("configs") / "backends" / f"lazyslide-{model}.toml"
                source = (
                    f"--config {config}"
                    if (project_root() / config).is_file()
                    else f"--model {lazyslide_key} --no-amp"
                )
                console.print(
                    f"  argo extract {slides} --backend lazyslide {source}{flag} --indices {rows}"
                )
            else:
                console.print(
                    f"  argo extract {slides} --backend mussel --model {model}{flag} "
                    f"--indices {rows}"
                )
        raise typer.Exit(1)

    kwargs = {"anchor": anchor} if anchor != "topleft" else {}
    report = compare_slides(pairs, **kwargs)
    stem = stem or f"compare_{model}"
    notes = (
        f"{labels[0]} vs {labels[1]}; LazySlide key `{lazyslide_key}`; tiles matched on "
        f"level-0 {anchor}; slides selected by `{select}` (seed {seed}) from `{slides}`."
    )
    csv_path, md_path = write_report(
        report, out, f"{labels[0]} vs {labels[1]}: {model}", stem=stem, notes=notes
    )
    console.print(f"[bold]Verdict:[/bold] {verdict(report)}")
    console.print(f"Report: {csv_path} , {md_path}")
    if slide_embedding:
        from .backends.compare import compare_slide_embeddings

        emb = pd.DataFrame(
            [
                {"slide_id": slide_id, **compare_slide_embeddings(a, b)}
                for slide_id, a, b in embeddings
            ]
        )
        emb_path = Path(out) / f"{stem}_slide.csv"
        emb.to_csv(emb_path, index=False)
        if "cosine" in emb:
            console.print(
                f"Slide embeddings: cosine median {emb['cosine'].median():.6f}, "
                f"min {emb['cosine'].min():.6f}"
            )
        console.print(f"Slide-embedding report: {emb_path}")


def _slide_embedding(
    slide_path: Path, backend: str, root: Optional[Path], model: str, lazyslide_key: str
):
    """Read one slide-encoder output (LazySlide ``agg_slide`` or Mussel slide h5)."""
    from .backends import readers
    from .backends.mussel import output_paths

    if backend == "lazyslide":
        store = readers.lazyslide_store(slide_path, root)
        return readers.read_lazyslide_slide_embedding(store, lazyslide_key)
    outputs = output_paths(slide_path, model, root)
    if not outputs.h5.is_file():
        raise FileNotFoundError(f"No slide embedding {outputs.h5}")
    return readers.read_mussel_slide_embedding(outputs.h5, outputs.pt)


@app.command()
def paladin(
    features: Optional[Path] = typer.Argument(
        None, help="Mussel H-optimus-0 feature directory or slide table"
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", envvar="PALADIN_CHECKPOINT", help="PALADIN/Aeon checkpoint"
    ),
):
    """PALADIN inference on Mussel features (stub until MSK weights are configured)."""
    from .envs import ENVS, env_command

    if checkpoint is None:
        console.print(
            "[red]PALADIN weights are MSK-internal and not configured.[/red] "
            "Set PALADIN_CHECKPOINT or pass --checkpoint."
        )
        raise typer.Exit(1)
    if not (ENVS["paladin"].venv_dir / "bin" / "python").exists():
        console.print(
            f"[red]PALADIN env missing at {ENVS['paladin'].venv_dir}[/red]; "
            "run `argo setup --env paladin`."
        )
        raise typer.Exit(1)
    cmd = env_command(
        "paladin",
        ["python", "-m", "paladin", "--checkpoint", str(checkpoint), str(features or "")],
    )
    console.print("[yellow]PALADIN inference is not wired yet.[/yellow] Would run:")
    console.print("  " + " ".join(cmd))


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
    _require_env("lazyslide")
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
    _require_env("lazyslide")
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
    _require_env("lazyslide")
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
    _require_env("lazyslide")
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


def _experiment_needs_lazyslide(
    config: Path, from_stage: Optional[str], until_stage: Optional[str]
) -> bool:
    """Return whether the selected experiment stages extract or aggregate features.

    An unreadable config returns ``False`` so ``run_experiment`` reports the error itself.
    """
    from .experiment import experiment_plan, load_experiment, select_experiment_plan

    try:
        plan = select_experiment_plan(
            experiment_plan(load_experiment(config)),
            from_stage=from_stage,
            until_stage=until_stage,
        )
    except (KeyError, ValueError, OSError):
        return False
    return any(stage["kind"] in {"extract", "aggregate"} for stage in plan)


@app.command("experiment")
def experiment_command(
    config: Path = typer.Argument(..., help="Experiment TOML file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the stage plan"),
    resume: bool = typer.Option(True, "--resume/--fresh"),
    from_stage: Optional[str] = typer.Option(
        None, "--from-stage", help="Resume at this stage ID after completed prerequisites"
    ),
    until_stage: Optional[str] = typer.Option(
        None, "--until-stage", help="Stop cleanly after this stage ID"
    ),
):
    """Run or resume the configuration-driven end-to-end experiment graph.

    A run whose selected stages include ``extract`` or ``aggregate`` executes in the
    LazySlide environment; dry runs and post-embedding runs stay in core.
    """
    from .experiment import run_experiment

    if not dry_run and _experiment_needs_lazyslide(config, from_stage, until_stage):
        _require_env("lazyslide")

    def report(stage: str, status: str) -> None:
        color = {"completed": "green", "failed": "red", "skipped": "yellow"}.get(status, "cyan")
        console.print(f"[{color}]{status:9s}[/{color}] {stage}")

    try:
        result = run_experiment(
            config,
            dry_run=dry_run,
            resume=resume,
            from_stage=from_stage,
            until_stage=until_stage,
            on_stage=report,
        )
    except (KeyError, ValueError, OSError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    if dry_run:
        console.print(f"Workspace: {result['workspace']}")
        console.print(f"Run directory: {result['run_dir']}")
        for stage in result["plan"]:
            console.print(f"  {stage['id']}")
    else:
        status = result["status"]
        color = "green" if status == "completed" else "yellow"
        console.print(f"[bold {color}]Experiment {status}[/bold {color}]: {result['name']}")
        if status == "paused":
            console.print(f"Resume from: {result['resume_from']}")


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
    # Shared HPC: keep scratch under the working directory, never the system /tmp.
    with tempfile.TemporaryDirectory(prefix=".argo-acceptance-", dir=Path.cwd()) as directory:
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
    _require_env("lazyslide")
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
    from_stage: Optional[str] = typer.Option(
        None, "--from-stage", help="Preflight this stage and its successors"
    ),
    until_stage: Optional[str] = typer.Option(
        None, "--until-stage", help="Preflight only through this stage"
    ),
):
    """Check environment, inputs, models, GPU, tiling provenance, and budgets."""
    # doctor verifies the LazySlide stack (versions, models, tiling), so it runs there.
    _require_env("lazyslide")
    import json

    from .doctor import run_doctor

    report = run_doctor(
        workspace=workspace,
        config_path=config,
        from_stage=from_stage,
        until_stage=until_stage,
    )
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
