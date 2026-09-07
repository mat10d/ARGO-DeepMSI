"""Elastic SLURM/Dask execution for the canonical feature extractor."""

from __future__ import annotations

import gc
import shlex
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .reproducibility import environment_snapshot, write_json

DEFAULT_MODELS = (
    "uni2",
    "virchow2",
    "conch_v1.5",
    "h-optimus-1",
    "gigapath",
    "hibou-b",
    "musk",
    "chief",
    "ctranspath",
    "phikonv2",
    "plip",
)


def _rss_gib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError):
        pass
    return 0.0


def _release_worker_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def process_slide(
    slide_path: str,
    models: Sequence[str],
    *,
    tile_px: int,
    mpp: float,
    amp: bool,
    device: str,
    num_workers: int,
    batch_size: int,
    overwrite: bool,
    tiling_policy: str,
) -> dict[str, Any]:
    """Worker entry point; one model per reopen keeps peak host memory bounded."""
    from .feature_extraction import extract_features_single_slide

    slide = Path(slide_path)
    started = time.time()

    def log(message: str) -> None:
        print(
            f"[RSS={_rss_gib():.1f} GiB t={time.time() - started:.1f}s] "
            f"[SLIDE={slide.name}] {message}",
            flush=True,
        )

    zarr_path = slide.with_suffix(".zarr")
    existing = set()
    if (zarr_path / "tables").exists():
        existing = {
            path.name.removesuffix("_tiles")
            for path in (zarr_path / "tables").iterdir()
            if path.is_dir() and path.name.endswith("_tiles")
        }
    completed = []
    for model in models:
        was_present = model in existing and not overwrite
        log(f"validate existing: {model}" if was_present else f"extract start: {model}")
        result = extract_features_single_slide(
            slide,
            models=[model],
            tile_px=tile_px,
            mpp=mpp,
            amp=amp,
            device=device,
            overwrite=overwrite,
            num_workers=num_workers,
            batch_size=batch_size,
            tiling_policy=tiling_policy,  # type: ignore[arg-type]
        )
        if result is None:
            raise RuntimeError(f"{slide.name}: extraction failed for {model}")
        if not was_present:
            completed.append(model)
        _release_worker_memory()
        log(f"validated: {model}" if was_present else f"extract done: {model}")
    status = "success" if completed else "skipped"
    return {"slide": slide.name, "status": status, "models": completed}


def run_dask_extraction(
    *,
    slide_table: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    partition: str = "nvidia-A6000-20",
    max_workers: int = 3,
    min_workers: int = 1,
    walltime: str = "24:00:00",
    cores: int = 16,
    memory: str = "256 GB",
    tile_px: int = 256,
    mpp: float = 0.5,
    amp: bool = True,
    device: str = "cuda",
    num_workers: int = 2,
    batch_size: int = 32,
    overwrite: bool = False,
    tiling_policy: str = "require-current",
    conda_env: str | None = None,
    workspace: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Submit one failure-isolated GPU task per slide and capture a run report."""
    from dask.distributed import Client, as_completed
    from dask_jobqueue import SLURMCluster
    from tqdm.auto import tqdm

    if not models:
        raise ValueError("At least one extraction model is required")
    models = tuple(dict.fromkeys(models))
    from .feature_extraction import list_available_models

    unknown_models = sorted(set(models) - set(list_available_models()))
    if unknown_models:
        raise ValueError(f"Unknown extraction models: {unknown_models}")
    if min_workers < 0 or max_workers < 1 or min_workers > max_workers:
        raise ValueError("Require 0 <= min_workers <= max_workers and max_workers >= 1")
    if tiling_policy not in {"reuse", "require-current"}:
        raise ValueError("tiling_policy must be 'reuse' or 'require-current'")

    workspace = (workspace or Path.cwd()).resolve()
    slide_table = slide_table.resolve()
    slides = pd.read_csv(slide_table)
    if "FILENAME" not in slides:
        raise ValueError(f"{slide_table} has no 'FILENAME' column")
    slides = slides.dropna(subset=["FILENAME"]).drop_duplicates("FILENAME")
    if slides.empty:
        raise ValueError(f"{slide_table} contains no slide paths")
    output_dir = output_dir or workspace / "results" / "runs" / "dask-extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prologue = [
        "export MALLOC_ARENA_MAX=2",
        f"export HF_HOME={shlex.quote(str(workspace / '.huggingface_cache'))}",
        f"export ARGO_WORKSPACE={shlex.quote(str(workspace))}",
        f"cd {shlex.quote(str(workspace))}",
    ]
    if conda_env:
        prologue[1:1] = [
            "source $(conda info --base)/etc/profile.d/conda.sh",
            f"conda activate {shlex.quote(conda_env)}",
        ]

    cluster = SLURMCluster(
        queue=partition,
        cores=cores,
        processes=1,
        memory=memory,
        nanny=False,
        job_extra_directives=["--gres=gpu:1", f"--time={walltime}"],
        worker_extra_args=["--resources", "GPU=1", "--memory-limit", "0"],
        log_directory=str(log_dir),
        job_script_prologue=prologue,
    )
    client = Client(cluster)
    cluster.adapt(minimum=min_workers, maximum=max_workers)
    futures = {
        client.submit(
            process_slide,
            str(row.FILENAME),
            list(models),
            tile_px=tile_px,
            mpp=mpp,
            amp=amp,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            overwrite=overwrite,
            tiling_policy=tiling_policy,
            resources={"GPU": 1},
            pure=False,
        ): str(row.FILENAME)
        for row in slides.itertuples()
    }
    results = []
    try:
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - isolate individual slide failures
                results.append(
                    {
                        "slide": Path(futures[future]).name,
                        "status": "failed",
                        "error": str(error),
                    }
                )
    finally:
        client.close()
        cluster.close()

    report = {
        "slide_table": slide_table,
        "models": list(models),
        "parameters": {
            "partition": partition,
            "min_workers": min_workers,
            "max_workers": max_workers,
            "walltime": walltime,
            "cores": cores,
            "memory": memory,
            "tile_px": tile_px,
            "mpp": mpp,
            "amp": amp,
            "device": device,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "overwrite": overwrite,
            "tiling_policy": tiling_policy,
        },
        "counts": {
            status: sum(result["status"] == status for result in results)
            for status in ("success", "skipped", "failed")
        },
        "slides": results,
        "environment": environment_snapshot(workspace),
    }
    write_json(output_dir / "extraction.json", report)
    return report
