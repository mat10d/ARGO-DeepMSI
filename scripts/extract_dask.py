"""Elastic SLURM feature extraction with dask-jobqueue.

Submits one Dask worker per slide and auto-scales GPU workers within the
given bounds. Per-slide failure isolation — a single OOM doesn't kill
other slides.

Usage
-----
    conda activate argo
    pip install -e ".[dask]"

    # Full extraction (foundation models)
    python scripts/extract_dask.py \
        --slide-table results/data/slide_table.csv \
        --partition nvidia-A6000-20 \
        --memory "256 GB" \
        --max-workers 3

    # QC-only pass (fast, can use more workers)
    python scripts/extract_dask.py \
        --slide-table results/data/slide_table.csv \
        --models grandqc-artifact grandqc-tissue \
        --memory "64 GB" \
        --max-workers 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = [
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
]


def _process_slide(slide_path: str, models: list[str], tile_px: int, mpp: float) -> dict:
    """Worker-side: extract features for a single slide.

    Imports happen inside the function so they execute on the Dask worker,
    not the submit process.
    """
    import gc

    from pathlib import Path as _Path
    from wsidata import open_wsi
    import lazyslide as zs

    p = _Path(slide_path)
    zarr_path = p.with_suffix(".zarr")

    # Check which models are already extracted
    existing: set[str] = set()
    if zarr_path.exists() and (zarr_path / "tables").exists():
        existing = {
            d.name.replace("_tiles", "")
            for d in (zarr_path / "tables").iterdir()
            if d.is_dir() and d.name.endswith("_tiles")
        }
    to_extract = [m for m in models if m not in existing]
    if not to_extract:
        return {"slide": p.name, "status": "skipped", "models": []}

    def _open(fresh_preprocess: bool):
        """Open the WSI, preprocessing if the zarr doesn't exist yet.

        Uses svs path + store=parent to avoid the fastslide KeyError on
        zarrs recorded with a reader backend we don't have installed.
        """
        if zarr_path.exists():
            w = open_wsi(str(p), store=str(p.parent), attach_thumbnail=False)
        else:
            w = open_wsi(str(p), attach_thumbnail=False)
        if fresh_preprocess and not zarr_path.exists():
            zs.pp.find_tissues(w)
            zs.pp.tile_tissues(w, tile_px=tile_px, mpp=mpp)
            w.write()
        return w

    # Preprocess once (creates the initial zarr with tissues/tiles) so
    # every subsequent open reads from a persistent store.
    if not zarr_path.exists():
        w0 = _open(fresh_preprocess=True)
        del w0
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # Extract one model at a time — open fresh, write, drop. This caps
    # the worker's RAM footprint at one model's feature table instead
    # of accumulating all 11 in memory before a single final write()
    # (observed ~190 GiB peak → OOM with the accumulate-then-write
    # pattern even with DataLoader num_workers trimmed).
    for model in to_extract:
        wsi = _open(fresh_preprocess=False)
        zs.tl.feature_extraction(
            wsi,
            model=model,
            amp=True,
            device="cuda",
            num_workers=2,
            batch_size=32,
            pbar=False,
        )
        wsi.write()
        del wsi

        # Aggressive memory cleanup between models. Three layers:
        # 1. Python GC — drop circular refs holding AnnData/zarr caches
        # 2. torch CUDA cache — free GPU memory blocks back to the allocator
        # 3. libc malloc_trim — actually return freed pages to the OS
        #    (glibc hoards freed memory in per-thread arenas by default)
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
        except Exception:
            pass

    return {"slide": p.name, "status": "success", "models": to_extract}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slide-table", type=Path, required=True)
    ap.add_argument("--partition", default="nvidia-A6000-20")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--min-workers", type=int, default=1)
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--memory", default="256 GB")
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--mpp", type=float, default=0.5)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument(
        "--conda-env", default="argo",
        help="Conda environment name to activate on workers",
    )
    ap.add_argument(
        "--project-dir",
        default="/lab/barcheese01/mdiberna/ARGO-DeepMSI",
        help="Repo root (for HF cache + conda activation)",
    )
    args = ap.parse_args()

    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, as_completed
    from tqdm.auto import tqdm

    project = Path(args.project_dir)
    log_dir = project / "scripts" / "logs" / "dask"
    log_dir.mkdir(parents=True, exist_ok=True)

    cluster = SLURMCluster(
        queue=args.partition,
        cores=args.cores,
        processes=1,
        memory=args.memory,
        # nanny=False: without the Nanny wrapper the Worker process itself
        # is the SLURM job's main process (non-daemonic), which lets the
        # DataLoader inside zs.tl.feature_extraction(num_workers>0) fork
        # children. With nanny=True (the default) the Worker is daemonic
        # and DataLoader forks raise "daemonic processes are not allowed
        # to have children". Trade-off: no automatic worker restarts if a
        # worker OOMs — per-slide failure still propagates back to the
        # driver as a future exception (handled below).
        nanny=False,
        job_extra_directives=[
            "--gres=gpu:1",
            f"--time={args.walltime}",
        ],
        worker_extra_args=[
            "--resources GPU=1",
            # Lower the memory pause threshold from 80% to 95% so Dask doesn't
            # pre-emptively pause tasks while Python/glibc are just slow to
            # return freed pages. The real protection is SLURM's cgroup OOM
            # killer, not Dask's watermark. With MALLOC_ARENA_MAX=2 and
            # malloc_trim between models, actual RSS tracks usage more closely.
            "--memory-limit", "0",  # disable Dask's memory management entirely;
                                     # rely on SLURM cgroup limits instead
        ],
        log_directory=str(log_dir),
        job_script_prologue=[
            # Limit glibc malloc arenas to prevent per-thread memory hoarding.
            # Default is 8×nCPU (~128 arenas × many MB each); 2 arenas caps
            # the RSS overhead from freed-but-not-returned allocations.
            "export MALLOC_ARENA_MAX=2",
            f"source $(conda info --base)/etc/profile.d/conda.sh",
            f"conda activate {args.conda_env}",
            f"export HF_HOME={project}/.huggingface_cache",
            f"cd {project}",
        ],
    )
    client = Client(cluster)
    cluster.adapt(minimum=args.min_workers, maximum=args.max_workers)

    slides = pd.read_csv(args.slide_table)
    print(f"Slides: {len(slides)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Workers: {args.min_workers}–{args.max_workers} × {args.memory}")
    print(f"Dashboard: {client.dashboard_link}")
    print()

    futures = [
        client.submit(
            _process_slide,
            row["FILENAME"],
            args.models,
            args.tile_px,
            args.mpp,
            resources={"GPU": 1},
            pure=False,
        )
        for _, row in slides.iterrows()
    ]

    n_ok = n_skip = n_fail = 0
    failed_slides = []
    t0 = time.time()

    for fut in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = fut.result()
            if result["status"] == "success":
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            n_fail += 1
            failed_slides.append(str(e)[:200])
            tqdm.write(f"  FAIL: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/3600:.1f}h. success={n_ok} skipped={n_skip} failed={n_fail}")

    if failed_slides:
        fail_log = log_dir / "failed_slides.txt"
        with open(fail_log, "w") as f:
            f.write("\n".join(failed_slides))
        print(f"Failed slides logged to: {fail_log}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
