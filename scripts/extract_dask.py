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

    # Open WSI — use svs path + store=parent to avoid fastslide KeyError
    # when the zarr was created by a different reader backend.
    if zarr_path.exists():
        wsi = open_wsi(str(p), store=str(p.parent), attach_thumbnail=False)
    else:
        wsi = open_wsi(str(p), attach_thumbnail=False)

    # Preprocess only for new slides (existing zarr already has tissues/tiles)
    if not zarr_path.exists():
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

    for model in to_extract:
        # num_workers=0: dask's worker processes are daemonic, and
        # DataLoader(num_workers>0) forks daemonic children, which the
        # stdlib forbids ("daemonic processes are not allowed to have
        # children"). The dask parallelism already gives us slide-level
        # concurrency, so tile-level DataLoader workers aren't needed.
        zs.tl.feature_extraction(
            wsi,
            model=model,
            amp=True,
            device="cuda",
            num_workers=0,
            batch_size=64,
            pbar=False,
        )

    wsi.write()
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
        job_extra_directives=[
            "--gres=gpu:1",
            f"--time={args.walltime}",
        ],
        worker_extra_args=["--resources GPU=1"],
        log_directory=str(log_dir),
        job_script_prologue=[
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
