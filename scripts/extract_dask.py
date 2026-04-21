"""Elastic SLURM feature extraction with dask-jobqueue.

Alternative to `scripts/extract.sh` (fixed 2-group array). This script submits
one Dask worker per slide and auto-scales GPU workers within the given bounds,
so we don't pay the walltime cost of whichever group is slowest.

Usage
-----
    conda activate argo
    python scripts/extract_dask.py \
        --slide-table results/data/slide_table.csv \
        --partition nvidia-A6000-20 \
        --max-workers 10

The pattern mirrors the recipe in docs/lazyslide_reference_guide.md section 7.
"""

from __future__ import annotations

import argparse
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
    """Worker-side: run extraction for a single slide. Imports inside the
    function so they happen on the worker, not the submit process."""
    from pathlib import Path as _Path
    from wsidata import open_wsi
    import lazyslide as zs

    p = _Path(slide_path)
    zarr_path = p.with_suffix(".zarr")

    # Decide which models are missing
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

    wsi = open_wsi(str(zarr_path if zarr_path.exists() else p), attach_thumbnail=False)
    if not zarr_path.exists():
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

    for model in to_extract:
        zs.tl.feature_extraction(
            wsi,
            model=model,
            amp=True,
            device="cuda",
            num_workers=4,
            batch_size=64,
            pbar=False,
        )

    wsi.write()
    return {"slide": p.name, "status": "success", "models": to_extract}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slide-table", type=Path, required=True)
    ap.add_argument("--partition", default="nvidia-A6000-20")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--min-workers", type=int, default=1)
    ap.add_argument("--walltime", default="12:00:00")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--memory", default="64 GB")
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--mpp", type=float, default=0.5)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--project-dir", default="/lab/barcheese01/mdiberna/ARGO-DeepMSI")
    args = ap.parse_args()

    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, as_completed
    from tqdm.auto import tqdm

    cluster = SLURMCluster(
        queue=args.partition,
        cores=args.cores,
        processes=1,
        memory=args.memory,
        job_extra_directives=[f"--gres=gpu:1", f"--time={args.walltime}"],
        worker_extra_args=["--resources GPU=1"],
        log_directory=str(Path(args.project_dir) / "scripts" / "logs" / "dask"),
        job_script_prologue=[
            f"source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh",
            "conda activate argo",
            f"export HF_HOME={args.project_dir}/.huggingface_cache",
        ],
    )
    client = Client(cluster)
    cluster.adapt(minimum=args.min_workers, maximum=args.max_workers)

    slides = pd.read_csv(args.slide_table)
    print(f"Submitting {len(slides)} slides × {len(args.models)} models")
    print(f"Dashboard: {client.dashboard_link}")

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
    for fut in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = fut.result()
            if result["status"] == "success":
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  FAIL: {e}")

    print(f"Done. success={n_ok} skipped={n_skip} failed={n_fail}")
    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
