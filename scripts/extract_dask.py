"""Compatibility wrapper for ``argo extract-dask``.

The implementation lives in :mod:`argo_deepmsi.dask_extraction` so local and
SLURM extraction share the same LazySlide preprocessing and provenance logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from argo_deepmsi.dask_extraction import DEFAULT_MODELS, run_dask_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--partition", default="nvidia-A6000-20")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--memory", default="256 GB")
    parser.add_argument("--tile-px", type=int, default=256)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conda-env")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tiling-policy", choices=("reuse", "require-current"), default="require-current"
    )
    args = parser.parse_args()
    report = run_dask_extraction(
        slide_table=args.slide_table,
        models=args.models,
        partition=args.partition,
        max_workers=args.max_workers,
        min_workers=args.min_workers,
        walltime=args.walltime,
        cores=args.cores,
        memory=args.memory,
        tile_px=args.tile_px,
        mpp=args.mpp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conda_env=args.conda_env,
        workspace=args.workspace,
        output_dir=args.output,
        overwrite=args.overwrite,
        tiling_policy=args.tiling_policy,
    )
    counts = report["counts"]
    print(
        f"Done: success={counts['success']} skipped={counts['skipped']} failed={counts['failed']}"
    )


if __name__ == "__main__":
    main()
