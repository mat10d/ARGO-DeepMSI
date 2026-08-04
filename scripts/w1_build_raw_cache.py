"""Materialize the fixed aligned raw-tile subset used by W1 PEFT."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from argo_deepmsi.w1_peft import build_fixed_raw_tile_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--w1-dir", type=Path, default="results/experiments/w1_ctranspath"
    )
    parser.add_argument("--raw-tiles", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    index = pd.read_csv(args.w1_dir / "slide_index.csv")
    path = build_fixed_raw_tile_cache(
        index,
        feature_cache_dir=args.w1_dir / "bag_cache_uniform_1024",
        outdir=args.w1_dir / f"raw_tile_cache_{args.raw_tiles}",
        raw_tiles=args.raw_tiles,
        workers=args.workers,
    )
    print(f"W1 raw tile cache: {path}")


if __name__ == "__main__":
    main()
