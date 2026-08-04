"""Freeze the W1 patient-fold and CTransPath feature-index contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from argo_deepmsi.w1 import build_bag_cache, write_w1_data_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default="results/data/cohort_clean.csv")
    parser.add_argument(
        "--slide-table",
        type=Path,
        default="results/data/slide_table_pyramidal.csv",
    )
    parser.add_argument(
        "--outdir", type=Path, default="results/experiments/w1_ctranspath"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="also materialize a deterministic 1024-tile reservoir per slide",
    )
    parser.add_argument("--cache-tiles", type=int, default=1024)
    args = parser.parse_args()
    folds, index, contract = write_w1_data_contract(
        cohort_path=args.cohort,
        slide_table_path=args.slide_table,
        outdir=args.outdir,
        n_splits=args.folds,
        seed=args.seed,
    )
    print(
        f"W1 contract: {len(folds)} patients / {len(index)} slides / "
        f"{contract.n_positive_patients} MSI-H; sha256={contract.cohort_sha256}"
    )
    if args.build_cache:
        cache_dir = args.outdir / f"bag_cache_uniform_{args.cache_tiles}"
        data_path, _ = build_bag_cache(
            index,
            outdir=cache_dir,
            max_tiles=args.cache_tiles,
            seed=args.seed,
        )
        print(f"W1 tile reservoir: {data_path}")


if __name__ == "__main__":
    main()
