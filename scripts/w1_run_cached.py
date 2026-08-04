"""Run one prespecified W1 cached-CTransPath candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from argo_deepmsi.models.w1_heads import CACHED_CANDIDATES
from argo_deepmsi.w1_training import run_cached_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=sorted(CACHED_CANDIDATES))
    parser.add_argument(
        "--w1-dir", type=Path, default="results/experiments/w1_ctranspath"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outdir = run_cached_candidate(
        args.candidate, w1_dir=args.w1_dir, device=args.device, seed=args.seed
    )
    print(f"W1 candidate complete: {outdir}")


if __name__ == "__main__":
    main()
