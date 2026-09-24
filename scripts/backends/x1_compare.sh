#!/bin/bash
#SBATCH --job-name=mdiberna_argo_x1_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=20
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# X1: every backend pairing on the 8 study slides (core env, CPU). Reports land in
# results/analysis/backends/compare_*.{csv,md} (+ *_slide.csv for TITAN slide embeddings).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
S=results/analysis/backends/stores
SLIDES=results/analysis/backends/x1_slides.csv
cmp() { uv run --frozen argo compare-backends --slides "$SLIDES" --n 8 --select sites --seed 0 "$@"; }

# primary: Mussel (224 px effective) vs LazySlide 224 px fp32, matched grid
cmp -m hoptimus0 --a lazyslide=$S/lazyslide_224 --b mussel=$S/mussel --stem compare_hoptimus0_224
# documented defaults: LazySlide 256 px vs Mussel 224 px (centre-anchored: tile sizes differ)
cmp -m hoptimus0 --a lazyslide=$S/lazyslide_256 --b mussel=$S/mussel --anchor center \
    --stem compare_hoptimus0_256
# fp16 drift: LazySlide 224 fp32 vs LazySlide 224 amp
cmp -m hoptimus0 --a lazyslide=$S/lazyslide_224 --b lazyslide=$S/lazyslide_224_amp \
    --stem compare_hoptimus0_fp16
# grid-only: LazySlide 224 vs 256 (same backend, centre-anchored)
cmp -m hoptimus0 --a lazyslide=$S/lazyslide_224 --b lazyslide=$S/lazyslide_256 --anchor center \
    --stem compare_hoptimus0_lazyslide_224_vs_256
# TITAN: CONCH v1.5 512 px tiles + TITAN slide embeddings
cmp -m titan --a lazyslide=$S/lazyslide_512 --b mussel=$S/mussel --slide-embedding \
    --stem compare_titan
