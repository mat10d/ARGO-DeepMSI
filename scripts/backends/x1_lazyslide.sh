#!/bin/bash
#SBATCH --job-name=mdiberna_argo_x1_lazyslide
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# X1 backend equivalence: LazySlide runs for one slide of results/analysis/backends/x1_slides.csv
# (array index = row). Each run writes to its own out-root (one tiling generation per root)
# under results/analysis/backends/stores/, never next to the slides.
#
#   b  lazyslide_256      lazyslide-hoptimus0.toml       (documented: 256 px, fp32)
#   c  lazyslide_224      lazyslide-hoptimus0-224.toml   (matched grid, fp32)
#   d  lazyslide_224_amp  lazyslide-hoptimus0-224.toml --amp (ARGO historical fp16)
#   e  lazyslide_512      lazyslide-titan.toml           (CONCH v1.5 512 px + TITAN)
#
# Usage (from the repo root):
#   sbatch --array=0-7 scripts/backends/x1_lazyslide.sh [run ...]   # default: b c d e

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
ROW="${SLURM_ARRAY_TASK_ID:-0}"
SLIDES=results/analysis/backends/x1_slides.csv
STORES=results/analysis/backends/stores
TIMINGS=results/analysis/backends/timings
mkdir -p logs "$TIMINGS" .tmp_build
export HF_HOME="$PWD/.huggingface_cache" HF_HUB_OFFLINE=1 TMPDIR="$PWD/.tmp_build"

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(b c d e)
echo "Job ${SLURM_JOB_ID:-local} row $ROW on ${SLURM_NODELIST:-$(hostname)}: ${RUNS[*]}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

status=0
for run in "${RUNS[@]}"; do
    case "$run" in
        b) name=lazyslide_256; args=(--config configs/backends/lazyslide-hoptimus0.toml) ;;
        c) name=lazyslide_224; args=(--config configs/backends/lazyslide-hoptimus0-224.toml) ;;
        d) name=lazyslide_224_amp; args=(--config configs/backends/lazyslide-hoptimus0-224.toml --amp) ;;
        e) name=lazyslide_512; args=(--config configs/backends/lazyslide-titan.toml) ;;
        *) echo "unknown run $run"; exit 2 ;;
    esac
    start=$(date +%s.%N)
    uv run --frozen --project envs/lazyslide argo extract "$SLIDES" --backend lazyslide \
        "${args[@]}" --out-root "$STORES/$name" --indices "$ROW"
    rc=$?
    end=$(date +%s.%N)
    printf "run\tname\trow\tjob\tnode\trc\twall_s\n%s\t%s\t%s\t%s\t%s\t%s\t%.1f\n" \
        "$run" "$name" "$ROW" "${SLURM_JOB_ID:-local}" "${SLURM_NODELIST:-$(hostname)}" "$rc" \
        "$(echo "$end - $start" | bc)" > "$TIMINGS/${name}_${ROW}.tsv.new"
    # keep the first successful timing; a rerun that only skips existing outputs is not a timing
    if [ $rc -ne 0 ] || [ ! -s "$TIMINGS/${name}_${ROW}.tsv" ]; then
        mv "$TIMINGS/${name}_${ROW}.tsv.new" "$TIMINGS/${name}_${ROW}.tsv"
    else
        rm -f "$TIMINGS/${name}_${ROW}.tsv.new"
    fi
    [ $rc -ne 0 ] && status=$rc
done
exit $status
