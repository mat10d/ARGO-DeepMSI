#!/bin/bash
#SBATCH --job-name=mdiberna_argo_x1_mussel
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# X1 backend equivalence: Mussel runs (documented defaults) for one slide of
# results/analysis/backends/x1_slides.csv (array index = row), written under
# results/analysis/backends/stores/mussel/ (never next to the slides).
#
#   a  mussel-hoptimus0.toml  (OPTIMUS, effective 224 px @ 0.5 um/px, fp32)
#   e  mussel-titan.toml      (CONCH1_5 512 px + TITAN_SLIDE)
#
# Usage (from the repo root):
#   sbatch --array=0-7 scripts/backends/x1_mussel.sh [run ...]   # default: a e

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
ROW="${SLURM_ARRAY_TASK_ID:-0}"
SLIDES=results/analysis/backends/x1_slides.csv
STORES=results/analysis/backends/stores
TIMINGS=results/analysis/backends/timings
mkdir -p logs "$TIMINGS" .tmp_build
export HF_HOME="$PWD/.huggingface_cache" HF_HUB_OFFLINE=1 TMPDIR="$PWD/.tmp_build"

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(a e)
echo "Job ${SLURM_JOB_ID:-local} row $ROW on ${SLURM_NODELIST:-$(hostname)}: ${RUNS[*]}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

status=0
for run in "${RUNS[@]}"; do
    case "$run" in
        a) name=mussel_hoptimus0; config=configs/backends/mussel-hoptimus0.toml ;;
        e) name=mussel_titan; config=configs/backends/mussel-titan.toml ;;
        *) echo "unknown run $run"; exit 2 ;;
    esac
    start=$(date +%s.%N)
    uv run --frozen argo extract "$SLIDES" --backend mussel --config "$config" \
        --out-root "$STORES/mussel" --indices "$ROW"
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
# Mussel/NFS leaves empty pymp-* dirs behind
find .tmp_build -maxdepth 1 -name 'pymp-*' -type d -empty -delete 2>/dev/null
exit $status
