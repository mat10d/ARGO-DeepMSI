#!/usr/bin/env bash
# GPU concurrency gate — HARD CAP of 3 concurrent SLURM GPU jobs.
# Usage:  bash ralph/gpu_gate.sh sbatch scripts/my_gpu_job.sh [args...]
# Blocks (polling) until fewer than $GPU_MAX GPU jobs are running/pending for
# this user, then submits. Never lets a 4th GPU job into the queue.
set -uo pipefail

GPU_MAX="${GPU_MAX:-3}"
POLL_SECS="${POLL_SECS:-60}"
USER_NAME="$(whoami)"

# Count this user's GPU jobs (running + pending) via gres/tres.
count_gpu_jobs() {
  # squeue: any job requesting gpu in its TRES, states R (running) or PD (pending)
  squeue -u "$USER_NAME" -h -t R,PD -o "%b|%T" 2>/dev/null \
    | grep -i -c 'gpu' || true
}

while :; do
  n="$(count_gpu_jobs)"
  if [ "${n:-0}" -lt "$GPU_MAX" ]; then
    break
  fi
  echo "[gpu_gate] $n/$GPU_MAX GPU jobs in flight — waiting ${POLL_SECS}s..." >&2
  sleep "$POLL_SECS"
done

echo "[gpu_gate] slot free ($(count_gpu_jobs)/$GPU_MAX) — submitting: $*" >&2
exec "$@"
