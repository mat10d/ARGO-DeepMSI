#!/usr/bin/env bash
# Ralph loop driver for ARGO-DeepMSI. Runs on a fry node inside the argo conda env
# with the `claude` CLI on PATH. The driver itself is CPU/orchestration; GPU work is
# submitted by iterations via ralph/gpu_gate.sh (hard cap 3).
#
#   Usage:  bash ralph/loop.sh [MAX_ITERS]
#
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p ralph/logs

# Activate argo env so the iteration's python/pytest resolve.
CONDA_ROOT="${CONDA_ROOT:-/lab/barcheese01/mdiberna/miniconda3}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate argo
fi

MAX_ITERS="${1:-200}"
MODEL="${RALPH_MODEL:-opus}"

for i in $(seq 1 "$MAX_ITERS"); do
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  log="ralph/logs/iter_$(printf '%03d' "$i")_${ts}.log"

  # --- cross-turn GPU wait ------------------------------------------------
  # A GPU task cannot fit in one one-shot `claude -p` turn. The submitting
  # iteration writes the SLURM job/array id to ralph/.waiting_on, commits its
  # WIP, and ends. Here the DRIVER blocks until that job clears the queue, then
  # invokes the next iteration as the HARVEST turn (marker still present so the
  # agent knows to reduce+finish, then it removes the marker).
  if [ -f ralph/.waiting_on ]; then
    jid="$(tr -d '[:space:]' < ralph/.waiting_on)"
    if [ -n "$jid" ] && squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      echo "[loop] iter $i: parked on SLURM job $jid — waiting for it to finish" | tee -a ralph/logs/driver.log
      while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 60; done
      echo "[loop] job $jid cleared queue — running harvest iteration $i" | tee -a ralph/logs/driver.log
    fi
  fi

  echo "=== ralph iter $i @ $ts (model=$MODEL) ===" | tee -a ralph/logs/driver.log

  # One iteration: feed the fixed prompt; Claude Code runs headless with auto-approval.
  cat ralph/AGENT.md | claude -p --dangerously-skip-permissions --model "$MODEL" \
      > "$log" 2>&1 || echo "[loop] claude exited non-zero on iter $i (see $log)" | tee -a ralph/logs/driver.log

  # Stop conditions written by the agent into the journal. Markers are REAL only
  # when they start a line (anchored) — this avoids matching the header comment
  # "# terminal markers: LOOP-COMPLETE ... | HALT-BLOCKED ...". Also require the
  # marker to be in the last 3 lines (i.e. just written), not anywhere in history.
  if tail -n 3 ralph/JOURNAL.md 2>/dev/null | grep -qE '^LOOP-COMPLETE '; then
    echo "[loop] LOOP-COMPLETE detected — stopping at iter $i." | tee -a ralph/logs/driver.log
    break
  fi
  if tail -n 3 ralph/JOURNAL.md 2>/dev/null | grep -qE '^HALT-BLOCKED '; then
    echo "[loop] HALT-BLOCKED detected — stopping for human review at iter $i." | tee -a ralph/logs/driver.log
    break
  fi

  # Safety: a clean iteration must leave a clean tree (it commits its own work).
  # Untracked SLURM logs under scripts/logs and ralph/logs are ignorable noise.
  if ! git diff --quiet || ! git diff --cached --quiet \
     || [ -n "$(git status --porcelain --untracked-files=all | grep -vE 'ralph/logs/|scripts/logs/|ralph/\.waiting_on')" ]; then
    echo "[loop] Dirty working tree after iter $i — halting for human review." | tee -a ralph/logs/driver.log
    git status --short | tee -a ralph/logs/driver.log
    break
  fi
done

echo "[loop] driver exiting." | tee -a ralph/logs/driver.log
