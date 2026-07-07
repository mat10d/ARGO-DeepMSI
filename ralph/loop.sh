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

MAX_ITERS="${1:-200}"
MODEL="${RALPH_MODEL:-opus}"

for i in $(seq 1 "$MAX_ITERS"); do
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  log="ralph/logs/iter_$(printf '%03d' "$i")_${ts}.log"
  echo "=== ralph iter $i @ $ts (model=$MODEL) ===" | tee -a ralph/logs/driver.log

  # One iteration: feed the fixed prompt; Claude Code runs headless with auto-approval.
  cat ralph/AGENT.md | claude -p --dangerously-skip-permissions --model "$MODEL" \
      > "$log" 2>&1 || echo "[loop] claude exited non-zero on iter $i (see $log)" | tee -a ralph/logs/driver.log

  # Stop conditions written by the agent into the journal.
  if tail -n 5 ralph/JOURNAL.md 2>/dev/null | grep -q "LOOP-COMPLETE"; then
    echo "[loop] LOOP-COMPLETE detected — stopping at iter $i." | tee -a ralph/logs/driver.log
    break
  fi
  if tail -n 5 ralph/JOURNAL.md 2>/dev/null | grep -q "HALT-BLOCKED"; then
    echo "[loop] HALT-BLOCKED detected — stopping for human review at iter $i." | tee -a ralph/logs/driver.log
    break
  fi

  # Safety: a clean iteration must leave a clean tree (it commits its own work).
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[loop] Dirty working tree after iter $i — halting for human review." | tee -a ralph/logs/driver.log
    git status --short | tee -a ralph/logs/driver.log
    break
  fi
done

echo "[loop] driver exiting." | tee -a ralph/logs/driver.log
