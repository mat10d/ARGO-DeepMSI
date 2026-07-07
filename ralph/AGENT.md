# You are extending ARGO-DeepMSI in the NO-NEW-TRAINING-DATA regime.

Read `ralph/CONTRACT.md` first — every rule there is binding.

## Environment
Run all python in the `argo` conda env: `conda activate argo` (root
`/lab/barcheese01/mdiberna/miniconda3`) or use
`/lab/barcheese01/mdiberna/miniconda3/envs/argo/bin/python` directly. Importing the argo
package is slow (~40s, pulls torch/lazyslide) — expect that overhead per verify run. For GPU
work submit SLURM jobs through `ralph/gpu_gate.sh` (hard cap 3); CPU partitions are `24`/`20`/`18`,
GPU partitions include `nvidia-A6000-20`/`nvidia-A100-20`, accounts `wibrusers`/`weissman`.

## CRITICAL: each iteration is a single one-shot turn — run everything FOREGROUND
You are invoked as `claude -p` (headless, one prompt, no follow-up). When your turn
ends, YOU ARE GONE — there is no "later", no wake-up, no monitor callback. Therefore:
- **NEVER background a job and say "waiting".** Run long jobs (SLURM leaderboard regen,
  training, extraction) in the FOREGROUND and BLOCK until they finish within THIS turn:
  use `srun ... <cmd>` (blocking) or `sbatch --wait ... <script>`, not `sbatch &` /
  `nohup &` / `submit + return`. A backgrounded job is orphaned and killed the moment
  your turn ends (this already happened once and cost an iteration).
- **A leaderboard regen takes 15-30 min — that is fine, block on it.** One long
  foreground iteration is correct; a short iteration that defers work is a bug.
- **Every turn must end in a terminal, consistent state:** either (a) task fully done,
  verify green, committed, JOURNAL line appended; or (b) task reverted and marked
  `blocked` with a reason; or (c) if genuinely mid-multistep, commit a coherent WIP with
  the task left `doing` and a `checkpoint:` note in BACKLOG.yaml — but NEVER leave an
  uncommitted dirty tree (the driver halts on it).
- The terminal markers you append must start the line exactly: `LOOP-COMPLETE <UTC>` or
  `HALT-BLOCKED <UTC> <ids>` (the driver greps `^LOOP-COMPLETE ` / `^HALT-BLOCKED `).

## Leaderboard is your responsibility, not the gate's
Whenever your iteration lands a new scorer or changes any scorer's scores, you MUST
regenerate the leaderboard yourself before running verify:
```
python -m argo_deepmsi.eval.qc_comparison --outdir results/comparison
```
(add `--qc-csv results/data/problem_slides.csv --slide-table results/data/slide_table_pyramidal.csv`
once the Q-phase has built `cohort_clean.csv`). This re-runs every scorer's compute_batch and
takes 15+ min, so it is NOT in verify.sh — verify only checks the leaderboard is present and
newer than all scorer outputs. If you skip it, the freshness gate fails.

Then, each iteration, do EXACTLY ONE thing:

1. Read `ralph/BACKLOG.yaml`. Read the last 30 lines of `ralph/JOURNAL.md`.
2. Pick the highest-`priority` task with `status: todo` whose `deps` are all `done`.
   - If none are unblocked: run `bash ralph/verify.sh`. If it passes AND every task is `done`,
     append a line `LOOP-COMPLETE <UTC>` to `ralph/JOURNAL.md` and STOP.
   - If some tasks are `blocked` and nothing is runnable, append `HALT-BLOCKED <UTC> <ids>` and STOP.
   - Do NOT invent new work not in the backlog.
3. Set that task `status: doing` in BACKLOG.yaml. Implement it — and ONLY it. Stay inside the
   repo shape in CONTRACT.md §Repo shape.
4. For any GPU work: submit via SLURM through `ralph/gpu_gate.sh` (hard cap 3 concurrent).
   Never submit a 4th GPU job while 3 are in flight.
5. Run `bash ralph/verify.sh`. It must exit 0 (tests + metric-contract + leaderboard regen +
   no-regression floor).
   - If it fails: fix YOUR change until green. If you cannot without weakening a test/gate,
     revert your change (`git checkout -- .` / `git clean -fd` for new files you added) and set
     the task `status: blocked` with a one-line `reason:` in BACKLOG.yaml. NEVER weaken a gate.
6. On green: set `status: done`; write `docs/experiments/<id>.md` (<= 1 page: method, headline
   result line vs the MSIntuit target, per-site + per-bag-size table, verdict); append one line
   to `ralph/JOURNAL.md` (`<UTC> <id> done <one-line-result>`); `git commit` referencing the id.

## The competitive target (report every result against it)
- MSIntuit: sensitivity 0.96-0.98, specificity 0.46-0.47 (rule-out screener), Cohen's kappa 0.82.
- FM-MSI benchmark (ScienceDirect PII S0895611125001892): CONCH + few-shot/cluster adaptation,
  externally validated. Its exact spec@sens is NOT yet transcribed from the paper — read the
  full text and record the numbers before using it as a comparator. Do not cite a figure you
  have not read from source.
- Current best: calibrated max/sqrt(n), patient AUROC 0.710 (clean, 198 patients). Beat it — and
  report it in MSIntuit's units (spec @ sens 0.95/0.96), not AUROC alone.

## Output discipline
Output NOTHING to chat except the task id you did and its result line. The ledger
(BACKLOG.yaml + JOURNAL.md) is the source of truth, not your memory.
