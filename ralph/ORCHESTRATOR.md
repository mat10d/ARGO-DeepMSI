# ARGO-DeepMSI — Autonomous Orchestrator Brief

You are a **resident** Claude Code session running in a tmux window on a fry login node.
You will work continuously, unattended, until the backlog is complete or you must halt for
a human. Read this whole file, then read `ralph/CONTRACT.md` (binding invariants), then begin.

## 0. Execution model — you are RESIDENT, not one-shot
Unlike the `ralph/loop.sh` worker (which is a fresh `claude -p` per iteration), you persist
across the entire run. This changes one thing: **you CAN wait on long jobs in-session.** When
you submit a SLURM job, poll `squeue` yourself and block until it finishes, then harvest — no
`.waiting_on` handoff file, no cross-turn dance. But because you persist, YOU are responsible
for your own context hygiene (§7). The ledger on disk — `ralph/BACKLOG.yaml` + `ralph/JOURNAL.md`
— is always the source of truth, never your memory.

## 1. Mission
Prove that an open-source, no-new-training-data pipeline on our own ~200-patient Nigerian CRC
cohort can be **competitive with MSIntuit CRC** (Owkin, Nat Commun 2023) as an MSI pre-screening
rule-out test. We do NOT train on TCGA / CPTAC / PAIP / any external slide set — fitting a head
on OUR cohort is fine; external pretraining/fitting is forbidden (enforced by
`tests/test_no_external_data.py`). Borrow their *ideas* (tumor-region segmentation, artifact QC,
metric alignment), not their data.

## 2. Ground truth — environment & paths
- Repo: `/lab/barcheese01/mdiberna/ARGO-DeepMSI`, branch `autorun-scaffold`. Work only here.
- Python: the `argo` conda env. Activate with
  `source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh && conda activate argo`,
  or call `/lab/barcheese01/mdiberna/miniconda3/envs/argo/bin/python` directly. Importing the
  argo package is slow (~40s — it pulls torch/lazyslide); expect that overhead on every verify.
- Always invoke tools through the env interpreter: `python -m pytest ...`, NOT bare `pytest`
  (a stale system pytest shadows the env's on PATH and crashes).
- SLURM: CPU partitions `24`/`20`/`18`; GPU partitions include `nvidia-A6000-20`,
  `nvidia-A100-20`. Accounts: `wibrusers`, `weissman`. Put `#SBATCH` directives at the top of
  every job script (partition + account are REQUIRED — there is no default partition).

## 3. The loop you run
Repeat until a stop condition (§8):
1. Read `ralph/BACKLOG.yaml` and the tail of `ralph/JOURNAL.md`.
2. Pick the highest-`priority` task with `status: todo` whose `deps` are all `done`.
   Set it `status: doing`. Do ONLY that task. Never invent work not in the backlog.
3. Implement it inside the repo shape defined in `CONTRACT.md`. One scorer per file, etc.
4. If it needs GPU: submit via `bash ralph/gpu_gate.sh sbatch <script>` (the gate blocks until
   fewer than 3 GPU jobs are in flight — the hard cap is 3, never exceed it). Then **poll
   `squeue -j <jobid> -h` and wait** until it clears. Then reduce/harvest the outputs.
5. If the task changed any scorer's scores or added a scorer, regenerate the leaderboard
   yourself (§6) — the gate does NOT do this for you.
6. Run `bash ralph/verify.sh`. It MUST exit 0. Check the exit code directly (`echo $?`), never
   trust a piped-to-`tail` exit status. If it fails: fix YOUR change until green. If you cannot
   without weakening a test or gate, **revert your change** (`git checkout -- .`; `git clean -fd`
   for new files) and set the task `status: blocked` with a one-line `reason:`. NEVER weaken a gate.
7. On green: set `status: done`; write `docs/experiments/<id>.md` (≤1 page: method, the headline
   result line in MSIntuit units, a per-site + per-bag-size table, verdict vs the champion);
   append one line to `ralph/JOURNAL.md` — `<UTC> <id> done <one-line-result>`; `git commit`
   referencing the task id. **Every task ends committed with a clean tree.**

## 4. GPU discipline
- Hard cap **3 concurrent GPU jobs**. Always submit through `ralph/gpu_gate.sh`.
- GPU jobs can run hours. That is fine — you are resident, so wait on them. Do not fake progress.
- Prefer sharded arrays (one GPU per shard, ≤3 shards) for whole-cohort passes.

## 5. The competitive target — report EVERY result in these units
- **MSIntuit**: sensitivity 0.96–0.98, specificity 0.46–0.47 (rule-out screener), Cohen's κ 0.82.
  (Verified from the paper abstract.)
- **FM-MSI benchmark** (ScienceDirect PII S0895611125001892): CONCH + few-shot/cluster adaptation,
  externally validated. Its exact spec@sens has NOT been transcribed from the full text — read
  the paper before citing any figure from it. Do not invent numbers.
- **Current champion**: `calibrated_pool` (max/√n, zero params) — patient AUROC **0.710** clean,
  but only **spec 0.064 @ sens 0.96** and NPV 0.923. Beat it, and report spec@sens95/96 + NPV,
  not AUROC alone. The gap to MSIntuit's spec 0.46 @ sens 0.96 is the whole game.

## 6. Verify & leaderboard
- Gate: `bash ralph/verify.sh` → tests + regime guard + metric-contract + leaderboard freshness
  + no-regression floor (best clean AUROC must stay ≥ 0.710, tol 0.005).
- Leaderboard regen (YOUR job after any scoring change; 15–30 min, run it and wait):
  `python -m argo_deepmsi.eval.qc_comparison --outdir results/comparison`
  (add `--qc-csv results/data/problem_slides.csv --slide-table results/data/slide_table_pyramidal.csv`
  once the Q-phase has built `cohort_clean.csv`).

## 7. Context hygiene (because you persist)
- After finishing each task, treat the ledger as your memory. Do not carry a task's details
  forward — re-read `BACKLOG.yaml`/`JOURNAL.md` at the start of the next.
- If your context grows large, summarize progress into `ralph/JOURNAL.md`, then compact and
  continue from the ledger. The run must be resumable from disk alone at any moment.

## 8. Stop conditions — halt and say so clearly
- **All tasks `done` and verify green** → append `LOOP-COMPLETE <UTC>` to `ralph/JOURNAL.md`,
  print a final summary, stop.
- **Nothing runnable and some tasks `blocked`** → append `HALT-BLOCKED <UTC> <ids>`, stop.
- **Any invariant would have to be violated to proceed** (need external data, need a 4th GPU,
  a gate can't pass without weakening it) → stop and explain; do not proceed.

## 9. Do NOT
- Train on or download any external slide set. Fabricate metrics. Weaken or delete a test/gate.
- Exceed 3 GPU jobs. Leave an uncommitted dirty tree. Edit files outside the repo.
- Work more than one backlog task at a time, or invent tasks not in the backlog.

Begin: read `ralph/CONTRACT.md`, then `ralph/BACKLOG.yaml`, then start the loop at §3.
