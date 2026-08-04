# You are extending ARGO-DeepMSI in the NO-NEW-TRAINING-DATA regime.

## C1 RESET — current primary evaluation contract
The canonical primary cohort is all 217 patients / 803 feature-complete slides. QC flags are
retained for the historical 181-patient sensitivity analysis but no longer define primary
eligibility. Cohort-trained model/configuration selection must use the nested outer/inner
patient-grouped machinery in `argo_deepmsi.eval.validation`; the final outer predictions may
not select an encoder, correction, aggregation, or threshold. Patient cohort is distinct from
slide processing location: the 83 paired retrospective patients form one `retrospective`
patient cohort.

## D-PHASE (historical) — every inherited choice is a HYPOTHESIS
The A-phase is done (A0 Harmony is the OAUTHC frontier at 0.683; normalization A1/A2 falsified;
A3 adaptation real but capped; A6 names "site-routed ensemble" as next). Before building that
readout, D-phase tests the pipeline choices we INHERITED as defaults — each kept only if it
improves target-site (OAUTHC + small-site) held-out rule-out performance, reported PER-SITE:
  - D1: error-structure + COMPLEMENTARITY across scorers. Decisive gate for a case-conditioned
    readout: do different scorers fail on DIFFERENT patients (headroom) or the SAME ones
    (representational ceiling)? Reports oracle-selection ceiling.
  - D2: QC-exclusion ablation (hard / none / SOFT reliability-weight). QC-flag is ~99% OAUTHC vs
    ~20% MSK, so "exclude bad QC" ~= "shrink OAUTHC" — is it even helping?
  - D3: tumor-filter + aggregation ablation.
D-tasks assume NOTHING about the inherited QC/tumor choices. If a D-result overturns an
assumption baked into A0-A6, note it — those may need re-running on a better-justified cohort.

## MISSION UPDATE (A-phase) — recover OAUTHC, do NOT abstain on it
The original plan concluded "abstain on OAUTHC." That is RETIRED: OAUTHC is 48% of patients and
44% of MSI-H positives and CANNOT be excluded or abstained on. Root cause is established in
docs/experiments/D0-oauthc-batch-rootcause.md: OAUTHC-prospective vs retrospective_oau is a
PROCESSING-PIPELINE batch effect (tissue cut at OAUTHC vs MSKCC), not unlearnable biology —
raw CONCH-TITAN separates the two pipelines at site-pred AUROC 1.00, Harmony drops it to 0.82
and lifts OAUTHC MSI-AUROC 0.61->0.69. The SAME-INSTITUTION ceiling to target is retro-OAU 0.80.

For every A-phase task: report **OAUTHC held-out patient AUROC + spec@sens95/96 FIRST**, then
overall + full per-site. Evaluate on the FULL cohort WITH OAUTHC included (no abstention). The
champion floor (best clean AUROC >= 0.710, tol 0.005) still holds as no-regression. Base model
unless a task says otherwise: conch_v1.5_titan + Harmony. Do NOT resurrect global site-invariance
equalization (R2/R3 proved it regresses to the mean) — corrections must be OAUTHC-TARGETED.

Provenance rule (this project has repeatedly caught fabricated external numbers): never write an
external metric, author/year, or journal name you have not retrieved from source IN THIS RUN.
If unread, write the flag, not the number.


Read `ralph/CONTRACT.md` first — every rule there is binding.

## Environment
Run all python in the `argo` conda env: `conda activate argo` (root
`/lab/barcheese01/mdiberna/miniconda3`) or use
`/lab/barcheese01/mdiberna/miniconda3/envs/argo/bin/python` directly. Importing the argo
package is slow (~40s, pulls torch/lazyslide) — expect that overhead per verify run. For GPU
work submit SLURM jobs through `ralph/gpu_gate.sh` (hard cap 3); CPU partitions are `24`/`20`/`18`,
GPU partitions include `nvidia-A6000-20`/`nvidia-A100-20`, accounts `wibrusers`/`weissman`.

## CRITICAL: each iteration is a single one-shot turn — you cannot "wait" or "resume"
You are invoked as `claude -p` (headless, one prompt, no follow-up). When your turn
ends, YOU ARE GONE — there is no "later", no wake-up, no monitor callback. NEVER say
"I'll resume when notified" or background a job and end — it will be orphaned. There are
exactly two ways to run long work:

**(A) CPU work that fits (< ~40 min): run it FOREGROUND, blocking, within this turn.**
Leaderboard regen (15-30 min), CPU extraction, small fits — use blocking `srun ... <cmd>`
or `sbatch --wait ... <script>`, finish, verify, commit, journal. One long foreground
iteration is correct; a short iteration that defers work is a bug.

**(B) GPU work (or any multi-hour job): SUBMIT, hand off to the driver, end cleanly.**
The driver waits across turns for you. Protocol:
  1. Submit via `bash ralph/gpu_gate.sh sbatch <script>` (respects the 3-GPU cap).
     Capture the job/array id from sbatch's stdout.
  2. Write JUST that id to `ralph/.waiting_on` (e.g. `echo 10300717 > ralph/.waiting_on`).
  3. Leave the task `status: doing`, add a `checkpoint:` note in BACKLOG.yaml saying what
     the pending job produces and how to reduce it.
  4. `git add -A && git commit` the WIP (scripts + code + marker). Do NOT append a JOURNAL
     `done` line yet. End your turn.
  5. The driver blocks until job id in `.waiting_on` clears the queue, then starts the next
     iteration — the HARVEST turn. In that turn: see `.waiting_on` present + task `doing`,
     REDUCE the job outputs (e.g. `python -m argo_deepmsi.eval.cohort --artifact-qc-dir ...`),
     run verify, and only then `rm ralph/.waiting_on`, set `status: done`, write the
     experiment doc, append the JOURNAL line, commit.

- **Every turn must end committed.** Either: task fully done (verify green, JOURNAL line,
  committed); OR a GPU handoff WIP committed with `.waiting_on` set + task `doing` (B above);
  OR task reverted and marked `blocked` with a `reason:`. NEVER leave an uncommitted dirty
  tree — the driver halts on it.
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
