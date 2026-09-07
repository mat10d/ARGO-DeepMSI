# ARGO-DeepMSI agent operating contract

This repository supports autonomous implementation and experiment iteration,
but an agent must preserve the scientific estimand, validation boundaries, and
reproducibility record. Read this file before changing code or launching work.

## Mission

Improve MSI prediction and understanding on the predefined Nigeria cohort.
Prefer a well-audited negative result over an optimistic result produced by
leakage, post-hoc cohort changes, or an incomparable training regime.

## Pathology-ML strategy families

Do not collapse these into a single generic "model" category. Record one of
these values in experiment TOML and in every train/scorer stage of a mixed run:

- `frozen_foundation_trained_head`: a frozen patch/slide foundation model plus
  a head trained on our cohort. Linear probes, random forests, SVMs, ABMIL, and
  newly initialized bag transformers belong here.
- `pretrained_end_to_end`: a pretrained slide-to-MSI prediction system evaluated
  without fitting on the target cohort.
- `adapted_pretrained_head`: a pretrained prediction head, prompt, or adapter
  updated using target-cohort training folds.
- `backbone_finetune`: some or all of the foundation backbone is updated using
  target-cohort training folds.
- `mixed`: an explicit comparison containing labelled stages from multiple
  families. Never use `mixed` to avoid labelling individual stages.

These strategies answer different scientific questions. Compare them in one
leaderboard only when the training data, split policy, and adaptation allowance
are stated explicitly.

## Non-negotiable invariants

- Patient groups, never slides, define CV boundaries.
- Preprocessing, encoder/head selection, calibration, and threshold selection
  happen inside the relevant training fold.
- The primary cohort and label definition cannot change inside an experiment.
- Never use test-site labels to choose a model, tune a hyperparameter, stop
  training, calibrate probabilities, or choose an operating point.
- Fit-on-cohort methods must emit out-of-fold predictions for reported training
  performance. Cached in-sample predictions are not evidence.
- Preserve old feature stores. LazySlide 0.12 tiling is a new preprocessing
  generation; do not mix its tile grid with legacy embeddings.
- Every expensive run needs a unique run name, a checked-in TOML config, a
  resource budget, and a successful `argo doctor` preflight.
- Do not publish scorer caches unless the run was explicitly designated as the
  new canonical result.
- Do not commit credentials, raw protected data, model caches, or rebuildable
  large tensors.

## Canonical commands

```bash
uv sync --frozen --extra dev --extra dask --extra waiv
uv run argo doctor --config configs/nigeria-v2.toml
uv run argo experiment configs/nigeria-v2.toml --dry-run
uv run argo experiment configs/nigeria-v2.toml
```

Discover capabilities instead of guessing names or parameters:

```bash
uv run argo models
uv run argo strategies
uv run argo scorers list
uv run argo scorers show NAME
uv run argo experiment-schema --output /tmp/argo-experiment.schema.json
```

## Autonomous experiment loop

1. Read the last run manifest, fold audit, comparison table, and relevant
   experiment note. State the unresolved hypothesis.
2. Classify the proposed method using the strategy families above.
3. Copy an existing TOML, give it a new run name, change one interpretable
   hypothesis at a time, and keep the resource budget finite.
4. Run `argo doctor --strict --config ...`, then `argo experiment ... --dry-run`.
5. Run the experiment. Resume the same config after infrastructure failures;
   use a new run name if any scientific parameter changes.
6. Inspect patient-level metrics, per-site results, fold audits, failures, and
   attention artifacts. Check for leakage and missing-cohort rows before ranking.
7. Record the result, including null/negative outcomes. Propose the next run only
   from completed, comparable evidence.

Stop and request human direction when the next step would change the cohort,
labels, held-out-site policy, external training-data allowance, or consume more
resources than the checked-in budget.

## Development loop

Before editing, inspect `git status` and preserve unrelated work. Add new
experimental methods as scorer modules or configurable runner stages, not as a
new one-off script. Keep package imports lazy so listing commands do not load
Torch or download weights.

Run this acceptance sequence before handing work off:

```bash
uv run ruff check argo_deepmsi tests scripts/extract_dask.py
uv run pytest -q
uv run argo self-test
uv lock --check
git diff --check
```

Network, GPU, gated-model, and large integration tests are opt-in pytest markers.
Run the relevant marker explicitly when changing that boundary. The synthetic
self-test replaces WSI/model inference and feature-store reads with deterministic
arrays; orchestration, aggregation transforms, bag construction, training,
scoring, comparison, provenance, and budget code are the production paths.

## Architecture map

- `argo_deepmsi/experiment.py`: ordered, resumable experiment execution.
- `argo_deepmsi/experiment_schema.py`: strategy vocabulary, strict TOML keys,
  semantic validation, and generated JSON Schema.
- `argo_deepmsi/doctor.py`: read-only machine/environment/data preflight.
- `argo_deepmsi/feature_extraction.py`: canonical tiling, extraction, manifests,
  and aggregation.
- `argo_deepmsi/bags.py`: deterministic arbitrary-encoder tile bags.
- `argo_deepmsi/scorer_runner.py`: uniform scorer execution and evaluation.
- `argo_deepmsi/scorers/`: one discoverable module per scoring method.
- `configs/`: checked-in experiment definitions; one hypothesis per run name.
- `results/runs/<name>/manifest.json`: source of truth for run state and provenance.

Historical scripts and notes are evidence, not the preferred execution API.
When a useful one-off behavior is found, encode its choices as validated CLI/TOML
parameters and add a regression test.
