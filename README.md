# ARGO-DeepMSI

Reproducible MSI prediction on the predefined Nigeria cohort using LazySlide.

The supported interface is one checked-in TOML file executed by `argo
experiment`. The runner owns preprocessing, fresh LazySlide feature extraction,
aggregation, cohort construction, bags, trained heads, scorers, comparison, and
stage-level restart state. Historical scripts are evidence, not the execution
API for a new run.

## Fresh-machine run

Install the core environment, then let it build the tool environments from their
locks:

```bash
uv sync --frozen --extra dev
uv run argo setup        # envs/lazyslide + envs/mussel; `--env paladin` is opt-in
uv run argo envs         # status of every environment
cp .env.template .env
```

The pipeline has four stages, each in its own locked environment:

| Stage | Environment | What |
|---|---|---|
| 1. setup | core | `argo setup`: sync envs, HF cache/token, credentials, data paths |
| 2. slide tasks | core | ingest, pyramidal conversion, QC, cohort freeze |
| 3. extract | `envs/lazyslide` (default) or `envs/mussel` | per-slide features in each tool's native format + provenance |
| 4a. analyses | core | Wagner, heads, domain shift, slide-count audit |
| 4b. PALADIN | `envs/paladin` | PALADIN on Mussel H-optimus-0 features (weights MSK-internal; `argo paladin` is a stub) |

Commands that need LazySlide (`extract`, `extract-dask`, `models`, `aggregate`,
`qc`, `visualize`, `run`, and `experiment` with extraction or aggregation stages)
re-execute themselves in `envs/lazyslide`. Mussel runs as a subprocess of
`argo extract --backend mussel`, configured by `configs/backends/*.toml`.
`argo compare-backends` measures how far the two backends' features diverge on
the same slides (`docs/experiments/X1-backend-equivalence.md`).

Set `HF_TOKEN`, `REDCAP_API_URL`, and `REDCAP_API_TOKEN` in `.env` or the job
environment. Download the protected whole-slide files and matching PathPresenter
CSV/Excel exports into `data/<site>/`. They are source inputs and are never
committed. The spreadsheet columns are normalized into the stable downstream
slide-table contract by the ingestion stage.

Then use only the configured runner:

```bash
uv run --frozen --project envs/lazyslide argo doctor --strict --config configs/nigeria-v2.toml
uv run argo experiment configs/nigeria-v2.toml --dry-run
uv run argo experiment configs/nigeria-v2.toml
```

`argo doctor` checks the LazySlide stack, so run it in `envs/lazyslide`. The
ingestion stage requires exactly 808 downloaded slides and 217 patients; the
cohort stage requires the fixed 217-patient/47-positive estimand. A partial
download or changed label population fails before it can silently define a new
experiment.

## Cluster checkpoint at tile embeddings

On SLURM, feature extraction is dispatched to Dask workers. Stop the driver
after the newly pinned LazySlide generation has embedded every slide:

```bash
uv run --frozen --project envs/lazyslide argo doctor --strict \
  --config configs/nigeria-v2.toml --until-stage extract
uv run argo experiment configs/nigeria-v2.toml --until-stage extract
```

The manifest is marked `paused`, not completed. Resume the same immutable config
from a suitable compute allocation:

```bash
uv run --frozen --project envs/lazyslide argo doctor --strict \
  --config configs/nigeria-v2.toml --from-stage aggregate:mean_pool
uv run argo experiment configs/nigeria-v2.toml \
  --from-stage aggregate:mean_pool
```

The runner refuses `--from-stage` when any prerequisite is unfinished. Repeating
the plain experiment command is also safe: completed stages are skipped.

Tile-feature stores are generated artifacts, but they are the practical
post-embedding checkpoint. Each store records LazySlide/wsidata, tiling, and
encoder provenance. `tiling_policy = "require-current"` prevents old and new
tile grids from being mixed.

## Agent experiments after bootstrap

An agent starts from
[`configs/nigeria-postembed-example.toml`](configs/nigeria-postembed-example.toml),
assigns a new `[run].name`, and changes one interpretable hypothesis in an
`[[aggregate]]`, `[[bag]]`, `[[train]]`, or `[[scorer]]` entry. New model logic
belongs in a discoverable scorer module or runner stage, never a one-off script.
Patient-grouped fold rules and strategy labels are enforced by the repository
contract in [AGENTS.md](AGENTS.md).

Discover the current capabilities instead of guessing:

```bash
uv run argo models
uv run argo strategies
uv run argo scorers list
uv run argo scorers show NAME
uv run argo experiment-schema --output /tmp/argo-experiment.schema.json
```

Every run writes its immutable config, environment, input hashes, stage state,
outputs, fold audits, and comparison under `results/runs/<name>/manifest.json`.
Changing a config after its run starts is rejected.

For details on dependency upgrades, model families, and manifests, see
[docs/reproducibility.md](docs/reproducibility.md).

## Results so far

[docs/summary.md](docs/summary.md) gives the current state: frozen Wagner/CTransPath
max/√n is AUROC 0.717 (0.631–0.799) on the 217-patient estimand, and a nested
cohort-trained probe is 0.530. [docs/negative-results-ledger.md](docs/negative-results-ledger.md)
lists every experiment run, with its cohort, validation design, and verdict.

## Repository layout

- `argo_deepmsi/`: package (runner, extraction, `backends/`, `models/`, `scorers/`, `eval/`).
- `envs/`: locked tool environments (`lazyslide`, `mussel`) and the `paladin` setup script.
- `configs/`: checked-in experiment TOMLs; `configs/backends/` holds per-tool extraction
  configs.
- `scripts/`: SLURM wrappers for the core pipeline; `scripts/domain_shift/` holds
  domain-shift evidence jobs and `scripts/qc/` holds QC and error-anatomy jobs. The
  wrappers still carry WI partition names; see "Porting to IRIS" in [CLAUDE.md](CLAUDE.md).
- `docs/experiments/`: detailed write-ups for retained evidence.

The pre-cleanup tree (autorun loop, W1 adaptation code, retired scorers, one-off
scripts, and their results) is frozen on branch `archive/pre-iris-2026-09`.

## Development check

```bash
uv run ruff check argo_deepmsi tests scripts/extract_dask.py
uv run pytest -q
uv run --frozen --project envs/lazyslide pytest -q
uv run argo self-test
uv lock --check
uv lock --check --project envs/lazyslide
uv lock --check --project envs/mussel
git diff --check
```

The core suite passes without LazySlide; `lazyslide`-marked tests run in the second
pytest call. GPU, network, gated-model, and large integration tests are opt-in markers.

`envs/lazyslide` pins `setuptools<81` because spatialdata → xarray_schema still
imports `pkg_resources`; without it a fresh install silently loses LazySlide. Torch
comes from the cu128 index (cu121 for Mussel), which runs on CUDA 12.x drivers. Ruff
lint selection is pinned to `E4`, `E7`, `E9`, `F`.
