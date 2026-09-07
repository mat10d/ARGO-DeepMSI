# Reproducible experiments and new-machine setup

The supported workflow is now a locked Python environment plus a TOML run file.
Ad-hoc scripts remain useful as historical records, but new experiments should
enter through `argo experiment` or `argo scorers run` so parameters and software
versions are captured automatically.

## Bootstrap a new machine

Install Python 3.11 and [uv](https://docs.astral.sh/uv/), clone this repository,
then run:

```bash
uv sync --frozen --extra dev --extra dask --extra waiv
uv run argo version
uv run argo models
uv run argo self-test
uv run pytest tests/test_package_boundaries.py tests/test_reproducible_workflows.py -q
```

`uv.lock` is the source of truth. Do not start a new machine from `pip install`
without the lock unless you are deliberately updating dependencies. Model weights
and slide data are not committed; restore the data paths in the slide table and set
`HF_TOKEN` for gated encoders.

The two bootstrap tests are offline. Once network access is available, validate
the real LazySlide integration against its public sample slide:

```bash
uv run pytest tests/test_argo_pipeline.py tests/test_lazyslide_api.py \
  -m 'network and not gpu and not model_download' -q
```

## Why LazySlide 0.12 is the new default

The project pins the LazySlide 0.12 minor series and wsidata 0.10–0.11. LazySlide
0.12 rewrote tile generation and made background filtering exact and vectorized.
It fixes out-of-bounds tiles, missed thin strips/small islands, and errors near
concave tissue boundaries. The resulting tile grids intentionally differ from
0.10, so they define a new preprocessing generation.

Every newly created slide zarr now contains `argo_manifest.json`, recording the
LazySlide/wsidata versions, tile size, MPP, and the software versions used for each
encoder. For a clean v2 run, use `tiling_policy = "require-current"`. It rejects an
old or unversioned zarr instead of adding new embeddings to a mismatched grid.

On the new machine, point the slide table at a location without old `.zarr` stores
and let 0.12 create them. If old stores are copied for comparison, keep them under a
separate path; do not overwrite them. This retains the old result as a reproducible
baseline and makes the 0.12 extraction an explicit new experiment.

## Run the Nigeria-only sweep

Review the shipped configuration first:

```bash
uv run argo doctor --strict --config configs/nigeria-v2.toml
uv run argo experiment configs/nigeria-v2.toml --dry-run
```

Change `[run].name`, GPU partition, and worker count, then run:

```bash
uv run argo experiment configs/nigeria-v2.toml
```

On a single workstation, change only `engine = "local"`. The runner ignores
the retained SLURM fields (`partition`, cluster worker bounds, cores, memory,
walltime, and `conda_env`) and records their names in the extraction-stage
manifest. `batch_size`, `num_workers`, `device`, and model selection still apply.

The run is stage-resumable. Repeating the command skips completed stages. Changing
the TOML after a run starts is rejected; give a changed hypothesis a new run name.
Outputs live under `results/runs/<name>/`:

- `config.toml`: immutable snapshot of the run definition;
- `manifest.json`: git revision/dirty state, dependency versions, input-table hashes,
  stage state, and artifacts;
- `bags/`: deterministic tile bags for any encoder or aligned concatenation;
- `training/`: configurable LR/RF/SVM comparisons;
- `scorers/`: scores, fold audits, metrics, and exact scorer parameters;
- `scorers/*/attention.npz`: optional fold-valid CLS-to-tile attention and source tile indices;
- `comparison.csv`: run-local ranking, including multiple variants of one scorer.

The example replaces the recent Waiv one-offs with parameterized stages: base-vs-
Waiv nested probes, Phaet/Mascaret/concatenated ABMIL, and from-scratch Wagner-style
transformers. To try another embedder, add it to `[extract].models`, aggregation,
and a bag variant; no Python edit is required.

Extraction is all-or-nothing by default at the experiment boundary: every slide
is attempted and the failure report is saved, then the stage fails rather than
letting a partial cohort flow downstream. Set `allow_failures = true` only for an
intentional diagnostic run.

## Keep training regimes explicit

Digital-pathology systems that share an encoder can still answer different
scientific questions. Experiment metadata distinguishes frozen foundation
features with a cohort-trained head, pretrained end-to-end predictors, adapted
pretrained heads, and backbone fine-tuning. Run `argo strategies` for the exact
machine-readable names. A run spanning multiple families uses
`strategy = "mixed"` and must label every train/scorer stage individually.

The optional `[run.budget]` table bounds model count, total stages, modeling
stages, epochs, and repeats, and records whether network access or model downloads
are authorized. Budget violations fail during validation, before GPU work starts.
Generate the editor/agent schema with:

```bash
uv run argo experiment-schema --output /tmp/argo-experiment.schema.json
```

`argo doctor` is read-only. It checks the locked pathology stack, Git state,
disk, requested accelerator, input tables and slide paths, model registration,
gated-model credentials, and existing feature-store tiling provenance. `--json`
provides a stable report for automation, while `--strict` makes warnings fail.

## Iterate on one scorer

```bash
uv run argo scorers list
uv run argo scorers show nested_linear_probe
uv run argo scorers run nested_linear_probe \
  --run-name probe-robust-v1 \
  --param 'embeddings=["phaet_mean","mascaret_mean"]' \
  --param repeats=5
```

`--recompute` is the default. `--cache` is only for replaying a canonical cached
result and cannot be combined with new parameters. Add `--publish` only when a run
should replace the canonical scorer cache used by the legacy comparison tooling.

## Updating the environment intentionally

```bash
uv lock --upgrade-package lazyslide --upgrade-package wsidata
uv sync --extra dev --extra dask --extra waiv
uv run pytest tests/test_model_registry.py tests/test_lazyslide_api.py \
  -m 'not gpu and not model_download and not requires_hf_token' -v
```

Commit `pyproject.toml` and `uv.lock` together. If a tiling dependency changes,
start a new run name and new feature-store location rather than mixing generations.
