# ARGO-DeepMSI

Reproducible MSI prediction on the predefined Nigeria cohort using LazySlide.

The supported interface is one checked-in TOML file executed by `argo
experiment`. The runner owns preprocessing, fresh LazySlide feature extraction,
aggregation, cohort construction, bags, trained heads, scorers, comparison, and
stage-level restart state. Historical scripts are evidence, not the execution
API for a new run.

## Fresh-machine run

Install the locked environment:

```bash
uv sync --frozen --extra dev --extra dask --extra waiv
cp .env.template .env
```

Set `HF_TOKEN`, `REDCAP_API_URL`, and `REDCAP_API_TOKEN` in `.env` or the job
environment. Download the protected whole-slide files and matching PathPresenter
CSV/Excel exports into `data/<site>/`. They are source inputs and are never
committed. The spreadsheet columns are normalized into the stable downstream
slide-table contract by the ingestion stage.

Then use only the configured runner:

```bash
uv run argo doctor --strict --config configs/nigeria-v2.toml
uv run argo experiment configs/nigeria-v2.toml --dry-run
uv run argo experiment configs/nigeria-v2.toml
```

The ingestion stage requires exactly 808 downloaded slides and 217 patients;
the cohort stage requires the fixed 217-patient/47-positive estimand. A partial
download or changed label population fails before it can silently define a new
experiment.

## Cluster checkpoint at tile embeddings

On SLURM, feature extraction is dispatched to Dask workers. Stop the driver
after the newly pinned LazySlide generation has embedded every slide:

```bash
uv run argo doctor --strict --config configs/nigeria-v2.toml --until-stage extract
uv run argo experiment configs/nigeria-v2.toml --until-stage extract
```

The manifest is marked `paused`, not completed. Resume the same immutable config
from a suitable compute allocation:

```bash
uv run argo doctor --strict --config configs/nigeria-v2.toml \
  --from-stage aggregate:mean_pool
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

## Development check

```bash
uv run ruff check argo_deepmsi tests scripts/extract_dask.py
uv run pytest -q
uv run argo self-test
uv lock --check
git diff --check
```

GPU, network, gated-model, and large integration tests are opt-in markers.
