# ARGO-DeepMSI

MSI prediction from whole slide images using [LazySlide](https://github.com/rendeirolab/LazySlide).

## Installation

```bash
# Python 3.11; exact versions come from uv.lock
uv sync --frozen --extra dev --extra dask --extra waiv

# Configure HuggingFace token (for gated models)
cp .env.template .env
# Edit .env and add your HF_TOKEN
```

## Running the Pipeline

### 1. Data Ingestion

Fetch clinical data from REDCap and slide metadata from Halo Link:

```bash
argo ingest
```

Creates:
- `results/data/clinical_table.csv` - Patient MSI labels (one row per patient)
- `results/data/slide_table.csv` - Slide paths and processing metadata

**Slide table columns:**
- `PATIENT` - Patient ID (matches clinical_table)
- `FILENAME` - Absolute path to slide file
- `SITE` - Site identifier (e.g., UITH, retrospective_msk, retrospective_oau)
- `cut_location` - Where slide was sectioned (e.g., MSKCC, UITH, OAUTHC)
- `stain_location` - Where slide was stained (e.g., MSKCC, UITH, OAUTHC)
- `image_location` - Where slide was scanned (always Nigeria for this dataset)

**Note:** Retrospective patients (142-series) may have multiple slides with different
staining locations (MSK vs Nigeria), but they remain the same patient in clinical_table.

### 2. Pyramidal Conversion

Convert any non-pyramidal slides to tiled pyramidal TIFFs (`argo pyramidal` under the hood — requires `libvips`):

```bash
sbatch scripts/pyramidal.sh results/data/slide_table.csv
```

Writes `results/data/slide_table_pyramidal.csv` pointing at the converted files; originals are preserved.

### 3. Feature Extraction

One failure-isolated task per slide, auto-scaling between 1–N GPU workers:

```bash
uv run argo extract-dask results/data/slide_table_pyramidal.csv --max-workers 3
```

Monitor progress:
```bash
squeue -u $USER
tail -f results/runs/dask-extraction/logs/*.err
```

Creates: `data/SITE/slide.zarr/tables/{model}_tiles/` for each slide.

### 4. Aggregation

The wrapper discovers extracted models from the slide zarr stores:

```bash
sbatch scripts/aggregate.sh
```

Creates: `results/embeddings/{model}_{method}/`
- `embeddings.npy` - Slide embedding matrix
- `metadata.csv` - Slide metadata

### 5. Training

The wrapper discovers every completed embedding directory:

```bash
sbatch scripts/train.sh
```

Creates: `results/models/{embedding_type}/`
- `classifier_comparison.csv` - Performance metrics
- `training_data.csv` - Patient-slide-label mappings

## Reproducible experiments

The preferred interface for a new sweep is one versioned TOML file:

```bash
uv run argo doctor --strict --config configs/nigeria-v2.toml
uv run argo experiment configs/nigeria-v2.toml --dry-run
uv run argo experiment configs/nigeria-v2.toml
```

The run is resumable and captures the config, git state, dependency versions,
stage outputs, scorer parameters, and fold audits under `results/runs/<name>/`.
It includes the recent Waiv nested-probe, ABMIL, and from-scratch transformer
experiments as tunable jobs. See [the reproducibility guide](docs/reproducibility.md).
Autonomous work follows [AGENTS.md](AGENTS.md), which fixes the validation
invariants and keeps frozen-head, pretrained end-to-end, adapted-head, and
backbone-finetuning experiments scientifically distinct.

### Feature Extraction

Pass any number of `--model` options; no source edit is needed. LazySlide 0.12
models are discovered from its registry, including newly added encoders. Scale
parallelism with `--min-workers` / `--max-workers`.

### Aggregation and Training

`scripts/aggregate.sh` scans the first available zarr for extracted models.
`scripts/train.sh` scans `results/embeddings/` for completed embedding sets.
Neither script requires a hardcoded model list.

## Available Models

Use the CLI as the source of truth for the current model and aggregation catalog:

```bash
argo models
argo models --check --non-gated-only  # optionally verify installed weights
```

For neural aggregators (prism, titan), use the CLI:
```bash
argo aggregate virchow --method prism
```

## Development

Run static checks and the CPU test suite before submitting changes:

```bash
ruff check argo_deepmsi tests scripts/extract_dask.py
pytest -m "not gpu and not integration and not model_download and not network and not requires_hf_token"
```

Local-data integration, network, model-download, and GPU tests are opt-in through
their corresponding pytest markers.

## References

- **LazySlide**: https://github.com/rendeirolab/LazySlide
- **Paper**: https://doi.org/10.1101/2025.05.28.656548
