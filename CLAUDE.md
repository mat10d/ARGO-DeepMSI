# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

ARGO-DeepMSI is a pipeline for MSI (Microsatellite Instability) prediction from whole slide images, built on [LazySlide](https://github.com/rendeirolab/LazySlide). Single Python environment, single CLI entry point. Nigerian colorectal cancer cohorts from UITH and OAUTHC (retrospective MSK & OAU), all imaged in Nigeria.

## Architecture

```
argo_deepmsi/
├── __main__.py         # enables `python -m argo_deepmsi`
├── cli.py              # Typer CLI (ingest/extract/aggregate/qc/train/visualize/run)
├── data_ingestion.py   # REDCap + Halo Link → clinical_table + slide_table
├── feature_extraction.py  # LazySlide extraction + zarr aggregation + QC filter
├── visualization.py    # Slide viz (cached-zarr aware), UMAP/t-SNE
├── training.py         # sklearn classifiers + MLP; StratifiedGroupKFold on patient_id
└── io_utils.py         # Path management

scripts/
├── extract.sh          # SLURM array (2 groups) for extraction
├── extract_dask.py     # elastic SLURM via dask-jobqueue (alternative)
├── aggregate.sh        # SLURM array (one task per model)
└── train.sh            # SLURM array (one task per embedding dir)

tests/
├── conftest.py            # shared GTEx fixture download
├── test_lazyslide_api.py  # validates LazySlide API contract (18 CPU + 1 GPU)
└── test_argo_pipeline.py  # exercises our wiring end-to-end (12 tests)
```

## Essential Commands

```bash
# Install (base pipeline + dev)
pip install -e ".[dev]"
# With elastic-SLURM dask extraction
pip install -e ".[dev,dask]"

# CLI commands
argo --help                  # Show all commands
argo models                  # List models + aggregation methods
argo ingest                  # REDCap + Halo Link → clinical_table + slide_table
argo extract <table>         # Extract features (zarr per slide, next to svs)
argo qc <table> --model grandqc-artifact   # Filter slides by QC scores
argo aggregate <models>      # Patch → slide embeddings (mean/max/prism/titan/...)
argo visualize               # UMAP + slide plots
argo train <embeddings>      # Train LR/RF/SVM + save
argo run <slides> <clinical> # Full pipeline for one or more models

# Tests
pytest tests/ -v -m "not gpu"    # ~8 min cold, ~1 min after GTEx cached
pytest tests/ -v -m gpu          # adds the one GPU extraction check
```

## Key Dependencies

- **lazyslide / wsidata**: WSI I/O, preprocessing, feature extraction, batch aggregation (`agg_wsi`)
- **anndata + zarr**: scverse data layer; all feature tables live in zarr
- **scanpy + umap-learn**: UMAP / leiden on slide embeddings
- **scikit-learn**: classifiers + `StratifiedGroupKFold`
- **torch / transformers / fairscale / musk**: foundation models
- **dask + distributed + dask-jobqueue**: optional, for `scripts/extract_dask.py`
- **typer / rich**: CLI

## LazySlide Usage Pattern (actual API)

```python
from wsidata import open_wsi, agg_wsi   # NOT zs.WSI
import lazyslide as zs

# Open slide (skip thumbnail for batch jobs)
wsi = open_wsi("path/to/slide.svs", attach_thumbnail=False)

# Preprocessing
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

# Feature extraction — always pass num_workers + batch_size for GPU/CPU overlap
zs.tl.feature_extraction(
    wsi, model="uni2", amp=True, device="cuda",
    num_workers=4, batch_size=64, pbar=False,
)
wsi.write()

# Access features as AnnData (n_tiles × n_features)
features = wsi["uni2_tiles"]
```

**Reopening a cached zarr:** use the `open_wsi(svs_path, store=svs_path.parent, attach_thumbnail=False)` pattern. Opening the `.zarr` directory directly (`open_wsi(zarr_path)`) can KeyError on the recorded reader (e.g. `fastslide`) if that reader isn't installed locally. `extract_features_single_slide` uses this pattern in the "zarr already exists" branch (applied 2026-04-20 after the prior direct-zarr-open variant tripped the `fastslide` KeyError on every already-extracted slide in the first full-cohort run).

## Supported Models

**Patch-level, no auth:** ctranspath, plip, phikon, phikonv2

**Patch-level, gated (HF auth):** uni, uni2, virchow, virchow2, conch, conch_v1.5, gigapath, h-optimus-0, h-optimus-1, hibou-b, hibou-l, chief, musk, medsiglip, omiclip

**Quality control (integrated, see `argo qc`):** grandqc-artifact, grandqc-tissue, pathprofilerqc, focus, focuslitenn

**Aggregation:**
- Simple pooling (statistical): mean, max, median, sum
- Neural encoders (need patch features first): prism (virchow2), titan (conch_v1.5), chief (chief), madeleine (conch)
- Direct slide-level: gigapath-slide-encoder, chief-slide-encoder, gigatime

**Listed in `argo_deepmsi.feature_extraction.PATCH_MODELS / QC_MODELS / SLIDE_ENCODERS`.**

## Output Structure

```
results/
├── data/           # clinical_table.csv, slide_table.csv
├── embeddings/     # per model+method dir, containing:
│   ├── embeddings.npy        # (n_slides × n_features) float32
│   ├── metadata.csv          # slide_id, patient_id, site, n_tiles, zarr_path
│   └── embeddings.h5ad       # AnnData for scverse interop
├── visualizations/ # plots
└── models/         # trained classifiers
```

Patch-level features live inside each slide's `<slide>.zarr/tables/<model>_tiles/` (AnnData in zarr), not a central features directory. Aggregation reads from the per-slide zarrs.

## Environment Variables

```bash
export HF_HOME="/path/to/.huggingface_cache"
export HF_HUB_OFFLINE=1    # For offline compute nodes
```

## REDCap Integration

`.env` file in project root:
```
REDCAP_API_TOKEN=your_token
REDCAP_API_URL=https://redcap.example.com/api/
```

`data_ingestion.py` handles:
- Fetching records via POST to REDCap API
- Extracting MSI status from prospective (`cmo_msi_status`) vs retrospective (`msi_status_mmr`) fields
- Loading Halo Link CSV exports (`halo_link_*.csv`)
- Generating clinical and slide tables with MSI labels

## Pipeline Design Invariants

- **All slide paths flow through `io_utils.py` helpers** — don't hardcode.
- **Cross-validation uses `StratifiedGroupKFold(groups=patient_id)`** — slide-level CV leaks across patients who contribute multiple slides. `compare_classifiers` will `raise ValueError` if `groups` is missing.
- **`load_training_data` prefers `embeddings.h5ad`** when present, falls back to `embeddings.npy + metadata.csv`. It also guards against row-count mismatches that silently mis-align rows.
- **Feature extraction is incremental** — re-running `argo extract` with additional models only runs the missing ones per slide; preprocessing is skipped when the zarr already exists.
- **Simple-pooling aggregation is slide-outer, model-inner** — each zarr is opened once per aggregation pass regardless of how many models are requested.
- **Neural encoder aggregation opens only the zarr** (no redundant SVS re-read).
- **Visualization functions use `_open_cached`** — never re-extract features when a zarr already has them.

## QC Workflow (implemented)

```bash
# 1. Extract QC models alongside (or before) feature models
argo extract results/data/slide_table.csv --model grandqc-artifact

# 2. Filter by per-tile reduction over the QC table
argo qc results/data/slide_table.csv --model grandqc-artifact \
    --reduce mean --threshold 0.5 \
    --output results/data/slide_table_qc.csv

# 3. Extract full features on the filtered table
argo extract results/data/slide_table_qc.csv --model uni2 --model virchow2
```

`filter_slides_by_qc` is model-agnostic: any zarr table named `{model}_tiles` with a 2D `X` works.

## SLURM Parallelization

Two paths:

1. **`scripts/extract.sh`** — static 2-group SLURM array (default). Simple, resilient, each job owns ~half the slides and all models. Guards against empty groups. Mem currently `256G` per task (bumped from 128G after the first full run OOM'd); A6000-20 user cap is 768G total across 3 GPUs, so ≤384G per task is safe if we ever go to 2-task + headroom.
2. **`scripts/extract_dask.py`** — elastic dask-jobqueue, one worker per slide, auto-adapts GPU worker count between `--min-workers` and `--max-workers`. Requires `pip install -e ".[dask]"`. Not yet used in a production run.
3. **`scripts/extract_retry_g0.sh`** — single-task retry template for when one SLURM-array group fails and you want to resume it without cancelling the healthy sibling. Takes a pre-filtered CSV (e.g. `scripts/logs/slide_group_0_retry.csv`) and runs all models against it at 384G.

`scripts/aggregate.sh` and `scripts/train.sh` are array jobs (one task per model / embedding dir) with bounds that match the enabled lists, and guards for empty array slots. Keep the three arrays in sync — extract/aggregate/train should all reference the same 11 models (currently `uni2, virchow2, conch_v1.5, h-optimus-1, gigapath, hibou-b, musk, chief, ctranspath, phikonv2, plip`).

## Known Problem Slides

Some slides blow up extraction memory regardless of model choice. Keep a running list here; reintroduce after pyramidal conversion with `vips`/`bioformats`, or drop from the cohort.

| Slide | Dimensions | Symptom | Status |
|---|---|---|---|
| `LASUTH/HP1353_21_1.svs` | 78048 × 75453, not pyramidal (n_level=1) | OOM during `zs.pp.find_tissues` at both 128G and 256G; kills the SLURM task before the per-slide `except` can swallow it | Excluded from `scripts/logs/slide_group_0_retry.csv` during the 2026-04-20 run. Revisit by generating a pyramidal copy before re-extracting. |

**Why per-slide `try/except` doesn't save you here:** the process itself gets `Killed` by the OOM killer, so `extract_features_single_slide`'s exception handler never runs — the whole SLURM task dies. Pre-filter known offenders out of the slide table before submitting.

## Testing

- `tests/conftest.py` shares a session-scoped GTEx slide fixture downloaded from `rendeirolab/lazyslide-data` on HuggingFace. All tests are CPU-only unless marked `@pytest.mark.gpu`.
- `tests/test_lazyslide_api.py` (Operon) — 19 tests that lock the LazySlide API surface we rely on: `open_wsi`, `find_tissues`, `tile_tissues`, `feature_extraction(num_workers/batch_size)`, `feature_aggregation`, `agg_wsi`, `zs.pl.tissue/tiles`, zarr direct read, incremental extraction.
- `tests/test_argo_pipeline.py` — 12 tests exercising our wiring end-to-end on the same GTEx slide with `resnet50`: `extract_features_single_slide` (fresh + incremental), `aggregate_simple_pooling` (with h5ad + numeric sanity), `load_training_data` (h5ad preference, row-count guard), `compare_classifiers` (requires groups, runs StratifiedGroupKFold), `filter_slides_by_qc`, `_open_cached`.

## Future Work (not yet implemented)

These are LazySlide capabilities we don't yet exploit. They're net-new features, not refactors of existing code:

- **Spatial analysis** — `zs.pp.tile_graph`, `zs.tl.spatial_domain`, `zs.tl.spatial_features`. Potentially useful for tumor-stroma interface and immune-infiltrate distribution in MSI prediction.
- **Vision-language queries** — `zs.tl.text_embedding` + `zs.tl.text_image_similarity` on CONCH/PLIP for zero-shot tissue characterization and interpretable features.
- **Multimodal fusion** — combining image embeddings with clinical text (pathology reports, demographics).

See `docs/lazyslide_gap_analysis.md` for the gap against the full ecosystem, `docs/lazyslide_reference_guide.md` for the verified API recipes, and `docs/refactor_status.md` for the full cross-reference of every Operon-review recommendation against what landed.

## Notes on Specific Models

**MUSK** (`musk`): Gated patch-level model from Stanford. Installed via git dep (`musk @ git+https://github.com/lilab-stanford/MUSK`). Requires `fairscale`.

**PLIP** (`plip`): Non-gated vision-language patch-level model. Previously had sporadic extraction failures — monitor in current runs.
