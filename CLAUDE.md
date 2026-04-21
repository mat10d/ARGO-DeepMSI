# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

ARGO-DeepMSI is a pipeline for MSI (Microsatellite Instability) prediction from whole slide images, built on [LazySlide](https://github.com/rendeirolab/LazySlide). Single Python environment, single CLI entry point. Nigerian colorectal cancer cohorts from UITH and OAUTHC (retrospective MSK & OAU), all imaged in Nigeria.

## Architecture

```
argo_deepmsi/
├── __main__.py         # enables `python -m argo_deepmsi`
├── cli.py              # Typer CLI (ingest/pyramidal/extract/qc/aggregate/visualize/train/run)
├── data_ingestion.py   # REDCap + Halo Link → clinical_table + slide_table
├── slide_prep.py       # Non-pyramidal → tiled pyramidal TIFF (via libvips)
├── feature_extraction.py  # LazySlide extraction + zarr aggregation + QC filter
├── visualization.py    # Slide viz (cached-zarr aware), UMAP/t-SNE
├── training.py         # sklearn classifiers + MLP; StratifiedGroupKFold on patient_id
└── io_utils.py         # Path management

scripts/
├── pyramidal.sh        # SLURM wrapper for `argo pyramidal` (CPU, one-shot)
├── extract_dask.py     # elastic SLURM via dask-jobqueue (GPU, per-slide)
├── extract_dask.sh     # SLURM wrapper so the dask driver isn't on the head node
├── aggregate.sh        # auto-discovers models from zarrs; loops argo aggregate
└── train.sh            # auto-discovers embeddings dirs; loops argo train

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
argo pyramidal <table>       # Convert non-pyramidal slides → tiled pyramidal TIFF
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

- **`scripts/pyramidal.sh`** — one-shot CPU job wrapping `argo pyramidal`. Run once on a new slide table; serial (the vips tile/compress step is cheap relative to extraction).
- **`scripts/extract_dask.py`** — elastic dask-jobqueue for feature extraction. One worker per slide, auto-adapts GPU worker count between `--min-workers` and `--max-workers`. Per-slide failure isolation via future exceptions. Uses `open_wsi(svs_path, store=parent)` internally (zarr-reopen fix), preprocesses once then extracts one-model-per-open to cap RAM, and applies three-layer cleanup between models (`gc.collect → torch.cuda.empty_cache → malloc_trim`). Worker prologue sets `MALLOC_ARENA_MAX=2` to stop glibc from hoarding freed pages. Dask's own memory watermark is disabled (`--memory-limit 0`) — SLURM cgroup is the only backstop. Requires `pip install -e ".[dask]"`.
- **`scripts/extract_dask.sh`** — SLURM wrapper so the dask driver itself doesn't sit on the head node.
- **`scripts/aggregate.sh` / `scripts/train.sh`** — auto-discovery (no hardcoded model list). `aggregate.sh` scans the first zarr's `tables/*_tiles` dirs; `train.sh` scans `results/embeddings/*/` for anything with `embeddings.npy`/`embeddings.h5ad`. Works with 3 models or 30 — no edits between Phase 1 and Phase 2.

## Execution Plan (two-phase)

Full rationale + iterate-while-waiting list in `docs/remaining_tasks.md`.

**Phase 1 — baseline on the current 3 GPUs (~1 day):**

```
1. sbatch scripts/pyramidal.sh results/data/slide_table.csv
2. sbatch scripts/extract_dask.sh results/data/slide_table_pyramidal.csv \
       --models uni2 virchow2 conch_v1.5 --max-workers 3 --memory "256 GB"
3. sbatch scripts/aggregate.sh
4. sbatch scripts/train.sh
```

**Phase 1e (iterate while Phase 2 queues):** neural aggregators (PRISM on virchow2, TITAN on conch_v1.5), XGBoost / attention-MIL classifiers, fix QC via `zs.seg.artifact()` polygons, scanpy UMAP on h5ads, spatial analysis.

**Phase 2 — full sweep on MSK cluster (10-20 GPUs):** same `extract_dask.sh` with the full 11-model list and higher `--max-workers`. The 3 Phase-1 models are already in every zarr → extraction skips them, only the 8 new models run. Aggregate/train scripts auto-discover the new models with zero edits.

QC filtering (`grandqc`) is deferred — LazySlide's `zs.tl.feature_extraction(model="grandqc-artifact")` dispatch is broken upstream. Proper fix is to refactor `filter_slides_by_qc` against `zs.seg.artifact()` polygon output; done during Phase 1e.

## Known Problem Slides

Non-pyramidal WSIs (n_levels=1) OOM during `zs.pp.find_tissues` because LazySlide's thumbnail fallback tries to load the full-resolution image into memory. Mitigated by `argo pyramidal` (see `argo_deepmsi/slide_prep.py`, run in-cluster via `sbatch scripts/pyramidal.sh`), which converts such slides to tiled pyramidal TIFFs via `libvips` (`conda install -c conda-forge libvips`) and writes a `*_pyramidal.csv` slide table pointing at the converted files. The original SVS is preserved alongside (`.pyramidal.tiff` sibling).

| Slide | Dimensions | Symptom | Status |
|---|---|---|---|
| `LASUTH/HP1353_21_1.svs` | 78048 × 75453, n_levels=1 | OOM during `zs.pp.find_tissues` at 128G and 256G | Handled by pyramidal conversion (2026-04-21). |

**Why per-slide `try/except` doesn't rescue an OOM:** the process itself gets `Killed` by the OOM killer, so the exception handler in `extract_features_single_slide` never runs. The group-array path dies with the slide; the dask path isolates the failure to that worker but still loses the slide unless it's been converted to pyramidal first.

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
