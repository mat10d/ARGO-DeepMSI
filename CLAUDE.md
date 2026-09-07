# CLAUDE.md

The repository-wide operating contract is [AGENTS.md](AGENTS.md). Its scientific
invariants, strategy taxonomy, canonical commands, and acceptance checks apply to
all agent-driven work; the notes below provide additional implementation context.

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
├── experiment.py       # TOML experiment orchestration + resumable manifests
├── scorer_runner.py    # parameterized execution for the lazy scorer registry
├── bags.py             # deterministic arbitrary-encoder MIL bag builder
├── dask_extraction.py  # SLURM/Dask front-end to the canonical extractor
└── io_utils.py         # Path management

scripts/
├── pyramidal.sh        # SLURM wrapper for `argo pyramidal` (CPU, one-shot)
├── extract_dask.py     # elastic SLURM via dask-jobqueue (GPU, per-slide)
├── extract_dask.sh     # SLURM wrapper so the dask driver isn't on the head node
├── aggregate.sh        # auto-discovers models from zarrs; loops argo aggregate
└── train.sh            # auto-discovers embeddings dirs; loops argo train

tests/
├── conftest.py                    # opt-in markers + anonymous GTEx fixture
├── test_lazyslide_api.py          # real-slide LazySlide API contract
├── test_argo_pipeline.py          # end-to-end project wiring
└── test_reproducible_workflows.py # manifests, bags, scorer runner, attention
```

## Essential Commands

```bash
# Reproduce the locked Python 3.11 environment
uv sync --frozen --extra dev --extra dask --extra waiv

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
argo experiment <config>     # resumable, versioned experiment sweep
argo scorers list/show/run   # inspect or execute any scorer without a wrapper
argo extract-dask <table>    # multi-GPU extraction through the same core path

# Tests (bare pytest skips network/GPU/model-download/local-data integrations)
pytest -q
pytest -m network -v             # public GTEx integration
pytest -m gpu -v                 # GPU extraction checks
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

Run `argo models`; it is the source of truth. Patch encoders and neural slide
encoders are capability-discovered from `lazyslide-models`, so a LazySlide catalog
addition does not require a second hardcoded list here. Segmentation, QC, and
style-transfer models are deliberately not advertised as patch embedders.

**Waiv robust encoders (gated, non-commercial academic; `argo_deepmsi/models/waiv.py`):** phaet (fine-tuned Phikon-v2, 1024-d), mascaret (fine-tuned Midnight-12k, 1536-d). Custom remote-code HF repos registered into LazySlide's `MODEL_REGISTRY` on import of `argo_deepmsi.models`. Requires accepting the license at huggingface.co/wearewaiv. See `docs/competitive/waiv-robustify-2607.md`.

**Quality control (integrated, see `argo qc`):** grandqc-artifact, grandqc-tissue, pathprofilerqc, focus, focuslitenn

**Aggregation:** mean/max/median/sum plus every installed registry entry with the
`slide_encoder` capability (for example PRISM, TITAN, CHIEF, Madeleine, GigaPath,
and MOOZY in the locked LazySlide generation).

## Output Structure

```
results/
├── data/           # clinical_table.csv, slide_table.csv
├── embeddings/     # per model+method dir, containing:
│   ├── embeddings.npy        # (n_slides × n_features) float32
│   ├── metadata.csv          # slide_id, patient_id, site, n_tiles, zarr_path
│   └── embeddings.h5ad       # AnnData for scverse interop
├── visualizations/ # plots
├── models/         # trained classifiers
└── runs/           # immutable configs, manifests, fold audits, scores, attention
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
- **Feature extraction is incremental within one tiling generation** — re-running `argo extract` with additional models only runs missing ones. The CLI defaults to `tiling_policy="require-current"`; legacy reuse must be explicit.
- **LazySlide 0.12 is the current preprocessing generation** — its corrected exact/vectorized tiling intentionally differs from older grids. New stores write `argo_manifest.json`; use `tiling_policy="require-current"` in comparative runs and never mix old/unversioned grids with new features.
- **Experiment parameters live in TOML, not one-off wrappers** — use `configs/nigeria-v2.toml` and a new `[run].name` per hypothesis. Completed stages resume from `results/runs/<name>/manifest.json`; a changed config cannot reuse that run directory.
- **Scorers are lazy and filename-discovered** — one module + `register(...)` is enough. Do not eagerly import scorer/model stacks from package `__init__.py` files.
- **Simple-pooling aggregation is slide-outer, model-inner** — each zarr is opened once per aggregation pass regardless of how many models are requested.
- **Neural encoder aggregation opens only the zarr** (no redundant SVS re-read).
- **Visualization functions use `_open_cached`** — never re-extract features when a zarr already has them.

## QC Workflow

`argo qc` filters an existing `{model}_tiles` numeric table and remains useful for
legacy stores. LazySlide 0.12 QC/segmentation models are not patch embedders and
must be run through the corresponding `zs.seg`/prediction API before filtering;
the model catalog no longer offers invalid `argo extract --model <segmentation>`
combinations.

## SLURM Parallelization

- **`scripts/pyramidal.sh`** — one-shot CPU job wrapping `argo pyramidal`. Run once on a new slide table; serial (the vips tile/compress step is cheap relative to extraction).
- **`argo extract-dask` / `argo_deepmsi/dask_extraction.py`** — elastic dask-jobqueue for feature extraction. One worker per slide, auto-adapts GPU worker count between `--min-workers` and `--max-workers`. It calls the canonical extractor one model per reopen to cap RAM, preserves per-slide failure isolation, and applies `gc.collect → torch.cuda.empty_cache → malloc_trim` between models. `scripts/extract_dask.py` is only a compatibility wrapper. Requires the `dask` extra.
- **`scripts/extract_dask.sh`** — SLURM wrapper so the dask driver itself doesn't sit on the head node.
- **`scripts/aggregate.sh` / `scripts/train.sh`** — auto-discovery (no hardcoded model list). `aggregate.sh` scans the first zarr's `tables/*_tiles` dirs; `train.sh` scans `results/embeddings/*/` for anything with `embeddings.npy`/`embeddings.h5ad`. Works with 3 models or 30 — no edits between Phase 1 and Phase 2.

## Canonical Experiment Plan

Use `configs/nigeria-v2.toml` instead of composing phase-specific wrappers:

```bash
uv run argo experiment configs/nigeria-v2.toml --dry-run
uv run argo experiment configs/nigeria-v2.toml
```

Change `[run].name` for each hypothesis. The config controls encoder selection,
tiling, Dask resources, aggregation, bags, conventional classifiers, nested probes,
ABMIL, from-scratch transformers, and attention capture. Completed stages resume;
configuration drift is rejected. See `docs/reproducibility.md`.

## Known Error Modalities (empirical — 2026-04-22 cohort run)

Each mode below is something we hit on the 808-slide Nigerian cohort. Mitigations are landed; the residual failure rate from data-level issues is ~0.6% (5 / 808 slides).

### Slide-level failures

| Modality | Symptom | Root cause | Mitigation | Residual impact |
|---|---|---|---|---|
| **Non-pyramidal WSI** | OOM inside `zs.pp.find_tissues` loading a huge thumbnail | `n_levels == 1` forces LazySlide to load the full-res image | `argo pyramidal` (`scripts/pyramidal.sh`) converts to tiled pyramidal TIFF via libvips | 13/808 converted; none failed after |
| **MPP-less `generic-tiff`** | OOM during `tile_tissues` (not the thumbnail step — the tile-grid builder) | Source SVS has no MPP metadata; vips defaults output to 1000 μm/px → millions of bogus tiles | `argo pyramidal` now probes source MPP via openslide; if None or >100 μm/px, stamps `xres/yres` → 0.25 μm/px (40×) into the output TIFF | 12/808 slides were affected; all rescued |
| **openslide can't parse SVS** | Driver logs `FAIL: "Cannot find reader 'fastslide' in registry."` — the reopen path tries the svs after initial open failed | Corrupt / format-outside-openslide SVS; nothing lazyslide can do | Pre-filter unreadable slides from the table (or accept per-slide failures in dask) | 4 slides: `H-503-23-A8, H82-23-11, H-503-23-A9, H800-23-B3` — permanently skipped |
| **MPP upsampling refusal** | `FAIL: Requested mpp=0.5 is smaller than the slide mpp=0.526316. Up-sampling is not supported.` | LazySlide refuses to interpolate from coarser native MPP to finer target | Options: (a) let `tile_tissues` use native mpp for these edge cases, (b) drop the slide | 1 slide (`H-208-3-22`) |

### System-level (memory / dask)

- **Unbounded RSS creep across slides on a single worker** — glibc per-thread arenas hoard freed pages, pytorch CUDA allocator caches by default, zarr keeps block residuals. Symptom: worker RSS climbs to ~80% of whatever limit you give it (190 GiB at 256G cap, 286 GiB at 384G cap) and SLURM OOM-kills it.
  - **Mitigations** (in `scripts/extract_dask.py`):
    - `MALLOC_ARENA_MAX=2` in worker `job_script_prologue` — caps glibc arena count to prevent per-thread hoarding
    - Three-layer cleanup between models: `gc.collect()` → `torch.cuda.empty_cache()` → `ctypes.CDLL("libc.so.6").malloc_trim(0)`
    - `SLURMCluster(nanny=False)` — dask Worker is the SLURM job's main process (non-daemonic), so PyTorch DataLoader can fork children
    - `worker_extra_args=["--memory-limit", "0"]` — disable dask's 80% watermark. RSS reported by the OS is inflated vs. live allocations; SLURM cgroup is the real backstop
  - **Outcome:** 11.3h run with 3 workers, zero respawns, RSS held flat at 1-2 GiB per slide across 808 slides.

- **DataLoader daemonic-fork error** — `daemonic processes are not allowed to have children`. Happens when `nanny=True` (default): the Worker is a daemonic child of the Nanny, and stdlib forbids daemonic processes from forking, which kills `num_workers > 0` DataLoaders. Mitigation: `nanny=False` (above).

- **Partial zarr from mid-write OOM** — if a worker dies while writing a `{model}_tiles` table, the next worker's incremental check (just "does the dir exist?") would erroneously skip the partial write. Not observed after the MPP fix landed, but a size-sanity check on the table before skipping would make the pipeline more robust under failure. Not a priority while OOMs aren't happening.

### Why per-slide `try/except` doesn't rescue OOMs

The Python process itself gets `Killed` by the OOM killer, so the exception handler in `_process_slide` never runs. With `nanny=False`, the SLURM job also ends — dask `adapt()` requests a new SLURM slot to replace it, paying full model-load cost on the replacement. Pre-filter known-bad slides out of the slide table to avoid this entirely.

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
