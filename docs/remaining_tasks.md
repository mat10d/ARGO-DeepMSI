# Execution Runbook

Two-phase approach: lean run now on 3 GPUs, full sweep later on MSK cluster.

---

## Phase 1: Baseline (3 GPU cluster, ~1 day)

3 top-performing models, end-to-end. Gets you AUROC numbers, proves the
pipeline, and creates zarrs that Phase 2 builds on incrementally.

### 1a. Convert non-pyramidal slides

```bash
sbatch scripts/pyramidal.sh results/data/slide_table.csv
```

Writes `results/data/slide_table_pyramidal.csv`. Idempotent.

### 1b. Extract 3 foundation models

```bash
python scripts/extract_dask.py \
    --slide-table results/data/slide_table_pyramidal.csv \
    --models uni2 virchow2 conch_v1.5 \
    --max-workers 3 \
    --memory "256 GB"
```

~3× faster than 11 models. Each slide gets a zarr with 3 feature tables.

### 1c. Aggregate

```bash
sbatch scripts/aggregate.sh
```

Auto-discovers which models have been extracted. Produces
`results/embeddings/{model}_mean/` with npy + csv + h5ad.

### 1d. Train

```bash
sbatch scripts/train.sh
```

Auto-discovers which embeddings exist. Produces
`results/models/{embedding}/classifier_comparison.csv`.

### 1e. Iterate (while extraction for Phase 2 is queued)

With baseline numbers in hand:
- Try different aggregation methods (max, median) on the same zarrs
- Try neural aggregators: PRISM (on virchow2), TITAN (on conch_v1.5)
- Add classifiers (XGBoost, attention MIL)
- Fix QC: refactor `filter_slides_by_qc` to consume `zs.seg.artifact()` polygon output
- Scanpy UMAP on the h5ad embeddings colored by MSI status + site
- Spatial analysis on a few representative slides

---

## Phase 2: Full sweep (MSK cluster, 10-20 GPUs)

All foundation models. The 3 models from Phase 1 are already in every
zarr — extraction skips them and only runs the new ones.

### 2a. Extract remaining models

```bash
python scripts/extract_dask.py \
    --slide-table results/data/slide_table_pyramidal.csv \
    --models uni2 virchow2 conch_v1.5 \
              h-optimus-1 gigapath hibou-b musk \
              chief ctranspath phikonv2 plip \
    --max-workers 15 \
    --partition <msk-gpu-partition> \
    --memory "256 GB"
```

Incremental: only the 8 new models run. With 15 workers, ~1 day.

### 2b. Expand further

Add any new models by appending to `--models`. The zarr/aggregation/training
pipeline is model-agnostic — no code changes needed.

Candidates beyond the current 11:
- `h-optimus-0`, `hibou-l` (larger variants)
- `gpfm`, `path_orchestra`, `histoplus`, `rosie` (newer models)
- `nulite`, `pathprofiler` (specialized)
- Any future LazySlide-registered model

### 2c. Aggregate + Train

Same commands as Phase 1 — the scripts auto-discover whatever models
are present.

```bash
sbatch scripts/aggregate.sh
sbatch scripts/train.sh
```

---

## QC (deferred — upstream LazySlide bug)

`zs.tl.feature_extraction(wsi, model="grandqc-artifact")` is broken in
the current LazySlide version (dispatcher passes `model_path` to a
constructor that doesn't accept it).

**Fix path:** refactor `filter_slides_by_qc` to use `zs.seg.artifact()`
which produces polygon shapes instead of per-tile AnnData tables. The
QC score becomes "fraction of tile area covered by artifact polygons"
rather than a direct per-tile scalar.

Do this during Phase 1e iteration — it doesn't block baseline results.

---

## Architecture guarantee

The pipeline is plug-and-play because:
1. **Zarr is the contract** — every model writes `{slide}.zarr/tables/{model}_tiles`
2. **Extraction is incremental** — existing models are skipped
3. **Aggregation auto-discovers** — reads whatever models exist in zarrs
4. **Training auto-discovers** — iterates over `results/embeddings/*/`
5. **`extract_dask.py --models`** accepts any list — no code changes to add models
