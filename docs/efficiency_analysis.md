# ARGO-DeepMSI: Efficiency Deep-Dive

## Scale of the Problem

- **~808 slides** across 3 sites (UITH, OAUTHC/retrospective)
- **9–12 foundation models** per slide (uni2, virchow2, conch_v1.5, h-optimus-1, gigapath, hibou-b, chief, ctranspath, phikonv2, ...)
- **~10,000–20,000 tiles per slide** at 256px / 0.5 mpp
- Total: **~7,000–10,000 slide×model extraction jobs**, each doing GPU inference on thousands of tiles

Every wasted minute per slide compounds across this matrix. Below is a stage-by-stage analysis.

---

## 1. Feature Extraction — The GPU Bottleneck

### 1a. `num_workers=0` (default) wastes CPU/GPU overlap

LazySlide's `feature_extraction()` accepts `num_workers` and `batch_size` parameters:

```python
# LazySlide API signature:
feature_extraction(wsi, model=..., batch_size=32, num_workers=0, ...)
```

The current code **never passes `num_workers` or `batch_size`**:

```python
# feature_extraction.py line 411
zs.tl.feature_extraction(wsi, model=model, amp=amp, device=device)
```

With `num_workers=0`, tile loading happens in the main thread — the GPU sits idle while each batch of tiles is read from the SVS, decoded, and transformed. For large slides (~15K+ tiles), this CPU→GPU bottleneck is massive.

**Fix:** Pass `num_workers=4` (or `--cpus-per-task` / 2) and `batch_size=64` or higher:
```python
zs.tl.feature_extraction(
    wsi, model=model, amp=amp, device=device,
    num_workers=4, batch_size=64,
)
```

**Estimated impact:** 2–4× faster per slide (tile decoding and GPU inference overlap).

### 1b. Model loading happens once per slide, not once per job

In `extract_features_single_slide()`, the model loop is:

```python
for model in models_to_extract:
    zs.tl.feature_extraction(wsi, model=model, ...)  # loads model each time
```

LazySlide internally loads the model weights each call. For 9 models × 270 slides per SLURM group, that's **~2,430 model loads** per job. Most of these models are large ViTs (uni2 = ~300M params, virchow2 = ~631M params) — each load takes seconds to tens of seconds.

However — this is partly mitigated by LazySlide likely caching the model in memory within the same process. The real cost is the **first load per model per process**. Since all models are loaded sequentially for each slide, the first slide pays the full cost but subsequent slides reuse the in-memory model. **This is actually handled correctly** by the architecture — each SLURM job runs one process, and all slides in the group share the same process.

**However**, there's a subtlety: `wsi.write()` and `wsi = open_wsi(...)` in the loop probably don't clear the model cache, but if the garbage collector reclaims the previous WSI's model reference, reloading happens. Worth verifying.

### 1c. Preprocessing is NOT skipped when opening existing zarr

```python
if zarr_path.exists():
    wsi = open_wsi(str(zarr_path))  # loads existing zarr
else:
    wsi = open_wsi(str(slide_path))

if not zarr_path.exists():  # BUG: this was True before the open_wsi above!
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, ...)
```

Wait — this is actually fine. When zarr exists and we `open_wsi(zarr_path)`, the tissue/tile data is already in the zarr from the first run. The `not zarr_path.exists()` check correctly skips preprocessing. ✅

BUT: when a zarr exists with some models and we're adding new ones, we open the zarr and skip preprocessing — but the tiles should already be there from the first extraction. This is correct. ✅

### 1d. `wsi.write()` rewrites the ENTIRE zarr every time

After extracting new models:
```python
wsi.write()  # line 415
```

This writes ALL tables (including already-extracted models) back to zarr. For a slide with 15K tiles × 9 models already extracted, adding model #10 rewrites ~9 existing model tables + 1 new one. That's **~10× more I/O than necessary**.

**Fix:** Check if LazySlide supports incremental writes. The `wsidata` library may support `wsi.write(tables=[f"{model}_tiles"])` or selective table writes. If not, this is a feature request to the LazySlide team.

**Estimated impact:** 5–10× reduction in zarr write time when adding models incrementally.

---

## 2. Aggregation — The I/O Bottleneck

### 2a. Per-slide AnnData loading is massively inefficient

The `aggregate_simple_pooling()` hot path:

```python
for idx, row in tqdm(df.iterrows(), total=len(df)):
    adata = ad.read_zarr(str(adata_path))  # reads full AnnData per slide
    embedding = np.asarray(adata.X.mean(axis=0)).flatten()
```

For 808 slides with, say, plip features (512D × 15K tiles), each `ad.read_zarr()`:
- Opens a zarr store
- Reads the X matrix (~15K × 512 floats = ~30 MB)
- Reads all obs, var, uns, obsm metadata
- Constructs a full AnnData object

To compute `mean(axis=0)`, you only need the X matrix. Everything else (obs, var, obsm, uns) is wasted I/O and memory.

**Fix: Use `wsidata.agg_wsi()` instead.** LazySlide provides exactly this function:

```python
from wsidata import agg_wsi
dataset["store"] = [f"data/{s}.zarr" for s in dataset["slide_id"]]
agg_data = agg_wsi(dataset, "conch", store_col="store", agg_key="agg_slide")
```

This is purpose-built for batch aggregation across multiple zarr stores and is far more efficient. The current code reinvents this wheel less efficiently.

**Alternative fix:** If you need custom pooling, read just the X matrix directly from zarr:

```python
import zarr
z = zarr.open(str(adata_path), mode='r')
X = z['X'][:]  # or z['X']  for lazy access
embedding = X.mean(axis=0)
```

This skips the entire AnnData construction overhead.

**Estimated impact:** 3–5× faster aggregation, significantly lower peak memory.

### 2b. Aggregation loops over models sequentially, re-reading slide table each time

```python
for model in models:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        adata = ad.read_zarr(...)  # opens zarr store again for each model
```

For 9 models × 808 slides = **7,272 zarr opens**. Each zarr store contains ALL models' tables. Opening the store is overhead that's paid 9 times per slide.

**Fix:** Restructure to iterate slides in the outer loop, models in the inner loop:

```python
for idx, row in tqdm(df.iterrows(), total=len(df)):
    zarr_store = zarr.open(zarr_path, mode='r')
    for model in models:
        X = zarr_store[f'tables/{model}_tiles/X'][:]
        embeddings[model].append(X.mean(axis=0))
```

This opens each zarr store exactly once instead of N_models times.

**Estimated impact:** ~9× fewer zarr opens, ~2× overall aggregation speedup.

### 2c. `np.vstack()` on a list of embeddings is fine but could use pre-allocation

```python
embedding_matrix = np.vstack(df["embedding"].values)  # line 541
```

For 808 slides × 1024D, this is fine (~6MB). Not a bottleneck. ✅

---

## 3. Neural Encoder Aggregation — Double-Open Anti-Pattern

### 3a. Opens BOTH the original SVS and the zarr for every slide

```python
wsi = open_wsi(str(svs_path))        # opens original SVS
zarr_wsi = open_wsi(str(zarr_path))  # opens zarr copy
wsi.tables[feature_key] = zarr_wsi.tables[feature_key]  # copies features
```

For each of 808 slides, this:
1. Opens the original SVS (slow — SVS files are huge, 1–5 GB each)
2. Opens the zarr store
3. Copies the AnnData table between the two WSI objects
4. Then runs `zs.tl.feature_aggregation()`

The SVS is opened because neural encoders "need spatial context" — but LazySlide stores spatial coordinates in `obsm['spatial']` within the zarr. Check if `feature_aggregation` actually reads pixel data from the SVS, or just uses the spatial coordinates. If it only uses coordinates, opening the SVS is unnecessary.

**Fix:** Try running aggregation directly on the zarr WSI:
```python
wsi = open_wsi(str(zarr_path))  # just the zarr
zs.tl.feature_aggregation(wsi, feature_key=model, encoder=encoder, device=device)
```

**Estimated impact:** Eliminates ~808 SVS opens (seconds each) → minutes saved total.

---

## 4. Visualization — Full Re-Extraction from Scratch

### 4a. Three visualization functions re-extract features instead of loading from zarr

`visualize_tile_clusters()` (line 460–463):
```python
wsi = open_wsi(str(slide_path))
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)
zs.tl.feature_extraction(wsi, model=model, device=device)
```

`visualize_feature_heatmap()` (line 537–540): Same pattern.

`explore_slide()` calls both, then also calls `visualize_slide()` which does preprocessing again.

Each of these re-runs the **entire GPU inference pipeline** from scratch, even though the zarr already has all features. For a single `explore_slide()` call, you get:
- 1× tissue detection
- 1× tiling
- 3× full feature extraction (clusters, heatmap, overview)

That's **~3 GPU inference runs per visualization**, when the answer is already on disk.

**Fix:** All visualization functions should check for existing zarr and `open_wsi(zarr_path)`:

```python
zarr_path = slide_path.with_suffix('.zarr')
if zarr_path.exists():
    wsi = open_wsi(str(zarr_path))  # loads pre-computed everything
else:
    wsi = open_wsi(str(slide_path))
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, ...)
    zs.tl.feature_extraction(wsi, model=model, ...)
    wsi.write()
```

**Estimated impact:** Visualization goes from **minutes (GPU-bound)** to **seconds (disk read only)**.

---

## 5. SLURM Parallelization Strategy

### 5a. 3-group parallelization is conservative

The current approach splits 808 slides into 3 groups of ~270 slides. Each group runs on 1 GPU for up to 72 hours. With 9 models per slide, each taking ~2–5 min:

**Worst case:** 270 slides × 9 models × 5 min = ~20,250 min = **337 hours** (single GPU)

That **far exceeds** the 72-hour SLURM limit. Either:
- The incremental extraction helps (most models already extracted), or
- Some models are fast enough that 5 min/slide is pessimistic

LazySlide's docs recommend **Dask + dask-jobqueue for SLURM** parallelization:

```python
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(queue="gpu", cores=8, gres="gpu:1", ...)
client = Client(cluster)
cluster.adapt(minimum=1, maximum=10)
futures = [client.submit(process_slide, s, resources={"GPU": 1}) for s in slides]
```

**Fix:** Consider more granular parallelization — e.g., 10–20 groups instead of 3, or use dask-jobqueue for elastic scaling. Alternatively, split by model rather than (or in addition to) by slide: run all 808 slides for model A, then all for model B, etc. This avoids model loading overhead.

### 5b. No checkpointing within extraction

If a SLURM job dies at slide 200/270, all 200 completed slides are fine (zarr is written per-slide), but the job restarts from slide 1. The incremental check skips already-extracted slides, so this is mostly OK — the overhead is just re-scanning 200 zarr directories.

But there's no **progress file** to skip the scan. For 270 slides × 9 models, checking each zarr's tables/ directory means 2,430 directory listings at job restart.

**Fix:** Write a simple progress CSV after each slide:
```python
with open("progress.csv", "a") as f:
    f.write(f"{slide_path},{','.join(models_extracted)}\n")
```

---

## 6. `argo run` Pipeline — Serial When It Could Overlap

### 6a. Extract → Aggregate → Train runs per-model sequentially

```python
for model in models:
    extract_features_batch(slide_table, models=[model], ...)
    aggregate_features(slide_table, models=[model], method="mean")
    compare_classifiers(X, y)
```

This means model B's extraction doesn't start until model A's training is done. Since extraction is GPU-bound, aggregation is I/O-bound, and training is CPU-bound, these could overlap:

- Extract model B while aggregating model A
- Aggregate model B while training model A

**Fix:** For the full pipeline, use concurrent.futures or similar to overlap extraction with aggregation/training from previous models.

---

## Summary: Impact Priority Matrix

| Issue | Stage | Est. Speedup | Effort |
|-------|-------|-------------|--------|
| Pass `num_workers=4, batch_size=64` to `feature_extraction()` | Extraction | **2–4×** per slide | 1 line |
| Use `wsidata.agg_wsi()` for simple pooling | Aggregation | **3–5×** | Medium refactor |
| Read zarr X matrix directly (skip AnnData overhead) | Aggregation | **2–3×** | Small refactor |
| Outer loop = slides, inner loop = models in aggregation | Aggregation | **~2×** (fewer zarr opens) | Small refactor |
| Load from zarr in visualization functions | Visualization | **Minutes → seconds** | Medium refactor |
| Skip SVS open in neural encoder aggregation | Neural Agg | **Significant** (TBD) | Test needed |
| Investigate `wsi.write()` selective table writes | Extraction | **5–10× write I/O** | Depends on LazySlide |
| Increase SLURM parallelism (10–20 groups) | Extraction | **Linear scaling** | Config change |
| Add progress tracking file | Extraction | Reliability, not speed | Small |
| Overlap extract/aggregate/train stages | Full pipeline | **~30% walltime** | Medium |

### Top 3 Quick Wins (< 1 hour to implement)

1. **`num_workers=4, batch_size=64`** in feature extraction — just pass the parameters
2. **Load from zarr in visualization** — check for `.zarr` before calling `find_tissues`/`feature_extraction`
3. **Read zarr X directly** in aggregation — `zarr.open(path)['X'][:]` instead of `ad.read_zarr()`

### Top 3 Structural Improvements (1–3 hours)

1. **Use `wsidata.agg_wsi()`** for batch slide-level aggregation
2. **Restructure aggregation loops** (slide-outer, model-inner)
3. **Selective zarr writes** when adding models incrementally

---

## 7. Upcoming Phases — Why These Fixes Matter Even More

Based on the current CLAUDE.md roadmap, three new model categories are planned:

### 7a. QC Models (grandqc-artifact, grandqc-tissue, focus, etc.)

**Current plan:** Run QC *before* feature extraction to filter bad slides.

**Efficiency implication:** If QC runs through the same `extract_features_single_slide()` path, it will:
- Load each slide with `open_wsi()` for QC
- Write QC results to zarr
- Then load the slide *again* for feature extraction

**Better approach:** Run QC as the first model in the same `models_to_extract` list. The current architecture already supports this — just add QC model names alongside patch models in `extract.sh`. The slide is preprocessed once, QC and feature extraction happen in the same open/write cycle. **But** the QC-then-filter workflow described in CLAUDE.md (`extract QC → filter slide_table → extract features`) requires two separate passes, which doubles SVS open overhead.

**Recommendation:** Add a `--qc-filter` flag that runs QC models, filters inline, and only proceeds to feature extraction for passing slides — all in one pass:

```python
# In extract_features_single_slide:
if qc_models:
    for qc_model in qc_models:
        zs.tl.feature_extraction(wsi, model=qc_model, ...)
    if not passes_qc(wsi, qc_models):
        logger.info(f"Skipping {slide_path.name} — failed QC")
        wsi.write()  # save QC results for records
        return zarr_path
# Proceed with feature models...
```

### 7b. Direct Slide-Level Extractors (gigapath-slide-encoder, chief-slide-encoder)

These bypass patch extraction entirely. **But** the current `extract_features_single_slide()` always runs `find_tissues → tile_tissues` before any model. For slide-level models, tiling is unnecessary overhead.

**Fix:** Check if the model is slide-level before tiling:
```python
if not zarr_path.exists():
    zs.pp.find_tissues(wsi)
    if any(PATCH_MODELS.get(m) for m in models_to_extract):
        zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)
```

### 7c. Scale Projection

With the full model catalog from CLAUDE.md:
- **9 patch models** (current)
- **~6 QC models** (planned)  
- **2–3 slide-level models** (planned)
- **2 VL models** (planned)

That's potentially **~20 models per slide × 808 slides = 16,160 model-slide operations**. The `num_workers=0` bottleneck alone could waste **hundreds of GPU-hours** across this matrix.

---

## 8. Minor Efficiency Notes

### 8a. `verify_slides_exist()` does a full recursive filesystem walk

```python
for root, _, files in os.walk(base_dir):
    for file in files:
        found_files[file] = str(Path(root) / file)
```

For large data directories with thousands of SVS files (each 1–5 GB), this builds a dict of every filename. It's fine for ~800 slides but could be slow if the data directory contains other large files. Not a major concern, just worth noting.

### 8b. `iterrows()` used in aggregation hot path

```python
for idx, row in tqdm(df.iterrows(), total=len(df)):
```

`iterrows()` is notoriously slow for large DataFrames due to per-row Series construction. For 808 rows this is negligible, but for future scaling, `df.itertuples()` is ~10× faster.

### 8c. CLAUDE.md shows `zs.WSI()` but code uses `wsidata.open_wsi()`

The LazySlide usage example in CLAUDE.md:
```python
wsi = zs.WSI("path/to/slide.svs")
```

But the actual code correctly uses:
```python
from wsidata import open_wsi
wsi = open_wsi(str(slide_path))
```

The CLAUDE.md example should be updated to match the real API to avoid confusion for anyone (or Claude Code) working from it.
