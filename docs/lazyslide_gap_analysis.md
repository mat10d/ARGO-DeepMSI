# LazySlide Ecosystem Usage Gap Analysis
## ARGO-DeepMSI vs. What LazySlide Actually Provides

---

## What You USE from LazySlide

| Feature | API | Status |
|---------|-----|--------|
| Slide I/O | `wsidata.open_wsi()` | ✅ Used |
| Tissue detection | `zs.pp.find_tissues()` | ✅ Used |
| Tiling | `zs.pp.tile_tissues()` | ✅ Used |
| Feature extraction | `zs.tl.feature_extraction()` | ⚠️ Used, but without `num_workers`/`batch_size` |
| Neural aggregation | `zs.tl.feature_aggregation()` | ✅ Used (for PRISM/TITAN path) |
| Visualization | `zs.pl.wsi()`, `zs.pl.tiles()`, `zs.pl.tissues()` | ✅ Used |
| Zarr persistence | `wsi.write()` | ✅ Used |

That's it. That's 7 out of ~25+ capabilities.

---

## What You REINVENT Instead of Using LazySlide

### 1. Batch slide-level aggregation — `wsidata.agg_wsi()`

**What LazySlide provides:**
```python
from wsidata import agg_wsi

# One function call: reads all zarrs, aggregates, returns AnnData
dataset["store"] = [f"data/{s}.zarr" for s in slide_ids]
agg_data = agg_wsi(dataset, "conch", store_col="store", agg_key="agg_slide")
# agg_data is an AnnData: (n_slides × n_features), with obs = slide metadata
```

**What you wrote instead (100+ lines in feature_extraction.py):**
```python
for model in models:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        adata = ad.read_zarr(str(adata_path))  # full AnnData per slide
        embedding = np.asarray(adata.X.mean(axis=0)).flatten()
        embeddings.append({...})
    df_result = pd.DataFrame(embeddings)
    np.save(output_dir / "embeddings.npy", np.vstack(...))
    metadata_df.to_csv(output_dir / "metadata.csv")
```

**Impact:** Your version is slower (full AnnData construction per slide), less memory-efficient, and stores results as separate npy+csv files instead of a single AnnData that's compatible with the rest of the scverse ecosystem. The `agg_wsi()` output is directly usable with scanpy for downstream analysis.

---

### 2. SLURM parallelization — LazySlide Nextflow / Dask

**What LazySlide provides:**

*Option A: Nextflow pipeline (ready-made)*
```bash
# rendeirolab/lazyslide-nextflow — a complete pipeline
nextflow run rendeirolab/lazyslide-nextflow \
    --input your_data_dir/input.csv \
    --tile_px 256 \
    --models 'uni2,virchow2,conch_v1.5'
```

*Option B: Dask + dask-jobqueue (recommended in LazySlide tutorial)*
```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(
    queue="gpu", cores=8, processes=1, memory="20 GB",
    job_extra_directives=["--gres=gpu:1", "--time=2:00:00"],
    worker_extra_args=["--resources GPU=1"],
)
client = Client(cluster)
cluster.adapt(minimum=1, maximum=10)  # auto-scale!

futures = [
    client.submit(process_slide, slide, resources={"GPU": 1})
    for slide in slides
]
```

**What you wrote instead (scripts/extract.sh, 150+ lines):**
- Manual SLURM array job splitting (calculate group boundaries, temp CSV files)
- Hardcoded paths (`/lab/barcheese01/mdiberna/...`)
- No auto-scaling, no fault tolerance beyond zarr-level incremental checks

---

### 3. Training data format — scverse AnnData vs. npy+csv

**What LazySlide's ecosystem expects:**

After `agg_wsi()`, you get an AnnData object:
```python
agg_data  # AnnData: (808 slides × 1024 features)
agg_data.obs  # slide_id, patient_id, site, MSI_status, ...
agg_data.X    # embedding matrix

# Then use scanpy directly:
import scanpy as sc
sc.pp.neighbors(agg_data)
sc.tl.umap(agg_data)
sc.pl.umap(agg_data, color='MSI_status')
```

**What you do instead:**
```python
embeddings = np.load("embeddings.npy")      # raw numpy
metadata = pd.read_csv("metadata.csv")       # separate file
clinical = pd.read_csv("clinical_table.csv") # another separate file
merged = metadata.merge(clinical, ...)        # manual merge (with index bug)
X = embeddings[matched_indices]               # manual alignment
```

You lose scverse interoperability entirely. UMAP, leiden clustering, differential analysis — all of scanpy's toolkit — would work out of the box on an AnnData, but you can't use any of it because your embeddings live in a disconnected npy file.

---

### 4. Quality control — built-in QC models

**What LazySlide provides:**
```python
# Built-in QC during preprocessing
zs.tl.feature_extraction(wsi, model="grandqc-artifact")
zs.tl.feature_extraction(wsi, model="grandqc-tissue")
zs.tl.feature_extraction(wsi, model="focus")

# QC scores are stored in the zarr alongside features
# Filter before expensive feature extraction
```

**What you have:** Planned in CLAUDE.md, not implemented. No slide quality filtering at all — you extract features from every slide regardless of quality, then discover bad slides during training when performance is unexpectedly poor.

---

### 5. Spatial analysis — completely unused

**What LazySlide provides:**
```python
# Tile-level spatial domain analysis
zs.tl.spatial_domain(wsi, feature_key="uni2", resolution=0.2)

# Spatial feature smoothing (UTAG-style)
zs.pp.tile_graph(wsi)
zs.tl.spatial_features(wsi, feature_key="uni2")

# This could reveal spatial patterns in MSI vs MSS tissue
```

**What you have:** Nothing. Your `analyze_tiles()` function in feature_extraction.py does basic scanpy neighbors/UMAP/leiden, but doesn't use the spatial coordinates at all. For MSI prediction, spatial patterns (e.g., tumor-stroma interface, immune infiltrate distribution) could be informative.

---

### 6. Vision-language queries — completely unused

**What LazySlide provides:**
```python
# Zero-shot tissue classification
embed = zs.tl.text_embedding(["tumor", "stroma", "necrosis", "mucin"], "conch")
zs.tl.text_image_similarity(wsi, embed, "conch")

# Score tiles by pathological terms
scores = zs.metrics.topk_score(adata, k=100)
# → "This slide has high mucin content" without any labels
```

**What you have:** You extract PLIP and CONCH features but only use them as generic embeddings for classification. You never leverage their vision-language capabilities for interpretability, QC, or feature enrichment.

---

### 7. The `fetch` and `iter` accessors

**What LazySlide provides:**
```python
# Clean AnnData retrieval
adata = wsi.fetch.features_anndata("uni2")

# Iterate over tissues/tiles
for tissue in wsi.iter.tissues():
    for tile in wsi.iter.tiles(tissue):
        ...

# PyTorch dataloader creation
dataset = wsi.dataset.tiles(feature_key="uni2")
loader = DataLoader(dataset, batch_size=32)
```

**What you have:** Direct dictionary-style access (`wsi["uni2_tiles"]`), which works but misses the structured accessor pattern that handles edge cases.

---

## Summary: You're Using LazySlide as OpenSlide++

```
What LazySlide IS:
┌─────────────────────────────────────────────────────────┐
│  WSI I/O → Preprocessing → Feature Extraction →         │
│  Aggregation → Spatial Analysis → VL Queries →           │
│  QC → Visualization → scverse Integration →              │
│  Nextflow/Dask HPC → AnnData ecosystem                  │
└─────────────────────────────────────────────────────────┘

What you use:
┌─────────────────────────────────────────────────────────┐
│  WSI I/O → Preprocessing → Feature Extraction            │
│  ↓                                                       │
│  (exit LazySlide, enter custom numpy/pandas/sklearn)     │
└─────────────────────────────────────────────────────────┘
```

You're treating LazySlide as "the thing that reads SVS files and runs models on tiles" — about 30% of what it does. Everything downstream (aggregation, training data assembly, parallelization, visualization) is hand-rolled in ways that are:

1. **Less efficient** (manual zarr reads vs. `agg_wsi()`)
2. **Less interoperable** (npy+csv vs. AnnData)
3. **Less maintainable** (custom SLURM scripts vs. Nextflow/Dask)
4. **Missing features** (no QC, no spatial analysis, no VL queries)

---

## What "Full Use" Would Look Like

### Phase 1: Core fixes (high impact, low effort)
- Use `num_workers=4, batch_size=64` in feature extraction
- Replace manual aggregation loop with `wsidata.agg_wsi()`
- Store results as AnnData (not npy+csv) for scverse compatibility
- Load from zarr in visualization functions

### Phase 2: Leverage the ecosystem (medium effort)
- Use Dask + dask-jobqueue for SLURM instead of bash scripts
- Add QC models before feature extraction
- Use `wsi.fetch.features_anndata()` consistently
- Use scanpy on the AnnData output (UMAP, clustering, DE-like analysis)

### Phase 3: Advanced capabilities (higher effort, high value)
- Spatial domain analysis — morphological patterns in MSI vs MSS
- Vision-language queries — zero-shot tissue characterization
- Text-image similarity scoring — interpretable features for MSI
- Multi-omics integration if transcriptomics data becomes available
