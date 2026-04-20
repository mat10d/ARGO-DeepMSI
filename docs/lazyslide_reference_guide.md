# LazySlide Practical Reference for ARGO-DeepMSI
## Verified API patterns from docs, tutorials, and Nature Methods paper
## Version: LazySlide >=0.9.0 / wsidata >=0.7.0

> **Purpose:** This file is a concrete, copy-pasteable reference for refactoring
> ARGO-DeepMSI to fully leverage the LazySlide ecosystem. Every code pattern below
> has been verified against the official LazySlide readthedocs, the "Integration with
> slide-level labels" tutorial, and the LazySlide Nature Methods paper (2026).
> Where LazySlide does not cover a need, we note it and provide our own solution.

---

## 1. Core Imports

```python
from wsidata import open_wsi, agg_wsi
import lazyslide as zs
import anndata as ad
import scanpy as sc
```

- `open_wsi` is from `wsidata`, NOT from `lazyslide` directly
- `agg_wsi` is from `wsidata` -- this is the batch aggregation function
- LazySlide convention: `import lazyslide as zs`

---

## 2. Opening a Slide

```python
# Basic open (creates WSIData object)
wsi = open_wsi("path/to/slide.svs")

# For batch processing: skip thumbnail generation for speed
wsi = open_wsi("path/to/slide.svs", attach_thumbnail=False)

# Specify where zarr is stored (default: next to the SVS)
wsi = open_wsi("path/to/slide.svs", store="path/to/output_dir")
```

**Key behavior:** When you `open_wsi("slide.svs")` and `slide.zarr` exists in the same
directory, LazySlide automatically picks up the existing zarr with all prior results
(tissues, tiles, features). You do NOT need to open the zarr separately.

---

## 3. Preprocessing

```python
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

# Optional: filter background-heavy tiles
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5, background_fraction=0.5)

# Optional: build spatial graph for tiles (needed for spatial analysis later)
zs.pp.tile_graph(wsi)
```

Preprocessing results are stored in the WSIData object in memory.
Call `wsi.write()` to persist to zarr. On subsequent opens, preprocessing is
already in the zarr and does NOT need to be re-run.

---

## 4. Feature Extraction

```python
# CRITICAL: always pass num_workers and batch_size for GPU efficiency
zs.tl.feature_extraction(
    wsi,
    model="uni2",
    amp=True,
    device="cuda",
    num_workers=4,    # >0 enables parallel tile loading (CPU/GPU overlap)
    batch_size=64,    # default is 32, increase for faster throughput
    pbar=True,        # show progress bar (set False in batch scripts)
)
```

### Full API signature (verified from readthedocs):
```
feature_extraction(
    wsi,
    model=None,           # model name string
    model_path=None,      # path to custom model file
    model_name=None,      # custom name for key_added
    jit=False,
    token=None,           # HuggingFace token for gated models
    load_kws=None,
    transform=None,
    device=None,           # "cuda", "cpu", or None (auto-detect)
    amp=None,
    autocast_dtype=None,
    tile_key="tiles",
    key_added=None,
    batch_size=32,         # inference batch size
    num_workers=0,         # dataloader workers (SET THIS >0!)
    pbar=None,
    return_features=False,
)
```

Features are stored in `wsi.tables["{model}_tiles"]` as AnnData.
Shape: (n_tiles, n_features) -- e.g., (15000, 1024) for UNI2.

### Multi-model extraction pattern:
```python
wsi = open_wsi(slide_path, attach_thumbnail=False)
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

for model in ["uni2", "virchow2", "conch_v1.5"]:
    zs.tl.feature_extraction(
        wsi, model=model, amp=True, device="cuda",
        num_workers=4, batch_size=64, pbar=False,
    )

wsi.write()  # single write saves all models
```

---

## 5. Feature Aggregation -- Neural Encoders (per-slide)

```python
# PRISM requires virchow or virchow2 features already extracted
zs.tl.feature_aggregation(wsi, feature_key="virchow2", encoder="prism", device="cuda")

# TITAN requires conch_v1.5 features
zs.tl.feature_aggregation(wsi, feature_key="conch_v1.5", encoder="titan", device="cuda")

# Default mean aggregation (stores in varm['agg_slide'])
zs.tl.feature_aggregation(wsi, feature_key="uni2")
```

---

## 6. Batch Slide-Level Aggregation -- `agg_wsi()` (CRITICAL)

This replaces the entire manual aggregation loop.

```python
from wsidata import agg_wsi
from pathlib import Path
import pandas as pd

slide_table = pd.read_csv("results/data/slide_table.csv")
slide_table["store"] = slide_table["FILENAME"].apply(
    lambda f: str(Path(f).with_suffix(".zarr"))
)

# One-liner: reads all zarrs, aggregates, returns AnnData
agg_data = agg_wsi(
    slide_table,
    feature_key="uni2",
    store_col="store",
    agg_key="agg_slide",
)
# agg_data.X  -> (n_slides, n_features) embedding matrix
# agg_data.obs -> slide metadata from slide_table
```

### Using the AnnData output with scanpy:
```python
clinical = pd.read_csv("results/data/clinical_table.csv")
agg_data.obs = agg_data.obs.merge(
    clinical[["PATIENT", "isMSIH"]],
    left_on="PATIENT", right_on="PATIENT", how="left"
)

sc.pp.neighbors(agg_data)
sc.tl.umap(agg_data)
sc.pl.umap(agg_data, color="isMSIH", save="_msi_status.png")
sc.tl.leiden(agg_data, resolution=0.5)
```

### If agg_wsi does not support your pooling method:
```python
import zarr
import numpy as np

embeddings = []
for _, row in slide_table.iterrows():
    zarr_path = Path(row["FILENAME"]).with_suffix(".zarr")
    store = zarr.open(str(zarr_path / "tables" / "uni2_tiles"), mode="r")
    X = store["X"][:]  # just the feature matrix, no AnnData overhead
    embeddings.append(X.mean(axis=0))  # or max, median, etc.

embedding_matrix = np.vstack(embeddings)
```

---

## 7. SLURM Parallelization with Dask

### LazySlide-recommended pattern (from their multi-slide tutorial):

```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm
import pandas as pd

def process_slide(slide_path, models, tile_px=256, mpp=0.5):
    from pathlib import Path
    from wsidata import open_wsi
    import lazyslide as zs

    slide_path = Path(slide_path)
    wsi = open_wsi(str(slide_path), attach_thumbnail=False)

    # Check existing models in zarr
    existing = set()
    if hasattr(wsi, "tables"):
        existing = {k.replace("_tiles", "") for k in wsi.tables if k.endswith("_tiles")}

    to_extract = [m for m in models if m not in existing]
    if not to_extract:
        return {"slide": slide_path.name, "status": "skipped"}

    if "tiles" not in getattr(wsi, "shapes", {}):
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp)

    for model in to_extract:
        zs.tl.feature_extraction(
            wsi, model=model, amp=True, device="cuda",
            num_workers=4, batch_size=64, pbar=False,
        )

    wsi.write()
    return {"slide": slide_path.name, "status": "success", "new_models": to_extract}

# ADAPT THESE TO YOUR CLUSTER
cluster = SLURMCluster(
    queue="nvidia-2080ti-20",
    cores=8,
    processes=1,
    memory="64 GB",
    job_extra_directives=["--gres=gpu:1", "--time=12:00:00"],
    worker_extra_args=["--resources GPU=1"],
    log_directory="./dask-logs",
    job_script_prologue=[
        "source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh",
        "conda activate argo",
        "export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache",
    ],
)

client = Client(cluster)
cluster.adapt(minimum=1, maximum=10)

slide_table = pd.read_csv("results/data/slide_table.csv")
models = ["uni2", "virchow2", "conch_v1.5", "h-optimus-1",
          "gigapath", "hibou-b", "chief", "ctranspath", "phikonv2"]

futures = [
    client.submit(process_slide, row["FILENAME"], models, resources={"GPU": 1})
    for _, row in slide_table.iterrows()
]

for future in tqdm(as_completed(futures), total=len(futures)):
    result = future.result()
    if result["status"] != "success":
        print(f"  {result['slide']}: {result['status']}")

client.close()
cluster.close()
```

### Key parameters to adapt for your cluster:
- `queue` -> your GPU partition name
- `--gres=gpu:1` -> your GPU resource spec
- `job_script_prologue` -> your conda/env setup commands
- `cluster.adapt(minimum=1, maximum=10)` -> max concurrent GPU workers

---

## 8. Quality Control Models

```python
zs.tl.feature_extraction(wsi, model="grandqc-artifact", amp=True, device="cuda")
zs.tl.feature_extraction(wsi, model="grandqc-tissue", amp=True, device="cuda")

# QC results stored in wsi.tables["grandqc-artifact_tiles"]
qc_adata = wsi["grandqc-artifact_tiles"]
# qc_adata.X contains per-tile artifact scores
```

Run QC as the FIRST models. Check scores before expensive foundation models.

---

## 9. Vision-Language Queries

```python
msi_terms = [
    "tumor infiltrating lymphocytes", "mucin", "Crohn-like reaction",
    "poorly differentiated", "medullary pattern", "dirty necrosis",
    "peritumoral lymphocytes", "normal colonic mucosa", "stroma", "necrosis",
]

# Requires CONCH or PLIP features already extracted
text_embed = zs.tl.text_embedding(msi_terms, "conch")
zs.tl.text_image_similarity(wsi, text_embed, "conch")

# Results in: wsi.tables["conch_tiles_text_similarity"]
# Score slides by pathological terms:
scores = zs.metrics.topk_score(wsi["conch_tiles_text_similarity"], k=100)
```

---

## 10. Spatial Analysis

```python
zs.pp.tile_graph(wsi)
zs.tl.spatial_features(wsi, feature_key="uni2")
zs.tl.spatial_domain(wsi, feature_key="uni2", resolution=0.2)
zs.pl.tiles(wsi, feature_key="uni2", color="leiden", alpha=0.5)
```

---

## 11. Visualization from Zarr (NEVER re-extract)

```python
from pathlib import Path

def visualize_slide(slide_path, model="uni2"):
    slide_path = Path(slide_path)
    # Auto-loads existing zarr with all pre-computed features
    wsi = open_wsi(str(slide_path))

    feature_key = f"{model}_tiles"
    if feature_key not in wsi.tables:
        raise ValueError(f"No {model} features found. Run extraction first.")

    # All of these use pre-computed data -- no GPU needed
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    zs.pl.wsi(wsi, ax=axes[0])
    zs.pl.tissues(wsi, ax=axes[1])
    zs.pl.tiles(wsi, feature_key=model, color=["0"], ax=axes[2])
    return fig
```

---

## 12. Gotchas and Pitfalls

### API changes between versions:
- Before v0.6.0: `n_workers` param; After: `num_workers` (harmonized)
- Old: `zs.WSI("path")` -> Current: `from wsidata import open_wsi`
- Old: `wsi.sdata[key]` -> Current: `wsi.tables[key]`

### Known limitations:
- `wsi.write()` rewrites ALL tables (no selective write)
- `feature_extraction` processes one model per call (loop models yourself)
- `num_workers > 0` may cause segfaults with some OpenSlide builds (try 0 if so)
- Gated models need `HF_TOKEN` in env (set in .env or SLURM prologue)

### Complete pipeline flow:
```
1. INGEST    -> clinical_table.csv + slide_table.csv (unchanged)
2. EXTRACT   -> Dask+SLURM, num_workers=4, batch_size=64
3. AGGREGATE -> agg_wsi() returns AnnData
4. ANALYZE   -> scanpy UMAP/leiden on slide embeddings
5. TRAIN     -> X=agg_data.X, y from obs, GroupKFold on PATIENT
6. VISUALIZE -> load from zarr, never re-extract
```

---

## Sources

- LazySlide docs: https://lazyslide.readthedocs.io/en/latest/
- wsidata docs: https://wsidata.readthedocs.io/en/latest/
- Multi-slide tutorial: https://lazyslide.readthedocs.io/en/latest/tutorials/multiple_slides.html
- Feature extraction API: https://lazyslide.readthedocs.io/en/v0.10.0/api/_autogen/lazyslide.tl.feature_extraction.html
- Nature Methods paper: https://doi.org/10.1038/s41592-026-03044-7
- dask-jobqueue: https://jobqueue.dask.org/en/latest/
