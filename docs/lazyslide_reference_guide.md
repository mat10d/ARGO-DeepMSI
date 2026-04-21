# LazySlide API Reference for ARGO-DeepMSI

> Verified patterns from LazySlide readthedocs, tutorials, and Nature Methods paper (2026).
> Pin: `lazyslide>=0.9.0`, `wsidata>=0.7.0`. Last updated against v0.10.0.

---

## Core Imports

```python
from wsidata import open_wsi, agg_wsi   # NOT from lazyslide
import lazyslide as zs
import anndata as ad
import scanpy as sc
```

---

## Opening Slides

```python
wsi = open_wsi("slide.svs")                          # basic
wsi = open_wsi("slide.svs", attach_thumbnail=False)   # batch (faster)
wsi = open_wsi("slide.svs", store="output_dir")       # custom zarr location
```

Auto-detection: if `slide.zarr` exists next to `slide.svs`, `open_wsi` loads it
with all prior tissues/tiles/features. No need to open the zarr separately.

**Gotcha:** Opening a `.zarr` directly (`open_wsi("slide.zarr")`) can KeyError if
the recorded reader (e.g. `fastslide`) isn't installed. Open the SVS with
`store=svs.parent` instead.

---

## Preprocessing

```python
zs.pp.find_tissues(wsi)                                    # tissue detection
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)              # tile grid
zs.pp.tile_tissues(wsi, 256, mpp=0.5, background_fraction=0.5)  # filter bg tiles
zs.pp.tile_graph(wsi)                                      # spatial adjacency (for spatial analysis)
wsi.write()                                                 # persist to zarr
```

---

## Feature Extraction

```python
zs.tl.feature_extraction(
    wsi,
    model="uni2",         # any model name from LazySlide registry
    amp=True,             # mixed precision
    device="cuda",
    num_workers=4,        # CPU/GPU overlap (default 0 — always override)
    batch_size=64,        # default 32
    pbar=False,           # quiet for batch scripts
)
```

Multi-model, single preprocess:
```python
wsi = open_wsi(slide_path, attach_thumbnail=False)
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
for model in ["uni2", "virchow2", "conch_v1.5"]:
    zs.tl.feature_extraction(wsi, model=model, amp=True, device="cuda",
                              num_workers=4, batch_size=64, pbar=False)
wsi.write()  # single write saves all
```

Results in `wsi.tables["{model}_tiles"]` as AnnData `(n_tiles, n_features)`.

---

## Feature Aggregation

### Per-slide neural encoders
```python
zs.tl.feature_aggregation(wsi, feature_key="virchow2", encoder="prism", device="cuda")
zs.tl.feature_aggregation(wsi, feature_key="conch_v1.5", encoder="titan", device="cuda")
zs.tl.feature_aggregation(wsi, feature_key="uni2")  # default mean, stores in varm["agg_slide"]
```

### Batch aggregation with `agg_wsi()` (NOT YET ADOPTED)

LazySlide provides a one-liner for batch slide-level aggregation:
```python
from wsidata import agg_wsi

slide_table["store"] = slide_table["FILENAME"].apply(lambda f: str(Path(f).with_suffix(".zarr")))
agg_data = agg_wsi(slide_table, "uni2", store_col="store", agg_key="agg_slide")
# Returns AnnData: (n_slides, n_features) with obs from slide_table
```

We currently use a custom slide-outer/model-inner loop with direct zarr X reads
(fast, tested). `agg_wsi()` would simplify this further and return a native AnnData.
Blocked on verifying it works with our zarr layout — see remaining work below.

### Direct zarr reads (current fallback)
```python
import zarr
store = zarr.open(str(zarr_path / "tables" / "uni2_tiles"), mode="r")
X = store["X"][:]          # just the feature matrix, no AnnData overhead
embedding = X.mean(axis=0) # or max, median, sum
```

---

## Dask + SLURM Parallelization

Already implemented in `scripts/extract_dask.py`. The pattern from LazySlide's tutorial:

```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed

cluster = SLURMCluster(
    queue="nvidia-2080ti-20", cores=8, processes=1, memory="64 GB",
    job_extra_directives=["--gres=gpu:1", "--time=12:00:00"],
    worker_extra_args=["--resources GPU=1"],
    log_directory="./dask-logs",
    job_script_prologue=["source .../conda.sh", "conda activate argo", "export HF_HOME=..."],
)
client = Client(cluster)
cluster.adapt(minimum=1, maximum=10)  # auto-scale

futures = [client.submit(process_slide, path, models, resources={"GPU": 1})
           for path in slide_paths]
for f in tqdm(as_completed(futures), total=len(futures)):
    print(f.result())
```

---

## Quality Control

Already implemented via `argo qc` CLI and `filter_slides_by_qc()`.

```python
zs.tl.feature_extraction(wsi, model="grandqc-artifact", amp=True, device="cuda")
# QC scores in wsi.tables["grandqc-artifact_tiles"].X
```

---

## Vision-Language Queries (NOT YET IMPLEMENTED)

```python
terms = ["tumor infiltrating lymphocytes", "mucin", "Crohn-like reaction",
         "poorly differentiated", "medullary pattern", "dirty necrosis",
         "peritumoral lymphocytes", "normal colonic mucosa", "stroma", "necrosis"]

text_embed = zs.tl.text_embedding(terms, "conch")
zs.tl.text_image_similarity(wsi, text_embed, "conch")
# Results: wsi.tables["conch_tiles_text_similarity"]

scores = zs.metrics.topk_score(wsi["conch_tiles_text_similarity"], k=100)
```

Potential value for MSI: interpretable tissue characterization without labeled training data.

---

## Spatial Analysis (NOT YET IMPLEMENTED)

```python
zs.pp.tile_graph(wsi)
zs.tl.spatial_features(wsi, feature_key="uni2")
zs.tl.spatial_domain(wsi, feature_key="uni2", resolution=0.2)
zs.pl.tiles(wsi, feature_key="uni2", color="leiden", alpha=0.5)
```

Potential value: tumor-stroma interface patterns, immune infiltrate distribution.

---

## Gotchas

- **`num_workers > 0`** can segfault with some OpenSlide builds. Fall back to 0 for that slide.
- **`wsi.write()`** rewrites ALL tables. No selective write yet.
- **Gated models** need `HF_TOKEN` in env.
- **API drift:** before v0.6.0 the param was `n_workers`. Current: `num_workers`.
- **Old API:** `zs.WSI()` is gone. Use `from wsidata import open_wsi`.
- **`wsi.sdata[key]`** is gone. Use `wsi.tables[key]`.

---

## Remaining Work

These are the LazySlide ecosystem features not yet adopted:

| Feature | Status | Blocker |
|---------|--------|---------|
| `agg_wsi()` batch aggregation | Not adopted | Need to verify with our zarr layout |
| Spatial analysis (`tile_graph`, `spatial_domain`) | Not implemented | Net-new science feature |
| Vision-language queries (`text_embedding`, `text_image_similarity`) | Not implemented | Net-new science feature |
| Multimodal fusion (image + clinical text) | Not implemented | Needs pathology report data |

Current custom aggregation (slide-outer/model-inner with zarr X reads) is tested and
performant. `agg_wsi()` would be a simplification, not a correctness fix.

---

## Sources

- LazySlide docs: https://lazyslide.readthedocs.io/en/latest/
- wsidata docs: https://wsidata.readthedocs.io/en/latest/
- Multi-slide tutorial: https://lazyslide.readthedocs.io/en/latest/tutorials/multiple_slides.html
- Feature extraction API: https://lazyslide.readthedocs.io/en/v0.10.0/api/_autogen/lazyslide.tl.feature_extraction.html
- Nature Methods paper: https://doi.org/10.1038/s41592-026-03044-7
- dask-jobqueue: https://jobqueue.dask.org/en/latest/
