# Refactor Status — Operon Markdowns vs. Landed Work

Cross-reference of every concrete recommendation in the four docs Operon
produced (`code_review.md`, `efficiency_analysis.md`, `lazyslide_gap_analysis.md`,
`lazyslide_reference_guide.md`) against what actually shipped on branch
`lazyslide-refactor`.

Legend: **✅** landed · **⚠️** deferred with explicit rationale · **❌** not
addressed (tracked here for follow-up).

---

## 1. `code_review.md`

### High priority

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Patient-level CV (StratifiedKFold → StratifiedGroupKFold on `patient_id`) | ✅ | `training.py`; enforced in `compare_classifiers` (raises without `groups`) — commit `ea0bf72` |
| 2 | Index-alignment bug in `load_training_data` | ✅ | `training.py` uses `reset_index(drop=True)` + row-count guard + `merged.index.values` — commit `ea0bf72` |
| 3 | Visualization re-extracts features from scratch | ✅ | `_open_cached` in `visualization.py` loads cached zarr; all four viz functions use it — commit `ea0bf72` + zarr-reopen fix in `6c3738d` |
| 4 | Missing `resnet50` in `PATCH_MODELS` | ❌ | Not added. The test suite uses `resnet50` successfully via LazySlide's TimmModel path (not registered in `zs.models.MODEL_REGISTRY`). README still lists it as non-gated. Low priority — either add a `ModelConfig(name="resnet50", type="timm", requires_auth=False)` entry or drop from README. |

### Medium priority

| # | Item | Status | Where |
|---|---|---|---|
| 5 | `visualize` CLI doesn't pass labels to UMAP | ✅ | `cli.py` now routes through `load_training_data` — commit `ea0bf72` |
| 6 | `scripts/train.sh` / `aggregate.sh` array bounds mismatch | ✅ | Array sizes aligned to enabled model counts + empty-slot guards — commit `ea0bf72` |
| 7 | `plt.suptitle` / `plt.tight_layout` in `data_ingestion.py` | ✅ | Both converted to `fig.*` — commit `ea0bf72` |
| 8 | No empty-group guard in `extract.sh` | ✅ | `NUM_SLIDES -le 0` guard added — commit `ea0bf72` |

### Low priority

| # | Item | Status | Notes |
|---|---|---|---|
| 9 | `aggregate.sh` doesn't need GPU (unless switched to a neural encoder) | ⚠️ | Partition remains `short` (no GPU) because default `METHOD=mean`. If you change `METHOD` to `prism`/`titan` you'll need a GPU partition. Documented in `CLAUDE.md`'s aggregation section. |
| 10 | Dead `wsi.sdata[key]` path in `process_slide_with_aggregation` | ✅ | Rewritten to read from `wsi.tables[feature_key].uns['agg_slide']` — commit `ea0bf72` |
| 11 | `simple_methods` list included `std`/`var` but they weren't implemented | ✅ | Removed from the list; `_POOL_FNS` now enforces the supported set — commit `ea0bf72` |
| 12 | Hardcoded `/lab/barcheese01/mdiberna/...` paths in SLURM scripts | ❌ | Still present in `extract.sh`, `aggregate.sh`, `train.sh`, `extract_dask.py`. Portability issue for re-use on other clusters or a different user's checkout. Fix: compute repo root from `$(dirname $(dirname ${BASH_SOURCE[0]}))` and `cd` there; source conda from `$(conda info --base)` instead of hardcoded path. |
| 13 | Missing `__main__.py` | ✅ | Added — commit `0912acc` |

---

## 2. `efficiency_analysis.md`

### Top quick wins

| # | Item | Status | Where |
|---|---|---|---|
| 1a | Pass `num_workers=4`, `batch_size=64` to `feature_extraction` | ✅ | Plumbed through `process_slide`, `extract_features_single_slide`, `extract_features_batch` — commit `ea0bf72` |
| 2a | Skip `AnnData` overhead in simple pooling; read zarr X directly | ✅ | `aggregate_simple_pooling` uses `zarr.open(...)["tables"][key]["X"][:]` with sparse fallback — commit `ea0bf72` |
| 4a | Visualization loads from zarr instead of re-extracting | ✅ | `_open_cached` helper — commit `ea0bf72` |

### Structural improvements

| # | Item | Status | Where |
|---|---|---|---|
| 2b | Slide-outer / model-inner aggregation loop | ✅ | Zarr opened once per slide regardless of how many models — commit `ea0bf72` |
| 3a | Neural encoder: drop SVS+zarr double-open | ✅ | `aggregate_neural_encoders` opens the zarr only — commit `ea0bf72` |
| 1d | Selective `wsi.write()` for incremental extraction | ⚠️ | LazySlide (0.10) does not expose selective table writes. `wsi.write()` rewrites all tables. Called out in `docs/lazyslide_reference_guide.md` §12 as an upstream feature request; not fixable here. |

### Still not addressed

| # | Item | Status | Notes |
|---|---|---|---|
| 5a | Finer-grained SLURM parallelism (10–20 groups vs 2) | ⚠️ | `scripts/extract_dask.py` (commit `0912acc`) is the elastic-scaling alternative — goes to `--max-workers 10` by default and scales one job per slide. The 2-group bash array stays as the simpler fallback. Net: the recommendation IS addressed, just via a different mechanism than "bump the array size". |
| 5b | Progress tracking file (persistent per-slide `progress.csv`) | ❌ | Not added. Current reliance is on the incremental zarr-existence check: on restart, each slide's zarr is listed and existing models are skipped. This is O(slides × models) directory listings on restart, which is fast enough for ~800 slides × 11 models. A progress.csv would be faster on very large cohorts. Defer unless we scale past a few thousand slides. |
| 6a | `argo run` overlap of extract / aggregate / train across models | ❌ | Still strictly serial per model. `argo run` is not used in the current SLURM flow (that uses the three separate sbatch scripts); the sequential `run` command is mainly for local smoke testing. Not a production bottleneck — leave until someone actually uses `argo run` at scale. |
| 7a | QC-then-filter workflow in a single pass (avoid double SVS open) | ⚠️ | We added QC as a set of registered models and `filter_slides_by_qc` + `argo qc` CLI. Running QC first and filtering before the expensive models is a two-pass flow (documented in `CLAUDE.md` QC section). The single-pass `--qc-filter` variant isn't implemented — deferred until we see whether the two-pass flow is in practice a bottleneck. |
| 7b | Conditional-tile-skip for slide-level extractors | ❌ | Direct slide-level extractors (`gigapath-slide-encoder`, `chief-slide-encoder`, `gigatime`) don't need patch tiling. `extract_features_single_slide` runs `tile_tissues` unconditionally. Fix is straightforward (check `if any(m in PATCH_MODELS for m in models_to_extract)` before tiling) but we don't currently use slide-level extractors in `scripts/extract.sh`, so deferred. |
| 8a | `verify_slides_exist` full recursive walk | ❌ | Minor. `os.walk` over ~800 SVS files is <1s in practice. Defer. |
| 8b | `iterrows()` in aggregation hot path | ❌ | Replaced one spot already; others remain. For 800 slides this is noise compared to I/O. Defer. |
| 8c | `CLAUDE.md` had stale `zs.WSI("...")` example | ✅ | Rewritten to `wsidata.open_wsi(...)` with current API — commit `794a977` |

---

## 3. `lazyslide_gap_analysis.md`

### Phase 1 core fixes

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Use `num_workers=4, batch_size=64` | ✅ | (see `efficiency_analysis.md` §1a above) |
| 2 | `wsidata.agg_wsi()` fast path | ✅ | `aggregate_with_agg_wsi()` available for slides that already have a pre-computed `agg_slide` key — commit `0912acc` |
| 3 | AnnData output (scverse-interop) | ✅ | `_save_embeddings` writes `embeddings.h5ad` alongside npy+csv; `load_training_data` prefers h5ad — commit `0912acc` |
| 4 | Visualization loads from zarr | ✅ | (see `code_review.md` §3) |

### Phase 2 ecosystem leverage

| # | Item | Status | Where |
|---|---|---|---|
| 5 | Dask + dask-jobqueue for SLURM | ✅ (partial) | `scripts/extract_dask.py` + `[dask]` extra — commit `0912acc`. `aggregate.sh` and `train.sh` are CPU-bound short jobs; didn't port them. |
| 6 | QC models integration | ✅ | QC models in `PATCH_MODELS` with `type="qc"`; `filter_slides_by_qc` + `argo qc` CLI; documented workflow — commit `0912acc` |
| 7 | `wsi.fetch.features_anndata()` consistency | ⚠️ | We use direct `zarr.open` + X slicing in aggregation, which is strictly faster than going through the fetch accessor (no AnnData construction overhead). The fetch pattern would be cleaner stylistically but slower. Intentional deviation from the recommendation. |
| 8 | Use scanpy on AnnData output | ⚠️ | AnnData output exists; scanpy is an installed dep and the test suite verifies scanpy on `agg_wsi` output works (`test_agg_wsi_scanpy_integration`). We don't yet use scanpy's UMAP/leiden as the default downstream — it's available for users who want it. |

### Phase 3 advanced capabilities

| # | Item | Status | Notes |
|---|---|---|---|
| 9 | Spatial domain analysis (`zs.pp.tile_graph`, `zs.tl.spatial_domain`) | ❌ | Explicitly listed under "Future Work" in `CLAUDE.md`. Not a refactor of existing code — it's a net-new analysis pipeline step. Needs a research decision before implementation. |
| 10 | Vision-language queries (`text_embedding`, `text_image_similarity`) | ❌ | Same. Future Work. |
| 11 | Multimodal integration with clinical text | ❌ | Same. Future Work. |

---

## 4. `lazyslide_reference_guide.md`

This doc is primarily a reference for correct API usage, not a recommendation
list. The test suites enforce its patterns:

| Section | Content | Enforced by |
|---|---|---|
| 2–4 | `open_wsi`, preprocessing, feature extraction with num_workers | `test_lazyslide_api.py::TestSlideOpening/Preprocessing/FeatureExtraction` |
| 5–6 | Neural aggregation + `agg_wsi` batch | `test_lazyslide_api.py::TestFeatureAggregation` |
| 7 | Dask-jobqueue SLURM recipe | Implemented in `scripts/extract_dask.py` |
| 8 | QC models | `test_argo_pipeline.py::TestQCFilter`, `argo qc` |
| 9 | Vision-language queries | ❌ Not implemented; Future Work |
| 10 | Spatial analysis | ❌ Not implemented; Future Work |
| 11 | Visualization from zarr | `test_argo_pipeline.py::TestOpenCached` + `visualization.py::_open_cached` |
| 12 | Gotchas (API drift, `num_workers>0` segfaults, selective write limits) | Documented; the `fastslide`-reader zarr-reopen gotcha was actually hit and fixed in `6c3738d` (`open_wsi(svs, store=parent)` pattern) |

---

## Summary — what's still open

### Minor / worth doing eventually

1. **resnet50 in `PATCH_MODELS`** (or remove from README) — code_review §4
2. **De-hardcode SLURM script paths** (use `$(dirname ${BASH_SOURCE})` + `conda info --base`) — code_review §12
3. **Skip tiling for direct slide-level extractors** — efficiency §7b
4. **Progress CSV for extraction** — efficiency §5b. Only matters at scale > few thousand slides.

### Explicitly deferred (Future Work per `CLAUDE.md`)

5. **Spatial domain analysis**
6. **Vision-language queries on CONCH/PLIP**
7. **Multimodal (image + clinical text) fusion**

### Upstream issues (not fixable in this repo)

8. **Selective `wsi.write()`** — LazySlide feature request
9. **`hf_access` reading `HF_TOKEN` env** — worked around via the `huggingface_hub.login()` call in `argo_deepmsi/__init__.py` (commit `3f96ee4`)

Nothing in the "still open" set blocks the full GPU run documented in
`docs/handoff_full_run.md`.
