# Plan — split environments and dual extraction backends (LazySlide + Mussel)

**Status:** approved by Matteo 2026-09-24, not yet implemented. Branch: `backends` (from `iris`
@ `5afc448`). This file is the handoff for the implementation session; update its checklist
as work lands.

## Goal

ARGO-DeepMSI stays the single orchestrator (setup, credentials, data/cohort tasks, all
analyses including one-offs). Slide encoding is delegated to external tools, each in its own
locked environment, **used exactly as documented upstream (no parameter divergence, no
forks, no clones)**:

| Stage | Environment | What |
|---|---|---|
| 1. setup | core | `argo setup`: create/verify every env, HF cache/token, data paths |
| 2. slide tasks | core | ingest, pyramidal conversion, QC, cohort freeze |
| 3. extract | `envs/lazyslide` **or** `envs/mussel` | per-slide features in each tool's native format + ARGO provenance |
| 4a. analyses | core | Wagner, heads, domain shift, slide-count audit, interpretability — on LazySlide features |
| 4b. PALADIN | `envs/paladin` | PALADIN/Aeon on Mussel H-optimus-0 features (weights are MSK-internal) |

LazySlide is the default backend. Anyone who wants PALADIN uses Mussel. ARGO ships tests
that **characterise** (not force) divergence between backends.

Timing decision: build now with Mussel's `default` preset; when MSK engineering confirms the
Mosaic/PALADIN parameters, only `configs/backends/mussel-hoptimus0.toml` changes and the
comparison is rerun.

## Evidence gathered (2026-09-24)

**Mussel** (`github.com/pathology-data-mining/Mussel`, GPL-3.0, pin commit
`d4cfce9437d92706811b465ff46cfb247c462249`, 2026-08-21; package name `mussel-pathology`):
- Documented install into an existing env: `Mussel[torch-gpu]` (torch < 2.6 from the
  `download.pytorch.org/whl/cu121` index; its own pyproject maps torch/torchvision to that
  index via `[tool.uv.sources]` — a dependent project must replicate that mapping).
  `requires-python >=3.10,<3.12`. Pins `transformers<4.46`, `numcodecs<0.16` → cannot share
  the LazySlide env (transformers 5.16, numcodecs 0.16.5, zarr 3).
- CLI: `tessellate_extract_features slide_path=... output_h5_path=... output_pt_path=...
  model_type=<TYPE> [seg_config=<preset>] [key=value ...]` (Hydra-style overrides).
- Defaults: `seg_config=default` (classic HSV, full mask), `mpp=0.5`, `patch_size=256`,
  `overlap=0`, `min_tissue_proportion=0.0`, `tissue_area_threshold=100` tiles,
  `hole_area_threshold=16`, `batch_size=64`, `num_workers=4`, `embedding_precision=float32`.
- ⚠ The command Andy sent uses `seg_config=stain` — the README's *stain-classification*
  preset: 32-tile cap, bounded neural sampling, no full-slide mask. Almost certainly not what
  Mosaic used (PALADIN `max_seq_len` 20000). Follow-up sent asking engineering to confirm.
- H-optimus-0: `ModelType.OPTIMUS` = `hf-hub:bioptimus/H-optimus-0` (README calls it
  `HOPTIMUS0`; confirm the accepted CLI name). timm `create_model(pretrained=True,
  init_values=1e-5, dynamic_img_size=False)`; preprocessing `Resize(224, BICUBIC)` →
  `ToTensor` → `Normalize(mean=(0.707223,0.578729,0.703617), std=(0.211883,0.230117,0.177517))`.
- TITAN: patch encoder `CONCH1_5` (`MahmoodLab/TITAN`) + `slide_model_type=TITAN_SLIDE`;
  needs coords and patch_size. Check the README section on TITAN's patch-size handling.
- Outputs: `.h5` with datasets `coords` (N×2, level-0 top-left; attrs incl. `patch_size`) and
  `features` (optional); `.pt` = torch tensor N×D. Reader reference:
  `mussel/utils/file.py::load_features_from_h5` (bfloat16 stored as `|V2`).
- Tools useful later for interpretability: `export_tiles`, `annotate`,
  `create_class_embeddings`, `clustering_benchmark`, `convert`.

**LazySlide** (current env: lazyslide 0.12, lazyslide-models; h-optimus-0 in
`lazyslide_models/vision/h_optimus.py`):
- Same timm construction (`init_values=1e-5, dynamic_img_size=False`) and same normalisation
  constants; transform `Resize((224,224), BICUBIC)` + `CenterCrop(224)` — identical to
  Mussel's for square tiles.
- Tiling is ARGO's `find_tissues` + `tile_tissues(tile_px=256, mpp=0.5)` → same nominal grid
  size as Mussel's default, but a **different tissue segmentation**, so tile sets/origins differ.
- ARGO currently calls `feature_extraction(..., amp=True)` (fp16 autocast) — Mussel is fp32.
  For the equivalence test run LazySlide at its documented default and record amp.
- Zarr layout: `<slide>.zarr/tables/<model>_tiles` (AnnData, obs `tile_id`),
  `<slide>.zarr/shapes/tiles/shapes.parquet` (`tile_id`, `tissue_id`, WKB polygon `geometry`,
  identity transform → level-0 pixels). Reader: pyarrow + shapely bounds.
- H-optimus-0 HF access works for account `mat10d`; TITAN access works.

**Environment facts**
- Whitehead GPU driver is CUDA 12.6: the current lock's `torch==2.14.0` (PyPI, CUDA 13)
  fails at `.cuda()`. The old conda env `argo` (torch 2.10+cu128) works. → Pin torch to the
  cu128 index in `envs/lazyslide` (and core if GPU is needed there), cu121 in `envs/mussel`.
- Current root env mixes core + LazySlide; code already guards LazySlide with
  `LAZYSLIDE_AVAILABLE` in `feature_extraction.py`.
- LazySlide-importing modules: `feature_extraction.py`, `visualization.py`,
  `models/_lazyslide.py`, `models/ctranspath.py`, `models/waiv.py` (+transformers),
  `eval/error_anatomy.py` (wsidata open_wsi), scripts `qc/artifact_qc.py`,
  `qc/tumor_tile_classifier.py`, `domain_shift/stain_norm_oauthc.py`, `wagner_zeroshot.py`
  (wsidata). `dask_extraction.py` drives LazySlide extraction.
- PALADIN (`github.com/kmboehm/paladin`): torch 2.4.1 (cu121), lightning 2.3.0,
  torchmetrics 1.4.2, numpy 1.26.4, `nn-template-core==0.4.0 --no-deps`; inference script
  hardcodes an internal checkpoint path; weights not public.

## Design

```
ARGO-DeepMSI/
├── pyproject.toml + uv.lock      # core: CLI, setup, cohort, readers, eval, heads (no lazyslide)
├── envs/
│   ├── lazyslide/pyproject.toml + uv.lock   # argo-deepmsi (path, editable) + lazyslide stack
│   ├── mussel/pyproject.toml + uv.lock      # mussel-pathology[torch-gpu] @ git pin + cu121 index
│   └── paladin/                             # setup script replicating paladin README (not locked)
├── configs/backends/
│   ├── lazyslide-hoptimus0.toml   # documented defaults
│   ├── mussel-hoptimus0.toml      # Mussel default preset; swap in MSK engineering values
│   └── mussel-titan.toml / lazyslide-titan.toml
└── argo_deepmsi/
    ├── envs.py                    # env registry, `uv run --frozen --project envs/<name>` runner,
    │                              # auto-dispatch of LazySlide commands, env status
    └── backends/
        ├── provenance.py          # backend, tool version/commit, model, full params, env lock hash
        ├── mussel.py              # config → CLI argv; per-slide run; output layout; provenance
        ├── readers.py             # TileFeatures(features, coords_level0, tile_px, mpp, backend, model, provenance)
        │                          # from LazySlide zarr or Mussel h5/pt (no format conversion)
        └── compare.py             # backend divergence metrics + report
```

- **Dispatch:** a `requires_env("lazyslide")` guard on CLI commands that need LazySlide
  (`extract --backend lazyslide`, `extract-dask`, `models`, `aggregate`, `qc`, `visualize`,
  `run`, and `experiment` when extraction/aggregation stages are enabled). If LazySlide is not
  importable, re-exec `uv run --frozen --project envs/lazyslide argo <same argv>` with an
  `ARGO_ENV=lazyslide` loop guard. The lazyslide env contains core too (path dependency), so
  every command works there.
- **Mussel output layout:** next to the slide, like zarrs:
  `<slide_dir>/<slide_stem>.mussel/<MODEL>.features.h5|.pt` + `<MODEL>.provenance.json`.
  Per-slide failure isolation; skip existing complete outputs; SLURM array wrapper
  `scripts/extract_mussel.sh` (GPU, cu121 env).
- **Core dependency changes:** drop lazyslide, lazyslide-models, wsidata, scanpy, torchstain,
  transformers, einops(-exts), fairscale, sentencepiece, musk, llvmlite/numba pins,
  setuptools<81 pin, and the `waiv`/`dask`/`conch`/`omiclip` extras → all move to
  `envs/lazyslide`. Add to core: `h5py`, `pyarrow`, `shapely`, `openslide-python` +
  `openslide-bin` (slide tasks, image stats). Keep torch (heads, Wagner on cached features),
  anndata + zarr (reading LazySlide tables), umap-learn, inmoose, scikit-learn.
- **Tests:** core suite must pass without LazySlide installed. LazySlide-API tests run in
  `envs/lazyslide` (marker `lazyslide`). New synthetic tests: Mussel argv builder from config,
  reader for fake h5/pt and fake zarr (WKB tiles), coordinate matching + metrics, dispatch
  command construction and loop guard, provenance round-trip.

## Backend equivalence study (the acceptance experiment)

`argo compare-backends --slides <table> --n 8 --model hoptimus0` (+ TITAN), 8 slides spanning
every site including a pyramidal-converted/MPP-stamped slide and a native SVS:
1. Mussel at config defaults → h5/pt. LazySlide at documented defaults → zarr table.
2. Tile-grid agreement: tile counts, tissue overlap, fraction of tiles matched by level-0
   top-left within ¼ tile.
3. Matched-tile cosine similarity (median, 5th percentile); max abs diff.
4. Slide-mean cosine and correlation; TITAN slide-embedding cosine.
5. Report (`results/analysis/backends/compare_<model>.csv` + markdown) and a verdict:
   identical up to float noise on matched tiles (expected, given identical model +
   preprocessing) with grid differences explained by segmentation, or a located divergence.
Also run once with LazySlide `amp=True` (ARGO's current setting) to quantify fp16 drift.

## Implementation checklist

- [ ] `envs/lazyslide` project (+ cu128 torch index), lock, sync, GPU smoke (`torch.zeros(1).cuda()`)
- [ ] `envs/mussel` project (git pin + cu121 index), lock, sync, `tessellate_extract_features --help`
- [ ] core pyproject slimmed; core lock; `pytest -q` passes without LazySlide
- [ ] `argo_deepmsi/envs.py` + `argo setup` / `argo envs` + dispatch guard
- [ ] `backends/{provenance,mussel,readers,compare}.py` + CLI `extract --backend`, `compare-backends`
- [ ] configs/backends/*.toml; `scripts/extract_mussel.sh`
- [ ] synthetic tests; acceptance in core and in `envs/lazyslide`
- [ ] run equivalence study (GPU) on 8 slides; write `docs/experiments/X1-backend-equivalence.md`
- [ ] `envs/paladin` setup script + `argo paladin` stub (clear error until a checkpoint path is configured)
- [ ] update CLAUDE.md / AGENTS.md / README / `docs/iris-runbook.md` (four-stage status table)
- [ ] commit on `backends`, merge to `iris` after review

## Open questions (do not block the build)

1. MSK engineering's exact Mussel parameters for Mosaic/PALADIN (seg preset, mpp,
   patch_size, overlap, min_tissue_proportion, Mussel version) — asked via Andy.
2. Accepted `model_type` string for H-optimus-0 (`OPTIMUS` vs `HOPTIMUS0`).
3. Cluster: CDSI is the likely home (engineering + embeddings there); IRIS pending.
