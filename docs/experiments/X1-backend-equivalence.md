# X1 — LazySlide vs Mussel backend equivalence (H-optimus-0, TITAN)

## Question
Can H-optimus-0 tile features extracted by LazySlide (ARGO's default backend) stand in for
features extracted by Mussel (the MSK tool whose H-optimus-0 features feed PALADIN), and
how much do the two backends' tile grids differ? Both tools are run exactly as documented
upstream (no forks, no monkeypatching); ARGO only builds the documented calls.

## Setup
**Slides.** 8 slides, one or more from every SITE, chosen by `compare-backends`' `sites`
selection with seed 0 on `results/data/slide_table_pyramidal.csv` — the table ARGO
extracts from (non-pyramidal / MPP-less originals replaced by their `.pyramidal.tiff`), so
both backends read the identical file. Subset: `results/analysis/backends/x1_slides.csv`
(`scripts/backends/x1_select_slides.py`).

| slide | site | file | native mpp |
|---|---|---|---|
| HP1353_21_1 | LASUTH | `.pyramidal.tiff` (vips-converted) | 0.250 |
| HP2074_21_5 | LASUTH | `.pyramidal.tiff` (vips-converted) | 0.250 |
| MSI-B985-22 | LUTH | SVS | 0.262 |
| MSI-B985-3-22- | LUTH | SVS | 0.262 |
| H1079-22-9 | OAUTHC | SVS | 0.260 |
| H700-22-1 | UITH | SVS | 0.260 |
| T218_1-1 | retrospective_msk | SVS | 0.260 |
| T207-1-1-USS-2 | retrospective_oau | SVS | 0.260 |

**Environments.**
- `envs/lazyslide`: lazyslide 0.12.0, lazyslide-models 0.0.4, wsidata 0.11.0 (OpenSlide
  reader, openslide-bin 4.0.1.2), opencv-python-headless 5.0.0.93, torch 2.11.0+cu128,
  torchvision 0.26.0, timm 1.0.30, transformers 5.17.0.
- `envs/mussel`: mussel-pathology 1.4.5 @ `d4cfce9` (tiffslide 3.0.1 reader, Pillow 12.3.0),
  torch 2.5.1+cu121, torchvision 0.20.1, timm 1.0.30, transformers 4.45.2.
- GPU: RTX A6000 (driver 560.35.03, CUDA 12.6), `nvidia-A6000-20`, 1 GPU / 8 CPU / 64 GB.

**Runs** (each written to its own out-root under `results/analysis/backends/stores/`; the
production zarrs next to the slides were never touched):

| run | backend / config | tile | precision | out-root |
|---|---|---|---|---|
| a | Mussel `mussel-hoptimus0.toml` (`model_type=OPTIMUS`, all else default) | 224 px @ 0.5 (MODEL_PATCH_SIZES) | fp32 | `stores/mussel` |
| b | LazySlide `lazyslide-hoptimus0.toml` (documented default) | 256 px @ 0.5 | fp32 | `stores/lazyslide_256` |
| c | LazySlide `lazyslide-hoptimus0-224.toml` (matched grid) | 224 px @ 0.5 | fp32 | `stores/lazyslide_224` |
| d | same as c + `--amp` (ARGO historical) | 224 px @ 0.5 | fp16 autocast | `stores/lazyslide_224_amp` |
| e | Mussel `mussel-titan.toml` (CONCH1_5 + TITAN_SLIDE) vs LazySlide `lazyslide-titan.toml` (`conch_v1.5` + `titan`) | 512 px @ 0.5 | fp32 | `stores/mussel`, `stores/lazyslide_512` |

LazySlide runs: `argo extract --backend lazyslide --config <toml> --out-root <root>`
(provenance JSON beside each store: `<stem>.zarr.<model_key>.provenance.json`). Mussel
runs: `argo extract --backend mussel --config <toml> --out-root stores/mussel`.
Jobs: LazySlide array 11016484 (+ smoke 11016460), Mussel array 11016487, pixel probe
11016958, comparisons 11016959 (rerun interactively after adding `coverage_iou`).

**Metrics** (`argo compare-backends`, `argo_deepmsi/backends/compare.py`): tile counts;
`coverage_iou` = IoU of the level-0 area covered by the two tile sets (origin-independent);
tiles matched one-to-one by level-0 anchor within ¼ tile; matched-tile cosine (median, p05);
slide-mean cosine over all tiles. For (a vs c) the matched pairs are *not* identical boxes
(grid origins differ by up to ¼ tile), so the feature-level question is answered by the
pixel probe below, which replays both backends on identical boxes.

## Results

### 1. Grids: same nominal size, different tiling (a vs c, a vs b)
| slide | tiles LS224 | tiles Mussel | coverage IoU | grid offset dx, dy (px) | slide-mean cos (a vs c) | slide-mean cos (a vs b, 256) |
|---|---|---|---|---|---|---|
| HP2074_21_5.pyr | 14906 | 15205 | 0.968 | 0, 128 | 0.9994 | 0.9973 |
| MSI-B985-3-22- | 21675 | 21266 | 0.781 | 143, −66 | 0.9893 | 0.9905 |
| H1079-22-9 | 5188 | 5696 | 0.901 | 24, 50 | 0.9835 | 0.9905 |
| H700-22-1 | 7612 | 8045 | 0.919 | 13, −161 | 0.9933 | 0.9906 |
| T218_1-1 | 6725 | 7225 | 0.893 | 115, 73 | 0.9820 | 0.9887 |
| T207-1-1-USS-2 | 4299 | 4619 | 0.903 | 52, −128 | 0.9791 | 0.9880 |
| HP1353_21_1.pyr | 4718 | 4884 | 0.884 | 80, 0 | 0.9952 | 0.9904 |
| MSI-B985-22 | 22076 | 23931 | 0.831 | −129, −104 | 0.9959 | 0.9947 |
| **median** | | | **0.897** | | **0.9913** (min 0.979) | **0.9905** (min 0.988) |

Mussel keeps 2–8% more tiles (consistent with its default HSV mask plus
`min_tissue_proportion=0` being more inclusive than LazySlide's `find_tissues`; not dissected), the covered tissue agrees at IoU 0.78–0.97, and
the grid origins differ per tissue contour, so on most slides almost no tile box coincides
(0 exact-coordinate matches on HP1353_21_1; 2 slides have *no* tile within ¼ tile —
verdict string "no matched tiles"). Matched-within-¼-tile cosine (median 0.91–0.99) mixes a
spatial shift of up to 112 px with any processing difference and should not be read as
backend noise. LazySlide 224 vs 256 on the same backend (`compare_hoptimus0_lazyslide_224_vs_256`)
agrees less than either LazySlide grid agrees with Mussel (slide-mean cos median 0.973,
min 0.956): **tile size is a bigger lever than the backend**.

### 2. Located divergence on identical boxes (pixel probe)
`scripts/backends/x1_pixel_probe.py` samples 64 Mussel tile boxes per slide and replays each
backend's documented read + preprocessing on those exact boxes in its own env, then swaps
reader output, preprocessing, and model runtime between the two paths
(`results/analysis/backends/pixel_probe.csv`). Replays reproduce each backend's stored
features exactly (cos 1.0 for both), so the probe is faithful.

| slide | LS box / Mussel box (level-0 px) | decoded pixels identical | resize Δ (mean abs, u8) | **total cos** median (p05) | box+reader only | resize only | runtime only |
|---|---|---|---|---|---|---|---|
| HP1353_21_1.pyr | 448 / 448 | 100% | 0.12 | **0.9991** (0.9969) | 1.0000 | 0.9991 | 1.0000 |
| HP2074_21_5.pyr | 448 / 448 | 100% | 0.11 | **0.9994** (0.9982) | 1.0000 | 0.9994 | 1.0000 |
| MSI-B985-22 | 427 / 428 | 100% (common crop) | 0.16 | **0.9981** (0.9959) | 0.9993 | 0.9988 | 1.0000 |
| MSI-B985-3-22- | 427 / 428 | 100% | 0.68 | **0.9862** (0.9544) | 0.9992 | 0.9857 | 1.0000 |
| H1079-22-9 | 430 / 431 | 100% | 0.70 | **0.9824** (0.9635) | 0.9989 | 0.9819 | 1.0000 |
| H700-22-1 | 430 / 431 | 100% | 1.17 | **0.9940** (0.9771) | 0.9988 | 0.9952 | 1.0000 |
| T218_1-1 | 430 / 431 | 100% | 1.45 | **0.9673** (0.9071) | 0.9982 | 0.9770 | 1.0000 |
| T207-1-1-USS-2 | 430 / 431 | 100% | 1.42 | **0.9748** (0.9172) | 0.9984 | 0.9777 | 1.0000 |

Step by step:
1. **Reader / decode — identical.** OpenSlide (LazySlide) and tiffslide (Mussel) return
   bit-identical RGB for every probed box on both SVS and vips pyramidal TIFFs (RGBA→RGB
   included). Level selection is the same (level 0 on every slide).
2. **Tile edge rounding — small.** Both compute the level-0 edge as 224 × 0.5 / slide mpp
   (430.83 px at 0.25997 mpp), but wsidata truncates it (430) while Mussel rounds (431); the
   origin is the same. Alone this costs ~0.001 cosine (column "box+reader only", same
   Mussel preprocessing on each backend's read). Exactly 448 on 0.25-mpp TIFFs → no effect.
3. **Resize — the located divergence.** LazySlide's `TileImagesDataset` downsamples the
   level-0 read with `cv2.resize(tile, (224, 224))` (INTER_LINEAR, no antialiasing) before
   the model transform, so H-optimus-0's documented `Resize(224, BICUBIC, antialias)` is a
   no-op; Mussel applies that bicubic, antialiased PIL resize to the full-resolution read.
   Same raw pixels, LazySlide resize vs Mussel resize: cos median 0.977–0.999, p05 down to
   0.894. The gap tracks image sharpness/aliasing (mean resize Δ 0.1 u8 on the 2× TIFFs vs
   ~1.4 u8 on the sharp retrospective scans); at an exact 2× factor bilinear ≈ box filter,
   which is why the pyramidal TIFFs are near-identical.
4. **Model runtime — identical.** The same input tensors through LazySlide's model
   (torch 2.11, `encode_image`, `inference_mode`) and Mussel's (`OptimusModel`, torch 2.5)
   give cos ≥ 0.999999 (max abs diff ≤ 0.024 on 1536-d features). Same timm construction
   and normalisation constants.

### 3. fp16 drift (c vs d)
LazySlide `amp=True` vs `amp=False` on the same 224 grid: 100% tiles matched, cos median
1.000000, p05 ≥ 0.99999, slide-mean cos 1.000000, max abs diff ≤ 0.042. **fp16 autocast is
negligible** for H-optimus-0 (verdict: identical up to float noise) and runs 2.4× faster.

### 4. TITAN (e)
CONCH v1.5 tiles at 512 px: coverage IoU median 0.82; matched-tile cos median 0.95–0.997
(shift-confounded as above; the TIFF slide with coinciding columns reaches 0.997). TITAN
slide embeddings (768-d), LazySlide vs Mussel: **cos median 0.9909 (range 0.9826–0.9963)**
(`compare_titan_slide.csv`). Mussel feeds TITAN every CONCH tile of its (larger) mask with
level-0 coordinates and patch size 985/1024; LazySlide uses its own tiles. Part of this
gap is the resize path (Mussel CONCH1_5: 512 → `Resize(448, bilinear)`; LazySlide conch_v1.5:
`cv2.resize` to 512 then the model's own transform) — not separately probed here.

### 5. Wall time per slide (A6000, median over 8 slides; `timings_summary.csv`)
| run | wall s / slide (incl. start-up + model load) | median tiles |
|---|---|---|
| Mussel H-optimus-0 (batch 64, 4 workers) | 128 | 7635 |
| LazySlide H-optimus-0 256 px fp32 (batch 32, 0 workers) | 405 | 5462 |
| LazySlide H-optimus-0 224 px fp32 | 447 | 7169 |
| LazySlide H-optimus-0 224 px fp16 | 185 | 7169 |
| Mussel TITAN (CONCH1_5 + TITAN) | 95 | 1527 |
| LazySlide TITAN (conch_v1.5 512 px + titan) | 185 | 1335 |

LazySlide's documented `num_workers=0` leaves the GPU waiting on single-process tile reads;
throughput, not features, is what the documented LazySlide defaults cost.

## Verdicts
- **Model/runtime and decoding: interchangeable.** Given the same RGB tile tensor, the two
  environments produce the same H-optimus-0 features (cos ≥ 0.999999); OpenSlide and
  tiffslide decode identical pixels.
- **Features on identical boxes: not interchangeable at float precision.** Median cos
  0.967–0.999 (p05 as low as 0.907) per slide, entirely explained by LazySlide's
  non-antialiased `cv2.resize` (plus a 1-px edge truncation on ~0.26-mpp SVS). On 0.25-mpp
  pyramidal TIFFs (exact 2× downsample) the backends agree to 0.999.
- **Grids: segmentation-driven, not equivalent.** Coverage IoU 0.78–0.97 and per-contour
  origins mean tile sets are different samples of the same tissue; slide-mean features still
  agree at cos 0.979–0.999 on the matched 224-px grid.
- **fp16: negligible.** **TITAN: slide-embedding cos ~0.99.**

## Implications
- **LazySlide-extracted H-optimus-0 should not be fed to PALADIN as a drop-in replacement
  for Mussel features.** PALADIN/Aeon were trained on Mussel-pipeline features; the tile-level
  shift (≤0.97 cos on the sharp retrospective scans) is larger than fp16 noise by 3–4 orders
  of magnitude and the tile sets differ. Use `argo extract --backend mussel` for PALADIN, as
  planned. For ARGO's own analyses (heads trained on ARGO features) LazySlide remains fine.
- ARGO's historical `amp=True` is not a concern for H-optimus-0.
- **When MSK engineering sends the Mosaic/PALADIN parameters**, only
  `configs/backends/mussel-hoptimus0.toml` (and `mussel-titan.toml`) change: seg preset,
  mpp, patch_size (note 256 is a sentinel that Mussel replaces with 224 for OPTIMUS),
  overlap, `min_tissue_proportion`, Mussel version/commit. Then rerun the Mussel array and
  every comparison (below). If engineering's tiles are not 224 px @ 0.5, add a matching
  LazySlide variant config and out-root. Also ask whether Mosaic used tiffslide or cuCIM
  (Mussel prefers cuCIM when installed; this env has none) — the decode step was identical
  here but was only tested for OpenSlide vs tiffslide.
- A LazySlide run can match Mussel on grid geometry (224 px) but not on resize without
  changing LazySlide's code path; per the backend policy we characterise rather than patch.

## Rerun
```bash
uv run --frozen python scripts/backends/x1_select_slides.py        # 8-slide subset
sbatch --array=0-7 scripts/backends/x1_mussel.sh                   # runs a, e (Mussel)
sbatch --array=0-7 scripts/backends/x1_lazyslide.sh                # runs b, c, d, e (LazySlide)
sbatch scripts/backends/x1_pixel_probe.sh                          # identical-box probe
sbatch scripts/backends/x1_compare.sh                              # compare_*.csv/md
uv run --frozen python scripts/backends/x1_timings.py              # timings.csv / timings_summary.csv
```
Outputs: `results/analysis/backends/compare_hoptimus0_{224,256,fp16,lazyslide_224_vs_256}.{csv,md}`,
`compare_titan.{csv,md}`, `compare_titan_slide.csv`, `pixel_probe.csv`, `timings*.csv`;
stores under `results/analysis/backends/stores/` (≈2.9 GB, not for git).
