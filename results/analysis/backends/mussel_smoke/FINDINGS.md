# Mussel smoke — findings (2026-09-24)
Env envs/mussel: py3.11.13, mussel-pathology 1.4.5 @ d4cfce94…, torch 2.5.1+cu121, torchvision 0.20.1+cu121, timm 1.0.30 (via open-clip-torch 3.3.0), transformers 4.45.2, hf-hub 0.36.2, h5py 3.16.0, numpy 2.4.6, tiffslide 3.0.1. RTX A6000 driver 560.35.03 OK.
Files: default_config.yaml, run_smoke.sh, inspect_outputs.py, per-slide dirs, timing_*.txt, gpu_mem_*.log.

## CLI
Env vars: TMPDIR=$PWD/.tmp_build HF_HOME=$PWD/.huggingface_cache HF_HUB_OFFLINE=1 (offline works; models cached).
H-optimus-0: uv run --frozen --project envs/mussel tessellate_extract_features slide_path=… output_h5_path=…/OPTIMUS.features.h5 output_pt_path=…/OPTIMUS.features.pt model_type=OPTIMUS
TITAN: … slide_path=… output_dir=…/titan model_type=CONCH1_5 slide_model_type=TITAN_SLIDE (both set => multi-model mode => internal batch mode; needs output_dir or output_h5_path)
Mussel uses tempfile.mkdtemp() => /tmp unless TMPDIR set. NFS causes harmless "OSError [Errno 16] .nfs…" finalizer tracebacks; clean empty pymp-* dirs.

## model_type
Parsed via ModelType[name] (exact, case-sensitive). H-optimus-0 = OPTIMUS (HOPTIMUS0 does not exist; HOPTIMUS1 = H-optimus-1). TITAN = CONCH1_5 + TITAN_SLIDE (model_type inferable).

## Patch size / magnification
patch_size=256 is a sentinel replaced by MODEL_PATCH_SIZES: OPTIMUS=224, CONCH1_5/TITAN_SLIDE=512 (all @ mpp 0.5). 256 cannot be forced for OPTIMUS.
Read at level 0 with native size round(ps*mpp/slide_mpp) (431 @0.26mpp SVS, 448 @0.25 TIFF, 985 TITAN) then resized. OPTIMUS: Resize(224,bicubic)->ToTensor->Normalize((0.707223,0.578729,0.703617),(0.211883,0.230117,0.177517)). CONCH1_5: 512 -> Resize(448,bilinear). fp32, no amp.
TITAN gets coords.attrs["patch_size"] (level-0, 985) + level-0 coords; max_slide_patches=null (no subsample; O(N^2)).
tissue_area_threshold (100) / hole_area_threshold (16) in tiles -> mask depends on patch size: 5 contours (OPTIMUS) vs 2 (CONCH) on the same slide.

## Presets
default: classic HSV thr20 blur7 morph0, seg_level -1 (auto ~64x), mpp0.5, overlap0, min_tissue 0, area100 hole16 holes8, ref512, full_mask, no max_tiles.
stain: neural DeepLabV3, bounded_neural, min_tissue 0.75, max_tiles 32, max_candidate 256 => ≤32 tiles/slide; unsuitable for MIL/PALADIN.
biopsy 15/11/2 area1 hole1 holes2; resection 15/11/4; tcga 8/7/4 area16 hole4.

## Runs (nvidia-A6000-20, 1 GPU, 8 CPU, 64G)
11016361 1210-2-16.svs OPTIMUS 6276x1536 wall 259s (extract ~82s) GPU 8150MiB RSS 9.7GB
11016362 HP1353_21_1.pyramidal.tiff OPTIMUS 4884x1536 wall 247s (~71s) 8150MiB 9.7GB (native_mpp 0.25, mpp_is_fallback False)
11016364 1210-2-16.svs CONCH1_5+TITAN_SLIDE 1044x768 + slide 768, wall 114s, 3896MiB
Wall includes ~2 min uv/import start-up + ~55s model load.

## Output schema
h5 (no root attrs):
 coords int64 (N,2) gzip, (x,y) level-0 top-left.
 coords.attrs: patch_size (level-0 px), step_size (level-0), patch_size_to_resize_to_for_desired_mpp (e.g. 224), patch_level=0, mpp=0.5, native_mpp, mpp_is_fallback, level_dim=[w,h], name=slide stem, seg_level, seg_model, selection_mode, segment_threshold, segment_max_value, median_blur_ksize, morphology_ex_kernel, ref_patch_size, tissue_area_threshold, hole_area_threshold (pixel areas), max_num_holes, overlap, min_tissue_proportion, max_tiles(-1=None), max_tiles_strategy, max_tiles_seed, max_candidate_tiles(-1=None).
 features <f4 (N,D) gzip chunks (128,D); bfloat16 precision => |V2, decode via view(ml_dtypes.bfloat16).astype(float32).
 Rows aligned; failed tiles dropped from both.
pt: torch.Tensor float32 (N,D), torch.load(weights_only=True), bit-identical to h5 features, no coords.
TITAN mode under D: CONCH1_5/{h5/<id>.features.h5, pt/<id>.features.pt, tile_h5/<id>.patch.h5 (duplicate)}, TITAN_SLIDE/h5/<id>.features.h5 (features <f4 (768,), no coords/attrs), TITAN_SLIDE/pt/<id>.features.pt Tensor (768,). TITAN values fp16-representable. CONCH extraction ran twice.

## Reader notes
Prefer h5 features, fall back to .pt; handle |V2. Match grids on level-0 tile centres or IoU, not ¼-tile top-left (224 vs 256 px, per-contour origin, different mask).
