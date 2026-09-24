# IRIS-prep session ledger (2026-09-23)

High-level running record of the pre-port cleanup and the domain-shift analyses. Newest
status at the top of each section.

## Status

| Area | State |
|---|---|
| Archive | `archive/pre-iris-2026-09` pushed to origin (full pre-cleanup tree) |
| Clean branch | `iris` committed (49268d0) and pushed to origin |
| Acceptance | ruff ✅ · pytest 110 passed / 78 opt-in skips ✅ · `argo self-test` ✅ · `uv lock --check` ✅ |
| Docs | ledger, summary, README/CLAUDE/AGENTS updated; research landscape written |
| Emails | final drafts saved locally (`docs/collaboration/`, gitignored): HPC contact (data placement only), Mosaic TITAN contact, K43 postdoc (v3, leads with the slide-count finding); awaiting more context from Matteo on the HPC thread |
| Jobs | image stats 8/8 ✅ (804 slides; 4 known-unreadable) · image summary ✅ · embedding shift ✅ · DANN 7/7 ✅ · calibration ✅ · site smoothing ✅ |

## ⚠ Headline finding: the reference model fails on OAUTHC-prospective, and max/√n hid it

`python -m argo_deepmsi.eval.domain_shift slidecount` → `slide_count_audit_wagner.csv`.

| Site | pts / MSI-H | max/√n | mean | 1 random slide | slide-level | slide count alone (−n) |
|---|---|---:|---:|---:|---:|---:|
| ALL | 217 / 47 | **0.717** | 0.659 | **0.650** | 0.572 | 0.555 |
| OAUTHC-prospective | 81 / 17 | 0.616 | 0.451 | **0.468** | 0.439 | **0.626** |
| retrospective, MSKCC-stained | 94 / 20 | 0.776 | 0.774 | 0.773 | 0.777 | 0.520 |
| retrospective, OAUTHC-stained | 83 / 19 | 0.826 | 0.790 | 0.800 | 0.804 | 0.523 |
| LASUTH / LUTH / UITH | 13–19 / 3–4 | 0.75–0.93 | 0.79–0.93 | 0.75–0.92 | 0.69–0.92 | — |

- At OAUTHC, MSI-H patients contribute fewer slides (4.9 vs 6.1 per patient; median 1 vs 2),
  so **slide count alone scores 0.626 — as high as Wagner's 0.616 there.** Wagner's
  image signal at OAUTHC is at chance (one slide per patient 0.47; slide-level 0.44).
- The max/√n "gain" (0.659 → 0.717) is partly this slide-count shortcut. Without it, the
  reference is **~0.65 overall**: ~0.78–0.80 on MSKCC-sectioned tissue, chance on
  OAUTHC-prospective.
- **Staining lab does not matter**: the same 83 retrospective patients score 0.79 whether
  MSKCC- or OAUTHC-stained (Spearman 0.85 between arms).
- Ruled out at OAUTHC: label noise (MSIsensor scores cleanly separated, MSS ≤ 9.9 vs MSI-H
  ≥ 10; definite-only 0.434), non-tumor slides (QC-passing slides 0.48–0.50), tissue amount
  (OAUTHC slides carry more tissue than retrospective; no tercile works).
- Consistent with the image-level result that OAUTHC local processing is the largest shift
  (re-cut FID 82 vs restain 42). Open: specimen type (biopsy vs resection is missing for
  72/81 OAUTHC patients), fixation/sectioning practice, block selection — questions for the
  Nigerian team and a pathologist review of OAUTHC slides.
- Consequences: (1) report slide-count-neutral estimates (mean / one-slide) alongside
  max/√n, and treat −n as a shortcut check on every new cohort; (2) the domain-shift aim's
  core problem is OAUTHC-prospective tissue processing, not staining; (3) "smoothing across
  sites" must be judged on OAUTHC's slide-count-free AUROC.

## Cleanup decisions

- Removed: autorun loop, W1 adaptation runtime, 10 negative-result scorers, ~50 one-off
  scripts, old archive docs, results for retired analyses. All recoverable from the archive.
- Kept and grouped: `scripts/` core pipeline, `scripts/domain_shift/` (everything behind the
  collaborator's domain-shift aim), `scripts/qc/` (cohort QC producers).
- Restored after dependency checks: `vl_text_cosine` (FLEX probe needs its prototypes),
  tumor-tile head in `results/analysis/c5_phase1c/` (QC scripts load it), paired-staining and
  site-holdout write-ups (now `docs/experiments/P1`, `P2`).
- Fixed: fresh installs silently lost LazySlide (setuptools 84 removed `pkg_resources`) →
  pinned `setuptools<81`; ruff 0.16 default rules → pinned E4/E7/E9/F; every SLURM wrapper
  now cluster-neutral (`uv run --frozen`, `$SLURM_SUBMIT_DIR`), only partition names remain.

## Findings so far (primary cohort, 217 patients / 803 slides)

**Step 1 — quantify shift (embedding space, `results/analysis/domain_shift/embed_shift.csv`)**
- Site is decodable from every raw encoder: macro site AUROC 0.95–0.996; the two natural
  contrasts (same patients restained MSKCC vs OAUTHC; same stain lab, MSKCC vs OAUTHC cut)
  are separable at AUROC ≈ 1.00.
- Fréchet distance vs a patient-level permutation null: every Nigerian site is **4–10×**
  the null away from MSK-processed slides, while **MSI-H vs MSS is at the null (0.8–1.1×)**
  in every encoder. Processing dominates the embedding geometry; MSI is invisible at the
  distribution level.
- Restaining alone: 4–9× null for most encoders, but **Mascaret ≈ null (1.05×)** — the
  only raw encoder invariant to the staining lab. Sectioning (cut) shift remains 5.3× for it.
- Harmony drops the stain contrast *below* the null (0.2–0.35×) — overcorrection.
- Wagner reference per site: OAUTHC 0.616 (0.45–0.76), retrospective 0.782, small sites
  0.75–0.93 (n ≤ 19). OAUTHC specificity at the global threshold is 0.109.

**Step 2 — mitigate (nested DANN vs identical head without adversary; 7 encoders)**
- DANN is within ±0.011 AUROC of its matched control for every encoder (e.g. CTransPath
  0.577 vs 0.587, UNI2 0.580 vs 0.569, Phaet 0.630 vs 0.632, Virchow2 0.568 vs 0.573). No encoder gains.
- Best learned head overall: Phaet MLP 0.632 (0.53–0.72) — still below frozen Wagner 0.717.
- Leave-OAUTHC-out (train on all other sites, score OAUTHC): 0.37–0.60 for every variant,
  including DANN with unlabelled OAUTHC slides in the adversary. Wagner, trained on no
  Nigerian data, scores OAUTHC at 0.616 — a pretrained external model transfers at least as
  well as anything trained on our other sites.
- Macenko on OAUTHC: already a documented negative (A2: 0.61 → 0.48).

**Step 3 — calibrate (Wagner patient scores; 50 × nested 5-fold; `calibration_*.csv`)**
- Isotonic **hurts**: AUROC 0.717 → 0.694 (collapses 217 scores into ~30 ties), calibration
  slope 0.35 (overconfident), specificity at the transported sens-0.95 threshold 0.106 → 0.047.
- Platt keeps ranking (0.710), slope 0.85, ECE 0.059. MSK's own PALADIN code uses Platt.
- The operating point does not transport across sites: a threshold set on training folds
  gives OAUTHC sensitivity **0.878** (target 0.95) while retrospective specificity is 0.059.
  Calibration cannot fix this; it is a site-conditional ranking/offset problem.
- Same pattern for every learned head (isotonic ECE 0.047–0.066 vs Platt 0.030–0.045).

**Image level (`image_*.csv`; 804 slides, 32 tiles each, Inception-v3 + colour + Macenko)**
- Inception FID vs MSK-processed slides: OAUTHC 91, small sites 124–175. Same-patient
  restain 42; same-stain-lab re-cut **82**. MSI-H vs MSS tiles: **23.5**, against a split-half
  noise floor of 15.5 (KID: site 28–110 vs MSI 4.7 vs noise 0.1, ×10⁻³).
- Colour KL tells the same story: re-cut 1.9 vs restain 0.43 (null ≈ 0.03).
  → **Local sectioning/processing at OAUTHC is ~2× the shift of the staining lab.** MSK
  staining is bluer/lighter (mean B 231 vs 169 OAUTHC); OAUTHC slides are darker overall,
  consistent with thicker sections or fixation differences (hypothesis for pathology review).
- Eight colour/stain descriptors alone identify site at macro AUROC 0.88 (restain 0.98,
  re-cut 0.96).
- Colour does **not** drive Wagner's false-positive tendency: among MSS slides |Spearman|
  ≤ 0.15 for stain intensity and colour. The one signal is tissue area on OAUTHC (ρ 0.22):
  bigger tissue bags score higher — a bag-size effect, not a colour effect.

## Research inputs (see `docs/research/domain-shift-landscape.md`)

- PALADIN: H-optimus (very likely -0 via Mussel), 20×, 1536-d; per-subtype transformer; CRC
  MSI internal AUROC 0.97, external 0.93; ~71k MSK-IMPACT patients; Platt calibration in
  their code; weights unreleased. Encoder version to confirm with MSK.
- Nearest external analogue: a transformer MSI model on 197 South African CRC reached 0.91
  only after local threshold recalibration.
- With 47 positives, AUROC SE ≈ 0.046; per-site SE ≈ 0.14 at ~5 positives; +0.05 paired
  gains are under-powered → pre-register and confirm on the sealed cohort.

## Pushing past Mascaret: label-free site smoothing (done; `site_smoothing/`)

Judged on slide-count-neutral AUROC (patient mean / one random slide), per the headline finding.

| Wagner variant (label-free) | ALL mean | ALL 1-slide | OAUTHC mean | OAUTHC 1-slide | retro mean |
|---|---:|---:|---:|---:|---:|
| identity | 0.659 | 0.648 | 0.451 | 0.471 | 0.783 |
| per-site diagonal → MSK tiles (AdaBN analogue) | 0.653 | 0.643 | 0.485 | 0.496 | 0.772 |
| same, half strength | 0.652 | 0.645 | 0.466 | 0.487 | 0.775 |
| per-site diagonal → pooled tiles | 0.649 | 0.637 | 0.449 | 0.477 | 0.787 |
| per-site CORAL → MSK tiles | 0.659 | 0.644 | 0.470 | 0.477 | 0.778 |
| per-bag instance norm → MSK | 0.620 | 0.609 | 0.493 | 0.498 | 0.755 |
| patient mean + per-site Z-norm (score level) | 0.667 | — | — | — | — |
| patient mean + per-site rank (score level) | 0.672 | — | — | — | — |

- Input moment matching moves OAUTHC by ≤ +0.04 (still chance) and costs elsewhere.
  Feature-statistic alignment cannot restore signal the model does not extract from
  OAUTHC-processed tissue.
- Score-level per-site normalisation gives +0.01 pooled (removes offsets, cannot add
  within-site ranking). Cheap, label-free, worth keeping as a deployment option.
- Shift-invariant bag descriptors (within-slide standardised upper tail), nested LR:
  Mascaret 0.478 / Phaet 0.511 vs their plain means 0.599 / 0.600 — the bag-level offset
  that cancels carries MSI signal too. Negative. CTransPath: 0.495 vs mean 0.549.
- Conclusion: with the signal absent at OAUTHC, smoothing redistributes offsets but cannot
  create discrimination. The lever is upstream — what happens to OAUTHC tissue — plus a
  pretrained model that reads it (PALADIN) or in-domain adaptation on OAUTHC tiles.

## IRIS handoff

`docs/iris-runbook.md` is the ledger to give Claude Code on IRIS: phases, gates, encoder
list (CTransPath, Mascaret, Phaet first; H-optimus-0 and CONCH→TITAN at 512 px pending
collaborator answers; drop UNI2/Virchow2/PRISM), and what each email reply unblocks.
Finding while writing it: **our existing TITAN embeddings used 256 px CONCH tiles, not
TITAN's native 512 px @ 20×**, so they are non-canonical and not comparable to MSK's TITAN.

## Next

1. Acceptance rerun with new tests (site smoothing, slide-count audit); pytest temp dir and
   `argo self-test` scratch now stay inside the repo (shared-HPC rule).
2. ~~Commit `iris` + push~~ done (49268d0).
3. Cluster (CDSI vs IRIS) pending the Mosaic contact's answer on where embeddings/code live;
   IRIS needs a TheSpot request. Then follow `docs/iris-runbook.md`.

## 2026-09-24 — Split environments and extraction backends (branch `backends`)

Design: `docs/backends-plan.md`. ARGO stays the orchestrator; slide encoders run in their own
locked environments, used as documented upstream.

- **Core** (root `pyproject.toml`, `uv sync --frozen --extra dev`): no LazySlide; torch
  2.11+cu128, anndata, zarr v3, h5py, pyarrow, shapely, openslide. The `dask`, `waiv`,
  `conch`, `omiclip` extras are gone from core.
- **`envs/lazyslide`**: argo-deepmsi (editable) + LazySlide 0.12 stack, Waiv deps, dask;
  `conch`/`omiclip` are its optional extras. Torch 2.11+cu128 runs on Whitehead's CUDA 12.6
  driver (GPU smoke passed), which retires the old-conda-env workaround for the CUDA-13
  torch 2.14 lock. LazySlide commands re-exec into it (`ARGO_ENV` guard, `ARGO_NO_DISPATCH=1`).
- **`envs/mussel`**: Mussel @ `d4cfce9`, torch 2.5.1+cu121, subprocess of
  `argo extract --backend mussel`. Smoke findings (`results/analysis/backends/mussel_smoke/`):
  H-optimus-0 is `OPTIMUS`; effective tile 224 px @ 0.5 mpp; `seg_config=stain` caps at 32
  tiles/slide (unsuitable for PALADIN); `TMPDIR` must be set or Mussel writes to `/tmp`.
- **`envs/paladin`**: `bash envs/paladin/setup.sh` venv; weights MSK-internal; `argo paladin`
  is a stub.
- New commands: `argo setup`, `argo envs`, `argo compare-backends`. SLURM wrappers whose
  targets import LazySlide/wsidata/transformers/torchstain/torchvision now run through
  `uv run --frozen --project envs/lazyslide`; `scripts/extract_mussel.sh` added.
- Equivalence study: `docs/experiments/X1-backend-equivalence.md`. Rerun it when MSK
  engineering confirms the Mussel parameters (only `configs/backends/mussel-hoptimus0.toml`
  changes).
