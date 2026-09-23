# Domain-shift evidence — the three proposed steps, tested on the Nigerian cohort

Answers a proposed three-step domain-shift aim (quantify → mitigate → lock/calibrate) with
measurements on the current development cohort: **217 patients / 803 slides / 47 MSI-H**,
six processing sites, all slides imaged in Nigeria. Code: `argo_deepmsi/eval/domain_shift.py`,
`argo_deepmsi/eval/site_smoothing.py`, `scripts/domain_shift/`. Tables:
`results/analysis/domain_shift/`. Literature context: `docs/research/domain-shift-landscape.md`.

## Headline: the reference model does not read OAUTHC-prospective tissue

Slide-count-neutral evaluation of frozen Wagner/CTransPath
(`python -m argo_deepmsi.eval.domain_shift slidecount` → `slide_count_audit_wagner.csv`):

| Site | pts / MSI-H | max/√n | patient mean | 1 random slide | slide count alone |
|---|---|---:|---:|---:|---:|
| all | 217 / 47 | 0.717 | 0.659 | 0.650 | 0.555 |
| OAUTHC-prospective | 81 / 17 | 0.616 | 0.451 | 0.468 | 0.626 |
| retrospective, MSKCC-stained | 94 / 20 | 0.776 | 0.774 | 0.773 | 0.520 |
| retrospective, OAUTHC-stained | 83 / 19 | 0.826 | 0.790 | 0.800 | 0.523 |

- OAUTHC MSI-H patients contributed fewer slides, so max/√n rewards slide count; the
  image signal at OAUTHC is at chance. Slide-count-neutral pooled AUROC is ~0.65.
- Staining lab does not change performance (same patients 0.79 vs 0.79).
- Label noise, non-tumor slides and tissue amount do not explain the OAUTHC failure
  (see `docs/iris-prep-ledger.md`). Specimen type and local processing remain open.

## Two natural experiments in the cohort

| Contrast | Groups | Held fixed | Varies |
|---|---|---|---|
| **Restain** | retrospective_msk vs retrospective_oau | patients, MSKCC sectioning, scanner | staining lab (MSKCC vs OAUTHC) |
| **Re-cut** | retrospective_oau vs OAUTHC-prospective | staining lab (OAUTHC), scanner | sectioning/processing lab (MSKCC vs OAUTHC), patients |

These isolate processing factors without new data and are the strongest preliminary data
for a domain-shift aim.

## Step 1 — Quantify

**Image level** (32 tissue tiles per slide, 804 readable slides; `image_*.csv`):

| Comparison | Inception FID | KID ×10³ | RGB KL (null) |
|---|---:|---:|---:|
| noise floor (OAUTHC split-half) | 15.5 | 0.1 | — |
| MSI-H vs MSS tiles | 23.5 | 4.7 | — |
| restain (same patients) | 41.8 | 28.2 | 0.43 (0.03) |
| re-cut (same stain lab) | 82.3 | 61.2 | 1.91 (0.03) |
| OAUTHC vs MSK-processed | 90.7 | 62.8 | 4.64 (0.05) |
| small sites vs MSK-processed | 124–175 | 87–110 | 1.8–6.2 |

- Local sectioning/processing shifts images about **twice** as much as the staining lab.
- Eight colour/stain descriptors alone identify site at macro AUROC 0.88.
- Colour does not drive the reference model's false positives (|Spearman| ≤ 0.15 among MSS
  slides); only tissue area on OAUTHC correlates (ρ 0.22), a bag-size effect.

**Embedding level** (Fréchet distance in the top 32 PCs vs a patient-level permutation null;
site classifier with patient-grouped CV; `embed_shift.csv`):

| Encoder | site macro AUROC | restain FD ×null | re-cut FD ×null | MSI vs MSS FD ×null |
|---|---:|---:|---:|---:|
| CTransPath | 0.989 | 8.1 | 5.4 | 1.01 |
| UNI2 | 0.994 | 9.3 | 5.8 | 0.82 |
| Virchow2 | 0.994 | 4.3 | 6.0 | 0.83 |
| CONCH v1.5 → TITAN* | 0.974 | 3.9 | 5.5 | 0.82 |
| Phaet | 0.996 | 6.7 | 5.3 | 0.97 |
| **Mascaret** | 0.985 | **1.05** | 5.3 | 0.84 |
| Virchow2 → PRISM | 0.952 | 3.7 | 5.3 | 0.99 |

\*Our TITAN used 256 px CONCH tiles, not TITAN's native 512 px @ 20×.

- Every site is 4–10× the null away from MSK-processed slides; **MSI-H vs MSS is at the
  null** in every encoder. Processing dominates the embedding geometry.
- Mascaret is the only encoder whose embedding ignores the staining lab; none ignores
  sectioning.
- FID on ImageNet features agrees in direction but not magnitude with the model's own
  feature space; for model-relevant shift, measure in the deployed encoder's space.

**Baseline performance** (frozen Wagner/CTransPath): see the headline table — use
slide-count-neutral pooling (patient mean or one slide per patient) as the primary per-site
estimate; small sites (≤ 4 MSI-H each) are not interpretable individually.

## Step 2 — Mitigate

| Method | Result | Source |
|---|---|---|
| Macenko normalisation of OAUTHC to a retro-OAU reference, re-extracted | OAUTHC 0.61 → **0.48** (destroys signal) | A2 |
| Feature-space ComBat / site alignment | site axis erased (site AUROC 1.00 → 0.08), OAUTHC MSI unchanged | A1 |
| Harmony | modest OAUTHC gain on the historical non-nested cohort; over-corrects restain below null here | A0, `embed_shift.csv` |
| **Multi-source DANN** (gradient reversal over site, λ chosen in inner CV) vs identical head without adversary, 7 encoders | within ±0.011 AUROC of the control for every encoder | `dann_*/dann_nested.csv` |
| DANN trained on other sites, scored on OAUTHC, ± unlabelled OAUTHC in the adversary | 0.37–0.60 for all variants; frozen Wagner (no Nigerian training) scores OAUTHC 0.616 | `dann_*/dann_leave_oauthc_out.csv` |
| CycleGAN colour transfer | not run: image-to-image GANs can hallucinate or erase morphology (Cohen et al. 2018), and pixel normalisation already removed MSI signal here | — |

With 47 positives, adversarial invariance has too little label signal to anchor on, and
the site axis is entangled with the MSI-relevant variance.

## Step 3 — Lock and calibrate (nested, 50 × 5-fold, patient level)

| Wagner scores | AUROC | ECE | slope | spec @ transported sens-0.95 threshold |
|---|---:|---:|---:|---:|
| uncalibrated | 0.717 | — | — | 0.106 |
| Platt | 0.710 | 0.059 | 0.85 | 0.106 |
| isotonic | 0.694 | 0.060 | 0.35 | 0.047 |

- Isotonic overfits at this n (collapses 217 scores to ~30 ties, slope 0.35) and halves
  specificity; Platt is the safe choice (and is what PALADIN's code uses).
- The operating point does not transport across sites: OAUTHC sensitivity 0.878 at a
  threshold targeting 0.95, retrospective specificity 0.059. This is a site-conditional
  offset, which calibration cannot correct.
- Nested five-fold CV is necessary but not sufficient: with 47 positives the AUROC SE is
  ≈ 0.046, so the lock should be confirmed on newly accrued patients.

## Label-free site smoothing (`site_smoothing/`)

| Wagner variant | all (patient mean) | OAUTHC (patient mean) |
|---|---:|---:|
| identity | 0.659 | 0.451 |
| per-site diagonal moment matching → MSK tiles (AdaBN analogue) | 0.653 | 0.485 |
| per-site CORAL → MSK tiles | 0.659 | 0.470 |
| per-bag instance normalisation | 0.620 | 0.493 |
| per-site score rank-normalisation | 0.672 | — |

Shift-invariant bag descriptors (nested LR) fell below plain means for every encoder
(Mascaret 0.478 vs 0.599, Phaet 0.511 vs 0.600, CTransPath 0.495 vs 0.549). Smoothing
moves site offsets but cannot create discrimination the model does not extract.

## Recommended protocol for the aim

1. **Quantify** in the deployed model's feature space (site AUROC, permutation-calibrated
   FD/KID) plus image-level colour/FID, always with the restain and re-cut contrasts and an
   MSI-vs-MSS yardstick.
2. **Baseline** with pretrained external models first (Wagner; PALADIN if available), per
   site with bootstrap CIs, using slide-count-neutral pooling and a slide-count-only
   control.
3. **Mitigate** only against a matched control under nested CV; prioritise label-free
   site smoothing and acquisition-robust encoders over pixel normalisation or adversarial
   training.
4. **Calibrate** with Platt; report per-site realised sensitivity at a fixed operating
   point; confirm on a sealed temporal set.
