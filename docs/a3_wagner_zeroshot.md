# A3 — Wagner et al. zero-shot generalization

**Status:** ✅ done 2026-04-22 (sbatch 9728123, 55 min on 1× A6000 with SDPA + CPU-OOM fallback)

**Scripts:** `scripts/wagner_zeroshot.py` + `scripts/wagner_zeroshot.sh`

**Reference:**
- Wagner, Reisenbüchler, West et al. "Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study." *Cancer Cell* 41(9), 2023. DOI 10.1016/j.ccell.2023.08.002
- Repo: https://github.com/peng-lab/HistoBistro
- Pre-trained weights: `CancerCellCRCTransformer/trained_models/MSI_high_CRC_model.pth` (already mirrored locally at `old/HistoBistro/CancerCellCRCTransformer/trained_models/`)

## What this tests

Western → Africa generalization without any fine-tuning. Wagner et al. trained the transformer on ~13,000 Western CRC patients from 16 cohorts (DACHS, NLCS, QUASAR, TCGA, MCO, etc.) using CTransPath tile features + a CLS-pooled 2-layer transformer aggregator. We run that exact network on our 803-slide Nigerian cohort's CTransPath features and compare the resulting AUROC to the published ~0.95 Western AUROC. The gap between them is the generalization penalty.

## Methodology

Architecture (extracted by inspecting the state dict, confirmed against `old/HistoBistro/models/aggregators/transformer.py`):

```
Transformer(
    input_dim=768,      # CTransPath feature dim
    dim=512, depth=2, heads=8, dim_head=64, mlp_dim=512,
    pool='cls',
    num_classes=1,      # single logit → sigmoid for P(MSI-H)
)
```

Checkpoint wraps weights with a `model.` prefix (pytorch-lightning convention); the loader strips it.

Inference loop (one slide at a time):
1. Open `<slide>.zarr` via `open_wsi(svs_path, store=parent)`.
2. Read `wsi.tables["ctranspath_tiles"].X` → `(n_tiles, 768)` float32.
3. Forward pass through `WagnerTransformer.eval()` → logit → sigmoid.
4. Store `p_msih` per slide.
5. Aggregate to patient level (mean probability across a patient's slides).

No fine-tuning, no gradient updates, no head swap — the Western classifier as published.

## Known caveats to report in the paper

1. **Feature-extraction protocol.** Wagner's CTransPath features were extracted from their own preprocessing pipeline; ours come from LazySlide's `zs.tl.feature_extraction(model="ctranspath")`. Normalization and tile-sampling can differ. A follow-up is to verify using their exact preprocessing, which *should* narrow any gap that's really just pipeline mismatch.
2. **Tile count per slide** varies between cohorts. Wagner padded to a fixed `num_tiles`; we pass the full tile set per slide (their classifier is tile-count invariant through attention pooling, so this is a degree of freedom, not a bug).
3. **Stain and scanner protocols** differ substantially between the Western training cohorts and the Nigerian test set. Part of the A3 gap is this expected shift; A1 already shows the foundation features themselves are largely stain-robust within-patient, so the gap attributable to site/scanner/batch effects is bounded above by the A3 number.

## Outputs

```
results/analysis/wagner_zeroshot/
├── slide_scores.csv     # slide_id, patient_id, site, stain_location, y, p_msih, n_tiles
├── patient_scores.csv   # patient_id, p_msih (mean of their slides), y, n_slides
├── metrics.json         # slide/patient AUROC+AUPRC, per-site AUROC
└── roc.png              # slide vs patient ROC curves
```

## Run

```bash
sbatch scripts/wagner_zeroshot.sh
```

Runs on a single A4000 GPU in <15 min for 803 slides (per-slide transformer pass is cheap).

## Expected paper figure

ROC curve overlay: our Nigerian cohort curves vs. Wagner's published Western AUROC (~0.95). The visual gap is the generalization penalty. Per-site AUROC table quantifies where generalization breaks down the hardest.

## Results (2026-04-22)

### Whole-cohort

| Metric | Value |
|---|---:|
| Slide-level AUROC | 0.572 |
| Slide-level AUPRC | 0.240 |
| **Patient-level AUROC** | **0.659** |
| Patient-level AUPRC | 0.386 |
| N slides | 803 |
| N patients | 217 |
| MSI-H prevalence (slide / patient) | 0.19 / 0.22 |

Patient-level (mean P(MSI-H) across a patient's slides) beats slide-level by ~9 AUROC points, and notably **beats our Phase 1 supervised baseline of 0.603** (`conch_v1.5_mean + LR`, 5-fold StratifiedGroupKFold). A Western-trained zero-shot model is outperforming a within-cohort supervised classifier at the patient level — a meaningful finding in its own right.

### Per-site (slide-level)

| Site | N | MSI-H prev | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| LASUTH | 15 | 0.20 | **0.917** | 0.833 |
| retrospective_oau | 172 | 0.22 | 0.804 | 0.603 |
| retrospective_msk | 97 | 0.21 | 0.777 | 0.639 |
| LUTH | 31 | 0.13 | 0.750 | 0.602 |
| UITH | 12 | 0.50 | 0.694 | 0.731 |
| OAUTHC | 476 | 0.17 | **0.439** | 0.160 |

### Interpretation

1. **Retrospective slides (same patients, MSK vs OAUTHC stain) give matched AUROC (0.78 vs 0.80).** This is an independent Wagner-classifier replication of A1's foundation-model stain-robustness finding: at the feature + classifier level, the two stain protocols yield equivalent MSI discrimination.

2. **OAUTHC prospective slides (60% of the cohort) are the hard case — AUROC 0.44.** The whole-cohort result of 0.572 is dragged down by OAUTHC alone. Every other site is 0.69–0.92 AUROC zero-shot, which approaches Wagner's published Western performance.

3. **The OAUTHC failure is not a stain effect.** A1 + the retrospective_oau 0.80 AUROC here both say OAUTHC-stained slides from retrospective patients predict well. What distinguishes the OAUTHC prospective slides is:
   - Different scanner / acquisition protocol than retrospective slides (possible)
   - Different tumor sampling / grading distribution (possible)
   - Different MSI-H labelling protocol (REDCap fields differ: `cmo_msi_status` for prospective vs `msi_status_mmr` for retrospective — see `data_ingestion.py`)
   - Some other batch effect we haven't yet isolated

### Next follow-ups (ranked)

1. **Audit MSI label quality in OAUTHC prospective.** If labels themselves are noisy, *any* classifier will fail there regardless of embedding quality. Compare MSI-H prevalence, IHC vs PCR labelling, and label confidence against the retrospective_oau set.
2. **Feature-space analysis.** UMAP/PHATE of ctranspath features colored by SITE — is OAUTHC prospective clearly separated from retrospective_oau in embedding space? If yes, domain-adaptation is needed. If no, labels are the culprit.
3. **Few-shot fine-tuning.** Tune the Wagner classifier on 50–100 OAUTHC slides; measure AUROC-lift on held-out OAUTHC. Likely a small-N, high-impact improvement.
4. **StainX normalization** targeted specifically at OAUTHC prospective if the feature-space is separable.
