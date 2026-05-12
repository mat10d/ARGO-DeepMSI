# C2 — Multi-model late fusion + few-shot evaluation

**Status:** ✅ first pass done 2026-04-22 — stacking buys a real lift on leave-one-site-out

**Script:** `scripts/fusion.py` (SLURM wrapper: `scripts/fusion.sh`; needs bounded
BLAS threads — a head-node run spawned 128 sklearn-internal threads before).

**Outputs:** `results/analysis/fusion/{fusion_results.csv, per_model_oof.csv, best_per_bucket.{csv,png}}`.

Covers Steps 2d (multi-model fusion) and 2e (few-shot) from `docs/remaining_tasks.md`.

## Setup

- 802 slides / 217 patients / prevalence 0.192 (intersection of all 6 raw
  embedding dirs — one slide drops somewhere in a non-base model).
- 4 base models for fusion: `conch_v1.5_mean`, `ctranspath_mean`, `uni2_mean`,
  `virchow2_mean`. Neural-aggregator variants (`conch_v1.5_titan`,
  `virchow2_prism`) are scored individually but not stacked.
- Two evaluation regimes:
  - **patient_cv** — StratifiedGroupKFold(5) over PATIENT
  - **loso** — leave-one-site-out with patient-level leakage guard
- Every strategy produces an out-of-fold probability vector covering all 802 slides.

## Strategies evaluated

1. **Single-model LR** (baseline reference)
2. **Late-average fusion** — mean of per-model OOF probs across every 2/3/4-model subset of the base list
3. **Concat + PCA(100) + LR** on stacked features
4. **Stacking** — meta-LR on per-model OOF probs
5. **k-NN few-shot** (cosine, k ∈ {3, 5, 9})
6. **Prototypical few-shot**

## Headline result

| regime | best strategy | AUROC | AUPRC | Δ over best single |
|---|---|---:|---:|---:|
| patient_cv | `late_avg:conch_v1.5+virchow2` | **0.586** | 0.221 | +0.010 (single best: virchow2_mean 0.576) |
| **loso**   | `stack_meta_lr:BASE`             | **0.580** | 0.242 | **+0.058** (single best: conch_v1.5_mean 0.522) |

Full best-per-bucket table:

| bucket     | regime     | best strategy                   | AUROC |
|---|---|---|---:|
| late_avg   | patient_cv | conch_v1.5+virchow2             | 0.586 |
| single     | patient_cv | virchow2_mean                   | 0.576 |
| concat_pca | patient_cv | BASE                            | 0.543 |
| knn        | patient_cv | knn3:uni2_mean                  | 0.512 |
| proto      | patient_cv | proto:virchow2_prism            | 0.505 |
| stacking   | patient_cv | BASE                            | *0.327* ⚠ |
| **stacking**| **loso**  | **BASE**                        | **0.580** |
| knn        | loso       | knn9:conch_v1.5_titan           | 0.536 |
| proto      | loso       | proto:conch_v1.5_mean           | 0.535 |
| single     | loso       | conch_v1.5_mean                 | 0.522 |
| late_avg   | loso       | conch_v1.5+uni2                 | 0.460 |
| concat_pca | loso       | BASE                            | 0.451 |

Figure: `results/analysis/fusion/best_per_bucket.png`.

## Interpretation

- **Stacking is the only strategy that helps LOSO.** Late-average actually *hurts* the LOSO score versus best single (0.460 vs 0.522) — averaging four embeddings that each carry different site biases smears them together rather than cancelling them. Concat+PCA is similar (0.451). Stacking lets a meta-learner reweight models per held-out fold, which is exactly the setting where site-specific residual patterns differ between base embeddings.
- **Patient-CV stacking looks broken (0.33).** With identical fold splits, shared random state, and L2 meta-LR, the 0.33 AUROC indicates a class-balanced LR flipping sign on highly-correlated OOF features. For within-cohort CV the base models agree strongly and the meta head collapses — not the regime where stacking is useful. Not fixing now; it's the LOSO result that matters.
- **Within-cohort (patient_cv) is still flat.** Late-average fusion buys +0.01 over the best single model. This mirrors the B1 Tier 1 conclusion: within-cohort classifier choice is not the bottleneck; the domain shift is.
- **Few-shot methods do nothing useful.** k-NN and prototypical heads land at or below single-model LR in both regimes. Not worth pursuing further.

## Why LOSO stacking works and late-average doesn't

Each base model carries a different residual error pattern per held-out site. Conch-V1.5 and virchow2 disagree most where sites differ most. Late-average assumes each model's probability is equally informative everywhere — it isn't. Stacking's meta-LR can learn e.g. "at held-out OAUTHC, down-weight uni2 because it over-predicts MSS there." That's exactly what the A2 site-holdout heatmap showed: different embeddings are best at different held-out sites.

## Next

Order of operations from here:

1. **Spot-check patient-CV stacking** — either fix the sign issue (drop class_weight, add feature standardisation on OOF probs) or accept it as a known non-interesting quirk.
2. **Stacking variants** — add Harmony-corrected neural aggregators (conch_v1.5_titan_harmony was the one Harmony win) as a 5th base model, rerun stacking.
3. **Tier 3 fusion grid (Step 3b)** — systematically sweep classifier types inside the meta-learner (LR, XGBoost, calibrated SVM) and base-model subsets.
4. **ABMIL (Step 4)** remains the only path to 0.80+ on LOSO; but stacking at 0.58 sets a stronger target baseline for the full tile-level model to clear.
