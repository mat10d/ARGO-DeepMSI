# score_fusion

**Module:** `argo_deepmsi.scorers.score_fusion`
**Resolution:** slide
**Trained on our cohort:** yes (component LR heads, fit 5-fold)

## Mechanism

Late-average fusion of two single-embedding logistic-regression OOF
heads, both trained 5-fold patient-grouped on our cohort:

    p_msih_fused = 0.5 · p(conch_v1.5_mean) + 0.5 · p(virchow2_mean)

Members chosen from the leaderboard in `fusion_results.csv`:
those two embeddings give the highest single-model AUROC under
patient_cv, and they're sufficiently uncorrelated that averaging adds
~1 pt over either alone.

Patient aggregation = max / √n.

## Inputs

- `results/analysis/fusion/per_model_oof.csv` — per-slide OOF preds
  from each embedding × regime

## Outputs

- `results/scorers/score_fusion/slide_scores.csv`
- Legacy raw output: `results/analysis/fusion/`

## Results

| metric            | full cohort | QC-clean (refit) |
|-------------------|-------------|------------------|
| slide AUROC       | 0.586       | tbd              |
| patient AUROC     | 0.601       | 0.578            |

Underperforms `wagner_zeroshot` (0.710). The per-embedding LR heads
each individually pull 0.52–0.58 AUROC; late-averaging two weak heads
does not produce a strong head. QC refit shifts the number by ~+2 pts
vs post-hoc filter (0.560 → 0.578) — the dirty per-embedding folds
were learning some noise.

## When to consider revisiting

Stacking (`stack_meta_lr:BASE` in `fusion_results.csv`) hit 0.580 under
LOSO but 0.586 under patient_cv. The full stacking head exposes more
variance — worth re-running through the comparison harness as an
alternate `score_fusion` variant if/when fusion becomes a serious
candidate.
