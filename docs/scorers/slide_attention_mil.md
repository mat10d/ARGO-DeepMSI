# slide_attention_mil

**Module:** `argo_deepmsi.scorers.slide_attention_mil`
**Resolution:** patient
**Trained on our cohort:** yes (attention head, 5-fold patient-grouped)

## Mechanism

Small attention-MIL head trained on our cohort: each patient's bag is
all of that patient's slide-level frozen-embedding features
(default: `virchow2_mean`). A learned attention pooler combines slides
into one patient embedding, then a linear head produces `p_msih`.

Trained 5-fold under `StratifiedGroupKFold(patient_id)` with
class-weighted BCE. No external pretraining beyond the frozen embedder.

## Architecture

    z_i  = SlideEmbedding(slide_i)           # frozen (virchow2_mean)
    α_i  = softmax(MLP(z_i))                  # learned attention
    p    = Linear(Σ_i α_i z_i)                # patient logit
    p_msih = sigmoid(p)

Tested across 4 embeddings × {raw, +n_slides feature, +wagner_p feature}
in `c5_phase2/ablation_results.csv`.

## Inputs

- `results/embeddings/<embedding>/embeddings.npy` (per-slide features)
- `clinical_table.csv`

## Outputs

- `results/scorers/slide_attention_mil/patient_scores.csv` (best config)
- `results/scorers/slide_attention_mil/attention_weights.csv` — per-slide
  attention coefficients (interpretability hook)
- Legacy raw output: `results/analysis/c5_phase2/`

## Results

| metric              | full cohort | QC-clean (refit) |
|---------------------|-------------|------------------|
| patient AUROC       | 0.665       | **0.669**        |

QC-clean refit barely moves the needle (+0.4 pt) and still trails
`wagner_zeroshot` / `calibrated_pool` (both 0.710). An earlier
post-hoc-filter version had this at 0.682 — that was eval-only filtering
of the dirty-cohort OOFs, not a true clean retrain.

On OAUTHC specifically the attention head fails — OAUTHC patient AUROC
drops below chance, mirroring the C5/Phase-2 finding that learned slide
attention cannot beat max/√n.

## Why this matters

This is the headline "small head trained on our cohort" data point.
That a learned attention head loses to a zero-parameter max/√n
aggregator on the same Wagner scores is the strongest evidence we have
that the failure mode is at the *representation* level (foundation
model embeddings encode site > biology on Nigerian tissue), not at the
*aggregation* level.

The attention weights themselves carry useful interpretability signal —
the model learns to ignore Wagner on OAUTHC (ρ = −0.05) and trust it
on retrospective sites (ρ = +0.28–0.30), then can't find an alternative
MSI cue.
