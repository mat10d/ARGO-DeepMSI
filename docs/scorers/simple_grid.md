# simple_grid

**Module:** `argo_deepmsi.scorers.simple_grid`
**Resolution:** slide
**Trained on our cohort:** yes (small head, 5-fold patient-grouped CV)

## Mechanism

Sweeps lightweight classifier heads on each frozen embedding and picks
the best-performing config by patient AUROC (max/√n). All combinations
of:

    classifier      ∈ {logistic_regression, random_forest, xgboost}
    embedding       ∈ {conch_v1.5_mean, conch_v1.5_titan, ctranspath_mean,
                       uni2_mean, virchow2_mean, virchow2_prism}
    representation  ∈ {raw, pca100}

= **36 configs** per cohort. Each config is fit 5-fold under
`StratifiedGroupKFold(patient_id)`, class-balanced. The OOF predictions
of the best config become the primary slide score; the full sweep table
is dumped to `grid.csv`.

"Simple" because everything here is a small head on frozen features —
no foundation-model finetuning, no torch, no GPU. ~50 min CPU per cohort.

## Inputs

- Per-embedding `results/embeddings/<name>/embeddings.npy + metadata.csv`
- `clinical_table.csv` for labels

## Outputs

- `results/scorers/simple_grid/slide_scores.csv` — OOF p_msih of best config
- `results/scorers/simple_grid/grid.csv` — all 36 configs ranked by patient AUROC
- `results/scorers/simple_grid/best_config.txt` — one-liner of the winner

## Results

Best config (refit on QC-clean cohort):

    embedding=conch_v1.5_titan
    classifier=logistic_regression
    representation=pca100
    patient_auroc=0.6697

| metric                  | full cohort | QC-clean (refit) |
|-------------------------|-------------|------------------|
| patient AUROC (best)    | 0.622       | **0.670**        |

**+4.8 pts on the clean refit — the largest QC-driven lift of any
scorer.** Three of the top-3 configs share the same embedding
(`conch_v1.5_titan`); LR + PCA-100 narrowly beats RF + PCA-100. Tied
with `slide_attention_mil` (0.669) at rank 3 in the leaderboard, despite
a much simpler architecture (no learned attention pooler, no torch).

Top-5 clean configs:

| embedding         | classifier          | repr   | patient AUROC |
|-------------------|---------------------|--------|---------------|
| conch_v1.5_titan  | logistic_regression | pca100 | **0.670**     |
| conch_v1.5_titan  | logistic_regression | raw    | 0.649         |
| conch_v1.5_titan  | random_forest       | pca100 | 0.646         |
| virchow2_prism    | random_forest       | raw    | 0.625         |
| virchow2_prism    | logistic_regression | raw    | 0.625         |

## Interpretation

Two observations matter:

1. **The TITAN slide encoder (768-d, contrastively trained on CONCH v1.5
   patches) is the only embedding that lets a linear head ≥ 0.65.** Per-tile
   embeddings (conch_v1.5_mean, virchow2_mean) max out at ~0.60 with the
   same heads. The slide-level contrastive pretraining matters more than
   the choice of classifier.

2. **A 5-fold logistic regression on PCA-100 features ties a torch-trained
   attention-MIL** at the patient level. That's evidence the architectural
   complexity of `slide_attention_mil` is overkill for a 200-patient cohort.

Still 4.0 pts under `wagner_zeroshot` (0.710), so the lightweight-head
direction doesn't dethrone the zero-shot baseline. But it's the strongest
trained-on-our-cohort scorer in the leaderboard.

## Caveats

- 36-config grid is dense but not exhaustive — we don't sweep regularization
  strength, n_estimators, or learning rate. A Bayesian optimisation pass
  could squeeze 1–2 more pts.
- SVM_RBF was deliberately dropped (slow on 768-d × 200, never wins).
- The best config is "logistic regression on PCA-100 of TITAN slide
  embeddings" — borderline trivial. That this trivial recipe beats every
  other no-training-on-external-data approach except Wagner zero-shot is
  itself a meaningful negative result for the more complex baselines.
