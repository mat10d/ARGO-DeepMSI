# transductive_smoothing

**Module:** `argo_deepmsi.scorers.transductive_smoothing`
**Resolution:** slide
**Trained on our cohort:** no (graph propagation, no parameters)

## Mechanism

Smooths the per-tile MSI vs MSS scores from `vl_text_cosine` through a
per-slide kNN affinity graph in CONCH v1.5 feature space, then
aggregates to a slide score. The hypothesis (Zanella et al. 2024 —
Histo-TransCLIP): visually similar tiles should agree on their VL
score; smoothing kills isolated outliers and amplifies regional
consensus.

Per slide:

1. Compute raw per-tile score `s0[i] = cos(x_i, MSI) − cos(x_i, MSS)`.
2. Build cosine kNN graph (k = 10) on L2-normalised CONCH v1.5
   features (brute force, sklearn).
3. Form row-stochastic transition matrix W and iterate
   `s ← (1 − α) s0 + α W s` for T iterations (α = 0.7, T = 5).
4. Aggregate smoothed `s` per slide via mean / top-k / max.

Patient aggregation = max / √n.

## Inputs

- Per-slide `tables/conch_v1.5_tiles`
- Same TITAN text prototypes as `vl_text_cosine`

## Outputs

- `results/scorers/transductive_smoothing/slide_scores.csv`

## Results

| variant         | slide AUROC | patient AUROC |
|-----------------|-------------|---------------|
| `smooth_max` (primary) | 0.537 | 0.582 |
| `smooth_topk`   | 0.529       | 0.583         |
| `smooth_mean`   | 0.469       | 0.548         |
| `raw_topk` (control)   | 0.535 | 0.608         |
| `raw_max` (control)    | 0.556 | 0.598         |

QC-clean patient AUROC: **0.555** (rank 6).

**Smoothing strictly hurts** vs the raw `vl_text_cosine` scores:
patient AUROC drops 1–3 pts across every aggregation. The OAUTHC
slide-level lift that `vl_text_cosine` produced (0.604 → 0.580 after
smoothing) is the strongest signal that the graph prior is wrong here.

## Interpretation

The smoothing prior — "visually similar tiles should agree" — is the
wrong inductive bias for this cohort. `vl_text_cosine`'s `tile_max`
was catching rare, isolated high-confidence positive tiles. Label
propagation washes those out.

That's mechanistically consistent with the C5 finding that MSI-H signal
in this cohort is sparse and concentrated in a few tiles, not in
contiguous regions where graph smoothing helps.

## Decision

Ruled out as standalone. Held as the negative-result reference for any
future write-up.

## Caveats

- Tested one `(k, α, T)` point. A scan could find a regime where
  smoothing ties `raw`, but the Histo-TransCLIP claim was that
  smoothing *helps* zero-shot VL — that doesn't replicate here.
- Brute-force kNN per slide is the bottleneck (~30 s/slide). HNSW
  would speed it up but doesn't change the conclusion.
