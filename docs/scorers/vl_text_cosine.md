# vl_text_cosine

**Module:** `argo_deepmsi.scorers.vl_text_cosine`
**Resolution:** slide
**Trained on our cohort:** no (zero-shot text-image cosine)

## Mechanism

Per slide we compute per-tile cosine similarity between L2-normalised
`conch_v1.5_tiles` features and class-prototype text embeddings, then
aggregate to a slide score.

- Text encoder: **TITAN.encode_text** (CONCH v1.5 paired text tower,
  768-d, L2-normalised). MahmoodLab/TITAN on HuggingFace.
- Class prototypes: 7 MSI-H prompts + 6 MSS prompts (curated from
  pathology literature), mean-pooled per class and re-normalised.
- Per-tile score: `s_i = cos(x_i, MSI) − cos(x_i, MSS)`
- Slide-level variants (all four columns in `slide_scores.csv`):
  - `tile_max` (primary) — max over tiles
  - `tile_topk` — mean over top-k (k=64)
  - `tile_mean` — mean over all tiles
  - `slide_score` — same formula on the TITAN slide embedding

TITAN's text↔slide alignment is the one it was trained for; its
text↔patch alignment is heuristic. Both are computed for comparison.

## Inputs

- Per-slide `tables/conch_v1.5_tiles` in each zarr
- Existing `results/embeddings/conch_v1.5_titan/embeddings.npy` (slide-level)
- HF auth for `MahmoodLab/TITAN` (gated)

## Outputs

- `results/scorers/vl_text_cosine/slide_scores.csv`
- `results/scorers/vl_text_cosine/prompts.json`
- `results/scorers/vl_text_cosine/text_embeddings.npy`

## Results

| variant              | slide AUROC | patient AUROC |
|----------------------|-------------|---------------|
| `tile_max` (primary) | 0.556       | 0.598         |
| `tile_topk`          | 0.535       | 0.608         |
| `tile_mean`          | 0.471       | 0.551         |
| `slide_score`        | 0.414       | 0.461         |

QC-clean patient AUROC: **0.562** (rank 4, see `docs/summary.md`).

`slide_score` collapsing below chance argues TITAN's slide-level text
alignment carries the wrong axis on our cohort — likely a
Western-vs-this-cohort confound rather than MSI vs MSS biology. The
tile-level variants bypass TITAN's slide encoder and salvage modest
signal.

## OAUTHC lift

`tile_max` is the only scorer (besides `wagner_zeroshot` itself) that
gets OAUTHC slide AUROC > 0.5 in the full-cohort eval (0.604 vs
Wagner's 0.439). After QC exclusion this gap closes; on the clean
cohort `tile_max` and Wagner are close to par.

## Caveats

- Prompt set is literature-derived but not search-tuned. Swapping the
  prompt family (architectural vs cellular) could shift OAUTHC AUROC
  by ±0.05.
- TITAN was trained predominantly on Western tissue. Null result
  doesn't rule out the VL hypothesis — would need a tile-text-aligned
  model trained on diverse cohorts.
