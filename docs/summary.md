# ARGO-DeepMSI — No-Training Scorer Leaderboard

**Cohort:** Nigerian colorectal cancer, 803 slides → **198 patients** after
pathologist QC exclusion (`results/data/problem_slides.csv`, 514 slides
flagged: 502 non-tumor, 50 out-of-focus, 28 stain-faded).

**Constraint:** scorers may train on our 198-patient cohort but not on
external slide sets. Wagner zero-shot (pretrained on ~13K Western
patients, applied here without fine-tuning) is the reference baseline.

## Leaderboard (patient AUROC, QC-clean, with proper refit)

Every scorer that trains a head on our cohort was **refit** with the
flagged slides removed from its training data (not just filtered at
eval). Zero-shot scorers (Wagner, vl_text_cosine, transductive) are
filter-only since they have no parameters to refit.

| Rank | Scorer                  | Clean | Dirty | Trained on us | Resolution |
|------|-------------------------|-------|-------|---------------|------------|
| 1    | wagner_zeroshot         | 0.710 | 0.717 | no       | slide      |
| 1    | calibrated_pool         | 0.710 | 0.717 | no       | patient    |
| 3    | slide_attention_mil     | 0.669 | 0.665 | yes      | patient    |
| 4    | score_fusion            | 0.578 | 0.601 | yes      | slide      |
| 5    | vl_text_cosine          | 0.562 | 0.598 | no       | slide      |
| 6    | transductive_smoothing  | 0.555 | 0.582 | no       | slide      |
| 7    | nuclear_morphology      | 0.513 | 0.529 | yes      | slide      |

Reproducible: `python -m argo_deepmsi.eval.qc_comparison`. Raw artefacts
under `results/comparison/`:

- `leaderboard.csv` — the table above
- `per_site_clean.csv` — every scorer × site
- `auroc_bar_overall.png` — bar chart with Wagner baseline
- `roc_overlay.png` — top-5 ROC curves
- `auroc_grid_per_site.png` — heatmap scorer × site

## Headline finding

`wagner_zeroshot` and `calibrated_pool` **tie at 0.710** on the clean
cohort. An earlier post-hoc-filter version of this table showed
calibrated_pool at 0.729 — that was an artifact of the MSS null curve
being built from a dirty MSS pool (i.e. flagged MSS slides inflated the
null Emax, making the `max − Emax_MSS` subtraction look better than it
was). Rebuilding the null curve from clean MSS slides eliminates the
gap.

In other words: **on the QC-clean cohort, no aggregator, no head, no
text-cosine, no morphology feature set beats applying Wagner's
pretrained transformer and taking max/√n over a patient's slides.**
Pathologist QC nets a small overall hit (Wagner clean 0.710 vs dirty
0.717) — explained by removing some informative slides along with the
non-tumor ones — but the gap between methods stays roughly the same
shape: the trained-on-our-cohort variants don't catch up.

`slide_attention_mil` at 0.682 remains the strongest cohort-trained
head — 2.8 pt behind Wagner. The attention head learns useful
interpretable weights (it correctly ignores Wagner on OAUTHC) but
cannot find a replacement signal in foundation-model embeddings that
encode site > biology.

## Per-site (patient AUROC, clean cohort)

Latest numbers — regenerate with `python -m argo_deepmsi.eval.qc_comparison`
and read `results/comparison/per_site_clean.csv`. The top half is the
no-training and zero-shot tier; the bottom half is cohort-trained.

Notes:

- OAUTHC remains the failure cohort across every scorer; `calibrated_pool`'s
  0.61 is the best any approach gets and is still ~30 pts below
  retrospective-site AUROC.
- The `retrospective_oau` column is empty for patient-resolution scorers
  because their per-patient aggregation tables collapse multi-site
  patients to a single "site" label (an artifact of the legacy
  `calibrated_aggregation.py` / `c5_phase2.py` output schema, not a
  scorer issue).
- Small-n cells (LASUTH n=11, LUTH n=16, UITH n=10) are noisy; treat
  ±0.10 AUROC swings as within-sample variance.

## Per-scorer details

- `docs/scorers/wagner_zeroshot.md` — pretrained Western baseline
- `docs/scorers/calibrated_pool.md` — **headline winner**
- `docs/scorers/slide_attention_mil.md` — best cohort-trained head
- `docs/scorers/vl_text_cosine.md` — VL zero-shot (A1)
- `docs/scorers/transductive_smoothing.md` — kNN label propagation on vl_text_cosine scores
- `docs/scorers/nuclear_morphology.md` — NuLite + LR (D)
- `docs/scorers/score_fusion.md` — late-average over embeddings

## Scorer interface

Uniform contract in `argo_deepmsi/scorers/base.py`:

```python
class Scorer(ABC):
    name: str
    description: str
    needs_training_on_our_data: bool      # external training is banned
    resolution: Literal["slide", "patient"]
    score_columns: list[ScoreColumn]      # multiple variants per scorer OK

    def fit(train_df) -> None: ...        # default no-op
    def compute_batch(slide_table) -> pd.DataFrame: ...
    def predict_batch(use_cache=True) -> pd.DataFrame: ...
    def publish() -> Path: ...            # writes canonical CSV + metadata
```

Add a new scorer:

1. Subclass `Scorer` in `argo_deepmsi/scorers/<name>.py`
2. Implement `compute_batch` returning the canonical schema
3. `register(name, NewScorer)` at module bottom
4. Add `from . import <name>` in `argo_deepmsi/scorers/__init__.py`
5. Re-run `python -m argo_deepmsi.eval.qc_comparison`

`needs_training_on_our_data=True` is allowed (sklearn / attention heads
fit on our 198 patients). External pretraining is permitted only for
frozen foundation-model embedders; no external slide-level training.

## Roadmap (what's not yet attempted in this no-training space)

- **Multi-scorer fusion** — `calibrated_pool` + `slide_attention_mil`
  weighted by site-specific reliability (Wagner ρ ≈ 0 on OAUTHC,
  ρ ≈ 0.3 on retrospective sites)
- **Exemplar memory** — kNN to held-out MSI-H tile exemplars (the
  no-training analog of FLEX feature alignment)
- **Better VL prompts** — search-tuned prompt sets on a val fold;
  the initial `vl_text_cosine` prompt set was hand-curated, not search-tuned
- **TCGA tile features** — *would* require external training; explicitly
  out of scope until the user lifts that constraint
