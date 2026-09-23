# ARGO-DeepMSI — Current state

**Primary estimand (C1/V1, frozen 2026-08-03):** 217 patients / 803 feature-complete slides /
47 MSI-H. Every experiment ever run, with cohort, validation design, verdict, and code location,
is in [`negative-results-ledger.md`](negative-results-ledger.md).

| result | patient AUROC (95% CI) | notes |
|---|---:|---|
| **Wagner/CTransPath zero-shot, max/√n** | **0.717 (0.631–0.799)** | spec 0.141 @ sens 0.95; OAUTHC 0.616; the reference and no-regression floor — **partly a slide-count artifact, see below** |
| Wagner/CTransPath zero-shot, patient mean (slide-count-neutral) | 0.659 | OAUTHC **0.451** (chance); retrospective 0.77–0.79; one random slide/patient 0.650 |
| Nested four-encoder linear probe (V1) | 0.530 (0.429–0.636) | honest cohort-trained baseline; OAUTHC 0.510 |
| Waiv Phaet + Mascaret, nested probe (B2) | 0.619 (0.524–0.712) | first frozen encoder above the nested baseline; OAUTHC 0.594 |
| Best W1 adaptation (W1-6 residual adapter) | 0.661 | paired Δ −0.055 (−0.132 to +0.015) vs frozen Wagner; not promoted |
| Historical 181-patient hard-QC sensitivity set | 0.713 (0.625–0.795) | QC restoration delta +0.019 (−0.006 to +0.047) |

**Slide-count shortcut (2026-09-23; `docs/domain-shift-evidence.md`).** At OAUTHC-prospective,
MSI-H patients contributed fewer slides; slide count alone scores 0.626 there, equal to Wagner's
max/√n 0.616, while Wagner's image signal at OAUTHC is at chance (slide-level 0.44). Staining
lab does not matter (same retrospective patients 0.79 vs 0.79 across MSKCC/OAUTHC stain). Report
slide-count-neutral pooling as primary and a slide-count-only control on every cohort.

**Error anatomy (E1–E4).** At sens 0.95, Wagner makes **146 false positives vs 2 false
negatives**. The failure is specificity. The 148 errors split into four enrichment-gated causes:

| cause | % of errors | lever |
|---|---:|---|
| borderline-score | 39% | operating point / selective abstention |
| unexplained confident FP | 29% | new signal or abstention; attention on tumor is normal (E3), so not another head |
| label-suspect (enriched 1.66×) | 22% | re-adjudication worklist (written) |
| low-tumor-content (enriched 2.80×) | 11% | re-tile / QC-refix worklist (written) |

Paired MSK/OAU scans flip only 4.8% of calls, so acquisition is mostly not the driver. **Do not
add aggregation or attention heads on frozen CTransPath.** Reproduce with
`python -m argo_deepmsi.eval.error_anatomy` and `scripts/qc/error_anatomy_c.sh`.

> **Data freshness:** A read-only live REDCap audit on 2026-08-03 found 17 newly labelled
> clinical patients, but none has a slide in the current corpus. All 217 benchmark patients and
> every audited label/assay field are unchanged. The benchmark is current; the broader clinical
> snapshot should be refreshed before those patients acquire slides.

## Current path forward

1. **Treat this as a data/label problem, not another head search.** Reconcile the 40 prospective
   `Indeterminate` cases now encoded as MSS and audit the OAUTHC-vs-retrospective assay pathway.
   The full-cohort label-certainty sensitivity is already frozen; adjudicated labels are the next
   information-bearing input.
2. **Give better encoders a pretrained aggregator.** Waiv beats the nested baseline under
   low-capacity heads but collapses under a from-scratch transformer (0.542). The next
   representation step is distillation from the frozen Wagner teacher or an externally
   pretrained slide head (see [`future-roadmap.md`](future-roadmap.md)), not cold training.
3. **Prioritize validation over optimization.** Freeze Wagner max/√n and evaluate it on the final
   larger Nigerian cohort before that cohort is used for any fitting, selection, or thresholding.
4. **Do not continue the exhausted branches:** feature-level batch correction, stain
   normalization, learned aggregation, late fusion, prompt tuning, supervised Wagner/CTransPath
   adaptation, and non-nested encoder grids have already produced negative or selection-biased
   results.

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
4. Run it with `argo scorers run <name>` or add a `[[scorer]]` job to an experiment TOML

`needs_training_on_our_data=True` is allowed only with patient-grouped, fold-local fitting.
External pretraining is permitted only for frozen foundation-model embedders; no external
slide-level training. Per-scorer notes: `docs/scorers/wagner_zeroshot.md`,
`docs/scorers/calibrated_pool.md`, `docs/scorers/vl_text_cosine.md`.

## Historical (non-nested, superseded)

The leaderboard below is the April–July 2026 **198-patient eCRF-clean** board. Every
cohort-trained row chose its configuration on the OOF predictions it reports, so none is a
confirmatory estimate (V1). It is kept only for provenance. Most scorers in it were removed from
`iris`; their code and pages are on the archive branch `archive/pre-iris-2026-09`, and every row
is summarised in the ledger.

| scorer | clean | dirty | trained on us | code |
|---|---:|---:|---|---|
| wagner_zeroshot | 0.710 | 0.717 | no | iris |
| calibrated_pool | 0.710 | 0.717 | no | iris |
| simple_grid | 0.670 | 0.622 | yes | archive |
| slide_attention_mil | 0.669 | 0.665 | yes | archive |
| score_fusion | 0.578 | 0.601 | yes | archive |
| vl_text_cosine | 0.562 | 0.598 | no | iris |
| transductive_smoothing | 0.555 | 0.582 | no | archive |
| nuclear_morphology | 0.513 | 0.529 | yes | archive |

- An earlier post-hoc-filter board showed calibrated_pool at 0.729. That came from building the
  MSS null curve on a dirty MSS pool; on clean MSS slides it ties Wagner.
- On the 181-patient hard-QC board (Q4) the champion was 0.7127 and OAUTHC 0.579. That cohort is
  now only a sensitivity analysis.
- `python -m argo_deepmsi.eval.qc_comparison` now builds the unified primary-cohort board over
  the scorers registered on `iris`. Rows for removed scorers can only be reproduced from the
  archive branch.
