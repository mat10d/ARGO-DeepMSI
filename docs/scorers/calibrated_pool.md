# calibrated_pool

**Module:** `argo_deepmsi.scorers.calibrated_pool`
**Resolution:** patient
**Trained on our cohort:** no (closed-form empirical MSS null curve, no learned head)

## Mechanism

Patient-level aggregation operators applied to Wagner per-slide
`p_msih`. The headline operator is **max / √n_slides** — the C5 best
baseline that beat every learned aggregator we tried, including
attention-MIL and prism / titan slide encoders.

Variants exposed (in `score_columns`):

- `max_over_sqrtn` (**primary**) — max p(MSI-H) / √n
- `raw_max`, `raw_mean`, `top3_mean` — standard pooling controls
- `max_minus_Emax_MSS` — max minus the empirical MSS Emax at this n
  (calibration against the null distribution of "what does max look
  like when none of these slides are MSI-H")

The MSS null curve is an empirical quantile table computed from our
cohort's MSS patients; no model fit, no learned parameters.

## Inputs

- `results/analysis/wagner_zeroshot/slide_scores.csv` (Wagner per-slide)
- `results/analysis/calibrated_aggregation/mss_null_curve.csv` (MSS Emax bins)

## Outputs

- `results/scorers/calibrated_pool/patient_scores.csv`
- Legacy raw output: `results/analysis/calibrated_aggregation/`

## Results

| metric              | full cohort | QC-clean |
|---------------------|-------------|----------|
| patient AUROC (`max_over_sqrtn`) | 0.717 | 0.710 |
| patient AUROC (`max_minus_Emax_MSS`) | similar | similar |

**Ties wagner_zeroshot on the clean cohort.** The MSS null curve is
rebuilt from clean MSS slides, so `max − Emax_MSS` adds nothing beyond
what `max/√n` already captures.

An earlier post-hoc-filter version of the leaderboard had
calibrated_pool at 0.729 (a +1.9 pt lift over Wagner). That was
an artifact: building the MSS Emax curve from a dirty MSS pool inflated
the null, and subtracting it from the clean cohort's max gave fake
signal. Rebuilding the null curve on clean MSS slides erases the gap.

## Why it doesn't beat raw Wagner

`max / √n` is itself the right inductive bias for "MSI signal lives in
one patch of one slide" — it rewards a single high-confidence positive
without rewarding patients who simply have more slides scanned. The
MSS null curve is a sensible *interpretation aid* (lets you state how
anomalous a given max is vs the null) but doesn't add discriminative
power on top of the ranking that `max/√n` already produces.
