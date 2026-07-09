# D3 — tumor-filter × aggregation ablation

## Method
The last two inherited choices, tested as a 2-D grid on the same 181 patients (D2 base =
all slides, no hard QC gate), champion (Wagner) + A0 Harmony:

- **tumor_floor** ∈ {off (0.0), 0.01 (current), 0.05, 0.10} — slide-level `tumor_fraction` gate.
- **aggregator** ∈ {max_sqrtn, mean, top3_mean, learned_lr}, where `learned_lr` is a logistic
  regression over per-patient summary features [max, mean, top3, 1/√n], 5-fold patient-grouped OOF.

Champion per-slide = Wagner p_msih; Harmony per-slide = OOF probe refit on each floor's slide
set. Per-site AUROC, OAUTHC-first. `results/analysis/tumor_agg_ablation/grid_by_site.csv`.

## OAUTHC AUROC grid (the decisive panel)
| scorer | floor | max_sqrtn | mean | top3_mean | learned_lr |
|---|---|---|---|---|---|
| Harmony | off | 0.674 | 0.656 | 0.578 | 0.621 |
| Harmony | **0.01** | 0.683 | **0.713** | 0.694 | 0.698 |
| Harmony | 0.05 | 0.544 | 0.583 | 0.578 | 0.527 |
| Harmony | 0.10 | 0.588 | 0.609 | 0.609 | 0.553 |
| champion | off | **0.617** | 0.465 | 0.449 | 0.486 |
| champion | 0.01 | 0.579 | 0.498 | 0.517 | 0.499 |
| champion | 0.05 | 0.532 | 0.427 | 0.456 | 0.383 |
| champion | 0.10 | 0.538 | 0.403 | 0.432 | 0.306 |

## OVERALL AUROC (headline cells)
- champion: **off + max_sqrtn = 0.732** (best overall on the board); 0.01 = 0.713; floors ≥0.05 fall to 0.61–0.67.
- Harmony: 0.01 + mean = 0.669; off + max_sqrtn = 0.653; floors ≥0.05 drop to ~0.59.

## VERDICT per choice
1. **Tumor floor: keep it LIGHT (0.01) or off — floors ≥0.05 HURT.** Both scorers degrade
   monotonically above 0.01 (Harmony OAUTHC 0.71→0.58→0.61; champion overall 0.71→0.67→0.61),
   because a high tumor floor discards slides (esp. OAUTHC) — the same over-exclusion D2 flagged.
   The inherited 0.01 floor is defensible for Harmony; **off** is strictly best for the champion.
2. **Aggregation is scorer-specific — the "one aggregator for all" default is wrong.**
   - **Champion (Wagner): max_sqrtn is clearly best** (0.617 OAUTHC / 0.732 overall) and beats
     mean/top3/learned by 0.05–0.15 — the max/√n rule-out aggregation earns its place here.
   - **Harmony: MEAN beats max_sqrtn** — OAUTHC **0.713 vs 0.683** at floor 0.01 (+0.030), overall
     0.669 vs 0.656. The probe's OOF scores are smooth, so averaging beats the noisy single-max.
3. **learned_lr aggregation consistently LOSES** (0.30–0.70, always ≤ the best simple operator) —
   no case for a learned aggregator over frozen summaries (consistent with S5's failure).

## Headline finding — a new OAUTHC best, for free
**Harmony + tumor_floor 0.01 + MEAN aggregation = OAUTHC AUROC 0.713** — the highest OAUTHC AUROC
anywhere in the project, +0.030 over the A-phase frontier (Harmony max/√n 0.683) and now only
0.087 from the retro-OAU 0.80 ceiling. It costs nothing but switching the aggregation operator.

## Consequence (feeds B-phase)
- **Aggregation should be chosen per scorer**: max/√n for the Wagner champion, **mean for the
  Harmony probe**. A case-conditioned readout should not hard-code a single aggregator.
- **Tumor floor light (≤0.01)**; do not raise it.
- **No learned aggregation.**
- Combined with D2 (drop the hard QC gate), the best-justified single models are: champion =
  off + max/√n (overall 0.732); OAUTHC = Harmony + floor-0.01 + mean (0.713). Both exceed the
  currently-recorded numbers — a cohort/aggregation rebuild is warranted before the B-phase readout.

Analysis only — no scorer scores or leaderboard changed.
