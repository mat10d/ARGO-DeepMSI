# F1 — stacked fusion of the top-3 distinct-signal scorers

**Method.** Stack the three best-performing, distinct-signal scorers on the clean cohort with a
logistic-regression meta-learner (`argo_deepmsi/scorers/fusion_top3.py`):

- `calibrated_pool` (0.713) — Wagner zero-shot CONCH, max/√n pooling (the champion)
- `slidefm_linearprobe` (0.646) — LR probe on TITAN slide embeddings
- `flex_bottleneck` (0.623) — FLEX site-adversarial bottleneck on TITAN

(`wagner_zeroshot` excluded as an exact duplicate of the champion; `simple_grid` as a
near-duplicate of the TITAN probe; the below-chance few-shot recipes S2/S3/S5 excluded as
noise.) Each base score is aggregated to the patient (max/√n) and aligned by patient_id; the
LR meta-learner is fit under patient-grouped CV over the three OOF scores (clean stacked
generalization). Patient-resolution.

**Headline — fusion does NOT beat the champion.** Overall patient AUROC **0.703**, vs champion
**0.713** (Δ **−0.009**). The three signals are all CONCH-derived and correlated, and the two
weaker components add more noise than complementary signal; the LR meta-learner on 181 patients
cannot recover a net gain. The zero-param champion remains the best single model.

**vs MSIntuit** (sens 0.96–0.98 @ spec 0.46–0.47). Fusion: spec@sens95 **0.107**, spec@sens96
**0.043**, NPV@sens95 0.882 — *worse* than the champion at the high-sensitivity rule-out point
(champion spec@sens95 0.179) and far from MSIntuit. Fusion is not the deployment model.

**But a nuanced split — fusion helps at mid-sensitivity and on OAUTHC.** Not everything is
worse:

| metric | fusion | champion |
|--------|-------:|---------:|
| AUROC | 0.703 | 0.713 |
| AUPRC | **0.412** | 0.313 |
| spec@sens90 | **0.329** | 0.250 |
| spec@sens95 | 0.107 | **0.179** |
| OAUTHC AUROC | **0.641** | 0.579 |

Fusion has higher AUPRC and spec@sens90, and lifts OAUTHC from 0.579 → 0.641 (the slidefm/flex
components carry a little OAUTHC signal the champion lacks). So the fusion trades high-sensitivity
specificity (worse) for mid-sensitivity precision and a modestly fairer per-site profile. For a
*rule-out* screener (which lives at sens 0.95–0.98) this trade is unfavourable — but if the use
case were a *referral/enrichment* tool at sens 0.90, fusion would be preferable.

**Per-site AUROC:** LUTH 0.962, LASUTH 0.889, retrospective_oau 0.833, retrospective_msk 0.736,
OAUTHC 0.641, UITH 0.500. **Per-bag-size:** 1→0.704, 2→0.756, 3–4→0.751, 5+→0.273.

**Verdict.** Fusion is not a win for the rule-out mission: it does not beat the champion on
AUROC and is worse at the high-sensitivity operating point that matters. This closes the modeling
question consistently with the whole S-phase — **no combination of the available FM signals
improves on the zero-param max/√n champion for rule-out MSI screening on this cohort.** The
recommended model remains `calibrated_pool`, deployed selectively (T1/T3). Fusion is retained as
a documented alternative for a mid-sensitivity enrichment use case.

**Files.** `results/scorers/fusion_top3/{patient_scores.csv, metrics.json}` — metrics.json carries
the full block, per-component AUROC, the champion delta, and the MSIntuit target. Patient-
resolution (no board regen here; the final board freeze is F2).
