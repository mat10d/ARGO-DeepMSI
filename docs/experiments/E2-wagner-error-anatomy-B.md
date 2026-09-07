# E2 — Wagner error anatomy, Stage B (paired-scan natural experiment)

**Date:** 2026-08-04
**Design:** the 83 retrospective patients whose tissue was processed/scanned at **both** MSK and
OAU are the same biology imaged twice. Comparing Wagner's per-acquisition score isolates
acquisition sensitivity from biology/label.
**Reproduce:** `run_stage_b(threshold=0.1375)` → `results/analysis/error_anatomy/B_*`.

## Result

| metric | value |
|---|---:|
| paired patients | 83 |
| mean \|Δp\| (OAU − MSK) | 0.112 |
| median \|Δp\| | 0.103 |
| **call-flip rate at threshold 0.1375** | **4.8% (4/83)** |
| Pearson r(p_MSK, p_OAU) | 0.811 |
| direction (OAU > MSK) | 8/83 |
| mean p_MSK / p_OAU | 0.411 / 0.316 |

## Reading

- **Acquisition rarely flips the call.** Only ~5% of paired patients change MSI-H/MSS call between
  the MSK and OAU acquisition of the same tissue, and the two scans are strongly correlated
  (r = 0.81). Wagner's prediction on the retrospective cohort is driven mostly by biology, not by
  the scanner/stain.
- **There is a real but modest acquisition bias.** OAU acquisitions score systematically *lower*
  (mean p 0.411 → 0.316; only 8/83 rise). The OAU pathway shifts Wagner toward MSS by ~0.10 in
  p_msih — enough to occasionally cost sensitivity, not enough to explain the large specificity
  failure.
- **Implication for the OAUTHC gap and for Waiv.** If acquisition shift were the dominant driver,
  an acquisition-robust encoder would be the fix. This experiment argues acquisition shift is
  modest, consistent with A1 (feature batch-correction) and A2 (stain normalization) both failing
  to recover OAUTHC. It tempers the prior that the Waiv encoders (extraction in progress, job
  10641312) will close OAUTHC — that remains a clean, still-worth-running test, but the paired
  evidence says the OAUTHC deficit is more entangled biology/label than a removable nuisance.

## Caveat

This pairing is the **retrospective** MSK/OAU acquisition, not the **prospective OAUTHC** cohort
that is the principal weakness. It bounds acquisition sensitivity for retrospective-OAU scanning
and is the cleanest natural experiment available, but prospective OAUTHC acquisition may differ.
Pairs are same-patient tissue, not guaranteed to be the identical physical section.
