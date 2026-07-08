# A0 — Harmony CONCH-TITAN linear probe (OAUTHC-recovery baseline)

## Method
Promote the D0 root-cause finding into a registered scorer. Class-balanced logistic
regression (the same head as S1) on the **Harmony batch-corrected** CONCH-TITAN slide
embedding `conch_v1.5_titan_harmony` (50-d), 5-fold patient-grouped OOF, aggregated to
patient by **max/√n**. Evaluated on the full Q4 clean cohort **with OAUTHC included**
(428 slides / 181 patients, `in_clean_set=1`). Frozen features only — Harmony was fit on
OUR cohort's features (no external data); the FM weights are frozen extractors. The 6 FMs'
Harmony variants were also added to the `simple_grid` sweep so the board reflects them.

Root cause (D0): OAUTHC-prospective vs retrospective_oau is a **processing-pipeline batch
effect** (tissue cut at OAUTHC vs MSKCC), not biology. Raw `conch_v1.5_titan` separates the
two pipelines at site-pred AUROC 1.00; Harmony drops that to 0.82. The classifier stops
latching onto the local-histology signature and recovers MSI signal on OAUTHC.

## Headline (OAUTHC FIRST — the A-phase mission)
**OAUTHC held-out patient AUROC 0.683, spec 0.275 @ sens 0.95 (= @ sens 0.96), n=64 pt,
prevalence 0.203.** Harmony lifts OAUTHC from the raw-TITAN LR probe's 0.608 (S1 by_site) —
**+0.075**, matching D0's diagnostic +0.074 — toward the same-institution ceiling
retro-OAU 0.80. Not yet at ceiling; A1–A5 push further.

Overall (full cohort, OAUTHC included): patient AUROC 0.656, spec 0.264 @ sens 0.95, NPV
0.949. This is **below** the zero-param champion `calibrated_pool` (0.713) — expected: this
is the OAUTHC-recovery baseline, not a global-race entry. The no-regression floor (best
clean AUROC ≥ 0.7127) is unaffected (champion still tops the board).

## Per-site (patient AUROC, clean cohort)
| site | n | prevalence | AUROC | spec@sens95 |
|---|---|---|---|---|
| retrospective_msk | 60 | — | 0.697 | — |
| **OAUTHC** | **64** | **0.203** | **0.683** | **0.275** |
| retrospective_oau | 23 | — | 0.611 | — |
| LASUTH | 9 | — | 0.778 | — |
| LUTH | 15 | — | 0.654 | — |
| UITH | 10 | — | 0.417 | — |

## Per-bag-size (patient AUROC)
| bucket | n | AUROC |
|---|---|---|
| 1 slide | 77 | 0.629 |
| 2 slides | 35 | 0.705 |
| 3–4 slides | 56 | 0.687 |
| 5+ slides | 13 | 0.273 |

## Verdict vs targets
- **vs MSIntuit** (spec 0.46–0.47 @ sens 0.96, κ 0.82): not competitive on rule-out spec —
  OAUTHC spec@sens96 0.275, overall 0.236. Still the whole gap.
- **vs raw CONCH-TITAN**: Harmony is a real OAUTHC gain (+0.075 AUROC) with no new data, and
  now the honest evaluated A-phase baseline (the D0 raw-vs-harmony sweep was diagnostic only).
- **vs retro-OAU ceiling 0.80**: OAUTHC 0.683 leaves ~0.12 AUROC on the table for A1–A5
  (stronger/targeted batch correction, image-level stain-norm, OAUTHC-specific few-shot).

Base model for the rest of the A-phase: `conch_v1.5_titan` + Harmony.
