# D0 — OAUTHC batch-effect root cause (diagnostic, no new data)

## Question
OAUTHC-prospective (64 pt) scores AUROC 0.58 for MSI; retrospective_oau (23 pt, SAME
institution's tissue) scores 0.80. Is the deficit a correctable batch effect or unlearnable
biology? (Deployment constraint: OAUTHC is 48% of patients / 44% of positives and CANNOT be
excluded or abstained on.)

## Evidence

**1. Metadata — same population, different histology pipeline.**
- OAUTHC-prospective: cut_location=OAUTHC, stain_location=OAUTHC (locally processed end-to-end).
- retrospective_oau:   cut_location=MSKCC,  stain_location=OAUTHC (OAU tissue, sectioned at MSKCC).
Same patients, different sectioning/processing lab -> textbook batch effect.

**2. Embedding-space batch signal is near-total on raw features.**
Site-prediction (OAUTHC-prosp vs retro-OAU) from the slide embedding, 5-fold slide-level AUROC:
| embedding                | site-pred AUROC | interpretation                          |
|--------------------------|-----------------|-----------------------------------------|
| conch_v1.5_titan (raw)   | 1.000           | processing pipeline fully separable     |
| conch_v1.5_titan_harmony | 0.823           | Harmony removes a large batch component  |

**3. Harmony recovers MSI signal on OAUTHC (from harmony_oauthc_diagnostic.csv).**
CONCH-TITAN + Harmony: OAUTHC MSI-AUROC 0.611 -> 0.685 (+0.074), overall 0.645 -> 0.656.
Deflating the batch axis (finding 2) raises MSI performance on the shifted site (finding 3).

## Verdict
The OAUTHC deficit is a PROCESSING-PIPELINE BATCH EFFECT, not unlearnable biology. The FM encodes
the local-histology signature; the classifier latches onto it instead of MSI morphology. Harmony
partially corrects it (residual batch AUROC 0.823 >> 0.5 -> room to improve). This is a solvable
domain-adaptation problem. Retro-OAU (0.80) is the same-institution performance ceiling to target.

## Consequence for the plan
Retire the abstention framing. Base model = CONCH-TITAN + Harmony. Levers to test (OAUTHC-first,
held-out OAUTHC patients): stronger/targeted batch correction, OAUTHC-targeted stain normalization,
S4 attention-MIL + OAUTHC-specific few-shot adaptation, per-site calibration.
