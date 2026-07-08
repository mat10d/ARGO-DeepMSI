# R2 — FLEX knowledge-guided information bottleneck for site-invariance

**Method.** FLEX (Feature-Level Enhancement for Cross-domain generalization; Nat Commun 2025,
s41467-025-66300-y) is the published fix for the cross-site failure R1 diagnosed on OAUTHC. It
learns a task-specific information bottleneck guided by *textual concepts* from a pathology
FM's text encoder — text concepts are site-free, so aligning image features to them should
strip site signatures while keeping MSI signal. Our compact implementation
(`argo_deepmsi/scorers/flex_bottleneck.py`) on frozen TITAN (CONCH v1.5) slide features:

- **encoder φ**: 768 → 128 bottleneck z
- **MSI head**: z → p(MSI-H)  (BCE)
- **site adversary**: gradient-reversal(z) → site  (λ_site=0.5) — the bottleneck that *removes*
  site signal
- **text anchor**: g(z) → 768, cosine-aligned to the class's site-free MSI/MSS text prototype
  (λ_text=0.3) — the knowledge guidance

Patient-grouped CV, few-shot curve K∈{1,2,4,8,16,all}. Frozen features only (CPU).

**Overall headline.** On the clean cohort (181 patients): patient AUROC **0.623**, AUPRC 0.285,
**spec@sens90 0.221**, spec@sens95 0.164, spec@sens96 0.129, NPV@sens95 0.920. This is the
**2nd-best learned scorer** (behind S1 0.646, above S4 0.607) and carries the best spec@sens90
of any learned scorer — but it is still below the zero-param champion (0.713 / spec@sens95
0.179).

**Site-invariance verdict — the method fails at its one job.** Per-site AUROC and Δ vs the
champion (`calibrated_pool`):

| site | n | FLEX AUROC | champion | Δ |
|------|--:|----------:|---------:|---:|
| LASUTH | 9 | 0.500 | 0.889 | **−0.389** |
| LUTH | 15 | 0.500 | 1.000 | **−0.500** |
| OAUTHC | 64 | 0.514 | 0.579 | **−0.065** |
| UITH | 10 | 0.583 | 0.750 | −0.167 |
| retrospective_msk | 60 | 0.752 | 0.793 | −0.042 |
| retrospective_oau | 23 | 0.711 | 0.800 | −0.089 |

**Every per-site delta is negative** (mean |Δ| = 0.209). Critically, on **OAUTHC — the target —
FLEX is *worse* than the champion** (0.514 vs 0.579). The site adversary did make performance
more *uniform* across sites (spread compressed from the champion's 0.579–1.000 to 0.500–0.752),
but it achieved uniformity by **regression to the mean**: it removed site signal *and*
MSI-predictive signal, pulling the strong sites down rather than lifting the weak one. This is
the canonical adversarial-domain-adaptation failure — invariance bought at the cost of the task.

**Why FLEX doesn't transfer here.** The published FLEX operates over 9,900 slides and 16 tasks;
our cohort has 181 patients and two of the six sites have n=9 and n=15. A gradient-reversal
adversary cannot learn a meaningful site-invariant subspace from 6 sites where the target site
(OAUTHC) dominates and the others are tiny — it just degrades to a lower-variance, lower-signal
predictor. The text anchor (site-free by construction) is the right idea, but the zero-shot text
alignment is itself weak on this cohort (S3 showed CONCH slide-text alignment is ≈ chance), so it
provides little useful guidance.

**Few-shot curve (patient AUROC):** K=1→0.470, 2→0.480, 4→0.475, 8→0.570, 16→0.557, all→0.623.
Rises only at full-shot; not few-shot capable (consistent with the S-phase).

**Per-bag-size (patient AUROC):** 1→0.550, 2→0.718, 3–4→0.716, 5+→0.318 (n=13).

**Verdict.** FLEX is a competitive *overall* learned scorer (0.623, strong operating point) but a
**failed site-invariance fix**: it does not close the OAUTHC gap and regresses every site below
the champion. Site-invariance via adversarial bottleneck is the wrong tool at this cohort size —
the R-phase should pivot to **R3 (supervised-UMAP / fmMAP)**, which suppresses site as a
*projection* constraint rather than an adversarial one, and the **T-phase abstention** (T1),
which handles OOD sites by *declining* rather than forcing a site-invariant prediction.

**Files.** `results/scorers/flex_bottleneck/{slide_scores.csv, metrics.json, few_shot_curve.csv}`.
Leaderboard re-raced to include R2.
