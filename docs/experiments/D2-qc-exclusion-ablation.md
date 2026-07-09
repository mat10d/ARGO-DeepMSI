# D2 — QC-exclusion ablation (hard / none / soft reliability-weight)

## Method
The pipeline inherited a hard QC exclusion (`in_clean_set`) as step 1. Because QC-flagging is
~99% OAUTHC vs ~20% MSK, "exclude bad QC" ≈ "shrink OAUTHC" (OAUTHC 481→182 slides). D2 tests
whether that exclusion is justified. On the **same 181 in_clean patients** (fixed canonical-site
partition), aggregate the champion (Wagner max/√n) and the A0 Harmony probe under three regimes:

- **hard** — only `in_clean_set==1` slides (current default).
- **none** — every slide of those patients (no QC exclusion).
- **soft** — every slide, reliability-weighted in max/√n: `w = FLOOR + (1-FLOOR)·(1-artifact)·
  min(tumor/0.1,1)·min(n_tiles/200,1)` (`argo_deepmsi/eval/reliability_weight.py`).

Harmony's OOF probe is refit on each regime's slide set (hard set vs full set; soft shares the
full-set fit, differing only in weighted aggregation). CPU-only. Output:
`results/analysis/qc_ablation/qc_regime_by_site.csv`.

## Headline (OAUTHC + overall FIRST)
| scorer | regime | OAUTHC slides | OAUTHC AUROC | OAUTHC spec@95 | OVERALL AUROC | OVERALL spec@95 |
|---|---|---|---|---|---|---|
| Harmony | hard | 182 | 0.683 | 0.275 | 0.656 | 0.264 |
| Harmony | **none** | **413** | 0.674 | **0.314** | 0.653 | 0.286 |
| Harmony | soft | 413 | 0.682 | 0.275 | 0.642 | 0.243 |
| champion | hard | 182 | 0.579 | 0.000 | 0.713 | 0.179 |
| champion | **none** | **413** | **0.617** | 0.000 | **0.732** | 0.150 |
| champion | soft | 413 | 0.588 | 0.000 | 0.655 | 0.129 |

## VERDICT — hard QC exclusion does NOT earn its place; keep all slides
1. **The exclusion HURTS the champion.** Dropping ~230 OAUTHC slides costs the champion
   **−0.038 OAUTHC AUROC (0.617→0.579)** and **−0.019 overall (0.732→0.713)**. The current
   0.713 champion is an *artifact of the exclusion* — on the full slide set it scores 0.732.
2. **The exclusion is a wash for Harmony** on AUROC (0.683 vs 0.674) but hard gives slightly
   *worse* OAUTHC specificity than none (0.275 vs 0.314). No AUROC is bought by discarding
   2.3× the OAUTHC slides.
3. **Soft reliability-weighting (as specified) does not beat `none`.** It matches on OAUTHC
   AUROC (0.682/0.588) but is slightly worse overall for both scorers — down-weighting by
   artifact/tumor/size removes signal rather than noise here. Simple inclusion wins.

**Bottom line: the inherited hard QC exclusion should be dropped.** It was silently shrinking
OAUTHC (182 vs 413 slides) for no gain and, for the champion, a real loss. `none` ≥ `hard`
everywhere that matters.

## Consequence for A0–A6 (flagged per D-phase rule)
This overturns the `in_clean_set` assumption baked into Q1–Q4 and every A-scorer. The A-phase
numbers (Harmony OAUTHC 0.683, champion 0.579/0.713) were computed on the *hard* cohort; on the
better-justified *none* cohort the champion is 0.617 OAUTHC / **0.732 overall** and Harmony holds
0.674 OAUTHC with better specificity. **Recommendation:** rebuild the canonical cohort without the
hard QC gate (or with the artifact/tumor floors relaxed) and re-race the board — the champion's
true ceiling looks ~0.02 higher than recorded. Deferred to a cohort-rebuild task, not done here
(D2 is the ablation + verdict only; no scorer scores or leaderboard were changed).

## Caveats
- Per-site rows use a canonical per-patient site (mode over the patient's slides) so the partition
  is identical across regimes; this reassigns some patients vs the A-phase in_clean-slide site
  (e.g. retro_oau 73 pt here vs 23 in A-phase), so per-site absolute AUROC is not 1:1 comparable to
  A-phase — but the within-D2 hard/none/soft comparison is clean.
- `soft` shares Harmony's full-set OOF fit with `none`; only aggregation differs.
