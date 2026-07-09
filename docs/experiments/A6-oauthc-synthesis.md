# A6 — OAUTHC-recovery synthesis (A-phase capstone)

## Mission recap
OAUTHC-prospective is 48% of patients / 44% of MSI-H positives and cannot be abstained on.
D0 established its MSI deficit is a processing-pipeline batch effect (tissue cut at OAUTHC vs
MSKCC), not unlearnable biology, with the same-institution ceiling retro-OAU = 0.80. The
A-phase asked: how far can OAUTHC be recovered with **no new training data**?

## OAUTHC-recovery leaderboard (held-out OAUTHC patient AUROC, n=64 pt / 13 MSI-H)
| approach | OAUTHC AUROC | OAUTHC spec@sens95 | overall AUROC | gap to 0.80 ceiling |
|---|---|---|---|---|
| **A0 Harmony (base)** | **0.683** | **0.275** | 0.656 | 0.117 |
| A1 batch-correction (oauthc_target) | 0.615 | 0.118 | 0.654 | 0.185 |
| raw CONCH-TITAN probe | 0.608 | 0.118 | 0.646 | 0.192 |
| A3 tile-MIL OAUTHC-adapt | 0.590 | 0.000 | 0.613 | 0.210 |
| champion (calibrated_pool) | 0.579 | — | **0.713** | 0.221 |
| A2 stain-norm (image) | 0.484 | 0.078 | 0.591 | 0.316 |

(Figure: `results/comparison/figure_oauthc_recovery.png`; CSV:
`results/comparison/oauthc_recovery_leaderboard.csv`.)

## How far did OAUTHC recover?
**A0 Harmony is the single biggest OAUTHC gain and the best OAUTHC scorer on the entire board:
0.579 → 0.683 (+0.104), closing ~47% of the 0.221 gap to the retro-OAU same-institution
ceiling — using a zero-parameter, no-new-data batch correction.** Everything past Harmony
failed to add OAUTHC signal:

- **A1 (feature-space batch correction):** near-perfectly erased the OAUTHC-vs-retro-OAU batch
  axis (site-pred 1.00 → 0.08) yet OAUTHC AUROC barely moved (0.615) — batch separability and
  MSI recovery are **decoupled**.
- **A2 (image-level stain-norm + re-extraction):** drove OAUTHC **below chance (0.484)** —
  pixel normalization strips the stain/morphology cues that carry MSI signal.
- **A3 (OAUTHC-specific few-shot MIL):** the good-site head transfers to OAUTHC below chance
  (K=0 = 0.445); OAUTHC-label adaptation recovers it to ~0.59–0.61 — **real and necessary, but
  the tile-MIL ceiling is below Harmony**.
- **A4 (TITAN descriptions):** blocked — the checkpoint has no text-generation head (verified).
- **A5 (per-site calibration):** the OAUTHC rule-out operating point is **already safe** at the
  global Harmony threshold (sens 1.000, NPV 1.000, spec 0.275); recalibration only costs
  sensitivity. The residual OAUTHC limit is **discrimination, not calibration**.

**Convergent conclusion:** the OAUTHC batch effect and MSI signal ride the same stain/morphology
channel (consistent with R2/R3 entanglement). Any correction aggressive enough to remove the
batch axis also removes MSI signal; the most that helps is Harmony's *soft, partial* correction.
Beyond that, only more OAUTHC labels (A3-style adaptation) add signal, and only up to ~0.61.

## Head-to-head vs MSIntuit (full cohort WITH OAUTHC — no abstention)
`results/comparison/head_to_head_msintuit.csv`:
- **Champion (calibrated_pool):** overall AUROC 0.713, κ 0.927, NPV 0.926, but OAUTHC AUROC 0.579.
- **A0 Harmony (deploy base for OAUTHC):** overall 0.656, **OAUTHC 0.683**; at the OAUTHC
  operating point, **sens 1.0 / NPV 1.0 / spec 0.275** — a *safe but low-yield* rule-out.
- **MSIntuit target:** spec 0.46–0.47 @ sens 0.96, κ 0.82 (external, PRECISE).
- **FM-MSI benchmark:** closed-access; operating points NOT transcribed — named comparator only
  (provenance rule; not cited as a figure).

**Is the rule-out bar reachable with OAUTHC included? No — not on specificity.** Harmony makes
OAUTHC rule-out *safe* (zero missed MSI-H among ruled-out patients in this cohort, NPV 1.0) but
only at **spec 0.275**, well short of MSIntuit's 0.46. No single model both tops the pooled board
and recovers OAUTHC: the champion wins overall (0.713) but stalls on OAUTHC (0.579); Harmony wins
OAUTHC (0.683) at a lower pooled AUROC (0.656).

## Honest bottom line + what's left
1. **OAUTHC is partially recovered, not solved:** +0.104 AUROC from Harmony (≈47% of the ceiling
   gap), with a safe-but-low-specificity rule-out. The abstention framing is correctly retired —
   OAUTHC is served, just not yet at MSIntuit specificity.
2. **The limit is intrinsic at n=64:** batch and MSI are entangled on the same channel; frozen-
   feature correction plateaus at Harmony, and image/feature normalization actively hurts.
3. **Deployment recommendation:** use **A0 Harmony as the OAUTHC base** with the **global rule-out
   threshold** (safe: sens/NPV 1.0). For the good sites the zero-param champion remains stronger —
   a **site-routed ensemble** (champion on good sites, Harmony on OAUTHC) is the natural next step
   but is new modeling, out of A-phase scope.
4. **To close the last 0.12 to the retro-OAU ceiling** most likely needs more OAUTHC MSI-H labels
   (only 13 here) or same-pipeline OAUTHC tissue — a data question, not a method one.
