# A3 — Attention-MIL with OAUTHC-specific few-shot adaptation

## Method
S4 (clam_tilemil) was the only learned recipe that beat the champion on OAUTHC (0.629 vs
0.579), but its few-shot curve subsampled K patients per class from the WHOLE pool — a
GLOBAL few-shot that (correctly) failed at low N. A3 gives few-shot the **OAUTHC-specific**
test it never got. Two evaluations, both on S4's frozen tumor-CONCH bags + gated-attention
ABMIL (Ilse 2018), 5-fold patient-grouped, max/√n, clean cohort (OAUTHC included):

1. **Full-cohort adapted OOF (primary).** Base ABMIL ensemble (M=3) trained per fold; for
   OAUTHC test patients the fold's base is fine-tuned on that fold's OAUTHC-train patients
   before predicting. No leakage — adaptation only sees train-fold patients.
2. **OAUTHC K-shot adaptation curve (headline).** A base ensemble is trained ONCE on ALL
   non-OAUTHC (good-site) patients, then adapted on K OAUTHC support patients (per class,
   3 draws averaged) and evaluated on OAUTHC held-out. K=0 = zero-shot transfer.

Frozen features; adaptation fits on our own OAUTHC patients. No external data. CPU-only.

## Headline (OAUTHC FIRST) — adaptation is real and NECESSARY, but caps below Harmony
**Full-cohort adapted OAUTHC AUROC 0.590** (spec 0.000 @ sens 0.95, n=64 pt); overall 0.613
(< champion 0.713; floor holds).

**OAUTHC K-shot adaptation curve** — the good-site head does NOT transfer, but even a few
OAUTHC patients recover it:

| K OAUTHC support / class | OAUTHC patient AUROC |
|---|---|
| 0 (zero-shot good-site head) | **0.445** (below chance) |
| 2 | 0.557 |
| 4 | 0.594 |
| 8 | 0.561 |
| 16 | n/a (OAUTHC has ~13 MSI-H patients — infeasible) |
| all OAUTHC-train | 0.612 |

The zero-shot good-site MIL head lands **below chance on OAUTHC (0.445)** — direct evidence
that a head trained on the good-site pipeline actively mis-generalizes to OAUTHC. Adapting on
as few as 2–4 OAUTHC patients per class lifts it to ~0.59, saturating near 0.61 at K=all.

## Per-site (full-cohort adapted, patient AUROC)
| site | n | AUROC |
|---|---|---|
| LASUTH | 9 | 0.778 |
| retrospective_msk | 60 | 0.727 |
| **OAUTHC** | **64** | **0.590** |
| LUTH | 15 | 0.538 |
| retrospective_oau | 23 | 0.522 |
| UITH | 10 | 0.417 |

## Interpretation
1. **OAUTHC-specific adaptation is the right *kind* of fix** — unlike the global few-shot of
   S1–S5 (flat/below chance) and unlike normalization (A1/A2, which erase signal), fine-tuning
   the head on a handful of OAUTHC patients moves OAUTHC from 0.445 to 0.61. The signal is
   learnable with OAUTHC labels; it just isn't transferable zero-shot.
2. **But the tile-MIL ceiling (~0.61) is below the slide-FM Harmony frontier (0.683, A0).**
   The adapted ABMIL matches S4 (0.629) rather than beating it; adaptation recovers the
   below-chance transfer but does not exceed what Harmony already gets with zero OAUTHC-specific
   fitting. The gain from adaptation ≈ the gain Harmony gets for free.
3. **Practical read:** if OAUTHC labels are available at deploy time, a few-shot-adapted head
   is a viable path; but on the frozen no-adaptation setting, Harmony remains the base model.

## Verdict vs targets
- **vs A0 Harmony (OAUTHC 0.683)**: A3 does not beat it (0.590 full-cohort / 0.612 K-all curve).
- **vs S4 clam_tilemil (OAUTHC 0.629)**: comparable; adaptation ≈ recovers the transfer gap.
- **vs retro-OAU ceiling 0.80 / MSIntuit spec 0.46**: still short (OAUTHC spec 0.000 @ sens 0.95).
- **Consequence**: adaptation with OAUTHC labels is the only lever so far that turns a
  below-chance OAUTHC transfer into a real signal, but it plateaus at the tile-MIL level.
  Combine with A5 per-site calibration for an honest OAUTHC operating point; Harmony stays base.
