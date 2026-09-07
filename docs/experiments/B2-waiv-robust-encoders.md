# B2 — Waiv acquisition-robust encoders (Phaet + Mascaret)

**Date:** 2026-08-05
**Status:** extraction complete (803/803 readable slides); honest nested race done.
**Encoders:** Phaet (fine-tuned Phikon-v2, 1024-d), Mascaret (fine-tuned Midnight-12k, 1536-d),
from Waiv (arXiv:2607.22861). Frozen pretrained weights, mean-pooled per slide.
**Reproduce:** `sbatch scripts/agg_waiv.sh` (aggregate) → `sbatch scripts/waiv_race.sh` (nested race).

## Result — the first frozen encoder to beat the base-4 nested baseline

Same repeated nested patient-grouped LR probe as the V1 reset (fold-local scaling + selection;
patient aggregation = mean), so these are directly comparable to the honest 0.530 baseline.

| encoder set | overall AUROC | 95% CI | OAUTHC |
|---|---:|---:|---:|
| base4 (conch_v1.5, ctranspath, uni2, virchow2) | 0.530 | 0.430–0.629 | 0.510 |
| **waiv (phaet + mascaret)** | **0.619** | 0.524–0.712 | **0.594** |
| all6 (base4 + waiv) | 0.573 | 0.473–0.675 | 0.571 |

Reference points (different model classes, not apples-to-apples): supervised Wagner champion
overall **0.717** / OAUTHC 0.616; Harmony A0 OAUTHC 0.683 (non-nested, optimistic).

Per-site (waiv): LASUTH 0.833, retrospective 0.686, OAUTHC 0.594, LUTH 0.396, UITH 0.333
(small-site cells noisy).

## Reading

- **Genuine, honestly-nested gain.** Waiv lifts the frozen mean-pool probe +0.089 overall
  (0.530→0.619) and +0.084 on OAUTHC (0.510→0.594) — the first frozen encoder to clear the base-4
  nested baseline, and it reaches Wagner's OAUTHC level (0.616) that no frozen probe had matched
  honestly.
- **Not primarily an acquisition effect.** Stage B (E2) showed paired MSK/OAU scans flip only ~5%
  of calls, so the Waiv gain is unlikely to be acquisition-invariance alone — phaet/mascaret appear
  to be simply stronger pathology encoders for this tissue. (Tempered prior updated.)
- **all6 < waiv alone** — adding the four weaker encoders dilutes the nested selection (it
  sometimes picks a weak base encoder per fold), the same selection-variance V1 documented. Use the
  Waiv encoders on their own, not pooled with weak ones.

## Caveats

- Still below the supervised Wagner aggregator (0.717); this is the fair *frozen-encoder*
  comparison, not a champion-beating claim. The real test is Waiv under a supervised aggregator.
- CI wide (n=47 MSI-H); overall/base4 CIs partly overlap though point estimates separate clearly.
- OAUTHC 0.594 is the honest nested value; do not compare it to non-nested Harmony 0.683.

## Supervised aggregator test (2026-08-05)

Ran the S4 attention-MIL recipe (tumor-only ≤500 tiles, patient-grouped 5-fold OOF, M=3 fusion,
max/√n) with the encoder swapped for Waiv. Cohort: `in_clean_set` (487 slides / 195 patients),
same basis as the S4 CONCH-ABMIL 0.607.

| aggregator | overall AUROC | OAUTHC |
|---|---:|---:|
| phaet ABMIL | **0.643** | 0.548 |
| mascaret ABMIL | 0.623 | 0.530 |
| waiv_concat (2560-d) ABMIL | 0.614 | 0.578 |
| CONCH ABMIL (S4) | 0.607 | — |
| Wagner champion (all tiles, primary cohort) | 0.717 | 0.616 |

**Verdict:** the encoder gain replicates under a supervised head — phaet-ABMIL 0.643 > CONCH-ABMIL
0.607 and > phaet frozen mean-pool 0.619 — but **no Waiv-ABMIL variant beats the Wagner champion
(0.717)**, and tumor-only restriction *hurts* OAUTHC vs Waiv's own frozen mean-pool (0.53–0.58 <
0.594). Waiv is a better encoder; a quick tumor-only ABMIL is not enough to overtake Wagner.

**Confounds vs Wagner (not yet a final head-to-head):** tumor-only ≤500 tiles (Wagner uses all),
`in_clean_set` 487/195 (not the 803/217 primary), and a from-scratch ABMIL (not Wagner's
pretrained-then-adapted transformer). Reproduce: `scripts/waiv_mil.py build` then
`eval <phaet|mascaret|waiv_concat>`; results in `results/analysis/error_anatomy/waiv_mil/`.

## From-scratch Wagner-style aggregator on Waiv (2026-08-06) — the decisive test

Trained a Wagner-style slide transformer (same architecture) FROM SCRATCH on Waiv all-tile bags
(2048-tile cap, 20 epochs, inner early-stopping), frozen W1 fold contract (803/217).
`scripts/waiv_wagner.py`; results in `results/analysis/error_anatomy/waiv_wagner/`.

| Waiv aggregator | overall AUROC | OAUTHC |
|---|---:|---:|
| frozen mean-pool (nested LR) | 0.619 | 0.594 |
| ABMIL (tumor-only) | 0.643 | 0.55 |
| **from-scratch Wagner transformer** | **0.542 (CI 0.449–0.641)** | 0.445 |
| W1-3 (CTransPath transformer, warm-started) | 0.653 | — |
| Wagner (pretrained aggregator) | 0.717 | 0.616 |

**Decisive verdict: the binding constraint is the aggregator's pretraining, not the encoder.**
The high-capacity Wagner transformer trained from scratch on 47 MSI-H patients **collapses to
~chance (0.542)** — the *worst* Waiv result. W1-3 reached 0.653 only by warm-starting from Wagner's
pretrained weights; Waiv cannot warm-start (different input dim), so it trains cold and overfits.

Three results now cohere: (1) Waiv is a genuinely **better encoder** — it wins under *low-capacity*
aggregators (mean-pool 0.619 > 0.530; ABMIL 0.643 > 0.607); (2) Wagner's 0.717 edge is its
aggregator pretrained on ~13K Western slides; (3) a high-capacity head with no pretraining
collapses on 47 positives. Capacity/pretraining, not the encoder, is the wall.

## Next (updated)

1. **Do NOT train a Waiv aggregator from scratch.** The right move is to give Waiv a *pretrained*
   aggregator: **distill the pretrained Wagner champion into a Waiv-based student** (roadmap Phase 4
   teacher–student), or pretrain a Waiv slide-head on external labeled data.
2. Meanwhile the deployable Waiv model is the **low-capacity** one (ABMIL 0.643 / mean-pool 0.619),
   as a complementary encoder — not a Wagner replacement.
3. **Re-run the error anatomy (E1–E3) on Waiv features** — does the confident-FP hard core shrink?
4. Keep nested; never let a select-on-OOF Waiv number set the champion.
