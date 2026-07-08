# A1 — Batch-correction sweep on frozen CONCH-TITAN (OAUTHC-targeted)

## Method
Harmony (A0) removes only part of the OAUTHC-prospective vs retro-OAU batch axis
(residual site-pred AUROC 0.82). Sweep stronger / targeted feature-space corrections on
the **raw** `conch_v1.5_titan` (768-d) embedding, keep the one with the highest **OAUTHC
held-out patient AUROC**. All corrections are unsupervised w.r.t. MSI (site + features
only) and fit transductively on the scored cohort — the same protocol A0/Harmony used.
LR probe, 5-fold patient-grouped OOF, max/√n patient aggregation, clean cohort (428
slides / 181 patients, OAUTHC included).

- `raw` — no correction (reference).
- `combat_global` — parametric-EB ComBat (inmoose `pycombat_norm`), batch = site.
- `oauthc_target_retrooau` — OAUTHC-prospective location/scale aligned to the
  retrospective_oau reference (same OAU patients, cut at MSKCC); other sites untouched.
- `oauthc_target_pooled` — OAUTHC-prospective aligned to the pooled non-OAUTHC reference.

## Headline (OAUTHC FIRST)
Winner = **`oauthc_target_retrooau`**: **OAUTHC AUROC 0.615, spec 0.118 @ sens 0.95/0.96,
n=64 pt.** It **near-perfectly erases the measurable batch axis** (site-pred AUROC
0.9997 → 0.079) — yet OAUTHC MSI-AUROC barely moves (raw 0.608 → 0.615). Overall 0.654
(< champion 0.713; no-regression floor holds).

## Batch-sep vs MSI-AUROC trade table (the key result)
| method | OAUTHC AUROC | overall AUROC | residual batch AUROC ↓ |
|---|---|---|---|
| **oauthc_target_retrooau** | **0.615** | 0.654 | **0.079** |
| raw | 0.608 | 0.646 | 1.000 |
| oauthc_target_pooled | 0.593 | 0.646 | 0.961 |
| combat_global | 0.552 | 0.607 | 0.157 |
| — Harmony (A0, for reference) | 0.683 | 0.656 | 0.823 |

## Interpretation
1. **Batch separability and MSI recovery are decoupled.** A one-site linear location/scale
   alignment drives the OAUTHC-vs-retro-OAU site-pred AUROC from ~1.0 to 0.08 (batch signal
   essentially gone) while lifting OAUTHC MSI-AUROC by only +0.007. Erasing the *measurable*
   batch axis does not recover MSI biology — the two are entangled at the linear-feature level.
2. **Harmony still leads OAUTHC (0.683) despite a much larger residual batch axis (0.82).**
   Its soft, factor-based correction preserves more MSI-relevant variance than an aggressive
   linear match that also removes MSI signal riding on the same directions. Linear "stronger"
   correction is not better here.
3. **Global ComBat regresses to the mean** (OAUTHC 0.552, overall 0.607) — consistent with
   R2/R3's finding that global site-invariance equalization hurts. Targeting only the
   odd-one-out (OAUTHC) beats global correction, but still can't beat Harmony.

## Verdict vs targets
- **vs retro-OAU ceiling 0.80**: the best batch-corrected linear probe reaches OAUTHC 0.615 —
  ~0.19 short. Feature-space linear batch correction is **not** the lever that closes the gap.
- **vs A0 Harmony (OAUTHC 0.683)**: A1 does not beat A0. The registered `batch_corrected_probe`
  keeps `oauthc_target_retrooau` as the OAUTHC-best batch-correction method, but Harmony remains
  the A-phase base model.
- **vs MSIntuit** (spec 0.46 @ sens 0.96): not competitive on rule-out spec (OAUTHC 0.118).
- **Consequence**: the residual OAUTHC gap is not a *linearly-removable* batch offset. Push to
  image-level correction (A2 stain-norm + re-extraction) and OAUTHC-specific adaptation (A3),
  not more feature-space batch math.

Dependency added: `inmoose>=0.9.0` (pycombat ComBat) — see pyproject.toml + JOURNAL.
