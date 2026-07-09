# A2 — Image-level Macenko stain-norm of OAUTHC + CONCH-TITAN re-extraction

## Method
The pixel-level counterpart to A1's feature-space correction. Every OAUTHC-prospective
clean slide (182 slides / 64 patients) was **Macenko stain-normalized** to a fixed
retrospective_oau reference tile (`results/data/stain_ref_retrooau.png`, from T148-1-1-USS)
and **re-extracted through CONCH v1.5 → TITAN** — so the correction happens on the tile
pixels, upstream of the frozen encoder. Every non-OAUTHC site keeps its original
`conch_v1.5_titan` vector. LR probe (class-balanced, 5-fold patient-grouped, max/√n),
clean cohort with OAUTHC included.

Implementation: a transform hook composes Macenko in front of CONCH's own tile transform
(`scripts/stain_norm_oauthc.py`), writing a separate `conch_v1.5_stainnorm_tiles` table,
then `feature_aggregation(encoder="titan")`. Ran as a 3-shard GPU array (job 10306692) —
**182/182 slides re-extracted, zero failures** (~6 min/slide). Frozen FM weights; Macenko
fit on OUR retro-OAU tile. No external data.

## Headline (OAUTHC FIRST) — a clean negative result
**OAUTHC AUROC 0.484 (below chance), spec 0.078 @ sens 0.95/0.96, n=64 pt.** Image-level
stain normalization **actively destroys** OAUTHC MSI signal — worse than every prior point:

| approach | OAUTHC AUROC | overall AUROC |
|---|---|---|
| raw CONCH-TITAN | 0.608 | 0.646 |
| A1 batch-correction (oauthc_target_retrooau) | 0.615 | 0.654 |
| **A0 Harmony (base model)** | **0.683** | 0.656 |
| **A2 stain-norm (this)** | **0.484** | 0.591 |

Overall 0.591 (< champion 0.713; no-regression floor holds). Non-OAUTHC sites are
essentially unchanged (only OAUTHC vectors were swapped); the pooled drop is OAUTHC — the
largest site — collapsing to chance and dragging the joint probe.

## Per-site (patient AUROC, clean cohort)
| site | n | AUROC | spec@sens95 |
|---|---|---|---|
| LASUTH | 9 | 0.778 | 0.667 |
| retrospective_msk | 60 | 0.671 | 0.152 |
| retrospective_oau | 23 | 0.644 | 0.333 |
| LUTH | 15 | 0.615 | 0.615 |
| UITH | 10 | 0.500 | 0.167 |
| **OAUTHC** | **64** | **0.484** | **0.078** |

## Interpretation
1. **Aggressive pixel normalization removes MSI-relevant morphology.** Macenko re-stains
   OAUTHC tiles to the retro-OAU color statistics; CONCH then encodes tiles stripped of the
   stain/texture cues that carried MSI signal, pushing OAUTHC below chance. The correction
   is not selective — it erases biology with batch.
2. **A1 + A2 triangulate the same conclusion from both directions.** A1 (feature-space)
   near-erased the batch axis yet barely moved OAUTHC MSI-AUROC; A2 (image-space) went
   further on "normalization" and drove OAUTHC *below chance*. Neither removing the batch
   axis in features nor re-staining pixels recovers OAUTHC — the OAUTHC deficit is **not
   fixable by domain-invariant normalization** at n=64. Harmony's soft, partial correction
   (0.683) remains the frontier precisely because it does the least damage to MSI signal.
3. **The batch effect and MSI signal share the same stain/morphology channel on OAUTHC.**
   This is consistent with R2/R3 (site & MSI entangled) — you cannot subtract the OAUTHC
   processing signature without subtracting its MSI signal too.

## Verdict vs targets
- **vs retro-OAU ceiling 0.80 / A0 Harmony 0.683**: A2 regresses hard (0.484). Image-level
  normalization is the wrong lever. `harmony_probe` stays the A-phase base model.
- **vs MSIntuit** (spec 0.46 @ sens 0.96): not competitive (OAUTHC spec 0.078).
- **Consequence**: stop trying to *normalize away* the OAUTHC shift. The remaining levers
  are adaptation that keeps OAUTHC signal (A3 OAUTHC-specific few-shot MIL) and honest
  per-site operating-point calibration (A5), not further correction/normalization.

Dependency added: `torchstain>=1.3.0` (Macenko) — see pyproject.toml + JOURNAL.
GPU: 3-shard array, ~9 GPU-hours total, 182/182 slides, 0 failures.
