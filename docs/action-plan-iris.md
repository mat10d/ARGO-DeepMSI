# Action plan — IRIS phase (final Nigerian cohort)

**Date:** 2026-09-23 · **Branch:** `iris` (clean) · **Frozen history:** `archive/pre-iris-2026-09`
**Evidence behind this plan:** [negative-results ledger](negative-results-ledger.md),
[domain-shift evidence](domain-shift-evidence.md), [future roadmap](future-roadmap.md).

## Where we are

- One model carries reliable signal on Nigerian slides: the **pretrained** Wagner/CTransPath
  MSI transformer, applied without fitting. Its reported 0.717 (max/√n) is partly a
  slide-count artifact; slide-count-neutral it is **~0.65** — ~0.78–0.80 on MSKCC-sectioned
  tissue and **chance on OAUTHC-prospective** (our largest site). Staining lab does not
  matter (same patients 0.79 vs 0.79); OAUTHC local processing is the open factor.
- Everything *trained on our 47 positives* is at or below ~0.64 once validation is nested.
  The binding constraint is the label budget for the aggregator, not the tile encoder.
- Sites are almost perfectly identifiable from every foundation-model embedding, but
  removing the site axis (ComBat, alignment, Macenko) does not restore MSI signal and can
  destroy it. Within-patient restaining barely moves mean-pooled encoders.

The problem is therefore **a low-n, many-instance problem with bag-level shift**: a few
hundred labelled bags, each of thousands of tiles, with a weak label and a site nuisance
shared by every tile of a bag. The plan scales the three things that move that regime —
independent labelled patients, pretrained bag-level models, and unlabelled in-domain
tiles — and holds validation fixed.

## Principles (non-negotiable for this phase)

1. **Seal before looking.** Newly accrued patients form a temporal test set that no model
   selection touches. Freeze its membership before any score is computed.
2. **Pretrained bag-level models first.** Evaluate models trained on thousands of external
   patients (Wagner, PALADIN) before fitting anything on ours.
3. **One estimand.** Patient-level AUROC with patient-stratified bootstrap, specificity at
   sensitivity 0.95, per-site breakdown, nested selection whenever anything is fit.
   Slide-count-neutral pooling is primary; always report the slide-count-only control.
4. **Match collaborators' preprocessing exactly** (tile size, magnification, encoder
   version) so MSK-side and Nigeria-side embeddings are comparable.

## Phase 0 — Port and reproduce (week 1)

| Step | Done when |
|---|---|
| `uv sync --frozen --extra dev --extra dask --extra waiv` on IRIS; `argo doctor`, `pytest -q`, `argo self-test` green | acceptance sequence passes on IRIS |
| Transfer SVS/TIFF + existing zarrs via the transfer node into lab storage; checksum manifest | every slide in `slide_table.csv` has a matching checksum |
| Re-run `argo ingest` on IRIS paths (slide tables store absolute paths) | tables regenerated, row counts unchanged |
| Reproduce Wagner on the 803 slides | max/√n 0.717 ± 0.005 and patient-mean 0.659, identical patient set |
| Set IRIS partitions in wrappers / `configs/nigeria-v2.toml` | GPU smoke extraction on 3 slides |

## Phase 1 — Data freeze v3 (weeks 1–3; labels expected end of September)

- **OAUTHC processing audit:** obtain specimen type (biopsy vs resection; missing for 72/81
  OAUTHC patients), fixation and sectioning practice, and block-selection rules; arrange a
  pathologist review of a sample of OAUTHC-prospective slides against retrospective ones.

- Pull updated REDCap labels (the ~105 prospective patients whose slides are not yet local;
  `scripts/audit_redcap_freshness.py`, `scripts/build_jhu_crc_pathology_crosswalk.py`).
- Ingest and QC new slides with the existing pyramidal/MPP fixes (`argo pyramidal`).
- **Assign roles at freeze:** current 217 = development; newly labelled patients = sealed
  temporal test; unlabelled Nigerian slides = self-supervision pool. Record in the cohort
  manifest with label source and certainty (keep `Indeterminate` separate, not MSS).

## Phase 2 — Encoders on the full cohort (weeks 2–4, GPU)

Extract once, incrementally, into the per-slide zarrs:

| Priority | Encoder | Why |
|---|---|---|
| 1 | CTransPath | Wagner input (reference model) |
| 1 | Mascaret, Phaet (Waiv) | only staining-invariant encoder; best learned heads |
| 2 | H-optimus (version to confirm with MSK) | PALADIN input, matched to MSK's Mussel preprocessing |
| 2 | CONCH v1.5 → TITAN at 512 px / 20× | match the MSK CRC TITAN embeddings (ours used 256 px) |
| — | UNI2, Virchow2, PRISM | dropped: no nested advantage (see `docs/iris-runbook.md`) |

## Phase 3 — Pretrained end-to-end evaluation (weeks 3–6)

Pre-register the analysis, then score development and sealed sets with:
1. Wagner/CTransPath (locked reference).
2. **PALADIN** (MSK), either run by MSK on our H-optimus features or with a shared
   checkpoint. The strongest available prior: a pretrained, subtype-conditioned aggregator
   trained on MSK-IMPACT scale data.
3. Their ensemble (rank average, no fitting).

Report per site and at a fixed high-sensitivity operating point. Promotion requires the
sealed set.

## Phase 4 — Domain-shift aim (with the MSK postdoc's K43)

Use the measurement protocol that worked here, and treat mitigation as hypotheses:
- **Measure** in the model's feature space (site predictability, permutation-calibrated
  Fréchet distance), plus image-level colour/stain statistics and Inception FID/KID as the
  conventional reference. Use the natural contrasts: same patients restained
  (MSKCC vs OAUTHC stain) and same stain lab re-sectioned (MSKCC vs OAUTHC cut).
- **Mitigate** only against a matched control under nested validation (DANN vs same head
  without adversary; stain augmentation vs none). Macenko normalisation is already a
  documented negative on this cohort.
- **Calibrate** with Platt / fixed operating points; report per-site realised sensitivity.

## Phase 5 — Escaping the label bottleneck (weeks 6+, depends on GPUs)

Ordered by expected value given our evidence:
1. **Warm-start / distil from pretrained aggregators** (Wagner, PALADIN) into a student on a
   stronger encoder — the only learned route that has worked here (W1 warm-start 0.653 vs
   cold 0.54).
2. **In-domain self-supervision** on unlabelled Nigerian tiles (continued pretraining or
   adapters), audited for site predictability before/after.
3. **Label-efficient MIL**: low-capacity heads (mean/max, ABMIL) only, nested, with repeated
   CV and paired bootstrap against the reference; no from-scratch transformers at this n.

## Phase 6 — Ancestry comparison (Aim 2)

Run the identical pipeline on MSK African-ancestry and non-African-ancestry CRC with matched
TITAN/H-optimus settings; separate ancestry from site/processing by comparing within
processing pipeline where possible.

## Gates

| Gate | Advance when |
|---|---|
| G-port | Wagner reproduces within tolerance on IRIS |
| G-freeze | sealed temporal set frozen before any scoring |
| G-PALADIN | PALADIN inference path agreed with MSK (features or checkpoint) |
| G-promote | any new model beats the locked reference on the sealed set, paired bootstrap |
