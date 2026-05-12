# A1 — Within-patient paired staining analysis

**Status:** ✅ done 2026-04-22

**Script:** `scripts/paired_staining.py` + `scripts/paired_staining_summary.py`

**Outputs:** `results/analysis/paired_staining/<embedding>/{paired_scores.csv, metrics.json, paired_scatter.png}` and `results/analysis/paired_staining/{summary.csv, summary_paired.png}`

## What this tests

83 retrospective patients have slides stained both at **MSKCC** and at **OAUTHC** (Nigeria). Same patient, same scanner (all imaging done in Nigeria), different staining protocol. We use `cross_val_predict(..., cv=StratifiedGroupKFold, groups=PATIENT)` to produce leakage-free OOF `P(MSI-H)` scores, then compare per-patient mean scores across the two stain origins.

The leakage guard works because ingestion assigns a single canonical `PATIENT` id (e.g. `P_0001`) to both the `retrospective_msk` record (`142-1`) and the `retrospective_oau` record (`142-95`) of the same person. `StratifiedGroupKFold(groups=PATIENT)` then holds out *both* the MSK- and Nigeria-stained slides of that patient together — neither prediction has ever seen the patient.

## Results (N=83 paired patients, 19 MSI-H / 64 MSS)

| Embedding | Cohen's κ (binary) | Pearson r | Wilcoxon p | Mean \|Δ\| | Mean (MSK−NG) |
|---|---:|---:|---:|---:|---:|
| **virchow2_mean**      | **0.94** | **0.94** | 0.186 | 0.069 | −0.013 |
| uni2_mean              | 0.68     | 0.78     | 0.849 | 0.133 | +0.035 |
| conch_v1.5_mean        | 0.54     | 0.69     | 0.311 | 0.188 | −0.020 |
| conch_v1.5_titan       | 0.36     | 0.45     | 0.183 | 0.213 | +0.004 |
| virchow2_prism         | 0.28     | 0.31     | **0.035** | 0.285 | −0.063 |

Figures: per-embedding `paired_scatter.png`, combined `summary_paired.png`.

## Interpretation

- **virchow2_mean is near-perfectly stain-invariant** (r=0.94, κ=0.94, |Δ|=0.07) — at the foundation-model level, MSK staining and Nigeria staining are essentially interchangeable for MSI-H prediction on these patients.
- **uni2_mean and conch_v1.5_mean are also stain-robust** (r=0.78, 0.69). None of the three mean-pooled foundation models show a significant Wilcoxon shift; differences are symmetric noise, not systematic bias.
- **Neural slide encoders (PRISM, TITAN) are the *least* stain-robust.** PRISM on virchow2 drops from r=0.94 (mean pool) to r=0.31 (PRISM), *and* shows a significant Wilcoxon shift (p=0.035) favoring MSK-stained scores. TITAN on conch_v1.5 similarly collapses to r=0.45. The neural aggregator is introducing stain-dependent sensitivity that the mean-pooled foundation features don't have.

## Why this matters

Favorable: at the foundation-model + mean-pool tier, **the pipeline does not require stain normalization for Nigerian deployment** when using virchow2. The paper's domain-shift story can report this as a positive generalization finding.

Caution for Phase 2: PRISM/TITAN-style slide encoders, which are trained on specific (Western) slide-level distributions, may *add* a domain-shift failure mode. Their use on African cohorts should be audited for stain-bias before being trusted as the primary classifier backbone.

## Rerun

```bash
python scripts/paired_staining.py --embeddings results/embeddings/<model>
python scripts/paired_staining_summary.py
```
