# A2 — Leave-one-site-out evaluation

**Status:** ✅ first pass done 2026-04-22 — results are **concerning**, needs follow-up

**Script:** `scripts/site_holdout.py`

**Outputs:** `results/analysis/site_holdout/<embedding>/per_site.csv` and `results/analysis/site_holdout/{summary.csv, summary_heatmap.png}`

## What this tests

For each site S (`LASUTH`, `LUTH`, `OAUTHC`, `UITH`, `retrospective_msk`, `retrospective_oau`):
1. Remove from training every slide of every patient who has at least one slide in S (explicit patient-level leakage guard — important for retrospective patients whose PATIENT id spans `retrospective_msk` and `retrospective_oau`).
2. Train class-balanced LR on the remaining slides.
3. Test on the held-out site's slides.

Metrics: AUROC, AUPRC, balanced accuracy.

## Results (test-set AUROC)

| Embedding           | LASUTH (15) | LUTH (31) | OAUTHC (476) | UITH (12) | retrospective_msk (269) | retrospective_oau (258) |
|---|---:|---:|---:|---:|---:|---:|
| conch_v1.5_mean     | 0.50 | 0.21 | **0.53** | 0.25 | **0.54** | **0.57** |
| conch_v1.5_titan    | 0.64 | 0.42 | 0.49 | 0.17 | 0.48 | 0.51 |
| uni2_mean           | 0.33 | 0.53 | 0.34 | 0.75 | 0.44 | 0.44 |
| virchow2_mean       | 0.39 | 0.52 | 0.36 | 0.44 | 0.48 | 0.54 |
| virchow2_prism      | 0.28 | 0.46 | 0.49 | 0.36 | 0.49 | 0.50 |

Figure: `results/analysis/site_holdout/summary_heatmap.png`.

## Interpretation — this is a real problem

Against the within-patient StratifiedGroupKFold baseline (conch_v1.5_mean AUROC=0.60), **leave-one-site-out drops every model to near or below chance** on the large held-out sites (OAUTHC n=476, retrospective_msk n=269, retrospective_oau n=258).

- **OAUTHC** is 60% of the cohort. Holding it out removes most of the training signal. The best any model does is `conch_v1.5_mean` @ 0.53.
- **The retrospective sets** are intriguing — `retrospective_msk` (0.54) and `retrospective_oau` (0.57) with `conch_v1.5_mean`. Since the **same patients** appear in both sets (different stain protocol only), holding one out still leaves the other for training from the patient-leakage guard's perspective — but the guard drops *both* versions of those patients when either site is held out, by design. So these are honest cross-site numbers.
- **Small held-out sites (LASUTH n=15, LUTH n=31, UITH n=12) are too small to be reliable** — AUROC variance is enormous.

## What this says about the pipeline vs. the paired-staining result

A1 (paired staining, within-patient) says foundation models are stain-robust. A2 (leave-one-site-out) says they *don't* generalize across sites in this cohort. The gap between those two results is the **batch effect that remains after stain is controlled for** — cohort composition, fixation protocols, scanner-level quirks beyond stain, or differences in tumor grade / sampling practice.

Practical read: the Phase 1 within-patient CV AUROC of 0.60 is optimistic. Real deployment to an unseen Nigerian site currently performs at chance. This is the problem the paper needs to either solve or carefully characterize.

## Next actions (follow-up)

Priority for Phase 1e (ranked by expected impact × effort):

1. **Per-site class prevalence audit** — MSI-H prevalence varies 13–50% across small sites. Verify the 0.21–0.50 AUROCs on the small held-out sites aren't just small-N noise by reporting bootstrapped CIs.
2. **Site-aware feature-space analysis** — UMAP or PHATE on `embeddings.h5ad` colored by site, not just MSI label. Is there a site cluster structure that dominates the MSI cluster structure? That's the mechanism if so.
3. **Batch-effect correction** — ComBat or Harmony on the slide embeddings, keyed by SITE. Then re-evaluate site-holdout.
4. **Site-stratified training** — weight training samples so each site contributes equally (or use a site-level mixed-effects classifier).

## Rerun

```bash
python scripts/site_holdout.py             # sweep all embedding dirs
python scripts/site_holdout.py --embeddings results/embeddings/<model>
```
