# C1 — Failure diagnosis on the Wagner zero-shot + Harmony batch correction

**Status:** ✅ first pass done 2026-04-22

**Scripts:** `scripts/failure_diagnosis.py`, `scripts/harmony_integrate.py`

**Outputs:**
- `results/analysis/failure_diagnosis/{slide_stats.csv, tercile_auroc.csv, error_coefs.csv, per_site_errors.csv, oauthc_audit.md, *.png, summary.json}`
- `results/analysis/harmony/{summary.csv, site_holdout_long.csv, site_holdout_compare.png, run.log}`
- Corrected embeddings in `results/embeddings/<emb>_harmony/` (auto-picked up by `site_holdout.py`, `train.sh`, `autoresearch_tier1.py`).

Covers Step 1 (1a, 1c, 1d) and Step 5 of `docs/remaining_tasks.md`. **Step 1b (MSIsensor correlation) is blocked on missing IMPACT data** — no IMPACT / MSIsensor tables are present in `data/` or `results/data/`.

---

## 1a — Biopsy / resection stratification (tissue-area proxy)

Using `n_tiles` per slide as the tissue-area proxy (each tile is 256 px × 0.5 μm/px = 128 μm, so area = `n_tiles × 16 384 μm²`). Overall Wagner AUROC by tissue-area tercile:

| Tercile | n | tile range | prevalence | AUROC | AUPRC |
|---|---:|---|---:|---:|---:|
| small  | 268 | 55 – 2 248   | 0.134 | **0.582** | 0.228 |
| medium | 267 | 2 258 – 5 471 | 0.195 | 0.473 | 0.238 |
| large  | 268 | 5 484 – 26 046 | 0.246 | **0.615** | 0.283 |

Two findings worth noting:

1. **Non-monotonic.** If biopsy-vs-resection were the driver, we'd see AUROC rise with tissue area. Instead the middle tercile dips to 0.47 while both ends clear 0.58. The "tiny-biopsy hypothesis" doesn't hold up at the overall level.
2. **Prevalence rises with tissue area** (0.13 → 0.19 → 0.25). Larger resected specimens in this cohort skew MSI-H. This is a sampling-bias story that affects any downstream threshold calibration.

Bottom line: specimen type is not the dominant failure mode in aggregate. See per-tercile per-site breakdown in `tercile_auroc.csv` for the subtleties (some small sites do follow the intuitive pattern).

---

## 1c — Error profiling (logit(correct ~ metadata))

Class-balanced logistic regression of prediction correctness on metadata covariates. Top ranked by |coef|:

| Feature | Coef | Reading |
|---|---:|---|
| `y` (being MSI-H positive) | **−1.24** | Wagner is biased toward MSS; MSI-H cases dominate the errors. |
| `cut_location_OAUTHC` | −0.76 | OAUTHC-cut slides are the dominant error cluster. |
| `site_OAUTHC` | −0.58 | Consistent with the per-site AUROC of 0.44. |
| `stain_location_OAUTHC` | −0.43 | Same. |
| `site_UITH` / `stain_location_UITH` | −0.41 | UITH also struggles (AUROC 0.69 on n=12). |
| `site_retrospective_msk` / `stain_location_MSKCC` | −0.24 | Mild. |

Per-site accuracy / AUROC breakdown is in `per_site_errors.csv`.

Two takeaways:

- The classifier is **MSS-biased**: it's not randomly wrong on MSI-H, it's systematically missing them. Matches the 0.572 slide-level AUROC with 0.192 prevalence.
- Every strongly-negative covariate is an OAUTHC-derived feature. The site, cut location, and stain location are all OAUTHC. This is the error hot-spot.

---

## 1d — OAUTHC prospective deep dive

|  | n slides | n patients | prevalence | Wagner AUROC |
|---|---:|---:|---:|---:|
| OAUTHC (prospective) | 476 | 81 | 0.174 | **0.439** |
| retrospective_oau    | 172 | 83 | 0.221 | 0.804 |

Full audit: `results/analysis/failure_diagnosis/oauthc_audit.md`.

### Patient overlap: zero

**No patients appear in both OAUTHC-prospective and retrospective_oau.** The prospective and retrospective cohorts at the same hospital are **disjoint patient populations**. That disqualifies the runbook hypothesis of auditing `cmo_msi_status` vs `msi_status_mmr` concordance per-patient — that audit is impossible without extra record linkage we don't currently have.

### Tissue area

`n_tiles` is LARGER in OAUTHC prospective (median 4 435) than retrospective_oau (2 731). Biopsy-vs-resection can't explain the AUROC gap. If anything the prospective cohort has more specimen area.

### Label derivation

Label source is **perfectly confounded with site** — every OAUTHC prospective slide uses `cmo_msi_status`, every retrospective_oau slide uses `msi_status_mmr`. You cannot test label-source effects independently in this data.

### What this tells us

OAUTHC prospective is not small-biopsy driven and not re-labellable within this dataset. The AUROC collapse there must be driven by something about the cohort + protocol + label procurement jointly. Plausible suspects:

- Different MSI-H prevalence / definition (prospective uses "Instable / Stable / Indeterminate" which we binarise with Indeterminate → MSS — see `argo_deepmsi/data_ingestion.py:107`).
- Fixation / block-age / scanner protocol differences we don't have metadata on.
- Case selection bias — whichever criteria routed patients into the prospective arm at OAUTHC may not match the Western training distribution that Wagner learned.

---

## Step 5 — Harmony batch correction (site-keyed)

Applied scanpy → harmonypy to the **PCA(50)** of each standardised foundation-model embedding, keyed on `SITE`. Correction produced in `results/embeddings/<emb>_harmony/`. Evaluation: leave-one-site-out class-balanced LR on both raw and corrected features. Mean AUROC across the 6 held-out sites:

| Embedding           |   raw |  harmony |      Δ |
|---------------------|------:|---------:|-------:|
| conch_v1.5_mean     | 0.435 |    0.471 | +0.037 |
| **conch_v1.5_titan**| 0.450 |    **0.549** | **+0.099** |
| ctranspath_mean     | 0.375 |    0.323 | −0.053 |
| uni2_mean           | 0.473 |    0.412 | −0.061 |
| virchow2_mean       | 0.455 |    0.429 | −0.026 |
| virchow2_prism      | 0.430 |    0.434 | +0.004 |

Figure: `results/analysis/harmony/site_holdout_compare.png`.

### Reading

- **Harmony does NOT generically rescue the cohort.** Mean Δ across embeddings is ~0 to slightly negative; two of the best single-model raw features (uni2_mean, virchow2_mean) get *worse*.
- **Neural aggregators respond differently from mean pooling.** TITAN aggregation of CONCH-V1.5 gets the largest positive Δ (+0.099) — because TITAN's 768-D output is much more batch-sensitive (A1 r=0.45 vs virchow2_mean r=0.94). For aggregators that already soak up batch, Harmony has something to take out.
- **Mean-pooled features are already stain-robust (A1 r≥0.78)** and their residual variability is MSI-relevant signal, not batch — Harmony ends up regressing away the signal it was meant to preserve. Consistent with the "disjoint patient populations" finding in 1d: when site and biology are confounded in the ground truth, Harmony has no way to distinguish them.
- **Best post-correction AUROC is ~0.55** (conch_v1.5_titan + Harmony). Not a publishable generalisation result. Site confounding is a *symptom*, not the *root cause*, of the Nigerian generalisation gap.

### Implication for Step 2

- Step 2b ("if site-driven → Harmony rescues it") is refuted as a general rescue. Keep the corrected embeddings around for conch_v1.5_titan only; they're a candidate for fusion, not a primary fix.
- Step 2a (specimen-type filter) is also weakly supported — 1a shows tissue area is non-monotonic, and 1d shows OAUTHC prospective has *more* tissue than retrospective_oau. Filtering by tile count won't rescue the main failure mode.
- Step 2c (continuous MSI regression) is **still blocked** on missing IMPACT data.
- Step 2d (multi-model late fusion) and 2e (few-shot k-NN / prototypical) become the remaining cheap-no-risk plays before we commit engineering effort to ABMIL / TCGA pre-training.

---

## Files produced

```
results/analysis/failure_diagnosis/
├── slide_stats.csv         # per-slide: Wagner p, y, correct, n_tiles, site, stains, label_source
├── tercile_auroc.csv       # overall + per-site AUROC by tissue-area tercile
├── error_coefs.csv         # logit(correct ~ covariates) ranked by |coef|
├── per_site_errors.csv     # n, prevalence, pred_rate, accuracy, AUROC per site
├── oauthc_audit.md         # 1d audit narrative
├── tercile_auroc.png
├── tissue_vs_p.png
├── error_coefs.png
└── summary.json

results/analysis/harmony/
├── summary.csv                  # mean LOSO AUROC: raw vs harmony vs Δ per embedding
├── site_holdout_long.csv        # per (embedding, variant, site) AUROC/AUPRC/bal_acc
├── site_holdout_compare.png     # bars raw vs harmony, per embedding × site
└── run.log

results/embeddings/
└── <emb>_harmony/              # harmony-corrected features for each input embedding
    ├── embeddings.npy          # (n_slides × 50) float32 — PCA + Harmony adjusted
    ├── embeddings.h5ad         # AnnData mirror (obs: PATIENT, SITE, isMSIH)
    └── metadata.csv            # slide_id, patient_id, site, n_tiles, zarr_path
```

The `_harmony` embedding dirs follow the same layout as the raw ones, so `scripts/site_holdout.py`, `scripts/train.sh`, and `scripts/autoresearch_tier1.py` pick them up automatically.
