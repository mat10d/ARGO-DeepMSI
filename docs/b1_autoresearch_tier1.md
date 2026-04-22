# B1 — Autoresearch Tier 1 grid

**Status:** ✅ two passes done 2026-04-22. Pass 1 (sbatch 9728087, 5 embeddings, 24.5 min). Pass 2 (sbatch 9728177, 6 embeddings with `ctranspath_mean` added, 14.9 min). Rerun in future any time new embeddings are aggregated — the script auto-discovers.

**Scripts:** `scripts/autoresearch_tier1.py` + `scripts/autoresearch_tier1.sh`

**Outputs:** `results/autoresearch/tier1/{results.csv, best_config.yaml, auroc_heatmap.png}`

## Grid

- **Embeddings (6 after pass 2):** `conch_v1.5_mean`, `conch_v1.5_titan`, `ctranspath_mean`, `uni2_mean`, `virchow2_mean`, `virchow2_prism`.
- **Classifiers:** `LogisticRegression`, `SVC(rbf)`, `RandomForest(500 trees)`, `XGBoost(300 trees, md=5, lr=0.05)`. All class-balanced / `scale_pos_weight=4.26`.
- **Feature engineering:** `raw` and `PCA(100)`.
- **CV:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, groups = `patient_id`.
- **Metrics:** AUROC (primary), AUPRC.

Total configs: 6 × 4 × 2 = 48.

## Top 6 configs (AUROC)

| # | Embedding | Classifier | Feat-eng | AUROC | AUPRC |
|---|---|---|---|---:|---:|
| 1 | conch_v1.5_mean | LR | raw | 0.603 ± 0.170 | 0.295 |
| 2 | ctranspath_mean | SVM-rbf | pca100 | 0.581 ± 0.158 | 0.295 |
| 3 | ctranspath_mean | SVM-rbf | raw | 0.580 ± 0.162 | 0.297 |
| 4 | conch_v1.5_mean | LR | pca100 | 0.580 ± 0.162 | 0.275 |
| 5 | conch_v1.5_titan | LR | pca100 | 0.580 ± 0.195 | 0.310 |
| 6 | virchow2_mean | LR | raw | 0.578 ± 0.141 | 0.262 |

Best: **conch_v1.5_mean + LR + raw = 0.603 AUROC** — identical to the initial `train.sh` baseline. ctranspath_mean is competitive but doesn't dislodge conch.

## What Tier 1 actually tells us

**The classifier / feature-engineering choice is not the bottleneck.** Across 40 balanced configs, AUROC stays in the 0.43–0.60 band, with LR top across every embedding. Neither SVM-RBF, RF, nor XGBoost reliably beats LR on any embedding. PCA(100) helps `conch_v1.5_titan` (0.54 → 0.58) and `uni2_mean` slightly, but doesn't move the top 3. Huge per-fold std (±0.11 – 0.23) dominates every difference, so the observed "wins" are largely noise.

This confirms what A1/A2 already suggested:
- **Flat mean pooling of foundation features is the ceiling of linear/tree classifiers at this cohort size.** To move past 0.60, we need either a smarter aggregator (attention-MIL on the zarr tile features — B4 in the runbook) or a classifier that can make use of the full distribution of tile embeddings.
- **Class-balanced LR is the correct baseline — it is already what every other config is implicitly trying to match.**

## What to look at next

1. **Rerun Tier 1 with `ctranspath_mean` in the grid** (now available). Aggregation finished 2026-04-22. Should take another ~25 min, auto-discovered by the script.
2. **Tier 2 — multi-model fusion.** Late-average across models' P(MSI-H) (e.g. conch_v1.5_mean + virchow2_mean + uni2_mean) and feature-concatenation (concat of the 3 mean-pooled feature vectors → LR). Both are cheap additions in the same script.
3. **ABMIL (B4)** — the real lever at 217 patients / 803 slides. Reads tile features directly from zarrs, no re-extraction, ~50k params. Expected to unlock 0.70+ based on published MSI-from-H&E numbers.

## Rerun

```bash
sbatch scripts/autoresearch_tier1.sh   # auto-discovers all results/embeddings/*
```

Sbatch wrapper is on partition=20, 16 CPU, 96 GB, 2h. SVC-RBF on 2560D (virchow2) is the slow part — whole grid takes ~25 min on 16 cores.
