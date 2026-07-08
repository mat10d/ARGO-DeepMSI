# Q3 — tumor-area filter (CTransPath NCT-CRC TUM head)

**Method.** Smoke-off of two tumor-area filters on a fixed 20-slide / 6-site probe
(`scripts/tumor_filter_smokeoff.py`, GPU job 10302126):

- **(a) GrandQC** `zs.seg.tissue(model='grandqc')` — tissue-vs-background only; **no
  tumor class**, so it scores 0 tumor-localization recall. Rejected.
- **(b) CTransPath NCT-CRC-HE 9-class LR head** (`results/analysis/c5_phase1c/tissue_head_ctranspath_nonorm.joblib`,
  holdout TUM recall 0.91) — a real `TUM` class; 0.95 tumor-localization recall on the
  probe. **Winner.**

Applied the winner to every in-clean-set slide (`scripts/tumor_tiles_apply.py`, CPU,
reads cached `ctranspath_tiles` from each zarr — no GPU, no re-extraction): per-tile
`argmax==TUM` → `results/data/tumor_tiles/<slide>.npy` + `tumor_fraction.csv`
(509/509 slides, median tumor_fraction 0.109). Reduced into `cohort_clean.csv` via
`argo_deepmsi.eval.cohort --tumor-tiles-dir` which adds `n_tumor_tiles`,
`tumor_fraction` and **drops** slides with `tumor_fraction < floor` (no tumor tiles ⇒
no MSI signal).

**Floor choice (largest floor keeping champion clean patient AUROC ≥ 0.710).** Swept
`calibrated_pool` on the tumor-filtered clean set:

| floor | n_slide | n_pat | patient AUROC | spec@sens95 | spec@sens96 |
|------:|--------:|------:|--------------:|------------:|------------:|
| 0.0001 (drop 22 zero-tumor) | 487 | 195 | 0.7109 | 0.157 | 0.065 |
| 0.005 | 445 | 186 | 0.7144 | 0.172 | 0.076 |
| **0.01 (chosen)** | **428** | **181** | **0.7127** | **0.179** | **0.079** |
| 0.02 | 388 | 168 | 0.7012 | 0.092 | 0.092 |
| 0.05 | 317 | 139 | 0.6670 | 0.062 | 0.062 |

Floor **0.01** is the largest that holds AUROC ≥ 0.710 (0.02 breaks it). It drops 81/509
slides (17 patients) that are <1 % tumor, lifting **spec@sens95 0.154 → 0.179** on the
same champion — a real purity gain, not just size loss.

**Headline vs MSIntuit** (target: sens 0.96–0.98 @ spec 0.46–0.47, κ 0.82). Champion on
the tumor-filtered clean cohort (428 slides / 181 patients): patient AUROC **0.7127**,
spec@sens95 **0.179**, spec@sens96 **0.079**, NPV@sens95 **0.926**. Still far from the
MSIntuit operating point — this task purifies the cohort; the S-phase scorers are where
the spec@sens gap is meant to close.

**Per-site drops** (tumor floor 0.01):

| site | before → after | dropped |
|------|---------------:|--------:|
| LASUTH | 11 → 9 | 2 |
| LUTH | 21 → 18 | 3 |
| OAUTHC | 196 → 182 | 14 |
| UITH | 12 → 12 | 0 |
| retrospective_msk | 97 → 61 | 36 |
| retrospective_oau | 172 → 146 | 26 |

Drops are heaviest on the retrospective MSK/OAU slides (many low-tumor sections), not
skewed toward the OAUTHC-problem axis. Clean prevalence after: 0.236 (181 patients).

**Verdict.** CTransPath TUM head chosen over GrandQC (which has no tumor class); floor
0.01 applied. Champion clean patient AUROC held at 0.7127 (≥ 0.710) with an improved
spec@sens95, so **no-regression holds** and the filter is kept. Manifest records
method + floor + smoke-off metrics + per-site drops. Q4 will freeze the manifest, wire
`in_clean_set` into the board re-race, and set the no-regression floor.
