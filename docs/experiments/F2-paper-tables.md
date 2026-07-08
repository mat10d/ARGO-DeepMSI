# F2 — manuscript tables & figures (head-to-head vs MSIntuit)

Final deliverable freeze. No new modeling — synthesises the frozen board + R1 (κ) + T3
(fairness) into the manuscript table and figures (`scripts/build_paper_tables.py`).

## Head-to-head vs MSIntuit (`results/comparison/head_to_head_msintuit.csv`)

| method | cohort | sens | spec@90 | spec@95 | spec@96 | NPV@95 | κ | AUROC |
|--------|--------|-----:|--------:|--------:|--------:|-------:|--:|------:|
| **ARGO champion** (calibrated_pool) | Nigerian CRC (ours, 181 pt) | 0.95 | 0.193 | **0.179** | 0.079 | **0.926** | **0.927** | 0.713 |
| ARGO fusion (top-3 stack) | ours | 0.95 | 0.329 | 0.107 | 0.043 | 0.882 | — | 0.703 |
| **MSIntuit CRC** (Owkin 2023) | external PRECISE | 0.96–0.98 | — | — | **0.46–0.47** | high | 0.82 | — |
| FM-MSI benchmark (CONCH, CMIG 2025) | external TCGA/PAIP | 0.90/0.94 | 0.65 | — | — | — | — | — |

## The two honest headlines

**1. On the rule-out operating point, we do NOT match MSIntuit.** At sens ≈ 0.95–0.96 our best
model holds **spec ≈ 0.08–0.18**, against MSIntuit's **0.46–0.47** — a 3–6× gap in the metric
that defines a pre-screening rule-out test. The FM-MSI benchmark's CONCH (spec 0.65 @ sens 0.90
on TCGA/PAIP) is likewise far above us. On *our* Nigerian cohort, an open-source, no-new-training
pipeline is **not competitive with MSIntuit** for rule-out specificity.

**2. On robustness we exceed it, and NPV is respectable.** Our inter-condition κ = **0.927** (R1)
is *above* MSIntuit's inter-scanner κ = 0.82 — the model's calls are stable under crop/stain
perturbation. NPV@sens95 = 0.926 means a negative call is usually right. The problem is *not*
brittleness; it is raw discriminative specificity.

## Why — the S+R+T evidence chain

- **S-phase (5 scorers):** no learned/few-shot FM recipe (linear probe 0.646, ProtoNet 0.521,
  Tip-Adapter 0.444, attention-MIL 0.607, Deep-Sets 0.516) beat the zero-param champion (0.713).
  The champion (max/√n over Wagner zero-shot CONCH scores) is the best available signal.
- **R-phase:** the champion is robust (R1 κ 0.927) but site-spiky (OAUTHC AUROC 0.579 vs 0.9–1.0
  elsewhere). Two independent site-invariance fixes — FLEX adversarial bottleneck (R2) and
  supervised-UMAP projection (R3) — **both failed identically** (every per-site Δ negative,
  OAUTHC never improved): site and MSI signal are entangled in frozen CONCH features at n=181, so
  removing site removes MSI.
- **T-phase:** framed as a selective rule-out screener, the champion safely rules out ~49 % of
  patients at ~9 % false-omission (T1) — real workload value. But covariate shift is pervasive
  and does not predict error (T2), and the **fairness gate FAILS on OAUTHC** (T3): even
  group-conditional conformal cannot make OAUTHC safe (out-of-sample FOR 0.198), because it has
  no recoverable MSI signal.

## Recommendation (the deployable story)

Deploy `calibrated_pool` as a **selective rule-out screener on the serviceable sites**
(retrospective MSK/OAU, LUTH — 63–74 % safe coverage at the FOR target), and **abstain on
OAUTHC**, which requires either more data or a non-CONCH signal. Do not claim MSIntuit-parity;
claim a robust (κ 0.93), high-NPV rule-out tool with an explicit, audited fairness boundary.
`fusion_top3` is retained only as a mid-sensitivity (sens 0.90) enrichment alternative
(higher AUPRC/spec@sens90, better OAUTHC).

## Figures (`results/comparison/figure_*.png`)

- `figure_msintuit_gap.png` — spec@sens95/96 for champion & fusion vs the MSIntuit target band.
- `figure_leaderboard.png` — clean patient AUROC of all 14 scorers (champion in red).
- `figure_per_site.png` — per-site AUROC, champion vs fusion (OAUTHC the floor).

**Files.** `results/comparison/{head_to_head_msintuit.csv, figure_msintuit_gap.png,
figure_leaderboard.png, figure_per_site.png}`. Fairness gate: **FAIL** (OAUTHC) — recorded, not
hidden.
