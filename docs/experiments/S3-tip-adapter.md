# S3 — training-free Tip-Adapter (+ CoOp-lite) on CONCH

**Method.** Extend the zero-shot `vl_text_cosine` (hand prompts) with a few-shot cache,
using cached artifacts only (no GPU, no text-encoder reload):

- **image features**: TITAN (CONCH v1.5) slide embeddings (`conch_v1.5_titan`, 768-d), L2-norm.
- **text prototypes**: the MSI / MSS prompt prototypes already encoded by `vl_text_cosine`
  (`text_embeddings.npy`, 2×768).

**Tip-Adapter (training-free, primary).** For a test slide feature f:
`text_delta = cos(f,MSI) − cos(f,MSS)` (zero-shot) and a class-balanced cache delta
`mean_{s∈MSI} exp(−β(1−cos(f,F_s))) − mean_{s∈MSS} …` (β=5.5, paper default). The two
terms are z-scored per fold (train-only StandardScaler) and summed at equal weight, then
sigmoid → score. No gradient, β fixed → training-free. Cache built strictly train-only;
few-shot curve K∈{1,2,4,8,16,all} rebuilds it from K patients/class (K=0 ⇒ zero-shot).

**CoOp-lite (learned, secondary).** Logistic regression on the two prompt-channel cosines
[cos(f,MSI), cos(f,MSS)] — a minimal learned re-weighting of the hand prompts. (Full CoOp
prompt-context tuning would require differentiating through TITAN's text transformer; scoped
out in favour of the training-free cache.)

**Headline vs MSIntuit** (sens 0.96–0.98 @ spec 0.46–0.47) **and the FM benchmark** (CONCH
spec 0.65@sens0.90, 0.45@sens0.94 on TCGA/PAIP). On our clean cohort (181 patients):
patient AUROC **0.444**, spec@sens95 **0.036**, NPV 0.714.

**Verdict — negative result (below chance).** Tip-Adapter **0.444**, CoOp-lite 0.502,
zero-shot text 0.438 — all at or below chance, the worst block on the board. The
`_cache_delta` unit test confirms the cache mechanism is correct on separable data, so this
reflects the real signal: **TITAN's slide-level text alignment does not transfer to MSI on
this cohort** (0.438, anti-correlated), and a few-shot cache over the same slide embeddings
cannot rescue a feature axis that is itself uninformative. The few-shot curve is flat at
≈0.45 across every K.

Note the contrast with the leaderboard's `vl_text_cosine` (0.563): that entry uses the
per-tile *max* of (MSI−MSS) cosine, whereas S3's zero-shot term uses the TITAN *slide*
embedding's text delta — the slide-level text projection is markedly weaker than the tile-max
on this data. Tip-Adapter here is built on slide embeddings (for a like-for-like cache), so it
inherits that weaker axis.

**Few-shot learning curve (Tip-Adapter patient AUROC):**

| K / class | 1 | 2 | 4 | 8 | 16 | all (37) |
|-----------|--:|--:|--:|--:|---:|---------:|
| Tip-Adapter | 0.464 | 0.453 | 0.469 | 0.457 | 0.445 | **0.444** |

Flat and below chance — adding shots does not help because the underlying feature axis is
uninformative for MSI at slide level.

**Per-site (patient AUROC / spec@sens95):**

| site | n | prev | AUROC | spec@sens95 |
|------|--:|-----:|------:|------------:|
| LASUTH | 9 | 0.33 | 0.500 | 0.333 |
| LUTH | 15 | 0.13 | 0.885 | 0.769 |
| OAUTHC | 64 | 0.20 | 0.487 | 0.000 |
| UITH | 10 | 0.40 | 0.125 | 0.000 |
| retrospective_msk | 60 | 0.23 | 0.474 | 0.000 |
| retrospective_oau | 23 | 0.22 | 0.256 | 0.056 |

No coherent site signal (LUTH 0.885 on n=15 is small-n noise; UITH 0.125 and
retrospective_oau 0.256 are well below chance).

**Per-bag-size (patient AUROC):** 1→0.495, 2→0.226, 3–4→0.503, 5+→0.045. Uniformly ≤ chance.

**Verdict in context.** Third consecutive supervised/few-shot recipe below the zero-param
champion (S1 0.646 → S2 0.521 → S3 0.444). Text-prompt-derived slide features are the weakest
axis yet on this cohort. **Recommendation:** the remaining S-phase should move to tile-level
MIL with a learned attention head (S4) and cardinality-calibrated aggregation (S5), which
carry a stronger inductive bias than distance/text/prototype heads on frozen slide vectors.

**Files.** `results/scorers/tip_adapter/{slide_scores.csv, metrics.json, few_shot_curve.csv}`.
Leaderboard re-raced to include S3.
