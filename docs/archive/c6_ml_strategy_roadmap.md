# ARGO-DeepMSI: Strategic ML Roadmap — Beyond Slide Attention

**Date:** April 28, 2026
**Context:** C5 Phase 2 closed negative. All aggregation-level and filtering approaches exhausted. Calibrated max/√n (AUROC 0.717, zero params) remains best. The OAUTHC 7+ bin (AUROC 0.16–0.38) is unrescuable from slide-level embeddings.

---

## 1. The Diagnosed Problem

The Phase 2 attention experiment proved something important: **the failure is not at the aggregation level — it's at the representation level.** The attention gate correctly learned to ignore Wagner on OAUTHC (ρ = −0.05) and trust it on retrospective sites (ρ = +0.28–0.30). But once it ignores Wagner, the foundation model embeddings offer no alternative MSI signal.

This is consistent with de Jong et al. (2025), who showed that all 20 pathology foundation models encode medical center information, leading to systematic diagnostic errors. Our UMAPs confirm this: embeddings cluster by site, not by MSI status. The OAUTHC prospective cluster is an embedding island with no MSI-discriminative structure.

**Root cause chain:**
1. Foundation model embeddings encode **site > biology** for this cohort
2. Mean-pooling 10,000+ tiles into one 768-2560D vector **averages away** the sparse MSI morphological signal
3. With 41 MSI-H patients, no classifier can disentangle site from MSI in this averaged space
4. The problem is not "bad slides" or "wrong aggregation" — it's that the **tile-to-slide lossy compression destroys the MSI signal before any classifier sees it**

---

## 2. The Four Strategic Approaches

Ranked by novelty × feasibility × expected impact:

### Strategy A: Vision-Language Zero-Shot MSI Scoring (★★★ — Most Novel)

**Core idea:** Use CONCH v1.5's text-image alignment to score individual tiles against MSI-specific morphological descriptions, bypassing the embedding-to-classifier pipeline entirely.

**Why it could work:**
- MSI-H has well-characterized tile-level morphological features: dense tumor-infiltrating lymphocytes (TILs), poorly differentiated/mucinous adenocarcinoma, Crohn's-like lymphoid reaction, dirty necrosis
- Text descriptions are **inherently site-invariant** — "dense lymphocytic infiltrate at tumor border" means the same thing regardless of staining protocol or scanner
- CONCH achieved 86-94% zero-shot balanced accuracy on slide-level cancer subtyping tasks; tile-level tissue recognition is its strongest modality
- FLEX (Nature Comms 2025) showed that aligning patch features with clean textual concepts via InfoNCE explicitly "forces the model to unlearn site-specific shortcuts"
- No training required — truly zero-shot, no N=41 limitation

**Implementation:**
```python
# Per tile: compute cosine similarity to MSI-H vs MSS text embeddings
# MSI-H prompts (curated from pathology literature):
msi_prompts = [
    "colorectal adenocarcinoma with dense tumor-infiltrating lymphocytes",
    "poorly differentiated colorectal carcinoma with mucinous features",
    "medullary-type colorectal carcinoma with prominent lymphoid response",
    "tumor with Crohn's-like peritumoral lymphoid reaction",
    "signet ring cell features in colorectal adenocarcinoma",
]
mss_prompts = [
    "well-differentiated colorectal adenocarcinoma",
    "moderately differentiated tubular adenocarcinoma",
    "colorectal adenocarcinoma without significant lymphocytic infiltrate",
    "conventional colorectal adenocarcinoma with glandular architecture",
]
# Score = mean(sim(tile, msi_prompts)) - mean(sim(tile, mss_prompts))
# Aggregate tile scores → slide score → patient score via max/√n
```

**What makes it potentially groundbreaking:** No one has done VL zero-shot MSI scoring on an African cohort. If text-guided scoring is site-invariant where Wagner is not, this would be a major methodological contribution — showing that vision-language priors transfer across demographics where vision-only models fail.

**Risk:** CONCH v1.5 was not trained on Nigerian tissue either. The VL alignment may still fail on unfamiliar morphologies. But text descriptions provide a domain-independent anchor that pure vision models lack.

**Compute cost:** Low. Just needs CONCH v1.5 inference on existing tiles (already in zarr). No training. ~2 GPU-hours.

---

### Strategy B: FLEX-style Information Bottleneck (★★★ — Most Directly Relevant)

**Core idea:** Apply the FLEX framework (Nature Comms 2025) to compress foundation model features through a variational information bottleneck that suppresses site-specific signals while retaining task-relevant (MSI) biology.

**Why it could work:**
- FLEX improved OOD performance in 14/16 tasks when applied to CONCH features with ABMIL
- It explicitly addresses our diagnosed problem: site signatures contaminate features
- Uses text prompts as "clean" anchors (site-free reference points) to guide the bottleneck
- Compatible with ABMIL — works at the tile level, where MSI signal lives
- Validated on TCGA CRC (one of their 16 tasks) — our exact cancer type

**Implementation:**
- Pretrain FLEX bottleneck on TCGA-CRC (publicly available, ~600 patients with MSI labels)
- Apply to Nigerian cohort tiles (zero-shot or few-shot)
- The bottleneck learns to project CONCH features into a subspace where site = noise and MSI = signal
- This is the principled fix for what de Jong et al. showed: foundation models are unrobust to center differences

**Key advantage over what we've tried:** Harmony/ComBat operate on slide-level embeddings (already lossy). FLEX operates on tile-level features before aggregation — it denoises at the source.

**Compute cost:** Medium. Needs TCGA-CRC feature extraction + FLEX training (~1 GPU-day). Then inference on Nigerian tiles.

---

### Strategy C: TCGA-Pretrained Tile-Level ABMIL (★★☆ — Most Validated)

**Core idea:** Train ABMIL on TCGA-CRC/STAD tiles (thousands of MSI-labeled slides) to learn tile-level attention for MSI prediction, then fine-tune or directly apply to Nigerian slides.

**Why it could work:**
- This is the standard recipe that achieves 0.85+ AUROC in published MSI papers
- TCGA has ~630 CRC patients with MSI labels + ~320 STAD patients — abundant training data
- ABMIL learns which tiles carry MSI signal (TILs, poorly diff areas) via attention
- Transfer works because MSI morphology is biologically conserved across populations — the same mutations produce the same tissue phenotypes
- FEATHER (ICML 2025) provides pretrained ABMIL weights for UNI/CONCH/Virchow2 on a 108-class pan-cancer task — we could start from these pretrained aggregator weights and fine-tune for MSI

**Implementation:**
1. Download TCGA-CRC + TCGA-STAD slides (public, GDC portal)
2. Extract tile features using existing models (UNI2/CONCH/Virchow2)
3. Train ABMIL on TCGA with MSI labels (5-fold CV, ~2-4 GPU-hours)
4. Apply frozen ABMIL to Nigerian slides → patient-level predictions
5. Optional: fine-tune ABMIL on Nigerian cohort (41 MSI-H, may help or hurt)

**Or shortcut:** Use FEATHER pretrained weights (MIL-Lab, Mahmood Lab) → fine-tune for MSI on TCGA → apply to Nigeria. Skips step 1-2 if UNI2 features are already public.

**Risk:** The well-known one — TCGA is predominantly White/European tissue. The domain shift to Nigerian tissue is exactly what has defeated every other approach. But tile-level ABMIL learns *which morphological patterns* correlate with MSI, not *which site correlates with MSI*. The biological signal (TILs, mucinous differentiation) should transfer.

**Compute cost:** High. TCGA slide download (~500GB) + feature extraction (1-2 GPU-days) + ABMIL training (hours). Total: ~1 week wall time.

---

### Strategy D: Multi-Scale + Morphological Feature Engineering (★★☆ — Most Interpretable)

**Core idea:** Instead of relying on black-box embeddings, compute interpretable morphological features that pathologists actually use for MSI assessment, which are inherently more robust to domain shift.

**Why it could work:**
- MSI-H has specific quantifiable features: TIL density, tumor/stroma ratio, poorly differentiated fraction, mucin content
- These features are biologically grounded — they don't change with scanner or stain
- Existing tools can compute them: HoVer-Net (nuclear segmentation), TILs scoring models, tissue composition classifiers
- Multi-scale analysis (10×, 20×, 40×) captures different MSI signatures: low-mag = lymphoid aggregates, high-mag = nuclear pleomorphism
- More interpretable for clinicians — important for adoption in Nigerian healthcare settings

**Implementation:**
- Run HoVer-Net or similar on tiles to get nuclear segmentation
- Compute per-slide: TIL density, nuclear pleomorphism score, tumor/stroma ratio
- Use these as features (5-20D) instead of or alongside 768D embeddings
- At 5-20D, N=217 is plenty for a robust classifier
- This is what Baumann et al. (MSAI-Path, Mod Pathol 2025) did for their "explainable MSI" approach

**Risk:** Feature engineering is labor-intensive and may miss subtle signals that deep learning captures. Also, HoVer-Net may itself fail on Nigerian tissue.

**Compute cost:** Medium. HoVer-Net inference on 803 slides (~1 GPU-day).

---

## 3. Recommended Execution Order

```
PHASE 1 — Zero-shot experiments (no training, days not weeks):
  ├─ A1: CONCH v1.5 VL zero-shot MSI scoring with text prompts
  │     → Compute tile-level MSI text similarity scores
  │     → Aggregate via max/√n
  │     → Compare to Wagner baseline per-site
  │
  └─ A2: Transductive refinement (Histo-TransCLIP style)
        → Propagate VL pseudo-labels through tile affinity graph
        → No training, just inference-time refinement

DECISION: If A1 AUROC > 0.72 on any site → pursue VL approach
          If A1 shows site-invariance even at lower AUROC → combine with FLEX

PHASE 2 — Lightweight adaptation (1-2 GPU-days):
  ├─ B: FLEX information bottleneck on CONCH features
  │     → Train on TCGA-CRC
  │     → Apply to Nigerian tiles
  │     → Test with ABMIL aggregation
  │
  └─ D: Morphological features (TILs, differentiation grade)
        → HoVer-Net + tissue composition
        → Low-D interpretable features

PHASE 3 — Full ABMIL pipeline (1 week):
  └─ C: TCGA-pretrained tile-level ABMIL
        → Download TCGA-CRC/STAD
        → Extract features, train ABMIL
        → Transfer to Nigerian cohort
        → Fine-tune with Nigerian labels
```

---

## 4. Why This is a High-Impact Paper Regardless of Outcome

The paper narrative doesn't require solving OAUTHC. It requires *understanding* the failure:

**If VL zero-shot works (Strategy A):**
> "Vision-language foundation models achieve site-invariant MSI prediction on the first West African colorectal cancer cohort, where vision-only models catastrophically fail. Text-guided feature alignment provides a domain-independent anchor that overcomes the site confounding that affects all tested pathology foundation models."

**If FLEX works (Strategy B):**
> "Information-bottleneck-based feature disentanglement rescues MSI prediction across 6 African sites, demonstrating that the foundation model robustness crisis identified by de Jong et al. (2025) can be addressed through task-specific feature selection."

**If TCGA transfer works (Strategy C):**
> "Tile-level ABMIL pretrained on TCGA-CRC transfers to Nigerian tissue with minimal fine-tuning, achieving [X] AUROC — the first demonstration of cross-continental MSI prediction transfer."

**If nothing works:**
> "We present comprehensive negative results showing that the Western-to-African computational pathology transfer gap persists across 28 foundation models, 3 aggregation paradigms, 6 domain adaptation strategies, and vision-language zero-shot scoring. This gap represents a fundamental challenge for equitable AI in pathology, requiring purpose-built African training cohorts rather than adaptation of Western models."

That last version is still a high-impact paper. The systematic negative result, the bag-size inflation discovery, and the first West African MSI validation are all novel contributions. But I'd bet on Strategy A or B working — the VL text-guided approaches are specifically designed for the site-invariance problem we've diagnosed.

---

## 5. Key Papers to Cite

| Paper | Relevance |
|---|---|
| **FLEX** (Huang et al., Nature Comms 2025) | Information bottleneck for site-invariance — directly applicable |
| **de Jong et al. 2025** | "All 20 FMs encode medical center info" — validates our diagnosis |
| **CONCH** (Lu et al., Nature Med 2024) | VL model for text-guided zero-shot — Strategy A backbone |
| **FEATHER/MIL-Lab** (Mahmood Lab, ICML 2025) | Pretrained ABMIL aggregators — Strategy C shortcut |
| **Histo-TransCLIP** (Zanella et al., 2024) | Transductive VL boost — Strategy A enhancement |
| **AdvDINO** (2025) | Domain-adversarial SSL — alternative to FLEX for Strategy B |
| **Wagner et al. (Cancer Cell 2023)** | Our zero-shot baseline, 13K patient training |
| **Aldera et al. (J Clin Pathol 2025)** | Only other African MSI study (South Africa, AUROC 0.91) |
| **MSAI-Path** (Baumann et al., Mod Pathol 2025) | Explainable MSI approach — Strategy D reference |
| **Midnight** (MICCAI 2025) | SOTA FM from only 12K TCGA slides — shows quality > quantity |

---

## 6. What Your Friends Need to Do (Option 1)

While we pursue Option 2, the pathologist slide review should focus on:

1. **OAUTHC prospective: flag non-tumor slides.** Even 30 minutes of pathologist time on the 18 worst-performing patients (7+ slides) would tell us whether the slides contain tumor tissue at all. This is the single highest-value annotation task.

2. **Confirm MSI labels on borderline cases.** P_0152 (MSIsensor 30.3, 17 slides, all Wagner < 0.5) is the most informative patient. If the label is wrong, the 7+ cohort failure dissolves. If the label is right, the morphology is genuinely unrecognizable to all Western models.

3. **Staining protocol documentation.** Are OAUTHC prospective slides prepared differently from other sites? The embeddings clearly separate OAUTHC from everything else — if there's a fixation or staining difference, that's the root cause and the explanation for the paper.

These three items are high-value, low-effort, and complementary to the ML work.
