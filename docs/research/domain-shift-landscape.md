# Domain shift, tiny label budgets, and MSI prediction in the Nigerian cohort: a literature review

**Date:** 2026-09-23 · **Branch:** `iris` · **Purpose:** feeds the action plan and the reply to the
MSK collaborator's K43 proposal. Every citation below was checked against PubMed, arXiv, bioRxiv,
or the publisher record. Anything that could not be confirmed is marked **unverified**.

**Our evidence (from `docs/summary.md` and `docs/experiments/`):** 217 patients, 803 slides,
47 MSI-H patients (21.7%), 6 sites. Frozen Wagner/CTransPath zero-shot with max/√n patient
pooling reaches AUROC 0.717 (0.631–0.799), with specificity 0.14 at sensitivity 0.95 (146 false
positives vs 2 false negatives). Nested frozen-FM probes reach 0.53–0.62. Sites are almost
perfectly identifiable from the embeddings. Macenko normalisation dropped OAUTHC from 0.61 to
0.48. ComBat and location/scale alignment removed the site axis but did not bring back MSI
signal. A from-scratch transformer aggregator collapsed (0.542). Paired MSK-vs-OAUTHC staining of
the same tissue flips only 4.8% of calls (P1). The error anatomy (E1–E4) attributes 39% of errors
to borderline scores, 29% to unexplained confident false positives, 22% to suspect labels, and
11% to low tumour content.

**Main conclusion.** Our bottleneck is **label-level sample size, label quality, and
specificity at a fixed operating point**. Pixel-level domain shift is a smaller factor. The
collaborator's Step 1 would measure shift and Step 2 would remove it, but three of our own
experiments already show that removing the site signal does not recover MSI. A pretrained
aggregator is the thing that has worked. PALADIN's MSK-trained COAD-MSI model would be exactly
that kind of asset, but only if MSK shares the weights or runs inference for us. Training
PALADIN's architecture on our data is the regime where our from-scratch transformer collapsed.

---

## 0. PALADIN / Mosaic and TITAN: confirmed facts

Sources: the preprint PDF (Boehm et al., bioRxiv 10.1101/2025.08.14.670351, v1 posted
2025-08-16), the public code at `github.com/kmboehm/paladin`, the Mussel tiling repo, and the
Hugging Face model cards.

| item | what we could confirm | source |
|---|---|---|
| Cohort | 378,123 H&E WSIs from 79,149 tumours in 71,142 patients, all with MSK-IMPACT sequencing; 163 OncoTree subtypes | preprint abstract and Results |
| Split | 70% train / 10% val / 20% test by patient, stratified by the most granular OncoTree code. MSI is itself a stratification variable for stomach, colorectal, and endometrial cancers | preprint, lines 117–119 |
| Tiling | 20× magnification, **112 µm × 112 µm tiles** (4.74 billion tiles), tiled with MSK's Mussel | preprint |
| Encoder | **Not named in the v1 main text.** The Methods are referenced but are not in the PDF. The code reads a column called `optimus_features_tensor_path` with `tile_emb_dim: 1536`. In Mussel, `model_type=OPTIMUS` maps to `hf-hub:bioptimus/H-optimus-0`, which implies **H-optimus-0**. This is an inference that we should confirm with Boehm. Both H-optimus-0 and H-optimus-1 take 224 px at 0.5 µm/px (= 112 µm) and output 1536-d features, so the tile geometry fits either one | code; Mussel `model_factory.py`; HF model cards |
| Aggregator (public code) | `AcontextualAggregator`: linear 1536→768 projection, then a CLS token, a **2-layer transformer** (768-d, 12 heads, FFN 3072, dropout 0.1), and an MLP head. The paper trains **one Paladin model per subtype × target × primary/metastasis** and says they are "conditioned on granular subtype and site of disease". The data config also loads 224-d OncoTree and target embeddings. Training uses max 20k tiles, beta-binomial loss by default, AdamW, SWA, and early stopping | code; preprint lines 103–105 and 120–121 |
| Calibration | The public `CalibrationCallback` uses **Platt / sigmoid** calibration via `CalibratedClassifierCV(LogisticRegression(C=1e-4), cv=5, method="sigmoid")` on validation logits. It does not use isotonic regression. The paper reports calibration curves and net-benefit curves | code, `pl_modules/callbacks.py` |
| MSI results | **COAD MSI: internal AUROC 0.97 (0.96–0.99), n=561 (110 MSI+); external 0.93 (0.87–0.98), n=322 (43 MSI+).** STAD MSI: 0.80 internal / 0.94 external. Only 165 of 3,541 histology–target pairs (4.7%) reached AUROC ≥ 0.80 | preprint Table 1 |
| External test set | The external test set appears to be TCGA: the STK11 and FGFR3 analyses refer to "the external test set" and "the TCGA test set" interchangeably. The exact external composition is in the unreleased Methods | preprint lines 331–348 |
| Ancestry / race | **No race, ancestry, or ethnicity analysis** appears in the v1 main text | full-text grep |
| Weights | "Paladin and Aeon are undergoing additional software engineering and documentation prior to public release." The code is public; the weights are not | preprint, Code availability |

**What this means for the collaborator's "baseline PALADIN performance" step.** Without the
weights, this step can only mean one of two things:
- **(a) MSK runs the COAD-MSI Paladin model on our tiles.** This is high value, because it gives
  us a second strong teacher trained on MSK data. To make it possible we must extract
  H-optimus-0 features at 0.5 µm/px on 224 px tiles.
- **(b) We retrain the architecture on our 47 positives.** This is low value. A 2-layer, 768-d
  transformer on 47 positive bags is the setting in which our from-scratch aggregator collapsed.

Their external COAD result is also worth quoting in the reply: 0.93 on about 43 positives, a
Western test set, and a CI width of about 0.11. That is the same statistical regime as our 47
positives.

**TITAN** (Ding et al., arXiv 2411.19666; HF `MahmoodLab/TITAN`) needs **CONCH v1.5 patch
features from 512 × 512 px patches at 20×**, with coordinates given at level 0
(`patch_size_lv0` = 1024 for 40× slides, 512 for 20× slides). It was pretrained on 335,645 WSIs.
The licence is CC-BY-NC-ND 4.0 and access is gated to institutional accounts. The model card
does not state the output dimension.

---

## 1. Critique of the proposed steps, one by one

### Step 1a: Fréchet Inception Distance (FID)
- FID (Heusel et al. 2017) compares Gaussian fits of **ImageNet-trained Inception-v3**
  features. Those features reflect natural-image texture and colour statistics, not tissue
  morphology. A large FID tells us the images differ, but not whether the MSI-relevant
  morphology differs.
- FID is a biased estimator that depends on sample size. Bińkowski et al. (2018) proposed KID,
  an unbiased MMD-based alternative. With about 25 slides per small site, any FID comparison
  across sites mixes up shift and sample size.
- **We already know the answer Step 1 would give.** The site is almost perfectly identifiable
  from every FM embedding we tested. This matches de Jong et al. (2025), who found that FM
  embeddings are "more strongly organized by medical centers than by biological factors", and
  Kömen et al. (2024), who found that hospital signatures dominate distances in feature space
  and are not removed by stain normalisation.
- **Better replacements:**
  1. Fréchet distance or KID computed in the space of the model we actually deploy
     (CTransPath tiles, Wagner slide embedding).
  2. Site-classifier AUROC.
  3. The **robustness index** from PathoROB (Kömen et al., Nat Commun 2026) and de Jong et al.,
     which measures whether biology or centre dominates nearest neighbours.
  4. Most useful of all: **task-relevant shift**, meaning the per-site score distribution of
     MSS patients and the per-site false-positive rate at the global threshold. We also have a
     natural experiment in P1 (same tissue, MSK vs OAUTHC staining): the flip rate there
     (4.8%) is a direct causal estimate of how much staining shift affects the output.

### Step 1b: Colour-histogram KL divergence
- The value depends heavily on bin choice, on background and pen marks, and on whether the
  computation is restricted to tissue. KL is also asymmetric and infinite when a bin is empty.
- It measures stain, which P1 has already shown is not the main driver of our errors.
  Howard et al. (2021) showed that TCGA site signatures are detectable even after colour
  normalisation and augmentation, so colour statistics under-measure the shift a model can
  actually see.
- **Verdict:** report it once as a descriptive figure. Use a symmetric, bounded measure (for
  example Jensen–Shannon on tissue-masked optical-density histograms) and do not base any
  decision on it.

### Step 2a: Macenko / Reinhard / Vahadane stain normalisation
- **Direct evidence from our own data:** Macenko destroyed MSI signal on OAUTHC (0.61 → 0.48).
  A likely mechanism is that the reference-stain fit removes chromatin and eosin cues that
  carry MSI morphology (mucin, lymphocytes, poor differentiation), and that the frozen encoders
  were not trained on normalised inputs.
- **Literature:**
  - Tellez et al. (2019) compared augmentation and normalisation across 9 labs and 4 organs,
    and recommended stain **augmentation** as the main tool.
  - Boschman et al. (2022) found normalisation "does not consistently improve" single-centre
    performance. It did help external-centre accuracy for CNNs *trained from scratch* (ovarian
    +0.25 AUC), which is a different setting from frozen foundation models.
  - Echle et al. (2020) saw only 0.95 → 0.96 external AUROC from colour normalisation.
  - Kömen et al. (2024) showed stain normalisation does not remove FM hospital signatures.
- **One legitimate use:** match whatever preprocessing the *pretrained reference model* used at
  training time. That is a question of preprocessing parity, not of domain adaptation.
- **Verdict:** do not pursue. It has already been tested and failed.

### Step 2b: CycleGAN colour or style transfer
- Cohen et al. (2018) showed that distribution-matching losses (CycleGAN-type) **hallucinate or
  remove features** to match the target distribution. In their example, tumours were added to
  or removed from MRI.
- For MSI, the diagnostic features are exactly the ones a GAN can invent or erase: tumour-
  infiltrating lymphocytes, mucin, and Crohn's-like reaction.
- With 47 positives we cannot validate that the translation preserves MSI morphology.
- StainGAN (Shaban et al. 2018) and later work improve realism, but realism is not the same as
  label preservation.
- P1 again shows that staining explains little of our error.
- **Verdict:** high risk, high cost, low expected value. At most, use it as a stress test.

### Step 2c: Multi-source domain-adversarial training (DANN)
- DANN (Ganin et al. 2016) learns features the domain classifier cannot use. In histopathology,
  Lafarge et al. (2017) found that DANN combined with colour augmentation beat standard
  approaches for **mitosis detection**. That task has dense patch-level labels, not 47 weak bag
  labels.
- **DomainBed** (Gulrajani & Lopez-Paz 2021) found that carefully tuned ERM matches or beats most
  domain-generalisation methods under fair model selection.
- **Why this is especially weak for us at the bag level:**
  1. The site is a bag-level variable shared by every tile, and our sites have about 5
     positives each outside OAUTHC. The adversary sees 6 domains and 217 bags, so its gradient
     is dominated by bag-level nuisance and noise.
  2. Prevalence differs by site, so site and label are partly confounded. Removing site
     information then removes label information as well.
  3. We already ran the non-learned version of this: ComBat and location/scale alignment
     erased the site axis and MSI did not come back. DANN is a learned version of the same
     intervention.
- **Verdict:** do not do it as a main aim. At most, include it as a DomainBed-style baseline
  under nested, site-held-out selection.

### Step 3: Lock after 5-fold nested CV, then calibrate with isotonic regression
- **Nested CV is correct and essential.** Varma & Simon (2006) showed that tuning on the same CV
  used for evaluation gives substantially optimistic error estimates, even on null data. We
  already run nested, patient-grouped CV (V1).
- Additions to propose:
  - Also report **leave-one-site-out** performance.
  - Evaluate the locked model once on the **untouched future Nigerian cohort** before that
    cohort is used for anything else.
- **Isotonic vs Platt:** Niculescu-Mizil & Caruana (2005) compared both. Isotonic regression is
  non-parametric and overfits when calibration data are scarce, where Platt's 2-parameter
  sigmoid does better. With 47 positives across 5 folds, isotonic fits a step function to about
  9 positives per fold. **Use Platt**, or better, a hierarchical or shrinkage per-site
  intercept. MSK's own PALADIN code uses sigmoid calibration.
- **Calibration does not change AUROC.** Our actual clinical failure is specificity 0.14 at 95%
  sensitivity, which is a question of threshold and ranking.
  - Echle et al. (2022) and Aldera et al. (2025) both needed **cohort- or region-specific
    thresholds**. Aldera recalibrated from 0.620 to 0.470 to reach 95.6% sensitivity in South
    Africa.
  - The right Step 3 is **site-aware operating-point selection under nested CV**, plus
    selective abstention for the 39% of errors that sit near the threshold.

---

## 2. Ideas from other fields, organised around our problem structure

Our regime: **n = 217 labelled bags (47 positive), with thousands of tiles per slide and 1–8
slides per patient. Labels are weak and at bag level, and the witness fraction is unknown and
sparse. The shift is at bag level: every tile in a bag shares its site and processing.** Each
idea below lists a concrete test on frozen features, followed by EV (expected value given our
evidence).

### 2a. MIL under tiny label budgets: which aggregators are sample-efficient?

| approach | evidence | test on our frozen features | EV |
|---|---|---|---|
| Mean / max pooling + linear head | Classic MIL (Dietterich 1997). Our V1 nested probe gives 0.53 | already done | baseline |
| ABMIL (Ilse et al. 2018) / CLAM (Lu et al. 2021; instance clustering as auxiliary pseudo-labels, reported as data-efficient) | gated attention adds about d×128 parameters; CLAM's instance loss acts as a regulariser | Low-capacity gated ABMIL at nested-CV scale has been done (S4 `clam_tilemil`). Don't expand it | low |
| TransMIL (Shao et al. 2021) / from-scratch transformers | parameter-heavy; our run collapsed to 0.542 | don't | negative |
| Top-k / LSE / softmax-temperature pooling of **teacher tile scores** | the audio and video weak-label literature (below) shows the pooling function matters most when instance scores are good | Sweep k and temperature over Wagner **tile logits**, inside nested CV. It has 1–2 parameters, so it can't overfit much | moderate, cheap |
| **Pretrained aggregators / warm start** | Shao et al. (2025, *Do MIL models transfer?*): across 11 MIL models and 21 pretraining tasks, pretrained MIL models "consistently outperform models trained from scratch", even across organs; pancancer pretraining beats slide FMs. Our own data agree: Wagner warm start works and cold start collapses | (i) MIL-Lab pretrained weights on matching encoders; (ii) TITAN/PRISM slide embeddings + linear head (done, weak); (iii) **MSK Paladin COAD-MSI weights** on H-optimus-0 tiles | **high** |
| Attention regularisation / prior | an instance-level prior (e.g., attend to tumour; E3 shows attention is already on tumour) | skip: E3 found attention is normal for the confident false positives | low |

**How many bags do aggregators need?** We found no study that gives a clean bags-vs-AUROC curve
for MSI with FM features below about 100 positives. Wagner et al. (2023) claim their pretrained
transformer pipeline is *data-efficient*. Neidlinger et al. (2025) report that the best FMs'
advantage "was less pronounced in low-data scenarios and low-prevalence tasks". **Our
experiments are themselves the best evidence available for this regime.**

### 2b. Analogous weakly-labelled bag problems in other fields

- **Video recognition with clip-level labels** (UntrimmedNets, Wang et al. 2017). They learn
  clip selection plus classification from video-level labels only, and they rely on strong
  *pretrained* clip features plus a light selection module.
  - Lesson: freeze instance features and learn only selection.
  - Test: the teacher-tile-logit pooling sweep above.
- **Weak-label audio event detection** (Kumar & Raj 2016). Audio events are cast as MIL over
  segments with recording-level labels, and the choice of pooling or aggregation is central.
  - Test: same as above.
- **Poverty mapping from satellite imagery with few labelled regions** (Jean et al. 2016). This
  is the closest analogy. Survey labels are scarce, so they **transferred from an abundant proxy
  label** (night-time lights) and explained up to 75% of variation in local economic outcomes
  across five African countries, **including Nigeria**.
  - Our proxy labels are MSI-associated morphologies that can be scored on *every* slide without
    MSI labels: TIL density, mucinous fraction, grade, and Crohn's-like reaction.
  - MSAI-Path (Baumann et al. 2025) builds MSI predictors from exactly these Bethesda-type
    features with logistic regression and random forests.
  - Test: quantify 4–6 proxy features per slide (segmentation or CONCH text prompts), then fit a
    ≤6-parameter logistic model under nested CV and a fusion with Wagner.
  - EV: moderate. It is interpretable, robust at low n, and directly addresses the question of
    whether the confident false positives are morphologically MSI-like.
- **Remote-sensing sensor and season shift** (Tuia et al. 2016). Their review of domain
  adaptation for remote sensing covers histogram matching and feature alignment for
  image-to-image spectral shift. The shift in their setting is **per image**, which matches our
  per-slide shift.
  - Lesson: alignment helps when class-conditional distributions line up after the shift, and
    fails under label shift.
  - Our site prevalences differ, which is why unconditional alignment (ComBat) could not help.
- **MRI scanner harmonisation** (ComBat: Johnson et al. 2007; Fortin et al. 2018). ComBat keeps
  biology only for *covariates you specify*.
  - Pitfall: to protect MSI you must pass MSI as a covariate, which is impossible at test time.
    Without it, ComBat shrinks site means that are partly driven by prevalence.
  - This is a mechanistic explanation of our A1 result. Harmony (Korsunsky et al. 2019) helped
    modestly, which is consistent with soft, cluster-wise correction doing less damage.
- **Drug discovery, multi-conformer MIL** (Dietterich et al. 1997; Zankov et al. 2021). Each
  molecule is a bag of conformers, and the "key instance" assumption works with a few hundred
  labelled bags when the *instance representation* is good.
  - Lesson: invest in instance quality and teacher scores, not in aggregator capacity.
- **Medical 3D volumes (slice bags) with small cohorts:** the analogy fits (bag of slices,
  scan-level label, scanner shift). **No specific citation was verified in this pass.**
- **Autonomous-driving weather shift and style randomisation** (Hendrycks & Dietterich 2019
  corruptions benchmark; Geirhos et al. 2019 texture bias; Jackson et al. 2019 style
  augmentation). Randomising style during training improves robustness, but only when *training
  the encoder*.
  - The frozen-feature analogue is MixStyle or DSU (below).
- **Dermatology across skin tones** (Daneshjou et al. 2022). Models underperformed on dark skin,
  and **fine-tuning on the diverse set closed the gap**.
  - Lesson: a small amount of in-distribution labelled data plus a strong pretrained model works
    better than adaptation from nothing.

### 2c. How domain shift interacts with bags

- **Bag-level feature-statistics retargeting (AdaBN analogue).** Schneider et al. (2020)
  improved corruption robustness by re-estimating normalisation statistics on target data. Our
  encoders use LayerNorm, so this is only possible at the **feature level**.
  - Test: shift and scale each *slide's* CTransPath tile features to the pooled cohort's tile
    statistics, **upstream of the frozen Wagner aggregator**. If A1 applied correction only to
    slide embeddings for new heads, this tile-level variant has not been tried yet.
  - Risk: per-slide standardisation also removes real slide-level biology, such as a globally
    lymphocyte-rich MSI tumour.
  - Prefer per-site statistics estimated on MSS-dominated data, and check P1 pairs as a
    positive control.
  - EV: moderate, cheap.
- **MixStyle / DSU at instance level** (Zhou et al. 2021; Li et al. 2022). Mix per-slide
  feature means and SDs across slides from *different sites* while training a small head, so
  that the head is exposed to site-style variation.
  - Test: add it to the nested-CV head training for Waiv and UNI2 features.
  - EV: low–moderate, because heads on these features are not the binding constraint.
- **CORAL / Deep CORAL** (Sun & Saenko 2016) aligns second-order statistics. It is a
  covariance-level ComBat, and **the same prevalence caveat applies**. EV: low.
- **Test-time adaptation (TENT; Wang et al. 2021).** TENT minimises entropy by updating norm
  affine parameters. With about 80% MSS and bag-level decisions, entropy minimisation pushes
  predictions toward the majority class. It is also untested on LayerNorm transformers in
  pathology. EV: low, and it can make specificity worse at a fixed sensitivity.
- **Treating the site as a bag-level nuisance (IRM, Arjovsky et al. 2019; group DRO, Sagawa et
  al. 2019).** Worst-group objectives need reliable per-group risk. With about 5 positives per
  small site, per-site risk is mostly noise. EV: low. Use it only as a reporting lens (worst-site
  AUROC with honest CIs).
- **Label shift across sites** (BBSE, Lipton et al. 2018; EM prior adjustment, Saerens et al.
  2002). Prevalence differs across sites. Within a site, AUROC is prior-invariant, but **pooled
  AUROC and a single global threshold are not invariant to site-specific score offsets**.
  - Test: estimate per-site priors with EM on held-out scores, adjust posteriors, and re-select
    thresholds.
  - Report stratified-by-site AUROC alongside pooled AUROC.
  - EV: moderate–high for specificity, cheap.
- **Patient-level bag-of-bags pooling.** Max over slides inflates the score for MSS patients
  with many slides, and slides per patient vary by site (OAUTHC contributes many). √n tempers
  this but may not remove it.
  - Test: plot false-positive rate against number of slides per site. Compare max/√n, mean, and
  a noisy-OR calibrated on the number of slides, all within nested CV.
  - EV: moderate, cheap, and it may explain part of the site-specific false positives.
- **Using the unlabelled tiles and slides.**
  - **Teacher distillation** (Hinton et al. 2015): train a student on Waiv or UNI2 features
    to reproduce frozen Wagner's *tile or slide logits* on all 803 slides, plus any unlabelled
    Nigerian slides, then fine-tune lightly on labels. This removes the label bottleneck for the
    aggregator and puts a better encoder behind it. It matches the roadmap. EV: **high**.
  - **In-domain continued SSL** (Gururangan et al. 2020, "don't stop pretraining"; Kang et al.
    2022 on in-domain pathology SSL; Filiot et al. 2026 Waiv robust fine-tuning; PathoROB
    post-hoc robustification). GPU cost is high, and the gain for MSI is uncertain given that
    Waiv, the robustness-tuned encoder, reached only 0.619. EV: moderate, expensive.
  - **Pseudo-labelling from the teacher.** With teacher specificity at 0.14, hard pseudo-labels
    would mostly propagate false positives. Use soft distillation targets instead. EV: low as
    hard labels.
  - **Stain augmentation** (HED jitter; RandStainNA, Shen et al. 2022) only applies if we train
    the tile encoder. R1's feature-space test-time augmentation already shows stable calls
    (κ = 0.93). EV: low for the frozen pipeline.

### 2d. Evaluation under low n

- **AUROC precision.** Using the Hanley–McNeil (1982) SE with 47 positives and 170 negatives:
  AUROC 0.717 has SE ≈ 0.046, so the 95% CI is about 0.63–0.81. This matches our bootstrap
  (0.631–0.799).
- **Per-site AUROC** at about 5 positives and 20 negatives has SE ≈ 0.14, which makes it
  uninterpretable. Do not rank methods on per-site AUROCs.
- **Power for a +0.05 improvement** (0.65 → 0.70), two-sided α = 0.05, paired test such as
  DeLong (1988) or a paired bootstrap:

  | correlation between the two models' scores | power |
  |---|---|
  | 0.5 | ≈ 0.19 |
  | 0.8 | ≈ 0.39 |
  | 0.9 | ≈ 0.66 |

  Our own W1 paired Δ CI (−0.132 to +0.015) shows this directly. We cannot detect realistic gains
  on this cohort, so **every head-to-head comparison should be pre-registered, few in number,
  and confirmed on the future cohort**.
- **Repeated nested CV** reduces variance from the choice of partition. It does not reduce
  sampling variance from having only 47 positives. Report the fold-to-fold spread, but keep the
  bootstrap CI as the primary uncertainty.
- **Selective prediction and conformal methods** (Angelopoulos & Bates 2021). Group-conditional
  (Mondrian) conformal by site gives per-site coverage guarantees, but with about 5 positives
  per site the prediction sets for small sites will be near-trivial. Pool the small sites.

---

## 3. Ancestry and generalisability (collaborator's Aim 2)

- **Race and demographic performance gaps exist and are only partly closed by FMs.** Vaidya et
  al. (2024, Nat Med) found white-vs-Black AUROC gaps of 3.0% (breast subtyping), 10.9% (lung
  subtyping), and 16.0% (IDH1 in glioma). Self-supervised FMs reduced the gaps but did not
  eliminate them. MSI was not studied.
- **Ancestry is confounded with site.** Howard et al. (2021) showed that TCGA submitting-site
  signatures survive colour normalisation and augmentation, bias accuracy for mutations and
  survival, and let **ethnicity be inferred from the site signature**. They proposed site-
  preserved splits. Dehkharghanian et al. (2023) found TCGA acquisition site predictable at up
  to 86% accuracy from deep features.
- **MSI models across populations:**
  - Kather et al. (2019): a TCGA-trained STAD model dropped to AUC 0.69 (0.52–0.82) on a
    Japanese cohort, and the authors noted that performance "does not necessarily extend beyond
    the cancer type and ethnicity present in the training set".
  - Echle et al. (2022) validated across nine cohorts "across different countries and
    ethnicities" but needed cohort-specific thresholds.
  - **Aldera et al. (2025, J Clin Pathol)** applied a transformer model trained on international
    cohorts (Kather group) to 197 South African CRCs from an ethnically heterogeneous
    population. AUROC was **0.91**. It required a region-specific threshold (0.470) to reach
    95.6% sensitivity at 69.6% specificity. False negatives were mostly left-sided, and
    isolated PMS2 or MSH6 loss reduced sensitivity.
  - PubMed searches found no deep-learning MSI validation in a West African or Nigerian cohort
    before ours.
- **Implication.** African ancestry alone does not force a model below 0.72: the South African
  cohort reached 0.91 with a similar class of model. Our gap is therefore more plausibly
  explained by **cohort-specific factors** than by ancestry per se:
  - label assay heterogeneity (IHC `msi_status_mmr` vs `cmo_msi_status`, and 40 "Indeterminate"
    cases coded as MSS);
  - tissue handling (fixation delays, which have not been measured);
  - tumour content;
  - the prevalence mix.
- **Design recommendation for Aim 2.** A naive comparison of "Nigerian vs MSK African-ancestry
  vs MSK non-African" confounds ancestry with fixation, sectioning, staining, and scanning. Our
  cohort already contains the controls needed to separate them:
  - `retrospective_oau` vs `retrospective_msk` is **the same Nigerian patients with MSK vs
    Nigerian staining**. It isolates processing while holding ancestry and tumour fixed. P1:
    flips were 4.8%.
  - `retrospective_msk` vs **MSK-processed African-ancestry US patients** holds processing
    roughly fixed and varies population and biology.
  - MSK African-ancestry vs MSK non-African-ancestry patients holds the site fixed and varies
    ancestry.

  Pre-specify these contrasts and report performance **stratified within site**, with
  ancestry-stratified CIs. Given the power analysis above, frame Aim 2 as *estimation with CIs*,
  not as hypothesis tests.

---

## 4. Top 5 next steps given our evidence

| # | action | why (our evidence + literature) | expected value | cost |
|---|---|---|---|---|
| 1 | **Get MSK's Paladin COAD-MSI model applied to our slides**: they run inference, or share weights under a DUA. We extract H-optimus-0 (224 px, 0.5 µm/px) features. Evaluate it zero-shot, then as a Wagner+Paladin rank-average ensemble | Pretrained aggregators are the only thing that has worked for us. Shao et al. (2025) found pretrained MIL beats scratch. Paladin COAD-MSI reached 0.93 external. Neidlinger et al. (2025) found ensembles of complementary FMs beat single models in 55% of tasks | **high**: a second independent teacher, and it turns the collaboration into data | low for us (one H-optimus-0 extraction pass), depends on MSK |
| 2 | **Label adjudication** of the label-suspect errors and the 40 "Indeterminate" cases coded MSS. Harmonise IHC vs molecular assay | 22% of errors are label-suspect (1.66× enriched). With 47 positives, a handful of label flips moves AUROC by more than any method difference we can detect | **high**: every other estimate depends on it | pathologist time |
| 3 | **Distil frozen Wagner into better encoders** (Waiv/UNI2 student on teacher logits over all tiles and slides, including unlabelled ones), then fine-tune lightly with warm start | Removes the label bottleneck for the aggregator. Hinton et al. (2015); consistent with our warm-start vs cold-start result | moderate–high | moderate GPU |
| 4 | **Bag-level fixes at the operating point**: per-site prior adjustment (Saerens EM / BBSE), shrinkage per-site thresholds, slide-count-aware patient pooling, and selective abstention, all chosen under nested CV and reported as stratified AUROC | The failure is specificity 0.14 at sensitivity 0.95. 39% of errors are borderline. Echle et al. (2022) and Aldera et al. (2025) both needed local thresholds | moderate–high for clinical utility (no AUROC change expected, except from pooling) | cheap |
| 5 | **Replace FID and KL with task-relevant shift metrics** for the grant: robustness index (PathoROB) on our encoders, site-classifier AUROC, KID in deployed-model space, per-site MSS score distributions, and the P1 paired flip rate. Pre-register the Aim 2 contrasts in §3 | Answers the collaborator's Step 1 in a way reviewers accept. Kömen et al. (2024), de Jong et al. (2025), and PathoROB show FID-style image statistics miss what matters | moderate: framing and credibility, not accuracy | cheap |

**Explicitly deprioritise:** Macenko/Reinhard/Vahadane (tested, harmful), CycleGAN
(hallucination risk, and staining is not the driver), bag-level DANN, IRM, or group DRO with 6
sites and about 5 positives per small site (DomainBed; ComBat already negative), isotonic
calibration at n = 47 (use Platt or shrinkage), TENT, and retraining PALADIN's transformer from
scratch on our labels.

---

## References

Checked against PubMed (PMID or DOI), arXiv (ID), bioRxiv, or the publisher record.

**PALADIN, encoders, and slide models**
- Boehm KM, …, Sanchez-Vega F. Integrated histopathologic modeling of detailed tumor subtypes and
  actionable biomarkers. *bioRxiv* 2025. doi:10.1101/2025.08.14.670351 (PMID 40832221).
  Code: github.com/kmboehm/paladin. Tiling: github.com/pathology-data-mining/Mussel.
- Bioptimus. H-optimus-0 and H-optimus-1 model cards: 224 px, 0.5 µm/px, 1536-d.
  huggingface.co/bioptimus.
- Ding T, et al. Multimodal whole slide foundation model for pathology (TITAN). arXiv:2411.19666
  (2024). Model card: huggingface.co/MahmoodLab/TITAN.
- Filiot A, et al. Robustifying pathology foundation models via fine-tuning. arXiv:2607.22861
  (2026).
- Neidlinger P, et al. Benchmarking foundation models as feature extractors for weakly supervised
  computational pathology. *Nat Biomed Eng* 2025;10(6):1113–1123. doi:10.1038/s41551-025-01516-3.

**MSI models**
- Kather JN, et al. Deep learning can predict microsatellite instability directly from histology
  in gastrointestinal cancer. *Nat Med* 2019;25:1054. doi:10.1038/s41591-019-0462-y.
- Echle A, et al. Clinical-grade detection of microsatellite instability in colorectal tumors by
  deep learning. *Gastroenterology* 2020;159:1406. doi:10.1053/j.gastro.2020.06.021.
- Echle A, et al. Artificial intelligence for detection of microsatellite instability in
  colorectal cancer—a multicentric analysis of a pre-screening tool for clinical application.
  *ESMO Open* 2022;7(2):100400. doi:10.1016/j.esmoop.2022.100400.
- Wagner SJ, et al. Transformer-based biomarker prediction from colorectal cancer histology: a
  large-scale multicentric study. *Cancer Cell* 2023;41. doi:10.1016/j.ccell.2023.08.002.
- Saillard C, et al. Validation of MSIntuit as an AI-based pre-screening tool for MSI detection
  from colorectal cancer histology slides. *Nat Commun* 2023;14:6695.
  doi:10.1038/s41467-023-42453-6.
- Aldera AP, et al. Deep learning predicts microsatellite instability status in colorectal
  carcinoma in an ethnically heterogeneous population in South Africa. *J Clin Pathol*
  2025/2026;79(1):50–56. doi:10.1136/jcp-2025-210053.
- Baumann E, et al. MSAI-Path: predicting microsatellite instability from routine histology
  slides without reinventing the wheel. *Mod Pathol* 2025;39(1):100932.
  doi:10.1016/j.modpat.2025.100932.

**Robustness, site effects, and bias**
- Howard FM, et al. The impact of site-specific digital histology signatures on deep learning
  model accuracy and bias. *Nat Commun* 2021;12:4423. doi:10.1038/s41467-021-24698-1.
- Dehkharghanian T, et al. Biased data, biased AI: deep networks predict the acquisition site of
  TCGA images. *Diagn Pathol* 2023;18:67. doi:10.1186/s13000-023-01355-3.
- Kömen J, et al. Do histopathological foundation models eliminate batch effects? A comparative
  study. arXiv:2411.05489 (2024).
- de Jong ED, et al. Current pathology foundation models are unrobust to medical center
  differences. arXiv:2501.18055 (2025).
- Kömen J, de Jong ED, et al. Towards robust foundation models for digital pathology (PathoROB).
  *Nat Commun* 2026;17. doi:10.1038/s41467-026-73923-2. arXiv:2507.17845.
- Vaidya A, et al. Demographic bias in misdiagnosis by computational pathology models. *Nat Med*
  2024;30:1174–1190. doi:10.1038/s41591-024-02885-z.
- Daneshjou R, et al. Disparities in dermatology AI performance on a diverse, curated clinical
  image set. *Sci Adv* 2022;8. doi:10.1126/sciadv.abq6147.

**Stain normalisation, augmentation, and translation**
- Macenko M, et al. A method for normalizing histology slides for quantitative analysis. *ISBI*
  2009:1107–1110. doi:10.1109/ISBI.2009.5193250.
- Reinhard E, et al. Color transfer between images. *IEEE Comput Graph Appl* 2001;21(5):34–41.
  doi:10.1109/38.946629.
- Vahadane A, et al. Structure-preserving color normalization and sparse stain separation for
  histological images. *IEEE TMI* 2016;35:1962. doi:10.1109/TMI.2016.2529665.
- Tellez D, et al. Quantifying the effects of data augmentation and stain color normalization in
  convolutional neural networks for computational pathology. arXiv:1902.06543 (2019).
- Boschman J, et al. The utility of color normalization for AI-based diagnosis of hematoxylin and
  eosin-stained pathology images. *J Pathol* 2022;256(1):15–24. doi:10.1002/path.5797.
- Shen Y, et al. RandStainNA. arXiv:2206.12694 (2022).
- Zhu J-Y, et al. Unpaired image-to-image translation using cycle-consistent adversarial
  networks. arXiv:1703.10593 (2017).
- Cohen JP, Luck M, Honari S. Distribution matching losses can hallucinate features in medical
  image translation. *MICCAI* 2018. arXiv:1805.08841.
- Shaban MT, et al. StainGAN. arXiv:1804.01601 (2018).

**Shift metrics, domain generalisation, and adaptation**
- Heusel M, et al. GANs trained by a two time-scale update rule converge to a local Nash
  equilibrium (FID). *NeurIPS* 2017. arXiv:1706.08500.
- Bińkowski M, et al. Demystifying MMD GANs (KID). arXiv:1801.01401 (2018).
- Ganin Y, et al. Domain-adversarial training of neural networks. *JMLR* 2016;17:1–35.
  arXiv:1505.07818.
- Lafarge MW, et al. Domain-adversarial neural networks to address the appearance variability of
  histopathology images. arXiv:1707.06183 (2017).
- Gulrajani I, Lopez-Paz D. In search of lost domain generalization. arXiv:2007.01434.
- Zhou K, et al. Domain generalization with MixStyle. arXiv:2104.02008 (2021).
- Li X, et al. Uncertainty modeling for out-of-distribution generalization (DSU).
  arXiv:2202.03958 (2022).
- Sun B, Saenko K. Deep CORAL. arXiv:1607.01719 (2016).
- Wang D, et al. Tent: fully test-time adaptation by entropy minimization. arXiv:2006.10726.
- Schneider S, et al. Improving robustness against common corruptions by covariate shift
  adaptation. arXiv:2006.16971 (2020).
- Arjovsky M, et al. Invariant risk minimization. arXiv:1907.02893 (2019).
- Sagawa S, et al. Distributionally robust neural networks for group shifts. arXiv:1911.08731.
- Lipton ZC, et al. Detecting and correcting for label shift with black box predictors.
  arXiv:1802.03916 (2018).
- Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori
  probabilities. *Neural Comput* 2002;14(1):21–41. doi:10.1162/089976602753284446.
- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using
  empirical Bayes methods (ComBat). *Biostatistics* 2007;8:118 (PMID 16632515).
- Fortin J-P, et al. Harmonization of cortical thickness measurements across scanners and sites.
  *NeuroImage* 2018;167:104–120. doi:10.1016/j.neuroimage.2017.11.024.
- Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony.
  *Nat Methods* 2019;16:1289–1296. doi:10.1038/s41592-019-0619-0.
- Tuia D, Persello C, Bruzzone L. Domain adaptation for the classification of remote sensing
  data. *IEEE Geosci Remote Sens Mag* 2016;4(2):41–57. doi:10.1109/MGRS.2016.2548504.
- Hendrycks D, Dietterich T. Benchmarking neural network robustness to common corruptions and
  perturbations. arXiv:1903.12261.
- Geirhos R, et al. ImageNet-trained CNNs are biased towards texture. arXiv:1811.12231.
- Jackson PT, et al. Style augmentation. arXiv:1809.05375.

**MIL, weak labels, and transfer**
- Dietterich TG, Lathrop RH, Lozano-Pérez T. Solving the multiple instance problem with
  axis-parallel rectangles. *Artif Intell* 1997;89:31–71. doi:10.1016/S0004-3702(96)00034-3.
- Ilse M, Tomczak JM, Welling M. Attention-based deep multiple instance learning.
  arXiv:1802.04712 (2018).
- Shao Z, et al. TransMIL. arXiv:2106.00908 (2021).
- Lu MY, et al. Data-efficient and weakly supervised computational pathology on whole-slide
  images (CLAM). *Nat Biomed Eng* 2021;5:555. doi:10.1038/s41551-020-00682-w.
- Campanella G, et al. Clinical-grade computational pathology using weakly supervised deep
  learning on whole slide images. *Nat Med* 2019;25:1301. doi:10.1038/s41591-019-0508-1.
- Shao D, et al. Do multiple instance learning models transfer? arXiv:2506.09022 (2025).
- Wang L, et al. UntrimmedNets for weakly supervised action recognition and detection.
  arXiv:1703.03329 (2017).
- Kumar A, Raj B. Audio event detection using weakly labeled data. arXiv:1605.02401 (2016).
- Jean N, et al. Combining satellite imagery and machine learning to predict poverty. *Science*
  2016;353:790. doi:10.1126/science.aaf7894.
- Zankov DV, et al. QSAR modeling based on conformation ensembles using a multi-instance learning
  approach. *J Chem Inf Model* 2021;61(10):4913–4923. doi:10.1021/acs.jcim.1c00692.
- Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv:1503.02531.
- Gururangan S, et al. Don't stop pretraining. arXiv:2004.10964 (2020).
- Kang M, et al. Benchmarking self-supervised learning on diverse pathology datasets.
  arXiv:2212.04690 (2022).

**Evaluation and calibration**
- Varma S, Simon R. Bias in error estimation when using cross-validation for model selection.
  *BMC Bioinformatics* 2006;7:91. doi:10.1186/1471-2105-7-91.
- Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. *ICML*
  2005. doi:10.1145/1102351.1102430.
- Hanley JA, McNeil BJ. The meaning and use of the area under a ROC curve. *Radiology*
  1982;143:29 (PMID 7063747).
- DeLong ER, et al. Comparing the areas under two or more correlated ROC curves. *Biometrics*
  1988;44:837 (PMID 3203132).
- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free
  uncertainty quantification. arXiv:2107.07511.
