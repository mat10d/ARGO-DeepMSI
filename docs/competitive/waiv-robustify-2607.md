# Waiv — "Robustifying pathology foundation models via fine-tuning"

**Source:** Filiot, Thaeter, Schmauch, Guillou. arXiv:2607.22861v1, 24 Jul 2026. Waiv
(Owkin spin-off; Filiot is the Phikon/Phikon-v2 author). PDF: `docs/2607.22861v1.pdf`.
All figures/numbers below read from that PDF in this run.

## What they claim
A **label-free fine-tuning recipe** that makes tile-level pathology FMs robust to
acquisition factors (scanner / stain / lab) **without a downstream trade-off**. Applied
to 10 FMs. Headline: PathoROB robustness index **0.72 → 0.87** avg (+23%); combined
cross-benchmark performance +43% over Patho-Bench/HEST/THUNDER. Biggest robustness lifts:
Phikon-v2 0.47→0.81, Prov-GigaPath 0.62→0.89, H-Optimus-0 0.81→0.92. They release two
fine-tuned encoders: **Phaet** (Phikon-v2) and **Mascaret** (Midnight-12k).

Cleanest evidence (Fig 2): Phikon-v2 features cluster by **medical center** (ARI 0.46);
after fine-tuning they cluster by **biology** (ARI 0.67; center collapses to 0.01) — on 5
Camelyon centers **not seen during fine-tuning**. Real generalization of learned invariance.

Mechanism (Figs 3–4): acquisition shift *looks* like a near-linear feature-space offset
(Fig 3, PLISM) **but removing it needs deep backbone adaptation** (Fig 4, SCORPION
cross-scanner retrieval; invariance builds across transformer depth). They explicitly state
feature-level ops ("statistics matching or CORAL-style covariance alignment") "can partially
reduce the shift but are not deep enough to remove it entirely."

## Why this matters to us
Their central mechanistic claim is **exactly what our A-phase found independently on OAUTHC**:
- A1 (ComBat + location/scale) erased the batch axis but MSI didn't move.
- A2 (stain normalization) went below chance.

The field leader just published the theoretical backing for our feature-level negative
results. Worth banking and citable.

## Skeptical caveats (why this is not a silver bullet for OAUTHC)
1. **Their robustness ≠ our task.** PathoROB measures whether biology dominates center in
   kNN neighborhood structure — a proxy. Their Limitations admit it "does not directly
   measure downstream robustness under domain shift for specific tasks such as biomarker
   prediction." MSI is a biomarker task; robustness-index lift ≠ OAUTHC AUROC lift.
2. **Our A1 argues against a pure-nuisance story.** Their method removes scanner/stain
   *nuisance* directions. We erased the batch axis and OAUTHC MSI didn't improve, suggesting
   OAUTHC's deficit may be entangled biology/label-quality, not a removable offset — exactly
   what their method targets, and exactly where it might not reach.
3. **They exclude our base.** No CONCH v1.5 / TITAN among the 10 backbones; they scope out
   vision-language / slide encoders. Phaet/Mascaret are **tile encoders** → an encoder swap,
   not a drop-in to `conch_v1.5_titan`. Downstream aggregation is ours to choose (mean/ABMIL).
4. **The recipe is withheld.** No methods section describes the fine-tuning objective/data.
   They ship **weights (product)**, not the **method (moat)**. We can consume the outputs;
   we cannot replicate what made them.

## Integration (landed 2026-07-31)
Both `wearewaiv/phaet` and `wearewaiv/mascaret` verified on HuggingFace:
- **Gated** — non-commercial academic license (needs institutional email + ORCID). A human
  must accept the license in-browser before any download.
- Custom remote code: `AutoModel.from_pretrained(repo, trust_remote_code=True)`, exposing
  `.encode(pixel_values) -> (B, D)`.
- **Phaet 1024-d**, transform Resize224/CenterCrop224/Normalize(ImageNet).
- **Mascaret 1536-d** (L2-norm CLS), transform Resize224/CenterCrop224/Normalize(0.5,0.5,0.5).

Wired as thin `ImageModel` subclasses in `argo_deepmsi/models/waiv.py`, registered into
LazySlide's global `MODEL_REGISTRY` (import `argo_deepmsi.models` fires the decorators).
No lazyslide upgrade (0.10.0 already accepts a custom `ImageModel`; the `.encode` string
path resolves via the registry). Registration triggered in both extraction entry points
(`feature_extraction.py`, `scripts/extract_dask.py` worker). Frozen pretrained weights →
allowed under CONTRACT (no new training data). `trust_remote_code=True` executes Waiv's
published encoder code.

The model card requires `transformers>=5.14,<6` and `safetensors>=0.8`; the current project
environment has Transformers 4.57.6. The compatible stack is therefore an explicit
`pip install -e ".[waiv]"` opt-in in a cloned/dedicated extraction environment and must be
installed before extraction once access is approved. This avoids changing the environment
used for the existing encoders.

## Planned experiment (blocked on provider approval)
On 2026-08-03 the configured Hugging Face identity (`mat10d`) could enumerate both repositories,
but downloading the actual Phaet snapshot returned HTTP 403: the gated access request is awaiting
review by the repository authors. This is no longer a local authentication/configuration issue.

1. `sbatch scripts/extract_dask.sh <full-808 table> --models phaet mascaret` — extracts
   `phaet_tiles`/`mascaret_tiles` into the existing per-slide zarrs (incremental; the 3
   existing models are skipped). Full cohort so D2's "drop the hard QC gate" applies and
   OAUTHC keeps ~413 slides.
2. Aggregate **mean** (D3: mean is best for the OAUTHC-relevant probe) → embeddings.
3. Race OAUTHC vs our Harmony **0.713** ceiling — the clean "does an encoder-level fix beat
   feature-level correction" test. Expectations tempered by caveat #2.
