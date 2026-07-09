# A4 — TITAN morphological descriptions (EXPLORATORY) — BLOCKED at the capability gate

## Gate-first result (checked BEFORE any modeling): capability ABSENT
The A4 hypothesis needed TITAN to **generate** a per-slide morphological description (a
caption / report) that could be read out for MSI. The gate — "does the TITAN checkpoint on
fry expose a text-generation/caption head, not just the contrastive text encoder?" — was
checked first, and the answer is **no**. A4 is therefore **blocked**; no descriptions were
generated or committed (fabricating them would violate the provenance rule).

## Evidence (from the cached checkpoint's own remote code, this run)
Cached at `.huggingface_cache/modules/transformers_modules/MahmoodLab/TITAN/<rev>/`:

- `modeling_titan.py` — `class Titan(PreTrainedModel)`. It does **not** subclass
  `GenerationMixin`, has **no** `generate` / `caption` / `decode` method, and holds no text
  decoder attribute. Its complete public API is:
  `return_conch`, `encode_slide_from_patch_features`, `encode_text` (contrastive text
  **encoder**, `normalize` only), `zero_shot_classifier`, `zero_shot`.
- `text_transformer.py` — a contrastive text encoder; no autoregressive decoder, no
  `generate`, no `lm_head`. The "CoCa" references in config/vision/text modules are
  contrastive-setup parameters (e.g. appended CLS-embed pooling), not an instantiated
  captioning decoder.
- `conch_v1_5.py` — a pure `VisionTransformer` patch encoder (no text head at all).

TITAN's language capability as shipped is **contrastive alignment + zero-shot
classification**, not free-text generation. There is no head to produce a morphological
description, so the described-morphology→MSI readout A4 proposed cannot be built from this
checkpoint without an external captioner (which the no-external-data regime forbids as a
new fitted/pretrained text model on top).

## Provenance note (per AGENT.md rule)
- Model identity verified from source this run: Hugging Face repo `MahmoodLab/TITAN`
  (cached remote code above). 
- TITAN paper author/year/journal were **not retrieved from source in this run** — flagged,
  not asserted. Any A6 citation of TITAN must fetch the reference from source first.

## Disposition
- **Status: blocked** (capability gate failed). This is a documented dead-end, not an
  implementation failure: the zero-shot/contrastive text path is already covered by the
  S3 `tip_adapter` / `vl_text_cosine` scorers (both underperformed), so there is no
  untried language lever left on this checkpoint.
- **For A6 synthesis:** treat A4 as a resolved null — the vision-language *generation* angle
  is unavailable on the fry TITAN checkpoint; the A-phase OAUTHC story rests on A0 (Harmony,
  frontier 0.683), A1/A2 (normalization falsified), A3 (adaptation real but caps ~0.61),
  and A5 (per-site calibration).
