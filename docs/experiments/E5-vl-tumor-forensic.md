# E5 — VL tumor forensic: can the language encoder adjudicate Wagner's failures?

**Date:** 2026-08-05
**Question:** for Wagner's confident MSS→MSI-H false positives, does a vision-language encoder
(CONCH v1.5 / TITAN text) *also* read the tumor region as MSI-H-like (genuine morphology mimicry),
or not (Wagner-specific)?
**Method:** per-tile cosine of cached `conch_v1.5_tiles` to curated MSI-H vs MSS prompts,
restricted to tumor tiles, aggregated per patient. Reproduce: `sbatch scripts/vl_tumor_forensic.sh`
→ `results/analysis/error_anatomy/vl_tumor/`.

## Result — the instrument is invalid on this cohort

| metric | value |
|---|---:|
| patients scored | 195 |
| **tumor VL-MSI AUROC on TRUE labels** | **0.518 (chance)** |
| mean tumor VL-MSI — true MSI-H | −0.0134 |
| mean tumor VL-MSI — true MSS (TN) | −0.0078 |
| mean tumor VL-MSI — confident FP | −0.0141 |

The zero-shot VL-tumor axis **cannot separate true MSI-H from true MSS** (AUROC 0.52), and the
reference means are indistinguishable and slightly inverted. The quality axis shows no group
difference.

**Do not interpret `fp_position = 1.11`** from the summary JSON: it normalizes by the
(MSS→MSI-H) reference gap, which here is near-zero and wrong-signed, so the ratio is a
divide-by-noise artifact, not evidence that false positives look MSI-H.

## Reading

- Because the VL-tumor axis fails on the true labels, it **cannot adjudicate** whether Wagner's
  false positives are genuine morphology mimicry. The natural-language lever does not help here.
- This is consistent with every prior VL/text result on this cohort: `vl_text_cosine` slide-level
  ~0.56, S3 Tip-Adapter/CoOp below chance, R2/R3 text-anchored bottlenecks failed. Zero-shot
  pathology-CLIP text alignment does not carry usable MSI signal on Nigerian tumor tissue with
  these prompts.
- The per-case P_0382 VL numbers (E-case) should therefore be read as noise, not signal — as
  cautioned when they were produced.

## Implication

- **Retire the zero-shot VL/NL lever** for failure adjudication (unless a fundamentally different
  prompt/model is proposed). It is a well-replicated dead end.
- The genuine hand-evaluation path is **image review** — a contact sheet of the actual
  top-attention tumor tiles for a pathologist — not a VL proxy.
- The remaining "new signal" lever for the hard confident-FP core is the **Waiv acquisition-robust
  encoders** (now fully extracted, 803 slides) and **adjudicated labels**, evaluated with nested
  patient-grouped validation.

## Note for reuse

`scripts/vl_tumor_forensic.py` guards nothing about instrument validity — always read the
`tumor_vl_msi_auroc_true_labels` field first; if it is ~0.5, the placement metrics below it are
meaningless by construction.
