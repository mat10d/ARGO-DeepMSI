# E3 — Wagner error anatomy, Stage C (spatial attention introspection)

**Date:** 2026-08-04
**Subset:** the confident, still-unexplained error patients from Stage A (attributed_cause ∈
{unexplained, site-structured}) + matched correct (TN) controls.
**Method:** a numerically-inert attention hook (`argo_deepmsi.models.wagner.capture_attention`)
records the last block's CLS→tile attention (only the CLS row is computed — O(n), so it does not
OOM on large slides). Wagner runs on **cached CTransPath features**, so Stage C is CPU-only and
does not contend with GPU extraction. Attention mass on tumor tiles uses the existing
`results/data/tumor_tiles/<slide>.npy` masks.
**Reproduce:** `sbatch scripts/error_anatomy_c.sh` → `results/analysis/error_anatomy/C_*`.

## Result

| group | slides | mean tumor-attention mass | mean tumor-tile fraction | attention / availability |
|---|---:|---:|---:|---:|
| control (TN) | 59 | 0.273 | 0.250 | 1.09× |
| error (confident FP) | 76 | 0.205 | 0.194 | 1.06× |

## Reading

- **The errors are not a spatial mis-attention problem.** Wagner puts attention on tumor tiles at
  essentially the same rate relative to how much tumor is present (~1.06–1.09× the tumor-tile
  fraction) whether it is right or wrong. Error slides simply have slightly less tumor available,
  and attention tracks that availability the same way in both groups.
- **This is the genuine representational residual.** On its confident false positives Wagner
  attends to the same tissue mix it does when correct, and still reads Nigerian MSS morphology as
  MSI-H-like. There is no "it looked at pen marks / fold / background" story to exploit — a
  tumor-restricted tiling or attention fix would not convert these errors.
- **Note on the low absolute tumor attention (~0.2–0.27).** Wagner spends most of its attention on
  non-tumor (stroma/immune) tissue in *both* groups. That is plausibly appropriate — MSI-H
  morphology (Crohn's-like reaction, TILs) lives partly in stroma — and it does not distinguish
  errors from correct calls, so it is not the failure lever.

## Implication for the recipe

The ~29% "unexplained confident FP" bucket is a **true hard core**, not a fixable QC/tiling
artifact. The honest levers for it are (a) selective abstention at deployment (T-phase), and
(b) new signal — an acquisition-robust or differently-pretrained encoder (the Waiv test, in
progress) or adjudicated labels — **not** another aggregation/attention head on the same
CTransPath features.

## Caveats

- Attention mass is descriptive evidence, not causal attribution.
- Slides without a cached tumor mask are skipped; the comparison is within the masked subset
  (76 error + 59 control slides).
- Controls are the highest-scoring TN patients (nearest the boundary), a conservative comparison.
