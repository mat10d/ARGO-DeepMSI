# wagner_zeroshot

**Module:** `argo_deepmsi.scorers.wagner_zeroshot`
**Resolution:** slide
**Trained on our cohort:** no (external pretrained model, applied zero-shot)

## Mechanism

Wagner et al. (HistoBistro, *Cancer Cell* 2023) MSI transformer trained
on ~13K Western CRC slides with CTransPath tile features. Architecture:

    Transformer(input_dim=768, dim=512, depth=2, heads=8, mlp_dim=512, pool='cls')

We apply it zero-shot to the Nigerian cohort using our own CTransPath
embeddings (`tables/ctranspath_tiles` in each slide's zarr). For each
slide we forward the full bag of tiles and take `sigmoid(cls_logit)` as
`p_msih`.

Patient aggregation is **max / √n_slides** (the C5 best baseline).

## Inputs

- Per-slide `tables/ctranspath_tiles` in each slide zarr
- `clinical_table.csv` for ground-truth `isMSIH`
- Pretrained weights at `old/HistoBistro/CancerCellCRCTransformer/trained_models/MSI_high_CRC_model.pth`

## Outputs

- `results/scorers/wagner_zeroshot/slide_scores.csv` (canonical schema)
- `results/scorers/wagner_zeroshot/metadata.json`
- Legacy raw output: `results/analysis/wagner_zeroshot/`

## Results

| metric            | full cohort | QC-clean |
|-------------------|-------------|----------|
| slide AUROC       | 0.572       | tbd      |
| patient AUROC     | 0.717       | 0.710    |

Per-site (clean, patient AUROC) — see `results/comparison/per_site_clean.csv`.

## Why this is the reference baseline

Trained on 13K patients in the West, no Nigerian exposure. Sets the
"how far does Western pretraining get us" floor. Every other scorer is
ranked against it.
