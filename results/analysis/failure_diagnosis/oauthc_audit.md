# 1d — OAUTHC Prospective Deep Dive

- OAUTHC (prospective): 476 slides, 81 patients, prevalence 0.174, Wagner AUROC 0.439.
- retrospective_oau: 172 slides, 83 patients, prevalence 0.221, Wagner AUROC 0.804.

## Patient overlap

- Patients with slides in BOTH OAUTHC prospective and retrospective_oau: **0**
- Prospective-only patients: 81
- Retro-only patients: 83

## Tissue area (n_tiles as proxy)

- **OAUTHC prospective** n_tiles: median=4435  p25=2062  p75=7417  min=211  max=24072
- **retrospective_oau** n_tiles: median=2731  p25=1366  p75=4932  min=87  max=10036
- **LASUTH** n_tiles: median=11410  p25=3750  p75=13096  min=1460  max=20121
- **LUTH** n_tiles: median=13952  p25=1786  p75=17218  min=55  max=26046
- **UITH** n_tiles: median=3610  p25=1676  p75=7063  min=745  max=12117
- **retrospective_msk** n_tiles: median=2849  p25=1403  p75=4999  min=97  max=9995

## Label derivation source by site

| site | prospective_cmo | retrospective_mmr | NA |
|---|---|---|---|
| LASUTH | 15 | 0 | 0 |
| LUTH | 31 | 0 | 0 |
| OAUTHC | 476 | 0 | 0 |
| UITH | 12 | 0 | 0 |
| retrospective_msk | 0 | 97 | 0 |
| retrospective_oau | 0 | 172 | 0 |

No patient overlap between OAUTHC prospective and retrospective_oau; label-concordance audit not possible without patient linkage.
