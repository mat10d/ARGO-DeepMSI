# ARGO-DeepMSI Folder Structure

## Design Principles

1. **`data/`** = INPUT DATA ONLY (never modified by pipeline)
2. **`results/`** = ALL PIPELINE OUTPUTS (organized by stage)
3. **Stage-based organization** = Clear provenance of all outputs
4. **No cryptic names** = Descriptive folder names, not "0", "2", etc.

---

## New Structure

```
ARGO-DeepMSI/
│
├── data/                           # INPUT DATA (raw, never modified)
│   ├── raw/                        # Raw whole slide images (SVS)
│   │   ├── OAUTHC/
│   │   ├── LUTH/
│   │   ├── LASUTH/
│   │   ├── UITH/
│   │   ├── retrospective_msk/
│   │   └── retrospective_oau/
│   └── metadata/                   # Metadata exports
│       ├── halo_link_oauthc_export.csv
│       ├── halo_link_luth_export.csv
│       └── ...
│
├── results/                        # ALL PIPELINE OUTPUTS
│   ├── stage1_data_ingestion/
│   │   ├── clinical_table.csv      # Patient-level: PATIENT, isMSIH
│   │   ├── slide_table.csv         # Slide-level: PATIENT, FILENAME, SITE
│   │   └── figures/
│   │       ├── patient_count_by_site.png
│   │       ├── slide_count_by_site.png
│   │       ├── slides_per_patient.png
│   │       ├── overall_msi_distribution.png
│   │       └── patient_summary_by_site.csv
│   │
│   ├── stage2_qc/
│   │   ├── qc_report.csv           # Slide-level QC results
│   │   └── figures/
│   │
│   ├── stage3_features/            # Extracted features (H5 files)
│   │   ├── ctranspath/             # By model
│   │   │   ├── OAUTHC/
│   │   │   │   └── slide_001.h5
│   │   │   ├── LUTH/
│   │   │   └── ...
│   │   ├── h-optimus-0/
│   │   ├── virchow2/
│   │   └── uni2/
│   │
│   ├── stage4_feature_validation/
│   │   ├── tables/
│   │   │   ├── all_clinical_table.csv         # Consolidated
│   │   │   ├── all_slide_table.csv            # FILENAME = path to H5
│   │   │   ├── OAUTHC_clinical_table.csv      # Site-specific
│   │   │   ├── OAUTHC_slide_table.csv
│   │   │   └── ...
│   │   ├── reports/
│   │   │   ├── missing_slides.csv             # Slides without features
│   │   │   ├── missing_patients.csv           # Patients without features
│   │   │   └── extraction_summary_by_site.csv
│   │   └── figures/
│   │       ├── processing_status_by_site.png
│   │       ├── processing_heatmap_slides.png
│   │       └── processing_heatmap_patients.png
│   │
│   ├── stage5_baseline/
│   │   ├── predictions.csv         # HistoBistro predictions
│   │   └── figures/
│   │       └── roc_curve.png
│   │
│   ├── stage6_training/            # MIL training outputs
│   │   ├── ctranspath/
│   │   │   └── crossval/
│   │   │       ├── split-0/
│   │   │       │   ├── patient-preds.csv
│   │   │       │   └── model.ckpt
│   │   │       ├── split-1/
│   │   │       └── split-2/
│   │   ├── virchow2/
│   │   └── ...
│   │
│   ├── stage7_statistics/
│   │   ├── ctranspath/
│   │   │   ├── metrics.csv         # AUROC, AUPRC, CI
│   │   │   └── confusion_matrix.csv
│   │   └── ...
│   │
│   └── stage8_visualization/
│       └── figures/
│           ├── roc_curves_comparison.png
│           ├── model_performance_heatmap.png
│           └── embedding_tsne.png
│
├── logs/                           # Execution logs
│   ├── data_ingestion/
│   │   └── data_ingestion_20250103_142530.log
│   ├── feature_validation/
│   └── ...
│
├── configs/                        # STAMP model configs
│   ├── ctranspath/
│   │   ├── config_all.yaml
│   │   ├── config_OAUTHC.yaml
│   │   └── ...
│   └── virchow2/
│
├── argo_deepmsi/                   # Python package
├── scripts/                        # CLI scripts
├── environments/                   # Conda environments
└── ...
```

---

## Migration from Old Structure

### Old → New Mapping

| Old Path | New Path | Reason |
|----------|----------|--------|
| `tables/0/clinical_table.csv` | `results/stage1_data_ingestion/clinical_table.csv` | Stage-based naming |
| `tables/2/all_clinical_table.csv` | `results/stage4_feature_validation/tables/all_clinical_table.csv` | Stage-based, organized |
| `data/{SITE}/features/{MODEL}/` | `results/stage3_features/{MODEL}/{SITE}/` | Features are outputs, not inputs |
| `data/{SITE}/results/crossval/` | `results/stage6_training/{MODEL}/crossval/` | Training outputs in results/ |
| `visualizations/0/` | `results/stage1_data_ingestion/figures/` | Stage-based organization |
| `visualizations/2/` | `results/stage4_feature_validation/figures/` | Stage-based organization |

### Breaking Changes

**Scripts affected** (will update):
- `scripts/1_data_ingestion.py` - Update output paths
- `scripts/4_feature_validation.py` - Update input/output paths
- `scripts/5_baseline_testing.py` - Update paths
- All SLURM scripts - Update feature paths

**Modules affected**:
- `argo_deepmsi/io_utils.py` - Update path utility functions
- `argo_deepmsi/data_ingestion.py` - Update output directory
- `argo_deepmsi/feature_validation.py` - Update feature search paths

---

## Benefits of New Structure

✅ **Clear separation**: Inputs (`data/`) vs outputs (`results/`)
✅ **Stage-based provenance**: Easy to see which stage produced what
✅ **Descriptive names**: No cryptic "0", "2" folders
✅ **Organized by model**: Features and training outputs grouped by model
✅ **Easy cleanup**: Delete `results/` to rerun pipeline
✅ **Scalable**: Easy to add new models or sites

---

## Implementation Notes

1. **Backward compatibility**: Old paths will be deprecated with warnings
2. **Migration script**: Create utility to move existing data to new structure
3. **Default behavior**: New structure used by default, old structure supported temporarily
4. **Documentation**: Update all docs to reflect new structure
