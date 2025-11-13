# ARGO-DeepMSI Development Roadmap

## Current Status (Updated 2025-11-13)
- **Branch**: `overhaul` (active refactoring)
- **Goal**: Clean, standardized, end-to-end MSI prediction pipeline with 12+ models
- **Strategy**: Refactor → Test → Expand
- **Latest Focus**: Documentation cleanup and streamlining

## Major Milestones Completed ✅

### Phase 1: Repository Structure (COMPLETED)
- ✅ Implemented installable `argo-deepmsi` package with `pyproject.toml`
- ✅ Refactored all scripts to use package functions (thin CLI wrappers)
- ✅ **Clean folder structure**:
  - `data/` - Input data only (raw WSI files, metadata)
  - `results/` - All pipeline outputs (stage-based naming)
  - Template-based STAMP configuration (no per-model config files)
- ✅ Path management centralized in `argo_deepmsi/io_utils.py`
- ✅ All 8 pipeline stages implemented with modular package architecture
- ✅ Upgraded to Python 3.11 across environments
- ✅ Template-based config generation for STAMP models

### Phase 2: Core Functionality (COMPLETED)
- ✅ Stage 1: Data ingestion (REDCap + Halo metadata)
- ✅ Stage 2: Quality control (placeholder)
- ✅ Stage 3: Feature extraction (template-based, 12+ models supported)
- ✅ Stage 4: Feature validation (extraction QC and reporting)
- ✅ Stage 5: Baseline testing (HistoBistro integration)
- ✅ Stage 6: MIL training (STAMP cross-validation)
- ✅ Stage 7: Statistics (AUROC/AUPRC with CI)
- ✅ Stage 8: Visualization (heatmaps, ROC curves)

## Current Focus 🔄

### Documentation Streamlining (IN PROGRESS)
- ✅ Updated CLAUDE.md to reflect 8-stage pipeline
- ✅ environments/README.md comprehensive and up-to-date
- ⏳ Cleaning up TODO.md (this file)
- ⏳ Archiving temporary session notes

### Next Steps
1. **Test end-to-end pipeline** - Run stages 1-8 on small dataset
2. **Feature extraction at scale** - Extract features for all 12 models
3. **Model comparison** - Compare performance across all models
4. **Merge to main** - Clean merge after validation

---

## Project Vision

Create a **clean, modular, reproducible** MSI prediction pipeline:

### Core Pipeline (8 Stages)
1. **Data Ingestion**: SVS files + metadata from REDCap → cleaned tables
2. **Quality Control**: Slide-level QC (tissue quality, artifacts, staining)
3. **Feature Extraction**: Extract features using ALL available STAMP models
4. **Feature Validation**: QC extracted features - which passed/failed and why
5. **Baseline Testing**: Test pre-trained HistoBistro model (validates data quality)
6. **MIL Training**: K-fold cross-validation with STAMP
7. **Statistics**: Performance metrics (AUROC, AUPRC, CI)
8. **Visualization**: Heatmaps, embeddings, interpretability

### Code Organization
```
argo_deepmsi/       # Installable Python package (core logic)
scripts/            # Thin CLI wrappers (call package functions)
configs/            # YAML configurations per model
```

**Principle**: Heavy logic in package, thin scripts for CLI

**Critical Design Principles:**
1. **NEVER modify STAMP or HistoBistro repositories**
   - Keep them as-is, treat as external dependencies
   - All custom code goes in `argo_deepmsi/` package and `scripts/`
   - This allows easy updates: `cd STAMP && git pull`
   - We call STAMP/HistoBistro as tools, not modify them

2. **argo-deepmsi is an installable package**
   - Heavy logic in `argo_deepmsi/` module
   - Scripts are thin wrappers calling package functions
   - Installed via: `pip install -e .` (editable mode)
   - Allows future expansion (models, layers, etc.)

---

## Phase 1: Repository Structure & Cleanup (COMPLETED ✅)

### 1.0 Environment Management (COMPLETED ✅)

**Three Separate Environments:**

1. **ARGO Environment** (conda, Python 3.11)
   - Purpose: Data ingestion, QC, validation, statistics, visualization
   - Location: `environments/argo.yml`
   - Used in: Stages 1, 2, 4, 7, 8

2. **STAMP Environment** (uv, Python 3.11+)
   - Purpose: Feature extraction and MIL training
   - Location: `STAMP/.venv/`
   - Used in: Stages 3, 6

3. **HistoBistro Environment** (conda, Python 3.10)
   - Purpose: Pre-trained model inference
   - Location: `environments/histobistro.yml`
   - Used in: Stage 5

**Completed:**
- ✅ Upgraded ARGO environment to Python 3.11
- ✅ Organized environment files in `environments/` directory
- ✅ Comprehensive environment setup guide in `environments/README.md`
- ✅ All environment variables documented
- ✅ STAMP installation instructions (must install on compute node)
- ✅ Hugging Face authentication documented

### 1.1 Package Module Structure (COMPLETED ✅)

**Implemented as `argo_deepmsi/` package:**
```
argo_deepmsi/
├── __init__.py
├── data_ingestion.py      # REDCap API, SVS discovery, table creation
├── quality_control.py     # Slide QC functions
├── feature_extraction.py  # STAMP preprocessing wrappers
├── feature_validation.py  # Feature QC and reporting
├── baseline_testing.py    # HistoBistro inference
├── training.py            # STAMP crossval wrappers
├── statistics.py          # Metrics calculation
├── visualization.py       # Plotting, heatmaps, embeddings
├── config_utils.py        # Config generation and validation
└── io_utils.py            # File I/O, logging, paths
```

**Completed:**
- ✅ Created installable `argo_deepmsi` package with `pyproject.toml`
- ✅ All modules implemented with proper structure
- ✅ Error handling and logging throughout
- ✅ Docstrings and type hints added
- ✅ Scripts refactored as thin wrappers calling package functions

### 1.2 Refactor Existing Scripts (COMPLETED ✅)

**Scripts refactored as thin wrappers:**
- ✅ `scripts/1_data_ingestion.py` → calls `argo_deepmsi.data_ingestion`
- ✅ `scripts/2_quality_control.py` → calls `argo_deepmsi.quality_control`
- ✅ `scripts/3_feature_extraction.sh` → SLURM wrapper with template-based config generation
- ✅ `scripts/4_feature_validation.py` → calls `argo_deepmsi.feature_validation`
- ✅ `scripts/5_baseline_testing.py` → calls `argo_deepmsi.baseline_testing`
- ✅ `scripts/6_mil_training.sh` → SLURM wrapper for STAMP cross-validation
- ✅ `scripts/7_statistics.py` → calls `argo_deepmsi.statistics`
- ✅ `scripts/8_visualization.py` → calls `argo_deepmsi.visualization`

**All scripts follow the pattern:**
```python
import argparse
from argo_deepmsi import module_name, io_utils

def main():
    args = parse_args()
    # Call package functions (logic in package, not script)
    result = module_name.main_function(args)
    print(f"✓ Complete. Results saved to {result.output_dir}")

if __name__ == "__main__":
    main()
```

### 1.3 SLURM Scripts (COMPLETED ✅)

**Implemented SLURM wrappers:**
- ✅ `scripts/3_feature_extraction.sh` - Array job for 6 sites, GPU required
- ✅ `scripts/3_feature_extraction_all.sh` - Sequential submission of all models
- ✅ `scripts/6_mil_training.sh` - Cross-validation training
- ✅ All scripts include: environment activation, GPU checks, config generation, error handling

### 1.4 Directory Structure (COMPLETED ✅)

**Implemented clean folder structure:**
```
ARGO-DeepMSI/
├── README.md, CLAUDE.md, TODO.md    # Documentation
├── pyproject.toml                   # Package definition
├── argo_deepmsi/                    # Installable package (all logic)
├── scripts/                         # Thin CLI wrappers
│   ├── 1_data_ingestion.py
│   ├── 2_quality_control.py
│   ├── 3_feature_extraction.sh
│   ├── 3_feature_extraction_all.sh
│   ├── 4_feature_validation.py
│   ├── 5_baseline_testing.py
│   ├── 6_mil_training.sh
│   ├── 7_statistics.py
│   ├── 8_visualization.py
│   ├── generate_config.py
│   └── test_model_access.py
├── configs/templates/               # YAML templates
├── data/                            # INPUT: raw WSI files, metadata
├── results/                         # OUTPUT: all pipeline outputs by stage
├── logs/                            # Execution logs
├── environments/                    # Environment YAML files
├── STAMP/                           # External dependency
└── HistoBistro/                     # External dependency
```

**Key improvements:**
- ✅ Clean separation: `data/` (inputs) vs `results/` (outputs)
- ✅ Stage-based naming: `results/stage{N}_{NAME}/`
- ✅ Template-based configs (no per-model files)
- ✅ Centralized path management in `argo_deepmsi/io_utils.py`

---

## Phase 2: Core Functionality (COMPLETED ✅)

All 8 pipeline stages have been implemented with modular package architecture.

### 2.1 Stage 1: Data Ingestion (COMPLETED ✅)

**Implemented in `argo_deepmsi/data_ingestion.py`:**
- ✅ REDCap API integration
- ✅ Halo Link CSV processing
- ✅ Clinical and slide table generation
- ✅ MSI status extraction and validation
- ✅ Output to `results/stage1_data_ingestion/`

### 2.2 Stage 2: Quality Control (COMPLETED ✅)

**Implemented in `argo_deepmsi/quality_control.py`:**
- ✅ Placeholder implementation (ready for QC tool integration)
- ✅ Output to `results/stage2_qc/`

### 2.3 Stage 3: Feature Extraction (COMPLETED ✅)

**Implemented template-based feature extraction:**
- ✅ Template-based config generation (`scripts/generate_config.py`)
- ✅ SLURM array job for parallel processing (`scripts/3_feature_extraction.sh`)
- ✅ Batch processing script for all models (`scripts/3_feature_extraction_all.sh`)
- ✅ Model accessibility testing (`scripts/test_model_access.py`)
- ✅ Support for 12 STAMP models (no-auth and gated)
- ✅ Output to `results/stage3_features/{MODEL}/{SITE}/`

### 2.4 Stage 4: Feature Validation (COMPLETED ✅)

**Implemented in `argo_deepmsi/feature_validation.py`:**
- ✅ Feature file existence checking across all models
- ✅ Feature integrity validation (dimensions, NaN/Inf detection)
- ✅ Per-site and per-model extraction statistics
- ✅ Extraction success rate reporting
- ✅ Output to `results/stage4_feature_validation/`

### 2.5 Stage 5: Baseline Testing (COMPLETED ✅)

**Implemented in `argo_deepmsi/baseline_testing.py`:**
- ✅ HistoBistro model integration
- ✅ Pre-trained model inference for data validation
- ✅ Performance metrics (AUROC, sensitivity, NPV)
- ✅ Output to `results/stage5_baseline/`

### 2.6 Stage 6: MIL Training (COMPLETED ✅)

**Implemented via SLURM wrapper:**
- ✅ Template-based training config generation
- ✅ STAMP cross-validation integration (`scripts/6_mil_training.sh`)
- ✅ k-fold cross-validation (default 3 splits)
- ✅ Output to `results/stage6_training/{MODEL}/crossval/`

### 2.7 Stage 7: Statistics (COMPLETED ✅)

**Implemented in `argo_deepmsi/statistics.py`:**
- ✅ Aggregation of cross-validation predictions
- ✅ AUROC/AUPRC calculation with 95% CI
- ✅ Per-site and per-model metrics
- ✅ Model comparison tables
- ✅ Output to `results/stage7_statistics/{MODEL}/`

### 2.8 Stage 8: Visualization (COMPLETED ✅)

**Implemented in `argo_deepmsi/visualization.py`:**
- ✅ ROC and PR curve generation
- ✅ Confusion matrix plotting
- ✅ STAMP heatmap generation wrapper
- ✅ Multi-model comparison plots
- ✅ Output to `results/stage8_visualization/`

---

## Phase 3: Testing & Validation (NEXT)

### 3.1 End-to-End Pipeline Testing

**Goal**: Validate the entire pipeline works on real data

**Test Plan:**
1. ⏳ Create ARGO environment and test Stage 1 (data ingestion)
2. ⏳ Run feature extraction for multiple models (ctranspath, virchow2, uni2)
3. ⏳ Validate feature extraction success rates (Stage 4)
4. ⏳ Run HistoBistro baseline (Stage 5) - compare with published 0.99 NPV
5. ⏳ Run MIL training (Stage 6) for at least one model
6. ⏳ Generate statistics and visualizations (Stages 7-8)
7. ⏳ Verify outputs at each stage are correct
8. ⏳ Document any issues or bugs found

### 3.2 Large-Scale Feature Extraction

**Goal**: Extract features for all 12 STAMP models

**Tasks:**
- [ ] Run `bash scripts/3_feature_extraction_all.sh`
- [ ] Monitor extraction jobs (check SLURM queue, logs)
- [ ] Validate feature extraction success rates per model
- [ ] Identify any problematic slides or models
- [ ] Document model-specific issues (memory, authentication, etc.)

### 3.3 Documentation (PARTIALLY COMPLETED ✅)

**Completed:**
- ✅ README.md - Comprehensive user guide with 8-stage pipeline
- ✅ CLAUDE.md - Updated AI assistant guide (Nov 2025)
- ✅ environments/README.md - Complete environment setup guide
- ✅ FEATURE_EXTRACTION_GUIDE.md - Technical extraction details
- ✅ QUICKSTART_FEATURE_EXTRACTION.md - Quick start guide

**Remaining:**
- [ ] Create CHANGELOG.md - Track version history and major changes
- [ ] Add examples/ directory with sample configs and outputs
- [ ] Create CONTRIBUTING.md (if planning external contributions)

---

## Phase 4: Merge to Main (PENDING)

**Pre-Merge Checklist:**
- ✅ All 8 pipeline stages implemented
- ✅ Package structure complete with docstrings
- ✅ SLURM scripts implemented
- ⏳ End-to-end pipeline tested on real data
- ✅ Documentation complete (README, CLAUDE, guides)
- ✅ Clean folder structure (data/ vs results/)
- [ ] Validate on full dataset
- [ ] No sensitive data in repo (.env excluded)

**Merge Process:**
```bash
# When ready:
git checkout main
git merge overhaul --no-ff -m "Major refactor: 8-stage pipeline with modular package architecture"
git tag -a v2.0.0 -m "Clean, standardized multi-model MSI prediction pipeline"
git push origin main --tags
```

---

## Phase 5: Future Work (Post-Merge)

### 5.1 Multi-Model Evaluation
- [ ] Extract features for all 12 STAMP models
- [ ] Train and evaluate all models
- [ ] Compare performance (AUROC, AUPRC, CI)
- [ ] Identify top-performing models
- [ ] Statistical significance testing across models

### 5.2 Advanced Features
- [ ] Hyperparameter tuning for top models
- [ ] Ensemble methods (model averaging, stacking)
- [ ] External validation on independent cohorts
- [ ] Prospective validation

### 5.3 Quality Control Enhancements
- [ ] Integrate HistoQC or similar QC tools
- [ ] Quantify impact of QC on model performance
- [ ] Retrain models on QC-filtered data

---

## Key Milestones

| Milestone | Status |
|-----------|--------|
| Package structure created | ✅ Complete |
| Scripts refactored (thin wrappers) | ✅ Complete |
| Clean folder structure | ✅ Complete |
| Template-based configs | ✅ Complete |
| All 8 stages implemented | ✅ Complete |
| Documentation updated | ✅ Complete |
| **End-to-end pipeline tested** | ⏳ **In Progress** |
| Large-scale feature extraction | ⏳ Pending |
| Multi-model evaluation | ⏳ Pending |
| Merge to main | ⏳ Pending |

---

## Success Criteria

**For merging to main:**
1. ✅ Clean, modular code structure (argo_deepmsi package + thin scripts)
2. ⏳ All 8 stages run end-to-end on real data without errors
3. ✅ Feature validation identifies extraction issues clearly
4. ⏳ HistoBistro baseline validates data quality (target: ~0.99 NPV)
5. ⏳ At least 3 models successfully extracted features
6. ⏳ STAMP MIL training completes for at least one model
7. ✅ Comprehensive documentation exists
8. ⏳ Results reproducible from clean environment setup

**Quality Standards Met:**
- ✅ No hardcoded paths (centralized in io_utils.py)
- ✅ Proper error handling throughout
- ✅ Logging at all stages
- ✅ Template-driven configs (no magic numbers)
- ✅ Clean git history and documentation
