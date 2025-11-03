# ARGO-DeepMSI Repository Cleanup & Standardization

## Current Status
- **Branch**: `overhaul` (development branch)
- **Goal**: Clean, standardized, end-to-end MSI prediction pipeline
- **Strategy**: Refactor → Test → Merge → Expand

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

## Phase 1: Repository Structure & Cleanup

### 1.0 Environment Management

**Goal**: Clean, reproducible environments for each component

**Three Separate Environments:**

1. **ARGO-DeepMSI Environment** (for utils + scripts)
   - Python: 3.10+
   - Manager: conda
   - Purpose: Data ingestion, QC, validation, statistics, visualization
   - Key dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, requests

2. **STAMP Environment** (for feature extraction + MIL training)
   - Python: 3.12 (per STAMP v2.3)
   - Manager: uv (per STAMP requirements)
   - Purpose: Feature extraction and MIL training
   - Key dependencies: torch, transformers, timm, lightning, huggingface_hub

3. **HistoBistro Environment** (for baseline validation)
   - Python: 3.10
   - Manager: conda
   - Purpose: Pre-trained model inference
   - Key dependencies: torch, pytorch-lightning, etc.

**Current Status:**
- ✅ `argo_env.yml` exists (Python 3.9, pandas, scikit-learn, matplotlib, requests)
- ✅ `HistoBistro/environment_simple.yaml` exists (Python 3.10.9, PyTorch 2.0, Lightning)
- ✅ STAMP has its own environment (STAMP/.venv, managed via uv)

**Tasks:**
- [ ] Review and potentially update `argo_env.yml` (currently Python 3.9)
  - Consider upgrading to Python 3.10 for better compatibility
  - Add any missing dependencies for QC and validation
- [ ] Organize environment files:
  - Move `argo_env.yml` → `environments/argo.yml`
  - Create symlink or document HistoBistro environment location
- [ ] Create `scripts/setup_environments.sh` automation script:
  ```bash
  # 1. Create ARGO environment
  conda env create -f environments/argo.yml

  # 2. Setup STAMP (uv-based)
  cd STAMP && uv sync --extra build --extra gpu

  # 3. Setup HistoBistro
  conda env create -f HistoBistro/environment_simple.yaml
  ```
- [ ] Document environment activation in README:
  - Stage 1-2, 4, 6-7: `conda activate argo`
  - Stage 3, 5: `source STAMP/.venv/bin/activate`
  - HistoBistro baseline: `conda activate histobistro`
- [ ] Add environment variables to documentation:
  - `HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache`
  - `REDCAP_API_TOKEN` (from .env)
- [ ] Test all three environments install and work correctly

**Final Environment Structure:**
```
ARGO-DeepMSI/
├── environments/
│   ├── argo.yml              # Main utils env (Python 3.9 → 3.10)
│   ├── setup.sh              # Environment setup automation
│   └── README.md             # Environment docs
├── STAMP/
│   ├── .venv/                # STAMP environment (uv-managed, Python 3.12)
│   └── pyproject.toml        # STAMP dependencies
└── HistoBistro/
    └── environment_simple.yaml  # HistoBistro env (Python 3.10.9)
```

### 1.1 Create Utils Module Structure
```
utils/
├── __init__.py
├── data_ingestion.py      # REDCap API, SVS discovery, table creation
├── quality_control.py     # Slide QC functions
├── feature_extraction.py  # STAMP preprocessing wrappers
├── model_validation.py    # HistoBistro inference
├── training.py            # STAMP crossval wrappers
├── statistics.py          # Metrics calculation
├── visualization.py       # Plotting, heatmaps, embeddings
├── config_utils.py        # Config generation and validation
└── io_utils.py            # File I/O, logging, paths
```

**Tasks:**
- [ ] Create `utils/` directory structure
- [ ] Move reusable code from scripts into utils
- [ ] Add proper error handling and logging
- [ ] Add docstrings to all functions
- [ ] Add type hints
- [ ] Create `utils/__init__.py` with clean imports

### 1.2 Refactor Existing Scripts

**Before (monolithic scripts):**
```
scripts/0.prepare.py  (does everything inline)
```

**After (modular scripts):**
```python
# scripts/1_data_ingestion.py
from utils.data_ingestion import fetch_redcap_data, create_clinical_table
from utils.io_utils import setup_logging

def main():
    logger = setup_logging("data_ingestion")
    clinical_df = fetch_redcap_data(...)
    # etc.
```

**Script Implementation (Thin Wrappers):**
- [ ] `scripts/1_data_ingestion.py` → calls `argo_deepmsi.data_ingestion`
- [ ] `scripts/2_quality_control.py` → calls `argo_deepmsi.quality_control`
- [ ] `scripts/3_feature_extraction.py` → calls `argo_deepmsi.feature_extraction`
- [ ] `scripts/4_feature_validation.py` → calls `argo_deepmsi.feature_validation`
- [ ] `scripts/5_baseline_testing.py` → calls `argo_deepmsi.baseline_testing`
- [ ] `scripts/6_mil_training.py` → calls `argo_deepmsi.training`
- [ ] `scripts/7_statistics.py` → calls `argo_deepmsi.statistics`
- [ ] `scripts/8_visualization.py` → calls `argo_deepmsi.visualization`

**Example Script Structure:**
```python
# scripts/4_feature_validation.py
import argparse
from argo_deepmsi import feature_validation, io_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-table", required=True)
    parser.add_argument("--feature-dir", required=True)
    args = parser.parse_args()

    # All logic in package
    report = feature_validation.generate_extraction_report(
        args.slide_table, args.feature_dir
    )
    report.save("results/feature_validation/")

if __name__ == "__main__":
    main()
```

### 1.3 Standardize SLURM Scripts

Create consistent SLURM wrappers:
```
scripts/slurm/
├── 2_qc_array.sh              # Quality control (array job)
├── 3_extract_features.sh      # Feature extraction (array job, per model)
├── 4_baseline_validation.sh   # HistoBistro baseline
├── 5_mil_training.sh          # STAMP crossval
└── submit_all.sh              # Master submission script
```

**Tasks:**
- [ ] Create `scripts/slurm/` directory
- [ ] Standardize SLURM headers (partition, resources, logging)
- [ ] Add pre-flight checks (GPU, env, configs)
- [ ] Add error handling and notifications
- [ ] Create master orchestration script

### 1.4 Directory Structure

**Final structure:**
```
ARGO-DeepMSI/
├── README.md                    # User-facing documentation
├── TODO.md                      # This file
├── CLAUDE.md                    # AI assistant guide
├── PIPELINE.md                  # Technical pipeline details
├── pyproject.toml               # Package definition
├── setup.py                     # Package installation
│
├── argo_deepmsi/                # Installable Python package
│   ├── __init__.py              # Package init, version
│   ├── data_ingestion.py        # Stage 1
│   ├── quality_control.py       # Stage 2
│   ├── feature_extraction.py    # Stage 3
│   ├── feature_validation.py    # Stage 4 (NEW)
│   ├── baseline_testing.py      # Stage 5 (HistoBistro)
│   ├── training.py              # Stage 6 (STAMP MIL)
│   ├── statistics.py            # Stage 7
│   ├── visualization.py         # Stage 8
│   ├── config_utils.py
│   └── io_utils.py
│
├── scripts/                     # Executable scripts
│   ├── 1_data_ingestion.py
│   ├── 2_quality_control.py
│   ├── 3_feature_extraction.py
│   ├── 4_feature_validation.py      # NEW: Validate extracted features
│   ├── 5_baseline_testing.py        # HistoBistro pre-trained model
│   ├── 6_mil_training.py
│   ├── 7_statistics.py
│   ├── 8_visualization.py
│   └── slurm/                       # SLURM batch scripts
│       ├── 2_qc_array.sh
│       ├── 3_extract_features.sh
│       ├── 5_baseline_testing.sh
│       ├── 6_mil_training.sh
│       └── submit_all.sh
│
├── configs/                     # Model configurations
│   ├── template.yaml            # Base template
│   ├── ctranspath/              # Per-site configs
│   ├── virchow2/
│   ├── uni2/
│   └── ...
│
├── data/                        # Processed data (gitignored)
│   ├── OAUTHC/
│   ├── LUTH/
│   ├── LASUTH/
│   ├── UITH/
│   ├── retrospective_msk/
│   ├── retrospective_oau/
│   └── all/                     # Consolidated
│
├── tables/                      # Clinical/slide metadata
│   ├── clinical_table.csv
│   ├── slide_table.csv
│   └── qc_report.csv
│
├── results/                     # Pipeline outputs
│   ├── qc/                      # QC reports
│   ├── features/                # Feature summaries
│   ├── baseline/                # HistoBistro results
│   ├── crossval/                # STAMP crossval results
│   ├── statistics/              # Metrics
│   └── figures/                 # Visualizations
│
├── logs/                        # Execution logs
│   ├── data_ingestion/
│   ├── qc/
│   ├── feature_extraction/
│   └── ...
│
├── environments/
│   ├── argo.yml                 # Main environment
│   ├── stamp.txt                # STAMP requirements
│   └── histobistro.yml
│
├── STAMP/                       # Git submodule
├── HistoBistro/                 # Git submodule
│
└── tests/                       # Unit tests (future)
    └── test_utils.py
```

**Tasks:**
- [ ] Create all necessary directories
- [ ] Add `.gitkeep` for empty dirs
- [ ] Update `.gitignore` appropriately
- [ ] Document structure in README

---

## Phase 2: Implement Core Functionality

### 2.1 Stage 1: Data Ingestion

**Goal**: Clean, validated clinical and slide tables

**Utils Functions:**
```python
# utils/data_ingestion.py
def fetch_redcap_data(api_url, api_token) -> pd.DataFrame
def discover_svs_files(data_dir) -> List[Path]
def create_clinical_table(redcap_df) -> pd.DataFrame
def create_slide_table(svs_files, clinical_df) -> pd.DataFrame
def validate_tables(clinical_df, slide_df) -> bool
```

**Script:**
```bash
python scripts/1_data_ingestion.py \
  --redcap-url $REDCAP_URL \
  --redcap-token $REDCAP_TOKEN \
  --data-dir data/ \
  --output-dir tables/
```

**Tasks:**
- [ ] Implement `utils/data_ingestion.py`
- [ ] Implement `scripts/1_data_ingestion.py`
- [ ] Add logging and error handling
- [ ] Test on existing data
- [ ] Document expected outputs

### 2.2 Stage 2: Quality Control

**Goal**: Filter slides based on tissue quality, staining, artifacts

**Utils Functions:**
```python
# utils/quality_control.py
def run_qc_pipeline(wsi_path, qc_tool="histoqc") -> Dict
def parse_qc_results(qc_output) -> pd.DataFrame
def apply_qc_thresholds(qc_df, thresholds) -> pd.DataFrame
def generate_qc_report(qc_df) -> str
```

**Script:**
```bash
# Sequential
python scripts/2_quality_control.py \
  --slide-table tables/slide_table.csv \
  --wsi-dir data/{SITE}/raw/ \
  --output-dir results/qc/

# Parallel (SLURM)
sbatch scripts/slurm/2_qc_array.sh
```

**Tasks:**
- [ ] Research QC tool options (HistoQC, STAMP, custom)
- [ ] Implement `utils/quality_control.py`
- [ ] Implement `scripts/2_quality_control.py`
- [ ] Create SLURM array job
- [ ] Define QC pass/fail thresholds
- [ ] Generate QC visualization report

### 2.3 Stage 3: Feature Extraction (ALL STAMP Models)

**Goal**: Extract features for all available models

**Available STAMP v2.3 Models (18 total):**
1. CTransPath ✓
2. H-optimus-0 ✓
3. H-optimus-1 ✓
4. Virchow2 (Priority)
5. UNI2 (Priority)
6. CONCHv1.5 (Priority)
7. Gigapath (Priority)
8. MUSK
9. mSTAR
10. DinoBloom
11. PLIP
12. Virchow (v1)
13. UNI (v1)
14. CONCH
15. CHIEF-CTransPath
16. Empty (baseline)
17-18. Others TBD

**Utils Functions:**
```python
# utils/feature_extraction.py
def generate_stamp_configs(model_name, sites, base_config) -> List[Path]
def run_stamp_preprocessing(config_path) -> bool
def validate_features(feature_dir, slide_table) -> pd.DataFrame
def consolidate_features(site_dirs, output_dir) -> None
```

**Script:**
```bash
# Single model, single site
python scripts/3_feature_extraction.py \
  --model virchow2 \
  --site OAUTHC \
  --config configs/virchow2/config_OAUTHC.yaml

# All models, all sites (SLURM)
sbatch scripts/slurm/3_extract_features.sh virchow2
```

**Tasks:**
- [ ] Implement `utils/feature_extraction.py`
- [ ] Implement `scripts/3_feature_extraction.py`
- [ ] Create config generator: `utils/config_utils.py`
- [ ] Generate configs for all 18 models × 6 sites
- [ ] Create SLURM array job with model parameter
- [ ] Add resume capability for failed jobs
- [ ] Implement feature validation checks
- [ ] Document expected feature formats

### 2.4 Stage 4: Feature Validation

**Goal**: Understand which slides successfully extracted features, which failed, and why

**What to Check:**
- Which slides have features extracted successfully
- Which slides failed during extraction (errors, timeouts, corrupted files)
- Feature file integrity (correct dimensions, no NaN/Inf values)
- Feature statistics per slide (mean, std, distribution)
- Missing tiles or incomplete coverage
- Per-site extraction success rates
- Cross-model comparison (do same slides fail across models?)

**Utils Functions:**
```python
# utils/feature_validation.py
def check_feature_files_exist(slide_table, feature_dir) -> pd.DataFrame
def validate_feature_integrity(feature_path) -> Dict[str, Any]
def parse_extraction_logs(log_dir) -> pd.DataFrame
def generate_extraction_report(slide_table, feature_dirs, models) -> pd.DataFrame
def plot_extraction_statistics(report_df) -> Figure
def identify_problematic_slides(report_df, threshold=0.8) -> List[str]
```

**Script:**
```bash
python scripts/4_feature_validation.py \
  --slide-table tables/slide_table.csv \
  --feature-base-dir data/ \
  --models ctranspath,virchow2,uni2 \
  --output-dir results/feature_validation/
```

**Output:**
- `results/feature_validation/extraction_report.csv` - Per-slide, per-model status
- `results/feature_validation/failed_slides.csv` - Slides that failed extraction
- `results/feature_validation/statistics.json` - Summary statistics
- `results/feature_validation/figures/` - Visualization of success rates

**Tasks:**
- [ ] Implement `utils/feature_validation.py`
- [ ] Implement `scripts/4_feature_validation.py`
- [ ] Check feature file existence across all models
- [ ] Validate feature dimensions and integrity
- [ ] Parse STAMP preprocessing logs for errors
- [ ] Generate per-site, per-model extraction statistics
- [ ] Create visualization of extraction success rates
- [ ] Identify and document problematic slides
- [ ] Generate actionable recommendations (re-run, exclude, etc.)

### 2.5 Stage 5: Baseline Testing (HistoBistro)

**Goal**: Validate data quality by testing with published pre-trained model

**Why This Matters:**
- Proves our feature extraction is correct
- Establishes performance benchmark to beat
- Validates our data against published 0.99 sensitivity/NPV results
- Identifies any systematic data issues early

**Pre-trained Model:**
- `HistoBistro/CancerCellCRCTransformer/trained_models/MSI_high_CRC_model.pth`
- Trained on 13K+ patients from 16 cohorts
- Published performance: 0.99 sensitivity, 0.99 NPV
- Uses CTransPath features (768-dim)

**Utils Functions:**
```python
# utils/baseline_testing.py
def load_histobistro_model(checkpoint_path) -> nn.Module
def prepare_features_for_histobistro(feature_dir) -> Dict
def run_histobistro_inference(model, features) -> pd.DataFrame
def calculate_metrics(predictions, ground_truth) -> Dict
def generate_baseline_report(metrics) -> str
def compare_with_published_results(metrics, published_metrics) -> Dict
```

**Script:**
```bash
python scripts/5_baseline_testing.py \
  --model-checkpoint HistoBistro/CancerCellCRCTransformer/trained_models/MSI_high_CRC_model.pth \
  --feature-dir data/all/features/xiyuewang-ctranspath-*/ \
  --clinical-table tables/clinical_table.csv \
  --output-dir results/baseline/
```

**Tasks:**
- [ ] Implement `utils/baseline_testing.py`
- [ ] Implement `scripts/5_baseline_testing.py`
- [ ] Test HistoBistro model loads correctly
- [ ] Validate feature format compatibility
- [ ] Generate predictions for all patients
- [ ] Calculate AUROC, AUPRC, Sensitivity, NPV with 95% CI
- [ ] Compare with published results
- [ ] Generate baseline report

### 2.6 Stage 6: MIL Training (STAMP K-Fold)

**Goal**: Train STAMP MIL models with k-fold cross-validation

**Utils Functions:**
```python
# utils/training.py
def prepare_stamp_crossval_config(model_name, n_splits=3) -> Path
def run_stamp_crossval(config_path) -> bool
def extract_crossval_results(output_dir) -> pd.DataFrame
def save_best_checkpoints(crossval_dir, output_dir) -> None
```

**Script:**
```bash
# Single model
python scripts/6_mil_training.py \
  --model virchow2 \
  --config configs/virchow2/config_all.yaml \
  --n-splits 3

# Via SLURM
sbatch scripts/slurm/6_mil_training.sh virchow2
```

**Tasks:**
- [ ] Implement `utils/training.py`
- [ ] Implement `scripts/6_mil_training.py`
- [ ] Create SLURM script for training
- [ ] Test on one model (CTransPath)
- [ ] Validate checkpoint saving
- [ ] Extract predictions from all folds
- [ ] Document training hyperparameters

### 2.7 Stage 7: Statistics

**Goal**: Generate comprehensive performance metrics

**Utils Functions:**
```python
# utils/statistics.py
def aggregate_crossval_predictions(split_dirs) -> pd.DataFrame
def calculate_auroc_with_ci(y_true, y_pred) -> Tuple[float, float, float]
def calculate_auprc_with_ci(y_true, y_pred) -> Tuple[float, float, float]
def calculate_confusion_metrics(y_true, y_pred, threshold) -> Dict
def compare_models(results_dict) -> pd.DataFrame
def generate_statistics_report(metrics) -> str
```

**Script:**
```bash
python scripts/7_statistics.py \
  --model virchow2 \
  --crossval-dir data/all/results/crossval/ \
  --output-dir results/statistics/
```

**Tasks:**
- [ ] Implement `utils/statistics.py`
- [ ] Implement `scripts/7_statistics.py`
- [ ] Aggregate predictions across folds
- [ ] Calculate metrics with 95% CI (bootstrap)
- [ ] Generate per-site statistics
- [ ] Create comparison table across models
- [ ] Statistical significance testing (DeLong)

### 2.8 Stage 8: Visualization

**Goal**: Generate heatmaps, embeddings, interpretability plots

**Utils Functions:**
```python
# utils/visualization.py
def generate_roc_curves(results_dict) -> Figure
def generate_pr_curves(results_dict) -> Figure
def plot_confusion_matrix(y_true, y_pred) -> Figure
def generate_stamp_heatmaps(config_path, checkpoint) -> None
def plot_embeddings_umap(features, labels) -> Figure
def visualize_top_tiles(heatmap_dir) -> Figure
```

**Script:**
```bash
# ROC/PR curves
python scripts/8_visualization.py \
  --mode curves \
  --results-dir results/statistics/ \
  --output-dir results/figures/

# Heatmaps via STAMP
python scripts/8_visualization.py \
  --mode heatmaps \
  --model virchow2 \
  --config configs/virchow2/config_all.yaml \
  --checkpoint data/all/results/training/best_model.ckpt
```

**Tasks:**
- [ ] Implement `utils/visualization.py`
- [ ] Implement `scripts/8_visualization.py`
- [ ] Generate ROC/PR curves for all models
- [ ] Create STAMP heatmaps wrapper
- [ ] Generate UMAP/t-SNE embeddings
- [ ] Create top-k tile visualizations
- [ ] Generate publication-quality figures

---

## Phase 3: End-to-End Testing

### 3.1 Integration Testing

**Test Pipeline:**
1. Run data ingestion
2. Run QC on small subset
3. Extract features for ONE model (CTransPath - already done)
4. Validate extracted features (check which passed/failed)
5. Run HistoBistro baseline (validates data quality)
6. Run STAMP crossval (1 split, small dataset)
7. Generate statistics
8. Create visualizations

**Tasks:**
- [ ] Create test dataset (small subset)
- [ ] Run end-to-end pipeline manually
- [ ] Verify outputs at each stage
- [ ] Check logs for errors
- [ ] Validate metrics make sense
- [ ] Document any issues found

### 3.2 Create Master Pipeline Script

```bash
# scripts/run_pipeline.sh
#!/bin/bash
# Master pipeline orchestrator

set -e  # Exit on error

# Stage 1: Data Ingestion
python scripts/1_data_ingestion.py --config config.yaml

# Stage 2: Quality Control
sbatch --wait scripts/slurm/2_qc_array.sh

# Stage 3: Feature Extraction (all models)
for model in virchow2 uni2 conch1_5; do
  sbatch --wait scripts/slurm/3_extract_features.sh $model
done

# Stage 4: Feature Validation
python scripts/4_feature_validation.py

# Stage 5: Baseline Testing
python scripts/5_baseline_testing.py

# Stage 6: MIL Training (all models)
for model in virchow2 uni2 conch1_5; do
  sbatch --wait scripts/slurm/6_mil_training.sh $model
done

# Stage 7: Statistics
python scripts/7_statistics.py --all-models

# Stage 8: Visualization
python scripts/8_visualization.py --all-models
```

**Tasks:**
- [ ] Create `scripts/run_pipeline.sh`
- [ ] Add error handling and logging
- [ ] Add progress notifications
- [ ] Add resume capability
- [ ] Test on full dataset
- [ ] Document runtime estimates

### 3.3 Documentation

**Create comprehensive docs:**
- [ ] **README.md**: User guide, quick start, examples
- [ ] **PIPELINE.md**: Technical details, data flow, formats
- [ ] **CLAUDE.md**: Update with new structure
- [ ] **CONTRIBUTING.md**: Code style, adding models
- [ ] **CHANGELOG.md**: Track major changes

**README Structure:**
```markdown
# ARGO-DeepMSI

## Overview
[What it does, who it's for]

## Installation
[Environment setup]

## Quick Start
[Minimal working example]

## Pipeline Stages
[Detailed stage-by-stage guide]

## Available Models
[List of 18 STAMP models]

## Results
[Where to find outputs]

## Troubleshooting
[Common issues]

## Citation
[How to cite]
```

**Tasks:**
- [ ] Write comprehensive README.md
- [ ] Create PIPELINE.md with data flow diagrams
- [ ] Update CLAUDE.md
- [ ] Add inline code documentation
- [ ] Create examples directory

---

## Phase 4: Merge to Main

### 4.1 Pre-Merge Checklist

- [ ] All 8 pipeline stages implemented and tested
- [ ] Utils module complete with docstrings
- [ ] SLURM scripts tested on cluster
- [ ] End-to-end pipeline runs successfully
- [ ] Documentation complete
- [ ] Code follows style guidelines
- [ ] Git history clean (consider squashing)
- [ ] No sensitive data in repo (.env excluded)

### 4.2 Merge Process

```bash
# Final checks
git status
git log --oneline

# Merge to main
git checkout main
git merge overhaul --no-ff -m "Refactor: Standardize pipeline with modular utils structure"

# Tag release
git tag -a v2.0.0 -m "Clean, standardized multi-model pipeline"
git push origin main --tags
```

### 4.3 Post-Merge

- [ ] Update main branch README
- [ ] Archive old scripts (if needed)
- [ ] Update documentation links
- [ ] Notify collaborators
- [ ] Plan Phase 5 (expansion)

---

## Phase 5: Expansion (Post-Merge)

### 5.1 Additional Models

- [ ] Run feature extraction for remaining STAMP models (8-18)
- [ ] Evaluate performance across all models
- [ ] Identify top performers

### 5.2 Advanced Features

- [ ] Hyperparameter tuning
- [ ] Ensemble methods
- [ ] External validation datasets
- [ ] Clinical deployment pipeline

### 5.3 Advanced QC

- [ ] Integrate advanced QC tools
- [ ] Retrain models on QC-filtered data
- [ ] Quantify QC impact

---

## Key Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| Utils module created | Week 1 | ⏳ Not started |
| Scripts refactored | Week 1-2 | ⏳ Not started |
| Stage 1-2 working | Week 2 | ⏳ Not started |
| Stage 3-5 working | Week 2-3 | ⏳ Not started |
| Stage 6-7 working | Week 3 | ⏳ Not started |
| End-to-end tested | Week 3 | ⏳ Not started |
| Documentation complete | Week 4 | ⏳ Not started |
| Merge to main | Week 4 | ⏳ Not started |

---

## Success Criteria

**Before merging, ensure:**
1. ✅ Clean, modular code structure (utils + scripts)
2. ✅ All 8 stages run end-to-end without errors
3. ✅ Feature validation identifies extraction issues clearly
4. ✅ HistoBistro baseline validates data quality (close to 0.99 NPV)
5. ✅ At least 3 new models successfully extracted features
6. ✅ STAMP MIL training completes for all models
7. ✅ Comprehensive documentation exists
8. ✅ Results reproducible from clean repo clone

**Quality gates:**
- Code reviewed
- No hardcoded paths
- Proper error handling
- Logging at all stages
- Config-driven (no magic numbers)
