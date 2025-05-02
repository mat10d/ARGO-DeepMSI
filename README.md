# ARGO-DeepMSI: MSI Status Prediction from Histopathology

A deep learning framework for microsatellite instability (MSI) prediction from whole slide images of colorectal cancer.

## Project Overview

ARGO-DeepMSI implements a multi-stage pipeline to process whole slide images, extract features, and train deep learning models to predict MSI status. The workflow integrates REDCap data collection, multi-site processing, and cross-validation using multiple specialized environments.

## Environment Setup

### 1. ARGO Environment

The base environment used for data collection, processing, and validation:

```bash
# Create and activate environment for data collection/processing
conda env create -f argo_env.yml
conda activate argo
```

### 2. STAMP Environment

Used for feature extraction and model training:

```bash
# Create and activate conda environment for STAMP with Python 3.11
conda create -n stamp-env python=3.11
conda activate stamp-env

pip install "stamp[all] @ git+https://github.com/KatherLab/STAMP"
pip install seaborn wandb dgl torchdata==0.9.0 # for HistoBistro
```

### 3. TRIDENT Environment

For alternative feature extraction:

```bash
# Create and activate conda environment for TRIDENT with Python 3.10
conda create -n trident-env python=3.10
conda activate trident-env

pip install git+https://github.com/mahmoodlab/trident.git
```

### 4. HistoBistro Repository

For model comparison and benchmarking:

```bash
# Install directly into ARGO-DeepMSI
git clone https://github.com/peng-lab/HistoBistro.git

cd HistoBistro

# Create and activate conda environment for HistoBistro
conda env create --file environment_simple.yaml
conda activate histobistro
```

## Workflow Steps

### Step 0: AWS Slide Download and Data Collection

1. For AWS slide download and data transfer instructions, refer to `AWS.md`.

2. Create a `.env` file in the project root with your REDCap credentials:
   ```
   REDCAP_API_TOKEN=your_token_here
   REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
   ```

3. Process the raw data to create standardized tables:
   ```bash
   # Activate ARGO environment
   conda activate argo
   
   # Run the data processing script
   python scripts/0.prepare.py
   ```

   This script will:
   - Fetch patient data from REDCap
   - Extract MSI status information
   - Load and standardize Halo Link data files
   - Create clinical and slide tables
   - Verify slide existence and generate file paths
   - Clean and merge tables for consistency
   - Generate visualizations of MSI distribution by site

### Step 1: STAMP Processing

1. Initialize STAMP configs for each site:
   ```bash
   # Activate STAMP environment
   conda activate stamp-env
   
   # Make a directory for configs (if needed)
   mkdir -p configs
   
   # Initialize configs for each site
   stamp --config configs/config_OAUTHC.yaml init
   stamp --config configs/config_LUTH.yaml init
   stamp --config configs/config_UITH.yaml init
   stamp --config configs/config_LASUTH.yaml init
   stamp --config configs/config_retrospective_msk.yaml init
   stamp --config configs/config_retrospective_oau.yaml init
   ```

2. Edit the configuration files in the `configs/` directory to match your data paths

3. Run preprocessing using the SLURM job scheduler:
   ```bash
   cd scripts
   sbatch 1.preprocess.sh
   ```

### Step 2: Validation of Preprocessing

Verify that all slides were processed correctly:

```bash
# Switch to ARGO environment
conda activate argo

# Run validation script
python scripts/2.preprocess_eval.py
```

This script will:
- Validate processing status of all slides
- Generate visualizations of processing completion
- Create split tables by site
- Prepare tables for the next steps

### Step 3: Consolidating Features for Cross-Validation

To prepare for cross-validation, we consolidate features from all sites into a single directory:

```bash
cd scripts
sbatch 3.cross_validation.sh
```

This script includes the following consolidation code:

```bash
# Create a target directory for consolidated features
mkdir -p "$TARGET_DIR"

# Copy all feature data from individual repos to the consolidated directory
for dir in ../data/*/features/xiyuewang-ctranspath-7c998680-02627079/; do
    # Skip the target directory itself to avoid copying a directory into itself
    if [[ "$dir" != "../data/all/features/xiyuewang-ctranspath-7c998680-02627079/" ]]; then
        echo "Copying files from $dir to $TARGET_DIR"
        rsync -av "$dir"/* "$TARGET_DIR"
    fi
done
```

After consolidation, the script runs cross-validation on the consolidated dataset.

### Step 4: Statistical Analysis

Generate performance statistics with:

```bash
cd scripts
sbatch 4.statistics.sh
```

This will create:
- ROC curves with confidence intervals
- Precision-recall curves
- AUROC, AUPRC with 95% confidence intervals
- Performance metrics by split and aggregated
- All outputs saved in `results/statistics`

### Step 6-7: Using HistoBistro Models (Optional)

If you want to compare with or use HistoBistro models:

1. Prepare data for HistoBistro format (script 6):
   ```bash
   # Switch to ARGO environment
   conda activate argo
   python scripts/6.prepare_histobistro.py
   ```

2. Run HistoBistro evaluation (script 7):
   ```bash
   sbatch scripts/7.histobistro.sh
   ```

## Project Structure

```
argo-deepmsi/
├── scripts/
│   ├── 0.prepare.py         # REDCap data collection
│   ├── 1.preprocess.sh      # STAMP feature extraction
│   ├── 2.preprocess_eval.py # Processing validation
│   ├── 3.cross_validation.sh # Consolidation and cross-validation
│   ├── 4.statistics.sh      # Generate performance metrics
│   ├── 6.prepare_histobistro.py # Format data for HistoBistro
│   └── 7.histobistro.sh     # Run HistoBistro comparison
├── data/                    # All slide data (gitignored)
│   ├── OAUTHC/
│   │   ├── raw/             # Original SVS files
│   │   ├── features/        # Extracted features
│   │   ├── results/         # Model results
│   │   └── .cache/          # STAMP cache
│   ├── LUTH/
│   │   └── ...
│   ├── UITH/
│   │   └── ...
│   ├── LASUTH/
│   │   └── ...
│   ├── retrospective_msk/
│   │   └── ...
│   ├── retrospective_oau/
│   │   └── ...
│   └── all/                 # Consolidated data
│       ├── features/        # Combined features
│       └── results/         # Combined results
├── tables/                  # Generated CSV tables
│   ├── 0/                   # Initial tables
│   │   ├── clinical_table.csv
│   │   └── slide_table.csv
│   └── 2/                   # Post-processing tables
│       ├── all_clinical_table.csv
│       ├── all_slide_table.csv
│       ├── OAUTHC_clinical_table.csv
│       └── ...
├── configs/                 # STAMP configuration files
│   ├── config_LUTH.yaml
│   ├── config_OAUTHC.yaml
│   ├── config_UITH.yaml
│   ├── config_LASUTH.yaml
│   ├── config_retrospective_msk.yaml
│   ├── config_retrospective_oau.yaml
│   └── config_all.yaml      # Config for consolidated data
├── HistoBistro/             # Optional comparison framework -- contains updated: config.yaml, data_config.yaml, environment_simple.yaml
├── .env                     # Environment variables (gitignored) 
├── .gitignore               # Git ignore file
├── AWS.md                   # AWS data transfer guide
├── argo_env.yml             # ARGO conda environment specification
└── README.md                # Project documentation
```

## Notes

- ARGO environment (conda) is used for data collection and validation
- STAMP environment (venv) is used only for STAMP processing
- STAMP configurations are institution-specific and stored in `configs/`
- AWS credentials should be configured system-wide using `aws configure`
- All generated tables are stored in `tables/` directory
- The script numbering (0, 1, 2, 3, 4, 6, 7) directly corresponds to the workflow steps

## References

- [STAMP](https://github.com/KatherLab/STAMP): Slide-level Transformer with Attention Multiple-instance learning for Pathology
- [HistoBistro](https://github.com/peng-lab/HistoBistro): Benchmarking platform for histopathology AI models
- [TRIDENT](https://github.com/mahmoodlab/trident): Tool for Rapid and Integrated Development Environment