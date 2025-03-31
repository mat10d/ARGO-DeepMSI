# ARGO-deepmsi Workflow

## Environment Setup

### 1. ARGO Environment
```bash
# Create and activate environment for data collection/processing
conda env create -f argo_env.yml
conda activate argo
```

### 2. STAMP Environment
```bash
# Create and activate conda environment for STAMP with python 3.11
conda create -n stamp-env python=3.11
conda activate stamp-env

pip install "stamp[all] @ git+https://github.com/KatherLab/STAMP"
```

### 3. TRIDENT Environment

```bash
# Create and activate conda environment for TRIDENT with python 3.10
conda create -n trident-env python=3.10
conda activate trident-env

pip install git+https://github.com/mahmoodlab/trident.git
```

## Workflow Steps

### Step 1: AWS Slide Download

See `AWS.md`.

### Step 2: Collect and Process MSI Data

1. Create a `.env` file in the project root directory with your REDCap credentials:
```
REDCAP_API_TOKEN=your_token_here
REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
```

2. Run the data processing script to:
   - Fetch patient data from REDCap using environment variables
   - Extract MSI status information 
   - Load and standardize Halo Link data files
   - Create clinical and slide tables
   - Verify slide existence and generate file paths
   - Clean and merge tables for consistency
   - Generate visualizations of MSI distribution by site
   - Split data by site for further analysis

```bash
conda activate argo
python src/process.py
```

The script will create organized data tables in the `tables` directory and visualizations in the `visualizations` directory, with detailed console output showing the data processing steps.

### Step 3: STAMP Processing

1. Initialize STAMP configs:
```bash
# Activate STAMP environment
conda activate stamp-env

# Make a directory to store configs
mkdir configs

# Initialize configs
stamp --config configs/config_OAUTHC.yaml init
stamp --config configs/config_LUTH.yaml init
stamp --config configs/config_UITH.yaml init
stamp --config configs/config_LASUTH.yaml init
stamp --config configs/config_retrospective_MSK.yaml init
stamp --config configs/config_retrospective_OAU.yaml init
```

2. Edit STAMP configs

3. Run preprocessing:
```bash
# Get a 

# Run preprocessing
stamp --config configs/config_OAUTHC.yaml preprocess
stamp --config configs/config_LUTH.yaml preprocess
stamp --config configs/config_UITH.yaml preprocess
stamp --config configs/config_LASUTH.yaml preprocess
stamp --config configs/config_retrospective_MSK.yaml preprocess
stamp --config configs/config_retrospective_OAU.yaml preprocess
```

### Step 4: Validation
```bash
# Switch to ARGO environment
conda activate argo

# Run validation
python src/validate_processing.py
```

## Project Structure
```
# Project Structure
```
argo-deepmsi/
├── src/
│   ├── collect_redcap.py      # REDCap data collection
│   └── validate_processing.py  # Processing validation
├── data/                      # All slide data (gitignored)
│   ├── OAUTHC/
│   │   ├── raw/              # Original SVS files
│   │   ├── processed/        # STAMP output
│   │   └── cache/           # STAMP cache
│   ├── LUTH/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── cache/
│   ├── UITH/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── cache/
│   ├── LASUTH/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── cache/
│   ├── retrospective_msk/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── cache/
│   └── retrospective_oau/
│       ├── raw/
│       ├── processed/
│       └── cache/
├── tables/                    # Generated CSV tables
│   ├── prospective_clinical_table.csv
│   ├── prospective_slide_table.csv
│   ├── retrospective_msk_clinical_table.csv
│   ├── retrospective_msk_slide_table.csv
│   ├── retrospective_oau_clinical_table.csv
│   └── retrospective_oau_slide_table.csv
├── configs/                   # STAMP configuration files
│   ├── config_LUTH.yaml
│   ├── config_OAUTHC.yaml
│   ├── config_UITH.yaml
│   ├── config_LASUTH.yaml
│   ├── config_retrospective_MSK.yaml
│   └── config_retrospective_OAU.yaml
├── .env                      # Environment variables 
├── .gitignore                # Git ignore file
├── argo-env.yml              # Conda environment specification
└── README.md                 # Project documentation
```

## Notes
- ARGO environment (conda) is used for data collection and validation
- STAMP environment (venv) is used only for STAMP processing
- STAMP configurations are institution-specific and stored in `configs/`
- AWS credentials should be configured system-wide using `aws configure`
- All generated tables are stored in `tables/` directory