# ARGO-deepmsi Workflow

## Environment Setup

### 1. ARGO Environment
```bash
# Create and activate environment for data collection/processing
conda env create -f argo-env.yml
conda activate argo
```

### 2. STAMP Environment
```bash
# Create and activate environment for STAMP
python -m venv .venv-stamp
source .venv-stamp/bin/activate
pip install "stamp[all] @ git+https://github.com/KatherLab/STAMP"
```

## Workflow Steps

### Step 1: REDCap Data Collection
1. Create `.env` file with REDCap credentials:
```
REDCAP_API_TOKEN=your_token_here
REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
```

2. Run REDCap data collection script:
```bash
conda activate argo
python src/collect_redcap.py
```

### Step 2: AWS Slide Download
```bash
# Configure AWS CLI if not already done
aws configure

# Create data directories
mkdir -p data/{OAUTHC,LUTH,UITH,LASUTH}/{raw,processed,cache}

# Download slides for each institution
aws s3 cp s3://bucket-name/slides/oauthc/ data/OAUTHC/raw/
aws s3 cp s3://bucket-name/slides/luth/ data/LUTH/raw/
aws s3 cp s3://bucket-name/slides/uith/ data/UITH/raw/
aws s3 cp s3://bucket-name/slides/lasuth/ data/LASUTH/raw/
```

### Step 3: STAMP Processing

1. Initialize STAMP configs:
```bash
# Activate STAMP environment
source .venv-stamp/bin/activate

# Initialize configs
stamp init --config configs/config_OAUTHC.yaml
stamp init --config configs/config_LUTH.yaml
stamp init --config configs/config_UITH.yaml
```

2. Run preprocessing:
```bash
# Run for each institution
stamp preprocess --config configs/config_OAUTHC.yaml
stamp preprocess --config configs/config_LUTH.yaml
stamp preprocess --config configs/config_UITH.yaml
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
ARGO-DeepMSI/
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
│   └── LASUTH/
│       ├── raw/
│       ├── processed/
│       └── cache/
├── tables/                    # Generated CSV tables
├── configs/                   # STAMP configuration files
│   ├── config_OAUTHC.yaml
│   ├── config_LUTH.yaml
│   ├── config_UITH.yaml
│   └── config_LASUTH.yaml
├── argo-env.yml              # Conda environment specification
└── .env                      # Environment variables
```

## Notes
- ARGO environment (conda) is used for data collection and validation
- STAMP environment (venv) is used only for STAMP processing
- STAMP configurations are institution-specific and stored in `configs/`
- AWS credentials should be configured system-wide using `aws configure`
- All generated tables are stored in `tables/` directory