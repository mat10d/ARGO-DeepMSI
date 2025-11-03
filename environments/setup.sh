#!/bin/bash
# ARGO-DeepMSI Environment Setup Script
# This script sets up all three environments needed for the pipeline

set -e  # Exit on error

echo "=========================================="
echo "ARGO-DeepMSI Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the root directory (parent of environments/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Root directory: $ROOT_DIR"
echo ""

# Function to check if conda environment exists
check_conda_env() {
    conda env list | grep -q "^$1 "
}

# Function to check if directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        echo -e "${RED}Error: Directory $1 not found${NC}"
        return 1
    fi
    return 0
}

echo "=========================================="
echo "Step 1: Setting up ARGO environment"
echo "=========================================="

if check_conda_env "argo"; then
    echo -e "${YELLOW}ARGO environment already exists. Updating...${NC}"
    conda env update -f "$ROOT_DIR/environments/argo.yml" --prune
else
    echo "Creating ARGO environment..."
    conda env create -f "$ROOT_DIR/environments/argo.yml"
fi

echo -e "${GREEN}✓ ARGO environment ready${NC}"
echo ""

echo "=========================================="
echo "Step 2: Setting up STAMP environment"
echo "=========================================="

if ! check_dir "$ROOT_DIR/STAMP"; then
    echo -e "${RED}Error: STAMP directory not found. Please clone STAMP first.${NC}"
    exit 1
fi

cd "$ROOT_DIR/STAMP"

# Clear triton cache (important for STAMP)
if [ -d "$HOME/.triton" ]; then
    echo "Clearing triton cache..."
    rm -rf "$HOME/.triton"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Installing STAMP dependencies with uv..."
uv sync --extra build --extra gpu

echo -e "${GREEN}✓ STAMP environment ready${NC}"
cd "$ROOT_DIR"
echo ""

echo "=========================================="
echo "Step 3: Setting up HistoBistro environment"
echo "=========================================="

if ! check_dir "$ROOT_DIR/HistoBistro"; then
    echo -e "${RED}Error: HistoBistro directory not found. Please clone HistoBistro first.${NC}"
    exit 1
fi

if check_conda_env "histobistro"; then
    echo -e "${YELLOW}HistoBistro environment already exists. Updating...${NC}"
    conda env update -f "$ROOT_DIR/HistoBistro/environment_simple.yaml" --prune
else
    echo "Creating HistoBistro environment..."
    conda env create -f "$ROOT_DIR/HistoBistro/environment_simple.yaml"
fi

echo -e "${GREEN}✓ HistoBistro environment ready${NC}"
echo ""

echo "=========================================="
echo "Step 4: Setting up environment variables"
echo "=========================================="

# Create .env file if it doesn't exist
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "Creating .env file..."
    cat > "$ROOT_DIR/.env" << EOF
# Hugging Face cache directory
export HF_HOME=$ROOT_DIR/.huggingface_cache

# REDCap API credentials (update these)
export REDCAP_API_URL=https://redcap.oauife.edu.ng/api/
export REDCAP_API_TOKEN=your_token_here
EOF
    echo -e "${YELLOW}⚠ Please update .env with your REDCap API token${NC}"
else
    echo ".env file already exists"
fi

# Create HF cache directory
mkdir -p "$ROOT_DIR/.huggingface_cache"

echo -e "${GREEN}✓ Environment variables configured${NC}"
echo ""

echo "=========================================="
echo "Step 5: Verifying installations"
echo "=========================================="

echo "Testing ARGO environment..."
conda run -n argo python -c "import pandas, numpy, sklearn, matplotlib, seaborn, requests; print('  ✓ ARGO packages OK')" || echo -e "${RED}  ✗ ARGO test failed${NC}"

echo "Testing STAMP environment..."
cd "$ROOT_DIR/STAMP"
source .venv/bin/activate
python -c "import torch, transformers, timm, lightning; print('  ✓ STAMP packages OK'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" || echo -e "${RED}  ✗ STAMP test failed${NC}"
deactivate
cd "$ROOT_DIR"

echo "Testing HistoBistro environment..."
conda run -n histobistro python -c "import torch, pytorch_lightning; print('  ✓ HistoBistro packages OK')" || echo -e "${RED}  ✗ HistoBistro test failed${NC}"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment activation commands:"
echo "  ARGO:        conda activate argo"
echo "  STAMP:       source STAMP/.venv/bin/activate"
echo "  HistoBistro: conda activate histobistro"
echo ""
echo "Next steps:"
echo "  1. Update .env with your REDCap API token"
echo "  2. Login to Hugging Face: hf auth login"
echo "  3. Request access to gated models (UNI, Virchow, etc.)"
echo "  4. Run: python scripts/1_data_ingestion.py"
echo ""
