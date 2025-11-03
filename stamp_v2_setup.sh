#!/bin/bash

# Simple STAMP setup for HPC (just redirecting HF cache)
echo "=== Simple STAMP Setup for HPC ==="

# Starting from your ARGO-DeepMSI directory
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI

# Only set Hugging Face cache to your lab space (where the large model files go)
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
mkdir -p "$HF_HOME"

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6

# Add nvcc to PATH
export PATH=$CUDA_HOME/bin:$PATH

# Verify it works
nvcc --version

echo "Using Hugging Face cache directory: $HF_HOME"

# Check if system dependencies are available
echo "Checking system dependencies..."

# Check for OpenSlide
if command -v openslide-write-png &> /dev/null; then
    echo "✓ OpenSlide found"
elif [ -f "/usr/lib/x86_64-linux-gnu/libopenslide.so" ] || [ -f "/usr/local/lib/libopenslide.so" ]; then
    echo "✓ OpenSlide library found"
elif module avail openslide 2>&1 | grep -q openslide; then
    echo "✓ OpenSlide available via module system"
    echo "Loading OpenSlide module..."
    module load openslide
else
    echo "⚠ OpenSlide not found - you may need to load it via module system"
    echo "  Try: module load openslide"
fi

# Remove old STAMP if it exists
if [ -d "STAMP" ]; then
    echo "Removing existing STAMP directory..."
    rm -rf STAMP
    fi

# Clone STAMP repository
echo "Cloning STAMP repository..."
git clone https://github.com/KatherLab/STAMP.git STAMP

# Enter directory
cd STAMP/

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv (user installation)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install STAMP with all extras (using default uv cache since you cleared it)
echo "Installing STAMP (this may take 10-20 minutes)..."
uv sync --all-extras

if [ $? -eq 0 ]; then
    echo "✓ STAMP installation successful!"
else
    echo "✗ STAMP installation failed"
    exit 1
fi

# Test the installation
echo "Testing STAMP installation..."
source .venv/bin/activate

# Test basic functionality
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import stamp
    print('✓ STAMP imported successfully')
except ImportError as e:
    print(f'✗ STAMP import failed: {e}')
    exit(1)

try:
    import torch
    print(f'✓ PyTorch {torch.__version__} available')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('✗ PyTorch not available')

try:
    from stamp.preprocessing.feature_extractors import AVAILABLE_FEATURE_EXTRACTORS
    print(f'✓ Found {len(AVAILABLE_FEATURE_EXTRACTORS)} available extractors')
    
    # Check specifically for H-optimus models
    h_optimus_models = [ext for ext in AVAILABLE_FEATURE_EXTRACTORS if 'optimus' in ext.lower()]
    if h_optimus_models:
        print(f'✓ H-optimus models found: {h_optimus_models}')
    else:
        print('⚠ No H-optimus models found')
        
    # List all extractors
    print('All available extractors:')
    for ext in sorted(AVAILABLE_FEATURE_EXTRACTORS):
        print(f'  - {ext}')
        
except Exception as e:
    print(f'⚠ Error checking extractors: {e}')
"

# Test STAMP CLI
echo "Testing STAMP CLI..."
stamp --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ STAMP CLI working"
else
    echo "✗ STAMP CLI not working"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Hugging Face cache will be stored in: $HF_HOME"
echo ""
echo "Next steps:"
echo "1. cd /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP"
echo "2. source .venv/bin/activate"
echo "3. export HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
echo "4. hf auth login  # Required for H-optimus models"
echo ""
echo "To activate STAMP from anywhere:"
echo "source /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP/.venv/bin/activate"