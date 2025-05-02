#!/bin/bash
#SBATCH --job-name=stamp_statistics
#SBATCH --partition=20       
#SBATCH --output=out/stamp_statistics_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G          
#SBATCH --time=1:00:00

# Load conda environment
source ~/.bashrc
conda activate stamp-env

# Switch to the directory where the script is located
cd ../HistoBistro

# Print GPU information
echo "==== GPU INFO ===="
nvidia-smi
echo "================="

# Run a quick Python script to check PyTorch GPU access
echo "==== PYTORCH GPU CHECK ===="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA current device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(0))
"
echo "=========================="

python test.py --config config.yaml

