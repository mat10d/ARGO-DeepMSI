#!/bin/bash
#SBATCH --job-name=stamp_crossval_hoptimus0
#SBATCH --partition=nvidia-A100-20          
#SBATCH --output=out/stamp_crossval_hoptimus0_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G          
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

echo "=== H-optimus-0 Cross-Validation ==="

# Set environment variables
export HF_HOME="/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH

# Create cache directories
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Define target directory where all H-optimus-0 features will be consolidated
TARGET_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI/data/all/features/h-optimus-0/"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Consolidating H-optimus-0 features..."

# Copy all H-optimus-0 feature data from individual sites to the consolidated directory
# Need to find the actual hash subdirectories
sites=("OAUTHC" "LUTH" "LASUTH" "UITH" "retrospective_msk" "retrospective_oau")

for site in "${sites[@]}"; do
    SITE_FEATURES_DIR="../data/$site/features"
    
    if [ -d "$SITE_FEATURES_DIR" ]; then
        echo "Checking site: $site"
        
        # Look for h-optimus-0 directories
        for feature_dir in "$SITE_FEATURES_DIR"/h-optimus-0*/; do
            if [ -d "$feature_dir" ]; then
                echo "Found H-optimus-0 directory: $feature_dir"
                
                # Check if this directory has H5 files directly
                if ls "$feature_dir"*.h5 1> /dev/null 2>&1; then
                    echo "  H5 files found directly in: $feature_dir"
                    echo "  Copying files from $feature_dir to $TARGET_DIR"
                    rsync -av "$feature_dir"*.h5 "$TARGET_DIR"
                else
                    # Look for subdirectories (hash directories)
                    for subdir in "$feature_dir"*/; do
                        if [ -d "$subdir" ] && ls "$subdir"*.h5 1> /dev/null 2>&1; then
                            echo "  H5 files found in subdirectory: $subdir"
                            echo "  Copying files from $subdir to $TARGET_DIR"
                            rsync -av "$subdir"*.h5 "$TARGET_DIR"
                            break
                        fi
                    done
                fi
            fi
        done
    else
        echo "Warning: $SITE_FEATURES_DIR not found"
    fi
done

# Count consolidated files
file_count=$(ls -1 "$TARGET_DIR"*.h5 2>/dev/null | wc -l)
echo "Consolidated $file_count H5 files to $TARGET_DIR"

if [ $file_count -eq 0 ]; then
    echo "Error: No H5 files found to consolidate!"
    echo "Please check that H-optimus-0 preprocessing completed successfully"
    exit 1
fi

# Activate STAMP environment
echo "Activating STAMP environment..."
source /lab/barcheese01/mdiberna/ARGO-DeepMSI/STAMP/.venv/bin/activate


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

# Check Hugging Face authentication
echo "==== HUGGING FACE CHECK ===="
python -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    user = api.whoami()
    print(f'✓ Logged in as: {user[\"name\"]}')
except Exception as e:
    print(f'⚠ HF authentication issue: {e}')
"
echo "=========================="

# Define base directory and config path
BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"
CONFIG_FILE="$BASE_DIR/configs/h-optimus-0/config_all.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    echo "Expected config structure:"
    echo "  $BASE_DIR/configs/h-optimus-0/config_all.yaml"
    echo ""
    echo "Please ensure you have:"
    echo "1. Created H-optimus-0 specific configs"
    echo "2. Updated table paths in the config to use ../tables/h_optimus_0/"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"

# Change to project directory
cd "$BASE_DIR"

# Run cross-validation
echo "==== STARTING CROSS-VALIDATION ===="
echo "Extractor: h-optimus-0"
echo "Config: $CONFIG_FILE"
echo "Feature directory: $TARGET_DIR"
echo "Time: $(date)"
echo "====================================="

stamp --config "$CONFIG_FILE" crossval

crossval_exit_code=$?

echo "====================================="
echo "CROSS-VALIDATION COMPLETED"
echo "Extractor: h-optimus-0"
echo "Exit code: $crossval_exit_code"
echo "Time: $(date)"
echo "====================================="

if [ $crossval_exit_code -eq 0 ]; then
    echo "✓ Cross-validation completed successfully for h-optimus-0"
    echo "Results saved to: $BASE_DIR/data/all/results/crossval_h_optimus_0/"
    
    # Run statistics
    echo ""
    echo "==== STARTING STATISTICS GENERATION ===="
    echo "Time: $(date)"
    echo "========================================"
    
    stamp --config "$CONFIG_FILE" statistics
    
    stats_exit_code=$?
    
    echo "========================================"
    echo "STATISTICS GENERATION COMPLETED"
    echo "Exit code: $stats_exit_code"
    echo "Time: $(date)"
    echo "========================================"
    
    if [ $stats_exit_code -eq 0 ]; then
        echo "✓ Statistics generated successfully for h-optimus-0"
        echo "Statistics saved to: $BASE_DIR/data/all/results/statistics_h_optimus_0/"
        
        # Run heatmaps (optional - only if we have a trained model)
        echo ""
        echo "==== STARTING HEATMAP GENERATION ===="
        echo "Time: $(date)"
        echo "==================================="
        
        # Check if we have a checkpoint for heatmaps
        checkpoint_path="$BASE_DIR/data/all/results/training_h_optimus_0/model.ckpt"
        if [ -f "$checkpoint_path" ]; then
            echo "Found checkpoint at: $checkpoint_path"
            stamp --config "$CONFIG_FILE" heatmaps
            
            heatmaps_exit_code=$?
            
            echo "==================================="
            echo "HEATMAP GENERATION COMPLETED"
            echo "Exit code: $heatmaps_exit_code"
            echo "Time: $(date)"
            echo "==================================="
            
            if [ $heatmaps_exit_code -eq 0 ]; then
                echo "✓ Heatmaps generated successfully for h-optimus-0"
                echo "Heatmaps saved to: $BASE_DIR/data/all/results/heatmaps_h_optimus_0/"
                final_exit_code=0
            else
                echo "✗ Heatmap generation failed (exit code: $heatmaps_exit_code)"
                final_exit_code=$heatmaps_exit_code
            fi
        else
            echo "⚠ No trained model checkpoint found at: $checkpoint_path"
            echo "Skipping heatmap generation. To generate heatmaps:"
            echo "1. First run: stamp --config $CONFIG_FILE train"
            echo "2. Then run: stamp --config $CONFIG_FILE heatmaps"
            final_exit_code=0
        fi
        
        # Summary
        echo ""
        echo "========================================="
        echo "H-OPTIMUS-0 PIPELINE SUMMARY"
        echo "========================================="
        echo "Cross-validation: $([ $crossval_exit_code -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
        echo "Statistics:       $([ $stats_exit_code -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
        if [ -f "$checkpoint_path" ]; then
            echo "Heatmaps:         $([ $heatmaps_exit_code -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
        else
            echo "Heatmaps:         ⚠ SKIPPED (no model checkpoint)"
        fi
        echo "========================================="
        echo "Overall:          $([ $final_exit_code -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED")"
        echo "Time completed:   $(date)"
        echo "========================================="
        
    else
        echo "✗ Statistics generation failed for h-optimus-0 (exit code: $stats_exit_code)"
        final_exit_code=$stats_exit_code
    fi
    
else
    echo "✗ Cross-validation failed for h-optimus-0 (exit code: $crossval_exit_code)"
    echo "Skipping statistics and heatmaps due to cross-validation failure"
    final_exit_code=$crossval_exit_code
fi

exit $final_exit_code