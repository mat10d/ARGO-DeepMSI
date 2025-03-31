#!/bin/bash
#SBATCH --job-name=stamp_msi
#SBATCH --output=stamp_msi_%A_%a.out
#SBATCH --error=stamp_msi_%A_%a.err
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Define sites array
sites=("OAUTHC" "LUTH" "LASUTH" "UITH" "retrospective_msk" "retrospective_oau")

# Get the current site based on array index
SITE=${sites[$SLURM_ARRAY_TASK_ID]}
echo "Processing site: $SITE"

# Load modules/environment (modify as needed for your system)
module load python/3.8
source /path/to/your/stamp-env/bin/activate

# Define base directory and config path
BASE_DIR="/lab/barcheese01/mdiberna/ARGO-DeepMSI"
CONFIG_FILE="$BASE_DIR/data/$SITE/config.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file for $SITE not found at $CONFIG_FILE"
    exit 1
fi

# Define which step to run (uncomment as needed)
# Comment out steps you've already completed or don't want to run

Step 1: Feature extraction
echo "Running preprocessing for $SITE..."
stamp --config "$CONFIG_FILE" preprocess

# # Step 2: Cross-validation
# echo "Running cross-validation for $SITE..."
# stamp --config "$CONFIG_FILE" crossval

# # Step 3: Train final model
# echo "Training final model for $SITE..."
# stamp --config "$CONFIG_FILE" train

# # Step 4: Generate statistics
# echo "Generating statistics for $SITE..."
# stamp --config "$CONFIG_FILE" statistics

# # Step 5: Generate heatmaps
# echo "Generating heatmaps for $SITE..."
# stamp --config "$CONFIG_FILE" heatmap

# echo "All processing completed for $SITE"