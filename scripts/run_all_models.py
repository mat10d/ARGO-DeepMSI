#!/usr/bin/env python3
"""
Run feature extraction across all available models.

This script can:
1. Run all models sequentially on a single GPU
2. Submit SLURM array jobs for parallel processing
3. Run specific model subsets

Usage:
    # Run all models sequentially (interactive)
    python scripts/run_all_models.py results/data/slide_table.csv

    # Run only non-gated models
    python scripts/run_all_models.py results/data/slide_table.csv --no-auth-only

    # Submit as SLURM array job
    python scripts/run_all_models.py results/data/slide_table.csv --slurm

    # Specific models only
    python scripts/run_all_models.py results/data/slide_table.csv --models uni2 virchow2 h-optimus-0
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from argo_deepmsi.feature_extraction import (
    PATCH_MODELS,
    SLIDE_MODELS,
    list_available_models,
    extract_features_batch,
)
from argo_deepmsi.io_utils import setup_logging, get_features_dir, ensure_dir


# Model groups for convenience
MODEL_GROUPS = {
    "no_auth": ["resnet50", "ctranspath", "plip"],
    "gated": ["uni", "uni2", "virchow", "virchow2", "conch", "gigapath", "h-optimus-0", "h-optimus-1"],
    "recommended": ["uni2", "virchow2", "h-optimus-0", "gigapath", "ctranspath"],
    "all_patch": list(PATCH_MODELS.keys()),
    "slide": list(SLIDE_MODELS.keys()),
}


def get_models_to_run(
    models: Optional[List[str]] = None,
    no_auth_only: bool = False,
    group: Optional[str] = None,
) -> List[str]:
    """Determine which models to run based on arguments."""
    if models:
        return models
    if no_auth_only:
        return MODEL_GROUPS["no_auth"]
    if group and group in MODEL_GROUPS:
        return MODEL_GROUPS[group]
    return MODEL_GROUPS["all_patch"]


def run_sequential(
    slide_table: Path,
    models: List[str],
    device: str = "cuda",
    max_slides: Optional[int] = None,
    overwrite: bool = False,
):
    """Run extraction for all models sequentially."""
    import pandas as pd

    logger = setup_logging("extract_all")
    df = pd.read_csv(slide_table)

    logger.info(f"Running extraction for {len(models)} models on {len(df)} slides")

    results = {}
    for i, model in enumerate(models, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(models)}] Model: {model}")
        logger.info(f"{'='*60}")

        try:
            output_dir = get_features_dir(model)
            ensure_dir(output_dir)

            model_results = extract_features_batch(
                slide_table=df,
                model=model,
                output_dir=output_dir,
                device=device,
                amp=True,
                overwrite=overwrite,
                max_slides=max_slides,
            )

            success_rate = model_results["success"].mean() * 100
            results[model] = {
                "success": model_results["success"].sum(),
                "total": len(model_results),
                "rate": success_rate,
            }
            logger.info(f"Completed {model}: {success_rate:.1f}% success rate")

        except Exception as e:
            logger.error(f"Failed {model}: {e}")
            results[model] = {"error": str(e)}

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for model, result in results.items():
        if "error" in result:
            logger.info(f"  {model}: FAILED - {result['error']}")
        else:
            logger.info(f"  {model}: {result['success']}/{result['total']} ({result['rate']:.1f}%)")

    return results


def submit_slurm(
    slide_table: Path,
    models: List[str],
    partition: str = "gpu",
    time: str = "24:00:00",
    memory: str = "64G",
    max_concurrent: int = 4,
):
    """Submit SLURM array job for parallel extraction."""
    n_models = len(models)

    # Create SLURM script
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name=argo-extract
#SBATCH --output=logs/slurm/extract_%A_%a.out
#SBATCH --error=logs/slurm/extract_%A_%a.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --array=0-{n_models-1}%{max_concurrent}

set -e

MODELS=({' '.join(f'"{m}"' for m in models)})
MODEL=${{MODELS[$SLURM_ARRAY_TASK_ID]}}

echo "Running model: $MODEL"
echo "Slide table: {slide_table}"

cd {PROJECT_ROOT}
source ~/.bashrc
conda activate argo

export HF_HOME="{PROJECT_ROOT}/.huggingface_cache"

argo extract "{slide_table}" --model "$MODEL" --device cuda --amp
'''

    # Write script
    script_path = PROJECT_ROOT / "logs" / "slurm" / "submit_extract.sh"
    ensure_dir(script_path.parent)
    script_path.write_text(slurm_script)

    print(f"Generated SLURM script: {script_path}")
    print(f"Models to run ({n_models}): {', '.join(models)}")
    print(f"Max concurrent jobs: {max_concurrent}")
    print()

    # Submit
    response = input("Submit job? [y/N]: ").strip().lower()
    if response == "y":
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
    else:
        print(f"Script saved to {script_path} - submit manually with: sbatch {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run feature extraction across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Groups:
  --group no_auth      Only models that don't require HF auth
  --group gated        Only gated HF models
  --group recommended  Best performing models (uni2, virchow2, h-optimus-0, gigapath, ctranspath)
  --group all_patch    All patch-level extractors (default)
  --group slide        Slide-level aggregators (prism, threads)

Examples:
  # Run recommended models sequentially
  python scripts/run_all_models.py slide_table.csv --group recommended

  # Submit SLURM array job for all models
  python scripts/run_all_models.py slide_table.csv --slurm

  # Test with 5 slides only
  python scripts/run_all_models.py slide_table.csv --max-slides 5 --models uni2 virchow2
        """
    )

    parser.add_argument("slide_table", type=Path, help="Path to slide table CSV")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--group", choices=list(MODEL_GROUPS.keys()), help="Predefined model group")
    parser.add_argument("--no-auth-only", action="store_true", help="Only run models without HF auth")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-slides", type=int, help="Max slides to process (for testing)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM array job")
    parser.add_argument("--partition", default="gpu", help="SLURM partition")
    parser.add_argument("--time", default="24:00:00", help="SLURM time limit")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent SLURM jobs")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("Available Models:\n")
        print("Patch-Level Extractors:")
        for name, config in PATCH_MODELS.items():
            auth = "(HF auth)" if config.requires_auth else ""
            print(f"  {name:15} {auth:12} {config.description}")
        print("\nSlide-Level Aggregators:")
        for name, config in SLIDE_MODELS.items():
            print(f"  {name:15} {'(HF auth)':12} {config.description}")
        print("\nModel Groups:")
        for group, models in MODEL_GROUPS.items():
            print(f"  {group:15} {', '.join(models)}")
        return

    # Determine models
    models = get_models_to_run(args.models, args.no_auth_only, args.group)

    print(f"Models to run ({len(models)}): {', '.join(models)}")
    print(f"Slide table: {args.slide_table}")
    print()

    if args.slurm:
        submit_slurm(
            args.slide_table,
            models,
            partition=args.partition,
            time=args.time,
            max_concurrent=args.max_concurrent,
        )
    else:
        run_sequential(
            args.slide_table,
            models,
            device=args.device,
            max_slides=args.max_slides,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
