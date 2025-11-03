#!/usr/bin/env python3
"""
Stage 7: Statistics

Calculates performance metrics (AUROC, AUPRC, sensitivity, specificity, NPV)
with 95% confidence intervals from cross-validation predictions.

NOTE: This script wraps STAMP's statistics functionality.

Usage:
    python scripts/7_statistics.py --model ctranspath \
                                   [--config configs/ctranspath/config_all.yaml]

Environment: STAMP (source STAMP/.venv/bin/activate)
"""

import argparse
from pathlib import Path
from argo_deepmsi import io_utils


def main():
    """Run statistics calculation via STAMP."""
    parser = argparse.ArgumentParser(
        description="Stage 7: Statistics - Calculate performance metrics from cross-validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., ctranspath, virchow2, h-optimus-0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to STAMP config file (default: configs/{model}/config_all.yaml)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = io_utils.setup_logging(
        name="statistics",
        log_dir=io_utils.get_project_root() / "logs" / "statistics"
    )

    logger.info("=" * 80)
    logger.info("ARGO-DeepMSI: Stage 7 - Statistics")
    logger.info("=" * 80)

    try:
        # Determine config path
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = io_utils.get_configs_dir(args.model) / "config_all.yaml"

        # Validate config exists
        config_path = io_utils.validate_file_exists(config_path, "STAMP config")

        logger.info(f"Using STAMP config: {config_path}")
        logger.info("This script is a wrapper around STAMP's statistics functionality")
        logger.info("")
        logger.info("STAMP will:")
        logger.info("1. Load cross-validation predictions from all splits")
        logger.info("2. Calculate AUROC, AUPRC with 95% CI")
        logger.info("3. Calculate sensitivity, specificity, NPV at optimal threshold")
        logger.info("4. Generate ROC and PR curves")
        logger.info("5. Save results to statistics output directory")
        logger.info("")

        # Instructions for running STAMP
        logger.info("=" * 80)
        logger.info("To run STAMP statistics:")
        logger.info("")
        logger.info("Option 1: Use this config directly with STAMP CLI")
        logger.info(f"  stamp --config {config_path} statistics")
        logger.info("")
        logger.info("Option 2: Submit SLURM job")
        logger.info(f"  sbatch scripts/slurm/7_statistics.sh {args.model}")
        logger.info("")
        logger.info("=" * 80)

        # TODO: Could directly invoke STAMP Python API here instead of requiring CLI
        # from stamp.statistics import calculate_metrics
        # results = calculate_metrics(config_path)

    except Exception as e:
        logger.error(f"Statistics preparation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
