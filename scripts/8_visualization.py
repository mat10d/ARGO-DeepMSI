#!/usr/bin/env python3
"""
Stage 8: Visualization

Generates visualizations for pipeline stages:
- Stage 1: Data ingestion plots (patient/slide counts by site)
- Stage 4: Feature validation plots (processing status)
- Stage 8: Results visualization (ROC curves, heatmaps, embeddings)

Usage:
    # For data ingestion visualizations
    python scripts/8_visualization.py stage1 \
        --clinical tables/0/clinical_table.csv \
        --slides tables/0/slide_table.csv \
        --output-dir results/figures/stage1

    # For feature validation visualizations
    python scripts/8_visualization.py stage4 \
        --clinical tables/0/clinical_table.csv \
        --slides tables/2/all_slide_table.csv \
        --output-dir results/figures/stage4

Environment: ARGO (conda activate argo)
"""

import argparse
import pandas as pd
from pathlib import Path
from argo_deepmsi import io_utils, visualization, feature_validation


def visualize_stage1(args, logger):
    """Generate Stage 1 (Data Ingestion) visualizations."""
    # Validate input files
    clinical_path = io_utils.validate_file_exists(args.clinical, "Clinical table")
    slide_path = io_utils.validate_file_exists(args.slides, "Slide table")

    # Load tables
    logger.info(f"Loading clinical table from {clinical_path}")
    clinical_table = pd.read_csv(clinical_path)

    logger.info(f"Loading slide table from {slide_path}")
    slide_table = pd.read_csv(slide_path)

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else io_utils.get_results_dir("figures/stage1")

    # Generate visualizations
    visualization.generate_data_ingestion_visualizations(
        clinical_table=clinical_table,
        slide_table=slide_table,
        output_dir=output_dir
    )

    logger.info(f"✓ Stage 1 visualizations saved to {output_dir}")


def visualize_stage4(args, logger):
    """Generate Stage 4 (Feature Validation) visualizations."""
    # Validate input files
    clinical_path = io_utils.validate_file_exists(args.clinical, "Clinical table")
    slide_path = io_utils.validate_file_exists(args.slides, "Slide table")

    # Load tables
    logger.info(f"Loading clinical table from {clinical_path}")
    clinical_table = pd.read_csv(clinical_path)

    logger.info(f"Loading slide table from {slide_path}")
    slide_table = pd.read_csv(slide_path)

    # Check processing status (need to regenerate merged_data with 'processed' column)
    logger.info("Checking processing status...")
    merged_data = feature_validation.check_processing_status(
        clinical_table=clinical_table,
        slide_table=slide_table,
        features_base_dir=args.features_dir,
        extractor_name=args.extractor
    )

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else io_utils.get_results_dir("figures/stage4")

    # Generate visualizations
    visualization.generate_feature_validation_visualizations(
        merged_data=merged_data,
        output_dir=output_dir
    )

    logger.info(f"✓ Stage 4 visualizations saved to {output_dir}")


def main():
    """Run visualization generation."""
    parser = argparse.ArgumentParser(
        description="Stage 8: Visualization - Generate plots for pipeline stages"
    )

    # Subcommands for different stages
    subparsers = parser.add_subparsers(dest="stage", required=True,
                                      help="Pipeline stage to visualize")

    # Stage 1: Data Ingestion
    parser_stage1 = subparsers.add_parser("stage1", help="Data ingestion visualizations")
    parser_stage1.add_argument("--clinical", type=str, required=True,
                              help="Path to clinical table CSV")
    parser_stage1.add_argument("--slides", type=str, required=True,
                              help="Path to slide table CSV")
    parser_stage1.add_argument("--output-dir", type=str, default=None,
                              help="Output directory (default: results/figures/stage1/)")

    # Stage 4: Feature Validation
    parser_stage4 = subparsers.add_parser("stage4", help="Feature validation visualizations")
    parser_stage4.add_argument("--clinical", type=str, required=True,
                              help="Path to clinical table CSV")
    parser_stage4.add_argument("--slides", type=str, required=True,
                              help="Path to slide table CSV (pre-validation)")
    parser_stage4.add_argument("--extractor", type=str, default="ctranspath",
                              help="Extractor name (default: ctranspath)")
    parser_stage4.add_argument("--features-dir", type=str, default=None,
                              help="Base directory for features (default: data/)")
    parser_stage4.add_argument("--output-dir", type=str, default=None,
                              help="Output directory (default: results/figures/stage4/)")

    args = parser.parse_args()

    # Setup logging
    logger = io_utils.setup_logging(
        name=f"visualization_{args.stage}",
        log_dir=io_utils.get_project_root() / "logs" / "visualization"
    )

    logger.info("=" * 80)
    logger.info(f"ARGO-DeepMSI: Stage 8 - Visualization ({args.stage.upper()})")
    logger.info("=" * 80)

    try:
        if args.stage == "stage1":
            visualize_stage1(args, logger)
        elif args.stage == "stage4":
            visualize_stage4(args, logger)
        else:
            raise ValueError(f"Unknown stage: {args.stage}")

        logger.info("=" * 80)
        logger.info("Visualization generation completed successfully")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
