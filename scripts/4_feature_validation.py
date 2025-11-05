#!/usr/bin/env python3
"""
Stage 4: Feature Validation

Validates feature extraction results, identifies which slides passed/failed extraction,
and prepares tables for MIL training.

Usage:
    # Validate ALL models (default)
    python scripts/4_feature_validation.py \
        --clinical results/stage1_data_ingestion/clinical_table.csv \
        --slides results/stage1_data_ingestion/slide_table.csv

    # Validate ONE specific model
    python scripts/4_feature_validation.py \
        --clinical results/stage1_data_ingestion/clinical_table.csv \
        --slides results/stage1_data_ingestion/slide_table.csv \
        --extractor ctranspath

Environment: ARGO (conda activate argo)

Outputs (per model):
    results/stage4_feature_validation/{model}/tables/all_clinical_table.csv
    results/stage4_feature_validation/{model}/tables/all_slide_table.csv
    results/stage4_feature_validation/{model}/reports/missing_slides.csv
    results/stage4_feature_validation/{model}/reports/extraction_summary_by_site.csv

    For ctranspath specifically, also generates:
    results/stage4_feature_validation/ctranspath/tables/all_slide_table_histobistro.csv
"""

import argparse
import pandas as pd
from argo_deepmsi import io_utils, feature_validation


def main():
    """Run feature validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Feature Validation - Check extraction results and prepare tables"
    )
    parser.add_argument(
        "--clinical",
        type=str,
        required=True,
        help="Path to clinical table CSV"
    )
    parser.add_argument(
        "--slides",
        type=str,
        required=True,
        help="Path to slide table CSV"
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        help="Extractor name to validate (default: all models in results/stage3_features/)"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Base directory for features (default: results/stage3_features/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed tables (default: results/stage4_feature_validation/tables/)"
    )
    parser.add_argument(
        "--no-split-by-site",
        action="store_true",
        help="Don't save site-specific tables (only save consolidated 'all' table)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = io_utils.setup_logging(
        name="feature_validation",
        log_dir=io_utils.get_project_root() / "logs" / "feature_validation"
    )

    logger.info("=" * 80)
    logger.info("ARGO-DeepMSI: Stage 4 - Feature Validation")
    logger.info("=" * 80)

    try:
        # Validate input files exist
        clinical_path = io_utils.validate_file_exists(args.clinical, "Clinical table")
        slide_path = io_utils.validate_file_exists(args.slides, "Slide table")

        # Load tables
        logger.info(f"Loading clinical table from {clinical_path}")
        clinical_table = pd.read_csv(clinical_path)
        logger.info(f"Loaded {len(clinical_table)} patients")

        logger.info(f"Loading slide table from {slide_path}")
        slide_table = pd.read_csv(slide_path)
        logger.info(f"Loaded {len(slide_table)} slides")

        # Determine which models to process
        if args.extractor:
            # Single model specified
            extractors_to_process = [args.extractor]
            logger.info(f"Processing single extractor: {args.extractor}")
        else:
            # Default: process all models found in stage3_features/
            from pathlib import Path
            features_base_dir = Path(args.features_dir) if args.features_dir else io_utils.get_stage_dir(3)
            if features_base_dir.exists():
                extractors_to_process = [d.name for d in features_base_dir.iterdir() if d.is_dir()]
                logger.info(f"No --extractor specified, processing all {len(extractors_to_process)} models")
                logger.info(f"Models found: {', '.join(extractors_to_process)}")
            else:
                logger.error(f"Features directory not found: {features_base_dir}")
                raise FileNotFoundError(f"No features found in {features_base_dir}")

        # Process each extractor
        for extractor in extractors_to_process:
            logger.info("=" * 80)
            logger.info(f"Processing extractor: {extractor}")
            logger.info("=" * 80)

            # Run feature validation pipeline
            result_tables = feature_validation.prepare_tables_for_training(
                clinical_table=clinical_table,
                slide_table=slide_table,
                output_dir=args.output_dir,
                features_base_dir=args.features_dir,
                extractor_name=extractor,
                save_by_site=not args.no_split_by_site
            )

            logger.info(f"Completed validation for {extractor}")
            if 'all' in result_tables:
                clinical, slides = result_tables['all']
                logger.info(f"  - {len(clinical)} patients, {len(slides)} slides with features")

        logger.info("=" * 80)
        logger.info(f"Feature validation completed for {len(extractors_to_process)} extractor(s)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Feature validation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
