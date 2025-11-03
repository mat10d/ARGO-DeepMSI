#!/usr/bin/env python3
"""
Stage 5: Baseline Testing

Tests pre-trained HistoBistro model on CTransPath features to validate data quality.
HistoBistro achieves 0.99 NPV on published data, so similar performance validates
that our data and features are of good quality.

Usage:
    python scripts/5_baseline_testing.py --clinical tables/2/all_clinical_table.csv \
                                          --slides tables/2/all_slide_table.csv \
                                          [--output-dir results/baseline]

Environment: HistoBistro (conda activate histobistro)
"""

import argparse
import pandas as pd
from pathlib import Path
from argo_deepmsi import io_utils


def prepare_histobistro_tables(clinical_table: pd.DataFrame, slide_table: pd.DataFrame,
                               output_dir: Path) -> tuple:
    """Prepare tables in HistoBistro format (Excel with specific column names).

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT and FILENAME columns
        output_dir: Directory to save Excel files

    Returns:
        Tuple of (clinical_excel_path, slide_excel_path)
    """
    # HistoBistro expects:
    # - Clinical table: MSIH (not isMSIH), values should be "MSIH" and "nonMSIH"
    # - Format: Excel (.xlsx)

    clinical_histobistro = clinical_table.copy()
    clinical_histobistro['isMSIH'] = clinical_histobistro['isMSIH'].replace({
        'MSS': 'nonMSIH',
        'MSI-H': 'MSIH'
    })

    slide_histobistro = slide_table.copy()

    # Save as Excel
    io_utils.ensure_dir(output_dir)
    clinical_path = output_dir / "clinical_table_histobistro.xlsx"
    slide_path = output_dir / "slide_table_histobistro.xlsx"

    clinical_histobistro.to_excel(clinical_path, index=False)
    slide_histobistro.to_excel(slide_path, index=False)

    return clinical_path, slide_path


def main():
    """Run baseline testing with HistoBistro."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Baseline Testing - Validate data with pre-trained HistoBistro model"
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
        help="Path to slide table CSV (must have CTransPath features in FILENAME column)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for baseline results (default: results/baseline/)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = io_utils.setup_logging(
        name="baseline_testing",
        log_dir=io_utils.get_project_root() / "logs" / "baseline_testing"
    )

    logger.info("=" * 80)
    logger.info("ARGO-DeepMSI: Stage 5 - Baseline Testing")
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

        # Set output directory
        output_dir = Path(args.output_dir) if args.output_dir else io_utils.get_results_dir("baseline")

        # Prepare tables in HistoBistro format
        logger.info("Preparing tables in HistoBistro format...")
        clinical_excel, slide_excel = prepare_histobistro_tables(clinical_table, slide_table, output_dir)
        logger.info(f"Saved HistoBistro-format tables to {output_dir}")

        # Instructions for running HistoBistro
        logger.info("=" * 80)
        logger.info("Tables prepared for HistoBistro baseline testing")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Activate HistoBistro environment: conda activate histobistro")
        logger.info("2. Navigate to HistoBistro directory: cd HistoBistro")
        logger.info("3. Run inference with prepared tables:")
        logger.info(f"   python scripts/inference.py \\")
        logger.info(f"     --clinical {clinical_excel} \\")
        logger.info(f"     --slides {slide_excel} \\")
        logger.info(f"     --output {output_dir / 'predictions.csv'}")
        logger.info("")
        logger.info("Expected performance: NPV ~0.99 (from published results)")
        logger.info("If NPV is significantly lower, check data quality and feature extraction")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Baseline testing preparation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
