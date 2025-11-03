#!/usr/bin/env python3
"""
Stage 1: Data Ingestion

Fetches patient data from REDCap, loads Halo Link slide exports,
and creates clinical and slide tables.

Usage:
    python scripts/1_data_ingestion.py [--output-dir tables/0]

Environment: ARGO (conda activate argo)
"""

import argparse
from argo_deepmsi import io_utils, data_ingestion


def main():
    """Run data ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Data Ingestion - Fetch REDCap data and create tables"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tables (default: tables/0/)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="REDCap API URL (default: from .env file)"
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="REDCap API token (default: from .env file)"
    )
    parser.add_argument(
        "--halo-dir",
        type=str,
        default=None,
        help="Directory containing Halo Link CSV files (default: data/)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = io_utils.setup_logging(
        name="data_ingestion",
        log_dir=io_utils.get_project_root() / "logs" / "data_ingestion"
    )

    logger.info("=" * 80)
    logger.info("ARGO-DeepMSI: Stage 1 - Data Ingestion")
    logger.info("=" * 80)

    try:
        # Run data ingestion pipeline
        clinical_table, slide_table = data_ingestion.process_redcap_data(
            output_dir=args.output_dir,
            api_url=args.api_url,
            api_token=args.api_token,
            halo_base_dir=args.halo_dir
        )

        logger.info("=" * 80)
        logger.info("Data ingestion completed successfully")
        logger.info(f"  - Clinical table: {len(clinical_table)} patients")
        logger.info(f"  - Slide table: {len(slide_table)} slides")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
