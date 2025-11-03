"""
Feature validation module for ARGO-DeepMSI pipeline.

Validates feature extraction results, identifies missing features,
and generates processing status reports.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging

from .io_utils import get_project_root, get_data_dir, ensure_dir


logger = logging.getLogger(__name__)


def find_feature_file(
    slide_filename: str,
    site: str,
    features_base_dir: Path,
    extractor_name: Optional[str] = None
) -> Optional[str]:
    """Find feature file path for a given slide.

    Args:
        slide_filename: Slide filename (without path)
        site: Site name
        features_base_dir: Base directory for features
        extractor_name: Name of the extractor (ctranspath, h-optimus-0, etc.).
                       If None, searches for any extractor.

    Returns:
        Full path to the feature file or None if not found
    """
    # Extract base filename without extension
    base_filename = Path(slide_filename).stem

    # Check for feature directories
    site_path = features_base_dir / site / 'features'
    if not site_path.exists():
        return None

    # Look for feature directories matching the extractor
    feature_dirs = []
    for item in site_path.iterdir():
        if not item.is_dir():
            continue

        # If extractor specified, match it
        if extractor_name:
            if extractor_name.lower() in item.name.lower() or extractor_name.replace('-', '_') in item.name:
                # Check for H5 files directly or in subdirectories
                if any(f.suffix == '.h5' for f in item.iterdir() if f.is_file()):
                    feature_dirs.append(item)
                else:
                    # Look for hash subdirectories
                    for subitem in item.iterdir():
                        if subitem.is_dir() and any(f.suffix == '.h5' for f in subitem.iterdir() if f.is_file()):
                            feature_dirs.append(subitem)
        else:
            # Default: look for ctranspath or xiyuewang
            if 'ctranspath' in item.name or 'xiyuewang' in item.name:
                feature_dirs.append(item)

    # Check each feature directory for the H5 file
    for feature_dir in feature_dirs:
        h5_path = feature_dir / f"{base_filename}.h5"
        if h5_path.exists():
            return str(h5_path)

    return None


def add_feature_paths(
    slide_table: pd.DataFrame,
    features_base_dir: Optional[Path] = None,
    extractor_name: Optional[str] = None
) -> pd.DataFrame:
    """Add feature file paths to the slide table.

    Args:
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory for features (if None, uses project data dir)
        extractor_name: Name of the extractor to search for

    Returns:
        Updated slide table with FEATURE_PATH column
    """
    if features_base_dir is None:
        features_base_dir = get_data_dir()
    else:
        features_base_dir = Path(features_base_dir)

    result = slide_table.copy()
    result['FEATURE_PATH'] = None

    logger.info("Adding feature paths to slide table...")
    feature_count = 0

    # Check each slide for associated feature file
    for idx, row in result.iterrows():
        filename = row['FILENAME']
        site = row.get('SITE')

        if pd.isna(filename) or not isinstance(filename, str) or pd.isna(site):
            continue

        # Find feature file
        feature_path = find_feature_file(filename, site, features_base_dir, extractor_name)
        if feature_path:
            result.at[idx, 'FEATURE_PATH'] = feature_path
            feature_count += 1

    logger.info(f"Feature path addition complete: {feature_count} found, {len(result) - feature_count} not found")

    # Log examples
    found_features = result[result['FEATURE_PATH'].notna()]
    if not found_features.empty:
        logger.info("Examples of found features:")
        for _, row in found_features.head(3).iterrows():
            logger.info(f"  - {Path(row['FILENAME']).name} → {row['FEATURE_PATH']}")

    return result


def check_processing_status(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    features_base_dir: Optional[Path] = None,
    extractor_name: Optional[str] = None
) -> pd.DataFrame:
    """Check which slides have been processed (H5 files exist).

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory where feature H5 files are stored
        extractor_name: Name of the extractor to check

    Returns:
        Merged DataFrame with 'processed' column indicating status
    """
    if features_base_dir is None:
        features_base_dir = get_data_dir()
    else:
        features_base_dir = Path(features_base_dir)

    # Ensure SITE column exists
    if 'SITE' not in slide_table.columns:
        raise ValueError("SITE column not found in slide_table. Cannot check processing status.")

    # Merge clinical and slide tables to get MSI status for each slide
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')
    merged_data['processed'] = False

    logger.info("Checking for processed slides (H5 files)...")
    processed_count = 0

    # Find all feature directories across all sites
    feature_dirs = {}
    for site in merged_data['SITE'].unique():
        site_path = features_base_dir / site / 'features'
        if site_path.exists():
            # Look for subdirectories containing extractor name
            for item in site_path.iterdir():
                if not item.is_dir():
                    continue

                # Match extractor if specified
                if extractor_name:
                    if extractor_name.lower() in item.name.lower():
                        feature_dirs[site] = item
                        logger.info(f"Found feature directory for {site}: {item}")
                        break
                else:
                    # Default: ctranspath or xiyuewang
                    if 'ctranspath' in item.name or 'xiyuewang' in item.name:
                        feature_dirs[site] = item
                        logger.info(f"Found feature directory for {site}: {item}")
                        break

    if not feature_dirs:
        logger.warning(f"No feature directories found in {features_base_dir}")
        return merged_data

    logger.info(f"Found feature directories for {len(feature_dirs)} sites")

    # Check each slide for H5 files
    for idx, row in merged_data.iterrows():
        filename = row['FILENAME']
        if pd.isna(filename) or not isinstance(filename, str):
            continue

        # Extract just the filename without path or extension
        base_filename = Path(filename).stem

        # Check for H5 file in the site's feature directory
        site = row['SITE']
        if site in feature_dirs:
            feature_dir = feature_dirs[site]
            h5_path = feature_dir / f"{base_filename}.h5"
            if h5_path.exists():
                merged_data.at[idx, 'processed'] = True
                processed_count += 1

    pct = (processed_count / len(merged_data) * 100) if len(merged_data) > 0 else 0
    logger.info(f"Processing status: {processed_count} of {len(merged_data)} slides processed ({pct:.1f}%)")

    return merged_data


def generate_extraction_report(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    features_base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    extractor_name: Optional[str] = None
) -> pd.DataFrame:
    """Generate extraction report showing which slides passed/failed feature extraction.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory where feature H5 files are stored
        output_dir: Directory to save report files (default: results/feature_validation/)
        extractor_name: Name of the extractor

    Returns:
        DataFrame with processing status for all slides
    """
    if output_dir is None:
        output_dir = get_project_root() / "results" / "feature_validation"
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    # Check processing status
    merged_data = check_processing_status(clinical_table, slide_table, features_base_dir, extractor_name)

    # Create missing slides report
    missing_slides = merged_data[merged_data['processed'] == False].copy()
    missing_slides = missing_slides[['PATIENT', 'SITE', 'FILENAME', 'isMSIH']]
    missing_slides.columns = ['PATIENT', 'SITE', 'FILENAME', 'MSI_STATUS']

    missing_path = output_dir / "missing_slides.csv"
    missing_slides.to_csv(missing_path, index=False)
    logger.info(f"Saved list of {len(missing_slides)} missing slides to {missing_path}")

    # Create summary statistics by site
    site_stats = merged_data.groupby('SITE').agg(
        Total=('FILENAME', 'count'),
        Processed=('processed', 'sum')
    ).reset_index()

    site_stats['Percentage'] = (site_stats['Processed'] / site_stats['Total'] * 100).round(1)
    site_stats = site_stats.sort_values('Total', ascending=False)

    summary_path = output_dir / "extraction_summary_by_site.csv"
    site_stats.to_csv(summary_path, index=False)
    logger.info(f"Saved extraction summary to {summary_path}")

    # Create patient-level summary
    patient_stats = merged_data.groupby(['PATIENT', 'isMSIH', 'SITE']).agg(
        processed_any=('processed', 'any'),
        processed_count=('processed', 'sum'),
        total_slides=('FILENAME', 'count')
    ).reset_index()

    patient_stats['percentage'] = (patient_stats['processed_count'] / patient_stats['total_slides'] * 100).round(1)

    missing_patients = patient_stats[patient_stats['processed_any'] == False].copy()
    missing_patients = missing_patients[['PATIENT', 'isMSIH', 'SITE', 'total_slides']]
    missing_patients.columns = ['PATIENT', 'MSI_STATUS', 'SITE', 'SLIDE_COUNT']

    missing_patients_path = output_dir / "missing_patients.csv"
    missing_patients.to_csv(missing_patients_path, index=False)
    logger.info(f"Saved missing patient stats to {missing_patients_path}")

    # Print detailed summary
    logger.info("\nProcessing Status Summary by Site:")
    logger.info("-" * 60)
    logger.info(f"{'Site':<15} {'MSI-H':<20} {'MSS':<20} {'Total':<15}")
    logger.info("-" * 60)

    for site in sorted(merged_data['SITE'].unique()):
        site_data = merged_data[merged_data['SITE'] == site]
        total_processed = site_data['processed'].sum()
        total_slides = len(site_data)
        total_pct = (total_processed / total_slides * 100) if total_slides > 0 else 0

        # MSI-H stats
        msih_data = site_data[site_data['isMSIH'] == 'MSI-H']
        msih_processed = msih_data['processed'].sum()
        msih_total = len(msih_data)
        msih_pct = (msih_processed / msih_total * 100) if msih_total > 0 else 0

        # MSS stats
        mss_data = site_data[site_data['isMSIH'] == 'MSS']
        mss_processed = mss_data['processed'].sum()
        mss_total = len(mss_data)
        mss_pct = (mss_processed / mss_total * 100) if mss_total > 0 else 0

        msih_str = f"{msih_processed}/{msih_total} ({msih_pct:.1f}%)"
        mss_str = f"{mss_processed}/{mss_total} ({mss_pct:.1f}%)"
        total_str = f"{total_processed}/{total_slides} ({total_pct:.1f}%)"

        logger.info(f"{site:<15} {msih_str:<20} {mss_str:<20} {total_str:<15}")

    logger.info("-" * 60)
    total_processed = merged_data['processed'].sum()
    total_slides = len(merged_data)
    overall_pct = (total_processed / total_slides * 100) if total_slides > 0 else 0
    logger.info(f"Overall: {total_processed}/{total_slides} ({overall_pct:.1f}%)")
    logger.info("-" * 60)

    return merged_data


def split_tables_by_site(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame
) -> Tuple[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], List[str]]:
    """Split clinical and slide tables by site.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns

    Returns:
        Tuple of (split_tables dict, sites list)
        split_tables: Dictionary with site names as keys and tuples of (clinical_table, slide_table) as values
        sites: List of site names
    """
    # Ensure SITE column exists
    if 'SITE' not in slide_table.columns:
        raise ValueError("SITE column not found in slide_table. Cannot split tables by site.")

    # Get list of unique sites
    sites = slide_table['SITE'].unique().tolist()
    logger.info(f"Found {len(sites)} unique sites: {', '.join(sites)}")

    # Dictionary to store split tables
    split_tables = {}

    # Process each site
    for site in sites:
        # Filter slides for this site
        site_slides = slide_table[slide_table['SITE'] == site].copy()

        # Get unique patients at this site
        site_patients = set(site_slides['PATIENT'].unique())

        # Filter clinical data for patients at this site
        site_clinical = clinical_table[clinical_table['PATIENT'].isin(site_patients)].copy()

        # Store in dictionary
        split_tables[site] = (site_clinical, site_slides)

        # Log MSI distribution for this site
        msi_counts = site_clinical['isMSIH'].value_counts()

        logger.info(f"\nSite: {site}")
        logger.info(f"  - Patients: {len(site_clinical)}")
        logger.info(f"  - Slides: {len(site_slides)}")
        for status, count in msi_counts.items():
            logger.info(f"  - {status}: {count} patients ({count/len(site_clinical)*100:.1f}%)")

    return split_tables, sites


def prepare_tables_for_training(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_dir: Optional[Path] = None,
    features_base_dir: Optional[Path] = None,
    extractor_name: Optional[str] = None,
    save_by_site: bool = True
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Complete feature validation pipeline: check extraction, prepare tables for training.

    This replaces FILENAME with FEATURE_PATH and optionally splits by site.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        output_dir: Directory to save output tables (default: tables/2/)
        features_base_dir: Base directory for features
        extractor_name: Name of the extractor
        save_by_site: Whether to save site-specific tables

    Returns:
        Dictionary with 'all' key and optional site-specific keys containing (clinical, slide) tuples
    """
    if output_dir is None:
        output_dir = get_project_root() / "tables" / "2"
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    # Step 1: Add feature paths
    logger.info("Adding feature paths to slide table...")
    slide_table_with_features = add_feature_paths(slide_table, features_base_dir, extractor_name)

    # Step 2: Generate extraction report
    logger.info("Generating extraction report...")
    generate_extraction_report(clinical_table, slide_table_with_features, features_base_dir,
                               get_project_root() / "results" / "feature_validation", extractor_name)

    # Step 3: Prepare "all" table (consolidated across all sites)
    all_slide_table = slide_table_with_features.copy()
    # Remove slides without features
    all_slide_table = all_slide_table.dropna(subset=['FEATURE_PATH'])
    # Replace FILENAME with FEATURE_PATH
    all_slide_table['FILENAME'] = all_slide_table['FEATURE_PATH']
    all_slide_table = all_slide_table.drop(columns=['FEATURE_PATH'])

    # Save consolidated tables
    clinical_table.to_csv(output_dir / "all_clinical_table.csv", index=False)
    all_slide_table.to_csv(output_dir / "all_slide_table.csv", index=False)
    logger.info(f"Saved consolidated tables with {len(clinical_table)} patients and {len(all_slide_table)} slides")

    result = {'all': (clinical_table, all_slide_table)}

    # Step 4: Split by site if requested
    if save_by_site:
        logger.info("Splitting tables by site...")
        split_tables, sites = split_tables_by_site(clinical_table, slide_table_with_features)

        for site, (site_clinical, site_slides) in split_tables.items():
            # Prepare slide table (replace FILENAME with FEATURE_PATH)
            site_slides_prepared = site_slides.copy()
            site_slides_prepared = site_slides_prepared.dropna(subset=['FEATURE_PATH'])
            site_slides_prepared['FILENAME'] = site_slides_prepared['FEATURE_PATH']
            site_slides_prepared = site_slides_prepared.drop(columns=['FEATURE_PATH'])

            # Save site-specific tables
            site_clinical.to_csv(output_dir / f"{site}_clinical_table.csv", index=False)
            site_slides_prepared.to_csv(output_dir / f"{site}_slide_table.csv", index=False)

            logger.info(f"Saved {site} tables with {len(site_clinical)} patients and {len(site_slides_prepared)} slides")
            result[site] = (site_clinical, site_slides_prepared)

    logger.info("Feature validation complete")
    return result
