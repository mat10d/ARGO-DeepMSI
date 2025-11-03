"""
Data ingestion module for ARGO-DeepMSI pipeline.

Handles REDCap data fetching, Halo Link data loading, and clinical/slide table creation.
"""

import os
import requests
import pandas as pd
import glob
from pathlib import Path
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import logging

from .io_utils import get_project_root, ensure_dir


logger = logging.getLogger(__name__)


def fetch_redcap_data(api_url: Optional[str] = None, api_token: Optional[str] = None) -> pd.DataFrame:
    """Fetch patient data from REDCap API.

    Args:
        api_url: REDCap API URL (if None, loads from environment)
        api_token: REDCap API token (if None, loads from environment)

    Returns:
        DataFrame with REDCap patient records.

    Raises:
        Exception: If REDCap API request fails.
    """
    if api_url is None or api_token is None:
        load_dotenv()
        api_token = api_token or os.getenv('REDCAP_API_TOKEN')
        api_url = api_url or os.getenv('REDCAP_API_URL')

    if not api_token or not api_url:
        raise ValueError("REDCap API credentials not found. Set REDCAP_API_TOKEN and REDCAP_API_URL in .env file")

    payload = {
        'token': api_token,
        'content': 'record',
        'format': 'json',
        'type': 'flat',
        'rawOrLabel': 'raw',
        'rawOrLabelHeaders': 'raw',
        'exportDataAccessGroups': 'true'
    }

    logger.info(f"Fetching data from REDCap API: {api_url}")
    response = requests.post(api_url, data=payload)

    if response.status_code != 200:
        raise Exception(f"Error accessing REDCap API: {response.text}")

    data = response.json()
    df = pd.DataFrame(data)
    df.replace('', pd.NA, inplace=True)

    logger.info(f"Fetched {len(df)} records from REDCap")
    return df


def create_clinical_table(redcap_data: pd.DataFrame) -> pd.DataFrame:
    """Create clinical table with patient IDs and MSI status.

    Args:
        redcap_data: DataFrame from REDCap API

    Returns:
        DataFrame with columns: PATIENT, isMSIH, batch_number, redcap_data_access_group
    """
    clinical_table = pd.DataFrame(columns=['PATIENT', 'isMSIH'])

    # Split data into prospective (batch != 1,2) and retrospective (batch = 1,2)
    prospective_data = redcap_data[
        (redcap_data['batch_number'] != '1') &
        (redcap_data['batch_number'] != '2')
    ].copy()

    retrospective_data = redcap_data[
        (redcap_data['batch_number'] == '1') |
        (redcap_data['batch_number'] == '2')
    ].copy()

    # Process prospective data (uses cmo_msi_status field)
    if not prospective_data.empty:
        prospective_msi = prospective_data[['record_id', 'cmo_msi_status', 'batch_number', 'redcap_data_access_group']].copy()
        prospective_msi['isMSIH'] = prospective_msi['cmo_msi_status'].map({
            'Instable': 'MSI-H',
            'Stable': 'MSS',
            'Indeterminate': 'MSS',
            'Stable, Indeterminate': 'MSS'
        })
        prospective_msi.rename(columns={'record_id': 'PATIENT'}, inplace=True)
        prospective_msi = prospective_msi[['PATIENT', 'isMSIH', 'batch_number', 'redcap_data_access_group']]
        clinical_table = pd.concat([clinical_table, prospective_msi], ignore_index=True)
        logger.info(f"Processed {len(prospective_msi)} prospective patients")

    # Process retrospective data (uses msi_status_mmr field)
    if not retrospective_data.empty:
        retrospective_msi = retrospective_data[['record_id', 'msi_status_mmr', 'batch_number', 'redcap_data_access_group']].copy()
        retrospective_msi['isMSIH'] = retrospective_msi['msi_status_mmr'].map({'1': 'MSI-H', '2': 'MSS'})
        retrospective_msi.rename(columns={'record_id': 'PATIENT'}, inplace=True)
        retrospective_msi = retrospective_msi[['PATIENT', 'isMSIH', 'batch_number', 'redcap_data_access_group']]
        clinical_table = pd.concat([clinical_table, retrospective_msi], ignore_index=True)
        logger.info(f"Processed {len(retrospective_msi)} retrospective patients")

    # Ensure PATIENT column is string
    clinical_table['PATIENT'] = clinical_table['PATIENT'].astype(str)

    logger.info(f"Created clinical table with {len(clinical_table)} total patients")
    return clinical_table


def load_halo_link_data(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all Halo Link CSV export files from site directories.

    Args:
        base_dir: Base directory to search for halo_link_*.csv files
                 (if None, searches data/*/halo_link_*.csv in all site directories)

    Returns:
        Combined DataFrame from all Halo Link files with standardized columns.
    """
    if base_dir is None:
        # Search in all site directories: data/*/halo_link_*.csv
        base_dir = get_project_root() / "data"
    else:
        base_dir = Path(base_dir)

    # Search recursively for halo_link_*.csv in site subdirectories
    halo_files = list(base_dir.glob('*/halo_link_*.csv'))

    if not halo_files:
        logger.warning(f"No Halo Link files found in {base_dir}")
        return pd.DataFrame()

    halo_dfs = []

    for file in halo_files:
        site_name = file.stem.replace('halo_link_', '').replace('_export', '')
        try:
            df = pd.read_csv(file)
            df['site'] = site_name
            halo_dfs.append(df)
            logger.info(f"Loaded Halo data for {site_name} ({len(df)} records)")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not halo_dfs:
        return pd.DataFrame()

    combined_halo = pd.concat(halo_dfs, ignore_index=True)

    # Standardize column names
    standard_col_map = {
        'Slide ID': 'slide_id',
        'Study Image ID': 'image_id',
        'Name': 'filename',
        'Image Location': 'image_location',
        'Pathology REDCap ID': 'redcap_id'
    }

    combined_halo.rename(
        columns={k: v for k, v in standard_col_map.items() if k in combined_halo.columns},
        inplace=True
    )

    logger.info(f"Combined {len(halo_dfs)} Halo Link files, total {len(combined_halo)} records")
    return combined_halo


def create_slide_table(halo_data: pd.DataFrame) -> pd.DataFrame:
    """Create slide table relating patients to their slide files.

    Args:
        halo_data: DataFrame with Halo Link data

    Returns:
        DataFrame with columns: PATIENT, FILENAME, SITE
    """
    slide_table = pd.DataFrame(columns=['PATIENT', 'FILENAME', 'SITE'])

    # Check if we have the necessary columns
    if 'redcap_id' not in halo_data.columns or 'filename' not in halo_data.columns:
        logger.warning("Missing required columns in Halo data for slide table")

        if 'redcap_id' not in halo_data.columns:
            logger.warning("- Missing 'redcap_id' column (Pathology REDCap ID)")
        if 'filename' not in halo_data.columns:
            logger.warning("- Missing 'filename' column (Name)")

        # Try to find alternative columns
        patient_id_cols = [col for col in halo_data.columns if 'id' in col.lower() and 'redcap' in col.lower()]
        filename_cols = [col for col in halo_data.columns if 'name' in col.lower() or 'file' in col.lower()]

        if patient_id_cols and filename_cols:
            logger.info(f"Using alternative columns: {patient_id_cols[0]} and {filename_cols[0]}")
            temp_df = halo_data[[patient_id_cols[0], filename_cols[0]]].copy()
            temp_df.columns = ['PATIENT', 'FILENAME']
            slide_table = pd.concat([slide_table, temp_df], ignore_index=True)
        else:
            return slide_table
    else:
        # Extract relevant columns
        temp_df = halo_data[['redcap_id', 'filename', 'site']].copy()
        temp_df.columns = ['PATIENT', 'FILENAME', 'SITE']
        slide_table = pd.concat([slide_table, temp_df], ignore_index=True)

    # Drop rows with missing values
    slide_table = slide_table.dropna(subset=['PATIENT', 'FILENAME'])

    # Ensure PATIENT column is string
    slide_table['PATIENT'] = slide_table['PATIENT'].astype(str)

    logger.info(f"Created slide table with {len(slide_table)} slides")
    return slide_table


def verify_slides_exist(
    slide_table: pd.DataFrame,
    slide_dirs: Optional[List[Path]] = None
) -> pd.DataFrame:
    """Verify if slides exist in specified directories and add absolute paths.

    Args:
        slide_table: DataFrame with PATIENT and FILENAME columns
        slide_dirs: List of directories to search
                   (if None, searches data/*/raw/ in all site directories)

    Returns:
        Updated DataFrame with slide_exists and slide_path columns.
    """
    result = slide_table.copy()

    # Add columns for slide existence and path
    result['slide_exists'] = False
    result['slide_path'] = None

    # Default directories if not specified
    if not slide_dirs:
        # Search in all site-specific raw directories: data/*/raw/
        data_root = get_project_root() / "data"
        slide_dirs = []
        if data_root.exists():
            # Find all site directories with raw/ subdirectories
            for site_dir in data_root.iterdir():
                if site_dir.is_dir():
                    raw_dir = site_dir / "raw"
                    if raw_dir.exists():
                        slide_dirs.append(raw_dir)
    else:
        slide_dirs = [Path(d) for d in slide_dirs]

    logger.info(f"Starting slide verification in {len(slide_dirs)} base directories")

    # Create dictionary of all found files for faster lookup
    found_files = {}

    for base_dir in slide_dirs:
        if not base_dir.is_dir():
            logger.warning(f"Directory {base_dir} does not exist")
            continue

        logger.info(f"Scanning directory: {base_dir}")
        for root, _, files in os.walk(base_dir):
            for file in files:
                absolute_path = Path(root) / file
                found_files[file] = str(absolute_path.resolve())
                # Also add version without .svs extension for easier matching
                if file.endswith('.svs'):
                    found_files[file[:-4]] = str(absolute_path.resolve())

    logger.info(f"Found {len(found_files)} unique filenames in all directories")

    # Check each slide
    found_count = 0
    for idx, row in result.iterrows():
        filename = row['FILENAME']
        if pd.isna(filename) or not isinstance(filename, str):
            continue

        # Check if file exists in our dictionary
        if filename in found_files:
            result.at[idx, 'slide_exists'] = True
            result.at[idx, 'slide_path'] = found_files[filename]
            found_count += 1
        # Try with .svs extension if not found
        elif filename + '.svs' in found_files:
            result.at[idx, 'slide_exists'] = True
            result.at[idx, 'slide_path'] = found_files[filename + '.svs']
            found_count += 1

    logger.info(f"Slide verification complete: {found_count} found, {len(result) - found_count} not found")

    # Log examples
    found_slides = result[result['slide_exists'] == True]
    if not found_slides.empty:
        logger.info("Examples of found slides:")
        for _, row in found_slides.head(3).iterrows():
            logger.info(f"  - {row['FILENAME']} → {row['slide_path']}")

    missing_slides = result[result['slide_exists'] == False]
    if not missing_slides.empty:
        logger.warning("Examples of missing slides:")
        for _, row in missing_slides.head(3).iterrows():
            logger.warning(f"  - {row['FILENAME']} (Patient {row['PATIENT']})")

    return result


def clean_tables(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean up tables to ensure consistency.

    Removes:
    - Patients with missing MSI status
    - Slides for patients without MSI status
    - Patients without any slides

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT and FILENAME columns

    Returns:
        Tuple of cleaned (clinical_table, slide_table)
    """
    clinical = clinical_table.copy()
    slides = slide_table.copy()

    logger.info(f"Initial counts: {len(clinical)} patients, {len(slides)} slides")

    # 1. Remove patients with missing MSI status
    initial_clinical_count = len(clinical)
    clinical = clinical.dropna(subset=['isMSIH'])
    dropped_clinical = initial_clinical_count - len(clinical)
    if dropped_clinical > 0:
        logger.info(f"Removed {dropped_clinical} patients with missing MSI status")

    # 2. Get list of valid patients (those with MSI status)
    valid_patients = set(clinical['PATIENT'].unique())

    # 3. Remove slides for patients without MSI status
    initial_slide_count = len(slides)
    slides = slides[slides['PATIENT'].isin(valid_patients)]
    dropped_slides = initial_slide_count - len(slides)
    if dropped_slides > 0:
        logger.info(f"Removed {dropped_slides} slides for patients without MSI status")

    # 4. Update clinical table to include only patients with slides
    patients_with_slides = set(slides['PATIENT'].unique())
    initial_clinical_count = len(clinical)
    clinical = clinical[clinical['PATIENT'].isin(patients_with_slides)]
    dropped_clinical = initial_clinical_count - len(clinical)
    if dropped_clinical > 0:
        logger.info(f"Removed {dropped_clinical} patients without slides")

    # 5. Final summary
    logger.info(f"Final counts: {len(clinical)} patients, {len(slides)} slides")

    # 6. Log MSI distribution
    if not clinical.empty:
        msi_counts = clinical['isMSIH'].value_counts()
        for status, count in msi_counts.items():
            logger.info(f"  - {status}: {count} patients ({count/len(clinical)*100:.1f}%)")

    return clinical, slides


def process_redcap_data(
    output_dir: Optional[Path] = None,
    api_url: Optional[str] = None,
    api_token: Optional[str] = None,
    halo_base_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Complete data ingestion pipeline: REDCap → clinical/slide tables.

    This is the main entry point that orchestrates the entire data ingestion process.

    Args:
        output_dir: Directory to save output tables (default: results/stage1_data_ingestion/)
        api_url: REDCap API URL (if None, loads from environment)
        api_token: REDCap API token (if None, loads from environment)
        halo_base_dir: Directory with Halo Link CSV files (if None, uses data/metadata/)

    Returns:
        Tuple of (clinical_table, slide_table) DataFrames

    Raises:
        Exception: If REDCap fetch or data processing fails
    """
    if output_dir is None:
        from .io_utils import get_stage_dir
        output_dir = get_stage_dir(1)  # results/stage1_data_ingestion/
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    # Step 1: Fetch REDCap data
    logger.info("Fetching data from REDCap...")
    redcap_data = fetch_redcap_data(api_url, api_token)

    # Step 2: Create clinical table
    logger.info("Extracting clinical information...")
    clinical_table = create_clinical_table(redcap_data)
    clinical_table.to_csv(output_dir / "clinical_table_full.csv", index=False)
    logger.info(f"Saved clinical table with {len(clinical_table)} patients")

    # Step 3: Load Halo Link data
    logger.info("Loading Halo Link data...")
    halo_data = load_halo_link_data(halo_base_dir)

    # Step 4: Create slide table
    logger.info("Creating slide table...")
    slide_table = create_slide_table(halo_data)

    # Step 5: Verify slides exist
    logger.info("Verifying slides exist...")
    slide_table = verify_slides_exist(slide_table)
    slide_table['FILENAME'] = slide_table['slide_path']
    slide_table = slide_table.drop(columns=['slide_path', 'slide_exists'])
    slide_table.to_csv(output_dir / "slide_table_full.csv", index=False)
    logger.info(f"Saved slide table with {len(slide_table)} slides")

    # Step 6: Clean tables
    logger.info("Cleaning tables...")
    clinical_table, slide_table = clean_tables(clinical_table, slide_table)

    # Save final cleaned tables
    clinical_table.to_csv(output_dir / "clinical_table.csv", index=False)
    slide_table.to_csv(output_dir / "slide_table.csv", index=False)

    logger.info("Data ingestion complete")
    logger.info(f"  - Clinical table: {len(clinical_table)} patients")
    logger.info(f"  - Slide table: {len(slide_table)} slides")

    return clinical_table, slide_table
