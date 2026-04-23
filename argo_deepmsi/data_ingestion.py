"""
Data ingestion module for ARGO-DeepMSI pipeline.

Handles REDCap data fetching, Halo Link data loading, and clinical/slide table creation.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from .io_utils import get_project_root, ensure_dir


logger = logging.getLogger(__name__)


def fetch_redcap_data(
    api_url: Optional[str] = None, api_token: Optional[str] = None
) -> pd.DataFrame:
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
        api_token = api_token or os.getenv("REDCAP_API_TOKEN")
        api_url = api_url or os.getenv("REDCAP_API_URL")

    if not api_token or not api_url:
        raise ValueError(
            "REDCap API credentials not found. Set REDCAP_API_TOKEN and REDCAP_API_URL in .env file"
        )

    payload = {
        "token": api_token,
        "content": "record",
        "format": "json",
        "type": "flat",
        "rawOrLabel": "raw",
        "rawOrLabelHeaders": "raw",
        "exportDataAccessGroups": "true",
    }

    logger.info(f"Fetching data from REDCap API: {api_url}")
    response = requests.post(api_url, data=payload)

    if response.status_code != 200:
        raise Exception(f"Error accessing REDCap API: {response.text}")

    data = response.json()
    df = pd.DataFrame(data)
    df.replace("", pd.NA, inplace=True)

    logger.info(f"Fetched {len(df)} records from REDCap")
    return df


def create_clinical_table(redcap_data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Create clinical table with patient IDs and MSI status.

    Assigns sequential patient IDs (P_0001, P_0002, etc.) sorted by record_id.
    For retrospective patients (batch 1, 2), uses crc_redcap_number to properly
    merge patients with multiple slides processed at different locations.

    Args:
        redcap_data: DataFrame from REDCap API

    Returns:
        Tuple of:
        - DataFrame with columns: PATIENT, record_id, crc_redcap_number, isMSIH, batch_number, redcap_data_access_group
        - dict mapping all record_ids to PATIENT IDs (for slide table creation)
    """
    all_records = []
    record_id_mapping = {}  # Maps all record_ids to their sequential PATIENT ID

    # Split data into prospective (batch != 1,2) and retrospective (batch = 1,2)
    prospective_data = redcap_data[
        (redcap_data["batch_number"] != "1") & (redcap_data["batch_number"] != "2")
    ].copy()

    retrospective_data = redcap_data[
        (redcap_data["batch_number"] == "1") | (redcap_data["batch_number"] == "2")
    ].copy()

    # Optional quantitative / methodological fields that downstream analyses
    # benefit from (cmo_msi_score is MSIsensor-style %, populated for the
    # prospective CMO-molecular arm only; IHC protein calls are populated for
    # retrospective MMR-IHC records). Retained verbatim where present.
    extra_fields = [
        "cmo_msi_score", "msi_method", "sample_type",
        "mlh1", "msh2", "msh6", "pms2",
        "tissue_processing_site", "slide_staining_site", "slide_imaging_site",
    ]

    def _carry_extras(df_in):
        return df_in[[c for c in extra_fields if c in df_in.columns]].copy()

    # Process prospective data (uses cmo_msi_status field)
    if not prospective_data.empty:
        prospective_msi = prospective_data[
            ["record_id", "cmo_msi_status", "batch_number", "redcap_data_access_group"]
        ].copy()
        prospective_msi["isMSIH"] = prospective_msi["cmo_msi_status"].map(
            {
                "Instable": "MSI-H",
                "Stable": "MSS",
                "Indeterminate": "MSS",
                "Stable, Indeterminate": "MSS",
            }
        )
        prospective_msi["crc_redcap_number"] = None  # No CRC number for prospective
        prospective_msi["true_patient_id"] = prospective_msi["record_id"]  # For grouping
        prospective_msi = prospective_msi.reset_index(drop=True)
        prospective_msi = pd.concat(
            [prospective_msi, _carry_extras(prospective_data).reset_index(drop=True)],
            axis=1,
        )
        all_records.append(prospective_msi)
        logger.info(f"Processed {len(prospective_msi)} prospective patients")

    # Process retrospective data (uses msi_status_mmr field)
    if not retrospective_data.empty:
        # Check if crc_redcap_number exists
        if "crc_redcap_number" not in retrospective_data.columns:
            logger.warning(
                "crc_redcap_number not found in REDCap data - using record_id for retrospective patients"
            )
            retrospective_data["crc_redcap_number"] = retrospective_data["record_id"]

        retrospective_msi = retrospective_data[
            [
                "record_id",
                "crc_redcap_number",
                "msi_status_mmr",
                "batch_number",
                "redcap_data_access_group",
            ]
        ].copy()
        retrospective_msi["isMSIH"] = retrospective_msi["msi_status_mmr"].map(
            {"1": "MSI-H", "2": "MSS"}
        )
        retrospective_msi["cmo_msi_status"] = None  # Not applicable to retro arm
        retrospective_msi["true_patient_id"] = retrospective_msi["crc_redcap_number"]
        retrospective_msi = retrospective_msi.reset_index(drop=True)
        retrospective_msi = pd.concat(
            [retrospective_msi, _carry_extras(retrospective_data).reset_index(drop=True)],
            axis=1,
        )
        all_records.append(retrospective_msi)
        logger.info(f"Processed {len(retrospective_msi)} retrospective records")

    # Combine all records
    combined = pd.concat(all_records, ignore_index=True)

    # Group by true_patient_id to get unique patients
    # Keep all record_ids for mapping later
    patient_groups = combined.groupby("true_patient_id")

    clinical_records = []
    for true_id, group in patient_groups:
        # Take first record as representative
        record = group.iloc[0].copy()
        # Store all record_ids for this patient for mapping
        all_record_ids = group["record_id"].tolist()
        record["all_record_ids"] = all_record_ids
        clinical_records.append(record)

    clinical_table = pd.DataFrame(clinical_records)

    # Sort by the first (representative) record_id to ensure deterministic ordering
    clinical_table = clinical_table.sort_values("record_id").reset_index(drop=True)

    # Assign sequential patient IDs
    clinical_table["PATIENT"] = [f"P_{i + 1:04d}" for i in range(len(clinical_table))]

    # Create mapping from all record_ids to PATIENT
    for _, row in clinical_table.iterrows():
        patient_id = row["PATIENT"]
        for record_id in row["all_record_ids"]:
            record_id_mapping[str(record_id)] = patient_id

    # Select final columns. Core columns first so existing consumers
    # (clinical_table[["PATIENT","isMSIH"]].merge(…)) keep working; optional
    # quantitative / methodological fields append after and are NaN where
    # the record's assay arm doesn't populate them.
    core_cols = [
        "PATIENT",
        "record_id",
        "crc_redcap_number",
        "isMSIH",
        "batch_number",
        "redcap_data_access_group",
    ]
    optional_cols = [
        "cmo_msi_status", "cmo_msi_score", "msi_method", "msi_status_mmr",
        "mlh1", "msh2", "msh6", "pms2", "sample_type",
        "tissue_processing_site", "slide_staining_site", "slide_imaging_site",
    ]
    present_optional = [c for c in optional_cols if c in clinical_table.columns]
    clinical_table = clinical_table[core_cols + present_optional].copy()

    # Ensure string types
    clinical_table["PATIENT"] = clinical_table["PATIENT"].astype(str)
    clinical_table["record_id"] = clinical_table["record_id"].astype(str)

    logger.info(f"Created clinical table with {len(clinical_table)} unique patients")
    logger.info(f"Created mapping for {len(record_id_mapping)} record IDs")

    return clinical_table, record_id_mapping


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
    halo_files = list(base_dir.glob("*/halo_link_*.csv"))

    if not halo_files:
        logger.warning(f"No Halo Link files found in {base_dir}")
        return pd.DataFrame()

    halo_dfs = []

    for file in halo_files:
        site_name = file.stem.replace("halo_link_", "").replace("_export", "")
        try:
            df = pd.read_csv(file)
            df["site"] = site_name
            halo_dfs.append(df)
            logger.info(f"Loaded Halo data for {site_name} ({len(df)} records)")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not halo_dfs:
        return pd.DataFrame()

    combined_halo = pd.concat(halo_dfs, ignore_index=True)

    # Standardize column names
    standard_col_map = {
        "Slide ID": "slide_id",
        "Study Image ID": "image_id",
        "Name": "filename",
        "Image Location": "image_location",
        "Pathology REDCap ID": "redcap_id",
        "Cut location": "cut_location",
        "Stain location": "stain_location",
    }

    combined_halo.rename(
        columns={k: v for k, v in standard_col_map.items() if k in combined_halo.columns},
        inplace=True,
    )

    logger.info(f"Combined {len(halo_dfs)} Halo Link files, total {len(combined_halo)} records")
    return combined_halo


def create_slide_table(
    halo_data: pd.DataFrame, record_id_mapping: Optional[dict] = None
) -> pd.DataFrame:
    """Create slide table relating patients to their slide files.

    Uses record_id_mapping to convert Halo's record_ids to sequential PATIENT IDs.

    Args:
        halo_data: DataFrame with Halo Link data
        record_id_mapping: dict mapping record_ids to PATIENT IDs (from create_clinical_table)

    Returns:
        DataFrame with columns: PATIENT, record_id, FILENAME, SITE,
                               cut_location, stain_location, image_location
    """
    slide_table = pd.DataFrame(
        columns=[
            "PATIENT",
            "record_id",
            "FILENAME",
            "SITE",
            "cut_location",
            "stain_location",
            "image_location",
        ]
    )

    # Check if we have the necessary columns
    required_cols = ["redcap_id", "filename", "site"]

    if not all(col in halo_data.columns for col in required_cols):
        logger.warning("Missing required columns in Halo data for slide table")

        if "redcap_id" not in halo_data.columns:
            logger.warning("- Missing 'redcap_id' column (Pathology REDCap ID)")
        if "filename" not in halo_data.columns:
            logger.warning("- Missing 'filename' column (Name)")
        if "site" not in halo_data.columns:
            logger.warning("- Missing 'site' column")

        # Try to find alternative columns
        patient_id_cols = [
            col for col in halo_data.columns if "id" in col.lower() and "redcap" in col.lower()
        ]
        filename_cols = [
            col for col in halo_data.columns if "name" in col.lower() or "file" in col.lower()
        ]

        if patient_id_cols and filename_cols:
            logger.info(f"Using alternative columns: {patient_id_cols[0]} and {filename_cols[0]}")
            temp_df = halo_data[[patient_id_cols[0], filename_cols[0]]].copy()
            temp_df.columns = ["record_id", "FILENAME"]
            temp_df["PATIENT"] = "Unknown"
            # Fill missing columns with defaults
            temp_df["SITE"] = "Unknown"
            temp_df["cut_location"] = "Unknown"
            temp_df["stain_location"] = "Unknown"
            temp_df["image_location"] = "Nigeria"
            slide_table = pd.concat([slide_table, temp_df], ignore_index=True)
        else:
            return slide_table
    else:
        # Extract relevant columns
        extract_cols = ["redcap_id", "filename", "site"]

        # Add processing columns if available
        if "cut_location" in halo_data.columns:
            extract_cols.append("cut_location")
        if "stain_location" in halo_data.columns:
            extract_cols.append("stain_location")

        temp_df = halo_data[extract_cols].copy()

        # Rename columns to match slide table schema
        temp_df.rename(
            columns={"redcap_id": "record_id", "filename": "FILENAME", "site": "SITE"}, inplace=True
        )

        # Add image_location (all slides imaged in Nigeria)
        temp_df["image_location"] = "Nigeria"

        # Fill missing processing metadata with "Unknown"
        if "cut_location" not in temp_df.columns:
            temp_df["cut_location"] = "Unknown"
        if "stain_location" not in temp_df.columns:
            temp_df["stain_location"] = "Unknown"

        # Map record_id to PATIENT using the provided mapping
        if record_id_mapping:
            temp_df["PATIENT"] = temp_df["record_id"].astype(str).map(record_id_mapping)
            mapped_count = temp_df["PATIENT"].notna().sum()
            unmapped_count = temp_df["PATIENT"].isna().sum()
            logger.info(f"Mapped {mapped_count} slides to PATIENT IDs")
            if unmapped_count > 0:
                logger.warning(
                    f"Warning: {unmapped_count} slides could not be mapped to PATIENT IDs"
                )
                # Keep original record_id for unmapped slides
                temp_df.loc[temp_df["PATIENT"].isna(), "PATIENT"] = temp_df.loc[
                    temp_df["PATIENT"].isna(), "record_id"
                ]
        else:
            logger.warning("No record_id mapping provided - using record_id as PATIENT")
            temp_df["PATIENT"] = temp_df["record_id"]

        slide_table = pd.concat([slide_table, temp_df], ignore_index=True)

    # Drop rows with missing values in core columns
    slide_table = slide_table.dropna(subset=["PATIENT", "FILENAME"])

    # Ensure string types
    slide_table["PATIENT"] = slide_table["PATIENT"].astype(str)
    slide_table["record_id"] = slide_table["record_id"].astype(str)

    # Reorder columns
    slide_table = slide_table[
        [
            "PATIENT",
            "record_id",
            "FILENAME",
            "SITE",
            "cut_location",
            "stain_location",
            "image_location",
        ]
    ]

    logger.info(
        f"Created slide table with {len(slide_table)} slides for {slide_table['PATIENT'].nunique()} patients"
    )
    return slide_table


def verify_slides_exist(
    slide_table: pd.DataFrame, slide_dirs: Optional[List[Path]] = None
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
    result["slide_exists"] = False
    result["slide_path"] = None

    # Default directories if not specified
    if not slide_dirs:
        # Search in all site-specific directories: data/*/
        data_root = get_project_root() / "data"
        slide_dirs = []
        if data_root.exists():
            # Find all site directories
            for site_dir in data_root.iterdir():
                if site_dir.is_dir():
                    slide_dirs.append(site_dir)
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
                if file.endswith(".svs"):
                    found_files[file[:-4]] = str(absolute_path.resolve())

    logger.info(f"Found {len(found_files)} unique filenames in all directories")

    # Check each slide
    found_count = 0
    for idx, row in result.iterrows():
        filename = row["FILENAME"]
        if pd.isna(filename) or not isinstance(filename, str):
            continue

        # Check if file exists in our dictionary
        if filename in found_files:
            result.at[idx, "slide_exists"] = True
            result.at[idx, "slide_path"] = found_files[filename]
            found_count += 1
        # Try with .svs extension if not found
        elif filename + ".svs" in found_files:
            result.at[idx, "slide_exists"] = True
            result.at[idx, "slide_path"] = found_files[filename + ".svs"]
            found_count += 1

    logger.info(
        f"Slide verification complete: {found_count} found, {len(result) - found_count} not found"
    )

    # Log examples
    found_slides = result[result["slide_exists"]]
    if not found_slides.empty:
        logger.info("Examples of found slides:")
        for _, row in found_slides.head(3).iterrows():
            logger.info(f"  - {row['FILENAME']} → {row['slide_path']}")

    missing_slides = result[~result["slide_exists"]]
    if not missing_slides.empty:
        logger.warning("Examples of missing slides:")
        for _, row in missing_slides.head(3).iterrows():
            logger.warning(f"  - {row['FILENAME']} (Patient {row['PATIENT']})")

    return result


def clean_tables(
    clinical_table: pd.DataFrame, slide_table: pd.DataFrame
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
    clinical = clinical.dropna(subset=["isMSIH"])
    dropped_clinical = initial_clinical_count - len(clinical)
    if dropped_clinical > 0:
        logger.info(f"Removed {dropped_clinical} patients with missing MSI status")

    # 2. Get list of valid patients (those with MSI status)
    valid_patients = set(clinical["PATIENT"].unique())

    # 3. Remove slides for patients without MSI status
    initial_slide_count = len(slides)
    slides = slides[slides["PATIENT"].isin(valid_patients)]
    dropped_slides = initial_slide_count - len(slides)
    if dropped_slides > 0:
        logger.info(f"Removed {dropped_slides} slides for patients without MSI status")

    # 4. Update clinical table to include only patients with slides
    patients_with_slides = set(slides["PATIENT"].unique())
    initial_clinical_count = len(clinical)
    clinical = clinical[clinical["PATIENT"].isin(patients_with_slides)]
    dropped_clinical = initial_clinical_count - len(clinical)
    if dropped_clinical > 0:
        logger.info(f"Removed {dropped_clinical} patients without slides")

    # 5. Final summary
    logger.info(f"Final counts: {len(clinical)} patients, {len(slides)} slides")

    # 6. Log MSI distribution
    if not clinical.empty:
        msi_counts = clinical["isMSIH"].value_counts()
        for status, count in msi_counts.items():
            logger.info(f"  - {status}: {count} patients ({count / len(clinical) * 100:.1f}%)")

    return clinical, slides


def plot_ingestion_diagnostics(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    slide_table_full: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate diagnostic plots for data ingestion.

    Creates publication-ready plots showing:
    - Slides/patients per processing location (grouped by site with MSI breakdown)
    - MSI status distribution (overall and by site)
    - Missing slides summary

    Args:
        clinical_table: Clinical data (one row per patient)
        slide_table: Slide data (cleaned, only existing slides)
        slide_table_full: Slide data before cleaning (includes missing slides)
        output_dir: Directory to save plots (results/data/)
    """
    # Set publication-ready style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts for publications
    plt.rcParams["ps.fonttype"] = 42

    # Fix site labels: All MSKCC slides are from OAUTHC patients
    def fix_site_labels(df):
        """Map retrospective MSKCC sites to OAUTHC."""
        df = df.copy()
        site_mapping = {
            "retrospective_msk": "OAUTHC",
            "retrospective_oau": "OAUTHC",
            "OAU": "OAUTHC",  # Consolidate any OAU to OAUTHC
        }
        df["SITE"] = df["SITE"].replace(site_mapping)
        return df

    # Apply site fixes
    slide_table = fix_site_labels(slide_table)
    clinical_table = clinical_table.copy()

    # Add MSI status to slide table by merging with clinical
    slide_table = slide_table.merge(clinical_table[["PATIENT", "isMSIH"]], on="PATIENT", how="left")
    # Fill missing MSI with "Unknown"
    slide_table["isMSIH"] = slide_table["isMSIH"].fillna("Unknown")

    # Map processing locations to site for coloring
    # MSKCC cut/stain locations are OAUTHC patients
    def get_site_for_location(row, location_col):
        """Get site for coloring based on processing location."""
        location = row[location_col]
        if location == "MSKCC":
            return "OAUTHC"
        else:
            return row["SITE"]

    slide_table["cut_site"] = slide_table.apply(
        lambda x: get_site_for_location(x, "cut_location"), axis=1
    )
    slide_table["stain_site"] = slide_table.apply(
        lambda x: get_site_for_location(x, "stain_location"), axis=1
    )
    slide_table["image_site"] = slide_table["SITE"]  # Image location is always by actual site

    # Plot 1: Slides per Location (4 panels: cut, stain, image, MSI)
    logger.info("Creating slides per location plot...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # First 3 panels: processing locations colored by site
    for ax, col, site_col, title in zip(
        axes[:3],
        ["cut_location", "stain_location", "image_location"],
        ["cut_site", "stain_site", "image_site"],
        ["Cut Location", "Stain Location", "Image Location"],
    ):
        # Count slides per location, grouped by site
        counts = slide_table.groupby([col, site_col]).size().reset_index(name="count")

        sns.barplot(data=counts, x=col, y="count", hue=site_col, ax=ax, dodge=True)
        ax.set_title(f"Slides per {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Number of Slides", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    # 4th panel: MSI status colored by site
    msi_counts = slide_table.groupby(["isMSIH", "SITE"]).size().reset_index(name="count")
    sns.barplot(data=msi_counts, x="isMSIH", y="count", hue="SITE", ax=axes[3], dodge=True)
    axes[3].set_title("Slides per MSI Status", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("MSI Status", fontsize=11)
    axes[3].set_ylabel("Number of Slides", fontsize=11)
    axes[3].tick_params(axis="x", rotation=0)
    axes[3].legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    fig.suptitle("Slide Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "ingestion_slides_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  - Created ingestion_slides_distribution.png")

    # Plot 2: Patients per Location (4 panels: cut, stain, image, MSI)
    logger.info("Creating patients per location plot...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # First 3 panels: processing locations colored by site
    for ax, col, site_col, title in zip(
        axes[:3],
        ["cut_location", "stain_location", "image_location"],
        ["cut_site", "stain_site", "image_site"],
        ["Cut Location", "Stain Location", "Image Location"],
    ):
        # Get unique patient-location-site combinations
        unique_patients = slide_table[[col, site_col, "PATIENT"]].drop_duplicates()
        # Count unique patients per location and site
        patient_counts = unique_patients.groupby([col, site_col])["PATIENT"].nunique().reset_index()
        patient_counts.columns = [col, site_col, "count"]

        sns.barplot(data=patient_counts, x=col, y="count", hue=site_col, ax=ax, dodge=True)
        ax.set_title(f"Patients per {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Number of Patients", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    # 4th panel: MSI status colored by site
    # Get unique patient-MSI-site combinations
    unique_patient_msi = slide_table[["PATIENT", "isMSIH", "SITE"]].drop_duplicates()
    msi_patient_counts = (
        unique_patient_msi.groupby(["isMSIH", "SITE"])["PATIENT"].nunique().reset_index()
    )
    msi_patient_counts.columns = ["isMSIH", "SITE", "count"]

    sns.barplot(data=msi_patient_counts, x="isMSIH", y="count", hue="SITE", ax=axes[3], dodge=True)
    axes[3].set_title("Patients per MSI Status", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("MSI Status", fontsize=11)
    axes[3].set_ylabel("Number of Patients", fontsize=11)
    axes[3].tick_params(axis="x", rotation=0)
    axes[3].legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    fig.suptitle("Patient Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "ingestion_patients_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  - Created ingestion_patients_distribution.png")


def generate_ingestion_report(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    slide_table_full: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate markdown report with ingestion statistics.

    Args:
        clinical_table: Clinical data (one row per patient)
        slide_table: Slide data (cleaned, only existing slides)
        slide_table_full: Slide data before cleaning (includes missing slides)
        output_dir: Directory to save report (results/data/)
    """

    # Fix site labels in tables
    def fix_site_labels(df):
        df = df.copy()
        site_mapping = {
            "retrospective_msk": "OAUTHC",
            "retrospective_oau": "OAUTHC",
            "OAU": "OAUTHC",  # Consolidate any OAU to OAUTHC
        }
        df["SITE"] = df["SITE"].replace(site_mapping)
        return df

    slide_table = fix_site_labels(slide_table)
    slide_table_full = fix_site_labels(slide_table_full)

    # Calculate statistics
    n_patients = len(clinical_table)
    n_slides = len(slide_table)

    # Get full table counts for cleaning summary (before cleaning)
    # Read the saved clinical_table_full to get pre-cleaning count
    clinical_full_path = output_dir / "clinical_table_full.csv"
    if clinical_full_path.exists():
        clinical_full = pd.read_csv(clinical_full_path)
        n_patients_full = len(clinical_full)
    else:
        n_patients_full = n_patients

    n_slides_full = len(slide_table_full)

    # Count slides that exist
    if "slide_exists" in slide_table_full.columns:
        n_slides_found = slide_table_full["slide_exists"].sum()
        n_slides_missing = n_slides_full - n_slides_found
    else:
        n_slides_found = n_slides_full
        n_slides_missing = 0

    # Include Unknown MSI status for reporting
    clinical_report = clinical_table.copy()
    clinical_report["isMSIH"] = clinical_report["isMSIH"].fillna("Unknown")
    msi_counts = clinical_report["isMSIH"].value_counts()

    slides_per_patient = slide_table.groupby("PATIENT").size()
    mean_slides = slides_per_patient.mean()
    median_slides = slides_per_patient.median()
    max_slides = slides_per_patient.max()
    n_multi = (slides_per_patient > 1).sum()

    # Missing slides
    if "slide_exists" in slide_table_full.columns:
        missing_by_site = slide_table_full.groupby("SITE")["slide_exists"].agg(["count", "sum"])
        missing_by_site["missing"] = missing_by_site["count"] - missing_by_site["sum"]
        missing_by_site["pct_found"] = missing_by_site["sum"] / missing_by_site["count"] * 100
    else:
        missing_by_site = None

    # Calculate patients removed
    n_patients_removed = n_patients_full - n_patients

    # Generate markdown
    report = f"""# Data Ingestion Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

- **Total Patients (final):** {n_patients}
- **Total Slides (final):** {n_slides}

### MSI Status Distribution

- **MSI-H Patients:** {msi_counts.get("MSI-H", 0)} ({msi_counts.get("MSI-H", 0) / n_patients * 100:.1f}%)
- **MSS Patients:** {msi_counts.get("MSS", 0)} ({msi_counts.get("MSS", 0) / n_patients * 100:.1f}%)
- **Unknown MSI:** {msi_counts.get("Unknown", 0)} ({msi_counts.get("Unknown", 0) / n_patients * 100:.1f}%)

## Data Cleaning Summary

The cleaning step removes patients without MSI status and slides that don't exist on disk:

### Patients
- **Initial (from REDCap):** {n_patients_full} patients
- **Removed (no MSI status or no slides):** {n_patients_removed} patients
- **Final (ready for ML):** {n_patients} patients

### Slides
- **Initial (from Halo Link):** {n_slides_full} slides
- **Missing (not found on disk):** {n_slides_missing} slides
- **Removed (patients without MSI):** {n_slides_found - n_slides} slides
- **Final (ready for feature extraction):** {n_slides} slides

## Slides per Patient

- **Mean:** {mean_slides:.2f}
- **Median:** {median_slides:.0f}
- **Max:** {max_slides}
- **Patients with >1 slide:** {n_multi} ({n_multi / n_patients * 100:.1f}%)

## Patients with Multiple Processing Locations

"""

    # Multi-processing patients
    patient_stains = slide_table.groupby("PATIENT")["stain_location"].nunique()
    multi_stain = patient_stains[patient_stains > 1]
    report += f"- **Patients with multiple staining locations:** {len(multi_stain)}\n"

    if len(multi_stain) > 0:
        example = multi_stain.index[0]
        locs = slide_table[slide_table["PATIENT"] == example]["stain_location"].unique()
        report += f"  - Example: Patient {example} has slides stained at {', '.join(locs)}\n"

    report += "\n## Missing Slides\n\n"

    if missing_by_site is not None:
        report += "| Site | Total Slides | Found | Missing | Found % |\n"
        report += "|------|--------------|-------|---------|---------|\\n"
        for site, row in missing_by_site.iterrows():
            report += f"| {site} | {int(row['count'])} | {int(row['sum'])} | {int(row['missing'])} | {row['pct_found']:.1f}% |\n"

        total_missing = missing_by_site["missing"].sum()
        total_slides = missing_by_site["count"].sum()
        pct_missing = (total_missing / total_slides * 100) if total_slides > 0 else 0
        report += f"\n**Total missing:** {int(total_missing)} slides ({pct_missing:.1f}%)\n"
    else:
        report += "*No slide verification data available.*\n"

    # Processing location summary
    report += "\n## Processing Location Summary\n\n"

    report += "### Cut Locations\n"
    cut_counts = slide_table["cut_location"].value_counts()
    for loc, count in cut_counts.items():
        report += f"- {loc}: {count} slides\n"

    report += "\n### Stain Locations\n"
    stain_counts = slide_table["stain_location"].value_counts()
    for loc, count in stain_counts.items():
        report += f"- {loc}: {count} slides\n"

    report += "\n### Image Locations\n"
    image_counts = slide_table["image_location"].value_counts()
    for loc, count in image_counts.items():
        report += f"- {loc}: {count} slides\n"

    # MSI status by site
    report += "\n## MSI Status by Site\n\n"
    # Merge slide table with clinical to get MSI per site
    slide_with_msi = slide_table.merge(
        clinical_report[["PATIENT", "isMSIH"]], on="PATIENT", how="left"
    )
    slide_with_msi["isMSIH"] = slide_with_msi["isMSIH"].fillna("Unknown")

    # Get unique patients per site
    site_patient_msi = slide_with_msi[["SITE", "PATIENT", "isMSIH"]].drop_duplicates()

    report += "| Site | MSI-H | MSS | Unknown | Total |\n"
    report += "|------|-------|-----|---------|-------|\n"

    for site in sorted(site_patient_msi["SITE"].unique()):
        site_data = site_patient_msi[site_patient_msi["SITE"] == site]
        msih = (site_data["isMSIH"] == "MSI-H").sum()
        mss = (site_data["isMSIH"] == "MSS").sum()
        unknown = (site_data["isMSIH"] == "Unknown").sum()
        total = len(site_data)
        report += f"| {site} | {msih} | {mss} | {unknown} | {total} |\n"

    # Save report
    report_path = output_dir / "ingestion_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved ingestion report: {report_path}")


def process_redcap_data(
    output_dir: Optional[Path] = None,
    api_url: Optional[str] = None,
    api_token: Optional[str] = None,
    halo_base_dir: Optional[Path] = None,
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
        from .io_utils import get_results_dir

        output_dir = get_results_dir() / "data"
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    # Step 1: Fetch REDCap data
    logger.info("Fetching data from REDCap...")
    redcap_data = fetch_redcap_data(api_url, api_token)

    # Step 2: Create clinical table
    logger.info("Extracting clinical information...")
    clinical_table, record_id_mapping = create_clinical_table(redcap_data)
    clinical_table.to_csv(output_dir / "clinical_table_full.csv", index=False)
    logger.info(f"Saved clinical table with {len(clinical_table)} patients")

    # Step 3: Load Halo Link data
    logger.info("Loading Halo Link data...")
    halo_data = load_halo_link_data(halo_base_dir)

    # Step 4: Create slide table
    logger.info("Creating slide table...")
    slide_table = create_slide_table(halo_data, record_id_mapping)

    # Step 5: Verify slides exist
    logger.info("Verifying slides exist...")
    slide_table = verify_slides_exist(slide_table)

    # Save full table with slide_exists column for diagnostics (BEFORE dropping columns)
    slide_table.to_csv(output_dir / "slide_table_full.csv", index=False)

    # Update FILENAME to absolute path and clean for final table
    slide_table["FILENAME"] = slide_table["slide_path"]
    slide_table = slide_table.drop(columns=["slide_path", "slide_exists"])
    logger.info(f"Verified {len(slide_table)} slides")

    # Step 6: Clean tables
    logger.info("Cleaning tables...")
    clinical_table, slide_table = clean_tables(clinical_table, slide_table)

    # Save final cleaned tables
    clinical_table.to_csv(output_dir / "clinical_table.csv", index=False)
    slide_table.to_csv(output_dir / "slide_table.csv", index=False)

    # Step 7: Generate diagnostic plots and report
    logger.info("Generating diagnostic visualizations...")
    try:
        # Read slide_table_full that was saved earlier (has slide_exists column)
        slide_table_full = pd.read_csv(output_dir / "slide_table_full.csv")

        plot_ingestion_diagnostics(
            clinical_table=clinical_table,
            slide_table=slide_table,
            slide_table_full=slide_table_full,
            output_dir=output_dir,
        )
        generate_ingestion_report(
            clinical_table=clinical_table,
            slide_table=slide_table,
            slide_table_full=slide_table_full,
            output_dir=output_dir,
        )
        logger.info("Diagnostic plots and report saved")
    except Exception as e:
        logger.warning(f"Failed to generate diagnostics: {e}")
        # Continue execution even if plotting fails

    logger.info("Data ingestion complete")
    logger.info(f"  - Clinical table: {len(clinical_table)} patients")
    logger.info(f"  - Slide table: {len(slide_table)} slides")

    return clinical_table, slide_table
