"""
Visualization module for ARGO-DeepMSI pipeline.

Provides plotting functions for data ingestion, feature validation,
and results visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict
import logging

from .io_utils import ensure_dir


logger = logging.getLogger(__name__)

# Color scheme for MSI status
MSI_COLORS = {
    'MSI-H': '#FF5733',
    'MSS': '#3366FF'
}


def plot_patient_counts_by_site(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_path: Path,
    title: str = "Patient Count by Site and MSI Status"
) -> None:
    """Plot stacked bar chart of patient counts by site and MSI status.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        output_path: Path to save the figure
        title: Plot title
    """
    # Merge tables
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')

    # Count unique patients per site and MSI status
    patient_counts = merged_data.groupby(['SITE', 'isMSIH'])['PATIENT'].nunique().reset_index()
    patient_counts = patient_counts.rename(columns={'PATIENT': 'Count'})

    # Create pivot table
    patient_pivot = patient_counts.pivot(index='SITE', columns='isMSIH', values='Count').fillna(0)

    # Sort by total count
    patient_pivot['Total'] = patient_pivot.sum(axis=1)
    patient_pivot = patient_pivot.sort_values('Total', ascending=False)
    patient_pivot = patient_pivot.drop(columns=['Total'])

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    patient_pivot.plot(kind='bar', stacked=True, color=MSI_COLORS, ax=ax)

    # Add count labels
    for i, (site, row) in enumerate(patient_pivot.iterrows()):
        total = row.sum()
        cumulative = 0
        for status, count in row.items():
            if count > 0:
                label_pos = cumulative + (count / 2)
                ax.text(i, label_pos, f"{int(count)}",
                       ha='center', va='center', color='white')
                cumulative += count

        # Add total on top
        ax.text(i, total + 0.5, f"Total: {int(total)}", ha='center', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel('Site', fontsize=14)
    plt.ylabel('Number of Patients', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='MSI Status')
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved patient count plot to {output_path}")


def plot_slide_counts_by_site(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_path: Path,
    title: str = "Slide Count by Site and MSI Status"
) -> None:
    """Plot stacked bar chart of slide counts by site and MSI status.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        output_path: Path to save the figure
        title: Plot title
    """
    # Merge tables
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')

    # Count slides per site and MSI status
    slide_counts = merged_data.groupby(['SITE', 'isMSIH']).size().reset_index(name='Count')

    # Create pivot table
    slide_pivot = slide_counts.pivot(index='SITE', columns='isMSIH', values='Count').fillna(0)

    # Sort by total count
    slide_pivot['Total'] = slide_pivot.sum(axis=1)
    slide_pivot = slide_pivot.sort_values('Total', ascending=False)
    slide_pivot = slide_pivot.drop(columns=['Total'])

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    slide_pivot.plot(kind='bar', stacked=True, color=MSI_COLORS, ax=ax)

    # Add count labels
    for i, (site, row) in enumerate(slide_pivot.iterrows()):
        total = row.sum()
        cumulative = 0
        for status, count in row.items():
            if count > 0:
                label_pos = cumulative + (count / 2)
                ax.text(i, label_pos, f"{int(count)}",
                       ha='center', va='center', color='white')
                cumulative += count

        # Add total on top
        ax.text(i, total + 0.5, f"Total: {int(total)}", ha='center', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel('Site', fontsize=14)
    plt.ylabel('Number of Slides', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='MSI Status')
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved slide count plot to {output_path}")


def plot_slides_per_patient_distribution(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_path: Path,
    title: str = "Distribution of Slides per Patient by MSI Status"
) -> None:
    """Plot box plot of slides per patient distribution.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        output_path: Path to save the figure
        title: Plot title
    """
    # Merge tables
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')

    # Count slides per patient
    slides_per_patient = merged_data.groupby(['PATIENT', 'isMSIH']).size().reset_index(name='SlideCount')

    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='isMSIH', y='SlideCount', data=slides_per_patient, palette=MSI_COLORS, ax=ax)
    sns.stripplot(x='isMSIH', y='SlideCount', data=slides_per_patient,
                 color='black', alpha=0.4, jitter=True, ax=ax)

    plt.title(title, fontsize=16)
    plt.xlabel('MSI Status', fontsize=14)
    plt.ylabel('Number of Slides per Patient', fontsize=14)
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved slides per patient distribution to {output_path}")


def plot_msi_distribution(
    clinical_table: pd.DataFrame,
    output_path: Path,
    title: str = "MSI Status Distribution Across All Sites"
) -> None:
    """Plot pie chart of MSI status distribution.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        output_path: Path to save the figure
        title: Plot title
    """
    # Get MSI distribution
    msi_distribution = clinical_table['isMSIH'].value_counts()

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [MSI_COLORS.get(status, '#999999') for status in msi_distribution.index]
    ax.pie(msi_distribution, labels=msi_distribution.index, autopct='%1.1f%%',
           colors=colors, startangle=90, explode=[0.05] * len(msi_distribution))

    plt.title(title, fontsize=16)
    plt.axis('equal')

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MSI distribution to {output_path}")


def generate_patient_summary_by_site(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_path: Path
) -> pd.DataFrame:
    """Generate patient summary statistics by site.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        output_path: Path to save the CSV

    Returns:
        DataFrame with summary statistics by site
    """
    # Merge tables
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')

    # Generate summary for each site
    patient_summary = []
    for site in merged_data['SITE'].unique():
        site_data = merged_data[merged_data['SITE'] == site]
        site_patients = site_data['PATIENT'].nunique()

        # MSI-H stats
        msih_patients = site_data[site_data['isMSIH'] == 'MSI-H']['PATIENT'].nunique()
        msih_pct = (msih_patients / site_patients * 100) if site_patients > 0 else 0

        # MSS stats
        mss_patients = site_data[site_data['isMSIH'] == 'MSS']['PATIENT'].nunique()
        mss_pct = (mss_patients / site_patients * 100) if site_patients > 0 else 0

        # Slide counts
        site_slides = len(site_data)
        msih_slides = len(site_data[site_data['isMSIH'] == 'MSI-H'])
        mss_slides = len(site_data[site_data['isMSIH'] == 'MSS'])

        # Average slides per patient
        avg_slides = site_slides / site_patients if site_patients > 0 else 0

        patient_summary.append({
            'Site': site,
            'Total Patients': site_patients,
            'MSI-H Patients': msih_patients,
            'MSI-H %': msih_pct,
            'MSS Patients': mss_patients,
            'MSS %': mss_pct,
            'Total Slides': site_slides,
            'MSI-H Slides': msih_slides,
            'MSS Slides': mss_slides,
            'Avg Slides/Patient': avg_slides
        })

    # Create DataFrame
    patient_summary_df = pd.DataFrame(patient_summary)

    ensure_dir(output_path.parent)
    patient_summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved patient summary to {output_path}")

    return patient_summary_df


def plot_processing_status_by_site(
    merged_data: pd.DataFrame,
    output_path: Path,
    title: str = "Total vs. Processed Slides by Site"
) -> None:
    """Plot processing status (total vs processed slides) by site.

    Args:
        merged_data: DataFrame with 'SITE', 'FILENAME', and 'processed' columns
        output_path: Path to save the figure
        title: Plot title
    """
    # Prepare data
    site_stats = merged_data.groupby('SITE').agg(
        Total=('FILENAME', 'count'),
        Processed=('processed', 'sum')
    ).reset_index()

    # Calculate percentage
    site_stats['Percentage'] = (site_stats['Processed'] / site_stats['Total'] * 100).round(1)
    site_stats = site_stats.sort_values('Total', ascending=False)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(site_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, site_stats['Total'], width, label='Total Slides', color='#AAAAAA')
    bars2 = ax.bar(x + width/2, site_stats['Processed'], width, label='Processed Slides', color='#44BB99')

    # Add percentage labels
    for i, (_, row) in enumerate(site_stats.iterrows()):
        ax.text(i + width/2, row['Processed'] + 5, f"{row['Percentage']}%",
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Site', fontsize=14)
    ax.set_ylabel('Number of Slides', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(site_stats['SITE'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved processing status plot to {output_path}")


def plot_processing_heatmap(
    merged_data: pd.DataFrame,
    output_path: Path,
    mode: str = "slides",
    title: Optional[str] = None
) -> None:
    """Plot heatmap of processing status by site and MSI status.

    Args:
        merged_data: DataFrame with processing status information
        output_path: Path to save the figure
        mode: "slides" or "patients" - what to count
        title: Plot title (if None, uses default based on mode)
    """
    if mode not in ["slides", "patients"]:
        raise ValueError("mode must be 'slides' or 'patients'")

    # Prepare data based on mode
    if mode == "slides":
        site_msi_data = []
        for site in merged_data['SITE'].unique():
            site_data = merged_data[merged_data['SITE'] == site]
            for msi_status in site_data['isMSIH'].unique():
                msi_data = site_data[site_data['isMSIH'] == msi_status]
                total = len(msi_data)
                processed = msi_data['processed'].sum()

                site_msi_data.append({
                    'Site': site,
                    'MSI Status': msi_status,
                    'Total': total,
                    'Processed': processed,
                    'Percentage': (processed / total * 100) if total > 0 else 0
                })

        value_col = 'Percentage'
        default_title = "Percentage of Slides Processed by Site and MSI Status"

    else:  # patients
        site_msi_data = []
        for site in merged_data['SITE'].unique():
            site_data = merged_data[merged_data['SITE'] == site]
            for msi_status in site_data['isMSIH'].unique():
                msi_data = site_data[site_data['isMSIH'] == msi_status]
                total_patients = msi_data['PATIENT'].nunique()
                processed_patients = msi_data[msi_data['processed'] == True]['PATIENT'].nunique()

                site_msi_data.append({
                    'Site': site,
                    'MSI Status': msi_status,
                    'Total': total_patients,
                    'Processed': processed_patients,
                    'Percentage': (processed_patients / total_patients * 100) if total_patients > 0 else 0
                })

        value_col = 'Percentage'
        default_title = "Percentage of Patients with Processed Slides by Site and MSI Status"

    site_msi_df = pd.DataFrame(site_msi_data)

    # Create pivot table for heatmap
    heatmap_data = site_msi_df.pivot_table(
        index='Site',
        columns='MSI Status',
        values=value_col,
        aggfunc='mean'
    ).fillna(0)

    # Sort by MSI-H percentage
    if 'MSI-H' in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values('MSI-H', ascending=False)

    # Create annotation text
    annot_data = np.empty_like(heatmap_data, dtype=object)
    for i, site in enumerate(heatmap_data.index):
        for j, status in enumerate(heatmap_data.columns):
            row = site_msi_df[(site_msi_df['Site'] == site) & (site_msi_df['MSI Status'] == status)]
            if not row.empty:
                percentage = row['Percentage'].values[0]
                processed = row['Processed'].values[0]
                total = row['Total'].values[0]
                annot_data[i, j] = f"{percentage:.1f}%\n({int(processed)}/{int(total)})"
            else:
                annot_data[i, j] = "0.0%\n(0/0)"

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap="YlGnBu",
                cbar_kws={'label': 'Percentage Processed'}, ax=ax)

    plt.title(title or default_title, fontsize=16)
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved processing heatmap to {output_path}")


def generate_data_ingestion_visualizations(
    clinical_table: pd.DataFrame,
    slide_table: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate all data ingestion visualizations.

    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        output_dir: Directory to save visualizations
    """
    logger.info("Generating data ingestion visualizations...")

    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Generate all plots
    plot_patient_counts_by_site(clinical_table, slide_table, output_dir / "patient_count_by_site.png")
    plot_slide_counts_by_site(clinical_table, slide_table, output_dir / "slide_count_by_site.png")
    plot_slides_per_patient_distribution(clinical_table, slide_table, output_dir / "slides_per_patient.png")
    plot_msi_distribution(clinical_table, output_dir / "overall_msi_distribution.png")
    generate_patient_summary_by_site(clinical_table, slide_table, output_dir / "patient_summary_by_site.csv")

    logger.info(f"All visualizations saved to {output_dir}")


def generate_feature_validation_visualizations(
    merged_data: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate all feature validation visualizations.

    Args:
        merged_data: DataFrame with processing status (must have 'processed' column)
        output_dir: Directory to save visualizations
    """
    logger.info("Generating feature validation visualizations...")

    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Generate plots
    plot_processing_status_by_site(merged_data, output_dir / "processing_status_by_site.png")
    plot_processing_heatmap(merged_data, output_dir / "processing_heatmap_slides.png", mode="slides")
    plot_processing_heatmap(merged_data, output_dir / "processing_heatmap_patients.png", mode="patients")

    logger.info(f"All visualizations saved to {output_dir}")
