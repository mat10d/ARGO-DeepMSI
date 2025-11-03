#!/usr/bin/env python3
"""
Preprocessing evaluation for H-optimus-0 features.
Updated to handle H-optimus-0 feature directories and save to h-optimus-0 specific folders.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def find_feature_file(slide_filename, site, features_base_dir, extractor_name="h-optimus-0"):
    """Find feature file path for a given slide.
    
    Args:
        slide_filename: Slide filename (without path)
        site: Site name
        features_base_dir: Base directory for features
        extractor_name: Name of the extractor (h-optimus-0, h-optimus-1, etc.)
    
    Returns:
        Full path to the feature file or None if not found
    """
    # Extract base filename without extension
    base_filename = os.path.basename(slide_filename)
    base_filename = os.path.splitext(base_filename)[0]
    
    # Check for feature directories
    site_path = os.path.join(features_base_dir, site, 'features')
    if not os.path.exists(site_path):
        return None
    
    # Look for feature directories matching the extractor
    feature_dirs = []
    for item in os.listdir(site_path):
        item_path = os.path.join(site_path, item)
        if os.path.isdir(item_path):
            # Check for H-optimus directories (including hash subdirectories)
            if extractor_name == "h-optimus-0" and ("h-optimus-0" in item or "h_optimus_0" in item):
                # Check if this directory has H5 files directly, or if we need to go into a subdirectory
                if any(f.endswith('.h5') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                    feature_dirs.append(item_path)
                else:
                    # Look for subdirectories (like hash directories)
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            if any(f.endswith('.h5') for f in os.listdir(subitem_path) if os.path.isfile(os.path.join(subitem_path, f))):
                                feature_dirs.append(subitem_path)
                                break
            elif extractor_name == "h-optimus-1" and ("h-optimus-1" in item or "h_optimus_1" in item):
                # Same logic for h-optimus-1
                if any(f.endswith('.h5') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                    feature_dirs.append(item_path)
                else:
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            if any(f.endswith('.h5') for f in os.listdir(subitem_path) if os.path.isfile(os.path.join(subitem_path, f))):
                                feature_dirs.append(subitem_path)
                                break
            # Also check for original ctranspath for comparison
            elif extractor_name == "ctranspath" and ('ctranspath' in item or 'xiyuewang' in item):
                feature_dirs.append(item_path)
    
    # Check each feature directory for the H5 file
    for feature_dir in feature_dirs:
        h5_path = os.path.join(feature_dir, f"{base_filename}.h5")
        if os.path.exists(h5_path):
            return h5_path
    
    return None


def add_feature_paths(slide_table, features_base_dir, extractor_name="h-optimus-0"):
    """Add feature file paths to the slide table.
    
    Args:
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory for features
        extractor_name: Name of the extractor to look for
    
    Returns:
        Updated slide table with feature file paths
    """
    # Make a copy to avoid modifying the original
    result = slide_table.copy()
    
    # Add column for feature path
    result['FEATURE_PATH'] = None
    
    print(f"Adding {extractor_name} feature paths to slide table...")
    feature_count = 0
    
    # Check each slide for associated feature file
    for idx, row in result.iterrows():
        filename = row['FILENAME']
        site = row['SITE']
        
        if pd.isna(filename) or not isinstance(filename, str) or pd.isna(site):
            continue
        
        # Find feature file
        feature_path = find_feature_file(filename, site, features_base_dir, extractor_name)
        if feature_path:
            result.at[idx, 'FEATURE_PATH'] = feature_path
            feature_count += 1
    
    # Print summary
    print(f"{extractor_name} feature path addition complete: {feature_count} features found, {len(result) - feature_count} not found")
    
    # Print some examples of found features with their paths
    found_features = result[result['FEATURE_PATH'].notna()]
    if not found_features.empty:
        print(f"\nExamples of found {extractor_name} features (showing up to 3):")
        for _, row in found_features.head(3).iterrows():
            print(f"  - {os.path.basename(row['FILENAME'])} → {row['FEATURE_PATH']}")
    
    return result


def generate_processing_histograms(clinical_table, slide_table, features_base_dir, extractor_name="h-optimus-0", output_dir="../visualizations/h-optimus-0"):
    """Generate histograms showing processed vs. total slides by site and MSI status.
    
    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory where feature H5 files are stored
        extractor_name: Name of the extractor
        output_dir: Directory to save visualizations
    
    Returns:
        Merged data with processing status
    """   
    # Make sure we have the SITE column in the slide table
    if 'SITE' not in slide_table.columns:
        print("ERROR: SITE column not found in slide_table. Cannot generate site-based histograms.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge clinical and slide tables to get MSI status for each slide
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')
    
    # Add a column to indicate if the slide has been processed (H5 file exists)
    merged_data['processed'] = False
    
    # Check existence of H5 files for each slide
    print(f"Checking for processed slides ({extractor_name} H5 files)...")
    processed_count = 0
    
    # Find all feature directories across all sites
    feature_dirs = {}
    for site in merged_data['SITE'].unique():
        site_path = os.path.join(features_base_dir, site, 'features')
        if os.path.exists(site_path):
            # Look for subdirectories containing the extractor name
            for item in os.listdir(site_path):
                item_path = os.path.join(site_path, item)
                if os.path.isdir(item_path):
                    # Check for the specific extractor
                    if extractor_name == "h-optimus-0" and ("h-optimus-0" in item or "h_optimus_0" in item):
                        # Check if H5 files are directly in this directory
                        if any(f.endswith('.h5') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                            feature_dirs[site] = item_path
                            print(f"Found {extractor_name} feature directory for {site}: {item_path}")
                            break
                        else:
                            # Look for subdirectories (like hash directories)
                            for subitem in os.listdir(item_path):
                                subitem_path = os.path.join(item_path, subitem)
                                if os.path.isdir(subitem_path):
                                    if any(f.endswith('.h5') for f in os.listdir(subitem_path) if os.path.isfile(os.path.join(subitem_path, f))):
                                        feature_dirs[site] = subitem_path
                                        print(f"Found {extractor_name} feature directory for {site}: {subitem_path}")
                                        break
                            if site in feature_dirs:
                                break
                    elif extractor_name == "h-optimus-1" and ("h-optimus-1" in item or "h_optimus_1" in item):
                        # Same logic for h-optimus-1
                        if any(f.endswith('.h5') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                            feature_dirs[site] = item_path
                            print(f"Found {extractor_name} feature directory for {site}: {item_path}")
                            break
                        else:
                            for subitem in os.listdir(item_path):
                                subitem_path = os.path.join(item_path, subitem)
                                if os.path.isdir(subitem_path):
                                    if any(f.endswith('.h5') for f in os.listdir(subitem_path) if os.path.isfile(os.path.join(subitem_path, f))):
                                        feature_dirs[site] = subitem_path
                                        print(f"Found {extractor_name} feature directory for {site}: {subitem_path}")
                                        break
                            if site in feature_dirs:
                                break
                    elif extractor_name == "ctranspath" and ('ctranspath' in item or 'xiyuewang' in item):
                        feature_dirs[site] = item_path
                        print(f"Found {extractor_name} feature directory for {site}: {item_path}")
                        break
    
    if not feature_dirs:
        print(f"WARNING: No {extractor_name} feature directories found in subfolders of {features_base_dir}")
        return merged_data
    
    print(f"Found {extractor_name} feature directories for {len(feature_dirs)} sites")
    
    # Check each slide for H5 files
    for idx, row in merged_data.iterrows():
        filename = row['FILENAME']
        if pd.isna(filename) or not isinstance(filename, str):
            continue
            
        # Extract just the filename without path or extension
        base_filename = os.path.basename(filename)
        base_filename = os.path.splitext(base_filename)[0]
        
        # Check for H5 file in the site's feature directory
        site = row['SITE']
        if site in feature_dirs:
            feature_dir = feature_dirs[site]
            h5_path = os.path.join(feature_dir, f"{base_filename}.h5")
            if os.path.exists(h5_path):
                merged_data.at[idx, 'processed'] = True
                processed_count += 1
    
    print(f"{extractor_name} processing status: {processed_count} of {len(merged_data)} slides processed ({processed_count/len(merged_data)*100:.1f}%)")
    
    # Create missing slides list
    missing_slides = merged_data[merged_data['processed'] == False].copy()
    missing_slides = missing_slides[['PATIENT', 'SITE', 'FILENAME', 'isMSIH']]
    missing_slides.columns = ['PATIENT', 'SITE', 'FILENAME', 'MSI_STATUS']
    missing_slides.to_csv(f'{output_dir}/missing_slides_{extractor_name.replace("-", "_")}.csv', index=False)
    print(f"Saved list of {len(missing_slides)} missing slides to {output_dir}/missing_slides_{extractor_name.replace('-', '_')}.csv")
    
    # Set up the visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Slides by site - total vs. processed
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data for plotting
    site_stats = merged_data.groupby('SITE').agg(
        Total=('FILENAME', 'count'),
        Processed=('processed', 'sum')
    ).reset_index()
    
    # Calculate processing percentage
    site_stats['Percentage'] = (site_stats['Processed'] / site_stats['Total'] * 100).round(1)
    
    # Sort by total count descending
    site_stats = site_stats.sort_values('Total', ascending=False)
    
    # Create grouped bar chart
    x = np.arange(len(site_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, site_stats['Total'], width, label='Total Slides', color='#AAAAAA')
    bars2 = ax.bar(x + width/2, site_stats['Processed'], width, label='Processed Slides', color='#44BB99')
    
    # Add percentage labels on top of processed bars
    for i, (_, row) in enumerate(site_stats.iterrows()):
        ax.text(i + width/2, row['Processed'] + 5, f"{row['Percentage']}%", 
                ha='center', va='bottom', fontsize=9)
    
    # Add labels, title and axis ticks
    ax.set_xlabel('Site', fontsize=14)
    ax.set_ylabel('Number of Slides', fontsize=14)
    ax.set_title(f'Total vs. Processed Slides by Site ({extractor_name})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(site_stats['SITE'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/processing_status_by_site_{extractor_name.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"Saved processing status by site to {output_dir}/processing_status_by_site_{extractor_name.replace('-', '_')}.png")
    
    # Print detailed summary
    print(f"\n{extractor_name} Processing Status Summary by Site:")
    print("-" * 60)
    print(f"{'Site':<15} {'MSI-H':<20} {'MSS':<20} {'Total':<15}")
    print("-" * 60)
    
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
        
        print(f"{site:<15} {msih_str:<20} {mss_str:<20} {total_str:<15}")
    
    print("-" * 60)
    total_processed = merged_data['processed'].sum()
    total_slides = len(merged_data)
    overall_pct = (total_processed / total_slides * 100) if total_slides > 0 else 0
    print(f"Overall: {total_processed}/{total_slides} ({overall_pct:.1f}%)")
    print("-" * 60)
    
    # Return the merged data with processing information
    return merged_data


def split_tables_by_site(clinical_table, slide_table, extractor_name="h-optimus-0"):
    """Split clinical and slide tables by site.
    
    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        extractor_name: Name of the extractor for output naming
    
    Returns:
        Dictionary with site names as keys and tuples of (clinical_table, slide_table) as values
    """
    # Make sure we have the SITE column in the slide table
    if 'SITE' not in slide_table.columns:
        print("ERROR: SITE column not found in slide_table. Cannot split tables by site.")
        return {}, []
    
    # Get list of unique sites
    sites = slide_table['SITE'].unique()
    print(f"Found {len(sites)} unique sites: {', '.join(sites)}")
    
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
        
        # Generate MSI distribution for this site
        msi_counts = site_clinical['isMSIH'].value_counts()
        
        print(f"\nSite: {site}")
        print(f"  - Patients: {len(site_clinical)}")
        print(f"  - Slides: {len(site_slides)}")
        for status, count in msi_counts.items():
            print(f"  - {status}: {count} patients ({count/len(site_clinical)*100:.1f}%)")
    
    return split_tables, sites


if __name__ == "__main__":
    # Configuration
    EXTRACTOR_NAME = "h-optimus-0"  # Change this to "h-optimus-1" when processing that model
    EXTRACTOR_CLEAN = EXTRACTOR_NAME.replace("-", "_")
    
    # Set base paths
    tables_dir = '../tables'  
    base_dir = '/lab/barcheese01/mdiberna/ARGO-DeepMSI/data'
    
    # Create output directories for this extractor
    output_tables_dir = f'../tables/{EXTRACTOR_CLEAN}'
    output_viz_dir = f'../visualizations/{EXTRACTOR_CLEAN}'
    os.makedirs(output_tables_dir, exist_ok=True)
    os.makedirs(output_viz_dir, exist_ok=True)
    
    # Step 1: Read clinical and slide tables
    print(f"Reading data tables for {EXTRACTOR_NAME} processing...")
    clinical_path = os.path.join(tables_dir, '0/clinical_table.csv')
    slide_path = os.path.join(tables_dir, '0/slide_table.csv')
    
    print(f"Reading clinical table from {clinical_path}")
    clinical_table = pd.read_csv(clinical_path)
    
    print(f"Reading slide table from {slide_path}")
    slide_table = pd.read_csv(slide_path)
    
    print(f"Found {len(clinical_table)} patients and {len(slide_table)} slides in tables")
        
    # Step 2: Add feature paths to slide table for the specific extractor
    print(f"\nStep 2: Adding {EXTRACTOR_NAME} feature paths to slide table...")
    slide_table_with_features = add_feature_paths(slide_table, base_dir, EXTRACTOR_NAME)
    
    # Step 3: Generate processing status visualizations
    print(f"\nStep 3: Generating {EXTRACTOR_NAME} processing status visualizations...")
    merged_data = generate_processing_histograms(clinical_table, slide_table_with_features, base_dir, EXTRACTOR_NAME, output_viz_dir)
    
    # Step 4: Split tables by site
    print(f"\nStep 4: Splitting tables by site for {EXTRACTOR_NAME}...")
    split_tables, sites = split_tables_by_site(clinical_table, slide_table_with_features, EXTRACTOR_NAME)
    
    # Step 5: Save processed data    
    print(f"\nStep 5: Saving {EXTRACTOR_NAME} processed data...")
    
    # Save the clinical table
    clinical_table_path = os.path.join(output_tables_dir, 'all_clinical_table.csv')
    clinical_table.to_csv(clinical_table_path, index=False)
    print(f"Saved clinical table to {clinical_table_path}")

    # For the all_slide_table, swap FILENAME and FEATURE_PATH
    all_slide_table = slide_table_with_features.copy()
    # First, ensure there are no NaN values in FEATURE_PATH
    all_slide_table = all_slide_table.dropna(subset=['FEATURE_PATH'])
    # Now, replace FILENAME with FEATURE_PATH
    all_slide_table['FILENAME'] = all_slide_table['FEATURE_PATH']
    all_slide_table = all_slide_table.drop(columns=['FEATURE_PATH'])

    # Save the updated slide table with feature paths
    updated_slide_path = os.path.join(output_tables_dir, 'all_slide_table.csv')
    all_slide_table.to_csv(updated_slide_path, index=False)
    print(f"Saved updated slide table with {EXTRACTOR_NAME} feature paths to {updated_slide_path}")

    # Now handle the site-specific tables
    for site, (site_clinical, site_slides) in split_tables.items():
        # Save clinical table
        site_clinical_path = os.path.join(output_tables_dir, f'{site}_clinical_table.csv')
        site_clinical.to_csv(site_clinical_path, index=False)
        
        # Prepare slide table by directly replacing FILENAME with FEATURE_PATH
        site_slides_with_features = site_slides.copy()
        # First, ensure there are no NaN values in FEATURE_PATH
        site_slides_with_features = site_slides_with_features.dropna(subset=['FEATURE_PATH'])
        # Now, replace FILENAME with FEATURE_PATH
        site_slides_with_features['FILENAME'] = site_slides_with_features['FEATURE_PATH']
        site_slides_with_features = site_slides_with_features.drop(columns=['FEATURE_PATH'])
        
        # Save slide table
        site_slide_path = os.path.join(output_tables_dir, f'{site}_slide_table.csv')
        site_slides_with_features.to_csv(site_slide_path, index=False)
        
        print(f"Saved {site} tables with {len(site_clinical)} patients and {len(site_slides_with_features)} slides")
    
    print(f"\n{EXTRACTOR_NAME} processing analysis complete!")
    print(f"Tables saved to: {output_tables_dir}")
    print(f"Visualizations saved to: {output_viz_dir}")
    print(f"\nNext steps:")
    print(f"1. Review processing status in {output_viz_dir}")
    print(f"2. Run cross-validation using the updated configs")
    print(f"3. Compare results with other extractors")