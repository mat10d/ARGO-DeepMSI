import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def find_feature_file(slide_filename, site, features_base_dir):
    """Find feature file path for a given slide.
    
    Args:
        slide_filename: Slide filename (without path)
        site: Site name
        features_base_dir: Base directory for features
    
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
    
    # Look for feature directories (ctranspath or xiyuewang)
    feature_dirs = []
    for item in os.listdir(site_path):
        item_path = os.path.join(site_path, item)
        if os.path.isdir(item_path) and ('ctranspath' in item or 'xiyuewang' in item):
            feature_dirs.append(item_path)
    
    # Check each feature directory for the H5 file
    for feature_dir in feature_dirs:
        h5_path = os.path.join(feature_dir, f"{base_filename}.h5")
        if os.path.exists(h5_path):
            return h5_path
    
    return None


def add_feature_paths(slide_table, features_base_dir):
    """Add feature file paths to the slide table.
    
    Args:
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory for features
    
    Returns:
        Updated slide table with feature file paths
    """
    # Make a copy to avoid modifying the original
    result = slide_table.copy()
    
    # Add column for feature path
    result['FEATURE_PATH'] = None
    
    print(f"Adding feature paths to slide table...")
    feature_count = 0
    
    # Check each slide for associated feature file
    for idx, row in result.iterrows():
        filename = row['FILENAME']
        site = row['SITE']
        
        if pd.isna(filename) or not isinstance(filename, str) or pd.isna(site):
            continue
        
        # Find feature file
        feature_path = find_feature_file(filename, site, features_base_dir)
        if feature_path:
            result.at[idx, 'FEATURE_PATH'] = feature_path
            feature_count += 1
    
    # Print summary
    print(f"Feature path addition complete: {feature_count} features found, {len(result) - feature_count} not found")
    
    # Print some examples of found features with their paths
    found_features = result[result['FEATURE_PATH'].notna()]
    if not found_features.empty:
        print(f"\nExamples of found features (showing up to 3):")
        for _, row in found_features.head(3).iterrows():
            print(f"  - {os.path.basename(row['FILENAME'])} → {row['FEATURE_PATH']}")
    
    return result


def generate_processing_histograms(clinical_table, slide_table, features_base_dir):
    """Generate histograms showing processed vs. total slides by site and MSI status.
    
    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
        features_base_dir: Base directory where feature H5 files are stored
    
    Returns:
        Merged data with processing status
    """   
    # Make sure we have the SITE column in the slide table
    if 'SITE' not in slide_table.columns:
        print("ERROR: SITE column not found in slide_table. Cannot generate site-based histograms.")
        return None
    
    # Merge clinical and slide tables to get MSI status for each slide
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')
    
    # Add a column to indicate if the slide has been processed (H5 file exists)
    merged_data['processed'] = False
    
    # Check existence of H5 files for each slide
    print("Checking for processed slides (H5 files)...")
    processed_count = 0
    
    # Find all feature directories across all sites
    feature_dirs = {}
    for site in merged_data['SITE'].unique():
        site_path = os.path.join(features_base_dir, site, 'features')
        if os.path.exists(site_path):
            # Look for subdirectories containing extractor name (like ctranspath or xiyuewang)
            for item in os.listdir(site_path):
                item_path = os.path.join(site_path, item)
                if os.path.isdir(item_path) and ('ctranspath' in item or 'xiyuewang' in item):
                    feature_dirs[site] = item_path
                    print(f"Found feature directory for {site}: {item_path}")
                    break
    
    if not feature_dirs:
        print(f"WARNING: No feature directories found in subfolders of {features_base_dir}")
        return merged_data
    
    print(f"Found feature directories for {len(feature_dirs)} sites")
    
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
    
    print(f"Processing status: {processed_count} of {len(merged_data)} slides processed ({processed_count/len(merged_data)*100:.1f}%)")
    
    # Create missing slides list
    missing_slides = merged_data[merged_data['processed'] == False].copy()
    missing_slides = missing_slides[['PATIENT', 'SITE', 'FILENAME', 'isMSIH']]
    missing_slides.columns = ['PATIENT', 'SITE', 'FILENAME', 'MSI_STATUS']
    missing_slides.to_csv('../visualizations/2/missing_slides.csv', index=False)
    print(f"Saved list of {len(missing_slides)} missing slides to ../visualizations/2/missing_slides.csv")
    
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
    ax.set_title('Total vs. Processed Slides by Site', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(site_stats['SITE'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../visualizations/2/processing_status_by_site.png', dpi=300, bbox_inches='tight')
    print("Saved processing status by site to ../visualizations/2/processing_status_by_site.png")
       
    # Plot 2: Site-level MSI processing comparison
    # Prepare data
    site_msi_data = []
    for site in merged_data['SITE'].unique():
        site_data = merged_data[merged_data['SITE'] == site]
        
        # Get unique patients for this site
        site_patients = site_data['PATIENT'].nunique()
        
        for msi_status in site_data['isMSIH'].unique():
            msi_data = site_data[site_data['isMSIH'] == msi_status]
            total = len(msi_data)
            processed = msi_data['processed'].sum()
            
            # Count processed and total patients for this MSI status
            msi_patients = msi_data['PATIENT'].nunique()
            processed_patients = msi_data[msi_data['processed'] == True]['PATIENT'].nunique()
            
            site_msi_data.append({
                'Site': site,
                'MSI Status': msi_status,
                'Total': total,
                'Processed': processed,
                'Percentage': (processed / total * 100) if total > 0 else 0,
                'Total Patients': msi_patients,
                'Processed Patients': processed_patients
            })
    
    site_msi_df = pd.DataFrame(site_msi_data)
    
    # Filter to sites with at least 5 slides
    popular_sites = site_stats[site_stats['Total'] >= 5]['SITE'].tolist()
    site_msi_df = site_msi_df[site_msi_df['Site'].isin(popular_sites)]
    
    # Create two heatmaps: one for slides and one for patients
    
    # 1. Slide percentage heatmap
    plt.figure(figsize=(14, 10))
    
    # Create pivot table for percentage heatmap
    percentage_data = site_msi_df.pivot_table(
        index='Site', 
        columns='MSI Status', 
        values='Percentage',
        aggfunc='mean'
    ).fillna(0)
    
    # Sort by MSI-H percentage
    if 'MSI-H' in percentage_data.columns:
        percentage_data = percentage_data.sort_values('MSI-H', ascending=False)
    
    # Create annotation text with count information
    annot_data = np.empty_like(percentage_data, dtype=object)
    for i, site in enumerate(percentage_data.index):
        for j, status in enumerate(percentage_data.columns):
            # Get the corresponding row from site_msi_df
            row = site_msi_df[(site_msi_df['Site'] == site) & (site_msi_df['MSI Status'] == status)]
            if not row.empty:
                percentage = row['Percentage'].values[0]
                processed = row['Processed'].values[0]
                total = row['Total'].values[0]
                annot_data[i, j] = f"{percentage:.1f}%\n({processed}/{total})"
            else:
                annot_data[i, j] = "0.0%\n(0/0)"
    
    # Create heatmap with custom annotations
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(percentage_data, annot=annot_data, fmt="", cmap="YlGnBu", 
                cbar_kws={'label': 'Percentage Processed'}, ax=ax)
    
    plt.title('Percentage of Slides Processed by Site and MSI Status', fontsize=16)
    plt.tight_layout()
    
    plt.savefig('../visualizations/2/processing_heatmap_slides.png', dpi=300, bbox_inches='tight')
    print("Saved slides processing heatmap to ../visualizations/2/processing_heatmap_slides.png")
    
    # 2. Patient percentage heatmap
    plt.figure(figsize=(14, 10))
    
    # Calculate patient percentage
    site_msi_df['Patient Percentage'] = (site_msi_df['Processed Patients'] / site_msi_df['Total Patients'] * 100).round(1)
    
    # Create pivot table for patient percentage heatmap
    patient_percentage_data = site_msi_df.pivot_table(
        index='Site', 
        columns='MSI Status', 
        values='Patient Percentage',
        aggfunc='mean'
    ).fillna(0)
    
    # Sort by MSI-H percentage (same order as slide heatmap)
    if 'MSI-H' in patient_percentage_data.columns:
        patient_percentage_data = patient_percentage_data.reindex(percentage_data.index)
    
    # Create annotation text with patient count information
    patient_annot_data = np.empty_like(patient_percentage_data, dtype=object)
    for i, site in enumerate(patient_percentage_data.index):
        for j, status in enumerate(patient_percentage_data.columns):
            # Get the corresponding row from site_msi_df
            row = site_msi_df[(site_msi_df['Site'] == site) & (site_msi_df['MSI Status'] == status)]
            if not row.empty:
                percentage = row['Patient Percentage'].values[0]
                processed = row['Processed Patients'].values[0]
                total = row['Total Patients'].values[0]
                patient_annot_data[i, j] = f"{percentage:.1f}%\n({processed}/{total})"
            else:
                patient_annot_data[i, j] = "0.0%\n(0/0)"
    
    # Create heatmap with custom annotations
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(patient_percentage_data, annot=patient_annot_data, fmt="", cmap="YlGnBu", 
                cbar_kws={'label': 'Percentage of Patients with Processed Slides'}, ax=ax)
    
    plt.title('Percentage of Patients with Processed Slides by Site and MSI Status', fontsize=16)
    plt.tight_layout()
    
    plt.savefig('../visualizations/2/processing_heatmap_patients.png', dpi=300, bbox_inches='tight')
    print("Saved patients processing heatmap to ../visualizations/2/processing_heatmap_patients.png")
    
    # Plot 3: Patient-level processing status
    # Group by patient and get processing status
    patient_stats = merged_data.groupby(['PATIENT', 'isMSIH']).agg(
        processed_any=('processed', 'any'),
        processed_count=('processed', 'sum'),
        total_slides=('FILENAME', 'count'),
        site=('SITE', 'first')  # Keep track of the site for each patient
    ).reset_index()
    
    # Calculate percentage processed per patient
    patient_stats['percentage'] = (patient_stats['processed_count'] / patient_stats['total_slides'] * 100).round(1)
    
    # Create a summary by MSI status
    patient_summary = patient_stats.groupby('isMSIH').agg(
        total_patients=('PATIENT', 'count'),
        patients_with_processed_slides=('processed_any', 'sum'),
        percentage=('processed_any', lambda x: (x.sum() / len(x) * 100))
    ).reset_index()
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(patient_summary))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, patient_summary['total_patients'], width, label='Total Patients', color='#AAAAAA')
    bars2 = ax.bar(x + width/2, patient_summary['patients_with_processed_slides'], width, 
                  label='Patients with Processed Slides', color='#44BB99')
    
    # Add percentage labels
    for i, row in enumerate(patient_summary.itertuples()):
        ax.text(i + width/2, row.patients_with_processed_slides + 0.5, 
                f"{row.percentage:.1f}%", ha='center', va='bottom', fontsize=9)
        
        # Add count labels inside the bars
        ax.text(i - width/2, row.total_patients / 2, 
                f"{int(row.total_patients)}", ha='center', va='center', fontsize=10, color='black')
        ax.text(i + width/2, row.patients_with_processed_slides / 2, 
                f"{int(row.patients_with_processed_slides)}", ha='center', va='center', fontsize=10, color='white' if row.patients_with_processed_slides > 5 else 'black')
    
    ax.set_xlabel('MSI Status', fontsize=14)
    ax.set_ylabel('Number of Patients', fontsize=14)
    ax.set_title('Patients with Processed Slides by MSI Status', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(patient_summary['isMSIH'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../visualizations/2/patients_with_processed_slides.png', dpi=300, bbox_inches='tight')
    print("Saved patient processing status to ../visualizations/2/patients_with_processed_slides.png")
    
    # Create and save missing patient stats
    missing_patient_stats = patient_stats[patient_stats['processed_any'] == False].copy()
    missing_patient_stats = missing_patient_stats[['PATIENT', 'isMSIH', 'site', 'total_slides']]
    missing_patient_stats.columns = ['PATIENT', 'MSI_STATUS', 'SITE', 'SLIDE_COUNT']
    missing_patient_stats.to_csv('../visualizations/2/missing_patient_processing_stats.csv', index=False)
    print("Saved missing patient stats to ../visualizations/2/missing_patient_processing_stats.csv")
    
    # Calculate the number of missing patients by site and MSI status
    missing_by_site_msi = missing_patient_stats.groupby(['SITE', 'MSI_STATUS']).size().reset_index(name='MISSING_PATIENT_COUNT')
    missing_by_site_msi.to_csv('../visualizations/2/missing_patients_by_site.csv', index=False)
    print("Saved missing patients by site to ../visualizations/2/missing_patients_by_site.csv")
    
    # Print detailed summary
    print("\nProcessing Status Summary by Site:")
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


def split_tables_by_site(clinical_table, slide_table):
    """Split clinical and slide tables by site.
    
    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
    
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
    # Set base paths
    tables_dir = '../tables'  
    base_dir = '/lab/barcheese01/mdiberna/ARGO-DeepMSI/data'
    
    # Create output directories
    os.makedirs('../tables/2', exist_ok=True)
    os.makedirs('../visualizations/2', exist_ok=True)
    
    # Step 1: Read clinical and slide tables (both full and clean versions)
    print("Reading data tables...")
    clinical_path = os.path.join(tables_dir, '0/clinical_table.csv')
    slide_path = os.path.join(tables_dir, '0/slide_table.csv')
    full_clinical_path = os.path.join(tables_dir, '0/clinical_table_full.csv')
    full_slide_path = os.path.join(tables_dir, '0/slide_table_full.csv')
    
    # Check if required files exist
    required_files = {
        'Clinical table': clinical_path,
        'Slide table': slide_path,
        'Full clinical table': full_clinical_path,
        'Full slide table': full_slide_path
    }
    
    print(f"Reading clinical table from {clinical_path}")
    clinical_table = pd.read_csv(clinical_path)
    
    print(f"Reading slide table from {slide_path}")
    slide_table = pd.read_csv(slide_path)
    
    print(f"Reading full clinical table from {full_clinical_path}")
    full_clinical_table = pd.read_csv(full_clinical_path)
    
    print(f"Reading full slide table from {full_slide_path}")
    full_slide_table = pd.read_csv(full_slide_path)
    
    print(f"Found {len(clinical_table)} patients and {len(slide_table)} slides in clean tables")
    print(f"Found {len(full_clinical_table)} patients and {len(full_slide_table)} slides in full tables")
        
    # Step 2: Add feature paths to slide table
    print("\nStep 2: Adding feature paths to slide table...")
    slide_table_with_features = add_feature_paths(slide_table, base_dir)
    
    # Step 3: Generate processing status visualizations
    print("\nStep 3: Generating processing status visualizations...")
    merged_data = generate_processing_histograms(clinical_table, slide_table_with_features, base_dir)
    
    # Step 4: Split tables by site
    print("\nStep 4: Splitting tables by site...")
    split_tables, sites = split_tables_by_site(clinical_table, slide_table_with_features)
    
   # Step 5: Save processed data    
    print("\nStep 5: Saving processed data...")
    # Save the clinical table to tables/2/ folder first
    clinical_table_path = os.path.join(tables_dir, '2/all_clinical_table.csv')
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
    updated_slide_path = os.path.join(tables_dir, '2/all_slide_table.csv')
    all_slide_table.to_csv(updated_slide_path, index=False)
    print(f"Saved updated slide table with feature paths to {updated_slide_path}")

    # Now handle the site-specific tables
    for site, (site_clinical, site_slides) in split_tables.items():
        # Save clinical table
        site_clinical_path = os.path.join(tables_dir, f'2/{site}_clinical_table.csv')
        site_clinical.to_csv(site_clinical_path, index=False)
        
        # Prepare slide table by directly replacing FILENAME with FEATURE_PATH
        site_slides_with_features = site_slides.copy()
        # First, ensure there are no NaN values in FEATURE_PATH
        site_slides_with_features = site_slides_with_features.dropna(subset=['FEATURE_PATH'])
        # Now, replace FILENAME with FEATURE_PATH
        site_slides_with_features['FILENAME'] = site_slides_with_features['FEATURE_PATH']
        site_slides_with_features = site_slides_with_features.drop(columns=['FEATURE_PATH'])
        
        # Save slide table
        site_slide_path = os.path.join(tables_dir, f'2/{site}_slide_table.csv')
        site_slides_with_features.to_csv(site_slide_path, index=False)
        
        print(f"Saved {site} tables with {len(site_clinical)} patients and {len(site_slides_with_features)} slides")
            
    print("\nProcessing analysis complete!")