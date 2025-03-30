import os
import requests
import pandas as pd
import glob
from dotenv import load_dotenv
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ignore SettingWithCopyWarning
warnings.filterwarnings('ignore')

def fetch_redcap_data(api_url=None, api_token=None):
    """Fetch data from REDCap API."""
    if api_url is None or api_token is None:
        # Load environment variables
        load_dotenv()
        api_token = os.getenv('REDCAP_API_TOKEN')
        api_url = os.getenv('REDCAP_API_URL')
    payload = {
        'token': api_token,
        'content': 'record',
        'format': 'json',
        'type': 'flat',
        'rawOrLabel': 'raw',
        'rawOrLabelHeaders': 'raw',
        'exportDataAccessGroups': 'true'
    }
    response = requests.post(api_url, data=payload)
    if response.status_code != 200:
        raise Exception(f"Error accessing REDCap API: {response.text}")
    
    data = response.json()
    df = pd.DataFrame(data)
    df.replace('', pd.NA, inplace=True)
    return df

def create_cinical_table(redcap_data):
    """Create a clinical table with patient IDs and MSI status.
    
    Returns:
        DataFrame with record_id (as PATIENT) and MSI status (isMSIH)
    """
    # Initialize the clinical table with proper columns
    clinical_table = pd.DataFrame(columns=['PATIENT', 'isMSIH'])
    
    # Check for prospective data (batch_number is not 1 or 2)
    prospective_data = redcap_data[
        (redcap_data['batch_number'] != '1') & 
        (redcap_data['batch_number'] != '2')
    ].copy()
    # Check for retrospective data (batch_number is 1 or 2)
    retrospective_data = redcap_data[
        (redcap_data['batch_number'] == '1') | 
        (redcap_data['batch_number'] == '2')
    ].copy()
    
    if not prospective_data.empty:
        # Process prospective data
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
    
    if not retrospective_data.empty:
        # Process retrospective data
        retrospective_msi = retrospective_data[['record_id', 'msi_status_mmr', 'batch_number', 'redcap_data_access_group']].copy()
        retrospective_msi['isMSIH'] = retrospective_msi['msi_status_mmr'].map({'1': 'MSI-H', '2': 'MSS'})
        retrospective_msi.rename(columns={'record_id': 'PATIENT'}, inplace=True)
        retrospective_msi = retrospective_msi[['PATIENT', 'isMSIH', 'batch_number', 'redcap_data_access_group']]
        clinical_table = pd.concat([clinical_table, retrospective_msi], ignore_index=True)
    
    # Ensure PATIENT column is string
    clinical_table['PATIENT'] = clinical_table['PATIENT'].astype(str)
    
    return clinical_table

def load_halo_link_data(base_dir):
    """Load all Halo Link data files from the data directory."""
    halo_files = glob.glob(os.path.join(base_dir, 'halo_link_*.csv'))
    
    if not halo_files:
        print("Warning: No Halo Link files found in data directory")
        return pd.DataFrame()
    
    # Initialize an empty list to store dataframes
    halo_dfs = []
    
    for file in halo_files:
        site_name = os.path.basename(file).replace('halo_link_', '').replace('_export.csv', '')
        try:
            df = pd.read_csv(file)
            df['site'] = site_name
            halo_dfs.append(df)
            print(f"Loaded halo data for {site_name} ({len(df)} records)")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine all halo data
    if not halo_dfs:
        return pd.DataFrame()
    
    combined_halo = pd.concat(halo_dfs, ignore_index=True)
    
    # Standardize column names based on the sample data
    standard_col_map = {
        'Slide ID': 'slide_id',
        'Study Image ID': 'image_id',
        'Name': 'filename',
        'Image Location': 'image_location',
        'Pathology REDCap ID': 'redcap_id'
    }
    
    combined_halo.rename(columns={k: v for k, v in standard_col_map.items() if k in combined_halo.columns}, 
                         inplace=True)
    
    return combined_halo

def create_slide_table(halo_data):
    """Create a slide table relating patients to their slide files.
    
    Args:
        halo_data: DataFrame with Halo Link data
    
    Returns:
        DataFrame with PATIENT and FILENAME columns
    """
    # Initialize the slide table
    slide_table = pd.DataFrame(columns=['PATIENT', 'FILENAME'])
    
    # Check if we have the necessary columns
    if 'redcap_id' not in halo_data.columns or 'filename' not in halo_data.columns:
        print("Warning: Missing required columns in Halo data for slide table")
        if 'redcap_id' not in halo_data.columns:
            print("- Missing 'redcap_id' column (Pathology REDCap ID)")
        if 'filename' not in halo_data.columns:
            print("- Missing 'filename' column (Name)")
        
        # Try to find alternative columns
        patient_id_cols = [col for col in halo_data.columns if 'id' in col.lower() and 'redcap' in col.lower()]
        filename_cols = [col for col in halo_data.columns if 'name' in col.lower() or 'file' in col.lower()]
        
        if patient_id_cols and filename_cols:
            print(f"Using alternative columns: {patient_id_cols[0]} and {filename_cols[0]}")
            temp_df = halo_data[[patient_id_cols[0], filename_cols[0]]].copy()
            temp_df.columns = ['PATIENT', 'FILENAME']
            slide_table = pd.concat([slide_table, temp_df], ignore_index=True)
        else:
            return slide_table
    else:
        # Extract the relevant columns and rename them
        temp_df = halo_data[['redcap_id', 'filename', 'site']].copy()
        temp_df.columns = ['PATIENT', 'FILENAME', 'SITE']
        slide_table = pd.concat([slide_table, temp_df], ignore_index=True)
    
    # Drop rows with missing values
    slide_table = slide_table.dropna(subset=['PATIENT', 'FILENAME'])
    
    # Ensure PATIENT column is string
    slide_table['PATIENT'] = slide_table['PATIENT'].astype(str)
    
    return slide_table

def verify_slides_exist(slide_table, slide_dirs=None):
    """Verify if slides exist in the specified directories or their subdirectories.
    
    Args:
        slide_table: DataFrame with PATIENT and FILENAME columns
        slide_dirs: List of directories to search for slides (if None, use common locations)
    
    Returns:
        Updated slide table with existence status and absolute paths
    """
    # Make a copy to avoid modifying the original
    result = slide_table.copy()
    
    # Add columns for slide existence and path
    result['slide_exists'] = False
    result['slide_path'] = None
    
    # If no directories specified, try common locations
    if not slide_dirs:
        slide_dirs = [
            '/lab/barcheese01/mdiberna/ARGO-DeepMSI/data',
            'data',
            'slides'
        ]
    
    print(f"Starting slide verification in {len(slide_dirs)} base directories and their subdirectories")
    
    # Create a dictionary to store all found files for faster lookup
    found_files = {}
    
    # Find all slide files in all directories and subdirectories
    for base_dir in slide_dirs:
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory {base_dir} does not exist")
            continue
            
        print(f"Scanning directory: {base_dir}")
        for root, _, files in os.walk(base_dir):
            for file in files:
                # Store the absolute path for each file
                absolute_path = os.path.abspath(os.path.join(root, file))
                found_files[file] = absolute_path
                # Also add version without .svs extension for easier matching
                if file.endswith('.svs'):
                    found_files[file[:-4]] = absolute_path
    
    print(f"Found {len(found_files)} unique filenames in all directories")
    
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
    
    # Print summary
    print(f"Slide verification complete: {found_count} found, {len(result) - found_count} not found")
    
    # Print some examples of found slides with their paths
    found_slides = result[result['slide_exists'] == True]
    if not found_slides.empty:
        print(f"\nExamples of found slides (showing up to 3):")
        for _, row in found_slides.head(3).iterrows():
            print(f"  - {row['FILENAME']} → {row['slide_path']}")
    
    # Print some examples of missing slides if any
    missing_slides = result[result['slide_exists'] == False]
    if not missing_slides.empty:
        print(f"\nExamples of missing slides (showing up to 3):")
        for _, row in missing_slides.head(3).iterrows():
            print(f"  - {row['FILENAME']} (Patient {row['PATIENT']})")
    
    return result

def clean_tables(clinical_table, slide_table):
    """Clean up tables to ensure consistency.
    
    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT and FILENAME columns
    
    Returns:
        Tuple of cleaned (clinical_table, slide_table)
    """
    # Make copies to avoid modifying originals
    clinical = clinical_table.copy()
    slides = slide_table.copy()
    
    print("\nCleaning data tables...")
    print(f"Initial counts: {len(clinical)} patients, {len(slides)} slides")
    
    # 1. Remove patients with missing MSI status from clinical table
    initial_clinical_count = len(clinical)
    clinical = clinical.dropna(subset=['isMSIH'])
    dropped_clinical = initial_clinical_count - len(clinical)
    if dropped_clinical > 0:
        print(f"Removed {dropped_clinical} patients with missing MSI status")
    
    # 2. Get list of valid patients (those with MSI status)
    valid_patients = set(clinical['PATIENT'].unique())
    
    # 3. Remove slides for patients without MSI status from slide table
    initial_slide_count = len(slides)
    slides = slides[slides['PATIENT'].isin(valid_patients)]
    dropped_slides = initial_slide_count - len(slides)
    if dropped_slides > 0:
        print(f"Removed {dropped_slides} slides for patients without MSI status")
    
    # 4. Update clinical table to include only patients with slides
    patients_with_slides = set(slides['PATIENT'].unique())
    initial_clinical_count = len(clinical)
    clinical = clinical[clinical['PATIENT'].isin(patients_with_slides)]
    dropped_clinical = initial_clinical_count - len(clinical)
    if dropped_clinical > 0:
        print(f"Removed {dropped_clinical} patients without slides")
    
    # 5. Final summary
    print(f"Final counts: {len(clinical)} patients, {len(slides)} slides")
    
    # 6. Check MSI distribution
    if not clinical.empty:
        msi_counts = clinical['isMSIH'].value_counts()
        for status, count in msi_counts.items():
            print(f"  - {status}: {count} patients ({count/len(clinical)*100:.1f}%)")
    
    return clinical, slides

def generate_histograms(clinical_table, slide_table):
    """Generate histograms showing slide count and patient count by site and MSI status.
    
    Args:
        clinical_table: DataFrame with PATIENT and isMSIH columns
        slide_table: DataFrame with PATIENT, FILENAME, and SITE columns
    """
    # Create directory for visualizations
    os.makedirs('../visualizations', exist_ok=True)
    
    # Make sure we have the SITE column in the slide table
    if 'SITE' not in slide_table.columns:
        print("ERROR: SITE column not found in slide_table. Cannot generate site-based histograms.")
        return
    
    # Merge clinical and slide tables to get MSI status for each slide
    merged_data = pd.merge(slide_table, clinical_table, on='PATIENT', how='inner')
    
    # Set up the visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'MSI-H': '#FF5733', 'MSS': '#3366FF'}
    
    print("\nGenerating visualizations...")
    
    # Plot 1: Patient count by site and MSI status
    plt.figure(figsize=(12, 6))
    
    # Count unique patients per site and MSI status
    patient_counts = merged_data.groupby(['SITE', 'isMSIH'])['PATIENT'].nunique().reset_index()
    patient_counts = patient_counts.rename(columns={'PATIENT': 'Count'})
    
    # Create a pivot table for easier plotting
    patient_pivot = patient_counts.pivot(index='SITE', columns='isMSIH', values='Count').fillna(0)
    
    # Plot the stacked bar chart
    patient_pivot.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
    
    plt.title('Patient Count by Site and MSI Status', fontsize=16)
    plt.xlabel('Site', fontsize=14)
    plt.ylabel('Number of Patients', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='MSI Status')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('../visualizations/patient_count_by_site.png', dpi=300, bbox_inches='tight')
    print("Saved patient count histogram to ../visualizations/patient_count_by_site.png")
    
    # Plot 2: Slide count by site and MSI status
    plt.figure(figsize=(12, 6))
    
    # Count slides per site and MSI status
    slide_counts = merged_data.groupby(['SITE', 'isMSIH']).size().reset_index(name='Count')
    
    # Create a pivot table for easier plotting
    slide_pivot = slide_counts.pivot(index='SITE', columns='isMSIH', values='Count').fillna(0)
    
    # Plot the stacked bar chart
    slide_pivot.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
    
    plt.title('Slide Count by Site and MSI Status', fontsize=16)
    plt.xlabel('Site', fontsize=14)
    plt.ylabel('Number of Slides', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='MSI Status')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('../visualizations/slide_count_by_site.png', dpi=300, bbox_inches='tight')
    print("Saved slide count histogram to ../visualizations/slide_count_by_site.png")
    
    # Plot 3: Distribution of slides per patient, grouped by MSI status
    plt.figure(figsize=(12, 6))
    
    # Count slides per patient
    slides_per_patient = merged_data.groupby(['PATIENT', 'isMSIH']).size().reset_index(name='SlideCount')
    
    # Create a box plot
    sns.boxplot(x='isMSIH', y='SlideCount', data=slides_per_patient, palette=colors)
    
    # Add individual data points for better visualization
    sns.stripplot(x='isMSIH', y='SlideCount', data=slides_per_patient, 
                 color='black', alpha=0.4, jitter=True)
    
    plt.title('Distribution of Slides per Patient by MSI Status', fontsize=16)
    plt.xlabel('MSI Status', fontsize=14)
    plt.ylabel('Number of Slides per Patient', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('../visualizations/slides_per_patient.png', dpi=300, bbox_inches='tight')
    print("Saved slides per patient distribution to ../visualizations/slides_per_patient.png")
    
    # Plot 4: MSI status distribution across all sites
    plt.figure(figsize=(10, 6))
    
    # Get the total counts of MSI-H and MSS patients
    msi_distribution = clinical_table['isMSIH'].value_counts()
    
    # Create a pie chart
    plt.pie(msi_distribution, labels=msi_distribution.index, autopct='%1.1f%%', 
            colors=[colors.get(status, '#999999') for status in msi_distribution.index], 
            startangle=90, explode=[0.05] * len(msi_distribution))
    
    plt.title('MSI Status Distribution Across All Sites', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the figure
    plt.savefig('../visualizations/overall_msi_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved overall MSI distribution to ../visualizations/overall_msi_distribution.png")
    
    print("\nAll visualizations generated successfully.")
    print("Files are saved in the 'visualizations' directory.")

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

# Create a main function to run the script
if __name__ == "__main__":
    os.makedirs('../tables', exist_ok=True)
    
    # Step 1: Fetch MSI data from REDCap
    print("Fetching data from REDCap...")
    redcap_data = fetch_redcap_data()
    
    # Step 2: Extract clinical information (MSI status)
    print("Extracting clinical information...")
    clinical_table = create_cinical_table(redcap_data)
    clinical_table.to_csv("../tables/clini_table_full.csv", index=False)
    print(f"Saved clinical table with {len(clinical_table)} patients")
    
    # Step 3: Load Halo Link data
    print("Loading Halo Link data...")
    halo_data = load_halo_link_data(base_dir='/lab/barcheese01/mdiberna/ARGO-DeepMSI/data')
    
    # Step 4: Create slide table
    print("Creating slide table...")
    slide_table = create_slide_table(halo_data)
    
    # Step 5: Verify slides exist and generate summary
    print("Verifying slides exist...")
    slide_table = verify_slides_exist(slide_table)
    slide_table['FILENAME'] = slide_table['slide_path']
    slide_table.drop(columns=['slide_path', 'slide_exists'], inplace=True)
    slide_table.to_csv("../tables/slide_table_full.csv", index=False)
    print(f"Saved slide table with {len(slide_table)} slides")
    
    print("\nSummary Report:")
    print(f"- Clinical table contains {len(clinical_table)} patients")
    print(f"- slide table contains {len(slide_table)} slides")
    
    # Step 6: Clean tables to ensure consistency
    print("Cleaning tables...")
    clinical_table, slide_table = clean_tables(clinical_table, slide_table)
    slide_table.to_csv("../tables/slide_table.csv", index=False)
    clinical_table.to_csv("../tables/clini_table.csv", index=False)
    
    # Step 7: Generate histograms and visualizations
    print("Generating histograms and visualizations...")
    generate_histograms(clinical_table, slide_table)
    
    # Step 8: Split tables by site
    print("Splitting tables by site...")
    split_tables, site = split_tables_by_site(clinical_table, slide_table)
    for site, (clinical, slides) in split_tables.items():
        clinical.to_csv(f"../tables/{site}_clinical_table.csv", index=False)
        slides.to_csv(f"../tables/{site}_slide_table.csv", index=False)
        print(f"Saved {site} tables with {len(clinical)} patients and {len(slides)} slides")

    print("Data processing and visualization complete.")