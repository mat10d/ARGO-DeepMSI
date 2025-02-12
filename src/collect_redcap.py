import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import warnings

# Ignore SettingWithCopyWarning
warnings.filterwarnings('ignore')

def fetch_redcap_data():
    """Fetch data from REDCap API."""
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
    df['record_id'] = pd.to_numeric(df['record_id'], errors='coerce')
    return df

def process_retrospective(msi_data, institution):
    """Process retrospective data for a specific institution."""
    proforma_complete_index = list(msi_data.columns).index('proforma_complete')
    retrospective = msi_data.iloc[:, :proforma_complete_index]
    retrospective = retrospective[
        (retrospective['batch_number'] == '1') | 
        (retrospective['batch_number'] == '2')
    ]
    
    # Filter by institution
    if institution == 'msk':
        retrospective = retrospective[retrospective['batch_number'] == '1']
    elif institution == 'oau':
        retrospective = retrospective[retrospective['batch_number'] == '2']
    
    retrospective = retrospective.dropna(axis=1, how='all')
    retrospective['patient_id'] = 'crc_' + retrospective['crc_redcap_number']
    
    # Generate tables
    clinical_table = retrospective[['patient_id', 'ismsih']].copy()
    clinical_table['isMSIH'] = clinical_table['ismsih'].map({'1': 'MSI-H', '0': 'MSS'})
    clinical_table = clinical_table[['patient_id', 'isMSIH']]
    
    slide_table = retrospective[['patient_id', 'filename']].copy()
    slide_table = slide_table[slide_table['filename'].notna()]
    
    return clinical_table, slide_table

def process_prospective(msi_data):
    """Process prospective data."""
    proforma_complete_index = list(msi_data.columns).index('proforma_complete')
    prospective = msi_data.iloc[:, proforma_complete_index + 1:]
    prospective = pd.concat([
        msi_data['record_id'], 
        msi_data['batch_number'], 
        prospective
    ], axis=1)
    
    prospective = prospective[
        (prospective['batch_number'] != '1') & 
        (prospective['batch_number'] != '2')
    ]
    prospective = prospective.dropna(axis=1, how='all')
    prospective['patient_id'] = (
        prospective['r01_redcap_data_access_group'] + 
        '_' + 
        prospective['r01_record_id']
    )
    
    # Generate tables
    clinical_table = prospective[['patient_id', 'cmo_msi_status', 'r01_redcap_data_access_group']].copy()
    clinical_table['isMSIH'] = clinical_table['cmo_msi_status'].map({
        'Instable': 'MSI-H',
        'Stable': 'MSS',
        'Indeterminate': 'MSS',
        'Stable, Indeterminate': 'MSS'
    })
    clinical_table = clinical_table[['patient_id', 'isMSIH', 'r01_redcap_data_access_group']]
    clinical_table.rename(columns={'r01_redcap_data_access_group': 'site'}, inplace=True)
    
    slide_table = prospective[['patient_id', 'slide_name', 'r01_redcap_data_access_group']].copy()
    slide_table['slide_name'] = slide_table['slide_name'].str.split(', ')
    slide_table = slide_table.explode('slide_name')
    slide_table['slide_name'] = slide_table['slide_name'].str.rstrip('.svs')
    slide_table.rename(columns={
        'slide_name': 'FILENAME',
        'r01_redcap_data_access_group': 'site'
    }, inplace=True)
    
    return clinical_table, slide_table

def main():
    """Main function to process REDCap data and generate tables."""
    # Create tables directory if it doesn't exist
    os.makedirs('tables', exist_ok=True)
    
    # Fetch data
    print("Fetching data from REDCap...")
    msi_data = fetch_redcap_data()
    
    # Process retrospective data
    print("Processing retrospective data...")
    for institution in ['msk', 'oau']:
        clinical_table, slide_table = process_retrospective(msi_data, institution)
        
        # Save tables
        prefix = f"retrospective_{institution}"
        clinical_table.to_csv(f"tables/{prefix}_clinical_table.csv", index=False)
        slide_table.to_csv(f"tables/{prefix}_slide_table.csv", index=False)
        print(f"Saved {prefix} tables")
    
    # Process prospective data
    print("Processing prospective data...")
    clinical_table, slide_table = process_prospective(msi_data)
    
    # Save prospective tables
    clinical_table.to_csv("tables/prospective_clinical_table.csv", index=False)
    slide_table.to_csv("tables/prospective_slide_table.csv", index=False)
    print("Saved prospective tables")

if __name__ == "__main__":
    main()