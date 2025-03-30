import os
import pandas as pd
from pathlib import Path

def load_tables():
    """Load all clinical and slide tables."""
    tables = {}
    tables_dir = Path("tables")
    
    for table_path in tables_dir.glob("*.csv"):
        name = table_path.stem
        tables[name] = pd.read_csv(table_path)
    
    return tables

def check_raw_files(slide_table, institution):
    """Check if all raw SVS files exist."""
    missing_files = []
    raw_dir = Path(f"data/{institution}/raw")
    
    for _, row in slide_table.iterrows():
        filename = row['FILENAME']
        filepath = raw_dir / f"{filename}.svs"
        
        if not filepath.exists():
            missing_files.append(filename)
    
    return missing_files

def check_processed_files(slide_table, institution):
    """Check if all processed H5 files exist."""
    missing_files = []
    h5_dir = Path(f"data/{institution}/processed/STAMP_macenko_xiyuewang-ctranspath-7c998680")
    
    for _, row in slide_table.iterrows():
        filename = row['FILENAME']
        filepath = h5_dir / f"{filename}.h5"
        
        if not filepath.exists():
            missing_files.append(filename)
    
    return missing_files

def check_stamp_logs(institution):
    """Check STAMP logs for processing errors."""
    errors = []
    cache_dir = Path(f"data/{institution}/cache")
    
    # Check typical STAMP error indicators in log files
    # This would need to be customized based on how STAMP logs errors
    log_file = cache_dir / "stamp.log"
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if "ERROR" in line or "Exception" in line:
                    errors.append(line.strip())
    
    return errors

def validate_institution(institution, clinical_table, slide_table):
    """Validate files and processing for one institution."""
    print(f"\nValidating {institution}...")
    
    # Check raw files
    missing_raw = check_raw_files(slide_table, institution)
    if missing_raw:
        print(f"Missing raw files ({len(missing_raw)}):")
        for filename in missing_raw:
            print(f"  - {filename}.svs")
    else:
        print("✓ All raw files present")
    
    # Check processed files
    missing_processed = check_processed_files(slide_table, institution)
    if missing_processed:
        print(f"Missing processed files ({len(missing_processed)}):")
        for filename in missing_processed:
            print(f"  - {filename}.h5")
    else:
        print("✓ All processed files present")
    
    # Check STAMP logs
    errors = check_stamp_logs(institution)
    if errors:
        print(f"STAMP processing errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ No STAMP processing errors")

def main():
    """Main validation function."""
    print("Starting validation...")
    
    # Load all tables
    tables = load_tables()
    
    # Validate prospective data
    prospective_sites = ['oauthc', 'luth', 'uith', 'lasuth']
    for site in prospective_sites:
        site_slides = tables['prospective_slide_table'][
            tables['prospective_slide_table']['site'] == site
        ]
        site_clinical = tables['prospective_clinical_table'][
            tables['prospective_clinical_table']['site'] == site
        ]
        validate_institution(site.upper(), site_clinical, site_slides)
    
    # Validate retrospective data
    validate_institution(
        'retrospective_msk',
        tables['retrospective_msk_clinical_table'],
        tables['retrospective_msk_slide_table']
    )
    
    validate_institution(
        'retrospective_oau',
        tables['retrospective_oau_clinical_table'],
        tables['retrospective_oau_slide_table']
    )

if __name__ == "__main__":
    main()