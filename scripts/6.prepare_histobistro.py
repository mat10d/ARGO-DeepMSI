import pandas as pd

# Load your current clinical table
clinical_df = pd.read_csv("../tables/2/all_clinical_table.csv")

# Convert MSI status values
clinical_df['isMSIH'] = clinical_df['isMSIH'].replace({
    'MSS': 'nonMSIH',
    'MSI-H': 'MSIH'
})

# Save the updated table
clinical_df.to_csv("../tables/2/all_clinical_table_histobistro.csv", index=False)