import pandas as pd

# Load your current clinical table
clinical_df = pd.read_csv("../tables/2/all_clinical_table.csv")

# Convert MSI status values
clinical_df['isMSIH'] = clinical_df['isMSIH'].replace({
    'MSS': 'nonMSIH',
    'MSI-H': 'MSIH'
})

# Save the updated table as an excel file
clinical_df.to_excel("../tables/2/all_clinical_table_histobistro.xlsx", index=False)

# Load your current slide table
slide_df = pd.read_csv("../tables/2/all_slide_table.csv")

# For the FILENAME column, remove strip all the path text and keep only the file name
slide_df['FILENAME'] = slide_df['FILENAME'].str.split('/').str[-1]
# Remove the .svs extension from the FILENAME column
slide_df['FILENAME'] = slide_df['FILENAME'].str.replace('.h5', '', regex=False)

# Save the updated table as a csv file
slide_df.to_csv("../tables/2/all_slide_table_histobistro.csv", index=False)