import pandas as pd

# Load the DataFrame (assuming 'df' is your DataFrame)
file_path = "/home/dcsadmin/Documents/DeleteSKU/master_data"
df = pd.read_csv(file_path, encoding='windows-1252')

# # Remove specified columns
# columns_to_remove = ['itemcode', 'catcode', 'packaging', 'flavour', 'colour', 'quantity', 'uomdesc', 'pack_size', 'launchdate', 'audittype', 'brand']
# df = df.drop(columns=columns_to_remove, errors='ignore')

# Remove the last 10000 rows
df = df[:-22000]

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)