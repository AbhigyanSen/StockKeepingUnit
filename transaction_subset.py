import pandas as pd

file_path = r'Data\DataCleaned\transaction_cleaned.csv'
df = pd.read_csv(file_path)

# Trim the DataFrame to 100 rows
trimmed_df = df.head(100)
trimmed_df.to_csv('transaction_subset.csv', index=False)
print("Completed")