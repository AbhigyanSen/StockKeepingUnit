import pandas as pd

# Load the CSV
df = pd.read_csv(r"Data\master.csv")

# Step 1: Drop specified columns
cols_to_drop = ['itemcode', 'catcode', 'category', 'flavor', 'color', 'launchdate']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Step 2: Replace G -> GM and L -> LTR in uomdesc (case-sensitive)
df['uomdesc'] = df['uomdesc'].replace({'G': 'GM', 'L': 'LTR'})

# Step 3: Replace nulls or empty strings with 'NOT' in specified columns
for col in ['qty', 'uomdesc', 'pack_size']:
    if col in df.columns:
        df[col] = df[col].fillna('NOT')
        df[col] = df[col].replace('', 'NOT')

# Step 4: Remove .0 from qty column
df['qty'] = df['qty'].apply(lambda x: int(x) if pd.notnull(x) else x)

# Save cleaned file
df.to_csv("master_cleaned.csv", index=False)