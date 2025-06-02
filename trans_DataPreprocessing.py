import pandas as pd
import re

# Load the CSV
df = pd.read_csv(r"Data\transaction.csv")

# Step 1: Drop specific columns (ensure exact match, ignore if not found)
columns_to_drop = ['PERIOD','AUDITTYPE','STORECODE','DLRCODE','ITEMCODE','CATEGORY','MRP','COMMENTS','IMAGE']

df = df.drop(columns=columns_to_drop, errors='ignore')

# Step 2: Drop existing QTY and UNIT if they already exist to avoid duplicates
for col in ['QTY', 'UNIT']:
    if col in df.columns:
        df = df.drop(columns=col)

# Step 3: Extract QTY and UNIT from PACKSIZE
def extract_qty_unit(packsize):
    if pd.isna(packsize) or not isinstance(packsize, str):
        return pd.Series([None, None])
    
    match = re.match(r'^\s*(\d+(?:\.\d+)?)([A-Za-z]+)', packsize)
    if match:
        qty = match.group(1)
        unit = match.group(2).upper()
        return pd.Series([qty, unit])
    else:
        return pd.Series([None, None])

df[['QTY', 'UNIT']] = df['PACKSIZE'].apply(extract_qty_unit)

# Step 4: Normalize UNIT values
df['UNIT'] = df['UNIT'].replace({'G': 'GM', 'L': 'LTR'})

# Step 5: Replace null/empty values in QTY, UNIT, PACKSIZE, PACKTYPE with '10000'
for col in ['QTY', 'UNIT', 'PACKSIZE', 'PACKTYPE']:
    df[col] = df[col].fillna('10000')
    df[col] = df[col].replace('', '10000')

# Step 6: Reorder to place QTY and UNIT just before PACKSIZE
cols = df.columns.tolist()
if 'PACKSIZE' in cols:
    # Remove QTY and UNIT if already present elsewhere
    cols = [c for c in cols if c not in ['QTY', 'UNIT']]
    packsize_index = cols.index('PACKSIZE')
    new_order = cols[:packsize_index] + ['QTY', 'UNIT'] + cols[packsize_index:]
    df = df[new_order]

# Save cleaned file
df.to_csv("transaction_cleaned.csv", index=False)