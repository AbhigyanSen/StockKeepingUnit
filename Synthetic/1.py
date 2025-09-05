import pandas as pd

# Read the original CSV
df = pd.read_csv(r"D:\Projects\SKU\StockKeepingUnit\FinalMatches\final_matches_sep-24.csv")

# Transaction columns
transaction_cols = [
    "t_NEW_CODES", "t_CATEGORY", "t_MANUFACTURE", "t_BRAND",
    "t_ITEMDESC", "t_MRP", "t_PACKSIZE", "t_PACKTYPE"
]

# Master columns
master_cols = [
    "t_NEW_CODES", "matched_itemcode", "m_catcode", "m_company", "m_mbrand",
    "m_brand", "m_sku", "m_packtype", "m_base_pack", "m_flavor", "m_color",
    "m_wght", "m_uom", "m_mrp"
]

# Create transaction and master DataFrames
transaction_df = df[transaction_cols].drop_duplicates()
master_df = df[master_cols].drop_duplicates()

# Save into separate CSV files
transaction_df.to_csv("transaction.csv", index=False)
master_df.to_csv("master.csv", index=False)

print("✅ transaction.csv and master.csv created successfully.")