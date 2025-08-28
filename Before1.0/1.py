import pandas as pd
import re
import os

# ---------- File Paths ----------
BASE_DIR = r"D:\Projects\SKU\StockKeepingUnit\Data"

transaction_filename = "trans_dec-24.csv"     # CHANGE this when switching files
transaction_path = os.path.join(BASE_DIR, "DataCSV", transaction_filename)

output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

if transaction_filename.lower().startswith("trans_"):
    output_filename = transaction_filename.replace("trans_", "matches_", 1)
else:
    output_filename = "matches_" + transaction_filename

output_path = os.path.join(output_dir, output_filename)
master_path = os.path.join(BASE_DIR, "DataCSV", "master.csv")

print("📂 Master:", master_path)
print("📂 Transaction:", transaction_path)
print("📂 Output:", output_path)
print("")

# ---------- Load Data ----------
master_orig = pd.read_csv(master_path).fillna("")
transaction_orig = pd.read_csv(transaction_path).fillna("")

# ---------- Columns to Drop ----------
master_colToDrop = [
    'itemcode', 'category', 'subcat', 'ssubcat', 'multipack',
    'flavor', 'color', 'hpkcnv', 'msu', 'launchdate', 'status',
    'factcode', 'activeitem', 'active', 'nepalcomm', 'flag',
    'audittype', 'price_seg', 'filter_seg', 'DELYM'
]
transaction_colToDrop = [
    'DATE', 'PERIOD', 'AUDITYPE', 'STORECODE', 'DLRCODE', 'ITEMCODE',
    'NEW_CODES', 'COMMENTS', 'IMAGE', 'CODE COMMENT', 'FLAG'
]

master = master_orig.drop(columns=[c for c in master_colToDrop if c in master_orig.columns])
transaction = transaction_orig.drop(columns=[c for c in transaction_colToDrop if c in transaction_orig.columns])

# ✅ Debug: Print available columns
print("🔎 Master: \t", list(master.columns))
print("🔎 Transaction:\t", list(transaction.columns))
print("")

# ---------- Utility Functions ----------
def normalize_string(s):
    """Remove spaces, make uppercase, keep alphanumerics only."""
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())

def contains_match(value, df, column):
    """Return DataFrame rows where value is contained in df[column]."""
    value = str(value).upper().strip()
    mask = df[column].astype(str).str.upper().str.contains(value, na=False)
    return df[mask]

def parse_packsize(val):
    """Parse packsize into qty + uom if possible."""
    if not isinstance(val, str):
        val = str(val)
    val = val.strip().upper()
    m = re.match(r"(\d+)\s*([A-Z]*)", val)
    if m:
        return m.group(1), m.group(2)
    return val, ""

# ---------- Matching ----------
results = []
for t_idx, t_row in transaction.iterrows():
    print(f"🔄 Processing Transaction Row {t_idx+1}: {t_row['ITEMDESC']}")
    # print("🔎 Master Columns:", list(master.columns))
    # print("🔎 Transaction Columns:", list(transaction.columns))
    t_orig_row = transaction_orig.iloc[t_idx]

    # Step 1: CATEGORY -> catcode
    A = master[master["catcode"].astype(str).str.upper() == str(t_row["CATEGORY"]).upper()]
    if A.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["CATEGORY"])
        continue

    # Step 2: MANUFACTURE -> company
    B = contains_match(t_row["MANUFACTURE"], A, "company")
    if B.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["MANUFACTURE"])
        continue

    # Step 3: BRAND -> brand, fallback mbrand
    brand_val = str(t_row["BRAND"]).upper().strip()

    # Step 3.1: match BRAND against master.brand
    C = contains_match(brand_val, B, "brand")

    # Step 3.2: if no match OR multiple rows, also try mbrand
    if len(C) == 0:
        C = contains_match(brand_val, B["mbrand"])
    elif len(C) > 1:
        mbrand_matches = contains_match(brand_val, C, "mbrand")
        if len(mbrand_matches) > 0:
            C = mbrand_matches
    if C.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["BRAND"])
        continue

    # Step 4: PACKSIZE+PACKTYPE -> packtype+qty+uom
    t_qty, t_uom = parse_packsize(t_row["PACKSIZE"])
    trans_pack = normalize_string(f"{t_qty}{t_uom}{t_row['PACKTYPE']}")

    def make_master_pack(row):
        return normalize_string(f"{row['qty']}{row.get('uom','')}{row['packtype']}")

    C = B.copy()
    C.loc[:, "pack_combo"] = C.apply(make_master_pack, axis=1)
    D = C[C["pack_combo"].str.contains(trans_pack, na=False)]
    if D.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["PACKS"])
        continue

    # Step 5: ITEMDESC -> sku+packtype+qty+uom
    trans_itemdesc = normalize_string(t_row["ITEMDESC"])

    def make_master_item(row):
        return normalize_string(f"{row['sku']}{row['packtype']}{row['qty']}{row.get('uom','')}")

    D["item_combo"] = D.apply(make_master_item, axis=1)
    E = D[D["item_combo"].str.contains(trans_itemdesc, na=False)]
    if E.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["ITEM"])
        continue

    # Take first match (or highest priority)
    m_idx = E.index[0]
    m_orig_row = master_orig.iloc[m_idx]
    results.append(list(t_orig_row) + list(m_orig_row) + [""])

# ---------- Save Results ----------
columns = list(transaction_orig.columns) + list(master_orig.columns) + ["ERROR"]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Matching complete. Results saved to {output_path}")