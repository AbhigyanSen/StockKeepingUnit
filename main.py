import pandas as pd
import re
import os

# ---------- File Paths ----------
BASE_DIR = r"D:\Projects\SKU\StockKeepingUnit\Data"

# take only filename (after DataCSV)
transaction_filename = "trans_sep-22.csv"                                       # CHANGE when switching files
transaction_path = os.path.join(BASE_DIR, "DataCSV", transaction_filename)

# output folder inside Data
output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

# auto-generate output filename (trans_xxx.csv → matches_xxx.csv)
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

# ---------- Load Original Data ----------
master_orig = pd.read_csv(master_path).fillna("")
transaction_orig = pd.read_csv(transaction_path).fillna("")

# ---------- Define Columns to Drop ----------
master_colToDrop = [
    'itemcode', 'category', 'subcat', 'ssubcat', 'mbrand', 'multipack',
    'flavor', 'color', 'hpkcnv', 'msu', 'launchdate', 'status',
    'factcode', 'activeitem', 'active', 'nepalcomm', 'flag',
    'audittype', 'price_seg', 'filter_seg', 'DELYM'
]
transaction_colToDrop = [
    'DATE', 'PERIOD', 'AUDITYPE', 'STORECODE', 'DLRCODE', 'ITEMCODE',
    'NEW_CODES', 'COMMENTS', 'IMAGE', 'CODE COMMENT', 'FLAG'
]

# ---------- Cleaned Copies for Matching ----------
master = master_orig.drop(columns=[c for c in master_colToDrop if c in master_orig.columns])
transaction = transaction_orig.drop(columns=[c for c in transaction_colToDrop if c in transaction_orig.columns])

# ---------- Utility: Parse Packsize ----------
def parse_packsize(val):
    if not isinstance(val, str):
        val = str(val)
    val = val.strip().upper()
    if val.isdigit():
        return val, ""
    m = re.match(r"(\d+)\s*([A-Z]*)", val)
    if m:
        return m.group(1), m.group(2) if m.group(2) else ""
    return val, ""

# ---------- Utility: Scoring ----------
def score_and_rank_matches(transaction_row, potential_matches):
    if potential_matches.empty:
        return None

    potential_matches = potential_matches.copy()
    potential_matches['score'] = 0.0

    # MRP
    mrp_match = (potential_matches['mrp'].astype(str) == str(transaction_row['MRP']))
    potential_matches.loc[mrp_match, 'score'] += 10.0

    # BRAND
    t_brand = str(transaction_row['BRAND']).strip().upper()
    def brand_score(master_brand):
        mb = str(master_brand).strip().upper()
        if mb == t_brand:
            return 5.0
        elif t_brand in mb or mb in t_brand:
            return 3.0
        return 0.0
    potential_matches['brand_score'] = potential_matches['brand'].apply(brand_score)
    potential_matches['score'] += potential_matches['brand_score']

    # ITEMDESC similarity
    t_words = set(re.sub(r'[^A-Z0-9\s]', '', str(transaction_row['ITEMDESC']).strip().upper()).split())
    if t_words:
        potential_matches["concat_str"] = (
            potential_matches["company"].astype(str).str.strip().str.upper() + " " +
            potential_matches["brand"].astype(str).str.strip().str.upper() + " " +
            potential_matches["packtype"].astype(str).str.strip().str.upper() + " " +
            potential_matches["qty"].astype(str).str.strip().str.replace(".0", "", regex=False)
        )
        potential_matches['itemdesc_score'] = potential_matches.apply(
            lambda row: len(t_words.intersection(set(str(row['concat_str']).split()))) / len(t_words),
            axis=1
        )
        potential_matches['score'] += potential_matches['itemdesc_score'] * 10

    return potential_matches.sort_values(by='score', ascending=False).iloc[0]

# ---------- Matching ----------
results = []
for t_idx, t_row in transaction.iterrows():
    print(f"🔄 Processing Transaction Row {t_idx+1}: {t_row['ITEMDESC']}")
    t_orig_row = transaction_orig.iloc[t_idx]   # get original transaction row
    
    A = master[master["catcode"].astype(str).str.upper() == str(t_row["CATEGORY"]).upper()]
    if A.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["CATCODE", ""])
        continue

    B = A[A["company"].astype(str).str.upper() == str(t_row["MANUFACTURE"]).upper()]
    if B.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["MANUFACTURE", ""])
        continue

    brand_upper = str(t_row["BRAND"]).upper()
    C = B[B["brand"].astype(str).str.upper() == brand_upper]
    if C.empty:
        C = B[B["brand"].astype(str).str.upper().apply(
            lambda mb: brand_upper in mb or mb in brand_upper
        )]

    if not C.empty:
        D = C[C["packtype"].astype(str).str.upper() == str(t_row["PACKTYPE"]).upper()]
        if not D.empty:
            t_qty, t_uom = parse_packsize(t_row["PACKSIZE"])
            E = D[D["qty"].astype(float) == float(t_qty)]
            if not E.empty:
                best_match = score_and_rank_matches(t_row, E)
                if best_match is not None:
                    m_idx = best_match.name
                    m_orig_row = master_orig.iloc[m_idx]   # original master row
                    results.append(list(t_orig_row) + list(m_orig_row) + ["", best_match['score']])
                    continue

    # Fallback
    D_fallback = B[B["packtype"].astype(str).str.upper() == str(t_row["PACKTYPE"]).upper()]
    if D_fallback.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["PACKTYPE", ""])
        continue

    t_qty_f, t_uom_f = parse_packsize(t_row["PACKSIZE"])
    E_fallback = D_fallback[D_fallback["qty"].astype(float) == float(t_qty_f)]
    if E_fallback.empty:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["PACKSIZE", ""])
        continue

    best_match_fallback = score_and_rank_matches(t_row, E_fallback)
    if best_match_fallback is not None and best_match_fallback['score'] > 0:
        m_idx = best_match_fallback.name
        m_orig_row = master_orig.iloc[m_idx]
        results.append(list(t_orig_row) + list(m_orig_row) + ["", best_match_fallback['score']])
    else:
        results.append(list(t_orig_row) + [""] * len(master_orig.columns) + ["ITEMDESC", ""])

# ---------- Save Results ----------
columns = list(transaction_orig.columns) + list(master_orig.columns) + ["ERROR", "SCORE"]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Matching complete. Results saved to {output_path}")