import pandas as pd
import re
import os

# ---------- Load Data ----------
BASE_DIR = r"D:\Projects\SKU\StockKeepingUnit\Data\DataCleaned"

master = pd.read_csv(os.path.join(BASE_DIR, "master_cleaned.csv")).fillna("")
transactions = pd.read_csv(os.path.join(BASE_DIR, "transaction_cleaned1.csv")).fillna("")

# ---------- Utility: Parse packsize ----------
def parse_packsize(val):
    """
    Splits packsize like '180 ML', '250ML', '20' → (qty, uom)
    qty: str
    uom: str
    """
    if not isinstance(val, str):
        val = str(val)
    val = val.strip().upper()

    # Handle cases with just a number
    if val.isdigit():
        return val, ""

    # Separate numbers and alphabets
    m = re.match(r"(\d+)\s*([A-Z]*)", val)
    if m:
        qty = m.group(1)
        uom = m.group(2) if m.group(2) else ""
        return qty, uom
    return val, ""

# ---------- Utility: Scoring and Matching ----------
def score_and_rank_matches(transaction_row, potential_matches):
    """
    Calculates a score for each potential match and returns the best one.
    """
    if potential_matches.empty:
        return None

    potential_matches = potential_matches.copy()
    
    # Initialize score column
    potential_matches['score'] = 0.0

    # Score based on MRP
    mrp_match = (potential_matches['mrp'].astype(str) == str(transaction_row['MRP']))
    potential_matches.loc[mrp_match, 'score'] += 10.0  # High score for MRP match

    # Score based on BRAND (exact vs fuzzy)
    t_brand = str(transaction_row['BRAND']).strip().upper()
    def brand_score(master_brand):
        mb = str(master_brand).strip().upper()
        if mb == t_brand:
            return 5.0  # exact match
        elif t_brand in mb or mb in t_brand:
            return 3.0  # fuzzy substring match
        else:
            return 0.0

    potential_matches['brand_score'] = potential_matches['brand'].apply(brand_score)
    potential_matches['score'] += potential_matches['brand_score']

    # Score based on ITEMDESC similarity
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

    # Return the row with the highest score
    best_match = potential_matches.sort_values(by='score', ascending=False).iloc[0]
    return best_match

# ---------- Matching Logic ----------
results = []
for t_idx, t_row in transactions.iterrows():
    print("="*80)
    print(f"🔄 Processing Transaction Row {t_idx+1}: {t_row['ITEMDESC']}")
    print("-"*80)

    # Step 1: CATEGORY
    A = master[master["catcode"].astype(str).str.strip().str.upper() == str(t_row["CATEGORY"]).strip().upper()]
    print(f"Step 1: CATEGORY filter -> kept {len(A)}, filtered out {len(master) - len(A)}")
    if A.empty:
        print("❌ No CATEGORY match. Filtered out rows:")
        print(master[~master.index.isin(A.index)])
        results.append(list(t_row) + [""] * len(master.columns) + ["CATCODE"] + [""])
        continue

    # Step 2: MANUFACTURE
    B = A[A["company"].astype(str).str.strip().str.upper() == str(t_row["MANUFACTURE"]).strip().upper()]
    print(f"Step 2: MANUFACTURE filter -> kept {len(B)}, filtered out {len(A) - len(B)}")
    if B.empty:
        print("❌ No MANUFACTURE match. Filtered out rows:")
        print(A[~A.index.isin(B.index)])
        results.append(list(t_row) + [""] * len(master.columns) + ["MANUFACTURE"] + [""])
        continue

    # Step 3: BRAND
    brand_upper = str(t_row["BRAND"]).strip().upper()
    C = B[B["brand"].astype(str).str.strip().str.upper() == brand_upper]  # exact match
    if C.empty:
        C = B[B["brand"].astype(str).str.strip().str.upper().apply(
            lambda mb: brand_upper in mb or mb in brand_upper
        )]
    print(f"Step 3: BRAND filter -> kept {len(C)}, filtered out {len(B) - len(C)}")
    if C.empty:
        print("❌ No BRAND match. Filtered out rows:")
        print(B[~B.index.isin(C.index)])

    # If Brand matches (exact or fuzzy), proceed with Packtype and Packsize
    if not C.empty:
        D = C[C["packtype"].astype(str).str.strip().str.upper() == str(t_row["PACKTYPE"]).strip().upper()]
        print(f"Step 4: PACKTYPE filter -> kept {len(D)}, filtered out {len(C) - len(D)}")
        if not D.empty:
            t_qty, t_uom = parse_packsize(t_row["PACKSIZE"])
            # E = D[
            #     (D["qty"].astype(float) == float(t_qty)) & 
            #     (D["uom"].astype(str).str.strip().str.upper() == t_uom)
            # ]
            E = D[D["qty"].astype(float) == float(t_qty)]
            print(f"Step 5: PACKSIZE filter -> kept {len(E)}, filtered out {len(D) - len(E)}")

            if not E.empty:
                best_match = score_and_rank_matches(t_row, E)
                if best_match is not None:
                    drop_cols = [c for c in ["concat_str", "score", "itemdesc_score", "brand_score"] if c in best_match.index]
                    results.append(list(t_row) + list(best_match.drop(drop_cols)) + [""] + [best_match['score']])
                    continue
    
    # ---------- Fallback ----------
    print("⚠️ Entering Fallback Matching...")

    # Fallback Step 3.1: PACKTYPE
    D_fallback = B[B["packtype"].astype(str).str.strip().str.upper() == str(t_row["PACKTYPE"]).strip().upper()]
    print(f"Fallback PACKTYPE filter -> kept {len(D_fallback)}, filtered out {len(B) - len(D_fallback)}")
    if D_fallback.empty:
        print("❌ No PACKTYPE match. Filtered out rows:")
        print(B[~B.index.isin(D_fallback.index)])
        results.append(list(t_row) + [""] * len(master.columns) + ["PACKTYPE"] + [""])
        continue

    # Fallback Step 3.2: PACKSIZE
    t_qty_f, t_uom_f = parse_packsize(t_row["PACKSIZE"])
    # E_fallback = D_fallback[
    #     (D_fallback["qty"].astype(float) == float(t_qty_f)) & 
    #     (D_fallback["uom"].astype(str).str.strip().str.upper() == t_uom_f)
    # ]
    E_fallback = D_fallback[D_fallback["qty"].astype(float) == float(t_qty_f)]
    print(f"Fallback PACKSIZE filter -> kept {len(E_fallback)}, filtered out {len(D_fallback) - len(E_fallback)}")
    if E_fallback.empty:
        print("❌ No PACKSIZE match. Filtered out rows:")
        print(D_fallback[~D_fallback.index.isin(E_fallback.index)])
        results.append(list(t_row) + [""] * len(master.columns) + ["PACKSIZE"] + [""])
        continue

    # Fallback Step 6: ITEMDESC
    best_match_fallback = score_and_rank_matches(t_row, E_fallback)
    if best_match_fallback is not None and best_match_fallback['score'] > 0:
        drop_cols = [c for c in ["concat_str", "score", "itemdesc_score", "brand_score"] if c in best_match_fallback.index]
        results.append(list(t_row) + list(best_match_fallback.drop(drop_cols)) + [""] + [best_match_fallback['score']])
    else:
        print("❌ ITEMDESC did not match any rows")
        results.append(list(t_row) + [""] * len(master.columns) + ["ITEMDESC"] + [""])
        
# ---------- Save Results ----------
columns = list(transactions.columns) + list(master.columns) + ["ERROR"] + ["SCORE"]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("demo_matches.csv", index=False)
print("✅ Matching complete. Results saved to demo_matches.csv.")