import pandas as pd
import re
import os

# ---------- Load Data ----------
BASE_DIR = r"D:\Projects\SKU\StockKeepingUnit\Data\DataCleaned"

master = pd.read_csv(os.path.join(BASE_DIR, "master_cleaned.csv")).fillna("")
transactions = pd.read_csv(os.path.join(BASE_DIR, "transaction_cleaned.csv")).fillna("")

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

    # Separate numbers and alphabets
    m = re.match(r"(\d+)\s*([A-Z]*)", val)
    if m:
        qty = m.group(1)
        uom = m.group(2) if m.group(2) else ""
        return qty, uom
    return val, ""

# ---------- Utility: Step 6 (ITEMDESC matching) ----------
def match_itemdesc(E, t_itemdesc):
    """
    Matches ITEMDESC from transaction against concatenated string:
    company + brand + packtype + qty
    """
    E = E.copy()
    E["concat_str"] = (
        E["company"].astype(str).str.strip().str.upper() + " " +
        E["brand"].astype(str).str.strip().str.upper() + " " +
        E["packtype"].astype(str).str.strip().str.upper() + " " +
        E["qty"].astype(str).str.strip().str.upper()
    )
    F = E[E["concat_str"] == str(t_itemdesc).strip().upper()]
    return F

# ---------- Matching Logic ----------
results = []

for t_idx, t_row in transactions.iterrows():
    print(f"🔄 Processing Transaction Row {t_idx+1}: {t_row['ITEMDESC']}")

    # Step 1: CATEGORY
    A = master[master["catcode"].astype(str).str.strip().str.upper() ==
               str(t_row["CATEGORY"]).strip().upper()]
    if A.empty:
        results.append(list(t_row) + [""] * len(master.columns) + ["CATCODE"])
        continue

    # Step 2: MANUFACTURE
    B = A[A["company"].astype(str).str.strip().str.upper() ==
           str(t_row["MANUFACTURE"]).strip().upper()]
    if B.empty:
        results.append(list(t_row) + [""] * len(master.columns) + ["MANUFACTURE"])
        continue

    # Step 3: BRAND
    C = B[B["brand"].astype(str).str.strip().str.upper() ==
          str(t_row["BRAND"]).strip().upper()]

    if C.empty:
        # Step 3.1: PACKTYPE
        D = B[B["packtype"].astype(str).str.strip().str.upper() ==
               str(t_row["PACKTYPE"]).strip().upper()]
        if D.empty:
            results.append(list(t_row) + [""] * len(master.columns) + ["PACKTYPE"])
            continue

        # Step 3.2: PACKSIZE
        t_qty, t_uom = parse_packsize(t_row["PACKSIZE"])
        if t_uom == "":
            E = D[D["qty"].astype(str).str.replace(".0", "", regex=False) == t_qty]
        else:
            E = D[(D["qty"].astype(str).str.replace(".0", "", regex=False) == t_qty) &
                  (D["uom"].astype(str).str.strip().str.upper() == t_uom)]

        if E.empty:
            results.append(list(t_row) + [""] * len(master.columns) + ["PACKSIZE"])
            continue

        # Step 6: ITEMDESC
        F = match_itemdesc(E, t_row["ITEMDESC"])
        if F.empty:
            results.append(list(t_row) + [""] * len(master.columns) + ["ITEMDESC"])
            continue

        # Matches found
        for _, m_row in F.iterrows():
            results.append(list(t_row) + list(m_row.drop("concat_str")) + [""])
        continue

    # Step 4: PACKTYPE
    D = C[C["packtype"].astype(str).str.strip().str.upper() ==
           str(t_row["PACKTYPE"]).strip().upper()]
    if D.empty:
        results.append(list(t_row) + [""] * len(master.columns) + ["PACKTYPE"])
        continue

    # Step 5: PACKSIZE
    t_qty, t_uom = parse_packsize(t_row["PACKSIZE"])
    E = D[(D["qty"].astype(str).str.replace(".0", "", regex=False) == t_qty) &
          (D["uom"].astype(str).str.strip().str.upper() == t_uom)]
    if E.empty:
        results.append(list(t_row) + [""] * len(master.columns) + ["PACKSIZE"])
        continue

    # Step 6: ITEMDESC
    F = match_itemdesc(E, t_row["ITEMDESC"])
    if F.empty:
        results.append(list(t_row) + [""] * len(master.columns) + ["ITEMDESC"])
        continue

    # Matches found
    for _, m_row in F.iterrows():
        results.append(list(t_row) + list(m_row.drop("concat_str")) + [""])

# ---------- Save Results ----------
columns = list(transactions.columns) + list(master.columns) + ["ERROR"]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("matches.csv", index=False)
print("✅ Matching complete. Results saved to matches.csv.")