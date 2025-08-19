import pandas as pd
import requests
import re
import time
from difflib import SequenceMatcher

# Load Data
master = pd.read_csv("/home/dcsadmin/Documents/del_SKU/StockKeepingUnit/Data/DataCleaned/master_cleaned.csv").fillna("")
transactions = pd.read_csv("/home/dcsadmin/Documents/del_SKU/StockKeepingUnit/Data/DataCleaned/transaction_cleaned.csv").fillna("")

PACKTYPE_EQUIVALENCE = {
    "CDB": ["CDB", "CDBOX"]
}

# ----- UTILITIES -----

def extract_product_name_ollama(description, model="mistral"):
    prompt = (
        f"Extract only the product name from the following description. "
        f"Do not include quantity, unit, or pack type:\n\n{description}\n\nProduct name:"
    )
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False})
        if response.status_code == 200:
            raw = response.json()["response"].strip()
            cleaned = re.sub(r"\(.*?\)", "", raw).strip()
            return cleaned
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def split_item_desc(desc):
    tokens = re.split(r"(FREE|SAVE|DISCOUNT|RS\s*\d+)", desc, flags=re.IGNORECASE)
    if len(tokens) > 1:
        main = tokens[0].strip()
        offer = ' '.join(tokens[1:]).strip()
        return main, offer
    return desc.strip(), ""

def similar_str(a, b, threshold=0.85):
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def packsize_match(p1, p2):
    return re.sub(r'\.0+', '', p1.replace(" ", "").upper()) == re.sub(r'\.0+', '', p2.replace(" ", "").upper())

# ----- MAIN PROCESSING -----

results = []

for idx, row in transactions.iterrows():
    # Collect transaction row values in lowercase keys
    input_vals = {
        "manufacture": str(row['MANUFACTURE']).strip(),
        "brand": str(row['BRAND']).strip(),
        "category": str(row['CATEGORY']).strip(),
        "itemdesc": str(row['ITEMDESC']).strip(),
        "qty": row['qty'],
        "uomdesc": str(row['uomdesc']).strip(),
        "packsize": str(row['PACKSIZE']).strip(),
        "packtype": str(row['PACKTYPE']).strip()
    }

    # 1. Match by MANUFACTURE
    A = master[master['company'].str.strip().str.upper() == input_vals["manufacture"].upper()]
    if A.empty:
        results.append(list(row) + [1] + [''] * len(master.columns) + ['MANUFACTURE', ''])
        continue

    # 2. Match by BRAND
    B = A[A['brand'].str.strip().str.upper() == input_vals["brand"].upper()]
    if B.empty:
        results.append(list(row) + [1] + [''] * len(master.columns) + ['BRAND', ''])
        continue

    # 3. Match by CATEGORY → catcode
    C = B[B['catcode'].astype(str).str.strip().str.upper() == input_vals["category"].upper()]
    if C.empty:
        results.append(list(row) + [1] + [''] * len(master.columns) + ['CATEGORY', ''])
        continue

    # 4. Match by qty
    D = C[C['qty'] == input_vals["qty"]]
    if not D.empty:
        E = D[D['uomdesc'].str.strip().str.upper() == input_vals["uomdesc"].upper()]
        if not E.empty:
            F = E
        else:
            F = D[D['pack_size'].apply(lambda x: packsize_match(str(x), input_vals["packsize"]))]
    else:
        F = C[C['pack_size'].apply(lambda x: packsize_match(str(x), input_vals["packsize"]))]

    if F.empty:
        results.append(list(row) + [1] + [''] * len(master.columns) + ['PACKSIZE', ''])
        continue

    # 5. Match by PACKTYPE
    match_packtypes = PACKTYPE_EQUIVALENCE.get(input_vals["packtype"], [input_vals["packtype"]])
    G = F[F['packaging'].isin(match_packtypes)]
    if G.empty:
        G = F[F['packaging'].apply(lambda x: similar_str(str(x), input_vals["packtype"]))]
        if G.empty:
            results.append(list(row) + [1] + [''] * len(master.columns) + ['PACKTYPE', ''])
            continue

    # --- LLM extraction & scoring remains the same ---
    product_name = extract_product_name_ollama(input_vals["itemdesc"])
    time.sleep(1)
    if product_name.startswith("Error:"):
        results.append(list(row) + [1] + [''] * len(master.columns) + ['ITEMDESC', product_name])
        continue

    trans_main, trans_offer = split_item_desc(product_name)

    def score_match(master_desc):
        master_name = extract_product_name_ollama(master_desc)
        time.sleep(1)
        master_main, master_offer = split_item_desc(master_name)
        main_score = SequenceMatcher(None, trans_main.upper(), master_main.upper()).ratio()
        offer_score = SequenceMatcher(None, trans_offer.upper(), master_offer.upper()).ratio()
        return (main_score + offer_score) / 2, master_name

    # Score all matches
    match_scores = []
    for _, f_row in G.iterrows():
        score, m_name = score_match(f_row["itemdesc"])
        match_scores.append((score, m_name, f_row))

    # Sort and select top 3
    match_scores = sorted(match_scores, key=lambda x: x[0], reverse=True)
    top_matches = match_scores[:3]

    if top_matches and top_matches[0][0] >= 0.85:
        top = top_matches[0][2]
        results.append(list(row) + [0] + list(top[master.columns]) + ['', top_matches[0][1]])
    else:
        results.append(list(row) + [2] + [''] * len(master.columns) + ['ITEMDESC', product_name])

    # --- LLM extraction & scoring remains the same ---
    product_name = extract_product_name_ollama(input_vals["itemdesc"])
    time.sleep(1)
    if product_name.startswith("Error:"):
        results.append(list(row) + [1] + [''] * len(master.columns) + ['ITEMDESC', product_name])
        continue

    trans_main, trans_offer = split_item_desc(product_name)

    def score_match(master_desc):
        master_name = extract_product_name_ollama(master_desc)
        time.sleep(1)
        master_main, master_offer = split_item_desc(master_name)
        main_score = SequenceMatcher(None, trans_main.upper(), master_main.upper()).ratio()
        offer_score = SequenceMatcher(None, trans_offer.upper(), master_offer.upper()).ratio()
        return (main_score + offer_score) / 2, master_name

    # Score all matches
    match_scores = []
    for _, f_row in G.iterrows():
        score, m_name = score_match(f_row["itemdesc"])
        match_scores.append((score, m_name, f_row))


    # Sort and select top 3
    match_scores = sorted(match_scores, key=lambda x: x[0], reverse=True)
    top_matches = match_scores[:3]

    if top_matches and top_matches[0][0] >= 0.85:
        top = top_matches[0][2]
        results.append(list(row) + [0] + list(top[master.columns]) + ['', top_matches[0][1]])
    else:
        results.append(list(row) + [2] + [''] * len(master.columns) + ['ITEMDESC', product_name])

# Final Columns
columns = list(transactions.columns) + ['MATCHED'] + list(master.columns) + ['ERROR', 'ProductName']

# Debug mismatch checker
for i, r in enumerate(results):
    if len(r) != len(columns):
        print(f"⚠️ Row {i} has {len(r)} columns, expected {len(columns)}")

# Save
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("matches.csv", index=False)
print("✅ Matching complete. Results saved to matches.csv.")