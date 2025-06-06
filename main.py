import pandas as pd
import requests
import re
import time
from difflib import SequenceMatcher

# Load CSVs
master = pd.read_csv(r"D:\Projects\SKU\StockKeepingUnit\Labelled_Data\master.csv")
transactions = pd.read_csv(r"D:\Projects\SKU\StockKeepingUnit\transaction_FROMLABELLED.csv")

PACKTYPE_EQUIVALENCE = {
    "CDB": ["CDB", "CDBOX"]
}

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

def similar_str(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def packsize_match(p1, p2):
    return re.sub(r'\.0+', '', p1.replace(" ", "").upper()) == re.sub(r'\.0+', '', p2.replace(" ", "").upper())

results = []
for idx, row in transactions.iterrows():
    input_vals = {
        "MANUFACTURE": str(row['MANUFACTURE']).strip(),
        "BRAND": str(row['BRAND']).strip(),
        "ITEMDESC": str(row['ITEMDESC']).strip(),
        "QTY": row['QTY'],
        "UNIT": str(row['UNIT']).strip(),
        "PACKSIZE": str(row['PACKSIZE']).strip(),
        "PACKTYPE": str(row['PACKTYPE']).strip()
    }

    A = master[master['company'].str.strip().str.upper() == input_vals["MANUFACTURE"].upper()]
    if A.empty:
        results.append([*row, 1] + [''] * len(master.columns) + ['MANUFACTURE', ''])
        continue

    B = A[A['brand'].str.strip().str.upper() == input_vals["BRAND"].upper()]
    if B.empty:
        results.append([*row, 1] + [''] * len(master.columns) + ['BRAND', ''])
        continue

    C = B[B['qty'] == input_vals["QTY"]]
    if not C.empty:
        D = C[C['uomdesc'].str.strip().str.upper() == input_vals["UNIT"].upper()]
        if not D.empty:
            E = D
        else:
            # Try packsize similarity
            E = C[C['pack_size'].apply(lambda x: packsize_match(str(x), input_vals["PACKSIZE"]))]
    else:
        # Try packsize similarity directly
        E = B[B['pack_size'].apply(lambda x: packsize_match(str(x), input_vals["PACKSIZE"]))]

    if E.empty:
        results.append([*row, 1] + [''] * len(master.columns) + ['PACKSIZE', ''])
        continue

    match_packtypes = PACKTYPE_EQUIVALENCE.get(input_vals["PACKTYPE"], [input_vals["PACKTYPE"]])
    F = E[E['packaging'].isin(match_packtypes)]

    if F.empty:
        # Try spelling similarity
        F = E[E['packaging'].apply(lambda x: similar_str(x, input_vals["PACKTYPE"]))]
        if F.empty:
            results.append([*row, 1] + [''] * len(master.columns) + ['PACKTYPE', ''])
            continue

    product_name = extract_product_name_ollama(input_vals["ITEMDESC"])
    time.sleep(1)
    if product_name.startswith("Error:"):
        results.append([*row, 1] + [''] * len(master.columns) + ['ITEMDESC', product_name])
        continue

    F = F.copy()
    F["llm_product_name"] = F["itemdesc"].astype(str).apply(extract_product_name_ollama)
    time.sleep(1)

    match_rows = F[F["llm_product_name"].str.strip().str.upper() == product_name.strip().upper()]
    if not match_rows.empty:
        match = match_rows.iloc[0]
        results.append([*row, 0] + list(match) + ['', product_name])
    else:
        results.append([*row, 2] + [''] * len(master.columns) + ['ITEMDESC', product_name])

# Columns setup
columns = list(transactions.columns) + ['MATCHED'] + list(master.columns) + ['ERROR', 'ProductName']
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("matches.csv", index=False)
print("Matching complete. Results saved to matches.csv.")