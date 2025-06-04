import pandas as pd
import requests
import time
import re

# Normalize packtype (add more mappings as needed)
PACKTYPE_EQUIVALENCE = {
    "CDB": ["CDB", "CDBOX"]
}

# LLM product name extractor with bracket-removal
def extract_product_name_ollama(description, model="mistral"):
    prompt = (
        f"Extract only the product name from the following description. "
        f"Do not include quantity, unit, or pack type:\n\n"
        f"{description}\n\nProduct name:"
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            raw = response.json()["response"].strip()
            cleaned = re.sub(r"\(.*?\)", "", raw).strip()
            return cleaned
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Load data
master = pd.read_csv(r"Data\DataCleaned\master_cleaned.csv")
transactions = pd.read_csv(r"transaction_subset.csv")

# Output storage
results = []

for idx, row in transactions.iterrows():
    input_manufacturer = str(row['MANUFACTURE']).strip()
    input_brand = str(row['BRAND']).strip()
    input_itemdesc = str(row['ITEMDESC']).strip()
    input_qty = int(row['QTY'])
    input_unit = str(row['UNIT']).strip()
    input_packsize = str(row['PACKSIZE']).strip()
    input_packtype = str(row['PACKTYPE']).strip()

    # Step 1: Filter MANUFACTURER
    A = master[master['company'] == input_manufacturer]
    if A.empty:
        results.append([*row, 1, '', '', '', '', '', '', '', 'MANUFACTURE', ''])
        continue

    # Step 2: Filter BRAND
    B = A[A['brand'] == input_brand]
    if B.empty:
        results.append([*row, 1, '', '', '', '', '', '', '', 'BRAND', ''])
        continue

    # Step 3: Filter QTY
    C = B[B['qty'] == input_qty]
    if C.empty:
        results.append([*row, 1, '', '', '', '', '', '', '', 'QTY', ''])
        continue

    # Step 4: Filter PACKTYPE with equivalence
    match_packtypes = PACKTYPE_EQUIVALENCE.get(input_packtype, [input_packtype])
    D = C[C['packaging'].isin(match_packtypes)]
    if D.empty:
        results.append([*row, 1, '', '', '', '', '', '', '', 'PACKTYPE', ''])
        continue

    # Step 5: Get LLM Product Name for ITEMDESC
    product_name = extract_product_name_ollama(input_itemdesc)
    time.sleep(1)
    if product_name.startswith("Error:"):
        results.append([*row, 1, '', '', '', '', '', '', '', 'ITEMDESC', product_name])
        continue

    # Step 6: Get product name for all D[itemdesc]
    D = D.copy()
    D["llm_product_name"] = D["itemdesc"].astype(str).apply(extract_product_name_ollama)
    time.sleep(1)

    # Step 7: Match by cleaned product name
    match_rows = D[D["llm_product_name"].str.strip().str.upper() == product_name.strip().upper()]
    if match_rows.empty:
        results.append([*row, 1, '', '', '', '', '', '', '', 'ITEMDESC', product_name])
    else:
        match = match_rows.iloc[0]
        results.append([
            *row, 0,
            match['itemdesc'], match['company'], match['brand'], match['packaging'],
            match['qty'], match['uomdesc'], match['pack_size'], '', product_name
        ])

# Define final columns
columns = list(transactions.columns) + [
    'MATCHED', 'MATCH_ITEMDESC', 'MATCH_COMPANY', 'MATCH_BRAND', 'MATCH_PACKAGING',
    'MATCH_QTY', 'MATCH_UNIT', 'MATCH_PACKSIZE', 'ERROR', 'PRODUCT_NAME'
]

# Save to matches.csv
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("matches.csv", index=False)
print("Matching complete. Results saved to matches.csv.")