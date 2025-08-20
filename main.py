import pandas as pd
import requests
import re
import time
from difflib import SequenceMatcher

# ---------- Load Data ----------
master = pd.read_csv("/home/dcsadmin/Documents/del_SKU/StockKeepingUnit/Data/DataCleaned/master_cleaned.csv").fillna("")
transactions = pd.read_csv("/home/dcsadmin/Documents/del_SKU/StockKeepingUnit/Data/DataCleaned/transaction_cleaned.csv").fillna("")

# ---------- Config ----------
PACKTYPE_EQUIVALENCE = {
    "CDB": ["CDB", "CDBOX"],
    "HL": ["HL", "HARDLINE", "HL PACK"]  # extend as needed
}

# ---------- Scoring Mechanism ----------
MATCH_WEIGHTS = {
    "manufacture": 0.4,
    "brand": 0.3,
    "category": 0.1,
    "packsize": 0.1,
    "packtype": 0.1
}

MATCH_THRESHOLD = 0.75  # below this → fallback to LLM


# ---------- Utilities ----------
def extract_product_name_ollama(description, model="mistral"):
    prompt = (
        f"Extract only the product name from the following description. "
        f"Do not include quantity, unit, or pack type:\n\n{description}\n\nProduct name:"
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        if response.status_code == 200:
            raw = response.json()["response"].strip()
            cleaned = re.sub(r"\(.*?\)", "", raw).strip()
            return cleaned
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def similar_str(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def packsize_match(p1, p2):
    return re.sub(r"\.0+", "", str(p1).replace(" ", "").upper()) == re.sub(
        r"\.0+", "", str(p2).replace(" ", "").upper()
    )


def score_row(master_row, trans_row):
    """Score a master row against transaction row"""
    score = 0

    # Manufacture
    score += MATCH_WEIGHTS["manufacture"] * similar_str(
        master_row["company"], trans_row["manufacture"]
    )

    # Brand
    score += MATCH_WEIGHTS["brand"] * similar_str(
        master_row["brand"], trans_row["brand"]
    )

    # Category
    score += (
        MATCH_WEIGHTS["category"]
        if str(master_row["catcode"]).strip().upper()
        == str(trans_row["category"]).strip().upper()
        else 0
    )

    # Packsize
    score += (
        MATCH_WEIGHTS["packsize"]
        if packsize_match(master_row["qty"], trans_row["packsize"])
        else 0
    )

    # Packtype
    match_packtypes = PACKTYPE_EQUIVALENCE.get(
        trans_row["packtype"], [trans_row["packtype"]]
    )
    score += (
        MATCH_WEIGHTS["packtype"]
        if str(master_row["packtype"]).upper() in [s.upper() for s in match_packtypes]
        else 0
    )

    return score


# ---------- Main Processing ----------
results = []

for idx, row in transactions.iterrows():
    trans_vals = {
        "manufacture": str(row["MANUFACTURE"]).strip(),
        "brand": str(row["BRAND"]).strip(),
        "category": str(row["CATEGORY"]).strip(),
        "itemdesc": str(row["ITEMDESC"]).strip(),
        "qty": row.get("qty", ""),
        "uomdesc": row.get("uomdesc", ""),
        "packsize": str(row["PACKSIZE"]).strip(),
        "packtype": str(row["PACKTYPE"]).strip(),
    }

    # Score every master row
    match_scores = []
    for _, m_row in master.iterrows():
        s = score_row(m_row, trans_vals)
        match_scores.append((s, m_row))

    # Sort by score
    match_scores = sorted(match_scores, key=lambda x: x[0], reverse=True)

    if match_scores and match_scores[0][0] >= MATCH_THRESHOLD:
        # ✅ High confidence match
        best = match_scores[0][1]
        results.append(list(row) + [0, match_scores[0][0]] + list(best[master.columns]) + ["", ""])
    else:
        # ❌ Use LLM as fallback
        product_name = extract_product_name_ollama(trans_vals["itemdesc"])
        time.sleep(1)
        if product_name.startswith("Error:"):
            results.append(
                list(row)
                + [1, 0.0]
                + [""] * len(master.columns)
                + ["ITEMDESC", product_name]
            )
            continue

        # Compare with master using LLM-extracted product names
        llm_scores = []
        for _, m_row in master.iterrows():
            master_name = extract_product_name_ollama(m_row["sku"])
            time.sleep(1)
            score = similar_str(product_name, master_name)
            llm_scores.append((score, m_row, master_name))

        llm_scores = sorted(llm_scores, key=lambda x: x[0], reverse=True)

        if llm_scores and llm_scores[0][0] >= 0.75:
            best = llm_scores[0][1]
            results.append(
                list(row)
                + [0, llm_scores[0][0]]
                + list(best[master.columns])
                + ["", llm_scores[0][2]]
            )
        else:
            results.append(
                list(row)
                + [2, 0.0]
                + [""] * len(master.columns)
                + ["ITEMDESC", product_name]
            )

# ---------- Save ----------
columns = (
    list(transactions.columns)
    + ["MATCHED", "MATCH_SCORE"]
    + list(master.columns)
    + ["ERROR", "ProductName"]
)
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("matches.csv", index=False)
print("✅ Matching complete. Results saved to matches.csv.")