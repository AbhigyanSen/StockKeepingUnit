import os
import numpy as np
import pandas as pd
import ollama
import faiss
import json
from tqdm import tqdm
import re

# ----------------- CONFIG -----------------
MODEL_NAME = "nomic-embed-text"
TRANSACTION_FILE = 'Data/DataCSV/transaction/dec-24.csv'

# Ensure required directories exist
os.makedirs("output", exist_ok=True)

# Files from master step
FAISS_INDEX_FILE = "./temp/master_index.faiss"
METADATA_FILE = "./temp/metadata.json"

# Dynamic output filenames
input_filename = os.path.splitext(os.path.basename(TRANSACTION_FILE))[0]
OUTPUT_CSV = f"./output/matches_{input_filename}.csv"
FINAL_OUTPUT_CSV = f"./output/final_matches_{input_filename}.csv"

# Column definitions
m_columns = ['itemcode', 'catcode', 'company', 'mbrand', 'brand', 'sku',
             'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']
t_columns = ['CATEGORY', 'MANUFACTURE', 'BRAND', 'ITEMDESC', 'MRP', 'PACKSIZE', 'PACKTYPE']

# ----------------- LOAD DATA -----------------
transaction = pd.read_csv(TRANSACTION_FILE)
transaction = transaction[t_columns]

# ----------------- FUNCTION: EMBEDDING -----------------
def get_embedding(text):
    """Call local Ollama client to get embedding for a given text."""
    try:
        response = ollama.embeddings(model=MODEL_NAME, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"|ERROR| Error embedding text: {e}")
        return None

# ----------------- LOAD FAISS & METADATA -----------------
index = faiss.read_index(FAISS_INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

itemcodes = list(metadata.keys())

# ----------------- QUERY TRANSACTIONS -----------------
print(f"\n|INFO| Querying {len(transaction)} transaction rows...")
results_list = []

for _, row in tqdm(transaction.iterrows(), total=len(transaction), desc="|INFO| Transaction Queries"):
    query_text = " ".join(str(val) for val in row if pd.notna(val))
    query_emb = get_embedding(query_text)

    if query_emb is not None:
        query_np = np.array([query_emb]).astype('float32')
        distances, indices = index.search(query_np, k=10)

        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            itemcode = itemcodes[idx]
            metadata_item = metadata[itemcode]

            result_entry = {
                "rank": rank + 1,
                "matched_itemcode": itemcode,
                "distance": dist
            }

            # Add transaction columns
            for col in transaction.columns:
                result_entry[f"t_{col}"] = row[col]

            # Add master metadata
            for k, v in metadata_item.items():
                result_entry[f"m_{k}"] = v

            results_list.append(result_entry)
    else:
        result_entry = {
            "rank": None,
            "matched_itemcode": None,
            "distance": None
        }
        for col in transaction.columns:
            result_entry[f"t_{col}"] = row[col]
        for k in m_columns:
            if k != "itemcode":
                result_entry[f"m_{k}"] = None
        results_list.append(result_entry)

# ----------------- SAVE INITIAL RESULTS -----------------
results_df = pd.DataFrame(results_list)
t_cols_prefixed = [f"t_{c}" for c in t_columns]
m_cols_prefixed = [f"m_{c}" for c in m_columns if c != "itemcode"]
ordered_cols = t_cols_prefixed + ["rank", "matched_itemcode", "distance"] + m_cols_prefixed
results_df = results_df[ordered_cols]
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"|INFO| Saved transaction matches to {OUTPUT_CSV}")

# ----------------- HELPER: Extract numeric from PACKSIZE -----------------
def extract_numeric(val):
    """Extract the first numeric value from a string, else return None."""
    if pd.isna(val):
        return None
    match = re.search(r"\d+(\.\d+)?", str(val))
    return float(match.group()) if match else None

# ----------------- PROCESS FILTERED OUTPUT -----------------
final_results = []
grouped = results_df.groupby([col for col in results_df.columns if col.startswith("t_")])

for _, group in grouped:
    t_packsize_val = group.iloc[0]["t_PACKSIZE"]
    t_num = extract_numeric(t_packsize_val)
    chosen_row = None

    if t_num is not None:
        for _, row in group.iterrows():
            try:
                m_wght_val = float(row["m_wght"])
                if m_wght_val == t_num:
                    chosen_row = row
                    break
            except (ValueError, TypeError):
                continue

    if chosen_row is None:
        chosen_row = group.sort_values("rank").iloc[0]

    final_results.append(chosen_row)

final_df = pd.DataFrame(final_results)
final_df.to_csv(FINAL_OUTPUT_CSV, index=False)
print(f"|INFO| Saved final filtered matches to {FINAL_OUTPUT_CSV}")