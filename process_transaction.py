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
TRANSACTION_FOLDER = 'Demo'   # <-- folder path here

# Ensure required directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("FinalMatches", exist_ok=True)

# Files from master step
INDEX_FOLDER = "./datastore/cat_indexes"
METADATA_FILE = "./datastore/metadata.json"

# Column definitions
m_columns = ['itemcode', 'catcode', 'company', 'mbrand', 'brand', 'sku',
             'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']

# ----------------- FUNCTION: EMBEDDING -----------------
def get_embedding(text):
    """Call local Ollama client to get embedding for a given text."""
    try:
        response = ollama.embeddings(model=MODEL_NAME, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"|ERROR| Error embedding text: {e}")
        return None

# ----------------- LOAD METADATA -----------------
with open(METADATA_FILE, "r") as f:
    metadata_nested = json.load(f)

# ----------------- HELPER: Extract numeric from PACKSIZE -----------------
def extract_numeric(val):
    """Extract the first numeric value from a string, else return None."""
    if pd.isna(val):
        return None
    match = re.search(r"\d+(\.\d+)?", str(val))
    return float(match.group()) if match else None

# ----------------- MAIN PROCESSING FUNCTION -----------------
def process_transaction_file(file_path):
    input_filename = os.path.splitext(os.path.basename(file_path))[0]
    OUTPUT_CSV = f"./output/matches_{input_filename}.csv"
    FINAL_OUTPUT_CSV = f"./FinalMatches/final_matches_{input_filename}.csv"

    print(f"\n|INFO| Processing file: {file_path}")
    transaction = pd.read_csv(
        file_path,
        dtype={
            "PERIOD": str,
            "AUDITYPE": str,
            "STORECODE": str,
            "DLRCODE": str,
            "ITEMCODE": str,
            "NEW_CODES": str,
            "CATEGORY": str,
            "MRP": str
        }
    )

    # Clean CATEGORY, NEW_CODES, MRP values (remove .0)
    transaction["CATEGORY"] = transaction["CATEGORY"].str.replace(r"\.0$", "", regex=True)
    transaction["NEW_CODES"] = transaction["NEW_CODES"].str.replace(r"\.0$", "", regex=True)
    transaction["MRP"] = transaction["MRP"].str.replace(r"\.0$", "", regex=True)

    # Keep full transaction columns
    all_t_columns = list(transaction.columns)

    results_list = []

    print(f"|INFO| Querying {len(transaction)} transaction rows...")
    for tx_id, row in tqdm(transaction.iterrows(), total=len(transaction), desc="|INFO| Transaction Queries"):
        # Only embed selected columns
        embed_cols = ["CATEGORY", "MANUFACTURE", "BRAND", "ITEMDESC", "MRP", "PACKSIZE", "PACKTYPE"]
        row_values = [str(row.get(c, "")).strip() for c in embed_cols if pd.notna(row.get(c, "")) and str(row.get(c, "")).strip() != ""]
        query_text = " ".join(row_values)

        if not row_values:
            # Blank row
            result_entry = {"t_row_id": tx_id, "rank": None, "matched_itemcode": None, "distance": None}
            for col in all_t_columns:
                result_entry[f"t_{col}"] = row[col]
            for k in m_columns:
                if k != "itemcode":
                    result_entry[f"m_{k}"] = None
            results_list.append(result_entry)
            continue

        query_emb = get_embedding(query_text)
        catcode_target = str(row.get("CATEGORY") or row.get("t_CATEGORY") or "").strip()

        # --- Catcode remapping ---
        if catcode_target == "134":
            catcode_target = "118"

        # Collect indexes to search
        index_paths = []
        if catcode_target and os.path.exists(f"{INDEX_FOLDER}/index_{catcode_target}.faiss"):
            # Normal case: valid category and index exists
            index_paths = [(catcode_target, f"{INDEX_FOLDER}/index_{catcode_target}.faiss")]
        else:
            # Fallback: search all indexes
            print(f"|WARN| No valid index for catcode={catcode_target}, searching across all indexes for row {tx_id}")
            for fname in os.listdir(INDEX_FOLDER):
                if fname.endswith(".faiss"):
                    cc = fname.replace("index_", "").replace(".faiss", "")
                    index_paths.append((cc, os.path.join(INDEX_FOLDER, fname)))

        # Search across selected indexes
        for cc, index_path in index_paths:
            index = faiss.read_index(index_path)
            itemcodes = list(metadata_nested.get(cc, {}).keys())

            if query_emb is not None and query_emb.shape[0] == index.d:
                query_np = np.array([query_emb]).astype('float32')
                distances, indices = index.search(query_np, k=10)

                for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
                    if idx >= len(itemcodes):
                        continue
                    itemcode = itemcodes[idx]
                    metadata_item = metadata_nested[cc][itemcode]

                    result_entry = {
                        "t_row_id": tx_id,
                        "rank": rank,
                        "matched_itemcode": itemcode,
                        "distance": dist
                    }
                    for col in all_t_columns:
                        result_entry[f"t_{col}"] = row[col]
                    for k, v in metadata_item.items():
                        result_entry[f"m_{k}"] = v

                    results_list.append(result_entry)

        if query_emb is None:
            # Embedding failed
            result_entry = {"t_row_id": tx_id, "rank": None, "matched_itemcode": None, "distance": None}
            for col in all_t_columns:
                result_entry[f"t_{col}"] = row[col]
            for k in m_columns:
                if k != "itemcode":
                    result_entry[f"m_{k}"] = None
            results_list.append(result_entry)

    # ----------------- SAVE INITIAL RESULTS -----------------
    results_df = pd.DataFrame(results_list)
    t_cols_prefixed = [f"t_{c}" for c in all_t_columns]
    m_cols_prefixed = [f"m_{c}" for c in m_columns if c != "itemcode"]

    ordered_cols = ["t_row_id"] + t_cols_prefixed + ["rank", "matched_itemcode", "distance"] + m_cols_prefixed

    # Keep only those that exist in results_df
    existing_cols = [c for c in ordered_cols if c in results_df.columns]
    results_df = results_df[existing_cols]

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"|INFO| Saved transaction matches to {OUTPUT_CSV}")

    # ----------------- PROCESS FILTERED OUTPUT -----------------
    final_results = []
    grouped = results_df.groupby("t_row_id")

    for _, group in grouped:
        t_packsize_val = group.iloc[0].get("t_PACKSIZE", None)
        t_num = extract_numeric(t_packsize_val)

        selected_rows = []
        if t_num is not None:
            matches = []
            for _, row in group.iterrows():
                try:
                    m_wght_val = float(row["m_wght"])
                    if m_wght_val == t_num:
                        matches.append(row)
                except (ValueError, TypeError):
                    continue
            if matches:
                selected_rows = pd.DataFrame(matches).sort_values("rank").head(3).to_dict("records")

        if not selected_rows:
            selected_rows = group.sort_values("rank").head(3).to_dict("records")

        final_results.extend(selected_rows)

    # final_df = pd.DataFrame(final_results).drop(columns=["t_row_id"])
    final_df = pd.DataFrame(final_results)                                  # keep t_row_id
    final_df.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"|INFO| Saved final filtered top-3 matches to {FINAL_OUTPUT_CSV}")


# ----------------- LOOP THROUGH ALL FILES -----------------
if __name__ == "__main__":
    csv_files = [f for f in os.listdir(TRANSACTION_FOLDER) if f.endswith(".csv")]
    if not csv_files:
        print(f"|WARN| No CSV files found in {TRANSACTION_FOLDER}")
    else:
        for file in csv_files:
            process_transaction_file(os.path.join(TRANSACTION_FOLDER, file))