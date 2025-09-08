import pandas as pd
import json
import subprocess
import os
import glob

# --- Config ---
INPUT_FOLDER = r"/home/dcsadmin/Documents/del_SKU_AS/StockKeepingUnit/FinalMatches/"  # folder containing all CSV files
OUTPUT_FILE = "Exact.csv"

# --- Helper functions ---
def ask_mistral(prompt):
    """Send prompt to Mistral running locally via ollama"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

def find_match_with_mistral(trans_row, master_df):
    candidates = master_df[master_df["t_NEW_CODES"] == trans_row["t_NEW_CODES"]].copy()
    if candidates.empty:
        return None
    
    # Drop t_NEW_CODES for clarity
    trans_dict = trans_row.drop(labels=["t_NEW_CODES"]).to_dict()
    cand_dicts = candidates.drop(columns=["t_NEW_CODES"]).to_dict(orient="records")
    
    prompt = f"""
You are given a transaction item and possible master items. 
Find the best matching master itemcode. If no good match, return None.

Transaction:
{json.dumps(trans_dict, indent=2)}

Candidates:
{json.dumps(cand_dicts, indent=2)}

Return only the matched_itemcode (number) or None.
"""
    response = ask_mistral(prompt)
    
    try:
        if response.lower().strip() == "none":
            return None
        return int(response.strip())
    except:
        return None

# --- Main processing loop ---
results = []

for file_path in glob.glob(os.path.join(INPUT_FOLDER, "*.csv")):
    filename = os.path.basename(file_path)
    print(f"\n?? Processing file: {filename}\n" + "="*80)

    # Read CSV
    df = pd.read_csv(file_path)

    # Transaction columns
    transaction_cols = [
        "t_NEW_CODES", "t_CATEGORY", "t_MANUFACTURE", "t_BRAND",
        "t_ITEMDESC", "t_MRP", "t_PACKSIZE", "t_PACKTYPE"
    ]

    # Master columns
    master_cols = [
        "t_NEW_CODES", "matched_itemcode", "m_catcode", "m_company", "m_mbrand",
        "m_brand", "m_sku", "m_packtype", "m_base_pack", "m_flavor", "m_color",
        "m_wght", "m_uom", "m_mrp"
    ]

    # Build transaction and master
    transaction = df[transaction_cols].drop_duplicates()
    master = df[master_cols].drop_duplicates()

    total_rows = len(transaction)
    matched_count = 0
    exact_match_count = 0

    for idx, row in transaction.iterrows():
        match = find_match_with_mistral(row, master)
        
        print("Transaction row index:", idx)
        print("Transaction:", row.to_dict())
        print("Matched ItemCode:", match)
        print("-" * 60)
        
        if match is not None:
            matched_count += 1
            if row["t_NEW_CODES"] == match:
                exact_match_count += 1

    match_percentage = (matched_count / total_rows * 100) if total_rows > 0 else 0
    accuracy_percentage = (exact_match_count / matched_count * 100) if matched_count > 0 else 0

    print(f"\n? Summary for {filename}:")
    print(f"Matches found: {matched_count}/{total_rows} ({match_percentage:.2f}%)")
    print(f"Exact code accuracy: {exact_match_count}/{matched_count} ({accuracy_percentage:.2f}%)")
    print("="*80)

    # Append results
    results.append({
        "filename": filename,
        "matches": f"{matched_count}/{total_rows}",
        "match_percentage": f"{match_percentage:.2f}%",
        "accuracy": f"{exact_match_count}/{matched_count}",
        "accuracy_percentage": f"{accuracy_percentage:.2f}%"
    })

# --- Save summary CSV ---
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"\n?? All results saved to {OUTPUT_FILE}")
