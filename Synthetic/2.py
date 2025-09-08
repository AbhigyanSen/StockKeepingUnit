import pandas as pd
import json
import subprocess

# Load CSVs
transaction = pd.read_csv("transaction.csv")
master = pd.read_csv("master.csv")

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
    
    # Build the prompt for Mistral
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

# --- Main loop with tracking ---
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

# --- Final reporting ---
match_percentage = (matched_count / total_rows * 100) if total_rows > 0 else 0
accuracy_percentage = (exact_match_count / matched_count * 100) if matched_count > 0 else 0

print(f"\nMatches found: {matched_count}/{total_rows} ({match_percentage:.2f}%)")
print(f"Exact code accuracy: {exact_match_count}/{matched_count} ({accuracy_percentage:.2f}%)")