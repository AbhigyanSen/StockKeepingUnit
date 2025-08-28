import re
import pandas as pd

# ----------------- FUNCTION: EXTRACT NUMERIC -----------------
def extract_numeric(value):
    """
    Extract numeric part from a string like '100GM', '52PCS', '1 NO'.
    Returns float or None if nothing found.
    """
    if pd.isna(value):
        return None
    s = str(value)
    match = re.search(r"\d+(\.\d+)?", s)
    if match:
        return float(match.group())
    return None

# ----------------- FUNCTION: FILTER TOP MATCHES -----------------
def filter_top_matches(results_df, top_k=2):
    """
    For each transaction, return top_k rows sorted by:
    1. closeness of TR_PACKSIZE numeric vs M_wght
    2. FAISS distance (as fallback)
    """
    filtered = []

    for tr_text, group in results_df.groupby(
        ["TR_CATEGORY", "TR_MANUFACTURE", "TR_BRAND", "TR_ITEMDESC", "TR_MRP", "TR_PACKSIZE", "TR_PACKTYPE"]
    ):
        group = group.copy()

        # Extract numeric values
        tr_packnum = extract_numeric(group["TR_PACKTYPE"].iloc[0])  # transaction value
        group["M_wght_num"] = group["M_wght"].apply(extract_numeric)

        if tr_packnum is not None:
            # Sort by difference in size first, then distance
            group["pack_diff"] = abs(group["M_wght_num"].fillna(1e9) - tr_packnum)
            group = group.sort_values(by=["pack_diff", "distance"], ascending=[True, True])
        else:
            # No numeric to compare, fallback to FAISS distance
            group = group.sort_values(by="distance", ascending=True)

        filtered.append(group.head(top_k))

    return pd.concat(filtered, ignore_index=True)


# Load original matches
results_df = pd.read_csv("t_TR_matches.csv")

# Apply filtering
filtered_df = filter_top_matches(results_df, top_k=1)

# Save final output
filtered_df.to_csv("t_TR_matches_filtered.csv", index=False)
print("Saved filtered matches to t_TR_matches_filtered.csv")