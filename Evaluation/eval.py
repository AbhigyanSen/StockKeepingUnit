import pandas as pd
import os

# Input file
input_file = r"output\final_matches_oct-24.csv"

# Extract filename without path/extension for naming
filename = os.path.splitext(os.path.basename(input_file))[0].replace("final_matches_", "")

# Load CSV
df = pd.read_csv(input_file)

# Keep only needed columns
df = df[["t_NEW_CODES", "matched_itemcode", "rank"]]

# Pivot so each t_NEW_CODES has rank1, rank2, rank3 matched codes
pivoted = df.pivot_table(index="t_NEW_CODES",
                         columns="rank",
                         values="matched_itemcode",
                         aggfunc="first")

# Rename columns for clarity
pivoted.columns = [f"rank_{c}" for c in pivoted.columns]
pivoted = pivoted.reset_index()

# Accuracy calculations
total = len(pivoted)

# Top-1 accuracy
top1 = (pivoted["t_NEW_CODES"] == pivoted["rank_1"]).sum() / total

# Top-2 accuracy
top2 = ((pivoted["t_NEW_CODES"] == pivoted["rank_1"]) |
        (pivoted["t_NEW_CODES"] == pivoted["rank_2"])).sum() / total

# Top-3 accuracy
top3 = ((pivoted["t_NEW_CODES"] == pivoted["rank_1"]) |
        (pivoted["t_NEW_CODES"] == pivoted["rank_2"]) |
        (pivoted["t_NEW_CODES"] == pivoted["rank_3"])).sum() / total

# ---- Save eval.csv (append mode) ----
eval_file = "eval.csv"
row = pd.DataFrame([[filename, round(top1, 4), round(top2, 4), round(top3, 4)]],
                   columns=["filename", "Top1", "Top2", "Top3"])

if os.path.exists(eval_file):
    row.to_csv(eval_file, mode="a", header=False, index=False)
else:
    row.to_csv(eval_file, index=False)

# ---- Save mismatched rows ----
# For each group, determine match rank (if any)
def flag_group(group):
    tcode = group["t_NEW_CODES"].iloc[0]
    matches = group.loc[group["matched_itemcode"] == tcode, "rank"]
    if matches.empty:
        flag_val = 4
    else:
        flag_val = matches.min()  # first rank that matched
    group = group.copy()
    group["FLAG"] = flag_val
    return group

mismatches = df.groupby("t_NEW_CODES", group_keys=False).apply(flag_group)

# Keep only those groups where FLAG != 1 (i.e., not a perfect rank-1 match)
mismatches = mismatches[mismatches["FLAG"] != 1]

mismatch_file = f"Evaluation/eval_{filename}.csv"
os.makedirs("Evaluation", exist_ok=True)
mismatches.to_csv(mismatch_file, index=False)

print(f"Results saved in {eval_file}")
print(f"Mismatched rows saved in {mismatch_file}")