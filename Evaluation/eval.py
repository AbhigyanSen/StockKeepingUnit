import pandas as pd
import os
import glob

# ---- CONFIG ----
# Folder containing final_matches_*.csv files
INPUT_FOLDER = "output"
EVAL_FILE = "eval.csv"
MISMATCH_FOLDER = "Evaluation"
os.makedirs(MISMATCH_FOLDER, exist_ok=True)


def process_file(file_path):
    # Extract filename (e.g., oct-24 from final_matches_oct-24.csv)
    filename = os.path.splitext(os.path.basename(file_path))[0].replace("final_matches_", "")

    # Load CSV
    df = pd.read_csv(file_path)

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
    if total == 0:
        print(f"|WARNING| {file_path} is empty, skipping...")
        return

    # Handle missing rank columns safely
    top1 = (pivoted["t_NEW_CODES"] == pivoted.get("rank_1")).sum() / total if "rank_1" in pivoted else 0
    top2 = (
        ((pivoted["t_NEW_CODES"] == pivoted.get("rank_1")) |
         (pivoted["t_NEW_CODES"] == pivoted.get("rank_2"))).sum() / total
        if any(col in pivoted for col in ["rank_1", "rank_2"]) else 0
    )
    top3 = (
        ((pivoted["t_NEW_CODES"] == pivoted.get("rank_1")) |
         (pivoted["t_NEW_CODES"] == pivoted.get("rank_2")) |
         (pivoted["t_NEW_CODES"] == pivoted.get("rank_3"))).sum() / total
        if any(col in pivoted for col in ["rank_1", "rank_2", "rank_3"]) else 0
    )

    # ---- Save eval.csv (append mode) ----
    row = pd.DataFrame([[filename, round(top1, 4), round(top2, 4), round(top3, 4)]],
                       columns=["filename", "Top1", "Top2", "Top3"])

    if os.path.exists(EVAL_FILE):
        row.to_csv(EVAL_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(EVAL_FILE, index=False)

    # ---- Save mismatched rows ----
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

    # Keep only those groups where FLAG != 1
    mismatches = mismatches[mismatches["FLAG"] != 1]

    mismatch_file = os.path.join(MISMATCH_FOLDER, f"eval_{filename}.csv")
    mismatches.to_csv(mismatch_file, index=False)

    print(f"|INFO| Processed {file_path}")
    print(f"       Results appended to {EVAL_FILE}")
    print(f"       Mismatched rows saved in {mismatch_file}")


# ---- MAIN LOOP ----
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "final_matches_*.csv"))

    if not csv_files:
        print(f"|WARNING| No files found in {INPUT_FOLDER}")
    else:
        print(f"|INFO| Found {len(csv_files)} files to process")
        for file in csv_files:
            process_file(file)